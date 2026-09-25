"""Persistence backends.

Render Free has an **ephemeral filesystem**: local writes vanish on redeploy,
restart or spin-down. Any state that must outlive the process therefore cannot
live on disk alone. This module defines one storage contract and three
backends:

``DiskStore``
    Fast local JSON documents. Always available.
``GistStore``
    Mirrors the same documents into a private GitHub Gist, so state survives
    an ephemeral filesystem. Deliberately *not* used as a high-frequency
    general-purpose filesystem — writes are coalesced and only important
    documents are mirrored.
``ChainedStore``
    Disk first (fast path), Gist as the durable mirror. Reads prefer disk and
    fall back to Gist when the container is cold.

Secrets are never written: every document is passed through the redactor
before it is serialised.
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import httpx

from .config import PersistenceConfig
from .errors import PersistenceError
from .redaction import redactor

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


class StateStore(abc.ABC):
    """A tiny document store: get / set / delete by key."""

    name = "store"

    @abc.abstractmethod
    async def get(self, key: str, default: Any = None) -> Any: ...

    @abc.abstractmethod
    async def set(self, key: str, value: Any) -> None: ...

    @abc.abstractmethod
    async def delete(self, key: str) -> None: ...

    async def keys(self) -> list[str]:
        return []

    async def healthy(self) -> tuple[bool, str]:
        return True, "ok"

    async def aclose(self) -> None:
        return None


class DiskStore(StateStore):
    """JSON documents under a directory. Atomic writes via temp + rename."""

    name = "disk"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self.root / f"{safe}.json"

    async def get(self, key: str, default: Any = None) -> Any:
        path = self._path(key)
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("disk read failed", extra={"key": key, "error": str(exc)})
            return default

    async def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        payload = json.dumps(redactor().scrub_deep(value), default=str, ensure_ascii=False, indent=2)
        with self._lock:
            try:
                fd, tmp = tempfile.mkstemp(dir=str(self.root), suffix=".tmp")
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
            except OSError as exc:
                raise PersistenceError(f"disk write failed for {key}: {exc}") from exc

    async def delete(self, key: str) -> None:
        path = self._path(key)
        with self._lock:
            if path.exists():
                try:
                    path.unlink()
                except OSError as exc:
                    raise PersistenceError(f"disk delete failed for {key}: {exc}") from exc

    async def keys(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))


class GistStore(StateStore):
    """Durable mirror in a single private GitHub Gist.

    All documents live as separate files inside one Gist, which keeps the
    request count low (one PATCH per write, one GET per cold read) and avoids
    the Gist-per-document sprawl that would waste API quota.
    """

    name = "gist"

    #: Marker written into every agent-created gist, so a later boot can find the
    #: state gist again instead of littering the account with duplicates.
    DESCRIPTION_MARKER = "auton-agent durable state"

    def __init__(
        self,
        gist_id: str | None,
        api_key: str,
        prefix: str = "auton-agent",
        *,
        allow_create: bool = True,
    ) -> None:
        # A gist id is intentionally OPTIONAL. The operator is told not to hardcode
        # one: on first boot the agent discovers or creates its own state gist and
        # caches the id locally, so GIST_ID is only ever a way to re-attach to a
        # known gist after a wipe.
        self.gist_id = gist_id or ""
        self.api_key = api_key
        self.prefix = prefix
        self.allow_create = allow_create
        self._client: httpx.AsyncClient | None = None
        self._cache: dict[str, Any] = {}
        self._cache_at: float = 0.0
        self._lock = asyncio.Lock()
        #: Where a self-created gist id is remembered so it survives restarts.
        self._id_cache_path = Path(os.environ.get("STATE_ROOT", ".agentstate")) / "gist_id"

    # -- self-provisioning ----------------------------------------------

    async def resolve_gist_id(self) -> str:
        """Return a usable gist id, discovering or creating one as needed.

        Order of preference:

        1. an explicitly configured ``GIST_ID``;
        2. an id cached from a previous boot (survives restarts when STATE_ROOT is
           on a writable path);
        3. an existing gist in the account carrying our description marker;
        4. a freshly created private gist.

        Discovery before creation is what keeps this idempotent after a redeploy
        wipes the local cache.
        """
        if self.gist_id:
            self._remember_id(self.gist_id)
            return self.gist_id

        cached = self._read_cached_id()
        if cached:
            self.gist_id = cached
            return cached

        client = await self._http()
        try:
            resp = await client.get(
                f"{GITHUB_API}/gists", headers=self._headers(), params={"per_page": 100}
            )
            if resp.status_code < 400:
                for gist in resp.json() or []:
                    if (gist.get("description") or "").strip().startswith(self.DESCRIPTION_MARKER):
                        found = gist.get("id")
                        if found:
                            log.info("re-attached to existing state gist", extra={"gist": found})
                            self.gist_id = found
                            self._remember_id(found)
                            return found
        except httpx.HTTPError as exc:
            raise PersistenceError(f"could not list gists: {exc}") from exc

        if not self.allow_create:
            raise PersistenceError(
                "no gist found and creation is disabled; set GIST_ID explicitly"
            )
        created = await self._create_gist()
        self.gist_id = created
        self._remember_id(created)
        log.info("created a new private state gist", extra={"gist": created})
        return created

    async def _create_gist(self) -> str:
        client = await self._http()
        body = {
            "description": f"{self.DESCRIPTION_MARKER} (auto-managed, safe to keep private)",
            "public": False,
            "files": {
                f"{self.prefix}__manifest.json": {
                    "content": json.dumps(
                        {
                            "managed_by": "auton-agent",
                            "purpose": "durable agent state; do not edit by hand",
                            "created_at": time.time(),
                        },
                        indent=2,
                    )
                }
            },
        }
        try:
            resp = await client.post(f"{GITHUB_API}/gists", headers=self._headers(), json=body)
        except httpx.HTTPError as exc:
            raise PersistenceError(f"gist creation failed: {exc}") from exc
        if resp.status_code >= 400:
            raise PersistenceError(
                f"gist creation failed: http {resp.status_code} {resp.text[:200]}"
            )
        return resp.json()["id"]

    def _remember_id(self, gist_id: str) -> None:
        try:
            self._id_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._id_cache_path.write_text(gist_id, encoding="utf-8")
        except OSError:  # pragma: no cover - read-only fs is not fatal
            pass

    def _read_cached_id(self) -> str:
        try:
            return self._id_cache_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    # -- plumbing -------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"token {self.api_key}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "auton-agent",
        }

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=25.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _filename(self, key: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return f"{self.prefix}__{safe}.json"

    async def _fetch(self, *, force: bool = False) -> dict[str, Any]:
        async with self._lock:
            if not force and self._cache and (time.time() - self._cache_at) < 30:
                return self._cache
            client = await self._http()
            try:
                resp = await client.get(f"{GITHUB_API}/gists/{self.gist_id}", headers=self._headers())
            except httpx.HTTPError as exc:
                raise PersistenceError(f"gist unreachable: {exc}") from exc
            if resp.status_code == 404:
                raise PersistenceError(
                    f"gist {self.gist_id} not found or not accessible with the configured token"
                )
            if resp.status_code >= 400:
                raise PersistenceError(f"gist read failed: http {resp.status_code}")
            payload = resp.json()
            files = payload.get("files") or {}
            documents: dict[str, Any] = {}
            for filename, meta in files.items():
                if not filename.startswith(f"{self.prefix}__"):
                    continue
                key = filename[len(self.prefix) + 2 :]
                if key.endswith(".json"):
                    key = key[:-5]
                content = (meta or {}).get("content")
                if content is None and (meta or {}).get("truncated"):
                    raw = (meta or {}).get("raw_url")
                    if raw:
                        try:
                            r2 = await client.get(raw, headers=self._headers())
                            content = r2.text
                        except httpx.HTTPError:
                            content = None
                try:
                    documents[key] = json.loads(content) if content else None
                except json.JSONDecodeError:
                    documents[key] = content
            self._cache = documents
            self._cache_at = time.time()
            return documents

    async def _patch(self, files: Mapping[str, Any]) -> None:
        client = await self._http()
        body = {
            "files": {
                self._filename(k): {
                    "content": json.dumps(redactor().scrub_deep(v), default=str, ensure_ascii=False, indent=2)
                }
                for k, v in files.items()
            }
        }
        try:
            resp = await client.patch(
                f"{GITHUB_API}/gists/{self.gist_id}", headers=self._headers(), json=body
            )
        except httpx.HTTPError as exc:
            raise PersistenceError(f"gist write failed: {exc}") from exc
        if resp.status_code >= 400:
            raise PersistenceError(f"gist write failed: http {resp.status_code} {resp.text[:200]}")
        async with self._lock:
            for k, v in files.items():
                self._cache[k] = v
            self._cache_at = time.time()

    # -- contract -------------------------------------------------------

    async def get(self, key: str, default: Any = None) -> Any:
        documents = await self._fetch()
        return documents.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        await self._patch({key: value})

    async def delete(self, key: str) -> None:
        client = await self._http()
        try:
            resp = await client.patch(
                f"{GITHUB_API}/gists/{self.gist_id}",
                headers=self._headers(),
                json={"files": {self._filename(key): None}},
            )
        except httpx.HTTPError as exc:
            raise PersistenceError(f"gist delete failed: {exc}") from exc
        if resp.status_code >= 400:
            raise PersistenceError(f"gist delete failed: http {resp.status_code}")
        async with self._lock:
            self._cache.pop(key, None)

    async def keys(self) -> list[str]:
        return sorted((await self._fetch()).keys())

    async def healthy(self) -> tuple[bool, str]:
        # Resolve the gist id first: on a fresh deployment the id is not known until
        # it has been discovered or created, and probing an empty id would report a
        # false "not accessible".
        try:
            await self.resolve_gist_id()
        except PersistenceError as exc:
            return False, exc.message
        try:
            await self._fetch(force=True)
        except PersistenceError as exc:
            return False, exc.message
        return True, "ok"


class ChainedStore(StateStore):
    """Disk-first with a durable Gist mirror.

    Writes go to disk synchronously (fast, always works) and to the mirror
    asynchronously; a mirror failure is logged but never fails the write,
    because losing durability is better than losing the task.
    """

    name = "chained"

    def __init__(self, primary: StateStore, mirror: StateStore, mirror_keys: tuple[str, ...] | None = None) -> None:
        self.primary = primary
        self.mirror = mirror
        self.mirror_keys = mirror_keys

    def _should_mirror(self, key: str) -> bool:
        if self.mirror_keys is None:
            return True
        return key in self.mirror_keys

    async def get(self, key: str, default: Any = None) -> Any:
        value = await self.primary.get(key, None)
        if value is not None:
            return value
        if not self._should_mirror(key):
            return default
        try:
            mirrored = await self.mirror.get(key, None)
        except PersistenceError as exc:
            log.warning("mirror read failed", extra={"key": key, "error": exc.message})
            return default
        if mirrored is not None:
            # Warm the fast path for subsequent reads.
            try:
                await self.primary.set(key, mirrored)
            except PersistenceError:
                pass
            return mirrored
        return default

    async def set(self, key: str, value: Any) -> None:
        await self.primary.set(key, value)
        if not self._should_mirror(key):
            return
        try:
            await self.mirror.set(key, value)
        except PersistenceError as exc:
            log.warning("mirror write failed", extra={"key": key, "error": exc.message})

    async def delete(self, key: str) -> None:
        await self.primary.delete(key)
        if not self._should_mirror(key):
            return
        try:
            await self.mirror.delete(key)
        except PersistenceError as exc:
            log.warning("mirror delete failed", extra={"key": key, "error": exc.message})

    async def keys(self) -> list[str]:
        return await self.primary.keys()

    async def healthy(self) -> tuple[bool, str]:
        ok_p, msg_p = await self.primary.healthy()
        try:
            ok_m, msg_m = await self.mirror.healthy()
        except PersistenceError as exc:
            ok_m, msg_m = False, exc.message
        return (ok_p and ok_m), f"primary={msg_p} mirror={msg_m}"

    async def aclose(self) -> None:
        await self.primary.aclose()
        await self.mirror.aclose()


def build_store(config: PersistenceConfig, state_root: Path) -> StateStore:
    """Select the persistence backend named by configuration."""
    disk = DiskStore(state_root)
    backend = (config.backend or "disk").lower()

    if backend == "disk" or not config.gist_ready:
        if backend in ("gist", "chained") and not config.gist_ready:
            log.warning(
                "persistence backend %s requested but GIST_API_KEY is missing; using disk only",
                backend,
            )
        return disk

    gist = GistStore(
        config.gist_id,  # may be None — the store provisions its own gist
        config.gist_api_key or "",
        config.filename_prefix,
    )

    if backend == "gist":
        return gist

    # Only durable-critical documents are mirrored, to stay well inside API quota.
    return ChainedStore(
        disk,
        gist,
        mirror_keys=(
            "agent.memory",
            "agent.tasks",
            "agent.profile",
            "agent.skills_index",
            "deployment.state",
        ),
    )
