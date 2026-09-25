"""Filesystem tools. Every path goes through the PathGuard."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

from ..errors import PermissionDenied, ToolError
from ..registry import Tool, ToolContext, prop, schema

log = logging.getLogger(__name__)

MAX_READ_BYTES = 200_000
LIST_LIMIT = 400
SEARCH_HITS = 80
#: Directories never worth walking.
_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", "dist", "build", ".tox", ".idea", ".vscode",
}


class ListDirTool(Tool):
    name = "list_dir"
    category = "filesystem"
    description = (
        "List the contents of a directory inside the agent workspace. Returns names, "
        "sizes and whether each entry is a directory. Use this before reading files so "
        "you know what actually exists."
    )

    @property
    def parameters(self) -> dict[str, Any]:
        return schema(
            {
                "path": prop("string", "Directory path relative to the workspace root. Use '.' for the root.", default="."),
                "recursive": prop("boolean", "Walk subdirectories too.", default=False),
                "max_depth": prop("integer", "Depth limit when recursive.", minimum=1, maximum=6, default=3),
            },
            description="List a directory in the workspace.",
        )

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> dict[str, Any]:
        root = ctx.path_guard.resolve(args.get("path") or ".")
        if not root.exists():
            return {"path": ctx.path_guard.relative(root), "exists": False, "entries": []}
        if not root.is_dir():
            raise ToolError(f"{ctx.path_guard.relative(root)} is a file, not a directory")

        entries: list[dict[str, Any]] = []
        recursive = bool(args.get("recursive"))
        max_depth = int(args.get("max_depth", 3))

        def walk(directory: Path, depth: int) -> None:
            if len(entries) >= LIST_LIMIT or depth > max_depth:
                return
            try:
                children = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
            except PermissionError:
                return
            for child in children:
                if len(entries) >= LIST_LIMIT:
                    return
                if child.name in _SKIP_DIRS:
                    continue
                try:
                    stat = child.stat()
                    is_dir = child.is_dir()
                except OSError:
                    continue
                entries.append(
                    {
                        "name": ctx.path_guard.relative(child),
                        "type": "dir" if is_dir else "file",
                        "size": 0 if is_dir else stat.st_size,
                    }
                )
                if is_dir and recursive:
                    walk(child, depth + 1)

        walk(root, 1)
        return {
            "path": ctx.path_guard.relative(root),
            "exists": True,
            "count": len(entries),
            "truncated": len(entries) >= LIST_LIMIT,
            "entries": entries,
        }


class ReadFileTool(Tool):
    name = "read_file"
    category = "filesystem"
    description = (
        "Read a text file from the workspace. Returns the content with 1-based line numbers, "
        "and supports an offset/limit window so you can inspect one region of a large file "
        "instead of loading all of it."
    )

    @property
    def parameters(self) -> dict[str, Any]:
        return schema(
            {
                "path": prop("string", "File path relative to the workspace root."),
                "start_line": prop("integer", "First line to return (1-based).", minimum=1),
                "max_lines": prop("integer", "Maximum lines to return.", minimum=1, maximum=2000, default=400),
            },
            required=["path"],
            description="Read a workspace file.",
        )

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> dict[str, Any]:
        path = ctx.path_guard.resolve(args["path"])
        if not path.exists():
            raise ToolError(f"file not found: {args['path']}")
        if path.is_dir():
            raise ToolError(f"{args['path']} is a directory; use list_dir")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ToolError(f"cannot stat {args['path']}: {exc}") from exc

        raw = path.read_bytes()[:MAX_READ_BYTES]
        if b"\x00" in raw[:4096]:
            raise ToolError(
                f"{args['path']} looks binary ({size} bytes). Use run_command with an "
                f"appropriate tool (file, head, xxd) instead."
            )
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()

        start = int(args.get("start_line", 1) or 1)
        count = int(args.get("max_lines", 400) or 400)
        window = lines[start - 1 : start - 1 + count]
        numbered = "\n".join(f"{start + i:>5} | {line}" for i, line in enumerate(window))

        return {
            "path": ctx.path_guard.relative(path),
            "total_lines": len(lines),
            "returned_lines": len(window),
            "start_line": start,
            "truncated": (start - 1 + count) < len(lines),
            "bytes": size,
            "content": numbered,
        }


class WriteFileTool(Tool):
    name = "write_file"
    category = "filesystem"
    description = (
        "Create or overwrite a file with the given text. Parent directories are created. "
        "Overwriting an existing file is allowed and is reported."
    )

    @property
    def parameters(self) -> dict[str, Any]:
        return schema(
            {
                "path": prop("string", "File path relative to the workspace root."),
                "content": prop("string", "Full file content to write."),
            },
            required=["path", "content"],
            description="Write a file.",
        )

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> dict[str, Any]:
        path = ctx.path_guard.resolve(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.exists()
        content = args["content"]
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"write failed: {exc}") from exc
        written = path.stat().st_size
        # Verify: read the byte count back rather than trusting the write call.
        return {
            "path": ctx.path_guard.relative(path),
            "bytes_written": written,
            "overwrote_existing": existed,
            "verified": path.stat().st_size == written,
        }


class EditFileTool(Tool):
    name = "edit_file"
    category = "filesystem"
    description = (
        "Replace an exact string in a file (the deterministic, cheap edit). The search text "
        "must appear exactly once unless replace_all is true. Use this instead of rewriting "
        "a whole file for a one-line change."
    )

    @property
    def parameters(self) -> dict[str, Any]:
        return schema(
            {
                "path": prop("string", "File path relative to the workspace root."),
                "old_text": prop("string", "Exact text to replace. Must match byte-for-byte."),
                "new_text": prop("string", "Replacement text.", default=""),
                "replace_all": prop("boolean", "Replace every occurrence instead of requiring one.", default=False),
            },
            required=["path", "old_text"],
            description="Exact-string edit of a file.",
        )

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> dict[str, Any]:
        path = ctx.path_guard.resolve(args["path"])
        if not path.exists():
            raise ToolError(f"file not found: {args['path']}")
        text = path.read_text(encoding="utf-8", errors="replace")
        old = args["old_text"]
        new = args.get("new_text", "")

        occurrences = text.count(old)
        if occurrences == 0:
            raise ToolError(
                f"old_text not found in {args['path']}. Read the file first and copy the "
                f"exact current text."
            )
        if occurrences > 1 and not args.get("replace_all"):
            raise ToolError(
                f"old_text appears {occurrences} times in {args['path']}; add surrounding "
                f"context to make it unique, or set replace_all=true."
            )
        updated = text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8")
        return {
            "path": ctx.path_guard.relative(path),
            "replacements": occurrences if args.get("replace_all") else 1,
            "bytes": len(updated.encode("utf-8")),
            "changed": updated != text,
        }


class SearchFilesTool(Tool):
    name = "search_files"
    category = "filesystem"
    description = (
        "Search file contents across the workspace with a regular expression. Returns "
        "matching lines with file and line numbers, plus optional surrounding context. "
        "Use this to locate code before reading it."
    )

    @property
    def parameters(self) -> dict[str, Any]:
        return schema(
            {
                "pattern": prop("string", "Regular expression to search for."),
                "path": prop("string", "Directory or file to search, relative to the workspace root.", default="."),
                "glob": prop("string", "Filename filter, e.g. '*.py'.", default="*"),
                "context_lines": prop("integer", "Lines of context around each hit.", minimum=0, maximum=6, default=0),
                "max_results": prop("integer", "Maximum hits to return.", minimum=1, maximum=200, default=SEARCH_HITS),
            },
            required=["pattern"],
            description="Regex search across workspace files.",
        )

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> dict[str, Any]:
        root = ctx.path_guard.resolve(args.get("path") or ".")
        try:
            regex = re.compile(args["pattern"])
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from exc
        glob = args.get("glob") or "*"
        context = int(args.get("context_lines", 0) or 0)
        limit = int(args.get("max_results", SEARCH_HITS) or SEARCH_HITS)

        files: list[Path] = [root] if root.is_file() else list(root.rglob(glob))
        hits: list[dict[str, Any]] = []
        scanned = 0
        skipped: list[str] = []

        for candidate in files:
            if len(hits) >= limit:
                break
            if not candidate.is_file() or any(part in _SKIP_DIRS for part in candidate.parts):
                continue
            try:
                if candidate.stat().st_size > 2_000_000:
                    skipped.append(ctx.path_guard.relative(candidate))
                    continue
                text = candidate.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            scanned += 1
            lines = text.splitlines()
            for index, line in enumerate(lines):
                if len(hits) >= limit:
                    break
                if regex.search(line):
                    record: dict[str, Any] = {
                        "file": ctx.path_guard.relative(candidate),
                        "line": index + 1,
                        "text": line[:400],
                    }
                    if context:
                        lo = max(0, index - context)
                        hi = min(len(lines), index + context + 1)
                        record["context"] = "\n".join(lines[lo:hi])[:1200]
                    hits.append(record)

        return {
            "pattern": args["pattern"],
            "files_scanned": scanned,
            "hit_count": len(hits),
            "truncated": len(hits) >= limit,
            "skipped_large": skipped[:10],
            "hits": hits,
        }
