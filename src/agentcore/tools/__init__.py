"""Tools — the agent's actual capabilities.

Grouped by domain so the registry stays legible:

    fs/          read, write, edit, list, search files
    exec/        run shell commands, run Python
    web/         search the web, fetch a URL, download a file
    state/       working memory, long-term memory, task state, artifacts
    meta/        load a skill, self-report the environment

Every tool is one file, declares a JSON schema, and returns either a value or
a structured error. Adding a capability means adding a class and registering
it — the loop, prompts and interfaces never change.
"""

from __future__ import annotations

from .fs import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from .exec import RunCommandTool, RunPythonTool
from .web import DownloadFileTool, FetchUrlTool, WebSearchTool
from .state import (
    ArtifactWriteTool,
    LongTermMemoryTool,
    TaskStateTool,
    WorkingMemoryTool,
)
from .meta import LoadSkillTool, SelfReportTool


def build_tools():
    """Instantiate the default tool set (fresh objects per registry)."""
    return [
        # filesystem
        ListDirTool(),
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        SearchFilesTool(),
        # execution
        RunCommandTool(),
        RunPythonTool(),
        # web
        WebSearchTool(),
        FetchUrlTool(),
        DownloadFileTool(),
        # state
        WorkingMemoryTool(),
        LongTermMemoryTool(),
        TaskStateTool(),
        ArtifactWriteTool(),
        # meta
        LoadSkillTool(),
        SelfReportTool(),
    ]


__all__ = ["build_tools"]
