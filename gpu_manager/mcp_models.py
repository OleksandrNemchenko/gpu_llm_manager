"""MCP-інструменти моделей HuggingFace (фаза 2): пошук, опис з оцінкою «чи влізе», завантаження, список, видалення."""

from __future__ import annotations

from typing import Any
from zoneinfo import ZoneInfo

from mcp.server.mcpserver import MCPServer

from .mcp_common import DESTRUCTIVE, READ, READ_EXTERNAL, WRITE, call, clock_text, dump
from .models import ModelStore


def register_model_tools(mcp: MCPServer, models: ModelStore, tz: ZoneInfo) -> None:
    """Додає інструменти hf_* і model_* до вже створеного MCP-сервера."""
    when = clock_text(tz)

    def download_view(d: dict[str, Any]) -> dict[str, Any]:
        return {"repo": d["repo"], "status": d["status"], "percent": d["percent"],
                "gib": [d["done_gib"], d["total_gib"]], "user": d["user"], "started": when(d["started"]),
                **({"error": f"{d['error_code']}: {d['error']}"} if d["error_code"] else {})}

    @mcp.tool(annotations=READ_EXTERNAL)
    def hf_search(query: str, limit: int = 10) -> str:
        """Search HuggingFace models by name, most downloaded first: repo, downloads, gated, params, task."""
        found = call(models.search, query, limit)
        return dump([{"repo": m["repo"], "downloads": m["downloads"], "gated": m["gated"], "params": m["params"],
                      "task": m["task"]} for m in found])

    @mcp.tool(annotations=READ_EXTERNAL)
    def hf_model(repo: str, revision: str | None = None, fraction: float | None = None) -> str:
        """Before downloading: size of the files vLLM needs, gated or not, whether it is already local, and a ROUGH
        fit estimate per card size (MiB): usable GiB and max context tokens on one card using `fraction` of its
        memory (default from config, ~0.9; use e.g. 0.45 to share a card between two models). The first real
        start decides; this only filters out clearly oversized models."""
        return dump(call(models.info, repo, revision, fraction))

    @mcp.tool(annotations=WRITE)
    def model_download(repo: str, user: str, revision: str | None = None) -> str:
        """Queue a model download into the server's HuggingFace cache (resumes after interruptions). Returns the
        download state; poll with model_downloads. Refused when the disk would drop below the configured reserve.
        Read gpu_guide first if you have not in this session."""
        return dump(download_view(call(models.download, repo, user, revision)))

    @mcp.tool(annotations=READ)
    def model_downloads() -> str:
        """All downloads, newest first: status (queued, downloading, done, failed, cancelled), percent, [done, total] GiB."""
        return dump([download_view(d) for d in models.downloads()])

    @mcp.tool(annotations=WRITE)
    def model_download_cancel(repo: str, user: str) -> str:
        """Stop a download. Partial files stay; model_download later resumes them."""
        return dump(download_view(call(models.cancel, repo, user)))

    @mcp.tool(annotations=READ)
    def models_local() -> str:
        """Models on the server disk: repo, size GiB, state (ready, downloading, partial)."""
        return dump([{"repo": m["repo"], "gib": m["size_gib"], "state": m["state"]} for m in models.local()])

    @mcp.tool(annotations=DESTRUCTIVE)
    def model_delete(repo: str, user: str) -> str:
        """Delete a model from the server disk (irreversible; re-download needed). Ask the human first unless they
        asked for it. An active download must be cancelled first."""
        return dump(call(models.delete, repo, user))
