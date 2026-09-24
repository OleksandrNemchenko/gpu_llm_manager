"""MCP-інструменти запуску моделей на vLLM (фаза 3): старт, стоп, стан, логи для розбору помилок агентом."""

from __future__ import annotations

from typing import Any
from zoneinfo import ZoneInfo

from mcp.server.mcpserver import MCPServer

from .mcp_common import READ, WRITE, GpuIndex, call, clock_text, dump
from .messages import EN, text
from .runner import ModelRunner


def register_server_tools(mcp: MCPServer, runner: ModelRunner, tz: ZoneInfo) -> None:
    """Додає model_start, model_move, model_stop, models_running, model_logs."""
    when = clock_text(tz)

    def view(s: dict[str, Any]) -> dict[str, Any]:
        return {"name": s["name"], "repo": s["repo"], "gpu": s["gpu"], "port": s["port"], "status": s["status"],
                "fraction": s["fraction"], "max_model_len": s["max_model_len"], "user": s["user"],
                "started": when(s["started"]), **({"usage": s["usage"]} if s["usage"] else {}),
                **({"auto_changes": s["auto_changes"]} if s["auto_changes"] else {}),
                **({"hint": f"{s['error_code']}: {text(s['error_code'], EN, **s['error_params'])}"} if s["error_code"] else {})}

    @mcp.tool(annotations=WRITE)
    def model_start(repo: str, gpu: GpuIndex, user: str, fraction: float | None = None,
                    max_model_len: int | None = None, extra_args: list[str] | None = None,
                    name: str | None = None, port: int | None = None) -> str:
        """Start a downloaded model on one GPU with vLLM. Port: lowest free, assigned by the server. Defaults aim at
        maximum memory and context: `fraction` = the model's last good value, else all free GPU memory (the server
        steps down 0.95 → 0.92 → 0.9 if vLLM needs room); `max_model_len` = last good value, else the model's own
        maximum (if the KV cache cannot hold it, the server restarts with vLLM's exact maximum). `extra_args`: extra `vllm serve` flags, e.g. ["--trust-remote-code"]. Starting takes
        minutes: poll models_running until status is running; on failed read model_logs and its hint, adjust, retry.
        Clients reach the model through the manager, not the port. Read gpu_guide first if you have not."""
        return dump(view(call(runner.start, repo, gpu, user, fraction, max_model_len, extra_args, name, port)))

    @mcp.tool(annotations=WRITE)
    def model_move(name: str, user: str, gpu: GpuIndex = None, port: int | None = None) -> str:
        """Move a running model to another GPU and/or port: it is stopped and started again under the same name
        (clients keep using the name). On failure it goes back to where it was. Poll models_running afterwards."""
        return dump(view(call(runner.move, name, user, gpu, port)))

    @mcp.tool(annotations=WRITE)
    def model_stop(name: str, user: str) -> str:
        """Stop a running model and free its GPU memory. It will not come back after a restart."""
        return dump(view(call(runner.stop, name, user)))

    @mcp.tool(annotations=READ)
    def models_running() -> str:
        """Models started by the manager: name, repo, gpu, port, status (starting, running, failed), usage
        (requests running / waiting, kv_usage 0..1), hint on failure."""
        return dump([view(s) for s in runner.servers()])

    @mcp.tool(annotations=READ)
    def model_logs(name: str, lines: int = 60, errors_only: bool = False) -> str:
        """Tail of the model's vLLM log from its last start (errors_only: only error lines) and the failure hint.
        Use it to find why a start failed, then change fraction / max_model_len / extra_args and start again."""
        out = call(runner.logs, name, lines, errors_only)
        hint = f"{out['hint']}: {text(out['hint'], EN, **out['hint_params'])}" if out["hint"] else None
        return dump({"name": out["name"], "status": out["status"], "hint": hint, "lines": out["lines"]})
