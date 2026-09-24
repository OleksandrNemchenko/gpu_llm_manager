"""MCP-оболонка: тонкі інструменти над GpuManager для агентів (Claude Code, Codex).

Відповіді — стислий JSON без відступів, бо кожен символ результату — вхідні токени агента (прохання
користувача економити токени). Тексти — англійською: їх бачить термінал (рішення 8.3)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from zoneinfo import ZoneInfo

from mcp.server.mcpserver import MCPServer

from .core import GpuManager
from .mcp_agents import register_agent_tools
from .mcp_common import READ, WRITE, GpuIndex, call, clock_text, dump
from .mcp_models import register_model_tools
from .mcp_servers import register_server_tools
from .messages import EN, text
from .models import ModelStore
from .prompts import PromptStore
from .runner import ModelRunner

# Перший рядок — найважливіший: Codex рахує головними перші 512 символів інструкцій.
INSTRUCTIONS = """FIRST, once per session and before any other tool: call gpu_guide and follow it (rules, your `user`, file paths, copy commands).
GPU manager of a multi-GPU server shared by a small trusted team.
Card statuses: free; reserved (someone marked it as in use); reserved_expired (the term passed but the mark
stays until a human releases it); busy (processes run but nobody reserved it); unknown (no telemetry).
Every tool that changes something takes `user`: the person's Ubuntu login, one of the allowed users.
Releasing someone else's reservation needs force=true and is written to the journal; ask the person first.
Times are local (see `tz`)."""




def build_mcp(manager: GpuManager, tz: ZoneInfo, guide: Callable[[], str] | None = None,
              models: ModelStore | None = None, runner: ModelRunner | None = None,
              prompts: PromptStore | None = None) -> MCPServer:
    """Створює MCP-сервер з інструментами gpu_*; tz — пояс для часу у відповідях.

    guide — повертає інструкцію для агентів (gpu_guide); рахується під час виклику, бо стан вхідної
    теки може змінитися без перезапуску. None — інструкції немає.
    models — сховище моделей (фаза 2), runner — запуск моделей (фаза 3); None — їхні інструменти не реєструються."""
    mcp = MCPServer("gpu-manager", instructions=INSTRUCTIONS)
    when = clock_text(tz)

    def reservation(g: dict[str, Any]) -> dict[str, Any] | None:
        r = g["reservation"]
        if r is None:
            return None
        return {"user": r["user"], "purpose": r["purpose"], "since": when(r["since"]), "until": when(r["until"]),
                "expired": r["expired"]}

    def brief(g: dict[str, Any], cmdline: bool = False) -> dict[str, Any]:
        return {
            "gpu": g["index"],
            "status": g["status"],
            "temp_c": g["temperature_c"],
            "util_pct": g["util_pct"],
            "power_w": g["power_w"],
            "mem_mib": [g["memory_used_mib"], g["memory_total_mib"]],
            "reservation": reservation(g),
            "processes": [
                {"pid": p["pid"], "user": p["user"], "name": p["name"], "mib": p["used_mib"],
                 **({"cmd": p["cmdline"]} if cmdline else {})}
                for p in g["processes"]
            ],
            **({"error": g["error"]} if g["error"] else {}),
        }

    @mcp.tool(annotations=READ)
    def gpu_status(gpu: GpuIndex | None = None) -> str:
        """Telemetry and state of every GPU, or of one: temperature, load, power, memory [used, total] MiB,
        reservation and processes with their Unix owners."""
        if gpu is not None:
            return dump({"tz": tz.key, "gpus": [brief(call(manager.gpu, gpu))]})
        return dump({"tz": tz.key, "gpus": [brief(g) for g in manager.overview()["gpus"]]})

    @mcp.tool(annotations=READ)
    def gpu_free() -> str:
        """GPUs nobody reserved and nothing runs on: the answer to "which cards are free?"."""
        return dump([{"gpu": g["index"], "mem_total_mib": g["memory_total_mib"], "temp_c": g["temperature_c"]}
                     for g in manager.free()])

    @mcp.tool(annotations=READ)
    def gpu_who(gpu: GpuIndex) -> str:
        """Who uses one GPU: the reservation (who, purpose, term) and every process with owner and command line."""
        g = call(manager.gpu, gpu)
        return dump({"tz": tz.key, **brief(g, cmdline=True), "users": g["users"]})

    @mcp.tool(annotations=WRITE)
    def gpu_reserve(gpu: GpuIndex, user: str, purpose: str = "", hours: float | None = None) -> str:
        """Mark a GPU as in use by `user` (e.g. simulation, YOLO training). `hours` is an optional term; after it
        the mark shows as expired but is NOT removed. Reserving again as the same user updates purpose and term.
        Read gpu_guide first if you have not in this session."""
        result = call(manager.reserve, gpu, user, purpose, hours)
        warnings = [text(w["code"], EN, **w["params"]) for w in result["warnings"]]
        return dump({"reserved": True, "gpu": brief(result["gpu"]), **({"warnings": warnings} if warnings else {})})

    @mcp.tool(annotations=WRITE)
    def gpu_release(gpu: GpuIndex, user: str, force: bool = False) -> str:
        """Mark a GPU as free again. Someone else's reservation needs force=true (journaled).
        Read gpu_guide first if you have not in this session."""
        result = call(manager.release, gpu, user, force)
        return dump({"released": result["released"], "previous_owner": (result["previous"] or {}).get("user"),
                     "gpu": brief(result["gpu"])})

    @mcp.tool(annotations=READ)
    def gpu_history(gpu: GpuIndex | None = None, minutes: float = 60, points: int = 60) -> str:
        """Telemetry history as column arrays, downsampled to `points` per GPU: t (unix time), temp_c (max),
        util_pct (mean), power_w (mean), mem_mib (max). Per-second data covers the last hour; older comes
        from 30-second aggregates."""
        data = call(manager.history, gpu, minutes, points)
        return dump({"tz": tz.key, "series": {str(k): v for k, v in data.items()}})

    @mcp.tool(annotations=READ)
    def gpu_guide() -> str:
        """Rules for agents: who `user` is, how to reserve cards, where and how to copy files. Read once per session."""
        return guide() if guide is not None else "No guide configured."

    @mcp.tool(annotations=READ)
    def gpu_journal(limit: int = 20) -> str:
        """Recent actions, newest first: who reserved or released which GPU, downloaded or deleted which model."""
        return dump([{**e, "ts": when(e["ts"]), **({"until": when(e["until"])} if e.get("until") else {})}
                     for e in manager.journal(limit)])

    if models is not None:
        register_model_tools(mcp, models, tz)
    if runner is not None:
        register_server_tools(mcp, runner, tz)
        if prompts is not None:
            register_agent_tools(mcp, runner, prompts)
    return mcp
