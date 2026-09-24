"""Веб-API запуску моделей (фаза 3). Ті самі виклики ModelRunner, що й у MCP; відмови й підказки — українською."""

from __future__ import annotations

from typing import Any

import anyio.to_thread
from starlette.requests import Request
from starlette.routing import Route

from .messages import UK, text
from .runner import ModelRunner
from .web import guarded, json_body

_LOG_LINES_DEFAULT = 80


def server_routes(runner: ModelRunner) -> list[Route]:
    def with_hint(s: dict[str, Any]) -> dict[str, Any]:
        return {**s, "hint": text(s["error_code"], UK, **s["error_params"]) if s["error_code"] else None}

    async def servers(request: Request) -> Any:
        return [with_hint(s) for s in runner.servers()]

    async def start(request: Request) -> Any:
        d = await json_body(request)
        fraction = d.get("fraction")
        ctx = d.get("max_model_len")
        args = (str(d["repo"]), d["gpu"], str(d["user"]), float(fraction) if fraction not in (None, "") else None,
                int(ctx) if ctx not in (None, "") else None, d.get("extra_args"), d.get("name") or None,
                d.get("port"))
        return with_hint(await anyio.to_thread.run_sync(lambda: runner.start(*args)))

    async def stop(request: Request) -> Any:
        d = await json_body(request)
        return with_hint(await anyio.to_thread.run_sync(runner.stop, str(d["name"]), str(d["user"])))

    async def move(request: Request) -> Any:
        d = await json_body(request)
        return with_hint(await anyio.to_thread.run_sync(
            lambda: runner.move(str(d["name"]), str(d["user"]), d.get("gpu"), d.get("port"))))

    async def ports(request: Request) -> Any:
        return await anyio.to_thread.run_sync(runner.free_ports)

    async def logs(request: Request) -> Any:
        q = request.query_params
        out = await anyio.to_thread.run_sync(runner.logs, q["name"], int(q.get("lines", _LOG_LINES_DEFAULT)),
                                             q.get("errors") == "1")
        return {**out, "hint": text(out["hint"], UK, **out["hint_params"]) if out["hint"] else None}

    return [
        Route("/api/servers", guarded(servers)),
        Route("/api/servers/start", guarded(start), methods=["POST"]),
        Route("/api/servers/stop", guarded(stop), methods=["POST"]),
        Route("/api/servers/logs", guarded(logs)),
        Route("/api/servers/move", guarded(move), methods=["POST"]),
        Route("/api/servers/ports", guarded(ports)),
    ]
