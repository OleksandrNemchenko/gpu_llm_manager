"""Веб-оболонка: сторінка і її JSON API. Ті самі виклики GpuManager, що й у MCP.

Тексти відмов і попереджень — українською, бо їх читає людина на сторінці."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio.to_thread
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from . import pdfconvert
from .build_info import build_info
from .core import GpuManager
from .messages import UK, ManagerError, text

_STATIC = Path(__file__).parent / "static"
_JOURNAL_DEFAULT = 30
_HISTORY_POINTS_DEFAULT = 120


class HostGuard:
    """Автентифікації немає свідомо (довірені користувачі в TailScale), тож два дешеві захисти від
    сторонньої сторінки в браузері користувача:
    - перелік дозволених Host: DNS rebinding не дістанеться до API;
    - POST на /api/ і /v1/ лише JSON і лише зі своїм Origin: HTML-форма чи fetch no-cors з чужого сайту не спрацює
      (CSRF). Клієнти OpenAI шлють application/json без Origin — їх це не зачіпає."""

    def __init__(self, app: ASGIApp, host_names: tuple[str, ...], port: int) -> None:
        self.app = app
        self.hosts = {f"{h}:{port}" for h in host_names} | set(host_names)
        self.origins = {f"http://{h}" for h in self.hosts}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
            if headers.get("host") not in self.hosts:
                await JSONResponse({"error": "unknown Host"}, status_code=421)(scope, receive, send)
                return
            if scope["method"] == "POST" and scope["path"].startswith(("/api/", "/v1/")):
                if not headers.get("content-type", "").startswith("application/json"):
                    await JSONResponse({"error": "JSON only"}, status_code=415)(scope, receive, send)
                    return
                origin = headers.get("origin")
                if origin is not None and origin not in self.origins:
                    await JSONResponse({"error": "foreign Origin"}, status_code=403)(scope, receive, send)
                    return
        await self.app(scope, receive, send)


async def json_body(request: Request) -> dict[str, Any]:
    """Тіло POST як об'єкт JSON; інакше ManagerError bad_request."""
    try:
        data = await request.json()
    except json.JSONDecodeError as exc:
        raise ManagerError("bad_request", detail=f"invalid JSON ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise ManagerError("bad_request", detail="JSON body must be an object")
    return data


def guarded(fn: Any) -> Any:
    """Обгортка маршруту: результат — JSON; відмова — 400 {"code", "error" українською}."""

    async def endpoint(request: Request) -> Response:
        try:
            return JSONResponse(await fn(request))
        except ManagerError as exc:
            return JSONResponse({"code": exc.code, "error": exc.text(UK)}, status_code=400)
        except (KeyError, TypeError, ValueError) as exc:
            err = ManagerError("bad_request", detail=repr(exc))
            return JSONResponse({"code": err.code, "error": err.text(UK)}, status_code=400)

    return endpoint


def web_routes(manager: GpuManager, public_host: str) -> list[Route]:
    """Маршрути сторінки й API. Відмова — 400 {"code", "error"} з текстом українською.

    public_host — адреса:порт для інструкції MCP на сторінці, коли її відкрили через localhost."""

    async def index(request: Request) -> Response:
        return FileResponse(_STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    async def overview(request: Request) -> Any:
        return {**manager.overview(), "public_host": public_host}

    async def journal(request: Request) -> Any:
        return manager.journal(int(request.query_params.get("limit", _JOURNAL_DEFAULT)))

    async def history(request: Request) -> Any:
        q = request.query_params
        gpu = int(q["gpu"]) if "gpu" in q else None
        data = manager.history(gpu, float(q.get("minutes", 60)), int(q.get("points", _HISTORY_POINTS_DEFAULT)))
        return {str(k): v for k, v in data.items()}

    async def reserve(request: Request) -> Any:
        d = await json_body(request)
        hours = d.get("hours")
        # gpu передається як є: перевіряє ядро (int() зробив би з true карту 1)
        result = manager.reserve(d["gpu"], str(d["user"]), str(d.get("purpose") or ""),
                                 float(hours) if hours not in (None, "") else None)
        return {**result, "warnings": [text(w["code"], UK, **w["params"]) for w in result["warnings"]]}

    async def release(request: Request) -> Any:
        d = await json_body(request)
        return manager.release(d["gpu"], str(d["user"]), d.get("force", False) is True)

    build = build_info()  # один раз при старті

    async def version(request: Request) -> Any:
        return build

    async def convert_pdf(request: Request) -> Any:
        """PDF з чату -> текст або картинки сторінок (mode: text | images); важке — у потоці."""
        d = await json_body(request)
        raw = pdfconvert.decode(str(d["data"]))
        fn = pdfconvert.to_images if d.get("mode") == "images" else pdfconvert.to_text
        return await anyio.to_thread.run_sync(fn, raw)

    async def healthz(request: Request) -> Response:
        # "\n" у кінці — щоб `curl` у терміналі не склеював відповідь із запрошенням оболонки (прохання користувача)
        return Response('{"ok":true}\n', media_type="application/json")

    return [
        Route("/", index),
        Route("/healthz", healthz),
        Route("/api/overview", guarded(overview)),
        Route("/api/version", guarded(version)),
        Route("/api/journal", guarded(journal)),
        Route("/api/history", guarded(history)),
        Route("/api/reserve", guarded(reserve), methods=["POST"]),
        Route("/api/release", guarded(release), methods=["POST"]),
        Route("/api/convert/pdf", guarded(convert_pdf), methods=["POST"]),
    ]
