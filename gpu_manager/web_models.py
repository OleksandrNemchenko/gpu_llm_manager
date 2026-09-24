"""Веб-API моделей HuggingFace (фаза 2) для сторінки. Ті самі виклики ModelStore, що й у MCP; відмови — українською."""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from starlette.routing import Route

from .models import ModelStore
from .web import guarded, json_body

_SEARCH_DEFAULT = 10


def model_routes(models: ModelStore) -> list[Route]:
    """GET пошук/опис/список, POST завантажити/скасувати/видалити."""

    async def search(request: Request) -> Any:
        q = request.query_params
        return models.search(q.get("q", ""), int(q.get("limit", _SEARCH_DEFAULT)))

    async def info(request: Request) -> Any:
        q = request.query_params
        fraction = q.get("fraction")
        return models.info(q["repo"], q.get("revision") or None, float(fraction) if fraction else None)

    async def local(request: Request) -> Any:
        return {"local": models.local(), "downloads": models.downloads()}

    async def download(request: Request) -> Any:
        d = await json_body(request)
        return models.download(str(d["repo"]), str(d["user"]), d.get("revision") or None)

    async def cancel(request: Request) -> Any:
        d = await json_body(request)
        return models.cancel(str(d["repo"]), str(d["user"]))

    async def delete(request: Request) -> Any:
        d = await json_body(request)
        return models.delete(str(d["repo"]), str(d["user"]))

    return [
        Route("/api/models/search", guarded(search)),
        Route("/api/models/info", guarded(info)),
        Route("/api/models", guarded(local)),
        Route("/api/models/download", guarded(download), methods=["POST"]),
        Route("/api/models/cancel", guarded(cancel), methods=["POST"]),
        Route("/api/models/delete", guarded(delete), methods=["POST"]),
    ]
