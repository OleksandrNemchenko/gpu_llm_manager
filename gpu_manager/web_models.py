"""Веб-API моделей HuggingFace (фаза 2) для сторінки. Ті самі виклики ModelStore, що й у MCP; відмови — українською.

Виклики HF і диска — у робочому потоці: у циклі подій вони зупиняли б увесь сервер (сторінку, MCP, потоки /v1),
поки HF відповідає."""

from __future__ import annotations

from typing import Any

import anyio.to_thread
from starlette.requests import Request
from starlette.routing import Route

from .models import ModelStore
from .web import guarded, json_body

_SEARCH_DEFAULT = 10


def model_routes(models: ModelStore) -> list[Route]:
    """GET пошук/опис/список, POST завантажити/скасувати/видалити."""

    async def search(request: Request) -> Any:
        q = request.query_params
        query, limit = q.get("q", ""), int(q.get("limit", _SEARCH_DEFAULT))
        return await anyio.to_thread.run_sync(lambda: models.search(query, limit))

    async def info(request: Request) -> Any:
        q = request.query_params
        fraction = q.get("fraction")
        repo, revision, fr = q["repo"], q.get("revision") or None, float(fraction) if fraction else None
        return await anyio.to_thread.run_sync(lambda: models.info(repo, revision, fr))

    async def local(request: Request) -> Any:
        return await anyio.to_thread.run_sync(lambda: {"local": models.local(), "downloads": models.downloads()})

    async def download(request: Request) -> Any:
        d = await json_body(request)
        repo, user, revision = str(d["repo"]), str(d["user"]), d.get("revision") or None
        return await anyio.to_thread.run_sync(lambda: models.download(repo, user, revision))

    async def cancel(request: Request) -> Any:
        d = await json_body(request)
        return await anyio.to_thread.run_sync(models.cancel, str(d["repo"]), str(d["user"]))

    async def delete(request: Request) -> Any:
        d = await json_body(request)
        return await anyio.to_thread.run_sync(models.delete, str(d["repo"]), str(d["user"]))

    return [
        Route("/api/models/search", guarded(search)),
        Route("/api/models/info", guarded(info)),
        Route("/api/models", guarded(local)),
        Route("/api/models/download", guarded(download), methods=["POST"]),
        Route("/api/models/cancel", guarded(cancel), methods=["POST"]),
        Route("/api/models/delete", guarded(delete), methods=["POST"]),
    ]
