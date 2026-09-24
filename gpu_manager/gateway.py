"""Шлюз для агентів (фаза 4): OpenAI-сумісний /v1 на порту менеджера. Модель обирається за назвою, порт знає
менеджер (моделі слухають лише 127.0.0.1). `модель@промпт` підставляє іменований системний промпт першим
повідомленням. Потокові відповіді (stream) пересилаються як є."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .messages import EN, ManagerError
from .prompts import PromptStore
from .runner import RUNNING, ModelRunner

# Генерація довгої відповіді може тривати хвилини; з'єднання з моделлю — лише локальне.
_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=60.0, pool=None)
# Без межі з'єднань: vLLM сам тримає сотні одночасних запитів, а типові 100 давали б 502 на 101-му потоці.
_LIMITS = httpx.Limits(max_connections=None, max_keepalive_connections=20)
_PASS_HEADERS = ("content-type",)


def _error(exc: ManagerError, status: int = 400) -> JSONResponse:
    """Помилка у форматі OpenAI: клієнти агентів уміють її показати."""
    return JSONResponse({"error": {"message": exc.text(EN), "type": "invalid_request_error", "code": exc.code}},
                        status_code=status)


def resolve(runner: ModelRunner, prompts: PromptStore, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """(порт, тіло для моделі): `модель@промпт` -> модель + системне повідомлення першим."""
    requested = str(body.get("model", ""))
    name, _, prompt = requested.partition("@")
    srv = next((s for s in runner.servers() if s["name"] == name and s["status"] == RUNNING), None)
    if srv is None:
        raise ManagerError("model_not_running", name=name or "?")
    out = {**body, "model": name}
    if prompt:
        system = {"role": "system", "content": prompts.get(prompt)}
        if "messages" in out:
            out["messages"] = [system, *out["messages"]]
        elif isinstance(out.get("prompt"), str):  # /v1/completions: промпт — префікс тексту
            out["prompt"] = f"{system['content']}\n\n{out['prompt']}"
        elif isinstance(out.get("prompt"), list) and all(isinstance(p, str) for p in out["prompt"]):
            out["prompt"] = [f"{system['content']}\n\n{p}" for p in out["prompt"]]
        elif "prompt" in out:  # токени: текст промпту до них не допишеш
            raise ManagerError("bad_request", detail="a named prompt needs a text prompt, not tokens")
    return int(srv["port"]), out


def gateway_routes(runner: ModelRunner, prompts: PromptStore) -> tuple[list[Route], httpx.AsyncClient]:
    """Маршрути /v1 і клієнт до моделей: його закриває lifespan застосунку."""
    client = httpx.AsyncClient(timeout=_TIMEOUT, limits=_LIMITS)

    async def models(request: Request) -> Response:
        data = [{"id": s["name"], "object": "model", "owned_by": "gpu-manager"}
                for s in runner.servers() if s["status"] == RUNNING]
        return JSONResponse({"object": "list", "data": data})

    async def proxy(request: Request) -> Response:
        try:
            body = json.loads(await request.body())
            if not isinstance(body, dict):
                raise TypeError("body must be an object")
            port, body = resolve(runner, prompts, body)
        except ManagerError as exc:
            return _error(exc, 404 if exc.code in ("model_not_running", "prompt_not_found") else 400)
        except (ValueError, TypeError) as exc:
            return _error(ManagerError("bad_request", detail=str(exc)))
        upstream = client.build_request("POST", f"http://127.0.0.1:{port}{request.url.path}", json=body)
        try:
            r = await client.send(upstream, stream=True)
        except httpx.HTTPError as exc:
            return _error(ManagerError("bad_request", detail=f"model unreachable: {exc}"), 502)
        headers = {k: v for k, v in r.headers.items() if k.lower() in _PASS_HEADERS}

        async def chunks() -> AsyncIterator[bytes]:
            async for part in r.aiter_raw():
                yield part

        return StreamingResponse(chunks(), status_code=r.status_code, headers=headers,
                                 background=BackgroundTask(r.aclose))

    return [
        Route("/v1/models", models),
        Route("/v1/chat/completions", proxy, methods=["POST"]),
        Route("/v1/completions", proxy, methods=["POST"]),
        Route("/v1/embeddings", proxy, methods=["POST"]),
    ], client
