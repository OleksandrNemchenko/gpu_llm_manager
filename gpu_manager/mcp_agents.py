"""MCP-інструменти для агентів (фаза 4): іменовані системні промпти і питання до локальної моделі.

llm_ask повертає лише текст відповіді: агентові не потрібні службові поля OpenAI, а кожен символ — його токени."""

from __future__ import annotations

from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer

from .gateway import resolve
from .mcp_common import DESTRUCTIVE, READ, WRITE, call, dump
from .messages import ManagerError
from .prompts import PromptStore
from .runner import ModelRunner

# Скільки чекати відповіді локальної моделі, секунд: довгі відповіді генеруються хвилинами.
_ASK_TIMEOUT_S = 600


def register_agent_tools(mcp: MCPServer, runner: ModelRunner, prompts: PromptStore) -> None:
    """Додає prompt_save, prompts_list, prompt_delete, llm_ask."""

    @mcp.tool(annotations=WRITE)
    def prompt_save(name: str, text: str, user: str) -> str:
        """Save a named system prompt once; later use it as model "<model>@<name>" in /v1 or system_prompt=<name>
        in llm_ask, instead of resending the text. Replaces an existing prompt with the same name."""
        item = call(prompts.save, name, text, user)
        return dump({"saved": True, "name": item["name"]})

    @mcp.tool(annotations=READ)
    def prompts_list() -> str:
        """Named system prompts: name and the first 80 characters."""
        return dump([{"name": p["name"], "preview": p["preview"], "user": p["user"]} for p in prompts.list()])

    @mcp.tool(annotations=DESTRUCTIVE)
    def prompt_delete(name: str, user: str) -> str:
        """Delete a named system prompt."""
        return dump(call(prompts.delete, name, user))

    @mcp.tool(annotations=WRITE)
    async def llm_ask(model: str, prompt: str, system_prompt: str | None = None, system: str | None = None,
                max_tokens: int = 1024, temperature: float = 0.2) -> str:
        """Ask a running local model (name from models_running) and get only the answer text. system_prompt = name
        of a saved prompt (cheap); system = literal system text (costs your tokens each time). Other agents and
        programs can use the same models through the OpenAI-compatible /v1 of this server."""
        messages: list[dict[str, Any]] = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        body = {"model": f"{model}@{system_prompt}" if system_prompt else model, "messages": messages,
                "max_tokens": max_tokens, "temperature": temperature}
        port, body = call(resolve, runner, prompts, body)
        try:
            async with httpx.AsyncClient(timeout=_ASK_TIMEOUT_S) as http:
                r = await http.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            return call(_raise, ManagerError("bad_request", detail=f"model unreachable: {exc}"))
        if r.status_code != 200:
            return call(_raise, ManagerError("bad_request", detail=r.text[:300]))
        choice = (r.json().get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content")
        if content is None:  # напр. модель витратила max_tokens на міркування
            return call(_raise, ManagerError("empty_answer", reason=choice.get("finish_reason") or "unknown"))
        return str(content)


def _raise(exc: ManagerError) -> str:
    raise exc
