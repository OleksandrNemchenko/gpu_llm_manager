"""Спільне для MCP-модулів: стислий JSON (економія токенів агента) і переклад ManagerError у ToolError."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import WithJsonSchema

from .messages import EN, ManagerError

# Номер карти: агент бачить у схемі integer, але значення доходить до ядра без перетворень. Інакше нестрогий
# pydantic зробив би з true карту 1 ще до перевірки (знайдено сліпими тестами), а StrictInt дав би відмову не
# з кодом unknown_gpu, як на HTTP-вході.
GpuIndex = Annotated[Any, WithJsonSchema({"type": "integer", "description": "GPU index"})]

READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
# Видалення з диска — незворотне, агент має бачити це в анотації.
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False)
# Запити до HuggingFace виходять у зовнішній світ.
READ_EXTERNAL = ToolAnnotations(read_only_hint=True, open_world_hint=True)


def dump(data: Any) -> str:
    """JSON без пробілів і переносів: кожен символ результату — вхідний токен агента."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Виклик ядра; ManagerError -> ToolError з текстом "<code>: <англійською>"."""
    try:
        return fn(*args, **kwargs)
    except ManagerError as exc:
        raise ToolError(f"{exc.code}: {exc.text(EN)}") from exc


def clock_text(tz: ZoneInfo) -> Callable[[float | None], str | None]:
    """Перетворювач unix-часу в "YYYY-MM-DD HH:MM" у поясі tz."""

    def when(ts: float | None) -> str | None:
        return datetime.fromtimestamp(ts, tz).strftime("%Y-%m-%d %H:%M") if ts is not None else None

    return when
