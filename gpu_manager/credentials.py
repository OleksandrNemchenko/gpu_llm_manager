"""Секрети з secrets.json (права 600). Файл читає лише програма; у логи й відповіді значення не потрапляють."""

from __future__ import annotations

import json
from pathlib import Path

# Заглушка з secrets.json до того, як людина вписала справжній токен.
_PLACEHOLDER_PREFIX = "PASTE_"


def hf_token(path: Path) -> str | None:
    """Токен HuggingFace або None: файлу немає, він зіпсований, або там досі заглушка."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))["hf_token"]["value"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(value, str) or not value.strip() or value.startswith(_PLACEHOLDER_PREFIX):
        return None
    return value.strip()
