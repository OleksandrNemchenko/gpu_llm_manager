"""Правило назви моделі й промпту — одне для всіх (SPEC-GPU-003 §2.1, §3)."""

from __future__ import annotations

import re

from .messages import ManagerError

# fullmatch, а не match з $: `$` пропускає кінцевий \n, і назва з переносом потрапила б у шлях логу.
_NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,47}")


def check_name(name: str) -> str:
    """Повертає name або ManagerError bad_name."""
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ManagerError("bad_name", name=name)
    return name


def slug(text: str) -> str:
    """Назва з довільного тексту (остання частина repo): малі літери, заборонене -> '-'."""
    return re.sub(r"[^a-z0-9._-]", "-", text.lower())[:40].strip("-._") or "model"  # перший символ — лише a-z0-9
