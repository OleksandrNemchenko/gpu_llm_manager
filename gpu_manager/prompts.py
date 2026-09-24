"""Іменовані системні промпти (фаза 4): агент надсилає назву замість тексту — токени агента не витрачаються на
повтор промпту (домовленість з обговорення, табл. 7.3). Використання: модель `назва_моделі@назва_промпту`."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from .core import require_user
from .journal import Journal
from .messages import ManagerError
from .names import check_name
from .store import StateStore

SECTION = "prompts"
# Межа тексту промпту, символів: іменований промпт — інструкція, а не документ.
_MAX_TEXT = 20000


class PromptStore:
    def __init__(self, users: tuple[str, ...], store: StateStore, journal: Journal,
                 clock: Callable[[], float] = time.time) -> None:
        self._users = users
        self._store = store
        self._journal = journal
        self._clock = clock
        self._lock = threading.Lock()
        self._items: dict[str, dict[str, Any]] = store.read(SECTION) or {}

    def save(self, name: str, text: str, user: str) -> dict[str, Any]:
        """Створює або замінює промпт name."""
        require_user(self._users, user)
        check_name(name)
        if not text.strip() or len(text) > _MAX_TEXT:
            raise ManagerError("bad_prompt", max=_MAX_TEXT)
        item = {"name": name, "text": text.strip(), "user": user, "updated": self._clock()}
        with self._lock:
            self._items = {**self._items, name: item}
            self._store.write(SECTION, self._items)
        self._journal.record(user, "prompt_save", name=name)
        return item

    def delete(self, name: str, user: str) -> dict[str, Any]:
        require_user(self._users, user)
        with self._lock:
            if name not in self._items:
                raise ManagerError("prompt_not_found", name=name)
            self._items = {k: v for k, v in self._items.items() if k != name}
            self._store.write(SECTION, self._items)
        self._journal.record(user, "prompt_delete", name=name)
        return {"deleted": True, "name": name}

    def get(self, name: str) -> str:
        with self._lock:
            item = self._items.get(name)
        if item is None:
            raise ManagerError("prompt_not_found", name=name)
        return str(item["text"])

    def list(self) -> list[dict[str, Any]]:
        """Назви з першими 80 символами тексту: повний текст агентові не потрібен, щоб вибрати промпт."""
        with self._lock:
            items = sorted(self._items.values(), key=lambda x: x["name"])
        return [{"name": i["name"], "preview": i["text"][:80], "user": i["user"], "updated": i["updated"]} for i in items]
