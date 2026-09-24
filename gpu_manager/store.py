"""Файл стану, який пише лише програма (data/state.json): бронювання, черга завантажень, запущені моделі, промпти.

Перезаписується атомарно (тимчасовий файл + rename), тож падіння посеред запису не лишає пів файлу."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


class StateError(Exception):
    pass


def fsync_dir(path: Path) -> None:
    """fsync теки після os.replace: без нього сам rename може не пережити вимкнення живлення, і після старту
    лишиться попередня версія файлу, хоча відповідь уже сказала «збережено»."""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class StateStore:
    """Сховище розділів стану. Потокобезпечне; read повертає копію, щоб зміни викликача не просочились."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Старт із порожнього стану тихо стер би чужі бронювання; вирішує людина.
            raise StateError(f"{self._path} is unreadable ({exc}); fix or move it by hand") from exc
        if not isinstance(data, dict):
            raise StateError(f"{self._path}: top level must be an object")
        return data

    def read(self, section: str) -> Any:
        with self._lock:
            return copy.deepcopy(self._data.get(section))

    def write(self, section: str, value: Any) -> None:
        """Замінює розділ і одразу пише файл. Якщо запис не вдався — пам'ять лишається старою."""
        with self._lock:
            data = {**self._data, section: copy.deepcopy(value)}
            self._dump(data)
            self._data = data

    def _dump(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=f".{self._path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        with contextlib.suppress(OSError):  # файл уже новий; невдалий fsync теки не робить запис невдалим
            fsync_dir(self._path.parent)
