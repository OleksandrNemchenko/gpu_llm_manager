"""Журнал дій (data/journal.jsonl): хто, що, з якою картою і коли.

Потрібен, бо користувачі довірені й без токенів: захист від помилок — не заборона, а видимість.
Розмір обмежений двома межами з конфігу (прохання користувача): останні keep_entries записів і лише
за keep_days днів; видаляється все, що виходить за будь-яку з них."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

_SECONDS_PER_DAY = 86400


class Journal:
    def __init__(
        self, path: Path, keep_entries: int, keep_days: float, clock: Callable[[], float] = time.time
    ) -> None:
        """keep_entries — скільки останніх записів лишати; keep_days — за скільки днів. Обидва > 0."""
        self._path = path
        self._keep_entries = keep_entries
        self._keep_s = keep_days * _SECONDS_PER_DAY
        self._clock = clock
        self._lock = threading.Lock()

    def record(self, user: str, action: str, **details: Any) -> dict[str, Any]:
        """Дописує запис {ts, user, action, **details}, чистить журнал за межами й повертає запис."""
        entry = {"ts": round(self._clock(), 3), "user": user, "action": action, **details}
        with self._lock:
            entries = self._read() + [entry]
            self._write(self._keep(entries))
        return entry

    def tail(self, limit: int) -> list[dict[str, Any]]:
        """Останні limit записів у межах зберігання, найновіший перший."""
        with self._lock:
            entries = self._keep(self._read())
        return entries[::-1][:limit]

    def _keep(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        oldest = self._clock() - self._keep_s
        return [e for e in entries if e.get("ts", 0) >= oldest][-self._keep_entries:]

    def _read(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        out = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # рядок, обірваний аварійною зупинкою, не має ховати решту
            if isinstance(item, dict):
                out.append(item)
        return out

    def _write(self, entries: list[dict[str, Any]]) -> None:
        # Файл переписується цілком і атомарно: чистка й дописування — одна дія, а обірваний рядок
        # від аварійної зупинки зникає при першому ж записі.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=f".{self._path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
