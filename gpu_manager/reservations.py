"""Бронювання карт: хто позначив карту як зайняту, навіщо і до коли.

Прострочене бронювання лише позначається, але не знімається само: тихо звільнити
карту посеред чийогось навчання гірше, ніж показати застарілу позначку."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from .messages import ManagerError
from .store import StateStore

SECTION = "reservations"
_SECONDS_PER_HOUR = 3600
# Найдовший строк, годин (рік): довший — описка; «безстроково» — hours не задано. Без межі inf/NaN чи 1e8 годин
# потрапляли в state.json і ламали показ бронювання для всіх (JSON не має Infinity, datetime — року понад 9999).
MAX_HOURS = 8760


@dataclass(frozen=True)
class Reservation:
    gpu: int
    user: str
    purpose: str
    since: float
    until: float | None  # None — без строку

    def expired(self, now: float) -> bool:
        return self.until is not None and now >= self.until


class ReservationBook:
    """Бронювання, що переживають перезапуск: кожна зміна одразу йде у StateStore."""

    def __init__(self, store: StateStore, clock: Callable[[], float] = time.time) -> None:
        self._store = store
        self._clock = clock
        self._lock = threading.Lock()
        raw = store.read(SECTION) or {}
        self._items = {int(gpu): Reservation(**item) for gpu, item in raw.items()}

    @property
    def store(self) -> StateStore:
        return self._store

    def get(self, gpu: int) -> Reservation | None:
        with self._lock:
            return self._items.get(gpu)

    def reserve(self, gpu: int, user: str, purpose: str, hours: float | None) -> tuple[Reservation, Reservation | None]:
        """Бронює карту за user.

        hours — строк у годинах або None; повертає (нове, попереднє). Повторне бронювання тим самим
        користувачем оновлює мету й строк, зберігаючи since. Чуже — ManagerError reserved_by_other."""
        if hours is not None and not (math.isfinite(hours) and 0 < hours <= MAX_HOURS):
            raise ManagerError("bad_hours", max=MAX_HOURS)
        with self._lock:
            current = self._items.get(gpu)
            if current is not None and current.user != user:
                raise ManagerError("reserved_by_other", gpu=gpu, owner=current.user, purpose=current.purpose)
            now = self._clock()
            new = Reservation(
                gpu=gpu,
                user=user,
                purpose=purpose.strip(),
                since=current.since if current is not None else now,
                until=now + hours * _SECONDS_PER_HOUR if hours is not None else None,
            )
            self._commit({**self._items, gpu: new})
            return new, current

    def release(self, gpu: int, user: str, force: bool) -> Reservation | None:
        """Знімає бронювання; повертає зняте або None, якщо карта не була заброньована.
        Чуже бронювання без force — ManagerError release_needs_force."""
        with self._lock:
            current = self._items.get(gpu)
            if current is None:
                return None
            if current.user != user and not force:
                raise ManagerError("release_needs_force", gpu=gpu, owner=current.user)
            self._commit({g: r for g, r in self._items.items() if g != gpu})
            return current

    def _commit(self, items: dict[int, Reservation]) -> None:
        # Спершу файл: якщо запис не вдасться, пам'ять лишиться такою ж, як файл.
        self._store.write(SECTION, {str(g): asdict(r) for g, r in sorted(items.items())})
        self._items = items
