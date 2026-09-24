"""Ядро менеджера: єдине місце, через яке йдуть обидві оболонки (веб і MCP), тож вони не можуть розійтися.
Ядро не знає мов: відмови — ManagerError з кодом, попередження — код і параметри."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from .config import Config
from .gpu import GpuBackend, GpuProcess, GpuSample
from .history import History
from .journal import Journal
from .messages import ManagerError
from .reservations import Reservation, ReservationBook
from .store import StateStore

FREE = "free"
RESERVED = "reserved"
RESERVED_EXPIRED = "reserved_expired"
BUSY = "busy"  # процеси чи зайнята пам'ять, але карту ніхто не бронював
UNKNOWN = "unknown"  # немає телеметрії

# Верхні межі запитів: захист від випадкового «дай мільйон точок».
_JOURNAL_MAX = 500
_HISTORY_POINTS_MAX = 1000


class GpuManager:
    def __init__(
        self,
        cfg: Config,
        backend: GpuBackend,
        book: ReservationBook,
        journal: Journal,
        history: History,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._cfg = cfg
        self._backend = backend
        self._book = book
        self._journal = journal
        self._history = history
        self._clock = clock
        self._info = backend.info()
        self._lock = threading.Lock()
        # Зміна бронювання і її запис у журнал — одна дія: інакше одночасні дії (MCP іде в робочих потоках)
        # потрапляли б у журнал не в тому порядку, у якому змінились бронювання.
        self._change_lock = threading.Lock()
        # Один tick за раз: move бере свіжий замір з потоку запиту, і замір фонового опитування, почату ще до зупинки
        # моделі, записався б після свіжого — новий старт знову побачив би пам'ять зупиненої моделі.
        self._tick_lock = threading.Lock()
        self._samples: dict[int, GpuSample] = {}
        self._procs: dict[int, list[GpuProcess]] = {}
        # Моделі менеджера на карті (фаза 3); підключає ModelRunner. Карта з моделлю — busy навіть до появи її
        # процесу в NVML (старт ~20 с), інакше gpu_free порадив би карту, яку вже зайнято.
        self.models_on: Callable[[int], list[dict[str, Any]]] = lambda gpu: []

    @property
    def gpu_count(self) -> int:
        return len(self._info)

    @property
    def users(self) -> tuple[str, ...]:
        return self._cfg.users

    @property
    def store(self) -> StateStore:
        """Спільний state.json: у ньому ж живе черга завантажень моделей."""
        return self._book.store

    @property
    def journal_log(self) -> Journal:
        """Спільний журнал дій: і карти, і моделі."""
        return self._journal

    def close(self) -> None:
        """Закриває базу історії при виході сервісу."""
        self._history.close()

    def tick(self) -> None:
        """Один крок опитування: фонова задача кожні sample_interval_s і move після зупинки моделі (свіжий замір).
        Карта-заглушка (NVML не прочитав її при старті) перечитується, доки не прочитається."""
        with self._tick_lock:
            if any(not g.memory_total_mib for g in self._info):
                self._info = self._backend.info()
            samples = self._backend.sample(self._clock())
            procs = self._backend.processes()
            with self._lock:
                self._samples = {s.index: s for s in samples}
                self._procs = procs
            self._history.add(samples)

    # ---- читання ---------------------------------------------------------------------------------

    def overview(self) -> dict[str, Any]:
        """Усі карти: телеметрія, процеси, бронювання, статус."""
        now = self._clock()
        return {"ts": now, "users": list(self._cfg.users), "gpus": [self._view(i, now) for i in range(self.gpu_count)]}

    def gpu(self, gpu: int) -> dict[str, Any]:
        self._check_gpu(gpu)
        return self._view(gpu, self._clock())

    def free(self) -> list[dict[str, Any]]:
        """Карти зі статусом free: без бронювання, без процесів, без зайнятої пам'яті."""
        return [g for g in self.overview()["gpus"] if g["status"] == FREE]

    def journal(self, limit: int) -> list[dict[str, Any]]:
        return self._journal.tail(max(1, min(limit, _JOURNAL_MAX)))

    def history(self, gpu: int | None, minutes: float, points: int) -> dict[int, dict[str, list[Any]]]:
        """Історія однієї карти (gpu) або всіх (None) за minutes хвилин, до points точок на карту."""
        if gpu is not None:
            self._check_gpu(gpu)
        if minutes <= 0 or points <= 0:
            raise ManagerError("bad_history")
        gpus = [gpu] if gpu is not None else list(range(self.gpu_count))
        return self._history.query(gpus, minutes, min(points, _HISTORY_POINTS_MAX))

    def _view(self, i: int, now: float) -> dict[str, Any]:
        info = self._info[i]
        with self._lock:
            sample = self._samples.get(i)
            procs = list(self._procs.get(i, []))
        res = self._book.get(i)
        models = self.models_on(i)
        return {
            **asdict(info),
            "temperature_c": sample.temperature_c if sample else None,
            "util_pct": sample.util_pct if sample else None,
            "power_w": sample.power_w if sample else None,
            "memory_used_mib": sample.memory_used_mib if sample else None,
            "error": sample.error if sample else "no sample yet",
            "status": self._status(sample, procs, res, now, models),
            "users": sorted({p.user for p in procs if p.user} | ({res.user} if res else set())),
            "processes": [asdict(p) for p in procs],
            "reservation": {**asdict(res), "expired": res.expired(now)} if res else None,
            "models": models,
        }

    def _status(self, sample: GpuSample | None, procs: list[GpuProcess], res: Reservation | None, now: float,
                models: list[dict[str, Any]] | None = None) -> str:
        if res is not None:
            return RESERVED_EXPIRED if res.expired(now) else RESERVED
        if sample is None or sample.error is not None:
            return UNKNOWN
        if procs or models or (sample.memory_used_mib or 0) >= self._cfg.busy_memory_mib:
            return BUSY
        return FREE

    # ---- зміни -----------------------------------------------------------------------------------

    def reserve(self, gpu: int, user: str, purpose: str = "", hours: float | None = None) -> dict[str, Any]:
        """Бронює карту за user; повертає {reserved, gpu, warnings}. warnings — [{code, params}]:
        напр. foreign_processes, коли на карті вже працюють процеси інших користувачів."""
        self._check_gpu(gpu)
        self._check_user(user)
        with self._change_lock:
            new, previous = self._book.reserve(gpu, user, purpose, hours)
            self._journal.record(
                user, "reserve_update" if previous else "reserve", gpu=gpu, purpose=new.purpose, until=new.until
            )
        view = self._view(gpu, self._clock())
        foreign = [p for p in view["processes"] if p["user"] != user]
        warnings = []
        if foreign:
            listed = ", ".join(f"{p['user']} ({p['name']}, pid {p['pid']})" for p in foreign)
            warnings.append({"code": "foreign_processes", "params": {"gpu": gpu, "processes": listed}})
        return {"reserved": True, "gpu": view, "warnings": warnings}

    def release(self, gpu: int, user: str, force: bool = False) -> dict[str, Any]:
        """Знімає бронювання; повертає {released, previous, gpu}. Незаброньована карта — released=False."""
        self._check_gpu(gpu)
        self._check_user(user)
        with self._change_lock:
            removed = self._book.release(gpu, user, force)
            if removed is not None and removed.user == user:
                self._journal.record(user, "release", gpu=gpu)
            elif removed is not None:
                self._journal.record(user, "release_forced", gpu=gpu, owner=removed.user, purpose=removed.purpose)
        if removed is None:
            return {"released": False, "previous": None, "gpu": self.gpu(gpu)}
        return {"released": True, "previous": asdict(removed), "gpu": self.gpu(gpu)}

    # ---- перевірки -------------------------------------------------------------------------------

    def _check_gpu(self, gpu: int) -> None:
        if not isinstance(gpu, int) or isinstance(gpu, bool) or not 0 <= gpu < self.gpu_count:
            raise ManagerError("unknown_gpu", gpu=gpu, last=self.gpu_count - 1)

    def _check_user(self, user: str) -> None:
        require_user(self._cfg.users, user)


def require_user(users: tuple[str, ...], user: str) -> None:
    """ManagerError unknown_user, якщо user не з users.allowed. Спільна для карт і моделей."""
    if user not in users:
        raise ManagerError("unknown_user", user=user, allowed=", ".join(users))
