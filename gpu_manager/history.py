"""Історія телеметрії: щосекундні точки в пам'яті за останню годину, усереднені — у SQLite на тижні.

Відповідь — стислі масиви-колонки, проріджені до заданої кількості точок: агент, що читає годину історії,
платить за десятки чисел, а не за тисячі (домовленість 1.2 про артефакт на запит)."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .gpu import GpuSample

Point = tuple[float, float | None, float | None, float | None, float | None]  # ts, temp, util, power, mem

# Старі рядки SQLite чистяться раз на годину: частіше — марна робота, рідше — зайвий ріст файлу.
_PRUNE_EVERY_S = 3600
_SECONDS_PER_DAY = 86400


class History:
    def __init__(
        self,
        db_path: Path,
        ring_keep_s: int,
        db_interval_s: int,
        db_keep_days: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """ring_keep_s — скільки секунд тримати в пам'яті; db_interval_s — період усереднених точок у SQLite;
        db_keep_days — скільки днів їх зберігати."""
        self._ring_keep_s = ring_keep_s
        self._db_interval_s = db_interval_s
        self._db_keep_s = db_keep_days * _SECONDS_PER_DAY
        self._clock = clock
        self._lock = threading.Lock()
        self._ring: dict[int, deque[Point]] = {}
        self._window_start: float | None = None
        self._last_prune = 0.0
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS samples (ts REAL, gpu INTEGER, temp REAL, util REAL, power REAL, mem REAL)"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS samples_gpu_ts ON samples (gpu, ts)")
        self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def add(self, samples: Iterable[GpuSample]) -> None:
        """Додає заміри одного опитування. Заміри з помилкою пропускаються."""
        now = self._clock()
        with self._lock:
            for s in samples:
                if s.error is not None:
                    continue
                ring = self._ring.setdefault(s.index, deque())
                ring.append((s.ts, s.temperature_c, s.util_pct, s.power_w, s.memory_used_mib))
                while ring and ring[0][0] < now - self._ring_keep_s:
                    ring.popleft()
            if self._window_start is None:
                self._window_start = now
            elif now - self._window_start >= self._db_interval_s:
                self._flush(self._window_start, now)
                self._window_start = now

    def _flush(self, start: float, now: float) -> None:
        rows = []
        for gpu, ring in self._ring.items():
            points = [p for p in ring if p[0] >= start]
            if points:
                agg = _aggregate(points)
                rows.append((now, gpu, agg[1], agg[2], agg[3], agg[4]))
        self._db.executemany("INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?)", rows)
        if now - self._last_prune >= _PRUNE_EVERY_S:
            self._db.execute("DELETE FROM samples WHERE ts < ?", (now - self._db_keep_s,))
            self._last_prune = now
        self._db.commit()

    def query(self, gpus: Iterable[int], minutes: float, points: int) -> dict[int, dict[str, list[Any]]]:
        """Історія за minutes хвилин, не більше points точок на карту.

        Період до ring_keep_s — з пам'яті (щосекундні), довший — із SQLite. Колонки: t, temp_c (максимум),
        util_pct і power_w (середнє), mem_mib (максимум)."""
        now = self._clock()
        since = now - minutes * 60
        out = {}
        with self._lock:
            for gpu in gpus:
                if minutes * 60 <= self._ring_keep_s:
                    series: list[Point] = [p for p in self._ring.get(gpu, ()) if p[0] >= since]
                else:
                    series = self._db.execute(
                        "SELECT ts, temp, util, power, mem FROM samples WHERE gpu = ? AND ts >= ? ORDER BY ts",
                        (gpu, since),
                    ).fetchall()
                out[gpu] = _columns(_downsample(series, since, now, max(1, points)))
        return out


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate(points: list[Point]) -> Point:
    """Найгірше для температури й пам'яті, середнє для завантаження й потужності: пік перегріву не губиться."""
    temps = [p[1] for p in points if p[1] is not None]
    mems = [p[4] for p in points if p[4] is not None]
    return (
        points[-1][0],
        max(temps) if temps else None,
        _mean([p[2] for p in points if p[2] is not None]),
        _mean([p[3] for p in points if p[3] is not None]),
        max(mems) if mems else None,
    )


def _downsample(series: list[Point], since: float, now: float, points: int) -> list[Point]:
    if len(series) <= points:
        return list(series)
    width = (now - since) / points
    buckets: dict[int, list[Point]] = {}
    for p in series:
        buckets.setdefault(min(points - 1, int((p[0] - since) / width)), []).append(p)
    return [_aggregate(buckets[b]) for b in sorted(buckets)]


def _round(value: float | None) -> int | None:
    return None if value is None else round(value)


def _columns(series: list[Point]) -> dict[str, list[Any]]:
    return {
        "t": [round(p[0]) for p in series],
        "temp_c": [_round(p[1]) for p in series],
        "util_pct": [_round(p[2]) for p in series],
        "power_w": [_round(p[3]) for p in series],
        "mem_mib": [_round(p[4]) for p in series],
    }
