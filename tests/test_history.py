"""§6 SPEC-GPU-001 (через GET /api/history, §7.2.5): історія телеметрії.

Щосекундні заміри тести роблять самі: годинник +1 с, потім GpuManager.tick() (§10) — усе ДО старту
TestClient, щоб фонове опитування застосунку не перетиналося з ручними tick(). Колонка t — unix-секунди
(§6), тому межі періоду перевіряються відносно фальшивого годинника.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from .conftest import HISTORY_COLUMNS, N_GPUS, T0, ok_json, refusal

pytestmark = pytest.mark.e2e

BASE_TEMP = 40.0  # температура gpu 0 у фальшивому бекенді за замовчуванням


def _history(client: Any, **params: Any) -> dict[str, Any]:
    data = ok_json(client.get("/api/history", params=params), f"GET /api/history {params}")
    assert isinstance(data, dict), f"/api/history: expected a JSON object keyed by gpu, got {type(data).__name__}"
    return data


def _series(data: dict[str, Any], gpu: int) -> dict[str, list[Any]]:
    key = str(gpu)
    assert key in data, f"/api/history: expected key {key!r}, got keys {sorted(data)}"
    series = data[key]
    missing = [col for col in HISTORY_COLUMNS if col not in series]
    assert not missing, f"/api/history[{key}]: missing columns {missing}, got {sorted(series)}"
    return series


def _one_spike(env: Any, gpu: int, when: float, field: str, spike: float, base: float) -> Callable[[float], None]:
    """before_tick для Env.tick_for: показник field дорівнює spike лише в момент when, інакше base."""

    def _set(now: float) -> None:
        env.backend.set_reading(gpu, **{field: spike if now == when else base})

    return _set


def _query(env: Any, **params: Any) -> dict[str, list[Any]]:
    """Стартує застосунок і повертає серію gpu 0 для запиту з params."""
    with env.client() as c:
        return _series(_history(c, gpu=0, **params), 0)


# --- Форма відповіді ---------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §6")
def test_history_columns_have_equal_length(env):
    """§6: відповідь на карту — колонки t, temp_c, util_pct, power_w, mem_mib однакової довжини."""
    env.tick_for(120)
    series = _query(env, minutes=2, points=1000)
    lengths = {col: len(series[col]) for col in HISTORY_COLUMNS}
    assert len(set(lengths.values())) == 1 and lengths["t"] > 0, (
        f"history columns: expected equal non-zero lengths, got {lengths}"
    )


@pytest.mark.req("SPEC-GPU-001 §6")
def test_history_t_is_unix_seconds_within_period(env):
    """§6: t — unix-секунди; усі точки запиту на 2 хв лежать у [now − 120, now]."""
    env.tick_for(120)
    now = env.clock()
    series = _query(env, minutes=2, points=1000)
    outside = [t for t in series["t"] if not (now - 120 <= t <= now)]
    assert series["t"] and not outside, (
        f"t outside [{now - 120}, {now}] (unix seconds): {outside[:5]} of {len(series['t'])} points"
    )


@pytest.mark.req("SPEC-GPU-001 §6")
def test_history_single_gpu_returns_only_that_gpu(env):
    """§6, §7.2.5: запит для однієї карти — у відповіді лише її ключ."""
    env.tick_for(30)
    with env.client() as c:
        keys = sorted(_history(c, gpu=2, minutes=1, points=10))
    assert keys == ["2"], f"history?gpu=2: expected keys ['2'], got {keys!r}"


@pytest.mark.req("SPEC-GPU-001 §6")
def test_history_without_gpu_returns_all_gpus(env):
    """§6, §7.2.5: без gpu — усі карти."""
    env.tick_for(30)
    with env.client() as c:
        keys = sorted(_history(c, minutes=1, points=10))
    expected = [str(i) for i in range(N_GPUS)]
    assert keys == expected, f"history without gpu: expected keys {expected!r}, got {keys!r}"


@pytest.mark.req("SPEC-GPU-001 §6")
def test_history_respects_points_limit(env):
    """§6: не більше заданої кількості точок на карту."""
    env.tick_for(120)
    series = _query(env, minutes=2, points=10)
    lengths = {col: len(series[col]) for col in HISTORY_COLUMNS}
    assert all(0 < n <= 10 for n in lengths.values()), f"points=10: expected 1..10 points per column, got {lengths}"


# --- Проріджування ------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §6")
def test_history_downsampling_keeps_temperature_peak(env):
    """§6: проріджування бере максимум температури в групі — пік перегріву не губиться."""
    env.tick_for(120, before_tick=_one_spike(env, 0, T0 + 60, "temperature_c", 95.0, BASE_TEMP))
    series = _query(env, minutes=2, points=4)
    peak = max(series["temp_c"])
    assert peak == 95.0, f"one 95 C sample among 40 C, points=4: expected max temp_c 95.0, got {peak!r}"


@pytest.mark.req("SPEC-GPU-001 §6")
def test_history_downsampling_keeps_memory_peak(env):
    """§6: проріджування бере максимум пам'яті в групі."""
    env.tick_for(120, before_tick=_one_spike(env, 0, T0 + 60, "memory_used_mib", 30000, 100))
    series = _query(env, minutes=2, points=4)
    peak = max(series["mem_mib"])
    assert peak == 30000, f"one 30000 MiB sample among 100 MiB, points=4: expected max mem_mib 30000, got {peak!r}"


@pytest.mark.req("SPEC-GPU-001 §6")
def test_history_downsampling_averages_utilisation(env):
    """§6: завантаження проріджується середнім — одиночний сплеск 100 % дає 0 < значення < 100."""
    env.tick_for(120, before_tick=_one_spike(env, 0, T0 + 60, "util_pct", 100.0, 0.0))
    series = _query(env, minutes=2, points=4)
    top = max(series["util_pct"])
    assert 0.0 < top < 100.0, (
        f"one 100 % sample among 0 % in groups of ~30, points=4: expected a group mean in (0, 100) %, got {top!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §6")
def test_history_downsampling_averages_power(env):
    """§6: потужність проріджується середнім — одиночний сплеск 300 Вт на тлі 30 Вт дає 30 < значення < 300."""
    env.tick_for(120, before_tick=_one_spike(env, 0, T0 + 60, "power_w", 300.0, 30.0))
    series = _query(env, minutes=2, points=4)
    top = max(series["power_w"])
    assert 30.0 < top < 300.0, (
        f"one 300 W sample among 30 W in groups of ~30, points=4: expected a group mean in (30, 300) W, got {top!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §6")
def test_history_error_samples_excluded(env):
    """§6: заміри з помилкою в історію не потрапляють (помилкові мають помітні значення 99 °C)."""

    def _alternate(now: float) -> None:
        if int(now) % 2 == 0:
            env.backend.set_reading(0, temperature_c=99.0, util_pct=99.0, power_w=299.0, error="NVML: GPU is lost")
        else:
            env.backend.set_reading(0, temperature_c=BASE_TEMP, util_pct=0.0, power_w=30.0, error=None)

    env.tick_for(120, before_tick=_alternate)
    series = _query(env, minutes=2, points=1000)
    temps = set(series["temp_c"])
    assert series["temp_c"] and temps == {BASE_TEMP}, (
        f"error samples carry 99 C: expected only {BASE_TEMP} C in history, got values {sorted(temps)}"
    )


# --- Кільце в пам'яті й SQLite -----------------------------------------------------------------------------

RING_S = 120  # gpu.ring_keep_s у тестах перемикання джерела
DB_INTERVAL_S = 60  # gpu.history_db_interval_s у тих самих тестах
SPAN_S = 300  # скільки секунд опитування; довше за кільце в 2,5 раза


def _ring_db_env(make_env: Callable[..., Any]) -> Any:
    return make_env({"gpu.ring_keep_s": RING_S, "gpu.history_db_interval_s": DB_INTERVAL_S})


@pytest.mark.req("SPEC-GPU-001 §6")
def test_history_within_ring_has_per_second_points(make_env):
    """§6: період ≤ ring_keep_s — з пам'яті, де точки щосекундні (≈120 точок за 120 с)."""
    env = _ring_db_env(make_env)
    env.tick_for(SPAN_S)
    series = _query(env, minutes=RING_S // 60, points=1000)
    # 120 щосекундних точок у вікні 120 с; запас на невизначену специфікацією обробку меж вікна.
    assert len(series["t"]) >= 100, (
        f"period {RING_S} s <= ring_keep_s: expected ~{RING_S} per-second points, got {len(series['t'])}"
    )


@pytest.mark.req("SPEC-GPU-001 §6")
def test_history_beyond_ring_uses_db_summaries(make_env):
    """§6: період > ring_keep_s — із SQLite, де одна зведена точка на history_db_interval_s."""
    env = _ring_db_env(make_env)
    env.tick_for(SPAN_S)
    series = _query(env, minutes=SPAN_S // 60, points=1000)
    n = len(series["t"])
    # 300 с / 60 с = 5 зведених точок; ±2 на невизначені специфікацією межі інтервалів.
    assert 3 <= n <= 7, f"period {SPAN_S} s > ring_keep_s: expected 3..7 summary points (one per {DB_INTERVAL_S} s), got {n}"


@pytest.mark.req("SPEC-GPU-001 §6")
def test_history_beyond_ring_reaches_older_than_ring(make_env):
    """§6: SQLite зберігає точки, старші за ring_keep_s."""
    env = _ring_db_env(make_env)
    env.tick_for(SPAN_S)
    now = env.clock()
    series = _query(env, minutes=SPAN_S // 60, points=1000)
    oldest = min(series["t"]) if series["t"] else None
    assert oldest is not None and oldest < now - RING_S, (
        f"period {SPAN_S} s: expected a point older than now - {RING_S} = {now - RING_S}, oldest t is {oldest!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §6")
def test_history_db_summary_keeps_temperature_peak(make_env):
    """§6: пік перегріву не губиться і в зведеній точці SQLite (пік — поза вікном кільця)."""
    env = _ring_db_env(make_env)
    env.tick_for(SPAN_S, before_tick=_one_spike(env, 0, T0 + 90, "temperature_c", 95.0, BASE_TEMP))
    series = _query(env, minutes=SPAN_S // 60, points=1000)
    peak = max(series["temp_c"]) if series["temp_c"] else None
    assert peak == 95.0, f"one 95 C sample 210 s ago, period from SQLite: expected max temp_c 95.0, got {peak!r}"


@pytest.mark.req("SPEC-GPU-001 §6")
def test_history_db_file_in_data_dir(make_env):
    """§6: зведені точки пишуться в data_dir/history.sqlite."""
    env = _ring_db_env(make_env)
    env.tick_for(SPAN_S)
    path = env.data_dir / "history.sqlite"
    assert path.is_file(), f"after {SPAN_S} s of polling with a {DB_INTERVAL_S} s DB interval: expected {path}"


# --- Типові параметри (§7.2.5) ---------------------------------------------------------------------------------

DEFAULT_SPAN_S = 7200  # 2 год опитування — удвічі довше за типовий період 60 хв
DEFAULT_STEP_S = 10  # крок опитування; 360 точок за останню годину — більше за типові 120


@pytest.mark.req("SPEC-GPU-001 §7.2.5")
def test_history_defaults_cover_all_gpus(env):
    """§7.2.5: без параметрів — усі карти."""
    env.tick_for(DEFAULT_SPAN_S, step=DEFAULT_STEP_S)
    with env.client() as c:
        keys = sorted(_history(c))
    expected = [str(i) for i in range(N_GPUS)]
    assert keys == expected, f"history without params: expected keys {expected!r}, got {keys!r}"


@pytest.mark.req("SPEC-GPU-001 §7.2.5")
def test_history_default_points_is_120(env):
    """§7.2.5: типово 120 точок на карту (з 360 доступних)."""
    env.tick_for(DEFAULT_SPAN_S, step=DEFAULT_STEP_S)
    with env.client() as c:
        series = _series(_history(c), 0)
    n = len(series["t"])
    # Верхня межа — сама вимога; нижня (> 60) відрізняє типові 120 від 60 (типове MCP) — див. звіт.
    assert 60 < n <= 120, f"history without points from 360 samples: expected 61..120 points, got {n}"


@pytest.mark.req("SPEC-GPU-001 §7.2.5")
def test_history_default_period_is_60_minutes(env):
    """§7.2.5: типовий період — 60 хв: найстаріша точка не старша за годину і не молодша за 50 хв."""
    env.tick_for(DEFAULT_SPAN_S, step=DEFAULT_STEP_S)
    now = env.clock()
    with env.client() as c:
        series = _series(_history(c), 0)
    oldest = min(series["t"]) if series["t"] else None
    # Запас — ширина однієї групи проріджування (3600 с / 120 точок), бо представник t групи не визначений.
    slack = 3600 / 120
    assert oldest is not None and now - 3600 - slack <= oldest < now - 3000, (
        f"default period: expected oldest t in [{now - 3600 - slack}, {now - 3000}), got {oldest!r}"
    )


# --- Відмови (§6, §9) ------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §6")
@pytest.mark.parametrize(
    "params",
    [{"minutes": 0}, {"minutes": -5}, {"points": 0}, {"points": -1}],
    ids=["minutes-zero", "minutes-negative", "points-zero", "points-negative"],
)
def test_history_non_positive_params_refused(env, params):
    """§6: minutes ≤ 0 або points ≤ 0 → bad_history."""
    with env.client() as c:
        refusal(c.get("/api/history", params=params), "bad_history")


@pytest.mark.req("SPEC-GPU-001 §9.1")
def test_history_unknown_gpu_refused(env):
    """§9.1: історія неіснуючої карти → unknown_gpu."""
    with env.client() as c:
        refusal(c.get("/api/history", params={"gpu": N_GPUS}), "unknown_gpu")
