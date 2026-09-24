"""§3 SPEC-GPU-001: статус карти (reserved / reserved_expired / unknown / busy / free) і пріоритет рядків.

Статус читається з GET /api/overview (§7.2.3). Телеметрію й процеси фальшивий бекенд віддає вже
під час старту: lifespan build_app сам робить tick() (§10).
"""

from __future__ import annotations

import pytest

from .conftest import BUSY_MIB, card, gpu_proc, ok_json, refusal, reserve

pytestmark = pytest.mark.e2e

HOUR_S = 3600
TEN_YEARS_S = 10 * 365 * 86400  # «ніколи» для бронювання без строку


@pytest.mark.req("SPEC-GPU-001 §3.5")
def test_status_idle_card_is_free(env):
    """§3.5: без бронювання, процесів і зайнятої пам'яті карта — free."""
    with env.client() as c:
        status = card(c, 0)["status"]
    assert status == "free", f"idle gpu 0: expected status 'free', got {status!r}"


@pytest.mark.req("SPEC-GPU-001 §3.4")
def test_status_process_makes_card_busy(env, backend):
    """§3.4: процес на карті робить її busy, навіть коли пам'яті зайнято менше за поріг."""
    backend.set_processes(1, gpu_proc(4242, "bob", used_mib=100))
    backend.set_reading(1, memory_used_mib=100)
    with env.client() as c:
        status = card(c, 1)["status"]
    assert status == "busy", f"gpu 1 with a process and 100 MiB used: expected 'busy', got {status!r}"


@pytest.mark.req("SPEC-GPU-001 §3.4")
@pytest.mark.parametrize(
    ("used_mib", "expected"),
    [(BUSY_MIB - 1, "free"), (BUSY_MIB, "busy"), (BUSY_MIB + 1, "busy")],
    ids=["below-threshold", "at-threshold", "above-threshold"],
)
def test_status_memory_threshold(env, backend, used_mib, expected):
    """§3.4: зайнята пам'ять ≥ busy_memory_mib без процесів — busy; на 1 MiB менше — free."""
    backend.set_reading(2, memory_used_mib=used_mib)
    with env.client() as c:
        status = card(c, 2)["status"]
    assert status == expected, (
        f"gpu 2 with {used_mib} MiB used (threshold {BUSY_MIB} MiB): expected {expected!r}, got {status!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §3.3")
def test_status_without_telemetry_is_unknown(env, backend):
    """§3.3: без бронювання й без телеметрії карти — unknown."""
    backend.drop_telemetry(3)
    with env.client() as c:
        status = card(c, 3)["status"]
    assert status == "unknown", f"gpu 3 absent from sample(): expected 'unknown', got {status!r}"


@pytest.mark.req("SPEC-GPU-001 §3.3")
def test_status_with_telemetry_error_is_unknown(env, backend):
    """§3.3: без бронювання й з помилковим заміром — unknown."""
    backend.set_error(3)
    with env.client() as c:
        status = card(c, 3)["status"]
    assert status == "unknown", f"gpu 3 with GpuSample.error set: expected 'unknown', got {status!r}"


@pytest.mark.req("SPEC-GPU-001 §3")
def test_status_unknown_beats_processes(env, backend):
    """§3, пріоритет рядків: unknown (рядок 3) важливіший за busy (рядок 4)."""
    backend.set_processes(3, gpu_proc(4243, "bob"))
    backend.drop_telemetry(3)
    with env.client() as c:
        status = card(c, 3)["status"]
    assert status == "unknown", f"gpu 3 without telemetry but with a process: expected 'unknown', got {status!r}"


@pytest.mark.req("SPEC-GPU-001 §3")
def test_status_reservation_beats_processes(env, backend):
    """§3, пріоритет: бронювання важливіше за процеси й зайняту пам'ять."""
    backend.set_processes(1, gpu_proc(4242, "bob", used_mib=5000))
    backend.set_reading(1, memory_used_mib=5000)
    with env.client() as c:
        ok_json(reserve(c, 1, "alice", purpose="eval"), "reserve gpu 1 by alice")
        status = card(c, 1)["status"]
    assert status == "reserved", f"reserved gpu 1 with foreign process: expected 'reserved', got {status!r}"


@pytest.mark.req("SPEC-GPU-001 §3")
def test_status_reservation_beats_missing_telemetry(env, backend):
    """§3, пріоритет: бронювання (рядок 1) важливіше за відсутню телеметрію (рядок 3)."""
    backend.drop_telemetry(2)
    with env.client() as c:
        ok_json(reserve(c, 2, "alice", purpose="eval"), "reserve gpu 2 by alice")
        status = card(c, 2)["status"]
    assert status == "reserved", f"reserved gpu 2 without telemetry: expected 'reserved', got {status!r}"


@pytest.mark.req("SPEC-GPU-001 §3.1")
def test_status_reserved_one_second_before_until(env):
    """§3.1: до моменту until (now < until) карта — reserved."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", hours=1), "reserve gpu 0 for 1 h")
        env.clock.advance(HOUR_S - 1)
        status = card(c, 0)["status"]
    assert status == "reserved", f"1 s before until: expected 'reserved', got {status!r}"


@pytest.mark.req("SPEC-GPU-001 §3.2")
def test_status_expired_exactly_at_until(env):
    """§3.2: у момент until (now ≥ until, рівність включно) карта — reserved_expired."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", hours=1), "reserve gpu 0 for 1 h")
        env.clock.advance(HOUR_S)
        status = card(c, 0)["status"]
    assert status == "reserved_expired", f"now == until: expected 'reserved_expired', got {status!r}"


@pytest.mark.req("SPEC-GPU-001 §3.2")
def test_reservation_expired_flag_at_until(env):
    """§3.2, §7.2: поле reservation.expired стає true у момент until."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", hours=1), "reserve gpu 0 for 1 h")
        env.clock.advance(HOUR_S)
        reservation = card(c, 0)["reservation"]
    assert reservation is not None and reservation.get("expired") is True, (
        f"now == until: expected reservation.expired == true, got {reservation!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §3")
def test_status_expired_reservation_beats_processes(env, backend):
    """§3, пріоритет: прострочене бронювання (рядок 2) важливіше за процеси (рядок 4)."""
    backend.set_processes(1, gpu_proc(4242, "bob"))
    with env.client() as c:
        ok_json(reserve(c, 1, "alice", hours=1), "reserve gpu 1 for 1 h")
        env.clock.advance(2 * HOUR_S)
        status = card(c, 1)["status"]
    assert status == "reserved_expired", (
        f"expired reservation with a foreign process: expected 'reserved_expired', got {status!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §3.2")
def test_expired_reservation_not_released_by_polling(env):
    """§3: прострочене бронювання не знімається автоматично — навіть після кількох tick() за добу."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="keep", hours=1), "reserve gpu 0 for 1 h")
        env.clock.advance(HOUR_S + 86400)
        for _ in range(3):
            env.manager.tick()
        item = card(c, 0)
    reservation = item["reservation"]
    assert item["status"] == "reserved_expired" and reservation is not None and reservation.get("user") == "alice", (
        f"a day after until: expected alice's reservation kept with status 'reserved_expired', "
        f"got status {item['status']!r}, reservation {reservation!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §3.2")
def test_expired_reservation_still_blocks_other_user(env):
    """§3 + §4.6: прострочене бронювання — не вільна карта; чуже бронювання → reserved_by_other."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", hours=1), "reserve gpu 0 for 1 h")
        env.clock.advance(2 * HOUR_S)
        refusal(reserve(c, 0, "bob", purpose="mine now"), "reserved_by_other")


@pytest.mark.req("SPEC-GPU-001 §3.1")
def test_reservation_without_hours_never_expires(env):
    """§3.1: бронювання без строку (until = null) лишається reserved і через 10 років."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="forever"), "reserve gpu 0 without hours")
        env.clock.advance(TEN_YEARS_S)
        status = card(c, 0)["status"]
    assert status == "reserved", f"10 years after a reservation without until: expected 'reserved', got {status!r}"


@pytest.mark.req("SPEC-GPU-001 §7.2")
def test_card_users_are_sorted_process_owners_and_reserver(env, backend):
    """§7.2, поле users: відсортовані логіни власників процесів + власник бронювання."""
    backend.set_processes(1, gpu_proc(5001, "dave"), gpu_proc(5002, "bob"))
    with env.client() as c:
        ok_json(reserve(c, 1, "carol", purpose="shared"), "reserve gpu 1 by carol")
        users = card(c, 1)["users"]
    assert users == ["bob", "carol", "dave"], f"gpu 1 users: expected ['bob', 'carol', 'dave'], got {users!r}"
