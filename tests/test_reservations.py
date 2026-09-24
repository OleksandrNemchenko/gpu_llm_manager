"""§4 SPEC-GPU-001: бронювання й звільнення карт через HTTP API (§7.2.6–7.2.7), збереження стану.

Час у записі бронювання — unix-секунди фальшивого годинника: since = now, until = now + hours·3600
(§4.3). Час старту годинника — T0.
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import (
    BASE_URL,
    N_GPUS,
    STRANGER,
    T0,
    card,
    gpu_proc,
    ok_json,
    refusal,
    release,
    reservations,
    reserve,
)

pytestmark = pytest.mark.e2e

HOUR_S = 3600
RESERVATION_FIELDS = ("gpu", "user", "purpose", "since", "until", "expired")


def _projection(record: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any] | None:
    """Лише задані ключі запису (None лишається None) — щоб порівнювати без зайвих полів."""
    if record is None:
        return None
    return {k: record.get(k) for k in keys}


def _reserved_cards(client: Any) -> dict[int, Any]:
    """Карти, на яких є бронювання: {index: reservation}; порожній словник — жодна не заброньована."""
    return {i: r for i, r in reservations(client).items() if r is not None}


# --- §4.1 користувач -------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §4.1")
def test_reserve_unknown_user_refused(env):
    """§4.1: user не з users.allowed → unknown_user; карта лишається без бронювання."""
    with env.client() as c:
        refusal(reserve(c, 0, STRANGER, purpose="x"), "unknown_user")
        reservation = card(c, 0)["reservation"]
    assert reservation is None, f"after unknown_user refusal: expected no reservation, got {reservation!r}"


@pytest.mark.req("SPEC-GPU-001 §4.1")
def test_release_unknown_user_refused(env):
    """§4.1: звільнення від логіна не з users.allowed → unknown_user; бронювання лишається."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="x"), "reserve gpu 0 by alice")
        refusal(release(c, 0, STRANGER, force=True), "unknown_user")
        reservation = card(c, 0)["reservation"]
    assert reservation is not None and reservation.get("user") == "alice", (
        f"after unknown_user refusal: expected alice's reservation kept, got {reservation!r}"
    )


# --- §4.2 номер карти ------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §4.2")
@pytest.mark.parametrize("gpu", [N_GPUS, -1, 100], ids=["n", "negative", "far"])
def test_reserve_gpu_out_of_range_refused(env, gpu):
    """§4.2: gpu поза 0..N-1 → unknown_gpu; жодна карта не заброньована."""
    with env.client() as c:
        refusal(reserve(c, gpu, "alice"), "unknown_gpu")
        reserved = _reserved_cards(c)
    assert reserved == {}, f"after unknown_gpu refusal for gpu={gpu!r}: expected no reservations, got {reserved!r}"


@pytest.mark.req("SPEC-GPU-001 §4.2")
@pytest.mark.parametrize("gpu", [True, False, "0", "1", 1.5], ids=["true", "false", "str-0", "str-1", "float"])
def test_reserve_gpu_not_integer_refused(env, gpu):
    """§4.2: gpu не ціле (true/false, рядок, дробове) → unknown_gpu; жодна карта не заброньована."""
    with env.client() as c:
        refusal(reserve(c, gpu, "alice"), "unknown_gpu")
        reserved = _reserved_cards(c)
    assert reserved == {}, f"after unknown_gpu refusal for gpu={gpu!r}: expected no reservations, got {reserved!r}"


@pytest.mark.req("SPEC-GPU-001 §4.2")
@pytest.mark.parametrize("gpu", [True, False, N_GPUS], ids=["true", "false", "n"])
def test_release_gpu_invalid_refused(env, gpu):
    """§4.2: звільнення з gpu не з 0..N-1 (зокрема true/false) → unknown_gpu."""
    with env.client() as c:
        refusal(release(c, gpu, "alice"), "unknown_gpu")


# --- §4.3 нове бронювання ---------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §4.3")
def test_reserve_free_card_creates_record(env):
    """§4.3: запис {gpu, user, purpose, since=now, until=now+hours·3600}, expired=false."""
    with env.client() as c:
        ok_json(reserve(c, 1, "alice", purpose="llm eval", hours=2), "reserve gpu 1 for 2 h")
        reservation = card(c, 1)["reservation"]
    expected = {
        "gpu": 1,
        "user": "alice",
        "purpose": "llm eval",
        "since": T0,
        "until": T0 + 2 * HOUR_S,
        "expired": False,
    }
    got = _projection(reservation, RESERVATION_FIELDS)
    assert got == expected, f"reservation of gpu 1: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-001 §4.3")
def test_reserve_without_hours_has_null_until(env):
    """§4.3: hours не задано → until = null."""
    with env.client() as c:
        ok_json(reserve(c, 1, "alice", purpose="open-ended"), "reserve gpu 1 without hours")
        reservation = card(c, 1)["reservation"]
    assert reservation is not None and reservation.get("until") is None and "until" in reservation, (
        f"reservation without hours: expected until == null, got {reservation!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §4.3")
@pytest.mark.req("SPEC-GPU-001 §7.2.6")
@pytest.mark.parametrize("hours", ["", None], ids=["empty-string", "null"])
def test_reserve_blank_hours_means_no_until(env, hours):
    """§7.2.6: hours = "" або null — бронювання без строку (until = null)."""
    with env.client() as c:
        ok_json(reserve(c, 1, "alice", hours=hours), f"reserve gpu 1 with hours={hours!r}")
        reservation = card(c, 1)["reservation"]
    assert reservation is not None and reservation.get("until") is None, (
        f"hours={hours!r}: expected until == null, got {reservation!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §7.2.6")
def test_reserve_hours_numeric_string(env):
    """§7.2.6: hours як рядок-число "3" → until = now + 3·3600."""
    with env.client() as c:
        ok_json(reserve(c, 1, "alice", hours="3"), "reserve gpu 1 with hours='3'")
        until = card(c, 1)["reservation"]["until"]
    assert until == T0 + 3 * HOUR_S, f"hours='3': expected until {T0 + 3 * HOUR_S}, got {until!r}"


@pytest.mark.req("SPEC-GPU-001 §4.3")
def test_reserve_fractional_hours(env):
    """§4.3: hours = 1.5 → until = now + 5400 с."""
    with env.client() as c:
        ok_json(reserve(c, 1, "alice", hours=1.5), "reserve gpu 1 with hours=1.5")
        until = card(c, 1)["reservation"]["until"]
    assert until == T0 + 5400, f"hours=1.5: expected until {T0 + 5400}, got {until!r}"


# --- §4.4 hours ≤ 0 -----------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §4.4")
@pytest.mark.parametrize("hours", [0, -1, -0.5, "0", "-2"], ids=["zero", "minus-one", "minus-half", "str-zero", "str-minus"])
def test_reserve_non_positive_hours_refused(env, hours):
    """§4.4: hours ≤ 0 (зокрема рядком) → bad_hours; бронювання не створюється."""
    with env.client() as c:
        refusal(reserve(c, 1, "alice", hours=hours), "bad_hours")
        reservation = card(c, 1)["reservation"]
    assert reservation is None, f"after bad_hours for hours={hours!r}: expected no reservation, got {reservation!r}"


# --- §4.5 повторне бронювання ------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §4.5")
def test_re_reserve_keeps_since(env):
    """§4.5: повторне бронювання тим самим користувачем не змінює since."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="a", hours=1), "first reserve")
        env.clock.advance(600)
        ok_json(reserve(c, 0, "alice", purpose="b", hours=2), "second reserve by the same user")
        since = card(c, 0)["reservation"]["since"]
    assert since == T0, f"re-reserve 600 s later: expected since unchanged at {T0}, got {since!r}"


@pytest.mark.req("SPEC-GPU-001 §4.5")
def test_re_reserve_updates_purpose_and_until(env):
    """§4.5: повторне бронювання тим самим користувачем оновлює purpose і until."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="a", hours=1), "first reserve")
        env.clock.advance(600)
        ok_json(reserve(c, 0, "alice", purpose="b", hours=2), "second reserve by the same user")
        got = _projection(card(c, 0)["reservation"], ("user", "purpose", "until"))
    expected = {"user": "alice", "purpose": "b", "until": T0 + 600 + 2 * HOUR_S}
    assert got == expected, f"re-reserve: expected {expected!r}, got {got!r}"


# --- §4.6 чуже бронювання --------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §4.6")
def test_reserve_other_users_card_refused_and_unchanged(env):
    """§4.6: бронювання чужої заброньованої карти → reserved_by_other; бронювання не змінюється."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="a", hours=1), "reserve gpu 0 by alice")
        env.clock.advance(60)
        refusal(reserve(c, 0, "bob", purpose="b", hours=5), "reserved_by_other")
        got = _projection(card(c, 0)["reservation"], ("user", "purpose", "since", "until"))
    expected = {"user": "alice", "purpose": "a", "since": T0, "until": T0 + HOUR_S}
    assert got == expected, f"after reserved_by_other: expected reservation unchanged {expected!r}, got {got!r}"


# --- §4.7 чужі процеси ---------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §4.7")
def test_reserve_with_foreign_processes_succeeds_with_warning(env, backend):
    """§4.7: на карті процес іншого власника — бронювання відбувається, у відповіді одне попередження."""
    backend.set_processes(1, gpu_proc(4242, "bob"))
    with env.client() as c:
        body = ok_json(reserve(c, 1, "alice", purpose="x"), "reserve gpu 1 with bob's process")
        owner = card(c, 1)["reservation"]["user"]
    warnings = body.get("warnings")
    assert body.get("reserved") is True and owner == "alice", (
        f"reserve over a foreign process: expected reserved by alice, got body {body!r}, owner {owner!r}"
    )
    assert isinstance(warnings, list) and len(warnings) == 1, (
        f"reserve over a foreign process: expected exactly 1 warning (foreign_processes), got {warnings!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §4.7")
def test_reserve_with_own_processes_has_no_warning(env, backend):
    """§4.7: процеси самого user — не чужі, попереджень немає."""
    backend.set_processes(1, gpu_proc(4242, "alice"))
    with env.client() as c:
        body = ok_json(reserve(c, 1, "alice", purpose="x"), "reserve gpu 1 with alice's own process")
    assert body.get("warnings") == [], f"own processes only: expected warnings == [], got {body.get('warnings')!r}"


@pytest.mark.req("SPEC-GPU-001 §4.7")
def test_reserve_empty_card_has_no_warning(env):
    """§4.7, §7.2.6: карта без процесів — warnings порожній список."""
    with env.client() as c:
        body = ok_json(reserve(c, 1, "alice", purpose="x"), "reserve empty gpu 1")
    assert body.get("warnings") == [], f"empty card: expected warnings == [], got {body.get('warnings')!r}"


# --- §4.8–4.10 звільнення ---------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §4.8")
def test_release_own_reservation(env):
    """§4.8: звільнення власного бронювання — released: true, previous — зняте бронювання."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="done soon"), "reserve gpu 0 by alice")
        body = ok_json(release(c, 0, "alice"), "release own gpu 0")
    previous = body.get("previous") or {}
    assert body.get("released") is True and previous.get("user") == "alice", (
        f"release own: expected released true and previous.user 'alice', got {body!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §4.8")
def test_release_own_leaves_card_unreserved(env):
    """§4.8: після звільнення власного бронювання карта без бронювання й має статус free."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice"), "reserve gpu 0 by alice")
        ok_json(release(c, 0, "alice"), "release own gpu 0")
        item = card(c, 0)
    assert item["reservation"] is None and item["status"] == "free", (
        f"after release: expected no reservation and status 'free', got {item['reservation']!r}, {item['status']!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §4.9")
def test_release_unreserved_card_is_not_an_error(env):
    """§4.9: звільнення незаброньованої карти — HTTP 200, released: false, previous: null."""
    with env.client() as c:
        body = ok_json(release(c, 2, "alice"), "release unreserved gpu 2")
    assert body.get("released") is False and "previous" in body and body["previous"] is None, (
        f"release unreserved: expected released false and previous null, got {body!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §4.10")
def test_release_other_without_force_refused(env):
    """§4.10: звільнення чужого без force → release_needs_force; бронювання лишається."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="mine"), "reserve gpu 0 by alice")
        refusal(release(c, 0, "bob"), "release_needs_force")
        reservation = card(c, 0)["reservation"]
    assert reservation is not None and reservation.get("user") == "alice", (
        f"after release_needs_force: expected alice's reservation kept, got {reservation!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §4.10")
def test_release_other_with_force_true(env):
    """§4.10: звільнення чужого з force=true знімає бронювання."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="mine"), "reserve gpu 0 by alice")
        body = ok_json(release(c, 0, "bob", force=True), "forced release by bob")
        reservation = card(c, 0)["reservation"]
    assert body.get("released") is True and reservation is None, (
        f"forced release: expected released true and no reservation, got body {body!r}, reservation {reservation!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §4.10")
@pytest.mark.req("SPEC-GPU-001 §7.2.7")
@pytest.mark.parametrize("force", [False, "true", 1, "1", "yes"], ids=["false", "str-true", "one", "str-one", "yes"])
def test_release_force_must_be_json_true(env, force):
    """§7.2.7: примусово лише при force: true (саме JSON true); інше → release_needs_force."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="mine"), "reserve gpu 0 by alice")
        refusal(release(c, 0, "bob", force=force), "release_needs_force")
        reservation = card(c, 0)["reservation"]
    assert reservation is not None and reservation.get("user") == "alice", (
        f"force={force!r}: expected alice's reservation kept, got {reservation!r}"
    )


# --- §4.11 purpose ---------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §4.11")
def test_purpose_stored_trimmed(env):
    """§4.11: purpose зберігається з обрізаними пробілами на краях."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="   train llm   "), "reserve with padded purpose")
        purpose = card(c, 0)["reservation"]["purpose"]
    assert purpose == "train llm", f"purpose: expected 'train llm', got {purpose!r}"


# --- §4.12–4.13 стан на диску -----------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §4.12")
def test_reservation_survives_restart(env):
    """§4.12: бронювання переживає перезапуск сервісу (новий менеджер над тим самим data_dir)."""
    with env.client() as c:
        ok_json(reserve(c, 2, "carol", purpose="long run", hours=5), "reserve gpu 2 by carol")
        before = _projection(card(c, 2)["reservation"], RESERVATION_FIELDS)
    restarted = env.restart()
    with restarted.client() as c:
        after = _projection(card(c, 2)["reservation"], RESERVATION_FIELDS)
    assert after == before, f"after restart: expected reservation {before!r}, got {after!r}"


@pytest.mark.req("SPEC-GPU-001 §4.12")
def test_reservation_stored_in_state_json(env):
    """§4.12: бронювання зберігаються в data_dir/state.json."""
    with env.client() as c:
        ok_json(reserve(c, 2, "carol"), "reserve gpu 2 by carol")
    state = env.data_dir / "state.json"
    assert state.is_file() and state.stat().st_size > 0, f"expected non-empty {state} after a reservation"


def _start_service(cfg: Any, backend: Any, clock: Any) -> None:
    """Повний старт сервісу: build_manager, build_app, lifespan і один запит."""
    from starlette.testclient import TestClient

    from gpu_manager.app import build_app, build_manager

    manager = build_manager(cfg, backend, clock=clock)
    with TestClient(build_app(cfg, manager), base_url=BASE_URL) as client:
        client.get("/api/overview")


def _exception_names(exc: BaseException) -> list[str]:
    """Імена класів винятку, його причин і вкладених винятків групи (StateError може бути обгорнутий)."""
    names: list[str] = []
    stack: list[BaseException | None] = [exc]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        names.append(type(current).__name__)
        stack.extend(getattr(current, "exceptions", ()) or ())
        stack.extend((current.__cause__, current.__context__))
    return names


CORRUPTED_STATES = [b"", b"{not json", b"[]", b"42", b'"state"', b"null"]
CORRUPTED_IDS = ["empty-file", "not-json", "array", "number", "string", "null"]


@pytest.mark.req("SPEC-GPU-001 §4.13")
@pytest.mark.parametrize("content", CORRUPTED_STATES, ids=CORRUPTED_IDS)
def test_corrupted_state_refuses_to_start(write_config, config_dir, backend, clock, content):
    """§4.13: state.json не JSON або не об'єкт — сервіс не стартує, StateError."""
    from gpu_manager.config import load_config

    cfg = load_config(write_config({"paths.data_dir": str(config_dir / "data")}))
    (config_dir / "data" / "state.json").write_bytes(content)
    try:
        _start_service(cfg, backend, clock)
    except Exception as exc:  # noqa: BLE001 — модуль StateError специфікація не називає
        names = _exception_names(exc)
        assert "StateError" in names, f"state.json={content!r}: expected StateError, got exception chain {names}"
    else:
        pytest.fail(f"state.json={content!r}: expected the service to refuse start with StateError, it started")


@pytest.mark.req("SPEC-GPU-001 §4.13")
def test_corrupted_state_left_intact(write_config, config_dir, backend, clock):
    """§4.13: сервіс не починає з порожнього стану — пошкоджений state.json лишається як був."""
    from gpu_manager.config import load_config

    cfg = load_config(write_config({"paths.data_dir": str(config_dir / "data")}))
    state = config_dir / "data" / "state.json"
    content = b'{"reservations": [truncated'
    state.write_bytes(content)
    try:
        _start_service(cfg, backend, clock)
    except Exception:  # noqa: BLE001 — відмову старту перевіряє test_corrupted_state_refuses_to_start
        pass
    after = state.read_bytes()
    assert after == content, f"corrupted state.json must stay untouched: expected {content!r}, got {after!r}"
