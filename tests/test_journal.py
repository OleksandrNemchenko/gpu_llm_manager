"""§5 SPEC-GPU-001: журнал дій — поля записів, що пишеться й що ні, порядок, файл, обидві межі зберігання.

Журнал читається через GET /api/journal (§7.2.4). Записи розрізняються за purpose: бронювання
тієї самої карти тим самим користувачем дає reserve, а далі reserve_update — обидва з purpose.
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import (
    N_GPUS,
    STRANGER,
    T0,
    journal,
    ok_json,
    refusal,
    release,
    reserve,
)

pytestmark = pytest.mark.e2e

HOUR_S = 3600
DAY_S = 86400


def _purposes(entries: list[dict[str, Any]]) -> list[Any]:
    return [e.get("purpose") for e in entries]


def _project(entry: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: entry.get(k) for k in keys}


def _log_entries(env: Any, client: Any, count: int, gpu: int = 0, prefix: str = "p") -> None:
    """Пише count записів журналу: alice бронює gpu з purpose <prefix>0..; між записами +1 с."""
    for k in range(count):
        ok_json(reserve(client, gpu, "alice", purpose=f"{prefix}{k}"), f"reserve #{k}")
        env.clock.advance(1)


# --- Поля записів (§5.1–5.4) ------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §5.1")
def test_journal_reserve_entry(env):
    """§5.1: нове бронювання → {ts, user, action: reserve, gpu, purpose, until}."""
    with env.client() as c:
        ok_json(reserve(c, 1, "alice", purpose="job", hours=2), "reserve gpu 1")
        entries = journal(c)
    expected = {"ts": T0, "user": "alice", "action": "reserve", "gpu": 1, "purpose": "job", "until": T0 + 2 * HOUR_S}
    assert entries, "journal: expected 1 entry after a reservation, got none"
    got = _project(entries[0], tuple(expected))
    assert got == expected, f"reserve entry: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-001 §5.2")
def test_journal_reserve_update_entry(env):
    """§5.2: повторне бронювання тим самим користувачем → action reserve_update з purpose і until."""
    with env.client() as c:
        ok_json(reserve(c, 1, "alice", purpose="a", hours=1), "first reserve")
        env.clock.advance(60)
        ok_json(reserve(c, 1, "alice", purpose="b", hours=2), "second reserve")
        entries = journal(c)
    expected = {"user": "alice", "action": "reserve_update", "gpu": 1, "purpose": "b", "until": T0 + 60 + 2 * HOUR_S}
    got = _project(entries[0], tuple(expected))
    assert got == expected, f"reserve_update entry: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-001 §5.3")
def test_journal_release_entry(env):
    """§5.3: звільнення свого → action release."""
    with env.client() as c:
        ok_json(reserve(c, 1, "alice", purpose="a"), "reserve")
        env.clock.advance(5)
        ok_json(release(c, 1, "alice"), "release own")
        entries = journal(c)
    expected = {"user": "alice", "action": "release", "gpu": 1}
    got = _project(entries[0], tuple(expected))
    assert got == expected, f"release entry: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-001 §5.4")
def test_journal_release_forced_entry(env):
    """§5.4: звільнення чужого з force → release_forced, user — хто зняв, owner — чиє було, purpose."""
    with env.client() as c:
        ok_json(reserve(c, 1, "alice", purpose="a"), "reserve by alice")
        env.clock.advance(10)
        ok_json(release(c, 1, "bob", force=True), "forced release by bob")
        entries = journal(c)
    expected = {"user": "bob", "action": "release_forced", "gpu": 1, "owner": "alice", "purpose": "a"}
    got = _project(entries[0], tuple(expected))
    assert got == expected, f"release_forced entry: expected {expected!r}, got {got!r}"


# --- Що не пишеться (§5) ------------------------------------------------------------------------------


REFUSED_ACTIONS = {
    "unknown_user": lambda c: reserve(c, 1, STRANGER),
    "unknown_gpu": lambda c: reserve(c, N_GPUS, "alice"),
    "reserved_by_other": lambda c: reserve(c, 0, "bob"),
    "release_needs_force": lambda c: release(c, 0, "bob"),
    "bad_hours": lambda c: reserve(c, 1, "alice", hours=0),
    "bad_request": lambda c: c.post("/api/reserve", json={"gpu": 1}),
}


@pytest.mark.req("SPEC-GPU-001 §5")
@pytest.mark.parametrize("code", sorted(REFUSED_ACTIONS))
def test_journal_refusal_not_logged(env, code):
    """§5: відмови в журнал не пишуться."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="base"), "reserve gpu 0 by alice")
        before = journal(c)
        env.clock.advance(1)
        refusal(REFUSED_ACTIONS[code](c), code)
        after = journal(c)
    assert after == before, f"after a {code} refusal: expected journal unchanged {before!r}, got {after!r}"


@pytest.mark.req("SPEC-GPU-001 §5")
def test_journal_release_false_not_logged(env):
    """§5: released: false (звільнення незаброньованої карти) у журнал не пишеться."""
    with env.client() as c:
        body = ok_json(release(c, 2, "alice"), "release unreserved gpu 2")
        entries = journal(c)
    assert body.get("released") is False, f"precondition: expected released false, got {body!r}"
    assert entries == [], f"after released: false: expected an empty journal, got {entries!r}"


# --- Порядок і limit (§5, §7.2.4) ------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §5")
def test_journal_newest_first(env):
    """§5: журнал повертається від найновішого запису."""
    with env.client() as c:
        for gpu, purpose in enumerate(["first", "second", "third"]):
            ok_json(reserve(c, gpu, "alice", purpose=purpose), f"reserve gpu {gpu}")
            env.clock.advance(1)
        order = _purposes(journal(c))
    assert order == ["third", "second", "first"], f"journal order: expected newest first, got {order!r}"


@pytest.mark.req("SPEC-GPU-001 §7.2.4")
def test_journal_limit_returns_newest(env):
    """§7.2.4: limit=N — N найновіших записів."""
    with env.client() as c:
        _log_entries(env, c, 3)
        order = _purposes(journal(c, limit=2))
    assert order == ["p2", "p1"], f"journal?limit=2: expected ['p2', 'p1'], got {order!r}"


@pytest.mark.req("SPEC-GPU-001 §7.2.4")
def test_journal_default_limit_is_30(env):
    """§7.2.4: без limit повертається 30 найновіших записів."""
    with env.client() as c:
        _log_entries(env, c, 32)
        order = _purposes(journal(c))
    expected = [f"p{k}" for k in range(31, 1, -1)]
    assert order == expected, f"journal without limit: expected 30 newest {expected[:2]}..{expected[-1:]}, got {order!r}"


# --- Файл (§5) -------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §5")
def test_journal_file_in_data_dir(env):
    """§5: журнал пишеться у data_dir/journal.jsonl."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="x"), "reserve")
    path = env.data_dir / "journal.jsonl"
    assert path.is_file() and path.stat().st_size > 0, f"expected non-empty {path} after a logged action"


@pytest.mark.req("SPEC-GPU-001 §5")
def test_journal_survives_restart(env):
    """§5: записи журналу зберігаються у файлі й видні після перезапуску."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="before restart"), "reserve")
    restarted = env.restart()
    with restarted.client() as c:
        order = _purposes(journal(c))
    assert order == ["before restart"], f"after restart: expected ['before restart'], got {order!r}"


def _tear_journal(env: Any, newline_after: bool) -> None:
    """Дописує в journal.jsonl обірвану половину першого рядка — слід аварійної зупинки."""
    path = env.data_dir / "journal.jsonl"
    raw = path.read_bytes()
    first = raw.splitlines()[0]
    torn = first[: len(first) // 2]
    path.write_bytes(raw + torn + (b"\n" if newline_after else b""))


@pytest.mark.req("SPEC-GPU-001 §5")
def test_journal_torn_last_line_hides_nothing(env):
    """§5: обірваний останній рядок (без \\n) не ховає ні попередній запис, ні наступний після рестарту."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="before crash"), "reserve before crash")
    _tear_journal(env, newline_after=False)
    restarted = env.restart()
    restarted.clock.advance(10)
    with restarted.client() as c:
        ok_json(release(c, 0, "alice"), "release after restart")
        actions = [e.get("action") for e in journal(c, limit=100)]
    assert actions == ["release", "reserve"], (
        f"torn trailing line: expected ['release', 'reserve'] (both around the torn line), got {actions!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §5")
def test_journal_torn_middle_line_hides_nothing(env):
    """§5: обірваний рядок посередині файлу не ховає ні попередні, ні наступні записи."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="before crash"), "reserve before crash")
    _tear_journal(env, newline_after=True)
    restarted = env.restart()
    restarted.clock.advance(10)
    with restarted.client() as c:
        ok_json(release(c, 0, "alice"), "release after restart")
        actions = [e.get("action") for e in journal(c, limit=100)]
    assert actions == ["release", "reserve"], (
        f"torn middle line: expected ['release', 'reserve'], got {actions!r}"
    )


# --- Межа journal.keep_entries (§5) -------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §5")
def test_journal_keep_entries_keeps_only_newest(make_env):
    """§5: лишаються лише останні journal.keep_entries записів."""
    env = make_env({"journal.keep_entries": 3})
    with env.client() as c:
        _log_entries(env, c, 5)
        order = _purposes(journal(c, limit=100))
    assert order == ["p4", "p3", "p2"], f"keep_entries=3 after 5 entries: expected ['p4', 'p3', 'p2'], got {order!r}"


@pytest.mark.req("SPEC-GPU-001 §5")
def test_journal_keep_entries_dropped_not_back_after_restart(make_env):
    """§5: запис поза межею keep_entries видаляється і не повертається після перезапуску."""
    env = make_env({"journal.keep_entries": 3})
    with env.client() as c:
        _log_entries(env, c, 5)
    restarted = env.restart()
    with restarted.client() as c:
        order = _purposes(journal(c, limit=100))
    assert order == ["p4", "p3", "p2"], f"keep_entries=3 after restart: expected ['p4', 'p3', 'p2'], got {order!r}"


# --- Межа journal.keep_days (§5) ---------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §5")
def test_journal_keep_days_boundary_kept(make_env):
    """§5: запис з ts = now − keep_days·86400 ще лишається (нерівність ≥ включна)."""
    env = make_env({"journal.keep_days": 1})
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="old"), "reserve at T0")
        env.clock.advance(DAY_S)
        order = _purposes(journal(c, limit=100))
    assert order == ["old"], f"exactly keep_days old: expected ['old'] kept, got {order!r}"


@pytest.mark.req("SPEC-GPU-001 §5")
def test_journal_keep_days_older_not_returned(make_env):
    """§5: запис, старший за keep_days днів, не повертається — навіть без нових записів."""
    env = make_env({"journal.keep_days": 1})
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="old"), "reserve at T0")
        env.clock.advance(DAY_S + 1)
        order = _purposes(journal(c, limit=100))
    assert order == [], f"keep_days=1, entry 86401 s old: expected an empty journal, got {order!r}"


@pytest.mark.req("SPEC-GPU-001 §5")
def test_journal_keep_days_keeps_fresh_entries(make_env):
    """§5: межа keep_days прибирає старий запис і лишає свіжі."""
    env = make_env({"journal.keep_days": 1})
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="old"), "reserve at T0")
        env.clock.advance(DAY_S + 1)
        ok_json(reserve(c, 1, "alice", purpose="new"), "reserve a day later")
        order = _purposes(journal(c, limit=100))
    assert order == ["new"], f"keep_days=1: expected only ['new'], got {order!r}"


@pytest.mark.req("SPEC-GPU-001 §5")
def test_journal_keep_days_dropped_not_back_after_restart(make_env):
    """§5: запис поза межею keep_days не повертається після перезапуску."""
    env = make_env({"journal.keep_days": 1})
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="old"), "reserve at T0")
        env.clock.advance(DAY_S + 1)
        ok_json(reserve(c, 1, "alice", purpose="new"), "reserve a day later")
    restarted = env.restart()
    with restarted.client() as c:
        order = _purposes(journal(c, limit=100))
    assert order == ["new"], f"keep_days=1 after restart: expected only ['new'], got {order!r}"


@pytest.mark.req("SPEC-GPU-001 §5")
def test_journal_either_limit_removes(make_env):
    """§5: запис поза БУДЬ-ЯКОЮ межею видаляється — старий за днями й зайвий за кількістю разом."""
    env = make_env({"journal.keep_entries": 3, "journal.keep_days": 1})
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="stale"), "reserve at T0")
        env.clock.advance(DAY_S + 1)
        _log_entries(env, c, 4, gpu=1, prefix="n")
        order = _purposes(journal(c, limit=100))
    assert order == ["n3", "n2", "n1"], (
        f"keep_entries=3, keep_days=1: expected ['n3', 'n2', 'n1'] (stale by age, n0 by count), got {order!r}"
    )
