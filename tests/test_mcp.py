"""§8, §10 SPEC-GPU-001: MCP-інструменти — у процесі (build_mcp + call_tool) і через HTTP /mcp.

У процесі фонового опитування немає: телеметрію дає явний GpuManager.tick() (§10). Час у MCP —
рядок YYYY-MM-DD HH:MM у поясі display_timezone (§8); очікуване значення рахує local_minute().
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import pytest

from .conftest import (
    HISTORY_COLUMNS,
    MEM_TOTAL_MIB,
    N_GPUS,
    STRANGER,
    T0,
    TZ_NAME,
    USERS,
    card,
    gpu_proc,
    has_cyrillic,
    local_minute,
    mcp_call,
    mcp_json,
    mcp_refusal,
)

HOUR_S = 3600
# Вісім інструментів таблиці §8; build_mcp без models= (фаза 2) реєструє лише їх.
EXPECTED_TOOLS = {
    "gpu_status", "gpu_free", "gpu_who", "gpu_reserve", "gpu_release", "gpu_history", "gpu_journal", "gpu_guide",
}
SHORT_CARD_FIELDS = {"gpu", "status", "temp_c", "util_pct", "power_w", "mem_mib", "reservation", "processes"}
MCP_ACCEPT = "application/json, text/event-stream"  # обидва типи вимагає Streamable HTTP-транспорт MCP


@pytest.fixture
def server(env):
    """MCP-сервер над менеджером, що вже зробив один tick() (телеметрія є)."""
    env.manager.tick()
    return env.mcp()


def _short_card(server: Any, gpu: int) -> dict[str, Any]:
    gpus = mcp_json(server, "gpu_status", {"gpu": gpu})["gpus"]
    assert len(gpus) == 1, f"gpu_status(gpu={gpu}): expected one card, got {len(gpus)}"
    return gpus[0]


# --- Набір інструментів і формат результату ---------------------------------------------------------------


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8")
def test_mcp_lists_exactly_spec_tools(env):
    """§8, §10: list_tools() повертає рівно вісім інструментів таблиці §8 (з gpu_guide)."""
    tools = anyio.run(env.mcp().list_tools)
    names = {t.name for t in tools}
    assert names == EXPECTED_TOOLS, (
        f"MCP tools: missing {sorted(EXPECTED_TOOLS - names)}, unexpected {sorted(names - EXPECTED_TOOLS)}"
    )


# Інструменти 1–7 §8; gpu_guide (рядок 8) — Markdown, не JSON, його формат перевіряє test_guide.py.
TOOL_CALLS = [
    ("gpu_status", {}),
    ("gpu_status", {"gpu": 0}),
    ("gpu_free", {}),
    ("gpu_who", {"gpu": 1}),
    ("gpu_reserve", {"gpu": 2, "user": "carol", "purpose": "x"}),
    ("gpu_release", {"gpu": 0, "user": "alice"}),
    ("gpu_history", {"gpu": 0, "minutes": 1, "points": 5}),
    ("gpu_journal", {}),
]


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8")
@pytest.mark.parametrize(("name", "args"), TOOL_CALLS, ids=[f"{n}-{i}" for i, (n, _) in enumerate(TOOL_CALLS)])
def test_mcp_result_is_single_text_block_without_newlines(env, backend, name, args):
    """§8: результат кожного інструмента — один текстовий блок з JSON без переносів рядків."""
    # Командний рядок із переносом перевіряє, що JSON не несе сирого \n навіть з таких даних.
    backend.set_processes(1, gpu_proc(4242, "bob", cmdline="python3 -c 'a=1\nb=2'"))
    env.tick_for(30)
    server = env.mcp()
    mcp_json(server, "gpu_reserve", {"gpu": 0, "user": "alice", "purpose": "setup"})
    result = mcp_call(server, name, args)
    blocks = list(result.content)
    assert len(blocks) == 1 and getattr(blocks[0], "type", None) == "text", (
        f"{name}: expected exactly one text block, got {[getattr(b, 'type', type(b).__name__) for b in blocks]}"
    )
    text = blocks[0].text
    assert "\n" not in text, f"{name}: expected JSON without newlines, got {text[:200]!r}"
    try:
        json.loads(text)
    except ValueError as exc:
        pytest.fail(f"{name}: expected a JSON text block, parsing failed ({exc}): {text[:200]!r}")


# --- gpu_status і коротка карта (§8.1) -------------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.1")
def test_gpu_status_all_cards_with_tz(server):
    """§8.1: gpu_status без gpu → {tz, gpus} з усіма картами; tz — назва поясу."""
    data = mcp_json(server, "gpu_status")
    got = (data.get("tz"), [g.get("gpu") for g in data.get("gpus", [])])
    expected = (TZ_NAME, list(range(N_GPUS)))
    assert got == expected, f"gpu_status: expected (tz, gpus) {expected!r}, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.1")
def test_gpu_status_single_card(server):
    """§8.1: gpu_status з gpu → лише ця карта."""
    ids = [g.get("gpu") for g in mcp_json(server, "gpu_status", {"gpu": 2})["gpus"]]
    assert ids == [2], f"gpu_status(gpu=2): expected [2], got {ids!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8")
def test_short_card_has_all_fields(server):
    """§8: коротка карта — {gpu, status, temp_c, util_pct, power_w, mem_mib, reservation, processes}."""
    item = _short_card(server, 0)
    missing = SHORT_CARD_FIELDS - set(item)
    assert not missing, f"short card: missing {sorted(missing)}, got {sorted(item)}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8")
def test_short_card_mem_is_used_total_pair(env, backend, server):
    """§8: mem_mib короткої карти — [used, total]."""
    backend.set_reading(1, memory_used_mib=2048)
    env.manager.tick()
    mem = _short_card(server, 1)["mem_mib"]
    assert mem == [2048, MEM_TOTAL_MIB], f"short card mem_mib: expected [2048, {MEM_TOTAL_MIB}], got {mem!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8")
def test_short_card_process_fields(env, backend, server):
    """§8: процес у короткій карті — {pid, user, name, mib}."""
    backend.set_processes(1, gpu_proc(4242, "bob", used_mib=2048))
    env.manager.tick()
    procs = _short_card(server, 1)["processes"]
    got = [{k: p.get(k) for k in ("pid", "user", "name", "mib")} for p in procs]
    expected = [{"pid": 4242, "user": "bob", "name": "python3", "mib": 2048}]
    assert got == expected, f"short card processes: expected {expected!r}, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8")
def test_short_card_reservation_times_in_display_tz(server):
    """§8: reservation короткої карти — {user, purpose, since, until, expired}, час — рядок у display_timezone."""
    mcp_json(server, "gpu_reserve", {"gpu": 0, "user": "alice", "purpose": "eval", "hours": 2})
    reservation = _short_card(server, 0)["reservation"] or {}
    expected = {
        "user": "alice",
        "purpose": "eval",
        "since": local_minute(T0),
        "until": local_minute(T0 + 2 * HOUR_S),
        "expired": False,
    }
    got = {k: reservation.get(k) for k in expected}
    assert got == expected, f"short card reservation: expected {expected!r}, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8")
def test_short_card_reservation_until_null_without_hours(server):
    """§8, §4.3: бронювання без hours — until = null і в короткій карті."""
    mcp_json(server, "gpu_reserve", {"gpu": 0, "user": "alice"})
    reservation = _short_card(server, 0)["reservation"] or {}
    assert "until" in reservation and reservation["until"] is None, (
        f"reservation without hours: expected until null, got {reservation!r}"
    )


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8")
def test_short_card_error_for_failed_telemetry(env, backend, server):
    """§8: error? — є в короткій карті з помилковим заміром; статус unknown."""
    backend.set_error(3, "NVML: GPU is lost")
    env.manager.tick()
    item = _short_card(server, 3)
    assert item.get("error") and item.get("status") == "unknown", (
        f"card with a failed sample: expected non-empty error and status 'unknown', got {item!r}"
    )


# --- gpu_free (§8.2) ---------------------------------------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.2")
def test_gpu_free_lists_only_free_cards(env, backend, server):
    """§8.2: gpu_free — лише карти зі статусом free (не busy, не reserved, не unknown)."""
    backend.set_processes(1, gpu_proc(4242, "bob"))
    backend.drop_telemetry(3)
    env.manager.tick()
    mcp_json(server, "gpu_reserve", {"gpu": 2, "user": "alice"})
    ids = [item.get("gpu") for item in mcp_json(server, "gpu_free")]
    assert ids == [0], f"gpu_free with 1 busy, 2 reserved, 3 unknown: expected [0], got {ids!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.2")
@pytest.mark.req("SPEC-GPU-001 §3.2")
def test_gpu_free_excludes_expired_reservation(env, server):
    """§3.2, §8.2: прострочене бронювання не робить карту вільною."""
    mcp_json(server, "gpu_reserve", {"gpu": 0, "user": "alice", "hours": 1})
    env.clock.advance(2 * HOUR_S)
    env.manager.tick()  # свіжа телеметрія: застарівання заміру специфікація не визначає
    ids =[item.get("gpu") for item in mcp_json(server, "gpu_free")]
    assert ids == [1, 2, 3], f"gpu 0 with an expired reservation: expected gpu_free [1, 2, 3], got {ids!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.2")
def test_gpu_free_item_fields(server):
    """§8.2: елемент gpu_free — {gpu, mem_total_mib, temp_c}."""
    items = [i for i in mcp_json(server, "gpu_free") if i.get("gpu") == 0]
    got = {k: items[0].get(k) for k in ("gpu", "mem_total_mib", "temp_c")} if items else None
    expected = {"gpu": 0, "mem_total_mib": MEM_TOTAL_MIB, "temp_c": 40.0}
    assert got == expected, f"gpu_free item for gpu 0: expected {expected!r}, got {got!r}"


# --- gpu_who (§8.3) ---------------------------------------------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.3")
def test_gpu_who_has_tz_gpu_and_users(env, backend, server):
    """§8.3: gpu_who → {tz, …коротка карта, users}."""
    backend.set_processes(1, gpu_proc(4242, "bob"))
    env.manager.tick()
    data = mcp_json(server, "gpu_who", {"gpu": 1})
    got = {k: data.get(k) for k in ("tz", "gpu", "users")}
    expected = {"tz": TZ_NAME, "gpu": 1, "users": ["bob"]}
    assert got == expected, f"gpu_who(gpu=1): expected {expected!r}, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.3")
def test_gpu_who_processes_carry_cmd(env, backend, server):
    """§8.3: у gpu_who процеси мають cmd (командний рядок)."""
    backend.set_processes(1, gpu_proc(4242, "bob", cmdline="python3 serve.py --port 8001"))
    env.manager.tick()
    procs = mcp_json(server, "gpu_who", {"gpu": 1}).get("processes") or [{}]
    assert procs[0].get("cmd") == "python3 serve.py --port 8001", (
        f"gpu_who processes: expected cmd 'python3 serve.py --port 8001', got {procs!r}"
    )


# --- gpu_reserve (§8.4) ------------------------------------------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.4")
def test_gpu_reserve_result(server):
    """§8.4: gpu_reserve → {reserved: true, gpu: коротка карта} із бронюванням user."""
    data = mcp_json(server, "gpu_reserve", {"gpu": 2, "user": "alice", "purpose": "eval"})
    gpu_card = data.get("gpu") or {}
    owner = (gpu_card.get("reservation") or {}).get("user")
    assert data.get("reserved") is True and gpu_card.get("gpu") == 2 and gpu_card.get("status") == "reserved" and owner == "alice", (
        f"gpu_reserve: expected reserved true and card 2 reserved by alice, got {data!r}"
    )


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.4")
def test_gpu_reserve_purpose_defaults_to_empty(server):
    """§8.4: purpose="" за замовчуванням."""
    mcp_json(server, "gpu_reserve", {"gpu": 2, "user": "alice"})
    purpose = (_short_card(server, 2)["reservation"] or {}).get("purpose")
    assert purpose == "", f"gpu_reserve without purpose: expected purpose '', got {purpose!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.4")
def test_gpu_reserve_no_warnings_on_clean_card(server):
    """§8.4: warnings? — на карті без чужих процесів попереджень немає (поле відсутнє або порожнє)."""
    data = mcp_json(server, "gpu_reserve", {"gpu": 2, "user": "alice"})
    assert not data.get("warnings"), f"clean card: expected no warnings, got {data.get('warnings')!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.4")
@pytest.mark.req("SPEC-GPU-001 §4.7")
def test_gpu_reserve_warns_on_foreign_processes(env, backend, server):
    """§4.7, §8.4: чужий процес на карті — бронювання є, warnings — непорожній список рядків."""
    backend.set_processes(1, gpu_proc(4242, "bob"))
    env.manager.tick()
    data = mcp_json(server, "gpu_reserve", {"gpu": 1, "user": "alice"})
    warnings = data.get("warnings")
    assert data.get("reserved") is True and isinstance(warnings, list) and warnings and all(isinstance(w, str) for w in warnings), (
        f"reserve over bob's process: expected reserved true and a non-empty list of warning texts, got {data!r}"
    )


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8")
def test_gpu_reserve_warning_text_is_english(env, backend, server):
    """§8: тексти MCP — англійською; попередження без кирилиці."""
    backend.set_processes(1, gpu_proc(4242, "bob"))
    env.manager.tick()
    warnings = mcp_json(server, "gpu_reserve", {"gpu": 1, "user": "alice"}).get("warnings") or []
    cyrillic = [w for w in warnings if has_cyrillic(str(w))]
    assert warnings and not cyrillic, f"MCP warnings: expected English texts, got {warnings!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.4")
@pytest.mark.req("SPEC-GPU-001 §4.2")
@pytest.mark.parametrize("gpu", [True, False], ids=["true", "false"])
def test_gpu_reserve_bool_gpu_refused(server, gpu):
    """§4.2: gpu = true/false → unknown_gpu і через MCP (без приведення bool до 1/0)."""
    message = mcp_refusal(server, "gpu_reserve", {"gpu": gpu, "user": "alice"})
    reserved = [g.get("gpu") for g in mcp_json(server, "gpu_status")["gpus"] if g.get("reservation")]
    assert "unknown_gpu: " in message and reserved == [], (
        f"gpu={gpu!r}: expected ToolError 'unknown_gpu: ...' and no reservations, got {message!r}, reserved {reserved!r}"
    )


# --- gpu_release (§8.5) --------------------------------------------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.5")
def test_gpu_release_own(server):
    """§8.5: gpu_release свого → {released: true, previous_owner, gpu: коротка карта (free)}."""
    mcp_json(server, "gpu_reserve", {"gpu": 0, "user": "alice"})
    data = mcp_json(server, "gpu_release", {"gpu": 0, "user": "alice"})
    got = (data.get("released"), data.get("previous_owner"), (data.get("gpu") or {}).get("status"))
    assert got == (True, "alice", "free"), f"gpu_release own: expected (True, 'alice', 'free'), got {got!r} from {data!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.5")
@pytest.mark.req("SPEC-GPU-001 §4.9")
def test_gpu_release_unreserved(server):
    """§4.9, §8.5: звільнення незаброньованої — released: false, previous_owner: null."""
    data = mcp_json(server, "gpu_release", {"gpu": 3, "user": "alice"})
    assert data.get("released") is False and "previous_owner" in data and data["previous_owner"] is None, (
        f"gpu_release unreserved: expected released false, previous_owner null, got {data!r}"
    )


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.5")
@pytest.mark.req("SPEC-GPU-001 §4.10")
def test_gpu_release_force_defaults_to_false(server):
    """§8.5: force=false за замовчуванням — чуже бронювання без force → release_needs_force."""
    mcp_json(server, "gpu_reserve", {"gpu": 0, "user": "alice"})
    message = mcp_refusal(server, "gpu_release", {"gpu": 0, "user": "bob"})
    owner = (_short_card(server, 0)["reservation"] or {}).get("user")
    assert "release_needs_force: " in message and owner == "alice", (
        f"release without force: expected 'release_needs_force: ...' and alice's reservation kept, "
        f"got {message!r}, owner {owner!r}"
    )


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.5")
@pytest.mark.req("SPEC-GPU-001 §4.10")
def test_gpu_release_force_true(server):
    """§4.10, §8.5: force=true знімає чуже бронювання; previous_owner — чиє було."""
    mcp_json(server, "gpu_reserve", {"gpu": 0, "user": "alice"})
    data = mcp_json(server, "gpu_release", {"gpu": 0, "user": "bob", "force": True})
    assert data.get("released") is True and data.get("previous_owner") == "alice", (
        f"forced gpu_release: expected released true and previous_owner 'alice', got {data!r}"
    )


# --- gpu_history (§8.6) ----------------------------------------------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.6")
def test_gpu_history_shape(env, server):
    """§8.6: gpu_history → {tz, series: {"<gpu>": колонки однакової довжини}}."""
    env.tick_for(60)
    data = mcp_json(server, "gpu_history", {"gpu": 0, "minutes": 1, "points": 10})
    series = data.get("series") or {}
    columns = series.get("0") or {}
    lengths = {col: len(columns.get(col, [])) for col in HISTORY_COLUMNS}
    assert data.get("tz") == TZ_NAME and sorted(series) == ["0"] and len(set(lengths.values())) == 1 and lengths["t"] > 0, (
        f"gpu_history: expected tz {TZ_NAME!r}, series key '0', equal non-zero column lengths; "
        f"got tz {data.get('tz')!r}, keys {sorted(series)}, lengths {lengths}"
    )


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.6")
def test_gpu_history_default_points_is_60(env, server):
    """§8.6: points=60 за замовчуванням — не більше 60 точок з 120 доступних."""
    env.tick_for(120)
    columns = mcp_json(server, "gpu_history", {"gpu": 0})["series"]["0"]
    lengths = {col: len(columns[col]) for col in HISTORY_COLUMNS}
    assert all(0 < n <= 60 for n in lengths.values()), f"gpu_history default points: expected 1..60, got {lengths}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.6")
def test_gpu_history_all_cards_by_default(env, server):
    """§8.6: gpu? не задано — серії всіх карт."""
    env.tick_for(30)
    keys = sorted(mcp_json(server, "gpu_history")["series"])
    expected = [str(i) for i in range(N_GPUS)]
    assert keys == expected, f"gpu_history without gpu: expected keys {expected!r}, got {keys!r}"


# --- gpu_journal (§8.7) -------------------------------------------------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.7")
def test_gpu_journal_ts_is_local_minute_string(server):
    """§8, §8.7: ts запису — рядок YYYY-MM-DD HH:MM у поясі display_timezone."""
    mcp_json(server, "gpu_reserve", {"gpu": 0, "user": "alice", "purpose": "x"})
    entries = mcp_json(server, "gpu_journal")
    ts = entries[0].get("ts") if entries else None
    assert ts == local_minute(T0), f"gpu_journal ts: expected {local_minute(T0)!r} ({TZ_NAME}), got {ts!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8.7")
def test_gpu_journal_default_limit_is_20(env, server):
    """§8.7: limit=20 за замовчуванням."""
    for k in range(22):
        mcp_json(server, "gpu_reserve", {"gpu": 0, "user": "alice", "purpose": f"p{k}"})
        env.clock.advance(1)
    entries = mcp_json(server, "gpu_journal")
    purposes = [e.get("purpose") for e in entries]
    expected = [f"p{k}" for k in range(21, 1, -1)]
    assert purposes == expected, f"gpu_journal default: expected 20 newest {expected[:2]}..{expected[-1:]}, got {purposes!r}"


# --- MCP через HTTP /mcp (§8) ----------------------------------------------------------------------------------------------------------------


def _mcp_http(client: Any, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """POST /mcp без заголовка сесії; повертає (відповідь, JSON-RPC повідомлення з тим самим id)."""
    resp = client.post("/mcp", json=payload, headers={"Accept": MCP_ACCEPT})
    assert resp.status_code == 200, (
        f"POST /mcp {payload.get('method')} without a session: expected HTTP 200, got {resp.status_code}: {resp.text[:300]}"
    )
    if resp.headers.get("content-type", "").startswith("application/json"):
        return resp, resp.json()
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            message = json.loads(line[len("data:"):].strip())
            if message.get("id") == payload["id"]:
                return resp, message
    raise AssertionError(f"POST /mcp: no JSON-RPC message with id {payload['id']} in {resp.text[:300]!r}")


def _tool_call(request_id: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}}


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-001 §8")
def test_mcp_http_initialize_issues_no_session(env):
    """§8: MCP без сесій — відповідь на initialize не видає Mcp-Session-Id."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "spec-test", "version": "0"}},
    }
    with env.client() as c:
        resp, message = _mcp_http(c, payload)
    assert "result" in message and "mcp-session-id" not in resp.headers, (
        f"initialize: expected a result and no Mcp-Session-Id header, got headers {dict(resp.headers)!r}, message {message!r}"
    )


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-001 §8")
def test_mcp_http_tool_call_without_session(env):
    """§8: інструмент викликається через /mcp без сесії й без initialize (перезапуск непомітний клієнтам)."""
    with env.client() as c:
        _, message = _mcp_http(c, _tool_call(2, "gpu_status", {}))
    result = message.get("result") or {}
    text = (result.get("content") or [{}])[0].get("text", "")
    gpus = json.loads(text).get("gpus", []) if text else []
    assert not result.get("isError") and len(gpus) == N_GPUS, (
        f"tools/call gpu_status without a session: expected {N_GPUS} cards, got {message!r}"
    )


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-001 §8")
def test_mcp_http_refusal_sets_is_error(env):
    """§8: відмова через HTTP — результат з isError: true, текст містить "<code>: <English text>"."""
    with env.client() as c:
        _, message = _mcp_http(c, _tool_call(3, "gpu_reserve", {"gpu": 0, "user": STRANGER}))
    result = message.get("result") or {}
    text = (result.get("content") or [{}])[0].get("text", "")
    assert result.get("isError") is True and "unknown_user: " in text, (
        f"tools/call refused over HTTP: expected isError true and 'unknown_user: ...', got {message!r}"
    )


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-001 §8.8")
def test_mcp_http_gpu_guide_is_configured(make_env, tmp_path):
    """§8.8: gpu_guide сервісу (/mcp застосунку build_app) — справжня інструкція, а не "No guide configured."."""
    inbox = tmp_path / "gpu-inbox"
    inbox.mkdir()
    env = make_env({"files.inbox_dir": str(inbox)})
    with env.client() as c:
        _, message = _mcp_http(c, _tool_call(4, "gpu_guide", {}))
    result = message.get("result") or {}
    text = (result.get("content") or [{}])[0].get("text", "")
    missing = [login for login in USERS if login not in text]
    assert "Status: ready." in text and not missing and "No guide configured." not in text, (
        f"gpu_guide over /mcp with an existing inbox: expected the rendered guide with all logins and "
        f"'Status: ready.', missing logins {missing}, got {text[:300]!r}"
    )


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-001 §1")
def test_reservation_via_mcp_visible_over_http(env):
    """§1: два входи з однаковою поведінкою — бронювання через MCP видно в /api/overview."""
    env.manager.tick()
    mcp_json(env.mcp(), "gpu_reserve", {"gpu": 1, "user": "dave", "purpose": "via mcp"})
    with env.client() as c:
        reservation = card(c, 1)["reservation"] or {}
    got = (reservation.get("user"), reservation.get("purpose"))
    assert got == ("dave", "via mcp"), f"reservation made via MCP, seen over HTTP: expected ('dave', 'via mcp'), got {got!r}"
