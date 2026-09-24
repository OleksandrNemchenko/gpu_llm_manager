"""§9 SPEC-GPU-001: коди відмов — конверт HTTP (§7.2) і текст ToolError MCP (§8, §10).

Кожен код викликається найпростішим сценарієм із таблиці §9; тут перевіряється форма відмови
(статус, ключі тіла, мова тексту), а сама поведінка — у тестах розділів §4, §6.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from .conftest import (
    N_GPUS,
    STRANGER,
    has_cyrillic,
    mcp_json,
    mcp_refusal,
    ok_json,
    release,
    reserve,
)


def _http_reserved_by_other(c: Any) -> Any:
    ok_json(reserve(c, 0, "alice"), "reserve gpu 0 by alice")
    return reserve(c, 0, "bob")


def _http_release_needs_force(c: Any) -> Any:
    ok_json(reserve(c, 0, "alice"), "reserve gpu 0 by alice")
    return release(c, 0, "bob")


HTTP_REFUSALS: dict[str, Callable[[Any], Any]] = {
    "unknown_gpu": lambda c: reserve(c, N_GPUS, "alice"),
    "unknown_user": lambda c: reserve(c, 0, STRANGER),
    "reserved_by_other": _http_reserved_by_other,
    "release_needs_force": _http_release_needs_force,
    "bad_hours": lambda c: reserve(c, 0, "alice", hours=0),
    "bad_history": lambda c: c.get("/api/history", params={"minutes": 0}),
    "bad_request": lambda c: c.post("/api/reserve", content=b"{not json", headers={"Content-Type": "application/json"}),
}


def _refuse_over_http(env: Any, code: str) -> Any:
    with env.client() as c:
        return HTTP_REFUSALS[code](c)


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-001 §9")
@pytest.mark.req("SPEC-GPU-001 §7.2")
@pytest.mark.parametrize("code", sorted(HTTP_REFUSALS))
def test_http_refusal_is_400_with_code(env, code):
    """§7.2, §9: відмова — HTTP 400, у тілі поле code з кодом таблиці §9."""
    resp = _refuse_over_http(env, code)
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else None
    assert resp.status_code == 400 and isinstance(body, dict) and body.get("code") == code, (
        f"{code}: expected HTTP 400 with code {code!r}, got {resp.status_code}: {resp.text[:300]}"
    )


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-001 §7.2")
@pytest.mark.parametrize("code", sorted(HTTP_REFUSALS))
def test_http_refusal_body_has_only_code_and_error(env, code):
    """§7.2: тіло відмови — {"code": ..., "error": ...}."""
    body = _refuse_over_http(env, code).json()
    keys = sorted(body) if isinstance(body, dict) else body
    assert keys == ["code", "error"], f"{code}: expected body keys ['code', 'error'], got {keys!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-001 §7.2")
@pytest.mark.parametrize("code", sorted(HTTP_REFUSALS))
def test_http_refusal_error_text_is_ukrainian(env, code):
    """§7.2: error — текст українською (непорожній, містить кирилицю)."""
    body = _refuse_over_http(env, code).json()
    text = body.get("error") if isinstance(body, dict) else None
    assert isinstance(text, str) and has_cyrillic(text), f"{code}: expected a Ukrainian error text, got {text!r}"


def _mcp_reserved_by_other(server: Any) -> str:
    mcp_json(server, "gpu_reserve", {"gpu": 0, "user": "alice"})
    return mcp_refusal(server, "gpu_reserve", {"gpu": 0, "user": "bob"})


def _mcp_release_needs_force(server: Any) -> str:
    mcp_json(server, "gpu_reserve", {"gpu": 0, "user": "alice"})
    return mcp_refusal(server, "gpu_release", {"gpu": 0, "user": "bob"})


MCP_REFUSALS: dict[str, tuple[str, Callable[[Any], str]]] = {
    "reserve-unknown-gpu": ("unknown_gpu", lambda s: mcp_refusal(s, "gpu_reserve", {"gpu": N_GPUS, "user": "alice"})),
    "who-unknown-gpu": ("unknown_gpu", lambda s: mcp_refusal(s, "gpu_who", {"gpu": N_GPUS})),
    "status-unknown-gpu": ("unknown_gpu", lambda s: mcp_refusal(s, "gpu_status", {"gpu": N_GPUS})),
    "unknown-user": ("unknown_user", lambda s: mcp_refusal(s, "gpu_reserve", {"gpu": 0, "user": STRANGER})),
    "reserved-by-other": ("reserved_by_other", _mcp_reserved_by_other),
    "release-needs-force": ("release_needs_force", _mcp_release_needs_force),
    "bad-hours": ("bad_hours", lambda s: mcp_refusal(s, "gpu_reserve", {"gpu": 0, "user": "alice", "hours": 0})),
    "bad-history-minutes": ("bad_history", lambda s: mcp_refusal(s, "gpu_history", {"minutes": 0})),
    "bad-history-points": ("bad_history", lambda s: mcp_refusal(s, "gpu_history", {"points": 0})),
}


@pytest.fixture
def mcp_server(env):
    env.manager.tick()
    return env.mcp()


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §9")
@pytest.mark.req("SPEC-GPU-001 §10")
@pytest.mark.parametrize("case", sorted(MCP_REFUSALS))
def test_mcp_refusal_raises_tool_error_with_code(mcp_server, case):
    """§8, §10: відмова в процесі піднімає ToolError; str(exc) містить "<code>: <текст>"."""
    code, trigger = MCP_REFUSALS[case]
    message = trigger(mcp_server)
    tail = message.split(f"{code}: ", 1)[1] if f"{code}: " in message else ""
    assert tail.strip(), f"{case}: expected ToolError text containing '{code}: <text>', got {message!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §8")
@pytest.mark.parametrize("case", sorted(MCP_REFUSALS))
def test_mcp_refusal_text_is_english(mcp_server, case):
    """§8: тексти MCP — англійською; у тексті відмови немає кирилиці."""
    code, trigger = MCP_REFUSALS[case]
    message = trigger(mcp_server)
    assert f"{code}: " in message and not has_cyrillic(message), (
        f"{case}: expected an English '{code}: <text>' message, got {message!r}"
    )


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-001 §10")
def test_manager_error_class_importable():
    """§10: відмови в Python — gpu_manager.messages.ManagerError (клас винятку)."""
    from gpu_manager.messages import ManagerError

    assert isinstance(ManagerError, type) and issubclass(ManagerError, Exception), (
        f"gpu_manager.messages.ManagerError: expected an Exception subclass, got {ManagerError!r}"
    )
