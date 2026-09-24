"""§7 SPEC-GPU-002: MCP-інструменти моделей — у процесі, build_mcp(manager, tz, models=store).

Формат — SPEC-GPU-001 §8: результат — один текстовий блок з JSON без переносів рядків, тексти англійською,
час — рядок YYYY-MM-DD HH:MM у поясі display_timezone. Відмова в процесі піднімає ToolError, str(exc)
містить "<code>: <English text>" (SPEC-GPU-001 §10).
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import pytest

from .conftest import STRANGER, T0, has_cyrillic, local_minute, mcp_call, mcp_json, mcp_refusal, ok_json
from .model_fakes import (
    GATED_REPO,
    HUGE_RESERVE_GIB,
    MIB,
    MISSING_REPO,
    REPO,
    REPO_B,
    REPO_C,
    TINY_TOTAL,
    append_log,
    gib_close,
    log_path,
    put_snapshot,
)

pytestmark = [pytest.mark.component, pytest.mark.usefixtures("isolated_home")]

PHASE1_TOOLS = {
    "gpu_status", "gpu_free", "gpu_who", "gpu_reserve", "gpu_release", "gpu_history", "gpu_journal", "gpu_guide",
}
MODEL_TOOLS = {
    "hf_search", "hf_model", "model_download", "model_downloads", "model_download_cancel", "models_local", "model_delete",
}
SEARCH_FIELDS = {"repo", "downloads", "gated", "params", "task"}
INFO_FIELDS = {"repo", "revision", "gated", "files", "size_gib", "fraction", "fit", "local"}
DOWNLOAD_FIELDS = {"repo", "status", "percent", "gib", "user", "started"}
LOCAL_FIELDS = {"repo", "gib", "state"}
LOCAL_SIZES = {"model-00001-of-00002.safetensors": 768 * MIB, "model-00002-of-00002.safetensors": 256 * MIB}
FP4_REPO = "acme/fp4-llm"


@pytest.fixture
def server(models_env: Any) -> Any:
    return models_env.mcp()


def _put_local(env: Any, repo: str = REPO_B) -> None:
    put_snapshot(env.hf_home, repo, ["config.json", *LOCAL_SIZES], sizes=LOCAL_SIZES)


def _tool_names(server: Any) -> set[str]:
    return {t.name for t in anyio.run(server.list_tools)}


def _annotations(tool: Any) -> dict[str, Any]:
    """Анотації інструмента в протокольних (camelCase) іменах."""
    dumped = tool.model_dump(by_alias=True) if hasattr(tool, "model_dump") else {}
    return dumped.get("annotations") or {}


# --- Набір інструментів -------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §7")
def test_mcp_with_models_lists_phase1_and_model_tools(models_env):
    """§7: з models=store — інструменти фази 1 і сім інструментів таблиці §7."""
    names = _tool_names(models_env.mcp())
    expected = PHASE1_TOOLS | MODEL_TOOLS
    assert names == expected, f"MCP tools with models: missing {sorted(expected - names)}, unexpected {sorted(names - expected)}"


@pytest.mark.req("SPEC-GPU-002 §7")
def test_mcp_without_models_has_no_model_tools(env):
    """§7: без models інструменти моделей не реєструються."""
    names = _tool_names(env.mcp())
    extra = sorted(names & MODEL_TOOLS)
    assert extra == [], f"MCP tools without models: expected none of {sorted(MODEL_TOOLS)}, got {extra}"


@pytest.mark.req("SPEC-GPU-002 §7.7")
def test_mcp_model_delete_is_destructive(server):
    """§7.7: model_delete має анотацію destructiveHint: true."""
    tools = {t.name: t for t in anyio.run(server.list_tools)}
    tool = tools.get("model_delete")
    hint = _annotations(tool).get("destructiveHint") if tool is not None else "<no model_delete tool>"
    assert hint is True, f"model_delete annotations: expected destructiveHint true, got {hint!r}"


# Виклики всіх семи інструментів з валідними аргументами; стан (REPO качається, REPO_B локальна) готує тест.
RESULT_CALLS: list[tuple[str, dict[str, Any]]] = [
    ("hf_search", {"query": "llama"}),
    ("hf_model", {"repo": REPO}),
    ("model_download", {"repo": REPO_C, "user": "carol"}),
    ("model_downloads", {}),
    ("model_download_cancel", {"repo": REPO, "user": "alice"}),
    ("models_local", {}),
    ("model_delete", {"repo": REPO_B, "user": "bob"}),
]


@pytest.mark.req("SPEC-GPU-002 §7")
@pytest.mark.parametrize(("name", "args"), RESULT_CALLS, ids=[n for n, _ in RESULT_CALLS])
def test_mcp_model_result_is_single_text_block_without_newlines(models_env, name, args):
    """§7 (SPEC-GPU-001 §8): результат — один текстовий блок з JSON без переносів рядків."""
    env = models_env
    _put_local(env, REPO_B)
    env.store.download(REPO, "alice")
    result = mcp_call(env.mcp(), name, args)
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


# --- 1. hf_search -----------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §7.1")
def test_hf_search_items_are_short(server):
    """§7.1: елемент — рівно {repo, downloads, gated, params, task} (без likes, updated)."""
    items = mcp_json(server, "hf_search", {"query": "llama"})
    shapes = [sorted(item) for item in items] if isinstance(items, list) else items
    assert isinstance(items, list) and items and all(set(item) == SEARCH_FIELDS for item in items), (
        f"hf_search items: expected keys {sorted(SEARCH_FIELDS)}, got {shapes!r}"
    )


@pytest.mark.req("SPEC-GPU-002 §7.1")
def test_hf_search_default_limit_ten(server):
    """§7.1: limit=10 за замовчуванням."""
    items = mcp_json(server, "hf_search", {"query": "llama"})
    assert len(items) == 10, f"hf_search default limit: expected 10 items, got {len(items)}"


@pytest.mark.req("SPEC-GPU-002 §7.1")
def test_hf_search_limit_and_order(models_env, server):
    """§7.1: limit=3 — три перші результати hub у його порядку."""
    items = mcp_json(server, "hf_search", {"query": "llama", "limit": 3})
    got = [item.get("repo") for item in items]
    expected = [item["repo"] for item in models_env.hub.pool[:3]]
    assert got == expected, f"hf_search(limit=3): expected {expected!r}, got {got!r}"


# --- 2. hf_model ----------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §7.2")
def test_hf_model_is_info(server):
    """§7.2: hf_model → як info."""
    body = mcp_json(server, "hf_model", {"repo": REPO})
    missing = INFO_FIELDS - set(body)
    assert not missing and body.get("repo") == REPO, f"hf_model: expected info fields {sorted(INFO_FIELDS)} for {REPO!r}, got {body!r}"


@pytest.mark.req("SPEC-GPU-002 §7.2")
def test_hf_model_fraction_argument(server):
    """§7.2: fraction? доходить до info."""
    body = mcp_json(server, "hf_model", {"repo": REPO, "fraction": 0.5})
    assert body.get("fraction") == 0.5, f"hf_model(fraction=0.5): expected fraction 0.5, got {body.get('fraction')!r}"


@pytest.mark.req("SPEC-GPU-002 §7.2")
def test_hf_model_warnings_are_english(models_env, server):
    """§7 (SPEC-GPU-001 §8): тексти англійською — попередження оцінки без кирилиці."""
    models_env.hub.add(FP4_REPO, config={"quantization_config": {"quant_method": "nvfp4"}})
    body = mcp_json(server, "hf_model", {"repo": FP4_REPO})
    warnings = (body.get("fit") or {}).get("warnings")
    text = json.dumps(body, ensure_ascii=False)
    assert warnings and not has_cyrillic(text), f"hf_model for an nvfp4 model: expected English warnings, got {warnings!r}"


# --- 3–5. model_download / model_downloads / model_download_cancel ------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §7.3")
def test_model_download_result_fields(server):
    """§7.3: {repo, status, percent, gib: [done, total], user, started, error?}."""
    body = mcp_json(server, "model_download", {"repo": REPO, "user": "alice"})
    missing = DOWNLOAD_FIELDS - set(body)
    assert not missing, f"model_download: missing fields {sorted(missing)}, got {body!r}"


@pytest.mark.req("SPEC-GPU-002 §7.3")
def test_model_download_values(server):
    """§7.3: repo, user і стан після кроку черги (downloading)."""
    body = mcp_json(server, "model_download", {"repo": REPO, "user": "alice"})
    got = (body.get("repo"), body.get("user"), body.get("status"))
    assert got == (REPO, "alice", "downloading"), f"model_download: expected ({REPO!r}, 'alice', 'downloading'), got {body!r}"


@pytest.mark.req("SPEC-GPU-002 §7.3")
def test_model_download_gib_is_done_total_pair(server):
    """§7.3: gib — [done, total]; на старті done ≈ 0, total ≈ сума файлів."""
    gib = mcp_json(server, "model_download", {"repo": REPO, "user": "alice"}).get("gib")
    ok = isinstance(gib, list) and len(gib) == 2 and gib_close(gib[0], 0) and gib_close(gib[1], TINY_TOTAL)
    assert ok, f"model_download gib: expected [≈0, ≈{TINY_TOTAL / 2**30:.3f}], got {gib!r}"


@pytest.mark.req("SPEC-GPU-002 §7.3")
def test_model_download_started_is_local_minute(server):
    """§7.3 (SPEC-GPU-001 §8): started — рядок YYYY-MM-DD HH:MM у display_timezone."""
    got = mcp_json(server, "model_download", {"repo": REPO, "user": "alice"}).get("started")
    assert got == local_minute(T0), f"model_download started: expected {local_minute(T0)!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §7.3")
def test_model_download_no_error_when_fine(server):
    """§7.3: error? — у справного завантаження помилки немає."""
    body = mcp_json(server, "model_download", {"repo": REPO, "user": "alice"})
    assert not body.get("error"), f"model_download of a healthy download: expected no error, got {body.get('error')!r}"


@pytest.mark.req("SPEC-GPU-002 §7.4")
def test_model_downloads_list_newest_first(models_env, server):
    """§7.4: model_downloads — список того ж формату, найновіші першими."""
    mcp_json(server, "model_download", {"repo": REPO, "user": "alice"})
    models_env.clock.advance(60)
    mcp_json(server, "model_download", {"repo": REPO_B, "user": "bob"})
    items = mcp_json(server, "model_downloads")
    got = [item.get("repo") for item in items]
    shapes_ok = all(DOWNLOAD_FIELDS <= set(item) for item in items)
    assert got == [REPO_B, REPO] and shapes_ok, f"model_downloads: expected [{REPO_B!r}, {REPO!r}] with fields {sorted(DOWNLOAD_FIELDS)}, got {items!r}"


@pytest.mark.req("SPEC-GPU-002 §7.4")
def test_model_downloads_error_on_failure(models_env, server):
    """§7.3–7.4: error? — у невдалого завантаження є непорожній текст помилки."""
    env = models_env
    mcp_json(server, "model_download", {"repo": REPO, "user": "alice"})
    append_log(log_path(env.data_dir, REPO), ["ERROR ValueError: unexpected response from the hub"])
    env.spawn.last(REPO).finish(1)
    env.store.poll()
    items = [i for i in mcp_json(server, "model_downloads") if i.get("repo") == REPO]
    error = items[0].get("error") if items else "<no item>"
    assert isinstance(error, str) and error.strip(), f"model_downloads for a failed download: expected an error text, got {items!r}"


@pytest.mark.req("SPEC-GPU-002 §7.5")
def test_model_download_cancel_result(server):
    """§7.5: model_download_cancel → той самий формат, status cancelled."""
    mcp_json(server, "model_download", {"repo": REPO, "user": "alice"})
    body = mcp_json(server, "model_download_cancel", {"repo": REPO, "user": "alice"})
    got = (body.get("repo"), body.get("status"), DOWNLOAD_FIELDS <= set(body))
    assert got == (REPO, "cancelled", True), f"model_download_cancel: expected ({REPO!r}, 'cancelled', all fields), got {body!r}"


# --- 6–7. models_local / model_delete -------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §7.6")
def test_models_local_items_are_short(models_env, server):
    """§7.6: models_local → [{repo, gib, state}] — рівно ці поля."""
    _put_local(models_env, REPO_B)
    items = mcp_json(server, "models_local")
    assert isinstance(items, list) and items and all(set(i) == LOCAL_FIELDS for i in items), (
        f"models_local: expected items with keys {sorted(LOCAL_FIELDS)}, got {items!r}"
    )


@pytest.mark.req("SPEC-GPU-002 §7.6")
def test_models_local_values(models_env, server):
    """§7.6: repo, gib ≈ розмір на диску, state ready."""
    _put_local(models_env, REPO_B)
    items = mcp_json(server, "models_local")
    got = [(i.get("repo"), gib_close(i.get("gib"), sum(LOCAL_SIZES.values())), i.get("state")) for i in items]
    assert got == [(REPO_B, True, "ready")], f"models_local: expected [({REPO_B!r}, gib≈1.0, 'ready')], got {items!r}"


@pytest.mark.req("SPEC-GPU-002 §7.7")
def test_model_delete_result(models_env, server):
    """§7.7: model_delete → {deleted, repo, freed_gib}."""
    _put_local(models_env, REPO_B)
    body = mcp_json(server, "model_delete", {"repo": REPO_B, "user": "bob"})
    ok = body.get("deleted") is True and body.get("repo") == REPO_B and gib_close(body.get("freed_gib"), sum(LOCAL_SIZES.values()))
    assert ok, f"model_delete: expected {{'deleted': True, 'repo': {REPO_B!r}, 'freed_gib': ≈1.0}}, got {body!r}"


# --- Відмови (§8; формат — SPEC-GPU-001 §8, §10) -----------------------------------------------------------------------------------------------------


def _delete_active(env: Any, server: Any) -> str:
    env.store.download(REPO, "alice")
    put_snapshot(env.hf_home, REPO, ["config.json"])
    return mcp_refusal(server, "model_delete", {"repo": REPO, "user": "alice"})


def _search_hub_down(env: Any, server: Any) -> str:
    env.hub.unavailable = True
    return mcp_refusal(server, "hf_search", {"query": "llama"})


# case → (код, перевизначення конфігу, дія над (env, server), що повертає str(ToolError)).
MCP_REFUSALS: dict[str, tuple[str, dict[str, Any], Any]] = {
    "download-unknown-user": (
        "unknown_user", {}, lambda env, s: mcp_refusal(s, "model_download", {"repo": REPO, "user": STRANGER}),
    ),
    "download-gated": ("hf_gated", {}, lambda env, s: mcp_refusal(s, "model_download", {"repo": GATED_REPO, "user": "alice"})),
    "model-not-found": ("hf_not_found", {}, lambda env, s: mcp_refusal(s, "hf_model", {"repo": MISSING_REPO})),
    "search-hub-down": ("hf_unavailable", {}, _search_hub_down),
    "model-bad-fraction": ("bad_fraction", {}, lambda env, s: mcp_refusal(s, "hf_model", {"repo": REPO, "fraction": 0})),
    "download-disk-full": (
        "disk_full",
        {"models.min_free_disk_gib": HUGE_RESERVE_GIB},
        lambda env, s: mcp_refusal(s, "model_download", {"repo": REPO, "user": "alice"}),
    ),
    "delete-active": ("download_active", {}, _delete_active),
    "cancel-not-active": (
        "download_not_active", {}, lambda env, s: mcp_refusal(s, "model_download_cancel", {"repo": REPO, "user": "alice"}),
    ),
    "delete-not-local": ("model_not_local", {}, lambda env, s: mcp_refusal(s, "model_delete", {"repo": REPO, "user": "alice"})),
}


def _mcp_refuse(make_models_env: Any, case: str) -> tuple[str, str]:
    code, overrides, trigger = MCP_REFUSALS[case]
    env = make_models_env(overrides)
    return code, trigger(env, env.mcp())


@pytest.mark.req("SPEC-GPU-002 §7")
@pytest.mark.req("SPEC-GPU-002 §8")
@pytest.mark.parametrize("case", sorted(MCP_REFUSALS))
def test_mcp_model_refusal_has_code(make_models_env, case):
    """§7, §8 (SPEC-GPU-001 §10): відмова піднімає ToolError; текст містить "<code>: <текст>"."""
    code, message = _mcp_refuse(make_models_env, case)
    tail = message.split(f"{code}: ", 1)[1] if f"{code}: " in message else ""
    assert tail.strip(), f"{case}: expected ToolError text containing '{code}: <text>', got {message!r}"


@pytest.mark.req("SPEC-GPU-002 §7")
@pytest.mark.parametrize("case", sorted(MCP_REFUSALS))
def test_mcp_model_refusal_text_is_english(make_models_env, case):
    """§7 (SPEC-GPU-001 §8): тексти MCP — англійською; у відмові немає кирилиці."""
    code, message = _mcp_refuse(make_models_env, case)
    assert f"{code}: " in message and not has_cyrillic(message), f"{case}: expected an English '{code}: <text>' message, got {message!r}"


@pytest.mark.req("SPEC-GPU-002 §7")
@pytest.mark.req("SPEC-GPU-002 §6")
def test_mcp_download_visible_over_http(models_env):
    """§6–7: два входи з однаковою поведінкою — завантаження через MCP видно в GET /api/models."""
    env = models_env
    mcp_json(env.mcp(), "model_download", {"repo": REPO, "user": "dave"})
    with env.client() as c:
        body = ok_json(c.get("/api/models"), "GET /api/models")
    got = [(d.get("repo"), d.get("user")) for d in body.get("downloads", [])]
    assert got == [(REPO, "dave")], f"download made via MCP, seen over HTTP: expected [({REPO!r}, 'dave')], got {got!r}"
