"""§4 SPEC-GPU-003: HTTP-маршрути серверів моделей і MCP-інструменти фаз 3–4; поле models картки GPU (§2.12).

HTTP — starlette TestClient над build_app(cfg, manager, models, runner, prompts) у режимі контекстного
менеджера (lifespan працює; захист і формат відмов — SPEC-GPU-001 §7). MCP — у процесі, build_mcp(...,
runner=, prompts=) (SPEC-GPU-001 §10): відмова піднімає ToolError, str(exc) містить "<code>: <English text>".
Стан серверів готується напряму через ModelRunner до відкриття клієнта; підроблені юніти живі, health
незапущених не відповідає, тож фонове опитування сервісу під час тесту стану не змінює.
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import pytest

from .conftest import card, has_cyrillic, mcp_call, mcp_json, mcp_refusal, mcp_text, ok_json, refusal
from .model_fakes import REPO
from .runner_fakes import (
    DEFAULT_NAME,
    GPU,
    GPU_B,
    LINE_INFO,
    LINE_REMOTE_CODE,
    NOT_LOCAL_REPO,
    PORT_FIRST,
    STUB_ANSWER,
    fraction_of,
    max_len_of,
    port_of,
)

pytestmark = pytest.mark.usefixtures("isolated_home")

SERVERS = "/api/servers"
START = "/api/servers/start"
STOP = "/api/servers/stop"
LOGS = "/api/servers/logs"
MOVE = "/api/servers/move"
PORTS = "/api/servers/ports"
NAME = DEFAULT_NAME

PHASE1_TOOLS = {
    "gpu_status", "gpu_free", "gpu_who", "gpu_reserve", "gpu_release", "gpu_history", "gpu_journal", "gpu_guide",
}
MODEL_TOOLS = {
    "hf_search", "hf_model", "model_download", "model_downloads", "model_download_cancel", "models_local", "model_delete",
}
RUNNER_TOOLS = {
    "model_start", "model_move", "model_stop", "models_running", "model_logs",
    "prompt_save", "prompts_list", "prompt_delete", "llm_ask",
}
CODER = "coder"
CODER_TEXT = "You are a strict Python reviewer. Point out bugs first, style last."


def _server_names(body: Any) -> list[Any]:
    """Назви серверів з відповіді GET /api/servers.

    Форму відповіді §4 не задає (прогалина): приймається список серверів (як servers()) або {servers: [...]}.
    """
    items = body.get("servers") if isinstance(body, dict) else body
    assert isinstance(items, list), f"GET /api/servers: expected a JSON list of servers (or {{servers: [...]}}), got {body!r}"
    return [item.get("name") for item in items]


def _tool_names(server: Any) -> set[str]:
    return {t.name for t in anyio.run(server.list_tools)}


# --- HTTP ------------------------------------------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §4")
def test_http_start_launches_unit(runner_env):
    """§4: POST /api/servers/start {repo, gpu, user} → 200 і запуск юніта gm-model-<name>."""
    env = runner_env
    with env.client() as c:
        ok_json(c.post(START, json={"repo": REPO, "gpu": GPU, "user": "alice"}), "POST /api/servers/start")
    got = len(env.starts(NAME))
    assert got == 1, f"after POST {START}: expected 1 start of {env.unit(NAME)!r}, got {got}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §4")
def test_http_start_passes_optional_fields(runner_env):
    """§4: fraction?, max_model_len?, extra_args?, name?, port? з тіла доходять до start()."""
    env = runner_env
    body = {
        "repo": REPO, "gpu": GPU, "user": "alice", "port": PORT_FIRST + 42,
        "fraction": 0.5, "max_model_len": 4096, "extra_args": ["--enable-prefix-caching"], "name": "coder",
    }
    with env.client() as c:
        ok_json(c.post(START, json=body), "POST /api/servers/start")
    argv = env.argv("coder")
    got = (fraction_of(argv), max_len_of(argv), argv[-1], port_of(argv))
    expected = (0.5, 4096, "--enable-prefix-caching", PORT_FIRST + 42)
    assert got == expected, f"argv of 'coder' started over HTTP: expected (fraction, max_model_len, last arg, port) {expected!r}, got {argv!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §4")
@pytest.mark.req("SPEC-GPU-003 §2.2")
def test_http_ports_is_free_ports(make_runner_env):
    """§4: GET /api/servers/ports → free_ports(): діапазон 8000..8003, 8000 зайнятий моделлю, 8002 не вільний."""
    env = make_runner_env({"vllm.port_range": [PORT_FIRST, PORT_FIRST + 3]})
    env.start()
    env.port_free.busy.add(PORT_FIRST + 2)
    with env.client() as c:
        body = ok_json(c.get(PORTS), "GET /api/servers/ports")
    got = sorted(body) if isinstance(body, list) else body
    expected = [PORT_FIRST + 1, PORT_FIRST + 3]
    assert got == expected, f"GET {PORTS}: expected the list {expected!r}, got {body!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §4")
@pytest.mark.req("SPEC-GPU-003 §2.10a")
def test_http_move(runner_env):
    """§4: POST /api/servers/move {name, user, gpu?, port?} переносить модель — новий старт на іншій карті й порту."""
    env = runner_env
    env.start()
    env.ensure_running(NAME)
    with env.client() as c:
        ok_json(c.post(MOVE, json={"name": NAME, "user": "alice", "gpu": GPU_B, "port": PORT_FIRST + 10}), "POST /api/servers/move")
    last = env.starts(NAME)[-1]
    got = (len(env.starts(NAME)), last.env.get("CUDA_VISIBLE_DEVICES"), port_of(last.argv))
    expected = (2, str(GPU_B), PORT_FIRST + 10)
    assert got == expected, f"after POST {MOVE}: expected (starts, gpu, port) {expected!r}, got {got!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §4")
def test_http_servers_lists_started(runner_env):
    """§4: GET /api/servers показує запущений сервер."""
    env = runner_env
    env.start()
    with env.client() as c:
        body = ok_json(c.get(SERVERS), "GET /api/servers")
    got = _server_names(body)
    assert got == [NAME], f"GET /api/servers: expected [{NAME!r}], got {got!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §4")
def test_http_stop(runner_env):
    """§4: POST /api/servers/stop {name, user} зупиняє юніт; сервер зникає з GET /api/servers."""
    env = runner_env
    env.start()
    env.ensure_running(NAME)
    with env.client() as c:
        ok_json(c.post(STOP, json={"name": NAME, "user": "alice"}), "POST /api/servers/stop")
        stopped = env.unit(NAME) in env.launcher.stops
        names = _server_names(ok_json(c.get(SERVERS), "GET /api/servers"))
    got = (stopped, NAME in names)
    assert got == (True, False), f"after POST {STOP}: expected (unit stopped, not listed) (True, False), got {got!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §4")
def test_http_start_refusal_is_400_ukrainian(runner_env):
    """§4 (SPEC-GPU-001 §7.2): відмова start — HTTP 400 {code, error}, error українською."""
    with runner_env.client() as c:
        body = refusal(c.post(START, json={"repo": NOT_LOCAL_REPO, "gpu": GPU, "user": "alice"}), "model_not_local")
    text = body.get("error")
    assert isinstance(text, str) and has_cyrillic(text), f"model_not_local over HTTP: expected a Ukrainian error text, got {text!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §4")
def test_http_start_without_repo_bad_request(runner_env):
    """§4 (SPEC-GPU-001 §7.2): немає обов'язкового repo → 400 bad_request."""
    with runner_env.client() as c:
        refusal(c.post(START, json={"gpu": GPU, "user": "alice"}), "bad_request")


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §4")
def test_http_start_foreign_origin_refused(runner_env):
    """§4 (SPEC-GPU-001 §7.1.3): POST start з чужим Origin → 403, модель не запускається."""
    env = runner_env
    with env.client() as c:
        resp = c.post(START, json={"repo": REPO, "gpu": GPU, "user": "alice"}, headers={"Origin": "http://evil.example"})
    got = (resp.status_code, len(env.launcher.calls))
    assert got == (403, 0), f"start with a foreign Origin: expected (HTTP 403, 0 unit starts), got {got!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §4")
def test_http_logs_hint_is_ukrainian(runner_env):
    """§4: GET /api/servers/logs?name= — поле hint текстом українською (не кодом)."""
    env = runner_env
    env.start()
    env.ensure_failed(NAME, LINE_REMOTE_CODE)
    with env.client() as c:
        body = ok_json(c.get(LOGS, params={"name": NAME}), "GET /api/servers/logs")
    hint = body.get("hint") if isinstance(body, dict) else None
    assert isinstance(hint, str) and has_cyrillic(hint), f"logs hint over HTTP: expected Ukrainian text, got {hint!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §4")
@pytest.mark.req("SPEC-GPU-003 §2.11")
def test_http_logs_errors_filter(runner_env):
    """§4: errors=1 → лише рядки помилок."""
    env = runner_env
    env.start()
    env.append_log(NAME, [LINE_INFO, LINE_REMOTE_CODE])
    with env.client() as c:
        body = ok_json(c.get(LOGS, params={"name": NAME, "lines": 50, "errors": 1}), "GET /api/servers/logs")
    lines = [str(line) for line in (body.get("lines") or [])] if isinstance(body, dict) else []
    got = (any("trust_remote_code=True" in line for line in lines), any(" INFO " in f" {line}" for line in lines))
    assert got == (True, False), f"logs?errors=1: expected the ERROR line and no INFO lines, got {body!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §2.12")
def test_overview_card_has_models(runner_env):
    """§2.12: картка GPU в /api/overview має поле models = on_gpu(gpu)."""
    env = runner_env
    env.start()
    env.ensure_running(NAME)
    with env.client() as c:
        got = card(c, GPU).get("models")
    expected = [{"name": NAME, "port": PORT_FIRST, "status": "running", "user": "alice"}]
    assert got == expected, f"/api/overview gpu {GPU} models: expected {expected!r}, got {got!r}"


# --- MCP ---------------------------------------------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §4")
def test_mcp_tools_with_runner_and_prompts(runner_env):
    """§4: з runner і prompts — інструменти фаз 1–2 і дев'ять нових (з model_move)."""
    names = _tool_names(runner_env.mcp())
    expected = PHASE1_TOOLS | MODEL_TOOLS | RUNNER_TOOLS
    assert names == expected, f"MCP tools: missing {sorted(expected - names)}, unexpected {sorted(names - expected)}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §4")
def test_mcp_no_runner_tools_without_runner(runner_env):
    """§4: без runner і prompts нових інструментів немає."""
    names = _tool_names(runner_env.models_env.mcp())
    extra = sorted(names & RUNNER_TOOLS)
    assert extra == [], f"MCP tools without runner/prompts: expected none of {sorted(RUNNER_TOOLS)}, got {extra}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §4")
def test_mcp_model_start_launches_unit(runner_env):
    """§4: model_start(repo, gpu, user) запускає юніт моделі."""
    env = runner_env
    mcp_call(env.mcp(), "model_start", {"repo": REPO, "gpu": GPU, "user": "alice"})
    got = len(env.starts(NAME))
    assert got == 1, f"after model_start: expected 1 start of {env.unit(NAME)!r}, got {got}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §4")
def test_mcp_model_start_refusal_is_english(runner_env):
    """§4 (SPEC-GPU-001 §8, §10): відмова model_start — ToolError з "model_not_local: <English>"."""
    message = mcp_refusal(runner_env.mcp(), "model_start", {"repo": NOT_LOCAL_REPO, "gpu": GPU, "user": "alice"})
    tail = message.split("model_not_local: ", 1)[1] if "model_not_local: " in message else ""
    assert tail.strip() and not has_cyrillic(message), f"model_start refusal: expected 'model_not_local: <English>', got {message!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §4")
@pytest.mark.req("SPEC-GPU-003 §2.10a")
def test_mcp_model_move(runner_env):
    """§4: model_move(name, user, gpu?, port?) переносить модель на іншу карту."""
    env = runner_env
    env.start()
    env.ensure_running(NAME)
    mcp_call(env.mcp(), "model_move", {"name": NAME, "user": "alice", "gpu": GPU_B})
    last = env.starts(NAME)[-1]
    got = (len(env.starts(NAME)), last.env.get("CUDA_VISIBLE_DEVICES"))
    assert got == (2, str(GPU_B)), f"after model_move to gpu {GPU_B}: expected (2 starts, gpu {GPU_B}), got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §4")
def test_mcp_model_stop_stops_unit(runner_env):
    """§4: model_stop(name, user) зупиняє юніт."""
    env = runner_env
    env.start()
    env.ensure_running(NAME)
    mcp_call(env.mcp(), "model_stop", {"name": NAME, "user": "alice"})
    got = env.unit(NAME) in env.launcher.stops
    assert got is True, f"after model_stop: expected {env.unit(NAME)!r} in launcher stops, got stops {env.launcher.stops!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §4")
def test_mcp_models_running_lists_server(runner_env):
    """§4: models_running показує запущений сервер (формат §4 не задає — перевіряється лише наявність назви)."""
    env = runner_env
    env.start()
    env.ensure_running(NAME)
    text = mcp_text(mcp_call(env.mcp(), "models_running", {}))
    assert NAME in text, f"models_running: expected {NAME!r} in the result, got {text[:400]!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §4")
def test_mcp_model_logs_hint_is_code_and_english(runner_env):
    """§4: підказка в MCP — "<code>: <English>" (hint_remote_code: …)."""
    env = runner_env
    env.start()
    env.ensure_failed(NAME, LINE_REMOTE_CODE)
    body = mcp_json(env.mcp(), "model_logs", {"name": NAME})
    hint = body.get("hint") if isinstance(body, dict) else None
    prefix = "hint_remote_code: "
    tail = hint[len(prefix):] if isinstance(hint, str) and hint.startswith(prefix) else ""
    assert tail.strip() and not has_cyrillic(hint or ""), f"model_logs hint: expected '{prefix}<English>', got {hint!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §4")
def test_mcp_prompt_tools_round_trip(runner_env):
    """§4: prompt_save → є в prompts_list → prompt_delete → немає."""
    server = runner_env.mcp()

    def _listed() -> list[Any]:
        items = mcp_json(server, "prompts_list", {})
        assert isinstance(items, list), f"prompts_list: expected a JSON list, got {items!r}"
        return [item.get("name") for item in items]

    mcp_call(server, "prompt_save", {"name": CODER, "text": CODER_TEXT, "user": "alice"})
    before = _listed()
    mcp_call(server, "prompt_delete", {"name": CODER, "user": "alice"})
    after = _listed()
    got = (CODER in before, CODER in after)
    assert got == (True, False), f"prompts_list around prompt_delete: expected (listed, gone) (True, False), got before {before!r}, after {after!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §4")
@pytest.mark.req("SPEC-GPU-003 §3")
def test_mcp_llm_ask_model_not_running(runner_env):
    """§4: llm_ask незапущеної моделі — ToolError з "model_not_running: <English>"."""
    message = mcp_refusal(runner_env.mcp(), "llm_ask", {"model": "ghost", "prompt": "2+2?"})
    tail = message.split("model_not_running: ", 1)[1] if "model_not_running: " in message else ""
    assert tail.strip() and not has_cyrillic(message), f"llm_ask refusal: expected 'model_not_running: <English>', got {message!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §4")
def test_mcp_llm_ask_returns_only_answer_text(proxied_env, upstream):
    """§4: llm_ask → лише текст відповіді моделі (не JSON відповіді)."""
    text = mcp_text(mcp_call(proxied_env.mcp(), "llm_ask", {"model": NAME, "prompt": "2+2?"}))
    assert text == STUB_ANSWER, f"llm_ask: expected only the answer text {STUB_ANSWER!r}, got {text[:400]!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §4")
def test_mcp_llm_ask_default_sampling(proxied_env, upstream):
    """§4: типові max_tokens=1024 і temperature=0.2 доходять до моделі."""
    mcp_call(proxied_env.mcp(), "llm_ask", {"model": NAME, "prompt": "2+2?"})
    sent = upstream.last_json() or {}
    got = (sent.get("max_tokens"), sent.get("temperature"))
    assert got == (1024, 0.2), f"llm_ask request seen by the model: expected (max_tokens, temperature) (1024, 0.2), got {json.dumps(sent)[:400]}"
