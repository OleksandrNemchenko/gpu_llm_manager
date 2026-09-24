"""§3 SPEC-GPU-003: сховище системних промптів (PromptStore) і шлюз OpenAI-сумісного API (gateway).

PromptStore(users, store, journal, clock) будується над state.json і журналом менеджера фази 1 — як ModelStore
(SPEC-GPU-002 §9). gateway.resolve перевіряється напряму над ModelRunner з підробками швів; маршрути /v1/* —
через TestClient над build_app(cfg, manager, models, runner, prompts). Пересилання до моделі перевіряється на
крихітному HTTP-сервері на 127.0.0.1 (фікстура upstream): runner видає йому порт, бо vllm.port_range тесту
складається з одного цього порту.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from .conftest import STRANGER, ok_json
from .model_fakes import expect_manager_error
from .runner_fakes import DEFAULT_NAME, GPU_B, PORT_FIRST, STUB_COMPLETION

pytestmark = pytest.mark.usefixtures("isolated_home")

CODER = "coder"
CODER_TEXT = "You are a strict Python reviewer. Point out bugs first, style last. Answer in one paragraph."
USER_MSG = {"role": "user", "content": "Review: def double(x): return x * 3"}
PROMPT_ITEM_FIELDS = {"name", "preview", "user", "updated"}  # елемент list() (§3)
MAX_PROMPT_CHARS = 20000  # межа тексту промпту (§3)


def _text(value: Any) -> Any:
    """Текст промпту з результату get(): рядок або запис з полем text (форму результату §3 не задає)."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return value.get("text")
    return getattr(value, "text", value)


# --- PromptStore ---------------------------------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §3")
def test_prompt_save_then_get(runner_env):
    """§3: save(name, text, user), потім get(name) → той самий текст."""
    prompts = runner_env.prompts
    prompts.save(CODER, CODER_TEXT, "alice")
    got = _text(prompts.get(CODER))
    assert got == CODER_TEXT, f"get({CODER!r}) after save: expected {CODER_TEXT!r}, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §3")
@pytest.mark.parametrize("text", ["", "x" * (MAX_PROMPT_CHARS + 1)], ids=["empty", "20001-chars"])
def test_prompt_bad_text_refused(runner_env, text):
    """§3: порожній текст або довший за 20000 символів → bad_prompt."""
    expect_manager_error("bad_prompt", runner_env.prompts.save, CODER, text, "alice")


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §3")
def test_prompt_max_length_accepted(runner_env):
    """§3: рівно 20000 символів — ще можна."""
    prompts = runner_env.prompts
    text = "y" * MAX_PROMPT_CHARS
    prompts.save(CODER, text, "alice")
    got = _text(prompts.get(CODER))
    assert got == text, f"prompt of {MAX_PROMPT_CHARS} chars: expected it saved intact, got {len(got) if isinstance(got, str) else got!r} chars"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §3")
def test_prompt_bad_name_refused(runner_env):
    """§3: назва — за правилами назви сервера (§2.1): «../evil» → bad_name."""
    expect_manager_error("bad_name", runner_env.prompts.save, "../evil", CODER_TEXT, "alice")


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §3")
def test_prompt_unknown_user_refused(runner_env):
    """§3: PromptStore отримує users — логін не з users.allowed → unknown_user."""
    expect_manager_error("unknown_user", runner_env.prompts.save, CODER, CODER_TEXT, STRANGER)


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §3")
def test_prompt_get_missing_not_found(runner_env):
    """§3: get промпту, якого немає, → prompt_not_found."""
    expect_manager_error("prompt_not_found", runner_env.prompts.get, "nope")


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §3")
def test_prompt_delete_removes(runner_env):
    """§3: після delete(name, user) промпту немає — get → prompt_not_found."""
    prompts = runner_env.prompts
    prompts.save(CODER, CODER_TEXT, "alice")
    prompts.delete(CODER, "bob")
    expect_manager_error("prompt_not_found", prompts.get, CODER)


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §3")
def test_prompt_list_items(runner_env):
    """§3: list() → [{name, preview (80 символів), user, updated}]."""
    prompts = runner_env.prompts
    text = "Long system prompt for review number " * 5  # 185 символів
    prompts.save(CODER, text, "carol")
    items = prompts.list()
    shapes = [sorted(item) for item in items] if isinstance(items, list) else items
    assert isinstance(items, list) and len(items) == 1 and set(items[0]) == PROMPT_ITEM_FIELDS, (
        f"list(): expected one item with keys {sorted(PROMPT_ITEM_FIELDS)}, got {shapes!r}"
    )
    got = (items[0]["name"], items[0]["preview"], items[0]["user"])
    expected = (CODER, text[:80], "carol")
    assert got == expected, f"list() item: expected (name, preview = first 80 chars, user) {expected!r}, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §3")
def test_prompt_journal(runner_env):
    """§3: save і delete пишуться в журнал як prompt_save і prompt_delete від свого користувача."""
    env = runner_env
    env.prompts.save(CODER, CODER_TEXT, "alice")
    env.prompts.delete(CODER, "bob")
    got = [(e.get("action"), e.get("user")) for e in env.journal() if e.get("action") in {"prompt_save", "prompt_delete"}]
    expected = [("prompt_save", "alice"), ("prompt_delete", "bob")]
    assert got == expected, f"journal prompt entries: expected {expected!r}, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §3")
def test_prompt_survives_restart(runner_env):
    """§3: промпт живе в state.json менеджера (аргумент store) — після перезапуску сервісу він є."""
    env = runner_env
    env.prompts.save(CODER, CODER_TEXT, "alice")
    restarted = env.restart()
    got = _text(restarted.prompts.get(CODER))
    assert got == CODER_TEXT, f"get({CODER!r}) after a service restart: expected {CODER_TEXT!r}, got {got!r}"


# --- gateway.resolve -------------------------------------------------------------------------------------------------


def _resolve(env: Any, body: dict[str, Any]) -> Any:
    from gpu_manager.gateway import resolve

    return resolve(env.runner, env.prompts, body)


@pytest.fixture
def served(runner_env: Any) -> Any:
    """tiny-llm у стані running на порту 8000 і збережений промпт coder."""
    env = runner_env
    env.start()
    env.ensure_running(DEFAULT_NAME)
    env.prompts.save(CODER, CODER_TEXT, "alice")
    return env


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §3")
def test_resolve_running_model(served):
    """§3: model = назва running-сервера → його порт; повідомлення без промпту не змінюються."""
    port, body = _resolve(served, {"model": DEFAULT_NAME, "messages": [USER_MSG]})
    got = (port, body.get("messages"))
    assert got == (PORT_FIRST, [USER_MSG]), f"resolve({DEFAULT_NAME!r}): expected (port {PORT_FIRST}, messages unchanged), got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §3")
def test_resolve_prompt_first_system_message(served):
    """§3: model = назва@промпт → текст промпту першим повідомленням system."""
    _, body = _resolve(served, {"model": f"{DEFAULT_NAME}@{CODER}", "messages": [USER_MSG]})
    expected = [{"role": "system", "content": CODER_TEXT}, USER_MSG]
    got = body.get("messages")
    assert got == expected, f"messages for {DEFAULT_NAME}@{CODER}: expected {expected!r}, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §3")
def test_resolve_forwards_served_model_name(served):
    """§3, §2.6: vLLM знає модель лише як --served-model-name N — у тілі для нього model = N, без «@промпт»."""
    _, body = _resolve(served, {"model": f"{DEFAULT_NAME}@{CODER}", "messages": [USER_MSG]})
    got = body.get("model")
    assert got == DEFAULT_NAME, f"forwarded model for {DEFAULT_NAME}@{CODER}: expected {DEFAULT_NAME!r}, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §3")
def test_resolve_prompt_prefixes_completion_prompt(served):
    """§3: для поля prompt (completions) текст промпту — префікс."""
    original = "def add(a, b):"
    _, body = _resolve(served, {"model": f"{DEFAULT_NAME}@{CODER}", "prompt": original})
    got = body.get("prompt")
    ok = isinstance(got, str) and got.startswith(CODER_TEXT) and got.endswith(original)
    assert ok, f"completion prompt: expected {CODER_TEXT!r} + ... + {original!r}, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §3")
@pytest.mark.parametrize("model", ["slow", "ghost"], ids=["starting", "unknown"])
def test_resolve_not_running_refused(served, model):
    """§3: лише running — сервер у стані starting або невідомий → model_not_running."""
    served.start(name="slow", gpu=GPU_B)  # юніт живий, health не відповідає — starting
    expect_manager_error("model_not_running", _resolve, served, {"model": model, "messages": [USER_MSG]})


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §3")
def test_resolve_unknown_prompt_refused(served):
    """§3: невідомий промпт у назва@промпт → prompt_not_found."""
    expect_manager_error("prompt_not_found", _resolve, served, {"model": f"{DEFAULT_NAME}@nope", "messages": [USER_MSG]})


# --- Маршрути /v1 ------------------------------------------------------------------------------------------------------


def _v1_error(resp: Any, status: int, code: str | None) -> None:
    """Помилка шлюзу §3: HTTP status і тіло {error: {message, type, code}}; code=None — код не перевіряється."""
    assert resp.status_code == status, f"expected HTTP {status}, got {resp.status_code}: {resp.text[:300]}"
    try:
        body = resp.json()
    except ValueError:
        pytest.fail(f"expected a JSON error body, got non-JSON: {resp.text[:300]}")
    error = body.get("error") if isinstance(body, dict) else None
    assert isinstance(error, dict) and {"message", "type", "code"} <= set(error), (
        f"expected body {{error: {{message, type, code}}}}, got {body!r}"
    )
    if code is not None:
        assert error.get("code") == code, f"expected error.code {code!r}, got {error.get('code')!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §3")
def test_v1_models_lists_only_running(served):
    """§3: GET /v1/models → {object: list, data: [...]}; сервер у стані starting не показується."""
    served.start(name="slow", gpu=GPU_B)
    with served.client() as c:
        body = ok_json(c.get("/v1/models"), "GET /v1/models")
    ids = [item.get("id") for item in body.get("data", [])] if isinstance(body, dict) else body
    ok = isinstance(body, dict) and body.get("object") == "list" and DEFAULT_NAME in ids and not any(
        str(i).startswith("slow") for i in ids
    )
    assert ok, f"/v1/models: expected object 'list' with {DEFAULT_NAME!r} and without 'slow', got {body!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §3")
def test_v1_models_item_shape(served):
    """§3: елемент /v1/models — {id, object: model, owned_by}."""
    with served.client() as c:
        body = ok_json(c.get("/v1/models"), "GET /v1/models")
    items = [i for i in body.get("data", []) if i.get("id") == DEFAULT_NAME] if isinstance(body, dict) else []
    ok = len(items) == 1 and items[0].get("object") == "model" and "owned_by" in items[0]
    assert ok, f"/v1/models item for {DEFAULT_NAME!r}: expected {{id, object: 'model', owned_by}}, got {body!r}"


V1_BODIES: dict[str, dict[str, Any]] = {
    "/v1/chat/completions": {"messages": [USER_MSG]},
    "/v1/completions": {"prompt": "def add(a, b):"},
    "/v1/embeddings": {"input": "hello"},
}


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §3")
@pytest.mark.parametrize("path", sorted(V1_BODIES))
def test_v1_model_not_running_404(served, path):
    """§3: невідома / незапущена модель → 404, {error: {message, type, code: model_not_running}}."""
    with served.client() as c:
        resp = c.post(path, json={"model": "ghost", **V1_BODIES[path]})
    _v1_error(resp, 404, "model_not_running")


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §3")
def test_v1_prompt_not_found_404(served):
    """§3: невідомий промпт → 404, code prompt_not_found."""
    with served.client() as c:
        resp = c.post("/v1/chat/completions", json={"model": f"{DEFAULT_NAME}@nope", "messages": [USER_MSG]})
    _v1_error(resp, 404, "prompt_not_found")


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §3")
def test_v1_bad_body_400(served):
    """§3: некоректне тіло (не JSON) → 400 у форматі {error: {message, type, code}}."""
    with served.client() as c:
        resp = c.post("/v1/chat/completions", content=b'{"model": ', headers={"Content-Type": "application/json"})
    _v1_error(resp, 400, None)


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §3")
def test_v1_chat_forwarded_with_system_prompt(proxied_env, upstream):
    """§3: POST /v1/chat/completions пересилається на 127.0.0.1:<порт моделі> тим самим шляхом, промпт — першим
    повідомленням system."""
    env = proxied_env
    env.prompts.save(CODER, CODER_TEXT, "alice")
    with env.client() as c:
        resp = c.post("/v1/chat/completions", json={"model": f"{DEFAULT_NAME}@{CODER}", "messages": [USER_MSG]})
    assert resp.status_code == 200, f"proxied chat completion: expected HTTP 200, got {resp.status_code}: {resp.text[:300]}"
    sent = upstream.last_json() or {}
    got = (upstream.requests[-1]["path"], sent.get("messages"))
    expected = ("/v1/chat/completions", [{"role": "system", "content": CODER_TEXT}, USER_MSG])
    assert got == expected, f"request seen by the model on port {upstream.port}: expected (path, messages) {expected!r}, got {got!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §3")
def test_v1_chat_returns_model_response(proxied_env, upstream):
    """§3: відповідь моделі повертається клієнту як є."""
    with proxied_env.client() as c:
        resp = c.post("/v1/chat/completions", json={"model": DEFAULT_NAME, "messages": [USER_MSG]})
    got = resp.json() if resp.status_code == 200 else f"HTTP {resp.status_code}: {resp.text[:300]}"
    assert got == STUB_COMPLETION, f"proxied response: expected the model's JSON {STUB_COMPLETION!r}, got {got!r}"
