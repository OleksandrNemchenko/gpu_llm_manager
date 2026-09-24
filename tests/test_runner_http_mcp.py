"""§4 SPEC-GPU-003: HTTP-маршрути серверів моделей і MCP-інструменти фаз 3–4; поле models картки GPU (§2.12);
конвертація PDF для чату сторінки — POST /api/convert/pdf (§5; PDF будуються в тесті через pypdfium2).

HTTP — starlette TestClient над build_app(cfg, manager, models, runner, prompts) у режимі контекстного
менеджера (lifespan працює; захист і формат відмов — SPEC-GPU-001 §7). MCP — у процесі, build_mcp(...,
runner=, prompts=) (SPEC-GPU-001 §10): відмова піднімає ToolError, str(exc) містить "<code>: <English text>".
Стан серверів готується напряму через ModelRunner до відкриття клієнта; підроблені юніти живі, health
незапущених не відповідає, тож фонове опитування сервісу під час тесту стану не змінює.
"""

from __future__ import annotations

import base64
import io
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
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
    STUB_COMPLETION,
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
    """§4: POST /api/servers/start {repo, gpu, user} → 200 і запуск юніта gm-model-<name>.service."""
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
    got = (len(env.starts(NAME)), env.gpu_of(last), port_of(last.argv))
    expected = (2, GPU_B, PORT_FIRST + 10)
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
    got = (len(env.starts(NAME)), env.gpu_of(last))
    assert got == (2, GPU_B), f"after model_move to gpu {GPU_B}: expected (2 starts, gpu {GPU_B}), got {got!r}"


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


def _completion_without_text(content: Any) -> dict[str, Any]:
    """Відповідь моделі у форматі STUB_COMPLETION, де content — content (без тексту), finish_reason — length."""
    choice = {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "length"}
    return {**STUB_COMPLETION, "choices": [choice]}


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §4")
@pytest.mark.parametrize("content", ["", None], ids=["empty-string", "null"])
def test_mcp_llm_ask_empty_answer_refused(proxied_env, upstream, content):
    """§4: модель не дала тексту (content порожній або null) → llm_ask відмовляє: ToolError з кодом empty_answer."""
    upstream.response = _completion_without_text(content)
    message = mcp_refusal(proxied_env.mcp(), "llm_ask", {"model": NAME, "prompt": "2+2?"})
    assert "empty_answer" in message, f"llm_ask on an answer with content {content!r}: expected a refusal with 'empty_answer', got {message!r}"


# --- §5 конвертація PDF ---------------------------------------------------------------------------------------------------

CONVERT = "/api/convert/pdf"
JPEG_URL_PREFIX = "data:image/jpeg;base64,"  # §5: images — data URL JPEG
PDF_MAX_SIDE_PX = 2000  # §5: сторінка-картинка — не більше 2000 px по довшій стороні
SMALL_PAGE_PT = (72.0, 72.0)  # мала сторінка (1 × 1 дюйм): картинки з неї дешеві навіть для 100 сторінок
HUGE_PAGE_PT = (20000.0, 20000.0)  # у 72 dpi це 20000 px по стороні — удесятеро більше за межу §5
RANDOM_SEED = 0x5EED  # сід випадкових байтів «не PDF»
NOT_PDF_DATA = {
    "random-bytes": base64.b64encode(random.Random(RANDOM_SEED).randbytes(4096)).decode("ascii"),
    "not-base64": "%%% not base64 %%%",
}


def _pdf(pages: int, size: tuple[float, float] = SMALL_PAGE_PT) -> bytes:
    """PDF з pages порожніх сторінок розміром size (pt), зібраний pypdfium2 у пам'яті.

    Викликається лише з головного потоку тесту: pdfium не потокобезпечний."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    buffer = io.BytesIO()
    try:
        for _ in range(pages):
            pdf.new_page(*size).close()
        pdf.save(buffer)
    finally:
        pdf.close()
    return buffer.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _data_url(data: bytes) -> str:
    return "data:application/pdf;base64," + _b64(data)


def _convert(client: Any, data: str, mode: str) -> Any:
    """POST /api/convert/pdf {data, mode} (§5)."""
    return client.post(CONVERT, json={"data": data, "mode": mode})


def _jpeg(url: Any) -> tuple[Any, tuple[int, int]]:
    """Формат і розмір (px) картинки з data URL JPEG, розібраної Pillow; не такий data URL — помилка тесту."""
    from PIL import Image

    assert isinstance(url, str) and url.startswith(JPEG_URL_PREFIX), (
        f"page image: expected a {JPEG_URL_PREFIX!r}... data URL, got {str(url)[:80]!r}"
    )
    with Image.open(io.BytesIO(base64.b64decode(url[len(JPEG_URL_PREFIX):]))) as img:
        return img.format, img.size


def _images(body: Any) -> list[Any]:
    images = body.get("images") if isinstance(body, dict) else None
    assert isinstance(images, list), f"{CONVERT} images: expected {{pages, images: [...]}}, got {str(body)[:300]}"
    return images


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §5")
@pytest.mark.parametrize("form", ["data-url", "base64"])
def test_convert_pdf_text_pages_marked(runner_env, form):
    """§5: mode text → {pages, text}, сторінки в тексті позначені [page N] по порядку; data — data URL або base64."""
    pdf = _pdf(2)
    data = _data_url(pdf) if form == "data-url" else _b64(pdf)
    with runner_env.client() as c:
        body = ok_json(_convert(c, data, "text"), f"POST {CONVERT} mode text")
    text = body.get("text") if isinstance(body, dict) else None
    first = text.find("[page 1]") if isinstance(text, str) else -1
    second = text.find("[page 2]") if isinstance(text, str) else -1
    got = (body.get("pages") if isinstance(body, dict) else None, 0 <= first < second)
    assert got == (2, True), f"2-page PDF as {form}, mode text: expected pages 2 and '[page 1]' before '[page 2]', got {str(body)[:300]}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §5")
def test_convert_pdf_images_are_jpeg_data_urls(runner_env):
    """§5: mode images → {pages, images}; кожна сторінка — data URL JPEG, що розбирається як JPEG."""
    with runner_env.client() as c:
        body = ok_json(_convert(c, _b64(_pdf(2)), "images"), f"POST {CONVERT} mode images")
    got = (body.get("pages"), [_jpeg(url)[0] for url in _images(body)])
    assert got == (2, ["JPEG", "JPEG"]), f"2-page PDF, mode images: expected (pages, image formats) (2, ['JPEG', 'JPEG']), got {got!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §5")
def test_convert_pdf_huge_page_at_most_2000px(runner_env):
    """§5: сторінка 20000 × 20000 pt рендериться не більше ніж 2000 px по довшій стороні."""
    with runner_env.client() as c:
        body = ok_json(_convert(c, _b64(_pdf(1, HUGE_PAGE_PT)), "images"), f"POST {CONVERT} huge page")
    images = _images(body)
    assert len(images) == 1, f"1-page PDF: expected 1 image, got {len(images)}"
    _, size = _jpeg(images[0])
    assert 0 < max(size) <= PDF_MAX_SIDE_PX, f"image of a {HUGE_PAGE_PT} pt page: expected the longer side 1..{PDF_MAX_SIDE_PX} px, got {size} px"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §5")
@pytest.mark.parametrize("case", sorted(NOT_PDF_DATA))
def test_convert_pdf_not_pdf_refused(runner_env, case):
    """§5: data — не PDF (випадкові байти, сід 0x5EED) або не base64 → 400 bad_pdf."""
    with runner_env.client() as c:
        refusal(_convert(c, NOT_PDF_DATA[case], "text"), "bad_pdf")


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §5")
@pytest.mark.parametrize(("mode", "pages"), [("images", 101), ("text", 501)], ids=["images-101", "text-501"])
def test_convert_pdf_too_many_pages_refused(runner_env, mode, pages):
    """§5: сторінок понад 100 для images чи понад 500 для text → 400 bad_pdf."""
    with runner_env.client() as c:
        refusal(_convert(c, _b64(_pdf(pages)), mode), "bad_pdf")


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §5")
@pytest.mark.parametrize(("mode", "pages"), [("images", 100), ("text", 500)], ids=["images-100", "text-500"])
def test_convert_pdf_page_limit_accepted(runner_env, mode, pages):
    """§5: рівно 100 сторінок для images і 500 для text — ще можна."""
    with runner_env.client() as c:
        body = ok_json(_convert(c, _b64(_pdf(pages)), mode), f"POST {CONVERT} {pages} pages, mode {mode}")
    got = body.get("pages") if isinstance(body, dict) else body
    assert got == pages, f"{pages}-page PDF, mode {mode}: expected pages {pages}, got {got!r}"


def _pages_and_images(resp: Any) -> Any:
    """(HTTP-код, pages, кількість images) відповіді конвертації; не JSON-об'єкт — текст для повідомлення."""
    if resp.status_code != 200:
        return (resp.status_code, resp.text[:200])
    body = resp.json()
    if not isinstance(body, dict):
        return (resp.status_code, str(body)[:200])
    images = body.get("images")
    return (resp.status_code, body.get("pages"), len(images) if isinstance(images, list) else images)


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §5")
def test_convert_pdf_concurrent_both_succeed(runner_env):
    """§5: дві конвертації, надіслані одночасно, обидві вдаються, і кожна отримує свої сторінки (2 і 3)."""
    pdfs = {pages: _b64(_pdf(pages)) for pages in (2, 3)}  # PDF — до потоків: pdfium не потокобезпечний
    barrier = threading.Barrier(len(pdfs), timeout=30)  # обидва запити вирушають разом; тайм-аут — від зависання
    with runner_env.client() as c:

        def _post(pages: int) -> Any:
            barrier.wait()
            return _convert(c, pdfs[pages], "images")

        with ThreadPoolExecutor(max_workers=len(pdfs)) as pool:
            futures = {pages: pool.submit(_post, pages) for pages in pdfs}
            got = {pages: _pages_and_images(future.result(timeout=120)) for pages, future in futures.items()}
    expected = {2: (200, 2, 2), 3: (200, 3, 3)}
    assert got == expected, f"two concurrent conversions: expected {{pages: (HTTP, pages, images)}} {expected!r}, got {got!r}"
