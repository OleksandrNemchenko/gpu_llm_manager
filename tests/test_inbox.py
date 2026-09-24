"""§6 SPEC-GPU-003: файли для моделей — тека files.inbox_dir.

Навіщо файл: агент кладе файл у свою теку й передає моделі лише шлях. Тут перевіряються Inbox(root, users, host)
(info і parts) над текою в tmp_path, MCP files_inbox і llm_ask(files=…) у процесі (build_mcp(…, inbox=)) — запит
до моделі бачить StubUpstream з runner_fakes — і збирання Inbox застосунком build_app (через його /mcp).
PDF будуються в тесті через pypdfium2 (_pdf з test_runner_http_mcp); сторінки порожні.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anyio
import pytest

from .conftest import PUBLIC_HOST, STRANGER, USERS, mcp_call, mcp_json, mcp_refusal
from .model_fakes import MIB, expect_manager_error
from .runner_fakes import DEFAULT_NAME
from .test_mcp import _mcp_http, _tool_call
from .test_runner_http_mcp import JPEG_URL_PREFIX, _pdf

pytestmark = pytest.mark.usefixtures("isolated_home")

HOST = PUBLIC_HOST.rsplit(":", 1)[0]  # public_host без порту — host, який build_app дає Inbox (§6)
USER = "alice"  # має теку в root
OTHER = "bob"  # з users.allowed, теки в root немає
NAME = DEFAULT_NAME  # запущена модель proxied_env
MAX_TEXT_BYTES = MIB  # §6: текст — до 1 MiB
MAX_PATHS = 20  # §6: понад 20 шляхів → bad_file
PDF_PAGES = 2  # 2, а не 1: «<N> pages» без питання однини
PROMPT = "Summarize the attached files."
TEXT_CONTENT = "привіт, модель\nрядок два"  # кирилиця: вміст має читатися саме як UTF-8
NOT_UTF8 = b"caf\xe9 au lait"  # latin-1 «é»: байт 0xE9 без байтів продовження — не UTF-8
NOT_PDF = b"this is not a PDF\n" * 8
JPEG_MAGIC = b"\xff\xd8\xff"  # початок будь-якого файла JPEG

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
VIDEO_EXT = (".mp4", ".webm", ".mov", ".mkv", ".avi")
MEDIA_CASES = (
    [(ext, "image_url") for ext in IMAGE_EXT]
    + [(ext, "audio_url") for ext in AUDIO_EXT]
    + [(ext, "video_url") for ext in VIDEO_EXT]
)
INFO_FIELDS = {"dir", "ready", "file_url_prefix", "copy_local", "copy_remote", "use"}


# --- Помічники ------------------------------------------------------------------------------------------------------


def _inbox(root: Path) -> Any:
    """Inbox(root, users, host) (§6) з логінами тестового конфігу й HOST."""
    from gpu_manager.inbox import Inbox

    return Inbox(root, list(USERS), HOST)


def _put(path: Path, data: bytes | str) -> Path:
    """Пише файл path (теки створюються); str — як UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)
    return path


def _parts(inbox: Any, paths: Sequence[Path | str], **kwargs: Any) -> list[Any]:
    """inbox.parts([шляхи рядками], **kwargs) як список."""
    return list(inbox.parts([str(p) for p in paths], **kwargs))


def _file_url(path: Path) -> str:
    return f"file://{path}"


def _text_part(name: str, content: str) -> dict[str, Any]:
    """Очікувана текстова частина файла (§6): [file <назва>]\\n<вміст>."""
    return {"type": "text", "text": f"[file {name}]\n{content}"}


def _is_jpeg_data_url(url: Any) -> bool:
    """url — data URL JPEG: префікс data:image/jpeg;base64, і байти, що починаються як JPEG."""
    if not (isinstance(url, str) and url.startswith(JPEG_URL_PREFIX)):
        return False
    try:
        raw = base64.b64decode(url[len(JPEG_URL_PREFIX):], validate=True)
    except ValueError:
        return False
    return raw[:3] == JPEG_MAGIC


def _image_summary(parts: list[Any]) -> list[tuple[Any, bool]]:
    """(type, чи url — data URL JPEG) кожної частини."""
    return [
        (p.get("type"), _is_jpeg_data_url((p.get("image_url") or {}).get("url")))
        if isinstance(p, dict) else (p, False)
        for p in parts
    ]


def _mcp(env: Any, inbox: Any) -> Any:
    """build_mcp(manager, tz, models=, runner=, prompts=, inbox=) (§4, §6) — як RunnerEnv.mcp(), але з inbox."""
    from gpu_manager.mcp_tools import build_mcp

    return build_mcp(
        env.manager, env.cfg.display_timezone, models=env.store, runner=env.runner, prompts=env.prompts, inbox=inbox
    )


def _tool_names(server: Any) -> set[str]:
    return {t.name for t in anyio.run(server.list_tools)}


def _user_content(upstream: Any) -> Any:
    """content останнього повідомлення role=user у запиті, що дійшов до моделі."""
    sent = upstream.last_json()
    messages = sent.get("messages") if isinstance(sent, dict) else None
    assert isinstance(messages, list), f"request seen by the model: expected a messages list, got {str(sent)[:400]}"
    users = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    assert users, f"request seen by the model: expected a user message, got messages {str(messages)[:400]}"
    return users[-1].get("content")


# --- Фікстури ---------------------------------------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Тека files.inbox_dir у tmp_path з текою USER. Шлях розкритий: очікувані dir і file:// URL не залежать
    від того, чи tmp_path іде через символьне посилання."""
    path = tmp_path.resolve() / "gpu-inbox"
    (path / USER).mkdir(parents=True)
    return path


@pytest.fixture
def inbox(root: Path) -> Any:
    return _inbox(root)


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    """Справжній файл поза root."""
    return _put(tmp_path.resolve() / "outside" / "secret.txt", "secret")


# --- info(user) -------------------------------------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_info_has_all_fields(inbox):
    """§6: info(user) → {dir, ready, file_url_prefix, copy_local, copy_remote, use}."""
    info = inbox.info(USER)
    missing = sorted(INFO_FIELDS - set(info))
    assert missing == [], f"info({USER!r}): expected fields {sorted(INFO_FIELDS)}, missing {missing}; got {info!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_info_dir_is_root_user_slash(inbox, root):
    """§6: dir = <root>/<user>/ (з кінцевою скісною)."""
    got = inbox.info(USER).get("dir")
    expected = f"{root}/{USER}/"
    assert got == expected, f"info({USER!r}).dir: expected {expected!r}, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
@pytest.mark.parametrize(("user", "expected"), [(USER, True), (OTHER, False)], ids=["folder-exists", "folder-missing"])
def test_info_ready_when_user_folder_exists(inbox, user, expected):
    """§6: ready — тека <root>/<user>/ існує."""
    got = inbox.info(user).get("ready")
    assert got is expected, f"info({user!r}).ready: expected {expected!r}, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_info_ready_follows_folder_created_later(inbox, root):
    """§6: ready — стан теки на момент виклику: тека, створена після Inbox(...), дає ready True."""
    before = inbox.info(OTHER).get("ready")
    (root / OTHER).mkdir()
    after = inbox.info(OTHER).get("ready")
    assert (before, after) == (False, True), (
        f"info({OTHER!r}).ready before/after creating {root / OTHER}: expected (False, True), got {(before, after)!r}"
    )


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_info_file_url_prefix(inbox, root):
    """§6: file_url_prefix = file://<root>/<user>/."""
    got = inbox.info(USER).get("file_url_prefix")
    expected = f"file://{root}/{USER}/"
    assert got == expected, f"info({USER!r}).file_url_prefix: expected {expected!r}, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
@pytest.mark.parametrize("field", ["copy_local", "copy_remote"])
def test_info_copy_command_contains_dir(inbox, root, field):
    """§6: команда копіювання (локальна й по SSH) веде в теку dir = <root>/<user>/."""
    got = inbox.info(USER).get(field)
    expected = f"{root}/{USER}/"
    assert isinstance(got, str) and expected in got, f"info({USER!r}).{field}: expected to contain {expected!r}, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
@pytest.mark.parametrize("field", ["copy_local", "copy_remote"])
def test_info_copy_command_is_rsync(inbox, field):
    """§6: copy_* — команди rsync."""
    got = inbox.info(USER).get(field)
    assert isinstance(got, str) and "rsync" in got, f"info({USER!r}).{field}: expected an rsync command, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_info_copy_remote_has_user_at_host(inbox):
    """§6: copy_remote — по SSH на <user>@<host>:."""
    got = inbox.info(USER).get("copy_remote")
    expected = f"{USER}@{HOST}:"
    assert isinstance(got, str) and expected in got, f"info({USER!r}).copy_remote: expected to contain {expected!r}, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_info_unknown_user_refused(inbox):
    """§6: логін не з users → ManagerError unknown_user."""
    expect_manager_error("unknown_user", inbox.info, STRANGER)


# --- parts(paths, pdf_mode): види файлів --------------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
@pytest.mark.parametrize(("ext", "kind"), MEDIA_CASES, ids=[f"{ext[1:]}-{kind}" for ext, kind in MEDIA_CASES])
def test_parts_media_is_file_url(inbox, root, ext, kind):
    """§6: картинка / аудіо / відео → {"type": <kind>, <kind>: {"url": "file://<розкритий шлях>"}}; вміст читає модель."""
    path = _put(root / USER / f"clip{ext}", b"\x00fake media bytes")
    got = _parts(inbox, [path])
    expected = [{"type": kind, kind: {"url": _file_url(path)}}]
    assert got == expected, f"parts([{path.name!r}]): expected {expected!r}, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
@pytest.mark.parametrize("name", ["notes.txt", "script.py", "README"])
def test_parts_text_file(inbox, root, name):
    """§6: решта файлів — текст UTF-8 → {"type": "text", "text": "[file <назва>]\\n<вміст>"}."""
    path = _put(root / USER / name, TEXT_CONTENT)
    got = _parts(inbox, [path])
    expected = [_text_part(name, TEXT_CONTENT)]
    assert got == expected, f"parts([{name!r}]): expected {expected!r}, got {got!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
@pytest.mark.parametrize("kwargs", [{}, {"pdf_mode": "text"}], ids=["default-mode", "text-mode"])
def test_parts_pdf_text_mode(inbox, root, kwargs):
    """§6: .pdf у режимі text (він же типовий) → одна частина {"type": "text", "text": "[file <назва>, <N> pages]\\n…"}."""
    path = _put(root / USER / "doc.pdf", _pdf(PDF_PAGES))
    parts = _parts(inbox, [path], **kwargs)
    header = f"[file doc.pdf, {PDF_PAGES} pages]\n"
    got = [(p.get("type"), str(p.get("text", "")).startswith(header)) if isinstance(p, dict) else p for p in parts]
    assert got == [("text", True)], f"parts([doc.pdf], {kwargs!r}): expected one text part starting {header!r}, got {parts!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_parts_pdf_images_mode(inbox, root):
    """§6: .pdf у режимі images → image_url з data URL JPEG на кожну сторінку."""
    path = _put(root / USER / "doc.pdf", _pdf(PDF_PAGES))
    parts = _parts(inbox, [path], pdf_mode="images")
    got = _image_summary(parts)
    expected = [("image_url", True)] * PDF_PAGES
    assert got == expected, (
        f"parts([doc.pdf], pdf_mode='images'): expected {PDF_PAGES} image_url parts with {JPEG_URL_PREFIX!r} JPEG, "
        f"got (type, is JPEG data URL) {got!r}"
    )


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_parts_pdf_bogus_mode_refused(inbox, root):
    """§6: pdf_mode не text і не images → bad_request."""
    path = _put(root / USER / "doc.pdf", _pdf(PDF_PAGES))
    expect_manager_error("bad_request", inbox.parts, [str(path)], pdf_mode="bogus")


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
@pytest.mark.req("SPEC-GPU-003 §5")
def test_parts_not_a_pdf_refused(inbox, root):
    """§6, §5: файл .pdf, що не PDF → bad_pdf."""
    path = _put(root / USER / "fake.pdf", NOT_PDF)
    expect_manager_error("bad_pdf", inbox.parts, [str(path)])


# --- parts: шляхи ---------------------------------------------------------------------------------------------------


def _escaping_path(case: str, root: Path, outside: Path) -> str:
    """Шлях, що після розкриття лежить поза <root>/ (outside — справжній файл поза root)."""
    if case == "outside-root":
        return str(outside)
    if case == "dotdot-escape":
        # <root>/alice/../.. — це tmp_path; далі outside/secret.txt
        return f"{root}/{USER}/../../{outside.parent.name}/{outside.name}"
    if case == "symlink-inside-root-to-outside":
        link = root / USER / "escape.txt"
        link.symlink_to(outside)
        return str(link)
    if case == "sibling-with-root-prefix":
        # <root>-evil/… починається тим самим рядком, що й <root>, але не лежить усередині <root>/
        return str(_put(root.parent / f"{root.name}-evil" / USER / "notes.txt", "x"))
    raise AssertionError(f"unknown case {case!r}")


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
@pytest.mark.parametrize(
    "case", ["outside-root", "dotdot-escape", "symlink-inside-root-to-outside", "sibling-with-root-prefix"]
)
def test_parts_path_outside_root_refused(inbox, root, outside, case):
    """§6: шлях, що після розкриття (зокрема символьних посилань) не лежить усередині <root>/ → bad_file_path."""
    path = _escaping_path(case, root, outside)
    expect_manager_error("bad_file_path", inbox.parts, [path])


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
@pytest.mark.parametrize("case", ["missing", "directory"])
def test_parts_path_not_a_file_refused(inbox, root, case):
    """§6: шлях усередині root, що не є файлом (його немає або це тека) → bad_file_path."""
    path = root / USER / ("missing.txt" if case == "missing" else "folder")
    if case == "directory":
        path.mkdir()
    expect_manager_error("bad_file_path", inbox.parts, [str(path)])


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_parts_symlink_within_root_uses_resolved_path(inbox, root):
    """§6: символьне посилання всередині root на файл усередині root приймається; file:// URL — розкритий шлях."""
    real = _put(root / USER / "real.png", b"\x89PNG fake")
    link = root / USER / "link.png"
    link.symlink_to(real)
    got = _parts(inbox, [link])
    expected = [{"type": "image_url", "image_url": {"url": _file_url(real)}}]
    assert got == expected, f"parts([link.png -> real.png]): expected {expected!r}, got {got!r}"


# --- parts: межі тексту й кількості -----------------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_parts_non_utf8_refused(inbox, root):
    """§6: текстовий файл не в UTF-8 → bad_file."""
    path = _put(root / USER / "latin1.txt", NOT_UTF8)
    expect_manager_error("bad_file", inbox.parts, [str(path)])


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_parts_text_over_1mib_refused(inbox, root):
    """§6: текстовий файл більше 1 MiB (1 MiB + 1 байт) → bad_file."""
    path = _put(root / USER / "big.txt", b"a" * (MAX_TEXT_BYTES + 1))
    expect_manager_error("bad_file", inbox.parts, [str(path)])


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_parts_text_of_exactly_1mib_accepted(inbox, root):
    """§6: «до 1 MiB», відмова — лише «більше»: файл рівно 1 MiB дає текстову частину з усім вмістом."""
    content = "a" * MAX_TEXT_BYTES
    path = _put(root / USER / "exact.txt", content)
    parts = _parts(inbox, [path])
    ok = parts == [_text_part("exact.txt", content)]
    shown = [(p.get("type"), len(str(p.get("text", ""))), str(p.get("text", ""))[:30]) if isinstance(p, dict) else p for p in parts]
    assert ok, (
        f"parts([exact.txt of {MAX_TEXT_BYTES} bytes]): expected one text part '[file exact.txt]\\n' + {MAX_TEXT_BYTES} × 'a', "
        f"got (type, text length, text head) {shown!r}"
    )


def _text_files(root: Path, count: int) -> list[Path]:
    return [_put(root / USER / f"f{i:02d}.txt", f"file {i}") for i in range(count)]


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_parts_20_paths_accepted(inbox, root):
    """§6: рівно 20 шляхів — межа, що ще дозволена: 20 частин."""
    got = len(_parts(inbox, _text_files(root, MAX_PATHS)))
    assert got == MAX_PATHS, f"parts of {MAX_PATHS} text files: expected {MAX_PATHS} parts, got {got}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_parts_21_paths_refused(inbox, root):
    """§6: понад 20 шляхів (21 справжній файл) → bad_file."""
    paths = [str(p) for p in _text_files(root, MAX_PATHS + 1)]
    expect_manager_error("bad_file", inbox.parts, paths)


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_parts_order_follows_paths(inbox, root):
    """§6: частини йдуть у порядку paths, а не за назвою чи видом файла."""
    b_txt = _put(root / USER / "b.txt", "bee")
    a_png = _put(root / USER / "a.png", b"\x89PNG fake")
    c_mp3 = _put(root / USER / "c.mp3", b"ID3 fake")
    a_txt = _put(root / USER / "a.txt", "ay")
    got = _parts(inbox, [b_txt, a_png, c_mp3, a_txt])
    expected = [
        _text_part("b.txt", "bee"),
        {"type": "image_url", "image_url": {"url": _file_url(a_png)}},
        {"type": "audio_url", "audio_url": {"url": _file_url(c_mp3)}},
        _text_part("a.txt", "ay"),
    ]
    assert got == expected, f"parts in paths order [b.txt, a.png, c.mp3, a.txt]:\n expected {expected!r}\n got      {got!r}"


# --- MCP: files_inbox --------------------------------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_mcp_files_inbox_returns_info(runner_env, inbox, root):
    """§6: MCP files_inbox(user) → info(user) того самого Inbox."""
    body = mcp_json(_mcp(runner_env, inbox), "files_inbox", {"user": USER})
    expected = json.loads(json.dumps(inbox.info(USER), default=str))
    got = (body, body.get("dir") if isinstance(body, dict) else None)
    assert got == (expected, f"{root}/{USER}/"), f"files_inbox({USER!r}): expected info {expected!r}, got {body!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_mcp_files_inbox_unknown_user_refused(runner_env, inbox):
    """§6: files_inbox з логіном не з users → ToolError з unknown_user."""
    message = mcp_refusal(_mcp(runner_env, inbox), "files_inbox", {"user": STRANGER})
    assert "unknown_user" in message, f"files_inbox({STRANGER!r}): expected a refusal with 'unknown_user', got {message!r}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_mcp_files_inbox_listed_with_inbox(runner_env, inbox):
    """§6: build_mcp(…, inbox=Inbox(…)) має інструмент files_inbox."""
    names = _tool_names(_mcp(runner_env, inbox))
    assert "files_inbox" in names, f"MCP tools with an inbox: expected 'files_inbox', got {sorted(names)}"


@pytest.mark.component
@pytest.mark.req("SPEC-GPU-003 §6")
def test_mcp_files_inbox_absent_without_inbox(runner_env):
    """§6: build_mcp без inbox (типове inbox=None) — files_inbox немає."""
    names = _tool_names(runner_env.mcp())
    assert "files_inbox" not in names, f"MCP tools without an inbox: expected no 'files_inbox', got {sorted(names)}"


# --- MCP: llm_ask(files=…) ------------------------------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §6")
def test_llm_ask_files_content_is_parts_then_prompt(proxied_env, upstream, root):
    """§6: з files вміст повідомлення користувача — parts(files), а в кінці {"type": "text", "text": prompt}."""
    path = _put(root / USER / "notes.txt", TEXT_CONTENT)
    server = _mcp(proxied_env, _inbox(root))
    mcp_call(server, "llm_ask", {"model": NAME, "prompt": PROMPT, "files": [str(path)]})
    got = _user_content(upstream)
    expected = [_text_part("notes.txt", TEXT_CONTENT), {"type": "text", "text": PROMPT}]
    assert got == expected, f"llm_ask with files: user message content:\n expected {expected!r}\n got      {got!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §6")
def test_llm_ask_pdf_mode_images_reaches_parts(proxied_env, upstream, root):
    """§6: llm_ask(pdf_mode="images") → сторінки PDF ідуть у модель як image_url з data URL JPEG, далі prompt."""
    path = _put(root / USER / "doc.pdf", _pdf(PDF_PAGES))
    server = _mcp(proxied_env, _inbox(root))
    mcp_call(server, "llm_ask", {"model": NAME, "prompt": PROMPT, "files": [str(path)], "pdf_mode": "images"})
    content = _user_content(upstream)
    items = content if isinstance(content, list) else []
    got = (_image_summary(items[:-1]), items[-1] if items else None)
    expected = ([("image_url", True)] * PDF_PAGES, {"type": "text", "text": PROMPT})
    assert got == expected, (
        f"llm_ask with a {PDF_PAGES}-page PDF, pdf_mode='images': expected ((type, is JPEG data URL) per page, prompt part) "
        f"{expected!r}, got {got!r}; content {str(content)[:300]}"
    )


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §6")
@pytest.mark.parametrize("files", [None, []], ids=["files-omitted", "files-empty"])
def test_llm_ask_without_files_content_is_prompt_string(proxied_env, upstream, root, files):
    """§6: без files (не передано чи порожній список — типове files=[]) вміст повідомлення користувача — рядок prompt."""
    args: dict[str, Any] = {"model": NAME, "prompt": PROMPT}
    if files is not None:
        args["files"] = files
    mcp_call(_mcp(proxied_env, _inbox(root)), "llm_ask", args)
    got = _user_content(upstream)
    assert got == PROMPT, f"llm_ask {args!r}: expected user content to be the plain string {PROMPT!r}, got {got!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §6")
def test_llm_ask_files_without_inbox_refused(proxied_env, upstream, root):
    """§6: files у llm_ask, а build_mcp без inbox → ToolError з bad_request."""
    path = _put(root / USER / "notes.txt", TEXT_CONTENT)
    message = mcp_refusal(proxied_env.mcp(), "llm_ask", {"model": NAME, "prompt": PROMPT, "files": [str(path)]})
    assert "bad_request" in message, f"llm_ask with files and no inbox: expected a refusal with 'bad_request', got {message!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §6")
def test_llm_ask_bad_file_path_refused_without_asking_model(proxied_env, upstream, root, outside):
    """§6: шлях поза root у files → відмова bad_file_path; вміст повідомлення без parts(files) не складається,
    тож до моделі запит не йде."""
    before = len(upstream.requests)
    message = mcp_refusal(
        _mcp(proxied_env, _inbox(root)), "llm_ask", {"model": NAME, "prompt": PROMPT, "files": [str(outside)]}
    )
    got = ("bad_file_path" in message, len(upstream.requests) - before)
    assert got == (True, 0), (
        f"llm_ask with a file outside the inbox: expected (refusal has 'bad_file_path', requests to the model) (True, 0), "
        f"got {got!r}; message {message!r}"
    )


# --- build_app ------------------------------------------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-003 §6")
def test_app_mcp_files_inbox_from_config(make_runner_env, root):
    """§6: build_app передає Inbox(cfg.inbox_dir, cfg.users, <public_host без порту>): files_inbox через /mcp
    застосунку дає dir у files.inbox_dir і copy_remote на <user>@<host без порту>:."""
    env = make_runner_env({"files.inbox_dir": str(root)})
    with env.client() as c:
        _, message = _mcp_http(c, _tool_call(1, "files_inbox", {"user": USER}))
    result = message.get("result") or {}
    text = (result.get("content") or [{}])[0].get("text", "")
    try:
        body = json.loads(text) if not result.get("isError") else {}
    except ValueError:
        body = {}
    body = body if isinstance(body, dict) else {}
    got = (body.get("dir"), f"{USER}@{HOST}:" in str(body.get("copy_remote", "")))
    expected = (f"{root}/{USER}/", True)
    assert got == expected, (
        f"files_inbox over /mcp of build_app: expected (dir, copy_remote has {USER}@{HOST}:) {expected!r}, "
        f"got {got!r}; message {str(message)[:400]}"
    )
