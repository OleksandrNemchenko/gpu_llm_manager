"""§8 (рядок 8) і §10 SPEC-GPU-001: інструкція для агентів — render_guide(cfg) та інструмент gpu_guide.

render_guide(cfg) будує Markdown: логіни з users.allowed, files.inbox_dir, адресу public_host без порту
і стан теки (Status: ready. / Status: NOT set up …). build_mcp(manager, tz, guide) віддає через gpu_guide
текст функції guide; без guide — "No guide configured.". Тека вхідних файлів — завжди в tmp_path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from .conftest import PUBLIC_HOST, USERS, has_cyrillic, mcp_call, mcp_text

pytestmark = pytest.mark.component

DEFAULT_INBOX = "/srv/gpu-inbox"  # типове files.inbox_dir (§2.4)
READY = "Status: ready."
NOT_SET_UP = "Status: NOT set up"
NO_GUIDE = "No guide configured."
PUBLIC_ADDRESS = PUBLIC_HOST.rsplit(":", 1)[0]  # public_host без порту: 203.0.113.7
# Адреса як окреме слово й без «:<порт>» одразу після неї.
_ADDRESS_WITHOUT_PORT = re.compile(rf"(?<![\d.]){re.escape(PUBLIC_ADDRESS)}(?![\d]|:\d)")


def _render(cfg: Any) -> str:
    from gpu_manager.app import render_guide

    text = render_guide(cfg)
    assert isinstance(text, str), f"render_guide: expected str, got {type(text).__name__}"
    return text


@pytest.fixture
def inbox(tmp_path: Path) -> Path:
    """Шлях теки вхідних файлів у tmp_path; тест сам вирішує, чи створювати її."""
    return tmp_path / "gpu-inbox"


@pytest.fixture
def guide_env(make_env, inbox):
    """Сервіс, у конфігу якого files.inbox_dir вказує на inbox."""
    return make_env({"files.inbox_dir": str(inbox)})


# --- render_guide(cfg) ------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §8.8")
def test_render_guide_lists_every_allowed_login(guide_env):
    """§8.8: інструкція перелічує логіни з users.allowed."""
    text = _render(guide_env.cfg)
    missing = [login for login in USERS if not re.search(rf"\b{re.escape(login)}\b", text)]
    assert not missing, f"guide: expected every login of users.allowed, missing {missing} in {text[:300]!r}"


@pytest.mark.req("SPEC-GPU-001 §8.8")
def test_render_guide_shows_inbox_path(guide_env, inbox):
    """§8.8: інструкція містить шлях files.inbox_dir."""
    text = _render(guide_env.cfg)
    assert str(inbox) in text, f"guide: expected inbox path {str(inbox)!r}, got {text[:300]!r}"


@pytest.mark.req("SPEC-GPU-001 §8.8")
@pytest.mark.req("SPEC-GPU-001 §2.4")
def test_render_guide_default_inbox_path(make_env):
    """§2.4, §8.8: без files.inbox_dir інструкція показує типовий шлях /srv/gpu-inbox."""
    text = _render(make_env().cfg)
    assert DEFAULT_INBOX in text, f"guide without files.inbox_dir: expected {DEFAULT_INBOX!r}, got {text[:300]!r}"


@pytest.mark.req("SPEC-GPU-001 §8.8")
def test_render_guide_shows_public_host_without_port(guide_env):
    """§8.8: адреса public_host у інструкції — без порту."""
    text = _render(guide_env.cfg)
    assert _ADDRESS_WITHOUT_PORT.search(text), (
        f"guide: expected public address {PUBLIC_ADDRESS!r} without ':<port>', got {text[:300]!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §8.8")
def test_render_guide_status_ready_when_inbox_exists(guide_env, inbox):
    """§8.8: тека files.inbox_dir існує → "Status: ready." (і немає "Status: NOT set up")."""
    inbox.mkdir()
    text = _render(guide_env.cfg)
    assert READY in text and NOT_SET_UP not in text, (
        f"guide with an existing inbox: expected {READY!r} and no {NOT_SET_UP!r}, got {text[:300]!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §8.8")
def test_render_guide_status_not_set_up_when_inbox_missing(guide_env, inbox):
    """§8.8: теки files.inbox_dir немає → "Status: NOT set up …" (і немає "Status: ready.")."""
    assert not inbox.exists(), f"precondition: {inbox} must not exist"
    text = _render(guide_env.cfg)
    assert NOT_SET_UP in text and READY not in text, (
        f"guide with a missing inbox: expected {NOT_SET_UP!r} and no {READY!r}, got {text[:300]!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §8.8")
def test_render_guide_status_follows_inbox_created_later(guide_env, inbox):
    """§8.8: стан теки визначається під час побудови інструкції — тека, створена пізніше, дає ready."""
    before = _render(guide_env.cfg)
    inbox.mkdir()
    after = _render(guide_env.cfg)
    assert NOT_SET_UP in before and READY in after, (
        f"guide before/after creating the inbox: expected {NOT_SET_UP!r} then {READY!r}, "
        f"got {before[:120]!r} / {after[:120]!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §8")
def test_render_guide_is_english(guide_env, inbox):
    """§8: тексти MCP — англійською; в інструкції немає кирилиці."""
    inbox.mkdir()
    text = _render(guide_env.cfg)
    assert not has_cyrillic(text), f"guide: expected English text, found Cyrillic in {text[:300]!r}"


# --- gpu_guide через build_mcp -----------------------------------------------------------------------------

GUIDE_MARKDOWN = "# GPU guide\n\n- step one\n- step two\n"  # Markdown із переносами — не JSON


@pytest.mark.req("SPEC-GPU-001 §10")
def test_gpu_guide_without_guide_says_not_configured(env):
    """§10: build_mcp без guide — gpu_guide повертає "No guide configured."."""
    text = mcp_text(mcp_call(env.mcp(), "gpu_guide"))
    assert text.strip() == NO_GUIDE, f"gpu_guide without guide: expected {NO_GUIDE!r}, got {text[:200]!r}"


@pytest.mark.req("SPEC-GPU-001 §8.8")
@pytest.mark.req("SPEC-GPU-001 §10")
def test_gpu_guide_returns_guide_text_verbatim(env):
    """§8.8, §10: gpu_guide віддає текст guide() як є — Markdown із переносами, не JSON-рядок."""
    text = mcp_text(mcp_call(env.mcp(guide=lambda: GUIDE_MARKDOWN), "gpu_guide"))
    assert text == GUIDE_MARKDOWN, f"gpu_guide: expected the guide text verbatim {GUIDE_MARKDOWN!r}, got {text!r}"


@pytest.mark.req("SPEC-GPU-001 §8")
def test_gpu_guide_is_single_text_block(env):
    """§8: результат gpu_guide — один текстовий блок."""
    result = mcp_call(env.mcp(guide=lambda: GUIDE_MARKDOWN), "gpu_guide")
    blocks = list(result.content)
    assert len(blocks) == 1 and getattr(blocks[0], "type", None) == "text", (
        f"gpu_guide: expected exactly one text block, got {[getattr(b, 'type', type(b).__name__) for b in blocks]}"
    )


@pytest.mark.req("SPEC-GPU-001 §10")
def test_gpu_guide_calls_guide_on_every_call(env):
    """§10: guide — функція, тож gpu_guide викликає її щоразу (стан теки не застигає на старті)."""
    texts = iter(["first rendering", "second rendering"])
    server = env.mcp(guide=lambda: next(texts))
    first = mcp_text(mcp_call(server, "gpu_guide"))
    second = mcp_text(mcp_call(server, "gpu_guide"))
    assert (first, second) == ("first rendering", "second rendering"), (
        f"gpu_guide twice: expected ('first rendering', 'second rendering'), got {(first, second)!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §8.8")
@pytest.mark.req("SPEC-GPU-001 §10")
def test_gpu_guide_with_render_guide_reports_inbox_status(guide_env, inbox):
    """§8.8, §10: build_mcp(guide=lambda: render_guide(cfg)) — gpu_guide показує стан теки."""
    from gpu_manager.app import render_guide

    inbox.mkdir()
    text = mcp_text(mcp_call(guide_env.mcp(guide=lambda: render_guide(guide_env.cfg)), "gpu_guide"))
    assert READY in text, f"gpu_guide via render_guide with an existing inbox: expected {READY!r}, got {text[:300]!r}"
