"""§2 SPEC-GPU-001: конфіг config.json — обгортка налаштувань, обов'язкові поля, межі значень, шляхи.

Кожен тест пише власний config.json у tmp_path і викликає gpu_manager.config.load_config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from .conftest import (
    DELETE,
    HOSTS,
    MODEL_PORTS,
    PORT,
    REQUIRED_SETTINGS,
    TZ_NAME,
    USERS,
    config_doc,
)

pytestmark = pytest.mark.component

# Налаштування, на яких перевіряється формат обгортки (§2.2–2.3): різні розділи й типи значень.
WRAPPED_PATHS = ["server.port", "users.allowed", "journal.keep_days"]


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Поточна тека — у tmp_path: помилкове тлумачення відносного data_dir не зачепить репозиторій."""
    cwd = tmp_path / "isolated-cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)


def _load(path: Path) -> Any:
    from gpu_manager.config import load_config

    return load_config(path)


def _expect_config_error(path: Path, what: str) -> str:
    """Повертає текст ConfigError; якщо конфіг завантажився — тест падає з поясненням."""
    from gpu_manager.config import ConfigError, load_config

    try:
        cfg = load_config(path)
    except ConfigError as exc:
        return str(exc)
    pytest.fail(f"{what}: expected ConfigError, the config loaded as {cfg!r}")


def _node(doc: dict[str, Any], dotted: str) -> tuple[dict[str, Any], str]:
    """Розділ, що містить налаштування dotted, і ключ налаштування в ньому."""
    section, key = dotted.split(".")
    return doc[section], key


# --- Поля Config -------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §2.4")
@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("hosts", HOSTS),
        ("port", PORT),
        ("users", USERS),
        ("sample_interval_s", 1),
        ("ring_keep_s", 3600),
        ("history_db_interval_s", 60),
        ("history_db_keep_days", 30),
        ("busy_memory_mib", 1024),
        ("journal_keep_entries", 1000),
        ("journal_keep_days", 90),
    ],
)
def test_config_valid_file_maps_to_fields(write_config, field, expected):
    """§2.4, «Поля Config»: значення налаштування потрапляє у відповідне поле Config."""
    cfg = _load(write_config())
    actual = getattr(cfg, field)
    if isinstance(expected, list):
        actual = list(actual)
    assert actual == expected, f"Config.{field}: expected {expected!r}, got {actual!r}"


@pytest.mark.req("SPEC-GPU-001 §2.4")
def test_config_display_timezone_is_zoneinfo(write_config):
    """§2.4, «Поля Config»: display_timezone — ZoneInfo поясу з server.display_timezone."""
    tz = _load(write_config()).display_timezone
    assert isinstance(tz, ZoneInfo) and tz.key == TZ_NAME, (
        f"Config.display_timezone: expected ZoneInfo({TZ_NAME!r}), got {tz!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §2.4")
def test_config_model_ports_is_tuple(write_config):
    """§2 «Поля Config»: model_ports — кортеж (від, до) з vllm.port_range."""
    ports = _load(write_config()).model_ports
    assert isinstance(ports, tuple) and ports == tuple(MODEL_PORTS), (
        f"Config.model_ports: expected tuple {tuple(MODEL_PORTS)!r}, got {ports!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §2.9")
def test_config_host_names_hosts_then_extra_without_duplicates(write_config):
    """§2.9: host_names = server.hosts + server.extra_host_names, без повторів, у цьому порядку."""
    path = write_config({"server.extra_host_names": ["gpu.lan", "127.0.0.1", "gpu.lan"]})
    names = list(_load(path).host_names)
    expected = ["127.0.0.1", "203.0.113.7", "gpu.lan"]
    assert names == expected, f"Config.host_names: expected {expected!r}, got {names!r}"


@pytest.mark.req("SPEC-GPU-001 §2.4")
def test_config_extra_host_names_optional(write_config):
    """§2.4: server.extra_host_names необов'язкове; без нього host_names = server.hosts."""
    names = list(_load(write_config({"server.extra_host_names": DELETE})).host_names)
    assert names == HOSTS, f"Config.host_names without extra names: expected {HOSTS!r}, got {names!r}"


@pytest.mark.req("SPEC-GPU-001 §2.4")
def test_config_inbox_dir_setting_accepted(write_config, tmp_path):
    """§2.4: необов'язкове files.inbox_dir у звичайній обгортці — конфіг завантажується."""
    from gpu_manager.config import ConfigError

    try:
        _load(write_config({"files.inbox_dir": str(tmp_path / "inbox")}))
    except ConfigError as exc:
        pytest.fail(f"files.inbox_dir is an optional setting: expected the config to load, got ConfigError: {exc}")


@pytest.mark.req("SPEC-GPU-001 §2.3")
def test_config_inbox_dir_bare_value_refused_with_path(write_config, tmp_path):
    """§2.3: files.inbox_dir голим значенням → ConfigError зі шляхом (правило діє й на необов'язкові)."""
    doc = config_doc({"files.inbox_dir": str(tmp_path / "inbox")})
    doc["files"]["inbox_dir"] = str(tmp_path / "inbox")
    message = _expect_config_error(write_config(doc=doc), "bare value at files.inbox_dir")
    assert "files.inbox_dir" in message, f"ConfigError text must name 'files.inbox_dir', got {message!r}"


# --- Шляхи -----------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §2.8")
def test_config_relative_data_dir_resolved_from_config_dir(write_config, config_dir, tmp_path, monkeypatch):
    """§2.8: відносний paths.data_dir рахується від теки конфігу, а не від поточної теки процесу."""
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cfg = _load(write_config({"paths.data_dir": "state/data"}))
    expected = (config_dir / "state" / "data").resolve()
    assert isinstance(cfg.data_dir, Path), f"Config.data_dir: expected Path, got {type(cfg.data_dir).__name__}"
    assert cfg.data_dir.resolve() == expected, (
        f"Config.data_dir: expected {expected}, got {cfg.data_dir} (cwd was {elsewhere})"
    )


@pytest.mark.req("SPEC-GPU-001 §2.8")
def test_config_absolute_data_dir_kept(write_config, tmp_path):
    """§2.8: абсолютний paths.data_dir використовується як є."""
    target = tmp_path / "abs-data"
    cfg = _load(write_config({"paths.data_dir": str(target)}))
    assert Path(cfg.data_dir).resolve() == target.resolve(), (
        f"Config.data_dir: expected {target}, got {cfg.data_dir}"
    )


# --- Формат обгортки (§2.2–2.3) ------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §2.3")
@pytest.mark.parametrize("dotted", WRAPPED_PATHS)
def test_config_bare_value_refused_with_path(write_config, dotted):
    """§2.3: голе значення замість {value, comment} → ConfigError, у тексті — шлях налаштування."""
    doc = config_doc()
    section, key = _node(doc, dotted)
    section[key] = section[key]["value"]
    message = _expect_config_error(write_config(doc=doc), f"bare value at {dotted}")
    assert dotted in message, f"ConfigError text must name {dotted!r}, got {message!r}"


@pytest.mark.req("SPEC-GPU-001 §2.3")
@pytest.mark.parametrize("dotted", WRAPPED_PATHS)
def test_config_extra_key_refused_with_path(write_config, dotted):
    """§2.2–2.3: налаштування з третім ключем поряд із value і comment → ConfigError зі шляхом."""
    doc = config_doc()
    section, key = _node(doc, dotted)
    section[key]["unit"] = "extra"
    message = _expect_config_error(write_config(doc=doc), f"extra key at {dotted}")
    assert dotted in message, f"ConfigError text must name {dotted!r}, got {message!r}"


@pytest.mark.req("SPEC-GPU-001 §2.3")
@pytest.mark.parametrize("dotted", WRAPPED_PATHS)
def test_config_empty_comment_refused_with_path(write_config, dotted):
    """§2.2–2.3: порожній comment налаштування → ConfigError зі шляхом."""
    doc = config_doc()
    section, key = _node(doc, dotted)
    section[key]["comment"] = ""
    message = _expect_config_error(write_config(doc=doc), f"empty comment at {dotted}")
    assert dotted in message, f"ConfigError text must name {dotted!r}, got {message!r}"


@pytest.mark.req("SPEC-GPU-001 §2.2")
@pytest.mark.parametrize("dotted", WRAPPED_PATHS)
def test_config_setting_without_comment_refused_with_path(write_config, dotted):
    """§2.2–2.3: налаштування лише з value, без comment → ConfigError зі шляхом."""
    doc = config_doc()
    section, key = _node(doc, dotted)
    del section[key]["comment"]
    message = _expect_config_error(write_config(doc=doc), f"setting without comment at {dotted}")
    assert dotted in message, f"ConfigError text must name {dotted!r}, got {message!r}"


@pytest.mark.req("SPEC-GPU-001 §2.2")
@pytest.mark.parametrize("dotted", WRAPPED_PATHS)
def test_config_non_string_comment_refused_with_path(write_config, dotted):
    """§2.2: comment налаштування має бути рядком; число → ConfigError зі шляхом."""
    doc = config_doc()
    section, key = _node(doc, dotted)
    section[key]["comment"] = 5
    message = _expect_config_error(write_config(doc=doc), f"non-string comment at {dotted}")
    assert dotted in message, f"ConfigError text must name {dotted!r}, got {message!r}"


# --- Обов'язкові налаштування (§2.4) ------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §2.4")
@pytest.mark.parametrize("dotted", REQUIRED_SETTINGS)
def test_config_missing_required_setting_refused(write_config, dotted):
    """§2.4: відсутнє обов'язкове налаштування → ConfigError."""
    _expect_config_error(write_config({dotted: DELETE}), f"missing required {dotted}")


# --- Адреси й користувачі (§2.5) -------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §2.5")
@pytest.mark.parametrize("bad_host", ["0.0.0.0", "::", ""], ids=["any-ipv4", "any-ipv6", "empty"])
def test_config_wildcard_or_empty_host_refused(write_config, bad_host):
    """§2.5: server.hosts з 0.0.0.0, :: або порожнім рядком → ConfigError."""
    _expect_config_error(write_config({"server.hosts": ["127.0.0.1", bad_host]}), f"server.hosts with {bad_host!r}")


@pytest.mark.req("SPEC-GPU-001 §2.5")
def test_config_empty_hosts_refused(write_config):
    """§2.5: порожній server.hosts → ConfigError."""
    _expect_config_error(write_config({"server.hosts": []}), "empty server.hosts")


@pytest.mark.req("SPEC-GPU-001 §2.5")
def test_config_empty_users_refused(write_config):
    """§2.5: порожній users.allowed → ConfigError."""
    _expect_config_error(write_config({"users.allowed": []}), "empty users.allowed")


# --- Порти (§2.6) -----------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §2.6")
@pytest.mark.parametrize("port", [0, 1023, 65536])
def test_config_port_out_of_range_refused(write_config, port):
    """§2.6: server.port поза 1024..65535 → ConfigError."""
    _expect_config_error(write_config({"server.port": port}), f"server.port={port}")


@pytest.mark.req("SPEC-GPU-001 §2.6")
@pytest.mark.parametrize("port", [1024, 65535])
def test_config_port_range_boundaries_accepted(write_config, port):
    """§2.6: межі 1024 і 65535 для server.port допустимі (діапазон включний)."""
    cfg = _load(write_config({"server.port": port}))
    assert cfg.port == port, f"Config.port: expected {port}, got {cfg.port!r}"


@pytest.mark.req("SPEC-GPU-001 §2.6")
@pytest.mark.parametrize("port_range", [[1023, 1100], [65000, 65536]], ids=["below-1024", "above-65535"])
def test_config_model_ports_out_of_range_refused(write_config, port_range):
    """§2.6: vllm.port_range поза 1024..65535 → ConfigError."""
    _expect_config_error(write_config({"vllm.port_range": port_range}), f"vllm.port_range={port_range}")


@pytest.mark.req("SPEC-GPU-001 §2.6")
def test_config_model_ports_reversed_refused(write_config):
    """§2.6: vllm.port_range з від > до → ConfigError."""
    _expect_config_error(write_config({"vllm.port_range": [8100, 8000]}), "vllm.port_range=[8100, 8000]")


@pytest.mark.req("SPEC-GPU-001 §2.6")
@pytest.mark.parametrize("port", [8000, 8050, 8099], ids=["first", "middle", "last"])
def test_config_model_ports_containing_server_port_refused(write_config, port):
    """§2.6: vllm.port_range, що містить server.port (межі включно), → ConfigError."""
    path = write_config({"server.port": port, "vllm.port_range": [8000, 8099]})
    _expect_config_error(path, f"vllm.port_range [8000, 8099] containing server.port={port}")


@pytest.mark.req("SPEC-GPU-001 §2.6")
@pytest.mark.parametrize("port", [7999, 8100], ids=["just-below", "just-above"])
def test_config_model_ports_next_to_server_port_accepted(write_config, port):
    """§2.6: server.port одразу поза межами vllm.port_range — допустимо."""
    cfg = _load(write_config({"server.port": port, "vllm.port_range": [8000, 8099]}))
    assert cfg.model_ports == (8000, 8099), f"Config.model_ports: expected (8000, 8099), got {cfg.model_ports!r}"


@pytest.mark.req("SPEC-GPU-001 §2.6")
def test_config_single_port_model_range_accepted(write_config):
    """§2.6: від == до — не «від > до», отже діапазон з одного порту допустимий."""
    cfg = _load(write_config({"vllm.port_range": [8000, 8000]}))
    assert cfg.model_ports == (8000, 8000), f"Config.model_ports: expected (8000, 8000), got {cfg.model_ports!r}"


# --- Додатні межі (§2.7) ------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §2.7")
@pytest.mark.parametrize(
    ("dotted", "value"),
    [
        ("gpu.sample_interval_s", 0),
        ("gpu.sample_interval_s", -1),
        ("journal.keep_entries", 0),
        ("journal.keep_entries", -5),
        ("journal.keep_days", 0),
        ("journal.keep_days", -1),
    ],
)
def test_config_non_positive_limit_refused(write_config, dotted, value):
    """§2.7: sample_interval_s, journal.keep_entries або journal.keep_days ≤ 0 → ConfigError."""
    _expect_config_error(write_config({dotted: value}), f"{dotted}={value}")
