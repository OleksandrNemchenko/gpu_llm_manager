"""§2 SPEC-GPU-002: розділ models конфігу і токен HF (gpu_manager.credentials.hf_token).

Формат налаштувань — як у SPEC-GPU-001 §2 (обгортка {value, comment}); конфіг пишеться в tmp_path
фікстурою write_config. HOME підміняється на теку в tmp_path, тож типовий models.hf_home
(<домашня тека>/hf-cache) не вказує на справжній кеш HF сервера.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from .conftest import config_doc

pytestmark = pytest.mark.component

# Налаштування §2 з типовими значеннями таблиці §2 (рядки 2–5).
MODEL_DEFAULTS = [
    ("max_parallel_downloads", 2),
    ("min_free_disk_gib", 50),
    ("fit_memory_fraction", 0.9),
    ("fit_overhead_gib", 2.0),
]


@pytest.fixture(autouse=True)
def _isolated_home_and_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """HOME і поточна тека — у tmp_path: типовий hf_home і відносні шляхи не зачеплять справжні теки."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    cwd = tmp_path / "isolated-cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return home


def _settings(config_dir: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    settings: dict[str, Any] = {"paths.data_dir": str(config_dir / "data")}
    settings.update(overrides or {})
    return settings


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


# --- Типові значення (§2) -------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §2")
@pytest.mark.parametrize(("field", "expected"), MODEL_DEFAULTS)
def test_models_defaults_without_section(write_config, config_dir, field, expected):
    """§2: розділ models необов'язковий; без нього — типові значення таблиці §2."""
    cfg = _load(write_config(_settings(config_dir)))
    actual = getattr(cfg, field)
    assert actual == expected, f"Config.{field} without a models section: expected {expected!r}, got {actual!r}"


@pytest.mark.req("SPEC-GPU-002 §2.1")
def test_models_default_hf_home_in_home_dir(write_config, config_dir, _isolated_home_and_cwd):
    """§2.1: типовий models.hf_home — <домашня тека>/hf-cache, тип Path."""
    cfg = _load(write_config(_settings(config_dir)))
    expected = _isolated_home_and_cwd / "hf-cache"
    assert isinstance(cfg.hf_home, Path) and cfg.hf_home == expected, (
        f"Config.hf_home default: expected Path {expected}, got {cfg.hf_home!r}"
    )


@pytest.mark.req("SPEC-GPU-002 §2")
@pytest.mark.parametrize(("field", "expected"), MODEL_DEFAULTS)
def test_models_partial_section_keeps_other_defaults(write_config, config_dir, tmp_path, field, expected):
    """§2: розділ models лише з hf_home — решта налаштувань розділу типові."""
    cfg = _load(write_config(_settings(config_dir, {"models.hf_home": str(tmp_path / "cache")})))
    actual = getattr(cfg, field)
    assert actual == expected, f"Config.{field} with only models.hf_home set: expected {expected!r}, got {actual!r}"


# --- Значення з конфігу (§2) ------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §2")
@pytest.mark.parametrize(
    ("field", "dotted", "value"),
    [
        ("max_parallel_downloads", "models.max_parallel_downloads", 3),
        ("min_free_disk_gib", "models.min_free_disk_gib", 120),
        ("fit_memory_fraction", "models.fit_memory_fraction", 0.75),
        ("fit_overhead_gib", "models.fit_overhead_gib", 4.5),
    ],
)
def test_models_setting_maps_to_field(write_config, config_dir, field, dotted, value):
    """§2: значення налаштування розділу models потрапляє у відповідне поле Config."""
    cfg = _load(write_config(_settings(config_dir, {dotted: value})))
    actual = getattr(cfg, field)
    assert actual == value, f"Config.{field} from {dotted}={value!r}: expected {value!r}, got {actual!r}"


@pytest.mark.req("SPEC-GPU-002 §2.1")
def test_models_hf_home_from_config_is_path(write_config, config_dir, tmp_path):
    """§2.1: models.hf_home з конфігу (абсолютний шлях) — поле hf_home типу Path."""
    target = tmp_path / "custom-cache"
    cfg = _load(write_config(_settings(config_dir, {"models.hf_home": str(target)})))
    assert isinstance(cfg.hf_home, Path) and cfg.hf_home.resolve() == target.resolve(), (
        f"Config.hf_home: expected Path {target}, got {cfg.hf_home!r}"
    )


# Відносні models.hf_home і куди вони мають вести відносно теки конфігу (§2.1).
RELATIVE_HF_HOMES = [
    ("hf-cache", ("hf-cache",)),
    ("cache/hf", ("cache", "hf")),
    ("../hf-shared", ("..", "hf-shared")),
]


@pytest.mark.req("SPEC-GPU-002 §2.1")
@pytest.mark.parametrize(("value", "parts"), RELATIVE_HF_HOMES, ids=[v for v, _ in RELATIVE_HF_HOMES])
def test_models_hf_home_relative_resolved_from_config_dir(write_config, config_dir, value, parts):
    """§2.1: відносний models.hf_home — від теки конфігу, а не від поточної теки процесу (вона тут інша)."""
    cfg = _load(write_config(_settings(config_dir, {"models.hf_home": value})))
    expected = config_dir.joinpath(*parts).resolve()
    got = Path(cfg.hf_home).resolve()
    assert got == expected, f"Config.hf_home for relative {value!r}: expected {expected} (from the config dir), got {cfg.hf_home!r} -> {got}"


@pytest.mark.req("SPEC-GPU-002 §2.1")
@pytest.mark.parametrize("value", [v for v, _ in RELATIVE_HF_HOMES])
def test_models_hf_home_relative_is_absolute_in_config(write_config, config_dir, value):
    """§2.1: у Config поле hf_home — абсолютний Path, навіть коли в конфігу шлях відносний."""
    cfg = _load(write_config(_settings(config_dir, {"models.hf_home": value})))
    ok = isinstance(cfg.hf_home, Path) and cfg.hf_home.is_absolute()
    assert ok, f"Config.hf_home for relative {value!r}: expected an absolute Path, got {cfg.hf_home!r}"


@pytest.mark.req("SPEC-GPU-002 §2")
def test_models_secrets_path_next_to_config(write_config, config_dir):
    """§2: secrets_path = <тека конфігу>/secrets.json."""
    cfg = _load(write_config(_settings(config_dir)))
    expected = (config_dir / "secrets.json").resolve()
    assert Path(cfg.secrets_path).resolve() == expected, f"Config.secrets_path: expected {expected}, got {cfg.secrets_path!r}"


# --- Межі (§2.2–2.5) ---------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §2.2")
@pytest.mark.parametrize("value", [0, -1])
def test_models_max_parallel_below_one_refused(write_config, config_dir, value):
    """§2.2: models.max_parallel_downloads < 1 → ConfigError."""
    path = write_config(_settings(config_dir, {"models.max_parallel_downloads": value}))
    _expect_config_error(path, f"models.max_parallel_downloads={value}")


@pytest.mark.req("SPEC-GPU-002 §2.2")
def test_models_max_parallel_one_accepted(write_config, config_dir):
    """§2.2: межа 1 допустима."""
    cfg = _load(write_config(_settings(config_dir, {"models.max_parallel_downloads": 1})))
    assert cfg.max_parallel_downloads == 1, f"Config.max_parallel_downloads: expected 1, got {cfg.max_parallel_downloads!r}"


@pytest.mark.req("SPEC-GPU-002 §2.4")
@pytest.mark.parametrize("value", [0, -0.1, 1.01, 2])
def test_models_fraction_outside_range_refused(write_config, config_dir, value):
    """§2.4: models.fit_memory_fraction поза (0, 1] → ConfigError (0 — поза, ліва межа відкрита)."""
    path = write_config(_settings(config_dir, {"models.fit_memory_fraction": value}))
    _expect_config_error(path, f"models.fit_memory_fraction={value}")


@pytest.mark.req("SPEC-GPU-002 §2.4")
@pytest.mark.parametrize("value", [1, 1.0, 0.001])
def test_models_fraction_inside_range_accepted(write_config, config_dir, value):
    """§2.4: 1 (права межа включна) і мале додатне значення допустимі."""
    cfg = _load(write_config(_settings(config_dir, {"models.fit_memory_fraction": value})))
    assert cfg.fit_memory_fraction == value, f"Config.fit_memory_fraction: expected {value!r}, got {cfg.fit_memory_fraction!r}"


@pytest.mark.req("SPEC-GPU-002 §2.3")
@pytest.mark.parametrize("value", [-1, -0.5])
def test_models_min_free_disk_negative_refused(write_config, config_dir, value):
    """§2.3: models.min_free_disk_gib < 0 → ConfigError."""
    path = write_config(_settings(config_dir, {"models.min_free_disk_gib": value}))
    _expect_config_error(path, f"models.min_free_disk_gib={value}")


@pytest.mark.req("SPEC-GPU-002 §2.3")
def test_models_min_free_disk_zero_accepted(write_config, config_dir):
    """§2.3: межа 0 допустима (заборонено лише < 0)."""
    cfg = _load(write_config(_settings(config_dir, {"models.min_free_disk_gib": 0})))
    assert cfg.min_free_disk_gib == 0, f"Config.min_free_disk_gib: expected 0 GiB, got {cfg.min_free_disk_gib!r}"


@pytest.mark.req("SPEC-GPU-002 §2.5")
@pytest.mark.parametrize("value", [-0.1, -2])
def test_models_fit_overhead_negative_refused(write_config, config_dir, value):
    """§2.5: models.fit_overhead_gib < 0 → ConfigError."""
    path = write_config(_settings(config_dir, {"models.fit_overhead_gib": value}))
    _expect_config_error(path, f"models.fit_overhead_gib={value}")


@pytest.mark.req("SPEC-GPU-002 §2.5")
@pytest.mark.parametrize("value", [0, 0.0])
def test_models_fit_overhead_zero_accepted(write_config, config_dir, value):
    """§2.5: межа 0 допустима (заборонено лише < 0)."""
    cfg = _load(write_config(_settings(config_dir, {"models.fit_overhead_gib": value})))
    assert cfg.fit_overhead_gib == 0, f"Config.fit_overhead_gib: expected 0 GiB, got {cfg.fit_overhead_gib!r}"


@pytest.mark.req("SPEC-GPU-002 §2")
@pytest.mark.parametrize("dotted", ["models.max_parallel_downloads", "models.hf_home"])
def test_models_bare_value_refused_with_path(write_config, config_dir, tmp_path, dotted):
    """§2 (формат — SPEC-GPU-001 §2.3): голе значення в розділі models → ConfigError зі шляхом налаштування."""
    value = 2 if dotted.endswith("downloads") else str(tmp_path / "cache")
    doc = config_doc(_settings(config_dir, {dotted: value}))
    section, key = dotted.split(".")
    doc[section][key] = value
    message = _expect_config_error(write_config(doc=doc), f"bare value at {dotted}")
    assert dotted in message, f"ConfigError text must name {dotted!r}, got {message!r}"


# --- Токен HF (§2) -------------------------------------------------------------------------------------------------------


def _secrets(tmp_path: Path, content: Any) -> Path:
    """Пише secrets.json: bytes — як є, інакше — JSON."""
    path = tmp_path / "etc-secrets" / "secrets.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    return path


def _token(path: Path) -> Any:
    from gpu_manager.credentials import hf_token

    return hf_token(path)


def _wrapped(value: Any) -> dict[str, Any]:
    return {"hf_token": {"value": value, "comment": "HuggingFace access token"}}


@pytest.mark.req("SPEC-GPU-002 §2")
def test_hf_token_real_value_returned(tmp_path):
    """§2: hf_token(path) повертає hf_token.value з secrets.json."""
    got = _token(_secrets(tmp_path, _wrapped("token-value-123")))
    assert got == "token-value-123", f"hf_token: expected 'token-value-123', got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §2")
def test_hf_token_missing_file_is_none(tmp_path):
    """§2: файлу немає → None."""
    got = _token(tmp_path / "absent" / "secrets.json")
    assert got is None, f"hf_token of a missing file: expected None, got {got!r}"


NO_TOKEN_CONTENTS = {
    "not-json": b"{hf_token: oops",
    "empty-file": b"",
    "no-hf-token-key": {"other": {"value": "x", "comment": "c"}},
    "no-value-key": {"hf_token": {"comment": "HuggingFace access token"}},
    "empty-value": _wrapped(""),
    "placeholder": _wrapped("PASTE_YOUR_HF_TOKEN_HERE"),
    "bare-placeholder": _wrapped("PASTE_"),
    "json-array": [],
}


@pytest.mark.req("SPEC-GPU-002 §2")
@pytest.mark.parametrize("case", sorted(NO_TOKEN_CONTENTS))
def test_hf_token_absent_or_placeholder_is_none(tmp_path, case):
    """§2: не JSON, поля немає, значення порожнє або починається з PASTE_ → None."""
    got = _token(_secrets(tmp_path, NO_TOKEN_CONTENTS[case]))
    assert got is None, f"hf_token for {case}: expected None, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §2")
def test_hf_token_paste_inside_value_is_not_placeholder(tmp_path):
    """§2: заглушка — значення, що ПОЧИНАЄТЬСЯ з PASTE_; PASTE_ всередині — звичайний токен."""
    got = _token(_secrets(tmp_path, _wrapped("hf_PASTE_real0123")))
    assert got == "hf_PASTE_real0123", f"hf_token 'hf_PASTE_real0123': expected it returned, got {got!r}"
