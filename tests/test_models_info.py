"""§5.4.3–5.4.4 SPEC-GPU-002: ModelStore.info (оцінка до завантаження) і ModelStore.search (межі limit).

info() бере файли й config.json з підробленого hub (§4), обсяги карт — з card_mib(), частку й запас —
з аргументу або конфігу (§2), і повертає Fit як словник (§3). Очікуваний per_card рахує оракул
fit_card_oracle за формулою §3.4 з тих самих вхідних даних.
"""

from __future__ import annotations

from typing import Any

import pytest

from .model_fakes import (
    CARD_MIB,
    GATED_OK_REPO,
    MISSING_REPO,
    NO_WEIGHTS_FILES,
    NO_WEIGHTS_REPO,
    REPO,
    TINY_KV,
    TINY_MAX_POSITION,
    TINY_TOTAL,
    TINY_WEIGHTS,
    card_view,
    expect_manager_error,
    fit_card_oracle,
    gib_close,
    per_card_keys,
)

pytestmark = pytest.mark.component

INFO_FIELDS = {"repo", "revision", "gated", "files", "size_gib", "fraction", "fit", "local"}
FIT_FIELDS = {"weights_gib", "kv_kib_per_token", "max_context", "per_card", "warnings"}


def _info(env: Any, repo: str = REPO, **kwargs: Any) -> dict[str, Any]:
    return env.store.info(repo, **kwargs)


# --- §5.4.3 info() ---------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §5.4.3")
def test_info_fields(models_env):
    """§5.4.3: {repo, revision, gated, files, size_gib, fraction, fit, local}."""
    body = _info(models_env)
    missing = INFO_FIELDS - set(body)
    assert not missing, f"info(): missing fields {sorted(missing)}, got {sorted(body)}"


@pytest.mark.req("SPEC-GPU-002 §5.4.3")
def test_info_repo_value(models_env):
    """§5.4.3: repo — запитаний репозиторій."""
    got = _info(models_env).get("repo")
    assert got == REPO, f"info() repo: expected {REPO!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.3")
def test_info_fit_is_dict_with_fit_fields(models_env):
    """§5.4.3: fit — Fit як словник (поля §3)."""
    fit = _info(models_env).get("fit")
    ok = isinstance(fit, dict) and FIT_FIELDS <= set(fit)
    assert ok, f"info() fit: expected a dict with {sorted(FIT_FIELDS)}, got {fit!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.3")
def test_info_size_gib(models_env):
    """§5.4.3: size_gib — сума розмірів вибраних файлів з API."""
    got = _info(models_env).get("size_gib")
    assert gib_close(got, TINY_TOTAL), f"info() size_gib: expected ≈{TINY_TOTAL / 2**30:.3f} GiB, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.3")
@pytest.mark.parametrize(("repo", "expected"), [(REPO, False), (GATED_OK_REPO, True)], ids=["open", "gated"])
def test_info_gated_flag(models_env, repo, expected):
    """§5.4.3, §4: gated — з hub.files."""
    got = _info(models_env, repo).get("gated")
    assert got is expected, f"info({repo!r}) gated: expected {expected}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.3")
@pytest.mark.req("SPEC-GPU-002 §2.4")
def test_info_fraction_defaults_to_config(make_models_env):
    """§5.4.3, §2.4: fraction не задано — models.fit_memory_fraction (тут 0.8)."""
    env = make_models_env({"models.fit_memory_fraction": 0.8})
    got = _info(env).get("fraction")
    assert got == 0.8, f"info() fraction without an argument: expected 0.8 from the config, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.3")
def test_info_explicit_fraction_reported(models_env):
    """§5.4.3: задана fraction повертається у відповіді."""
    got = _info(models_env, fraction=0.5).get("fraction")
    assert got == 0.5, f"info(fraction=0.5) fraction: expected 0.5, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.3")
@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_info_fit_per_card_matches_formula(models_env):
    """§5.4.3, §3.4: per_card для карти A40 — за формулою з fraction 0.9 і запасом 2.0 з конфігу."""
    got = card_view(_info(models_env)["fit"], CARD_MIB)
    expected = fit_card_oracle(CARD_MIB, 0.9, TINY_WEIGHTS, 2.0, TINY_KV, TINY_MAX_POSITION)
    assert got == expected, f"info() per_card[{CARD_MIB}]: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.3")
@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_info_fit_uses_explicit_fraction(models_env):
    """§5.4.3: оцінка рахується із заданою fraction (0.5 → usable ≈ 20.0 GiB замість 38.0)."""
    got = card_view(_info(models_env, fraction=0.5)["fit"], CARD_MIB)["usable_gib"]
    expected = fit_card_oracle(CARD_MIB, 0.5, TINY_WEIGHTS, 2.0, TINY_KV, TINY_MAX_POSITION)["usable_gib"]
    assert got == expected, f"info(fraction=0.5) usable_gib: expected {expected}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.3")
@pytest.mark.req("SPEC-GPU-002 §2.5")
def test_info_fit_uses_config_overhead(make_models_env):
    """§5.4.3, §2.5: запас на активації — models.fit_overhead_gib (тут 3.5)."""
    env = make_models_env({"models.fit_overhead_gib": 3.5})
    got = card_view(_info(env)["fit"], CARD_MIB)["usable_gib"]
    expected = fit_card_oracle(CARD_MIB, 0.9, TINY_WEIGHTS, 3.5, TINY_KV, TINY_MAX_POSITION)["usable_gib"]
    assert got == expected, f"info() usable_gib with fit_overhead_gib 3.5: expected {expected}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.3")
@pytest.mark.req("SPEC-GPU-002 §5")
def test_info_fit_cards_from_card_mib(make_models_env):
    """§5, §5.4.3: обсяги карт — з card_mib(); по одному запису на різний обсяг."""
    env = make_models_env(card_sizes=[46068, 46068, 24576])
    got = per_card_keys(_info(env)["fit"])
    assert got == {46068, 24576}, f"info() per_card keys for cards [46068, 46068, 24576]: expected {{46068, 24576}}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.3")
@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_info_fit_weights_gib(models_env):
    """§5.4.3, §3: weights_gib — ваги *.safetensors (рівно 0.5 GiB)."""
    got = _info(models_env)["fit"].get("weights_gib")
    assert got == 0.5, f"info() fit weights_gib: expected 0.5, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.3")
@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_info_no_weights_repo_fits_no_card(make_models_env):
    """§5.4.3, §3: репозиторій лише з README і .gitattributes — у fit кожна карта fits false."""
    env = make_models_env(card_sizes=[46068, 81559])
    env.hub.add(NO_WEIGHTS_REPO, NO_WEIGHTS_FILES)
    fit = _info(env, NO_WEIGHTS_REPO)["fit"]
    got = {mib: card_view(fit, mib)["fits"] for mib in (46068, 81559)}
    assert got == {46068: False, 81559: False}, f"info({NO_WEIGHTS_REPO!r}) fits per card: expected all False, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.3")
@pytest.mark.parametrize("fraction", [0, -0.5, 1.5, 1.0001])
def test_info_bad_fraction_refused(models_env, fraction):
    """§5.4.3: fraction поза (0, 1] → bad_fraction."""
    expect_manager_error("bad_fraction", models_env.store.info, REPO, fraction=fraction)


@pytest.mark.req("SPEC-GPU-002 §5.4.3")
def test_info_fraction_one_accepted(models_env):
    """§5.4.3: права межа 1 включна."""
    got = _info(models_env, fraction=1.0).get("fraction")
    assert got == 1.0, f"info(fraction=1.0) fraction: expected 1.0, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.3")
@pytest.mark.req("SPEC-GPU-002 §4")
def test_info_unknown_repo_not_found(models_env):
    """§4, §5.4.3: моделі немає на HF → hf_not_found."""
    expect_manager_error("hf_not_found", models_env.store.info, MISSING_REPO)


# --- §5.4.4 search() -------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §5.4.4")
def test_search_default_limit_is_ten(models_env):
    """§5.4.4: search(query) без limit — 10 результатів."""
    got = models_env.store.search("llama")
    assert len(got) == 10, f"search() default limit: expected 10 results, got {len(got)}"


@pytest.mark.req("SPEC-GPU-002 §5.4.4")
@pytest.mark.parametrize(
    ("limit", "expected"),
    [(0, 1), (-3, 1), (1, 1), (25, 25), (50, 50), (51, 50), (1000, 50)],
)
def test_search_limit_clamped(models_env, limit, expected):
    """§5.4.4: limit обмежується до 1..50 (hub має 60 результатів)."""
    got = models_env.store.search("llama", limit=limit)
    assert len(got) == expected, f"search(limit={limit}): expected {expected} results, got {len(got)}"


@pytest.mark.req("SPEC-GPU-002 §5.4.4")
@pytest.mark.req("SPEC-GPU-002 §4")
def test_search_returns_hub_results_in_order(models_env):
    """§5.4.4, §4: результати — від hub, у його порядку (за популярністю)."""
    got = [item.get("repo") for item in models_env.store.search("llama", limit=5)]
    expected = [item["repo"] for item in models_env.hub.pool[:5]]
    assert got == expected, f"search(limit=5): expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §4")
@pytest.mark.req("SPEC-GPU-002 §8")
def test_search_hub_unavailable(models_env):
    """§4, §8: мережа чи інша помилка HF → hf_unavailable доходить до виклику."""
    models_env.hub.unavailable = True
    expect_manager_error("hf_unavailable", models_env.store.search, "llama")
