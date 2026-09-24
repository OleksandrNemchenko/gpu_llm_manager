"""§3 SPEC-GPU-002: чисті функції gpu_manager.hub — select_files, weight_bytes, kv_bytes_per_token, estimate_fit.

Функції чисті (§3), тому тести — рівня unit: без файлів, мережі й годинника. Очікувані значення оцінки
§3.4 — або підібрані «круглі» числа, пораховані вручну в коментарі, або оракул fit_card_oracle, що
записує формулу §3.4 дослівно.
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import has_cyrillic
from .model_fakes import GIB, NO_WEIGHTS_FILES, card_view, fit_card_oracle, fit_value, per_card_keys

pytestmark = pytest.mark.unit

EXCLUDED_EXTENSIONS = [".gguf", ".pth", ".pt", ".onnx", ".onnx_data", ".h5", ".msgpack", ".ot", ".tflite", ".mlmodel"]
EXCLUDED_DIRS = ["original", "onnx", "openvino", "coreml", "gguf"]
FP4_WARNING = "no hardware support below sm_100"  # текст попередження §3 для quant_method з fp4
FP8_WARNING = "weight-only W8A16"  # текст попередження §3 для quant_method == fp8
NO_WEIGHTS_WARNING = "no safetensors/bin weights"  # текст попередження §3 для weight_bytes = 0

# «Кругла» модель для ручного рахунку §3.4: ваги рівно 2 GiB, kv = 2·4·2·128·2 = 4096 байт = 4.0 KiB.
CLEAN_FILES = {"model.safetensors": 2 * GIB, "config.json": 512}
CLEAN_CONFIG: dict[str, Any] = {
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "hidden_size": 1024,
    "head_dim": 128,
}
CLEAN_KV = 4096


def _select(names: list[str]) -> Any:
    from gpu_manager.hub import select_files

    return select_files(list(names))


def _weights(files: dict[str, int]) -> Any:
    from gpu_manager.hub import weight_bytes

    return weight_bytes(dict(files))


def _kv(config: dict[str, Any]) -> Any:
    from gpu_manager.hub import kv_bytes_per_token

    return kv_bytes_per_token(config)


def _fit(
    files: dict[str, int] | None = None,
    config: dict[str, Any] | None = None,
    cards: tuple[int, ...] = (10240,),
    fraction: float = 0.5,
    overhead: float = 1.0,
) -> Any:
    """estimate_fit(files, config, card_mib, fraction, overhead_gib); типово — «кругла» модель на карті 10 GiB."""
    from gpu_manager.hub import estimate_fit

    return estimate_fit(
        dict(CLEAN_FILES if files is None else files),
        dict(CLEAN_CONFIG if config is None else config),
        list(cards),
        fraction,
        overhead,
    )


def _warnings(fit: Any) -> list[Any]:
    return list(fit_value(fit, "warnings"))


def _quantized(method: str) -> dict[str, Any]:
    config = dict(CLEAN_CONFIG)
    config["quantization_config"] = {"quant_method": method}
    return config


# --- §3.1 select_files ----------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §3.1")
@pytest.mark.parametrize("ext", EXCLUDED_EXTENSIONS)
def test_select_files_drops_excluded_extension(ext):
    """§3.1: файл з відкинутим розширенням не вибирається; решта лишається."""
    got = _select(["config.json", f"model{ext}"])
    assert got == ["config.json"], f"select_files(['config.json', 'model{ext}']): expected ['config.json'], got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.1")
@pytest.mark.parametrize("folder", EXCLUDED_DIRS)
def test_select_files_drops_excluded_directory(folder):
    """§3.1: файл у теці original/ onnx/ openvino/ coreml/ gguf/ не вибирається, хоч розширення й дозволене."""
    got = _select(["config.json", f"{folder}/params.json"])
    assert got == ["config.json"], f"select_files with {folder}/params.json: expected ['config.json'], got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.1")
@pytest.mark.parametrize("name", ["original_notes.md", "onnx_export.md", "gguf-readme.txt"])
def test_select_files_name_prefix_is_not_a_directory(name):
    """§3.1: відкидаються теки; файл верхнього рівня, чиє ім'я лише починається так само, лишається."""
    got = _select(["config.json", name])
    expected = sorted(["config.json", name])
    assert got == expected, f"select_files with top-level {name!r}: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.1")
def test_select_files_safetensors_present_drops_bin():
    """§3.1: є хоч один *.safetensors — *.bin відкидається."""
    got = _select(["pytorch_model.bin", "model.safetensors", "config.json"])
    assert got == ["config.json", "model.safetensors"], (
        f"select_files with safetensors and bin: expected ['config.json', 'model.safetensors'], got {got!r}"
    )


@pytest.mark.req("SPEC-GPU-002 §3.1")
def test_select_files_safetensors_present_drops_every_bin():
    """§3.1: з *.safetensors відкидається кожен *.bin, не лише ваги (training_args.bin теж)."""
    got = _select(["model.safetensors", "training_args.bin", "config.json"])
    assert got == ["config.json", "model.safetensors"], (
        f"select_files with safetensors and training_args.bin: expected ['config.json', 'model.safetensors'], got {got!r}"
    )


@pytest.mark.req("SPEC-GPU-002 §3.1")
def test_select_files_without_safetensors_keeps_bin():
    """§3.1: без жодного *.safetensors файли *.bin лишаються."""
    got = _select(["pytorch_model.bin", "config.json"])
    assert got == ["config.json", "pytorch_model.bin"], (
        f"select_files without safetensors: expected ['config.json', 'pytorch_model.bin'], got {got!r}"
    )


@pytest.mark.req("SPEC-GPU-002 §3.1")
def test_select_files_safetensors_index_alone_keeps_bin():
    """§3.1: model.safetensors.index.json — не *.safetensors; сам він не змушує відкинути *.bin."""
    names = ["pytorch_model.bin", "model.safetensors.index.json", "config.json"]
    got = _select(names)
    assert got == sorted(names), f"select_files with only a safetensors index: expected {sorted(names)!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.1")
def test_select_files_keeps_other_files():
    """§3.1: «решту лишає» — конфіги, токенізатор, README, .gitattributes."""
    names = [
        ".gitattributes",
        "README.md",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    ]
    got = _select(list(reversed(names)))
    assert got == sorted(names), f"select_files of plain files: expected {sorted(names)!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.1")
def test_select_files_result_is_sorted_list():
    """§3.1: результат — список, відсортований незалежно від порядку входу."""
    names = [
        "tokenizer.json",
        "model-00002-of-00002.safetensors",
        "config.json",
        "model-00001-of-00002.safetensors",
        ".gitattributes",
    ]
    got = _select(names)
    assert isinstance(got, list) and got == sorted(names), f"select_files: expected sorted list {sorted(names)!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.1")
def test_select_files_typical_repository():
    """§3.1: типовий репозиторій з вагами в кількох форматах — лишаються safetensors, конфіги й токенізатор."""
    names = [
        ".gitattributes",
        "README.md",
        "config.json",
        "generation_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model-00002-of-00002.bin",
        "pytorch_model.bin.index.json",
        "tf_model.h5",
        "flax_model.msgpack",
        "rust_model.ot",
        "model.onnx",
        "onnx/model.onnx_data",
        "onnx/config.json",
        "original/consolidated.00.pth",
        "original/params.json",
        "original/tokenizer.model",
        "openvino/openvino_model.xml",
        "coreml/model.mlmodel",
        "gguf/model-q4.gguf",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    expected = sorted(
        [
            ".gitattributes",
            "README.md",
            "config.json",
            "generation_config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
    )
    got = _select(names)
    assert got == expected, f"select_files of a typical repo: expected {expected!r}, got {got!r}"


# --- §3.2 weight_bytes -----------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §3.2")
def test_weight_bytes_sums_safetensors_and_bin():
    """§3.2: сума розмірів *.safetensors і *.bin; індекси й конфіги не рахуються."""
    files = {
        "a.safetensors": 100,
        "b.safetensors": 30,
        "c.bin": 50,
        "config.json": 7,
        "model.safetensors.index.json": 1_000,
        "tokenizer.json": 5_000,
    }
    got = _weights(files)
    assert got == 180, f"weight_bytes: expected 180 bytes (100 + 30 + 50), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.2")
@pytest.mark.parametrize("files", [{}, {"config.json": 7, "tokenizer.json": 5_000}], ids=["empty", "no-weights"])
def test_weight_bytes_zero_without_weight_files(files):
    """§3.2: без *.safetensors і *.bin — 0."""
    got = _weights(files)
    assert got == 0, f"weight_bytes({files!r}): expected 0 bytes, got {got!r}"


# --- §3.3 kv_bytes_per_token -------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §3.3")
def test_kv_bytes_all_fields_present():
    """§3.3: 2 × num_hidden_layers × num_key_value_heads × head_dim × 2."""
    config = {
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "head_dim": 128,
    }
    got = _kv(config)
    assert got == 131072, f"kv_bytes_per_token: expected 2*32*8*128*2 = 131072 bytes, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.3")
def test_kv_bytes_kv_heads_default_to_attention_heads():
    """§3.3: без num_key_value_heads береться num_attention_heads."""
    config = {"num_hidden_layers": 12, "num_attention_heads": 12, "hidden_size": 768}
    got = _kv(config)
    assert got == 36864, f"kv_bytes_per_token without kv heads: expected 2*12*12*64*2 = 36864 bytes, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.3")
def test_kv_bytes_head_dim_defaults_to_hidden_over_heads():
    """§3.3: без head_dim береться hidden_size // num_attention_heads."""
    config = {"num_hidden_layers": 40, "num_attention_heads": 40, "num_key_value_heads": 8, "hidden_size": 5120}
    got = _kv(config)
    assert got == 163840, f"kv_bytes_per_token without head_dim: expected 2*40*8*128*2 = 163840 bytes, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.3")
def test_kv_bytes_head_dim_default_is_floor_division():
    """§3.3: типовий head_dim — ціла частка (//): 1000 // 3 = 333."""
    config = {"num_hidden_layers": 2, "num_attention_heads": 3, "hidden_size": 1000}
    got = _kv(config)
    assert got == 7992, f"kv_bytes_per_token with hidden 1000 / 3 heads: expected 2*2*3*333*2 = 7992 bytes, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.3")
def test_kv_bytes_explicit_head_dim_wins():
    """§3.3: заданий head_dim береться як є, навіть якщо ≠ hidden_size // num_attention_heads."""
    config = {
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "hidden_size": 3072,
        "head_dim": 256,
    }
    got = _kv(config)
    assert got == 458752, f"kv_bytes_per_token with head_dim 256: expected 2*28*16*256*2 = 458752 bytes, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.3")
def test_kv_bytes_from_text_config():
    """§3.3: є text_config — рахується з нього (мультимодальна модель без полів на верхньому рівні)."""
    config = {
        "model_type": "llava",
        "text_config": {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096},
        "vision_config": {"num_hidden_layers": 24, "num_attention_heads": 16, "hidden_size": 1024},
    }
    got = _kv(config)
    assert got == 131072, f"kv_bytes_per_token from text_config: expected 2*32*8*128*2 = 131072 bytes, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.3")
def test_kv_bytes_text_config_wins_over_top_level():
    """§3.3: є text_config — поля верхнього рівня не використовуються."""
    config = {
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "hidden_size": 64,
        "text_config": {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096},
    }
    got = _kv(config)
    assert got == 131072, f"kv_bytes_per_token with both levels: expected 131072 bytes from text_config, got {got!r}"


MISSING_KV_FIELDS = {
    "empty": {},
    "no-layers": {"num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096},
    "no-heads-at-all": {"num_hidden_layers": 32, "hidden_size": 4096},
    "no-hidden-no-head-dim": {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8},
}


@pytest.mark.req("SPEC-GPU-002 §3.3")
@pytest.mark.parametrize("case", sorted(MISSING_KV_FIELDS))
def test_kv_bytes_missing_fields_is_none(case):
    """§3.3: бракує полів для формули → None."""
    got = _kv(MISSING_KV_FIELDS[case])
    assert got is None, f"kv_bytes_per_token({MISSING_KV_FIELDS[case]!r}): expected None, got {got!r}"


# --- §3.4 estimate_fit: арифметика -------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_hand_computed_card():
    """§3.4: 10240 MiB·0.5 = 5 GiB − 2 GiB ваг − 1 GiB запасу = 2 GiB; 2 GiB // 4096 = 524288 токенів."""
    got = card_view(_fit(), 10240)
    expected = {"usable_gib": 2.0, "fits": True, "max_context_tokens": 524288}
    assert got == expected, f"per_card[10240]: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_a40_matches_formula():
    """§3.4: модель ≈ 8B (safetensors ≈ 15 GiB) на A40 46068 MiB, fraction 0.9, запас 2 GiB — за формулою."""
    files = {
        "model-00001-of-00004.safetensors": 4_976_698_672,
        "model-00002-of-00004.safetensors": 4_999_802_720,
        "model-00003-of-00004.safetensors": 4_915_916_176,
        "model-00004-of-00004.safetensors": 1_168_138_808,
        "config.json": 654,
        "tokenizer.json": 9_085_657,
    }
    config = {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096}
    weights = 4_976_698_672 + 4_999_802_720 + 4_915_916_176 + 1_168_138_808
    got = card_view(_fit(files, config, (46068,), 0.9, 2.0), 46068)
    expected = fit_card_oracle(46068, 0.9, weights, 2.0, 131072, None)
    assert got == expected, f"per_card[46068] for an 8B model: expected {expected!r} (formula §3.4), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_usable_gib_one_decimal():
    """§3: usable_gib — 1 знак: 5 − 0.375 − 2.28125 = 2.34375 GiB → 2.3; 2516582400 // 6000 = 419430 токенів."""
    files = {"model.safetensors": 402_653_184}  # 0.375 GiB
    config = {"num_hidden_layers": 3, "num_attention_heads": 5, "num_key_value_heads": 5, "hidden_size": 500, "head_dim": 100}
    got = card_view(_fit(files, config, (10240,), 0.5, 2.28125), 10240)
    expected = {"usable_gib": 2.3, "fits": True, "max_context_tokens": 419430}
    assert got == expected, f"per_card[10240]: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_weights_gib_two_decimals():
    """§3: weights_gib — 2 знаки: 0.375 GiB → 0.38 (не 0.4 і не 0.375)."""
    got = fit_value(_fit({"model.safetensors": 402_653_184}), "weights_gib")
    assert got == 0.38, f"weights_gib for 0.375 GiB of weights: expected 0.38, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_kv_kib_one_decimal():
    """§3: kv_kib_per_token — 1 знак: 2·3·5·100·2 = 6000 байт = 5.859375 KiB → 5.9."""
    config = {"num_hidden_layers": 3, "num_attention_heads": 5, "num_key_value_heads": 5, "hidden_size": 500, "head_dim": 100}
    got = fit_value(_fit(config=config), "kv_kib_per_token")
    assert got == 5.9, f"kv_kib_per_token for 6000 bytes: expected 5.9, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_only_weight_files_reduce_usable():
    """§3.4: з usable віднімається weight_bytes — великий tokenizer.json на оцінку не впливає."""
    files = dict(CLEAN_FILES)
    files["tokenizer.json"] = GIB
    got = card_view(_fit(files), 10240)
    expected = {"usable_gib": 2.0, "fits": True, "max_context_tokens": 524288}
    assert got == expected, f"per_card[10240] with a 1 GiB tokenizer.json: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_context_capped_by_max_position():
    """§3.4: max_context_tokens обмежено max_position_embeddings (524288 → 8192)."""
    config = dict(CLEAN_CONFIG, max_position_embeddings=8192)
    got = card_view(_fit(config=config), 10240)["max_context_tokens"]
    assert got == 8192, f"max_context_tokens with max_position_embeddings 8192: expected 8192, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_cap_above_computed_changes_nothing():
    """§3.4: межа max_position_embeddings лише обмежує — більша за пораховане не підвищує."""
    config = dict(CLEAN_CONFIG, max_position_embeddings=10_000_000)
    got = card_view(_fit(config=config), 10240)["max_context_tokens"]
    assert got == 524288, f"max_context_tokens with max_position_embeddings 10^7: expected 524288, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_one_entry_per_distinct_card_size():
    """§3.4: запис per_card — для кожного різного обсягу карти, без повторів."""
    got = per_card_keys(_fit(cards=(46068, 46068, 24576, 46068)))
    assert got == {46068, 24576}, f"per_card keys for cards [46068, 46068, 24576, 46068]: expected {{46068, 24576}}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_each_card_size_computed_separately():
    """§3.4: кожен обсяг рахується окремо: 8192 MiB·0.5 = 4 GiB − 3 GiB = 1 GiB → 262144 токенів."""
    fit = _fit(cards=(10240, 8192))
    got = {mib: card_view(fit, mib) for mib in (10240, 8192)}
    expected = {
        10240: {"usable_gib": 2.0, "fits": True, "max_context_tokens": 524288},
        8192: {"usable_gib": 1.0, "fits": True, "max_context_tokens": 262144},
    }
    assert got == expected, f"per_card for 10240 and 8192 MiB: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_negative_usable_does_not_fit():
    """§3.4: 4096 MiB·0.5 = 2 GiB − 3 GiB = −1 GiB → fits false, 0 токенів."""
    got = card_view(_fit(cards=(4096,)), 4096)
    expected = {"usable_gib": -1.0, "fits": False, "max_context_tokens": 0}
    assert got == expected, f"per_card[4096]: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_zero_usable_does_not_fit():
    """§3.4: usable рівно 0 (6144 MiB·0.5 = 3 GiB − 3 GiB) → fits false (usable > 0 строго), 0 токенів."""
    got = card_view(_fit(cards=(6144,)), 6144)
    expected = {"usable_gib": 0.0, "fits": False, "max_context_tokens": 0}
    assert got == expected, f"per_card[6144] with usable exactly 0 bytes: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_one_byte_usable_fits():
    """§3.4: usable = 1 байт → fits true, max_context_tokens = 1 // 4096 = 0."""
    got = card_view(_fit({"model.safetensors": 2 * GIB - 1}, cards=(6144,)), 6144)
    expected = {"usable_gib": 0.0, "fits": True, "max_context_tokens": 0}
    assert got == expected, f"per_card[6144] with usable exactly 1 byte: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_unknown_kv_context_none():
    """§3.4: kv невідомий (config порожній) → max_context_tokens None, kv_kib_per_token None."""
    fit = _fit(config={})
    got = (card_view(fit, 10240)["max_context_tokens"], fit_value(fit, "kv_kib_per_token"))
    assert got == (None, None), f"unknown kv: expected (max_context_tokens None, kv_kib_per_token None), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_max_context_is_max_position_embeddings():
    """§3: max_context — max_position_embeddings конфігу."""
    got = fit_value(_fit(config=dict(CLEAN_CONFIG, max_position_embeddings=32768)), "max_context")
    assert got == 32768, f"max_context: expected 32768, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_max_context_none_without_max_position():
    """§3: без max_position_embeddings — max_context None."""
    got = fit_value(_fit(), "max_context")
    assert got is None, f"max_context without max_position_embeddings: expected None, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_kv_kib_whole_value():
    """§3: kv_kib_per_token для 4096 байт — 4.0."""
    got = fit_value(_fit(), "kv_kib_per_token")
    assert got == 4.0, f"kv_kib_per_token for {CLEAN_KV} bytes: expected 4.0, got {got!r}"


# --- §3.4 estimate_fit: попередження -----------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_plain_model_has_no_warnings():
    """§3: без квантування й з відомим kv — попереджень немає."""
    got = _warnings(_fit())
    assert got == [], f"warnings for a plain model: expected [], got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
@pytest.mark.parametrize("method", ["fp4", "nvfp4", "mxfp4", "modelopt_fp4"])
def test_estimate_fit_fp4_warns_no_hardware_support(method):
    """§3: quant_method, що містить fp4 → попередження «no hardware support below sm_100»."""
    got = _warnings(_fit(config=_quantized(method)))
    assert any(FP4_WARNING in str(w) for w in got), f"quant_method {method!r}: expected a warning with {FP4_WARNING!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_fp8_warns_weight_only():
    """§3: quant_method рівно fp8 → попередження «weight-only W8A16»."""
    got = _warnings(_fit(config=_quantized("fp8")))
    assert any(FP8_WARNING in str(w) for w in got), f"quant_method 'fp8': expected a warning with {FP8_WARNING!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_fp8_not_warned_as_fp4():
    """§3: fp8 не містить fp4 — попередження про fp4 немає."""
    got = _warnings(_fit(config=_quantized("fp8")))
    assert not any(FP4_WARNING in str(w) for w in got), f"quant_method 'fp8': expected no {FP4_WARNING!r} warning, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_fp8_variant_is_not_exact_fp8():
    """§3: попередження W8A16 — лише для «рівно fp8»; fbgemm_fp8 його не дає."""
    got = _warnings(_fit(config=_quantized("fbgemm_fp8")))
    assert not any("W8A16" in str(w) for w in got), f"quant_method 'fbgemm_fp8': expected no W8A16 warning, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
@pytest.mark.parametrize("method", ["awq", "gptq", "bitsandbytes", "compressed-tensors"])
def test_estimate_fit_other_quantization_no_warnings(method):
    """§3: інші методи квантування попереджень §3 не дають."""
    got = _warnings(_fit(config=_quantized(method)))
    assert got == [], f"quant_method {method!r}: expected no warnings, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_unknown_kv_warns_about_context():
    """§3: kv невідомий → є попередження (про контекст)."""
    got = _warnings(_fit(config={}))
    assert len(got) >= 1, f"unknown kv: expected a context warning, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_warnings_are_english_strings():
    """§3: попередження — рядки англійською (без кирилиці); тут одразу fp4 і невідомий kv."""
    got = _warnings(_fit(config={"quantization_config": {"quant_method": "nvfp4"}}))
    bad = [w for w in got if not isinstance(w, str) or has_cyrillic(w)]
    assert len(got) >= 2 and not bad, f"warnings: expected >= 2 English strings (fp4 + context), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_fp4_warning_keeps_usable_arithmetic():
    """§3.4: попередження не змінюють арифметики — fp4 на тій самій моделі дає той самий per_card."""
    got = card_view(_fit(config=_quantized("nvfp4")), 10240)
    expected = {"usable_gib": 2.0, "fits": True, "max_context_tokens": 524288}
    assert got == expected, f"per_card[10240] with nvfp4: expected {expected!r}, got {got!r}"


# --- §3 estimate_fit: немає ваг ---------------------------------------------------------------------------------------------------

# Карти, на яких за формулою §3.4 без ваг usable > 0: 10240·0.5 − 1 = 4 GiB і 81920·0.5 − 1 = 39 GiB.
ROOMY_CARDS = (10240, 81920)


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_no_weights_fits_no_card():
    """§3: weight_bytes = 0 (лише README і .gitattributes, як у репозиторію з GGUF) → fits false для кожної карти."""
    fit = _fit(NO_WEIGHTS_FILES, cards=ROOMY_CARDS)
    got = {mib: card_view(fit, mib)["fits"] for mib in ROOMY_CARDS}
    expected = {mib: False for mib in ROOMY_CARDS}
    assert got == expected, f"fits without weight files on cards {ROOMY_CARDS} MiB: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_no_weights_warns():
    """§3: weight_bytes = 0 → попередження «no safetensors/bin weights»."""
    got = _warnings(_fit(NO_WEIGHTS_FILES, cards=ROOMY_CARDS))
    assert any(NO_WEIGHTS_WARNING in str(w) for w in got), f"no weight files: expected a warning with {NO_WEIGHTS_WARNING!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §3.4")
def test_estimate_fit_bin_weights_not_reported_missing():
    """§3.2: *.bin — теж ваги; модель лише з pytorch_model.bin не дає попередження про відсутні ваги."""
    got = _warnings(_fit({"pytorch_model.bin": 2 * GIB, "config.json": 512}))
    assert not any(NO_WEIGHTS_WARNING in str(w) for w in got), f"bin-only weights: expected no {NO_WEIGHTS_WARNING!r} warning, got {got!r}"
