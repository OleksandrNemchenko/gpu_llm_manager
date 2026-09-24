"""§1 і §2.1–2.6 SPEC-GPU-003: конфіг vllm і запуск моделі на vLLM.

Відмови start(), вибір порту й назви, параметри (явні → профіль → типові), межа пам'яті карти, argv і env
процесу vLLM, лог моделі. ModelRunner збирається з підробками швів §2 (tests/runner_fakes.py): юніт
«запускає» FakeLauncher, argv і env перевіряються в записаному виклику launcher.start. Карти — фальшивий
бекенд фази 1 (46068 MiB кожна); зайнята пам'ять карти задається явно (RunnerEnv.set_used).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from .conftest import N_GPUS, STRANGER, T0
from .model_fakes import REPO, expect_manager_error
from .runner_fakes import (
    DEFAULT_NAME,
    FRACTION_EMPTY,
    FRACTION_OVER_MID,
    FRACTION_USED_HIGH,
    FRACTION_USED_MID,
    GPU,
    GPU_B,
    MARKER,
    MIXED_NAME,
    MIXED_REPO,
    NO_DEFAULT_CAP,
    NOT_LOCAL_REPO,
    ODD_NAME,
    ODD_REPO,
    PORT_FIRST,
    USED_HIGH_MIB,
    USED_MID_MIB,
    flag_value,
    max_num_seqs_values,
    normalize_argv,
)

MAX_NUM_SEQS_DEFAULT = 32  # типовий vllm.max_num_seqs (§1)
MEDIA_FLAG = "--allowed-local-media-path"  # §2.6: прапорець теки files.inbox_dir, лише коли тека існує

pytestmark = [pytest.mark.component, pytest.mark.usefixtures("isolated_home")]


def _profile(**over: Any) -> dict[str, Any]:
    """Повний запис профілю §2.7 для файла data/model_profiles.json; over — змінені поля."""
    profile: dict[str, Any] = {
        "fraction": 0.6,
        "max_model_len": 4096,
        "extra_args": [],
        "gpu": GPU,
        "updated": T0,
        "attempts": 1,
    }
    profile.update(over)
    return profile


# --- §1 конфіг vllm ------------------------------------------------------------------------------------------------


def _load(write_config: Any, config_dir: Path, overrides: dict[str, Any] | None = None) -> Any:
    from gpu_manager.config import load_config

    settings: dict[str, Any] = {"paths.data_dir": str(config_dir / "data")}
    settings.update(overrides or {})
    return load_config(write_config(settings))


@pytest.mark.req("SPEC-GPU-003 §1")
def test_vllm_bin_default_next_to_config(write_config, config_dir):
    """§1: типовий vllm.bin — .venv-vllm/bin/vllm відносно теки конфігу; поле vllm_bin — абсолютне."""
    cfg = _load(write_config, config_dir)
    expected = (config_dir / ".venv-vllm" / "bin" / "vllm").resolve()
    got = Path(cfg.vllm_bin)
    assert got.is_absolute() and got.resolve() == expected, (
        f"Config.vllm_bin without vllm.bin: expected absolute {expected}, got {cfg.vllm_bin!r}"
    )


@pytest.mark.req("SPEC-GPU-003 §1")
def test_vllm_bin_relative_resolved_from_config_dir(write_config, config_dir):
    """§1: відносний vllm.bin рахується від теки конфігу, а не від поточної теки."""
    cfg = _load(write_config, config_dir, {"vllm.bin": "tools/vllm"})
    expected = (config_dir / "tools" / "vllm").resolve()
    got = Path(cfg.vllm_bin)
    assert got.is_absolute() and got.resolve() == expected, (
        f"Config.vllm_bin for vllm.bin='tools/vllm': expected absolute {expected}, got {cfg.vllm_bin!r}"
    )


@pytest.mark.req("SPEC-GPU-003 §1")
@pytest.mark.parametrize(("field", "expected"), [("vllm_default_fraction", 0.95), ("vllm_start_timeout_s", 900)])
def test_vllm_numeric_defaults(write_config, config_dir, field, expected):
    """§1: типові default_fraction 0.95 і start_timeout_s 900."""
    cfg = _load(write_config, config_dir)
    got = getattr(cfg, field)
    assert got == expected, f"Config.{field} without the setting: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §1")
@pytest.mark.parametrize(
    ("field", "dotted", "value"),
    [
        ("vllm_default_fraction", "vllm.default_fraction", 0.8),
        ("vllm_default_fraction", "vllm.default_fraction", 0.05),  # нижня межа [0.05, 1] — ще дозволена
        ("vllm_default_fraction", "vllm.default_fraction", 1.0),  # верхня межа [0.05, 1] — ще дозволена
        ("vllm_start_timeout_s", "vllm.start_timeout_s", 300),
    ],
)
def test_vllm_numeric_setting_maps_to_field(write_config, config_dir, field, dotted, value):
    """§1: значення з конфігу потрапляє у відповідне поле Config."""
    cfg = _load(write_config, config_dir, {dotted: value})
    got = getattr(cfg, field)
    assert got == value, f"Config.{field} from {dotted}={value!r}: expected {value!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §1")
@pytest.mark.parametrize("value", [0.04, 1.01])
def test_vllm_default_fraction_outside_range_refused(write_config, config_dir, value):
    """§1: vllm.default_fraction поза [0.05, 1] → ConfigError."""
    from gpu_manager.config import ConfigError

    try:
        cfg = _load(write_config, config_dir, {"vllm.default_fraction": value})
    except ConfigError:
        return
    pytest.fail(f"vllm.default_fraction={value}: expected ConfigError (allowed [0.05, 1]), loaded {cfg.vllm_default_fraction!r}")


@pytest.mark.req("SPEC-GPU-003 §1")
def test_vllm_max_num_seqs_below_one_refused(write_config, config_dir):
    """§1: vllm.max_num_seqs < 1 → ConfigError (0 — перше недопустиме)."""
    from gpu_manager.config import ConfigError

    try:
        _load(write_config, config_dir, {"vllm.max_num_seqs": 0})
    except ConfigError:
        return
    pytest.fail("vllm.max_num_seqs=0: expected ConfigError (must be >= 1), the config loaded")


@pytest.mark.req("SPEC-GPU-003 §1")
@pytest.mark.parametrize("value", [0, -1], ids=["zero", "negative"])
def test_vllm_start_timeout_not_positive_refused(write_config, config_dir, value):
    """§1: vllm.start_timeout_s ≤ 0 → ConfigError (0 — межа, що вже недопустима)."""
    from gpu_manager.config import ConfigError

    try:
        cfg = _load(write_config, config_dir, {"vllm.start_timeout_s": value})
    except ConfigError:
        return
    pytest.fail(f"vllm.start_timeout_s={value}: expected ConfigError (must be > 0 s), loaded {cfg.vllm_start_timeout_s!r}")


@pytest.mark.req("SPEC-GPU-003 §1")
def test_vllm_cuda_home_maps_to_field(write_config, config_dir, tmp_path):
    """§1: vllm.cuda_home → поле vllm_cuda_home."""
    target = tmp_path / "cuda-12.8"
    cfg = _load(write_config, config_dir, {"vllm.cuda_home": str(target)})
    got = cfg.vllm_cuda_home
    assert got is not None and Path(got).resolve() == target.resolve(), f"Config.vllm_cuda_home: expected {target}, got {got!r}"


# --- §2.1 відмови start() ---------------------------------------------------------------------------------------------


def _reserve_by_bob(env: Any) -> None:
    env.reserve(GPU, "bob")


def _take_name_coder(env: Any) -> None:
    env.start(name="coder")


def _only_port_not_free(env: Any) -> None:
    env.port_free.busy.add(PORT_FIRST)


def _card_used_mid(env: Any) -> None:
    env.set_used(GPU, USED_MID_MIB)


# code → (перевизначення конфігу, підготовка стану, аргументи start() поверх типових REPO / GPU / alice).
START_REFUSALS: dict[str, tuple[dict[str, Any], Any, dict[str, Any]]] = {
    "unknown_user": ({}, None, {"user": STRANGER}),
    "unknown_gpu": ({}, None, {"gpu": N_GPUS}),
    "reserved_by_other": ({}, _reserve_by_bob, {}),
    "model_not_local": ({}, None, {"repo": NOT_LOCAL_REPO}),
    "bad_name": ({}, None, {"name": "../evil"}),
    "name_taken": ({}, _take_name_coder, {"name": "coder", "gpu": GPU_B}),
    # Діапазон з одного порту, і той зайнятий чужим процесом.
    "no_free_port": ({"vllm.port_range": [PORT_FIRST, PORT_FIRST]}, _only_port_not_free, {}),
    # §2.5: карта зайнята на 10000 MiB, явна частка 0.78 > допустимих 0.7774.
    "gpu_memory_low": ({}, _card_used_mid, {"fraction": FRACTION_OVER_MID}),
}
REFUSAL_PARAMS = [
    pytest.param(code, marks=pytest.mark.req("SPEC-GPU-003 §2.5")) if code == "gpu_memory_low" else code
    for code in sorted(START_REFUSALS)
]


@pytest.mark.req("SPEC-GPU-003 §2.1")
@pytest.mark.parametrize("code", REFUSAL_PARAMS)
def test_start_refusal(make_runner_env, code):
    """§2.1, §2.5: кожна відмова start() — ManagerError з кодом таблиці; юніт не запускається."""
    overrides, setup, kwargs = START_REFUSALS[code]
    env = make_runner_env(overrides)
    if setup is not None:
        setup(env)
    before = len(env.launcher.calls)
    expect_manager_error(code, env.start, **kwargs)
    new_calls = len(env.launcher.calls) - before
    assert new_calls == 0, f"{code}: expected no launcher.start for a refused start, got {new_calls} new call(s)"


@pytest.mark.req("SPEC-GPU-003 §2.1")
@pytest.mark.parametrize("fraction", [0, 1.5])
def test_start_bad_fraction(runner_env, fraction):
    """§2.1: частка поза (0, 1] → bad_fraction (0 — межа, що не входить)."""
    expect_manager_error("bad_fraction", runner_env.start, fraction=fraction)


# §2.1: повний список прапорців, якими керує менеджер (у extra_args — bad_extra_args).
FORBIDDEN_FLAGS = (
    "--host", "--port", "--uds", "--served-model-name", "--gpu-memory-utilization", "--kv-cache-memory-bytes",
    "--num-gpu-blocks-override", "--config", "--allowed-local-media-path", "--allowed-media-domains", "--api-key",
    "--root-path", "--ssl-keyfile", "--ssl-certfile", "--ssl-ca-certs", "--middleware",
)
# §2.1 «також скорочений»: префікс, що серед прапорців vLLM веде лише до цього прапорця. Для --config скорочення
# немає: у vLLM є --config-format, тож кожен коротший префікс --config неоднозначний і vLLM його сам відкине.
# Найменшої довжини скорочення специфікація не задає (прогалина) — тут скорочення довгі й однозначні.
FORBIDDEN_ABBREVIATIONS = {
    "--host": "--hos",
    "--port": "--por",
    "--uds": "--ud",
    "--served-model-name": "--served-model",
    "--gpu-memory-utilization": "--gpu-memory-util",
    "--kv-cache-memory-bytes": "--kv-cache-mem",
    "--num-gpu-blocks-override": "--num-gpu-blocks",
    "--allowed-local-media-path": "--allowed-local-media",
    "--allowed-media-domains": "--allowed-media-dom",
    "--api-key": "--api-ke",
    "--root-path": "--root-pa",
    "--ssl-keyfile": "--ssl-key",
    "--ssl-certfile": "--ssl-certf",
    "--ssl-ca-certs": "--ssl-ca-cert",
    "--middleware": "--middlew",
}
FLAG_VALUE = "1"  # значення прапорця: відмова — за назвою прапорця, значення неважливе


def _forbidden_flag_cases() -> dict[str, list[str]]:
    """Кожен заборонений прапорець у кожній формі §2.1: повний, «=значення», «--прапорець.ключ», з «_» замість «-»
    (якщо в назві є «-»), скорочений; плюс одна змішана форма (скорочення з «_» і «=значення»)."""
    cases: dict[str, list[str]] = {}
    for flag in FORBIDDEN_FLAGS:
        stem = flag[2:]
        cases[stem] = [flag, FLAG_VALUE]
        cases[f"{stem}=value"] = [f"{flag}={FLAG_VALUE}"]
        cases[f"{stem}.key"] = [f"{flag}.k", FLAG_VALUE]
        if "-" in stem:
            cases[f"{stem}-underscore"] = ["--" + stem.replace("-", "_"), FLAG_VALUE]
        if flag in FORBIDDEN_ABBREVIATIONS:
            cases[f"{stem}-abbrev"] = [FORBIDDEN_ABBREVIATIONS[flag], FLAG_VALUE]
    cases["kv-cache-mem-abbrev-underscore=value"] = [f"--kv_cache_mem={FLAG_VALUE}"]
    return cases


BAD_EXTRA_ARGS: dict[str, Any] = {
    "not-strings": [42],
    "string-not-list": "--enforce-eager",  # рядок — ітерується посимвольно, але це не список рядків
    # Перед забороненим — дозволений прапорець: перевіряється весь список, а не лише перший елемент.
    **{f"flag:{case}": ["--enforce-eager", *args] for case, args in _forbidden_flag_cases().items()},
}


@pytest.mark.req("SPEC-GPU-003 §2.1")
@pytest.mark.parametrize("case", sorted(BAD_EXTRA_ARGS))
def test_start_bad_extra_args(runner_env, case):
    """§2.1: extra_args — не список рядків (зокрема рядок замість списку) або з прапорцем, яким керує менеджер,
    у будь-якій формі (скорочений, з «_», «=значення», «--прапорець.ключ») → bad_extra_args."""
    expect_manager_error("bad_extra_args", runner_env.start, extra_args=BAD_EXTRA_ARGS[case])


@pytest.mark.req("SPEC-GPU-003 §2.1")
@pytest.mark.parametrize(
    "extra", [["--enforce-eager"], ["--max-num-seqs", "8"]], ids=["enforce-eager", "max-num-seqs-8"]
)
def test_start_harmless_extra_args_accepted(runner_env, extra):
    """§2.1: прапорець не зі списку заборонених приймається й доходить до argv кінцем extra_args (§2.6)."""
    env = runner_env
    env.start(extra_args=list(extra))
    argv = env.argv(DEFAULT_NAME)
    got = argv[-len(extra):]
    assert got == extra, f"extra_args {extra!r}: expected a start with them at the end of argv, got argv {argv!r}"


# --- §2.2 порт --------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-003 §2.2")
def test_port_lowest_in_range(runner_env):
    """§2.2: перша модель — найменший порт vllm.port_range."""
    env = runner_env
    env.start()
    got = env.port(DEFAULT_NAME)
    assert got == PORT_FIRST, f"port of the first model: expected {PORT_FIRST}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.2")
def test_port_skips_port_not_free(runner_env):
    """§2.2: порт, для якого port_free(port) — False, пропускається."""
    env = runner_env
    env.port_free.busy.add(PORT_FIRST)
    env.start()
    got = env.port(DEFAULT_NAME)
    assert got == PORT_FIRST + 1, f"port with {PORT_FIRST} taken by another process: expected {PORT_FIRST + 1}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.2")
def test_port_skips_port_of_active_model(runner_env):
    """§2.2: порт активної моделі зайнятий, навіть коли port_free(port) його не бачить."""
    env = runner_env
    env.start(name="first")
    env.start(name="second", gpu=GPU_B)
    got = env.port("second")
    assert got == PORT_FIRST + 1, f"port of the second model: expected {PORT_FIRST + 1}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.1")
@pytest.mark.req("SPEC-GPU-003 §2.2")
def test_explicit_port_used(runner_env):
    """§2.1–2.2: явний вільний порт з діапазону — модель стартує саме на ньому, а не на найменшому."""
    env = runner_env
    env.start(port=PORT_FIRST + 42)
    got = env.port(DEFAULT_NAME)
    assert got == PORT_FIRST + 42, f"start(port={PORT_FIRST + 42}): expected --port {PORT_FIRST + 42}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.2")
def test_explicit_port_outside_range_refused(runner_env):
    """§2.2: явний порт поза vllm.port_range (8000..8099) → bad_port."""
    expect_manager_error("bad_port", runner_env.start, port=9000)


def _port_used_by_active_model(env: Any) -> None:
    env.start(name="first", gpu=GPU_B)  # займає 8000


def _port_not_free(env: Any) -> None:
    env.port_free.busy.add(PORT_FIRST)


PORT_TAKEN_CASES = {"active-model": _port_used_by_active_model, "port-free-false": _port_not_free}


@pytest.mark.req("SPEC-GPU-003 §2.2")
@pytest.mark.parametrize("case", sorted(PORT_TAKEN_CASES))
def test_explicit_port_taken_refused(runner_env, case):
    """§2.2: явний порт, зайнятий активною моделлю або з port_free = False, → port_taken."""
    PORT_TAKEN_CASES[case](runner_env)
    expect_manager_error("port_taken", runner_env.start, port=PORT_FIRST)


@pytest.mark.req("SPEC-GPU-003 §2.2")
def test_free_ports(make_runner_env):
    """§2.2: free_ports() — усі порти діапазону без активної моделі й з port_free = True."""
    env = make_runner_env({"vllm.port_range": [PORT_FIRST, PORT_FIRST + 3]})
    env.start()  # 8000
    env.port_free.busy.add(PORT_FIRST + 2)
    got = sorted(env.runner.free_ports())
    expected = [PORT_FIRST + 1, PORT_FIRST + 3]
    assert got == expected, f"free_ports() with 8000 used by a model and 8002 not free: expected {expected!r}, got {got!r}"


# --- §2.3 назва -------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-003 §2.3")
def test_default_name_is_lowercased_repo_tail(runner_env):
    """§2.3: типова назва — остання частина repo малими літерами."""
    env = runner_env
    env.start(repo=MIXED_REPO)
    got = env.names()
    assert got == [MIXED_NAME], f"default name for {MIXED_REPO!r}: expected [{MIXED_NAME!r}], got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.3")
def test_default_name_taken_gets_suffix(runner_env):
    """§2.3: типова назва зайнята → суфікс -2."""
    env = runner_env
    env.start()
    env.start(gpu=GPU_B)
    got = sorted(env.names())
    expected = [DEFAULT_NAME, f"{DEFAULT_NAME}-2"]
    assert got == expected, f"two starts of {REPO!r} without a name: expected names {expected!r}, got {got!r}"


# repo → очікувана типова назва (§2.3): малими літерами, недозволені символи → «-», не довше 40, без «-», «.», «_»
# на краях, порожня → model.
DEFAULT_NAME_CASES = {
    "edges-stripped": (ODD_REPO, ODD_NAME),
    "disallowed-to-dash": ("acme/Tiny+LLM@v2", "tiny-llm-v2"),
    "at-most-40": ("acme/" + "b" * 45, "b" * 40),
    # 50 символів, «-» на 40-й позиції: обрізана до 40 назва закінчилася б «-», який теж знімається.
    "cut-then-edge-stripped": ("acme/" + "a" * 39 + "-" + "b" * 10, "a" * 39),
    "empty-is-model": ("acme/_._", "model"),
}


@pytest.mark.req("SPEC-GPU-003 §2.3")
@pytest.mark.parametrize("case", sorted(DEFAULT_NAME_CASES))
def test_default_name_rule(runner_env, case):
    """§2.3: типова назва з останньої частини repo за правилом §2.3."""
    repo, expected = DEFAULT_NAME_CASES[case]
    env = runner_env
    env.put_ready(repo)
    env.start(repo=repo)
    got = env.names()
    assert got == [expected], f"default name for {repo!r}: expected [{expected!r}], got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.3")
@pytest.mark.parametrize("name", ["x", "a" * 48, "v1.2_b-3"], ids=["one-char", "48-chars", "dot-underscore-dash"])
def test_explicit_name_within_rule_accepted(runner_env, name):
    """§2.3: назва за правилом [a-z0-9][a-z0-9._-]{0,47} приймається (48 символів — найдовша)."""
    env = runner_env
    env.start(name=name)
    got = env.names()
    assert got == [name], f"start(name={name!r}): expected servers() names [{name!r}], got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.3")
@pytest.mark.parametrize(
    "name",
    ["a" * 49, "-x", ".x", "_x", "Coder", "a@b", "a/b"],
    ids=["49-chars", "lead-dash", "lead-dot", "lead-underscore", "uppercase", "at-sign", "slash"],
)
def test_explicit_name_outside_rule_refused(runner_env, name):
    """§2.3: назва поза правилом [a-z0-9][a-z0-9._-]{0,47} → bad_name."""
    expect_manager_error("bad_name", runner_env.start, name=name)


# --- §2.4 параметри: явні → профіль → типові ------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-003 §2.4")
def test_default_fraction_is_free_fraction_rounded_down(runner_env):
    """§2.4: типова частка = (total − used − 256 MiB)/total донизу до 0.01: used 10000 → 0.77 (не 0.78)."""
    env = runner_env
    env.set_used(GPU, USED_MID_MIB)
    env.start()
    got = env.fractions(DEFAULT_NAME)
    assert got == [FRACTION_USED_MID], f"default fraction with {USED_MID_MIB} MiB used: expected [{FRACTION_USED_MID}], got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.4")
def test_default_fraction_capped_by_default_fraction(make_runner_env):
    """§2.4: типова частка не більша за vllm.default_fraction (порожня карта: 0.99 → 0.5)."""
    env = make_runner_env({"vllm.default_fraction": 0.5})
    env.start()
    got = env.fractions(DEFAULT_NAME)
    assert got == [0.5], f"default fraction with vllm.default_fraction=0.5 on an empty card: expected [0.5], got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §1")
@pytest.mark.req("SPEC-GPU-003 §2.4")
def test_default_fraction_095_caps_empty_card(runner_env):
    """§1, §2.4: без налаштування vllm.default_fraction частка порожньої карти — типові 0.95, а не 0.99 з пам'яті."""
    env = runner_env
    env.start()
    got = env.fractions(DEFAULT_NAME)
    assert got == [FRACTION_EMPTY], f"default fraction on an empty card without vllm.default_fraction: expected [{FRACTION_EMPTY}], got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.4")
def test_default_max_model_len_not_set(runner_env):
    """§2.4: типовий max_model_len не задається — у argv немає --max-model-len."""
    env = runner_env
    env.start()
    argv = env.argv(DEFAULT_NAME)
    assert "--max-model-len" not in argv, f"argv without max_model_len: expected no --max-model-len, got {argv!r}"


@pytest.mark.req("SPEC-GPU-003 §2.4")
def test_profile_overrides_defaults(runner_env):
    """§2.4: без явних параметрів беруться частка, max_model_len і extra_args з профілю repo."""
    env = runner_env
    env.write_profile(REPO, _profile(fraction=0.6, max_model_len=4096, extra_args=["--enable-prefix-caching"]))
    env.start()
    argv = env.argv(DEFAULT_NAME)
    got = (env.fractions(DEFAULT_NAME)[-1], env.max_lens(DEFAULT_NAME)[-1], argv[-1])
    expected = (0.6, 4096, "--enable-prefix-caching")
    assert got == expected, f"start with a profile only: expected (fraction, max_model_len, last arg) {expected!r}, got {got!r}; argv {argv!r}"


@pytest.mark.req("SPEC-GPU-003 §2.4")
def test_explicit_overrides_profile(runner_env):
    """§2.4: явні fraction і max_model_len важливіші за профіль."""
    env = runner_env
    env.write_profile(REPO, _profile(fraction=0.6, max_model_len=4096))
    env.start(fraction=0.5, max_model_len=2048)
    got = (env.fractions(DEFAULT_NAME)[-1], env.max_lens(DEFAULT_NAME)[-1])
    assert got == (0.5, 2048), f"explicit (0.5, 2048) over profile (0.6, 4096): expected (0.5, 2048), got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.4")
def test_profile_read_at_each_start(runner_env):
    """§2.4: профіль читається при кожному старті — записаний між стартами діє на наступний.

    Перша модель зупиняється до другого старту: активна, вона займала б 0.95 частки цієї карти, і друга
    отримала б лише 1 − 0.95 (§2.4, Σ часток активних моделей)."""
    env = runner_env
    env.start(name="before")
    env.runner.stop("before", "alice")
    env.write_profile(REPO, _profile(fraction=0.6))
    env.start(name="after")
    got = (env.fractions("before")[-1], env.fractions("after")[-1])
    expected = (FRACTION_EMPTY, 0.6)
    assert got == expected, f"fractions before/after writing a profile: expected {expected!r}, got {got!r}"


# --- §2.5 межа пам'яті карти ---------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-003 §2.5")
def test_explicit_fraction_within_free_accepted(runner_env):
    """§2.5: явна частка в межах free − 256 MiB приймається (used 10000: 0.77·46068 = 35472 ≤ 35812)."""
    env = runner_env
    env.set_used(GPU, USED_MID_MIB)
    env.start(fraction=FRACTION_USED_MID)
    got = env.fractions(DEFAULT_NAME)
    assert got == [FRACTION_USED_MID], f"explicit fraction {FRACTION_USED_MID} within free memory: expected a start with it, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.4")
@pytest.mark.req("SPEC-GPU-003 §2.5")
def test_explicit_fraction_above_default_fraction_accepted(runner_env):
    """§2.4–2.5: vllm.default_fraction (типові 0.95) — стеля лише неявної частки. Явна 0.97 на порожній карті
    46068 MiB не більша за вільну (45812/46068 = 0.9944 → 0.99) і стартує рівно з 0.97, без gpu_memory_low."""
    from gpu_manager.messages import ManagerError

    env = runner_env
    try:
        env.start(fraction=0.97)
    except ManagerError as exc:
        pytest.fail(
            f"explicit fraction 0.97 on an empty card, default_fraction 0.95: expected a start, "
            f"got ManagerError {getattr(exc, 'code', None)!r}: {exc}"
        )
    got = env.fractions(DEFAULT_NAME)
    assert got == [0.97], f"explicit fraction 0.97 above default_fraction 0.95: expected a start with [0.97], got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.4")
@pytest.mark.req("SPEC-GPU-003 §2.5")
def test_profile_fraction_over_free_clamped(runner_env):
    """§2.4–2.5: частка з профілю — лише верхня межа: понад вільну (used 10000 → 0.77) старт іде з вільною
    часткою, без gpu_memory_low (відмова — лише для явної)."""
    env = runner_env
    env.set_used(GPU, USED_MID_MIB)
    env.write_profile(REPO, _profile(fraction=FRACTION_OVER_MID, max_model_len=None))
    env.start()
    got = env.fractions(DEFAULT_NAME)
    assert got == [FRACTION_USED_MID], (
        f"profile fraction {FRACTION_OVER_MID} over the free share: expected a start with the free share "
        f"[{FRACTION_USED_MID}], got {got!r}"
    )


@pytest.mark.req("SPEC-GPU-003 §2.4")
@pytest.mark.req("SPEC-GPU-003 §2.5")
def test_explicit_fraction_over_share_left_by_active_model_refused(runner_env):
    """§2.4–2.5: вільна частка враховує частки активних моделей карти — після явних 0.5 лишається 0.5;
    явна 0.6 > 0.5 → gpu_memory_low."""
    env = runner_env
    env.start(name="first", fraction=0.5)
    expect_manager_error("gpu_memory_low", env.start, name="second", fraction=0.6)


@pytest.mark.req("SPEC-GPU-003 §2.5")
def test_implicit_fraction_below_minimum_refused(make_runner_env):
    """§2.4–2.5: активна модель з явними 0.97 лишає карті 1 − 0.97 = 0.03; типовий старт другої моделі на цій
    карті (неявна 0.03 < 0.05) → gpu_memory_low. vllm.default_fraction 1.0 не потрібна (стеля 0.95 обмежує лише
    неявну частку, явна 0.97 ≤ вільної 0.99), але не заважає."""
    env = make_runner_env(NO_DEFAULT_CAP)
    env.start(name="first", fraction=0.97)
    expect_manager_error("gpu_memory_low", env.start, name="second")


@pytest.mark.req("SPEC-GPU-003 §2.5")
def test_implicit_fraction_above_minimum_started(runner_env):
    """§2.5: неявна частка ≥ 0.05 не відмовляється — на майже повній карті (used 40000) старт іде з 0.12."""
    env = runner_env
    env.set_used(GPU, USED_HIGH_MIB)
    env.start()
    got = env.fractions(DEFAULT_NAME)
    assert got == [FRACTION_USED_HIGH], f"default fraction with {USED_HIGH_MIB} MiB used: expected [{FRACTION_USED_HIGH}], got {got!r}"


# --- §2.6 argv, env, лог ---------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-003 §2")
def test_unit_name(runner_env):
    """§2: юніт моделі — gm-model-<name>.service (суфікс явний)."""
    env = runner_env
    env.start()
    units = [c.unit for c in env.launcher.calls]
    expected = [f"gm-model-{DEFAULT_NAME}.service"]
    assert units == expected, f"launcher.start units: expected {expected!r}, got {units!r}"


@pytest.mark.req("SPEC-GPU-003 §2")
def test_unit_names_of_x_and_x_service_differ(runner_env):
    """§2: назви x і x.service — різні юніти: gm-model-x.service і gm-model-x.service.service."""
    env = runner_env
    env.start(name="x", gpu=GPU)
    env.start(name="x.service", gpu=GPU_B)
    got = sorted(c.unit for c in env.launcher.calls)
    expected = ["gm-model-x.service", "gm-model-x.service.service"]
    assert got == expected, f"launcher.start units for names 'x' and 'x.service': expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2")
@pytest.mark.req("SPEC-GPU-003 §2.9")
@pytest.mark.parametrize(("stopped", "kept"), [("x", "x.service"), ("x.service", "x")])
def test_stop_one_of_x_and_x_service_keeps_other(runner_env, stopped, kept):
    """§2, §2.9: stop однієї з моделей x / x.service не зачіпає юніта іншої: той лишається живим і в servers()."""
    env = runner_env
    env.start(name="x", gpu=GPU)
    env.start(name="x.service", gpu=GPU_B)
    env.runner.stop(stopped, "alice")
    alive = sorted(env.launcher.alive)
    got = (env.unit(kept) in env.launcher.alive, env.unit(stopped) in env.launcher.alive, kept in env.names())
    assert got == (True, False, True), (
        f"after stop({stopped!r}): expected (unit of {kept!r} alive, unit of {stopped!r} gone, {kept!r} listed) "
        f"(True, False, True), got {got!r}; alive units {alive!r}, servers {env.names()!r}"
    )


@pytest.mark.req("SPEC-GPU-003 §2.6")
@pytest.mark.parametrize("inbox_exists", [True, False], ids=["inbox-exists", "inbox-missing"])
def test_argv_exact(make_runner_env, tmp_path, inbox_exists):
    """§2.6: argv дослівно — bin serve <знімок> --host --port --served-model-name --gpu-memory-utilization
    --disable-uvicorn-access-log [--allowed-local-media-path <files.inbox_dir>, лише коли тека існує] --max-model-len
    --max-num-seqs (типове 32), далі extra_args у своєму порядку. files.inbox_dir — тека в tmp_path: з типовим
    /srv/gpu-inbox (SPEC-GPU-001 §2.4) argv залежав би від сервера, на якому йдуть тести."""
    inbox = tmp_path.resolve() / "gpu-inbox"
    if inbox_exists:
        inbox.mkdir()
    env = make_runner_env({"files.inbox_dir": str(inbox)})
    media = [MEDIA_FLAG, str(inbox)] if inbox_exists else []
    extra = ["--enable-prefix-caching", "--seed", "7"]
    env.start(fraction=0.5, max_model_len=4096, extra_args=extra)
    expected = [
        str(env.cfg.vllm_bin), "serve", str(env.snapshot(REPO)),
        "--host", "127.0.0.1",
        "--port", str(PORT_FIRST),
        "--served-model-name", DEFAULT_NAME,
        "--gpu-memory-utilization", "0.5",
        "--disable-uvicorn-access-log",
        *media,
        "--max-model-len", "4096",
        "--max-num-seqs", str(MAX_NUM_SEQS_DEFAULT),
        *extra,
    ]
    got = normalize_argv(env.argv(DEFAULT_NAME))
    assert got == normalize_argv(expected), f"vLLM argv:\n expected {normalize_argv(expected)!r}\n got      {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.6")
def test_argv_media_path_follows_inbox_created_later(make_runner_env, tmp_path):
    """§2.6: «якщо тека існує» — на момент старту: перший старт без теки files.inbox_dir іде без прапорця,
    старт після створення теки тим самим ModelRunner — з --allowed-local-media-path <тека>."""
    inbox = tmp_path.resolve() / "gpu-inbox"
    env = make_runner_env({"files.inbox_dir": str(inbox)})
    env.start(name="before", gpu=GPU)
    inbox.mkdir()
    env.start(name="after", gpu=GPU_B)
    got = (flag_value(env.argv("before"), MEDIA_FLAG), flag_value(env.argv("after"), MEDIA_FLAG))
    expected = (None, str(inbox))
    assert got == expected, (
        f"{MEDIA_FLAG} value (start before the inbox existed, start after it was created): "
        f"expected {expected!r}, got {got!r}"
    )


@pytest.mark.req("SPEC-GPU-003 §2.6")
@pytest.mark.req("SPEC-GPU-003 §1")
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [({}, str(MAX_NUM_SEQS_DEFAULT)), ({"vllm.max_num_seqs": 1}, "1")],
    ids=["default-32", "configured-1"],
)
def test_max_num_seqs_from_config(make_runner_env, overrides, expected):
    """§1, §2.6: без --max-num-seqs в extra_args argv має рівно один --max-num-seqs <vllm.max_num_seqs>
    (типово 32; 1 — нижня допустима межа)."""
    env = make_runner_env(overrides)
    env.start()
    got = max_num_seqs_values(env.argv(DEFAULT_NAME))
    assert got == [expected], f"--max-num-seqs values in argv: expected [{expected!r}], got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.6")
@pytest.mark.parametrize(
    "extra",
    [["--max-num-seqs", "4"], ["--max-num-seqs=4"], ["--max_num_seqs=4"]],
    ids=["separate-value", "equals-form", "underscore-equals-form"],
)
def test_max_num_seqs_not_added_when_in_extra_args(runner_env, extra):
    """§2.6: --max-num-seqs / --max_num_seqs (з окремим значенням чи «=N») уже є в extra_args — типовий
    не додається, лишається лише значення з extra_args."""
    env = runner_env
    env.start(extra_args=list(extra))
    got = max_num_seqs_values(env.argv(DEFAULT_NAME))
    assert got == ["4"], f"--max-num-seqs values with extra_args {extra!r}: expected ['4'] only, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.6")
def test_env_card_cache_offline(runner_env):
    """§2.6: env — CUDA_DEVICE_ORDER=PCI_BUS_ID (номер карти в CUDA = номер у NVML), CUDA_VISIBLE_DEVICES=<uuid карти
    з NVML> (uuid фальшивого бекенда; не успадкований від тестового процесу), HF_HOME, HF_HUB_OFFLINE=1,
    VLLM_NO_USAGE_STATS=1, CUDA_HOME."""
    env = runner_env
    env.start()
    call_env = env.launcher.calls[-1].env

    def _path(value: Any) -> Any:
        return None if value is None else str(Path(value).resolve())

    got = {
        "CUDA_DEVICE_ORDER": call_env.get("CUDA_DEVICE_ORDER"),
        "VLLM_NO_USAGE_STATS": call_env.get("VLLM_NO_USAGE_STATS"),
        "CUDA_VISIBLE_DEVICES": call_env.get("CUDA_VISIBLE_DEVICES"),
        "HF_HOME": _path(call_env.get("HF_HOME")),
        "HF_HUB_OFFLINE": call_env.get("HF_HUB_OFFLINE"),
        "CUDA_HOME": _path(call_env.get("CUDA_HOME")),
    }
    expected = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "VLLM_NO_USAGE_STATS": "1",
        "CUDA_VISIBLE_DEVICES": env.card_uuid(GPU),
        "HF_HOME": _path(env.cfg.hf_home),
        "HF_HUB_OFFLINE": "1",
        "CUDA_HOME": _path(env.cfg.vllm_cuda_home),
    }
    assert got == expected, f"vLLM env: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.6")
def test_env_card_number_when_uuid_empty(backend, make_runner_env):
    """§2.6: uuid карти з NVML порожній → CUDA_VISIBLE_DEVICES = номер карти. uuid прибирається в бекенді ще до
    збирання сервісу: менеджер читає info() при збиранні (SPEC-GPU-001 §10)."""
    backend.infos[GPU] = replace(backend.infos[GPU], uuid="")
    env = make_runner_env()
    env.start()
    got = env.launcher.calls[-1].env.get("CUDA_VISIBLE_DEVICES")
    assert got == str(GPU), f"vLLM env CUDA_VISIBLE_DEVICES for gpu {GPU} with an empty uuid: expected {str(GPU)!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.6")
def test_env_path_starts_with_cuda_bin(runner_env):
    """§2.6: PATH починається з <cuda_home>/bin: (далі — решта PATH)."""
    env = runner_env
    env.start()
    path = env.launcher.calls[-1].env.get("PATH") or ""
    prefix = str(Path(env.cfg.vllm_cuda_home) / "bin") + ":"
    assert path.startswith(prefix), f"vLLM env PATH: expected to start with {prefix!r}, got {path[:200]!r}"


@pytest.mark.req("SPEC-GPU-003 §2")
@pytest.mark.req("SPEC-GPU-003 §2.6")
def test_log_path_given_to_launcher(runner_env):
    """§2, §2.6: лог моделі — data/servers/<name>.log; launcher.start отримує його як Path (log: Path)."""
    env = runner_env
    env.start()
    log = env.launcher.calls[-1].log
    expected = env.log_file(DEFAULT_NAME).resolve()
    got = (isinstance(log, Path), Path(log).resolve() if isinstance(log, (str, Path)) else log)
    assert got == (True, expected), f"launcher.start log: expected Path {expected}, got {log!r}"


@pytest.mark.req("SPEC-GPU-003 §2.6")
def test_log_marker_on_start(runner_env):
    """§2.6: старт пише в лог моделі рядок-маркер «=== gpu-manager start»."""
    env = runner_env
    env.start()
    got = env.markers(DEFAULT_NAME)
    assert got == 1, f"{env.log_file(DEFAULT_NAME)}: expected 1 line starting with {MARKER!r} after one start, got {got}"
