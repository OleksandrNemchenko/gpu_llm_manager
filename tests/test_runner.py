"""§1 і §2.1–2.6 SPEC-GPU-003: конфіг vllm і запуск моделі на vLLM.

Відмови start(), вибір порту й назви, параметри (явні → профіль → типові), межа пам'яті карти, argv і env
процесу vLLM, лог моделі. ModelRunner збирається з підробками швів §2 (tests/runner_fakes.py): юніт
«запускає» FakeLauncher, argv і env перевіряються в записаному виклику launcher.start. Карти — фальшивий
бекенд фази 1 (46068 MiB кожна); зайнята пам'ять карти задається явно (RunnerEnv.set_used).
"""

from __future__ import annotations

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
    NOT_LOCAL_REPO,
    PORT_FIRST,
    USED_HIGH_MIB,
    USED_MID_MIB,
    max_num_seqs_values,
    normalize_argv,
)

MAX_NUM_SEQS_DEFAULT = 32  # типовий vllm.max_num_seqs (§1)

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
@pytest.mark.parametrize(("field", "expected"), [("vllm_default_fraction", 1.0), ("vllm_start_timeout_s", 900)])
def test_vllm_numeric_defaults(write_config, config_dir, field, expected):
    """§1: типові default_fraction 1.0 і start_timeout_s 900."""
    cfg = _load(write_config, config_dir)
    got = getattr(cfg, field)
    assert got == expected, f"Config.{field} without the setting: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §1")
@pytest.mark.parametrize(
    ("field", "dotted", "value"),
    [
        ("vllm_default_fraction", "vllm.default_fraction", 0.8),
        ("vllm_default_fraction", "vllm.default_fraction", 0.05),  # нижня межа [0.05, 1] — ще дозволена
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


BAD_EXTRA_ARGS = {
    "not-strings": [42],
    "host": ["--host", "0.0.0.0"],
    "port": ["--port", "9000"],
    "served-model-name": ["--served-model-name", "other"],
    "gpu-memory-utilization": ["--gpu-memory-utilization", "0.5"],
}


@pytest.mark.req("SPEC-GPU-003 §2.1")
@pytest.mark.parametrize("case", sorted(BAD_EXTRA_ARGS))
def test_start_bad_extra_args(runner_env, case):
    """§2.1: extra_args не з рядків або з прапорцем, яким керує менеджер, → bad_extra_args."""
    expect_manager_error("bad_extra_args", runner_env.start, extra_args=BAD_EXTRA_ARGS[case])


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

    Перша модель зупиняється до другого старту: активна, вона займала б 0.99 частки цієї карти, і друга
    отримала б лише 1 − 0.99 (§2.4, Σ часток активних моделей)."""
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
def test_implicit_fraction_below_minimum_refused(runner_env):
    """§2.4–2.5: активна модель з явними 0.97 лишає карті 1 − 0.97 = 0.03; типовий старт другої моделі на цій
    карті (неявна 0.03 < 0.05) → gpu_memory_low."""
    env = runner_env
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
    """§2: юніт моделі — gm-model-<name>."""
    env = runner_env
    env.start()
    units = [c.unit for c in env.launcher.calls]
    assert units == [f"gm-model-{DEFAULT_NAME}"], f"launcher.start units: expected ['gm-model-{DEFAULT_NAME}'], got {units!r}"


@pytest.mark.req("SPEC-GPU-003 §2.6")
def test_argv_exact(runner_env):
    """§2.6: argv дослівно — bin serve <знімок> --host --port --served-model-name --gpu-memory-utilization
    --disable-uvicorn-access-log --max-model-len --max-num-seqs (типове 32), далі extra_args у своєму порядку."""
    env = runner_env
    extra = ["--enable-prefix-caching", "--seed", "7"]
    env.start(fraction=0.5, max_model_len=4096, extra_args=extra)
    expected = [
        str(env.cfg.vllm_bin), "serve", str(env.snapshot(REPO)),
        "--host", "127.0.0.1",
        "--port", str(PORT_FIRST),
        "--served-model-name", DEFAULT_NAME,
        "--gpu-memory-utilization", "0.5",
        "--disable-uvicorn-access-log",
        "--max-model-len", "4096",
        "--max-num-seqs", str(MAX_NUM_SEQS_DEFAULT),
        *extra,
    ]
    got = normalize_argv(env.argv(DEFAULT_NAME))
    assert got == normalize_argv(expected), f"vLLM argv:\n expected {normalize_argv(expected)!r}\n got      {got!r}"


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
    """§2.6: env — CUDA_VISIBLE_DEVICES=<gpu> (не успадкований від тестового процесу), HF_HOME, HF_HUB_OFFLINE=1,
    CUDA_HOME."""
    env = runner_env
    env.start()
    call_env = env.launcher.calls[-1].env

    def _path(value: Any) -> Any:
        return None if value is None else str(Path(value).resolve())

    got = {
        "CUDA_VISIBLE_DEVICES": call_env.get("CUDA_VISIBLE_DEVICES"),
        "HF_HOME": _path(call_env.get("HF_HOME")),
        "HF_HUB_OFFLINE": call_env.get("HF_HUB_OFFLINE"),
        "CUDA_HOME": _path(call_env.get("CUDA_HOME")),
    }
    expected = {
        "CUDA_VISIBLE_DEVICES": str(GPU),
        "HF_HOME": _path(env.cfg.hf_home),
        "HF_HUB_OFFLINE": "1",
        "CUDA_HOME": _path(env.cfg.vllm_cuda_home),
    }
    assert got == expected, f"vLLM env: expected {expected!r}, got {got!r}"


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
