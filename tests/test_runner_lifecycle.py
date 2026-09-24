"""§2.7–2.12 SPEC-GPU-003: стани сервера моделі, профіль, тайм-аут старту, автоповтор, стоп, відновлення,
списки й логи, захист від видалення запущеної моделі.

Крок стану — RunnerEnv.step() (ModelRunner.poll(); назви методу §2 не дає — прогалина). Падіння vLLM на
старті моделюється так: тест дописує в лог моделі data_dir/servers/<name>.log справжній текст помилки vLLM
(після маркера старту, який пише сам runner) і гасить юніт у FakeLauncher; далі RunnerEnv.die_and_step()
робить кроки, доки runner не перезапустить юніт або не позначить сервер failed. Рядки помилок —
у tests/runner_fakes.py, кожен містить рівно один тригер §2.8.
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import MEM_TOTAL_MIB, T0
from .model_fakes import REPO, expect_manager_error, model_dir
from .runner_fakes import (
    COMPILE_LINES,
    DEFAULT_NAME,
    FRACTION_EMPTY,
    GPU,
    GPU_B,
    LADDER_BELOW_DEFAULT,
    LINE_INFO,
    LINE_INFO_REMOTE_CODE,
    LINE_KV_CACHE,
    LINE_LESS_THAN_DESIRED,
    LINE_NO_CACHE_BLOCKS,
    LINE_OOM,
    LINE_REMOTE_CODE,
    LINE_REMOTE_CODE_BARE,
    LINE_UNRECOGNIZED,
    PORT_FIRST,
    PROFILE_FIELDS,
    START_TIMEOUT_S,
    info_line,
    line_estimated_len,
    line_mamba,
    max_num_seqs_values,
    normalize_argv,
)

pytestmark = [pytest.mark.component, pytest.mark.usefixtures("isolated_home")]

NAME = DEFAULT_NAME
LOGS_FIELDS = {"name", "status", "hint", "hint_params", "lines"}  # §2.11


# --- §2.7 стани -------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-003 §2.7")
def test_start_status_is_starting(runner_env):
    """§2.7: одразу після start() — стан starting і один запуск юніта."""
    env = runner_env
    env.start()
    got = (env.status(NAME), len(env.starts(NAME)))
    assert got == ("starting", 1), f"right after start(): expected ('starting', 1 unit start), got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.7")
def test_not_running_until_healthy(runner_env):
    """§2.7: юніт активний, але health ще не відповідає — сервер лишається starting."""
    env = runner_env
    env.start()
    env.step()
    got = env.status(NAME)
    assert got == "starting", f"active unit, health False: expected 'starting', got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.7")
def test_running_when_active_and_healthy(runner_env):
    """§2.7: юніт активний і health відповідає → running."""
    env = runner_env
    env.start()
    got = env.run(NAME)
    assert got == "running", f"active unit, health True: expected 'running', got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.7")
def test_profile_not_written_while_starting(runner_env):
    """§2.7: профіль пишеться лише при переході в running — поки starting, запису для repo немає."""
    env = runner_env
    env.start()
    env.step()
    got = env.profiles().get(REPO)
    assert got is None, f"{env.profiles_path} while starting: expected no entry for {REPO!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.7")
def test_profile_written_on_running(runner_env):
    """§2.7: при переході в running зберігається профіль {fraction, max_model_len, extra_args, gpu, updated,
    attempts} під ключем repo; значення — з цього старту."""
    env = runner_env
    env.start()
    env.ensure_running(NAME)
    profile = env.profiles().get(REPO)
    assert isinstance(profile, dict) and PROFILE_FIELDS <= set(profile), (
        f"{env.profiles_path}: expected an entry for {REPO!r} with fields {sorted(PROFILE_FIELDS)}, got {profile!r}"
    )
    got = (round(float(profile["fraction"]), 4), profile["gpu"], profile["attempts"])
    expected = (FRACTION_EMPTY, GPU, 1)
    assert got == expected, f"profile of {REPO!r}: expected (fraction, gpu, attempts) {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.7")
@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_profile_keeps_auto_tuned_fraction(runner_env):
    """§2.7–2.8: після автоповтору профіль зберігає підібрану частку і кількість спроб."""
    env = runner_env
    env.start()
    env.die_and_step(NAME, [LINE_LESS_THAN_DESIRED])
    env.ensure_running(NAME)
    profile = env.profiles().get(REPO) or {}
    fraction = profile.get("fraction")
    got = (round(float(fraction), 4) if isinstance(fraction, (int, float)) else fraction, profile.get("attempts"))
    assert got == (0.95, 2), f"profile after one retry 0.99 -> 0.95: expected (fraction 0.95, attempts 2), got {got!r}"


def _profile_06() -> dict[str, Any]:
    """Запис профілю §2.7 з часткою 0.6 для REPO на карті GPU."""
    return {"fraction": 0.6, "max_model_len": None, "extra_args": [], "gpu": GPU, "updated": T0, "attempts": 1}


def _profile_fraction(env: Any) -> Any:
    """Частка з профілю REPO (4 знаки); немає запису чи частки — те, що є, для повідомлення тесту."""
    fraction = (env.profiles().get(REPO) or {}).get("fraction")
    return round(float(fraction), 4) if isinstance(fraction, (int, float)) else fraction


@pytest.mark.req("SPEC-GPU-003 §2.7")
def test_profile_fraction_kept_after_explicit_start(runner_env):
    """§2.7: явна частка з запиту (0.5) не потрапляє в профіль — після running у профілі лишається попередня 0.6."""
    env = runner_env
    env.write_profile(REPO, _profile_06())
    env.start(fraction=0.5)
    env.ensure_running(NAME)
    got = _profile_fraction(env)
    assert got == 0.6, f"profile fraction after an explicit 0.5 start reached running: expected 0.6 kept, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.7")
def test_profile_fraction_overwritten_by_derived(runner_env):
    """§2.4, §2.7: старт без явної частки — менеджер виводить min(профіль 0.6, вільна 0.56 при used 20000) = 0.56;
    після running ця виведена частка перезаписує 0.6 у профілі."""
    env = runner_env
    env.set_used(GPU, 20_000)  # (46068 − 20000 − 256)/46068 = 0.5603 → вільна 0.56
    env.write_profile(REPO, _profile_06())
    env.start()
    env.ensure_running(NAME)
    got = (env.fractions(NAME), _profile_fraction(env))
    assert got == ([0.56], 0.56), f"derived start under a 0.6 profile: expected (argv fractions [0.56], profile 0.56), got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.7")
def test_journal_model_start(runner_env):
    """§2.7: у журналі — model_start з користувачем і картою."""
    env = runner_env
    env.start(user="carol")
    env.ensure_running(NAME)
    got = [(e.get("user"), e.get("gpu")) for e in env.journal("model_start")]
    assert got == [("carol", GPU)], f"journal model_start entries: expected [('carol', {GPU})], got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.7")
@pytest.mark.req("SPEC-GPU-003 §1")
def test_start_timeout_not_reached(runner_env):
    """§2.7, §1: starting не довше start_timeout_s (600 с − 1) — ще starting, юніт не зупинено."""
    env = runner_env
    env.start()
    env.clock.advance(START_TIMEOUT_S - 1)
    env.step()
    got = (env.status(NAME), env.launcher.stops)
    assert got == ("starting", []), f"{START_TIMEOUT_S - 1} s after start: expected ('starting', no stops), got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.7")
@pytest.mark.req("SPEC-GPU-003 §1")
def test_start_timeout_stops_and_fails(runner_env):
    """§2.7, §1: starting довше start_timeout_s → юніт зупинено, failed з error_code hint_start_timeout."""
    env = runner_env
    env.start()
    env.clock.advance(START_TIMEOUT_S + 1)
    env.step()
    record = env.server(NAME)
    got = (record.get("status"), record.get("error_code"), env.unit(NAME) in env.launcher.stops)
    expected = ("failed", "hint_start_timeout", True)
    assert got == expected, f"{START_TIMEOUT_S + 1} s in starting: expected (status, error_code, unit stopped) {expected!r}, got {got!r}"


# --- §2.8 автоповтор -----------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_retry_estimated_len_rounded_down_to_1024(runner_env):
    """§2.8: «estimated maximum model length is 12345» → max_model_len 12288 (донизу до кратного 1024);
    частка не змінюється."""
    env = runner_env
    env.start()
    env.die_and_step(NAME, [line_estimated_len(12345)])
    got = (env.max_lens(NAME), env.fractions(NAME))
    expected = ([None, 12288], [FRACTION_EMPTY, FRACTION_EMPTY])
    assert got == expected, f"retry after an estimated length of 12345: expected (max_model_len, fraction) per attempt {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_retry_estimated_len_below_1024_taken_as_is(runner_env):
    """§2.8: N < 1024 — max_model_len = N як є (700, а не 0)."""
    env = runner_env
    env.start()
    env.die_and_step(NAME, [line_estimated_len(700)])
    got = env.max_lens(NAME)
    assert got == [None, 700], f"max_model_len per attempt after an estimated length of 700: expected [None, 700], got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.8")
@pytest.mark.req("SPEC-GPU-003 §2.6")
def test_retry_mamba_sets_max_num_seqs(runner_env):
    """§2.8: «exceeds available Mamba cache blocks (24)» → наступна спроба з --max-num-seqs 24 в extra_args;
    типового --max-num-seqs 32 поруч немає (§2.6)."""
    env = runner_env
    env.start()
    env.die_and_step(NAME, [line_mamba(32, 24)])
    got = [max_num_seqs_values(call.argv) for call in env.starts(NAME)]
    assert got == [["32"], ["24"]], f"--max-num-seqs values per attempt: expected [['32'], ['24']], got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_retry_mamba_replaces_existing_max_num_seqs(runner_env):
    """§2.8: --max-num-seqs з правила Mamba замінює попередній в extra_args (128 → 100), а не додається другим."""
    env = runner_env
    env.start(extra_args=["--max-num-seqs", "128"])
    env.die_and_step(NAME, [line_mamba(128, 100)])
    got = [max_num_seqs_values(call.argv) for call in env.starts(NAME)]
    assert got == [["128"], ["100"]], f"--max-num-seqs values per attempt: expected [['128'], ['100']], got {got!r}"


# Кожне падіння — нова межа Mamba: 32 → 24 → 20 → 16 → 12; п'яте падіння (N = 8) — кінець спроб.
MAMBA_CHAIN = ((32, 24), (24, 20), (20, 16), (16, 12), (12, 8))


@pytest.mark.req("SPEC-GPU-003 §2.8")
@pytest.mark.req("SPEC-GPU-003 §2.7")
def test_mamba_failure_hint_with_n(runner_env):
    """§2.7–2.8, коди підказок: спроби вичерпано на помилці Mamba → failed з error_code hint_mamba_seqs і n = 8."""
    env = runner_env
    env.start()
    for seqs, n in MAMBA_CHAIN:
        env.die_and_step(NAME, [line_mamba(seqs, n)])
    record = env.server(NAME)
    params = record.get("error_params")
    n = params.get("n") if isinstance(params, dict) else None
    got = (len(env.starts(NAME)), record.get("status"), record.get("error_code"), str(n))
    expected = (5, "failed", "hint_mamba_seqs", "8")
    assert got == expected, f"after 5 Mamba failures: expected (starts, status, error_code, n) {expected!r}, got {record!r}"


@pytest.mark.req("SPEC-GPU-003 §2.8")
@pytest.mark.req("SPEC-GPU-003 §2.7")
def test_profile_keeps_mamba_max_num_seqs(runner_env):
    """§2.7–2.8: --max-num-seqs, встановлений правилом Mamba, після running потрапляє в профіль разом з extra_args."""
    env = runner_env
    env.start()
    env.die_and_step(NAME, [line_mamba(32, 24)])
    env.ensure_running(NAME)
    extra = (env.profiles().get(REPO) or {}).get("extra_args")
    got = max_num_seqs_values(extra) if isinstance(extra, list) else extra
    assert got == ["24"], f"profile extra_args after a Mamba retry reached running: expected --max-num-seqs 24, got {extra!r}"


@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_retry_estimated_len_below_256_doubles_fraction(runner_env):
    """§2.8: N < 256 — як «No available memory»: частка ×2 (0.3 → 0.6), max_model_len не задається."""
    env = runner_env
    env.start(fraction=0.3)
    env.die_and_step(NAME, [line_estimated_len(200)])
    got = (env.fractions(NAME), env.max_lens(NAME))
    expected = ([0.3, 0.6], [None, None])
    assert got == expected, f"retry after an estimated length of 200: expected (fractions, max_model_lens) {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.4")
@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_retry_double_fraction_excludes_own_share(runner_env):
    """§2.4, §2.8: ×2 «у межах вільної» — вільна враховує інші активні моделі карти, але не власну частку:
    інша 0.5, своя 0.3 → 0.6 обрізається до 1 − 0.5 = 0.5 (з власною в сумі було б 0.2, без інших — 0.6)."""
    env = runner_env
    env.start(name="other", fraction=0.5)
    env.start(fraction=0.3)
    env.die_and_step(NAME, [LINE_NO_CACHE_BLOCKS])
    got = env.fractions(NAME)
    assert got == [0.3, 0.5], f"fraction per attempt next to an active 0.5 model: expected [0.3, 0.5], got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_error_line_by_error_class_without_error_level(runner_env):
    """§2.8: рядок помилки — і без «ERROR», якщо в ньому «…Error»: голий «RuntimeError: … trust_remote_code=True»
    дає hint_remote_code."""
    env = runner_env
    env.start()
    env.ensure_failed(NAME, LINE_REMOTE_CODE_BARE)
    got = env.server(NAME).get("error_code")
    assert got == "hint_remote_code", f"failure on a bare RuntimeError line: expected error_code 'hint_remote_code', got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_info_line_is_not_error_line(runner_env):
    """§2.8: рядок INFO з «trust_remote_code=True» — не рядок помилки, hint_remote_code не дає."""
    env = runner_env
    env.start()
    env.ensure_failed(NAME, LINE_INFO_REMOTE_CODE)
    got = env.server(NAME).get("error_code")
    assert got != "hint_remote_code", f"failure with only an INFO line mentioning trust_remote_code=True: expected no 'hint_remote_code', got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_retry_fraction_ladder(runner_env):
    """§2.8: «less than desired GPU memory utilization» → частка щоразу на щабель нижче: 0.99 → 0.95 → 0.92 → 0.9."""
    env = runner_env
    env.start()
    for _ in LADDER_BELOW_DEFAULT:
        env.die_and_step(NAME, [LINE_LESS_THAN_DESIRED])
    got = env.fractions(NAME)
    expected = [FRACTION_EMPTY, *LADDER_BELOW_DEFAULT]
    assert got == expected, f"fraction per attempt: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_retry_oom_lowers_fraction_first(runner_env):
    """§2.8: OOM, поки є нижчий щабель, → нижча частка; max_model_len не чіпається."""
    env = runner_env
    env.start()
    env.die_and_step(NAME, [LINE_OOM])
    got = (env.fractions(NAME), env.max_lens(NAME))
    expected = ([FRACTION_EMPTY, 0.95], [None, None])
    assert got == expected, f"retry after OOM at 0.99: expected (fractions, max_model_lens) {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_retry_oom_at_bottom_sets_len_8192(runner_env):
    """§2.8: частка 0.9 (нижче нікуди) + OOM, max_model_len не задано → 8192; частка лишається 0.9."""
    env = runner_env
    env.start(fraction=0.9)
    env.die_and_step(NAME, [LINE_OOM])
    got = (env.fractions(NAME), env.max_lens(NAME))
    expected = ([0.9, 0.9], [None, 8192])
    assert got == expected, f"retry after OOM at 0.9: expected (fractions, max_model_lens) {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_retry_kv_cache_at_bottom_halves_len(runner_env):
    """§2.8: частка 0.9 + «KV cache» → max_model_len удвічі менший: 16384 → 8192."""
    env = runner_env
    env.start(fraction=0.9, max_model_len=16384)
    env.die_and_step(NAME, [LINE_KV_CACHE])
    got = env.max_lens(NAME)
    assert got == [16384, 8192], f"max_model_len per attempt after a KV cache error: expected [16384, 8192], got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_retry_no_cache_blocks_doubles_fraction(runner_env):
    """§2.8: «No available memory for the cache blocks» → частка ×2 (0.3 → 0.6 на порожній карті)."""
    env = runner_env
    env.start(fraction=0.3)
    env.die_and_step(NAME, [LINE_NO_CACHE_BLOCKS])
    got = env.fractions(NAME)
    assert got == [0.3, 0.6], f"fraction per attempt: expected [0.3, 0.6], got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_retry_no_cache_blocks_capped_by_free(runner_env):
    """§2.8: ×2 лише в межах вільної пам'яті: used 20000 → частка 0.4 росте, але не вище (46068 − 20000)/46068."""
    env = runner_env
    used = 20_000
    env.set_used(GPU, used)
    env.start(fraction=0.4)
    env.die_and_step(NAME, [LINE_NO_CACHE_BLOCKS])
    got = env.fractions(NAME)
    cap = (MEM_TOTAL_MIB - used) / MEM_TOTAL_MIB
    ok = len(got) == 2 and got[1] is not None and 0.4 < got[1] <= cap
    assert ok, f"fraction per attempt with {used} MiB used: expected [0.4, x] with 0.4 < x <= {cap:.4f}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.8")
@pytest.mark.parametrize("kind", sorted(COMPILE_LINES))
def test_retry_compile_error_enforce_eager(runner_env, kind):
    """§2.8: помилка компіляції (Ninja build failed, nvcc fatal, InductorError, torch._dynamo) → --enforce-eager."""
    env = runner_env
    env.start()
    env.die_and_step(NAME, [COMPILE_LINES[kind]])
    got = ["--enforce-eager" in call.argv for call in env.starts(NAME)]
    assert got == [False, True], f"{kind}: expected --enforce-eager only in the second attempt, got per attempt {got!r}"


# Кожна смерть дає нову оцінку довжини: 20000 → 19456, 15000 → 14336, 12000 → 11264, 9000 → 8192; п'ята — кінець.
EXHAUST_ESTIMATES = (20000, 15000, 12000, 9000, 8192)


def _exhaust_attempts(env: Any) -> None:
    """Старт і п'ять падінь поспіль, кожне — з правилом автоповтору, що могло б спрацювати й далі."""
    env.start()
    for n in EXHAUST_ESTIMATES:
        env.die_and_step(NAME, [line_estimated_len(n)])
    env.step()
    env.step()


@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_retry_at_most_five_attempts(runner_env):
    """§2.8: не більше 5 спроб разом із першою; після п'ятого падіння — failed, шостого старту немає."""
    env = runner_env
    _exhaust_attempts(env)
    got = (len(env.starts(NAME)), env.status(NAME))
    assert got == (5, "failed"), f"after 5 failed attempts: expected (5 unit starts, 'failed'), got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.7")
@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_failed_after_attempts_has_error_code_and_params(runner_env):
    """§2.7: failed несе error_code / error_params — за останньою помилкою: hint_kv_len з n = 8192."""
    env = runner_env
    _exhaust_attempts(env)
    record = env.server(NAME)
    params = record.get("error_params")
    n = params.get("n") if isinstance(params, dict) else None
    got = (record.get("error_code"), str(n))
    assert got == ("hint_kv_len", "8192"), f"failed record: expected error_code 'hint_kv_len' with n 8192, got {record!r}"


@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_retry_changes_listed_in_auto_changes(runner_env):
    """§2.8: кожна зміна автоповтору — окремий запис у auto_changes."""
    env = runner_env
    env.start()
    env.die_and_step(NAME, [LINE_LESS_THAN_DESIRED])
    env.die_and_step(NAME, [LINE_LESS_THAN_DESIRED])
    changes = env.server(NAME).get("auto_changes")
    count = len(changes) if isinstance(changes, list) else None
    assert count == 2, f"auto_changes after two retries: expected a list of 2 changes, got {changes!r}"


@pytest.mark.req("SPEC-GPU-003 §2.7")
@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_retry_journaled_with_change(runner_env):
    """§2.7: кожен автоповтор — запис журналу model_retry з непорожнім change."""
    env = runner_env
    env.start()
    env.die_and_step(NAME, [LINE_LESS_THAN_DESIRED])
    entries = env.journal("model_retry")
    got = [bool(e.get("change")) for e in entries]
    assert got == [True], f"journal model_retry after one retry: expected one entry with a change, got {entries!r}"


@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_retry_reads_only_last_start_log(runner_env):
    """§2.8: правило шукається лише в лозі останнього старту — давня «less than desired» не дає третьої спроби."""
    env = runner_env
    env.start()
    env.die_and_step(NAME, [LINE_LESS_THAN_DESIRED])
    env.die_and_step(NAME, [LINE_REMOTE_CODE])
    record = env.server(NAME)
    got = (len(env.starts(NAME)), record.get("status"), record.get("error_code"))
    expected = (2, "failed", "hint_remote_code")
    assert got == expected, f"second death with a remote-code error: expected (starts, status, error_code) {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.7")
@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_unrecognized_error_fails_without_retry(runner_env):
    """§2.7–2.8: жоден тригер не збігся → одразу failed, без повторного старту."""
    env = runner_env
    env.start()
    env.die_and_step(NAME, [LINE_UNRECOGNIZED])
    got = (len(env.starts(NAME)), env.status(NAME))
    assert got == (1, "failed"), f"death with an unrecognized error: expected (1 start, 'failed'), got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.7")
def test_failure_journaled_model_failed(runner_env):
    """§2.7: failed — запис журналу model_failed з полем error."""
    env = runner_env
    env.start()
    env.ensure_failed(NAME, LINE_REMOTE_CODE)
    entries = env.journal("model_failed")
    got = [bool(e.get("error")) for e in entries]
    assert got == [True], f"journal model_failed: expected one entry with an error, got {entries!r}"


@pytest.mark.req("SPEC-GPU-003 §2.6")
@pytest.mark.req("SPEC-GPU-003 §2.8")
def test_log_marker_per_attempt(runner_env):
    """§2.6: кожен старт, зокрема автоповтор, — свій рядок-маркер у тому самому лозі."""
    env = runner_env
    env.start()
    env.die_and_step(NAME, [LINE_LESS_THAN_DESIRED])
    got = (len(env.starts(NAME)), env.markers(NAME))
    assert got == (2, 2), f"after one retry: expected (2 starts, 2 marker lines), got {got!r}"


# --- §2.9 стоп ----------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-003 §2.9")
def test_stop_running_stops_unit(runner_env):
    """§2.9: stop активного сервера зупиняє юніт; сервер зникає зі servers() (там лише активні й failed)."""
    env = runner_env
    env.start()
    env.ensure_running(NAME)
    env.runner.stop(NAME, "alice")
    got = (env.unit(NAME) in env.launcher.stops, NAME in env.names())
    assert got == (True, False), f"after stop: expected (unit stopped, not listed) (True, False), got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.9")
@pytest.mark.req("SPEC-GPU-003 §2.7")
def test_stop_journaled_model_stop(runner_env):
    """§2.7, §2.9: stop — запис журналу model_stop від того, хто зупинив."""
    env = runner_env
    env.start()
    env.ensure_running(NAME)
    env.runner.stop(NAME, "bob")
    got = [e.get("user") for e in env.journal("model_stop")]
    assert got == ["bob"], f"journal model_stop users: expected ['bob'], got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.9")
@pytest.mark.req("SPEC-GPU-003 §2.10")
def test_stopped_not_restored(runner_env):
    """§2.9: зупинений сервер не піднімається після перезапуску менеджера (restore)."""
    env = runner_env
    env.start()
    env.ensure_running(NAME)
    env.runner.stop(NAME, "alice")
    restarted = env.restart()
    restarted.runner.restore()
    got = len(restarted.starts(NAME))
    assert got == 1, f"unit starts after stop, restart and restore(): expected 1 (no restart), got {got}"


@pytest.mark.req("SPEC-GPU-003 §2.9")
def test_stop_failed_dismisses(runner_env):
    """§2.9: stop сервера у стані failed прибирає його зі списку."""
    env = runner_env
    env.start()
    env.ensure_failed(NAME, LINE_REMOTE_CODE)
    env.runner.stop(NAME, "alice")
    got = env.names()
    assert NAME not in got, f"after stop of a failed server: expected {NAME!r} gone from servers(), got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.9")
def test_stop_failed_keeps_model_watched(runner_env):
    """§2.9: launcher.stop повернув False → stop_failed; модель лишається під наглядом у стані starting."""
    env = runner_env
    env.start()
    env.ensure_running(NAME)
    env.launcher.stop_fails.add(env.unit(NAME))
    expect_manager_error("stop_failed", env.runner.stop, NAME, "alice")
    got = env.status(NAME)
    assert got == "starting", f"after a failed launcher.stop: expected {NAME!r} watched as 'starting', got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2")
@pytest.mark.req("SPEC-GPU-003 §2.9")
def test_active_unknown_skips_only_that_model(runner_env):
    """§2 заголовок, §2.9: active() → None пропускає лише цю модель у цьому проході — її стан не змінюється
    (і перезапуску немає), а друга модель у тому самому проході переходить у running."""
    env = runner_env
    env.start(name="alpha", gpu=GPU)
    env.start(name="bravo", gpu=GPU_B)
    env.probe.healthy.update({env.port("alpha"), env.port("bravo")})
    env.launcher.unknown.add(env.unit("alpha"))
    env.step()
    got = (env.status("alpha"), len(env.starts("alpha")), env.status("bravo"))
    expected = ("starting", 1, "running")
    assert got == expected, f"one pass with active(alpha) = None: expected (alpha status, alpha starts, bravo status) {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.9")
def test_stop_unknown_server_not_found(runner_env):
    """§2.9: stop сервера, якого немає, → server_not_found."""
    expect_manager_error("server_not_found", runner_env.runner.stop, "ghost", "alice")


# --- §2.10 відновлення ------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-003 §2.10")
def test_restore_restarts_dead_server_same_params(runner_env):
    """§2.10: активний запис без живого юніта після перезапуску менеджера запускається з тим самим argv
    (параметри й порт)."""
    env = runner_env
    env.start(fraction=0.5, max_model_len=4096)
    env.ensure_running(NAME)
    first = normalize_argv(env.argv(NAME))
    env.launcher.kill(env.unit(NAME))
    restarted = env.restart()
    restarted.runner.restore()
    calls = restarted.starts(NAME)
    assert len(calls) == 2, f"unit starts after restore() of a dead unit: expected 2, got {len(calls)}"
    got = normalize_argv(calls[-1].argv)
    assert got == first, f"restored argv:\n expected {first!r}\n got      {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.10")
def test_restore_leaves_live_unit(runner_env):
    """§2.10: юніт, що живий, restore() не перезапускає."""
    env = runner_env
    env.start()
    env.ensure_running(NAME)
    restarted = env.restart()
    restarted.runner.restore()
    got = len(restarted.starts(NAME))
    assert got == 1, f"unit starts after restore() with a live unit: expected 1, got {got}"


# --- §2.11 списки й логи ---------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-003 §2.11")
def test_servers_active_and_failed_sorted_by_gpu_port(runner_env):
    """§2.11: servers() — активні й failed (без зупинених), за (gpu, port).

    Три моделі на одній карті — з явною часткою 0.2: з типовою перша забрала б 0.99 і решта отримали б
    менше 0.05 (§2.4–2.5, Σ часток активних моделей карти)."""
    env = runner_env
    env.start(name="alpha", gpu=GPU_B)  # 8000
    env.start(name="bravo", gpu=GPU, fraction=0.2)  # 8001
    env.start(name="charlie", gpu=GPU, fraction=0.2)  # 8002 → failed
    env.start(name="delta", gpu=GPU, fraction=0.2)  # 8003 → зупинений
    env.ensure_failed("charlie", LINE_REMOTE_CODE)
    env.runner.stop("delta", "alice")
    got = env.names()
    assert got == ["bravo", "charlie", "alpha"], f"servers() order by (gpu, port): expected ['bravo', 'charlie', 'alpha'], got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.11")
def test_on_gpu_items(runner_env):
    """§2.11: on_gpu(gpu) → [{name, port, status, user}]."""
    env = runner_env
    env.start(user="bob")
    env.ensure_running(NAME)
    got = env.runner.on_gpu(GPU)
    expected = [{"name": NAME, "port": PORT_FIRST, "status": "running", "user": "bob"}]
    assert got == expected, f"on_gpu({GPU}): expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.11")
def test_in_use_while_running(runner_env):
    """§2.11: in_use(repo) істинне, поки модель repo запущена."""
    env = runner_env
    env.start()
    env.ensure_running(NAME)
    got = env.runner.in_use(REPO)
    assert bool(got) is True, f"in_use({REPO!r}) while running: expected truthy, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.11")
def test_in_use_false_after_stop(runner_env):
    """§2.11: після stop in_use(repo) хибне."""
    env = runner_env
    env.start()
    env.ensure_running(NAME)
    env.runner.stop(NAME, "alice")
    got = env.runner.in_use(REPO)
    assert bool(got) is False, f"in_use({REPO!r}) after stop: expected falsy, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.11")
def test_logs_fields(runner_env):
    """§2.11: logs(name, lines, errors_only) → {name, status, hint, hint_params, lines}."""
    env = runner_env
    env.start()
    result = env.runner.logs(NAME, 50, False)
    got = sorted(result) if isinstance(result, dict) else result
    assert got == sorted(LOGS_FIELDS), f"logs() keys: expected {sorted(LOGS_FIELDS)}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.11")
def test_logs_last_lines(runner_env):
    """§2.11: lines=3 — три останні рядки логу."""
    env = runner_env
    env.start()
    lines = [info_line(i) for i in range(10)]
    env.append_log(NAME, lines)
    got = [str(line).rstrip("\n") for line in env.runner.logs(NAME, 3, False)["lines"]]
    assert got == lines[-3:], f"logs(lines=3): expected {lines[-3:]!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.11")
def test_logs_errors_only(runner_env):
    """§2.11: errors_only — лише рядки помилок: рядок ERROR є, рядків INFO немає."""
    env = runner_env
    env.start()
    env.append_log(NAME, [LINE_INFO, LINE_REMOTE_CODE, info_line(1)])
    lines = [str(line) for line in env.runner.logs(NAME, 50, True)["lines"]]
    got = (any("trust_remote_code=True" in line for line in lines), any(" INFO " in f" {line}" for line in lines))
    assert got == (True, False), f"logs(errors_only=True): expected the ERROR line and no INFO lines, got {lines!r}"


@pytest.mark.req("SPEC-GPU-003 §2.11")
@pytest.mark.req("SPEC-GPU-003 §2")
def test_logs_hint_for_failed(runner_env):
    """§2.11, коди підказок §2: failed через «trust_remote_code=True» → hint hint_remote_code."""
    env = runner_env
    env.start()
    env.ensure_failed(NAME, LINE_REMOTE_CODE)
    result = env.runner.logs(NAME, 50, False)
    got = (result.get("status"), result.get("hint"))
    assert got == ("failed", "hint_remote_code"), f"logs() of a remote-code failure: expected ('failed', 'hint_remote_code'), got {got!r}"


# --- §2.12 видалення запущеної моделі ---------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-003 §2.12")
def test_delete_running_model_refused(runner_env):
    """§2.12: ModelStore.delete запущеної моделі → model_running; модель лишається на диску."""
    env = runner_env
    env.start()
    env.ensure_running(NAME)
    expect_manager_error("model_running", env.store.delete, REPO, "alice")
    path = model_dir(env.hf_home, REPO)
    assert path.is_dir(), f"after a refused delete: expected {path} to stay on disk"
