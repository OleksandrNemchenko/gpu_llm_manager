"""§5.1–5.2 SPEC-GPU-002: ModelStore — стани завантаження, черга, скасування, збереження стану, процес.

ModelStore збирається над менеджером фази 1 (model_fakes.ModelsEnv): hub — FakeHub, spawn — FakeSpawn,
кеш HF і data_dir — у tmp_path, годинник — FakeClock (§10). Підроблений процес завершується лише явно
(FakeProcess.finish), тож стан черги змінюється тільки в poll() або в кроці черги всередині download().

Перевірка диска §5.1.5 іде по справжньому вільному місцю тимчасової теки: шва для неї §10 не дає. Тому
тестовий запас models.min_free_disk_gib = 0, а тестова модель «важить» ≈ 0.5 GiB за розмірами з API
(на диск тести пишуть лише кілька байтів і розріджені файли). Журнал читається з data_dir/journal.jsonl.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from .conftest import STRANGER, T0, USERS
from .model_fakes import (
    GATED_OK_REPO,
    GATED_REPO,
    HUGE_RESERVE_GIB,
    MIB,
    MISSING_REPO,
    PIB,
    REPO,
    REPO_B,
    REPO_C,
    TINY_FILES,
    TINY_TOTAL,
    TOKEN,
    append_log,
    expect_manager_error,
    gib_close,
    log_path,
    put_incomplete,
    put_snapshot,
    sha_of,
)

pytestmark = pytest.mark.component

RECORD_FIELDS = {
    "repo", "revision", "user", "status", "done_gib", "total_gib", "percent", "started", "finished", "error_code", "error",
}
GRACE_S = 5  # §5.1.8: SIGKILL через 5 с після SIGTERM
HUGE_REPO = "acme/huge-llm"
# Модель «на 1 PiB» за API: без наявних файлів потрібне місце більше за будь-який диск (§5.1.5).
HUGE_FILES = {"config.json": 700, "model-00001-of-00002.safetensors": PIB, "model-00002-of-00002.safetensors": MIB}


def _one_slot(make_models_env: Any) -> Any:
    """Сервіс з max_parallel_downloads = 1."""
    return make_models_env({"models.max_parallel_downloads": 1})


def _queue(env: Any, *repos: str, step_s: float = 10) -> None:
    """Ставить repos у чергу від alice; між викликами годинник іде на step_s — started різні."""
    for i, repo in enumerate(repos):
        if i:
            env.clock.advance(step_s)
        env.store.download(repo, "alice")


def _fail(env: Any, repo: str, lines: list[str], code: int = 1) -> None:
    """Процес repo пише рядки в лог і завершується з кодом code; далі poll()."""
    append_log(log_path(env.data_dir, repo), lines)
    env.spawn.last(repo).finish(code)
    env.store.poll()


# --- §5.1.1 користувач --------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §5.1.1")
def test_download_unknown_user_refused(models_env):
    """§5.1.1: user не з users.allowed → unknown_user."""
    expect_manager_error("unknown_user", models_env.store.download, REPO, STRANGER)


@pytest.mark.req("SPEC-GPU-002 §5.1.1")
def test_download_unknown_user_leaves_no_trace(models_env):
    """§5.1.1: відмова unknown_user — без процесу, без запису черги й без запису журналу."""
    env = models_env
    expect_manager_error("unknown_user", env.store.download, REPO, STRANGER)
    got = (env.spawn.repos, env.records(REPO), env.journal_for(REPO))
    assert got == ([], [], []), f"after unknown_user: expected (no processes, no records, no journal entries), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.1")
@pytest.mark.parametrize("user", USERS)
def test_download_allowed_user_recorded(models_env, user):
    """§5.1.1: будь-який логін з users.allowed ставить завантаження; запис несе цього user."""
    models_env.store.download(REPO, user)
    got = models_env.record(REPO).get("user")
    assert got == user, f"download by {user!r}: expected record user {user!r}, got {got!r}"


# --- §5.1.2 повтор активного -----------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §5.1.2")
def test_download_same_downloading_model_no_new_record(models_env):
    """§5.1.2: модель уже качається — повторний download не створює запису й процесу."""
    env = models_env
    env.store.download(REPO, "alice")
    env.clock.advance(10)
    env.store.download(REPO, "bob")
    got = (len(env.records(REPO)), len(env.spawn.for_repo(REPO)))
    assert got == (1, 1), f"second download of an active model: expected (1 record, 1 process), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.2")
def test_download_same_downloading_model_returns_current_state(models_env):
    """§5.1.2: повторний download повертає поточний стан — запис alice, downloading."""
    env = models_env
    env.store.download(REPO, "alice")
    env.clock.advance(10)
    state = env.store.download(REPO, "bob")
    got = (state.get("repo"), state.get("user"), state.get("status"))
    assert got == (REPO, "alice", "downloading"), f"repeated download: expected current state {(REPO, 'alice', 'downloading')!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.2")
def test_download_same_queued_model_no_new_record(make_models_env):
    """§5.1.2: queued — теж активний; повторний download не створює другого запису."""
    env = _one_slot(make_models_env)
    _queue(env, REPO, REPO_B)
    env.clock.advance(10)
    env.store.download(REPO_B, "bob")
    records = env.records(REPO_B)
    got = [(r.get("status"), r.get("user")) for r in records]
    assert got == [("queued", "alice")], f"second download of a queued model: expected [('queued', 'alice')], got {got!r}"


# --- §5.1.3 уже завантажена ------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §5.1.3")
def test_download_already_complete_is_done_at_once(models_env):
    """§5.1.3: усі вибрані файли вже є в знімку ревізії → одразу стан done."""
    env = models_env
    put_snapshot(env.hf_home, REPO, TINY_FILES)
    state = env.store.download(REPO, "alice")
    assert state.get("status") == "done", f"download of a complete snapshot: expected status 'done', got {state!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.3")
def test_download_already_complete_starts_no_process(models_env):
    """§5.1.3: для завантаженої моделі процес не запускається."""
    env = models_env
    put_snapshot(env.hf_home, REPO, TINY_FILES)
    env.store.download(REPO, "alice")
    env.store.poll()
    assert env.spawn.processes == [], f"complete snapshot: expected no process, got processes for {env.spawn.repos!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.3")
def test_download_already_complete_not_journaled(models_env):
    """§5.1.3: для завантаженої моделі в журнал не пишеться нічого."""
    env = models_env
    put_snapshot(env.hf_home, REPO, TINY_FILES)
    env.store.download(REPO, "alice")
    entries = env.journal_for(REPO)
    assert entries == [], f"complete snapshot: expected no journal entries for {REPO!r}, got {entries!r}"


# --- §5.1.4 gated ----------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §5.1.4")
def test_download_gated_without_access_refused(models_env):
    """§5.1.4: gated-модель без доступу → hf_gated."""
    expect_manager_error("hf_gated", models_env.store.download, GATED_REPO, "alice")


@pytest.mark.req("SPEC-GPU-002 §5.1.4")
def test_download_gated_without_access_not_queued(models_env):
    """§5.1.4: перевірка — до постановки в чергу: ні процесу, ні запису, ні журналу."""
    env = models_env
    expect_manager_error("hf_gated", env.store.download, GATED_REPO, "alice")
    got = (env.spawn.repos, env.records(GATED_REPO), env.journal_for(GATED_REPO))
    assert got == ([], [], []), f"after hf_gated: expected (no processes, no records, no journal entries), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.4")
def test_download_gated_with_access_accepted(models_env):
    """§5.1.4: gated-модель, до якої доступ є, качається як звичайна."""
    env = models_env
    env.store.download(GATED_OK_REPO, "alice")
    got = (env.status(GATED_OK_REPO), len(env.spawn.for_repo(GATED_OK_REPO)))
    assert got == ("downloading", 1), f"gated model with access: expected ('downloading', 1 process), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §4")
@pytest.mark.req("SPEC-GPU-002 §8")
def test_download_unknown_repo_refused(models_env):
    """§4, §8: моделі немає на HF — hf_not_found від hub доходить до виклику, процес не запускається."""
    env = models_env
    expect_manager_error("hf_not_found", env.store.download, MISSING_REPO, "alice")
    assert env.spawn.processes == [], f"after hf_not_found: expected no process, got {env.spawn.repos!r}"


# --- §5.1.5 диск ------------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §5.1.5")
def test_download_disk_reserve_exceeded_refused(make_models_env):
    """§5.1.5: вільно − потрібно < min_free_disk_gib (запас більший за диск) → disk_full."""
    env = make_models_env({"models.min_free_disk_gib": HUGE_RESERVE_GIB})
    expect_manager_error("disk_full", env.store.download, REPO, "alice")


@pytest.mark.req("SPEC-GPU-002 §5.1.5")
def test_download_disk_full_leaves_no_trace(make_models_env):
    """§5.1.5: відмова disk_full — без процесу, без запису черги й без журналу."""
    env = make_models_env({"models.min_free_disk_gib": HUGE_RESERVE_GIB})
    expect_manager_error("disk_full", env.store.download, REPO, "alice")
    got = (env.spawn.repos, env.records(REPO), env.journal_for(REPO))
    assert got == ([], [], []), f"after disk_full: expected (no processes, no records, no journal entries), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.5")
def test_download_needed_space_counts_missing_files(models_env):
    """§5.1.5: потрібно = сума файлів − наявне; 1 PiB, якого в кеші немає, не влазить → disk_full."""
    env = models_env
    env.hub.add(HUGE_REPO, HUGE_FILES)
    expect_manager_error("disk_full", env.store.download, HUGE_REPO, "alice")


@pytest.mark.req("SPEC-GPU-002 §5.1.5")
@pytest.mark.req("SPEC-GPU-002 §5.3")
def test_download_needed_space_excludes_present_files(models_env):
    """§5.1.5: файл на 1 PiB уже є в знімку — потрібно лише ≈ 1 MiB, завантаження стає в роботу.

    «Наявне» тут — за визначенням §5.3 (розміри з API файлів, що є в знімку): файл знімка — звичайний
    файл на 2 байти, як дозволяє §10.
    """
    env = models_env
    env.hub.add(HUGE_REPO, HUGE_FILES)
    put_snapshot(env.hf_home, HUGE_REPO, ["model-00001-of-00002.safetensors"])
    env.store.download(HUGE_REPO, "alice")
    got = env.status(HUGE_REPO)
    assert got == "downloading", f"1 PiB already present, 1 MiB missing: expected 'downloading', got {got!r}"


# --- §5.1.6 постановка в чергу ----------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §5.1.6")
def test_download_journal_entry(models_env):
    """§5.1.6: журнал download з repo, size_gib (≈ сума файлів) і user."""
    env = models_env
    env.store.download(REPO, "alice")
    entries = env.journal("download")
    got = [(e.get("repo"), e.get("user"), gib_close(e.get("size_gib"), TINY_TOTAL)) for e in entries]
    assert got == [(REPO, "alice", True)], (
        f"journal 'download': expected one entry ({REPO!r}, 'alice', size_gib≈{TINY_TOTAL / 2**30:.3f}), got {entries!r}"
    )


@pytest.mark.req("SPEC-GPU-002 §5.1.6")
def test_download_starts_process_at_once(models_env):
    """§5.1.6: одразу крок черги — вільний слот, процес запущено, стан downloading."""
    env = models_env
    env.store.download(REPO, "alice")
    got = (len(env.spawn.for_repo(REPO)), env.status(REPO))
    assert got == (1, "downloading"), f"right after download(): expected (1 process, 'downloading'), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.6")
def test_download_returns_state_after_queue_step(models_env):
    """§5.1.6: download() повертає стан після кроку черги — downloading, repo і user."""
    state = models_env.store.download(REPO, "alice")
    got = (state.get("repo"), state.get("user"), state.get("status"))
    assert got == (REPO, "alice", "downloading"), f"download() result: expected {(REPO, 'alice', 'downloading')!r}, got {state!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.11")
def test_download_record_has_all_fields(models_env):
    """§5.1.11: запис downloads() має всі поля {repo, revision, user, status, done_gib, total_gib, …}."""
    models_env.store.download(REPO, "alice")
    missing = RECORD_FIELDS - set(models_env.record(REPO))
    assert not missing, f"download record: missing fields {sorted(missing)}"


@pytest.mark.req("SPEC-GPU-002 §5.1.11")
@pytest.mark.req("SPEC-GPU-002 §9")
def test_download_record_started_from_clock(models_env):
    """§5.1.11, §9: started — час годинника в мить download(); finished активного — null."""
    models_env.store.download(REPO, "alice")
    record = models_env.record(REPO)
    got = (record.get("started"), record.get("finished"))
    assert got == (T0, None), f"active record: expected (started={T0}, finished=None), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.11")
def test_download_record_total_gib(models_env):
    """§5.1.11, §5.3: total_gib — сума розмірів вибраних файлів з API."""
    models_env.store.download(REPO, "alice")
    got = models_env.record(REPO).get("total_gib")
    assert gib_close(got, TINY_TOTAL), f"total_gib: expected ≈{TINY_TOTAL / 2**30:.3f} GiB, got {got!r}"


# --- §5.1.7 poll: завершення процесу ----------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §5.1.7")
def test_poll_exit_zero_marks_done(models_env):
    """§5.1.7: процес завершився з кодом 0 → done."""
    env = models_env
    env.store.download(REPO, "alice")
    env.spawn.last(REPO).finish(0)
    env.store.poll()
    assert env.status(REPO) == "done", f"exit code 0: expected status 'done', got {env.status(REPO)!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.7")
def test_poll_exit_zero_journals_download_done(models_env):
    """§5.1.7: done → журнал download_done."""
    env = models_env
    env.store.download(REPO, "alice")
    env.spawn.last(REPO).finish(0)
    env.store.poll()
    entries = env.journal("download_done")
    assert len(entries) == 1, f"journal 'download_done': expected 1 entry, got {entries!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.7")
@pytest.mark.req("SPEC-GPU-002 §5.1.11")
def test_poll_done_sets_finished_from_clock(models_env):
    """§5.1.7, §5.1.11: finished — час годинника в poll(), що побачив завершення; error_code — null."""
    env = models_env
    env.store.download(REPO, "alice")
    env.clock.advance(30)
    env.spawn.last(REPO).finish(0)
    env.store.poll()
    record = env.record(REPO)
    got = (record.get("finished"), record.get("error_code"))
    assert got == (T0 + 30, None), f"done record: expected (finished={T0 + 30}, error_code=None), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.7")
def test_poll_running_process_stays_downloading(models_env):
    """§5.1.7: процес ще працює — poll() не змінює стан і не запускає другого процесу."""
    env = models_env
    env.store.download(REPO, "alice")
    env.clock.advance(60)
    env.store.poll()
    got = (env.status(REPO), len(env.spawn.for_repo(REPO)))
    assert got == ("downloading", 1), f"running process after poll(): expected ('downloading', 1 process), got {got!r}"


# Останні рядки логу процесу (§5.2) і очікуваний error_code.
LOG_CASES: dict[str, tuple[list[str], str]] = {
    "gated": (["Fetching 9 files", "ERROR GatedRepoError: Access to model acme/tiny-llm is restricted"], "hf_gated"),
    "repo-not-found": (["ERROR RepositoryNotFoundError: 404 Client Error: Repository Not Found"], "hf_not_found"),
    "revision-not-found": (["ERROR RevisionNotFoundError: 404 Client Error: Revision Not Found"], "hf_not_found"),
    "no-space-last-line": (["ERROR OSError: [Errno 28] No space left on device"], "disk_full_during"),
    "no-space-earlier-line": (
        ["Traceback (most recent call last):", "OSError: [Errno 28] No space left on device", "ERROR RuntimeError: download aborted"],
        "disk_full_during",
    ),
    "other-error": (["ERROR ValueError: unexpected response from the hub"], "download_failed"),
    "no-error-line": (["Fetching 9 files", "Killed"], "download_failed"),
    "empty-log": ([], "download_failed"),
}


@pytest.mark.req("SPEC-GPU-002 §5.1.7")
@pytest.mark.req("SPEC-GPU-002 §5.2")
@pytest.mark.parametrize("case", sorted(LOG_CASES))
def test_poll_failure_error_code_from_log(models_env, case):
    """§5.1.7, §5.2: код ≠ 0 → failed; error_code — за класом винятку в рядку ERROR або «No space left» у лозі."""
    env = models_env
    lines, code = LOG_CASES[case]
    env.store.download(REPO, "alice")
    _fail(env, REPO, lines)
    record = env.record(REPO)
    got = (record.get("status"), record.get("error_code"))
    assert got == ("failed", code), f"{case}: expected ('failed', {code!r}), got {got!r}; log lines {lines!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.7")
def test_poll_failure_has_error_text(models_env):
    """§5.1.7: failed — поле error з непорожнім текстом."""
    env = models_env
    env.store.download(REPO, "alice")
    _fail(env, REPO, ["ERROR ValueError: unexpected response from the hub"])
    error = env.record(REPO).get("error")
    assert isinstance(error, str) and error.strip(), f"failed record: expected a non-empty error text, got {error!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.7")
def test_poll_failure_journals_download_failed_with_error(models_env):
    """§5.1.7: failed → журнал download_failed з полем error."""
    env = models_env
    env.store.download(REPO, "alice")
    _fail(env, REPO, ["ERROR ValueError: unexpected response from the hub"])
    entries = env.journal("download_failed")
    ok = len(entries) == 1 and isinstance(entries[0].get("error"), str) and entries[0]["error"].strip()
    assert ok, f"journal 'download_failed': expected 1 entry with a non-empty error, got {entries!r}"


# --- §5.1.7 poll: черга --------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §5.1.7")
def test_queue_runs_at_most_max_parallel(models_env):
    """§5.1.7: max_parallel_downloads = 2 — із трьох завантажень працюють два, третє queued."""
    env = models_env
    _queue(env, REPO, REPO_B, REPO_C)
    got = (env.spawn.repos, env.status(REPO_C))
    assert got == ([REPO, REPO_B], "queued"), f"three downloads, 2 slots: expected ([{REPO!r}, {REPO_B!r}], 'queued'), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.7")
def test_queue_single_slot_holds_second(make_models_env):
    """§5.1.7: max_parallel_downloads = 1 — друге завантаження чекає в queued."""
    env = _one_slot(make_models_env)
    _queue(env, REPO, REPO_B)
    got = (env.spawn.repos, env.status(REPO_B))
    assert got == ([REPO], "queued"), f"two downloads, 1 slot: expected ([{REPO!r}], 'queued'), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.7")
def test_queue_starts_queued_when_slot_frees(models_env):
    """§5.1.7: завантаження завершилось — poll() запускає queued."""
    env = models_env
    _queue(env, REPO, REPO_B, REPO_C)
    env.spawn.last(REPO).finish(0)
    env.store.poll()
    got = (env.status(REPO_C), len(env.spawn.for_repo(REPO_C)))
    assert got == ("downloading", 1), f"slot freed: expected {REPO_C!r} ('downloading', 1 process), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.7")
def test_queue_starts_in_started_order(make_models_env):
    """§5.1.7: queued запускаються в порядку started — спершу раніший."""
    env = _one_slot(make_models_env)
    _queue(env, REPO, REPO_B, REPO_C)
    env.spawn.last(REPO).finish(0)
    env.store.poll()
    got = (env.spawn.repos, env.status(REPO_C))
    assert got == ([REPO, REPO_B], "queued"), f"1 slot, A done: expected B started, C queued; got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.7")
def test_queue_failed_download_frees_slot(make_models_env):
    """§5.1.7: невдале завантаження теж звільняє слот — queued запускається."""
    env = _one_slot(make_models_env)
    _queue(env, REPO, REPO_B)
    _fail(env, REPO, ["ERROR ValueError: unexpected response from the hub"])
    got = (env.status(REPO_B), len(env.spawn.for_repo(REPO_B)))
    assert got == ("downloading", 1), f"A failed, 1 slot: expected B ('downloading', 1 process), got {got!r}"


# --- §5.1.8 скасування ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §5.1.8")
@pytest.mark.req("SPEC-GPU-002 §9")
def test_cancel_unknown_user_refused(models_env):
    """§9: cancel перевіряє user — логін не з users.allowed → unknown_user."""
    env = models_env
    env.store.download(REPO, "alice")
    expect_manager_error("unknown_user", env.store.cancel, REPO, STRANGER)


@pytest.mark.req("SPEC-GPU-002 §5.1.8")
@pytest.mark.req("SPEC-GPU-002 §9")
def test_cancel_unknown_user_keeps_download(models_env):
    """§9: після відмови unknown_user завантаження триває — процес не зупинено, стан downloading."""
    env = models_env
    env.store.download(REPO, "alice")
    proc = env.spawn.last(REPO)
    expect_manager_error("unknown_user", env.store.cancel, REPO, STRANGER)
    got = (env.status(REPO), proc.stopped)
    assert got == ("downloading", False), f"after cancel by a stranger: expected ('downloading', process not stopped), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.8")
def test_cancel_unknown_download_refused(models_env):
    """§5.1.8: скасування того, чого немає в черзі, → download_not_active."""
    expect_manager_error("download_not_active", models_env.store.cancel, REPO, "alice")


@pytest.mark.req("SPEC-GPU-002 §5.1.8")
def test_cancel_done_download_refused(models_env):
    """§5.1.8: done — не активний → download_not_active; стан лишається done."""
    env = models_env
    env.store.download(REPO, "alice")
    env.spawn.last(REPO).finish(0)
    env.store.poll()
    expect_manager_error("download_not_active", env.store.cancel, REPO, "alice")
    assert env.status(REPO) == "done", f"after refused cancel: expected status 'done', got {env.status(REPO)!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.8")
def test_cancel_downloading_marks_cancelled(models_env):
    """§5.1.8: скасування активного → стан cancelled (у відповіді й у downloads())."""
    env = models_env
    env.store.download(REPO, "alice")
    state = env.store.cancel(REPO, "alice")
    got = (state.get("status"), env.status(REPO))
    assert got == ("cancelled", "cancelled"), f"cancel: expected ('cancelled' returned, 'cancelled' listed), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.8")
def test_cancel_sends_sigterm_first(models_env):
    """§5.1.8: процес зупиняється SIGTERM (terminate) — першим сигналом."""
    env = models_env
    env.store.download(REPO, "alice")
    proc = env.spawn.last(REPO)
    env.store.cancel(REPO, "alice")
    signals = [e for e in proc.events if e in ("terminate", "kill")]
    assert signals[:1] == ["terminate"], f"cancel: expected SIGTERM (terminate) first, got signal sequence {signals!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.8")
def test_cancel_cooperative_process_not_killed(models_env):
    """§5.1.8: процес вийшов на SIGTERM — SIGKILL не надсилається й після 5 с."""
    env = models_env
    env.store.download(REPO, "alice")
    proc = env.spawn.last(REPO)
    env.store.cancel(REPO, "alice")
    env.clock.advance(GRACE_S + 1)
    env.store.poll()
    assert proc.kill_calls == 0, f"process exited on SIGTERM: expected no SIGKILL, got events {proc.events!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.8")
def test_cancel_stubborn_process_killed(models_env):
    """§5.1.8: процес ігнорує SIGTERM — через 5 с отримує SIGKILL (kill)."""
    env = models_env
    env.spawn.stubborn = True
    env.store.download(REPO, "alice")
    proc = env.spawn.last(REPO)
    env.store.cancel(REPO, "alice")
    env.clock.advance(GRACE_S + 1)
    env.store.poll()
    assert proc.kill_calls >= 1, f"process ignoring SIGTERM, {GRACE_S + 1} s later: expected SIGKILL, got events {proc.events!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.8")
def test_cancel_no_sigkill_before_grace(models_env):
    """§5.1.8: SIGKILL не раніше ніж через 5 с після SIGTERM.

    Два допустимі втілення: cancel() сам чекає wait(timeout=5) і тоді вбиває — тоді має бути wait з
    timeout 5; або вбиває пізніший poll() за годинником — тоді за 4 с kill ще немає.
    """
    env = models_env
    env.spawn.stubborn = True
    env.store.download(REPO, "alice")
    proc = env.spawn.last(REPO)
    env.store.cancel(REPO, "alice")
    if proc.kill_calls:
        waited = [t for t in proc.wait_timeouts if isinstance(t, (int, float)) and abs(t - GRACE_S) < 1e-6]
        assert waited, f"SIGKILL sent inside cancel(): expected a wait(timeout={GRACE_S}) before it, got events {proc.events!r}, timeouts {proc.wait_timeouts!r}"
    else:
        env.clock.advance(GRACE_S - 1)
        env.store.poll()
        assert proc.kill_calls == 0, f"{GRACE_S - 1} s after SIGTERM: expected no SIGKILL yet, got events {proc.events!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.8")
@pytest.mark.req("SPEC-GPU-002 §9")
def test_cancel_journals_download_cancel(models_env):
    """§5.1.8, §9: будь-який дозволений користувач скасовує чуже завантаження; журнал download_cancel від нього."""
    env = models_env
    env.store.download(REPO, "alice")
    env.store.cancel(REPO, "bob")
    entries = env.journal("download_cancel")
    got = [e.get("user") for e in entries]
    assert got == ["bob"], f"journal 'download_cancel': expected one entry by 'bob', got {entries!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.8")
def test_cancel_keeps_incomplete_files(models_env):
    """§5.1.8: недокачані файли після скасування лишаються."""
    env = models_env
    env.store.download(REPO, "alice")
    part = put_incomplete(env.hf_home, REPO, 64 * MIB)
    env.store.cancel(REPO, "alice")
    assert part.exists(), f"after cancel: expected the incomplete blob {part} to stay"


@pytest.mark.req("SPEC-GPU-002 §5.1.8")
def test_cancel_queued_never_starts(make_models_env):
    """§5.1.8: скасований queued не запускається й тоді, коли звільниться слот."""
    env = _one_slot(make_models_env)
    _queue(env, REPO, REPO_B)
    env.store.cancel(REPO_B, "alice")
    env.spawn.last(REPO).finish(0)
    env.store.poll()
    got = (env.status(REPO_B), env.spawn.for_repo(REPO_B))
    assert got == ("cancelled", []), f"cancelled queued download after a slot freed: expected ('cancelled', no process), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.8")
def test_cancelled_stays_cancelled_after_poll(models_env):
    """§5.1.8: процес, зупинений скасуванням, poll() не перетворює на failed."""
    env = models_env
    env.store.download(REPO, "alice")
    env.store.cancel(REPO, "alice")
    env.store.poll()
    got = (env.status(REPO), env.journal("download_failed"))
    assert got == ("cancelled", []), f"poll() after cancel: expected ('cancelled', no download_failed entries), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.8")
@pytest.mark.req("SPEC-GPU-002 §5.1.7")
def test_cancel_frees_slot_for_queued(make_models_env):
    """§5.1.7–5.1.8: скасований downloading звільняє слот — наступний poll() запускає queued."""
    env = _one_slot(make_models_env)
    _queue(env, REPO, REPO_B)
    env.store.cancel(REPO, "alice")
    env.store.poll()
    got = (env.status(REPO_B), len(env.spawn.for_repo(REPO_B)))
    assert got == ("downloading", 1), f"A cancelled, 1 slot: expected B ('downloading', 1 process), got {got!r}"


# --- §5.1.9 збереження й продовження ---------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §5.1.9")
def test_downloads_saved_in_state_json(models_env):
    """§5.1.9: стан черги — у data_dir/state.json, розділ downloads."""
    env = models_env
    env.store.download(REPO, "alice")
    section = env.state_downloads()
    assert REPO in json.dumps(section), f"state.json 'downloads': expected it to mention {REPO!r}, got {section!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.9")
def test_restart_resumes_interrupted_download(models_env):
    """§5.1.9: після перезапуску downloading продовжується — новий процес, стан downloading."""
    env = models_env
    env.store.download(REPO, "alice")
    restarted = env.restart()
    restarted.store.poll()
    got = (restarted.status(REPO), restarted.spawn.repos)
    assert got == ("downloading", [REPO]), f"after restart: expected ('downloading', a new process for {REPO!r}), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.9")
def test_restart_resume_respects_order_and_limit(make_models_env):
    """§5.1.9, §5.1.7: після перезапуску черга йде в порядку started і в межах max_parallel_downloads."""
    env = _one_slot(make_models_env)
    _queue(env, REPO, REPO_B)
    restarted = env.restart()
    restarted.store.poll()
    got = (restarted.spawn.repos, restarted.status(REPO_B))
    assert got == ([REPO], "queued"), f"after restart, 1 slot: expected ([{REPO!r}], B 'queued'), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.9")
def test_restart_keeps_finished_records(models_env):
    """§5.1.9: завершені записи переживають перезапуск (done лишається done, без нового процесу)."""
    env = models_env
    env.store.download(REPO, "alice")
    env.spawn.last(REPO).finish(0)
    env.store.poll()
    restarted = env.restart()
    restarted.store.poll()
    got = (restarted.status(REPO), restarted.spawn.repos)
    assert got == ("done", []), f"done record after restart: expected ('done', no process), got {got!r}"


# --- §5.1.10 stop_all -------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §5.1.10")
def test_stop_all_stops_every_process(models_env):
    """§5.1.10: stop_all() зупиняє всі процеси завантаження."""
    env = models_env
    _queue(env, REPO, REPO_B)
    env.store.stop_all()
    running = [p.repo for p in env.spawn.processes if not p.stopped]
    assert running == [], f"after stop_all(): expected every process stopped, still running {running!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.10")
def test_stop_all_keeps_saved_state(models_env):
    """§5.1.10: stop_all() не змінює збереженого стану (розділ downloads у state.json)."""
    env = models_env
    _queue(env, REPO, REPO_B)
    before = env.state_downloads()
    env.store.stop_all()
    after = env.state_downloads()
    assert after == before, f"state.json 'downloads' changed by stop_all(): before {before!r}, after {after!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.10")
@pytest.mark.req("SPEC-GPU-002 §5.1.9")
def test_stop_all_then_restart_resumes(models_env):
    """§5.1.10: зупинені stop_all() завантаження при старті продовжуються."""
    env = models_env
    env.store.download(REPO, "alice")
    env.store.stop_all()
    restarted = env.restart()
    restarted.store.poll()
    got = (restarted.status(REPO), restarted.spawn.repos)
    assert got == ("downloading", [REPO]), f"after stop_all() and restart: expected ('downloading', [{REPO!r}]), got {got!r}"


# --- §5.1.11 downloads() ------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §5.1.11")
def test_downloads_newest_first(models_env):
    """§5.1.11: downloads() — найновіші (started) першими."""
    env = models_env
    _queue(env, REPO, REPO_B, REPO_C)
    got = [r.get("repo") for r in env.store.downloads()]
    assert got == [REPO_C, REPO_B, REPO], f"downloads(): expected newest first {[REPO_C, REPO_B, REPO]!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.11")
def test_downloads_lists_every_state(models_env):
    """§5.1.11: downloads() — усі записи: done, failed і cancelled теж."""
    env = models_env
    _queue(env, REPO, REPO_B)
    env.spawn.last(REPO).finish(0)
    _fail(env, REPO_B, ["ERROR ValueError: unexpected response from the hub"])
    env.clock.advance(10)
    env.store.download(REPO_C, "alice")
    env.store.cancel(REPO_C, "alice")
    got = {r.get("repo"): r.get("status") for r in env.store.downloads()}
    expected = {REPO: "done", REPO_B: "failed", REPO_C: "cancelled"}
    assert got == expected, f"downloads(): expected {expected!r}, got {got!r}"


# --- §5.2 процес завантаження ------------------------------------------------------------------------------------------------------------------------


def _started(env: Any, repo: str = REPO) -> Any:
    env.store.download(repo, "alice")
    return env.spawn.last(repo)


def _stream_target(stream: Any) -> str:
    """Куди веде stdout/stderr процесу: шлях файла, «STDOUT» (злиття) або опис."""
    if stream is subprocess.STDOUT:
        return "STDOUT"
    if isinstance(stream, int):
        try:
            return str(Path(os.readlink(f"/proc/self/fd/{stream}")).resolve())
        except OSError:
            return f"<fd {stream}>"
    name = getattr(stream, "name", None)
    if isinstance(name, (str, os.PathLike)):
        return str(Path(name).resolve())
    return repr(stream)


@pytest.mark.req("SPEC-GPU-002 §5.2")
def test_worker_command_line(models_env):
    """§5.2: spawn([python, "-m", "gpu_manager.download_worker"], …)."""
    argv = [str(a) for a in _started(models_env).argv]
    ok = argv[1:] == ["-m", "gpu_manager.download_worker"] and Path(argv[0]).name.startswith("python")
    assert ok, f"worker argv: expected [<python>, '-m', 'gpu_manager.download_worker'], got {argv!r}"


@pytest.mark.req("SPEC-GPU-002 §5.2")
def test_worker_stdin_is_pipe(models_env):
    """§5.2: stdin=PIPE."""
    stdin = _started(models_env).kwargs.get("stdin")
    assert stdin == subprocess.PIPE, f"spawn stdin: expected subprocess.PIPE, got {stdin!r}"


@pytest.mark.req("SPEC-GPU-002 §5.2")
@pytest.mark.req("SPEC-GPU-002 §9")
def test_worker_stdin_json_payload(models_env):
    """§5.2, §9: у stdin — JSON {repo, revision = sha з hub.files, files = усі вибрані файли}."""
    proc = _started(models_env)
    payload = proc.stdin.payload()
    ok = (
        isinstance(payload, dict)
        and payload.get("repo") == REPO
        and payload.get("revision") == sha_of(REPO)
        and isinstance(payload.get("files"), list)
        and sorted(payload["files"]) == sorted(TINY_FILES)
    )
    assert ok, f"worker stdin: expected {{'repo': {REPO!r}, 'revision': {sha_of(REPO)!r}, 'files': {sorted(TINY_FILES)!r}}}, got {proc.stdin.text[:400]!r}"


@pytest.mark.req("SPEC-GPU-002 §5.2")
@pytest.mark.req("SPEC-GPU-002 §9")
def test_worker_stdin_revision_is_sha_for_named_revision(models_env):
    """§9: запитана ревізія "main" — у stdin іде sha, який повернув hub.files, а не назва."""
    models_env.store.download(REPO, "alice", revision="main")
    payload = models_env.spawn.last(REPO).stdin.payload()
    got = payload.get("revision") if isinstance(payload, dict) else payload
    assert got == sha_of(REPO), f"worker stdin revision for revision='main': expected sha {sha_of(REPO)!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.2")
@pytest.mark.req("SPEC-GPU-002 §9")
def test_worker_stdin_files_include_present_ones(models_env):
    """§9: files — усі вибрані файли, навіть якщо частина вже є в знімку ревізії."""
    env = models_env
    put_snapshot(env.hf_home, REPO, ["config.json", "model-00002-of-00002.safetensors"])
    env.store.download(REPO, "alice")
    payload = env.spawn.last(REPO).stdin.payload()
    got = sorted(payload.get("files") or []) if isinstance(payload, dict) else payload
    assert got == sorted(TINY_FILES), f"worker stdin files with 2 files present: expected all {sorted(TINY_FILES)!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.2")
def test_worker_stdin_closed(models_env):
    """§5.2: після запису JSON stdin закривається."""
    proc = _started(models_env)
    assert proc.stdin.closed, "worker stdin: expected closed after writing the JSON payload"


@pytest.mark.req("SPEC-GPU-002 §5.2")
def test_worker_env_hf_home(models_env):
    """§5.2: оточення процесу — HF_HOME=<hf_home>."""
    env_vars = _started(models_env).kwargs.get("env")
    got = env_vars.get("HF_HOME") if isinstance(env_vars, dict) else None
    ok = isinstance(got, str) and Path(got).resolve() == models_env.hf_home.resolve()
    assert ok, f"worker env HF_HOME: expected {models_env.hf_home}, got {got!r} (env passed: {type(env_vars).__name__})"


@pytest.mark.req("SPEC-GPU-002 §5.2")
def test_worker_env_token_when_present(make_models_env):
    """§5.2: токен є — HF_TOKEN у оточенні процесу дорівнює йому."""
    env = make_models_env(token=TOKEN)
    env_vars = _started(env).kwargs.get("env")
    got = env_vars.get("HF_TOKEN") if isinstance(env_vars, dict) else None
    assert got == TOKEN, f"worker env HF_TOKEN with a token: expected {TOKEN!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.2")
def test_worker_env_no_token_when_absent(models_env):
    """§5.2: токена немає — HF_TOKEN в оточенні процесу відсутній."""
    env_vars = _started(models_env).kwargs.get("env")
    ok = isinstance(env_vars, dict) and "HF_TOKEN" not in env_vars
    assert ok, f"worker env without a token: expected a dict without HF_TOKEN, got {env_vars if not isinstance(env_vars, dict) else sorted(env_vars)!r}"


@pytest.mark.req("SPEC-GPU-002 §5.2")
def test_worker_env_parent_token_not_passed(models_env, monkeypatch):
    """§5.2: HF_TOKEN — лише якщо токен є; токен з оточення сервісу не протікає процесу, коли функція токена дає None."""
    monkeypatch.setenv("HF_TOKEN", "env-token-not-real")
    env_vars = _started(models_env).kwargs.get("env")
    ok = isinstance(env_vars, dict) and "HF_TOKEN" not in env_vars
    assert ok, f"worker env with HF_TOKEN in the service environment and no token: expected no HF_TOKEN, got {env_vars.get('HF_TOKEN') if isinstance(env_vars, dict) else env_vars!r}"


@pytest.mark.req("SPEC-GPU-002 §5.2")
def test_worker_log_file_created(models_env):
    """§5.2: лог — <data_dir>/downloads/models--<org>--<name>.log."""
    _started(models_env)
    path = log_path(models_env.data_dir, REPO)
    assert path.is_file(), f"worker log: expected {path} to exist after the process was started"


@pytest.mark.req("SPEC-GPU-002 §5.2")
def test_worker_output_goes_to_log(models_env):
    """§5.2: stdout і stderr процесу — у лог-файл моделі."""
    kwargs = _started(models_env).kwargs
    expected = str(log_path(models_env.data_dir, REPO).resolve())
    out, err = _stream_target(kwargs.get("stdout")), _stream_target(kwargs.get("stderr"))
    ok = out == expected and err in (expected, "STDOUT")
    assert ok, f"worker stdout/stderr: expected both into {expected}, got stdout -> {out}, stderr -> {err}"


@pytest.mark.req("SPEC-GPU-002 §5.2")
def test_worker_log_per_model(models_env):
    """§5.2: у кожної моделі свій лог (ім'я з org і name)."""
    env = models_env
    _queue(env, REPO, REPO_B)
    paths = [log_path(env.data_dir, r) for r in (REPO, REPO_B)]
    missing = [str(p) for p in paths if not p.is_file()]
    assert not missing, f"per-model worker logs: expected {[str(p) for p in paths]}, missing {missing}"
