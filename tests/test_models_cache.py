"""§5.3–5.4.2 SPEC-GPU-002: прогрес завантаження за кешем HF, список завантажених моделей, видалення.

Кеш HF — тека в tmp_path з розкладкою §5.3: snapshots/<sha>/<файл> (готові файли) і blobs/*.incomplete
(недокачані). Файли знімка — звичайні файли (§10); де тест перевіряє розмір на диску, файли розріджені
(truncate): розмір великий, місця на диску вони не займають. Процес завантаження — підробка, він сам
нічого в кеш не пише: «хід завантаження» тест моделює, створюючи файли.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from .conftest import STRANGER, T0
from .model_fakes import (
    GIB,
    GIB_TOLERANCE,
    MIB,
    REPO,
    REPO_B,
    TINY_FILES,
    TINY_TOTAL,
    append_log,
    expect_manager_error,
    gib_close,
    log_path,
    model_dir,
    put_incomplete,
    put_snapshot,
    sha_of,
)

pytestmark = pytest.mark.component

LOCAL_FIELDS = {"repo", "size_gib", "revisions", "revision", "state", "last_modified"}
SHA_NEW = sha_of(REPO + "@new")  # новіша ревізія REPO, яку підроблений HF віддає після першого завантаження
NEWER_OUTCOMES = ["downloading", "failed", "cancelled"]  # чим скінчилось (ще не скінчилось) завантаження SHA_NEW
# Локальна модель для видалення: дві ваги по 768 і 256 MiB (розріджені) + дрібні файли ≈ 1 GiB.
LOCAL_SIZES = {"model-00001-of-00002.safetensors": 768 * MIB, "model-00002-of-00002.safetensors": 256 * MIB}
LOCAL_NAMES = ["config.json", *LOCAL_SIZES]
LOCAL_BYTES = sum(LOCAL_SIZES.values())  # рівно 1 GiB; дрібні файли по 2 байти в межах допуску


def _progress(env: Any, repo: str = REPO) -> dict[str, Any]:
    """Запис repo після poll() (прогрес міг оновлюватись лише в poll())."""
    env.store.poll()
    return env.record(repo)


def _local(env: Any) -> dict[str, dict[str, Any]]:
    """local() як словник repo → запис."""
    return {m.get("repo"): m for m in env.store.local()}


def _put_local_model(env: Any, repo: str = REPO) -> None:
    """Локальна модель ≈ 1 GiB з двома ревізіями (друга — без refs) і блобом."""
    put_snapshot(env.hf_home, repo, LOCAL_NAMES, sizes=LOCAL_SIZES)
    put_snapshot(env.hf_home, repo, ["config.json"], sha="e" * 40, ref=False)
    (model_dir(env.hf_home, repo) / "blobs" / ("a" * 64)).write_bytes(b"{}")


# --- §5.3 прогрес -----------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §5.3")
def test_progress_zero_before_any_file(models_env):
    """§5.3: у кеші ще нічого — done 0, percent 0."""
    env = models_env
    env.store.download(REPO, "alice")
    record = _progress(env)
    got = (record.get("done_gib"), record.get("percent"))
    assert got == (0, 0), f"nothing downloaded yet: expected (done_gib 0, percent 0), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.3")
def test_progress_counts_snapshot_files_by_api_size(models_env):
    """§5.3: готовий файл знімка рахується розміром з API (128 MiB), а не розміром на диску (2 байти)."""
    env = models_env
    env.store.download(REPO, "alice")
    put_snapshot(env.hf_home, REPO, ["model-00002-of-00002.safetensors"])
    got = _progress(env).get("done_gib")
    assert gib_close(got, 128 * MIB), f"done_gib with one 128 MiB file in the snapshot: expected ≈0.125 GiB, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.3")
def test_progress_counts_incomplete_blobs(models_env):
    """§5.3: blobs/*.incomplete додаються своїм розміром."""
    env = models_env
    env.store.download(REPO, "alice")
    put_incomplete(env.hf_home, REPO, 200 * MIB)
    got = _progress(env).get("done_gib")
    assert gib_close(got, 200 * MIB), f"done_gib with a 200 MiB incomplete blob: expected ≈0.195 GiB, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.3")
def test_progress_percent_one_decimal(models_env):
    """§5.3: percent = 100·done/total, 1 знак; done = 128 MiB у знімку + 90 MiB недокачаних."""
    env = models_env
    env.store.download(REPO, "alice")
    put_snapshot(env.hf_home, REPO, ["model-00002-of-00002.safetensors"])
    put_incomplete(env.hf_home, REPO, 90 * MIB)
    expected = round(100 * (128 * MIB + 90 * MIB) / TINY_TOTAL, 1)  # 42.4
    got = _progress(env).get("percent")
    assert got == expected, f"percent for 218 MiB of {TINY_TOTAL} bytes: expected {expected}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.3")
@pytest.mark.req("SPEC-GPU-002 §9")
def test_progress_done_gib_two_decimals(models_env):
    """§9: *_gib — 2 знаки: done = 218 MiB = 0.212890625 GiB → 0.21 (не 0.2 і не сире значення)."""
    env = models_env
    env.store.download(REPO, "alice")
    put_snapshot(env.hf_home, REPO, ["model-00002-of-00002.safetensors"])
    put_incomplete(env.hf_home, REPO, 90 * MIB)
    got = _progress(env).get("done_gib")
    assert got == 0.21, f"done_gib for 218 MiB: expected 0.21 (2 decimals), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.3")
def test_progress_capped_at_total(models_env):
    """§5.3: done не більше загального — недокачаних 2 GiB на модель ≈ 0.5 GiB дають 100 %, не більше."""
    env = models_env
    env.store.download(REPO, "alice")
    put_incomplete(env.hf_home, REPO, 2 * GIB)
    record = _progress(env)
    got = (record.get("percent"), gib_close(record.get("done_gib"), TINY_TOTAL))
    assert got == (100, True), f"incomplete blobs above total: expected (percent 100, done_gib ≈ total), got {record!r}"


@pytest.mark.req("SPEC-GPU-002 §5.3")
def test_progress_ignores_other_revision(models_env):
    """§5.3: рахуються файли знімка саме цієї ревізії; файл у знімку іншого sha — ні."""
    env = models_env
    env.store.download(REPO, "alice")
    put_snapshot(env.hf_home, REPO, ["model-00001-of-00002.safetensors"], sha="f" * 40, ref=False)
    got = _progress(env).get("done_gib")
    ok = isinstance(got, (int, float)) and got < GIB_TOLERANCE
    assert ok, f"file only in another revision's snapshot: expected done_gib ≈ 0, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.3")
def test_progress_done_state_is_100(models_env):
    """§5.3: у стані done — 100, навіть якщо файлів у кеші тест не створював."""
    env = models_env
    env.store.download(REPO, "alice")
    env.spawn.last(REPO).finish(0)
    got = _progress(env).get("percent")
    assert got == 100, f"percent in state done: expected 100, got {got!r}"


# --- §5.4.1 local() -------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §5.4.1")
def test_local_without_hub_dir_is_empty(models_env):
    """§5.4.1: немає теки <hf_home>/hub — []."""
    (models_env.hf_home / "hub").rmdir()
    got = models_env.store.local()
    assert got == [], f"local() without the hub directory: expected [], got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.1")
def test_local_lists_model_by_repo_name(models_env):
    """§5.4.1: тека models--acme--tiny-llm → repo "acme/tiny-llm"."""
    put_snapshot(models_env.hf_home, REPO, ["config.json"])
    got = [m.get("repo") for m in models_env.store.local()]
    assert got == [REPO], f"local(): expected [{REPO!r}], got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.1")
def test_local_skips_datasets(models_env):
    """§5.4.1: лише моделі, не датасети (datasets--… не показується)."""
    env = models_env
    put_snapshot(env.hf_home, REPO, ["config.json"])
    dataset = env.hf_home / "hub" / "datasets--acme--corpus" / "snapshots" / ("d" * 40)
    dataset.mkdir(parents=True)
    (dataset / "data.parquet").write_bytes(b"PAR1")
    got = [m.get("repo") for m in env.store.local()]
    assert got == [REPO], f"local() with a dataset in the cache: expected only [{REPO!r}], got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.1")
def test_local_ignores_cache_service_entries(models_env):
    """§5.4.1: службові записи кешу HF (.locks, version.txt) — не моделі."""
    env = models_env
    put_snapshot(env.hf_home, REPO, ["config.json"])
    (env.hf_home / "hub" / ".locks" / "models--acme--ghost").mkdir(parents=True)
    (env.hf_home / "hub" / "version.txt").write_text("1", encoding="utf-8")
    got = [m.get("repo") for m in env.store.local()]
    assert got == [REPO], f"local() with .locks and version.txt: expected only [{REPO!r}], got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.1")
def test_local_entry_fields(models_env):
    """§5.4.1: запис — {repo, size_gib, revisions, revision, state, last_modified}."""
    put_snapshot(models_env.hf_home, REPO, ["config.json"])
    entry = _local(models_env).get(REPO) or {}
    missing = LOCAL_FIELDS - set(entry)
    assert not missing, f"local() entry: missing fields {sorted(missing)}, got {entry!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.1")
def test_local_size_gib_from_disk(models_env):
    """§5.4.1: size_gib — розмір моделі на диску (≈ 1 GiB розріджених файлів знімка)."""
    put_snapshot(models_env.hf_home, REPO, LOCAL_NAMES, sizes=LOCAL_SIZES)
    got = (_local(models_env).get(REPO) or {}).get("size_gib")
    assert gib_close(got, LOCAL_BYTES), f"local() size_gib: expected ≈1.0 GiB, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.1")
def test_local_state_ready_without_record(models_env):
    """§5.4.1: запису черги немає — ready."""
    put_snapshot(models_env.hf_home, REPO, ["config.json"])
    got = (_local(models_env).get(REPO) or {}).get("state")
    assert got == "ready", f"local model without a download record: expected state 'ready', got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.1")
def test_local_state_ready_after_done(models_env):
    """§5.4.1: запис done — ready."""
    env = models_env
    env.store.download(REPO, "alice")
    put_snapshot(env.hf_home, REPO, TINY_FILES)
    env.spawn.last(REPO).finish(0)
    env.store.poll()
    got = (_local(env).get(REPO) or {}).get("state")
    assert got == "ready", f"local model with a done record: expected state 'ready', got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.1")
def test_local_state_downloading_while_downloading(models_env):
    """§5.4.1: активне завантаження (downloading) — state downloading."""
    env = models_env
    env.store.download(REPO, "alice")
    put_snapshot(env.hf_home, REPO, ["config.json"])
    put_incomplete(env.hf_home, REPO, 10 * MIB)
    got = (_local(env).get(REPO) or {}).get("state")
    assert got == "downloading", f"model being downloaded: expected state 'downloading', got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.1")
def test_local_state_downloading_while_queued(make_models_env):
    """§5.4.1, §5.1: queued — теж активний → state downloading."""
    env = make_models_env({"models.max_parallel_downloads": 1})
    env.store.download(REPO, "alice")
    env.clock.advance(10)
    env.store.download(REPO_B, "alice")
    put_snapshot(env.hf_home, REPO_B, ["config.json"])
    got = (env.status(REPO_B), (_local(env).get(REPO_B) or {}).get("state"))
    assert got == ("queued", "downloading"), f"queued model with files on disk: expected ('queued', state 'downloading'), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.1")
def test_local_state_partial_after_failure(models_env):
    """§5.4.1: запис failed — partial."""
    env = models_env
    env.store.download(REPO, "alice")
    put_snapshot(env.hf_home, REPO, ["config.json"])
    append_log(log_path(env.data_dir, REPO), ["ERROR ValueError: unexpected response from the hub"])
    env.spawn.last(REPO).finish(1)
    env.store.poll()
    got = (_local(env).get(REPO) or {}).get("state")
    assert got == "partial", f"model with a failed download: expected state 'partial', got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.1")
def test_local_state_partial_after_cancel(models_env):
    """§5.4.1: запис cancelled — partial."""
    env = models_env
    env.store.download(REPO, "alice")
    put_snapshot(env.hf_home, REPO, ["config.json"])
    env.store.cancel(REPO, "alice")
    got = (_local(env).get(REPO) or {}).get("state")
    assert got == "partial", f"model with a cancelled download: expected state 'partial', got {got!r}"


def _set_mtime(snapshot: Path, ts: float) -> None:
    """mtime теки знімка й усіх її файлів = ts: «найновіша» ревізія не залежить від порядку створення."""
    for item in [snapshot, *snapshot.rglob("*")]:
        os.utime(item, (ts, ts))


def _completed_then_newer(env: Any, outcome: str) -> None:
    """Менеджер докачав ревізію sha_of(REPO) (done); далі HF віддає SHA_NEW, і її завантаження — outcome.

    Знімок SHA_NEW частковий (config.json і недокачаний блоб), refs/main вказує на нього і він новіший за
    mtime: лише правило «ревізія, яку докачав менеджер» дає sha_of(REPO).
    """
    env.store.download(REPO, "alice")
    old = put_snapshot(env.hf_home, REPO, TINY_FILES)
    env.spawn.last(REPO).finish(0)
    env.store.poll()
    assert env.status(REPO) == "done", f"precondition: first download of {REPO!r} expected 'done', got {env.records(REPO)!r}"
    env.hub.repos[REPO].sha = SHA_NEW
    env.clock.advance(60)
    env.store.download(REPO, "alice")
    new = put_snapshot(env.hf_home, REPO, ["config.json"], sha=SHA_NEW)
    put_incomplete(env.hf_home, REPO, 64 * MIB, tag="new")
    _set_mtime(old, T0 - 7200)
    _set_mtime(new, T0 - 60)
    if outcome == "failed":
        append_log(log_path(env.data_dir, REPO), ["ERROR ValueError: unexpected response from the hub"])
        env.spawn.last(REPO).finish(1)
        env.store.poll()
    elif outcome == "cancelled":
        env.store.cancel(REPO, "alice")


@pytest.mark.req("SPEC-GPU-002 §5.4.1")
@pytest.mark.parametrize("outcome", NEWER_OUTCOMES)
def test_local_ready_with_completed_revision_and_newer_unfinished(models_env, outcome):
    """§5.4.1: є повна ревізія, яку менеджер докачав раніше — ready, навіть коли нова ще качається чи не докачалась."""
    env = models_env
    _completed_then_newer(env, outcome)
    got = (_local(env).get(REPO) or {}).get("state")
    assert got == "ready", f"completed revision + newer one {outcome}: expected state 'ready', got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.1")
@pytest.mark.parametrize("outcome", NEWER_OUTCOMES)
def test_local_revision_is_completed_not_newer_partial(models_env, outcome):
    """§5.4.1: revision — докачана менеджером (остання повна), а не часткова новіша, на яку вже вказує refs/main."""
    env = models_env
    _completed_then_newer(env, outcome)
    got = (_local(env).get(REPO) or {}).get("revision")
    assert got == sha_of(REPO), f"completed revision + newer one {outcome}: expected revision {sha_of(REPO)!r}, got {got!r} (newer is {SHA_NEW!r})"


@pytest.mark.req("SPEC-GPU-002 §5.4.1")
def test_local_revision_refs_main_without_record(models_env):
    """§5.4.1: запису менеджера немає — revision та, на яку вказує refs/main, хоч інша ревізія новіша."""
    env = models_env
    main = put_snapshot(env.hf_home, REPO, ["config.json"])
    other = put_snapshot(env.hf_home, REPO, ["config.json"], sha="f" * 40, ref=False)
    _set_mtime(main, T0 - 7200)
    _set_mtime(other, T0 - 60)
    got = (_local(env).get(REPO) or {}).get("revision")
    assert got == sha_of(REPO), f"refs/main -> {sha_of(REPO)!r}, newer {'f' * 40!r}: expected revision {sha_of(REPO)!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.1")
def test_local_revision_newest_without_refs(models_env):
    """§5.4.1: ні запису, ні refs/main — revision найновіша (тут за mtime; створена першою й менша за іменем)."""
    env = models_env
    newer = put_snapshot(env.hf_home, REPO, ["config.json"], sha="1" * 40, ref=False)
    older = put_snapshot(env.hf_home, REPO, ["config.json"], sha="f" * 40, ref=False)
    _set_mtime(older, T0 - 7200)
    _set_mtime(newer, T0 - 60)
    got = (_local(env).get(REPO) or {}).get("revision")
    assert got == "1" * 40, f"two revisions without refs/main: expected the newest {'1' * 40!r}, got {got!r}"


# --- §5.4.2 delete() ------------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §5.4.2")
def test_delete_active_download_refused(models_env):
    """§5.4.2: активне завантаження → download_active; файли лишаються."""
    env = models_env
    env.store.download(REPO, "alice")
    put_snapshot(env.hf_home, REPO, ["config.json"])
    expect_manager_error("download_active", env.store.delete, REPO, "alice")
    assert model_dir(env.hf_home, REPO).is_dir(), "after download_active: expected the model directory to stay"


@pytest.mark.req("SPEC-GPU-002 §5.4.2")
def test_delete_queued_download_refused(make_models_env):
    """§5.4.2, §5.1: queued — активний → download_active."""
    env = make_models_env({"models.max_parallel_downloads": 1})
    env.store.download(REPO, "alice")
    env.clock.advance(10)
    env.store.download(REPO_B, "alice")
    put_snapshot(env.hf_home, REPO_B, ["config.json"])
    expect_manager_error("download_active", env.store.delete, REPO_B, "alice")


@pytest.mark.req("SPEC-GPU-002 §5.4.2")
@pytest.mark.req("SPEC-GPU-002 §9")
def test_delete_unknown_user_refused(models_env):
    """§9: delete перевіряє user — логін не з users.allowed → unknown_user."""
    env = models_env
    _put_local_model(env)
    expect_manager_error("unknown_user", env.store.delete, REPO, STRANGER)


@pytest.mark.req("SPEC-GPU-002 §5.4.2")
@pytest.mark.req("SPEC-GPU-002 §9")
def test_delete_unknown_user_keeps_model(models_env):
    """§9: після відмови unknown_user модель лишається на диску."""
    env = models_env
    _put_local_model(env)
    expect_manager_error("unknown_user", env.store.delete, REPO, STRANGER)
    assert model_dir(env.hf_home, REPO).is_dir(), "after delete by a stranger: expected the model directory to stay"


@pytest.mark.req("SPEC-GPU-002 §5.4.2")
def test_delete_not_local_refused(models_env):
    """§5.4.2: моделі немає на диску → model_not_local."""
    expect_manager_error("model_not_local", models_env.store.delete, REPO, "alice")


@pytest.mark.req("SPEC-GPU-002 §5.4.2")
def test_delete_removes_model_directory(models_env):
    """§5.4.2: видаляються всі ревізії з блобами — тека моделі зникає повністю."""
    env = models_env
    _put_local_model(env)
    env.store.delete(REPO, "alice")
    path = model_dir(env.hf_home, REPO)
    assert not path.exists(), f"after delete: expected {path} gone, it still has {sorted(p.name for p in path.rglob('*'))}"


@pytest.mark.req("SPEC-GPU-002 §5.4.2")
def test_delete_keeps_other_models(models_env):
    """§5.4.2: видалення однієї моделі не чіпає інших."""
    env = models_env
    _put_local_model(env, REPO)
    _put_local_model(env, REPO_B)
    env.store.delete(REPO, "alice")
    assert model_dir(env.hf_home, REPO_B).is_dir(), f"after deleting {REPO!r}: expected {REPO_B!r} to stay"


@pytest.mark.req("SPEC-GPU-002 §5.4.2")
def test_delete_response(models_env):
    """§5.4.2: відповідь {deleted: true, repo, freed_gib ≈ розмір на диску}."""
    env = models_env
    _put_local_model(env)
    body = env.store.delete(REPO, "alice")
    ok = body.get("deleted") is True and body.get("repo") == REPO and gib_close(body.get("freed_gib"), LOCAL_BYTES)
    assert ok, f"delete(): expected {{'deleted': True, 'repo': {REPO!r}, 'freed_gib': ≈1.0}}, got {body!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.2")
@pytest.mark.req("SPEC-GPU-002 §9")
def test_delete_journals_model_delete(models_env):
    """§5.4.2, §9: будь-який дозволений користувач видаляє; журнал model_delete з freed_gib від нього."""
    env = models_env
    _put_local_model(env)
    env.store.delete(REPO, "bob")
    entries = env.journal("model_delete")
    got = [(e.get("user"), gib_close(e.get("freed_gib"), LOCAL_BYTES)) for e in entries]
    assert got == [("bob", True)], f"journal 'model_delete': expected one entry by 'bob' with freed_gib ≈1.0, got {entries!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.2")
def test_delete_refusal_not_journaled(models_env):
    """§5.4.2 (SPEC-GPU-001 §5): відмова model_not_local у журнал не пишеться."""
    env = models_env
    expect_manager_error("model_not_local", env.store.delete, REPO, "alice")
    assert env.journal("model_delete") == [], f"after model_not_local: expected no 'model_delete' entries, got {env.journal()!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.2")
def test_delete_removes_queue_record(models_env):
    """§5.4.2: запис черги моделі видаляється (тут — failed)."""
    env = models_env
    env.store.download(REPO, "alice")
    append_log(log_path(env.data_dir, REPO), ["ERROR ValueError: unexpected response from the hub"])
    env.spawn.last(REPO).finish(1)
    env.store.poll()
    _put_local_model(env)
    env.store.delete(REPO, "alice")
    got = env.records(REPO)
    assert got == [], f"after delete: expected no download record for {REPO!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.2")
@pytest.mark.req("SPEC-GPU-002 §5.4.1")
def test_delete_model_gone_from_local(models_env):
    """§5.4.2: видалена модель зникає з local()."""
    env = models_env
    _put_local_model(env)
    env.store.delete(REPO, "alice")
    got = [m.get("repo") for m in env.store.local()]
    assert REPO not in got, f"local() after delete: expected no {REPO!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §5.4.2")
def test_delete_twice_second_not_local(models_env):
    """§5.4.2: тека вже зникла — повторне видалення → model_not_local."""
    env = models_env
    _put_local_model(env)
    env.store.delete(REPO, "alice")
    expect_manager_error("model_not_local", env.store.delete, REPO, "alice")
