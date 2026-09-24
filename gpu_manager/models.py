"""Моделі HuggingFace на диску: пошук, оцінка «чи влізе», завантаження, список, видалення (фаза 2).

Завантаження — окремі процеси (download_worker) у черзі з обмеженням паралельності. Стан черги лежить у
state.json, тож після перезапуску менеджера недокачане продовжується (п. 8 обговорення: «піднімати останню
конфігурацію»). Прогрес рахується з диска: готові файли знімка + *.incomplete, поділені на розмір з API HF."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import scan_cache_dir

from .config import Config
from .core import require_user
from .hub import HubClient, estimate_fit
from .journal import Journal
from .messages import ManagerError
from .store import StateStore

SECTION = "downloads"
QUEUED, DOWNLOADING, DONE, FAILED, CANCELLED = "queued", "downloading", "done", "failed", "cancelled"
_ACTIVE = (QUEUED, DOWNLOADING)
_GIB = 1024**3
_SEARCH_MAX = 50
# Скільки чекати завершення процесу після SIGTERM, перш ніж добити SIGKILL, секунд.
_TERM_WAIT_S = 5
# Скільки останніх байтів логу завантаження читати, щоб знайти рядок "ERROR ...".
_LOG_TAIL_BYTES = 4096
# Клас винятку huggingface_hub у рядку ERROR -> код відмови для людини й агента.
_ERROR_CODES = {"GatedRepoError": "hf_gated", "RepositoryNotFoundError": "hf_not_found",
                "RevisionNotFoundError": "hf_not_found"}


@dataclass
class Download:
    repo: str
    revision: str  # commit sha, до якого прив'язане завантаження
    user: str
    files: dict[str, int]  # файл -> розмір у байтах (з API HF)
    status: str
    started: float
    finished: float | None = None
    error_code: str | None = None
    error: str | None = None

    @property
    def total(self) -> int:
        return sum(self.files.values())


def _repo_dir_name(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


class ModelStore:
    def __init__(
        self,
        cfg: Config,
        hub: HubClient,
        store: StateStore,
        journal: Journal,
        card_mib: Callable[[], list[int]],
        token: Callable[[], str | None],
        clock: Callable[[], float] = time.time,
        spawn: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        """card_mib — обсяги пам'яті карт (для оцінки «чи влізе»); token — токен HF для процесів завантаження;
        spawn — шов для тестів замість subprocess.Popen."""
        self._cfg = cfg
        self._hub = hub
        self._store = store
        self._journal = journal
        self._card_mib = card_mib
        self._token = token
        self._clock = clock
        self._spawn = spawn
        self._hub_dir = cfg.hf_home / "hub"
        self._log_dir = cfg.data_dir / "downloads"
        self._lock = threading.RLock()
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        # Чи запущена модель (фаза 3); підключає ModelRunner, бо він створюється після сховища.
        self.in_use: Callable[[str], bool] = lambda repo: False
        raw = store.read(SECTION) or {}
        self._items = {repo: Download(**d) for repo, d in raw.items()}
        for d in self._items.values():  # після перезапуску незавершені стають у чергу й продовжуються
            if d.status == DOWNLOADING:
                d.status = QUEUED

    # ---- HuggingFace -----------------------------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return self._hub.search(query, max(1, min(limit, _SEARCH_MAX)))

    def info(self, repo: str, revision: str | None = None, fraction: float | None = None) -> dict[str, Any]:
        """Опис моделі до завантаження: розмір вибраних файлів, gated, оцінка «чи влізе» на кожен обсяг карти.

        fraction — частка карти (напр. 0.45, щоб поділити карту між двома моделями); типово з конфігу."""
        sha, files, gated = self._hub.files(repo, revision)
        fr = fraction if fraction is not None else self._cfg.fit_memory_fraction
        if not 0 < fr <= 1:
            raise ManagerError("bad_fraction")
        fit = estimate_fit(files, self._hub.config(repo, sha), self._card_mib(), fr, self._cfg.fit_overhead_gib)
        return {"repo": repo, "revision": sha, "gated": gated, "files": len(files),
                "size_gib": round(sum(files.values()) / _GIB, 2), "fraction": fr, "fit": asdict(fit),
                "local": self._is_local(repo)}

    # ---- завантаження ----------------------------------------------------------------------------

    def download(self, repo: str, user: str, revision: str | None = None) -> dict[str, Any]:
        """Ставить модель у чергу завантаження. Уже активне завантаження тієї ж моделі не дублюється."""
        self._check_user(user)
        with self._lock:
            current = self._items.get(repo)
            if current is not None and current.status in _ACTIVE:
                return self._view(current)
        sha, files, gated = self._hub.files(repo, revision)
        if gated:
            self._hub.check_access(repo)
        need = max(0, sum(files.values()) - self._bytes_done(repo, sha, files))
        if need == 0 and self._complete(repo, sha, files):
            # Уже все на диску: процес не потрібен, а новий запис «download» у журналі був би неправдою.
            with self._lock:
                now = self._clock()
                d = Download(repo=repo, revision=sha, user=user, files=files, status=DONE, started=now, finished=now)
                self._items[repo] = d
                self._save()
            return self._view(d)
        free = shutil.disk_usage(self._existing_parent(self._cfg.hf_home)).free
        if free - need < self._cfg.min_free_disk_gib * _GIB:
            raise ManagerError("disk_full", need_gib=round(need / _GIB, 1), free_gib=round(free / _GIB, 1),
                               reserve_gib=self._cfg.min_free_disk_gib)
        with self._lock:
            d = Download(repo=repo, revision=sha, user=user, files=files, status=QUEUED, started=self._clock())
            self._items[repo] = d
            self._save()
        self._journal.record(user, "download", repo=repo, size_gib=round(d.total / _GIB, 2))
        self.poll()
        return self._view(d)

    def cancel(self, repo: str, user: str) -> dict[str, Any]:
        """Зупиняє завантаження; недокачані файли лишаються, повторне завантаження їх продовжить."""
        self._check_user(user)
        with self._lock:
            d = self._items.get(repo)
            if d is None or d.status not in _ACTIVE:
                raise ManagerError("download_not_active", repo=repo)
            self._stop(repo)
            d.status, d.finished = CANCELLED, self._clock()
            self._save()
        self._journal.record(user, "download_cancel", repo=repo)
        return self._view(d)

    def poll(self) -> None:
        """Крок черги: завершені процеси — у done/failed, вільні місця — наступним у черзі.
        Викликається фоновою задачею щосекунди і після кожної зміни."""
        with self._lock:
            changed = False
            for repo, proc in list(self._procs.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                del self._procs[repo]
                d = self._items[repo]
                d.finished = self._clock()
                if rc == 0:
                    d.status = DONE
                else:
                    d.status = FAILED
                    d.error_code, d.error = self._failure(repo)
                changed = True
                self._journal.record(d.user, "download_done" if rc == 0 else "download_failed", repo=repo,
                                     **({"error": d.error_code} if rc else {}))
            for d in sorted(self._items.values(), key=lambda x: x.started):
                if len(self._procs) >= self._cfg.max_parallel_downloads:
                    break
                if d.status == QUEUED and d.repo not in self._procs:
                    self._start(d)
                    changed = True
            if changed:
                self._save()

    def downloads(self) -> list[dict[str, Any]]:
        """Усі відомі завантаження з прогресом, найновіші першими."""
        with self._lock:
            items = sorted(self._items.values(), key=lambda x: x.started, reverse=True)
        return [self._view(d) for d in items]

    def stop_all(self) -> None:
        """Зупиняє процеси при виході менеджера; у state.json вони лишаються активними і продовжаться."""
        with self._lock:
            for repo in list(self._procs):
                self._stop(repo)

    def _start(self, d: Download) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "HF_HOME": str(self._cfg.hf_home), "HF_HUB_DISABLE_PROGRESS_BARS": "1",
               "HF_HUB_DISABLE_TELEMETRY": "1"}
        env.pop("HF_TOKEN", None)
        token = self._token()
        if token:
            env["HF_TOKEN"] = token
        log = open(self._log_path(d.repo), "ab")  # noqa: SIM115 — дескриптор успадковує процес
        try:
            proc = self._spawn([sys.executable, "-m", "gpu_manager.download_worker"], stdin=subprocess.PIPE,
                               stdout=log, stderr=log, env=env, cwd=str(Path(__file__).resolve().parent.parent))
        finally:
            log.close()
        assert proc.stdin is not None
        proc.stdin.write(json.dumps({"repo": d.repo, "revision": d.revision, "files": list(d.files)}).encode())
        proc.stdin.close()
        self._procs[d.repo] = proc
        d.status, d.error_code, d.error, d.finished = DOWNLOADING, None, None, None

    def _stop(self, repo: str) -> None:
        proc = self._procs.pop(repo, None)
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(_TERM_WAIT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def _failure(self, repo: str) -> tuple[str, str]:
        try:
            with open(self._log_path(repo), "rb") as f:
                f.seek(max(0, os.path.getsize(self._log_path(repo)) - _LOG_TAIL_BYTES))
                tail = f.read().decode("utf-8", "replace")
        except OSError:
            tail = ""
        line = next((ln for ln in reversed(tail.splitlines()) if ln.startswith("ERROR ")), "")
        cls = line[len("ERROR "):].split(":", 1)[0] if line else ""
        if "No space left" in tail:
            return "disk_full_during", line[:300]
        return _ERROR_CODES.get(cls, "download_failed"), (line or tail[-300:]).strip()[:300]

    def _log_path(self, repo: str) -> Path:
        return self._log_dir / f"{_repo_dir_name(repo)}.log"

    def _save(self) -> None:
        self._store.write(SECTION, {repo: asdict(d) for repo, d in self._items.items()})

    # ---- диск ------------------------------------------------------------------------------------

    def _bytes_done(self, repo: str, revision: str, files: dict[str, int]) -> int:
        repo_dir = self._hub_dir / _repo_dir_name(repo)
        snap = repo_dir / "snapshots" / revision
        done = 0
        for name, size in files.items():
            p = snap / name
            if p.exists():
                done += size
        blobs = repo_dir / "blobs"
        if blobs.is_dir():
            done += sum(p.stat().st_size for p in blobs.glob("*.incomplete"))
        return min(done, sum(files.values()))

    def _complete(self, repo: str, revision: str, files: dict[str, int]) -> bool:
        """Усі вибрані файли є в знімку ревізії (а не лише *.incomplete того ж обсягу)."""
        snap = self._hub_dir / _repo_dir_name(repo) / "snapshots" / revision
        return all((snap / name).exists() for name in files)

    def _view(self, d: Download) -> dict[str, Any]:
        done = d.total if d.status == DONE else self._bytes_done(d.repo, d.revision, d.files)
        return {"repo": d.repo, "revision": d.revision, "user": d.user, "status": d.status,
                "done_gib": round(done / _GIB, 2), "total_gib": round(d.total / _GIB, 2),
                "percent": round(100 * done / d.total, 1) if d.total else 100.0,
                "started": d.started, "finished": d.finished, "error_code": d.error_code, "error": d.error}

    def local(self) -> list[dict[str, Any]]:
        """Моделі в кеші HF: розмір на диску, ревізії, стан (ready / downloading / partial)."""
        if not self._hub_dir.is_dir():
            return []
        info = scan_cache_dir(self._hub_dir)
        with self._lock:
            items = dict(self._items)
        out = []
        for r in sorted(info.repos, key=lambda x: x.repo_id):
            if r.repo_type != "model":
                continue
            d = items.get(r.repo_id)
            state = "ready" if d is None or d.status == DONE else ("downloading" if d.status in _ACTIVE else "partial")
            revs = {x.commit_hash: x for x in r.revisions}
            out.append({"repo": r.repo_id, "size_gib": round(r.size_on_disk / _GIB, 2),
                        "revisions": sorted(revs), "revision": self._pick_revision(r.repo_path, revs, d),
                        "state": state, "last_modified": r.last_modified})
        return out

    def delete(self, repo: str, user: str) -> dict[str, Any]:
        """Видаляє модель з диска разом зі спільними блобами. Активне завантаження спершу треба скасувати."""
        self._check_user(user)
        with self._lock:
            d = self._items.get(repo)
            if d is not None and d.status in _ACTIVE:
                raise ManagerError("download_active", repo=repo)
            if self.in_use(repo):
                raise ManagerError("model_running", repo=repo)
            info = scan_cache_dir(self._hub_dir) if self._hub_dir.is_dir() else None
            found = next((r for r in info.repos if r.repo_id == repo and r.repo_type == "model"), None) if info else None
            if found is None:
                raise ManagerError("model_not_local", repo=repo)
            assert info is not None
            strategy = info.delete_revisions(*[x.commit_hash for x in found.revisions])
            freed = strategy.expected_freed_size
            strategy.execute()
            if found.repo_path.exists():  # недокачані *.incomplete і порожні теки ревізій delete_revisions лишає
                shutil.rmtree(found.repo_path, ignore_errors=True)
            self._items.pop(repo, None)
            self._save()
        self._journal.record(user, "model_delete", repo=repo, freed_gib=round(freed / _GIB, 2))
        return {"deleted": True, "repo": repo, "freed_gib": round(freed / _GIB, 2)}

    @staticmethod
    def _pick_revision(repo_path: Path, revs: dict[str, Any], d: Download | None) -> str:
        """Яку ревізію запускати: докачану менеджером -> на яку вказує refs/main -> найновішу. Без цього
        вибір з frozenset був би випадковим, і могла б запуститися стара чи неповна ревізія."""
        if d is not None and d.status == DONE and d.revision in revs:
            return d.revision
        try:
            main = (repo_path / "refs" / "main").read_text(encoding="utf-8").strip()
        except OSError:
            main = ""
        if main in revs:
            return main
        return max(revs, key=lambda h: revs[h].last_modified)

    def _is_local(self, repo: str) -> bool:
        return (self._hub_dir / _repo_dir_name(repo) / "snapshots").is_dir()

    @staticmethod
    def _existing_parent(path: Path) -> Path:
        while not path.exists() and path != path.parent:
            path = path.parent
        return path

    def _check_user(self, user: str) -> None:
        require_user(self._cfg.users, user)
