"""Підробки й помічники приймальних тестів SPEC-GPU-002 (фаза 2: моделі HuggingFace).

Навіщо файл: тести фази 2 пишуться наосліп, лише з docs/specs/SPEC-GPU-002-phase-2.md. Її §10 дає шви:
hub — підробка з методами §4, spawn — підробка subprocess.Popen (stdin з write/close, poll, terminate,
wait(timeout), kill), кеш HF — тека в tmp_path з розкладкою §5.3, clock — керований. Тут зібрано ці
підробки, збирання ModelStore (§5) над менеджером фази 1, оракул оцінки «чи влізе» за формулою §3.4 і
читання журналу дій (SPEC-GPU-001 §5: data_dir/journal.jsonl).

Мережа, справжні завантаження, справжні data/ і кеш HF сервера не використовуються: кеш HF і data_dir
тестів — лише в tmp_path.

Імпорти gpu_manager — ліниві (усередині функцій), як і в conftest.py: зламаний модуль фази 2 не валить
збирання тестів фази 1.
"""

from __future__ import annotations

import hashlib
import json
import signal
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from .conftest import BASE_URL, MEM_TOTAL_MIB, N_GPUS, Env

# --- Сталі ------------------------------------------------------------------------------------------
# Усі значення нижче обрані тестами (вхідні дані підробок і конфігу), а не взяті з коду.

MIB = 2**20
GIB = 2**30
PIB = 2**50
CARD_MIB = MEM_TOTAL_MIB  # обсяг кожної фальшивої карти, MiB (A40, як у фазі 1)

# Допуск для полів *_gib (size_gib, done_gib, total_gib, freed_gib, gib): §9 — 2 знаки після коми, тож
# значення відрізняється від точного не більше ніж на половину сотої (+1e-6 на похибку float).
GIB_TOLERANCE = 0.005 + 1e-6

REPO = "acme/tiny-llm"  # основна тестова модель
REPO_B = "acme/other-llm"
REPO_C = "acme/third-llm"
GATED_REPO = "gated-org/secret-llm"  # gated, доступу немає
GATED_OK_REPO = "gated-org/open-llm"  # gated, доступ є
MISSING_REPO = "nobody/no-such-model"  # немає на HF
TOKEN = "test-token-not-real"  # значення функції токена в тестах з токеном

# models.min_free_disk_gib, більший за будь-який справжній диск (≈ 1 EiB): робить disk_full без шва диска.
HUGE_RESERVE_GIB = 10**9
SEARCH_POOL_SIZE = 60  # скільки результатів має підроблений пошук (> 50, межі limit §5.4.4)
NETWORK_DOWN = "fake hub: network is down"  # параметр detail відмови hf_unavailable (§9)

# Файли основної моделі з розмірами «з API» (байти) — уже після select_files (§4: files()).
TINY_FILES: dict[str, int] = {
    ".gitattributes": 1_519,
    "config.json": 700,
    "generation_config.json": 180,
    "model-00001-of-00002.safetensors": 384 * MIB,
    "model-00002-of-00002.safetensors": 128 * MIB,
    "model.safetensors.index.json": 23_950,
    "special_tokens_map.json": 450,
    "tokenizer.json": 2_100_000,
    "tokenizer_config.json": 51_000,
}
TINY_TOTAL = sum(TINY_FILES.values())  # 539 048 711 байт ≈ 0.502 GiB
TINY_WEIGHTS = 384 * MIB + 128 * MIB  # *.safetensors — рівно 0.5 GiB
TINY_CONFIG: dict[str, Any] = {
    "model_type": "llama",
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 2048,
    "head_dim": 64,
    "max_position_embeddings": 131072,
}
TINY_KV = 2 * 16 * 8 * 64 * 2  # §3.3: 32 768 байт на токен
TINY_MAX_POSITION = 131072

CARD_KEYS = ("usable_gib", "fits", "max_context_tokens")  # поля запису per_card (§3)


def sha_of(repo: str) -> str:
    """Детермінований 40-символьний sha ревізії підробленого репозиторію."""
    return hashlib.sha1(repo.encode("utf-8")).hexdigest()


def models_settings(hf_home: Path) -> dict[str, Any]:
    """Налаштування розділу models тестового конфігу (§2): кеш — у tmp_path, запас диска — 0.

    Запас 0: перевірка диска §5.1.5 рахує справжнє вільне місце, шва для нього §10 не дає; з нулем
    тестовій моделі (≈ 0.5 GiB за API) досить ≈ 0.5 GiB вільного місця в tmp_path.
    """
    return {
        "models.hf_home": str(hf_home),
        "models.max_parallel_downloads": 2,
        "models.min_free_disk_gib": 0,
        "models.fit_memory_fraction": 0.9,
        "models.fit_overhead_gib": 2.0,
    }


# --- ManagerError ------------------------------------------------------------------------------------


def manager_error(code: str, **params: Any) -> BaseException:
    """Будує ManagerError(code, **params) з gpu_manager.messages — відмову підробленого hub (§4, §9).

    params — параметри тексту відмови (§9): repo для hf_not_found і hf_gated, detail для hf_unavailable.
    """
    from gpu_manager.messages import ManagerError

    return ManagerError(code, **params)


def expect_manager_error(code: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> BaseException:
    """Викликає func(*args, **kwargs), що має відмовити ManagerError з кодом code; повертає виняток."""
    from gpu_manager.messages import ManagerError

    what = getattr(func, "__name__", repr(func))
    try:
        result = func(*args, **kwargs)
    except ManagerError as exc:
        got = getattr(exc, "code", None)
        assert got == code, f"{what}{args!r}: expected ManagerError code {code!r}, got {got!r} ({exc})"
        return exc
    pytest.fail(f"{what}{args!r}: expected ManagerError {code!r}, got a result {result!r}")


# --- Підроблений hub (§4, §10) ---------------------------------------------------------------------------


@dataclass
class FakeRepo:
    """Один репозиторій підробленого HF: файли з розмірами, sha ревізії, gated і доступ, config.json."""

    files: dict[str, int]
    sha: str
    gated: bool = False
    access: bool = True
    config: dict[str, Any] = field(default_factory=dict)


def search_item(i: int) -> dict[str, Any]:
    """i-й результат підробленого пошуку у форматі §4 search; популярність спадає з i."""
    return {
        "repo": f"org{i:02d}/model-{i:02d}",
        "downloads": 100_000 - i * 1_000,
        "likes": 500 - i,
        "gated": i % 5 == 4,
        "task": "text-generation",
        "params": 1_000_000_000 + i,
        "updated": "2026-09-01T00:00:00Z",
    }


class FakeHub:
    """Підробка HubClient (§4, §10): search / files / check_access / config без мережі.

    Відмови — ManagerError з кодами §4: немає репозиторію — hf_not_found; gated без доступу —
    hf_gated (check_access і config); unavailable=True — hf_unavailable з будь-якого методу.
    calls — журнал викликів (метод, аргументи) для перевірок.
    """

    def __init__(self) -> None:
        self.repos: dict[str, FakeRepo] = {}
        self.calls: list[tuple[Any, ...]] = []
        self.unavailable = False
        self.pool = [search_item(i) for i in range(SEARCH_POOL_SIZE)]

    def add(
        self,
        repo: str,
        files: dict[str, int] | None = None,
        *,
        gated: bool = False,
        access: bool = True,
        config: dict[str, Any] | None = None,
    ) -> FakeRepo:
        """Додає репозиторій; типово — файли й config основної тестової моделі."""
        item = FakeRepo(
            files=dict(TINY_FILES if files is None else files),
            sha=sha_of(repo),
            gated=gated,
            access=access,
            config=dict(TINY_CONFIG if config is None else config),
        )
        self.repos[repo] = item
        return item

    def _repo(self, repo: str) -> FakeRepo:
        if self.unavailable:
            raise manager_error("hf_unavailable", detail=NETWORK_DOWN)
        if repo not in self.repos:
            raise manager_error("hf_not_found", repo=repo)
        return self.repos[repo]

    # --- методи §4 ---

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        self.calls.append(("search", query, limit))
        if self.unavailable:
            raise manager_error("hf_unavailable", detail=NETWORK_DOWN)
        return [dict(item) for item in self.pool[: max(0, int(limit))]]

    def files(self, repo: str, revision: str | None = None) -> tuple[str, dict[str, int], bool]:
        self.calls.append(("files", repo, revision))
        item = self._repo(repo)
        return item.sha, dict(item.files), item.gated

    def check_access(self, repo: str) -> None:
        """§9: нічого не повертає; немає доступу → ManagerError."""
        self.calls.append(("check_access", repo))
        item = self._repo(repo)
        if item.gated and not item.access:
            raise manager_error("hf_gated", repo=repo)

    def config(self, repo: str, revision: str | None = None) -> dict[str, Any]:
        self.calls.append(("config", repo, revision))
        item = self._repo(repo)
        if item.gated and not item.access:
            raise manager_error("hf_gated", repo=repo)
        return dict(item.config)


def standard_hub() -> FakeHub:
    """Hub з трьома звичайними моделями, gated без доступу і gated з доступом."""
    hub = FakeHub()
    hub.add(REPO)
    hub.add(REPO_B)
    hub.add(REPO_C)
    hub.add(GATED_REPO, gated=True, access=False)
    hub.add(GATED_OK_REPO, gated=True, access=True)
    return hub


# --- Підроблений процес завантаження (§5.2, §10) --------------------------------------------------------------


class FakeStdin:
    """stdin підробленого процесу (§10): write і close; записане зберігається для перевірки JSON (§5.2)."""

    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.closed = False

    def write(self, data: Any) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed stdin")
        text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)
        self.chunks.append(text)
        return len(data)

    def flush(self) -> None:
        """Нічого не робить: flush у §10 не названий, але справжній потік його має."""

    def close(self) -> None:
        self.closed = True

    @property
    def text(self) -> str:
        return "".join(self.chunks)

    def payload(self) -> Any:
        """Розібраний JSON зі stdin; None, якщо записано не JSON або ще нічого."""
        try:
            return json.loads(self.text)
        except ValueError:
            return None


class FakeProcess:
    """Підроблений процес download_worker (§10): poll / terminate / wait(timeout) / kill.

    Процес «працює», доки тест не викличе finish(code). stubborn=True — процес ігнорує SIGTERM
    (terminate), і wait(timeout) тоді піднімає TimeoutExpired одразу, без справжнього очікування.
    events — порядок викликів terminate / wait / kill; wait_timeouts — аргументи timeout.
    pid навмисно немає: §10 його не називає, а os.kill за вигаданим pid зачепив би справжній процес.
    """

    def __init__(self, argv: list[Any], extra_args: tuple[Any, ...], kwargs: dict[str, Any], *, stubborn: bool) -> None:
        self.argv = argv
        self.extra_args = extra_args
        self.kwargs = kwargs
        self.stdin = FakeStdin()
        self.returncode: int | None = None
        self.stubborn = stubborn
        self.events: list[str] = []
        self.wait_timeouts: list[Any] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.events.append("terminate")
        if not self.stubborn and self.returncode is None:
            self.returncode = -int(signal.SIGTERM)

    def kill(self) -> None:
        self.events.append("kill")
        if self.returncode is None:
            self.returncode = -int(signal.SIGKILL)

    def wait(self, timeout: float | None = None) -> int:
        self.events.append("wait")
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired([str(a) for a in self.argv], timeout if timeout is not None else 0)
        return self.returncode

    # --- керування з тесту ---

    def finish(self, code: int = 0) -> None:
        """Процес завершився з кодом code (0 — успіх)."""
        self.returncode = code

    @property
    def terminate_calls(self) -> int:
        return self.events.count("terminate")

    @property
    def kill_calls(self) -> int:
        return self.events.count("kill")

    @property
    def stopped(self) -> bool:
        return self.terminate_calls > 0 or self.kill_calls > 0

    @property
    def repo(self) -> Any:
        """repo з JSON, записаного в stdin (§5.2); None, якщо JSON ще немає."""
        payload = self.stdin.payload()
        return payload.get("repo") if isinstance(payload, dict) else None


class FakeSpawn:
    """Підробка spawn (= subprocess.Popen, §5, §10): кожен виклик створює FakeProcess і запам'ятовує його."""

    def __init__(self) -> None:
        self.processes: list[FakeProcess] = []
        self.stubborn = False  # нові процеси ігнорують SIGTERM

    def __call__(self, argv: Any, *args: Any, **kwargs: Any) -> FakeProcess:
        proc = FakeProcess(list(argv), args, dict(kwargs), stubborn=self.stubborn)
        self.processes.append(proc)
        return proc

    @property
    def repos(self) -> list[Any]:
        """repo кожного запущеного процесу в порядку запуску."""
        return [p.repo for p in self.processes]

    def for_repo(self, repo: str) -> list[FakeProcess]:
        return [p for p in self.processes if p.repo == repo]

    def last(self, repo: str) -> FakeProcess:
        procs = self.for_repo(repo)
        assert procs, f"spawn: expected a download process for {repo!r}, got processes for {self.repos!r}"
        return procs[-1]


# --- Кеш HF у tmp_path (§5.3) ----------------------------------------------------------------------------------


def model_dir(hf_home: Path, repo: str) -> Path:
    """<hf_home>/hub/models--<org>--<name> (§5.3)."""
    return Path(hf_home) / "hub" / ("models--" + repo.replace("/", "--"))


def put_snapshot(
    hf_home: Path,
    repo: str,
    names: Any,
    *,
    sha: str | None = None,
    sizes: dict[str, int] | None = None,
    ref: bool = True,
) -> Path:
    """Створює знімок ревізії snapshots/<sha>/<ім'я> звичайними файлами (§10), теку blobs і refs/main.

    sha — ревізія (типово sha_of(repo), як у FakeHub); sizes — розміри окремих файлів: такий файл
    створюється розрідженим (truncate) і на диску місця не займає; решта файлів — по 2 байти.
    ref=False — refs/main не пишеться (для «чужої» ревізії).
    """
    base = model_dir(hf_home, repo)
    revision = sha or sha_of(repo)
    snapshot = base / "snapshots" / revision
    for name in names:
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        size = (sizes or {}).get(name)
        with open(path, "wb") as handle:
            if size is None:
                handle.write(b"{}")
            else:
                handle.truncate(size)
    (base / "blobs").mkdir(parents=True, exist_ok=True)
    if ref:
        (base / "refs").mkdir(parents=True, exist_ok=True)
        (base / "refs" / "main").write_text(revision, encoding="utf-8")
    return snapshot


def put_incomplete(hf_home: Path, repo: str, size: int, tag: str = "part") -> Path:
    """Створює недокачаний блоб blobs/<hex>.incomplete розміром size байт (розріджений файл)."""
    blobs = model_dir(hf_home, repo) / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    path = blobs / f"{hashlib.sha256(f'{repo}:{tag}'.encode('utf-8')).hexdigest()}.incomplete"
    with open(path, "wb") as handle:
        handle.truncate(size)
    return path


def log_path(data_dir: Path, repo: str) -> Path:
    """Лог процесу завантаження: <data_dir>/downloads/models--<org>--<name>.log (§5.2)."""
    return Path(data_dir) / "downloads" / ("models--" + repo.replace("/", "--") + ".log")


def append_log(path: Path, lines: list[str]) -> None:
    """Дописує рядки в лог процесу — так «пише» підроблений download_worker (§5.2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


# --- Журнал (SPEC-GPU-001 §5) ---------------------------------------------------------------------------------------


def journal_entries(data_dir: Path) -> list[dict[str, Any]]:
    """Записи data_dir/journal.jsonl у порядку файлу; нерозібрані рядки пропускаються."""
    path = Path(data_dir) / "journal.jsonl"
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            entries.append(item)
    return entries


# --- Оцінка «чи влізе» (§3.4) --------------------------------------------------------------------------------------


def fit_value(fit: Any, name: str) -> Any:
    """Поле Fit за назвою §3: Fit може бути об'єктом з атрибутами або словником (info — «Fit як словник»)."""
    if isinstance(fit, Mapping):
        return fit[name]
    return getattr(fit, name)


def entry_value(entry: Any, key: str) -> Any:
    """Поле запису per_card (словник або об'єкт)."""
    if isinstance(entry, Mapping):
        return entry[key]
    return getattr(entry, key)


def per_card_keys(fit: Any) -> set[int]:
    """Обсяги карт (MiB), для яких є запис per_card; ключі JSON-рядки зводяться до int."""
    return {int(k) for k in fit_value(fit, "per_card")}


def card_view(fit: Any, mib: int) -> dict[str, Any]:
    """Запис per_card для карти mib MiB як словник {usable_gib, fits, max_context_tokens}."""
    per_card = fit_value(fit, "per_card")
    for key in (mib, str(mib)):
        if key in per_card:
            entry = per_card[key]
            return {k: entry_value(entry, k) for k in CARD_KEYS}
    raise AssertionError(f"per_card: expected an entry for {mib} MiB, got keys {list(per_card)!r}")


def fit_card_oracle(
    mib: int, fraction: float, weights: int, overhead_gib: float, kv: int | None, cap: int | None
) -> dict[str, Any]:
    """Очікуваний запис per_card дослівно за §3.4.

    usable = mib·2²⁰·fraction − weight_bytes − overhead_gib·2³⁰; fits = usable > 0;
    max_context_tokens = usable // kv, обмежене cap (max_position_embeddings), 0 при usable ≤ 0,
    None при невідомому kv; usable_gib — 1 знак (§3, опис Fit).
    """
    usable = mib * 2**20 * fraction - weights - overhead_gib * 2**30
    if kv is None:
        tokens = None
    elif usable <= 0:
        tokens = 0
    else:
        tokens = int(usable // kv)
        if cap is not None:
            tokens = min(tokens, cap)
    return {"usable_gib": round(usable / 2**30, 1), "fits": usable > 0, "max_context_tokens": tokens}


def gib_close(actual: Any, expected_bytes: float) -> bool:
    """actual (GiB) збігається з expected_bytes/2³⁰ у межах GIB_TOLERANCE."""
    return isinstance(actual, (int, float)) and not isinstance(actual, bool) and (
        abs(actual - expected_bytes / GIB) <= GIB_TOLERANCE
    )


# --- Сервіс з моделями ------------------------------------------------------------------------------------------


def state_and_journal(manager: Any) -> tuple[Any, Any]:
    """Об'єкти стану (state.json) і журналу фази 1 для конструктора ModelStore (§5).

    §9: store = manager.store, journal = manager.journal_log — той самий state.json і журнал, що в карт.
    """
    return manager.store, manager.journal_log


class ModelsEnv:
    """Один «запуск» сервісу з моделями: менеджер фази 1 (Env) і ModelStore над ним (§5).

    cfg — Config з розділом models; hub — FakeHub; spawn — FakeSpawn; token — значення, яке повертає
    функція токена (None — токена немає); card_sizes — обсяги карт у MiB, які повертає card_mib().
    restart() будує новий менеджер і новий ModelStore над тими самими конфігом, data_dir і кешем HF,
    з новою підробкою spawn — так тести моделюють перезапуск сервісу (§5.1.9–5.1.10).
    """

    def __init__(
        self,
        cfg: Any,
        backend: Any,
        clock: Any,
        hub: FakeHub,
        spawn: FakeSpawn,
        *,
        token: str | None = None,
        card_sizes: list[int] | None = None,
    ) -> None:
        from gpu_manager.models import ModelStore

        self.base = Env(cfg, backend, clock)
        self.cfg = cfg
        self.backend = backend
        self.clock = clock
        self.hub = hub
        self.spawn = spawn
        self.token_value = token
        self.card_sizes = list(card_sizes) if card_sizes is not None else [CARD_MIB] * N_GPUS
        state, journal = state_and_journal(self.base.manager)
        self.store = ModelStore(cfg, hub, state, journal, self._card_mib, self._token, clock=clock, spawn=spawn)

    def _card_mib(self) -> list[int]:
        return list(self.card_sizes)

    def _token(self) -> str | None:
        return self.token_value

    @property
    def manager(self) -> Any:
        return self.base.manager

    @property
    def data_dir(self) -> Path:
        return Path(self.cfg.data_dir)

    @property
    def hf_home(self) -> Path:
        return Path(self.cfg.hf_home)

    def restart(self) -> ModelsEnv:
        return ModelsEnv(
            self.cfg,
            self.backend,
            self.clock,
            self.hub,
            FakeSpawn(),
            token=self.token_value,
            card_sizes=self.card_sizes,
        )

    def client(self) -> Any:
        """TestClient над build_app(cfg, manager, models=store) (§6); використовувати як контекстний менеджер."""
        from starlette.testclient import TestClient

        from gpu_manager.app import build_app

        return TestClient(build_app(self.cfg, self.manager, models=self.store), base_url=BASE_URL)

    def mcp(self) -> Any:
        """MCPServer з build_mcp(manager, tz, models=store) (§7)."""
        from gpu_manager.mcp_tools import build_mcp

        return build_mcp(self.manager, self.cfg.display_timezone, models=self.store)

    # --- читання стану ---

    def records(self, repo: str) -> list[dict[str, Any]]:
        """Записи downloads() для repo (§5.1.11)."""
        return [r for r in self.store.downloads() if r.get("repo") == repo]

    def record(self, repo: str) -> dict[str, Any]:
        records = self.records(repo)
        assert len(records) == 1, f"downloads(): expected exactly one record for {repo!r}, got {len(records)}: {records!r}"
        return records[0]

    def status(self, repo: str) -> Any:
        return self.record(repo).get("status")

    def journal(self, action: str | None = None) -> list[dict[str, Any]]:
        """Записи журналу (порядок файлу); action — лише записи з цією дією."""
        entries = journal_entries(self.data_dir)
        return entries if action is None else [e for e in entries if e.get("action") == action]

    def journal_for(self, repo: str) -> list[dict[str, Any]]:
        """Записи журналу з полем repo == repo."""
        return [e for e in journal_entries(self.data_dir) if e.get("repo") == repo]

    def state_downloads(self) -> Any:
        """Розділ downloads файлу data_dir/state.json (§5.1.9); відсутній файл чи розділ — помилка тесту."""
        path = self.data_dir / "state.json"
        assert path.is_file(), f"expected {path} to exist after a download was queued"
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(doc, dict) and "downloads" in doc, (
            f"state.json: expected an object with a 'downloads' section, got keys {sorted(doc) if isinstance(doc, dict) else doc!r}"
        )
        return doc["downloads"]
