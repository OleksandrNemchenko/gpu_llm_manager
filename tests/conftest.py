"""Спільні фікстури приймальних тестів SPEC-GPU-001, фаза 1.

Навіщо файл: тести пишуться наосліп, лише з docs/specs/SPEC-GPU-001-phase-1.md. Тут зібрано все,
що потрібно для цього зі специфікації: §2 (формат config.json) і §10 (шви для тестів) —
фальшивий GpuBackend із dataclass-ів специфікації, керований годинник, тимчасовий config.json у
tmp_path і сервіс, зібраний через build_manager / build_app / build_mcp.

Справжні карти, справжній час, мережеві порти й тека data/ репозиторію не використовуються.

Імпорти gpu_manager — ліниві (усередині функцій): зламаний один модуль пакета не повинен валити
збирання тестів інших розділів специфікації.

Фаза 2 (SPEC-GPU-002, моделі HuggingFace): фікстури hf_home / fake_hub / fake_spawn / make_models_env
в кінці файлу; самі підробки — у tests/model_fakes.py.

Фази 3–4 (SPEC-GPU-003, моделі на vLLM і доступ агентів): фікстури make_runner_env / runner_env / upstream /
proxied_env — у самому кінці; підробки швів ModelRunner — у tests/runner_fakes.py.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pytest

# --- Сталі тестового середовища ------------------------------------------------------------------
# Усі значення нижче обрані тестами (це вхідні дані конфігу й фальшивого бекенда), а не взяті з коду.

SPEC_ID = "SPEC-GPU-001"
# Стартовий unix-час фальшивого годинника; кратний 60 — хвилинні рядки часу MCP (§8) не залежать
# від того, як реалізація округлює секунди.
T0 = 1_789_999_980.0
N_GPUS = 4  # кількість карт фальшивого бекенда: допустимі gpu — 0..3 (§4.2)
PORT = 1200  # server.port тестового конфігу
HOSTS = ["127.0.0.1", "203.0.113.7"]  # server.hosts: перша — loopback, друга — ні (§7.2 public_host)
EXTRA_HOST_NAMES = ["gpu.lan"]  # server.extra_host_names
PUBLIC_HOST = f"203.0.113.7:{PORT}"  # очікуваний public_host для HOSTS (§7.2)
BASE_URL = f"http://127.0.0.1:{PORT}"  # Host тестового клієнта — дозволений (§7.1.1)
USERS = ["alice", "bob", "carol", "dave"]  # users.allowed: 4 довірені користувачі (§1)
STRANGER = "mallory"  # логін поза users.allowed
BUSY_MIB = 1024  # gpu.busy_memory_mib
MEM_TOTAL_MIB = 46068  # memory_total_mib кожної фальшивої карти (A40)
TZ_NAME = "Europe/Kyiv"  # server.display_timezone
MODEL_PORTS = [8000, 8099]  # vllm.port_range
HISTORY_COLUMNS = ("t", "temp_c", "util_pct", "power_w", "mem_mib")  # колонки історії (§6)

DELETE = object()  # значення-маркер для перевизначень: прибрати налаштування з конфігу

_CYRILLIC = re.compile("[Ѐ-ӿ]")
_LOCAL_MINUTE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")


def pytest_configure(config: pytest.Config) -> None:
    """Реєструє маркери, якими користуються ці тести.

    pytest.ini у репозиторії відсутній, тож без цієї реєстрації кожен маркер давав би
    PytestUnknownMarkWarning. Місце реєстрації — тимчасове: маркери мають переїхати в pytest.ini.
    """
    for line in (
        "unit: pure logic without dependencies",
        "component: real classes with a minimum of dependencies, no hardware",
        "e2e: the full service stack in-process (ASGI app + manager + fake backend)",
        "hitl: reads real hardware (NVML); runs only by hand with GPU_MANAGER_LIVE=1",
        "req(id): traceability to a requirement or a specification item",
    ):
        config.addinivalue_line("markers", line)


# --- Конфіг (§2) ----------------------------------------------------------------------------------


def default_settings() -> dict[str, Any]:
    """Повний валідний набір налаштувань §2.4 у вигляді «шлях.налаштування → значення»."""
    return {
        "server.hosts": list(HOSTS),
        "server.port": PORT,
        "server.display_timezone": TZ_NAME,
        "server.extra_host_names": list(EXTRA_HOST_NAMES),
        "users.allowed": list(USERS),
        "gpu.sample_interval_s": 1,
        "gpu.ring_keep_s": 3600,
        "gpu.history_db_interval_s": 60,
        "gpu.history_db_keep_days": 30,
        "gpu.busy_memory_mib": BUSY_MIB,
        "vllm.port_range": list(MODEL_PORTS),
        "journal.keep_entries": 1000,
        "journal.keep_days": 90,
        "paths.data_dir": "data",
    }


# Обов'язкові налаштування §2.4 — усі, крім необов'язкового server.extra_host_names.
REQUIRED_SETTINGS = tuple(k for k in default_settings() if k != "server.extra_host_names")


def build_config_doc(settings: dict[str, Any]) -> dict[str, Any]:
    """Будує документ config.json у форматі §2.1–2.2 з пар «шлях → значення».

    settings — словник «розділ.налаштування → значення»; значення DELETE пропускається.
    Кожен розділ отримує ключ comment, кожне налаштування — обгортку {value, comment}.
    """
    doc: dict[str, Any] = {}
    for dotted, value in settings.items():
        if value is DELETE:
            continue
        *sections, key = dotted.split(".")
        node = doc
        for name in sections:
            node = node.setdefault(name, {"comment": f"Розділ «{name}»"})
        node[key] = {"value": value, "comment": f"Налаштування «{dotted}»"}
    return doc


def config_doc(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Валідний документ конфігу з перевизначеннями overrides (DELETE — прибрати налаштування)."""
    settings = default_settings()
    settings.update(overrides or {})
    return build_config_doc(settings)


def _precreate_data_dir(config_dir: Path, doc: dict[str, Any]) -> None:
    """Створює data_dir заздалегідь: чи створює її сервіс сам, §2 не каже (прогалина специфікації)."""
    try:
        value = doc["paths"]["data_dir"]["value"]
    except (KeyError, TypeError):
        return
    if isinstance(value, str) and value:
        target = Path(value)
        if not target.is_absolute():
            target = config_dir / target
        target.mkdir(parents=True, exist_ok=True)


# --- Годинник і бекенд (§10) ------------------------------------------------------------------------


class FakeClock:
    """Керований годинник (§10, аргумент clock): повертає unix-час, що змінюється лише явно."""

    def __init__(self, start: float = T0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        """Просуває час на seconds секунд і повертає новий поточний час."""
        self.now += seconds
        return self.now


class FakeBackend:
    """Фальшивий GpuBackend (§10): info() / sample(ts) / processes() з керованими даними.

    readings — показники кожної карти, з яких sample(ts) будує GpuSample; silent — карти, про які
    sample() мовчить (телеметрії немає, §3.3); procs — процеси кожної карти для processes().
    """

    def __init__(self, n_gpus: int = N_GPUS) -> None:
        from gpu_manager.gpu import GpuInfo

        self.n_gpus = n_gpus
        self.infos = [
            GpuInfo(
                index=i,
                name="NVIDIA A40",
                uuid=f"GPU-fake-{i:04d}",
                pci_bus_id=f"00000000:{0x41 + i:02X}:00.0",
                memory_total_mib=MEM_TOTAL_MIB,
                power_limit_w=300.0,
                ecc_enabled=True,
                compute_capability="8.6",
                temp_slowdown_c=88,
            )
            for i in range(n_gpus)
        ]
        # Температура 40+i робить карти розрізненними у відповідях.
        self.readings: dict[int, dict[str, Any]] = {
            i: {
                "temperature_c": 40.0 + i,
                "util_pct": 0.0,
                "power_w": 30.0,
                "memory_used_mib": 0,
                "error": None,
            }
            for i in range(n_gpus)
        }
        self.silent: set[int] = set()
        self.procs: dict[int, list[Any]] = {i: [] for i in range(n_gpus)}
        self.sample_calls = 0
        self.process_calls = 0

    def set_reading(self, index: int, **fields: Any) -> None:
        """Змінює показники карти index (temperature_c, util_pct, power_w, memory_used_mib, error)."""
        self.readings[index].update(fields)

    def set_error(self, index: int, message: str = "NVML: GPU is lost") -> None:
        """Робить замір карти index помилковим (GpuSample.error)."""
        self.readings[index]["error"] = message

    def drop_telemetry(self, index: int) -> None:
        """Прибирає карту index із результату sample(): телеметрії немає."""
        self.silent.add(index)

    def set_processes(self, index: int, *procs: Any) -> None:
        """Задає список процесів GpuProcess на карті index."""
        self.procs[index] = list(procs)

    # --- протокол GpuBackend (§10) ---

    def info(self) -> list[Any]:
        return list(self.infos)

    def sample(self, ts: float) -> list[Any]:
        from gpu_manager.gpu import GpuSample

        self.sample_calls += 1
        return [
            GpuSample(index=i, ts=ts, **self.readings[i])
            for i in range(self.n_gpus)
            if i not in self.silent
        ]

    def processes(self) -> dict[int, list[Any]]:
        self.process_calls += 1
        return {i: list(p) for i, p in self.procs.items()}


def gpu_proc(
    pid: int,
    user: str | None,
    *,
    used_mib: int = 512,
    name: str = "python3",
    cmdline: str | None = None,
    kind: str = "compute",
) -> Any:
    """Будує GpuProcess (§10) для фальшивого бекенда."""
    from gpu_manager.gpu import GpuProcess

    return GpuProcess(
        pid=pid,
        user=user,
        name=name,
        cmdline=cmdline if cmdline is not None else f"python3 train.py --run {pid}",
        used_mib=used_mib,
        kind=kind,
    )


# --- Сервіс ---------------------------------------------------------------------------------------


class Env:
    """Один «запуск» сервісу: конфіг, бекенд, годинник і менеджер з build_manager (§10).

    restart() будує новий менеджер над тими самими конфігом, бекендом і годинником — так
    тести моделюють перезапуск сервісу (§4.12, §5).
    """

    def __init__(self, cfg: Any, backend: FakeBackend, clock: FakeClock) -> None:
        from gpu_manager.app import build_manager

        self.cfg = cfg
        self.backend = backend
        self.clock = clock
        self.manager = build_manager(cfg, backend, clock=clock)

    @property
    def data_dir(self) -> Path:
        return Path(self.cfg.data_dir)

    def restart(self) -> Env:
        return Env(self.cfg, self.backend, self.clock)

    def client(self, base_url: str = BASE_URL) -> Any:
        """TestClient над build_app (§10); використовувати як контекстний менеджер (lifespan)."""
        from starlette.testclient import TestClient

        from gpu_manager.app import build_app

        return TestClient(build_app(self.cfg, self.manager), base_url=base_url)

    def mcp(self, guide: Callable[[], str] | None = None) -> Any:
        """MCPServer з build_mcp (§10) для виклику інструментів у процесі.

        guide — функція без аргументів для gpu_guide; None — build_mcp викликається без guide взагалі,
        щоб перевірити саме типову поведінку, а не явний guide=None.
        """
        from gpu_manager.mcp_tools import build_mcp

        if guide is None:
            return build_mcp(self.manager, self.cfg.display_timezone)
        return build_mcp(self.manager, self.cfg.display_timezone, guide=guide)

    def tick_for(
        self,
        seconds: int,
        step: int = 1,
        before_tick: Callable[[float], None] | None = None,
    ) -> None:
        """Просуває годинник кроками step секунд і після кожного кроку робить tick().

        before_tick(now) викликається перед кожним tick() — щоб змінити показники бекенда під час.
        """
        for _ in range(seconds // step):
            self.clock.advance(step)
            if before_tick is not None:
                before_tick(self.clock())
            self.manager.tick()


# --- Фікстури -------------------------------------------------------------------------------------


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    path = tmp_path / "etc"
    path.mkdir()
    return path


@pytest.fixture
def write_config(config_dir: Path) -> Callable[..., Path]:
    """Пише tmp_path/etc/config.json: з перевизначень overrides або готовий документ doc."""

    def _write(overrides: dict[str, Any] | None = None, *, doc: dict[str, Any] | None = None) -> Path:
        if doc is None:
            doc = config_doc(overrides)
        path = config_dir / "config.json"
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        _precreate_data_dir(config_dir, doc)
        return path

    return _write


@pytest.fixture
def make_env(
    write_config: Callable[..., Path], config_dir: Path, backend: FakeBackend, clock: FakeClock
) -> Callable[..., Env]:
    """Фабрика сервісу з конфігом, у якому застосовано перевизначення overrides.

    data_dir — абсолютний шлях у tmp_path: навіть реалізація, що помилково рахує відносний шлях від
    поточної теки, не зачепить справжню data/ репозиторію (§2.8 перевіряє test_config окремо).
    """

    def _make(overrides: dict[str, Any] | None = None) -> Env:
        from gpu_manager.config import load_config

        settings: dict[str, Any] = {"paths.data_dir": str(config_dir / "data")}
        settings.update(overrides or {})
        cfg = load_config(write_config(settings))
        return Env(cfg, backend, clock)

    return _make


@pytest.fixture
def env(make_env: Callable[..., Env]) -> Env:
    return make_env()


# --- HTTP (§7.2) ------------------------------------------------------------------------------------


def reserve(client: Any, gpu: Any, user: Any, **extra: Any) -> Any:
    """POST /api/reserve з тілом {gpu, user, ...extra}; hours=None дає JSON null."""
    return client.post("/api/reserve", json={"gpu": gpu, "user": user, **extra})


def release(client: Any, gpu: Any, user: Any, **extra: Any) -> Any:
    """POST /api/release з тілом {gpu, user, ...extra}."""
    return client.post("/api/release", json={"gpu": gpu, "user": user, **extra})


def ok_json(resp: Any, what: str = "request") -> Any:
    """Перевіряє HTTP 200 і повертає розібране JSON-тіло."""
    assert resp.status_code == 200, f"{what}: expected HTTP 200, got {resp.status_code}: {resp.text[:400]}"
    return resp.json()


def refusal(resp: Any, code: str) -> dict[str, Any]:
    """Перевіряє відмову §7.2: HTTP 400 і code у тілі; повертає тіло."""
    assert resp.status_code == 400, (
        f"expected HTTP 400 with code {code!r}, got HTTP {resp.status_code}: {resp.text[:400]}"
    )
    try:
        body = resp.json()
    except ValueError:
        pytest.fail(f"expected a JSON refusal body with code {code!r}, got non-JSON: {resp.text[:400]}")
    assert isinstance(body, dict) and body.get("code") == code, (
        f"expected refusal code {code!r}, got body {body!r}"
    )
    return body


def overview(client: Any) -> dict[str, Any]:
    return ok_json(client.get("/api/overview"), "GET /api/overview")


def card(client: Any, gpu: int) -> dict[str, Any]:
    """Карта з індексом gpu зі списку gpus відповіді /api/overview."""
    gpus = overview(client)["gpus"]
    for item in gpus:
        if item.get("index") == gpu:
            return item
    raise AssertionError(f"gpu {gpu} missing from /api/overview: indexes {[g.get('index') for g in gpus]}")


def reservations(client: Any) -> dict[int, Any]:
    """Бронювання всіх карт: {index: reservation або None}."""
    return {item.get("index"): item.get("reservation") for item in overview(client)["gpus"]}


def journal(client: Any, limit: int | None = None) -> list[dict[str, Any]]:
    params = {} if limit is None else {"limit": limit}
    data = ok_json(client.get("/api/journal", params=params), "GET /api/journal")
    assert isinstance(data, list), f"/api/journal: expected a JSON array, got {type(data).__name__}"
    return data


# --- MCP у процесі (§10) --------------------------------------------------------------------------


def mcp_call(server: Any, name: str, args: dict[str, Any] | None = None) -> Any:
    """await server.call_tool(name, args) через anyio.run (§10)."""
    import anyio

    async def _call() -> Any:
        return await server.call_tool(name, dict(args or {}))

    return anyio.run(_call)


def mcp_text(result: Any) -> str:
    """JSON-рядок відповіді інструмента: result.content[0].text (§10)."""
    return result.content[0].text


def mcp_json(server: Any, name: str, args: dict[str, Any] | None = None) -> Any:
    return json.loads(mcp_text(mcp_call(server, name, args)))


def mcp_refusal(server: Any, name: str, args: dict[str, Any] | None = None) -> str:
    """Викликає інструмент, що має відмовити; повертає str(ToolError) (§10)."""
    from mcp.server.mcpserver.exceptions import ToolError

    try:
        result = mcp_call(server, name, args)
    except ToolError as exc:
        return str(exc)
    try:
        shown = mcp_text(result)
    except (AttributeError, IndexError):
        shown = repr(result)
    pytest.fail(f"{name}({args!r}): expected ToolError, got a result: {shown[:400]}")


# --- Текст і час ----------------------------------------------------------------------------------


def has_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC.search(text))


def local_minute(ts: float) -> str:
    """Рядок часу MCP (§8): YYYY-MM-DD HH:MM у поясі display_timezone."""
    return datetime.fromtimestamp(ts, ZoneInfo(TZ_NAME)).strftime("%Y-%m-%d %H:%M")


def is_local_minute(value: Any) -> bool:
    return isinstance(value, str) and bool(_LOCAL_MINUTE.match(value))


# --- Фаза 2: моделі HuggingFace (SPEC-GPU-002) -------------------------------------------------------------


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """HOME — тека в tmp_path: типовий models.hf_home (<домашня тека>/hf-cache, SPEC-GPU-002 §2.1)
    конфігу без розділу models не вказує на справжній кеш HF сервера."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture
def hf_home(tmp_path: Path) -> Path:
    """Кеш HF тестів (models.hf_home) у tmp_path з порожньою текою hub (SPEC-GPU-002 §5.3, §10)."""
    path = tmp_path / "hf-cache"
    (path / "hub").mkdir(parents=True)
    return path


@pytest.fixture
def fake_hub() -> Any:
    """Підроблений HubClient (SPEC-GPU-002 §4, §10) зі стандартним набором репозиторіїв."""
    from .model_fakes import standard_hub

    return standard_hub()


@pytest.fixture
def fake_spawn() -> Any:
    """Підроблений spawn (= subprocess.Popen, SPEC-GPU-002 §5.2, §10)."""
    from .model_fakes import FakeSpawn

    return FakeSpawn()


@pytest.fixture
def make_models_env(
    write_config: Callable[..., Path],
    config_dir: Path,
    backend: FakeBackend,
    clock: FakeClock,
    hf_home: Path,
    fake_hub: Any,
    fake_spawn: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., Any]:
    """Фабрика сервісу з моделями (model_fakes.ModelsEnv) над конфігом з розділом models.

    overrides — перевизначення налаштувань (як у make_env); token — значення функції токена HF
    (None — токена немає); card_sizes — обсяги карт у MiB для card_mib().
    HF_TOKEN прибирається з оточення тестового процесу: токен процесу завантаження має братися лише
    з функції токена (§5.2), а не з оточення того, хто запустив тести.
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def _make(
        overrides: dict[str, Any] | None = None,
        *,
        token: str | None = None,
        card_sizes: list[int] | None = None,
    ) -> Any:
        from gpu_manager.config import load_config

        from .model_fakes import ModelsEnv, models_settings

        settings: dict[str, Any] = {"paths.data_dir": str(config_dir / "data")}
        settings.update(models_settings(hf_home))
        settings.update(overrides or {})
        cfg = load_config(write_config(settings))
        return ModelsEnv(cfg, backend, clock, fake_hub, fake_spawn, token=token, card_sizes=card_sizes)

    return _make


@pytest.fixture
def models_env(make_models_env: Callable[..., Any]) -> Any:
    return make_models_env()


# --- Фази 3–4: моделі на vLLM і доступ агентів (SPEC-GPU-003) ------------------------------------------------------


@pytest.fixture
def make_runner_env(make_models_env: Callable[..., Any], tmp_path: Path) -> Callable[..., Any]:
    """Фабрика сервісу з ModelRunner і PromptStore (runner_fakes.RunnerEnv) над сервісом з моделями фази 2.

    overrides — перевизначення налаштувань поверх налаштувань vllm тестів (runner_fakes.runner_settings);
    ready — repo, що лежать у кеші HF готовими (типово runner_fakes.READY_REPOS).
    Підробки launcher / probe / port_free — нові для кожного виклику фабрики.
    """

    def _make(overrides: dict[str, Any] | None = None, *, ready: Any = None) -> Any:
        from .runner_fakes import READY_REPOS, RunnerEnv, runner_settings

        settings = runner_settings(tmp_path)
        settings.update(overrides or {})
        env = RunnerEnv.build(make_models_env(settings))
        for repo in READY_REPOS if ready is None else ready:
            env.put_ready(repo)
        return env

    return _make


@pytest.fixture
def runner_env(make_runner_env: Callable[..., Any]) -> Any:
    return make_runner_env()


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> Any:
    """HTTP-сервер на 127.0.0.1 замість vLLM (runner_fakes.StubUpstream); закривається після тесту.

    Змінні проксі прибираються: запит шлюзу до 127.0.0.1 не повинен піти через проксі оточення.
    """
    from .runner_fakes import StubUpstream

    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    stub = StubUpstream()
    yield stub
    stub.close()


@pytest.fixture
def proxied_env(make_runner_env: Callable[..., Any], upstream: Any) -> Any:
    """Сервіс, у якому tiny-llm у стані running на порту upstream (vllm.port_range = [порт, порт])."""
    from .runner_fakes import DEFAULT_NAME

    env = make_runner_env({"vllm.port_range": [upstream.port, upstream.port]})
    env.start()
    env.ensure_running(DEFAULT_NAME)
    return env
