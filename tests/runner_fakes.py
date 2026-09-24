"""Підробки й помічники приймальних тестів SPEC-GPU-003 (фази 3–4: моделі на vLLM і доступ агентів).

Навіщо файл: тести фаз 3–4 пишуться наосліп, лише з docs/specs/SPEC-GPU-003-phases-3-4.md. Її §2 дає шви
ModelRunner: launcher (start / stop / active юніта), probe (health / metrics порту), port_free(port) і clock.
Тут зібрано підробки цих швів, збирання ModelRunner і PromptStore над сервісом з моделями фази 2
(model_fakes.ModelsEnv), справжні тексти помилок vLLM для правил автоповтору (§2.8) і крихітний HTTP-сервер
на 127.0.0.1 замість vLLM — для перевірки пересилання шлюзу (§3).

Справжнього vLLM, systemd і мережі немає: юніти «живуть» у FakeLauncher, «вивід vLLM» тест сам дописує в
лог моделі data_dir/servers/<name>.log (§2.6), карти — фальшивий бекенд фази 1 з керованою зайнятою пам'яттю.
Кеш HF і data_dir — лише в tmp_path.

Імпорти gpu_manager — ліниві (усередині функцій), як і в conftest.py: зламаний модуль фаз 3–4 не валить
збирання тестів фаз 1–2, а зламаний gpu_manager.prompts не валить тестів ModelRunner (і навпаки).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

from .conftest import BASE_URL, MODEL_PORTS, mcp_json
from .model_fakes import REPO, ModelsEnv, journal_entries, model_dir, put_snapshot, sha_of

# --- Сталі ------------------------------------------------------------------------------------------
# Усі значення нижче обрані тестами (вхідні дані конфігу й підробок) або виведені зі специфікації
# арифметикою, показаною в коментарі; з коду не взято нічого.

GPU = 2  # основна карта тестів (карти 2 і 3 — ті, що в репозиторії відведені під перевірки)
GPU_B = 3  # друга карта
PORT_FIRST = MODEL_PORTS[0]  # 8000 — найменший порт vllm.port_range тестового конфігу (§2.2)
UNIT_PREFIX = "gm-model-"  # юніт моделі — gm-model-<name>.service (§2)
UNIT_SUFFIX = ".service"  # суфікс юніта явний: назви x і x.service — різні юніти (§2)
MARKER = "=== gpu-manager start"  # рядок-маркер кожного старту в лозі моделі (§2.6)
START_TIMEOUT_S = 600  # vllm.start_timeout_s тестового конфігу; ≠ типових 900 (§1), щоб було видно, звідки значення
DEFAULT_NAME = "tiny-llm"  # типова назва для REPO "acme/tiny-llm" (§2.3)
MIXED_REPO = "Acme/Qwen3-Coder-8B"  # repo з великими літерами — для правила «малими літерами» (§2.3)
MIXED_NAME = "qwen3-coder-8b"
ODD_REPO = "acme/_Tiny_LLM_"  # repo з «_» на краях — правило «без -, ., _ на краях» (§2.3)
ODD_NAME = "tiny_llm"  # малими літерами, «_» з обох країв знято; «_» усередині дозволений
NOT_LOCAL_REPO = "acme/not-downloaded"  # моделі немає в кеші HF → model_not_local (§2.1)
READY_REPOS = (REPO, MIXED_REPO, ODD_REPO)  # моделі, що лежать у кеші тестів готовими (state ready, SPEC-GPU-002 §5.4.1)
READY_FILES = ("config.json", "model.safetensors", "tokenizer.json")
PROFILE_FIELDS = {"fraction", "max_model_len", "extra_args", "gpu", "updated", "attempts"}  # запис профілю (§2.7)

# Вільна частка карти 46068 MiB за §2.4: min((total − used − 256 MiB)/total, 1 − Σ), донизу до 0.01. Неявна частка —
# ще й не більше vllm.default_fraction (типові 0.95, §1): ця стеля обрізає лише майже порожню карту, нижче 0.95 частку
# задає пам'ять. Явну частку обмежує лише вільна.
NO_DEFAULT_CAP = {"vllm.default_fraction": 1.0}  # верхня межа §1: частку обрізають лише пам'ять і Σ (тести не про типову частку)
FRACTION_EMPTY = 0.95  # неявна, used 0: min(45812/46068 = 0.9944, default_fraction 0.95) → 0.95
FRACTION_EMPTY_NO_CAP = 0.99  # used 0 при NO_DEFAULT_CAP: 45812/46068 = 0.9944 → 0.99
USED_MID_MIB = 10_000
FRACTION_USED_MID = 0.77  # used 10000:  35812/46068 = 0.7774 → 0.77 (звичайне округлення дало б 0.78)
USED_HIGH_MIB = 40_000
FRACTION_USED_HIGH = 0.12  # used 40000:  5812/46068 = 0.1262 → 0.12
# §2.5 при used 10000: дозволено total·f ≤ free − 256 = 35812 MiB, тобто f ≤ 0.7774; 0.78·46068 = 35933 > 35812.
FRACTION_OVER_MID = 0.78
LADDER_BELOW_DEFAULT = (0.92, 0.9)  # щаблі §2.8 (1.0, 0.95, 0.92, 0.9) нижче типової частки 0.95 порожньої карти
LADDER_BELOW_NO_CAP = (0.95, 0.92, 0.9)  # щаблі §2.8 нижче частки 0.99 порожньої карти при NO_DEFAULT_CAP

# --- Рядки логу vLLM (§2.8) -------------------------------------------------------------------------------
# Справжні тексти помилок vLLM / PyTorch з префіксом рівня ERROR у форматі логу vLLM: що саме §2.8 вважає
# «рядком помилки», специфікація не каже (прогалина), тож кожен рядок-тригер тут однозначно помилковий. Виняток —
# pydantic-форма «контекст більший за межу моделі»: її тригер навмисно не в рядку помилки (§2.8 шукає його всюди).
# Кожен рядок (багаторядкова pydantic-форма — уся група) містить рівно один тригер §2.8 — інакше «перший збіг»
# був би неоднозначним.

_ERR = "ERROR 09-23 21:00:00 [core.py:708] "
_INFO = "INFO 09-23 21:00:00 [api_server.py:1024] "


def line_estimated_len(n: int) -> str:
    """Помилка vLLM «estimated maximum model length is N» (без слів «KV cache» і OOM)."""
    return _ERR + (
        "ValueError: To serve at least one request with the model's max seq len (131072), 16.00 GiB is needed. "
        f"Based on the available memory, the estimated maximum model length is {n}. "
        "Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine."
    )


def line_mamba(seqs: int, n: int) -> str:
    """Справжня помилка vLLM «max_num_seqs (seqs) exceeds available Mamba cache blocks (n)» (§2.8)."""
    return _ERR + (
        f"ValueError: max_num_seqs ({seqs}) exceeds available Mamba cache blocks ({n}). Each decode sequence "
        "requires one Mamba cache block, so CUDA graph capture cannot proceed. "
        f"Please lower max_num_seqs to at most {n} or increase gpu_memory_utilization."
    )


def line_ctx_over_model(user_len: int, n: int) -> str:
    """Помилка vLLM «greater than the derived max_model_len (<ключ>=N» (§2.8): заданий контекст більший за межу
    моделі. Рядок дослівний, без префікса рівня: рядок помилки — завдяки «ValueError»."""
    return (
        f"ValueError: User-specified max_model_len ({user_len}) is greater than the derived max_model_len "
        f"(max_position_embeddings={n} or model_max_length=None in model's config.json)."
    )


def lines_ctx_over_model_pydantic(user_len: int, n: int) -> list[str]:
    """Та сама помилка «greater than the derived max_model_len (<ключ>=N» у форматі vLLM 0.30 (§2.8): pydantic-виняток
    на кілька рядків. Тригер — у рядку «  Value error, …», де немає ні ERROR, ні …Error, ні …Exception, ні Traceback,
    ні fatal, тобто це не рядок помилки: правило має знайти тригер у будь-якому рядку логу останнього старту.
    За user_len = 8192, n = 4096 — дослівний вивід vLLM 0.30."""
    return [
        "pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig",
        f"  Value error, User-specified max_model_len ({user_len}) is greater than the derived max_model_len "
        f"(max_position_embeddings={n} or model_max_length=None in model's config.json). To allow overriding this "
        "maximum, set the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1. [type=value_error, input_value=ArgsKwargs((), "
        f"{{'max_model_len': {user_len}}}), input_type=ArgsKwargs]",
        "    For further information visit https://errors.pydantic.dev/2.13/v/value_error",
    ]


# Обидві форми помилки «контекст більший за межу моделі» (§2.8) як рядки логу, за (user_len, n): однорядкова
# «ValueError: …» і pydantic-форма vLLM 0.30. Ключ — id параметризації тестів.
CTX_OVER_MODEL_FORMS: dict[str, Callable[[int, int], list[str]]] = {
    "valueerror-line": lambda user_len, n: [line_ctx_over_model(user_len, n)],
    "pydantic-vllm-0.30": lines_ctx_over_model_pydantic,
}


LINE_LESS_THAN_DESIRED = _ERR + (
    "ValueError: Free memory on device (20.5/44.35 GiB) on startup is less than desired GPU memory utilization "
    "(0.99, 43.9 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes."
)
LINE_OOM = _ERR + (
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. "
    "GPU 0 has a total capacity of 44.35 GiB of which 1.10 GiB is free."
)
LINE_KV_CACHE = _ERR + (
    "ValueError: The model's max seq len (16384) is larger than the maximum number of tokens that can be stored "
    "in KV cache (9000). Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing "
    "the engine."
)
LINE_NO_CACHE_BLOCKS = _ERR + (
    "ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when "
    "initializing the engine."
)
# Чотири ознаки помилки компіляції §2.8 — по одній на рядок.
COMPILE_LINES = {
    "ninja": _ERR + "RuntimeError: Ninja build failed while building extension 'fused_moe_kernels'",
    "nvcc": _ERR + "nvcc fatal   : Unsupported gpu architecture 'compute_120a'",
    "inductor": _ERR + "torch._inductor.exc.InductorError: CalledProcessError: Command 'gcc' returned non-zero exit status 1.",
    "dynamo": _ERR + "torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised: RuntimeError: triton compilation failed",
}
# Жодне правило автоповтору не підходить; підказка — hint_remote_code («trust_remote_code=True», §2).
LINE_REMOTE_CODE = _ERR + (
    "ValueError: The repository acme/tiny-llm contains custom code which must be executed to correctly load the "
    "model. Please pass the argument `trust_remote_code=True` to allow custom code to be run."
)
# Той самий тригер підказки без «ERROR»: рядок помилки лише завдяки «…Error» (§2.8).
LINE_REMOTE_CODE_BARE = "RuntimeError: Loading acme/tiny-llm requires custom code; set `trust_remote_code=True` to allow it."
# Рядок INFO з тим самим текстом-тригером: не рядок помилки (немає ERROR, …Error, …Exception, Traceback, fatal).
LINE_INFO_REMOTE_CODE = _INFO + "Tokenizer of acme/tiny-llm asks for trust_remote_code=True; using the slow tokenizer"
# Жодне правило автоповтору й жодна названа підказка не підходять.
LINE_UNRECOGNIZED =_ERR + "RuntimeError: Engine core initialization failed. See root cause above."
LINE_INFO = _INFO + "Starting vLLM API server 0 on http://127.0.0.1:8000"


def info_line(i: int) -> str:
    """i-й звичайний (не помилковий) рядок логу vLLM."""
    return _INFO + f"Loading weights: shard {i} of 10 loaded"


# --- argv ------------------------------------------------------------------------------------------------------


def flag_value(argv: Any, flag: str) -> str | None:
    """Значення після прапорця flag в argv; None, якщо прапорця немає."""
    tokens = [str(a) for a in argv]
    if flag not in tokens:
        return None
    i = tokens.index(flag)
    return tokens[i + 1] if i + 1 < len(tokens) else None


def fraction_of(argv: Any) -> float | None:
    """--gpu-memory-utilization з argv як число (4 знаки); None, якщо прапорця немає."""
    value = flag_value(argv, "--gpu-memory-utilization")
    return None if value is None else round(float(value), 4)


def max_len_of(argv: Any) -> int | None:
    """--max-model-len з argv як ціле; None, якщо прапорця немає (§2.4: типово не задається)."""
    value = flag_value(argv, "--max-model-len")
    return None if value is None else int(value)


def max_num_seqs_values(argv: Any) -> list[str | None]:
    """Значення всіх входжень прапорця max-num-seqs в argv, по порядку: обидва написання (--max-num-seqs,
    --max_num_seqs) і обидві форми («прапорець N», «прапорець=N») — §2.6.

    Список, а не одне значення: тести §2.6 і §2.8 перевіряють, що прапорець у argv рівно один.
    """
    flags = ("--max-num-seqs", "--max_num_seqs")
    tokens = [str(a) for a in argv]
    values: list[str | None] = []
    for i, token in enumerate(tokens):
        if token in flags:
            values.append(tokens[i + 1] if i + 1 < len(tokens) else None)
        elif token.split("=", 1)[0] in flags and "=" in token:
            values.append(token.split("=", 1)[1])
    return values


def port_of(argv: Any) -> int | None:
    value = flag_value(argv, "--port")
    return None if value is None else int(value)


def normalize_argv(argv: Any) -> list[Any]:
    """argv для порівняння з очікуваним за §2.6.

    Усі елементи — рядки; шлях знімка (третій елемент) — розв'язаний (tmp_path може йти через symlink);
    значення --gpu-memory-utilization — число з 4 знаками («0.5» і «0.50» — одна частка). Решта — дослівно.
    """
    tokens: list[Any] = [str(a) for a in argv]
    if len(tokens) > 2:
        tokens[2] = str(Path(tokens[2]).resolve())
    if "--gpu-memory-utilization" in tokens:
        i = tokens.index("--gpu-memory-utilization") + 1
        if i < len(tokens):
            try:
                tokens[i] = round(float(tokens[i]), 4)
            except ValueError:
                pass
    return tokens


# --- Підробки швів §2 --------------------------------------------------------------------------------------------


@dataclass
class StartCall:
    """Один виклик launcher.start(unit, argv, env, log) (§2)."""

    unit: str
    argv: list[str]
    env: dict[str, str]
    log: Any


class FakeLauncher:
    """Підробка launcher (§2): start / stop / active юніта без systemd.

    start() робить юніт живим і запам'ятовує виклик (calls); stop() пише ім'я юніта в stops і гасить його
    (повертає None — не невдача); для юніта з stop_fails stop() повертає False, юніт лишається живим.
    active() для юніта з unknown повертає None (стан невідомий).
    kill(unit) — керування з тесту: юніт помер сам (як vLLM, що впав на старті, або після перезавантаження).
    before_alive(unit) — керування з тесту: викликається в start() після запису виклику, але до того, як юніт
    стане живим (stop, що прийшов посеред запуску, §2.9). on_stop(unit) — після вдалого stop() (звільнення
    пам'яті карти, §2.10a).
    Переживає RunnerEnv.restart(): юніти systemd живуть незалежно від менеджера (§2.10).
    """

    def __init__(self) -> None:
        self.calls: list[StartCall] = []
        self.stops: list[str] = []
        self.alive: set[str] = set()
        self.unknown: set[str] = set()  # юніти, для яких active() → None (стан невідомий, §2 заголовок)
        self.stop_fails: set[str] = set()  # юніти, для яких stop() → False (невдача, §2.9)
        self.before_alive: Callable[[str], None] | None = None
        self.on_stop: Callable[[str], None] | None = None

    def start(self, unit: Any, argv: Any, env: Any, log: Any) -> None:
        call = StartCall(
            unit=str(unit),
            argv=[str(a) for a in argv],
            env={str(k): str(v) for k, v in dict(env or {}).items()},
            log=log,
        )
        self.calls.append(call)
        if self.before_alive is not None:
            self.before_alive(call.unit)
        self.alive.add(call.unit)

    def stop(self, unit: Any) -> bool | None:
        self.stops.append(str(unit))
        if str(unit) in self.stop_fails:
            return False
        self.alive.discard(str(unit))
        if self.on_stop is not None:
            self.on_stop(str(unit))
        return None

    def active(self, unit: Any) -> bool | None:
        return None if str(unit) in self.unknown else str(unit) in self.alive

    # --- керування з тесту ---

    def kill(self, unit: str) -> None:
        """Юніт перестав бути активним без виклику stop()."""
        self.alive.discard(unit)

    def starts_of(self, unit: str) -> list[StartCall]:
        return [c for c in self.calls if c.unit == unit]


class FakeProbe:
    """Підробка probe (§2): health(port) — чи порт у healthy; metrics(port) — нульове навантаження."""

    def __init__(self) -> None:
        self.healthy: set[int] = set()

    def health(self, port: Any) -> bool:
        return int(port) in self.healthy

    def metrics(self, port: Any) -> dict[str, Any]:
        return {"running": 0, "waiting": 0, "kv_usage": 0.0}


class FakePortFree:
    """Підробка port_free(port) (§2.2): порт вільний, якщо його немає в busy (зайнятий «чужим» процесом)."""

    def __init__(self) -> None:
        self.busy: set[int] = set()

    def __call__(self, port: Any) -> bool:
        return int(port) not in self.busy


# --- Конфіг ------------------------------------------------------------------------------------------------------


def runner_settings(tmp_path: Path) -> dict[str, Any]:
    """Налаштування розділу vllm тестового конфігу (§1): bin і cuda_home — у tmp_path, start_timeout_s — 600.

    Файл bin і тека cuda_home справді створюються: чи перевіряє сервіс їхню наявність, §1 не каже.
    Файл bin ніхто не запускає — launcher підроблений.
    """
    vllm_bin = tmp_path / "vllm-venv" / "bin" / "vllm"
    vllm_bin.parent.mkdir(parents=True, exist_ok=True)
    vllm_bin.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    vllm_bin.chmod(0o755)
    cuda_home = tmp_path / "cuda"
    (cuda_home / "bin").mkdir(parents=True, exist_ok=True)
    return {
        "vllm.bin": str(vllm_bin),
        "vllm.cuda_home": str(cuda_home),
        "vllm.start_timeout_s": START_TIMEOUT_S,
    }


# --- Сервіс з ModelRunner і PromptStore --------------------------------------------------------------------------


class RunnerEnv:
    """Сервіс з моделями фази 2 (ModelsEnv) + ModelRunner (§2) і PromptStore (§3) над ним.

    ModelRunner(cfg, store, journal, models, gpus, launcher, probe, port_free, clock) — позиційно, у порядку §2;
    store / journal — manager.store / manager.journal_log (як у ModelStore, SPEC-GPU-002 §9).
    PromptStore(users, store, journal, clock) — так само.

    Зв'язки, які в сервісі робить build_runner(cfg, manager, models) (§2), тут робляться явно при створенні
    runner: models.in_use = runner.in_use, manager.models_on = runner.on_gpu. build_runner не бере швів
    (launcher, probe, port_free, clock), тож для тестів непридатний — прогалина специфікації.

    runner і prompts створюються ліниво, при першому зверненні.
    launcher / probe / port_free переживають restart(): так тест моделює перезапуск менеджера, під час
    якого юніти systemd живуть далі (§2.10).
    """

    def __init__(self, models_env: ModelsEnv, launcher: FakeLauncher, probe: FakeProbe, port_free: FakePortFree) -> None:
        self.models_env = models_env
        self.launcher = launcher
        self.probe = probe
        self.port_free = port_free
        self._runner: Any = None
        self._prompts: Any = None
        # Телеметрія карт (зайнята пам'ять) потрібна вже першому start() — формула частки §2.4.
        self.manager.tick()

    @classmethod
    def build(cls, models_env: ModelsEnv) -> RunnerEnv:
        return cls(models_env, FakeLauncher(), FakeProbe(), FakePortFree())

    # --- складові ---

    @property
    def cfg(self) -> Any:
        return self.models_env.cfg

    @property
    def clock(self) -> Any:
        return self.models_env.clock

    @property
    def backend(self) -> Any:
        return self.models_env.backend

    @property
    def manager(self) -> Any:
        return self.models_env.manager

    @property
    def store(self) -> Any:
        """ModelStore фази 2."""
        return self.models_env.store

    @property
    def hf_home(self) -> Path:
        return Path(self.cfg.hf_home)

    @property
    def data_dir(self) -> Path:
        return Path(self.cfg.data_dir)

    @property
    def runner(self) -> Any:
        if self._runner is None:
            from gpu_manager.runner import ModelRunner

            manager = self.manager
            self._runner = ModelRunner(
                self.cfg,
                manager.store,
                manager.journal_log,
                self.store,
                manager,
                self.launcher,
                self.probe,
                self.port_free,
                self.clock,
            )
            self.store.in_use = self._runner.in_use
            manager.models_on = self._runner.on_gpu
        return self._runner

    @property
    def prompts(self) -> Any:
        if self._prompts is None:
            from gpu_manager.prompts import PromptStore

            manager = self.manager
            self._prompts = PromptStore(self.cfg.users, manager.store, manager.journal_log, self.clock)
        return self._prompts

    def restart(self) -> RunnerEnv:
        """Новий менеджер, ModelStore, ModelRunner і PromptStore над тими самими конфігом, data_dir, кешем HF
        і юнітами (ті самі підробки launcher / probe / port_free)."""
        return RunnerEnv(self.models_env.restart(), self.launcher, self.probe, self.port_free)

    # --- дії ---

    def start(self, repo: str = REPO, gpu: Any = GPU, user: str = "alice", **kwargs: Any) -> Any:
        """runner.start(repo, gpu, user, **kwargs) (§2.1); kwargs — fraction, max_model_len, extra_args, name."""
        return self.runner.start(repo, gpu, user, **kwargs)

    def step(self) -> None:
        """Один крок станів серверів (starting → running, автоповтор, тайм-аут старту — §2.7–2.8).

        §2 не називає методу, що просуває стани, — прогалина. За зразком ModelStore.poll()
        (SPEC-GPU-002 §5.1.7) тести викликають ModelRunner.poll(); усі виклики — лише тут.
        """
        poll = getattr(self.runner, "poll", None)
        if not callable(poll):
            pytest.fail(
                "ModelRunner.poll() is missing: SPEC-GPU-003 §2 names no method that advances server states "
                "(starting -> running, auto-retry, start timeout); these tests assume poll() by analogy with "
                "ModelStore.poll() of SPEC-GPU-002 §5.1.7"
            )
        poll()

    def set_used(self, gpu: int, mib: int) -> None:
        """Зайнята пам'ять карти gpu — mib MiB; менеджер одразу робить tick() (свіжа телеметрія)."""
        self.backend.set_reading(gpu, memory_used_mib=mib)
        self.manager.tick()

    def reserve(self, gpu: int, user: str) -> None:
        """Бронювання карти через інструмент gpu_reserve фази 1 (SPEC-GPU-001 §8) того самого менеджера."""
        mcp_json(self.models_env.base.mcp(), "gpu_reserve", {"gpu": gpu, "user": user, "purpose": "phase 3 test"})

    def put_ready(self, repo: str) -> None:
        """Модель repo лежить у кеші HF готовою: знімок ревізії sha_of(repo) і refs/main."""
        put_snapshot(self.hf_home, repo, list(READY_FILES))

    def snapshot(self, repo: str) -> Path:
        """<hf_home>/hub/models--<org>--<name>/snapshots/<rev> готової моделі (§2.6)."""
        return model_dir(self.hf_home, repo) / "snapshots" / sha_of(repo)

    # --- сервери ---

    def servers(self) -> list[Any]:
        return list(self.runner.servers())

    def names(self) -> list[Any]:
        return [s.get("name") for s in self.servers()]

    def server(self, name: str) -> Any:
        for item in self.servers():
            if item.get("name") == name:
                return item
        raise AssertionError(f"servers(): expected a server named {name!r}, got names {self.names()!r}")

    def status(self, name: str) -> Any:
        return self.server(name).get("status")

    @staticmethod
    def unit(name: str) -> str:
        return UNIT_PREFIX + name + UNIT_SUFFIX

    def starts(self, name: str) -> list[StartCall]:
        """Усі виклики launcher.start для юніта моделі name (спроби старту по порядку)."""
        return self.launcher.starts_of(self.unit(name))

    def argv(self, name: str, index: int = -1) -> list[str]:
        calls = self.starts(name)
        assert calls, (
            f"launcher.start: expected a start of unit {self.unit(name)!r}, got units {[c.unit for c in self.launcher.calls]!r}"
        )
        return calls[index].argv

    def port(self, name: str) -> int | None:
        return port_of(self.argv(name))

    def card_uuid(self, gpu: int) -> str:
        """uuid карти gpu з фальшивого бекенда (те саме поле uuid, що в картці /api/overview, SPEC-GPU-001 §7.2) —
        очікуване CUDA_VISIBLE_DEVICES старту на цій карті (§2.6)."""
        infos = self.backend.info()
        for info in infos:
            if info.index == gpu:
                return str(info.uuid)
        raise AssertionError(f"fake backend: expected a card with index {gpu}, got indexes {[i.index for i in infos]!r}")

    def gpu_of(self, call: StartCall) -> Any:
        """Карта старту call за його CUDA_VISIBLE_DEVICES (§2.6): index карти, чий uuid з фальшивого бекенда (номер —
        лише коли uuid порожній) дорівнює значенню змінної.

        Не збіглося ні з однією картою — сире значення (None — змінної немає), щоб повідомлення тесту показало, що
        саме передано vLLM (напр. номер '3' замість uuid карти 3).
        """
        value = call.env.get("CUDA_VISIBLE_DEVICES")
        for info in self.backend.info():
            if value == (str(info.uuid) if info.uuid else str(info.index)):
                return info.index
        return value

    def fractions(self, name: str) -> list[float | None]:
        """--gpu-memory-utilization кожної спроби старту name."""
        return [fraction_of(c.argv) for c in self.starts(name)]

    def max_lens(self, name: str) -> list[int | None]:
        """--max-model-len кожної спроби старту name (None — не задано)."""
        return [max_len_of(c.argv) for c in self.starts(name)]

    def run(self, name: str) -> Any:
        """health порту name відповідає → один крок; повертає статус (§2.7: юніт активний і health → running)."""
        port = self.port(name)
        assert port is not None, f"argv of {name!r}: expected a --port value, got {self.argv(name)!r}"
        self.probe.healthy.add(port)
        self.step()
        return self.status(name)

    def ensure_running(self, name: str) -> None:
        """Передумова тесту: name переходить у running."""
        status = self.run(name)
        assert status == "running", f"precondition: expected {name!r} running once active and healthy, got {status!r}"

    # --- лог моделі й падіння ---

    def log_file(self, name: str) -> Path:
        """Лог моделі data_dir/servers/<name>.log (§2.6)."""
        return self.data_dir / "servers" / f"{name}.log"

    def append_log(self, name: str, lines: Any) -> None:
        """Дописує рядки в лог моделі — так «пише» підроблений vLLM."""
        path = self.log_file(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")

    def markers(self, name: str) -> int:
        """Скільки рядків-маркерів старту (§2.6) у лозі моделі."""
        path = self.log_file(name)
        if not path.exists():
            return 0
        text = path.read_text(encoding="utf-8", errors="replace")
        return sum(1 for line in text.splitlines() if line.startswith(MARKER))

    def die(self, name: str, lines: Any = ()) -> None:
        """vLLM моделі name пише lines у лог і падає: юніт перестає бути активним."""
        self.append_log(name, list(lines))
        self.launcher.kill(self.unit(name))

    def die_and_step(self, name: str, lines: Any) -> None:
        """Падіння з рядками lines, далі кроки, доки runner не перезапустить юніт або не позначить failed.

        Не більше трьох кроків: чи перезапуск відбувається в тому самому кроці, що й виявлення падіння,
        §2.8 не каже, тож тест не прив'язується до цього.
        """
        before = len(self.starts(name))
        self.die(name, lines)
        for _ in range(3):
            self.step()
            if len(self.starts(name)) > before or self.status(name) == "failed":
                return

    def ensure_failed(self, name: str, line: str) -> None:
        """Передумова тесту: name падає з рядком line без автоповтору і стає failed."""
        self.die_and_step(name, [line])
        status = self.status(name)
        assert status == "failed", f"precondition: expected {name!r} failed after {line[:70]!r}..., got {status!r}"

    # --- профілі й журнал ---

    @property
    def profiles_path(self) -> Path:
        """data/model_profiles.json (§2.4)."""
        return self.data_dir / "model_profiles.json"

    def profiles(self) -> dict[str, Any]:
        """Вміст файла профілів (об'єкт за ключем repo); немає файла — {}."""
        path = self.profiles_path
        if not path.exists():
            return {}
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(doc, dict), f"{path}: expected a JSON object keyed by repo, got {type(doc).__name__}"
        return doc

    def write_profile(self, repo: str, profile: dict[str, Any]) -> None:
        """Записує профіль repo у файл профілів (решта записів лишається)."""
        doc = self.profiles()
        doc[repo] = profile
        self.profiles_path.parent.mkdir(parents=True, exist_ok=True)
        self.profiles_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    def journal(self, action: str | None = None) -> list[dict[str, Any]]:
        """Записи журналу (порядок файлу); action — лише записи з цією дією."""
        entries = journal_entries(self.data_dir)
        return entries if action is None else [e for e in entries if e.get("action") == action]

    # --- входи сервісу ---

    def client(self) -> Any:
        """TestClient над build_app(cfg, manager, models, runner, prompts) (§4); як контекстний менеджер."""
        from starlette.testclient import TestClient

        from gpu_manager.app import build_app

        app = build_app(self.cfg, self.manager, models=self.store, runner=self.runner, prompts=self.prompts)
        return TestClient(app, base_url=BASE_URL)

    def mcp(self) -> Any:
        """MCPServer з build_mcp(manager, tz, models=, runner=, prompts=) (§4)."""
        from gpu_manager.mcp_tools import build_mcp

        return build_mcp(
            self.manager, self.cfg.display_timezone, models=self.store, runner=self.runner, prompts=self.prompts
        )


# --- Замість vLLM: HTTP-сервер на 127.0.0.1 (§3) ------------------------------------------------------------------

STUB_ANSWER = "stub answer: 4"
STUB_COMPLETION: dict[str, Any] = {
    "id": "chatcmpl-stub",
    "object": "chat.completion",
    "created": 0,
    "model": DEFAULT_NAME,
    "choices": [{"index": 0, "message": {"role": "assistant", "content": STUB_ANSWER}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
}


def _make_handler(upstream: StubUpstream) -> type[BaseHTTPRequestHandler]:
    """Клас обробника запитів, що пише кожен POST у upstream.requests і відповідає upstream.response."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — сигнатура базового класу
            """Тиша: вивід сервера в stderr тестам не потрібен."""

        def _body(self) -> bytes:
            """Тіло запиту — за Content-Length або chunked (§3: «потік як є» — шлюз може слати потоком)."""
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                parts: list[bytes] = []
                while True:
                    size_line = self.rfile.readline().split(b";")[0].strip()
                    size = int(size_line or b"0", 16)
                    if size == 0:
                        while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                            pass
                        return b"".join(parts)
                    parts.append(self.rfile.read(size))
                    self.rfile.readline()
            return self.rfile.read(int(self.headers.get("Content-Length") or 0))

        def do_POST(self) -> None:
            raw = self._body()
            try:
                payload = json.loads(raw.decode("utf-8"))
            except ValueError:
                payload = None
            upstream.requests.append({"path": self.path, "json": payload})
            data = json.dumps(upstream.response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


class StubUpstream:
    """Крихітний HTTP-сервер на 127.0.0.1 замість vLLM (§3: пересилання на 127.0.0.1:<port>).

    Приймає будь-який POST, запам'ятовує шлях і розібраний JSON тіла в requests і відповідає response
    (типово STUB_COMPLETION; тест може підмінити, напр. відповіддю без тексту — §4 empty_answer).
    Порт вибирає ОС (bind на 0). Сокет слухає вже після конструктора, тож запит, що прийшов раніше, ніж потік
    почав serve_forever, чекає в черзі, а не падає — синхронізації сном немає.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.response: dict[str, Any] = STUB_COMPLETION
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self))
        self.port = int(self.httpd.server_address[1])
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="stub-vllm", daemon=True)
        self._thread.start()

    def last_json(self) -> Any:
        """JSON тіла останнього запиту; запитів не було — помилка тесту з поясненням."""
        assert self.requests, f"stub vLLM on port {self.port}: expected a forwarded request, got none"
        return self.requests[-1]["json"]

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread.join(timeout=5)
