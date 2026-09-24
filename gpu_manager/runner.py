"""Моделі на vLLM (фаза 3): запуск і зупинка на вказаній карті, автопорт, кілька моделей на карті, стан, логи.

Кожна модель — окремий тимчасовий юніт systemd --user (gm-model-<name>.service), а не дочірній процес менеджера:
перезапуск менеджера (напр. після правки коду) не зупиняє моделей. Після перезавантаження машини юнітів
немає — менеджер піднімає моделі, що мали працювати. Порт — найменший вільний з vllm.port_range. Після першого
успішного старту параметри запам'ятовуються як профіль моделі в data/model_profiles.json (читається при кожному
старті — людина чи агент можуть вписати туди своє), і наступний старт без параметрів бере їх. Відомі помилки старту
менеджер виправляє сам, до _MAX_ATTEMPTS спроб; невідомі — агентові: стан failed, підказка і лог.

Під замком — лише стан у пам'яті та запис state.json; systemctl, HTTP і лог-файли — поза ним, інакше повільний
systemctl stop блокував би сторінку й потоки /v1."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Protocol

import httpx

from .config import Config
from .core import GpuManager, require_user
from .inbox import run_group
from .journal import Journal
from .messages import ManagerError
from .models import ModelStore
from .names import check_name, slug
from .store import StateStore

log = logging.getLogger(__name__)

SECTION = "servers"
PROFILES_FILE = "model_profiles.json"
# Скільки разів менеджер сам перезапускає модель з виправленими параметрами, перш ніж віддати агентові.
_MAX_ATTEMPTS = 5
# Драбина частки карти: спершу максимум пам'яті, при нестачі — на щабель нижче.
_FRACTION_LADDER = (1.0, 0.95, 0.92, 0.9)
# Контекст при нестачі пам'яті без поради vLLM: удвічі менше, але не менше цього; не заданий -> це ×4.
_MIN_AUTO_CTX = 2048
# Порада vLLM щодо контексту, менша за це, означає, що пам'яті на KV немає взагалі: треба більша частка.
_MIN_USEFUL_CTX = 256
STARTING, RUNNING, FAILED, STOPPED = "starting", "running", "failed", "stopped"
_UNIT_PREFIX = "gm-model-"
# Прапорці vLLM, які extra_args не можуть задати навіть скорочено (argparse приймає префікси, vLLM — `_` замість
# `-`, а JSON-аргументи — ще й `--прапорець.ключ`): адресою, портом, назвою й пам'яттю керує менеджер (інакше
# облік вільної частки карти хибний); решта відкриває доступ до файлів чи ламає шлюз.
_FORBIDDEN_FLAGS = ("--host", "--port", "--uds", "--served-model-name", "--gpu-memory-utilization",
                    "--kv-cache-memory-bytes", "--num-gpu-blocks-override", "--config",
                    "--allowed-local-media-path", "--allowed-media-domains", "--api-key", "--root-path",
                    "--ssl-keyfile", "--ssl-certfile", "--ssl-ca-certs", "--middleware")
# Менша частка карти безглузда навіть для крихітної моделі (ваги + CUDA-графи): краще відмова, ніж певне падіння.
_MIN_FRACTION = 0.05
# Запас пам'яті на карті понад частку моделі, MiB: драйвер і дрібні виділення, щоб vLLM не впав на старті.
_MEM_SLACK_MIB = 256
# Кратність, до якої округлюється автоматичний max_model_len: vLLM працює блоками KV-кешу.
_CTX_ROUND = 1024
_HTTP_TIMEOUT_S = 1.0
_START_MARK = "=== gpu-manager start "
# Типові помилки старту vLLM у лозі -> код підказки (тексти — у messages.py). Порядок важливий: перший збіг.
_HINTS: tuple[tuple[str, str], ...] = (
    ("larger than the available KV cache memory", "hint_kv_too_small"),
    ("larger than the maximum number of tokens that can be stored in KV cache", "hint_kv_too_small"),
    ("No available memory for the cache blocks", "hint_no_kv_memory"),
    ("less than desired GPU memory utilization", "hint_gpu_memory_taken"),
    ("CUDA out of memory", "hint_oom"),
    ("OutOfMemoryError", "hint_oom"),
    ("trust_remote_code=True", "hint_remote_code"),
    ("Address already in use", "hint_port_in_use"),
    ("Ninja build failed", "hint_compile"),
    ("nvcc fatal", "hint_compile"),
    ("InductorError", "hint_compile"),
    ("torch._dynamo", "hint_compile"),
    ("not supported", "hint_unsupported"),
)
# Рядки помилок: у INFO-рядку з конфігом теж є слова на кшталт trust_remote_code. `Error\b` без межі зліва —
# щоб ловити RuntimeError:, ValueError:, OutOfMemoryError: (API-сервер друкує їх без префікса ERROR).
_ERROR_LINE = re.compile(r"\bERROR\b|Error\b|Exception\b|\bTraceback\b|\bfatal\b")
# Гібридні моделі (Mamba): vLLM називає, скільки одночасних послідовностей уміщує кеш — стільки й ставимо.
_MAMBA_SEQS = re.compile(r"exceeds available Mamba cache blocks \((\d+)\)")
# Порада vLLM щодо довжини контексту в тексті помилки KV-кешу.
_SUGGESTED_LEN = re.compile(r"estimated maximum model length is (\d+)")
# Заданий контекст більший за межу моделі (напр. автоповтор поставив 8192 моделі з 4096): vLLM називає межу.
_MODEL_LEN_LIMIT = re.compile(r"greater than the derived max_model_len \(\w+=(\d+)")


def _unit(name: str) -> str:
    """Юніт моделі з явним суфіксом: без нього systemctl дописує .service сам, і назви `x` та `x.service` вели б
    до одного юніта — «прибрати» одну модель зупинило б іншу."""
    return f"{_UNIT_PREFIX}{name}.service"


class Launcher(Protocol):
    def start(self, unit: str, argv: list[str], env: dict[str, str], log: Path) -> None: ...

    def stop(self, unit: str) -> bool | None: ...

    def active(self, unit: str) -> bool | None: ...


class Probe(Protocol):
    def health(self, port: int) -> bool: ...

    def metrics(self, port: int) -> dict[str, float]: ...


class SystemdLauncher:
    """Юніти через systemd-run --user; вивід — у файл логу (дописується). group — запускати команду з цією групою
    через sg (як і сам менеджер): так vLLM читає теки користувачів у теці файлів (inbox.run_group)."""

    def __init__(self, group: str | None = None) -> None:
        self._group = group

    def start(self, unit: str, argv: list[str], env: dict[str, str], log: Path) -> None:
        if self._group:
            argv = [shutil.which("sg") or "/usr/bin/sg", self._group, "-c", "exec " + shlex.join(argv)]
        subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True, check=False)
        cmd = ["systemd-run", "--user", f"--unit={unit}", "--collect",
               f"--property=StandardOutput=append:{log}", f"--property=StandardError=append:{log}",
               "--property=KillMode=control-group", "--property=TimeoutStopSec=30",
               *[f"--setenv={k}={v}" for k, v in env.items()], *argv]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise ManagerError("start_failed", detail=(r.stderr or r.stdout).strip()[:300])

    def stop(self, unit: str) -> bool:
        """True — юніт зупинено або його вже немає."""
        r = subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True, text=True, check=False)
        return r.returncode == 0 or self.active(unit) is False

    def active(self, unit: str) -> bool | None:
        """True/False — живий/мертвий; None — невідомо (напр. збій D-Bus): цей прохід треба пропустити."""
        r = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True, check=False)
        state = r.stdout.strip()
        if state in ("active", "activating", "deactivating", "reloading", "refreshing"):
            return True
        if state in ("inactive", "failed"):
            return False
        return None


class HttpProbe:
    """/health і /metrics сервера vLLM на 127.0.0.1."""

    _WANTED: ClassVar[dict[str, str]] = {"vllm:num_requests_running": "running",
                                         "vllm:num_requests_waiting": "waiting",
                                         "vllm:kv_cache_usage_perc": "kv_usage"}

    def health(self, port: int) -> bool:
        try:
            return httpx.get(f"http://127.0.0.1:{port}/health", timeout=_HTTP_TIMEOUT_S).status_code == 200
        except httpx.HTTPError:
            return False

    def metrics(self, port: int) -> dict[str, float]:
        try:
            text = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=_HTTP_TIMEOUT_S).text
        except httpx.HTTPError:
            return {}
        out: dict[str, float] = {}
        for line in text.splitlines():
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name in self._WANTED:
                try:
                    out[self._WANTED[name]] = out.get(self._WANTED[name], 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    continue
        return out


def port_is_free(port: int) -> bool:
    """Чи не слухає порт ніхто на 127.0.0.1 (спроба з'єднання: зайнятий порт відповідає)."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) != 0


def _without_flag(args: list[str], flag: str) -> list[str]:
    """args без прапорця flag і його значення (і у формі --flag=value)."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        name = a.split("=", 1)[0].replace("_", "-")
        if name == flag:
            skip = "=" not in a
            continue
        out.append(a)
    return out


def check_extra_args(args: Any) -> list[str]:
    """Список рядків без заборонених прапорців (зокрема скорочених і з `_`); інакше bad_extra_args."""
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ManagerError("bad_extra_args", flags=", ".join(_FORBIDDEN_FLAGS))
    for a in args:
        if not a.startswith("--"):
            continue
        flag = a.split("=", 1)[0].split(".", 1)[0].replace("_", "-").lower()
        if any(f.startswith(flag) for f in _FORBIDDEN_FLAGS if len(flag) > 2):
            raise ManagerError("bad_extra_args", flags=", ".join(_FORBIDDEN_FLAGS))
    return list(args)


@dataclass
class Server:
    name: str
    repo: str
    revision: str
    gpu: int
    port: int
    fraction: float
    max_model_len: int | None
    extra_args: list[str]
    user: str
    started: float
    status: str
    want: bool = True  # має працювати: після рестарту менеджера / машини піднімається знову
    error_code: str | None = None
    error_params: dict[str, Any] = field(default_factory=dict)  # напр. {"n": 18192}
    attempts: int = 1
    auto_changes: list[str] = field(default_factory=list)  # що менеджер змінив сам між спробами
    ready_at: float | None = None
    usage: dict[str, float] = field(default_factory=dict)
    fraction_explicit: bool = False  # частку задали в запиті (напр. щоб ділити карту) — у профіль її не пишемо


class ModelRunner:
    def __init__(self, cfg: Config, store: StateStore, journal: Journal, models: ModelStore, gpus: GpuManager,
                 launcher: Launcher | None = None, probe: Probe | None = None,
                 port_free: Callable[[int], bool] = port_is_free, clock: Callable[[], float] = time.time) -> None:
        """launcher, probe, port_free, clock — шви для тестів (без systemd, мережі й справжнього часу)."""
        self._cfg = cfg
        self._store = store
        self._journal = journal
        self._models = models
        self._gpus = gpus
        self._launcher = launcher or SystemdLauncher(run_group(cfg.inbox_dir))
        self._probe = probe or HttpProbe()
        self._port_free = port_free
        self._clock = clock
        self._log_dir = cfg.data_dir / "servers"
        self._lock = threading.RLock()
        raw = store.read(SECTION) or {}
        self._items = {n: Server(**s) for n, s in raw.items()}
        self._profiles_path = cfg.data_dir / PROFILES_FILE

    # ---- запуск і зупинка ------------------------------------------------------------------------

    def start(self, repo: str, gpu: int, user: str, fraction: float | None = None, max_model_len: int | None = None,
              extra_args: list[str] | None = None, name: str | None = None, port: int | None = None) -> dict[str, Any]:
        """Запускає модель з диска на карті gpu. port — явний порт з vllm.port_range, інакше найменший вільний.
 Параметри: явні -> профіль моделі -> типові (максимум пам'яті й
        контексту). Частка з профілю — лише верхня межа: відмову дає тільки явна частка, що не влазить."""
        require_user(self._cfg.users, user)
        card = self._gpus.gpu(gpu)  # unknown_gpu, якщо карти немає
        res = card["reservation"]
        if res is not None and res["user"] != user:
            raise ManagerError("reserved_by_other", gpu=gpu, owner=res["user"], purpose=res["purpose"])
        local = {m["repo"]: m for m in self._models.local()}
        if repo not in local or local[repo]["state"] != "ready":
            raise ManagerError("model_not_local", repo=repo)
        if fraction is not None and (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                                     or not 0 < fraction <= 1):
            raise ManagerError("bad_fraction")
        profile = self._load_profiles().get(repo, {})
        args = check_extra_args(extra_args if extra_args is not None else profile.get("extra_args", []))
        if name is not None:
            check_name(name)
        ctx = max_model_len if max_model_len is not None else profile.get("max_model_len")
        with self._lock:
            free = self._free_share(gpu, card, cap=fraction is None)
            if fraction is not None:
                if fraction > free + 1e-9:
                    raise ManagerError("gpu_memory_low", gpu=gpu, need_gib=round(card["memory_total_mib"] * fraction / 1024, 1),
                                       free_gib=round(card["memory_total_mib"] * free / 1024, 1))
                fr = float(fraction)
            else:
                fr = min(float(profile.get("fraction") or self._cfg.vllm_default_fraction), free)
                if fr < _MIN_FRACTION:
                    raise ManagerError("gpu_memory_low", gpu=gpu,
                                       need_gib=round(card["memory_total_mib"] * _MIN_FRACTION / 1024, 1),
                                       free_gib=round(card["memory_total_mib"] * free / 1024, 1))
            srv = Server(name=self._name(repo, name), repo=repo, revision=local[repo]["revision"], gpu=gpu,
                         port=self._free_port(port), fraction=fr, max_model_len=ctx, extra_args=args, user=user,
                         started=self._clock(), status=STARTING, fraction_explicit=fraction is not None)
            self._items[srv.name] = srv  # місце й порт зайняті до systemd-run, щоб паралельний старт їх не взяв
            self._save()
        try:
            self._launch(srv)
        except (ManagerError, OSError) as exc:
            with self._lock:
                self._items.pop(srv.name, None)
                self._save()
            raise exc if isinstance(exc, ManagerError) else ManagerError("start_failed", detail=str(exc)[:300]) from exc
        self._journal.record(user, "model_start", name=srv.name, repo=repo, gpu=gpu, port=srv.port)
        self._stop_if_unwanted(srv.name)
        return self._view(srv)

    def stop(self, name: str, user: str) -> dict[str, Any]:
        """Зупиняє модель; вона більше не піднімається після рестарту. Впалу — прибирає зі списку."""
        require_user(self._cfg.users, user)
        with self._lock:
            srv = self._items.get(name)
            if srv is None or (not srv.want and srv.status != FAILED):
                raise ManagerError("server_not_found", name=name)
            dismiss = not srv.want  # впала модель: «зупинити» = прибрати зі списку
            if dismiss:
                del self._items[name]
            else:
                srv.want, srv.status, srv.usage = False, STOPPED, {}
            self._save()
        stopped = self._launcher.stop(_unit(name))  # і для впалої: раптом юніт ще живий
        if stopped is False and not dismiss:
            with self._lock:
                srv.want, srv.status = True, STARTING  # лишається під наглядом: poll побачить реальний стан
                self._save()
            raise ManagerError("stop_failed", name=name)
        if not dismiss:
            self._journal.record(user, "model_stop", name=name, repo=srv.repo, gpu=srv.gpu)
        return {**self._view(srv), "status": STOPPED}

    def move(self, name: str, user: str, gpu: int | None = None, port: int | None = None) -> dict[str, Any]:
        """Перенести запущену модель на іншу карту та/або порт: зупинити -> змінити -> запустити з тією ж назвою
        (агенти звертаються за назвою). Явна частка зберігається, виведену менеджер рахує для нової карти заново.
        Якщо новий старт відмовив — модель повертається на старе місце."""
        with self._lock:
            srv = self._items.get(name)
            if srv is None or not srv.want:
                raise ManagerError("server_not_found", name=name)
            old = replace(srv)
        new_gpu = old.gpu if gpu is None else gpu
        new_port = old.port if port is None else port
        if new_gpu == old.gpu and new_port == old.port:
            return self._view(old)
        card = self._gpus.gpu(new_gpu)  # unknown_gpu — ще до зупинки
        if new_gpu != old.gpu:  # пам'ять нової карти — теж до зупинки, щоб не перезапускати модель даремно
            with self._lock:
                free = self._free_share(new_gpu, card, cap=not old.fraction_explicit)
            need = old.fraction if old.fraction_explicit else _MIN_FRACTION
            if need > free + 1e-9:
                raise ManagerError("gpu_memory_low", gpu=new_gpu, need_gib=round(card["memory_total_mib"] * need / 1024, 1),
                                   free_gib=round(card["memory_total_mib"] * free / 1024, 1))
        self.stop(name, user)
        self._fresh_sample()
        keep = {"fraction": old.fraction if old.fraction_explicit else None, "max_model_len": old.max_model_len,
                "extra_args": old.extra_args, "name": name}
        try:
            out = self.start(old.repo, new_gpu, user, port=new_port, **keep)
        except ManagerError:
            try:
                self.start(old.repo, old.gpu, user, port=old.port, **keep)
            except ManagerError as back:
                log.warning("could not return %s to GPU %s port %s: %s", name, old.gpu, old.port, back)
            raise
        self._journal.record(user, "model_move", name=name, gpu=new_gpu, port=new_port, from_gpu=old.gpu,
                             from_port=old.port)
        return out

    def restore(self) -> None:
        """При старті менеджера: моделі, що мали працювати, але їхніх юнітів немає (перезавантаження), —
        запускаються знову з тими самими параметрами і портом; живі юніти просто підхоплюються."""
        with self._lock:
            wanted = [s for s in self._items.values() if s.want]
        for srv in wanted:
            if self._launcher.active(_unit(srv.name)) is not False:
                continue  # живий або невідомо — не чіпати
            try:
                with self._lock:
                    srv.status, srv.started, srv.ready_at, srv.error_code = STARTING, self._clock(), None, None
                self._launch(srv)
            except (ManagerError, OSError) as exc:
                with self._lock:
                    srv.status, srv.want = FAILED, False
                    srv.error_code = exc.code if isinstance(exc, ManagerError) else "start_failed"
        with self._lock:
            self._save()

    def poll(self) -> None:
        """Крок стану по кожній моделі окремо (збій однієї не зупиняє інших): живий юніт + /health 200 -> running;
        юніт зник -> автоповтор або failed з підказкою з логу; довгий старт понад start_timeout -> стоп і failed."""
        with self._lock:
            snapshot = [(s.name, s.port, s.started) for s in self._items.values() if s.want]
        for name, port, started in snapshot:
            try:
                self._poll_one(name, port, started)
            except Exception:
                log.exception("poll of model %s failed; will retry", name)

    def _poll_one(self, name: str, port: int, started: float) -> None:
        unit = _unit(name)
        alive = self._launcher.active(unit)
        if alive is None:
            return
        if not alive:
            code, params = self._hint(name)
            with self._lock:
                srv = self._current(name, started)
                if srv is None:
                    return
                change = self._plan_fix(srv, code, params)
                if change is None:
                    srv.status, srv.want, srv.usage = FAILED, False, {}
                    srv.error_code, srv.error_params = code, params
                self._save()
                snap = replace(srv)
            if change is None:
                self._journal.record(snap.user, "model_failed", name=name, repo=snap.repo, gpu=snap.gpu, error=code)
                return
            self._journal.record(snap.user, "model_retry", name=name, repo=snap.repo, gpu=snap.gpu, change=change)
            try:
                self._launch(snap)
            except (ManagerError, OSError) as exc:
                with self._lock:
                    if (cur := self._current(name, snap.started)) is not None:
                        cur.status, cur.want, cur.error_code = FAILED, False, "start_failed"
                        self._save()
                log.warning("relaunch of %s failed: %s", name, exc)
                return
            self._stop_if_unwanted(name, snap.started)
            return
        healthy = self._probe.health(port)
        usage = self._probe.metrics(port) if healthy else {}
        became_running: Server | None = None
        timed_out = False
        with self._lock:
            srv = self._current(name, started)
            if srv is None:
                return
            if healthy:
                if srv.status != RUNNING:
                    srv.status, srv.ready_at, srv.error_code, srv.error_params = RUNNING, self._clock(), None, {}
                    became_running = replace(srv)
                    self._save()
                srv.usage = usage
            elif srv.status == STARTING and self._clock() - srv.started > self._cfg.vllm_start_timeout_s:
                srv.status, srv.want, srv.error_code = FAILED, False, "hint_start_timeout"
                timed_out = True
                self._save()
        if became_running is not None:
            self._save_profile(became_running)
        if timed_out:
            self._launcher.stop(unit)
            self._journal.record(srv.user, "model_failed", name=name, repo=srv.repo, gpu=srv.gpu,
                                 error="hint_start_timeout")

    # ---- читання ---------------------------------------------------------------------------------

    def servers(self) -> list[dict[str, Any]]:
        """Моделі, що працюють, стартують або щойно впали (зупинені людиною не показуються)."""
        with self._lock:
            items = [s for s in self._items.values() if s.want or s.status == FAILED]
        return [self._view(s) for s in sorted(items, key=lambda s: (s.gpu, s.port))]

    def on_gpu(self, gpu: int) -> list[dict[str, Any]]:
        with self._lock:
            return [{"name": s.name, "port": s.port, "status": s.status, "user": s.user}
                    for s in self._items.values() if s.gpu == gpu and s.want]

    def in_use(self, repo: str) -> bool:
        with self._lock:
            return any(s.repo == repo and s.want for s in self._items.values())

    def logs(self, name: str, lines: int = 60, errors_only: bool = False) -> dict[str, Any]:
        """Хвіст логу останнього старту моделі; errors_only — лише рядки з помилками. Плюс код підказки."""
        with self._lock:
            srv = self._items.get(name)
        if srv is None:
            raise ManagerError("server_not_found", name=name)
        text = self._last_run_log(name)
        rows = [ln for ln in text.splitlines() if not errors_only or _ERROR_LINE.search(ln)]
        return {"name": name, "status": srv.status, "hint": srv.error_code, "hint_params": srv.error_params,
                "lines": rows[-max(1, min(lines, 500)):]}

    # ---- внутрішнє -------------------------------------------------------------------------------

    def _stop_if_unwanted(self, name: str, started: float | None = None) -> None:
        """Юніт щойно запущено, а модель тим часом зупинили: stop між записом starting і systemd-run бачив ще
        неіснуючий юніт і вважав його зупиненим. Без цього лишився б vLLM, якого менеджер не бачить, з пам'яттю карти.
        started — юніт автоповтору саме цього запуску: зупинити, якщо запис уже інший (напр. move вклинився).
        Без started (перший старт) нова бажана копія з тією ж назвою — від move: її не чіпаємо."""
        with self._lock:
            if started is not None:
                unwanted = self._current(name, started) is None
            else:
                srv = self._items.get(name)
                unwanted = srv is None or not srv.want
        if unwanted:
            self._launcher.stop(_unit(name))

    def _fresh_sample(self) -> None:
        """Свіжий замір карт після зупинки моделі: секундний замір ще бачить її пам'ять, і новий старт на тій самій
        карті (чи повернення при move) вирішив би, що пам'яті немає. systemctl stop повертається, коли всі процеси
        юніта завершились, тож драйвер пам'ять уже звільнив. Невдалий замір — лишається попередній."""
        try:
            self._gpus.tick()
        except Exception:
            log.exception("fresh GPU sample after stop failed; using the previous one")

    def _current(self, name: str, started: float) -> Server | None:
        """Той самий запуск, що був у знімку: між знімком і рішенням модель могли зупинити чи перезапустити."""
        srv = self._items.get(name)
        return srv if srv is not None and srv.want and srv.started == started else None

    def _launch(self, srv: Server) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"{srv.name}.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{_START_MARK}{time.strftime('%Y-%m-%d %H:%M:%S')} gpu={srv.gpu} port={srv.port} ===\n")
        snapshot = self._cfg.hf_home / "hub" / ("models--" + srv.repo.replace("/", "--")) / "snapshots" / srv.revision
        argv = [str(self._cfg.vllm_bin), "serve", str(snapshot), "--host", "127.0.0.1", "--port", str(srv.port),
                "--served-model-name", srv.name, "--gpu-memory-utilization", str(srv.fraction),
                # без журналу HTTP: менеджер питає /health і /metrics кожні 2 с, і лог перетворився б на шум
                "--disable-uvicorn-access-log"]
        if self._cfg.inbox_dir.is_dir():  # картинки, аудіо й відео з теки файлів модель читає сама (file://)
            argv += ["--allowed-local-media-path", str(self._cfg.inbox_dir)]
        if srv.max_model_len:
            argv += ["--max-model-len", str(srv.max_model_len)]
        # Для 1–3 користувачів 256 одночасних послідовностей vLLM — лише зайва пам'ять.
        if not any(a.split("=", 1)[0].replace("_", "-") == "--max-num-seqs" for a in srv.extra_args):
            argv += ["--max-num-seqs", str(self._cfg.vllm_max_num_seqs)]
        argv += srv.extra_args
        cuda = str(self._cfg.vllm_cuda_home)
        # CUDA_DEVICE_ORDER=PCI_BUS_ID: номер карти для CUDA збігається з номером NVML, яким карту перевіряли;
        # UUID точніший за номер: CUDA не рахує відпалої карти, і номери решти для неї зсунулися б.
        env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": self._cuda_id(srv.gpu), "HF_HOME": str(self._cfg.hf_home),
               "HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "CUDA_HOME": cuda, "PATH": f"{cuda}/bin:/usr/bin:/bin"}
        self._launcher.start(_unit(srv.name), argv, env, log_path)

    def _cuda_id(self, gpu: int) -> str:
        """UUID карти з NVML для CUDA_VISIBLE_DEVICES (vLLM і CUDA приймають UUID); номер — якщо UUID невідомий."""
        try:
            return self._gpus.gpu(gpu)["uuid"] or str(gpu)
        except ManagerError:
            return str(gpu)

    def _plan_fix(self, srv: Server, code: str, params: dict[str, Any]) -> str | None:
        """Відоме виправлення (під замком): змінює параметри й готує перезапуск з тим самим портом;
        None — виправлення немає або спроби вичерпано."""
        if srv.attempts >= _MAX_ATTEMPTS:
            return None
        if code in ("hint_kv_len", "hint_ctx_over_model"):
            srv.max_model_len, change = int(params["n"]), f"max_model_len={params['n']}"
        elif code in ("hint_gpu_memory_taken", "hint_oom") and (lower := self._lower_fraction(srv.fraction)):
            srv.fraction, change = lower, f"fraction={lower}"
        elif code in ("hint_kv_too_small", "hint_oom"):
            ctx = max(_MIN_AUTO_CTX, (srv.max_model_len or 2 * _MIN_AUTO_CTX * 4) // 2 // _CTX_ROUND * _CTX_ROUND)
            if srv.max_model_len is not None and ctx >= srv.max_model_len:
                return None
            srv.max_model_len, change = ctx, f"max_model_len={ctx}"
        elif code == "hint_no_kv_memory":
            fr = self._bigger_fraction(srv)
            if fr is None:
                return None
            srv.fraction, change = fr, f"fraction={fr}"
        elif code == "hint_mamba_seqs" and int(params["n"]) >= 1:
            n = int(params["n"])
            srv.extra_args, change = [*_without_flag(srv.extra_args, "--max-num-seqs"), "--max-num-seqs", str(n)], f"max_num_seqs={n}"
        elif code == "hint_compile" and "--enforce-eager" not in srv.extra_args:
            srv.extra_args, change = [*srv.extra_args, "--enforce-eager"], "--enforce-eager"
        else:
            return None
        srv.attempts += 1
        srv.auto_changes.append(change)
        srv.status, srv.started, srv.error_code, srv.error_params = STARTING, self._clock(), None, {}
        return change

    def _free_share(self, gpu: int, card: dict[str, Any], exclude: str | None = None, cap: bool = True) -> float:
        """Вільна частка карти: менше з телеметрії (з запасом) і з часток наших моделей на ній — ті, що ще
        стартують, пам'ять ще не зайняли, але vLLM візьме свою частку. Донизу з кроком 0.01. cap — ще й не більше
        vllm.default_fraction: стеля лише для частки, яку вибирає менеджер; явну людина чи агент вибрали свідомо."""
        total = card["memory_total_mib"]
        if not total:  # нечитна карта (NVML): пам'яті не знаємо — запуск на неї відмовить gpu_memory_low
            return 0.0
        by_memory = (total - (card["memory_used_mib"] or 0) - _MEM_SLACK_MIB) / total
        by_models = 1.0 - sum(s.fraction for s in self._items.values()
                              if s.want and s.gpu == gpu and s.name != exclude)
        limit = self._cfg.vllm_default_fraction if cap else 1.0
        return max(0.0, int(min(by_memory, by_models, limit) * 100) / 100)

    @staticmethod
    def _lower_fraction(current: float) -> float | None:
        """Наступний щабель драбини нижче за поточну частку; None — нижче 0.9 автоматично не йдемо."""
        return next((f for f in _FRACTION_LADDER if f < current - 1e-9), None)

    def _bigger_fraction(self, srv: Server) -> float | None:
        """Удвічі більша частка, але не більша за вільну (власна частка моделі не рахується: вона зупинена)."""
        free = self._free_share(srv.gpu, self._gpus.gpu(srv.gpu), exclude=srv.name)
        fr = round(min(srv.fraction * 2, free), 2)
        return fr if fr > srv.fraction else None

    def _load_profiles(self) -> dict[str, dict[str, Any]]:
        """Читається при кожному старті: файл можна правити руками без перезапуску менеджера."""
        try:
            data = json.loads(self._profiles_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            log.warning("%s is unreadable (%s); profiles ignored until fixed", self._profiles_path, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def _save_profile(self, srv: Server) -> None:
        """Дописує профіль моделі. Зіпсований файл не переписується: інакше пропали б профілі інших моделей."""
        try:
            profiles = json.loads(self._profiles_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            profiles = {}
        except (OSError, ValueError) as exc:
            log.warning("not saving profile of %s: %s is unreadable (%s)", srv.repo, self._profiles_path, exc)
            return
        if not isinstance(profiles, dict):
            log.warning("not saving profile of %s: %s is not an object", srv.repo, self._profiles_path)
            return
        prev = profiles.get(srv.repo)
        old: dict[str, Any] = prev if isinstance(prev, dict) else {}
        # Явна частка — рішення про поділ карти, а не властивість моделі: у профілі лишається виведена менеджером.
        fraction = old.get("fraction") if srv.fraction_explicit else srv.fraction
        profiles[srv.repo] = {"fraction": fraction, "max_model_len": srv.max_model_len,
                              "extra_args": srv.extra_args, "gpu": srv.gpu,
                              "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "attempts": srv.attempts}
        self._profiles_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._profiles_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self._profiles_path)

    def _name(self, repo: str, name: str | None) -> str:
        active = {n for n, s in self._items.items() if s.want}
        if name is not None:
            if name in active:
                raise ManagerError("name_taken", name=name)
            return name
        base = slug(repo.rsplit("/", 1)[-1])
        candidate, n = base, 2
        while candidate in active:
            candidate, n = f"{base}-{n}", n + 1
        return candidate

    def free_ports(self) -> list[int]:
        """Порти діапазону, які зараз можна взяти (для вибору на сторінці)."""
        with self._lock:
            taken = {s.port for s in self._items.values() if s.want}
        lo, hi = self._cfg.model_ports
        return [p for p in range(lo, hi + 1) if p not in taken and self._port_free(p)]

    def _free_port(self, wanted: int | None = None) -> int:
        taken = {s.port for s in self._items.values() if s.want}
        lo, hi = self._cfg.model_ports
        if wanted is not None:
            if isinstance(wanted, bool) or not isinstance(wanted, int) or not lo <= wanted <= hi:
                raise ManagerError("bad_port", port=wanted, lo=lo, hi=hi)
            if wanted in taken or not self._port_free(wanted):
                raise ManagerError("port_taken", port=wanted)
            return wanted
        for port in range(lo, hi + 1):
            if port not in taken and self._port_free(port):
                return port
        raise ManagerError("no_free_port", lo=lo, hi=hi)

    def _last_run_log(self, name: str) -> str:
        try:
            text = (self._log_dir / f"{name}.log").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[text.rfind(_START_MARK):] if _START_MARK in text else text

    def _hint(self, name: str) -> tuple[str, dict[str, Any]]:
        text = self._last_run_log(name)
        errors = "\n".join(ln for ln in text.splitlines() if _ERROR_LINE.search(ln))
        code = next((c for needle, c in _HINTS if needle in errors), "hint_see_logs")
        limit = _MODEL_LEN_LIMIT.search(text)  # рядок pydantic «  Value error, User-specified …» не має слова Error
        if limit:
            return "hint_ctx_over_model", {"n": int(limit.group(1))}
        mamba = _MAMBA_SEQS.search(errors)
        if mamba:
            return "hint_mamba_seqs", {"n": int(mamba.group(1))}
        m = _SUGGESTED_LEN.search(errors)
        if m:
            n = int(m.group(1))
            if n < _MIN_USEFUL_CTX:
                return "hint_no_kv_memory", {}
            return "hint_kv_len", {"n": n // _CTX_ROUND * _CTX_ROUND or n}
        return code, {}

    def _view(self, s: Server) -> dict[str, Any]:
        return {**{k: v for k, v in asdict(s).items() if k != "want"}, "active": s.want}

    def _save(self) -> None:
        self._store.write(SECTION, {n: asdict(s) for n, s in self._items.items()})
