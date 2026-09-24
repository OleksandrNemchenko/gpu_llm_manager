"""Завантаження config.json — файлу, який редагує людина.

Формат: розділи, у кожному — налаштування виду {"value": …, "comment": "…"}; ключ
"comment" на рівні розділу описує сам розділ. Налаштування без пояснення не приймається."""

from __future__ import annotations

import ipaddress
import json
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Вхідна тека за замовчуванням; налаштування необов'язкове.
_DEFAULT_INBOX = "/srv/gpu-inbox"
# Типові значення vLLM (фаза 3); пояснення — у config.json.
_VLLM_DEFAULTS = {"bin": ".venv-vllm/bin/vllm", "cuda_home": "/usr/local/cuda", "default_fraction": 0.95, "max_num_seqs": 32,
                  "start_timeout_s": 900}
# Типові значення розділу models (необов'язковий, щоб конфіг фази 1 лишався дійсним); пояснення — у config.json.
# hf_home за замовчуванням — <домашня тека>/hf-cache, рахується при читанні конфігу (не при імпорті модуля).
_MODELS_DEFAULTS = {"max_parallel_downloads": 2, "min_free_disk_gib": 50,
                    "fit_memory_fraction": 0.9, "fit_overhead_gib": 2.0}
# Порти нижче 1024 вимагають root; 65535 — межа TCP.
_MIN_PORT, _MAX_PORT = 1024, 65535


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    hosts: tuple[str, ...]
    port: int
    host_names: tuple[str, ...]  # усі імена, за якими відкривають сервер: дозволені значення заголовка Host
    display_timezone: ZoneInfo
    users: tuple[str, ...]
    sample_interval_s: float
    ring_keep_s: int
    history_db_interval_s: int
    history_db_keep_days: int
    busy_memory_mib: int
    model_ports: tuple[int, int]  # діапазон портів для vLLM, включно з межами
    journal_keep_entries: int
    journal_keep_days: float
    inbox_dir: Path  # спільна вхідна тека файлів: <inbox_dir>/<логін>/
    hf_home: Path  # кеш HuggingFace (HF_HOME)
    max_parallel_downloads: int
    min_free_disk_gib: float
    fit_memory_fraction: float
    fit_overhead_gib: float
    secrets_path: Path  # secrets.json поруч із конфігом
    vllm_bin: Path
    vllm_cuda_home: Path
    vllm_default_fraction: float
    vllm_start_timeout_s: float
    vllm_max_num_seqs: int
    data_dir: Path


def unwrap_settings(node: dict[str, Any], where: str = "") -> dict[str, Any]:
    """Розділ -> звичайний словник: відкидає "comment" розділу, замінює {"value", "comment"} на значення.

    where — шлях розділу для тексту помилки. Голе значення без коментаря — ConfigError."""
    out: dict[str, Any] = {}
    for key, item in node.items():
        if key == "comment":
            continue
        name = f"{where}.{key}" if where else key
        if not isinstance(item, dict):
            raise ConfigError(f'{name}: expected {{"value": …, "comment": "…"}} or a section, got {item!r}')
        if "value" in item:
            comment = item.get("comment")
            if set(item) != {"value", "comment"} or not isinstance(comment, str) or not comment.strip():
                raise ConfigError(f'{name}: a setting has exactly "value" and a non-empty "comment"')
            out[key] = item["value"]
        else:
            out[key] = unwrap_settings(item, name)
    return out


def _str_list(value: Any, name: str) -> tuple[str, ...]:
    """Список рядків з конфігу. Рядок замість списку розібрався б на літери: "alice" -> ('a', 'l', 'i', 'c', 'e')."""
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{name}: expected a list of strings, got {value!r}")
    return tuple(value)


def _refused_host(host: str) -> bool:
    """Адреса, яку не можна слухати: «усі інтерфейси» в будь-якому записі (0.0.0.0, 0, 0.0, ::) відкрила б
    керування картами всій локальній мережі, а не лише TailScale; IPv6 сервіс не підтримує (Host у дужках,
    IPV6_FREEBIND) і не стартував би; порожній рядок — теж «усі інтерфейси»."""
    if not host.strip():
        return True
    try:
        if ipaddress.ip_address(host).version == 6:
            return True
    except ValueError:
        pass
    try:
        return socket.inet_aton(host) == bytes(4)  # inet_aton приймає й скорочені записи: "0", "0.0"
    except OSError:
        return False  # ім'я хоста, напр. localhost


def _absolute(value: str, base: Path) -> Path:
    """Шлях з конфігу: ~ розкривається, відносний рахується від теки конфігу, а не від теки запуску менеджера
    (юніти моделей systemd запускаються з домашньої теки й відносних шляхів не зрозуміли б)."""
    p = Path(value).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


def load_config(path: Path) -> Config:
    """Читає й перевіряє конфіг. Відносні шляхи рахуються від теки конфігу. Будь-яка вада — ConfigError."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ConfigError("top level must be an object")
        raw = unwrap_settings(data)
    except (OSError, json.JSONDecodeError, ConfigError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    try:
        server, gpu = raw["server"], raw["gpu"]
        hosts = _str_list(server["hosts"], "server.hosts")
        ports = raw["vllm"]["port_range"]
        models = {"hf_home": str(Path.home() / "hf-cache"), **_MODELS_DEFAULTS, **raw.get("models", {})}
        vllm = {**_VLLM_DEFAULTS, **raw["vllm"]}
        vbin = Path(vllm["bin"])
        cfg = Config(
            hosts=hosts,
            port=int(server["port"]),
            host_names=tuple(dict.fromkeys([*hosts, *_str_list(server.get("extra_host_names", []),
                                                              "server.extra_host_names")])),
            display_timezone=ZoneInfo(server["display_timezone"]),
            users=_str_list(raw["users"]["allowed"], "users.allowed"),
            sample_interval_s=float(gpu["sample_interval_s"]),
            ring_keep_s=int(gpu["ring_keep_s"]),
            history_db_interval_s=int(gpu["history_db_interval_s"]),
            history_db_keep_days=int(gpu["history_db_keep_days"]),
            busy_memory_mib=int(gpu["busy_memory_mib"]),
            model_ports=(int(ports[0]), int(ports[1])),
            journal_keep_entries=int(raw["journal"]["keep_entries"]),
            journal_keep_days=float(raw["journal"]["keep_days"]),
            inbox_dir=_absolute(raw.get("files", {}).get("inbox_dir", _DEFAULT_INBOX), path.parent),
            hf_home=_absolute(models["hf_home"], path.parent),
            max_parallel_downloads=int(models["max_parallel_downloads"]),
            min_free_disk_gib=float(models["min_free_disk_gib"]),
            fit_memory_fraction=float(models["fit_memory_fraction"]),
            fit_overhead_gib=float(models["fit_overhead_gib"]),
            secrets_path=path.parent / "secrets.json",
            vllm_bin=(vbin if vbin.is_absolute() else path.parent / vbin).resolve(),
            vllm_cuda_home=Path(vllm["cuda_home"]),
            vllm_default_fraction=float(vllm["default_fraction"]),
            vllm_start_timeout_s=float(vllm["start_timeout_s"]),
            vllm_max_num_seqs=int(vllm["max_num_seqs"]),
            data_dir=_absolute(raw["paths"]["data_dir"], path.parent),
        )
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    except (KeyError, IndexError, TypeError, ValueError, ZoneInfoNotFoundError) as exc:
        raise ConfigError(f"{path}: invalid or missing key: {exc!r}") from exc
    refused = [h for h in cfg.hosts if _refused_host(h)]
    if refused:
        raise ConfigError(f"{path}: server.hosts must name concrete IPv4 addresses or host names, not {refused}")
    if not cfg.hosts or not cfg.users:
        raise ConfigError(f"{path}: server.hosts and users.allowed must not be empty")
    lo, hi = cfg.model_ports
    if not _MIN_PORT <= cfg.port <= _MAX_PORT:
        raise ConfigError(f"{path}: server.port {cfg.port} must be {_MIN_PORT}..{_MAX_PORT}")
    if not _MIN_PORT <= lo <= hi <= _MAX_PORT or lo <= cfg.port <= hi:
        raise ConfigError(f"{path}: vllm.port_range {lo}..{hi} must be within {_MIN_PORT}..{_MAX_PORT} and exclude server.port {cfg.port}")
    # Нуль тут — не «без межі», а тиха поломка: 0 с старту — кожна модель failed, 0 MiB — кожна карта busy.
    positive = {"gpu.sample_interval_s": cfg.sample_interval_s, "gpu.ring_keep_s": cfg.ring_keep_s,
                "gpu.history_db_interval_s": cfg.history_db_interval_s, "gpu.history_db_keep_days": cfg.history_db_keep_days,
                "gpu.busy_memory_mib": cfg.busy_memory_mib, "journal.keep_entries": cfg.journal_keep_entries,
                "journal.keep_days": cfg.journal_keep_days, "vllm.start_timeout_s": cfg.vllm_start_timeout_s}
    not_positive = [k for k, v in positive.items() if not v > 0]
    if not_positive:
        raise ConfigError(f"{path}: must be > 0: {', '.join(not_positive)}")
    if not cfg.min_free_disk_gib >= 0 or not cfg.fit_overhead_gib >= 0:
        raise ConfigError(f"{path}: models.min_free_disk_gib and models.fit_overhead_gib must be >= 0")
    if cfg.vllm_max_num_seqs < 1:
        raise ConfigError(f"{path}: vllm.max_num_seqs must be >= 1")
    if not 0.05 <= cfg.vllm_default_fraction <= 1:
        raise ConfigError(f"{path}: vllm.default_fraction must be in [0.05, 1]")
    if cfg.max_parallel_downloads < 1 or not 0 < cfg.fit_memory_fraction <= 1:
        raise ConfigError(f"{path}: models.max_parallel_downloads must be >= 1 and fit_memory_fraction in (0, 1]")
    return cfg
