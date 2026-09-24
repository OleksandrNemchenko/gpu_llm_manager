"""Завантаження config.json — файлу, який редагує людина.

Формат (рішення користувача): розділи, у кожному — налаштування виду {"value": …, "comment": "…"}; ключ
"comment" на рівні розділу описує сам розділ. Налаштування без пояснення не приймається."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Прослуховування всіх інтерфейсів відкрило б керування картами всій локальній мережі, а не лише TailScale.
_FORBIDDEN_HOSTS = frozenset({"0.0.0.0", "::", ""})
# Порти нижче 1024 вимагають root; 65535 — межа TCP.
# Вхідна тека за замовчуванням — та, що погоджена з користувачем (11.1); налаштування необов'язкове.
_DEFAULT_INBOX = "/srv/gpu-inbox"
# Типові значення розділу models (необов'язковий, щоб конфіг фази 1 лишався дійсним); пояснення — у config.json.
# Типові значення vLLM (фаза 3); пояснення — у config.json.
_VLLM_DEFAULTS = {"bin": ".venv-vllm/bin/vllm", "cuda_home": "/usr/local/cuda", "default_fraction": 1.0, "max_num_seqs": 32,
                  "start_timeout_s": 900}
# hf_home за замовчуванням — <домашня тека>/hf-cache, рахується при читанні конфігу (не при імпорті модуля).
_MODELS_DEFAULTS = {"max_parallel_downloads": 2, "min_free_disk_gib": 50,
                    "fit_memory_fraction": 0.9, "fit_overhead_gib": 2.0}
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


def load_config(path: Path) -> Config:
    """Читає й перевіряє конфіг. Відносний data_dir рахується від теки конфігу. Будь-яка вада — ConfigError."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ConfigError("top level must be an object")
        raw = unwrap_settings(data)
    except (OSError, json.JSONDecodeError, ConfigError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    try:
        server, gpu = raw["server"], raw["gpu"]
        hosts = tuple(server["hosts"])
        data_dir = Path(raw["paths"]["data_dir"])
        ports = raw["vllm"]["port_range"]
        models = {"hf_home": str(Path.home() / "hf-cache"), **_MODELS_DEFAULTS, **raw.get("models", {})}
        vllm = {**_VLLM_DEFAULTS, **raw["vllm"]}
        vbin = Path(vllm["bin"])
        cfg = Config(
            hosts=hosts,
            port=int(server["port"]),
            host_names=tuple(dict.fromkeys([*hosts, *server.get("extra_host_names", [])])),
            display_timezone=ZoneInfo(server["display_timezone"]),
            users=tuple(raw["users"]["allowed"]),
            sample_interval_s=float(gpu["sample_interval_s"]),
            ring_keep_s=int(gpu["ring_keep_s"]),
            history_db_interval_s=int(gpu["history_db_interval_s"]),
            history_db_keep_days=int(gpu["history_db_keep_days"]),
            busy_memory_mib=int(gpu["busy_memory_mib"]),
            model_ports=(int(ports[0]), int(ports[1])),
            journal_keep_entries=int(raw["journal"]["keep_entries"]),
            journal_keep_days=float(raw["journal"]["keep_days"]),
            inbox_dir=Path(raw.get("files", {}).get("inbox_dir", _DEFAULT_INBOX)),
            hf_home=Path(models["hf_home"]),
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
            data_dir=data_dir if data_dir.is_absolute() else path.parent / data_dir,
        )
    except (KeyError, IndexError, TypeError, ValueError, ZoneInfoNotFoundError) as exc:
        raise ConfigError(f"{path}: invalid or missing key: {exc!r}") from exc
    if _FORBIDDEN_HOSTS & set(cfg.hosts):
        raise ConfigError(f"{path}: server.hosts must name concrete addresses, not {sorted(_FORBIDDEN_HOSTS & set(cfg.hosts))}")
    if not cfg.hosts or not cfg.users:
        raise ConfigError(f"{path}: server.hosts and users.allowed must not be empty")
    lo, hi = cfg.model_ports
    if not _MIN_PORT <= cfg.port <= _MAX_PORT:
        raise ConfigError(f"{path}: server.port {cfg.port} must be {_MIN_PORT}..{_MAX_PORT}")
    if not _MIN_PORT <= lo <= hi <= _MAX_PORT or lo <= cfg.port <= hi:
        raise ConfigError(f"{path}: vllm.port_range {lo}..{hi} must be within {_MIN_PORT}..{_MAX_PORT} and exclude server.port {cfg.port}")
    if cfg.journal_keep_entries <= 0 or cfg.journal_keep_days <= 0:
        raise ConfigError(f"{path}: journal.keep_entries and journal.keep_days must be > 0")
    if cfg.vllm_max_num_seqs < 1:
        raise ConfigError(f"{path}: vllm.max_num_seqs must be >= 1")
    if not 0.05 <= cfg.vllm_default_fraction <= 1:
        raise ConfigError(f"{path}: vllm.default_fraction must be in [0.05, 1]")
    if cfg.max_parallel_downloads < 1 or not 0 < cfg.fit_memory_fraction <= 1:
        raise ConfigError(f"{path}: models.max_parallel_downloads must be >= 1 and fit_memory_fraction in (0, 1]")
    if cfg.sample_interval_s <= 0:
        raise ConfigError(f"{path}: gpu.sample_interval_s must be > 0")
    return cfg
