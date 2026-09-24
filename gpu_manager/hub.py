"""Робота з HuggingFace: пошук, опис моделі, вибір файлів для vLLM і оцінка «чи влізе в карту».

Оцінка умовна (рішення користувача): остаточно покаже перший запуск. Вона потрібна, щоб відсіяти явно
завеликі моделі ще до завантаження десятків гігабайтів."""

from __future__ import annotations

import fnmatch
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)
from huggingface_hub.utils import build_hf_headers

from .messages import ManagerError

_GIB = 1024**3
# Файли, які vLLM не читає: інші формати ваг і копії для інших рушіїв. Качати їх — марна витрата диска.
_NEVER = ("*.gguf", "*.pth", "*.pt", "*.onnx", "*.onnx_data", "*.h5", "*.msgpack", "*.ot", "*.tflite", "*.mlmodel",
          "original/*", "onnx/*", "openvino/*", "coreml/*", "gguf/*")
_WEIGHTS_ST = "*.safetensors"
_WEIGHTS_BIN = "*.bin"
# Скільки байтів займає одне значення KV-кешу: vLLM за замовчуванням тримає його у 16 бітах.
_KV_BYTES = 2
# Скільки чекати відповіді HF на запит config.json, секунд.
_HTTP_TIMEOUT_S = 20
# Формати FP4: апаратно — лише з compute capability 10.0 (Blackwell); на старіших картах vLLM або не
# запустить, або емулює повільно. Текст попередження не називає модель карти: він правдивий на будь-якому сервері.
_FP4_FORMATS = ("nvfp4", "mxfp4", "fp4")


def select_files(names: list[str]) -> list[str]:
    """Які файли репозиторію качати для vLLM.

    Відкидає формати, яких vLLM не читає; якщо є safetensors — відкидає й *.bin (ті самі ваги вдруге).
    Повертає відсортований список імен, придатний як точний allow_patterns."""
    has_st = any(fnmatch.fnmatch(n, _WEIGHTS_ST) for n in names)
    out = []
    for n in names:
        if any(fnmatch.fnmatch(n, pat) for pat in _NEVER):
            continue
        if has_st and fnmatch.fnmatch(n, _WEIGHTS_BIN):
            continue
        out.append(n)
    return sorted(out)


def weight_bytes(files: dict[str, int]) -> int:
    """Сума розмірів файлів ваг (safetensors або bin) серед вибраних."""
    return sum(size for n, size in files.items() if fnmatch.fnmatch(n, _WEIGHTS_ST) or fnmatch.fnmatch(n, _WEIGHTS_BIN))


def kv_bytes_per_token(config: dict[str, Any]) -> int | None:
    """Байтів KV-кешу на один токен контексту: 2 (K і V) × шари × KV-голови × розмір голови × 2 байти.

    Для мультимодальних моделей береться text_config. None — у config немає потрібних полів
    (напр. MLA у DeepSeek): оцінка контексту тоді невідома, а не нульова."""
    c = config.get("text_config") or config
    try:
        layers = int(c["num_hidden_layers"])
        heads = int(c["num_attention_heads"])
        kv_heads = int(c.get("num_key_value_heads") or heads)
        head_dim = int(c.get("head_dim") or int(c["hidden_size"]) // heads)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return 2 * layers * kv_heads * head_dim * _KV_BYTES


@dataclass(frozen=True)
class Fit:
    weights_gib: float
    kv_kib_per_token: float | None
    max_context: int | None  # межа моделі з config (max_position_embeddings)
    per_card: dict[int, dict[str, Any]]  # обсяг карти, MiB -> {usable_gib, fits, max_context_tokens}
    warnings: list[str]


def estimate_fit(files: dict[str, int], config: dict[str, Any], card_mib: list[int], fraction: float,
                 overhead_gib: float) -> Fit:
    """Умовна оцінка для однієї карти кожного обсягу з card_mib.

    usable = обсяг × fraction − ваги − overhead; max_context_tokens = usable / KV на токен (не більше межі моделі).
    fits — чи лишається хоч щось на KV-кеш."""
    weights = weight_bytes(files)
    kv = kv_bytes_per_token(config)
    c = config.get("text_config") or config
    model_ctx = c.get("max_position_embeddings")
    warnings = []
    quant = str((config.get("quantization_config") or {}).get("quant_method", "")).lower()
    if any(q in quant for q in _FP4_FORMATS):
        warnings.append(f"quantization {quant} has no hardware support below sm_100 (Blackwell): emulated or refused")
    elif quant == "fp8":
        warnings.append("fp8 below sm_89 (e.g. Ampere) runs as weight-only W8A16 (Marlin): memory saved, no speed-up")
    if kv is None:
        warnings.append("KV cache size unknown from config.json (non-standard attention); context estimate skipped")
    per_card = {}
    for mib in sorted(set(card_mib)):
        usable = mib * 1024 * 1024 * fraction - weights - overhead_gib * _GIB
        tokens = int(usable // kv) if kv and usable > 0 else (0 if usable <= 0 else None)
        if tokens is not None and isinstance(model_ctx, int):
            tokens = min(tokens, model_ctx)
        per_card[mib] = {"usable_gib": round(usable / _GIB, 1), "fits": usable > 0, "max_context_tokens": tokens}
    return Fit(
        weights_gib=round(weights / _GIB, 2),
        kv_kib_per_token=round(kv / 1024, 1) if kv else None,
        max_context=model_ctx if isinstance(model_ctx, int) else None,
        per_card=per_card,
        warnings=warnings,
    )


class HubClient:
    """Тонка обгортка над HfApi: мапить помилки HF у ManagerError і не пише в кеш моделей.

    token — функція, бо людина може вписати токен у secrets.json без перезапуску менеджера."""

    def __init__(self, token: Callable[[], str | None]) -> None:
        self._token = token

    def _api(self) -> HfApi:
        return HfApi(token=self._token() or False)

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Моделі за назвою, найпопулярніші першими."""
        try:
            found = self._api().list_models(search=query, sort="downloads", limit=limit,
                                            expand=["downloads", "likes", "gated", "pipeline_tag", "safetensors",
                                                    "lastModified"])
            return [{"repo": m.id, "downloads": m.downloads, "likes": m.likes, "gated": bool(m.gated),
                     "task": m.pipeline_tag, "params": m.safetensors.total if m.safetensors else None,
                     "updated": m.last_modified.isoformat() if m.last_modified else None} for m in found]
        except HfHubHTTPError as exc:
            raise ManagerError("hf_unavailable", detail=str(exc)[:200]) from exc

    def files(self, repo: str, revision: str | None) -> tuple[str, dict[str, int], bool]:
        """(commit sha, {файл: розмір} лише вибраних для vLLM, чи gated)."""
        try:
            info = self._api().model_info(repo, revision=revision, files_metadata=True)
        except RepositoryNotFoundError as exc:
            raise ManagerError("hf_not_found", repo=repo) from exc
        except HfHubHTTPError as exc:
            raise ManagerError("hf_unavailable", detail=str(exc)[:200]) from exc
        sizes = {s.rfilename: int(s.size or 0) for s in info.siblings or []}
        chosen = select_files(list(sizes))
        return str(info.sha), {n: sizes[n] for n in chosen}, bool(info.gated)

    def check_access(self, repo: str) -> None:
        """ManagerError hf_gated, якщо токена немає або ліцензію моделі не прийнято."""
        try:
            self._api().auth_check(repo)
        except GatedRepoError as exc:
            raise ManagerError("hf_gated", repo=repo) from exc
        except RepositoryNotFoundError as exc:
            raise ManagerError("hf_not_found", repo=repo) from exc
        except HfHubHTTPError as exc:
            raise ManagerError("hf_unavailable", detail=str(exc)[:200]) from exc

    def config(self, repo: str, revision: str) -> dict[str, Any]:
        """config.json моделі прямим HTTP-запитом: hf_hub_download поклав би його в кеш, і недокачана модель
        з'явилася б у списку локальних. Немає файла — порожній словник."""
        url = hf_hub_url(repo, "config.json", revision=revision)
        try:
            r = httpx.get(url, headers=build_hf_headers(token=self._token() or False), timeout=_HTTP_TIMEOUT_S,
                          follow_redirects=True)
        except httpx.HTTPError as exc:
            raise ManagerError("hf_unavailable", detail=str(exc)[:200]) from exc
        if r.status_code in (401, 403):
            raise ManagerError("hf_gated", repo=repo)
        if r.status_code != 200:
            return {}
        try:
            data = r.json()
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}
