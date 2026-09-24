"""Тексти відмов і попереджень двома мовами.

Навіщо: одна й та сама відмова йде і на вебсторінку (українською), і в термінал через MCP (англійською —
правило 30-python для всього, що бачить машина). Ядро знає лише код і параметри; мову обирає оболонка."""

from __future__ import annotations

from typing import Any

EN = "en"
UK = "uk"

# код -> (англійською, українською); параметри підставляються через str.format
_TEXTS: dict[str, tuple[str, str]] = {
    "unknown_gpu": ("No GPU {gpu}; valid: 0..{last}", "Немає GPU {gpu}; є 0..{last}"),
    "unknown_user": ("Unknown user {user!r}; allowed: {allowed}", "Невідомий користувач «{user}»; дозволені: {allowed}"),
    "reserved_by_other": (
        "GPU {gpu} is reserved by {owner} ({purpose}). {owner} must release it, or release it with force=true (journaled).",
        "GPU {gpu} зайняв {owner} ({purpose}). Звільнити може {owner} або ви — примусово.",
    ),
    "release_needs_force": (
        "GPU {gpu} is reserved by {owner}; releasing someone else's reservation needs force=true (journaled).",
        "GPU {gpu} зайняв {owner}. Чуже бронювання знімається лише примусово.",
    ),
    "bad_hours": ("hours must be > 0, or omitted for no term", "Строк має бути більше 0 або не вказаний"),
    "bad_history": ("minutes and points must be > 0", "Період і кількість точок мають бути більше 0"),
    "bad_request": ("Bad request: {detail}", "Некоректний запит: {detail}"),
    "hf_unavailable": ("HuggingFace is unavailable: {detail}", "HuggingFace недоступний: {detail}"),
    "hf_not_found": ("No such model on HuggingFace: {repo}", "Такої моделі на HuggingFace немає: {repo}"),
    "hf_gated": ("{repo} is gated: accept its license at https://huggingface.co/{repo} with the account of the token",
                 "{repo} — модель з доступом за ліцензією: прийміть її на https://huggingface.co/{repo} тим акаунтом, чий токен"),
    "bad_fraction": ("fraction must be in (0, 1]", "Частка карти має бути від 0 до 1"),
    "disk_full": ("Not enough disk: need {need_gib} GiB, free {free_gib} GiB, reserve {reserve_gib} GiB",
                  "Мало місця: треба {need_gib} GiB, вільно {free_gib} GiB, запас {reserve_gib} GiB"),
    "download_active": ("{repo} is downloading; cancel it first", "{repo} зараз завантажується; спершу скасуйте"),
    "download_not_active": ("{repo} is not downloading", "{repo} зараз не завантажується"),
    "model_not_local": ("{repo} is not on disk", "{repo} немає на диску"),
    "model_running": ("{repo} is running; stop it first", "{repo} зараз запущена; спершу зупиніть"),
    "start_failed": ("Could not start the model unit: {detail}", "Не вдалося запустити модель: {detail}"),
    "bad_extra_args": ("extra_args must be strings and must not set {flags}", "Додаткові аргументи — рядки й без {flags}"),
    "gpu_memory_low": ("GPU {gpu}: need {need_gib} GiB, free {free_gib} GiB; lower fraction or pick another GPU",
                       "GPU {gpu}: треба {need_gib} GiB, вільно {free_gib} GiB; зменште частку або візьміть іншу карту"),
    "empty_answer": ("The model returned no text (finish_reason: {reason}); raise max_tokens", "Модель не дала тексту ({reason}); збільште max_tokens"),
    "bad_pdf": ("Cannot read the PDF: {detail}", "Не вдалося прочитати PDF: {detail}"),
    "stop_failed": ("Could not stop {name}; it stays under watch", "Не вдалося зупинити {name}; модель лишається під наглядом"),
    "server_not_found": ("No running model named {name}", "Немає запущеної моделі {name}"),
    "bad_name": ("Bad name {name!r}: a-z, 0-9, . _ -, up to 48 chars", "Погана назва «{name}»: a-z, 0-9, . _ -, до 48 символів"),
    "name_taken": ("A running model is already named {name}", "Модель з назвою {name} уже запущена"),
    "bad_port": ("Port {port} is outside {lo}..{hi}", "Порт {port} поза {lo}..{hi}"),
    "port_taken": ("Port {port} is taken", "Порт {port} зайнятий"),
    "no_free_port": ("No free port in {lo}..{hi}", "Немає вільного порту в {lo}..{hi}"),
    "hint_kv_too_small": ("Context does not fit the KV cache: lower max_model_len or raise fraction",
                          "Контекст не вміщується в KV-кеш: зменште max_model_len або збільште частку"),
    "hint_kv_len": ("Context does not fit the KV cache: start again with max_model_len={n} (or raise fraction)",
                    "Контекст не вміщується в KV-кеш: запустіть з max_model_len={n} (або збільште частку)"),
    "hint_no_kv_memory": ("No memory left for the KV cache: raise fraction or use a smaller model",
                          "Не лишилося пам'яті на KV-кеш: збільште частку або візьміть меншу модель"),
    "hint_gpu_memory_taken": ("The GPU has less free memory than the fraction: lower fraction or pick another GPU",
                              "На карті менше вільної пам'яті, ніж частка: зменште частку або візьміть іншу карту"),
    "hint_oom": ("CUDA out of memory: lower max_model_len or raise fraction", "Не вистачило пам'яті GPU: зменште max_model_len або збільште частку"),
    "hint_remote_code": ("The model needs custom code: add --trust-remote-code to extra_args",
                         "Моделі потрібен власний код: додайте --trust-remote-code у додаткові аргументи"),
    "hint_port_in_use": ("The port was taken by another process; start again", "Порт зайняв інший процес; запустіть ще раз"),
    "hint_unsupported": ("vLLM does not support this model or format here; see model_logs",
                         "vLLM не підтримує цю модель чи формат тут; див. лог"),
    "hint_mamba_seqs": ("Too many parallel sequences for the Mamba cache: start again with --max-num-seqs {n}",
                        "Забагато паралельних послідовностей для Mamba-кешу: запустіть з --max-num-seqs {n}"),
    "hint_compile": ("Kernel compilation failed; retried with --enforce-eager", "Не вдалося скомпілювати ядра; спробувано з --enforce-eager"),
    "hint_start_timeout": ("The model did not become ready in time; see model_logs", "Модель не стала готовою вчасно; див. лог"),
    "hint_see_logs": ("The model stopped; see model_logs", "Модель зупинилася; див. лог"),
    "bad_prompt": ("Prompt text must be non-empty and at most {max} characters", "Текст промпту — непорожній, до {max} символів"),
    "prompt_not_found": ("No saved prompt {name}", "Немає збереженого промпту {name}"),
    "model_not_running": ("Model {name} is not running (see models_running)", "Модель {name} не запущена"),
    "foreign_processes": (
        "Other users' processes already run on GPU {gpu}: {processes}",
        "На GPU {gpu} уже працюють чужі процеси: {processes}",
    ),
}

# Порожня мета бронювання в тексті виглядала б як «()».
_NO_PURPOSE = {EN: "no purpose given", UK: "мету не вказано"}


def text(code: str, lang: str, **params: Any) -> str:
    """Текст повідомлення.

    code — ключ із _TEXTS; lang — EN або UK; params — підстановки.
    Невідомий код повертається як є: краще сирий код на екрані, ніж падіння на показі помилки."""
    pair = _TEXTS.get(code)
    if pair is None:
        return code
    if "purpose" in params and not params["purpose"]:
        params = {**params, "purpose": _NO_PURPOSE[lang]}
    template = pair[0] if lang == EN else pair[1]
    try:
        return template.format(**params)
    except (KeyError, IndexError, ValueError):
        # Бракує параметра: показати шаблон і те, що є, — помилка показу не має валити сам запит.
        return f"{template} {params}" if params else template


class ManagerError(Exception):
    """Відмова, яку викликач має прочитати: невідомий користувач, карта зайнята, поганий аргумент.

    code і params — для оболонки, що перекладе; str(exc) — англійською, для логів і терміналу."""

    def __init__(self, code: str, **params: Any) -> None:
        self.code = code
        self.params = params
        super().__init__(text(code, EN, **params))

    def text(self, lang: str) -> str:
        return text(self.code, lang, **self.params)
