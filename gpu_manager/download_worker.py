"""Окремий процес одного завантаження з HuggingFace.

Навіщо окремий процес: завантаження триває хвилини-години; його треба вміти скасувати (SIGTERM) і не блокувати
сервер. Докачування після обриву чи перезапуску менеджера робить сам huggingface_hub: недокачані файли
лишаються як *.incomplete і продовжуються.

Вхід — JSON у stdin: {"repo", "revision", "files"}; HF_HOME і HF_TOKEN — у змінних оточення.
Вихід — код 0 при успіху; при помилці останній рядок stderr: "ERROR <клас>: <повідомлення>"."""

from __future__ import annotations

import json
import sys


def main() -> int:
    job = json.loads(sys.stdin.read())
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(job["repo"], revision=job["revision"], allow_patterns=job["files"])
    except Exception as exc:  # noqa: BLE001 — будь-яка помилка йде в лог; менеджер розбере клас і позначить завантаження
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
