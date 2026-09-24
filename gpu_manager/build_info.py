"""Версія, коміт і дата коміту, з яких запущено сервер: показуються внизу сторінки.

Коміт читається з git один раз при старті — після нового коміту сервіс треба перезапустити."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from . import __version__

_REPO = Path(__file__).resolve().parent.parent
# Скільки чекати git, секунд: сторінка не має залежати від зависання git.
_GIT_TIMEOUT_S = 5


def build_info() -> dict[str, Any]:
    """{version, commit (скорочений хеш або None, якщо комітів ще немає), date (ISO або None), dirty}."""
    def git(*args: str) -> str | None:
        try:
            r = subprocess.run(["git", "-C", str(_REPO), *args], capture_output=True, text=True, timeout=_GIT_TIMEOUT_S,
                               check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    commit = git("rev-parse", "--short", "HEAD")
    return {
        "version": __version__,
        "commit": commit,
        "date": git("log", "-1", "--format=%cI") if commit else None,
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")) if commit else None,
    }
