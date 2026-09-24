"""Файли для моделей через спільну теку (SPEC-GPU-003 §6): агент кладе файл у свою теку й передає лише шлях.

Картинки, аудіо й відео модель (vLLM з --allowed-local-media-path) читає сама за посиланням file://; текст і PDF
менеджер вставляє в повідомлення сам. Вміст файлів іде в модель повз токени агента."""

from __future__ import annotations

import grp
import os
import pwd
from pathlib import Path
from typing import Any, cast

from . import pdfconvert
from .core import require_user
from .messages import ManagerError

# Розширення -> тип частини повідомлення OpenAI, яку vLLM завантажує сам за посиланням file://.
_MEDIA = {**dict.fromkeys((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"), "image_url"),
          **dict.fromkeys((".wav", ".mp3", ".flac", ".ogg", ".m4a"), "audio_url"),
          **dict.fromkeys((".mp4", ".webm", ".mov", ".mkv", ".avi"), "video_url")}
# Текстовий файл більший за це в контекст моделі однаково не влізе (1 MiB ≈ 250 тис. токенів).
_MAX_TEXT_BYTES = 1024 * 1024
# Файлів в одному запиті: більше — майже напевно помилка агента, а не задум.
_MAX_FILES = 20
_PDF_MODES = ("text", "images")
# Права для rsync: група читає теки й файли — інакше ні менеджер, ні модель їх не прочитають (теки 2750).
_CHMOD = "--chmod=Dg+rx,Fg+r"


class Inbox:
    """Тека файлів root з теками користувачів <root>/<логін>/; host — адреса сервера для копіювання по SSH."""

    def __init__(self, root: Path, users: tuple[str, ...], host: str) -> None:
        self._root = root
        self._users = users
        self._host = host

    def info(self, user: str) -> dict[str, Any]:
        """Куди агентові класти файли й як їх потім передати моделі."""
        require_user(self._users, user)
        d = f"{self._root}/{user}/"
        return {"dir": d, "ready": (self._root / user).is_dir(), "file_url_prefix": f"file://{d}",
                "copy_local": f"rsync -a {_CHMOD} <src> {d}<task>/",
                "copy_remote": f"rsync -a --partial {_CHMOD} <src> {user}@{self._host}:{d}<task>/",
                "use": (f"llm_ask(model, prompt, files=['{d}<task>/<file>']): images, audio and video are read by "
                        f"the model itself, text and PDF are inserted by the server. /v1: image_url / audio_url / "
                        f"video_url with url 'file://{d}<task>/<file>'.")}

    def parts(self, paths: list[str], pdf_mode: str = "text") -> list[dict[str, Any]]:
        """Частини повідомлення OpenAI для файлів з теки, у порядку paths."""
        if pdf_mode not in _PDF_MODES:
            raise ManagerError("bad_request", detail=f"pdf_mode must be one of {', '.join(_PDF_MODES)}")
        if len(paths) > _MAX_FILES:
            raise ManagerError("bad_file", path=f"{len(paths)} files", detail=f"at most {_MAX_FILES} per request")
        out: list[dict[str, Any]] = []
        for p in paths:
            out += self._part(self._checked(p), pdf_mode)
        return out

    def _checked(self, path: str) -> Path:
        """Розкритий шлях (з посиланнями) усередині теки; інакше bad_file_path — модель не читає чужого диска."""
        try:
            resolved = Path(path).expanduser().resolve()
            if self._root.resolve() in resolved.parents and resolved.is_file():
                return resolved
        except OSError:  # напр. немає прав на теку
            pass
        raise ManagerError("bad_file_path", path=path)

    @staticmethod
    def _part(path: Path, pdf_mode: str) -> list[dict[str, Any]]:
        kind = _MEDIA.get(path.suffix.lower())
        if kind is not None:
            return [{"type": kind, kind: {"url": path.as_uri()}}]
        if path.suffix.lower() == ".pdf":
            if path.stat().st_size > pdfconvert.MAX_BYTES:
                raise ManagerError("bad_pdf", detail=f"larger than {pdfconvert.MAX_BYTES // 2**20} MiB")
            raw = path.read_bytes()
            if pdf_mode == "images":
                images = cast(list[str], pdfconvert.to_images(raw)["images"])
                return [{"type": "image_url", "image_url": {"url": url}} for url in images]
            conv = pdfconvert.to_text(raw)
            return [{"type": "text", "text": f"[file {path.name}, {conv['pages']} pages]\n{conv['text']}"}]
        if path.stat().st_size > _MAX_TEXT_BYTES:
            raise ManagerError("bad_file", path=str(path), detail=f"larger than {_MAX_TEXT_BYTES // 1024} KiB")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ManagerError("bad_file", path=str(path), detail=str(exc)[:200]) from exc
        except UnicodeDecodeError as exc:
            raise ManagerError("bad_file", path=str(path), detail="not UTF-8 text (images, audio, video and PDF "
                               "are recognised by extension)") from exc
        return [{"type": "text", "text": f"[file {path.name}]\n{text}"}]


def run_group(root: Path) -> str | None:
    """Група, з якою запускати моделі, щоб вони читали теки користувачів (права 2750): група теки файлів, якщо
    користувач сервісу в ній, але вона не основна. Юніти systemd --user отримують лише групи, які менеджер systemd
    користувача мав при своєму старті, а його могли запустити до того, як користувача додали в групу.
    None — обгортка не потрібна чи неможлива (теки немає, користувач не в групі)."""
    try:
        gid = root.stat().st_gid
        group = grp.getgrgid(gid)
        me = pwd.getpwuid(os.getuid())
    except (OSError, KeyError):
        return None
    if gid == me.pw_gid or me.pw_name not in group.gr_mem:
        return None
    return group.gr_name
