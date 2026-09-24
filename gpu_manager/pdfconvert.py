"""Перетворення PDF для чату: текст сторінок або сторінки картинками (вибір людини біля вкладення).

vLLM не приймає файлових вкладень, тож PDF моделі можна дати лише текстом або картинками; що з цього їй
підходить — вирішує людина й модель, а не сервер."""

from __future__ import annotations

import base64
import io
import threading
from typing import Any

from .messages import ManagerError

# Масштаб рендеру: 1.5 від 72 dpi = 108 dpi — текст сторінки читається, а картинка лишається помірною.
_RENDER_SCALE = 1.5
# Довша сторона картинки сторінки, px: сторінка 200×200 дюймів дала б 1,4 ГБ растру (захист Pillow від «бомб»
# тут не спрацьовує — растр малює PDFium), а моделі більше й не треба.
_MAX_SIDE_PX = 2000
_JPEG_QUALITY = 85
# Більший PDF у чат не має сенсу: і текстом, і картинками він не влізе в контекст моделі.
MAX_BYTES = 100 * 1024 * 1024
_MAX_PAGES_TEXT = 500
_MAX_PAGES_IMAGES = 100  # сторінка-картинка ~0,1–0,3 МБ base64: більше — сотні МБ в одній репліці
# PDFium не можна викликати з кількох потоків одночасно навіть для різних документів (падає весь процес),
# а конвертації йдуть у робочих потоках: по черзі.
_PDFIUM = threading.Lock()


def decode(data_url: str) -> bytes:
    """Байти з data:...;base64,..."""
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url, validate=False)
    except (ValueError, IndexError) as exc:
        raise ManagerError("bad_pdf", detail="not base64") from exc
    if len(raw) > MAX_BYTES:
        raise ManagerError("bad_pdf", detail=f"larger than {MAX_BYTES // 2**20} MiB")
    return raw


def _convert(raw: bytes, max_pages: int, page_fn: Any) -> tuple[int, list[Any]]:
    """(сторінок, [page_fn(сторінка, номер)]) під замком PDFium; будь-яка помилка PDFium — bad_pdf."""
    import pypdfium2 as pdfium

    with _PDFIUM:
        try:
            pdf = pdfium.PdfDocument(raw)
        except pdfium.PdfiumError as exc:
            raise ManagerError("bad_pdf", detail=str(exc)[:200]) from exc
        try:
            pages = len(pdf)
            if pages > max_pages:
                raise ManagerError("bad_pdf", detail=f"{pages} pages; at most {max_pages} in this mode")
            out = []
            for i in range(pages):
                try:
                    out.append(page_fn(pdf[i], i))
                except pdfium.PdfiumError as exc:
                    raise ManagerError("bad_pdf", detail=f"page {i + 1}: {exc}"[:200]) from exc
            return pages, out
        finally:
            pdf.close()


def to_text(raw: bytes) -> dict[str, object]:
    """{pages, text}: текст кожної сторінки з позначкою [page N]."""
    pages, parts = _convert(raw, _MAX_PAGES_TEXT,
                            lambda page, i: f"[page {i + 1}]\n{page.get_textpage().get_text_range().strip()}")
    return {"pages": pages, "text": "\n\n".join(parts)}


def _jpeg(page: Any, i: int) -> str:
    w, h = page.get_size()
    scale = min(_RENDER_SCALE, _MAX_SIDE_PX / max(w, h, 1))
    buf = io.BytesIO()
    page.render(scale=scale).to_pil().convert("RGB").save(buf, format="JPEG", quality=_JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def to_images(raw: bytes) -> dict[str, object]:
    """{pages, images}: кожна сторінка — JPEG як data URL, не більше _MAX_SIDE_PX по довшій стороні."""
    pages, images = _convert(raw, _MAX_PAGES_IMAGES, _jpeg)
    return {"pages": pages, "images": images}
