"""Перетворення PDF для чату: текст сторінок або сторінки картинками (вибір людини біля вкладення).

vLLM не приймає файлових вкладень, тож PDF моделі можна дати лише текстом або картинками; що з цього їй
підходить — вирішує людина й модель, а не сервер."""

from __future__ import annotations

import base64
import io

from .messages import ManagerError

# Масштаб рендеру: 1.5 від 72 dpi = 108 dpi — текст сторінки читається, а картинка лишається помірною.
_RENDER_SCALE = 1.5
_JPEG_QUALITY = 85
# Більший PDF у чат не має сенсу: і текстом, і картинками він не влізе в контекст моделі.
_MAX_BYTES = 100 * 1024 * 1024


def decode(data_url: str) -> bytes:
    """Байти з data:...;base64,..."""
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url, validate=False)
    except (ValueError, IndexError) as exc:
        raise ManagerError("bad_pdf", detail="not base64") from exc
    if len(raw) > _MAX_BYTES:
        raise ManagerError("bad_pdf", detail=f"larger than {_MAX_BYTES // 2**20} MiB")
    return raw


def _open(raw: bytes):
    import pypdfium2 as pdfium

    try:
        return pdfium.PdfDocument(raw)
    except pdfium.PdfiumError as exc:
        raise ManagerError("bad_pdf", detail=str(exc)[:200]) from exc


def to_text(raw: bytes) -> dict[str, object]:
    """{pages, text}: текст кожної сторінки з позначкою [page N]."""
    pdf = _open(raw)
    try:
        parts = [f"[page {i + 1}]\n{pdf[i].get_textpage().get_text_range().strip()}" for i in range(len(pdf))]
        return {"pages": len(pdf), "text": "\n\n".join(parts)}
    finally:
        pdf.close()


def to_images(raw: bytes) -> dict[str, object]:
    """{pages, images}: кожна сторінка — JPEG як data URL."""
    pdf = _open(raw)
    try:
        images = []
        for i in range(len(pdf)):
            buf = io.BytesIO()
            page = pdf[i].render(scale=_RENDER_SCALE)  # pyright: ignore[reportArgumentType] — у заглушці int, float працює
            page.to_pil().convert("RGB").save(buf, format="JPEG", quality=_JPEG_QUALITY)
            images.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
        return {"pages": len(pdf), "images": images}
    finally:
        pdf.close()
