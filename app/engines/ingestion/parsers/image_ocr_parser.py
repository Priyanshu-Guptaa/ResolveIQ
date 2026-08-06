"""Image OCR parser -- pytesseract + Pillow.

Best-effort by design (RFC rev 1 §04 sign-off): OCR needs the
``tesseract`` binary on the host, not just the Python package. When it's
missing, this degrades to an empty-text result with a warning rather than
crashing the upload -- the file still becomes real (if textless) Evidence
instead of failing the whole request.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from app.domain.ingestion import FileKind, ParsedFile, ParseWarning

logger = logging.getLogger(__name__)

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def ocr_image_bytes(content: bytes) -> tuple[str, list[ParseWarning], dict]:
    """Shared by ImageOcrParser and PdfParser's scanned-page fallback."""
    warnings: list[ParseWarning] = []
    text = ""
    metadata: dict = {}
    try:
        import pytesseract
        from PIL import Image

        image = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(image)
        metadata = {"ocr_applied": True, "image_size": list(image.size)}
        if not text.strip():
            warnings.append(ParseWarning(message="OCR produced no text", severity="warning"))
    except ImportError as exc:
        warnings.append(ParseWarning(message=f"OCR dependency missing: {exc}", severity="warning"))
    except Exception as exc:  # noqa: BLE001 -- most commonly: tesseract binary not on PATH
        logger.warning("OCR unavailable or failed: %s", exc)
        warnings.append(
            ParseWarning(
                message=f"OCR unavailable on this host ({exc}) -- image stored without extracted text",
                severity="warning",
            )
        )
    return text, warnings, metadata


class ImageOcrParser:
    def can_parse(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in EXTENSIONS

    def parse(self, filename: str, content: bytes) -> ParsedFile:
        text, warnings, metadata = ocr_image_bytes(content)
        return ParsedFile(filename=filename, kind=FileKind.IMAGE, text=text, warnings=warnings, metadata=metadata)
