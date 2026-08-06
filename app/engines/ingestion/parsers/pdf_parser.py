"""PDF parser -- PyMuPDF (fitz), with a best-effort OCR fallback for
pages that look scanned (little to no extractable text layer).
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.domain.ingestion import FileKind, ParsedFile, ParseWarning
from app.engines.ingestion.parsers.image_ocr_parser import ocr_image_bytes

logger = logging.getLogger(__name__)

_MIN_CHARS_BEFORE_OCR_FALLBACK = 20
"""A page yielding fewer characters than this is treated as likely
scanned and gets an OCR attempt on its rendered image."""


class PdfParser:
    def can_parse(self, filename: str) -> bool:
        return Path(filename).suffix.lower() == ".pdf"

    def parse(self, filename: str, content: bytes) -> ParsedFile:
        warnings: list[ParseWarning] = []
        pages_text: list[str] = []
        ocr_pages = 0

        try:
            import fitz  # PyMuPDF

            document = fitz.open(stream=content, filetype="pdf")
            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text().strip()
                if len(page_text) < _MIN_CHARS_BEFORE_OCR_FALLBACK:
                    pixmap = page.get_pixmap(dpi=150)
                    ocr_text, ocr_warnings, _ = ocr_image_bytes(pixmap.tobytes("png"))
                    if ocr_text.strip():
                        page_text = ocr_text
                        ocr_pages += 1
                    else:
                        warnings.append(
                            ParseWarning(
                                message=f"Page {page_number}: little/no extractable text and OCR "
                                "found none -- likely a scanned page OCR couldn't read, or a blank page",
                                severity="warning",
                            )
                        )
                pages_text.append(f"--- Page {page_number} ---\n{page_text}")
            metadata = {"page_count": document.page_count, "ocr_fallback_pages": ocr_pages}
            document.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse .pdf %s: %s", filename, exc)
            warnings.append(ParseWarning(message=f"Failed to parse .pdf: {exc}", severity="error"))
            return ParsedFile(filename=filename, kind=FileKind.PDF, text="", warnings=warnings)

        return ParsedFile(
            filename=filename,
            kind=FileKind.PDF,
            text="\n\n".join(pages_text),
            warnings=warnings,
            metadata=metadata,
        )
