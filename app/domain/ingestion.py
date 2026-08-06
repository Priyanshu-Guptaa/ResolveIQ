"""Evidence Ingestion Pipeline domain models (RFC rev 1, §04).

The whole point of this pipeline: nothing downstream (entity extraction,
the Recommendation Engine's context text) ever sees raw bytes again.
Every uploaded file becomes a :class:`ParsedFile` with real extracted
text -- or, for a format nothing here understands, an honest "unsupported"
result with empty text, never a UTF-8 decode of binary content.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FileKind(str, Enum):
    TEXT = "text"
    DOCX = "docx"
    XLSX = "xlsx"
    PDF = "pdf"
    IMAGE = "image"
    EVTX = "evtx"
    ZIP = "zip"
    UNSUPPORTED = "unsupported"


class ParseWarning(BaseModel):
    message: str
    severity: str = "info"
    """"info" | "warning" | "error" -- error means parsing failed outright
    (text will be empty); warning means partial/degraded extraction
    (e.g. OCR unavailable, a page had no extractable text)."""


class ParsedFile(BaseModel):
    """The normalized output of the ingestion pipeline for one file.

    ``text`` is what Log Intelligence's entity extractor and the
    Recommendation Engine's context_text ever see -- never the original
    bytes. For a .zip, the top-level ParsedFile's ``text`` is empty and
    ``child_files`` holds one ParsedFile per archive entry (recursively
    flattened by the registry, not nested more than necessary).
    """

    filename: str
    kind: FileKind
    text: str = ""
    warnings: list[ParseWarning] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    """Parser-specific detail: page_count, sheet_names, ocr_applied, etc."""
