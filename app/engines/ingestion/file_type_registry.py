"""File-type detection and dispatch -- the entry point of the Evidence
Ingestion Pipeline.

``FileTypeRegistry.parse()`` is the only method callers use. It returns a
**list** of :class:`ParsedFile` because a .zip fans out into many; every
other input produces exactly one. Zip recursion lives here rather than in
a "ZipParser" implementing the same one-file-in/one-file-out Protocol as
everything else, because it's fundamentally a different shape of
operation (one input, many outputs) and needs registry access to dispatch
its children -- keeping it here keeps every individual parser simple and
uniform.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Protocol

from app.domain.ingestion import FileKind, ParsedFile, ParseWarning
from app.engines.ingestion.parsers.docx_parser import DocxParser
from app.engines.ingestion.parsers.evtx_parser import EvtxParser
from app.engines.ingestion.parsers.image_ocr_parser import ImageOcrParser
from app.engines.ingestion.parsers.pdf_parser import PdfParser
from app.engines.ingestion.parsers.text_parser import TextParser
from app.engines.ingestion.parsers.xlsx_parser import XlsxParser

logger = logging.getLogger(__name__)

_MAX_ZIP_DEPTH = 3
"""Caps nested-zip recursion (zip-of-zip-of-zip...) so a crafted archive
can't cause unbounded recursion."""
_MAX_ZIP_ENTRIES = 100
"""Caps total files extracted from one archive (including nested), so a
zip bomb-shaped upload can't make one request extract an unbounded number
of files."""


class FileParser(Protocol):
    """Anything that turns one file's bytes into one ParsedFile."""

    def can_parse(self, filename: str) -> bool:
        ...

    def parse(self, filename: str, content: bytes) -> ParsedFile:
        ...


class UnsupportedFileParser:
    """The fallback for anything no registered parser claims. This is the
    direct fix for Problem 1: a file that reaches here is stored as an
    empty-text, UNSUPPORTED-kind result -- it is never ``.decode("utf-8")``'d."""

    def parse(self, filename: str) -> ParsedFile:
        return ParsedFile(
            filename=filename,
            kind=FileKind.UNSUPPORTED,
            text="",
            warnings=[
                ParseWarning(
                    message=f"Unsupported file type ({Path(filename).suffix or 'no extension'}) -- "
                    "stored but not text-analyzed",
                    severity="warning",
                )
            ],
        )


class FileTypeRegistry:
    def __init__(self, parsers: list[FileParser] | None = None) -> None:
        self._parsers: list[FileParser] = parsers or [
            TextParser(),
            DocxParser(),
            XlsxParser(),
            PdfParser(),
            ImageOcrParser(),
            EvtxParser(),
        ]
        self._unsupported = UnsupportedFileParser()

    def parse(self, filename: str, content: bytes, *, _depth: int = 0, _budget: list[int] | None = None) -> list[ParsedFile]:
        """Top-level entry point. Returns 1 ParsedFile for any normal
        file, or 1+ for a .zip (flattened, not nested)."""
        if _budget is None:
            _budget = [_MAX_ZIP_ENTRIES]

        if Path(filename).suffix.lower() == ".zip":
            if _depth >= _MAX_ZIP_DEPTH:
                return [
                    ParsedFile(
                        filename=filename,
                        kind=FileKind.UNSUPPORTED,
                        text="",
                        warnings=[ParseWarning(message="Nested zip depth limit reached", severity="warning")],
                    )
                ]
            return self._parse_zip(filename, content, _depth, _budget)

        return [self._parse_one(filename, content)]

    def _parse_one(self, filename: str, content: bytes) -> ParsedFile:
        for parser in self._parsers:
            if parser.can_parse(filename):
                return parser.parse(filename, content)
        return self._unsupported.parse(filename)

    def _parse_zip(self, filename: str, content: bytes, depth: int, budget: list[int]) -> list[ParsedFile]:
        results: list[ParsedFile] = []
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                for entry_name in archive.namelist():
                    if entry_name.endswith("/"):
                        continue  # directory entry
                    if budget[0] <= 0:
                        results.append(
                            ParsedFile(
                                filename=entry_name,
                                kind=FileKind.UNSUPPORTED,
                                text="",
                                warnings=[
                                    ParseWarning(
                                        message=f"Archive entry limit ({_MAX_ZIP_ENTRIES}) reached",
                                        severity="warning",
                                    )
                                ],
                            )
                        )
                        break
                    budget[0] -= 1
                    entry_bytes = archive.read(entry_name)
                    results.extend(self.parse(entry_name, entry_bytes, _depth=depth + 1, _budget=budget))
        except zipfile.BadZipFile as exc:
            results.append(
                ParsedFile(
                    filename=filename,
                    kind=FileKind.UNSUPPORTED,
                    text="",
                    warnings=[ParseWarning(message=f"Corrupt zip file: {exc}", severity="error")],
                )
            )
        return results
