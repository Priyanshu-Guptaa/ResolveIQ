"""Parser for already-textual formats: .log, .txt, .csv, .json, .xml.

These never needed a special library -- they just needed to stop being
lumped in with binary formats that also got blindly UTF-8 decoded. .json
gets pretty-printed (and flagged if invalid) since that's a free
correctness check; .csv and .xml are passed through as text, which is
already exactly what the Log Intelligence Engine's regex-based extractor
and parser expect.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.domain.ingestion import FileKind, ParsedFile, ParseWarning

logger = logging.getLogger(__name__)

EXTENSIONS = {".log", ".txt", ".csv", ".json", ".xml"}


class TextParser:
    def can_parse(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in EXTENSIONS

    def parse(self, filename: str, content: bytes) -> ParsedFile:
        text = content.decode("utf-8", errors="replace")
        warnings: list[ParseWarning] = []
        suffix = Path(filename).suffix.lower()

        if suffix == ".json":
            try:
                text = json.dumps(json.loads(text), indent=2)
            except json.JSONDecodeError as exc:
                warnings.append(
                    ParseWarning(message=f"Invalid JSON ({exc}); stored as raw text", severity="warning")
                )

        return ParsedFile(filename=filename, kind=FileKind.TEXT, text=text, warnings=warnings)
