"""Windows Event Log parser -- python-evtx.

python-evtx's ``Evtx`` reader mmaps a real file, so the uploaded bytes are
written to a temporary file for the duration of parsing. Each record's
raw XML is reduced to a compact one-line summary (EventID, Level,
Provider, TimeCreated) rather than dumping full XML per event -- still
real structured data, just not overwhelming for a log with thousands of
records. Capped at ``_MAX_RECORDS`` so a huge .evtx can't make one upload
pathologically slow.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

from app.domain.ingestion import FileKind, ParsedFile, ParseWarning

logger = logging.getLogger(__name__)

_MAX_RECORDS = 2000
_FIELD_PATTERN = re.compile(
    r'<EventID[^>]*>(?P<event_id>\d+)</EventID>|'
    r"<Level>(?P<level>\d+)</Level>|"
    r'<Provider Name="(?P<provider>[^"]*)"|'
    r'<TimeCreated SystemTime="(?P<time>[^"]*)"'
)


class EvtxParser:
    def can_parse(self, filename: str) -> bool:
        return Path(filename).suffix.lower() == ".evtx"

    def parse(self, filename: str, content: bytes) -> ParsedFile:
        warnings: list[ParseWarning] = []
        lines: list[str] = []
        metadata: dict = {}
        tmp_path: str | None = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".evtx", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            from Evtx.Evtx import Evtx

            record_count = 0
            with Evtx(tmp_path) as evtx:
                for record in evtx.records():
                    record_count += 1
                    if record_count > _MAX_RECORDS:
                        warnings.append(
                            ParseWarning(
                                message=f"Truncated after {_MAX_RECORDS} records", severity="warning"
                            )
                        )
                        break
                    lines.append(_summarize_record(record.xml()))
            metadata = {"record_count": min(record_count, _MAX_RECORDS)}
            if not lines:
                warnings.append(ParseWarning(message="No event records found", severity="warning"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse .evtx %s: %s", filename, exc)
            warnings.append(ParseWarning(message=f"Failed to parse .evtx: {exc}", severity="error"))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return ParsedFile(
            filename=filename, kind=FileKind.EVTX, text="\n".join(lines), warnings=warnings, metadata=metadata
        )


def _summarize_record(xml_text: str) -> str:
    fields = {"event_id": "?", "level": "?", "provider": "?", "time": "?"}
    for match in _FIELD_PATTERN.finditer(xml_text):
        for key, value in match.groupdict().items():
            if value:
                fields[key] = value
    return f"[{fields['time']}] EventID={fields['event_id']} Level={fields['level']} Provider={fields['provider']}"
