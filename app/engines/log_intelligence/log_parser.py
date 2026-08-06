"""Log parsing: turning raw log text (any format) into a list of
:class:`~app.domain.entities.LogEvent` -- the common internal event model.

Rather than writing one parser per log format (Java, IIS, Kafka, ...), this
module uses one generic, format-tolerant line parser: it heuristically
detects a timestamp and level at the start of a line if present, folds
continuation lines (stack trace frames, wrapped messages) into the
preceding event, and always populates ``entities`` via the
:class:`~app.engines.log_intelligence.entity_extractor.EntityExtractor`.
This covers the formats in the sample data (Java stack traces, IIS,
Kafka consumer logs, meter/collector logs) without per-format code paths,
and new formats typically just need a new timestamp pattern below.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Protocol

from dateutil import parser as dateutil_parser

from app.domain.entities import LogEvent
from app.domain.enums import LogLevel
from app.engines.log_intelligence.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)

_LEVEL_ALIASES: dict[str, LogLevel] = {
    "TRACE": LogLevel.TRACE,
    "DEBUG": LogLevel.DEBUG,
    "INFO": LogLevel.INFO,
    "WARN": LogLevel.WARN,
    "WARNING": LogLevel.WARN,
    "ERROR": LogLevel.ERROR,
    "SEVERE": LogLevel.ERROR,
    "FATAL": LogLevel.FATAL,
    "CRITICAL": LogLevel.FATAL,
}

_LEVEL_PATTERN = re.compile(
    r"\b(" + "|".join(_LEVEL_ALIASES.keys()) + r")\b",
)

# A handful of timestamp shapes covering the sample formats: ISO-8601,
# log4j-style with comma millis, syslog-ish "Mon DD HH:MM:SS", and bare
# "YYYY-MM-DD HH:MM:SS".
_TIMESTAMP_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"),
    re.compile(r"^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"),
]

_COMPONENT_PATTERN = re.compile(r"\[([^\]]+)\]")

_CONTINUATION_PATTERN = re.compile(r"^\s*(at\s+\S|Caused by:|\.\.\.\s*\d+\s+more|\s{2,}\S)")
"""Lines matching this are folded into the previous event: stack-trace
frames, "Caused by" chains, and indented wrapped text."""


class LogParser(Protocol):
    """Anything that turns raw log content into normalized events."""

    def parse(self, raw_text: str, *, source_evidence_id: str | None = None) -> list[LogEvent]:
        ...


class GenericLogParser:
    """Format-tolerant line-oriented log parser.

    Not a strict grammar for any one format -- a pragmatic heuristic parser
    that works reasonably well across Java/log4j, IIS, Kafka, syslog-ish,
    and plain application logs, which is what Sprint 1 needs to prove the
    Log Intelligence Engine's value without a parser-per-format investment.
    """

    def __init__(self, entity_extractor: EntityExtractor) -> None:
        self._entity_extractor = entity_extractor

    def parse(self, raw_text: str, *, source_evidence_id: str | None = None) -> list[LogEvent]:
        if not raw_text:
            return []

        events: list[LogEvent] = []
        current: LogEvent | None = None

        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            if not line.strip():
                continue

            if current is not None and _CONTINUATION_PATTERN.match(line):
                current.raw_line += "\n" + line
                current.message += "\n" + line.strip()
                continue

            timestamp = _extract_timestamp(line)
            level = _extract_level(line)
            component = _extract_component(line)

            current = LogEvent(
                raw_line=line,
                timestamp=timestamp,
                level=level,
                message=line.strip(),
                source_component=component,
                line_number=line_number,
            )
            events.append(current)

        for event in events:
            event.entities = self._entity_extractor.extract(
                event.raw_line, source_evidence_id=source_evidence_id
            )

        logger.debug("Parsed %d log events from %d lines", len(events), raw_text.count("\n") + 1)
        return events


def _extract_timestamp(line: str) -> datetime | None:
    for pattern in _TIMESTAMP_PATTERNS:
        match = pattern.match(line.strip())
        if match:
            try:
                return dateutil_parser.parse(match.group(0))
            except (ValueError, OverflowError):
                continue
    return None


def _extract_level(line: str) -> LogLevel:
    match = _LEVEL_PATTERN.search(line)
    if not match:
        return LogLevel.UNKNOWN
    return _LEVEL_ALIASES.get(match.group(1).upper(), LogLevel.UNKNOWN)


def _extract_component(line: str) -> str | None:
    match = _COMPONENT_PATTERN.search(line)
    return match.group(1) if match else None
