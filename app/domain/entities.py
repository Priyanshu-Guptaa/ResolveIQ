"""Entity and log-event models produced by the Log Intelligence Engine.

These are the building blocks of ResolveIQ's "common internal event model" --
regardless of whether a log line came from a Java stack trace, an IIS log, or
a Kafka consumer, it is normalized into a :class:`LogEvent` carrying a list
of :class:`ExtractedEntity` objects.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.domain.enums import EntityType, LogLevel


class ExtractedEntity(BaseModel):
    """A single recognized entity (e.g. a Correlation ID or IP address)
    found within a piece of evidence.

    ``context_snippet`` retains a small window of surrounding text so a
    human reviewing recommendations can see *why* the entity was picked up,
    without having to re-open the source log.
    """

    entity_type: EntityType
    value: str
    context_snippet: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_evidence_id: Optional[str] = None

    def __hash__(self) -> int:  # allows de-duplication via set()
        return hash((self.entity_type, self.value))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtractedEntity):
            return NotImplemented
        return self.entity_type == other.entity_type and self.value == other.value


class LogEvent(BaseModel):
    """A single normalized log line/event, regardless of source format.

    This is the common internal event model referenced in the platform
    spec: every :class:`~app.engines.log_intelligence.log_parser.LogParser`
    implementation must produce a list of these, so downstream engines never
    need to know which format the raw log was in.
    """

    raw_line: str
    timestamp: Optional[datetime] = None
    level: LogLevel = LogLevel.UNKNOWN
    message: str = ""
    source_component: Optional[str] = None
    """Best-effort identification of the emitting component (thread name,
    pod name, host, logger name, ...) -- format-specific, kept as free text."""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    line_number: Optional[int] = None
