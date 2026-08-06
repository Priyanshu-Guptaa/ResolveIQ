"""Log Intelligence Engine: the orchestration layer over parsing + entity
extraction, applied to Evidence.

This is what other engines and the API talk to -- they never call
:class:`LogParser` or :class:`EntityExtractor` directly.
"""

from __future__ import annotations

import logging
from collections import Counter

from app.domain.entities import ExtractedEntity
from app.domain.evidence import Evidence
from app.domain.enums import EvidenceType
from app.engines.log_intelligence.entity_extractor import EntityExtractor
from app.engines.log_intelligence.log_parser import LogParser

logger = logging.getLogger(__name__)


class LogIntelligenceEngine:
    """Enriches Evidence with parsed log events and extracted entities.

    Works on any evidence, not just ``LOG_FILE``: task descriptions and
    manual notes are run through the entity extractor too (a description
    that mentions "meter 12345678" or a correlation ID is just as valuable
    an anchor as one found in a log line), while full log parsing
    (``log_events``) is reserved for ``LOG_FILE`` evidence.
    """

    def __init__(self, parser: LogParser, extractor: EntityExtractor) -> None:
        self._parser = parser
        self._extractor = extractor

    def analyze_evidence(self, evidence: Evidence) -> Evidence:
        """Populate ``evidence.entities`` (and ``log_events`` for log
        files) in place, and return it for convenient chaining."""
        if not evidence.raw_content:
            return evidence

        if evidence.evidence_type == EvidenceType.LOG_FILE:
            events = self._parser.parse(evidence.raw_content, source_evidence_id=evidence.id)
            evidence.log_events = events
            entities: list[ExtractedEntity] = []
            seen: set[tuple[str, str]] = set()
            for event in events:
                for entity in event.entities:
                    key = (entity.entity_type.value, entity.value.lower())
                    if key not in seen:
                        seen.add(key)
                        entities.append(entity)
            evidence.extracted_entities = entities
        else:
            evidence.extracted_entities = self._extractor.extract(
                evidence.raw_content, source_evidence_id=evidence.id
            )

        logger.info(
            "Log Intelligence: evidence %s (%s) -> %d entities, %d log events",
            evidence.id,
            evidence.evidence_type.value,
            len(evidence.extracted_entities),
            len(evidence.log_events),
        )
        return evidence

    @staticmethod
    def summarize_entities(entities: list[ExtractedEntity]) -> dict[str, int]:
        """Small helper for UI display: counts entities per type."""
        counts = Counter(entity.entity_type.value for entity in entities)
        return dict(counts.most_common())
