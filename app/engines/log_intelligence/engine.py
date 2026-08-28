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
from app.domain.enums import EntityType, EvidenceType
from app.domain.log_flow import LogEventCount, LogObservationSummary
from app.engines.log_intelligence.entity_extractor import EntityExtractor
from app.engines.log_intelligence.log_parser import LogParser

logger = logging.getLogger(__name__)

_MAX_TOP_EXCEPTIONS = 5
"""Same capping discipline as ``log_intelligence/flow.py``'s
``_MAX_SEARCH_HITS``/``PromptBuilder``'s snippet caps -- a chat answer
needs the handful of most-frequent patterns, not an exhaustive dump."""


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

    @staticmethod
    def summarize_observations(evidence_items: list[Evidence]) -> LogObservationSummary | None:
        """Chat Assistant Phase 33 -- the ``DETERMINISTIC PARSER ->
        COMPACT STRUCTURED EVIDENCE`` step: reduces every already-parsed
        ``LOG_FILE`` evidence item's ``log_events``/``extracted_entities``
        (populated by ``analyze_evidence`` above -- this method parses
        nothing itself, it only aggregates) into a small
        :class:`LogObservationSummary` safe to place in an LLM prompt.

        Returns ``None`` when there is no LOG_FILE evidence with any
        parsed events at all -- the caller (``ChatOrchestrator``) must
        then omit the LOG OBSERVATIONS section entirely rather than
        rendering an empty one, the same "only render a section when
        there is something to render" discipline ``PromptBuilder``
        already follows for root cause/resolution/validation steps.

        Deliberately produces ONLY counts and already-recognized entity
        values (see ``LogObservationSummary``'s own docstring for why
        that is what makes this safe against prompt injection) -- never
        a raw log line, never free-form message text, so an attacker-
        controlled sentence embedded in a log (e.g. "IGNORE ALL
        PREVIOUS INSTRUCTIONS...") has no field here it could ever
        occupy: it does not look like a severity level, a timestamp, or
        a recognized exception/error-code entity, so it is simply never
        aggregated in the first place -- not filtered out after the
        fact, structurally never captured."""
        log_evidence = [e for e in evidence_items if e.evidence_type == EvidenceType.LOG_FILE and e.log_events]
        if not log_evidence:
            return None

        level_counts: Counter[str] = Counter()
        exception_counts: Counter[str] = Counter()
        earliest = None
        latest = None
        total_events = 0

        for evidence in log_evidence:
            for event in evidence.log_events:
                total_events += 1
                level_counts[event.level.value] += 1
                if event.timestamp is not None:
                    if earliest is None or event.timestamp < earliest:
                        earliest = event.timestamp
                    if latest is None or event.timestamp > latest:
                        latest = event.timestamp
                for entity in event.entities:
                    if entity.entity_type == EntityType.EXCEPTION_TYPE:
                        exception_counts[entity.value] += 1

        return LogObservationSummary(
            source_evidence_ids=[e.id for e in log_evidence],
            analyzed_file_count=len(log_evidence),
            total_events=total_events,
            level_counts=[LogEventCount(label=label, count=count) for label, count in level_counts.most_common()],
            top_exceptions=[
                LogEventCount(label=label, count=count)
                for label, count in exception_counts.most_common(_MAX_TOP_EXCEPTIONS)
            ],
            earliest_timestamp=earliest,
            latest_timestamp=latest,
        )
