"""Recommendation Engine: turns an investigation's shared context into
"what should I do next?"

Two complementary signal sources feed every recommendation:

1. **Semantic similarity** to historical investigations/docs/bugs (via the
   Knowledge Engine) -- the strongest signal when a close historical match
   exists.
2. **Entity heuristics** over the investigation's merged entities -- a
   fallback (and supplement) that works even with zero historical
   precedent, using domain knowledge about what each entity type usually
   implies (e.g. a SQL session ID hints at a database investigation).

No LLM call is involved in Sprint 1 -- recommendations are deterministic
and explainable by design, matching the local-only embeddings decision.
A generative "explain this in natural language" layer is a natural, purely
additive Sprint 2+ enhancement on top of this same output shape.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings
from app.domain.entities import ExtractedEntity
from app.domain.enums import EntityType
from app.domain.investigation import InvestigationSession
from app.domain.recommendation import KnowledgeMatch, Recommendation, RootCauseHypothesis
from app.engines.knowledge.engine import KnowledgeEngine

logger = logging.getLogger(__name__)

_MAX_SNIPPET_CHARS = 600
"""Cap for any historical-match text (resolution, root_cause) surfaced
directly in a recommendation. Found necessary on real imported
ServiceNow tickets: `resolution` is the ticket's full "Comments and
Work notes" column -- every work-note entry ever added, sometimes
thousands of characters spanning weeks -- and dumping that verbatim
into "Next best step" (meant to be a short, scannable hint) reads as
unusable wall-of-text, not a recommendation. The full resolution is
still one click away (Historical Matches -> that investigation's own
record); this cap only bounds what's echoed inline."""


def _snippet(text: str) -> str:
    text = text.strip()
    if len(text) <= _MAX_SNIPPET_CHARS:
        return text
    return text[:_MAX_SNIPPET_CHARS].rstrip() + "…"


@dataclass(frozen=True)
class _EntityHeuristic:
    """A rule of thumb: "if entities of this type are present, it's
    plausibly this kind of problem." Kept as data so new heuristics are
    additions, not control-flow changes."""

    entity_type: EntityType
    root_cause_template: str
    confidence: float
    suggested_sql: str | None = None
    next_step_template: str | None = None


_ENTITY_HEURISTICS: list[_EntityHeuristic] = [
    _EntityHeuristic(
        EntityType.EXCEPTION_TYPE,
        "Unhandled application exception: {value}",
        confidence=0.45,
        next_step_template="Search other services/instances for the same exception "
        "'{value}' or the accompanying stack trace to find where it originates.",
    ),
    _EntityHeuristic(
        EntityType.SQL_SESSION,
        "Database session/blocking issue (SQL session {value})",
        confidence=0.4,
        suggested_sql="SELECT * FROM sys.dm_exec_requests WHERE session_id = {value}; "
        "-- check blocking_session_id and wait_type",
        next_step_template="Check whether SQL session {value} is being blocked "
        "(sys.dm_exec_requests.blocking_session_id) or holding long-running locks.",
    ),
    _EntityHeuristic(
        EntityType.KAFKA_TOPIC,
        "Kafka consumer/producer issue on topic {value}",
        confidence=0.35,
        next_step_template="Check consumer lag and broker health for topic '{value}'.",
    ),
    _EntityHeuristic(
        EntityType.RABBITMQ_QUEUE,
        "RabbitMQ queue backlog or connectivity issue on queue {value}",
        confidence=0.35,
        next_step_template="Check queue depth, consumer count, and connection churn for '{value}'.",
    ),
    _EntityHeuristic(
        EntityType.POD_NAME,
        "Kubernetes pod-level failure ({value})",
        confidence=0.3,
        next_step_template="Check pod events and prior container logs: "
        "kubectl describe pod {value} && kubectl logs {value} --previous",
    ),
    _EntityHeuristic(
        EntityType.IP_ADDRESS,
        "Possible network/connectivity issue involving {value}",
        confidence=0.2,
        next_step_template="Verify network reachability and firewall rules for {value}.",
    ),
    _EntityHeuristic(
        EntityType.METER_NUMBER,
        "Meter communication issue (meter {value})",
        confidence=0.3,
        next_step_template="Check the collector/head-end system for the last successful "
        "communication with meter {value}.",
    ),
]


class RecommendationEngine:
    def __init__(self, knowledge_engine: KnowledgeEngine, settings: Settings) -> None:
        self._knowledge = knowledge_engine
        self._settings = settings

    def generate(self, investigation: InvestigationSession) -> Recommendation:
        query_text = investigation.context_text
        entities = investigation.merged_entities

        similar_investigations: list[KnowledgeMatch] = []
        relevant_docs: list[KnowledgeMatch] = []
        known_bugs: list[KnowledgeMatch] = []

        if query_text.strip():
            top_k = self._settings.similarity_top_k
            similar_investigations = self._knowledge.search_historical_investigations(query_text, top_k)
            relevant_docs = self._knowledge.search_documentation(query_text, top_k)
            known_bugs = self._knowledge.search_known_bugs(query_text, top_k)

        root_causes = self._build_root_causes(similar_investigations, entities)
        overall_confidence = root_causes[0].confidence if root_causes else 0.0

        return Recommendation(
            investigation_id=investigation.id,
            root_causes=root_causes,
            overall_confidence=overall_confidence,
            similar_investigations=similar_investigations,
            relevant_documentation=relevant_docs,
            known_bugs=known_bugs,
            suggested_logs=self._suggest_logs(investigation, entities),
            suggested_sql=self._suggest_sql(entities),
            next_best_step=self._next_best_step(investigation, similar_investigations, entities),
        )

    # --- Root causes -----------------------------------------------------

    def _build_root_causes(
        self, similar_investigations: list[KnowledgeMatch], entities: list[ExtractedEntity]
    ) -> list[RootCauseHypothesis]:
        hypotheses: list[RootCauseHypothesis] = []

        # Strongest signal: a close historical match supplies its actual
        # (human-confirmed) root cause.
        for match in similar_investigations[:2]:
            if match.score < self._settings.min_similarity_for_root_cause:
                continue
            root_cause = match.metadata.get("root_cause")
            if not root_cause:
                continue
            hypotheses.append(
                RootCauseHypothesis(
                    description=_snippet(root_cause),
                    confidence=match.score,
                    rationale=f"Matches historical investigation '{match.title}' "
                    f"({match.score:.0%} similarity).",
                )
            )

        # Supplementary signal: entity-based heuristics, always included
        # (at lower confidence) so there's something actionable even with
        # no historical precedent.
        entity_by_type = {e.entity_type: e for e in entities}
        for heuristic in _ENTITY_HEURISTICS:
            entity = entity_by_type.get(heuristic.entity_type)
            if entity is None:
                continue
            hypotheses.append(
                RootCauseHypothesis(
                    description=heuristic.root_cause_template.format(value=entity.value),
                    confidence=heuristic.confidence,
                    rationale=f"Investigation evidence contains a {heuristic.entity_type.value} "
                    f"('{entity.value}').",
                )
            )

        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses[:5]

    # --- Next best step ----------------------------------------------------

    def _next_best_step(
        self,
        investigation: InvestigationSession,
        similar_investigations: list[KnowledgeMatch],
        entities: list[ExtractedEntity],
    ) -> str:
        has_logs = any(e.evidence_type.value == "log_file" for e in investigation.evidence)

        if not investigation.evidence:
            return "Paste the task description (or upload logs) to begin analysis."

        if not has_logs:
            return (
                "No logs uploaded yet. Upload application/system logs from around the time of "
                "the issue -- entity correlation (thread IDs, correlation IDs, exceptions) "
                "significantly improves recommendation confidence."
            )

        if similar_investigations and similar_investigations[0].score >= self._settings.min_similarity_for_root_cause:
            top = similar_investigations[0]
            next_step = top.metadata.get("next_step")
            if next_step:
                return _snippet(next_step)
            resolution = top.metadata.get("resolution")
            if resolution:
                return f"Based on similar past investigation '{top.title}', try: {_snippet(resolution)}"

        entity_by_type = {e.entity_type: e for e in entities}
        for heuristic in _ENTITY_HEURISTICS:
            entity = entity_by_type.get(heuristic.entity_type)
            if entity is not None and heuristic.next_step_template:
                return heuristic.next_step_template.format(value=entity.value)

        return (
            "No strong historical match or recognizable entity pattern yet. Add more specific "
            "evidence (exact error messages, IDs, timestamps) or broaden the log upload window."
        )

    # --- Suggestions -----------------------------------------------------

    def _suggest_logs(
        self, investigation: InvestigationSession, entities: list[ExtractedEntity]
    ) -> list[str]:
        suggestions: list[str] = []
        seen_components = {
            e.value
            for e in entities
            if e.entity_type in (EntityType.SERVICE_NAME, EntityType.HOST_NAME, EntityType.POD_NAME)
        }
        for component in sorted(seen_components):
            suggestions.append(f"Logs from '{component}' around the time of the issue")

        if any(e.entity_type == EntityType.CORRELATION_ID for e in entities):
            suggestions.append("Downstream/upstream service logs filtered by the same correlation ID")

        return suggestions[:5]

    def _suggest_sql(self, entities: list[ExtractedEntity]) -> list[str]:
        entity_by_type = {e.entity_type: e for e in entities}
        suggestions: list[str] = []
        for heuristic in _ENTITY_HEURISTICS:
            entity = entity_by_type.get(heuristic.entity_type)
            if entity is not None and heuristic.suggested_sql:
                suggestions.append(heuristic.suggested_sql.format(value=entity.value))
        return suggestions
