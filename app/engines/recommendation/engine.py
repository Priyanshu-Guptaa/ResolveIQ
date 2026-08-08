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
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.config import Settings
from app.domain.entities import ExtractedEntity
from app.domain.enums import EntityType
from app.domain.investigation import InvestigationSession
from app.domain.recommendation import KnowledgeMatch, Recommendation, RecommendedLogCollectionItem, RootCauseHypothesis
from app.engines.knowledge.engine import KnowledgeEngine

if TYPE_CHECKING:
    from app.domain.log_intelligence_kb import LogCollectionScenario
    from app.infrastructure.db.log_knowledge_repository import LogKnowledgeRepository

logger = logging.getLogger(__name__)

_PRIORITY_CRITICAL = "Critical"
_PRIORITY_RECOMMENDED = "Recommended"
_PRIORITY_OPTIONAL = "Optional"

_FULL_MATCH_SCORE = 1.0
_PARTIAL_MATCH_SCORE = 0.5
_MIN_TECHNOLOGY_WORD_LEN = 4
"""Below this, a single word from a technology name (e.g. "IP") is too
generic to treat as a meaningful partial-match signal on its own."""

_MAX_MATCHED_SCENARIOS = 3
_MAX_RECOMMENDED_LOG_ITEMS = 20
"""Caps applied the same way ``top_k``/snippet caps are used elsewhere
in this engine -- a matched scenario can have a long step list; this
keeps the "Recommended Log Collection" section scannable rather than
dumping every matched technology's full log inventory."""

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
    def __init__(
        self,
        knowledge_engine: KnowledgeEngine,
        settings: Settings,
        log_knowledge_repo: "LogKnowledgeRepository | None" = None,
    ) -> None:
        self._knowledge = knowledge_engine
        self._settings = settings
        self._log_knowledge = log_knowledge_repo

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
            recommended_logs=self._recommend_logs(investigation),
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

    # --- Log Intelligence: recommended log collection ---------------------

    def _recommend_logs(self, investigation: InvestigationSession) -> list[RecommendedLogCollectionItem]:
        """Consumes ``LogCollectionScenario`` records immediately (per
        explicit design refinement -- this does not wait for a future
        Recommendation Engine V2). Matching is deterministic keyword
        matching between the investigation's own text and each
        scenario's ``technology`` -- no embeddings, no LLM reasoning,
        consistent with every other rule in this engine."""
        if self._log_knowledge is None:
            return []
        context = investigation.context_text
        if not context.strip():
            return []

        scored = [
            (self._technology_match_score(scenario.technology, context), scenario)
            for scenario in self._log_knowledge.list_scenarios()
        ]
        matched = sorted((pair for pair in scored if pair[0] > 0), key=lambda pair: pair[0], reverse=True)

        items: list[RecommendedLogCollectionItem] = []
        best_full_match_claimed = False
        for score, scenario in matched[:_MAX_MATCHED_SCENARIOS]:
            if score >= _FULL_MATCH_SCORE and not best_full_match_claimed:
                label = _PRIORITY_CRITICAL
                best_full_match_claimed = True
            elif score >= _FULL_MATCH_SCORE:
                label = _PRIORITY_RECOMMENDED
            else:
                label = _PRIORITY_OPTIONAL
            items.extend(self._scenario_items(scenario, label))
            if len(items) >= _MAX_RECOMMENDED_LOG_ITEMS:
                break
        return items[:_MAX_RECOMMENDED_LOG_ITEMS]

    def _scenario_items(self, scenario: "LogCollectionScenario", label: str) -> list[RecommendedLogCollectionItem]:
        items: list[RecommendedLogCollectionItem] = []
        for step in sorted(scenario.steps, key=lambda s: s.priority):
            source = self._log_knowledge.get_log_source(step.log_source_id) if self._log_knowledge else None
            items.append(
                RecommendedLogCollectionItem(
                    priority_label=label,
                    order=step.priority,
                    component_name=step.component_name,
                    scenario_technology=scenario.technology,
                    scenario_type=scenario.scenario_type,
                    scenario_region=scenario.region,
                    explanation=step.explanation,
                    repository_platform=source.location.platform if source else "unknown",
                    repository_root_path=source.location.root_path if source else "",
                    repository_subdirectory=source.location.subdirectory if source else None,
                    filename_patterns=list(source.location.filename_patterns) if source else [],
                    log_source_id=step.log_source_id,
                    scenario_id=scenario.id,
                )
            )
        return items

    @staticmethod
    def _technology_match_score(technology: str, context: str) -> float:
        """1.0 for the full technology name appearing verbatim
        (word-boundary-safe) in the investigation text; 0.5 if only a
        significant individual word from a multi-word technology name
        appears (e.g. investigation mentions "Mesh" but not the full
        "RF Mesh" phrase); 0.0 otherwise."""
        tech = technology.strip()
        if not tech:
            return 0.0
        if re.search(rf"\b{re.escape(tech)}\b", context, re.IGNORECASE):
            return _FULL_MATCH_SCORE
        words = [w for w in re.findall(r"[A-Za-z]+", tech) if len(w) >= _MIN_TECHNOLOGY_WORD_LEN]
        if any(re.search(rf"\b{re.escape(w)}\b", context, re.IGNORECASE) for w in words):
            return _PARTIAL_MATCH_SCORE
        return 0.0

    def _suggest_sql(self, entities: list[ExtractedEntity]) -> list[str]:
        entity_by_type = {e.entity_type: e for e in entities}
        suggestions: list[str] = []
        for heuristic in _ENTITY_HEURISTICS:
            entity = entity_by_type.get(heuristic.entity_type)
            if entity is not None and heuristic.suggested_sql:
                suggestions.append(heuristic.suggested_sql.format(value=entity.value))
        return suggestions
