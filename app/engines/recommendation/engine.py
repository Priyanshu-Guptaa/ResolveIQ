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
from app.domain.enums import EntityType, EvidenceType
from app.domain.investigation import InvestigationSession
from app.domain.knowledge_relationships import KnowledgeObjectType, RelationshipType
from app.domain.recommendation import (
    InvestigationStage,
    InvestigationStrategy,
    KnowledgeMatch,
    MatchedComponent,
    Recommendation,
    RecommendedLogCollectionItem,
    RecommendedSolution,
    RequiredEvidenceItem,
    RootCauseHypothesis,
    SuggestedSqlItem,
)
from app.engines.external_knowledge.ranking import confidence_for_score
from app.engines.knowledge.engine import KnowledgeEngine
from app.engines.shared.text_matching import keyword_match_score

if TYPE_CHECKING:
    from app.domain.evidence import Evidence
    from app.domain.external_knowledge import ExternalKnowledgeResult
    from app.domain.log_intelligence_kb import LogCollectionScenario, LogSourceApplication
    from app.engines.external_knowledge.service import ExternalKnowledgeService
    from app.engines.knowledge_relationships.engine import KnowledgeRelationshipEngine
    from app.engines.sql_library.engine import SqlLibraryEngine
    from app.infrastructure.db.component_repository import ComponentProfileRepository
    from app.infrastructure.db.log_knowledge_repository import LogKnowledgeRepository

logger = logging.getLogger(__name__)

_PRIORITY_CRITICAL = "Critical"
_PRIORITY_RECOMMENDED = "Recommended"
_PRIORITY_OPTIONAL = "Optional"
_PRIORITY_BROWSED = "Browsed"
"""Not a real priority -- labels items from ``logs_for_technology()``'s
manual technology browse, which was never scored against the
investigation's issue text at all (see that method's docstring)."""

_FULL_MATCH_SCORE = 1.0
_PARTIAL_MATCH_SCORE = 0.5
_MIN_KEYWORD_WORD_LEN = 4
"""Below this, a single word from a technology/scenario-type name (e.g.
"IP") is too generic to treat as a meaningful partial-match signal on
its own."""

_SOLUTION_CONFIDENCE_FLOOR = 0.4
"""A TFS/Wiki match below this (ranking.py's own "Medium" floor)
doesn't clear the bar to drive the synthesized RecommendedSolution --
still shown in its own External Knowledge section, just not treated as
strong enough evidence to state a likely issue/resolution from."""

_MAX_MATCHED_SCENARIOS = 3
_MAX_RECOMMENDED_LOG_ITEMS = 20
"""Caps applied the same way ``top_k``/snippet caps are used elsewhere
in this engine -- a matched scenario can have a long step list; this
keeps the "Recommended Log Collection" section scannable rather than
dumping every matched technology's full log inventory."""

_MIN_COMPONENT_NAME_LEN_FOR_COLLECTED_MATCH = 4
"""A log source name shorter than this (rare, but the wiki has a few,
e.g. generic single-word names) is too likely to false-positive as a
substring of an unrelated uploaded filename to safely mark "already
collected" from a name match alone."""

_GENERIC_FILENAME_STEMS = {"log", "logfile", "logs"}
"""Filenames this generic are reused by dozens of different log
sources in the wiki (e.g. nearly every Windows service logs to
"logfile.log") -- an uploaded file with one of these names does not
reliably identify *which* source it came from, so it is deliberately
excluded from the "already collected" filename check (falling back to
the component-name-in-title check instead). Being wrong here has real
cost: a false "already collected" could make an engineer skip
gathering a log that's actually still missing."""

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


def _keyword_match_score(keyword: str, context: str) -> float:
    """Thin re-export -- the real implementation moved to
    ``app.engines.shared.text_matching.keyword_match_score`` so
    ``app.engines.external_knowledge.ranking`` (TFS/Wiki live search)
    reuses the exact same, already-fixed logic instead of a second
    copy. Kept as a module-level function here (not just an import
    alias) so every existing call site in this file is unchanged."""
    return keyword_match_score(keyword, context, min_word_len=_MIN_KEYWORD_WORD_LEN)


def _match_reason(scenario: "LogCollectionScenario", tech_score: float, type_score: float) -> str:
    """Deterministically explains *why this scenario* was selected --
    distinct from each step's own ``explanation`` (its position in the
    message flow). Built entirely from which score components fired;
    never phrased by an LLM."""
    tech_part = f'technology "{scenario.technology}"' if tech_score > 0 else None
    type_part = f'issue type "{scenario.scenario_type}"' if type_score >= _FULL_MATCH_SCORE else None
    matched = [p for p in (tech_part, type_part) if p]
    if tech_score >= _FULL_MATCH_SCORE and type_score >= _FULL_MATCH_SCORE:
        return f"Matched because the investigation mentions both {matched[0]} and {matched[1]} -- the specific operation this scenario covers."
    if tech_score >= _FULL_MATCH_SCORE:
        return f"Matched because the investigation mentions {matched[0]}; the specific issue type ({scenario.scenario_type}) wasn't identified, so this is one of several possible operations for that technology."
    return f"Weakly matched: the investigation mentions a related keyword for {matched[0] if matched else scenario.technology}, not the full technology name."


def _evidence_titles(investigation: InvestigationSession) -> list[str]:
    """Titles (original filenames, for uploads) of every LOG_FILE
    evidence item already attached to this investigation -- the signal
    "already collected" detection cross-references against."""
    return [e.title for e in investigation.evidence if e.evidence_type == EvidenceType.LOG_FILE and e.title]


def _is_already_collected(component_name: str, source: "LogSourceApplication | None", evidence_titles: list[str]) -> bool:
    """Conservative on purpose (false positives are worse than false
    negatives here -- see _GENERIC_FILENAME_STEMS): a component-name
    substring match against an uploaded filename covers the realistic
    case of a zip upload preserving "ComponentName/logfile.log"-shaped
    internal paths, or a descriptively-renamed file; a filename-pattern
    match is only trusted when that pattern isn't one of the generic
    names reused by dozens of unrelated sources."""
    name = component_name.strip()
    name_matches = len(name) >= _MIN_COMPONENT_NAME_LEN_FOR_COLLECTED_MATCH and any(
        name.lower() in title.lower() for title in evidence_titles
    )
    if name_matches:
        return True
    if source is None:
        return False
    for pattern in source.location.filename_patterns:
        stem = re.sub(r"\.\w+\*?$", "", pattern).lower()
        if stem in _GENERIC_FILENAME_STEMS:
            continue
        if any(pattern.lower() in title.lower() for title in evidence_titles):
            return True
    return False


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
        component_repo: "ComponentProfileRepository | None" = None,
        relationship_engine: "KnowledgeRelationshipEngine | None" = None,
        sql_library: "SqlLibraryEngine | None" = None,
        external_knowledge: "ExternalKnowledgeService | None" = None,
    ) -> None:
        self._knowledge = knowledge_engine
        self._settings = settings
        self._log_knowledge = log_knowledge_repo
        self._components = component_repo
        self._relationships = relationship_engine
        self._sql_library = sql_library
        self._external_knowledge = external_knowledge
        """None in older call sites/tests -- tfs_matches/wiki_matches
        stay None on the resulting Strategy in that case (see
        InvestigationStrategy's docstring for that field). A configured
        ExternalKnowledgeService always populates both fields, even
        when the underlying connector is unreachable (available=False),
        since External Knowledge itself being present is a build-time
        fact, not a per-request one."""

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
        recommended_logs = self._recommend_logs(investigation)
        next_action, next_action_rationale = self._next_action(investigation, similar_investigations, entities)

        strategy = self._build_strategy(
            investigation=investigation,
            entities=entities,
            root_causes=root_causes,
            similar_investigations=similar_investigations,
            known_bugs=known_bugs,
            relevant_docs=relevant_docs,
            recommended_logs=recommended_logs,
            next_action=next_action,
            next_action_rationale=next_action_rationale,
        )

        return Recommendation(
            investigation_id=investigation.id,
            root_causes=root_causes,
            overall_confidence=overall_confidence,
            similar_investigations=similar_investigations,
            relevant_documentation=relevant_docs,
            known_bugs=known_bugs,
            suggested_logs=self._suggest_logs(investigation, entities),
            recommended_logs=recommended_logs,
            suggested_sql=self._suggest_sql(entities),
            next_best_step=next_action,
            strategy=strategy,
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

    def _next_action(
        self,
        investigation: InvestigationSession,
        similar_investigations: list[KnowledgeMatch],
        entities: list[ExtractedEntity],
    ) -> tuple[str, str]:
        """Returns (action, rationale). ``action`` is unchanged from the
        pre-Strategy behavior and is what populates the legacy
        ``next_best_step`` field verbatim; ``rationale`` is new --
        surfaced only in ``InvestigationStrategy.next_action_rationale``
        -- a one-line "why this action" explanation for whichever branch
        produced it, built from the same signal, never re-derived
        independently (that would risk the two drifting apart)."""
        has_logs = any(e.evidence_type.value == "log_file" for e in investigation.evidence)

        if not investigation.evidence:
            return (
                "Paste the task description (or upload logs) to begin analysis.",
                "No evidence has been added to this investigation yet.",
            )

        if not has_logs:
            return (
                "No logs uploaded yet. Upload application/system logs from around the time of "
                "the issue -- entity correlation (thread IDs, correlation IDs, exceptions) "
                "significantly improves recommendation confidence.",
                "A task description is present, but no log file evidence has been uploaded yet.",
            )

        if similar_investigations and similar_investigations[0].score >= self._settings.min_similarity_for_root_cause:
            top = similar_investigations[0]
            rationale = f"Based on historical investigation '{top.title}' ({top.score:.0%} similarity)."
            next_step = top.metadata.get("next_step")
            if next_step:
                return _snippet(next_step), rationale
            resolution = top.metadata.get("resolution")
            if resolution:
                return f"Based on similar past investigation '{top.title}', try: {_snippet(resolution)}", rationale

        entity_by_type = {e.entity_type: e for e in entities}
        for heuristic in _ENTITY_HEURISTICS:
            entity = entity_by_type.get(heuristic.entity_type)
            if entity is not None and heuristic.next_step_template:
                return (
                    heuristic.next_step_template.format(value=entity.value),
                    f"Investigation evidence contains a {heuristic.entity_type.value} ('{entity.value}').",
                )

        return (
            "No strong historical match or recognizable entity pattern yet. Add more specific "
            "evidence (exact error messages, IDs, timestamps) or broaden the log upload window.",
            "Neither a confident historical match nor a recognizable entity pattern has been found yet.",
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
        scenario's ``technology`` *and* ``scenario_type`` -- no
        embeddings, no LLM reasoning, consistent with every other rule
        in this engine.

        Issue-aware priority (polishing-phase addition): a technology
        match alone used to be enough to earn "Critical", which meant
        every scenario for the matched technology (Command Request,
        Command Response, Firmware Download, ...) looked equally
        urgent regardless of what the investigation is actually about.
        Now Critical requires the scenario's *issue type* to match too
        -- the log collection for the specific operation described in
        the investigation, not just the general technology area.
        """
        if self._log_knowledge is None:
            return []
        context = investigation.context_text
        if not context.strip():
            return []

        evidence_titles = _evidence_titles(investigation)

        scored: list[tuple[float, float, float, "LogCollectionScenario"]] = []
        for scenario in self._log_knowledge.list_scenarios():
            tech_score = _keyword_match_score(scenario.technology, context)
            if tech_score <= 0:
                continue
            type_score = _keyword_match_score(scenario.scenario_type, context)
            scored.append((tech_score + type_score, tech_score, type_score, scenario))
        scored.sort(key=lambda row: row[0], reverse=True)

        items: list[RecommendedLogCollectionItem] = []
        for _combined, tech_score, type_score, scenario in scored[:_MAX_MATCHED_SCENARIOS]:
            if tech_score >= _FULL_MATCH_SCORE and type_score >= _FULL_MATCH_SCORE:
                label = _PRIORITY_CRITICAL
            elif tech_score >= _FULL_MATCH_SCORE:
                label = _PRIORITY_RECOMMENDED
            else:
                label = _PRIORITY_OPTIONAL
            reason = _match_reason(scenario, tech_score, type_score)
            items.extend(self._scenario_items(scenario, label, reason, evidence_titles))
            if len(items) >= _MAX_RECOMMENDED_LOG_ITEMS:
                break
        return items[:_MAX_RECOMMENDED_LOG_ITEMS]

    def logs_for_technology(
        self, investigation: InvestigationSession, technology: str
    ) -> list[RecommendedLogCollectionItem]:
        """Every documented scenario for exactly one technology,
        regardless of what the investigation's own text says -- the
        manual counterpart to ``_recommend_logs()``'s automatic
        technology+issue-type matching, for the real case that matching
        can't cover: the Log Intelligence Knowledge Base has no
        scenario_type at all for the actual operation under
        investigation (e.g. a data-extract job, not a meter command
        flow), so automatic matching has nothing relevant to surface no
        matter how the issue is worded. Same conservative
        already-collected detection, same per-step data -- just no
        issue-type filter and no priority ranking, since nothing here
        was actually matched to anything."""
        if self._log_knowledge is None:
            return []
        technology = technology.strip()
        if not technology:
            return []

        evidence_titles = _evidence_titles(investigation)
        reason = (
            f'Technology filter: every documented scenario for "{technology}" -- not matched to this '
            "investigation's issue type, since it was explicitly browsed rather than auto-matched."
        )
        items: list[RecommendedLogCollectionItem] = []
        for scenario in self._log_knowledge.list_scenarios():
            if scenario.technology.strip().lower() != technology.lower():
                continue
            items.extend(self._scenario_items(scenario, _PRIORITY_BROWSED, reason, evidence_titles))
            if len(items) >= _MAX_RECOMMENDED_LOG_ITEMS:
                break
        return items[:_MAX_RECOMMENDED_LOG_ITEMS]

    def available_log_technologies(self) -> list[str]:
        """Distinct technology names actually present in the Log
        Intelligence Knowledge Base -- real imported data, not a
        hardcoded list of the technologies this product happens to
        support today."""
        if self._log_knowledge is None:
            return []
        return sorted({s.technology for s in self._log_knowledge.list_scenarios() if s.technology.strip()})

    def _scenario_items(
        self,
        scenario: "LogCollectionScenario",
        label: str,
        match_reason: str,
        evidence_titles: list[str],
    ) -> list[RecommendedLogCollectionItem]:
        items: list[RecommendedLogCollectionItem] = []
        for step in sorted(scenario.steps, key=lambda s: s.priority):
            source = self._log_knowledge.get_log_source(step.log_source_id) if self._log_knowledge else None
            linked_id, linked_name = self._linked_component(step.log_source_id)
            items.append(
                RecommendedLogCollectionItem(
                    priority_label=label,
                    match_reason=match_reason,
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
                    already_collected=_is_already_collected(step.component_name, source, evidence_titles),
                    linked_component_id=linked_id,
                    linked_component_name=linked_name,
                )
            )
        return items

    def _linked_component(self, log_source_id: str) -> tuple[str | None, str | None]:
        """Product Intelligence integration: surfaces the real
        IMPLEMENTS_LOGGING_FOR relationship created at import time
        (app/engines/log_knowledge/importer.py), when one exists --
        never a new/independent match, just exposing what the
        relationship graph already knows."""
        if self._relationships is None:
            return None, None
        for rel in self._relationships.list_relationships(KnowledgeObjectType.LOG_SOURCE_APPLICATION, log_source_id):
            if rel.relationship.relationship_type != RelationshipType.IMPLEMENTS_LOGGING_FOR:
                continue
            other = rel.to_object if rel.relationship.from_type == KnowledgeObjectType.LOG_SOURCE_APPLICATION else rel.from_object
            if other.type == KnowledgeObjectType.COMPONENT:
                return other.id, other.title
        return None, None

    def _suggest_sql(self, entities: list[ExtractedEntity]) -> list[str]:
        entity_by_type = {e.entity_type: e for e in entities}
        suggestions: list[str] = []
        for heuristic in _ENTITY_HEURISTICS:
            entity = entity_by_type.get(heuristic.entity_type)
            if entity is not None and heuristic.suggested_sql:
                suggestions.append(heuristic.suggested_sql.format(value=entity.value))
        return suggestions

    # --- Investigation Strategy (Recommendation Engine V2) -----------------
    #
    # Everything below orchestrates the fields already computed above
    # (root_causes, similar_investigations, known_bugs, relevant_docs,
    # recommended_logs, next_action) into one explainable, ordered plan.
    # The only genuinely new matching logic is component matching
    # (_match_component) -- it reuses the exact same deterministic
    # keyword-matching helper Log Intelligence already uses. Nothing
    # here calls an LLM or introduces embeddings beyond what
    # KnowledgeEngine already provides for the three semantic-search
    # fields.

    def _build_strategy(
        self,
        *,
        investigation: InvestigationSession,
        entities: list[ExtractedEntity],
        root_causes: list[RootCauseHypothesis],
        similar_investigations: list[KnowledgeMatch],
        known_bugs: list[KnowledgeMatch],
        relevant_docs: list[KnowledgeMatch],
        recommended_logs: list[RecommendedLogCollectionItem],
        next_action: str,
        next_action_rationale: str,
    ) -> InvestigationStrategy:
        matched_component = self._match_component(investigation, entities)
        required_evidence = self._required_evidence(investigation, recommended_logs, entities)
        missing_evidence = [item for item in required_evidence if not item.satisfied]
        stage, stage_rationale = self._current_stage(investigation, missing_evidence, root_causes)
        progress, progress_summary = self._progress(stage, required_evidence, missing_evidence)

        tfs_matches = wiki_matches = recommended_solution = None
        if self._external_knowledge is not None:
            tfs_matches, wiki_matches = self._external_knowledge.gather(
                investigation, entities, matched_component, self._infer_technology(investigation)
            )
            recommended_solution = self._synthesize_recommendation(
                root_causes, similar_investigations, tfs_matches, wiki_matches, missing_evidence
            )

        return InvestigationStrategy(
            current_stage=stage,
            stage_rationale=stage_rationale,
            progress=progress,
            progress_summary=progress_summary,
            recommended_next_action=next_action,
            next_action_rationale=next_action_rationale,
            required_evidence=required_evidence,
            missing_evidence=missing_evidence,
            ordered_log_collection=recommended_logs,
            suggested_sql=self._suggested_sql_items(matched_component, entities),
            matched_component=matched_component,
            historical_investigations=similar_investigations,
            known_bugs=known_bugs,
            documentation=relevant_docs,
            tfs_matches=tfs_matches,
            wiki_matches=wiki_matches,
            recommended_solution=recommended_solution,
            decision_checkpoint=self._decision_checkpoint(root_causes),
        )

    def _infer_technology(self, investigation: InvestigationSession) -> str | None:
        """Real bug fix: the structured Summary Card ``technology``
        field is very often left blank even when the pasted task
        description names a real technology in free text (e.g. a real
        ServiceNow export literally containing "Technology: rf mesh")
        -- and External Knowledge's search-term builder previously used
        *only* that structured field, so a blank field meant TFS/Wiki
        got no technology anchor at all even though a real one was
        sitting in the evidence the whole time.

        Deliberately independent of ``_recommend_logs()``'s own
        technology+issue-type scenario matching (frozen -- "bug fixes
        only" per explicit instruction) rather than reusing its top
        result: that method breaks combined-score ties by whichever
        scenario happens to come first in Log Intelligence Knowledge
        Base iteration order, which is a real, separate quality issue
        of its own and not something this fix should silently inherit
        or paper over. This applies the exact same deterministic
        keyword_match_score used everywhere else in this file directly
        against the real, distinct technology names in the Knowledge
        Base, picking whichever single technology name actually scores
        highest against the investigation's own text -- simpler and
        answers a narrower question ("which technology name is
        actually mentioned") than the scenario matcher needs to."""
        if investigation.technology:
            return investigation.technology
        if self._log_knowledge is None:
            return None
        context = investigation.context_text
        best_technology: str | None = None
        best_score = 0.0
        for technology in self.available_log_technologies():
            score = keyword_match_score(technology, context)
            if score > best_score:
                best_score = score
                best_technology = technology
        return best_technology

    def _synthesize_recommendation(
        self,
        root_causes: list[RootCauseHypothesis],
        similar_investigations: list[KnowledgeMatch],
        tfs_result: "ExternalKnowledgeResult | None",
        wiki_result: "ExternalKnowledgeResult | None",
        missing_evidence: list[RequiredEvidenceItem],
    ) -> RecommendedSolution:
        """Correlates local KB + live TFS + live Wiki into one answer.
        Deterministic throughout: picks whichever real source(s) clear
        a confidence bar, quotes/references their actual content, and
        is honest (``insufficient_evidence=True``, no resolution text)
        when none do -- never invents a fix. See RecommendedSolution's
        own docstring for the full source-attribution contract."""
        what_to_check = [item.description for item in missing_evidence[:5]] or [
            "Confirm the affected component/technology and gather logs from around the time of the issue."
        ]

        best_local = (
            similar_investigations[0]
            if similar_investigations and similar_investigations[0].score >= self._settings.min_similarity_for_root_cause
            else None
        )
        best_tfs = (
            tfs_result.matches[0]
            if tfs_result and tfs_result.available and tfs_result.matches and tfs_result.matches[0].score >= _SOLUTION_CONFIDENCE_FLOOR
            else None
        )
        best_wiki = (
            wiki_result.matches[0]
            if wiki_result and wiki_result.available and wiki_result.matches and wiki_result.matches[0].score >= _SOLUTION_CONFIDENCE_FLOOR
            else None
        )

        if not (best_local or best_tfs or best_wiki):
            return RecommendedSolution(
                likely_issue="Not enough evidence yet to identify a likely root cause.",
                rationale="No local historical match, TFS case, or Wiki page currently meets the confidence "
                "bar -- more evidence is needed before recommending a specific fix.",
                what_to_check=what_to_check,
                recommended_resolution=None,
                insufficient_evidence=True,
                confidence="Insufficient",
            )

        # Real cross-source correlation, not a guess: a TFS Bug's own
        # LandisGyr.CRMID field carries the exact ServiceNow ticket
        # number(s) that reported it -- the same convention Task Import
        # already tags Historical Investigations with (tags:
        # ["ticket:<number>"]). When they match, this isn't "two
        # similar cases" -- it's the same real-world incident seen from
        # two systems.
        correlated_local: KnowledgeMatch | None = None
        if best_tfs is not None and best_tfs.tfs_case.crm_id:
            crm_tokens = [t.strip().upper() for t in best_tfs.tfs_case.crm_id.split("/") if t.strip()]
            for match in similar_investigations:
                tags = str(match.metadata.get("tags", "")).upper()
                if any(token and token in tags for token in crm_tokens):
                    correlated_local = match
                    break

        rationale_parts: list[str] = []
        likely_issue: str | None = None
        recommended_resolution: str | None = None
        confidence_score = 0.0

        # Prefer the already-computed root-cause hierarchy over deriving
        # "likely issue" from best_local directly: _build_root_causes()
        # already (a) skips historical matches with no recorded root
        # cause instead of falling back to their raw ticket title (which
        # is what caused the reported bug -- an unrelated customer's
        # ticket title being shown as if it diagnosed this case), and
        # (b) blends in entity-derived hypotheses that can outrank a
        # title-only match. Only fall back to best_local's title, with
        # an explicit "no confirmed root cause" caveat, when there is no
        # root-cause hypothesis at all.
        #
        # ``correlated_match`` tracks *which* similar_investigations
        # entry (if any) the surfaced likely_issue actually came from --
        # its ``resolution`` is only trustworthy to quote when it's the
        # same record, never a different top-similarity match that
        # happens to be unrelated (e.g. likely_issue driven by an entity
        # heuristic while best_local is about a completely different
        # defect) -- that mismatch is exactly what produced a resolution
        # about an unrelated "Grid location ID in a CSV extract" case
        # for a totally different RF Mesh command error in live testing.
        # ``local_contributed`` separately tracks whether local KB
        # informed ``likely_issue`` at all (root-cause-derived or
        # title-only fallback) -- used for the "Sources: Local KB" tag,
        # which should show whenever local KB genuinely shaped the
        # answer, even in the low-confidence fallback case where its
        # *resolution* still isn't trustworthy enough to quote.
        correlated_match: KnowledgeMatch | None = None
        local_contributed = False
        if root_causes:
            top_cause = root_causes[0]
            likely_issue = top_cause.description
            rationale_parts.append(
                top_cause.rationale.rstrip(".") if top_cause.rationale else "a matched root-cause hypothesis"
            )
            confidence_score = max(confidence_score, top_cause.confidence)
            if top_cause.rationale.startswith("Matches historical investigation"):
                local_contributed = True
                for match in similar_investigations[:2]:
                    root_cause_text = match.metadata.get("root_cause")
                    if root_cause_text and _snippet(root_cause_text) == top_cause.description:
                        correlated_match = match
                        break
        elif best_local is not None:
            # Deliberately does NOT set correlated_match here: we've just
            # said we don't trust this match's root cause enough to state
            # it as the likely issue (it has none on file, only a title/
            # customer/component overlap) -- so its resolution text isn't
            # trustworthy either, and must not be surfaced as if it
            # addresses this specific problem. Found live: a real CLECO
            # case where the top local match (86% similarity, same
            # customer/meter type, no root cause) was a *different*
            # defect (ST-03 advisory events) from the current one (meter
            # program change), yet its Closure Summary was still being
            # shown as "the" recommended resolution.
            likely_issue = f"Possibly related to a similar past case: \"{best_local.title}\" (no confirmed root cause on file)"
            rationale_parts.append(
                f"a similar local historical investigation ({best_local.score:.0%} similarity) with no recorded root cause"
            )
            confidence_score = max(confidence_score, best_local.score * 0.5)
            local_contributed = True

        if correlated_match is not None:
            local_resolution = correlated_match.metadata.get("resolution")
            if local_resolution:
                recommended_resolution = f"Based on a similar past case: {_snippet(local_resolution)}"

        if best_tfs is not None:
            case = best_tfs.tfs_case
            if likely_issue is None:
                likely_issue = case.title
            correlation_note = " -- the same real-world case as the local historical match above (matching CRM/ticket reference)" if correlated_local else ""
            rationale_parts.append(f"TFS-{case.tfs_id} ({case.state}, {best_tfs.confidence.lower()} confidence){correlation_note}")
            confidence_score = max(confidence_score, best_tfs.score)
            if case.resolution_text:
                recommended_resolution = f"Based on TFS-{case.tfs_id} ({case.state}): {case.resolution_text}"

        if best_wiki is not None:
            page = best_wiki.wiki_page
            rationale_parts.append(f'the Wiki page "{page.title}"')
            confidence_score = max(confidence_score, best_wiki.score)
            if recommended_resolution is None and page.excerpt:
                recommended_resolution = f'Per Wiki page "{page.title}": {page.excerpt}'

        if likely_issue is None:
            likely_issue = "Multiple weak signals found -- no single source clearly explains the issue yet."

        return RecommendedSolution(
            likely_issue=likely_issue,
            rationale="Based on " + " and ".join(rationale_parts) + ".",
            what_to_check=what_to_check,
            recommended_resolution=recommended_resolution,
            insufficient_evidence=recommended_resolution is None,
            confidence=confidence_for_score(confidence_score) if recommended_resolution else "Low",
            source_local=local_contributed,
            source_tfs=best_tfs is not None,
            source_wiki=best_wiki is not None,
            supporting_tfs_id=best_tfs.tfs_case.tfs_id if best_tfs else None,
            supporting_tfs_url=best_tfs.tfs_case.url if best_tfs else None,
            supporting_wiki_title=best_wiki.wiki_page.title if best_wiki else None,
            supporting_wiki_url=best_wiki.wiki_page.url if best_wiki else None,
        )

    def _match_component(
        self, investigation: InvestigationSession, entities: list[ExtractedEntity]
    ) -> MatchedComponent | None:
        """Deterministic name/keyword matching against the *complete*
        live Component Registry -- the same word-boundary matching
        Log Intelligence already uses for technology/scenario_type, now
        applied to component names. An extracted SERVICE_NAME/HOST_NAME
        entity that exactly equals a component's name is the strongest
        possible signal (1.0, real extracted data, not a guess); a
        keyword match against the investigation's own text is the
        fallback. Ties keep the first-encountered (registry order)
        component, consistent with every other tie-break in this file."""
        if self._components is None:
            return None
        components = self._components.list_all()
        if not components:
            return None
        context = investigation.context_text
        entity_values = {
            e.value.strip().lower()
            for e in entities
            if e.entity_type in (EntityType.SERVICE_NAME, EntityType.HOST_NAME)
        }

        best: MatchedComponent | None = None
        for component in components:
            name = component.name.strip()
            if not name:
                continue
            if name.lower() in entity_values:
                score = _FULL_MATCH_SCORE
                reason = (
                    f'Matched because an extracted service/host name in the investigation\'s evidence '
                    f'exactly equals the component "{component.name}".'
                )
            else:
                score = _keyword_match_score(name, context)
                if score >= _FULL_MATCH_SCORE:
                    reason = f'Matched because the investigation text mentions "{component.name}" by name.'
                elif score > 0:
                    reason = f'Weakly matched: the investigation text mentions a keyword related to "{component.name}".'
                else:
                    continue
            if best is None or score > best.confidence:
                best = MatchedComponent(
                    component_id=component.id, component_name=component.name, confidence=score, match_reason=reason
                )
        return best

    def _required_evidence(
        self,
        investigation: InvestigationSession,
        recommended_logs: list[RecommendedLogCollectionItem],
        entities: list[ExtractedEntity],
    ) -> list[RequiredEvidenceItem]:
        """Deduplicated at the component level (not per-file) from the
        ordered log collection -- a checklist view. Full detail
        (repository path, filenames, explanation) stays exclusively in
        ``ordered_log_collection``; this only points back to it via
        ``related_log_source_id`` so nothing is copied twice."""
        items: list[RequiredEvidenceItem] = []
        seen_components: set[str] = set()
        for log_item in recommended_logs:
            if log_item.component_name in seen_components:
                continue
            seen_components.add(log_item.component_name)
            items.append(
                RequiredEvidenceItem(
                    description=f"{log_item.component_name} logs ({log_item.scenario_technology} / {log_item.scenario_type})",
                    satisfied=log_item.already_collected,
                    source="log_intelligence",
                    related_log_source_id=log_item.log_source_id,
                )
            )

        has_logs = any(e.evidence_type == EvidenceType.LOG_FILE for e in investigation.evidence)
        for suggestion in self._suggest_logs(investigation, entities):
            items.append(RequiredEvidenceItem(description=suggestion, satisfied=has_logs, source="entity_heuristic"))
        return items

    def _current_stage(
        self,
        investigation: InvestigationSession,
        missing_evidence: list[RequiredEvidenceItem],
        root_causes: list[RootCauseHypothesis],
    ) -> tuple[InvestigationStage, str]:
        if not investigation.evidence:
            return InvestigationStage.TRIAGE, "No evidence has been added to this investigation yet."
        if missing_evidence:
            return (
                InvestigationStage.EVIDENCE_COLLECTION,
                f"{len(missing_evidence)} required evidence item(s) still need to be collected.",
            )
        if root_causes and root_causes[0].confidence >= self._settings.min_similarity_for_root_cause:
            return (
                InvestigationStage.ROOT_CAUSE_IDENTIFIED,
                f"A candidate root cause matched at {root_causes[0].confidence:.0%} confidence.",
            )
        return (
            InvestigationStage.ANALYSIS,
            "Required evidence is in hand; no root cause has matched with enough confidence yet.",
        )

    def _progress(
        self,
        stage: InvestigationStage,
        required_evidence: list[RequiredEvidenceItem],
        missing_evidence: list[RequiredEvidenceItem],
    ) -> tuple[float, str]:
        if stage == InvestigationStage.TRIAGE:
            return 0.0, "Investigation just started -- no evidence yet."
        if stage == InvestigationStage.ROOT_CAUSE_IDENTIFIED:
            return 1.0, "A root cause has been matched -- confirm it to close out the investigation."
        if stage == InvestigationStage.EVIDENCE_COLLECTION:
            collected = len(required_evidence) - len(missing_evidence)
            ratio = collected / len(required_evidence) if required_evidence else 0.0
            # Evidence collection is the first half of the journey to a
            # root cause; reaching ANALYSIS (all required evidence in
            # hand) is itself worth crossing the halfway point below.
            return ratio * 0.5, f"{collected} of {len(required_evidence)} required evidence item(s) collected."
        return 0.5, "All currently-known required evidence has been collected; analyzing for a root-cause match."

    def _suggested_sql_items(
        self, matched_component: MatchedComponent | None, entities: list[ExtractedEntity]
    ) -> list[SuggestedSqlItem]:
        """Real, governed SQL Library ``QueryTemplate`` records take
        priority -- matched via ``related_components``, a field that
        already exists and is already populated by migration, no new
        matching logic invented. Falls back to the existing
        entity-heuristic snippets only when no component matched or no
        template links to it, so a suggestion is still available for
        the (also pre-existing) fallback path."""
        items: list[SuggestedSqlItem] = []
        if matched_component is not None and self._sql_library is not None:
            for template in self._sql_library.list_templates():
                if matched_component.component_name in template.related_components:
                    items.append(
                        SuggestedSqlItem(
                            title=template.title,
                            sql_text=template.sql_text,
                            explanation=template.explanation,
                            source="sql_library",
                            template_id=template.id,
                        )
                    )
        if items:
            return items
        return [
            SuggestedSqlItem(title="Suggested query", sql_text=sql_text, source="entity_heuristic")
            for sql_text in self._suggest_sql(entities)
        ]

    @staticmethod
    def _decision_checkpoint(root_causes: list[RootCauseHypothesis]) -> str | None:
        """The specific thing to verify next -- built entirely from how
        many root-cause candidates exist; never a generic prompt."""
        if len(root_causes) >= 2:
            a, b = root_causes[0], root_causes[1]
            return (
                f'Two plausible causes match this investigation: "{a.description[:80]}" ({a.confidence:.0%}) '
                f'vs "{b.description[:80]}" ({b.confidence:.0%}). Use the recommended logs/SQL above to '
                f"determine which one applies."
            )
        if len(root_causes) == 1:
            return (
                f'Confirm "{root_causes[0].description[:80]}" using the recommended logs/SQL above before '
                f"considering this investigation resolved."
            )
        return None
