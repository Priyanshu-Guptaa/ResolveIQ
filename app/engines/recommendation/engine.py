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
from app.domain.provenance import EvidenceKind, EvidenceReference, ProvenanceRecord, ResolutionProvenance
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
from app.engines.knowledge.applicability import ApplicabilityRanker, RetrievalContext
from app.engines.knowledge.engine import KnowledgeEngine
from app.engines.shared.hierarchy import most_specific
from app.engines.shared.text_cleaning import strip_low_signal_boilerplate
from app.engines.shared.text_matching import keyword_match_score

if TYPE_CHECKING:
    from app.domain.evidence import Evidence, HistoricalInvestigationRecord, KnownBugRecord
    from app.domain.external_knowledge import ExternalKnowledgeResult, ExternalMatch
    from app.domain.log_intelligence_kb import LogCollectionScenario, LogSourceApplication
    from app.engines.external_knowledge.service import ExternalKnowledgeService
    from app.engines.knowledge_relationships.engine import KnowledgeRelationshipEngine
    from app.engines.sql_library.engine import SqlLibraryEngine
    from app.infrastructure.db.component_repository import ComponentProfileRepository
    from app.infrastructure.db.log_knowledge_repository import LogKnowledgeRepository
    from app.infrastructure.db.lookup_repository import LookupRepository

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

_ENGLISH_STOPWORDS = frozenset({
    "after", "again", "against", "because", "before", "being", "between", "does", "doing",
    "during", "each", "from", "further", "having", "here", "into", "more", "most", "once",
    "only", "other", "over", "same", "since", "some", "such", "than", "that", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "under", "until", "upon",
    "very", "were", "when", "where", "which", "while", "will", "with", "within", "without",
    "would", "your",
})
"""Standard, closed-class English function words at/above
``_MIN_KEYWORD_WORD_LEN`` -- excluded from
``_known_bug_has_topical_overlap``'s significant-word set (that method's
own docstring has the full rationale). Not a domain-specific blacklist
(the kind this codebase has explicitly rejected before, e.g.
``_match_component``'s fix) -- this is the standard, generic technique
for excluding words that carry no topical signal in *any* domain, the
same category of fix as ignoring "the"/"a" in a search index. Found
necessary live: the bare word "after" -- present in both "GPA...
commands failed to respond after patching" and "...command queue
stalls after mesh router firmware v3.4.0..." -- was on its own enough
to produce a spurious topical-overlap match between two genuinely
unrelated real investigations, exactly the class of false positive
this whole check exists to prevent."""

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

_APPLICABILITY_OVERFETCH_MULTIPLIER = 4
"""How much wider than ``similarity_top_k`` the local semantic search
over-fetches before ``ApplicabilityRanker`` re-ranks and truncates back
down -- see app/engines/knowledge/applicability.py's module docstring
(Part D's retrieval flow: widen the candidate pool, THEN apply
applicability, THEN truncate -- never re-rank only the already-
truncated top_k, which would have nothing left to promote a same-
customer/same-technology match that vector similarity alone ranked
6th)."""

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


def _match_reason(
    scenario: "LogCollectionScenario", tech_score: float, type_score: float, *, demoted_by: str | None = None
) -> str:
    """Deterministically explains *why this scenario* was selected --
    distinct from each step's own ``explanation`` (its position in the
    message flow). Built entirely from which score components fired;
    never phrased by an LLM.

    ``demoted_by`` is set when this scenario's technology was scored
    down from a full match specifically because a more specific
    technology variant (e.g. "RF Mesh IP") also fully matched the same
    text -- the RF-Mesh-vs-Mesh-IP fix (see
    ``RecommendationEngine._specificity_demoted_technologies``). Stated
    honestly rather than folded into the generic "weakly matched"
    wording, since it's a materially different (and more informative)
    reason than an ordinary partial keyword match."""
    if demoted_by:
        return (
            f'The investigation specifically mentions "{demoted_by}", a more specific technology than '
            f'"{scenario.technology}" -- this scenario covers the broader "{scenario.technology}" family, '
            f'not the specific variant named. Check the "{demoted_by}"-labeled scenario(s) first.'
        )
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
class _SynthesizedSources:
    """Real, already-computed internal state from ``_synthesize_recommendation``
    that Resolution Provenance (2026-08-13) needs -- exposed rather than
    re-derived a second time (which would risk drifting from the
    decision logic that actually produced ``RecommendedSolution``).
    Purely informational: nothing here changes what
    ``_synthesize_recommendation`` decides, it only names what it
    already decided. Empty (all None) when nothing cleared the
    confidence bar at all."""

    correlated_match: "KnowledgeMatch | None" = None
    """The local Historical Investigation whose ticket tag matched the
    best TFS case's CRM id -- real cross-source correlation, the only
    similarity-independent signal that can earn CONFIRMED on its own."""
    best_tfs: "ExternalMatch | None" = None
    best_wiki: "ExternalMatch | None" = None
    best_local: "KnowledgeMatch | None" = None
    """The top local match that cleared min_similarity_for_root_cause,
    regardless of whether it had a recorded root cause (i.e. including
    the title-only fallback case) -- distinct from
    ``_local_match_for_cause``'s root-cause-derived match, used only to
    tell POSSIBLE apart from UNKNOWN when no root-cause hypothesis
    exists at all (see ``_resolve_provenance_tier``)."""
    best_known_bug: "KnowledgeMatch | None" = None
    """The top Known Bug match that cleared min_similarity_for_root_cause
    -- added 2026-08-13 (Phase 0, Chat/Structured Resolution Knowledge
    architecture, closing the gap flagged in
    RESOLVEIQ_CHAT_AND_RESOLUTION_ARCHITECTURE.md Section 17/Section 6:
    "Known Bugs are a second-class resolution source"). Same
    "regardless of whether it had real resolution content" contract as
    ``best_local`` -- its ``workaround`` may or may not be populated."""


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
        lookup_repo: "LookupRepository | None" = None,
    ) -> None:
        self._knowledge = knowledge_engine
        self._settings = settings
        self._log_knowledge = log_knowledge_repo
        self._components = component_repo
        self._relationships = relationship_engine
        self._sql_library = sql_library
        self._external_knowledge = external_knowledge
        self._lookup = lookup_repo
        """None in older call sites/tests -- the RF-Mesh-vs-Mesh-IP
        specificity fix (``_specificity_demoted_technologies``) simply
        has no hierarchy to consult in that case and demotes nothing,
        same graceful-degradation contract as ``_external_knowledge``."""
        self._applicability = ApplicabilityRanker(relationship_engine, lookup_repo)
        """Context Dimensions phase (2026-08-12) -- re-ranks local
        semantic search results by real Customer/Region/Technology
        applicability before truncating to top_k. Degrades to a no-op
        (plain truncation) when neither dependency is wired, same
        contract as every other optional dependency in this
        constructor."""
        """None in older call sites/tests -- tfs_matches/wiki_matches
        stay None on the resulting Strategy in that case (see
        InvestigationStrategy's docstring for that field). A configured
        ExternalKnowledgeService always populates both fields, even
        when the underlying connector is unreachable (available=False),
        since External Knowledge itself being present is a build-time
        fact, not a per-request one."""

    def generate(self, investigation: InvestigationSession) -> Recommendation:
        # Cleaned before it becomes a search query (never for anything
        # else -- entity extraction, technology inference, etc. all use
        # investigation.context_text directly): a ServiceNow-templated
        # task description's disclaimer paragraph and empty label:value
        # header lines carry no discriminating signal but can consume a
        # large fraction of the embedding model's small (256-token)
        # window, crowding out the actual defect description. See
        # strip_low_signal_boilerplate's docstring for the real case
        # that surfaced this.
        query_text = strip_low_signal_boilerplate(investigation.context_text)
        entities = investigation.merged_entities
        technology = self._infer_technology(investigation)
        retrieval_context = self._resolve_retrieval_context(investigation, technology)

        similar_investigations: list[KnowledgeMatch] = []
        relevant_docs: list[KnowledgeMatch] = []
        known_bugs: list[KnowledgeMatch] = []

        if query_text.strip():
            top_k = self._settings.similarity_top_k
            overfetch_k = top_k * _APPLICABILITY_OVERFETCH_MULTIPLIER
            # Applicability-aware retrieval (Context Dimensions phase,
            # 2026-08-12): over-fetch beyond top_k, let ApplicabilityRanker
            # adjust for real Customer/Region/Technology applicability,
            # THEN truncate -- see app/engines/knowledge/applicability.py's
            # module docstring for why order matters here.
            similar_investigations = self._applicability.rerank(
                self._knowledge.search_historical_investigations(query_text, overfetch_k), retrieval_context, top_k=top_k
            )
            relevant_docs = self._applicability.rerank(
                self._knowledge.search_documentation(query_text, overfetch_k), retrieval_context, top_k=top_k
            )
            known_bugs = self._applicability.rerank(
                self._knowledge.search_known_bugs(query_text, overfetch_k), retrieval_context, top_k=top_k
            )
            # Resolution Provenance phase (2026-08-13): every match gets a
            # real "why was this recommended" -- applicability's own
            # reasons when they fired, a deterministic baseline
            # otherwise. See _annotate_match_reasons's docstring.
            similar_investigations = self._annotate_match_reasons(similar_investigations)
            relevant_docs = self._annotate_match_reasons(relevant_docs)
            known_bugs = self._annotate_match_reasons(known_bugs)

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
            technology=technology,
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

    def _annotate_match_reasons(self, matches: list[KnowledgeMatch]) -> list[KnowledgeMatch]:
        """Real gap fix (2026-08-13, Resolution Provenance phase,
        approved item 6): "Historical Investigation recommendations
        must have an explicit explanation for why they were selected,
        even when applicability did not contribute." Before this,
        ``KnowledgeMatch.reason`` was only ever set (via
        ``metadata["applicability_reasons"]``) when the Phase 1
        ApplicabilityRanker found a real Customer/Region/Technology
        signal to boost on -- a match with none of those (the common
        case for most of the corpus, which isn't yet classified) had
        no "why" at all beyond a bare score. This never invents a new
        signal: it restates the real score, plus (for historical
        investigations/known bugs specifically) whether the match
        carries a recorded root cause or resolution -- both already
        sitting in ``metadata`` from ``KnowledgeEngine.index_*``."""
        annotated: list[KnowledgeMatch] = []
        for match in matches:
            applicability_reasons = match.metadata.get("applicability_reasons")
            if applicability_reasons:
                reason = "; ".join(applicability_reasons) + f" (also semantically similar, {match.score:.0%})"
            else:
                parts = [f"Semantically similar to the investigation's text ({match.score:.0%} similarity)"]
                if match.metadata.get("root_cause"):
                    parts.append("carries a recorded root cause")
                if match.metadata.get("resolution"):
                    parts.append("carries a recorded resolution")
                reason = "; ".join(parts)
            annotated.append(match.model_copy(update={"reason": reason}))
        return annotated

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

        Specificity fix (Context Dimensions phase, 2026-08-12): a bare
        technology name like "RF Mesh" used to score a full match
        against text that actually said the more specific "RF Mesh
        IP" -- ``keyword_match_score``'s word-boundary check is
        satisfied by the space before "IP", so the shorter name (a
        strict prefix of the longer one) tied with it instead of
        losing to it. ``_specificity_demoted_technologies`` now demotes
        a parent technology's full match to a partial one whenever a
        real, governed child variant (``Technology.parent_technology_id``)
        also fully matches the same text -- so a scenario tagged bare
        "RF Mesh" no longer looks equally "Critical" as one tagged "RF
        Mesh IP" when the investigation specifically says "RF Mesh
        IP". See ``_match_reason``'s ``demoted_by`` parameter for the
        explanation this produces.
        """
        if self._log_knowledge is None:
            return []
        context = investigation.context_text
        if not context.strip():
            return []

        evidence_titles = _evidence_titles(investigation)
        demoted_technologies = self._specificity_demoted_technologies(context)

        scored: list[tuple[float, float, float, "LogCollectionScenario", str | None]] = []
        for scenario in self._log_knowledge.list_scenarios():
            tech_score = _keyword_match_score(scenario.technology, context)
            if tech_score <= 0:
                continue
            demoted_by = demoted_technologies.get(scenario.technology)
            if demoted_by and tech_score >= _FULL_MATCH_SCORE:
                tech_score = _PARTIAL_MATCH_SCORE
            else:
                demoted_by = None
            type_score = _keyword_match_score(scenario.scenario_type, context)
            scored.append((tech_score + type_score, tech_score, type_score, scenario, demoted_by))
        scored.sort(key=lambda row: row[0], reverse=True)

        items: list[RecommendedLogCollectionItem] = []
        for _combined, tech_score, type_score, scenario, demoted_by in scored[:_MAX_MATCHED_SCENARIOS]:
            if tech_score >= _FULL_MATCH_SCORE and type_score >= _FULL_MATCH_SCORE:
                label = _PRIORITY_CRITICAL
            elif tech_score >= _FULL_MATCH_SCORE:
                label = _PRIORITY_RECOMMENDED
            else:
                label = _PRIORITY_OPTIONAL
            reason = _match_reason(scenario, tech_score, type_score, demoted_by=demoted_by)
            items.extend(self._scenario_items(scenario, label, reason, evidence_titles))
            if len(items) >= _MAX_RECOMMENDED_LOG_ITEMS:
                break
        return items[:_MAX_RECOMMENDED_LOG_ITEMS]

    def _specificity_demoted_technologies(self, context: str) -> dict[str, str]:
        """Returns {parent_technology_name: child_technology_name} for
        every governed technology whose full-phrase match in ``context``
        should be treated as a broader-family match, not a specific
        one, because a more specific child (``parent_technology_id``)
        also fully matches the same text. Empty when no
        ``LookupRepository`` is wired (older call sites/tests) or no
        technology in the governed table has a populated hierarchy yet
        -- never raises, never demotes anything without real evidence
        that a more specific name is genuinely present in the text."""
        if self._lookup is None:
            return {}
        technologies = self._lookup.list_technologies()
        by_id = {t.id: t for t in technologies}
        demoted: dict[str, str] = {}
        for tech in technologies:
            if tech.parent_technology_id is None:
                continue
            parent = by_id.get(tech.parent_technology_id)
            if parent is None or parent.name in demoted:
                continue
            if (
                _keyword_match_score(tech.name, context) >= _FULL_MATCH_SCORE
                and _keyword_match_score(parent.name, context) >= _FULL_MATCH_SCORE
            ):
                demoted[parent.name] = tech.name
        return demoted

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
        technology: str | None = None,
    ) -> InvestigationStrategy:
        matched_component = self._match_component(investigation, entities)
        required_evidence = self._required_evidence(investigation, recommended_logs, entities)
        missing_evidence = [item for item in required_evidence if not item.satisfied]
        stage, stage_rationale = self._current_stage(investigation, missing_evidence, root_causes)
        progress, progress_summary = self._progress(stage, required_evidence, missing_evidence)
        suggested_sql_items = self._suggested_sql_items(matched_component, entities)

        tfs_matches = wiki_matches = recommended_solution = None
        sources = _SynthesizedSources()
        if self._external_knowledge is not None:
            tfs_matches, wiki_matches = self._external_knowledge.gather(
                investigation, entities, matched_component, technology
            )
            recommended_solution, sources = self._synthesize_recommendation(
                root_causes, similar_investigations, tfs_matches, wiki_matches, missing_evidence,
                known_bugs=known_bugs, investigation_context=investigation.context_text,
            )

        # Resolution Provenance (2026-08-13): always computed, even when
        # External Knowledge isn't wired -- root-cause/log/SQL evidence
        # don't depend on TFS/Wiki being configured, and the tier
        # calculation degrades gracefully (never guesses) when they
        # aren't present. See _build_provenance_record's docstring.
        provenance = self._build_provenance_record(
            root_causes=root_causes,
            similar_investigations=similar_investigations,
            recommended_logs=recommended_logs,
            suggested_sql=suggested_sql_items,
            recommended_solution=recommended_solution,
            sources=sources,
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
            suggested_sql=suggested_sql_items,
            matched_component=matched_component,
            historical_investigations=similar_investigations,
            known_bugs=known_bugs,
            documentation=relevant_docs,
            tfs_matches=tfs_matches,
            wiki_matches=wiki_matches,
            recommended_solution=recommended_solution,
            decision_checkpoint=self._decision_checkpoint(root_causes),
            provenance=provenance,
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
        result -- see ``_match_single_technology`` for the actual
        matching rule."""
        if investigation.technology:
            return investigation.technology
        if self._log_knowledge is None:
            return None
        return self._match_single_technology(investigation.context_text)

    def _match_single_technology(self, context: str) -> str | None:
        """Real bug fix (2026-08-13, Final Knowledge-Quality
        Acceptance Test, Findings #1/#2): the previous version scored
        every candidate technology and kept the "best" via a ``(score,
        len(name))`` tuple starting from a ``(0.0, 0)`` sentinel --
        since any non-empty technology name beats that sentinel even
        at a real score of 0.0, this fabricated a confident "match"
        (whichever technology happened to have the single longest
        name in the whole governed table -- "Tool Data (BCS / HHU)
        processing" in practice) for *every* investigation that never
        mentioned any real technology at all. The same length-based
        tie-break also picked the wrong *sibling* technology on a
        genuine partial-match tie: "Mesh IP" (no "RF" prefix) ties "RF
        Mesh", "RF Mesh IP", and "RF Mesh (DAS implementation)" at a
        shared 0.5 (via the single significant word "Mesh"), and the
        longest name -- "RF Mesh (DAS implementation)", which has
        nothing to do with "IP" at all -- always won.

        Fixed rule, generic (no technology names hardcoded anywhere),
        reusing the real ``Technology.parent_technology_id`` hierarchy
        via ``app.engines.shared.hierarchy.most_specific`` -- the same
        mechanism ``_specificity_demoted_technologies`` and the
        classification engine's ``_best_match`` already use:

        1. A candidate with a score of exactly 0.0 is evidence of
           *nothing* and is excluded outright -- it can never become
           "the match" merely for lacking competition.
        2. Among the candidates tied at the single highest *positive*
           score, the most specific one wins (a matched child beats a
           matched parent) -- e.g. text containing "RF Mesh IP" ties
           "RF Mesh" (1.0, a real prefix match) and "RF Mesh IP" (1.0,
           the exact phrase); "RF Mesh" is an ancestor of the matched
           "RF Mesh IP", so it's filtered out, leaving "RF Mesh IP" as
           the unique, correct answer.
        3. If more than one candidate remains after that filtering --
           two genuinely unrelated technologies tied, or (the "Mesh
           IP" case) two *sibling* technologies neither more specific
           than the other once their shared parent is filtered out --
           the evidence is genuinely insufficient to pick one.
           Returns ``None`` rather than guessing; never a length-based
           or alphabetical fallback.

        Uses the governed ``technologies`` table (via ``self._lookup``)
        when available -- it carries the real hierarchy Phase 1
        populated from this exact Log Intelligence Knowledge Base, so
        the candidate universe is unchanged, just now hierarchy-aware.
        Falls back to ``available_log_technologies()``'s flat name list
        (no hierarchy to resolve a tie with, so any genuine tie stays
        unresolved) only when no ``LookupRepository`` is wired --
        graceful degradation for older call sites/tests, same "don't
        guess" contract either way.

        Evidence threshold (2026-08-13, generic-single-word-overlap
        fix): scoring uses ``_technology_evidence_score``, not the
        shared ``keyword_match_score`` directly -- see that method's
        own docstring for why a generic single-word overlap (e.g.
        "processing" alone matching "Tool Data (BCS / HHU)
        processing") must not count as evidence on its own, while
        still allowing genuinely single-significant-word technology
        names (e.g. "RF Mesh") to match on that one word.
        """
        technologies = self._lookup.list_technologies() if self._lookup is not None else []
        if technologies:
            candidates = [(t.name, t.id) for t in technologies]
            parent_of = {t.id: t.parent_technology_id for t in technologies}
        else:
            candidates = [(name, name) for name in self.available_log_technologies()]
            parent_of = {}

        scores = {id_: self._technology_evidence_score(name, context) for name, id_ in candidates}
        positive = [(name, id_) for name, id_ in candidates if scores[id_] > 0]
        if not positive:
            return None

        best_score = max(scores[id_] for _name, id_ in positive)
        tied = [(name, id_) for name, id_ in positive if scores[id_] == best_score]
        if len(tied) == 1:
            return tied[0][0]

        survivor_ids = most_specific({id_ for _name, id_ in tied}, parent_of)
        survivors = [name for name, id_ in tied if id_ in survivor_ids]
        return survivors[0] if len(survivors) == 1 else None

    @staticmethod
    def _technology_evidence_score(name: str, context: str) -> float:
        """Real bug fix (2026-08-13): ``_match_single_technology`` used
        to score candidates via the shared ``keyword_match_score``,
        whose partial-match fallback accepts *any single* significant
        (>=4 char) word from a multi-word candidate name -- so
        "Command processing appears delayed" (nothing but the ordinary
        English word "processing" in common with the technology's
        name) was enough to confidently return "Tool Data (BCS / HHU)
        processing" as the matched technology, even though the far
        more distinctive words "Tool" and "Data" from that same name
        never appeared anywhere in the text.

        Generic evidence-coverage rule (no technology names or generic
        words hardcoded anywhere -- this is a property of how much of
        any given candidate's *own* name is actually present, computed
        identically for every candidate):

        1. An exact/full phrase match is always valid evidence
           (unchanged -- delegates to ``keyword_match_score``).
        2. Short of that, a partial match is only valid when *every
           one* of the candidate's own significant words is present in
           the text -- full coverage of that candidate's identifying
           vocabulary, not a fragment of it. A candidate whose name
           has only one significant word to begin with (e.g. "RF
           Mesh", whose only word >=4 chars is "Mesh") is unaffected:
           matching its one word is by definition 100% coverage of its
           own name, so it still counts -- this is what keeps the real
           RF-Mesh-family specificity/ambiguity resolution
           (``most_specific``, above) working exactly as before. A
           candidate with several significant words (e.g. "Tool Data
           (BCS / HHU) processing" -> {Tool, Data, processing}) now
           needs all of them, not just one.
        3. Anything less -- a single word out of several, or no
           significant words present at all -- scores 0.0: no
           evidence, excluded outright by the caller's positive-score
           filter, never "the best we found by default"."""
        full = keyword_match_score(name, context)
        if full >= _FULL_MATCH_SCORE:
            return _FULL_MATCH_SCORE
        significant_words = [w for w in re.findall(r"[A-Za-z]+", name) if len(w) >= _MIN_KEYWORD_WORD_LEN]
        if not significant_words:
            return 0.0
        if all(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", context, re.IGNORECASE) for w in significant_words):
            return _PARTIAL_MATCH_SCORE
        return 0.0

    @staticmethod
    def _known_bug_has_topical_overlap(known_bug_title: str, investigation_context: str) -> bool:
        """Real defect fix (2026-08-13, Phase 0 acceptance review,
        finding #1): ``best_known_bug`` used to be selected on raw
        semantic similarity score alone (``score >=
        min_similarity_for_root_cause``) plus a bare "does it have a
        workaround at all" check -- with no signal comparable to what
        TFS/Wiki matches get from ``ranking.py``'s multi-signal
        ``score_candidate()`` (component/technology/title-overlap/
        entity/customer weights). Demonstrated live against the real
        corpus: with the Known Bugs table this thin (4 rows, all
        fictional Sprint-1 seed content, none real), 7 of 10 real
        investigations tested surfaced a Known Bug driving a LIKELY
        tier and a quoted "recommended resolution" from embedding
        proximity alone -- including "IIS worker process crash on
        multipart uploads over 50MB" at 56% similarity for a generic
        "network connectivity" investigation with zero actual
        relevance, and "Kafka client rebalance storm" at 65% for an
        unrelated TEPCO ticket. A small, thin, generic knowledge source
        crosses a 0.35 cosine-similarity floor against almost any
        sufficiently technical-sounding text -- unlike the 728-row real
        Historical Investigation corpus this same floor was calibrated
        against, where a genuinely irrelevant top-1 match at that score
        is rare because there's usually real, on-topic content to win
        instead.

        This is a real, deterministic sanity check, not a new ranking
        subsystem: does the investigation's own text actually share at
        least one real, distinguishing word (>=4 chars, the same
        significance bar as every other word-level check in this file)
        with the Known Bug's own title? A genuine match on real content
        survives (e.g. "queue" shared between "Commands stuck in
        Pending... queue increasing" and "Collector command queue
        stalls..."); a same-magnitude score with zero shared vocabulary
        (the two false positives above) does not. Deliberately "any
        overlap" rather than ``_technology_evidence_score``'s stricter
        full-coverage-of-every-significant-word rule -- that rule
        exists to disambiguate between competing *named* technology
        candidates (RF Mesh vs RF Mesh IP); this is a plain relevance
        sanity check on a single already-selected top match, a
        different problem with a deliberately looser bar."""
        significant_words = {
            w.lower()
            for w in re.findall(r"[A-Za-z]+", known_bug_title)
            if len(w) >= _MIN_KEYWORD_WORD_LEN and w.lower() not in _ENGLISH_STOPWORDS
        }
        if not significant_words:
            return False
        context_words = {w.lower() for w in re.findall(r"[A-Za-z]+", investigation_context)}
        return bool(significant_words & context_words)

    def _resolve_retrieval_context(
        self, investigation: InvestigationSession, technology: str | None
    ) -> RetrievalContext:
        """Builds the ``RetrievalContext`` for one Analyze call --
        ``investigation.customer`` (free text, engineer-entered) is
        resolved against the governed, controlled Customer list by
        exact name/alias match only; a customer name that doesn't
        match any governed row yields ``customer_id=None`` (unknown,
        never guessed -- approved product decision #1: never infer a
        customer, and this specifically means never treating an
        unmatched free-text value as if it were a real, taggable
        customer). Region has no InvestigationSession field yet (no
        Workspace UI captures it today) -- left unresolved rather than
        inferred from customer or anything else; a real fix is adding
        that field, not guessing around its absence."""
        customer_id = customer_name = None
        if self._lookup is not None and investigation.customer and investigation.customer.strip():
            customer = self._lookup.get_customer_by_name(investigation.customer.strip())
            if customer is not None:
                customer_id, customer_name = customer.id, customer.name
        return RetrievalContext(
            customer_id=customer_id,
            customer_name=customer_name,
            region_id=None,
            region_name=None,
            technology_name=technology,
        )

    def _synthesize_recommendation(
        self,
        root_causes: list[RootCauseHypothesis],
        similar_investigations: list[KnowledgeMatch],
        tfs_result: "ExternalKnowledgeResult | None",
        wiki_result: "ExternalKnowledgeResult | None",
        missing_evidence: list[RequiredEvidenceItem],
        known_bugs: list[KnowledgeMatch] | None = None,
        investigation_context: str = "",
    ) -> tuple[RecommendedSolution, "_SynthesizedSources"]:
        """Correlates local KB + live TFS + live Wiki + Known Bugs into
        one answer. Deterministic throughout: picks whichever real
        source(s) clear a confidence bar, quotes/references their
        actual content, and is honest (``insufficient_evidence=True``,
        no resolution text) when none do -- never invents a fix. See
        RecommendedSolution's own docstring for the full
        source-attribution contract.

        ``known_bugs`` (2026-08-13, Phase 0 -- Chat/Structured
        Resolution Knowledge architecture) is a new, additive keyword
        parameter -- defaults to ``None``/empty so every pre-existing
        positional call site (tests included) keeps working unchanged.
        Closes the gap flagged in
        RESOLVEIQ_CHAT_AND_RESOLUTION_ARCHITECTURE.md Section 17/
        Section 6: Known Bugs were structurally present (indexed,
        searched, shown in ``InvestigationStrategy.known_bugs``) but
        never actually consulted here, so a strong Known Bug match
        could never surface as *the* recommended resolution or
        contribute a real provenance tier -- only TFS/Wiki/local could.

        ``investigation_context`` (2026-08-13, Phase 0 acceptance
        review, defect #1) -- also new, also additive, also defaults to
        ``""`` (empty context yields no topical-overlap evidence, so
        ``best_known_bug`` simply never qualifies for a caller that
        doesn't supply one -- the same graceful-degradation contract as
        every other optional dependency in this engine). Required by
        ``_known_bug_has_topical_overlap`` -- see that method's
        docstring for the real, live-demonstrated defect this closes:
        Known Bug selection used to trust raw semantic similarity alone
        (the exact failure mode this whole engine was built to avoid
        for every *other* source), which -- given the Known Bugs table
        is currently only 4 small, generic, fictional Sprint-1 rows --
        let an entirely unrelated bug's workaround get quoted as *the*
        recommended resolution and drive a LIKELY tier.

        Returns the ``RecommendedSolution`` alongside a
        ``_SynthesizedSources`` (2026-08-13, Resolution Provenance
        phase) -- purely an exposure of internal state this function
        already computes (``correlated_local``/``best_tfs``/
        ``best_wiki``/``best_known_bug``), added so
        ``_build_provenance_record`` doesn't need to re-derive the same
        cross-source-correlation logic a second time. The decision
        logic below is completely unchanged from before this field
        existed."""
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
        best_known_bug = (
            known_bugs[0]
            if known_bugs
            and known_bugs[0].score >= self._settings.min_similarity_for_root_cause
            and self._known_bug_has_topical_overlap(known_bugs[0].title, investigation_context)
            else None
        )

        if not (best_local or best_tfs or best_wiki or best_known_bug):
            return (
                RecommendedSolution(
                    likely_issue="Not enough evidence yet to identify a likely root cause.",
                    rationale="No local historical match, TFS case, Wiki page, or Known Bug currently meets the "
                    "confidence bar -- more evidence is needed before recommending a specific fix.",
                    what_to_check=what_to_check,
                    recommended_resolution=None,
                    insufficient_evidence=True,
                    confidence="Insufficient",
                ),
                _SynthesizedSources(),
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

        # Known Bug (Phase 0, 2026-08-13) -- same "quote real content,
        # never fabricate" discipline as the three sources above.
        # workaround/status live in KnowledgeMatch.metadata, populated
        # by KnowledgeEngine.index_known_bug (see that method).
        if best_known_bug is not None:
            workaround = best_known_bug.metadata.get("workaround")
            bug_status = best_known_bug.metadata.get("status")
            if likely_issue is None:
                likely_issue = best_known_bug.title
            status_note = f" (status: {bug_status})" if bug_status else ""
            rationale_parts.append(f'a known bug "{best_known_bug.title}"{status_note}')
            confidence_score = max(confidence_score, best_known_bug.score)
            if recommended_resolution is None and workaround:
                recommended_resolution = f'Per known bug "{best_known_bug.title}": {workaround}'

        if likely_issue is None:
            likely_issue = "Multiple weak signals found -- no single source clearly explains the issue yet."

        return (
            RecommendedSolution(
                likely_issue=likely_issue,
                rationale="Based on " + " and ".join(rationale_parts) + ".",
                what_to_check=what_to_check,
                recommended_resolution=recommended_resolution,
                insufficient_evidence=recommended_resolution is None,
                confidence=confidence_for_score(confidence_score) if recommended_resolution else "Low",
                source_local=local_contributed,
                source_tfs=best_tfs is not None,
                source_wiki=best_wiki is not None,
                source_known_bug=best_known_bug is not None,
                supporting_tfs_id=best_tfs.tfs_case.tfs_id if best_tfs else None,
                supporting_tfs_url=best_tfs.tfs_case.url if best_tfs else None,
                supporting_wiki_title=best_wiki.wiki_page.title if best_wiki else None,
                supporting_wiki_url=best_wiki.wiki_page.url if best_wiki else None,
                supporting_known_bug_id=best_known_bug.record_id if best_known_bug else None,
                supporting_known_bug_title=best_known_bug.title if best_known_bug else None,
            ),
            _SynthesizedSources(
                correlated_match=correlated_local,
                best_tfs=best_tfs,
                best_wiki=best_wiki,
                best_local=best_local,
                best_known_bug=best_known_bug,
            ),
        )

    # --- Resolution Provenance (2026-08-13) ---------------------------------
    #
    # Everything below wraps output _build_root_causes/_synthesize_recommendation/
    # _recommend_logs/_suggested_sql_items already produced into a uniform,
    # traceable EvidenceReference shape, and computes the Confirmed/Likely/
    # Possible/Unknown tier via a deterministic rule that never lets a
    # similarity score alone reach Confirmed. No new retrieval, no new
    # matching, no LLM -- see app/domain/provenance.py's module docstring.

    def _local_match_for_cause(
        self, cause: RootCauseHypothesis, similar_investigations: list[KnowledgeMatch]
    ) -> "KnowledgeMatch | None":
        """The specific similar_investigations entry backing one
        RootCauseHypothesis, found the exact same way
        _synthesize_recommendation's own correlated_match already does
        (root_cause snippet equality) -- reused here rather than a new
        mechanism, applied per-cause so a second root-cause hypothesis
        (also historical-investigation-derived) resolves to its own
        real match, not silently reuses the first one's."""
        if not cause.rationale.startswith("Matches historical investigation"):
            return None
        for match in similar_investigations[:2]:
            root_cause_text = match.metadata.get("root_cause")
            if root_cause_text and _snippet(root_cause_text) == cause.description:
                return match
        return None

    def _root_cause_evidence(
        self, root_causes: list[RootCauseHypothesis], similar_investigations: list[KnowledgeMatch]
    ) -> list[EvidenceReference]:
        """Answers "what evidence supports the proposed root cause" --
        one EvidenceReference per hypothesis _build_root_causes already
        produced, typed by where it actually came from (a real
        historical investigation vs. an entity heuristic with no
        governed record behind it)."""
        evidence: list[EvidenceReference] = []
        for cause in root_causes[:2]:
            local_match = self._local_match_for_cause(cause, similar_investigations)
            if local_match is not None:
                evidence.append(
                    EvidenceReference(
                        kind=EvidenceKind.HISTORICAL_INVESTIGATION,
                        source_id=local_match.record_id,
                        title=local_match.title,
                        score=local_match.score,
                        reason=cause.rationale,
                        contributes_to=["root_cause"],
                    )
                )
            else:
                evidence.append(
                    EvidenceReference(
                        kind=EvidenceKind.ENTITY_HEURISTIC,
                        source_id=cause.description,
                        title=cause.description,
                        score=cause.confidence,
                        reason=cause.rationale,
                        contributes_to=["root_cause"],
                    )
                )
        return evidence

    def _resolution_evidence(
        self,
        recommended_solution: "RecommendedSolution | None",
        sources: "_SynthesizedSources",
        primary_local_match: "KnowledgeMatch | None",
    ) -> list[EvidenceReference]:
        """Answers "what evidence supports the proposed resolution" --
        closes the real gap from the approved design (§2, question 7):
        TFS/Wiki already had structured supporting-reference fields on
        RecommendedSolution; the local historical-investigation source
        did not. All three are now uniformly represented."""
        if recommended_solution is None:
            return []
        evidence: list[EvidenceReference] = []
        if recommended_solution.source_local and primary_local_match is not None:
            reason = (
                "Backs the synthesized resolution text."
                if recommended_solution.recommended_resolution
                else "Backs the likely issue -- no confirmed root cause or resolution on file for this record."
            )
            evidence.append(
                EvidenceReference(
                    kind=EvidenceKind.HISTORICAL_INVESTIGATION,
                    source_id=primary_local_match.record_id,
                    title=primary_local_match.title,
                    score=primary_local_match.score,
                    reason=reason,
                    contributes_to=["resolution"],
                )
            )
        if recommended_solution.source_tfs and sources.best_tfs is not None:
            case = sources.best_tfs.tfs_case
            evidence.append(
                EvidenceReference(
                    kind=EvidenceKind.TFS_CASE,
                    source_id=str(case.tfs_id),
                    title=case.title,
                    url=case.url,
                    score=sources.best_tfs.score,
                    reason="; ".join(sources.best_tfs.match_reasons) or f"TFS-{case.tfs_id} ({case.state}).",
                    contributes_to=["resolution"],
                )
            )
        if recommended_solution.source_wiki and sources.best_wiki is not None:
            page = sources.best_wiki.wiki_page
            evidence.append(
                EvidenceReference(
                    kind=EvidenceKind.WIKI_PAGE,
                    source_id=page.page_id,
                    title=page.title,
                    url=page.url,
                    score=sources.best_wiki.score,
                    reason="; ".join(sources.best_wiki.match_reasons) or f'Wiki page "{page.title}".',
                    contributes_to=["resolution"],
                )
            )
        if recommended_solution.source_known_bug and sources.best_known_bug is not None:
            reason = (
                "Backs the synthesized resolution text."
                if recommended_solution.recommended_resolution
                else "Backs the likely issue -- no recorded workaround on file for this known bug."
            )
            evidence.append(
                EvidenceReference(
                    kind=EvidenceKind.KNOWN_BUG,
                    source_id=sources.best_known_bug.record_id,
                    title=sources.best_known_bug.title,
                    score=sources.best_known_bug.score,
                    reason=reason,
                    contributes_to=["resolution"],
                )
            )
        return evidence

    def _verified_local_record(
        self, primary_local_match: "KnowledgeMatch | None"
    ) -> "HistoricalInvestigationRecord | None":
        """Fetches the full, real HistoricalInvestigationRecord fresh
        from the repository (via KnowledgeEngine.get_historical_investigation
        -- already an existing public method, not a new lookup path) so
        its resolution_verified flag is authoritative rather than
        trusting Chroma's own (search-relevance-oriented, potentially
        stale) metadata. Returns None unless the record both exists and
        is genuinely marked verified -- an unverified or missing record
        is never treated as evidence of anything."""
        if primary_local_match is None:
            return None
        record = self._knowledge.get_historical_investigation(primary_local_match.record_id)
        if record is not None and record.resolution_verified:
            return record
        return None

    def _verified_known_bug_record(self, best_known_bug: "KnowledgeMatch | None") -> "KnownBugRecord | None":
        """The Known Bug analogue of ``_verified_local_record`` --
        added 2026-08-13, Phase 0. Same discipline: fetches fresh from
        the repository (never trusts Chroma's own metadata for this),
        returns None unless the record both exists and is genuinely
        marked verified."""
        if best_known_bug is None:
            return None
        record = self._knowledge.get_known_bug(best_known_bug.record_id)
        if record is not None and record.resolution_verified:
            return record
        return None

    def _resolve_provenance_tier(
        self,
        *,
        recommended_solution: "RecommendedSolution | None",
        sources: "_SynthesizedSources",
        root_causes: list[RootCauseHypothesis],
        similar_investigations: list[KnowledgeMatch],
    ) -> tuple[ResolutionProvenance, str]:
        """The approved rule, exactly: Confirmed only from real
        cross-source correlation or explicit human verification --
        never from a similarity score alone, however high. See
        app/domain/provenance.py's ResolutionProvenance docstring for
        the full rule; this is its one implementation."""
        primary_local_match = self._local_match_for_cause(root_causes[0], similar_investigations) if root_causes else None

        if sources.correlated_match is not None and sources.best_tfs is not None:
            return (
                ResolutionProvenance.CONFIRMED,
                f'Cross-source correlation: TFS-{sources.best_tfs.tfs_case.tfs_id} and local historical '
                f'investigation "{sources.correlated_match.title}" reference the same real-world case '
                f"(matching ticket/CRM reference) -- not a similarity score, two independent systems agreeing.",
            )

        verified_record = self._verified_local_record(primary_local_match)
        if verified_record is not None:
            note = f" -- {verified_record.resolution_verification_note}" if verified_record.resolution_verification_note else ""
            when = verified_record.resolution_verified_at.isoformat() if verified_record.resolution_verified_at else "an unrecorded date"
            return (
                ResolutionProvenance.CONFIRMED,
                f'Explicitly verified by {verified_record.resolution_verified_by or "an administrator"} on {when}{note}.',
            )

        verified_bug = self._verified_known_bug_record(sources.best_known_bug)
        if verified_bug is not None:
            note = f" -- {verified_bug.resolution_verification_note}" if verified_bug.resolution_verification_note else ""
            when = verified_bug.resolution_verified_at.isoformat() if verified_bug.resolution_verified_at else "an unrecorded date"
            return (
                ResolutionProvenance.CONFIRMED,
                f'Explicitly verified by {verified_bug.resolution_verified_by or "an administrator"} on {when}{note} '
                f'(known bug "{verified_bug.title}").',
            )

        if (
            primary_local_match is not None
            and primary_local_match.score >= self._settings.min_similarity_for_root_cause
            and primary_local_match.metadata.get("root_cause")
        ):
            return (
                ResolutionProvenance.LIKELY,
                f'Single local historical match ("{primary_local_match.title}", {primary_local_match.score:.0%} '
                f"similarity) with a recorded root cause -- no independent corroborating source.",
            )
        if sources.best_tfs is not None and sources.best_tfs.confidence == "High" and sources.best_tfs.tfs_case.resolution_text:
            return (
                ResolutionProvenance.LIKELY,
                f"High-confidence TFS match (TFS-{sources.best_tfs.tfs_case.tfs_id}) with real resolution "
                f"text -- no independent corroborating source.",
            )
        if sources.best_wiki is not None and sources.best_wiki.confidence == "High" and sources.best_wiki.wiki_page.excerpt:
            return (
                ResolutionProvenance.LIKELY,
                f'High-confidence Wiki match ("{sources.best_wiki.wiki_page.title}") with real guidance '
                f"content -- no independent corroborating source.",
            )
        if sources.best_known_bug is not None and sources.best_known_bug.metadata.get("workaround"):
            return (
                ResolutionProvenance.LIKELY,
                f'Single known-bug match ("{sources.best_known_bug.title}", {sources.best_known_bug.score:.0%} '
                f"similarity) with a recorded workaround -- no independent corroborating source.",
            )

        insufficient = recommended_solution is None or recommended_solution.insufficient_evidence
        has_any_evidence = (
            primary_local_match is not None
            or sources.best_local is not None
            or sources.best_tfs is not None
            or sources.best_wiki is not None
            or sources.best_known_bug is not None
        )
        if insufficient and not has_any_evidence:
            return ResolutionProvenance.UNKNOWN, "No source currently meets the confidence bar to support a root cause or resolution."

        return (
            ResolutionProvenance.POSSIBLE,
            "Evidence exists but doesn't reach a single strong, corroborated source -- treat as a hypothesis "
            "to verify, not a confirmed or likely resolution.",
        )

    def _build_provenance_record(
        self,
        *,
        root_causes: list[RootCauseHypothesis],
        similar_investigations: list[KnowledgeMatch],
        recommended_logs: list[RecommendedLogCollectionItem],
        suggested_sql: list[SuggestedSqlItem],
        recommended_solution: "RecommendedSolution | None",
        sources: "_SynthesizedSources",
    ) -> ProvenanceRecord:
        """The single entry point: wraps every already-computed
        recommendation into a traceable EvidenceReference and resolves
        the resolution's trust tier. Always returns a real
        ProvenanceRecord -- never None -- so a future consumer (the
        Investigation Workspace today, a Chat Assistant later) can
        always ask "why" and get a real, structured answer, even for a
        brand-new investigation with nothing in it yet (empty evidence
        lists, tier UNKNOWN)."""
        primary_local_match = self._local_match_for_cause(root_causes[0], similar_investigations) if root_causes else None

        log_evidence = [
            EvidenceReference(
                kind=EvidenceKind.LOG_COLLECTION_STEP,
                source_id=item.scenario_id,
                title=f"{item.component_name} ({item.scenario_technology} / {item.scenario_type})",
                reason=item.match_reason,
                contributes_to=["log_recommendation"],
            )
            for item in recommended_logs
        ]
        sql_evidence = [
            EvidenceReference(
                kind=EvidenceKind.SQL_TEMPLATE if item.source == "sql_library" else EvidenceKind.ENTITY_HEURISTIC,
                source_id=item.template_id or item.title,
                title=item.title,
                reason=item.match_reason,
                contributes_to=["sql_recommendation"],
            )
            for item in suggested_sql
        ]

        tier, rationale = self._resolve_provenance_tier(
            recommended_solution=recommended_solution,
            sources=sources,
            root_causes=root_causes,
            similar_investigations=similar_investigations,
        )

        return ProvenanceRecord(
            root_cause_evidence=self._root_cause_evidence(root_causes, similar_investigations),
            resolution_evidence=self._resolution_evidence(recommended_solution, sources, primary_local_match),
            log_recommendation_evidence=log_evidence,
            sql_recommendation_evidence=sql_evidence,
            resolution_provenance=tier,
            provenance_rationale=rationale,
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
        component, consistent with every other tie-break in this file.

        Real bug fix (2026-08-13, Final Knowledge-Quality Acceptance
        Test, Finding #3): this used to also accept a *partial* match
        -- a single significant word (>=4 chars) from a multi-word
        component name appearing anywhere in the text -- as "weakly
        matched" evidence, and still returned it as *the* matched
        component with no floor. Component names are ordinary English
        words far more often than technology names are ("Device Hub",
        "Network Hub", ...), so mentioning "network" in passing was
        enough to confidently claim "Network Hub" was matched. Fixed
        generically, not with a blacklist of generic words: a partial
        (single-word) match is no longer accepted as component-
        matching evidence at all -- only a real match on the
        component's *complete* name (or an exact extracted entity
        value, handled above) counts. This is a stricter version of
        the exact same ``keyword_match_score`` rule already used
        throughout this file, not a second matching mechanism."""
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
                if score < _FULL_MATCH_SCORE:
                    # A partial, single-generic-word match is not
                    # sufficiently distinctive evidence on its own --
                    # see this method's docstring.
                    continue
                reason = f'Matched because the investigation text mentions "{component.name}" by name.'
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
        the (also pre-existing) fallback path.

        Real gap fix (2026-08-13, Resolution Provenance phase, approved
        item 7): every item now carries ``match_reason`` -- why *this
        investigation* got *this* suggestion, distinct from
        ``explanation`` (the template's own static description of what
        the query does, unchanged). Iterates ``_ENTITY_HEURISTICS``
        directly for the fallback branch (rather than calling
        ``_suggest_sql``, which only returns bare SQL strings and loses
        which heuristic/entity produced each one) so the reason can
        name the real entity that triggered it; ``_suggest_sql`` itself
        is untouched and still backs the legacy flat
        ``Recommendation.suggested_sql`` field exactly as before."""
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
                            match_reason=(
                                f'Linked to the matched component "{matched_component.component_name}" via this '
                                f"SQL template's related_components."
                            ),
                        )
                    )
        if items:
            return items

        entity_by_type = {e.entity_type: e for e in entities}
        fallback_items: list[SuggestedSqlItem] = []
        for heuristic in _ENTITY_HEURISTICS:
            entity = entity_by_type.get(heuristic.entity_type)
            if entity is not None and heuristic.suggested_sql:
                fallback_items.append(
                    SuggestedSqlItem(
                        title="Suggested query",
                        sql_text=heuristic.suggested_sql.format(value=entity.value),
                        source="entity_heuristic",
                        match_reason=(
                            f'Investigation evidence contains a {heuristic.entity_type.value} entity ("{entity.value}").'
                        ),
                    )
                )
        return fallback_items

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
