"""Command Flow reconstruction and cross-file log search.

Both take a *fully-hydrated* ``InvestigationSession`` (every evidence
item's log_events/entities already populated) -- the same shape
``RecommendationEngine.generate()`` already requires, fetched once via
``InvestigationEngine.get_investigation()``, not a new full-hydration
path. See app/domain/log_flow.py for why these two features exist and
what they deliberately don't invent.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.domain.enums import EvidenceType
from app.domain.investigation import InvestigationSession
from app.domain.log_flow import CommandFlow, FlowStep, LogSearchHit, LogSearchResult

if TYPE_CHECKING:
    from app.domain.log_intelligence_kb import LogCollectionScenario, LogSourceApplication
    from app.infrastructure.db.log_knowledge_repository import LogKnowledgeRepository

_MAX_SEARCH_HITS = 500
"""Same capping discipline as everywhere else in this engine (context
text, snippets, top_k) -- a filter that still matches thousands of
lines needs a scannable result, not a second full dump. total_matches
stays exact; only the returned list is capped."""

_MAX_MESSAGE_CHARS = 500

_GENERIC_FILENAME_STEMS = {"log", "logfile", "logs"}
"""Same set the Recommendation Engine's already-collected check uses --
a bare "logfile.log" is shared by dozens of unrelated sources in the
wiki and can't identify which one produced an uploaded file on its own."""
_MIN_COMPONENT_NAME_LEN = 4


def _hit_from_event(evidence, event) -> LogSearchHit:
    return LogSearchHit(
        evidence_id=evidence.id,
        evidence_title=evidence.title,
        timestamp=event.timestamp,
        level=event.level.value,
        message=event.message[:_MAX_MESSAGE_CHARS],
        line_number=event.line_number,
    )


def _event_matches(event, *, entity_type: str | None, entity_value: str | None, keyword: str | None, level: str | None) -> bool:
    if level and event.level.value.lower() != level.lower():
        return False
    if keyword and keyword.lower() not in event.message.lower():
        return False
    if entity_type or entity_value:
        found = any(
            (not entity_type or e.entity_type.value.lower() == entity_type.lower())
            and (not entity_value or e.value.lower() == entity_value.lower())
            for e in event.entities
        )
        if not found:
            return False
    return True


def search_logs(
    investigation: InvestigationSession,
    *,
    entity_type: str | None = None,
    entity_value: str | None = None,
    keyword: str | None = None,
    level: str | None = None,
) -> LogSearchResult:
    """Cross-file, deterministic filter over every LOG_FILE evidence
    item's parsed events -- entity type/value, message keyword, and
    level, ANDed together (any subset may be given). Replaces "pick one
    file, see every line in it" with an actual filter."""
    hits: list[LogSearchHit] = []
    total = 0
    for evidence in investigation.evidence:
        if evidence.evidence_type != EvidenceType.LOG_FILE:
            continue
        for event in evidence.log_events:
            if not _event_matches(event, entity_type=entity_type, entity_value=entity_value, keyword=keyword, level=level):
                continue
            total += 1
            if len(hits) < _MAX_SEARCH_HITS:
                hits.append(_hit_from_event(evidence, event))

    query_parts = []
    if entity_type or entity_value:
        query_parts.append(f"{entity_type or 'any type'}={entity_value or 'any value'}")
    if keyword:
        query_parts.append(f'"{keyword}"')
    if level:
        query_parts.append(f"level={level}")
    return LogSearchResult(
        query_summary=" AND ".join(query_parts) or "(no filter)",
        total_matches=total,
        truncated=total > len(hits),
        hits=hits,
    )


def _match_log_source(evidence_title: str, sources: list["LogSourceApplication"]) -> "LogSourceApplication | None":
    """Given an uploaded file's name, resolves which real
    LogSourceApplication it is -- the mirror image of the
    Recommendation Engine's already-collected check (see
    app/engines/recommendation/engine.py's _is_already_collected):
    component-name-in-filename first (covers a zip upload preserving
    "ComponentName/logfile.log"-shaped internal paths), a genuinely
    distinctive (non-generic) filename pattern second. Same
    conservative discipline -- a shared generic filename never counts
    alone, since a wrong component attribution here would put a real
    log line in the wrong lane of the flow."""
    title_lower = evidence_title.lower()
    name_candidates = [
        s for s in sources if len(s.name) >= _MIN_COMPONENT_NAME_LEN and s.name.lower() in title_lower
    ]
    if name_candidates:
        return max(name_candidates, key=lambda s: len(s.name))
    for source in sources:
        for pattern in source.location.filename_patterns:
            stem = re.sub(r"\.\w+\*?$", "", pattern).lower()
            if stem in _GENERIC_FILENAME_STEMS:
                continue
            if pattern.lower() in title_lower:
                return source
    return None


def _direction_for_scenario_type(scenario_type: str) -> str:
    """Grounded in the wiki's own arrow-chain lines for each scenario
    (e.g. "Command Request (Outbound): CC Web -> CommandProcessor ->
    ... -> Meter" vs "Command Response (Inbound): Meter -> ... -> CC")
    -- Outbound/Request scenarios describe the command traveling toward
    the meter; Inbound/Response scenarios describe the reply traveling
    back. Never inferred from an individual log line's text."""
    lowered = scenario_type.lower()
    if "outbound" in lowered or "request" in lowered:
        return "outbound"
    if "inbound" in lowered or "response" in lowered:
        return "inbound"
    return "unknown"


def _best_scenario(
    component_names: set[str], scenarios: list["LogCollectionScenario"]
) -> "LogCollectionScenario | None":
    """The scenario that explains the most of the components actually
    seen reporting this correlating value -- not just any scenario one
    of them happens to appear in (most components appear in several
    technology variants of the same operation)."""
    best = None
    best_score = 0
    for scenario in scenarios:
        step_names = {step.component_name for step in scenario.steps}
        score = len(step_names & component_names)
        if score > best_score:
            best_score = score
            best = scenario
    return best


def reconstruct_flow(
    investigation: InvestigationSession,
    *,
    entity_type: str,
    entity_value: str,
    log_knowledge_repo: "LogKnowledgeRepository",
) -> CommandFlow:
    """Groups every log event carrying this correlating value (across
    every file in the investigation) by the real component that
    produced it, then orders and labels those components using the
    wiki-derived scenario they actually belong to. A component with no
    matching log entry is still listed, flagged as a gap -- never
    silently omitted, since that gap is exactly where the trail goes
    cold."""
    sources = log_knowledge_repo.list_log_sources()
    scenarios = log_knowledge_repo.list_scenarios()

    matches_by_component: dict[str, list[LogSearchHit]] = {}
    unresolved: list[LogSearchHit] = []
    for evidence in investigation.evidence:
        if evidence.evidence_type != EvidenceType.LOG_FILE:
            continue
        source = _match_log_source(evidence.title, sources)
        for event in evidence.log_events:
            if not _event_matches(event, entity_type=entity_type, entity_value=entity_value, keyword=None, level=None):
                continue
            hit = _hit_from_event(evidence, event)
            if source is not None:
                matches_by_component.setdefault(source.name, []).append(hit)
            else:
                unresolved.append(hit)

    matched_component_count = len(matches_by_component)
    if not matches_by_component:
        return CommandFlow(
            correlating_entity_type=entity_type,
            correlating_value=entity_value,
            matched_component_count=0,
            unresolved_events=unresolved,
        )

    scenario = _best_scenario(set(matches_by_component.keys()), scenarios)
    outbound: list[FlowStep] = []
    inbound: list[FlowStep] = []

    if scenario is not None:
        direction = _direction_for_scenario_type(scenario.scenario_type)
        lane = outbound if direction == "outbound" else inbound if direction == "inbound" else outbound
        for step in sorted(scenario.steps, key=lambda s: s.priority):
            events = matches_by_component.pop(step.component_name, [])
            lane.append(
                FlowStep(
                    order=step.priority,
                    component_name=step.component_name,
                    direction=direction,
                    has_log_entry=bool(events),
                    log_source_id=step.log_source_id,
                    events=events,
                )
            )

    # Anything the chosen scenario didn't explain still surfaces --
    # never silently dropped just for falling outside the "best" match.
    for events in matches_by_component.values():
        unresolved.extend(events)

    return CommandFlow(
        correlating_entity_type=entity_type,
        correlating_value=entity_value,
        matched_component_count=matched_component_count,
        scenario_id=scenario.id if scenario else None,
        scenario_technology=scenario.technology if scenario else None,
        scenario_type=scenario.scenario_type if scenario else None,
        outbound_steps=outbound,
        inbound_steps=inbound,
        unresolved_events=unresolved,
    )
