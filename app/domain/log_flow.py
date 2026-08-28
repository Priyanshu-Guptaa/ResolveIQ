"""Log Intelligence: cross-file search and Command Flow reconstruction.

Both replace "pick one file, see every line in it" with something
actually useful, and both stay within the project's deterministic-only
discipline: search is a plain filter over already-extracted
``LogEvent``/``ExtractedEntity`` data (see
``app/engines/log_intelligence/flow.py``); flow reconstruction adds
exactly one further step -- resolving each matched event's producing
component via the Log Intelligence Knowledge Base (the same wiki-
derived data the Recommendation Engine already uses) and ordering/
labeling the result using that component's real scenario ("Command
Request (Outbound)" vs "Command Response (Inbound)", and the
wiki-preserved step order). No new text parsing, no LLM, nothing
inferred beyond what the wiki and the log's own extracted entities
already state.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class LogSearchHit(BaseModel):
    """One matching log line, from either the search view or a flow
    step's matched events."""

    evidence_id: str
    evidence_title: str
    timestamp: datetime | None = None
    level: str
    message: str
    line_number: int | None = None


class LogSearchResult(BaseModel):
    query_summary: str
    total_matches: int
    truncated: bool
    """True when total_matches exceeds how many hits are actually
    returned (see _MAX_SEARCH_HITS) -- the count itself is always
    exact, only the returned list is capped."""
    hits: list[LogSearchHit] = Field(default_factory=list)


class FlowStep(BaseModel):
    """One component's position in a reconstructed command flow --
    present even when no log entry was found for it (has_log_entry
    False), since knowing exactly where a real, wiki-documented step
    is missing evidence is itself the diagnostic value of this view."""

    order: int
    """Wiki-preserved collection order within its scenario -- see
    ``LogCollectionStep.priority``."""
    component_name: str
    direction: str
    """"outbound" | "inbound" | "unknown" -- derived from the matched
    scenario's ``scenario_type`` text (Outbound/Request vs
    Inbound/Response), never guessed per-event."""
    has_log_entry: bool
    log_source_id: str | None = None
    events: list[LogSearchHit] = Field(default_factory=list)


class LogEventCount(BaseModel):
    """One (label, count) pair -- either an OBSERVED FACT (a direct
    tally, e.g. a severity level) or a DETERMINISTIC OBSERVATION (a
    computed pattern, e.g. how many times one exception type recurred).
    See ``LogObservationSummary``'s docstring for the distinction."""

    label: str
    count: int


class LogObservationSummary(BaseModel):
    """Chat Assistant Phase 33 -- a deterministic, provenance-preserving
    summary of already-parsed ``LOG_FILE`` evidence, safe to place in an
    LLM prompt because it is a strict allowlist transform: every field
    is either a direct count or an already-recognized entity value (see
    ``app.engines.log_intelligence.entity_extractor``'s closed pattern
    registry), never a raw log line or arbitrary free text copied
    through. Building this is the ``DETERMINISTIC PARSER -> COMPACT
    STRUCTURED EVIDENCE`` step of the architecture this phase
    implements -- see ``LogIntelligenceEngine.summarize_observations``.

    Deliberately contains ONLY two of this project's four evidence
    categories (see that method's docstring for all four):

    - OBSERVED FACT: ``level_counts`` (a direct tally of severities
      actually present in the parsed events).
    - DETERMINISTIC OBSERVATION: ``top_exceptions`` (a computed
      frequency count of recognized exception/error-code entities).

    It never contains a HYPOTHESIS or a TROUBLESHOOTING CHECK -- this
    project's Phase 25/32 Rule 9 guarantee (only an already-computed
    ``resolution_candidates``/``validation_steps`` entry may become an
    "available check") is preserved by construction: nothing here ever
    populates those fields, so log evidence can never make a
    troubleshooting action evidence-backed on its own, no matter how
    strongly a log's own text is worded (Rule 10 in
    ``app.engines.llm.prompt_builder`` reinforces this at the prompt
    level, but the real guarantee is architectural -- this summary
    object simply has no field a fabricated check could occupy)."""

    source_evidence_ids: list[str] = Field(default_factory=list)
    """Which LOG_FILE evidence items this summary was computed from --
    provenance, traceable back to real, already-persisted Evidence rows."""
    analyzed_file_count: int = 0
    total_events: int = 0
    level_counts: list[LogEventCount] = Field(default_factory=list)
    top_exceptions: list[LogEventCount] = Field(default_factory=list)
    earliest_timestamp: datetime | None = None
    latest_timestamp: datetime | None = None


class CommandFlow(BaseModel):
    """The reconstructed flow for one correlating value (a Command Log
    ID, Meter Number, or Endpoint ID actually found in this
    investigation's uploaded logs)."""

    correlating_entity_type: str
    correlating_value: str
    matched_component_count: int
    scenario_id: str | None = None
    scenario_technology: str | None = None
    scenario_type: str | None = None
    """None when no wiki scenario explains at least one of the matched
    components -- the raw matches still surface via
    ``unresolved_events``, just without wiki ordering/direction."""
    outbound_steps: list[FlowStep] = Field(default_factory=list)
    inbound_steps: list[FlowStep] = Field(default_factory=list)
    unresolved_events: list[LogSearchHit] = Field(default_factory=list)
    """Events carrying the correlating value whose producing component
    either couldn't be identified at all, or wasn't part of the chosen
    scenario -- shown separately rather than silently dropped, since
    "we found something but can't place it in the flow" is still real
    information, not nothing."""
