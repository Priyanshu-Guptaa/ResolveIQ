"""Conversation State -- the deterministic, zero-LLM persistence
foundation Phase 4's (not-yet-built) Chat Orchestrator will consume
(2026-08-14, Phase 3 -- RESOLVEIQ_CHAT_AND_RESOLUTION_ARCHITECTURE.md
§11 "Conversation Context Model").

Reconciled against the real Phase 2 ``ParsedQuery`` shape rather than
the architecture doc's original §11 sketch, which predates Phase 2's
actual implementation -- concretely:

- §11's ``ConversationSlots`` only tracks
  customer/region/technology/component/version (plus per-field
  "stated"/"inferred"/"unset" provenance on two of them). Phase 2's
  real ``ParsedQuery`` also extracts Product, Exception Type, and
  Ticket References -- all three of which the *current* Phase 3
  request explicitly requires preserving across turns (see this
  module's own field list). This implementation tracks every dimension
  Phase 2 actually extracts, uniformly, rather than the doc's smaller
  original set.
- §11 uses ``Literal["stated","inferred","unset"]`` per-field
  provenance. Phase 2 has no "inferred" extraction path at all --
  every extraction is either a real governed-value match (EXACT/
  PARTIAL) or nothing (None) or a genuine tie (AMBIGUOUS); there is no
  third, weaker "inferred" signal source anywhere in this codebase to
  distinguish from "stated." This module renames that axis to
  ``SlotOrigin = Literal["stated","carried_forward","unset"]`` --
  "stated" (this turn's own text established/reaffirmed it),
  "carried_forward" (still active only because an earlier turn stated
  it and nothing since replaced it), "unset" -- which is both truthful
  to what Phase 2 can actually tell you and is exactly the axis the
  current Phase 3 request's test list asks for ("context carry-forward"
  / "explicit context replacement").
- §11's ``ChatResponse``/``ChatIntent`` (the rendered answer + its
  own intent enum) are deliberately NOT built here -- the current
  Phase 3 request is explicit that answer generation, the Chat
  Orchestrator, and the Chat API/UI are out of scope until Phase 4.
  ``ChatMessage`` here persists a message and (for user turns) the
  real, unmodified Phase 2 ``ParsedQuery`` plus this phase's own
  ``ReferenceResolution`` -- never a rendered answer, never a second
  intent taxonomy (assistant turns reuse whatever ``QueryIntent`` the
  most recent user turn carried; no new ``ChatIntent`` enum exists).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.external_knowledge import ExternalKnowledgeResult
from app.domain.provenance import ResolutionProvenance
from app.domain.query_understanding import ParsedQuery, QueryIntent, SlotConfidence
from app.domain.recommendation import KnowledgeMatch, RecommendedLogCollectionItem, SuggestedSqlItem
from app.domain.structured_resolution import StructuredResolution
from app.engines.knowledge.applicability import RetrievalContext


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


ConversationFocus = Literal["problem", "historical_match", "resolution", "logs", "sql"]
"""What "this"/"that"/"it" currently refers to when a message contains
a deictic cue that asks about a *kind* of focus rather than a specific
prior record -- the same small, closed set §11 specifies, so reference
resolution (see ``ReferenceResolution``) is always a lookup against a
known value, never a guess. Nothing in Phase 3 itself decides what an
answer "centered on" -- Phase 3 has no answer-generation step. This is
set only via ``ConversationStateEngine.record_assistant_turn``, an
explicit hook Phase 4's Chat Orchestrator will call once it exists;
this phase's own tests call it directly to prove the mechanism works,
simulating what that future caller will supply."""

SlotOrigin = Literal["stated", "carried_forward", "unset"]
"""See module docstring's reconciliation note. "stated" = this turn's
own text established or reaffirmed the value; "carried_forward" = the
value is still active only because an earlier turn stated it and
nothing since has replaced it; "unset" = no value at all."""

_CONFIDENCE_RANK: dict[SlotConfidence, int] = {SlotConfidence.PARTIAL: 1, SlotConfidence.EXACT: 2}
"""Ordering used only to decide, on a same-value reaffirmation, which
of the previously-recorded and newly-extracted confidence to keep --
the stronger evidence wins, weaker evidence is never allowed to
downgrade an already-EXACT match (e.g. a later bare "Mesh IP" mention,
Phase 2 PARTIAL, must not downgrade an earlier exact "RF Mesh IP"
mention). AMBIGUOUS deliberately has no rank here -- an AMBIGUOUS
``ExtractedSlot`` never reaches a ``ConversationSlot`` at all (see
``ConversationStateEngine._merge_generic_slot``), so it is never a
value this table needs to compare against."""


class ConversationSlot(BaseModel):
    """One governed dimension's currently-active value across a
    conversation -- the persisted, turn-over-turn accumulation that
    Query Understanding's own ``ExtractedSlot`` (single-turn, never
    persisted by Phase 2) feeds into. Deliberately a distinct, smaller
    shape than ``ExtractedSlot``: a conversation slot only ever holds a
    *resolved* value or none -- there is no AMBIGUOUS state here, because
    an ambiguous single-turn extraction never becomes the active
    conversation value (see ``ConversationStateEngine._merge_generic_slot``).
    The ambiguity itself is never lost -- it is still fully visible on
    that turn's own persisted ``ChatMessage.parsed_query`` -- it is just
    never promoted to "the answer.\""""

    value_id: str | None = None
    value_name: str | None = None
    confidence: SlotConfidence | None = None
    """EXACT or PARTIAL only, by construction -- see class docstring.
    None only when ``value_id``/``value_name`` are also None."""
    origin: SlotOrigin = "unset"
    set_by_message_id: str | None = None
    """The user message whose extraction most recently established or
    reaffirmed this value -- the same "why do you believe this" trace
    every other provenance-bearing model in this codebase already
    carries (``EvidenceReference.source_id``, ``ExtractedSlot.evidence_snippet``)."""


class ConversationSlots(BaseModel):
    """What the conversation currently believes about the problem --
    persisted on ``ChatSession``, incrementally updated turn over turn.
    The chat-native, longer-lived counterpart to ``RetrievalContext``
    (which is single-shot, computed fresh per call) -- see
    ``ConversationStateEngine.to_retrieval_context`` for the narrow,
    pure-mapping bridge between the two (§13), and
    ``ConversationStateEngine.add_user_message`` for the merge rules
    that keep this "existing-context-wins" across an entire
    conversation, the same discipline Phase 2 already applies within a
    single call."""

    customer: ConversationSlot = Field(default_factory=ConversationSlot)
    region: ConversationSlot = Field(default_factory=ConversationSlot)
    technology: ConversationSlot = Field(default_factory=ConversationSlot)
    product: ConversationSlot = Field(default_factory=ConversationSlot)
    component: ConversationSlot = Field(default_factory=ConversationSlot)
    version: ConversationSlot = Field(default_factory=ConversationSlot)
    exception_type: ConversationSlot = Field(default_factory=ConversationSlot)

    ticket_references: list[str] = Field(default_factory=list)
    """Accumulated and deduplicated across every turn -- unlike the
    other slots, a ticket number is never "replaced," only added to: a
    real investigation routinely accumulates more than one real ticket
    reference over its life, none of them superseding an earlier one."""

    last_focus: ConversationFocus | None = None
    last_referenced_investigation_id: str | None = None
    last_referenced_tfs_id: int | None = None


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ReferenceState(str, Enum):
    """Whether a message's deictic reference ("this"/"that"/"that
    one"/...) could be safely resolved against the conversation's
    current state -- see ``ConversationStateEngine.resolve_reference``.
    Deliberately not an LLM coreference judgment: a fixed cue-phrase
    table plus a lookup against already-known state, same discipline as
    every other deterministic matcher in this codebase. There is no
    third "not applicable" member -- a message with no recognized cue
    simply has ``ChatMessage.reference_resolution = None`` (absence of
    evidence stays absence of evidence, the same rule Phase 2 already
    established for empty slots), never a manufactured "n/a" state."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"


class ReferenceResolution(BaseModel):
    """The result of checking one user message for a deictic reference
    against the conversation's state at the moment that message
    arrived (i.e. before that same message's own slot merge is
    applied)."""

    state: ReferenceState
    cue: str
    """The literal cue phrase that triggered resolution, e.g. "has this
    happened before" or "that one" -- always populated; this object is
    never constructed when no cue was found."""
    resolved_focus: ConversationFocus | None = None
    resolved_investigation_id: str | None = None
    resolved_tfs_id: int | None = None
    candidates: list[str] = Field(default_factory=list)
    """Human-readable description of what made this AMBIGUOUS (e.g.
    ["a previously referenced local match", "a previously referenced
    TFS case"]) -- always non-empty when ``state == AMBIGUOUS``, always
    empty otherwise."""


class ChatSession(BaseModel):
    id: str = Field(default_factory=_new_id)
    investigation_id: str | None = None
    """Set when opened from inside an Investigation Workspace -- shares
    that investigation's real, already-known context (customer/
    technology) instead of re-deriving one from scratch. ``None`` for a
    standalone chat session -- both are real, supported paths (see
    ``ConversationStateEngine.create_session``)."""
    slots: ConversationSlots = Field(default_factory=ConversationSlots)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ChatMessage(BaseModel):
    id: str = Field(default_factory=_new_id)
    session_id: str
    sequence: int
    """1-based, strictly increasing per session -- the real ordering
    key (assigned by the repository, never the caller's clock), so
    chronological reconstruction never depends on timestamp
    resolution."""
    role: MessageRole
    content: str
    parsed_query: ParsedQuery | None = None
    """Populated for user messages only -- the real, unmodified Phase 2
    ``QueryUnderstandingEngine.parse()`` output for this exact turn,
    persisted verbatim (never re-derived, summarized, or copied into a
    second shape) so a caller can always see exactly what was extracted
    from this message, independent of whether it was adopted into
    ``ConversationSlots``. Never populated for assistant turns -- an
    assistant message has no text to run Query Understanding against."""
    reference_resolution: ReferenceResolution | None = None
    """Populated for user messages that contain a recognized deictic
    cue -- ``None`` when no such cue was present (the common case),
    never a placeholder "not applicable" object."""
    created_at: datetime = Field(default_factory=_utcnow)


# =============================================================================
# ChatResponse (2026-08-14, Phase 4 -- Chat Orchestrator). Deliberately NOT
# a rendered string: every field below is either a real, unmodified object
# RecommendationEngine/StructuredResolutionEngine already produced (never
# recomputed, never duplicated), or a small piece of orchestration-only
# state (ambiguity/follow-up) that has no existing home elsewhere. See
# ChatOrchestrator (app/engines/chat/orchestrator.py) for how this is
# assembled -- composition only, no new matching/retrieval/ranking logic.
# =============================================================================


class ChatAmbiguityKind(str, Enum):
    """Why (if at all) the orchestrator stopped short of running
    retrieval and asked a clarifying question instead -- see
    ChatOrchestrator's module docstring for the exact, deterministic
    rule for each. Never an LLM judgment call."""

    NONE = "none"
    REFERENCE_AMBIGUOUS = "reference_ambiguous"
    """This message's own deictic reference (Phase 3's
    ReferenceResolution) had more than one plausible prior record and
    was never guessed."""
    SLOT_AMBIGUOUS = "slot_ambiguous"
    """A Customer/Technology extraction in this message tied between
    real, distinct candidates, with no already-established value on the
    conversation to fall back on -- the RF-Mesh-vs-RF-Mesh-IP
    disambiguation case from the approved architecture (§12)."""


class ChatAmbiguity(BaseModel):
    kind: ChatAmbiguityKind = ChatAmbiguityKind.NONE
    slot_name: str | None = None
    """Which dimension was ambiguous -- only set when
    kind == SLOT_AMBIGUOUS."""
    candidates: list[str] = Field(default_factory=list)
    """Human-readable description of the tied candidates -- always
    non-empty when ``kind != NONE``, always empty otherwise. Sourced
    directly from ``ExtractedSlot.ambiguous_candidates`` or
    ``ReferenceResolution.candidates`` -- never invented."""


class EnhancementStatus(str, Enum):
    """The lifecycle of one asynchronous LLM enhancement job (Chat
    Assistant Phase 37) -- see
    ``app.engines.chat.enhancement.ChatEnhancementService`` for the
    full state machine and why each terminal state exists."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"


class ChatEnhancementRef(BaseModel):
    """A pointer to an in-flight or finished LLM enhancement job,
    attached to ``ChatResponse.enhancement`` -- never the enhanced
    answer text itself (that is only ever returned by polling
    ``GET /chat/enhancements/{job_id}``, and only once it has passed
    every existing safety validator; see ``ChatEnhancementJob``).
    ``None`` on ``ChatResponse`` whenever no enhancement was scheduled
    -- always the case while ``Settings.llm_enabled`` or
    ``Settings.llm_async_enabled`` is ``False``, which is every
    configuration this project has run to date."""

    job_id: str
    status: EnhancementStatus


class ChatResponse(BaseModel):
    """The Chat Orchestrator's structured answer to one user message.
    Deliberately not a single string -- the UI renders ``answer_text``
    and every evidence field below independently, exactly as the
    Investigation Workspace already renders ``InvestigationStrategy``.

    Field-to-source map (nothing here is a second copy of anything --
    every non-orchestration field is the exact same object
    RecommendationEngine/StructuredResolutionEngine already computed):

    - ``resolution provenance`` / ``supporting evidence`` / ``validation
      steps`` -- all already inside ``structured_resolution``
      (``.confidence``/``.confidence_rationale``, ``.root_cause_evidence``
      + each ``resolution_candidates[].evidence``, ``.validation_steps``
      respectively) -- not flattened out to a second location.
    - ``recommended logs`` -- ``recommended_logs``
      (``InvestigationStrategy.ordered_log_collection``, unchanged).
    - ``suggested SQL`` -- ``suggested_sql``
      (``InvestigationStrategy.suggested_sql``, unchanged).
    - ``historical matches`` -- ``historical_investigations``/
      ``known_bugs``/``documentation`` (``InvestigationStrategy``'s own
      fields, unchanged).
    - ``TFS/Wiki matches`` -- ``tfs_matches``/``wiki_matches``
      (``InvestigationStrategy``'s own fields, unchanged, including
      ``available=False`` when a connector is unreachable)."""

    answer_text: str
    """Deterministic template output (see ChatOrchestrator's answer
    composer) -- never stronger confidence language than
    ``structured_resolution.confidence``/``resolution_provenance``
    allows. Never independently fabricated: every factual claim in this
    text is quoted or directly referenced from ``structured_resolution``
    (or, when that isn't available, ``RecommendedSolution``)."""
    answer_kind: str | None = None
    """``"knowledge"`` (Chat Knowledge-Synthesis feature) when
    ``answer_text`` came from ``ChatOrchestrator.
    _compose_knowledge_synthesis`` (a documentation/historical/known-bug/
    TFS/Wiki citation answer for an informational or historical
    question, used only when the investigation-confidence tier is
    POSSIBLE/UNKNOWN and no single strong resolution exists). ``"log_
    analysis"`` (Chat + Log Intelligence integration) when it came from
    ``ChatOrchestrator._compose_log_analysis_answer`` (a real, uploaded-
    log-derived timeline/errors/identifiers/correlation answer -- fires
    regardless of tier, since a timeline question is a fundamentally
    different question type than a root-cause question). ``None`` in
    every other case -- unchanged, tier-based boilerplate. Never changes
    ``resolution_provenance``'s own semantics: that field still always
    reflects the real, underlying investigation-confidence tier
    RecommendationEngine computed; this field only tells a caller (the
    UI) when NOT to present that tier as if it were a claim about this
    particular answer's own certainty (see Chat Knowledge-Synthesis
    feature report, Step 18)."""
    intent: QueryIntent
    active_context: RetrievalContext
    parsed_query: ParsedQuery
    ambiguity: ChatAmbiguity = Field(default_factory=ChatAmbiguity)
    follow_up_question: str | None = None
    """Set only when ``ambiguity.kind != NONE`` or the underlying
    provenance tier is UNKNOWN with a real next step to suggest --
    never a generic "can you clarify?" with no real reason."""

    structured_resolution: StructuredResolution | None = None
    resolution_provenance: ResolutionProvenance | None = None
    """Convenience mirror of ``structured_resolution.confidence`` (same
    value, not recomputed) -- ``None`` only when ``structured_resolution``
    itself is ``None`` (a build with the relationship engine unwired)."""

    historical_investigations: list[KnowledgeMatch] = Field(default_factory=list)
    known_bugs: list[KnowledgeMatch] = Field(default_factory=list)
    documentation: list[KnowledgeMatch] = Field(default_factory=list)
    recommended_logs: list[RecommendedLogCollectionItem] = Field(default_factory=list)
    suggested_sql: list[SuggestedSqlItem] = Field(default_factory=list)
    tfs_matches: ExternalKnowledgeResult | None = None
    wiki_matches: ExternalKnowledgeResult | None = None

    investigation_id: str | None = None
    """The REAL investigation this turn was scoped to, when the chat
    session is investigation-scoped -- never a synthesized/throwaway
    session id (a standalone chat's synthesized InvestigationSession is
    never persisted and never given an id a caller could look up)."""

    enhancement: ChatEnhancementRef | None = None
    """Chat Assistant Phase 37 -- set only when an LLM enhancement job
    was actually scheduled (``Settings.llm_enabled`` AND
    ``Settings.llm_async_enabled`` both True). ``answer_text`` above is
    ALWAYS the complete, valid, deterministic-or-already-LLM-validated
    answer regardless of this field -- ``enhancement`` never means "the
    real answer is still coming"; it means "a possibly-better phrasing
    may become available later via ``GET /chat/enhancements/{job_id}``,
    but what you already have is already correct.\""""
