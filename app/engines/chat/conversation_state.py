"""ConversationStateEngine -- the deterministic, zero-LLM conversation-
state foundation Phase 4's (not-yet-built) Chat Orchestrator will
consume (2026-08-14, Phase 3 -- RESOLVEIQ_CHAT_AND_RESOLUTION_
ARCHITECTURE.md §11/§12/§13; see app.domain.chat's module docstring
for how this reconciles against §11's original, pre-Phase-2 sketch).

Phase 3 scope boundary (explicit, per instruction): this engine never
generates an answer, never calls RecommendationEngine, and never
decides what an assistant answer "centered on" -- record_assistant_turn's
focus/referenced-id parameters are supplied by the caller (Phase 4's
future Chat Orchestrator), not computed here. What this engine DOES do,
fully and testably today:

1. Create/load ChatSessions (optionally seeded from an existing
   InvestigationSession's already-known customer/technology, reusing
   the exact same exact-name-match discipline
   RecommendationEngine._resolve_retrieval_context already uses --
   never a new resolution rule).
2. Process one new user message through the EXISTING, unmodified
   QueryUnderstandingEngine (Phase 2) -- this engine never re-implements
   slot extraction or intent classification.
3. Merge the resulting ParsedQuery's slots into the session's persisted
   ConversationSlots under the rules in _merge_generic_slot/
   _merge_technology_slot below (existing-context-wins, generalized
   across an entire conversation instead of one call; hierarchy-aware
   for Technology; an explicit correction cue is the only thing that
   lets a same-turn conflicting mention replace an already-established
   value).
4. Detect a deictic reference ("has this happened before"/"was that
   confirmed"/"what about that one") via a fixed cue-phrase table and
   resolve it against the session's current last_focus/
   last_referenced_investigation_id/last_referenced_tfs_id, or return
   it as AMBIGUOUS when there is nothing safe to resolve it against --
   never a guess (see resolve_reference).
5. Persist every message, in order, via ChatRepository.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from app.domain.chat import (
    ChatMessage,
    ChatSession,
    ConversationSlot,
    ConversationSlots,
    MessageRole,
    ReferenceResolution,
    ReferenceState,
)
from app.domain.query_understanding import ExtractedSlot, ParsedQuery, SlotConfidence
from app.engines.knowledge.applicability import RetrievalContext
from app.engines.shared.hierarchy import most_specific
from app.engines.shared.text_matching import phrase_present

if TYPE_CHECKING:
    from app.engines.query_understanding.engine import QueryUnderstandingEngine
    from app.infrastructure.db.chat_repository import ChatRepository
    from app.infrastructure.db.lookup_repository import LookupRepository
    from app.infrastructure.db.repository import InvestigationRepository

_CONFIDENCE_RANK: dict[SlotConfidence, int] = {SlotConfidence.PARTIAL: 1, SlotConfidence.EXACT: 2}


@dataclass(frozen=True)
class _CorrectionCue:
    phrase: str


_CORRECTION_CUE_PATTERNS: tuple[_CorrectionCue, ...] = (
    _CorrectionCue("actually,"),
    _CorrectionCue("actually this"),
    _CorrectionCue("actually it's"),
    _CorrectionCue("actually it is"),
    _CorrectionCue("no, this is"),
    _CorrectionCue("no this is"),
    _CorrectionCue("i meant"),
    _CorrectionCue("correction:"),
    _CorrectionCue("instead this is"),
    _CorrectionCue("this is actually for"),
    _CorrectionCue("let me correct"),
    _CorrectionCue("to correct myself"),
)
"""Fixed, data-driven phrase table (same idiom as Query Understanding's
own _INTENT_PATTERNS) -- the ONLY mechanism that lets a same-turn
mention conflicting with an already-established ConversationSlot
replace it. Deliberately narrow and literal, not a sentiment/intent
classifier: distinguishing "Actually, this is for CLECO" (a correction
-- item 8 of the approved Phase 3 examples) from "Does this apply to
CLECO?" (a check against a hypothetical -- item 4) is exactly the kind
of natural-language-pragmatics judgment call this codebase has never
attempted to guess at with a heuristic, and does not start here. A
conflicting mention with NO recognized correction cue is never
silently merged into the active slot AND never silently discarded --
it stays fully visible on that turn's own ChatMessage.parsed_query,
exactly as extracted; only the ACTIVE slot stays unchanged. A
correction phrased in some other way this table doesn't recognize
falls through to that same conservative default (preserve established
context) rather than guessing it was a correction -- a known, explicit
limitation, not a silent gap; see this session's Phase 3 report for
the full list of what genuinely cannot be resolved without an LLM."""

_ENTITY_REFERENCE_CUES: tuple[str, ...] = ("that one", "this one", "the other one")
"""Cues that ask about a SPECIFIC prior record (a local match, a TFS
case) rather than a KIND of focus -- resolved only via
last_referenced_investigation_id/last_referenced_tfs_id, never via
last_focus. See resolve_reference."""

_FOCUS_REFERENCE_CUES: tuple[tuple[str, str], ...] = (
    ("has this happened before", "problem"),
    ("have we seen this before", "problem"),
    ("seen this before", "problem"),
    ("happened before", "problem"),
    ("does this apply to", "problem"),
    ("what was the resolution", "resolution"),
    ("what was the fix", "resolution"),
    ("what's the fix", "resolution"),
    ("was that confirmed", "resolution"),
    ("was that fix confirmed", "resolution"),
    ("is this confirmed", "resolution"),
    ("which logs", "logs"),
    ("what logs", "logs"),
    ("which sql", "sql"),
    ("what sql", "sql"),
)
"""Cues that ask about a KIND of focus (§11's ConversationFocus) --
resolved against last_focus, except "problem" (the current investigation
subject is always singular and always available once a conversation has
started, so it never needs a prior-record pointer to resolve safely)."""

_SIMILAR_CASES_QUALIFIER_RE = re.compile(r"\b(?:in|among)\s+(?:similar|other|past|prior|previous)\s+cases?\b", re.IGNORECASE)
"""Real-Corpus Answer Quality & Final Chat Hardening phase -- "What was
the resolution in similar cases?" is a real, required golden question
(a self-contained HISTORICAL_LOOKUP request about OTHER past cases),
but it also contains the literal substring "what was the resolution",
a ``_FOCUS_REFERENCE_CUES`` entry meant for a genuinely different
question -- "What was the resolution?" asking about THIS investigation's
own already-established focus. In a fresh session with nothing
established yet, that collision made ``resolve_reference`` return
AMBIGUOUS with the unhelpful, generic "nothing has been established
yet to resolve this against" candidate, which ``ChatOrchestrator``
then rendered as "I found multiple possible interpretations of that
reference" -- never even reaching the real, already-built
HISTORICAL_LOOKUP path (see ``app.engines.chat.knowledge_question.
HISTORICAL_PHRASES``'s own "in similar cases" entry). This regex is
the one, narrow signal that distinguishes the two: any focus cue
qualified by "in similar/other/past/prior cases" is asking about OTHER
investigations, not this one, and must never be treated as a
reference needing this conversation's own prior context."""


class ConversationStateEngine:
    """See module docstring."""

    def __init__(
        self,
        chat_repo: "ChatRepository",
        query_understanding: "QueryUnderstandingEngine",
        lookup_repo: "LookupRepository",
        investigation_repo: "InvestigationRepository | None" = None,
    ) -> None:
        self._repo = chat_repo
        self._query_understanding = query_understanding
        self._lookup = lookup_repo
        self._investigations = investigation_repo

    # --- Session lifecycle ------------------------------------------------

    def create_session(self, investigation_id: str | None = None) -> ChatSession:
        """Standalone (investigation_id=None) and investigation-scoped
        sessions are both real, supported paths (§10/§11). When scoped,
        seeds ConversationSlots.customer/technology from the real
        InvestigationSession -- the exact same exact-name-match
        resolution RecommendationEngine._resolve_retrieval_context
        already uses (never a new rule): a customer/technology name
        that doesn't match a real governed row is left unresolved
        (technology falls back to the free-text name with no id, same
        as RetrievalContext.technology_name itself does today), never
        guessed."""
        slots = ConversationSlots()
        if investigation_id is not None and self._investigations is not None:
            investigation = self._investigations.get(investigation_id)
            if investigation is not None:
                if investigation.customer and investigation.customer.strip():
                    customer = self._lookup.get_customer_by_name(investigation.customer.strip())
                    if customer is not None:
                        slots = slots.model_copy(
                            update={
                                "customer": ConversationSlot(
                                    value_id=customer.id,
                                    value_name=customer.name,
                                    confidence=SlotConfidence.EXACT,
                                    origin="stated",
                                )
                            }
                        )
                if investigation.technology and investigation.technology.strip():
                    technology = self._lookup.get_technology_by_name(investigation.technology.strip())
                    slots = slots.model_copy(
                        update={
                            "technology": ConversationSlot(
                                value_id=technology.id if technology else None,
                                value_name=technology.name if technology else investigation.technology.strip(),
                                confidence=SlotConfidence.EXACT,
                                origin="stated",
                            )
                        }
                    )
        session = ChatSession(investigation_id=investigation_id, slots=slots)
        self._repo.save_session(session)
        return session

    def get_session(self, session_id: str) -> ChatSession | None:
        return self._repo.get_session(session_id)

    def list_messages(self, session_id: str) -> list[ChatMessage]:
        return self._repo.list_messages(session_id)

    # --- Turns --------------------------------------------------------------

    def add_user_message(self, session_id: str, text: str) -> ChatMessage:
        session = self._repo.get_session(session_id)
        if session is None:
            raise ValueError(f"No such chat session: {session_id}")
        slots = session.slots

        reference_resolution = self.resolve_reference(text, slots)

        existing_context = self.to_retrieval_context(slots)
        parsed = self._query_understanding.parse(text, existing_context=existing_context)

        message_id = str(uuid.uuid4())
        correction_cue_present = self._correction_cue(text) is not None

        new_slots = ConversationSlots(
            customer=self._merge_generic_slot(slots.customer, parsed.customer, message_id, correction_cue_present),
            region=self._merge_generic_slot(slots.region, parsed.region, message_id, correction_cue_present),
            technology=self._merge_technology_slot(slots.technology, parsed.technology, message_id, correction_cue_present),
            product=self._merge_generic_slot(slots.product, parsed.product, message_id, correction_cue_present),
            component=self._merge_generic_slot(slots.component, parsed.component, message_id, correction_cue_present),
            version=self._merge_generic_slot(slots.version, parsed.version, message_id, correction_cue_present),
            exception_type=self._merge_generic_slot(
                slots.exception_type, parsed.exception_type, message_id, correction_cue_present
            ),
            ticket_references=sorted(set(slots.ticket_references) | set(parsed.ticket_references)),
            last_focus=slots.last_focus,
            last_referenced_investigation_id=slots.last_referenced_investigation_id,
            last_referenced_tfs_id=slots.last_referenced_tfs_id,
        )

        message = ChatMessage(
            id=message_id,
            session_id=session_id,
            sequence=self._repo.next_sequence(session_id),
            role=MessageRole.USER,
            content=text,
            parsed_query=parsed,
            reference_resolution=reference_resolution,
        )
        self._repo.save_message(message)

        session.slots = new_slots
        session.updated_at = datetime.now(timezone.utc)
        self._repo.save_session(session)
        return message

    def record_assistant_turn(
        self,
        session_id: str,
        content: str,
        *,
        focus: str | None = None,
        referenced_investigation_id: str | None = None,
        referenced_tfs_id: int | None = None,
    ) -> ChatMessage:
        """Persists an assistant turn and, when supplied, updates
        last_focus/last_referenced_*. Phase 3 never computes these
        itself -- this is the explicit hook Phase 4's Chat Orchestrator
        will call once it exists, once it actually knows what its own
        answer centered on. Phase 3's own tests call this directly with
        explicit values to prove the persistence/reference-resolution
        mechanism works end-to-end, simulating that future caller."""
        session = self._repo.get_session(session_id)
        if session is None:
            raise ValueError(f"No such chat session: {session_id}")

        message = ChatMessage(
            session_id=session_id,
            sequence=self._repo.next_sequence(session_id),
            role=MessageRole.ASSISTANT,
            content=content,
        )
        self._repo.save_message(message)

        if focus is not None or referenced_investigation_id is not None or referenced_tfs_id is not None:
            slots = session.slots.model_copy(
                update={
                    "last_focus": focus if focus is not None else session.slots.last_focus,
                    "last_referenced_investigation_id": (
                        referenced_investigation_id
                        if referenced_investigation_id is not None
                        else session.slots.last_referenced_investigation_id
                    ),
                    "last_referenced_tfs_id": (
                        referenced_tfs_id if referenced_tfs_id is not None else session.slots.last_referenced_tfs_id
                    ),
                }
            )
            session.slots = slots
            session.updated_at = datetime.now(timezone.utc)
            self._repo.save_session(session)
        return message

    # --- RetrievalContext bridge (§13, pure mapping) -------------------------

    def to_retrieval_context(self, slots: ConversationSlots) -> RetrievalContext:
        """ConversationSlots -> RetrievalContext, a pure mapping -- no
        new ranking/resolution logic (§13: "EXTEND, trivial"). Product/
        Component/Version have no field on RetrievalContext (the same,
        already-documented Phase 2 limitation -- see
        app.domain.query_understanding.ParsedQuery.retrieval_context's
        own docstring); those three stay ConversationSlots-only here
        too, for the identical reason."""
        return RetrievalContext(
            customer_id=slots.customer.value_id,
            customer_name=slots.customer.value_name,
            region_id=slots.region.value_id,
            region_name=slots.region.value_name,
            technology_name=slots.technology.value_name,
        )

    # --- Slot merge ---------------------------------------------------------

    def _merge_generic_slot(
        self,
        active: ConversationSlot,
        extracted: ExtractedSlot | None,
        message_id: str,
        correction_cue_present: bool,
    ) -> ConversationSlot:
        """Existing-context-wins, generalized across an entire
        conversation (Phase 2 already applies this within one call via
        QueryUnderstandingEngine's own existing_context parameter; this
        is the same rule applied turn over turn). An AMBIGUOUS
        extraction never reaches the active slot -- it stays visible
        only on this turn's own ParsedQuery. A conflicting EXACT/PARTIAL
        extraction only replaces the active value when a recognized
        correction cue is present in this turn's text; otherwise the
        active value carries forward unchanged (origin relabeled
        "carried_forward", value/set_by_message_id untouched) and the
        conflicting mention is never silently merged into it."""
        carried = active.model_copy(update={"origin": "carried_forward"}) if active.value_id is not None else active

        if extracted is None or extracted.confidence == SlotConfidence.AMBIGUOUS:
            return carried

        if active.value_id is None:
            return ConversationSlot(
                value_id=extracted.value_id,
                value_name=extracted.value_name,
                confidence=extracted.confidence,
                origin="stated",
                set_by_message_id=message_id,
            )

        if extracted.value_id == active.value_id:
            stronger = (
                extracted.confidence
                if _CONFIDENCE_RANK[extracted.confidence] > _CONFIDENCE_RANK[active.confidence]
                else active.confidence
            )
            return active.model_copy(update={"confidence": stronger, "origin": "stated", "set_by_message_id": message_id})

        if correction_cue_present:
            return ConversationSlot(
                value_id=extracted.value_id,
                value_name=extracted.value_name,
                confidence=extracted.confidence,
                origin="stated",
                set_by_message_id=message_id,
            )

        return carried

    def _merge_technology_slot(
        self,
        active: ConversationSlot,
        extracted: ExtractedSlot | None,
        message_id: str,
        correction_cue_present: bool,
    ) -> ConversationSlot:
        """Wraps _merge_generic_slot with one Technology-specific rule:
        a same-turn mention of a real ANCESTOR of the already-active,
        more specific technology (e.g. "RF Mesh" mentioned while "RF
        Mesh IP" is active) is not new evidence of a different
        technology and never demotes the active slot -- reuses
        app.engines.shared.hierarchy.most_specific, the exact same
        mechanism classification.py/_match_single_technology already
        use for this identical question, never a new hierarchy rule.
        The ancestor mention is still fully visible on this turn's own
        ParsedQuery.technology -- nothing is hidden, it just doesn't
        win."""
        if (
            extracted is not None
            and extracted.confidence != SlotConfidence.AMBIGUOUS
            and active.value_id is not None
            and extracted.value_id is not None
            and extracted.value_id != active.value_id
            and not correction_cue_present
            and self._is_ancestor_technology(extracted.value_id, active.value_id)
        ):
            return active.model_copy(update={"origin": "carried_forward"})
        return self._merge_generic_slot(active, extracted, message_id, correction_cue_present)

    def _is_ancestor_technology(self, candidate_id: str, of_id: str) -> bool:
        technologies = self._lookup.list_technologies()
        parent_of = {t.id: t.parent_technology_id for t in technologies}
        survivors = most_specific({candidate_id, of_id}, parent_of)
        return survivors == {of_id}

    def _correction_cue(self, text: str) -> str | None:
        for cue in _CORRECTION_CUE_PATTERNS:
            if phrase_present(cue.phrase, text):
                return cue.phrase
        return None

    # --- Reference resolution -----------------------------------------------

    def resolve_reference(self, text: str, slots: ConversationSlots) -> ReferenceResolution | None:
        """Checks ``text`` for a recognized deictic cue and resolves it
        against ``slots`` as they stood BEFORE this same message's own
        slot merge -- a message can only refer back to what the
        conversation already knew, never to itself. Returns ``None``
        (no ``ReferenceResolution`` object at all) when no cue is
        recognized -- the same "absence stays absence" discipline Phase
        2 established for empty slots. Never guesses: an
        ``_ENTITY_REFERENCE_CUES`` match resolves only when exactly one
        of (last_referenced_investigation_id, last_referenced_tfs_id) is
        set; a ``_FOCUS_REFERENCE_CUES`` match (other than "problem",
        always available) resolves only when the conversation has
        actually recorded that focus or a referenced record. Anything
        else is returned AMBIGUOUS with a human-readable reason, never
        silently dropped and never fabricated."""
        for cue in _ENTITY_REFERENCE_CUES:
            if phrase_present(cue, text):
                available = [
                    name
                    for name, value in (
                        ("a previously referenced local investigation", slots.last_referenced_investigation_id),
                        ("a previously referenced TFS case", slots.last_referenced_tfs_id),
                    )
                    if value is not None
                ]
                if len(available) == 1:
                    return ReferenceResolution(
                        state=ReferenceState.RESOLVED,
                        cue=cue,
                        resolved_investigation_id=slots.last_referenced_investigation_id,
                        resolved_tfs_id=slots.last_referenced_tfs_id,
                    )
                return ReferenceResolution(
                    state=ReferenceState.AMBIGUOUS,
                    cue=cue,
                    candidates=available or ["nothing has been referenced yet in this conversation"],
                )

        for phrase, focus in _FOCUS_REFERENCE_CUES:
            if phrase_present(phrase, text):
                if _SIMILAR_CASES_QUALIFIER_RE.search(text):
                    # "...in similar/other/past/prior cases" -- a real,
                    # self-contained historical-lookup question about
                    # OTHER investigations, not a reference to this
                    # one's own established focus. See
                    # _SIMILAR_CASES_QUALIFIER_RE's own docstring.
                    continue
                if focus == "problem":
                    return ReferenceResolution(state=ReferenceState.RESOLVED, cue=phrase, resolved_focus="problem")
                if slots.last_focus == focus or slots.last_referenced_investigation_id or slots.last_referenced_tfs_id:
                    return ReferenceResolution(
                        state=ReferenceState.RESOLVED,
                        cue=phrase,
                        resolved_focus=focus,
                        resolved_investigation_id=slots.last_referenced_investigation_id,
                        resolved_tfs_id=slots.last_referenced_tfs_id,
                    )
                return ReferenceResolution(
                    state=ReferenceState.AMBIGUOUS,
                    cue=phrase,
                    candidates=["nothing has been established yet to resolve this against"],
                )

        return None
