"""ChatOrchestrator -- the single deterministic coordinator between a
chat message and every existing engine that already knows how to answer
one (2026-08-14, Phase 4 -- RESOLVEIQ_CHAT_AND_RESOLUTION_ARCHITECTURE.md
§10/§12/§17/§19/§26/§33; see this module's own inline notes for the
points where that document's original sketch predates Phase 2/3's real
implementations, reconciled here in favor of the real code).

Zero LLM, zero new retrieval/matching/ranking logic:

    user message -> ConversationStateEngine.add_user_message (Phase 3,
        which itself calls the unmodified Phase 2 QueryUnderstandingEngine)
    -> ambiguity check (a small, new, deterministic rule -- see
        _detect_ambiguity)
    -> RecommendationEngine.generate() (Phase 0-2C, completely unchanged --
        see _resolve_investigation_for_retrieval for the "Standalone
        Question" synthesis §10 already designed)
    -> deterministic answer composition from the resulting
        StructuredResolution/InvestigationStrategy (never a new evidence
        source, never independently invented text)
    -> ConversationStateEngine.record_assistant_turn (Phase 3)
    -> ChatResponse (app.domain.chat)

Trust-language discipline (§4, the one rule this module must never
violate): the answer text's confidence language is driven ENTIRELY by
``StructuredResolution.confidence`` (the same, unchanged four-tier
``ResolutionProvenance`` from Resolution Provenance/Phase 0/Phase 1 --
never re-derived, never loosened). CONFIRMED is the only tier allowed
"confirmed" language; LIKELY/POSSIBLE/UNKNOWN templates are hardcoded to
never use it, regardless of how high a similarity score might be
underneath -- the exact same "similarity alone can never produce
Confirmed" guarantee already enforced inside
``RecommendationEngine._resolve_provenance_tier``, now also enforced at
the point where that tier becomes user-facing prose.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.domain.chat import (
    ChatAmbiguity,
    ChatAmbiguityKind,
    ChatMessage,
    ChatResponse,
    ChatSession,
    ConversationSlots,
    MessageRole,
    ReferenceState,
)
from app.domain.enums import EvidenceType
from app.domain.evidence import Evidence
from app.domain.investigation import InvestigationSession
from app.domain.provenance import EvidenceKind, ResolutionProvenance
from app.domain.query_understanding import SlotConfidence
from app.engines.chat.scope_expansion import contains_unsupported_scope_expansion
from app.engines.chat.scope_question import split_out_scope_clause
from app.engines.chat.troubleshooting_question import split_out_troubleshooting_clause
from app.engines.investigation.engine import InvestigationNotFoundError
from app.engines.llm.prompt_builder import PromptBuilder, available_checks
from app.engines.llm.provider import LLMProviderError
from app.engines.log_intelligence.engine import LogIntelligenceEngine

if TYPE_CHECKING:
    from app.domain.log_flow import LogObservationSummary
    from app.domain.recommendation import InvestigationStrategy, RecommendedSolution
    from app.domain.structured_resolution import StructuredResolution
    from app.engines.chat.conversation_state import ConversationStateEngine
    from app.engines.investigation.engine import InvestigationEngine
    from app.engines.llm.provider import LLMProvider
    from app.engines.recommendation.engine import RecommendationEngine

logger = logging.getLogger(__name__)


def customer_scope_statement(structured: "StructuredResolution") -> str:
    """Chat Assistant Phase 30 -- the deterministic, ALWAYS-correct
    customer-impact-scope sentence, composed entirely outside the LLM
    and appended to every answer that has a real ``StructuredResolution``
    (see ``ChatOrchestrator._generate_answer``).

    Four straight phases (27, 28, 29) tried instead to make qwen2.5:3b
    safely PHRASE this fact from an in-prompt representation -- a prose
    header, a conditionally-omitted section, a compact status token,
    even a literal precomputed sentence with an explicit "copy this
    exactly, do not reason about scope" instruction. Every one measurably
    reduced fabrication but never eliminated it; the most literal attempt
    (Phase 30's own "precomputed answer, do not touch it" experiment,
    20 real calls) produced the WORST result of any variant tested across
    all four phases -- over 60% of responses either echoed the confidence
    tier ("Likely") as the scope answer or invented an explicit "Yes,
    other customers are affected", ignoring the precomputed text sitting
    right there in the prompt. That result closed the question: no
    in-prompt representation reliably prevents this model from
    re-deriving its own scope conclusion once it is generating the
    sentence that states it.

    The fix moves scope out of the LLM's responsibility entirely.
    ``PromptBuilder``'s rule 8 now has an explicit exception telling the
    model never to answer, mention, or speculate about customer-impact
    scope -- that sub-question is skipped by the LLM and answered here
    instead, by simple, deterministic string composition from the same
    real, already-governed ``ApplicabilitySummary.customer_names`` list
    every prior phase's guard used: empty means genuinely unknown, any
    non-empty list means exactly those names are known-affected and
    nothing further (never "only" those names, never "all customers" --
    a tag list of any length only ever establishes who IS known-affected,
    consistent with ``ApplicabilitySummary``'s own "not yet tagged, not
    'applies everywhere'" documented semantics)."""
    names = structured.applicability.customer_names
    if not names:
        return "Customer impact scope: not established by the supplied evidence -- whether this affects other customers is unknown."
    joined = ", ".join(names)
    verb = "is" if len(names) == 1 else "are"
    return (
        f"Customer impact scope: {joined} {verb} known to be affected. "
        "Whether any other customer is also affected is not established."
    )


def no_evidence_backed_check_statement() -> str:
    """Chat Assistant Phase 32 -- the deterministic, ALWAYS-correct "no
    evidence-backed troubleshooting check" statement, composed entirely
    outside the LLM and appended to the answer whenever the question
    asked a troubleshooting question AND zero evidence-backed checks
    exist for it (see ``ChatOrchestrator._generate_answer``).

    Same architecture as ``customer_scope_statement`` above, for the
    same measured reason. Phase 25's prompt-only guard (``PromptBuilder``
    rule 9 + the AVAILABLE EVIDENCE-BACKED CHECKS field) was never
    revisited until Phase 32 found it fabricated a generic
    troubleshooting suggestion in 39/40 fresh real qwen2.5:3b calls
    (97.5%) on a fixture with zero checks -- five stronger prompt-only
    variants (240 more real calls) never dropped below 60% unsafe, and
    withholding the raw root-cause text made it WORSE (100% unsafe),
    proving the model fabricates a topically-plausible check purely
    from the general problem framing, not from any specific root-cause
    wording. A root-cause-association attack (50 more real calls, five
    unrelated root causes) confirmed 50/50 fabricated every time. Only
    removing the troubleshooting clause from the LLM's input entirely
    (see ``app.engines.chat.troubleshooting_question``) eliminated it in
    every condition tested, while a genuine-checks positive control (20
    real calls) confirmed real checks are still surfaced correctly,
    unmodified, when they exist -- so this statement is only ever
    appended in the zero-checks case; Rule 9 itself is untouched and
    still governs the common case where checks DO exist."""
    return "No evidence-backed troubleshooting check can be determined from the supplied evidence."


class ChatSessionNotFoundError(Exception):
    pass


class EmptyMessageError(ValueError):
    pass


class ChatOrchestrator:
    """See module docstring. Never implements its own matching or
    retrieval -- every method here either delegates outright to an
    existing engine or composes already-computed data into
    ``ChatResponse``."""

    def __init__(
        self,
        state_engine: "ConversationStateEngine",
        recommendation_engine: "RecommendationEngine",
        investigation_engine: "InvestigationEngine | None" = None,
        llm_provider: "LLMProvider | None" = None,
    ) -> None:
        self._state = state_engine
        self._recommend = recommendation_engine
        self._investigations = investigation_engine
        self._llm = llm_provider
        """None (default) means the LLM path is never attempted --
        _generate_answer() falls straight to the existing, unchanged
        _compose_answer() every time, byte-identical to pre-Phase-1
        behavior. Set via DI (app/api/dependencies.py) only when
        Settings.llm_enabled is True."""
        self._prompt_builder = PromptBuilder()
        """Stateless -- always constructed, never None, regardless of
        whether an LLM provider is wired."""

    # --- Session lifecycle (thin passthrough to ConversationStateEngine) ----

    def create_session(self, investigation_id: str | None = None) -> ChatSession:
        if investigation_id is not None and self._investigations is not None:
            self._investigations.get_investigation(investigation_id)  # raises InvestigationNotFoundError if missing
        return self._state.create_session(investigation_id)

    def get_session(self, session_id: str) -> ChatSession | None:
        return self._state.get_session(session_id)

    def list_messages(self, session_id: str) -> list[ChatMessage]:
        return self._state.list_messages(session_id)

    # --- The one real orchestration method -----------------------------------

    def handle_message(self, session_id: str, text: str) -> ChatResponse:
        if not text or not text.strip():
            raise EmptyMessageError("Chat message must not be empty.")
        if self._state.get_session(session_id) is None:
            raise ChatSessionNotFoundError(session_id)

        message = self._state.add_user_message(session_id, text)
        session = self._state.get_session(session_id)
        assert session is not None  # just saved by add_user_message above
        slots = session.slots
        parsed = message.parsed_query
        assert parsed is not None  # add_user_message always populates this for user turns
        active_context = self._state.to_retrieval_context(slots)

        ambiguity = self._detect_ambiguity(message, slots)
        if ambiguity.kind != ChatAmbiguityKind.NONE:
            answer_text, follow_up = self._compose_ambiguous_response(ambiguity)
            self._state.record_assistant_turn(session_id, answer_text)
            return ChatResponse(
                answer_text=answer_text,
                intent=parsed.intent,
                active_context=active_context,
                parsed_query=parsed,
                ambiguity=ambiguity,
                follow_up_question=follow_up,
                investigation_id=session.investigation_id,
            )

        investigation_for_retrieval, response_investigation_id = self._resolve_investigation_for_retrieval(session)
        recommendation = self._recommend.generate(investigation_for_retrieval)
        strategy = recommendation.strategy
        log_observations = LogIntelligenceEngine.summarize_observations(investigation_for_retrieval.evidence)
        """Chat Assistant Phase 33 -- a deterministic, provenance-
        preserving summary of any LOG_FILE evidence already uploaded to
        this investigation (via the Investigation Workspace -- chat
        itself has no log-upload path of its own). ``None`` for every
        standalone (non-investigation-scoped) session, since the
        throwaway synthesized investigation never has LOG_FILE evidence
        -- see ``_resolve_investigation_for_retrieval``. Computed here,
        not inside ``RecommendationEngine.generate()``, deliberately:
        it never touches ranking/root-cause/resolution/confidence
        (Rule 31/context of this phase -- no duplicated or modified
        recommendation logic), it is purely an additional, optional
        fact block ``PromptBuilder`` may render alongside whatever
        ``RecommendationEngine`` already decided."""

        answer_text, follow_up = self._generate_answer(text, strategy, log_observations)
        focus, referenced_investigation_id, referenced_tfs_id = self._derive_focus(strategy)
        self._state.record_assistant_turn(
            session_id,
            answer_text,
            focus=focus,
            referenced_investigation_id=referenced_investigation_id,
            referenced_tfs_id=referenced_tfs_id,
        )

        structured = strategy.structured_resolution
        return ChatResponse(
            answer_text=answer_text,
            intent=parsed.intent,
            active_context=active_context,
            parsed_query=parsed,
            follow_up_question=follow_up,
            structured_resolution=structured,
            resolution_provenance=structured.confidence if structured is not None else None,
            historical_investigations=strategy.historical_investigations,
            known_bugs=strategy.known_bugs,
            documentation=strategy.documentation,
            recommended_logs=strategy.ordered_log_collection,
            suggested_sql=strategy.suggested_sql,
            tfs_matches=strategy.tfs_matches,
            wiki_matches=strategy.wiki_matches,
            investigation_id=response_investigation_id,
        )

    # --- Ambiguity (a small, new, deterministic rule -- see class docstring) -

    def _detect_ambiguity(self, message: ChatMessage, slots: ConversationSlots) -> ChatAmbiguity:
        """Two, and only two, deterministic reasons to stop before ever
        calling RecommendationEngine:

        1. This message's own deictic reference (Phase 3) had more than
           one plausible prior record -- searching on "that one" would
           be meaningless, and Phase 3 already refused to guess.
        2. A Technology or Customer extraction in this message tied
           between real, distinct governed candidates AND the
           conversation has nothing already established for that same
           dimension to fall back on -- the RF-Mesh-vs-RF-Mesh-IP
           disambiguation case the approved architecture (§12) names
           explicitly. Deliberately restricted to Technology/Customer
           (the two dimensions that actually drive ``RetrievalContext``)
           and deliberately restricted to "nothing already established"
           -- an incidental ambiguous mention elsewhere in a message,
           while a real customer/technology is already active, must not
           interrupt an otherwise-answerable question (Phase 3's own
           merge rules already keep the active slot untouched by an
           ambiguous extraction; this only decides whether to ask about
           it before running retrieval or just proceed with what's
           already known).

        Deliberately NOT triggered by "the message text is short/vague"
        -- inventing a vagueness threshold would be exactly the kind of
        unguided heuristic this codebase avoids. A genuinely
        underspecified question with no ambiguity signal still reaches
        RecommendationEngine and gets an honest UNKNOWN-tier answer
        (§4.C) -- which already says "I don't have enough evidence,"
        the same outcome without a fabricated pre-check."""
        if message.reference_resolution is not None and message.reference_resolution.state == ReferenceState.AMBIGUOUS:
            return ChatAmbiguity(kind=ChatAmbiguityKind.REFERENCE_AMBIGUOUS, candidates=message.reference_resolution.candidates)

        parsed = message.parsed_query
        for slot_name, extracted, active in (
            ("technology", parsed.technology, slots.technology),
            ("customer", parsed.customer, slots.customer),
        ):
            if extracted is not None and extracted.confidence == SlotConfidence.AMBIGUOUS and active.value_id is None:
                return ChatAmbiguity(
                    kind=ChatAmbiguityKind.SLOT_AMBIGUOUS, slot_name=slot_name, candidates=extracted.ambiguous_candidates
                )
        return ChatAmbiguity()

    def _compose_ambiguous_response(self, ambiguity: ChatAmbiguity) -> tuple[str, str]:
        candidates = ", ".join(ambiguity.candidates) if ambiguity.candidates else "more than one real possibility"
        if ambiguity.kind == ChatAmbiguityKind.REFERENCE_AMBIGUOUS:
            answer = "I found multiple possible interpretations of that reference. Please clarify which one you mean."
        else:
            answer = f"I found multiple possible {ambiguity.slot_name} matches for this question. Please clarify which one you mean."
        follow_up = f"Which of the following did you mean: {candidates}?"
        return answer, follow_up

    # --- Standalone vs. investigation-scoped retrieval (§10) ------------------

    def _resolve_investigation_for_retrieval(self, session: ChatSession) -> tuple[InvestigationSession, str | None]:
        """Investigation-scoped: reuse the REAL investigation, exactly
        as ``RecommendationEngine.generate()`` is already called from
        the Investigation Workspace -- no new matching/retrieval, and
        the chat message's text is never written back into that real
        investigation's evidence (no duplicate investigation, no silent
        mutation of the real one).

        Standalone: synthesize a throwaway ``InvestigationSession``
        (§10's approved option A) from every real user message in this
        conversation so far (not just the latest one -- a follow-up
        like "Has this happened before?" must still search against the
        substantive content from turn 1), seeded with whatever
        Customer/Technology the conversation has already resolved.
        Never persisted -- ``generate()`` doesn't require its input to
        be saved, and this session is discarded the moment this call
        returns."""
        if session.investigation_id is not None and self._investigations is not None:
            try:
                real = self._investigations.get_investigation(session.investigation_id)
                return real, session.investigation_id
            except InvestigationNotFoundError:
                pass  # fall through to standalone synthesis rather than error mid-answer

        messages = self._state.list_messages(session.id)
        user_texts = [m.content for m in messages if m.role == MessageRole.USER]
        title = user_texts[0] if user_texts else ""
        synthesized = InvestigationSession(
            title=title,
            customer=session.slots.customer.value_name,
            technology=session.slots.technology.value_name,
        )
        if user_texts:
            synthesized.add_evidence(
                Evidence(
                    investigation_id=synthesized.id,
                    evidence_type=EvidenceType.TASK_DESCRIPTION,
                    source="chat",
                    title="Conversation",
                    raw_content="\n".join(user_texts),
                )
            )
        return synthesized, None

    # --- Answer generation (Chat Assistant Phase 1 -- Qwen 4B/Ollama) ----------
    # _compose_answer() below (and _compose_answer_without_structured_resolution)
    # are UNCHANGED -- they remain the deterministic fallback for every case
    # the LLM path doesn't/can't handle. _generate_answer() is the new single
    # entry point handle_message() calls instead of _compose_answer() directly;
    # it only ever *adds* a first attempt in front of the existing behavior,
    # never replaces or alters it.

    def _generate_answer(
        self,
        question: str,
        strategy: "InvestigationStrategy",
        log_observations: "LogObservationSummary | None" = None,
    ) -> tuple[str, str | None]:
        """Tries LLM generation first when a provider is wired, enabled,
        and there's a real StructuredResolution to ground it in; falls
        back to the existing, untouched _compose_answer() in every
        other case -- provider absent/disabled, no structured
        resolution, or a real LLMProviderError. The deterministic path
        is always available and is never itself modified by this
        method.

        Chat Assistant Phase 31 -- BEFORE anything else, the question is
        split into its scope and non-scope parts (see
        app.engines.chat.scope_question). Four straight phases (27-30)
        tried to control customer-impact-scope fabrication by changing
        what the LLM was TOLD about scope -- a fact to phrase, a field
        to preserve, an instruction to skip it entirely -- and every one
        still left the model able to see and react to the literal scope
        question text, which it kept answering regardless of
        instruction (Phase 30's own "don't answer this" clause was
        measurably WORSE than doing nothing: 70%/55% unsafe vs. Phase
        29's ~10%/~30%). Phase 31's real-call validation of removing the
        scope clause from the LLM's input ENTIRELY (not just telling it
        to ignore something still visible) found 0/20 unsafe -- the
        model cannot fabricate an answer to a question it was never
        asked. The scope statement is now appended ONLY when the
        original question actually contained a scope clause (unlike
        Phase 30's unconditional append), matching what the user
        actually asked.

        Chat Assistant Phase 32 -- the same removal is now also applied
        to a troubleshooting-request clause (see
        app.engines.chat.troubleshooting_question), but ONLY when zero
        evidence-backed checks exist (``available_checks(structured)``
        is empty). Phase 25's prompt-only Rule 9 guard was never
        revisited until Phase 32 found it fabricated a generic
        troubleshooting suggestion in 39/40 fresh real qwen2.5:3b calls
        (97.5%) on a zero-checks fixture; five stronger prompt-only
        variants (240 more real calls) never dropped below 60% unsafe,
        including one that made things WORSE by withholding the raw
        root-cause text (100% unsafe -- proving the fabrication is a
        general compulsion to answer, not a reaction to specific
        wording). Only removing the clause entirely eliminated it (0
        real-call fabrications across every condition tested), while a
        genuine-checks positive control (20 real calls) confirmed real
        checks are still phrased correctly, unmodified, when they exist
        -- so the clause is deliberately left untouched, and Rule 9 left
        governing the question normally, whenever a real check exists.
        A multi-part attack (Phase 32 Step 9) also proved a naive
        "treat the whole question as troubleshooting-only" gate
        incorrectly bare-collapses questions like "Has this happened
        before, and what should I check first?" -- discarding the first
        part entirely, the exact Finding A regression this project has
        guarded against since Phase 26. Removing only the matched
        CLAUSE (mirroring ``split_out_scope_clause`` exactly, not a
        whole-question check) is what avoids that: the remaining
        question still reaches the LLM and Rule 8 still governs it.

        Chat Assistant Phase 33 -- ``log_observations``, when not
        ``None``, is passed straight through to ``PromptBuilder`` (see
        ``_generate_llm_answer``); it never affects the scope/
        troubleshooting-clause logic above, since it is a supplementary
        data section, not a fact that changes which sub-questions the
        LLM is permitted to see."""
        sanitized_question, had_scope = split_out_scope_clause(question)
        structured = strategy.structured_resolution

        checks_available = bool(available_checks(structured)) if structured is not None else False
        had_troubleshooting = False
        if not checks_available:
            sanitized_question, had_troubleshooting = split_out_troubleshooting_clause(sanitized_question)

        if (had_scope or had_troubleshooting) and not sanitized_question:
            # Nothing non-scope/non-troubleshooting is left to ask the
            # LLM at all.
            answer_text, follow_up = self._compose_answer(strategy)
            answer_text = self._append_deterministic_statements(answer_text, structured, had_scope, had_troubleshooting)
            return answer_text, follow_up

        if self._llm is not None and structured is not None and self._llm.is_configured():
            try:
                answer_text = self._generate_llm_answer(sanitized_question, structured, log_observations)
                if contains_unsupported_scope_expansion(answer_text):
                    # Chat Assistant Phase 35B -- deterministic post-
                    # generation gate, checked on the RAW LLM text
                    # before any deterministic statement is appended
                    # (see contains_unsupported_scope_expansion's own
                    # docstring for why the ordering matters). Phase 35's
                    # real-call validation found qwen2.5:3b will, on a
                    # normal (non-scope) multi-part question, sometimes
                    # spontaneously volunteer an unsupported claim like
                    # "occurred before with other customers in the APAC
                    # region" -- with no scope question anywhere in the
                    # input for Phase 31's clause-removal mechanism to
                    # act on, since none was asked. There is no clause to
                    # remove here; the claim appears inside the answer to
                    # a real, necessary question. Rejecting the whole LLM
                    # answer and falling back to the existing, unchanged
                    # deterministic path -- never rewriting it into a new
                    # claim -- is the same "prevent, don't just instruct"
                    # principle already proven necessary for the explicit-
                    # question version of this exact problem (Phases
                    # 27-30's four straight failed prompt-only attempts).
                    logger.warning(
                        "LLM answer contained an unsupported customer-scope expansion claim; "
                        "falling back to deterministic answer."
                    )
                else:
                    answer_text = self._append_deterministic_statements(answer_text, structured, had_scope, had_troubleshooting)
                    return answer_text, None
            except LLMProviderError as exc:
                # Environmental/provider failure only (connection, timeout,
                # HTTP error, malformed/empty response) -- never a bare
                # `except Exception`, so a real bug in PromptBuilder or here
                # still surfaces instead of being silently swallowed. Never
                # exposed to the end user -- the deterministic fallback below
                # returns a normal, successful answer, exactly like every
                # other graceful-degradation path in this codebase
                # (ExternalKnowledgeService, _reconcile_orphaned_columns).
                logger.warning("LLM generation failed, falling back to deterministic answer: %s", exc)
        answer_text, follow_up = self._compose_answer(strategy)
        answer_text = self._append_deterministic_statements(answer_text, structured, had_scope, had_troubleshooting)
        return answer_text, follow_up

    @staticmethod
    def _append_scope_statement(answer_text: str, structured: "StructuredResolution") -> str:
        return f"{answer_text}\n\n{customer_scope_statement(structured)}"

    @staticmethod
    def _append_no_checks_statement(answer_text: str) -> str:
        return f"{answer_text}\n\n{no_evidence_backed_check_statement()}"

    @classmethod
    def _append_deterministic_statements(
        cls,
        answer_text: str,
        structured: "StructuredResolution | None",
        had_scope: bool,
        had_troubleshooting: bool,
    ) -> str:
        """Chat Assistant Phase 32 -- appends whichever deterministic
        statements the original question actually asked for, in the
        order the two mechanisms were introduced (scope, then
        troubleshooting). Both are no-ops when their corresponding
        clause was never present, matching each statement's own
        "only when actually asked" contract."""
        if structured is None:
            return answer_text
        if had_scope:
            answer_text = cls._append_scope_statement(answer_text, structured)
        if had_troubleshooting:
            answer_text = cls._append_no_checks_statement(answer_text)
        return answer_text

    def _generate_llm_answer(
        self,
        question: str,
        structured: "StructuredResolution",
        log_observations: "LogObservationSummary | None" = None,
    ) -> str:
        """Chat Assistant Phase 31 -- ``question`` here is already the
        sanitized (scope-clause-stripped, and, since Phase 32, also
        troubleshooting-clause-stripped when no checks exist) text;
        this method additionally strips the same clauses from
        ``structured.problem`` before building the prompt, closing a
        real leak Phase 31's own orchestrator-level test caught:
        ``problem`` is populated from ``investigation.title``, which in
        the standalone-question synthesis path (see
        ``_resolve_investigation_for_retrieval``) is the user's RAW
        first message, independent of the ``question`` argument -- so a
        scope (or troubleshooting) clause could reach the LLM through
        the PROBLEM section even with the USER QUESTION section
        correctly sanitized. Only a prompt-construction-time COPY is
        sanitized (``model_copy``) -- the real ``structured`` object,
        and everything upstream of it (retrieval, matching, the
        ``ChatResponse`` returned to the caller), is never touched,
        preserving this module's "zero new retrieval/matching logic"
        guarantee.

        Chat Assistant Phase 33 -- ``log_observations`` (already
        computed deterministically in ``handle_message``, never
        recomputed or reinterpreted here) is passed straight through to
        ``PromptBuilder.build()``, which renders it as its own labeled,
        untrusted-data section governed by rule 10 -- this method makes
        no decision about it at all beyond forwarding it unchanged."""
        sanitized_problem, _ = split_out_scope_clause(structured.problem)
        if not available_checks(structured):
            sanitized_problem, _ = split_out_troubleshooting_clause(sanitized_problem)
        prompt_structured = structured.model_copy(update={"problem": sanitized_problem or structured.problem})
        system_prompt, user_prompt = self._prompt_builder.build(question, prompt_structured, log_observations)
        return self._llm.generate(user_prompt, system_prompt=system_prompt)

    # --- Deterministic answer composition (§4) ---------------------------------

    def _compose_answer(self, strategy: "InvestigationStrategy") -> tuple[str, str | None]:
        structured = strategy.structured_resolution
        if structured is None:
            return self._compose_answer_without_structured_resolution(strategy)

        tier = structured.confidence
        primary = next((c for c in structured.resolution_candidates if c.is_primary), None)
        subject = structured.root_cause or (primary.text if primary is not None else None)

        if tier == ResolutionProvenance.CONFIRMED:
            text = (
                f"Based on the available evidence, this issue is confirmed to be related to: {subject}."
                if subject
                else "Based on the available evidence, this has been confirmed."
            )
            if structured.confidence_rationale:
                text += f" ({structured.confidence_rationale})"
            if primary is not None:
                text += f" Recommended resolution: {primary.text}"
            return text, None

        if tier == ResolutionProvenance.LIKELY:
            text = (
                f"Based on the available evidence, this issue is likely related to: {subject}."
                if subject
                else "Based on the available evidence, a likely cause has been identified."
            )
            # Deliberately never the word "confirm"/"confirmed" anywhere in
            # this branch -- LIKELY must not use confirmed-equivalent
            # language even to negate it (§4's explicit rule).
            text += " This has not been independently verified."
            if primary is not None:
                text += f" A likely resolution, not yet independently verified: {primary.text}"
            return text, None

        if tier == ResolutionProvenance.POSSIBLE:
            text = f"A possible explanation is: {subject}." if subject else "A possible explanation may exist, but the evidence found is limited."
            text += (
                " Evidence is not yet sufficient to verify this -- treat it as a hypothesis to check, not a"
                " resolution to act on."
            )
            return text, None

        # UNKNOWN
        return "I don't have enough evidence to determine the cause.", self._unknown_follow_up(strategy)

    def _compose_answer_without_structured_resolution(self, strategy: "InvestigationStrategy") -> tuple[str, str | None]:
        """Defensive fallback for a build where the relationship engine
        (and therefore ``structured_resolution``) isn't wired -- not the
        normal production path (DI always wires it), but graceful
        degradation, same contract as every other optional dependency
        in this codebase. ``RecommendedSolution.confidence`` ("High"/
        "Medium"/"Low"/"Insufficient") is a different, coarser scale
        than ``ResolutionProvenance`` and must never be presented as if
        it were one of the four tiers -- this path always uses the same
        hedged, POSSIBLE-equivalent language regardless, the safe
        default when the richer tier isn't available."""
        recommended = strategy.recommended_solution
        if recommended is None or recommended.insufficient_evidence:
            return "I don't have enough evidence to determine the cause.", self._unknown_follow_up(strategy)
        text = f"A possible explanation is: {recommended.likely_issue}."
        if recommended.rationale:
            text += f" ({recommended.rationale})"
        text += " Evidence is not yet sufficient to verify this -- treat it as a hypothesis to check, not a resolution to act on."
        if recommended.recommended_resolution:
            text += f" A possible resolution, not independently verified: {recommended.recommended_resolution}"
        return text, None

    def _unknown_follow_up(self, strategy: "InvestigationStrategy") -> str | None:
        if strategy.ordered_log_collection:
            return "Consider collecting the recommended logs below to gather more evidence."
        if strategy.missing_evidence:
            names = ", ".join(item.description for item in strategy.missing_evidence[:3])
            return f"Providing more detail ({names}) may help narrow this down."
        return None

    # --- last_focus / last_referenced_* derivation (feeds Phase 3's own hook) -

    def _derive_focus(self, strategy: "InvestigationStrategy") -> tuple[str | None, str | None, int | None]:
        """Derives what THIS real answer actually centered on, from
        data ``RecommendationEngine`` already computed -- the exact
        "what did the answer focus on" decision Phase 3's
        ``record_assistant_turn`` was deliberately built to never make
        itself (see that method's docstring)."""
        structured = strategy.structured_resolution
        if structured is not None and structured.resolution_candidates:
            primary = next((c for c in structured.resolution_candidates if c.is_primary), structured.resolution_candidates[0])
            evidence = primary.evidence
            if evidence.kind == EvidenceKind.HISTORICAL_INVESTIGATION:
                return "resolution", evidence.source_id, None
            if evidence.kind == EvidenceKind.TFS_CASE:
                try:
                    return "resolution", None, int(evidence.source_id)
                except ValueError:
                    return "resolution", None, None
            return "resolution", None, None
        if strategy.historical_investigations:
            return "historical_match", strategy.historical_investigations[0].record_id, None
        return None, None, None
