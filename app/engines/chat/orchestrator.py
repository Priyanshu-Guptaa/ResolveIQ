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
    ChatEnhancementRef,
    ChatMessage,
    ChatResponse,
    ChatSession,
    ConversationSlots,
    MessageRole,
    ReferenceState,
)
from app.domain.enums import EvidenceType, LogLevel
from app.domain.evidence import Evidence
from app.domain.evidence_bundle import SufficiencyLevel
from app.domain.investigation import InvestigationSession
from app.domain.provenance import EvidenceKind, ResolutionProvenance
from app.domain.query_understanding import SlotConfidence
from app.engines.chat.confidence_expansion import contains_unsupported_confidence_claim
from app.engines.chat.grounding_validator import (
    check_claim_authority_preserved,
    check_no_material_loss,
    validate as validate_grounding,
)
from app.engines.chat.historical_expansion import contains_unsupported_resolution_claim
from app.engines.chat.knowledge_question import (
    TROUBLESHOOTING_ACTION_WORDS,
    contains_knowledge_question,
    extract_concept_words,
    extract_troubleshooting_subject_words,
    is_definitional_question,
    lexical_overlap,
    word_overlap,
)
from app.engines.chat.log_question import (
    contains_l2_task_note_question,
    contains_l3_escalation_question,
    contains_log_analysis_question,
    contains_log_comparison_question,
)
from app.engines.chat.query_intent import AnswerIntent, QueryContext, build_query_context, classify_intent
from app.engines.chat.retrieval_profile import (
    RETRIEVAL_PROFILES,
    build_evidence_bundle,
    has_majority_overlap,
    kind_priority,
    rank_items,
)
from app.engines.chat.scope_expansion import contains_unsupported_scope_expansion
from app.engines.chat.scope_question import split_out_scope_clause
from app.engines.chat.troubleshooting_expansion import contains_unsupported_troubleshooting_action
from app.engines.chat.troubleshooting_question import contains_troubleshooting_question, split_out_troubleshooting_clause
from app.engines.chat.troubleshooting_synthesis_question import contains_troubleshooting_synthesis_question
from app.engines.external_knowledge.extraction import truncate as truncate_extract
from app.engines.investigation.engine import InvestigationNotFoundError
from app.engines.llm.prompt_builder import PromptBuilder, available_checks
from app.engines.llm.provider import LLMProviderError
from app.engines.log_intelligence.engine import LogIntelligenceEngine
from app.engines.log_intelligence.flow import reconstruct_flow

if TYPE_CHECKING:
    from app.domain.evidence_bundle import EvidenceBundle
    from app.domain.log_flow import LogObservationSummary
    from app.domain.recommendation import InvestigationStrategy, RecommendedSolution
    from app.domain.structured_resolution import ResolutionCandidate, StructuredResolution
    from app.engines.chat.conversation_state import ConversationStateEngine
    from app.engines.chat.enhancement import ChatEnhancementService
    from app.engines.chat.log_upload import ChatLogUploadService
    from app.engines.investigation.engine import InvestigationEngine
    from app.engines.llm.provider import LLMProvider
    from app.engines.recommendation.engine import RecommendationEngine
    from app.infrastructure.db.log_knowledge_repository import LogKnowledgeRepository
    from app.infrastructure.db.lookup_repository import LookupRepository

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
        enhancement_service: "ChatEnhancementService | None" = None,
        async_enabled: bool = False,
        log_upload_service: "ChatLogUploadService | None" = None,
        log_knowledge_repo: "LogKnowledgeRepository | None" = None,
        lookup_repo: "LookupRepository | None" = None,
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
        self._enhancement_service = enhancement_service
        self._async_enabled = async_enabled
        """Chat Assistant Phase 37 -- both default to None/False, the
        same "purely additive" contract every LLM-related parameter on
        this constructor has had since Phase 1: with either left at its
        default, handle_message() calls the existing, unchanged
        _generate_answer() synchronously exactly as every prior phase
        has, and no ChatResponse.enhancement is ever set. Only when
        BOTH async_enabled is True AND a real llm_provider/
        enhancement_service are wired (Settings.llm_enabled AND
        Settings.llm_async_enabled, see app/api/dependencies.py) does
        handle_message() return the deterministic answer immediately
        and schedule LLM generation as a background job instead --
        see _should_enhance_asynchronously()."""
        self._log_upload_service = log_upload_service
        """Chat Assistant Phase 39 -- None (default) means the standalone
        branch of _resolve_investigation_for_retrieval behaves byte-
        identical to every prior phase: no session-uploaded log evidence
        exists to inject. Only wired via DI (app/api/dependencies.py)
        alongside the new POST /chat/sessions/{session_id}/logs endpoint.
        Investigation-scoped sessions never consult this service -- a
        chat-uploaded log for a real investigation is persisted straight
        into that investigation's own evidence via
        InvestigationEngine.add_file_evidence (the existing, unmodified
        Workspace upload path), so it is already present the next time
        this method re-fetches that real investigation below; no
        orchestrator change was needed for that branch."""
        self._log_knowledge_repo = log_knowledge_repo
        self._product_name_tokens: frozenset[str] = self._load_product_name_tokens(lookup_repo)
        """Phrase-Aware Relevance investigation -- real, governed product
        names (``LookupRepository.list_products()``, the same source
        ``RecommendationEngine``/``ApplicabilityRanker`` already use for
        applicability, never a new or hardcoded list), tokenized once at
        construction time via the same ``extract_concept_words`` every
        overlap check already uses. Used by ``_tier_answer_is_off_topic``/
        ``_compose_knowledge_synthesis`` to recognize that a concept word
        like "command"/"center" (from the real governed product "Command
        Center") is the SUBJECT of almost every document in a Command-
        Center-focused knowledge base, and therefore does not, by itself,
        distinguish one candidate from another the way a genuinely
        specific word ("settings") does -- see those methods' own
        docstrings for the full real-corpus finding. ``None`` (no
        ``lookup_repo`` wired, e.g. most existing tests) degrades to an
        empty set, which is a complete no-op: every concept word is then
        treated as before this phase, byte-identical to the prior,
        already-tested behavior."""
        """Grounded Conversational Intelligence phase -- None (default,
        every existing caller/test unchanged) means
        ``_compose_documented_flow_section`` never runs and correlation
        stays exactly as the L2/L3 Investigation Copilot phase left it
        (OBSERVED grouping + hedged INFERRED role narrative, never
        CONFIRMED). Only wired via DI (app/api/dependencies.py, the
        already-existing ``_log_knowledge_repository()`` singleton --
        no new repository implementation) lets that method additionally
        attempt the existing, frozen ``reconstruct_flow`` (app.engines.
        log_intelligence.flow) for a genuinely wiki-documented,
        CONFIRMED-tier component ordering when the log's own dominant
        correlating identifier unambiguously matches a real scenario.
        ``flow.py`` itself is never modified -- this is purely an
        additional caller of its existing public function."""

    @staticmethod
    def _load_product_name_tokens(lookup_repo: "LookupRepository | None") -> frozenset[str]:
        """One real, bounded DB read at construction time (this
        codebase's own governed product table has a handful of rows,
        the same table ``RecommendationEngine``/``ApplicabilityRanker``
        already query) -- never a per-message query, never a new
        caching layer. ``None`` (the same ``is not None`` convention
        ``RecommendationEngine`` already uses for this exact repository)
        degrades to an empty set, a complete no-op for every caller
        below."""
        if lookup_repo is None:
            return frozenset()
        tokens: set[str] = set()
        for product in lookup_repo.list_products():
            tokens.update(extract_concept_words(product.name))
        return frozenset(tokens)

    def _distinctive_concept_words(self, concept_words: list[str]) -> list[str]:
        """Phrase-Aware Relevance investigation -- the real corpus
        finding behind this method: "Explain process settings in
        Command Center." shares "process"/"command"/"center" (3 of 4
        concept words) with "IAD Move Checklist Answers_Master" -- a
        document about neither process settings nor anything the
        question is really about -- purely because "Command"/"Center"
        (the governed product's own name) appear in nearly every
        Command-Center-focused document, while the one word that
        actually distinguishes this question, "settings", appears in
        neither. Plain majority-overlap counting cannot tell "process"
        (a coincidental, generic hit) apart from "settings" (the real
        subject) -- both count the same. Filtering out the question's
        own product-name tokens before counting fixes this without a
        corpus-wide term-frequency table: "process"+"settings" (what's
        left) requires BOTH to match for a 2-word majority, so a
        document sharing only "process" is correctly rejected, while
        "Process Settings Configuration Guide for Command Center" (which
        genuinely has both) is correctly accepted.

        Falls back to the full, unfiltered ``concept_words`` whenever
        removing product-name tokens would leave nothing at all (a
        question that is ITSELF only the product's own name, e.g. "What
        is Command Center?") -- there is nothing left to be more
        specific than, so the original, already-tested majority rule is
        the right one to apply to the full set, exactly as before this
        method existed."""
        distinctive = [w for w in concept_words if w not in self._product_name_tokens]
        return distinctive or concept_words

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
        log_evidence = [
            e for e in investigation_for_retrieval.evidence if e.evidence_type == EvidenceType.LOG_FILE and e.log_events
        ]
        """Chat + Log Intelligence integration -- the REAL, already-parsed
        LogEvent/ExtractedEntity detail behind ``log_observations``'
        aggregate counts, for ``_compose_log_analysis_answer`` (never for
        the LLM prompt -- that remains ``log_observations`` only, the
        existing allowlist-transform safety property, untouched)."""
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

        enhancement_ref: ChatEnhancementRef | None = None
        if self._should_enhance_asynchronously(strategy):
            # Chat Assistant Phase 37 -- DETERMINISTIC ANSWER FIRST. The
            # entire knowledge pipeline above (retrieval, ranking,
            # StructuredResolution) already ran exactly once, synchronously,
            # deterministically -- it is never re-run for the async path.
            # The deterministic answer is composed and returned to the
            # caller in THIS call, never waiting on Ollama; the LLM
            # enhancement (if any) is scheduled as a background job and
            # polled separately (GET /chat/enhancements/{job_id}). See
            # _finalize_llm_answer for what that job actually runs -- the
            # exact same _attempt_llm_answer safety-validated decision the
            # synchronous path below uses, never a second, divergent
            # implementation.
            (
                answer_text,
                follow_up,
                answer_kind,
                sanitized_question,
                had_scope,
                had_troubleshooting,
                structured_for_job,
            ) = self._compose_deterministic_answer(text, strategy, log_evidence, investigation_for_retrieval)
            if not ((had_scope or had_troubleshooting) and not sanitized_question):
                # Something non-scope/non-troubleshooting remains to ask
                # the LLM -- otherwise there is nothing for a job to
                # attempt (mirrors _generate_answer's own bypass exactly).
                job = self._enhancement_service.submit(  # type: ignore[union-attr]  -- guarded by _should_enhance_asynchronously
                    session_id,
                    lambda sq=sanitized_question, st=structured_for_job, lo=log_observations, hs=had_scope, ht=had_troubleshooting, da=answer_text, strat=strategy: self._finalize_llm_answer(  # noqa: E501
                        sq, st, lo, hs, ht, da, strat
                    ),
                )
                enhancement_ref = ChatEnhancementRef(job_id=job.id, status=job.status)
            # §18 -- the real LLM attempt happens later, inside the
            # background job (_finalize_llm_answer), not in this
            # request -- "deferred" is the honest debug state here,
            # never "attempted"/"accepted" (this request never called
            # _attempt_llm_answer itself).
            llm_debug = {
                "llm_attempted": False,
                "llm_accepted": False,
                "llm_rejection_reason": None,
                "llm_deferred": enhancement_ref is not None,
            }
        else:
            answer_text, follow_up, answer_kind, llm_debug = self._generate_answer(
                text, strategy, log_observations, log_evidence, investigation_for_retrieval
            )
            llm_debug = {**llm_debug, "llm_deferred": False}

        focus, referenced_investigation_id, referenced_tfs_id = self._derive_focus(strategy)
        self._state.record_assistant_turn(
            session_id,
            answer_text,
            focus=focus,
            referenced_investigation_id=referenced_investigation_id,
            referenced_tfs_id=referenced_tfs_id,
        )

        structured = strategy.structured_resolution
        query_context = build_query_context(
            text,
            has_log_evidence=bool(log_evidence),
            parsed_query=parsed,
            context_text=investigation_for_retrieval.context_text,
        )
        # Evidence-Centered Knowledge Retrieval & Synthesis phase, §18 --
        # the full EvidenceBundle, built once here purely for debug
        # purposes (never a second source of truth for the answer text
        # itself, which the composers above already produced from their
        # own, equivalent bundle-driven ranking/authority/claims logic --
        # see retrieval_profile.py's module docstring). Cheap: pure
        # in-memory filtering/sorting over the same lists strategy
        # already holds, no new I/O (§24). Never exposed to normal users
        # by default -- see ChatResponse.debug's own docstring.
        next_actions = available_checks(structured) if structured is not None else []
        evidence_bundle = build_evidence_bundle(
            text,
            query_context,
            strategy,
            log_evidence=log_evidence,
            min_score=self._KNOWLEDGE_SYNTHESIS_MIN_SCORE,
            min_score_secondary=self._KNOWLEDGE_SYNTHESIS_MIN_SCORE_SECONDARY,
            next_actions=next_actions,
        )
        debug_info = {
            **query_context.as_debug_dict(),
            "answer_kind": answer_kind,
            "resolution_provenance": structured.confidence.value if structured is not None else None,
            "retrieval_profile": evidence_bundle.retrieval_profile,
            "candidates": {
                "documentation": len(strategy.documentation),
                "historical": len(strategy.historical_investigations),
                "known_bugs": len(strategy.known_bugs),
                "tfs": len(strategy.tfs_matches.matches) if strategy.tfs_matches is not None else 0,
                "wiki": len(strategy.wiki_matches.matches) if strategy.wiki_matches is not None else 0,
            },
            "selected_evidence": [item.title for item in evidence_bundle.all_evidence()],
            "rejected_evidence": evidence_bundle.rejected_evidence,
            # Final Support-Quality Pass, §5 -- claim_id/source_ids/
            # confidence/supported added for internal traceability
            # (final statement -> claim ID -> EvidenceItem -> source);
            # every existing key here is unchanged, so no existing
            # debug consumer that reads only "text"/"authority"/
            # "category"/"is_current" is affected.
            "claims": [
                {
                    "claim_id": c.claim_id,
                    "text": c.text,
                    "supported_by": c.supported_by,
                    "source_ids": c.source_ids,
                    "authority": c.authority.value,
                    "category": c.category,
                    "is_current": c.is_current,
                    "confidence": c.confidence,
                    "supported": c.supported,
                }
                for c in evidence_bundle.claims
            ],
            "contradictions": [
                {"description": c.description, "source_a": c.source_a, "source_b": c.source_b}
                for c in evidence_bundle.contradictions
            ],
            "sufficiency": evidence_bundle.sufficiency.value,
            "answer_plan": answer_kind,
            # Final Support-Quality Pass, §9 -- a real, unanswered-
            # question signal for future knowledge-base prioritization,
            # never exposed to normal users by default. Recorded here
            # (not inside any composer) so it reflects the SAME
            # independently-built EvidenceBundle every other debug field
            # already uses -- never a second, divergent sufficiency
            # judgment.
            "knowledge_gaps": self._knowledge_gaps_for(text, query_context, evidence_bundle),
            **llm_debug,
        }
        return ChatResponse(
            answer_text=answer_text,
            answer_kind=answer_kind,
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
            enhancement=enhancement_ref,
            debug=debug_info,
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

    _NO_PRIOR_CONTEXT_CANDIDATES = (
        "nothing has been referenced yet in this conversation",
        "nothing has been established yet to resolve this against",
    )
    """The two literal sentinel strings ``ConversationStateEngine.
    resolve_reference`` uses when a reference cue matched but there is
    genuinely ZERO prior context to resolve it against -- a
    fundamentally different situation from a real tie between multiple
    real candidates (§9's own distinction). See
    ``_compose_ambiguous_response``."""

    def _compose_ambiguous_response(self, ambiguity: ChatAmbiguity) -> tuple[str, str]:
        candidates = ", ".join(ambiguity.candidates) if ambiguity.candidates else "more than one real possibility"
        if ambiguity.kind == ChatAmbiguityKind.REFERENCE_AMBIGUOUS:
            if len(ambiguity.candidates) == 1 and ambiguity.candidates[0] in self._NO_PRIOR_CONTEXT_CANDIDATES:
                # Real-Corpus Answer Quality & Final Chat Hardening
                # phase, §9 -- this is not a tie between real
                # candidates at all; there is nothing established yet
                # in this conversation to resolve the reference
                # against. The old, generic "I found multiple possible
                # interpretations" framing was actively misleading here
                # (there was never more than one interpretation -- there
                # was zero context), and gave the user nothing concrete
                # to act on. A real, answerable clarifying question
                # instead.
                answer = (
                    "I can look for similar historical cases, but I don't have a specific investigation, product, "
                    "component, or issue to search against yet in this conversation."
                )
                follow_up = "Which product, component, or issue should I search for?"
                return answer, follow_up
            answer = "I found multiple possible interpretations of that reference. Please clarify which one you mean."
        else:
            answer = f"I found multiple possible {ambiguity.slot_name} matches for this question. Please clarify which one you mean."
        follow_up = f"Which of the following did you mean: {candidates}?"
        return answer, follow_up

    _NO_GAP_INTENTS = (AnswerIntent.FOLLOW_UP, AnswerIntent.UNKNOWN)
    """Final Support-Quality Pass, §9 -- a knowledge gap is only
    recorded for a question that was actually ASKING for an answer;
    FOLLOW_UP (a thin continuation with no subject of its own) and
    UNKNOWN (not recognized as any real question shape at all) are
    never real "the knowledge base lacks this" signals -- recording
    one for either would flood this list with noise no future
    knowledge-base-improvement effort could act on."""

    @staticmethod
    def _knowledge_gaps_for(question: str, query_context: "QueryContext", evidence_bundle: "EvidenceBundle") -> list[dict]:
        """Final Support-Quality Pass, §8/§9 -- a real, disclosed signal
        that this specific question was NOT well-served by the current
        knowledge base, for future KB-improvement prioritization. Never
        a second sufficiency judgment: reuses ``evidence_bundle.
        sufficiency`` (the exact same, already-computed value every
        other debug field and every composer's own fallback-to-honesty
        behavior already relies on) rather than re-deriving one. Only
        WEAK/INSUFFICIENT count as a gap -- MODERATE/STRONG/
        AUTHORITATIVE evidence means the question WAS answerable, even
        if the final answer also had to hedge or disclose uncertainty
        (that is a correct, working answer, not a gap). CONTRADICTORY
        is deliberately excluded too: evidence exists and conflicts,
        which is a data-quality issue (see the "Evidence conflict"
        sections), not an "evidence is missing" one. Never exposed to
        normal users by default (``ChatResponse.debug`` only)."""
        if query_context.intent in ChatOrchestrator._NO_GAP_INTENTS:
            return []
        if evidence_bundle.sufficiency not in (SufficiencyLevel.INSUFFICIENT, SufficiencyLevel.WEAK):
            return []
        subject = query_context.subject or question
        gap_type = (
            "INSUFFICIENT_AUTHORITATIVE_EVIDENCE"
            if evidence_bundle.sufficiency == SufficiencyLevel.INSUFFICIENT
            else "WEAK_EVIDENCE_RELEVANCE"
        )
        return [
            {
                "question": question,
                "gap_type": gap_type,
                "missing": f'documentation, historical cases, or known bugs that actually cover "{subject}"',
            }
        ]

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
        if self._log_upload_service is not None:
            # Chat Assistant Phase 39 -- any log(s) uploaded through this
            # standalone session's own chat-log-upload endpoint. Additive
            # only: with no service wired, or nothing uploaded yet, this
            # is a no-op and `synthesized` is unchanged from every prior
            # phase. Each Evidence here was already run through the
            # existing, unmodified LogIntelligenceEngine at upload time
            # (see ChatLogUploadService.upload) -- summarize_observations()
            # below picks it up automatically, no further change needed.
            for evidence in self._log_upload_service.get_evidence_for_session(session.id):
                synthesized.add_evidence(evidence)
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
        log_evidence: "list[Evidence] | None" = None,
        investigation: "InvestigationSession | None" = None,
    ) -> tuple[str, str | None, str | None, dict]:
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
        LLM is permitted to see.

        Final LLM Orchestration Hardening -- the routing this method
        follows changed from "try the LLM first, fall back to
        _compose_answer only on failure/rejection" to "compose the
        rich, evidence-grounded deterministic answer FIRST, always,
        then let the LLM (when wired) attempt to IMPROVE it, accepting
        that improvement only when it passes both the existing safety
        gates AND a new completeness check." This project's own real,
        live Ollama end-to-end testing found the exact failure this
        fixes: for "What is process settings in CC?", the deterministic
        knowledge synthesizer had real, cited documentation to answer
        from, but the OLD routing never even computed that answer --
        it tried the LLM first, which had no visibility into that
        documentation at all (see ``_generate_llm_answer``'s own scope:
        only ``StructuredResolution``, never Documentation/TFS/Wiki),
        and safely-but-uselessly said "cannot be answered based on the
        provided evidence." That LLM text passed every existing safety
        gate (nothing fabricated) yet was strictly worse than the
        answer ResolveIQ could already give. ``check_no_material_loss``
        (``app.engines.chat.grounding_validator``) is the new,
        deterministic, structural check for exactly this: does the
        LLM's candidate answer still contain every source citation/
        identifier/timestamp the rich deterministic answer already
        established? If anything is missing, the LLM's answer is
        rejected and the deterministic one is used instead -- "the LLM
        call succeeded" is deliberately never treated as "the LLM
        answer is acceptable" (this phase's own explicit instruction).
        Never a second LLM call to judge the first one's quality --
        only the same closed, regex/entity-extraction primitives
        ``validate()`` already uses.

        ``answer_kind``/``follow_up_question`` are now ALWAYS the
        deterministic composer's own values, even when the LLM's
        (checked, accepted) text is what is actually returned -- the
        LLM enhances wording, it does not change what KIND of answer
        this fundamentally is (a knowledge answer stays a knowledge
        answer for UI purposes, e.g.).

        Evidence-Centered Knowledge Retrieval & Synthesis phase, §18 --
        now returns a 4th element, ``llm_debug`` (``{"llm_attempted",
        "llm_accepted", "llm_rejection_reason"}``), the real gate
        decision this method already made, for ``ChatResponse.debug``.
        Its own only caller (``handle_message``) is updated to match;
        no test calls this private method directly."""
        sanitized_question, had_scope, had_troubleshooting, structured = self._prepare_question(question, strategy)

        # Domain-specific deterministic reasoning ALWAYS runs first
        # (knowledge synthesis / troubleshooting synthesis / log
        # analysis / L2/L3 / tier-based -- whichever _compose_answer
        # itself already decides applies), regardless of whether an
        # LLM is wired at all.
        #
        # Real-Corpus Answer Quality & Final Chat Hardening phase --
        # the ORIGINAL ``question`` is passed here, not ``sanitized_
        # question``. ``sanitized_question`` exists ONLY to protect
        # what the LLM sees (Rule 9's scope/troubleshooting-clause
        # removal, Phases 31/32) -- every deterministic composer
        # ``_compose_answer`` dispatches to is already safe by
        # construction (never invents an action, only cites
        # ``available_checks``/real evidence), so stripping the clause
        # before the DETERMINISTIC path even sees it serves no safety
        # purpose and was a real, discovered bug: "What should I check
        # next?" with zero evidence-backed checks had its ENTIRE text
        # removed before reaching ``_compose_troubleshooting_synthesis``,
        # so a real, on-topic historical match was silently retrieved
        # but never used, and the answer collapsed to the generic
        # "A possible explanation may exist..." boilerplate despite
        # strong evidence. ``sanitized_question`` is still exactly what
        # reaches the LLM below -- this change touches only which text
        # the deterministic composers themselves rank/match against.
        answer_text, follow_up, answer_kind = self._compose_answer(strategy, question, log_evidence, investigation)
        answer_text = self._append_deterministic_statements(answer_text, structured, had_scope, had_troubleshooting)
        # Evidence-Centered Knowledge Retrieval & Synthesis phase, §18 --
        # the real gate decision this method already makes, finally
        # surfaced structurally (``ChatResponse.debug``) instead of only
        # a log line. ``llm_attempted`` is True only when a real call to
        # ``_attempt_llm_answer`` was made (never for the "nothing left
        # to ask" bypass below, and never when no provider is wired).
        llm_debug = {"llm_attempted": False, "llm_accepted": False, "llm_rejection_reason": None}

        if (had_scope or had_troubleshooting) and not sanitized_question:
            # Nothing non-scope/non-troubleshooting is left to ask the
            # LLM at all.
            return answer_text, follow_up, answer_kind, llm_debug

        if self._llm is not None and structured is not None and self._llm.is_configured():
            try:
                llm_debug["llm_attempted"] = True
                llm_answer, rejection_reason = self._attempt_llm_answer(sanitized_question, structured, log_observations, strategy)
                if llm_answer is not None:
                    llm_answer = self._append_deterministic_statements(llm_answer, structured, had_scope, had_troubleshooting)
                    missing = check_no_material_loss(answer_text, llm_answer)
                    if not missing:
                        llm_debug["llm_accepted"] = True
                        return llm_answer, follow_up, answer_kind, llm_debug
                    logger.warning(
                        "LLM answer omitted %d real fact(s) present in the deterministic answer (%s); "
                        "using the deterministic answer instead.",
                        len(missing),
                        "; ".join(missing),
                    )
                    llm_debug["llm_rejection_reason"] = f"omitted {len(missing)} real fact(s) present in the deterministic answer"
                else:
                    llm_debug["llm_rejection_reason"] = rejection_reason
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
                llm_debug["llm_rejection_reason"] = f"provider error: {exc}"
        return answer_text, follow_up, answer_kind, llm_debug

    def _prepare_question(
        self, question: str, strategy: "InvestigationStrategy"
    ) -> tuple[str, bool, bool, "StructuredResolution | None"]:
        """Chat Assistant Phase 37 -- extracted, unchanged in behavior,
        from what used to be the start of ``_generate_answer`` (Phases
        31/32's scope/troubleshooting clause-splitting), so BOTH the
        existing synchronous path and the new asynchronous enhancement
        path (``_should_enhance_asynchronously``/``handle_message``)
        make this exact same deterministic decision instead of a second,
        divergent implementation. Returns
        ``(sanitized_question, had_scope, had_troubleshooting, structured)``."""
        sanitized_question, had_scope = split_out_scope_clause(question)
        structured = strategy.structured_resolution
        checks_available = bool(available_checks(structured)) if structured is not None else False
        had_troubleshooting = False
        if not checks_available:
            sanitized_question, had_troubleshooting = split_out_troubleshooting_clause(sanitized_question)
        return sanitized_question, had_scope, had_troubleshooting, structured

    def _attempt_llm_answer(
        self,
        question: str,
        structured: "StructuredResolution",
        log_observations: "LogObservationSummary | None",
        strategy: "InvestigationStrategy | None" = None,
    ) -> tuple[str | None, str | None]:
        """Chat Assistant Phase 37 -- the single, centralized "try the
        LLM and validate its output" decision, extracted unchanged from
        ``_generate_answer`` so it can be shared by BOTH the existing
        synchronous path and the new asynchronous enhancement path
        (``_finalize_llm_answer``) -- never duplicated, never a second,
        potentially-diverging safety check.

        Returns ``(answer_text, rejection_reason)``. ``answer_text`` is
        the validated answer text on success, ``None`` specifically
        when generation succeeded but the RAW output failed any of
        three deterministic post-generation gates -- Phase 49's
        confidence-upgrade gate (``contains_unsupported_confidence_
        claim``), Phase 35B's customer-scope-expansion gate
        (``contains_unsupported_scope_expansion``), or Phase 46's
        troubleshooting-action gate (``contains_unsupported_
        troubleshooting_action``, checked only when zero evidence-backed
        checks exist) -- all three checked here, on the raw text, before
        any deterministic statement is appended. The caller's job in
        every rejection case is to fall back to the deterministic
        answer, never to rewrite the rejected text into a new claim.
        ``rejection_reason`` (Evidence-Centered Knowledge Retrieval &
        Synthesis phase, §18) is ``None`` on success and a short,
        human-readable string on every rejection path, so
        ``ChatResponse.debug`` can surface ``llm_attempted``/``llm_
        accepted``/``llm_rejection_reason`` truthfully instead of only
        the pre-existing ``logger.warning`` call (kept, unchanged, on
        every path -- this adds a structured signal alongside it, never
        replaces it). Raises ``LLMProviderError`` (unchanged, from
        ``_generate_llm_answer``/``OllamaProvider``) for a genuine
        provider failure -- connection, timeout, empty response,
        malformed response -- which every caller must also treat as
        "fall back to the deterministic answer," exactly as this
        project always has.

        Final Hardening Pass, Objective 1 -- a fourth gate now runs
        after the three below: ``app.engines.chat.grounding_validator.
        validate``, checking every specific FACT VALUE the raw answer
        cites (an identifier, a timestamp, a computed duration, an
        error code, a case number, a configuration value) against
        ``evidence_text`` (the exact ``user_prompt`` the model was
        given -- see ``_generate_llm_answer``'s own docstring). A
        HIGH-severity finding (any claim type except an unsupported
        configuration value) rejects the whole answer, identically to
        the three gates below. A LOW-severity finding (an unsupported
        configuration value only -- Step 1F's own worked example) does
        NOT reject: the specific unsupported sentence is replaced with
        a safe, honest fallback clause, and the REPAIRED text is what
        this method returns -- see ``GroundingResult.repaired_text``'s
        own docstring for why a full rejection is not used there."""
        answer_text, evidence_text, llm_evidence_bundle = self._generate_llm_answer(question, structured, log_observations, strategy)
        if contains_unsupported_confidence_claim(answer_text, structured.confidence):
            # Chat Assistant Phase 49 -- deterministic post-generation
            # gate, checked first (before the scope/troubleshooting gates
            # below) on the RAW LLM text, before any deterministic
            # statement is appended. Phase 48's real 140-call qwen2.5:3b
            # benchmark found a real, live case -- "Has this happened
            # before? Confirmed." -- answered against a fixture whose
            # authoritative structured.confidence was LIKELY, not
            # CONFIRMED, reaching the simulated user with no protection
            # (Rule 4 was, until this phase, a prompt-only instruction --
            # see prompt_builder.py rule 4 -- with no deterministic
            # backstop, unlike Rule 9/Rule 11). Same "prevent, don't just
            # instruct" principle as both gates below: rejecting the
            # whole answer and falling back to the existing, unchanged
            # deterministic path -- which always renders the correct
            # tier via the same, unmodified _compose_answer -- rather
            # than rewriting the claim.
            logger.warning(
                "LLM answer contained an unsupported confidence-upgrade claim "
                "('confirmed'/'verified' below the Confirmed tier); falling back to deterministic answer."
            )
            return None, "unsupported confidence-upgrade claim"
        if contains_unsupported_scope_expansion(answer_text):
            # Chat Assistant Phase 35B -- deterministic post-generation
            # gate, checked on the RAW LLM text before any deterministic
            # statement is appended (see
            # contains_unsupported_scope_expansion's own docstring for
            # why the ordering matters). Phase 35's real-call validation
            # found qwen2.5:3b will, on a normal (non-scope) multi-part
            # question, sometimes spontaneously volunteer an unsupported
            # claim like "occurred before with other customers in the
            # APAC region" -- with no scope question anywhere in the
            # input for Phase 31's clause-removal mechanism to act on,
            # since none was asked. There is no clause to remove here;
            # the claim appears inside the answer to a real, necessary
            # question. Rejecting the whole LLM answer and falling back
            # to the existing, unchanged deterministic path -- never
            # rewriting it into a new claim -- is the same "prevent,
            # don't just instruct" principle already proven necessary
            # for the explicit-question version of this exact problem
            # (Phases 27-30's four straight failed prompt-only attempts).
            logger.warning(
                "LLM answer contained an unsupported customer-scope expansion claim; "
                "falling back to deterministic answer."
            )
            return None, "unsupported customer-scope expansion claim"
        if not available_checks(structured) and contains_unsupported_troubleshooting_action(answer_text):
            # Chat Assistant Phase 46 -- the troubleshooting analogue of
            # the gate directly above, gated on "zero evidence-backed
            # checks exist for this answer" exactly as
            # troubleshooting_question.py's own INPUT-side removal
            # already requires (see that module's docstring). Phase 32's
            # clause-removal only helps when the question text itself
            # matches its closed phrase table; Phase 45's real qwen2.5:3b
            # replay found that a differently-phrased troubleshooting
            # request ("What action should I take?", "How can I
            # investigate this problem?", and others) bypasses that
            # table entirely, reaches the LLM, and its raw, unprotected
            # output can contain exactly the generic, non-evidence-backed
            # suggestion ("ensure it is properly configured") Rule 9's
            # own prompt text already tells the model never to
            # substitute. This is the same "prevent, don't just
            # instruct" principle as the scope gate above, applied to
            # the one input path that gate does not cover. Never checked
            # when real evidence-backed checks exist -- a legitimate
            # answer is never rejected merely for containing
            # troubleshooting language.
            logger.warning(
                "LLM answer contained an unsupported generic troubleshooting action with no "
                "evidence-backed checks available; falling back to deterministic answer."
            )
            return None, "unsupported generic troubleshooting action with no evidence-backed checks"
        if contains_unsupported_resolution_claim(answer_text, structured.confidence):
            # Evidence-Centered Knowledge Retrieval & Synthesis phase,
            # §16 -- the historical-recommendation analogue of the
            # confidence gate above: at POSSIBLE/UNKNOWN tier, any
            # evidence this turn cites for "the fix" is at best a past
            # case's own recorded action (a HISTORICAL_RECOMMENDATION
            # claim, never promoted higher -- see retrieval_profile.py/
            # evidence_bundle.py), so settled-fix language here is
            # unsupported regardless of which specific case it
            # paraphrases. Never fires at CONFIRMED/LIKELY tier, where a
            # real current resolution genuinely exists.
            logger.warning(
                "LLM answer contained an unsupported settled-resolution claim below the Likely/"
                "Confirmed tier; falling back to deterministic answer."
            )
            return None, "unsupported historical-recommendation-as-current-resolution claim"

        # Final Hardening Pass, Objective 1 -- see this method's own
        # docstring note above for the HIGH-severity-rejects/LOW-
        # severity-repairs split.
        grounding = validate_grounding(answer_text, evidence_text, confidence=structured.confidence)
        if not grounding.valid:
            claim_types = ", ".join(sorted({c.claim_type for c in grounding.unsupported_claims}))
            logger.warning(
                "LLM answer contained %d unsupported fact claim(s) (%s); falling back to deterministic answer.",
                len(grounding.unsupported_claims),
                claim_types,
            )
            return None, f"grounding validation failed: unsupported {claim_types} claim(s)"
        if grounding.repaired_text is not None and grounding.repaired_text != answer_text:
            logger.warning(
                "LLM answer contained an unsupported configuration value; repaired in place."
            )
            answer_text = grounding.repaired_text

        # Final Support-Quality Pass, §5/§6 -- claim-authority
        # preservation, extending (never replacing) the grounding gate
        # above: a claim's own citation surviving is not enough if its
        # authority-appropriate caveat (historical/known-bug/inference)
        # was dropped along the way -- see ``check_claim_authority_
        # preserved``'s own docstring for exactly what this checks and
        # why it is claim-driven, not tier-driven (catches a drop the
        # tier-scoped gates above would not, e.g. inside an otherwise
        # LIKELY-tier answer that also cites a historical case inline).
        # ``llm_evidence_bundle`` is only ``None`` when ``strategy`` was
        # never supplied (a caller that predates §17's async-parity
        # wiring); this check is simply skipped then, never a false
        # rejection of an answer this method can't evaluate.
        if llm_evidence_bundle is not None:
            missing_caveats = check_claim_authority_preserved(llm_evidence_bundle.claims, answer_text)
            if missing_caveats:
                logger.warning(
                    "LLM answer dropped %d claim-authority caveat(s) (%s); falling back to deterministic answer.",
                    len(missing_caveats),
                    "; ".join(missing_caveats),
                )
                return None, f"missing claim-authority caveat(s): {'; '.join(missing_caveats)}"

        return answer_text, None

    def _compose_deterministic_answer(
        self,
        question: str,
        strategy: "InvestigationStrategy",
        log_evidence: "list[Evidence] | None" = None,
        investigation: "InvestigationSession | None" = None,
    ) -> tuple[str, str | None, str | None, str, bool, bool, "StructuredResolution | None"]:
        """Chat Assistant Phase 37 -- composes the same deterministic
        answer ``_generate_answer``'s own fallback branch would (real
        root cause/resolution/confidence, plus scope/no-checks
        deterministic statements when the question asked for them),
        without ever attempting the LLM. This is what
        ``handle_message`` returns IMMEDIATELY on the asynchronous path
        -- never waiting on Ollama -- and it is exactly what the
        synchronous path already falls back to on any LLM failure/
        rejection, so an asynchronous user's initial answer is never
        weaker than a synchronous user's worst case.

        Returns ``(answer_text, follow_up, answer_kind, sanitized_question,
        had_scope, had_troubleshooting, structured)`` -- the last four are
        handed straight to ``_finalize_llm_answer`` if an enhancement job
        is scheduled, so that job makes the identical clause-splitting
        decision this answer was already built from, never a second,
        possibly-different one.

        Real-Corpus Answer Quality & Final Chat Hardening phase -- same
        fix as ``_generate_answer``'s own identical call: the ORIGINAL
        ``question`` is passed to ``_compose_answer``, not
        ``sanitized_question`` -- see that method's own docstring note
        for why. ``sanitized_question`` is still returned unchanged and
        still exactly what ``_finalize_llm_answer`` hands the LLM."""
        sanitized_question, had_scope, had_troubleshooting, structured = self._prepare_question(question, strategy)
        answer_text, follow_up, answer_kind = self._compose_answer(strategy, question, log_evidence, investigation)
        answer_text = self._append_deterministic_statements(answer_text, structured, had_scope, had_troubleshooting)
        return answer_text, follow_up, answer_kind, sanitized_question, had_scope, had_troubleshooting, structured

    def _finalize_llm_answer(
        self,
        question: str,
        structured: "StructuredResolution | None",
        log_observations: "LogObservationSummary | None",
        had_scope: bool,
        had_troubleshooting: bool,
        deterministic_answer_text: str = "",
        strategy: "InvestigationStrategy | None" = None,
    ) -> str | None:
        """Chat Assistant Phase 37 -- the exact work an asynchronous
        enhancement job runs (see
        ``app.engines.chat.enhancement.ChatEnhancementService.submit``'s
        ``work`` callable contract). Calls ``_attempt_llm_answer`` --
        the SAME centralized, safety-validated decision the synchronous
        path uses, never a second implementation -- and, only on
        success, appends the same deterministic statements the
        synchronous path would, so a completed enhancement reads
        identically to how a synchronous LLM answer always has.
        Returns ``None`` (meaning "no safe enhancement available," which
        ``ChatEnhancementService`` reports as REJECTED) when the answer
        was unsafe; lets ``LLMProviderError`` propagate so that service
        can classify it as FAILED or TIMED_OUT instead.

        Final LLM Orchestration Hardening -- ``deterministic_answer_text``
        is the same rich answer ``_compose_deterministic_answer`` already
        produced (and the user already received synchronously) for this
        turn; the new ``check_no_material_loss`` gate (identical to the
        synchronous path's own -- see ``_generate_answer``'s docstring)
        rejects an "enhancement" that would actually be LESS informative
        than what the user already has. Defaults to ``""`` only so this
        stays source-compatible with ``_should_enhance_asynchronously``'s
        own precondition check, which never calls this method directly;
        every real caller (``handle_message``'s async branch) always
        supplies the real text.

        Real-Corpus Answer Quality & Final Chat Hardening phase, §17 --
        ``strategy``, when given, is threaded through to ``_attempt_
        llm_answer``/``_generate_llm_answer`` exactly like the
        synchronous path, so the async enhancement job builds and
        sends the SAME sanitized ``EvidenceBundle`` context to the LLM
        -- sync/async parity, never two divergent LLM architectures.
        Defaults to ``None`` (byte-identical to before this parameter
        existed) only for source-compatibility with the same
        precondition-check caller noted above; the real scheduling
        closure in ``handle_message`` always captures and passes the
        real ``strategy``."""
        if structured is None:  # pragma: no cover -- guarded by _should_enhance_asynchronously before a job is ever submitted
            return None
        answer_text, _rejection_reason = self._attempt_llm_answer(question, structured, log_observations, strategy)
        if answer_text is None:
            return None
        answer_text = self._append_deterministic_statements(answer_text, structured, had_scope, had_troubleshooting)
        if deterministic_answer_text:
            missing = check_no_material_loss(deterministic_answer_text, answer_text)
            if missing:
                logger.warning(
                    "Async LLM enhancement omitted %d real fact(s) present in the deterministic answer (%s); "
                    "rejecting the enhancement.",
                    len(missing),
                    "; ".join(missing),
                )
                return None
        return answer_text

    def _should_enhance_asynchronously(self, strategy: "InvestigationStrategy") -> bool:
        """Chat Assistant Phase 37 -- True only when every precondition
        for a safe, worthwhile background LLM job is met: async mode is
        on, a real provider is wired and configured, the enhancement
        service singleton is wired, and there is a real
        ``StructuredResolution`` to ground the enhancement in. False in
        every configuration this project has run to date (``async_enabled``
        defaults to False, and ``Settings.llm_async_enabled`` defaults to
        False) -- ``handle_message`` then takes the existing, unchanged
        synchronous ``_generate_answer`` path, byte-identical to every
        prior phase."""
        return (
            self._async_enabled
            and self._llm is not None
            and self._llm.is_configured()
            and self._enhancement_service is not None
            and strategy.structured_resolution is not None
        )

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
        strategy: "InvestigationStrategy | None" = None,
    ) -> tuple[str, str, "EvidenceBundle | None"]:
        """Returns ``(answer_text, evidence_text, evidence_bundle)`` --
        Final Hardening Pass, Objective 1/Step 4: ``evidence_text`` is
        the exact ``user_prompt`` string the provider was actually
        given, handed back so ``_attempt_llm_answer`` can pass it to
        the grounding validator (``app.engines.chat.grounding_
        validator.validate``) unchanged -- never a second, separately-
        reconstructed approximation of what the model saw.
        Provider-agnostic: this method still knows nothing about
        validation; it only stops discarding data it already built.

        Final Support-Quality Pass, §5/§6 -- ``evidence_bundle`` (the
        exact same one just sent to the LLM, or ``None`` when
        ``strategy`` was not given) is likewise handed back so
        ``_attempt_llm_answer`` can run ``check_claim_authority_
        preserved`` against its real ``claims`` -- never a second,
        separately-rebuilt bundle that could disagree with what the
        model actually saw.

        Chat Assistant Phase 31 -- ``question`` here is already the
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
        no decision about it at all beyond forwarding it unchanged.

        Evidence-Centered Knowledge Retrieval & Synthesis phase, §15 --
        ``strategy``, when given, is used to build a real
        ``EvidenceBundle`` (``retrieval_profile.build_evidence_bundle``,
        the exact same construction ``ChatResponse.debug`` uses -- never
        a second, divergent bundle) and pass it to ``PromptBuilder.
        build()`` as a new, sanitized, capped "ADDITIONAL EVIDENCE
        CONTEXT" section -- real Documentation/Historical/Known-Bug
        excerpts the LLM previously had NO visibility into at all (only
        ``StructuredResolution`` reached it before this phase). Optional
        and defaults to ``None`` only for source-compatibility with
        callers that predate this parameter. Real-Corpus Answer Quality
        & Final Chat Hardening phase, §17 -- BOTH real callers now
        supply it: the synchronous path (``_generate_answer``) always
        has ``strategy`` in scope and passes it, and the asynchronous
        enhancement path (``_finalize_llm_answer``) now threads it
        through its background-job closure too (``handle_message``'s
        scheduling lambda captures ``strat=strategy``) -- sync/async
        parity, never two divergent LLM architectures."""
        sanitized_problem, _ = split_out_scope_clause(structured.problem)
        if not available_checks(structured):
            sanitized_problem, _ = split_out_troubleshooting_clause(sanitized_problem)
        prompt_structured = structured.model_copy(update={"problem": sanitized_problem or structured.problem})
        evidence_bundle = None
        if strategy is not None:
            query_context = build_query_context(question)
            evidence_bundle = build_evidence_bundle(
                question,
                query_context,
                strategy,
                min_score=self._KNOWLEDGE_SYNTHESIS_MIN_SCORE,
                min_score_secondary=self._KNOWLEDGE_SYNTHESIS_MIN_SCORE_SECONDARY,
            )
        system_prompt, user_prompt = self._prompt_builder.build(question, prompt_structured, log_observations, evidence_bundle)
        return self._llm.generate(user_prompt, system_prompt=system_prompt), user_prompt, evidence_bundle

    # --- Deterministic answer composition (§4) ---------------------------------

    def _tier_answer_is_off_topic(
        self, question: str, subject: str | None, context_text: str = "", *, is_retrieval_derived: bool = True
    ) -> bool:
        """Real-Corpus Answer Quality & Final Chat Hardening phase --
        the single most severe bug the real seeded corpus exposed:
        ``RecommendationEngine``'s own root-cause/resolution matching
        runs entirely independently of chat intent classification, and
        can genuinely reach LIKELY (or even CONFIRMED) tier off a
        single historical/known-bug record that has NOTHING to do with
        the actual question asked (real example: "What is process
        settings in CC?" reached LIKELY tier off an unrelated "IIS
        worker process crash" known bug, purely because Chroma scored
        it as the nearest semantic match) -- silently pre-empting
        ``_compose_answer``'s knowledge-synthesis branch below, which
        never even runs because CONFIRMED/LIKELY both ``return`` before
        reaching it. This is the exact "task"/69%-similarity failure
        shape this codebase has fixed twice before at the composer
        level (see ``_compose_knowledge_synthesis``'s own docstring),
        now discovered one layer up, at the confidence-TIER level.

        True only when the question has real, extractable subject
        words of its own AND the tier's own ``subject`` (root_cause or
        primary resolution text) shares FEWER THAN HALF of them --
        never a guess, never a semantic-similarity threshold, the same
        ``extract_concept_words``/``lexical_overlap``/``retrieval_
        profile.has_majority_overlap`` primitives every other relevance
        check in this file already uses. A question with no extractable
        subject (a thin follow-up) never triggers this -- there is
        nothing to compare against, so the tier text is left alone,
        matching this method's own long-standing default.

        Real-Corpus Answer Quality & Final Chat Hardening phase --
        upgraded from a bare "overlap == 0" check after a second real
        corpus finding: "What is process settings in CC?" reached
        LIKELY tier off "IIS worker PROCESS crash on multipart
        uploads..." -- a single, coincidentally-shared generic word
        ("process") gave a real overlap of 1 out of 3 concept words
        ("process", "settings", "cc"), which the original bare
        overlap>0 check treated as "on topic" and left the off-topic
        answer untouched. The exact same majority-overlap discipline
        ``retrieval_profile.has_majority_overlap`` already applies to
        AUTHORITATIVE_DEFINITION/CONFIGURATION claims is reused here,
        not re-derived, so both fixes agree on what "really shares the
        subject" means.

        Deliberately scoped to CONFIRMED/LIKELY only, and only
        consulted by those two branches immediately before returning --
        never touches POSSIBLE/UNKNOWN's own, already-correct fallback
        to ``_compose_knowledge_synthesis`` a few lines below.

        Persistent Knowledge Index & Retrieval Quality investigation --
        a third real finding, this time surfaced by a controlled
        baseline-vs-hybrid-retrieval experiment rather than a bare
        baseline run: "Why did this request fail?" (no real subject of
        its own) reached LIKELY tier, under hybrid retrieval's different
        candidate ordering, off an entirely unrelated historical case
        ("...orders API to fail the whole request"). ``extract_concept_
        words`` keeps "why"/"request"/"fail" as if they were real
        subject content; they are not substring artifacts here --
        "request" and "fail" are genuine, whole, standalone words in
        that unrelated sentence, purely because ordinary English
        vocabulary for describing ANY failure naturally contains them.
        ``_compose_troubleshooting_synthesis`` already solved this exact
        class of question (frame words -- "why"/"should"/"next"/
        "request"/"fail" -- describe the SHAPE of a troubleshooting
        question, not its subject -- see ``TROUBLESHOOTING_FRAME_
        WORDS``) but only within its own normal/forced dispatch; this
        method, checked one layer up for CONFIRMED/LIKELY, still used
        the generic ``extract_concept_words``. Reused here, not
        re-derived, for the SAME troubleshooting-synthesis-shaped
        questions ``_off_topic_tier_override``'s own ``elif`` branch
        already recognizes (``contains_troubleshooting_synthesis_
        question``/``contains_troubleshooting_question``) -- every other
        question shape (entity/configuration/historical-lookup/...)
        is completely unaffected, since none of those phrases ever
        match and this falls through to the original, unmodified
        concept-word path."""
        if contains_troubleshooting_synthesis_question(question) or contains_troubleshooting_question(question):
            concept_words = ChatOrchestrator._troubleshooting_subject_words(question, context_text)
            if not subject:
                return False
            if not concept_words:
                # No real subject to check overlap against at all -- only
                # distrust the tier when its answer came from a RETRIEVAL
                # match (the only case this method ever guards). A log/
                # entity-derived root cause (``StructuredResolution.root_
                # cause_evidence[0].kind == EvidenceKind.ENTITY_HEURISTIC``
                # -- note ``StructuredResolution.source_kind`` is always
                # the fixed literal "investigation" for every chat-path
                # structured resolution regardless of where its root
                # cause actually came from, so it cannot be used for this
                # distinction) is grounded in the session's own uploaded
                # log, verified by Log Intelligence's own entity/event
                # extraction -- it was never found via text similarity to
                # the question, so having no question-side subject words
                # to compare against says nothing about whether it's
                # correct. A real regression this exact distinction
                # fixed: "Why did it fail?" in a log session,
                # with a genuine log-derived root cause ("Unhandled
                # application exception: CommandTimeoutException",
                # matching the log's own ERROR line) was wrongly replaced
                # with an honest-insufficiency answer, discarding a
                # correct, evidence-grounded hypothesis.
                return is_retrieval_derived
            overlap = word_overlap(concept_words, subject)
            return self._is_off_topic_overlap(overlap, len(concept_words))
        concept_words = extract_concept_words(question)
        if not concept_words or not subject:
            return False
        # Fix Remaining Off-Topic Answers -- acceptance review finding:
        # "Explain process settings in Command Center." (CONCEPT_
        # EXPLANATION, 4 concept words: "process"/"settings"/"command"/
        # "center") reached LIKELY tier off the SAME unrelated "IIS
        # worker PROCESS crash" known bug this method's own docstring
        # already names as its motivating example -- because 4 words is
        # already past ``_SHORT_QUESTION_WORD_COUNT`` (3), so ``_is_off_
        # topic_overlap`` used the lenient bare-``overlap > 0`` rule
        # meant for a longer, narrative, INVESTIGATIVE question ("RF
        # Mesh IP command timeout -- why did this fail?", where most of
        # the extra words really are filler around one distinctive
        # term). A knowledge-shaped question's concept words are never
        # filler that way -- "process"/"settings"/"command"/"center" are
        # ALL real, load-bearing parts of ONE compound subject, so word
        # count alone is the wrong signal here. Reuses ``_is_purely_
        # knowledge_shaped_question`` (already the exact test ``_off_
        # topic_tier_override`` uses one layer up to decide whether a
        # knowledge answer applies) rather than re-deriving a second
        # notion of "knowledge-shaped" -- for these intents, real
        # subject-content words are never long enough to need the
        # narrative-filler exception, so majority overlap (word-
        # boundary-aware, not substring) always applies regardless of
        # count. Every other question shape reaching this branch (e.g.
        # a non-troubleshooting-phrased ROOT_CAUSE question) is
        # completely unaffected -- still the original, unmodified
        # substring-``lexical_overlap`` + length-based rule.
        if ChatOrchestrator._is_purely_knowledge_shaped_question(question) and len(concept_words) >= 3:
            # Real regression this floor exists to prevent: "Tell me
            # about dashboard in CC" has only 2 concept words ("dashboard",
            # "cc"); the matching evidence shared "dashboard" (the real,
            # distinctive word) but not "cc" (the product's own generic
            # abbreviation) -- a GOOD relevance signal, not a bad one, yet
            # bare majority math (2 of 2 required) would reject it. With
            # only 1-2 words there is too little room for a spurious
            # coincidental match to hide among them the way there was in
            # the real 4-word finding this branch exists for -- so 1-2
            # word questions keep the original, more lenient rule below.
            words = self._distinctive_concept_words(concept_words)
            overlap = word_overlap(words, subject)
            return not has_majority_overlap(overlap, len(words))
        overlap = lexical_overlap(concept_words, subject)
        return self._is_off_topic_overlap(overlap, len(concept_words))

    _SHORT_QUESTION_WORD_COUNT = 3
    """Real-Corpus Answer Quality & Final Chat Hardening phase -- the
    empirically-found boundary between two real, opposite failure
    shapes: a SHORT, subject-only question ("What is AxeI meter?", 2
    concept words; "What is process settings in CC?", 3) has little
    else to disambiguate a coincidentally-shared GENERIC word ("meter",
    "process") from real relevance, so majority overlap is the right
    bar there. A LONGER, more sentence-like investigative question
    ("RF Mesh IP command timeout -- why did this fail?", 5 concept
    words after stopword removal) legitimately shares only ONE
    DISTINCTIVE word ("mesh") with a real, on-topic root cause phrased
    in different vocabulary for the rest of the sentence -- requiring a
    majority there produced a real, caught regression (``test_
    troubleshooting_synthesis_never_overrides_a_real_likely_tier``).
    This module cannot yet distinguish "generic" from "distinctive"
    words directly (no such classification exists anywhere in this
    codebase), so word COUNT is the practical, disclosed proxy: bare
    ``overlap > 0`` for longer questions (matches this codebase's
    pre-existing, already-tested default everywhere else), majority
    overlap only for short ones, where the real corpus findings
    actually were."""

    def _is_off_topic_overlap(self, overlap: int, concept_word_count: int) -> bool:
        """The one shared off-topic-overlap rule both ``_tier_answer_
        is_off_topic`` and ``_compose_troubleshooting_synthesis``'s
        ``force=True`` candidate filter use -- see ``_SHORT_QUESTION_
        WORD_COUNT``'s own docstring for why the threshold exists."""
        if concept_word_count <= self._SHORT_QUESTION_WORD_COUNT:
            return not has_majority_overlap(overlap, concept_word_count)
        return overlap <= 0

    @staticmethod
    def _is_knowledge_shaped_question(question: str) -> bool:
        """The exact gate condition ``_compose_answer``'s POSSIBLE/
        UNKNOWN branch has used since the Knowledge Answering &
        Evidence Synthesis phase, extracted into one named, shared
        method rather than re-typed at each of its call sites -- never
        a second, independently-maintained condition. Deliberately
        NOT used by the CONFIRMED/LIKELY off-topic override below --
        see ``_is_purely_knowledge_shaped_question``'s own docstring
        for why that override needs a stricter test."""
        return contains_knowledge_question(question) or classify_intent(question) in (
            AnswerIntent.CONFIGURATION,
            AnswerIntent.HOW_TO,
        )

    _PURE_KNOWLEDGE_INTENTS = (
        AnswerIntent.ENTITY_DEFINITION,
        AnswerIntent.PRODUCT_EXPLANATION,
        AnswerIntent.CONCEPT_EXPLANATION,
        AnswerIntent.CONFIGURATION,
        AnswerIntent.HOW_TO,
        AnswerIntent.HISTORICAL_LOOKUP,
    )
    """Real-Corpus Answer Quality & Final Chat Hardening phase --
    HISTORICAL_LOOKUP added: "What was the resolution in similar
    cases?" is, by construction, a question about OTHER investigations,
    so a CURRENT investigation's own off-topic LIKELY/CONFIRMED root
    cause must not silently stand in for it either. Safe for the exact
    same reason CONFIGURATION/HOW_TO already were: a compound,
    genuinely-investigative question (e.g. "...and what should I check
    first?") never classifies as HISTORICAL_LOOKUP in the first place --
    ``classify_intent`` checks troubleshooting/root-cause phrasing
    first, so this can never reintroduce the exact regression that
    motivated switching this set away from raw ``contains_knowledge_
    question`` (see ``_is_purely_knowledge_shaped_question``'s own
    docstring)."""

    def _compose_best_knowledge_answer(self, strategy: "InvestigationStrategy", question: str) -> str | None:
        """Real-Corpus Answer Quality & Final Chat Hardening phase --
        the one shared precedence order (CONFIGURATION/HOW_TO planner
        -> HISTORICAL_LOOKUP planner -> generic knowledge synthesis)
        every knowledge-shaped call site in this class now uses, never
        three independently-maintained copies of the same ordering."""
        configuration_synthesis = self._compose_configuration_synthesis(strategy, question)
        if configuration_synthesis is not None:
            return configuration_synthesis
        historical_lookup_synthesis = self._compose_historical_lookup_synthesis(strategy, question)
        if historical_lookup_synthesis is not None:
            return historical_lookup_synthesis
        return self._compose_knowledge_synthesis(strategy, question)

    def _off_topic_tier_override(
        self,
        strategy: "InvestigationStrategy",
        question: str,
        log_evidence: "list[Evidence] | None",
        context_text: str = "",
    ) -> tuple[str, str | None, str | None] | None:
        """Real-Corpus Answer Quality & Final Chat Hardening phase --
        the single place both the CONFIRMED and LIKELY branches above
        call once ``_tier_answer_is_off_topic`` has already determined
        the tier's own root cause/resolution shares no real subject
        with the question. Tries whichever REAL composer actually fits
        the question's own intent -- a purely knowledge-shaped question
        gets ``_compose_best_knowledge_answer``; a troubleshooting-
        synthesis-shaped question (including the real corpus finding
        this phase fixed, "What happens if X is wrong?") gets
        ``_compose_troubleshooting_synthesis`` with ``force=True``,
        bypassing that method's own LIKELY/CONFIRMED tier guard --
        exactly the escape hatch that guard's own docstring now
        documents, never a silent bypass. Returns ``None`` (caller
        keeps the original tier text) when neither applies or neither
        composer finds anything to say."""
        if self._is_purely_knowledge_shaped_question(question):
            synthesis = self._compose_best_knowledge_answer(strategy, question)
            if synthesis is not None:
                return synthesis, None, "knowledge"
        elif contains_troubleshooting_synthesis_question(question) or contains_troubleshooting_question(question):
            synthesis = self._compose_troubleshooting_synthesis(
                strategy, question, log_evidence, force=True, context_text=context_text
            )
            if synthesis is not None:
                return synthesis, None, None
        return None

    def _off_topic_disclosure_applies(self, question: str) -> bool:
        """Final Support-Quality Pass, §8 -- scopes ``_honest_off_topic_
        disclosure`` to exactly the class of question the real corpus
        finding was in: a SHORT, subject-only question (``<=
        _SHORT_QUESTION_WORD_COUNT`` concept words), where ``_tier_
        answer_is_off_topic``'s majority-overlap rule is a precise
        signal (proven by that finding). A LONGER, compound question
        (e.g. "What is the root cause, has this happened before, and
        what should I check first?") only ever uses the cruder bare-
        ``overlap > 0`` rule there, which was already known to false-
        positive on legitimate on-topic answers phrased in different
        vocabulary (see ``_SHORT_QUESTION_WORD_COUNT``'s own docstring
        and ``test_troubleshooting_synthesis_never_overrides_a_real_
        likely_tier``) -- before this phase, that false positive was
        harmless because the caller's fallback was a silent no-op
        (return the tier text unchanged regardless). Turning that
        fallback into an active honest-disclosure rewrite would convert
        a previously-harmless false positive into a real regression for
        long questions, so it is deliberately left out of scope here:
        only short questions, where the signal is trustworthy, get the
        new disclosure behavior."""
        return len(extract_concept_words(question)) <= self._SHORT_QUESTION_WORD_COUNT

    def _honest_off_topic_disclosure(
        self,
        strategy: "InvestigationStrategy",
        subject: str | None,
        primary: "ResolutionCandidate | None" = None,
    ) -> tuple[str, str | None, str | None]:
        """Final Support-Quality Pass, §8 -- the fallback both CONFIRMED
        and LIKELY branches now use when ``_tier_answer_is_off_topic``
        has already determined the tier's own root cause/resolution is
        off-topic AND ``_off_topic_tier_override`` found no real
        composer to substitute (neither a knowledge-shaped question nor
        a troubleshooting-synthesis-shaped one, or that composer's own
        majority-overlap bar found nothing either). Before this fix,
        that combination fell through to ``return text, None, None`` --
        silently presenting the SAME off-topic tier text this method's
        own caller had just determined was wrong, exactly the real
        corpus finding this phase's §8 investigation surfaced ("What
        happens if process settings are wrong?" answered with an
        unrelated IIS known bug, restated as fact, purely because no
        on-topic composer happened to have anything better to say).
        An honest "I don't have enough evidence" -- naming the
        off-topic record transparently rather than hiding it -- is
        always preferable to confidently restating an answer already
        known to be unrelated (this phase's closing principle: the
        system must never fabricate relevance merely because retrieval
        returned *something*).

        ``primary`` (when available) supplies a clean, human-readable
        ``EvidenceReference.title`` for the disclosure -- ``subject``
        itself is often ``primary.text``, the FULL templated resolution
        string (e.g. 'Per known bug "X": <full remediation text>'),
        which reads badly re-quoted inside this sentence. Falls back to
        ``subject`` verbatim only when no ``primary`` is available."""
        # Deliberately single-quoted, never double-quoted: grounding_
        # validator.check_no_material_loss's _QUOTED_RE treats every
        # double-quoted title in the deterministic answer as a
        # load-bearing source citation an LLM candidate must repeat
        # verbatim -- exactly backwards here, since this sentence exists
        # to say the record is NOT actually relevant. Double-quoting it
        # anyway (an earlier version of this fix did) turned that
        # explicitly-irrelevant title into a phantom required fact and
        # broke unrelated LLM-acceptance tests.
        display_name = primary.evidence.title if primary is not None else subject
        if display_name:
            text = (
                f"I don't have enough evidence in the current knowledge base to answer this specific question. "
                f"The most relevant record found, '{display_name}', does not actually address what was asked, so "
                f"presenting it as the answer would be misleading."
            )
        else:
            text = "I don't have enough evidence in the current knowledge base to answer this specific question."
        return text, self._unknown_follow_up(strategy), None

    @staticmethod
    def _is_purely_knowledge_shaped_question(question: str) -> bool:
        """Real-Corpus Answer Quality & Final Chat Hardening phase --
        the CONFIRMED/LIKELY off-topic override (``_tier_answer_is_
        off_topic``) needs a STRICTER test than ``_is_knowledge_shaped_
        question``: a real regression this phase's own first attempt
        caused, found by the existing test suite, not guessed --
        ``contains_knowledge_question`` matches on ANY substring phrase
        (e.g. "has this happened before"), so a genuinely compound,
        primarily-investigative question like "What is the root cause,
        has this happened before, and what should I check first?" also
        satisfies it, even though ``classify_intent`` itself -- whose
        OWN ordering checks troubleshooting/root-cause phrasing BEFORE
        ever considering a knowledge phrase match -- correctly resolves
        that exact same question to TROUBLESHOOTING, never a knowledge
        intent. Using raw ``contains_knowledge_question`` for the
        override let it fire for that compound question's real,
        legitimate LIKELY-tier answer (a genuine root cause that simply
        doesn't share words with "root cause"/"check first" themselves)
        and incorrectly replace it with a weaker knowledge-synthesis
        attempt. This method instead trusts ``classify_intent``'s own,
        already-ordered precedence completely: True only when the
        question's SINGLE resolved intent is itself one of the five
        purely-informational/configuration intents, never merely
        "contains a knowledge phrase somewhere"."""
        return classify_intent(question) in ChatOrchestrator._PURE_KNOWLEDGE_INTENTS

    def _compose_answer(
        self,
        strategy: "InvestigationStrategy",
        question: str = "",
        log_evidence: "list[Evidence] | None" = None,
        investigation: "InvestigationSession | None" = None,
    ) -> tuple[str, str | None, str | None]:
        """Returns ``(answer_text, follow_up, answer_kind)``.
        ``answer_kind`` is ``"log_analysis"`` when ``answer_text`` came
        from ``_compose_log_analysis_answer`` (real, uploaded log
        content -- see that method's own docstring), ``"knowledge"`` when
        it came from ``_compose_knowledge_synthesis`` instead of the
        tier-based boilerplate below, and ``None`` for the original,
        unmodified tier-based text -- see ``ChatResponse.answer_kind``'s
        docstring for how callers should use it (never a change to
        ``resolution_provenance``'s own semantics).

        Log-related questions are checked FIRST, ahead of the tier
        switch below, and fire regardless of tier: "analyze this log"/
        "show me the timeline"/"what errors do you see" are a
        fundamentally different question type than "what is the root
        cause" (Chat + Log Intelligence integration, §2) -- the tier-
        based text, even at LIKELY/CONFIRMED, has no way to answer them
        at all, since it was never designed to describe a timeline or
        list specific log lines. Comparison/L2/L3 are checked before the
        general log-analysis composer since they are more specific
        question shapes (L2/L3 Investigation Copilot phase, §7/11/12)."""
        if log_evidence:
            if len(log_evidence) >= 2 and contains_log_comparison_question(question):
                comparison = self._compose_log_comparison_answer(log_evidence, strategy)
                if comparison is not None:
                    return comparison, None, "log_analysis"
            if contains_l2_task_note_question(question):
                notes = self._compose_l2_task_notes(log_evidence, strategy)
                if notes is not None:
                    return notes, None, "log_analysis"
            if contains_l3_escalation_question(question):
                escalation = self._compose_l3_escalation_summary(log_evidence, strategy)
                if escalation is not None:
                    return escalation, None, "log_analysis"
            if contains_log_analysis_question(question) or contains_log_comparison_question(question):
                # A comparison-phrased question with fewer than two real
                # files (the branch above's own gate) still deserves a
                # real answer about the one log that IS present, never a
                # bare "insufficient evidence" -- see §7's own note that
                # multi-log handling is additive, not a replacement for
                # the single-log case.
                log_synthesis = self._compose_log_analysis_answer(log_evidence, strategy, investigation)
                if log_synthesis is not None:
                    return log_synthesis, None, "log_analysis"

        # Final Support-Quality Pass, §2 -- L2_GUIDANCE ("What should
        # L2 check?") is checked here, ahead of the tier switch below,
        # for the same reason log-related questions already are: it is
        # a fundamentally different question type than "what is the
        # root cause," and the tier-based text (even at LIKELY/
        # CONFIRMED, and even when that tier's own root cause is
        # off-topic) has no way to answer it. _compose_l2_guidance_
        # synthesis never returns None once it confirms L2_GUIDANCE
        # classification -- an honest "insufficient evidence" answer is
        # itself a real, complete answer, never a signal to keep
        # searching for a different composer.
        l2_guidance = self._compose_l2_guidance_synthesis(strategy, question, investigation)
        if l2_guidance is not None:
            return l2_guidance, None, "l2_guidance"

        structured = strategy.structured_resolution
        if structured is None:
            text, follow_up = self._compose_answer_without_structured_resolution(strategy)
            return text, follow_up, None

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
            if self._tier_answer_is_off_topic(
                question, subject, investigation.context_text if investigation is not None else "",
                is_retrieval_derived=not (
                    structured.root_cause
                    and structured.root_cause_evidence
                    and structured.root_cause_evidence[0].kind == EvidenceKind.ENTITY_HEURISTIC
                ),
            ):
                override = self._off_topic_tier_override(
                    strategy, question, log_evidence, investigation.context_text if investigation is not None else ""
                )
                if override is not None:
                    return override
                if self._off_topic_disclosure_applies(question):
                    return self._honest_off_topic_disclosure(strategy, subject, primary)
            return text, None, None

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
            if self._tier_answer_is_off_topic(
                question, subject, investigation.context_text if investigation is not None else "",
                is_retrieval_derived=not (
                    structured.root_cause
                    and structured.root_cause_evidence
                    and structured.root_cause_evidence[0].kind == EvidenceKind.ENTITY_HEURISTIC
                ),
            ):
                override = self._off_topic_tier_override(
                    strategy, question, log_evidence, investigation.context_text if investigation is not None else ""
                )
                if override is not None:
                    return override
                if self._off_topic_disclosure_applies(question):
                    return self._honest_off_topic_disclosure(strategy, subject, primary)
            return text, None, None

        # POSSIBLE and UNKNOWN both fall through to the tier boilerplate
        # below UNLESS this is a real knowledge/historical question (§2.A/
        # B of the Chat Knowledge-Synthesis feature) AND real, already-
        # retrieved documentation/historical/known-bug/TFS/Wiki evidence
        # actually exists to cite -- see _compose_knowledge_synthesis's own
        # docstring for the full rationale. CONFIRMED/LIKELY above are
        # completely untouched: when a real root cause/resolution already
        # exists, the existing tier-based text already is the right,
        # specific, grounded answer -- this only fires for the case the
        # real bug report was about, where that text would otherwise be
        # uselessly generic despite real evidence sitting unused.
        # Knowledge Answering & Evidence Synthesis phase -- widened to
        # also fire for CONFIGURATION/HOW_TO-shaped questions ("Where
        # do I configure it?"), a real gap this phase's own follow-up
        # test found: contains_knowledge_question's closed phrase list
        # (built for "tell me about"/"explain"/"what is" phrasing) never
        # recognized a bare configuration/how-to request at all, so a
        # perfectly reasonable follow-up fell all the way through to
        # the generic "I don't have enough evidence" tier text instead
        # of the real documentation this exact topic already has.
        # classify_intent is the single source of truth for this
        # distinction (never a second, independently-maintained check).
        if self._is_knowledge_shaped_question(question):
            # Real-Corpus Answer Quality & Final Chat Hardening phase,
            # §10 -- CONFIGURATION/HOW_TO and HISTORICAL_LOOKUP each get
            # their own, more specific answer-planner template before
            # falling back to the generic knowledge-synthesis shape;
            # each returns None (never partially renders) whenever the
            # question isn't its own intent or nothing clears the
            # relevance bar, so this is a pure, safe precedence
            # ordering, never a behavior change for anything that
            # doesn't match.
            synthesis = self._compose_best_knowledge_answer(strategy, question)
            if synthesis is not None:
                return synthesis, None, "knowledge"

        # Final Hardening Pass, Objective 2 -- "why did this fail?"-style
        # questions get the richer ranked-hypothesis answer instead of
        # the single-paragraph tier boilerplate below, when real
        # candidate evidence exists. answer_kind stays None (unlike the
        # knowledge/log-analysis branches) deliberately: this answer is
        # still fundamentally a POSSIBLE/UNKNOWN-tier investigative
        # answer -- the normal confidence badge/rationale caption should
        # keep showing, per Objective 2D's "preserve existing confidence
        # semantics."
        #
        # Real-Corpus Answer Quality & Final Chat Hardening phase --
        # widened to also fire for ``contains_troubleshooting_question``
        # ("what should I check"/"what should I check next"), a real
        # gap the real seeded corpus exposed: "What should I check
        # next?" found real, on-topic evidence (sufficiency STRONG) yet
        # still fell through to the bare "A possible explanation may
        # exist..." boilerplate, because only the EXPLANATION-shaped
        # phrase list (``contains_troubleshooting_synthesis_question``)
        # was wired here -- the ACTION-shaped phrase list (``contains_
        # troubleshooting_question``, otherwise used only for LLM-
        # prompt clause-splitting) was not. This composer's own "## What
        # to check next" section already exists specifically to answer
        # this question type. Both phrase lists are still mutually
        # exclusive (see troubleshooting_synthesis_question.py's own
        # docstring), so this is a pure OR-widening, never a behavior
        # change for a question already matching the first list. When
        # zero evidence-backed checks exist, ``_prepare_question`` has
        # already stripped this clause out of ``question`` upstream
        # (see ``_generate_answer``), so this widening only ever
        # activates when the clause is still genuinely present in the
        # text reaching this method.
        if contains_troubleshooting_synthesis_question(question) or contains_troubleshooting_question(question):
            synthesis = self._compose_troubleshooting_synthesis(
                strategy, question, log_evidence, context_text=investigation.context_text if investigation is not None else ""
            )
            if synthesis is not None:
                return synthesis, None, None

        if tier == ResolutionProvenance.POSSIBLE:
            text = f"A possible explanation is: {subject}." if subject else "A possible explanation may exist, but the evidence found is limited."
            text += (
                " Evidence is not yet sufficient to verify this -- treat it as a hypothesis to check, not a"
                " resolution to act on."
            )
            return text, None, None

        # UNKNOWN
        return "I don't have enough evidence to determine the cause.", self._unknown_follow_up(strategy), None

    _KNOWLEDGE_SYNTHESIS_MIN_SCORE = 0.35
    """Same value and rationale as Settings.min_similarity_for_root_cause
    (app/config.py) -- reused here, not re-derived, as the "is this match
    actually relevant, not just whatever Chroma's top_k happened to
    return" cutoff for citing Documentation/Historical Investigation
    matches, both real-tested (see below) to be strong, genuinely
    on-topic signals even at this permissive bar. Not wired to the live
    Settings object (ChatOrchestrator holds no Settings reference) -- a
    local constant keeps this change contained to one file rather than
    threading a new constructor dependency through every existing
    caller/test; revisit together if that threshold is ever retuned."""

    _KNOWLEDGE_SYNTHESIS_MIN_SCORE_SECONDARY = 0.6
    """A real live test ("tell me about dashboard in CC" against the
    real seeded sample corpus) found ``_KNOWLEDGE_SYNTHESIS_MIN_SCORE``
    too permissive specifically for Known Bugs/TFS/Wiki: Documentation
    and Historical Investigation matches were genuinely on-topic at
    0.68-0.77, but the two Known Bug matches that also cleared 0.35
    (0.54, 0.51 -- "Collector command queue stalls...",  "Kafka client
    rebalance storm...") were topically unrelated noise, an artifact of
    this sample corpus having very few Known Bug records at all so
    *something* is always nearest. A stricter bar for these three,
    lower-precedence categories keeps the synthesis's most prominent
    claims (documentation, historical context) reliably strong while
    only citing a known bug or TFS/Wiki match when it clears the same
    genuinely-on-topic range Documentation/Historical already showed."""

    def _compose_knowledge_synthesis(self, strategy: "InvestigationStrategy", question: str) -> str | None:
        """Chat Knowledge-Synthesis feature -- deterministic informational-
        answer composition from evidence ``RecommendationEngine.generate()``
        ALREADY retrieved (``strategy.documentation``/``historical_
        investigations``/``known_bugs``/``tfs_matches``/``wiki_matches``) --
        never a new retrieval call, never LLM-generated, never a paraphrase:
        every sentence either directly quotes a real ``KnowledgeMatch``/
        ``ExternalMatch`` title+snippet or lists real titles verbatim.
        Returns ``None`` (caller falls back to the existing tier-based
        boilerplate, unchanged) when nothing retrieved clears the
        relevance bar -- this function has no opinion about WHETHER the
        question deserves a knowledge answer, only WHAT to say once
        ``_compose_answer`` has already decided it does.

        RETRIEVAL SIMILARITY IS NOT ANSWER RELEVANCE (real live finding):
        "what is process setting in emerge" retrieved a personal task-list
        export ("task") as Chroma's single highest-scoring Documentation
        candidate (69% similarity) -- "task" shares no real subject with
        "process setting"/"emerge" at all, while a genuinely on-topic
        Historical Investigation ("Review Emerge Settings listed in
        CIL-98-3114 -- settings id 1126 missing in Emerge System Settings
        page") was sitting right there, unused, because the old version of
        this method always cited ``documentation[0]`` -- whichever
        Documentation candidate scored highest -- never comparing it
        against Historical Investigation candidates, and never checking
        whether it actually shares the question's own subject at all. This
        method now re-ranks the combined Documentation + Historical
        Investigation candidate pool by REAL LEXICAL OVERLAP with the
        question's own content words (``extract_concept_words``/
        ``lexical_overlap`` -- see that module's docstring) FIRST, semantic
        score only as a tiebreak, and degrades to an explicit, honest
        "couldn't find documentation that specifically covers this"
        admission (Step 9 of the Chat Intelligence Upgrade) rather than
        ever presenting a zero-overlap, merely-highest-scoring candidate as
        if it answered the question. Known Bugs/TFS/Wiki remain governed
        by the stricter secondary score bar alone (no title-based
        definition claim is ever made from those three categories, so the
        lexical re-rank doesn't apply there) -- unchanged from before.

        Deliberately excludes ``resolution_candidates``/``validation_
        steps``/``root_cause`` entirely -- an informational answer is not
        a troubleshooting or investigation answer, and must never present
        (or imply) either, keeping Rule 9's troubleshooting-safety
        contract and Rule 4's confidence-tier contract completely out of
        this code path's reach, by construction, not by extra checking."""
        min_score = self._KNOWLEDGE_SYNTHESIS_MIN_SCORE
        min_score_secondary = self._KNOWLEDGE_SYNTHESIS_MIN_SCORE_SECONDARY
        doc_matches = [m for m in strategy.documentation if m.score >= min_score]
        hist_matches = [m for m in strategy.historical_investigations if m.score >= min_score]
        bug_matches = [m for m in strategy.known_bugs if m.score >= min_score_secondary]
        tfs = strategy.tfs_matches
        wiki = strategy.wiki_matches
        tfs_matches = [m for m in (tfs.matches if tfs is not None and tfs.available else []) if m.score >= min_score_secondary]
        wiki_matches = [m for m in (wiki.matches if wiki is not None and wiki.available else []) if m.score >= min_score_secondary]

        if not (doc_matches or hist_matches or bug_matches or tfs_matches or wiki_matches):
            return None

        concept_words = extract_concept_words(question)
        # Evidence-Centered Knowledge Retrieval & Synthesis phase, §4:
        # this tiebreak now comes from the real, named, per-intent
        # RETRIEVAL_PROFILES table (retrieval_profile.py) instead of a
        # private two-entry dict local to this method -- the same
        # table also governs _compose_troubleshooting_synthesis below
        # and is independently unit-tested (tests/test_retrieval_
        # profile.py). Every profile this codebase defines ranks
        # "documentation"/"wiki" above "historical", so this substitution
        # is behavior-preserving for this composer by construction, not
        # by coincidence -- see RETRIEVAL_PROFILES's own docstring.
        _KIND_PRIORITY = kind_priority(RETRIEVAL_PROFILES.get(classify_intent(question), RETRIEVAL_PROFILES[AnswerIntent.UNKNOWN]))
        """(candidate, kind, overlap) for every Documentation/Historical
        candidate, ranked by real subject overlap first, score second --
        never the other way around (see docstring above). Knowledge
        Answering & Evidence Synthesis phase, §5: a historical case may
        prove something HAPPENED; it does not establish what a
        product/concept IS. Real subject overlap still decides first (a
        doc that only shares one word can still lose to a historical
        case that shares two, exactly as the existing "process setting
        in emerge"/"dashboard CC" regressions already require and this
        tiebreak never overrides) -- this only breaks a genuine TIE in
        overlap, in which case authoritative documentation is preferred
        over a historical incident, never the reverse."""
        ranked: list[tuple["KnowledgeMatch", str, int]] = sorted(
            (
                [(m, "documentation", lexical_overlap(concept_words, f"{m.title} {m.snippet[:500]}")) for m in doc_matches]
                + [(m, "historical", lexical_overlap(concept_words, f"{m.title} {m.snippet[:500]}")) for m in hist_matches]
            ),
            key=lambda item: (item[2], _KIND_PRIORITY[item[1]], item[0].score),
            reverse=True,
        )

        # Knowledge Answering & Evidence Synthesis phase -- restructured
        # into "## Answer / ## Why I'm saying this / ## Relevant
        # evidence / ## What is not confirmed", replacing the earlier
        # "Direct answer/Sources/Notes" labels with the phase's own
        # requested headers. The real, more important fix (§5 of this
        # phase) is semantic, not cosmetic: a HISTORICAL match may still
        # legitimately win here (unchanged ranking -- see the real
        # "process setting in emerge"/"dashboard CC" regressions this
        # re-ranking already fixed), but presenting one as if it
        # answered a genuine "what IS this" question, with no
        # acknowledgment that it is a past incident and not a
        # definition, is exactly this project's own real reported
        # failure ("a historical incident mentioning AxeI does not
        # define what an AxeI meter is"). Every "## Relevant evidence"
        # line now says what its source actually establishes (a
        # definition vs. merely "this was observed/investigated"), and
        # "## What is not confirmed" explicitly names that gap whenever
        # a historical case stands in for an entity/product/concept
        # definition question (``is_definitional_question``) with no
        # authoritative documentation behind it.
        direct_answer: str
        why_line: str
        primary_match, primary_kind = None, None
        if ranked:
            top_match, top_kind, top_overlap = ranked[0]
            # Deliberately NOT "or top_match.score >= some bar" -- the
            # real motivating bug ("task", 69% similarity) is itself
            # proof that a high raw semantic score is not a safe
            # override here. Only a real word-overlap match, or a
            # question with no extractable subject words to compare
            # against at all, counts as answerable.
            #
            # Acceptance review finding -- a bare "any overlap" bar is
            # not enough when the product's own name ("Command"/
            # "Center") is one of the question's concept words: it is so
            # ubiquitous across this corpus that nearly every document
            # shares it, so 1-2 of 4 words (never a majority) let an
            # unrelated document ("IAD Move Checklist Answers_Master")
            # through for "Explain process settings in Command Center."
            # Reuses the same majority-overlap, word-boundary-aware
            # primitives ``_tier_answer_is_off_topic``'s knowledge-shaped
            # branch already applies one layer up -- both must agree on
            # what "really shares the subject" means, never two
            # independently-calibrated notions of it.
            # Same word-count floor as _tier_answer_is_off_topic's sibling
            # fix, and for the identical reason: "Tell me about dashboard
            # in CC" (2 concept words) shares only "dashboard" (the real,
            # distinctive one) with "Access to Dashboard and Views in
            # CRM" -- a genuinely on-topic match a strict 2-of-2 majority
            # requirement would wrongly reject. 1-2 word questions keep
            # the original bare-overlap>0 rule; only 3+ word questions
            # (where the real "process settings"/"IIS worker PROCESS
            # crash" finding lived) require a real majority.
            if len(concept_words) >= 3:
                words = self._distinctive_concept_words(concept_words)
                top_word_overlap = word_overlap(words, f"{top_match.title} {top_match.snippet[:500]}")
                answerable = bool(words) and has_majority_overlap(top_word_overlap, len(words))
            else:
                answerable = top_overlap > 0 or not concept_words
            if answerable:
                primary_match, primary_kind = top_match, top_kind
                excerpt = truncate_extract(primary_match.snippet, 240)
                if primary_kind == "documentation":
                    direct_answer = f'Based on ResolveIQ\'s documentation "{primary_match.title}": {excerpt}'
                    why_line = f'ResolveIQ\'s documentation "{primary_match.title}" directly addresses this question.'
                else:
                    direct_answer = f'ResolveIQ has a related historical case on record, "{primary_match.title}": {excerpt}'
                    why_line = (
                        f"No authoritative documentation cleared the relevance bar for this question -- the most "
                        f'relevant record found is a historical case, "{primary_match.title}".'
                    )
            else:
                # Real evidence exists but none of it actually shares the
                # question's own subject -- an honest admission, never a
                # confident-sounding answer built from a coincidentally
                # highest-scoring but unrelated candidate.
                subject = " ".join(concept_words) if concept_words else "this"
                direct_answer = (
                    f"I found some ResolveIQ content that scored as semantically similar to \"{subject}\", but none of "
                    f"it actually shares that subject -- I couldn't find documentation or a historical case that "
                    f"specifically covers \"{subject}\"."
                )
                why_line = f'No retrieved evidence actually shares the real subject of "{subject}" above the relevance bar.'
        else:
            direct_answer = ""  # unreachable in practice: the caller already required at least one match to get here
            why_line = ""

        source_lines: list[str] = []
        if primary_match is not None:
            if primary_kind == "documentation":
                source_lines.append(f'- "{primary_match.title}" (documentation) -- directly addresses this question.')
            else:
                source_lines.append(
                    f'- "{primary_match.title}" (historical case) -- shows this subject was observed/investigated; '
                    f"not a product/concept definition."
                )
        for m in doc_matches:
            if m is not primary_match:
                source_lines.append(f'- "{m.title}" (documentation) -- related documentation.')
        for m in hist_matches[:3]:
            if m is not primary_match:
                source_lines.append(f'- "{m.title}" (historical case) -- a related case on record, not a definition.')
        for m in bug_matches[:2]:
            source_lines.append(f'- "{m.title}" (known bug) -- a related known bug on record, not a confirmed root cause.')
        for m in tfs_matches[:2]:
            if m.tfs_case is not None:
                source_lines.append(f'- "{m.tfs_case.title}" (live TFS) -- a related work item.')
        for m in wiki_matches[:2]:
            if m.wiki_page is not None:
                source_lines.append(f'- "{m.wiki_page.title}" (live Wiki) -- a related Wiki page.')

        not_confirmed_lines = [
            "This is informational context assembled from ResolveIQ's knowledge base -- not a validated root "
            "cause or resolution."
        ]
        if primary_kind == "historical" and is_definitional_question(question):
            not_confirmed_lines.append(
                f'"{primary_match.title}" is a historical case, not authoritative documentation -- ResolveIQ does '
                f"not have documentation on record that specifically defines this."
            )
        # Evidence-Centered Knowledge Retrieval & Synthesis phase, §6: a
        # historical case's own recorded resolution/next_step is a
        # HISTORICAL_RECOMMENDATION, never a confirmed current
        # resolution -- the anti-pattern this explicitly avoids is
        # rendering it as "Changing X is the solution." Additive only:
        # both regression fixtures for this composer pass resolution=""/
        # next_step="" deliberately (to stay at POSSIBLE/UNKNOWN tier),
        # so this never fires for them.
        if primary_kind == "historical" and (primary_match.metadata.get("resolution") or primary_match.metadata.get("next_step")):
            not_confirmed_lines.append(
                f'"{primary_match.title}" has a recorded resolution/next step from that past case -- this is a '
                f"historical recommendation, not a confirmed resolution for the current situation; it has not been "
                f"independently verified here."
            )

        # §8: bounded, practical contradiction detection over the same
        # Documentation/Wiki pool this answer already cites -- built
        # from the real EvidenceBundle (retrieval_profile.py), never a
        # second, divergent detector. Additive: fires only when two
        # AUTHORITATIVE_* sources genuinely disagree on a version token
        # or a "key = value" statement (see detect_contradictions's own
        # docstring for exactly what is checked) -- never for the
        # existing fixtures, which contain no such conflicting content.
        context_for_bundle = build_query_context(question)
        bundle = build_evidence_bundle(
            question, context_for_bundle, strategy, min_score=min_score, min_score_secondary=min_score_secondary
        )
        conflict_lines: list[str] = []
        if bundle.contradictions:
            conflict_lines.append("ResolveIQ found conflicting evidence:")
            for c in bundle.contradictions:
                conflict_lines.append(f'- {c.description} "{c.source_a}" says: {c.claim_a}. "{c.source_b}" says: {c.claim_b}.')

        sections = [f"## Answer\n{direct_answer}"]
        if why_line:
            sections.append(f"## Why I'm saying this\n{why_line}")
        if source_lines:
            sections.append("## Relevant evidence\n" + "\n".join(source_lines))
        sections.append("## What is not confirmed\n" + "\n".join(not_confirmed_lines))
        if conflict_lines:
            sections.append("## Evidence conflict\n" + "\n".join(conflict_lines))
        return "\n\n".join(sections)

    def _compose_configuration_synthesis(self, strategy: "InvestigationStrategy", question: str) -> str | None:
        """Real-Corpus Answer Quality & Final Chat Hardening phase, §10
        -- a dedicated answer planner for CONFIGURATION/HOW_TO
        questions ("Where do I configure X?", "How do I configure X?"),
        replacing the generic knowledge-synthesis "## Answer/## Why I'm
        saying this/## Relevant evidence" shape with the requested
        "## Answer/## Steps/## Important conditions/## Version-specific
        notes/## Evidence/## What is not confirmed" structure. Built
        entirely from the same, already-computed ``EvidenceBundle``
        (``retrieval_profile.build_evidence_bundle``) the knowledge
        composer and ``ChatResponse.debug`` both use -- never a new
        retrieval call, never an LLM, never a paraphrase: "## Steps"
        quotes the primary source's own excerpt verbatim rather than
        inventing a numbered procedure from unstructured text (this
        codebase's own "never fabricate structure that isn't really
        there" discipline). Returns ``None`` (caller falls back to the
        existing, unmodified ``_compose_knowledge_synthesis``) whenever
        the question is not CONFIGURATION/HOW_TO-classified, or nothing
        retrieved clears the relevance bar."""
        context = build_query_context(question)
        if context.intent not in (AnswerIntent.CONFIGURATION, AnswerIntent.HOW_TO):
            return None
        bundle = build_evidence_bundle(
            question, context, strategy,
            min_score=self._KNOWLEDGE_SYNTHESIS_MIN_SCORE, min_score_secondary=self._KNOWLEDGE_SYNTHESIS_MIN_SCORE_SECONDARY,
        )
        primary_pool = bundle.authoritative_documentation or bundle.documentation
        if not primary_pool and not bundle.historical_case_evidence:
            return None

        # Emerge acceptance-matrix finding -- documentation was preferred
        # over a historical case UNCONDITIONALLY, regardless of which one
        # actually shares the question's real subject. Real example: "How
        # do I start the services on Emerge?" cited "GSIS v3_5_0+ Post
        # Install Configuration Checklist" (2 of 3 concept words, 0.67
        # score, no "Emerge" anywhere in it) over "Start the services on
        # Emerge." (3 of 3 concept words, 0.90 score) -- an unrelated
        # system's checklist, purely because it happened to be a
        # DOCUMENTATION record and the correct answer was a HISTORICAL
        # one. Not a generic-word collision this composer's existing
        # relevance bar already guards against (GSIS clears the bar on
        # its own 2-word overlap) -- a cross-source-TYPE ordering bug,
        # independent of any product's governance status. Only overrides
        # the existing, already-tested documentation-first default when
        # the historical case's own top candidate shares STRICTLY MORE of
        # the question's real subject words (word-boundary, not
        # substring) than the chosen documentation candidate does, and
        # clears the same majority bar every other relevance check in
        # this file already requires -- a tie or documentation-ahead
        # keeps today's behavior exactly as it is, so a genuinely
        # authoritative, well-matching document is never displaced by a
        # merely-as-good historical case.
        concept_words = extract_concept_words(question)
        if primary_pool and bundle.historical_case_evidence and concept_words:
            doc_top, hist_top = primary_pool[0], bundle.historical_case_evidence[0]
            doc_overlap = word_overlap(concept_words, f"{doc_top.title} {doc_top.excerpt}")
            hist_overlap = word_overlap(concept_words, f"{hist_top.title} {hist_top.excerpt}")
            if hist_overlap > doc_overlap and has_majority_overlap(hist_overlap, len(concept_words)):
                primary_pool = bundle.historical_case_evidence

        primary = primary_pool[0] if primary_pool else bundle.historical_case_evidence[0]
        is_authoritative = primary in bundle.authoritative_documentation

        if is_authoritative:
            answer = f'Based on ResolveIQ\'s documentation "{primary.title}": {truncate_extract(primary.excerpt, 240)}'
        else:
            answer = (
                f'ResolveIQ does not have authoritative documentation on record for this configuration question -- '
                f'the most relevant record found is {"documentation" if primary.source_type == "documentation" else "a historical case"}, '
                f'"{primary.title}": {truncate_extract(primary.excerpt, 240)}'
            )

        steps_lines = [f'- Per "{primary.title}": {truncate_extract(primary.excerpt, 300)}']
        for item in primary_pool[1:3]:
            steps_lines.append(f'- Also see "{item.title}": {truncate_extract(item.excerpt, 200)}')

        conditions_lines: list[str] = []
        if context.customer:
            conditions_lines.append(f"- Customer context: {context.customer}")
        if context.region:
            conditions_lines.append(f"- Region context: {context.region}")
        if not conditions_lines:
            conditions_lines.append("- No customer/region-specific conditions are documented for this configuration.")

        if context.version or context.technology:
            version_line = "; ".join(v for v in (context.version, context.technology) if v)
            version_lines = [f"- {version_line}"]
        else:
            version_lines = ["- No version-specific documentation was found for this configuration."]

        evidence_lines = [f'- "{primary.title}" ({primary.source_type}) -- {primary.establishes}']
        for item in (bundle.authoritative_documentation + bundle.documentation + bundle.historical_case_evidence)[:5]:
            if item is not primary:
                evidence_lines.append(f'- "{item.title}" ({item.source_type}) -- {item.establishes}')

        not_confirmed_lines = [
            "This configuration guidance is assembled from ResolveIQ's knowledge base -- not a validated resolution "
            "for a specific incident."
        ]
        if not is_authoritative:
            not_confirmed_lines.append(
                f'"{primary.title}" is {"documentation with only partial overlap" if primary.source_type == "documentation" else "a historical case"}, '
                f"not authoritative configuration documentation -- treat this as a lead to verify, not a confirmed step."
            )
        if bundle.contradictions:
            not_confirmed_lines.append("ResolveIQ found conflicting documented configuration values -- see below.")

        sections = [
            f"## Answer\n{answer}",
            "## Steps\n" + "\n".join(steps_lines),
            "## Important conditions\n" + "\n".join(conditions_lines),
            "## Version-specific notes\n" + "\n".join(version_lines),
            "## Evidence\n" + "\n".join(evidence_lines),
            "## What is not confirmed\n" + "\n".join(not_confirmed_lines),
        ]
        if bundle.contradictions:
            conflict_lines = ["ResolveIQ found conflicting evidence:"]
            for c in bundle.contradictions:
                conflict_lines.append(f'- {c.description} "{c.source_a}" says: {c.claim_a}. "{c.source_b}" says: {c.claim_b}.')
            sections.append("## Evidence conflict\n" + "\n".join(conflict_lines))
        return "\n\n".join(sections)

    _HISTORICAL_LOOKUP_MAX_CASES = 3
    """Same capping discipline as every other ranked list in this
    module -- the strongest few cases, not an unbounded dump."""

    def _compose_historical_lookup_synthesis(self, strategy: "InvestigationStrategy", question: str) -> str | None:
        """Real-Corpus Answer Quality & Final Chat Hardening phase, §10
        -- a dedicated answer planner for HISTORICAL_LOOKUP questions
        ("Has this happened before?", "What was the resolution in
        similar cases?"), replacing the generic knowledge-synthesis
        shape with "## Similar cases found/## What this tells us/## What
        it does not establish" -- each case explicitly flagged
        historical-recommendation-vs-current-resolution (§9's own
        anti-pattern: never silently promoted to a confirmed fix).
        Built entirely from the same ``EvidenceBundle`` every other
        composer uses. Returns ``None`` when not HISTORICAL_LOOKUP-
        classified or nothing retrieved clears the relevance bar."""
        context = build_query_context(question)
        if context.intent != AnswerIntent.HISTORICAL_LOOKUP:
            return None
        bundle = build_evidence_bundle(
            question, context, strategy,
            min_score=self._KNOWLEDGE_SYNTHESIS_MIN_SCORE, min_score_secondary=self._KNOWLEDGE_SYNTHESIS_MIN_SCORE_SECONDARY,
        )
        if not bundle.historical_case_evidence:
            return None
        cases = bundle.historical_case_evidence[: self._HISTORICAL_LOOKUP_MAX_CASES]
        recommendation_titles = {
            c.supported_by[0]
            for c in bundle.claims
            if c.category == "recommendation" and c.supported_by
        }

        lines = ["## Similar cases found"]
        for idx, item in enumerate(cases, start=1):
            has_recommendation = item.title in recommendation_titles
            lines.append(f"\n### {idx}. {item.title}")
            lines.append(f"What happened: {truncate_extract(item.excerpt, 280)}")
            if has_recommendation:
                lines.append(
                    "Historical outcome: a resolution/next step was recorded for this past case -- this is a "
                    "HISTORICAL RECOMMENDATION, not a confirmed current resolution; it has not been independently "
                    "verified for this situation."
                )
            else:
                lines.append("Historical outcome: no recorded resolution/next step is on file for this case.")

        lines.append("\n## What this tells us")
        lines.append(
            f"ResolveIQ found {len(bundle.historical_case_evidence)} historical case(s) that share this subject -- "
            "these show the subject was previously observed/investigated."
        )

        lines.append("\n## What this does not establish")
        not_establish = [
            "None of these cases confirm the current situation has the same root cause -- similarity is not proof.",
        ]
        if recommendation_titles:
            not_establish.append(
                "A recorded historical resolution/next step is what a PAST case did, not a confirmed fix for the "
                "current situation."
            )
        lines.extend(f"- {line}" for line in not_establish)

        if bundle.contradictions:
            lines.append("\n## Evidence conflict")
            lines.append("ResolveIQ found conflicting evidence:")
            for c in bundle.contradictions:
                lines.append(f'- {c.description} "{c.source_a}" says: {c.claim_a}. "{c.source_b}" says: {c.claim_b}.')

        return "\n".join(lines)

    _L2_GUIDANCE_MAX_CHECKS = 3
    """Same capping discipline as every other ranked list in this
    module -- the strongest few checks, not an unbounded dump."""

    def _compose_l2_guidance_synthesis(
        self, strategy: "InvestigationStrategy", question: str, investigation: "InvestigationSession | None" = None
    ) -> str | None:
        """Final Support-Quality Pass, §2/§3/§4 -- a dedicated answer
        planner for L2_GUIDANCE questions ("What should L2 check?",
        "What should L2 verify?", ...), replacing the generic tier-
        based boilerplate ResolveIQ previously had NOTHING for (this
        intent classified UNKNOWN before this phase) with "## What is
        observed/## What L2 should check/## What this can establish/##
        What is not confirmed/## Next action". Built entirely from the
        same ``EvidenceBundle`` every other composer uses -- never a
        new retrieval call, never an LLM, never a paraphrase. Every
        recommended check is either a real, already-established
        evidence-backed action (``available_checks(structured)``) or a
        real citation to review ("Review \"<title>\"") -- never an
        invented generic action ("check logs, network, configuration").
        Returns ``None`` when the question is not L2_GUIDANCE-
        classified (caller falls through to other composers); returns
        an explicit, honest "I don't have enough evidence" answer
        (never ``None``) when L2_GUIDANCE-classified but nothing
        real supports a specific check -- this is a genuine answer,
        not a "no answer" signal, so ``answer_kind`` stays a real
        value and the caller never keeps searching for a different
        composer to paper over the gap.

        ``investigation`` (§4's own context-aware follow-up
        requirement) -- when given, its ``context_text`` (the
        conversation's own accumulated raw text) is passed to
        ``build_query_context`` so a genuinely thin follow-up like
        "What should L2 check?" (its own concept words are just
        "should"/"l2"/"check" -- none of which name a real subject)
        inherits the real prior subject/state ("AxeI meter"/
        "Discovered") the same anaphora-aware way every other follow-up
        in this codebase already does, rather than searching for
        evidence about "l2 check" itself. A fresh session's throwaway
        investigation has no such accumulated text, so New Chat still
        starts with nothing inherited."""
        context_text = investigation.context_text if investigation is not None else ""
        context = build_query_context(question, context_text=context_text)
        if context.intent != AnswerIntent.L2_GUIDANCE:
            return None
        structured = strategy.structured_resolution
        established_checks = available_checks(structured) if structured is not None else []
        # §4 continued -- L2_GUIDANCE_PHRASES are, by construction,
        # NEVER a real subject on their own ("should"/"l2"/"check" name
        # no real topic), so unlike a generic follow-up this composer
        # always merges in the prior turn's real concept words when any
        # exist, rather than relying on build_query_context's own
        # anaphora-gated merge (which correctly does NOT fire here --
        # there is no "it"/"this"/"that" in "What should L2 check?" for
        # it to key off). This only ever expands what evidence is
        # searched for; it never changes the L2_GUIDANCE classification
        # itself (already resolved above) or the literal question text
        # rendered in "## What is observed" below.
        concept_words = extract_concept_words(question)
        if context_text:
            context_words = extract_concept_words(context_text)
            merged = list(context_words)
            for word in concept_words:
                if word not in merged:
                    merged.append(word)
            concept_words = merged
        search_text = " ".join(concept_words) if concept_words else question
        bundle = build_evidence_bundle(
            search_text, context, strategy,
            min_score=self._KNOWLEDGE_SYNTHESIS_MIN_SCORE, min_score_secondary=self._KNOWLEDGE_SYNTHESIS_MIN_SCORE_SECONDARY,
            next_actions=established_checks,
        )
        candidate_pool = bundle.known_bug_evidence + bundle.documentation + bundle.authoritative_documentation + bundle.historical_case_evidence
        ranked = rank_items(candidate_pool, concept_words, bundle.retrieval_profile)
        if concept_words:
            # Same majority-overlap discipline §3/§13 of the prior
            # phase already established for troubleshooting-shaped
            # composers -- a single coincidentally-shared generic word
            # must never be presented as a real check to perform.
            ranked = [pair for pair in ranked if not self._is_off_topic_overlap(pair[1], len(concept_words))]

        subject_desc = " ".join(concept_words) if concept_words else "this symptom"
        if not established_checks and not ranked:
            missing = f'documented troubleshooting checks, known bugs, or historical cases covering "{subject_desc}"'
            return (
                "## What is observed\n"
                f"- {question}\n\n"
                "## What L2 should check\n"
                "I don't have enough evidence in the current knowledge base to recommend a specific L2 check for "
                "this symptom.\n\n"
                f"Missing evidence: {missing}.\n\n"
                "## What this can establish\n"
                "Nothing beyond the question itself -- no real evidence was found to check against.\n\n"
                "## What is not confirmed\n"
                "No evidence-backed check exists for this symptom in the current knowledge base.\n\n"
                "## Next action\n"
                "Escalate for manual investigation, or add documentation covering this scenario to the knowledge base."
            )

        lines = ["## What is observed", f"- {question}", "", "## What L2 should check"]
        idx = 1
        for action in established_checks[:2]:
            lines.append(f"\n{idx}. {action}")
            lines.append("   Why: An evidence-backed resolution/validation step already established for this investigation.")
            lines.append(f"   Evidence: {action}")
            lines.append("   Source: structured resolution")
            idx += 1
        remaining_slots = max(0, self._L2_GUIDANCE_MAX_CHECKS - (idx - 1))
        for item, _overlap in ranked[:remaining_slots]:
            lines.append(f'\n{idx}. Review "{item.title}"')
            lines.append(f"   Why: {item.establishes}")
            lines.append(f"   Evidence: {truncate_extract(item.excerpt, 200)}")
            lines.append(f"   Source: {item.source_type}")
            idx += 1

        lines.append("\n## What this can establish")
        lines.append(
            "The evidence above shows what has previously been checked, documented, or observed for similar "
            "symptoms -- it does not by itself confirm the root cause of the current situation."
        )
        lines.append("\n## What is not confirmed")
        lines.append("The root cause is not confirmed from the current evidence.")
        lines.append("\n## Next action")
        lines.append("Perform check #1 above first, then proceed through the remaining checks in order.")
        return "\n".join(lines)

    _TROUBLESHOOTING_SYNTHESIS_MAX_CANDIDATES = 3
    """Same capping discipline as everywhere else in this codebase --
    a ranked-hypothesis answer with a dozen entries is not more useful
    than one with the top few; the rest are still real, just not the
    strongest candidates."""

    def _troubleshooting_observed_lines(self, structured: "StructuredResolution") -> list[str]:
        """Final Hardening Pass, Objective 2A's "What is observed"
        section -- real, quoted evidence only (``structured.symptoms``,
        falling back to ``structured.problem``), never a new summary or
        inference. Deliberately does not read ``log_evidence`` directly
        here: log-shaped questions already have their own, much richer
        OBSERVED section in ``_compose_log_analysis_answer`` -- this
        method is reached only for the tier-based (non-log) troubleshooting
        path, so it stays scoped to what ``StructuredResolution`` itself
        carries, exactly like the rest of this composer."""
        if structured.symptoms:
            return [f"- {structured.symptoms}"]
        if structured.problem:
            return [f"- {structured.problem}"]
        return ["- No specific symptom evidence is recorded for this investigation."]

    @staticmethod
    def _troubleshooting_contradicting_line(log_evidence: "list[Evidence] | None") -> str:
        """Final Hardening Pass, Objective 2C: "Evidence against", or
        the required honest default when none exists -- NEVER a
        fabricated negative. The one real, deterministic signal this
        codebase already establishes for "something suggests recovery"
        is reused verbatim from ``_compose_log_analysis_answer``'s own
        OBSERVED recovery hint (the chronologically LAST parsed event
        being non-error after an earlier one was) -- never re-derived
        independently, so the two composers never disagree about what
        counts as a recovery signal."""
        if log_evidence:
            events = [event for evidence in log_evidence for event in evidence.log_events]
            timestamped = sorted((e for e in events if e.timestamp is not None), key=lambda e: e.timestamp)
            if timestamped and timestamped[-1].level not in (LogLevel.ERROR, LogLevel.FATAL):
                has_earlier_error = any(e.level in (LogLevel.ERROR, LogLevel.FATAL) for e in timestamped[:-1])
                if has_earlier_error:
                    return (
                        "The most recent observed log event is not an error, which MAY indicate recovery -- "
                        "this alone does not establish that the issue is resolved."
                    )
        return "No contradicting evidence identified in the current evidence."

    _SUBJECTLESS_TROUBLESHOOTING_INSUFFICIENCY = (
        "I don't have enough evidence in the current knowledge base to name a likely cause. This question doesn't say "
        "which component, symptom, or error it is about, so there is nothing specific to match evidence against, and I "
        "won't present an unrelated record as the explanation. Tell me what is affected (for example the component, the "
        "symptom, or an error message), or upload the relevant log, and I'll look for matching evidence."
    )

    @staticmethod
    def _troubleshooting_subject_words(question: str, context_text: str = "") -> list[str]:
        """The words ``_compose_troubleshooting_synthesis`` matches
        candidates against. A question that names its own subject uses
        only that. A follow-up whose own words are only question-shape/
        action words ("What should I check next?") inherits the
        investigation's subject words from ``context_text`` instead -- the
        same follow-up convention the L2 guidance planner uses.

        Fix Remaining Off-Topic Answers & Subjectless Follow-Ups phase,
        Part 2 -- returns ``[]`` (never the bare action words themselves)
        when NEITHER the question NOR the context names a real subject.
        An earlier version fell back to the action words alone (e.g.
        ``["check"]`` for a completely fresh "What should I check
        next?"), which let ``_compose_troubleshooting_synthesis`` match
        on the word "check" against WHATEVER historical record happened
        to contain it -- an arbitrary, unrelated case presented as "the
        most likely explanation" to a user who never gave any subject at
        all. Both callers already treat an empty return as "ask for the
        missing detail instead of guessing" (see ``_SUBJECTLESS_
        TROUBLESHOOTING_INSUFFICIENCY``), which is the correct behavior
        here: a fresh, genuinely subject-less question should prompt for
        the missing component/symptom/log, never retrieve or promote an
        arbitrary case."""
        own = extract_troubleshooting_subject_words(question)
        subject = [word for word in own if word not in TROUBLESHOOTING_ACTION_WORDS]
        if subject:
            return subject
        return [word for word in extract_troubleshooting_subject_words(context_text) if word not in TROUBLESHOOTING_ACTION_WORDS]

    def _compose_troubleshooting_synthesis(
        self,
        strategy: "InvestigationStrategy",
        question: str,
        log_evidence: "list[Evidence] | None" = None,
        *,
        force: bool = False,
        context_text: str = "",
    ) -> str | None:
        """Final Hardening Pass, Objective 2 -- a richer, ranked-
        hypothesis deterministic answer for "why did this fail?"-style
        questions (``contains_troubleshooting_synthesis_question``),
        built ENTIRELY from evidence ``RecommendationEngine.generate()``
        already retrieved (the exact same ``strategy.documentation``/
        ``historical_investigations``/``known_bugs`` this method's
        sibling ``_compose_knowledge_synthesis`` already uses -- no new
        retrieval, no LLM, no paraphrase). Reuses that method's own
        relevance bars/re-ranking (``_KNOWLEDGE_SYNTHESIS_MIN_SCORE``,
        ``_KNOWLEDGE_SYNTHESIS_MIN_SCORE_SECONDARY``, ``extract_concept_
        words``/``lexical_overlap``) rather than inventing a second,
        possibly-diverging relevance rule.

        Confidence discipline (Objective 2D): NEVER fires at LIKELY or
        CONFIRMED tier -- at those tiers ``structured.root_cause``/
        ``resolution_candidates`` are already the real, specific,
        established answer, and the existing tier-based text
        (unmodified) already states it correctly; this method would
        only add a weaker-sounding "possible causes" framing around an
        already-settled fact. Each individual candidate's own "Likely"/
        "Possible" label is a plain-English rendering of that
        candidate's own real, already-computed ``KnowledgeMatch.score``
        against the same secondary bar knowledge synthesis already
        uses for "strong" relevance -- never a new confidence
        calculation, never an LLM guess.

        Known-bug/historical-case separation (Objective 2E/2F): every
        candidate is labeled by its real source kind ("Relevant known
        bug"/"Similar historical case"/"Related documentation"), never
        asserted as *the* current root cause.

        Returns ``None`` (Objective 2G) whenever no real candidate
        clears the relevance bar at all, OR (after the same lexical
        re-ranking ``_compose_knowledge_synthesis`` uses) every
        candidate has zero real subject-word overlap with the
        question -- the caller then falls through to the existing,
        unmodified tier-based/L2-L3 text, which already states "what is
        observed"/"what is not established" honestly rather than this
        method manufacturing a hypothesis merely to look complete.

        ``force`` (Real-Corpus Answer Quality & Final Chat Hardening
        phase) -- bypasses the LIKELY/CONFIRMED tier guard below, used
        ONLY by ``_compose_answer``'s own off-topic override
        (``_tier_answer_is_off_topic``) for the exact real corpus
        finding that motivated it: "What happens if process settings
        are wrong?" reached LIKELY tier off an unrelated known bug, and
        this method's own tier guard -- built on the assumption that a
        LIKELY/CONFIRMED root cause is already correct -- silently kept
        the caller from ever trying a real, on-topic troubleshooting
        hypothesis instead. ``force`` is never set by this method's own
        normal callers (the default ``False`` preserves this contract
        exactly as before for every existing test)."""
        structured = strategy.structured_resolution
        if structured is None or (
            not force
            and structured.confidence in (
                ResolutionProvenance.LIKELY,
                ResolutionProvenance.CONFIRMED,
            )
        ):
            return None

        min_score = self._KNOWLEDGE_SYNTHESIS_MIN_SCORE
        min_score_secondary = self._KNOWLEDGE_SYNTHESIS_MIN_SCORE_SECONDARY
        hist_matches = [m for m in strategy.historical_investigations if m.score >= min_score]
        bug_matches = [m for m in strategy.known_bugs if m.score >= min_score]
        doc_matches = [m for m in strategy.documentation if m.score >= min_score]
        if not (hist_matches or bug_matches or doc_matches):
            return None

        # Subject words = what the question is actually ABOUT (question-
        # shape words like "why"/"should"/"next"/"request" removed; the
        # prior turns' subject inherited for a subject-less follow-up).
        # Matching below is whole-word, so neither a frame word nor an
        # incidental substring ("check" in "Checklist") can make an
        # unrelated record look relevant.
        # Uploaded log lines live in the same context text as the user's own
        # turns; their words ("timeout", "command", "request") are not a
        # subject the user named, so with log evidence present nothing is
        # inherited and the log-analysis paths keep owning that case.
        inherit_from = "" if log_evidence else context_text
        concept_words = self._troubleshooting_subject_words(question, inherit_from)
        subject_was_inherited = not any(
            word not in TROUBLESHOOTING_ACTION_WORDS for word in extract_troubleshooting_subject_words(question)
        ) and any(word not in TROUBLESHOOTING_ACTION_WORDS for word in concept_words)
        if not concept_words:
            if force or log_evidence:
                return None
            return self._SUBJECTLESS_TROUBLESHOOTING_INSUFFICIENCY
        kind_labels = {
            "historical": "Similar historical case",
            "known_bug": "Relevant known bug",
            "documentation": "Related documentation",
        }
        kind_means_phrase = {
            "known_bug": "a known bug",
            "documentation": "documented behavior",
            "historical": "a previously observed scenario",
        }
        # Evidence-Centered Knowledge Retrieval & Synthesis phase, §4:
        # sourced from the real, named RETRIEVAL_PROFILES table
        # (retrieval_profile.py) instead of a private dict. Pinned to
        # the TROUBLESHOOTING profile specifically (known_bug >
        # documentation > historical), not re-classified per question,
        # because this composer has always applied ONE uniform tiebreak
        # regardless of the TROUBLESHOOTING/ROOT_CAUSE sub-distinction
        # -- RETRIEVAL_PROFILES[ROOT_CAUSE] genuinely differs (it ranks
        # historical above documentation, reflecting that a root-cause
        # question benefits more from a similar past incident than a
        # generic doc), and silently switching between the two per
        # question would be an unreviewed, untested behavior change to
        # this already-hardened composer -- not something this phase's
        # own "preserve existing tested behavior" instruction allows
        # implicitly.
        _KIND_PRIORITY = kind_priority(RETRIEVAL_PROFILES[AnswerIntent.TROUBLESHOOTING])
        pool = (
            [(m, "historical") for m in hist_matches]
            + [(m, "known_bug") for m in bug_matches]
            + [(m, "documentation") for m in doc_matches]
        )
        ranked = sorted(
            pool,
            key=lambda item: (
                word_overlap(concept_words, f"{item[0].title} {item[0].snippet[:500]}"),
                _KIND_PRIORITY[item[1]],
                item[0].score,
            ),
            reverse=True,
        )
        # A candidate must share real subject words with the question.
        # Forced invocations (the tier's own answer was already off-topic)
        # and every normal invocation use the same length-aware rule
        # (``_is_off_topic_overlap``): a SHORT question needs a majority of
        # its subject words, a longer one at least one. Because frame words
        # and substring hits no longer count, "What should I check next?"
        # no longer ranks an RFC change-review FAQ, and a single shared
        # generic word ("process", "request") no longer suffices for a
        # short question.
        #
        # Subject words inherited from earlier turns are a larger, less
        # precise set than a question's own, so a candidate must match a
        # real MAJORITY of them (never one incidental word) to be shown.
        def _relevant(overlap: int) -> bool:
            if subject_was_inherited:
                return has_majority_overlap(overlap, len(concept_words))
            return not self._is_off_topic_overlap(overlap, len(concept_words))

        ranked = [
            item for item in ranked
            if _relevant(word_overlap(concept_words, f"{item[0].title} {item[0].snippet[:500]}"))
        ]
        if not ranked:
            return None
        candidates = ranked[: self._TROUBLESHOOTING_SYNTHESIS_MAX_CANDIDATES]

        lines: list[str] = ["## What is observed"]
        lines.extend(self._troubleshooting_observed_lines(structured))

        top_match, top_kind = candidates[0]
        lines.append("\n## What this likely means")
        lines.append(
            f'The available evidence points toward "{top_match.title}" ({kind_means_phrase[top_kind]}) as the most '
            f"likely explanation, though this is not yet confirmed."
        )

        lines.append("\n## Likely causes")
        contradicting = self._troubleshooting_contradicting_line(log_evidence)
        for idx, (match, kind) in enumerate(candidates, start=1):
            label = "Likely" if match.score >= min_score_secondary else "Possible"
            excerpt = truncate_extract(match.snippet, 220)
            lines.append(f"\n### {idx}. {match.title}")
            lines.append(f"Confidence: {label}")
            lines.append(f'\nEvidence supporting:\n- {kind_labels[kind]}: "{match.title}" -- {excerpt}')
            lines.append(f"\nEvidence against:\n- {contradicting}")

        lines.append("\n## What is not confirmed")
        lines.append("The root cause is not confirmed from the current evidence.")

        lines.append("\n## What to check next")
        checks = available_checks(structured)
        if checks:
            lines.extend(f"- {check}" for check in checks)
        else:
            lines.append("No evidence-backed troubleshooting check is currently available.")

        # §8: same bounded, practical contradiction check as
        # _compose_knowledge_synthesis, built from the real
        # EvidenceBundle -- additive only, fires only on a genuine
        # version/config-value conflict between two AUTHORITATIVE_*
        # documentation sources (never on the historical/known-bug
        # candidates this composer's own hypotheses are built from).
        context_for_bundle = build_query_context(question)
        bundle = build_evidence_bundle(
            question, context_for_bundle, strategy, log_evidence=log_evidence, min_score=min_score, min_score_secondary=min_score_secondary
        )
        if bundle.contradictions:
            lines.append("\n## Evidence conflict")
            lines.append("ResolveIQ found conflicting evidence:")
            for c in bundle.contradictions:
                lines.append(f'- {c.description} "{c.source_a}" says: {c.claim_a}. "{c.source_b}" says: {c.claim_b}.')

        return "\n".join(lines)

    _LOG_ANALYSIS_MAX_TIMELINE_EVENTS = 20
    _LOG_ANALYSIS_MAX_QUOTED_EVENTS = 5
    _LOG_ANALYSIS_RETRY_KEYWORDS = ("retry", "retries", "retrying", "retried")
    _LOG_ANALYSIS_TIMEOUT_KEYWORDS = ("timeout", "timed out", "time out", "time-out")
    _LOG_ANALYSIS_REQUEST_KEYWORDS = ("request sent", "command sent", "command request", "sending request", "request initiated")
    _LOG_ANALYSIS_RESPONSE_KEYWORDS = ("response received", "received response", "acknowledg", "command accepted", "reply received")
    _LOG_CORRELATION_ENTITY_TYPES = ("correlation_id", "request_id", "command_log_id", "session_id")
    _LOG_METER_ENTITY_TYPES = ("meter_number", "serial_number", "endpoint_id")
    """The only entity types this codebase's real, existing
    ``RegexEntityExtractor`` actually recognizes that plausibly identify
    "which meter/endpoint" (§8 of the L2/L3 Investigation Copilot phase)
    or "which correlating transaction" (§4/§5) -- see that module's own
    ``_PATTERN_REGISTRY``. Deliberately does NOT invent a Collector ID/
    Device ID/Customer Account Number grouping (§15 named these, but no
    real extractor pattern produces them) -- grouping by an entity type
    nothing actually extracts would silently do nothing, which is worse
    than being explicit that it isn't supported yet."""
    """Literal, closed keyword lists -- a mechanical text search over the
    log's own real message/raw_line content, never an inferred causal
    signal. See ``_compose_log_analysis_answer``'s own docstring for why
    this is reported as "N event(s) mention a retry/timeout", not "a
    retry/timeout occurred"."""

    def _compose_log_analysis_answer(
        self,
        log_evidence: "list[Evidence]",
        strategy: "InvestigationStrategy",
        investigation: "InvestigationSession | None" = None,
    ) -> str | None:
        """Chat + Log Intelligence integration -- deterministic,
        evidence-only log-analysis answer built directly from the REAL,
        already-parsed ``LogEvent``/``ExtractedEntity`` data every
        uploaded log already carries (``Evidence.log_events``, populated
        at upload time by the existing, unmodified ``LogIntelligenceEngine``
        -- see ``ChatLogUploadService.upload``/``InvestigationEngine.
        add_file_evidence``). Never a new parser, never an LLM, never a
        paraphrase: every timestamp, message, and identifier value quoted
        here is copied verbatim from a real ``LogEvent``/
        ``ExtractedEntity`` already computed by that engine.

        OBSERVED vs. INFERRED vs. UNKNOWN (§17/21 of the Chat + Log
        Intelligence integration): every claim this method makes is
        OBSERVED -- a direct fact read off the parsed events (a
        timestamp, a severity, a message, a literal "retry"/"timeout"
        keyword match, which entity values exist). It never claims a
        causal relationship ("the timeout CAUSED the failure") or a root
        cause -- the closest it comes is naming the first ERROR/FATAL
        event as a candidate "failure point," explicitly hedged
        ("does not by itself establish why"), never asserted as
        confirmed. Cross-source correlation (the closing paragraph,
        reusing the exact same real, already-retrieved Documentation/
        Historical/Known-Bug/TFS/Wiki matches ``_compose_knowledge_
        synthesis`` uses) is explicitly hedged too ("useful for
        investigation, but does not by itself prove the same root cause
        applies") -- historical similarity is never presented as proof.

        Returns ``None`` only when every uploaded log's parser produced
        zero events at all (e.g. a genuinely empty or unparseable
        upload) -- the caller then falls back to the existing tier-based
        text, unchanged."""
        events: list[tuple[str, "LogEvent"]] = [
            (evidence.title, event) for evidence in log_evidence for event in evidence.log_events
        ]
        if not events:
            return None

        sections: list[str] = []
        file_count = len(log_evidence)
        sections.append(
            f"Your uploaded log{'s' if file_count != 1 else ''} ({file_count} file{'s' if file_count != 1 else ''}) "
            f"contain{'s' if file_count == 1 else ''} {len(events)} parsed event(s)."
        )

        timestamped = sorted((pair for pair in events if pair[1].timestamp is not None), key=lambda pair: pair[1].timestamp)
        if timestamped:
            lines = [
                f"{event.timestamp.strftime('%H:%M:%S')}  {event.level.value}  {event.message[:160] or event.raw_line[:160]}"
                for _, event in timestamped[: self._LOG_ANALYSIS_MAX_TIMELINE_EVENTS]
            ]
            more = len(timestamped) - len(lines)
            timeline_text = "Timeline (observed, in order):\n" + "\n".join(lines)
            if more > 0:
                timeline_text += f"\n... and {more} more event(s)."
            sections.append(timeline_text)

        errors = [event for _, event in events if event.level in (LogLevel.ERROR, LogLevel.FATAL)]
        warnings = [event for _, event in events if event.level == LogLevel.WARN]
        if errors:
            quoted = "; ".join(f'"{e.message[:120] or e.raw_line[:120]}"' for e in errors[: self._LOG_ANALYSIS_MAX_QUOTED_EVENTS])
            sections.append(f"{len(errors)} ERROR/FATAL-level event(s) observed: {quoted}.")
        if warnings:
            quoted = "; ".join(f'"{e.message[:120] or e.raw_line[:120]}"' for e in warnings[: self._LOG_ANALYSIS_MAX_QUOTED_EVENTS])
            sections.append(f"{len(warnings)} WARN-level event(s) observed: {quoted}.")

        retry_count = sum(
            1 for _, e in events if any(k in (e.message or e.raw_line).lower() for k in self._LOG_ANALYSIS_RETRY_KEYWORDS)
        )
        timeout_count = sum(
            1 for _, e in events if any(k in (e.message or e.raw_line).lower() for k in self._LOG_ANALYSIS_TIMEOUT_KEYWORDS)
        )
        if retry_count:
            sections.append(f"{retry_count} event(s) mention a retry.")
        if timeout_count:
            sections.append(f"{timeout_count} event(s) mention a timeout.")

        if timestamped:
            sections.append(self._compose_failure_candidates_section(timestamped))

        entities_by_type: dict[str, set[str]] = {}
        for _, event in events:
            for entity in event.entities:
                entities_by_type.setdefault(entity.entity_type.value, set()).add(entity.value)
        if entities_by_type:
            id_lines = ", ".join(
                f"{etype}: {', '.join(sorted(values)[:5])}" for etype, values in sorted(entities_by_type.items())
            )
            sections.append(f"Identifiers found: {id_lines}.")

        correlation_section = self._compose_correlation_section(events)
        if correlation_section is not None:
            sections.append(correlation_section)

        if investigation is not None:
            documented_flow_section = self._compose_documented_flow_section(events, investigation)
            if documented_flow_section is not None:
                sections.append(documented_flow_section)

        meter_section = self._compose_multi_meter_section(events)
        if meter_section is not None:
            sections.append(meter_section)

        # Cross-source correlation -- the exact same real, already-
        # retrieved evidence _compose_knowledge_synthesis uses, never a
        # second retrieval, explicitly hedged (historical similarity is
        # never proof -- see docstring above).
        min_score, min_score_secondary = self._KNOWLEDGE_SYNTHESIS_MIN_SCORE, self._KNOWLEDGE_SYNTHESIS_MIN_SCORE_SECONDARY
        hist_matches = [m for m in strategy.historical_investigations if m.score >= min_score]
        bug_matches = [m for m in strategy.known_bugs if m.score >= min_score_secondary]
        tfs = strategy.tfs_matches
        wiki = strategy.wiki_matches
        tfs_matches = [m for m in (tfs.matches if tfs is not None and tfs.available else []) if m.score >= min_score_secondary]
        wiki_matches = [m for m in (wiki.matches if wiki is not None and wiki.available else []) if m.score >= min_score_secondary]
        correlation_bits: list[str] = []
        if hist_matches:
            correlation_bits.append(
                "historical case" + ("s" if len(hist_matches) > 1 else "") + f' ({", ".join(f"{m.title!r}" for m in hist_matches[:2])})'
            )
        if bug_matches:
            correlation_bits.append(f'known bug ({bug_matches[0].title!r})')
        if tfs_matches and tfs_matches[0].tfs_case is not None:
            correlation_bits.append(f'a TFS case ({tfs_matches[0].tfs_case.title!r})')
        if wiki_matches and wiki_matches[0].wiki_page is not None:
            correlation_bits.append(f'a Wiki page ({wiki_matches[0].wiki_page.title!r})')
        if correlation_bits:
            sections.append(
                "ResolveIQ found " + ", ".join(correlation_bits) + " with similar symptoms. This similarity is useful "
                "for investigation, but does not by itself prove the same root cause applies to your current log."
            )
        else:
            sections.append("No closely matching historical case, known bug, TFS item, or Wiki page was found.")

        return "\n\n".join(sections)

    def _classify_event_role(self, event: "LogEvent") -> str:
        """L2/L3 Investigation Copilot phase (§4/§5) -- a purely
        mechanical keyword/level classification of one event's likely
        role in a request/response exchange. Never a claim about what
        actually happened beyond "this event's own text/level matches
        this closed keyword set" -- the caller is responsible for every
        OBSERVED/INFERRED distinction built on top of this."""
        text = (event.message or event.raw_line).lower()
        if event.level in (LogLevel.ERROR, LogLevel.FATAL):
            return "error"
        if any(k in text for k in self._LOG_ANALYSIS_TIMEOUT_KEYWORDS):
            return "timeout"
        if any(k in text for k in self._LOG_ANALYSIS_RETRY_KEYWORDS):
            return "retry"
        if any(k in text for k in self._LOG_ANALYSIS_REQUEST_KEYWORDS):
            return "request"
        if any(k in text for k in self._LOG_ANALYSIS_RESPONSE_KEYWORDS):
            return "response"
        return "other"

    @staticmethod
    def _group_events_by_identifier(
        events: "list[tuple[str, LogEvent]]", entity_types: "tuple[str, ...]"
    ) -> "dict[tuple[str, str], list[tuple[str, LogEvent]]]":
        """Groups ``(filename, LogEvent)`` pairs by every recognized
        ``ExtractedEntity`` whose type is in ``entity_types`` -- an
        event carrying more than one qualifying entity is grouped under
        each of them (never double-counted as "the same event" across
        different real identifiers). Purely mechanical: the grouping key
        is a real, already-extracted entity value, never inferred."""
        groups: dict[tuple[str, str], list[tuple[str, "LogEvent"]]] = {}
        for title, event in events:
            for entity in event.entities:
                if entity.entity_type.value in entity_types:
                    groups.setdefault((entity.entity_type.value, entity.value), []).append((title, event))
        return groups

    def _compose_failure_candidates_section(self, timestamped: "list[tuple[str, LogEvent]]") -> str:
        """L2/L3 Investigation Copilot phase (§6) -- replaces the prior,
        simpler "first ERROR = candidate failure point" sentence with
        several distinct, separately-labeled OBSERVED candidates (first
        abnormal event, first ERROR/FATAL, first retry trigger, first
        timeout, and the log's final event with an explicit recovery-vs-
        terminal-failure note) -- never collapsed into one, and never
        promoted to a confirmed root cause. ``timestamped`` is already
        sorted chronologically and non-empty (guaranteed by the only
        caller)."""
        candidates: list[str] = []

        def _add(label: str, pair: "tuple[str, LogEvent] | None") -> None:
            if pair is None:
                return
            title, event = pair
            candidates.append(
                f'OBSERVED: {label} at {event.timestamp.strftime("%H:%M:%S")} (in "{title}"): '
                f'"{(event.message or event.raw_line)[:140]}".'
            )

        _add(
            "the first abnormal (WARN or worse) event",
            next(((t, e) for t, e in timestamped if e.level in (LogLevel.WARN, LogLevel.ERROR, LogLevel.FATAL)), None),
        )
        first_error = next(((t, e) for t, e in timestamped if e.level in (LogLevel.ERROR, LogLevel.FATAL)), None)
        _add("the first ERROR/FATAL-level event", first_error)
        _add("the first retry trigger", next(((t, e) for t, e in timestamped if self._classify_event_role(e) == "retry"), None))
        _add("the first timeout", next(((t, e) for t, e in timestamped if self._classify_event_role(e) == "timeout"), None))

        _, last_event = timestamped[-1]
        if last_event.level in (LogLevel.INFO, LogLevel.DEBUG, LogLevel.TRACE, LogLevel.UNKNOWN) and first_error is not None:
            candidates.append(
                f'OBSERVED: the final parsed event (at {last_event.timestamp.strftime("%H:%M:%S")}) is '
                f"{last_event.level.value}-level, after an earlier error -- this MAY indicate recovery, but the log "
                f"alone does not confirm the operation ultimately succeeded."
            )
        elif last_event.level in (LogLevel.ERROR, LogLevel.FATAL):
            candidates.append(
                f'OBSERVED: the final parsed event (at {last_event.timestamp.strftime("%H:%M:%S")}) is '
                f"{last_event.level.value}-level -- this log ends on a failure; whether that is the transaction's "
                f"true terminal state is not established beyond what was captured here."
            )

        return (
            "Failure candidates (each is an OBSERVED fact about the log, never a confirmed root cause):\n"
            + "\n".join(candidates)
        )

    @staticmethod
    def _format_timing_deltas(group_sorted: "list[tuple[str, LogEvent]]", roles: list[str]) -> str | None:
        """Grounded Conversational Intelligence phase (§11) -- real,
        computed time differences between CONSECUTIVE events in an
        already-time-sorted, already-role-classified correlation group.
        Pure arithmetic on real ``LogEvent.timestamp`` values already
        parsed by the existing, unmodified engine -- never an invented
        or estimated duration, and never rendered at all when two
        consecutive events don't both carry a real timestamp (a gap in
        timestamped coverage silently breaks the chain at that point,
        rather than pretending the missing side has a duration)."""
        bits: list[str] = []
        for (_, prev_event), (_, curr_event), curr_role in zip(group_sorted, group_sorted[1:], roles[1:]):
            if prev_event.timestamp is None or curr_event.timestamp is None:
                continue
            delta = (curr_event.timestamp - prev_event.timestamp).total_seconds()
            if delta < 0:
                # Out-of-order timestamps (a malformed/unsorted log) --
                # never report a negative or fabricated duration.
                continue
            delta_text = f"{delta:.0f}s" if delta == int(delta) else f"{delta:.1f}s"
            bits.append(f"{curr_role} {delta_text} later")
        if not bits:
            return None
        return f"OBSERVED timing: {', then '.join(bits)} (computed directly from real timestamps)."

    def _compose_correlation_section(self, events: "list[tuple[str, LogEvent]]") -> str | None:
        """L2/L3 Investigation Copilot phase (§4/§5) -- groups events by
        a real, shared correlating identifier (correlation ID, request
        ID, command-log ID, or session ID -- the only entity types this
        codebase's extractor actually recognizes for this purpose) and
        reports the OBSERVED grouping plus, only when the group's own
        event roles genuinely suggest a request/response or
        request/failure shape, one clearly-labeled INFERRED sentence,
        plus (Grounded Conversational Intelligence phase, §11) real,
        computed timestamp deltas between consecutive events in the
        group when timestamps support it. Deliberately never a
        CONFIRMED tier here for the ROLE/relationship claim (that
        requires ``_compose_documented_flow_section``'s real wiki-
        scenario cross-check, only available when a
        ``LogKnowledgeRepository`` is wired -- see that method's own
        docstring); the TIMING claim, by contrast, is real arithmetic
        and needs no such cross-check to be stated as fact."""
        groups = self._group_events_by_identifier(events, self._LOG_CORRELATION_ENTITY_TYPES)
        multi_event_groups = {key: group for key, group in groups.items() if len(group) >= 2}
        if not multi_event_groups:
            return None

        lines: list[str] = []
        for (etype, value), group in sorted(multi_event_groups.items())[:5]:
            group_sorted = sorted(group, key=lambda pair: (pair[1].timestamp is None, pair[1].timestamp))
            roles = [self._classify_event_role(e) for _, e in group_sorted]
            lines.append(f"OBSERVED: {len(group_sorted)} event(s) share {etype}={value!r}, roles in order: {', '.join(roles)}.")
            timing = self._format_timing_deltas(group_sorted, roles)
            if timing is not None:
                lines.append(timing)
            if "request" in roles and roles[-1] == "response":
                lines.append(
                    f"INFERRED: these {etype}={value!r} events appear to belong to the same request/response "
                    f"transaction, based on sharing this identifier and their time order -- not confirmed by "
                    f"documentation."
                )
            elif ("request" in roles or "retry" in roles) and roles[-1] in ("error", "timeout"):
                lines.append(
                    f"INFERRED: the transaction for {etype}={value!r} appears to have ended in failure, based on "
                    f"sharing this identifier and their time order -- not confirmed by documentation."
                )
        return "Correlation:\n" + "\n".join(lines)

    _LOG_FLOW_ENTITY_TYPES = ("meter_number", "serial_number", "endpoint_id", "command_log_id")
    """Grounded Conversational Intelligence phase (§12) -- deliberately
    a narrower list than ``_LOG_CORRELATION_ENTITY_TYPES``: the
    existing, frozen ``reconstruct_flow`` (app.engines.log_intelligence.
    flow) resolves each matched event's PRODUCING COMPONENT via
    ``LogSourceApplication``/``LogCollectionScenario`` records that are
    themselves keyed by meter/endpoint/command-log identity (see that
    module's own docstring) -- a bare correlation/request/session ID
    has no such component mapping in the real Log Collection Knowledge
    Base, so passing one to ``reconstruct_flow`` would only ever return
    ``matched_component_count=0``, never a real scenario."""

    def _compose_documented_flow_section(
        self, events: "list[tuple[str, LogEvent]]", investigation: "InvestigationSession"
    ) -> str | None:
        """Grounded Conversational Intelligence phase (§12) -- attempts
        the existing, frozen ``reconstruct_flow`` for a genuinely
        wiki-documented, CONFIRMED-tier component ordering, additive to
        (never a replacement for) ``_compose_correlation_section``'s own
        OBSERVED/INFERRED narrative above. ``flow.py`` itself is not
        modified; this only calls its existing public function with an
        AUTO-SELECTED identifier.

        Auto-selection discipline (this phase's own explicit
        requirement -- "only auto-select when unambiguous"): among
        ``_LOG_FLOW_ENTITY_TYPES`` groups with 2+ events, the candidate
        is the one with the STRICTLY largest event count -- a tie for
        the largest count means no single identifier obviously
        dominates the log, so this method deliberately does nothing
        rather than guess which one the user actually cares about.

        Returns ``None`` (no extra section -- the correlation section
        above already covers the log) whenever: no
        ``LogKnowledgeRepository`` is wired (``self._log_knowledge_repo
        is None``, the default), no candidate identifier is unambiguous,
        or ``reconstruct_flow`` itself found no real wiki scenario
        explaining the matched components (``scenario_id is None``) --
        a "no scenario matched" result is not treated as a reason to
        fabricate one."""
        if self._log_knowledge_repo is None:
            return None
        groups = self._group_events_by_identifier(events, self._LOG_FLOW_ENTITY_TYPES)
        candidates = [(key, group) for key, group in groups.items() if len(group) >= 2]
        if not candidates:
            return None
        candidates.sort(key=lambda item: len(item[1]), reverse=True)
        if len(candidates) > 1 and len(candidates[0][1]) == len(candidates[1][1]):
            return None  # ambiguous -- more than one identifier ties for "dominant"
        (etype, value), _ = candidates[0]

        flow = reconstruct_flow(
            investigation, entity_type=etype, entity_value=value, log_knowledge_repo=self._log_knowledge_repo
        )
        if flow.scenario_id is None or flow.matched_component_count == 0:
            return None

        lines = [
            f"CONFIRMED (via ResolveIQ's documented Log Collection Knowledge Base, scenario "
            f"{flow.scenario_technology or 'unspecified'}/{flow.scenario_type or 'unspecified'}): the following "
            f"component order applies to {etype}={value!r}:"
        ]
        for label, steps in (("Outbound", flow.outbound_steps), ("Inbound", flow.inbound_steps)):
            for step in steps:
                status = f"{len(step.events)} matching event(s) found" if step.has_log_entry else "no matching event found -- a gap in this log's coverage"
                lines.append(f"  {label} step {step.order}: {step.component_name} -- {status}.")
        if flow.unresolved_events:
            lines.append(
                f"  {len(flow.unresolved_events)} matching event(s) could not be placed in this documented flow."
            )
        return "\n".join(lines)

    def _compose_multi_meter_section(self, events: "list[tuple[str, LogEvent]]") -> str | None:
        """L2/L3 Investigation Copilot phase (§8) -- when a log genuinely
        contains more than one distinct meter/endpoint identifier
        (meter number, serial number, or endpoint ID -- see
        ``_LOG_METER_ENTITY_TYPES``'s own docstring for why no other
        identifier type is used here), reports a real per-identifier
        breakdown. Returns ``None`` for a single-identifier (or
        zero-identifier) log -- that case is already covered by the
        "Identifiers found" line, and a one-row "multi-meter" table
        would be misleading noise."""
        groups = self._group_events_by_identifier(events, self._LOG_METER_ENTITY_TYPES)
        if len(groups) < 2:
            return None
        lines = []
        for (etype, value), group in sorted(groups.items()):
            error_count = sum(1 for _, e in group if e.level in (LogLevel.ERROR, LogLevel.FATAL))
            retry_count = sum(1 for _, e in group if self._classify_event_role(e) == "retry")
            lines.append(f"{value} ({etype}): {len(group)} event(s), {error_count} error(s), {retry_count} retry mention(s).")
        return f"{len(groups)} distinct meter/endpoint identifier(s) found in this log:\n" + "\n".join(lines)

    def _compose_log_comparison_answer(self, log_evidence: "list[Evidence]", strategy: "InvestigationStrategy") -> str | None:
        """L2/L3 Investigation Copilot phase (§7) -- multi-log
        comparison, preserving per-file identity throughout rather than
        pooling every file's events into one timeline. Returns ``None``
        (caller falls through to the regular single-log/tier-based path)
        only when every file produced zero events."""
        per_file: list[tuple[str, list[LogEvent]]] = [
            (evidence.title, evidence.log_events) for evidence in log_evidence if evidence.log_events
        ]
        if not per_file:
            return None

        sections: list[str] = []
        per_file_errors: dict[str, set[str]] = {}
        per_file_ids: dict[str, set[str]] = {}
        for title, file_events in per_file:
            errors = [e for e in file_events if e.level in (LogLevel.ERROR, LogLevel.FATAL)]
            ids = {f"{ent.entity_type.value}={ent.value}" for e in file_events for ent in e.entities}
            per_file_errors[title] = {(e.message or e.raw_line)[:120] for e in errors}
            per_file_ids[title] = ids
            summary = [f"{len(file_events)} event(s), {len(errors)} error(s)/fatal(s)"]
            if errors:
                summary.append(f'first error: "{(errors[0].message or errors[0].raw_line)[:140]}"')
            if ids:
                summary.append(f"identifiers: {', '.join(sorted(ids)[:5])}")
            sections.append(f'"{title}": ' + "; ".join(summary) + ".")

        titles = list(per_file_errors.keys())
        common_errors = set.intersection(*per_file_errors.values()) if len(per_file_errors) > 1 else set()
        common_ids = set.intersection(*per_file_ids.values()) if len(per_file_ids) > 1 else set()
        if common_ids:
            sections.append(f"Common identifiers across all files: {', '.join(sorted(common_ids)[:5])}.")
        if common_errors:
            sections.append(f"Common error text across all files: {', '.join(repr(e) for e in sorted(common_errors)[:3])}.")
        for title in titles:
            unique_errors = per_file_errors[title] - set.union(*(v for k, v in per_file_errors.items() if k != title)) if len(titles) > 1 else per_file_errors[title]
            if unique_errors:
                sections.append(f'Errors seen only in "{title}": {", ".join(repr(e) for e in sorted(unique_errors)[:3])}.')
        if not common_errors and not common_ids:
            sections.append("No common identifiers or error text were found across the uploaded files.")
        sections.append(
            "This comparison is based only on the parsed events/identifiers above -- differing outcomes are not "
            "by themselves proof of differing root causes."
        )
        return "\n\n".join(sections)

    def _compose_l2_task_notes(self, log_evidence: "list[Evidence]", strategy: "InvestigationStrategy") -> str | None:
        """L2/L3 Investigation Copilot phase (§11) -- a structured L2
        task-note format, built ENTIRELY from already-computed real data
        (the same log events/entities and the same real, already-
        retrieved ``strategy`` evidence every other composer in this
        class uses). Every field that has no real, supporting data says
        so explicitly ("Not established from current evidence") rather
        than being omitted or guessed -- per this phase's own explicit
        instruction never to fabricate a missing field."""
        events: list[tuple[str, "LogEvent"]] = [
            (evidence.title, event) for evidence in log_evidence for event in evidence.log_events
        ]
        if not events:
            return None
        structured = strategy.structured_resolution
        errors = [e for _, e in events if e.level in (LogLevel.ERROR, LogLevel.FATAL)]
        timestamped = sorted((p for p in events if p[1].timestamp is not None), key=lambda p: p[1].timestamp)
        entities_by_type: dict[str, set[str]] = {}
        for _, event in events:
            for entity in event.entities:
                entities_by_type.setdefault(entity.entity_type.value, set()).add(entity.value)
        affected = ", ".join(f"{t}: {', '.join(sorted(v)[:5])}" for t, v in sorted(entities_by_type.items())) or "Not established from current evidence."
        environment = "Not established from current evidence."
        if structured is not None and structured.applicability is not None:
            app = structured.applicability
            bits = [b for b in (app.technology_name, ", ".join(app.customer_names) or None) if b]
            if bits:
                environment = "; ".join(bits)
        timeline_bits = (
            f"{timestamped[0][1].timestamp.strftime('%H:%M:%S')} to {timestamped[-1][1].timestamp.strftime('%H:%M:%S')} "
            f"({len(timestamped)} timestamped event(s))"
            if timestamped
            else "Not established from current evidence."
        )
        error_text = "; ".join(f'"{(e.message or e.raw_line)[:140]}"' for e in errors[:5]) or "No ERROR/FATAL-level events observed."
        potential_cause = structured.root_cause if structured is not None and structured.root_cause else "Not established from current evidence."
        checks = available_checks(structured) if structured is not None else []
        next_action = "; ".join(checks) if checks else "No evidence-backed troubleshooting check is currently available."
        correlation_line = self._compose_correlation_section(events)
        failure_line = self._compose_failure_candidates_section(timestamped) if timestamped else "Not established from current evidence."

        return "\n\n".join(
            [
                f"Issue: {log_evidence[0].title if len(log_evidence) == 1 else f'{len(log_evidence)} uploaded log files'} under investigation.",
                f"Environment: {environment}",
                f"Affected entities: {affected}",
                f"Timeline: {timeline_bits}",
                f"Errors: {error_text}",
                f"Investigation performed: {len(events)} log event(s) parsed and analyzed by ResolveIQ's Log Intelligence Engine.",
                f"Findings ({failure_line.splitlines()[0].rstrip(':')}):\n" + "\n".join(failure_line.splitlines()[1:]),
                f"Potential cause: {potential_cause} (not confirmed by this log alone).",
                (correlation_line or "Correlation: no shared correlating identifier was observed across multiple events."),
                f"Next action for L2: {next_action}",
                "Escalation to L3: recommended if the potential cause above is not established and the next action does not resolve the issue.",
            ]
        )

    def _compose_l3_escalation_summary(self, log_evidence: "list[Evidence]", strategy: "InvestigationStrategy") -> str | None:
        """L2/L3 Investigation Copilot phase (§12) -- same "never
        fabricate a missing field" discipline as ``_compose_l2_task_
        notes``, reorganized into the L3 escalation format. Reuses the
        exact same underlying data, never a second computation of any
        fact already established above."""
        events: list[tuple[str, "LogEvent"]] = [
            (evidence.title, event) for evidence in log_evidence for event in evidence.log_events
        ]
        if not events:
            return None
        structured = strategy.structured_resolution
        errors = [e for _, e in events if e.level in (LogLevel.ERROR, LogLevel.FATAL)]
        timestamped = sorted((p for p in events if p[1].timestamp is not None), key=lambda p: p[1].timestamp)
        entities_by_type: dict[str, set[str]] = {}
        for _, event in events:
            for entity in event.entities:
                entities_by_type.setdefault(entity.entity_type.value, set()).add(entity.value)
        affected = ", ".join(f"{t}: {', '.join(sorted(v)[:5])}" for t, v in sorted(entities_by_type.items())) or "Not established from current evidence."
        correlation_ids = ", ".join(
            f"{t}={v}" for t, values in sorted(entities_by_type.items()) if t in self._LOG_CORRELATION_ENTITY_TYPES for v in sorted(values)[:3]
        ) or "None identified."
        min_score, min_score_secondary = self._KNOWLEDGE_SYNTHESIS_MIN_SCORE, self._KNOWLEDGE_SYNTHESIS_MIN_SCORE_SECONDARY
        hist_titles = [m.title for m in strategy.historical_investigations if m.score >= min_score][:3]
        bug_titles = [m.title for m in strategy.known_bugs if m.score >= min_score_secondary][:2]
        relevant_cases = ", ".join(hist_titles + bug_titles) or "None found above the relevance bar."
        suspected_area = structured.root_cause if structured is not None and structured.root_cause else "Not established -- no root cause has been confirmed from current evidence."
        failure_line = self._compose_failure_candidates_section(timestamped) if timestamped else "Not established from current evidence."

        return "\n\n".join(
            [
                f"Problem statement: {log_evidence[0].title if len(log_evidence) == 1 else f'{len(log_evidence)} uploaded log files'} shows unresolved abnormal behavior; see Findings below.",
                "Environment: Not established from current evidence." if structured is None or structured.applicability is None else f"Environment: {structured.applicability.technology_name or 'Not established from current evidence.'}",
                f"Affected entities: {affected}",
                f"Timeline: {timestamped[0][1].timestamp.strftime('%H:%M:%S')} to {timestamped[-1][1].timestamp.strftime('%H:%M:%S')} ({len(timestamped)} timestamped event(s))." if timestamped else "Timeline: Not established from current evidence.",
                f"Error details: " + ("; ".join(f'"{(e.message or e.raw_line)[:140]}"' for e in errors[:5]) or "No ERROR/FATAL-level events observed."),
                f"Correlation IDs / identifiers: {correlation_ids}",
                f"Investigation performed: {len(events)} log event(s) parsed via ResolveIQ's Log Intelligence Engine; documentation, historical cases, known bugs, and TFS/Wiki were searched for correlation.",
                f"Findings:\n" + "\n".join(failure_line.splitlines()[1:]) if timestamped else "Findings: Not established from current evidence.",
                f"Suspected failure area: {suspected_area}",
                f"Relevant historical cases / defects: {relevant_cases} (similarity only -- not proof of the same root cause).",
                "What L2 already checked: log was uploaded and analyzed; evidence-backed checks (if any) were reviewed via Chat.",
                "What L3 needs to investigate: the suspected failure area above, and whether the correlation IDs/identifiers listed connect to server-side or device-side logs not available to ResolveIQ.",
                f"Attachments / log references: {', '.join(e.title for e in log_evidence)}.",
            ]
        )

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
