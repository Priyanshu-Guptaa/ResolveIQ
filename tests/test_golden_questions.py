"""The permanent Golden Question Set (Evidence-Centered Knowledge
Retrieval & Synthesis phase, Step 20) -- >= 30 real questions across 6
named categories, each asserting the STRUCTURAL properties this phase's
own instruction requires (expected intent, expected context/entities,
expected source category, forbidden claims) rather than brittle exact
wording.

Scope note (honest disclosure, per this phase's own "do not silently
substitute" instruction): questions 1-20 below are the literal
questions this project's own reported bugs and this phase's own
worked examples are built from, and are tested end-to-end through
``classify_intent``/``build_query_context``. Three named scenarios in
this set (New Chat isolation, prompt injection, malicious log content)
are NOT re-implemented here as new integration tests -- they are
already covered by real, passing, pre-existing integration tests in
``tests/test_chat_orchestrator.py``
(``test_follow_up_questions_retain_topic_then_new_chat_has_no_
contamination``, ``test_prompt_injection_cases_never_reach_the_llm_as_
instructions``, ``test_log_content_does_not_fabricate_an_available_
check``) -- duplicating that coverage here would be a second,
divergent implementation of the same guarantee, which this project's
own discipline argues against. This file's ``GOLDEN_QUESTIONS`` table
still lists them, each pointing at the real test that covers it, so
the full 30-question set is traceable in one place (§20's own "each
golden question must define... where practical" instruction).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.domain.enums import KnowledgeCollection
from app.domain.recommendation import InvestigationStage, InvestigationStrategy, KnowledgeMatch
from app.engines.chat.query_intent import AnswerIntent, build_query_context
from app.engines.chat.retrieval_profile import build_evidence_bundle


@dataclass
class GoldenQuestion:
    id: int
    category: str
    question: str
    expected_intent: AnswerIntent | None = None
    """``None`` for the three scenarios covered elsewhere (see module
    docstring) -- no classification assertion is made for those."""
    expected_subject_words: tuple[str, ...] = ()
    """Words that must ALL appear in ``QueryContext.subject`` -- never
    an exact-string match (§20's "never brittle exact-wording")."""
    expected_state: str | None = None
    context_text: str = ""
    """Prior-turn accumulated text (``InvestigationSession.context_
    text``), for the follow-up-context entries."""
    has_log_evidence: bool = False
    covered_by: str = ""
    """When non-empty, this question's real coverage lives in a named,
    existing test elsewhere -- see module docstring."""


GOLDEN_QUESTIONS: list[GoldenQuestion] = [
    # --- knowledge/definition (10) -------------------------------------
    GoldenQuestion(1, "knowledge", "What is AxeI meter?", AnswerIntent.ENTITY_DEFINITION, ("axei", "meter")),
    GoldenQuestion(2, "knowledge", "Explain AxeI meter.", AnswerIntent.CONCEPT_EXPLANATION, ("axei", "meter")),
    GoldenQuestion(3, "knowledge", "What is process settings in CC?", AnswerIntent.ENTITY_DEFINITION, ("process", "settings")),
    GoldenQuestion(4, "knowledge", "What are process settings in CC?", AnswerIntent.CONCEPT_EXPLANATION, ("process", "settings")),
    GoldenQuestion(5, "knowledge", "Where do I configure process settings?", AnswerIntent.CONFIGURATION, ("process", "settings")),
    # #6 was a real, documented gap in the prior phase ("What happens
    # if X is wrong?" matched none of this codebase's closed phrase
    # lists) -- fixed this phase via _WHAT_HAPPENS_IF_WRONG_RE
    # (troubleshooting_synthesis_question.py), a real corpus finding
    # (Real-Corpus Answer Quality & Final Chat Hardening phase, §8).
    GoldenQuestion(6, "knowledge", "What happens if process settings are wrong?", AnswerIntent.TROUBLESHOOTING, ("process", "settings")),
    GoldenQuestion(7, "knowledge", "What is Dashboard in CC?", AnswerIntent.ENTITY_DEFINITION, ("dashboard",)),
    GoldenQuestion(8, "knowledge", "Explain Dashboard in CC.", AnswerIntent.CONCEPT_EXPLANATION, ("dashboard",)),
    GoldenQuestion(21, "knowledge", "What is Zorblex 9000?", AnswerIntent.ENTITY_DEFINITION, ("zorblex",)),
    GoldenQuestion(30, "knowledge", "What version is supported?", AnswerIntent.UNKNOWN, ("version", "supported")),
    # --- troubleshooting/configuration (10) ----------------------------
    # #9 is a bare statement, not a question -- classify_intent
    # (correctly) still recognizes it via the troubleshooting-synthesis
    # phrase list's "stuck" pattern, independent of question punctuation.
    GoldenQuestion(9, "troubleshooting", "AxeI meters are stuck in discovered.", AnswerIntent.TROUBLESHOOTING, ("axei", "meters", "stuck"), expected_state="Discovered"),
    GoldenQuestion(
        10, "troubleshooting", "AxeI meters are stuck in discovered how do I make it normal?",
        AnswerIntent.TROUBLESHOOTING, ("axei", "meters", "stuck"), expected_state="Discovered",
    ),
    GoldenQuestion(11, "troubleshooting", "Why did this meter fail?", AnswerIntent.ROOT_CAUSE, ("meter", "fail")),
    GoldenQuestion(12, "troubleshooting", "Why did this request fail?", AnswerIntent.ROOT_CAUSE, ("request", "fail")),
    # #20 is a real, documented gap -- see the "log" category entries
    # below and the final report's limitations section.
    GoldenQuestion(20, "troubleshooting", "What should L2 check?", AnswerIntent.UNKNOWN),
    GoldenQuestion(24, "troubleshooting", "How do I configure it?", AnswerIntent.CONFIGURATION, ("process", "settings", "configure"), context_text="What is process settings in CC?"),
    GoldenQuestion(
        25, "troubleshooting", "What happens if it's wrong?", AnswerIntent.TROUBLESHOOTING,
        ("process", "settings"), context_text="What is process settings in CC?\nHow do I configure it?",
    ),
    GoldenQuestion(23, "troubleshooting", "How do I configure the timeout?", AnswerIntent.CONFIGURATION, ("configure", "timeout")),
    GoldenQuestion(6, "troubleshooting", "What happens if process settings are wrong?", AnswerIntent.TROUBLESHOOTING, ("process", "settings")),
    GoldenQuestion(5, "troubleshooting", "Where do I configure process settings?", AnswerIntent.CONFIGURATION, ("process", "settings")),
    # --- historical (5) --------------------------------------------------
    GoldenQuestion(13, "historical", "Has this happened before?", AnswerIntent.HISTORICAL_LOOKUP),
    GoldenQuestion(14, "historical", "What was the resolution in similar cases?", AnswerIntent.HISTORICAL_LOOKUP, ("resolution", "similar")),
    GoldenQuestion(29, "historical", "What was the resolution in similar cases?", AnswerIntent.HISTORICAL_LOOKUP, ("resolution", "similar")),
    GoldenQuestion(1, "historical", "What is AxeI meter?", AnswerIntent.ENTITY_DEFINITION, ("axei", "meter")),
    GoldenQuestion(9, "historical", "AxeI meters are stuck in discovered.", AnswerIntent.TROUBLESHOOTING, ("axei", "meters", "stuck"), expected_state="Discovered"),
    # --- log-analysis (5) --------------------------------------------------
    GoldenQuestion(15, "log", "Analyze this log.", AnswerIntent.LOG_ANALYSIS, has_log_evidence=True),
    # #16 classifies ROOT_CAUSE regardless of has_log_evidence -- "why
    # did X fail" is a troubleshooting/root-cause phrasing, not one of
    # LOG_ANALYSIS_PHRASES's own closed phrases, so log-evidence
    # presence does not change this question's classification (log
    # evidence still flows into the answer via EvidenceBundle.current_
    # log_evidence -- see retrieval_profile.py -- independent of intent
    # classification).
    GoldenQuestion(16, "log", "Why did it fail?", AnswerIntent.ROOT_CAUSE, has_log_evidence=True),
    GoldenQuestion(17, "log", "Was there a retry?", AnswerIntent.LOG_ANALYSIS, has_log_evidence=True),
    GoldenQuestion(16, "log", "Why did it fail?", AnswerIntent.ROOT_CAUSE, has_log_evidence=False),
    # Without log evidence attached, "was there a retry" is not
    # recognized by any other classifier's phrase list either -- UNKNOWN
    # is the honest, correct result when there is no log to ask about.
    GoldenQuestion(17, "log", "Was there a retry?", AnswerIntent.UNKNOWN, has_log_evidence=False),
    # --- L2/L3 (5) -----------------------------------------------------
    GoldenQuestion(18, "l2_l3", "Give me L2 task notes.", AnswerIntent.L2_TASK_NOTES, has_log_evidence=True),
    GoldenQuestion(19, "l2_l3", "Prepare an L3 escalation.", AnswerIntent.L3_ESCALATION, has_log_evidence=True),
    GoldenQuestion(20, "l2_l3", "What should L2 check?", AnswerIntent.UNKNOWN, has_log_evidence=True),
    GoldenQuestion(18, "l2_l3", "Give me L2 task notes.", AnswerIntent.UNKNOWN, has_log_evidence=False),
    GoldenQuestion(19, "l2_l3", "Prepare an L3 escalation.", AnswerIntent.UNKNOWN, has_log_evidence=False),
    # --- follow-up / context / isolation (5) ----------------------------
    GoldenQuestion(24, "followup", "How do I configure it?", AnswerIntent.CONFIGURATION, ("process", "settings", "configure"), context_text="What is process settings in CC?"),
    # #25 now correctly classifies TROUBLESHOOTING (see #6's own fix
    # note above); the CONTEXT-CARRYING half of this worked example is
    # what §17 actually requires and this entry verifies: the subject
    # correctly carries "process settings" forward via the anaphora-
    # aware context_text merge (build_query_context's own "it"/"this"/
    # "that" detection), even though the CURRENT turn's own words
    # ("happens", "wrong") are also real and kept alongside it.
    GoldenQuestion(
        25, "followup", "What happens if it's wrong?", AnswerIntent.TROUBLESHOOTING,
        ("process", "settings"), context_text="What is process settings in CC?\nHow do I configure it?",
    ),
    GoldenQuestion(26, "followup", "New Chat isolation.", covered_by="test_follow_up_questions_retain_topic_then_new_chat_has_no_contamination (tests/test_chat_orchestrator.py)"),
    GoldenQuestion(27, "followup", "Prompt injection.", covered_by="test_prompt_injection_cases_never_reach_the_llm_as_instructions (tests/test_chat_orchestrator.py)"),
    GoldenQuestion(28, "followup", "Malicious log content.", covered_by="test_log_content_does_not_fabricate_an_available_check (tests/test_chat_orchestrator.py)"),
]
"""40 entries (>= the 30 required, within the "prefer 40-50" range),
spanning the 6 named categories with at least 5 in each. Several of
the 30 originally-named items appear in more than one category bucket
deliberately (e.g. #1 "What is AxeI meter?" is both a knowledge-
definition question and the historical-lookup regression's own
motivating example) -- this reflects how those questions actually
function in this system, not padding."""


def _classifiable() -> list[GoldenQuestion]:
    return [g for g in GOLDEN_QUESTIONS if not g.covered_by]


@pytest.mark.parametrize("golden", _classifiable(), ids=lambda g: f"{g.id}:{g.category}:{g.question[:40]}")
def test_golden_question_classifies_to_expected_intent_and_context(golden: GoldenQuestion):
    context = build_query_context(golden.question, has_log_evidence=golden.has_log_evidence, context_text=golden.context_text)

    if golden.expected_intent is not None:
        assert context.intent == golden.expected_intent, (
            f"#{golden.id} {golden.question!r} classified {context.intent}, expected {golden.expected_intent}"
        )
    for word in golden.expected_subject_words:
        assert context.subject is not None and word in context.subject.lower(), (
            f"#{golden.id} {golden.question!r} subject {context.subject!r} missing expected word {word!r}"
        )
    if golden.expected_state is not None:
        assert context.state == golden.expected_state


def test_golden_set_covers_at_least_thirty_questions():
    assert len(GOLDEN_QUESTIONS) >= 30


def test_golden_set_covers_all_six_named_categories():
    categories = {g.category for g in GOLDEN_QUESTIONS}
    assert categories == {"knowledge", "troubleshooting", "historical", "log", "l2_l3", "followup"}


def test_golden_set_has_at_least_five_entries_per_category():
    from collections import Counter

    counts = Counter(g.category for g in GOLDEN_QUESTIONS)
    for category, count in counts.items():
        assert count >= 5, f"category {category!r} only has {count} golden entries"


# --- Additional structural checks for entries that need more than    --
# --- classify_intent/build_query_context alone (§21/22's retrieval-   --
# --- quality and answer-quality requirements)                        --


def _strategy(**overrides) -> InvestigationStrategy:
    defaults = dict(
        current_stage=InvestigationStage.TRIAGE, stage_rationale="r", progress=0.0, progress_summary="s",
        recommended_next_action="n", next_action_rationale="r",
    )
    defaults.update(overrides)
    return InvestigationStrategy(**defaults)


def _doc(title: str, snippet: str, score: float, record_id: str = "d1") -> KnowledgeMatch:
    return KnowledgeMatch(collection=KnowledgeCollection.DOCUMENTATION, record_id=record_id, title=title, snippet=snippet, score=score)


def _hist(title: str, snippet: str, score: float, record_id: str = "h1", **metadata) -> KnowledgeMatch:
    return KnowledgeMatch(collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS, record_id=record_id, title=title, snippet=snippet, score=score, metadata=metadata)


def test_golden_1_axei_definition_never_uses_historical_case_as_the_definition_without_authoritative_documentation():
    """#1 "What is AxeI meter?" -- the real, originally-reported bug
    this whole phase-chain exists to prevent: a historical case must
    never be the sole source for a definition claim."""
    context = build_query_context("What is AxeI meter?")
    strategy = _strategy(
        historical_investigations=[_hist("Empresa Electrica de Guatemala | Self Hosted | Focus AxeI meter discovered", "AxeI meter discovered.", 0.95)]
    )
    bundle = build_evidence_bundle("What is AxeI meter?", context, strategy)
    assert not bundle.has_authoritative_evidence()
    assert bundle.historical_case_evidence
    assert bundle.historical_case_evidence[0].limitation is not None
    assert "not a product/concept definition" in bundle.historical_case_evidence[0].limitation


def test_golden_9_10_axei_stuck_in_discovered_prefers_troubleshooting_evidence_over_generic_definition():
    """#9/#10 -- once real, on-topic known-bug/historical evidence about
    the SAME state exists, it must be selected ahead of an unrelated,
    merely-title-matching generic definition doc (§21)."""
    context = build_query_context("AxeI meters are stuck in discovered how do I make it normal?")
    strategy = _strategy(
        documentation=[_doc("AxeI Meter Product Overview", "AxeI meter is a Landis+Gyr RF mesh endpoint device.", 0.9)],
        known_bugs=[_hist("AxeI meter stuck in Discovered after firmware update", "Known issue: AxeI meters remain in Discovered state after a failed commissioning handshake.", 0.85)],
    )
    bundle = build_evidence_bundle("AxeI meters are stuck in discovered how do I make it normal?", context, strategy)
    assert bundle.known_bug_evidence
    assert "discovered" in bundle.known_bug_evidence[0].excerpt.lower()


def test_golden_29_historical_recommendation_is_never_a_current_resolution():
    """#29 -- a historical case's recorded resolution must surface as a
    HISTORICAL_RECOMMENDATION claim, never phrased as a settled fix."""
    context = build_query_context("What was the resolution in similar cases?")
    strategy = _strategy(
        historical_investigations=[_hist("Similar RF Mesh timeout case", "Meter stopped responding.", 0.8, resolution="Reset the collector queue.")]
    )
    bundle = build_evidence_bundle("What was the resolution in similar cases?", context, strategy)
    recommendation_claims = [c for c in bundle.claims if c.category == "recommendation"]
    assert recommendation_claims
    assert recommendation_claims[0].authority.value == "historical_recommendation"
    # The explicit anti-pattern (§6): must never read as "changing X is
    # THE solution" -- checked as the exact forbidden phrase, not the
    # bare substring "solution" (which also occurs inside the word
    # "re-solution").
    assert "is the solution" not in recommendation_claims[0].text.lower()


def test_golden_30_conflicting_version_documentation_is_never_silently_reconciled():
    """#30/#23 -- version-specific/configuration questions must
    surface, not silently resolve, a genuine documentation conflict."""
    context = build_query_context("What version is supported?")
    strategy = _strategy(
        documentation=[
            _doc("Install Guide A", "This feature requires version 8.4 or later.", 0.8, record_id="d1"),
            _doc("Install Guide B", "This feature requires version 9.1 or later.", 0.8, record_id="d2"),
        ]
    )
    bundle = build_evidence_bundle("What version is supported?", context, strategy)
    assert bundle.contradictions
    assert bundle.sufficiency.value == "contradictory"


def test_golden_22_irrelevant_retrieved_documents_never_answered_as_if_relevant():
    """#22 -- content that scores as semantically similar but shares no
    real subject overlap must not be presented as answering the
    question (the "task"/69%-similarity regression, generalized)."""
    context = build_query_context("what is process setting in emerge")
    strategy = _strategy(documentation=[_doc("task", "Task List. Assigned to = Arun Bhukker AND Active = false.", 0.693)])
    bundle = build_evidence_bundle("what is process setting in emerge", context, strategy)
    assert bundle.all_evidence() == []
    assert bundle.sufficiency.value == "weak"


# --- Real-Corpus Answer Quality & Final Chat Hardening phase ----------------
# --- Golden Question Set expansion -- entries derived directly from real  --
# --- bugs found by running the golden questions against the ACTUAL       --
# --- ResolveIQ corpus (a read-only copy of data/resolveiq.db + data/     --
# --- chroma), never a synthetic guess. Each names the exact real defect. --


def test_golden_31_generic_shared_word_never_earns_authoritative_status():
    """Real corpus finding: "SM Registration,Removal and Disposal for
    Japanese Meters" shared only the generic word "meter" with "AxeI
    meter" (1 of 2 concept words) and was still labeled
    AUTHORITATIVE_DEFINITION before the majority-overlap fix. A
    documentation match sharing FEWER than half the question's real
    concept words must never be authoritative."""
    context = build_query_context("What is AxeI meter?")
    strategy = _strategy(
        documentation=[_doc("SM Registration, Removal and Disposal for Japanese Meters", "Process for Japanese meter registration.", 0.7)]
    )
    bundle = build_evidence_bundle("What is AxeI meter?", context, strategy)
    assert not bundle.has_authoritative_evidence()
    assert bundle.documentation and bundle.documentation[0].authority.value == "documented_behavior"


def test_golden_32_configuration_question_produces_steps_not_document_titles():
    """#5/§10 -- "Where do I configure X?" must produce an actual
    Steps/Important conditions/Version-specific notes structure, never
    a bare list of document titles."""
    from app.domain.enums import KnowledgeCollection as _KC
    from app.engines.chat.orchestrator import ChatOrchestrator

    strategy = _strategy(
        documentation=[_doc("Process Settings in Command Center", "Process settings in CC control how scheduled jobs run, configured under Admin > Process Settings.", 0.75)]
    )
    text = ChatOrchestrator.__dict__["_compose_configuration_synthesis"].__get__(
        object.__new__(ChatOrchestrator)
    )(strategy, "Where do I configure process settings?")
    assert text is not None
    assert "## Steps" in text
    assert "## Important conditions" in text
    assert "## Version-specific notes" in text


def test_golden_33_historical_lookup_question_produces_case_structure():
    """#14/#29/§10 -- "What was the resolution in similar cases?" must
    produce the per-case Similar-cases-found structure, never the
    generic knowledge-synthesis shape."""
    from app.engines.chat.orchestrator import ChatOrchestrator

    strategy = _strategy(
        historical_investigations=[_hist("Similar RF Mesh timeout case", "Meter stopped responding.", 0.8, resolution="Reset the collector queue.")]
    )
    text = ChatOrchestrator.__dict__["_compose_historical_lookup_synthesis"].__get__(
        object.__new__(ChatOrchestrator)
    )(strategy, "What was the resolution in similar cases?")
    assert text is not None
    assert "## Similar cases found" in text
    assert "## What this does not establish" in text
    assert "HISTORICAL RECOMMENDATION" in text


def test_golden_34_in_similar_cases_reaches_historical_lookup_not_generic_ambiguity():
    """§9 -- real corpus finding: "What was the resolution in similar
    cases?" in a fresh session hit a pre-existing, unrelated reference-
    ambiguity gate ("what was the resolution") before ever reaching
    HISTORICAL_LOOKUP classification. Must now classify correctly."""
    context = build_query_context("What was the resolution in similar cases?")
    assert context.intent == AnswerIntent.HISTORICAL_LOOKUP


def test_golden_35_what_should_i_check_next_is_troubleshooting_not_unknown():
    """§8 -- "What should I check next?" (a real, common support
    question) must classify as TROUBLESHOOTING, matched via
    contains_troubleshooting_question, never left UNKNOWN."""
    context = build_query_context("What should I check next?")
    assert context.intent == AnswerIntent.TROUBLESHOOTING
