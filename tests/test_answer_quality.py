"""Unit tests for app.engines.chat.answer_quality (Final Support-
Quality Pass, §7) -- a deterministic, never-LLM-judged answer scorer."""

from __future__ import annotations

from app.domain.evidence_bundle import Claim, EvidenceBundle, SourceAuthority, SufficiencyLevel
from app.engines.chat.answer_quality import score_answer
from app.engines.chat.query_intent import AnswerIntent, build_query_context


def _bundle(**overrides) -> EvidenceBundle:
    defaults = dict(question="q", intent="entity_definition")
    defaults.update(overrides)
    return EvidenceBundle(**defaults)


def test_correct_intent_and_direct_answer_scores_perfectly():
    context = build_query_context("What is AxeI meter?")
    bundle = _bundle(
        sufficiency=SufficiencyLevel.AUTHORITATIVE,
        claims=[Claim(claim_id="claim-001", text="AxeI meter is defined here.", supported_by=["Doc A"], authority=SourceAuthority.AUTHORITATIVE_DEFINITION, category="definition")],
    )
    result = score_answer(
        question="What is AxeI meter?",
        answer_text="## Answer\nAxeI meter is a real device type documented in ResolveIQ.",
        actual_context=context,
        evidence_bundle=bundle,
        expected_intent=AnswerIntent.ENTITY_DEFINITION,
        required_elements=("AxeI",),
    )
    assert result.passed
    assert result.score == 1.0
    assert result.failed_dimensions == []


def test_wrong_intent_fails_intent_correctness_dimension():
    context = build_query_context("What is AxeI meter?")
    result = score_answer(
        question="What is AxeI meter?",
        answer_text="Some answer.",
        actual_context=context,
        expected_intent=AnswerIntent.TROUBLESHOOTING,
    )
    assert not result.passed
    assert "intent_correctness" in result.failed_dimensions


def test_search_results_page_failure_is_detected():
    context = build_query_context("What is AxeI meter?")
    result = score_answer(
        question="What is AxeI meter?",
        answer_text="Here are some related documents: Doc A, Doc B.",
        actual_context=context,
    )
    assert not result.passed
    assert "answers_the_question" in result.failed_dimensions


def test_weak_sufficiency_without_honest_admission_fails_uncertainty_dimension():
    context = build_query_context("What is Zorblex 9000?")
    bundle = _bundle(sufficiency=SufficiencyLevel.WEAK)
    result = score_answer(
        question="What is Zorblex 9000?",
        answer_text="Zorblex 9000 is definitely a meter model used in RF Mesh deployments.",
        actual_context=context,
        evidence_bundle=bundle,
    )
    assert not result.passed
    assert "uncertainty_correctness" in result.failed_dimensions


def test_weak_sufficiency_with_honest_admission_passes_uncertainty_dimension():
    context = build_query_context("What is Zorblex 9000?")
    bundle = _bundle(sufficiency=SufficiencyLevel.WEAK)
    result = score_answer(
        question="What is Zorblex 9000?",
        answer_text="I don't have enough evidence to determine what Zorblex 9000 is.",
        actual_context=context,
        evidence_bundle=bundle,
    )
    assert result.details["uncertainty_correctness"]


def test_historical_recommendation_without_caveat_fails_authority_dimension():
    context = build_query_context("What was the resolution in similar cases?")
    bundle = _bundle(
        claims=[Claim(claim_id="claim-001", text="Reset the queue.", supported_by=["Case A"], authority=SourceAuthority.HISTORICAL_RECOMMENDATION, category="recommendation")]
    )
    result = score_answer(
        question="What was the resolution in similar cases?",
        answer_text="The resolution is to reset the queue.",
        actual_context=context,
        evidence_bundle=bundle,
    )
    assert not result.details["source_authority_appropriate"]


def test_settled_fix_language_without_current_claim_fails_historical_vs_current_dimension():
    context = build_query_context("What was the resolution in similar cases?")
    bundle = _bundle(
        claims=[Claim(claim_id="claim-001", text="A past case recorded taking an action -- historical recommendation, not a confirmed current resolution.", supported_by=["Case A"], authority=SourceAuthority.HISTORICAL_RECOMMENDATION, category="recommendation")]
    )
    result = score_answer(
        question="What was the resolution in similar cases?",
        answer_text="This has been resolved by resetting the queue.",
        actual_context=context,
        evidence_bundle=bundle,
    )
    assert not result.details["historical_vs_current_correctness"]


def test_forbidden_claim_detected():
    context = build_query_context("What is AxeI meter?")
    result = score_answer(
        question="What is AxeI meter?",
        answer_text="AxeI meter is manufactured by Acme Corp and runs firmware v9.",
        actual_context=context,
        forbidden_claims=("manufactured by Acme Corp",),
    )
    assert not result.passed
    assert "no_forbidden_claims" in result.failed_dimensions


def test_contradictory_sufficiency_without_real_contradictions_fails_evidence_relevance():
    context = build_query_context("What is AxeI meter?")
    bundle = _bundle(sufficiency=SufficiencyLevel.CONTRADICTORY, contradictions=[])
    result = score_answer(
        question="What is AxeI meter?",
        answer_text="Some answer.",
        actual_context=context,
        evidence_bundle=bundle,
    )
    assert not result.details["evidence_relevance"]
