"""Tests for PromptBuilder (Chat Assistant Phase 1 -- Qwen 4B/Ollama
integration).

Pure unit tests -- no database, no HTTP, no LLM. Construct a realistic
StructuredResolution directly and assert the two prompts PromptBuilder
produces are deterministic and contain exactly the supplied evidence
(never more, never less).
"""

from __future__ import annotations

from app.domain.provenance import EvidenceKind, EvidenceReference, ResolutionProvenance, ValidationStep
from app.domain.structured_resolution import ApplicabilitySummary, ResolutionCandidate, StructuredResolution
from app.engines.llm.prompt_builder import PromptBuilder


def _structured_resolution(**overrides) -> StructuredResolution:
    root_cause_evidence = [
        EvidenceReference(
            kind=EvidenceKind.HISTORICAL_INVESTIGATION,
            source_id="hi-1",
            title="RF Mesh IP command timeout",
            reason="Matches historical investigation 'RF Mesh IP command timeout' (82% similarity).",
            score=0.82,
        )
    ]
    resolution_candidates = [
        ResolutionCandidate(
            text="Restart the collector service.",
            evidence=EvidenceReference(
                kind=EvidenceKind.HISTORICAL_INVESTIGATION,
                source_id="hi-1",
                title="RF Mesh IP command timeout",
                reason="The record's own recorded resolution.",
                score=0.82,
            ),
            is_primary=True,
        ),
        ResolutionCandidate(
            text="Apply configuration change Y instead of restarting.",
            evidence=EvidenceReference(
                kind=EvidenceKind.KNOWN_BUG,
                source_id="kb-1",
                title="Known RF Mesh IP collector bug",
                reason="The record's own recorded workaround.",
            ),
            is_primary=False,
        ),
    ]
    validation_steps = [
        ValidationStep(
            instruction="Confirm the collector's route table is restored.",
            source=EvidenceKind.HISTORICAL_INVESTIGATION,
            source_id="hi-1",
            source_title="RF Mesh IP command timeout",
        )
    ]
    defaults = dict(
        source_kind="historical_investigation",
        source_id="hi-1",
        problem="RF Mesh IP command timeout",
        symptoms="Meters stopped responding to commands.",
        applicability=ApplicabilitySummary(
            customer_names=["TEPCO"], region_names=["APAC"], component_names=["Network Hub"], technology_name="RF Mesh IP"
        ),
        root_cause="Collector lost network route to the mesh gateway.",
        root_cause_evidence=root_cause_evidence,
        resolution_candidates=resolution_candidates,
        validation_steps=validation_steps,
        confidence=ResolutionProvenance.LIKELY,
        confidence_rationale="Single local match at 82% similarity with a recorded root cause.",
    )
    defaults.update(overrides)
    return StructuredResolution(**defaults)


def test_prompt_contains_question():
    system_prompt, user_prompt = PromptBuilder().build("Has this happened before?", _structured_resolution())
    assert "Has this happened before?" in user_prompt


def test_prompt_contains_problem_and_symptoms():
    _, user_prompt = PromptBuilder().build("q", _structured_resolution())
    assert "RF Mesh IP command timeout" in user_prompt
    assert "Meters stopped responding to commands." in user_prompt


def test_prompt_contains_root_cause_and_its_evidence():
    _, user_prompt = PromptBuilder().build("q", _structured_resolution())
    assert "Collector lost network route to the mesh gateway." in user_prompt
    assert "hi-1" in user_prompt
    assert "82% similarity" in user_prompt


def test_prompt_contains_every_resolution_candidate_with_primary_designation():
    _, user_prompt = PromptBuilder().build("q", _structured_resolution())
    assert "Restart the collector service." in user_prompt
    assert "Apply configuration change Y instead of restarting." in user_prompt
    assert "[PRIMARY]" in user_prompt
    assert "[alternate]" in user_prompt


def test_prompt_contains_validation_steps():
    _, user_prompt = PromptBuilder().build("q", _structured_resolution())
    assert "Confirm the collector's route table is restored." in user_prompt


def test_prompt_contains_applicability():
    _, user_prompt = PromptBuilder().build("q", _structured_resolution())
    assert "TEPCO" in user_prompt
    assert "APAC" in user_prompt
    assert "Network Hub" in user_prompt
    assert "RF Mesh IP" in user_prompt


def test_prompt_contains_confidence_tier_and_rationale():
    _, user_prompt = PromptBuilder().build("q", _structured_resolution())
    assert "likely" in user_prompt
    assert "Single local match at 82% similarity with a recorded root cause." in user_prompt


def test_system_prompt_contains_grounding_and_trust_instructions():
    system_prompt, _ = PromptBuilder().build("q", _structured_resolution())
    assert "ONLY from the supplied evidence" in system_prompt
    assert "Never invent facts" in system_prompt
    assert "Confirmed" in system_prompt and "Likely" in system_prompt and "Possible" in system_prompt and "Unknown" in system_prompt
    assert "Never upgrade" in system_prompt
    assert '"confirmed"' in system_prompt or "confirmed" in system_prompt.lower()


def test_no_root_cause_produces_honest_absence_language_not_fabrication():
    structured = _structured_resolution(root_cause=None, root_cause_evidence=[])
    _, user_prompt = PromptBuilder().build("q", structured)
    assert "No root cause has been determined" in user_prompt
    assert "Collector lost network route" not in user_prompt


def test_no_resolution_candidates_produces_honest_absence_language():
    structured = _structured_resolution(resolution_candidates=[])
    _, user_prompt = PromptBuilder().build("q", structured)
    assert "No resolution candidates are available" in user_prompt


def test_unknown_confidence_tier_is_preserved_verbatim():
    structured = _structured_resolution(confidence=ResolutionProvenance.UNKNOWN, confidence_rationale="No source cleared any bar.")
    _, user_prompt = PromptBuilder().build("q", structured)
    assert "unknown" in user_prompt
    assert "No source cleared any bar." in user_prompt


def test_build_is_deterministic():
    structured = _structured_resolution()
    builder = PromptBuilder()
    first = builder.build("Has this happened before?", structured)
    second = builder.build("Has this happened before?", structured)
    assert first == second
