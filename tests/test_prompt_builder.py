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


def test_prompt_contains_problem():
    _, user_prompt = PromptBuilder().build("q", _structured_resolution())
    assert "RF Mesh IP command timeout" in user_prompt


def test_prompt_contains_root_cause():
    _, user_prompt = PromptBuilder().build("q", _structured_resolution())
    assert "Collector lost network route to the mesh gateway." in user_prompt


def test_prompt_contains_only_primary_resolution_candidate():
    """Phase 9 -- Phase 8's real-Qwen investigation found that showing
    non-primary candidates invites Qwen to re-rank/re-select a
    resolution it has no business choosing; ``is_primary`` is
    ResolveIQ's own already-made decision, so only that text is sent."""
    _, user_prompt = PromptBuilder().build("q", _structured_resolution())
    assert "Restart the collector service." in user_prompt
    assert "Apply configuration change Y instead of restarting." not in user_prompt


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
    """Phase 10 -- wording was compacted, but every one of these
    semantic guarantees (not the Phase 1/9 exact phrasing) must
    survive; see test_system_prompt_compaction_preserves_all_14_semantics
    below for the full enumerated list."""
    system_prompt, _ = PromptBuilder().build("q", _structured_resolution())
    assert "ONLY from the facts below" in system_prompt
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
    assert "No resolution candidate is available" in user_prompt


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


# --- Chat Assistant Phase 9 -- minimal deterministic-facts prompt --------
#
# Phase 8's real-Qwen investigation (read-only) proved the full-evidence
# shape invited Qwen to re-do work ResolveIQ already finished, and that a
# minimal shape which drops ``confidence_rationale`` loses the one fact
# that establishes a prior/historical occurrence. The tests below lock in
# both findings: (1) the reduced shape stays reduced, (2) the one field
# Phase 8 proved necessary is never dropped.


def test_symptoms_are_excluded():
    """Phase 9 Step 3 -- excluded: never needed to answer any question
    tested across Phases 3-8; the root cause and resolution already
    describe the operative facts."""
    _, user_prompt = PromptBuilder().build("q", _structured_resolution())
    assert "Meters stopped responding to commands." not in user_prompt


def test_raw_evidence_reference_details_are_excluded():
    """Phase 9 Step 3 -- excluded: source_id/score/reason detail blocks
    for root_cause_evidence and per-candidate evidence are UI-oriented
    and add tokens without adding anything Qwen needs to phrase an
    answer; the *fact* they represent is already carried in prose by
    ``confidence_rationale`` (see test_historical_context_is_preserved
    below)."""
    _, user_prompt = PromptBuilder().build("q", _structured_resolution())
    assert "0.82" not in user_prompt
    assert "Matches historical investigation" not in user_prompt
    assert "kb-1" not in user_prompt
    assert "The record's own recorded resolution." not in user_prompt


def test_historical_context_is_preserved_via_confidence_rationale():
    """Phase 9 Step 4 -- CRITICAL HISTORICAL-MATCH PROTECTION. Phase 8's
    Variant C dropped ``confidence_rationale`` and, as a direct result,
    Qwen incorrectly stated a fixture's issue "has not been documented
    previously" even though the fixture contained a real prior match.
    ``confidence_rationale`` is the deterministic fact that answers
    "why is the tier what it is / is this based on a prior occurrence" --
    it must never be silently dropped from the production prompt."""
    _, user_prompt = PromptBuilder().build("Has this happened before?", _structured_resolution())
    assert "Single local match at 82% similarity with a recorded root cause." in user_prompt


def test_confidence_rationale_omission_produces_honest_fallback_not_silence():
    """The rationale section is never simply absent -- when no rationale
    was supplied, the model is told so explicitly rather than the
    section vanishing (which could read as "no info was withheld")."""
    structured = _structured_resolution(confidence_rationale="")
    _, user_prompt = PromptBuilder().build("q", structured)
    assert "No rationale was supplied." in user_prompt


def test_applicability_is_a_single_compact_line_not_a_raw_structure():
    """Phase 8 Step 6/Test 6 -- applicability must be represented, but
    never as a duplicated raw structure. Each governed value appears
    exactly once, on one line, not repeated across multiple blocks."""
    _, user_prompt = PromptBuilder().build("q", _structured_resolution())
    assert user_prompt.count("TEPCO") == 1
    assert user_prompt.count("APAC") == 1


def test_prompt_size_does_not_grow_with_alternate_candidate_count():
    """Phase 9 Step 8 -- no raw candidate explosion: only the primary
    candidate's text ever reaches the prompt, so adding more alternates
    must not measurably grow it."""
    baseline = _structured_resolution()
    _, baseline_prompt = PromptBuilder().build("q", baseline)

    many_alternates = [baseline.resolution_candidates[0]] + [
        ResolutionCandidate(
            text=f"Alternate workaround number {i} with a reasonably long descriptive sentence attached to it.",
            evidence=EvidenceReference(
                kind=EvidenceKind.KNOWN_BUG, source_id=f"kb-{i}", title=f"Alternate bug {i}",
                reason="Some alternate workaround, never the selected one.",
            ),
            is_primary=False,
        )
        for i in range(10)
    ]
    swollen = _structured_resolution(resolution_candidates=many_alternates)
    _, swollen_prompt = PromptBuilder().build("q", swollen)

    assert len(swollen_prompt) == len(baseline_prompt)
    for i in range(10):
        assert f"Alternate workaround number {i}" not in swollen_prompt


def test_system_prompt_states_work_is_already_done():
    """Phase 9 Step 5/9 -- the system prompt must explicitly state that
    retrieval/ranking/confidence/resolution-selection are already
    performed, and forbid re-deriving them -- Phase 8's finding was that
    this framing, not prompt size alone, is what shortens reasoning.
    Wording compacted in Phase 10 -- see that test suite below for the
    exact current substrings and the full semantic checklist."""
    system_prompt, _ = PromptBuilder().build("q", _structured_resolution())
    assert "Retrieval, ranking, root-cause selection, resolution selection" in system_prompt
    assert "ALREADY DONE" in system_prompt
    assert "Do not re-rank" in system_prompt
    assert "Do not show your reasoning" in system_prompt


def test_thin_evidence_produces_all_three_honest_absence_messages():
    """Phase 9 Step 8/Test 10 -- the fully thin case (no root cause, no
    resolution, no validation steps) must represent every absence
    honestly, never fabricate placeholder content for any of them."""
    structured = _structured_resolution(
        root_cause=None, root_cause_evidence=[], resolution_candidates=[], validation_steps=[],
        confidence=ResolutionProvenance.UNKNOWN, confidence_rationale="No source cleared any bar.",
    )
    _, user_prompt = PromptBuilder().build("What should I check?", structured)
    assert "No root cause has been determined from the supplied evidence." in user_prompt
    assert "No resolution candidate is available from the supplied evidence." in user_prompt
    assert "No validation steps are available from the supplied evidence." in user_prompt
    assert "unknown" in user_prompt


# --- Chat Assistant Phase 10 -- system prompt compaction ------------------
#
# Phase 9's system prompt (1,378 chars) had real, measured redundancy
# across its intro paragraph and numbered rules. Phase 10 compacted it
# to 875 chars (~37% smaller) while keeping every one of the 14 semantic
# guarantees Phase 10 Step 3 enumerated -- SAFETY > RELIABILITY >
# BREVITY, deliberately stopping well short of Phase 8 Variant C's
# 326-char prompt (whose extreme compactness caused its one observed
# historical-match failure).


def test_system_prompt_compaction_preserves_all_14_semantics():
    """One test per Phase 10 Step 3 semantic guarantee (1-14), asserted
    against the actual current system prompt text -- not the exact
    Phase 9 wording, which was deliberately changed."""
    system_prompt, _ = PromptBuilder().build("q", _structured_resolution())

    # 1. Use only supplied evidence/facts.
    assert "ONLY from the facts below" in system_prompt
    # 2. Never invent facts.
    assert "Never invent facts" in system_prompt
    # 3. Never upgrade confidence.
    assert "Never upgrade it" in system_prompt
    # 4. Preserve the supplied confidence tier.
    assert "Preserve the exact confidence tier" in system_prompt
    assert all(tier in system_prompt for tier in ("Confirmed", "Likely", "Possible", "Unknown"))
    # 5. Do not claim "confirmed"/"verified" for sub-confirmed evidence.
    assert '"confirmed"' in system_prompt and '"verified"' in system_prompt
    # 6/7/8/9. Do not perform retrieval / re-rank / select another root
    # cause or resolution -- already done, framed once.
    assert "ALREADY DONE" in system_prompt
    assert "Do not re-rank" in system_prompt
    assert "different root cause or resolution" in system_prompt
    # 10. Do not calculate confidence.
    assert "calculate a new one" in system_prompt
    # 11. Do not expose internal reasoning.
    assert "Do not show your reasoning" in system_prompt
    # 12. Answer the user's actual question.
    assert "Answer the user's actual question" in system_prompt
    # 13. Be concise.
    assert "concisely" in system_prompt
    # 14. If evidence is insufficient, say so explicitly.
    assert "say so explicitly" in system_prompt


def test_system_prompt_size_stays_within_phase_16_bound():
    """Regression guard against silent re-growth back toward Phase 9's
    1,378 chars (or worse). Threshold reflects Phase 10's 875 chars plus
    Phase 16's validated rule 8 addition (1,130 chars measured, matching
    Phase 15's real-model-validated Variant B system prompt exactly) --
    not an arbitrary number. Phase 10's own Absolute Rules still apply:
    no chasing Phase 8 Variant C's 326-char extreme if it would cost a
    safety guarantee, and rule 8 is here specifically because Phase 15
    showed omitting it costs real completeness."""
    system_prompt, _ = PromptBuilder().build("q", _structured_resolution())
    assert len(system_prompt) <= 1200, f"system prompt grew to {len(system_prompt)} chars (Phase 16 target: <=1200)"
    assert len(system_prompt) < 1378, "system prompt regressed to Phase 9 size or larger"


# --- Chat Assistant Phase 16 -- multi-part question completeness ----------
#
# Phases 12-15 investigated qwen2.5:3b as an alternative-model candidate
# (never the configured ollama_model) and found it would silently answer
# only one part of a two-part question. Phase 15 validated a fully
# generic fix (no fixture-specific wording, no injected fact, no
# numbered-answer mandate) across 29 real-model calls with zero
# hallucination. Rule 8 below is that validated instruction, added
# because the completeness gap is a property of multi-part questions in
# general, not specific to any one model.


def test_system_prompt_contains_multi_part_completeness_rule():
    """The exact Phase 15-validated instruction must be present
    verbatim -- not the Phase 14 BENCH1-specific wording, not the
    rejected numbered-answer mandate, not the rejected post-hoc
    completeness-verification instruction."""
    system_prompt, _ = PromptBuilder().build("q", _structured_resolution())
    assert (
        "Identify each distinct part of the user's question and answer every "
        "part explicitly." in system_prompt
    )
    assert "Answer the parts in the same order they were asked." in system_prompt
    assert (
        "say that it is unknown or cannot be determined rather than guessing"
        in system_prompt
    )


def test_multi_part_rule_contains_no_fixture_specific_or_rejected_wording():
    """Guards against accidentally implementing one of Phase 15's
    rejected variants instead of the validated one: no BENCH1-specific
    phrasing (Phase 14), no injected historical-match fact, no numbered-
    answer mandate (rejected Variant C -- fabricated a confidence
    percentage), no post-hoc completeness-verification instruction
    (rejected Variant D -- fabricated an applicability claim)."""
    system_prompt, _ = PromptBuilder().build("q", _structured_resolution())
    assert "has this happened before" not in system_prompt.lower()
    assert "what should be checked first" not in system_prompt.lower()
    assert "Historical match:" not in system_prompt
    assert "Number the answers" not in system_prompt
    assert "verify that every distinct question" not in system_prompt.lower()


def test_multi_part_rule_does_not_alter_existing_sections():
    """The new rule is additive only -- every existing fact section and
    rule from Phases 1/9/10 remains present and unchanged."""
    structured = _structured_resolution()
    system_prompt, user_prompt = PromptBuilder().build("Has this happened before?", structured)
    # Existing facts (user prompt) untouched.
    assert "Collector lost network route to the mesh gateway." in user_prompt
    assert "Restart the collector service." in user_prompt
    assert "Apply configuration change Y instead of restarting." not in user_prompt
    assert "TEPCO" in user_prompt
    assert "Single local match at 82% similarity with a recorded root cause." in user_prompt
    # Existing rules (system prompt) untouched.
    assert "ONLY from the facts below" in system_prompt
    assert "Never invent facts" in system_prompt
    assert "Never upgrade it" in system_prompt
    assert '"confirmed"' in system_prompt
    assert "Do not show your reasoning" in system_prompt
