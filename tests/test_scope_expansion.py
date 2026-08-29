"""Tests for app.engines.chat.scope_expansion (Chat Assistant Phase
35B).

Pure unit tests -- deterministic pattern matching only, no LLM, no
database. See the module's own docstring for why this exists
(spontaneous, non-scope-question customer-scope-expansion fabrication,
distinct from app.engines.chat.scope_question's explicit-question
mechanism) and for its integration-order requirement.
"""

from __future__ import annotations

from app.engines.chat.scope_expansion import (
    SCOPE_EXPANSION_PHRASES,
    contains_unsupported_scope_expansion,
)


def test_detects_the_actual_phase_35_failure_text():
    text = (
        "This issue has likely occurred before with other customers in the APAC region "
        "using Network Hub technology."
    )
    assert contains_unsupported_scope_expansion(text)


def test_detects_the_actual_phase_35b_reproduction_text():
    text = (
        "This issue applies to customers in the APAC region, specifically to the "
        "Network Hub component using the RF Mesh IP technology."
    )
    assert contains_unsupported_scope_expansion(text)


def test_detects_every_minimum_required_phrase():
    required = [
        "This affects other customers.",
        "This affects all customers.",
        "This affects additional customers.",
        "There could be any other customer affected.",
        "This is a multiple customers issue.",
        "This has affected several customers.",
        "This has affected many customers.",
        "There are other regions affected.",
        "This suggests a broader deployment issue.",
        "This is an industry-wide problem.",
        "This is a widespread issue.",
    ]
    for text in required:
        assert contains_unsupported_scope_expansion(text), text


def test_does_not_flag_a_single_named_customer():
    ordinary = [
        "This has happened before with a customer named TEPCO in the APAC region.",
        "The customer applicability is TEPCO. Customer: TEPCO, Region: APAC.",
        "Root cause: Collector lost network route to the mesh gateway. This has happened before.",
        "You should first check if the collector's route table is restored.",
    ]
    for text in ordinary:
        assert not contains_unsupported_scope_expansion(text), text


def test_does_not_flag_multiple_legitimately_named_customers():
    """A fixture with several REAL, supplied customers must remain
    fully answerable (Absolute Rule 20) -- naming them is never
    'expansion', since expansion means claiming MORE than supplied."""
    text = "Customer: TEPCO, PG&E. Both TEPCO and PG&E are known to be affected by this issue."
    assert not contains_unsupported_scope_expansion(text)


def test_does_not_flag_the_deterministic_scope_statement_itself():
    """Integration-order regression guard: customer_scope_statement()'s
    own output uses overlapping vocabulary ("any other customer") to
    correctly express the ABSENCE of broader scope -- this function
    must only ever be applied to the raw LLM text before that
    statement is appended (see orchestrator.py's call site), but this
    test documents the actual text shape callers must never re-check."""
    deterministic_text = (
        "Customer impact scope: TEPCO is known to be affected. "
        "Whether any other customer is also affected is not established."
    )
    # This IS expected to match the phrase table (documenting exactly
    # why the ordering requirement in the module docstring exists) --
    # this test exists to make that fact explicit and regression-safe,
    # not to assert False here.
    assert contains_unsupported_scope_expansion(deterministic_text)


def test_scope_expansion_phrases_are_never_lone_generic_words():
    assert not any(phrase.strip() in ("customer", "customers", "region", "regions") for phrase in SCOPE_EXPANSION_PHRASES)


def test_named_region_pattern_requires_both_customers_and_region_words():
    assert not contains_unsupported_scope_expansion("This happened in the APAC region.")
    assert not contains_unsupported_scope_expansion("The customer is in APAC.")
    assert contains_unsupported_scope_expansion("This affected customers in the APAC region.")
    assert contains_unsupported_scope_expansion("This affected customers across the EMEA region.")


# --- Phase 45 -- replay of the exact real qwen2.5:3b outputs captured in ----
# --- Phase 44's real-Ollama smoke test (single-customer TEPCO fixture,      ---
# --- question: "Has this happened before, and does this affect other       ---
# --- customers?"). Verbatim text, not paraphrased or reconstructed.        ---


def test_detects_the_exact_phase_44_multipart_run_1_output():
    text = (
        "Has this happened before for TEPCO in APAC? This issue affects other customers. "
        "The root cause identified is a Collector lost network route to the mesh gateway. "
        "The recommended resolution is to Restart the Collector service. The validation "
        "step confirmed is to Confirm the Collector's route table is restored. The "
        "applicable customer is TEPCO, the applicable region is APAC, the applicable "
        "component is Network Hub, and the applicable technology is RF Mesh IP. The "
        "confidence level is Likely."
    )
    assert contains_unsupported_scope_expansion(text)


def test_detects_the_exact_phase_44_multipart_run_2_output():
    """Run 2's phrasing never says 'yes' outright, but still ends with
    the same unsupported 'other customers' language -- the gate is not
    required to parse grammatical ambiguity, only to catch the phrase."""
    text = (
        "Has this happened before for customer TEPCO in the APAC region? This issue "
        "affects the Network Hub component using RF Mesh IP technology. The collector "
        "lost its network route to the mesh gateway, leading to an RF Mesh IP command "
        "timeout. The root cause is a collector lost network route to the mesh gateway, "
        "and the resolution is to restart the collector service. Validation steps "
        "include confirming the collector's route table is restored. Evidence-backed "
        "checks are restarting the collector service and confirming the collector's "
        "route table is restored. The confidence level for this issue affecting other "
        "customers is Likely."
    )
    assert contains_unsupported_scope_expansion(text)


def test_detects_the_exact_phase_44_multipart_run_3_output():
    text = (
        "Has this happened before for TEPCO in APAC with component Network Hub using "
        "technology RF Mesh IP? This issue affects other customers as evidenced by the "
        "collected data. The root cause identified is a Collector lost network route to "
        "the mesh gateway. The resolution recommended is to restart the Collector "
        "service, which has been validated by confirming the Collector's route table is "
        "restored. No additional evidence-backed troubleshooting checks are required "
        "beyond what has been provided. The confidence level for this issue affecting "
        "other customers is likely based on the available evidence."
    )
    assert contains_unsupported_scope_expansion(text)
