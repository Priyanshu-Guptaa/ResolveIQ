"""Tests for app.engines.chat.grounding_validator (Final Hardening
Pass, Objective 1). Unit-level, no database, no LLM -- pure-function
tests against ``validate(answer_text, evidence_text, confidence=...)``.
"""

from __future__ import annotations

from app.domain.provenance import ResolutionProvenance
from app.engines.chat.grounding_validator import check_no_material_loss, validate


def test_valid_meter_id_passes():
    result = validate("Meter 12345678 failed.", "Meter: 12345678")
    assert result.valid
    assert result.unsupported_claims == []


def test_fabricated_meter_id_is_rejected():
    result = validate("Meter 99999999 failed.", "Meter: 12345678")
    assert not result.valid
    assert result.severity == "high"
    assert any(c.claim_type == "identifier" and c.excerpt == "99999999" for c in result.unsupported_claims)


def test_valid_timestamp_passes():
    result = validate("The timeout occurred at 10:01:12.", "10:01:03 request\n10:01:12 timeout")
    assert result.valid


def test_fabricated_timestamp_is_rejected():
    result = validate("The timeout occurred at 10:01:30.", "10:01:03 request\n10:01:12 timeout")
    assert not result.valid
    assert any(c.claim_type == "timestamp" for c in result.unsupported_claims)


def test_valid_timing_delta_passes():
    result = validate("Request timed out after 9 seconds.", "10:01:03 request\n10:01:12 timeout")
    assert result.valid


def test_fabricated_timing_delta_is_rejected():
    result = validate("Request timed out after 20 seconds.", "10:01:03 request\n10:01:12 timeout")
    assert not result.valid
    assert any(c.claim_type == "timing_delta" for c in result.unsupported_claims)


def test_valid_http_error_passes():
    result = validate("HTTP 500 occurred.", "Error: HTTP 500 Internal Server Error")
    assert result.valid


def test_fabricated_http_error_is_rejected():
    result = validate("HTTP 404 occurred.", "Error: HTTP 500 Internal Server Error")
    assert not result.valid
    assert any(c.claim_type == "error" for c in result.unsupported_claims)


def test_valid_case_id_passes():
    result = validate("See CS0122697 for details.", "Related ticket: CS0122697")
    assert result.valid


def test_fabricated_case_id_is_rejected():
    result = validate("See CS9999999 for details.", "Related ticket: CS0122697")
    assert not result.valid
    assert any(c.claim_type == "case_id" for c in result.unsupported_claims)


def test_documented_configuration_value_passes_unmodified():
    answer = "Set timeout to 30 seconds."
    result = validate(answer, "Documented timeout = 30 seconds")
    assert result.valid
    assert result.repaired_text == answer


def test_undocumented_configuration_value_is_repaired_not_rejected():
    """Step 1F's own worked example: repaired, never a full-answer
    rejection -- the overall answer is still ``valid``, but the
    specific unsupported instruction is replaced."""
    result = validate("Set timeout to 60 seconds.", "Documented timeout = 30 seconds")
    assert result.valid  # low severity -- repaired, not rejected
    assert result.severity == "low"
    assert "60 seconds" not in result.repaired_text
    assert "not established from current evidence" in result.repaired_text
    assert any(c.claim_type == "configuration_value" and c.severity == "low" for c in result.unsupported_claims)


def test_unsupported_root_cause_language_rejected_below_confirmed():
    result = validate(
        "The root cause is the collector database.",
        "Logs show database-related errors.",
        confidence=ResolutionProvenance.POSSIBLE,
    )
    assert not result.valid
    assert any(c.claim_type == "root_cause_language" for c in result.unsupported_claims)


def test_definitive_root_cause_language_allowed_at_confirmed_tier():
    result = validate(
        "The root cause is the collector database.",
        "Logs show database-related errors.",
        confidence=ResolutionProvenance.CONFIRMED,
    )
    assert result.valid


def test_hedged_root_cause_language_is_never_a_false_positive():
    """Step 1H's own GOOD example must never itself be rejected."""
    result = validate(
        "The logs show database-related errors. This makes the database path a possible failure area, but the "
        "root cause is not confirmed from the current evidence.",
        "Logs show database-related errors.",
        confidence=ResolutionProvenance.POSSIBLE,
    )
    assert result.valid
    assert result.unsupported_claims == []


def test_unsupported_mandate_language_is_rejected():
    result = validate(
        "The configuration must be set to strict mode.",
        "No configuration values are documented here.",
        confidence=ResolutionProvenance.POSSIBLE,
    )
    assert not result.valid
    assert any(c.claim_type == "mandate_language" for c in result.unsupported_claims)


def test_multiple_fabricated_facts_are_all_caught_together():
    """The exact Step 3G combo hallucination scenario: a fabricated
    meter, timestamp, and error code together in one answer."""
    result = validate(
        "Meter 99999999 failed at 11:11:11 with HTTP 404.",
        "Meter: 12345678\nTimestamp: 10:00:00\nError: HTTP 500",
    )
    assert not result.valid
    claim_types = {c.claim_type for c in result.unsupported_claims}
    assert claim_types == {"identifier", "timestamp", "error"}


def test_clean_grounded_answer_has_no_issues():
    result = validate(
        "Meter 12345678 reported a communication failure at 10:00:00. HTTP 500 was observed.",
        "Meter: 12345678\nTimestamp: 10:00:00\nError: HTTP 500",
        confidence=ResolutionProvenance.POSSIBLE,
    )
    assert result.valid
    assert result.unsupported_claims == []
    assert result.severity == "none"


def test_prose_evidence_identifier_is_recognized_without_a_colon():
    """The evidence side must be scanned with the SAME loose,
    separator-optional pattern as the answer side -- otherwise a real,
    legitimately-evidenced identifier phrased in prose (as this
    codebase's own StructuredResolution.problem/root_cause text often
    is) would be wrongly rejected as fabricated."""
    result = validate(
        "Meter 12345678 failed.",
        "Investigation title: Meter 12345678 stopped reporting reads.",
    )
    assert result.valid


# --- check_no_material_loss (Final LLM Orchestration Hardening, Step 10's ---
# --- "practical acceptance check" -- an LLM answer that technically       --
# --- generated successfully and contains no fabrication can still be      --
# --- unacceptable because it silently drops real evidence a rich          --
# --- deterministic answer already established.


def test_check_no_material_loss_flags_a_dropped_source_citation():
    """The exact real bug this phase fixes: a real, cited documentation
    match exists in the deterministic answer, but the (safe, non-
    fabricating) LLM answer never mentions it at all."""
    deterministic = (
        'Direct answer:\nBased on ResolveIQ\'s documentation "Access to Dashboard and Views in CRM": How to access.'
    )
    thin_llm_answer = "The question cannot be answered based on the provided evidence."
    missing = check_no_material_loss(deterministic, thin_llm_answer)
    assert any("Access to Dashboard and Views in CRM" in m for m in missing)


def test_check_no_material_loss_is_empty_for_plain_tier_prose_with_no_citations():
    """The tier-based CONFIRMED/LIKELY composer text never quotes a
    title and rarely cites a bare identifier/timestamp -- a generic
    LLM rewording of it must not be penalized for content that was
    never structurally required in the first place."""
    deterministic = (
        "Based on the available evidence, this issue is likely related to: Collector lost network route to the "
        "mesh gateway. This has not been independently verified."
    )
    generic_llm_answer = "Generated grounded answer."
    assert check_no_material_loss(deterministic, generic_llm_answer) == []


def test_check_no_material_loss_flags_dropped_identifiers_and_timestamps():
    deterministic = (
        "Timeline (observed, in order):\n10:00:00  INFO  event. meter_number: 12345678\n10:00:10  ERROR  event."
    )
    thin_llm_answer = "The log shows an error occurred."
    missing = check_no_material_loss(deterministic, thin_llm_answer)
    assert any("12345678" in m for m in missing)
    assert any("10:00:00" in m for m in missing)
    assert any("10:00:10" in m for m in missing)


def test_check_no_material_loss_is_empty_when_the_llm_answer_retains_everything():
    deterministic = 'Based on ResolveIQ\'s documentation "Access to Dashboard and Views in CRM": How to access.'
    good_llm_answer = 'Here is how to access it, per ResolveIQ\'s documentation "Access to Dashboard and Views in CRM".'
    assert check_no_material_loss(deterministic, good_llm_answer) == []
