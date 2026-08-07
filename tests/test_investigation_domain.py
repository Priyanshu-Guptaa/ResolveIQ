"""Pure domain-model tests for InvestigationSession -- no DB, no engine,
just the Pydantic model and its computed properties.
"""

from __future__ import annotations

from app.domain.enums import EvidenceType
from app.domain.evidence import Evidence
from app.domain.investigation import _MAX_CONTEXT_CHARS, InvestigationSession


def _investigation_with_evidence(*contents: str) -> InvestigationSession:
    investigation = InvestigationSession(title="Test investigation")
    for content in contents:
        investigation.add_evidence(
            Evidence(
                investigation_id=investigation.id,
                evidence_type=EvidenceType.LOG_FILE,
                raw_content=content,
            )
        )
    return investigation


def test_context_text_includes_title_and_all_small_evidence():
    investigation = _investigation_with_evidence("first note", "second note")
    ctx = investigation.context_text
    assert investigation.title in ctx
    assert "first note" in ctx
    assert "second note" in ctx


def test_context_text_is_capped_for_a_very_large_investigation():
    """Regression guard for a real bug: RecommendationEngine.generate()
    embeds context_text as its search query. A real investigation's
    evidence (97 items, several multi-MB log files) produced a 40MB
    context_text -- tokenizing that before the embedding model's own
    ~256-token window truncates it anyway made recommendations reliably
    time out. context_text itself must stay bounded regardless of how
    much evidence content exists."""
    investigation = _investigation_with_evidence("X" * 10_000_000, "Y" * 10_000_000, "Z" * 10_000_000)

    ctx = investigation.context_text

    assert len(ctx) <= _MAX_CONTEXT_CHARS + len(investigation.title) + 10  # a little slack for newlines


def test_context_text_prioritizes_title_and_earliest_evidence():
    """Cap is enforced while accumulating (earliest evidence first, since
    InvestigationSession.evidence is stored in creation order) -- an
    early, still-relevant piece of context (the task description, say)
    must never be crowded out entirely by whatever was uploaded last."""
    investigation = _investigation_with_evidence("EARLY_MARKER", "X" * 200_000)

    ctx = investigation.context_text

    assert "EARLY_MARKER" in ctx


def test_context_text_handles_evidence_with_no_content():
    investigation = _investigation_with_evidence("", "real content")
    ctx = investigation.context_text
    assert "real content" in ctx
