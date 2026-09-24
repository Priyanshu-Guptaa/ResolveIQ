"""Deterministic answer-quality scorer for the golden question set
(Final Support-Quality Pass, §7).

Explicitly NOT an LLM judge -- every dimension below is a closed,
structural check reusing already-computed, already-tested primitives
(``QueryContext``, ``EvidenceBundle``, plain substring checks), the
same "closed list, never a heuristic guess" idiom every classifier in
this codebase already follows. This module's own job is narrow: given
a question, the deterministic ``QueryContext``/``EvidenceBundle`` this
system already computed for it, and the final answer text, say WHICH
of ten real quality dimensions the answer satisfies -- never to judge
prose quality, tone, or anything requiring semantic understanding.

Deliberately tolerant of exact wording (§7's own "do not make the
scorer brittle" instruction): every text-presence check below looks
for one of several real, already-established phrasings this codebase's
own composers actually produce (e.g. the various "I don't have enough
evidence"/"couldn't find" honest-admission sentences already used
across ``_compose_knowledge_synthesis``/``_compose_l2_guidance_
synthesis``/etc.), never a single exact string."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.evidence_bundle import EvidenceBundle, SourceAuthority, SufficiencyLevel
from app.engines.chat.query_intent import AnswerIntent, QueryContext

_SEARCH_RESULTS_PAGE_PHRASES: tuple[str, ...] = (
    "here are some related documents",
    "here are the relevant documents",
    "here are some documents",
)
"""The one explicit failure shape §7/the closing principle names by
name: a response that is effectively a document list, never an actual
answer. Closed, narrow -- never a general "sounds unhelpful" heuristic."""

_UNCERTAINTY_PHRASES: tuple[str, ...] = (
    "don't have enough evidence",
    "do not have enough evidence",
    "couldn't find",
    "could not find",
    "i found some",
    "no authoritative documentation",
    "does not contain authoritative",
    "i don't have enough evidence in the current knowledge base",
)
"""Real, already-existing honest-admission phrasings this codebase's
own composers produce (``_compose_knowledge_synthesis``,
``_compose_l2_guidance_synthesis``, the WEAK/INSUFFICIENT fallback
paths) -- never a single exact sentence, several real ones."""

_SETTLED_FIX_PHRASES: tuple[str, ...] = (
    "is the solution",
    "is now resolved",
    "has been resolved",
    "this resolves the issue",
    "this resolves it",
)
"""The exact anti-pattern historical_expansion.py's own gate targets --
reused here as a read-only check, never a second, divergent gate."""


@dataclass
class QualityScore:
    """The scorer's own output -- see module docstring. ``details``
    carries every individual dimension's boolean result (not only the
    failures) so a caller can see exactly what WAS checked, not only
    what failed."""

    score: float
    passed: bool
    failed_dimensions: list[str] = field(default_factory=list)
    details: dict[str, bool] = field(default_factory=dict)


_DIMENSION_COUNT = 10


def score_answer(
    *,
    question: str,
    answer_text: str,
    actual_context: QueryContext,
    evidence_bundle: EvidenceBundle | None = None,
    expected_intent: AnswerIntent | None = None,
    required_elements: tuple[str, ...] = (),
    forbidden_claims: tuple[str, ...] = (),
) -> QualityScore:
    """Scores one real (question, answer) pair against 10 deterministic
    dimensions (§7's own numbered list). ``expected_intent``/
    ``required_elements``/``forbidden_claims`` are optional -- when
    omitted, that dimension is scored against the weaker, always-true
    "nothing contradicts it" bar rather than treated as a hard failure,
    so this function is usable even for a real production question with
    no pre-written golden expectations (Section 11's real-corpus
    validation), not only golden-set questions."""
    lower_answer = answer_text.lower()
    checks: dict[str, bool] = {}

    # 1. Intent correctness
    checks["intent_correctness"] = expected_intent is None or actual_context.intent == expected_intent

    # 2. Subject/context correctness -- a real question (not a thin
    # follow-up/unknown) should have resolved a real subject.
    if expected_intent is not None and expected_intent not in (AnswerIntent.FOLLOW_UP, AnswerIntent.UNKNOWN):
        checks["subject_context_correctness"] = bool(actual_context.subject)
    else:
        checks["subject_context_correctness"] = True

    # 3. Evidence relevance -- a CONTRADICTORY sufficiency must actually
    # carry real, surfaced contradictions, never a silent label.
    if evidence_bundle is not None and evidence_bundle.sufficiency == SufficiencyLevel.CONTRADICTORY:
        checks["evidence_relevance"] = bool(evidence_bundle.contradictions)
    else:
        checks["evidence_relevance"] = True

    # 4. Source authority appropriateness -- every HISTORICAL_
    # RECOMMENDATION claim must itself carry the historical/not-
    # confirmed caveat in its own text (never silently promoted).
    checks["source_authority_appropriate"] = True
    if evidence_bundle is not None:
        for claim in evidence_bundle.claims:
            if claim.authority == SourceAuthority.HISTORICAL_RECOMMENDATION:
                claim_lower = claim.text.lower()
                if "not a confirmed" not in claim_lower and "historical recommendation" not in claim_lower:
                    checks["source_authority_appropriate"] = False

    # 5. Required answer elements
    checks["required_elements_present"] = all(elem.lower() in lower_answer for elem in required_elements)

    # 6. Claim grounding -- every claim traces to at least one real
    # source (never an unsupported claim -- see Claim.supported/
    # supported_by's own "always real" contract).
    checks["claim_grounding"] = evidence_bundle is None or all(c.supported_by for c in evidence_bundle.claims)

    # 7. Uncertainty correctness -- WEAK/INSUFFICIENT evidence must
    # produce a real, honest admission, never a confident-sounding
    # answer built from evidence that doesn't actually support it.
    if evidence_bundle is not None and evidence_bundle.sufficiency in (SufficiencyLevel.WEAK, SufficiencyLevel.INSUFFICIENT):
        checks["uncertainty_correctness"] = any(phrase in lower_answer for phrase in _UNCERTAINTY_PHRASES)
    else:
        checks["uncertainty_correctness"] = True

    # 8. Historical-vs-current correctness -- settled-fix language is
    # only acceptable when at least one claim is genuinely CURRENT.
    has_settled_fix_language = any(phrase in lower_answer for phrase in _SETTLED_FIX_PHRASES)
    has_current_claim = evidence_bundle is not None and any(c.is_current for c in evidence_bundle.claims)
    checks["historical_vs_current_correctness"] = not has_settled_fix_language or has_current_claim

    # 9. Forbidden-claim detection
    checks["no_forbidden_claims"] = not any(phrase.lower() in lower_answer for phrase in forbidden_claims)

    # 10. Actual question coverage -- never the bare "here are some
    # related documents" search-results-page failure, and never empty.
    checks["answers_the_question"] = (
        bool(answer_text.strip()) and not any(p in lower_answer for p in _SEARCH_RESULTS_PAGE_PHRASES)
    )

    failed = [name for name, ok in checks.items() if not ok]
    score = (_DIMENSION_COUNT - len(failed)) / _DIMENSION_COUNT
    return QualityScore(score=score, passed=not failed, failed_dimensions=failed, details=checks)
