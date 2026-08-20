"""``PromptBuilder`` -- converts an already-computed
``StructuredResolution`` (plus the user's own question) into an LLM
prompt (Chat Assistant Phase 1 -- Qwen 4B/Ollama integration).

Pure text formatting only -- zero matching/retrieval/ranking/reasoning
logic of its own. Every fact this emits was already computed by
``RecommendationEngine``/``StructuredResolutionEngine`` by the time
this runs; this module only gives it a consistent, evidence-bounded
textual shape for an LLM to read. ``StructuredResolution`` (not
``InvestigationStrategy``) is the input specifically because it is
already this project's deliberate "read model" unifying every evidence
source into one shape (see ``app.domain.structured_resolution``'s own
module docstring) -- every text field on it (root cause, resolution
text, validation instructions) is already capped upstream (e.g.
``RecommendationEngine._snippet()``), so no additional capping is
needed here.

One fixed system prompt, not varied per request, so its behavior is
auditable independent of any one answer -- the same principle already
written into ``RESOLVEIQ_LLM_GATEWAY_DESIGN.md`` Section 3 (a design
for a later, larger phase, but the "fixed, versioned system prompt"
principle applies just as well to this smaller Phase 1 slice).
"""

from __future__ import annotations

from app.domain.structured_resolution import ResolutionCandidate, StructuredResolution

_SYSTEM_PROMPT = """\
You are ResolveIQ's chat assistant. You answer questions about a support \
investigation using ONLY the evidence supplied to you below -- you have no \
other knowledge of this organization's systems, customers, or history.

Rules you must follow exactly:
1. Answer ONLY from the supplied evidence. Never invent facts, details, or \
context that were not given to you.
2. Preserve the exact confidence tier supplied to you: Confirmed, Likely, \
Possible, or Unknown. Never upgrade it, and never state a higher tier than \
the one given.
3. Never use the words "confirmed" or "verified" (or equivalent language) \
for anything below the Confirmed tier, even to negate it.
4. If the evidence is thin, or the confidence tier is Unknown, say so \
explicitly rather than filling the gap with plausible-sounding text.
5. Do not claim knowledge of the customer, technology, incident, root \
cause, resolution, or validation steps beyond what is explicitly supplied \
in the evidence below.
6. If more than one resolution candidate is supplied, prefer the one \
marked PRIMARY, but you may mention the others exist.
"""


def _format_evidence_reference(label: str, reference) -> str:
    parts = [f"  - [{label}] {reference.title} (source_id={reference.source_id})"]
    if reference.reason:
        parts.append(f"    Reason: {reference.reason}")
    if reference.score is not None:
        parts.append(f"    Score: {reference.score:.2f}")
    return "\n".join(parts)


def _format_resolution_candidate(index: int, candidate: ResolutionCandidate) -> str:
    marker = "PRIMARY" if candidate.is_primary else "alternate"
    lines = [f"  Candidate {index} [{marker}]: {candidate.text}"]
    lines.append(_format_evidence_reference("evidence", candidate.evidence))
    return "\n".join(lines)


class PromptBuilder:
    """See module docstring. Stateless -- safe to construct once and
    reuse (``ChatOrchestrator`` does exactly that)."""

    def build(self, question: str, structured: StructuredResolution) -> tuple[str, str]:
        """Returns ``(system_prompt, user_prompt)``. Deterministic:
        the same ``(question, structured)`` pair always produces the
        exact same two strings."""
        sections: list[str] = []

        sections.append("=== USER QUESTION ===")
        sections.append(question.strip())

        sections.append("\n=== PROBLEM ===")
        sections.append(structured.problem)

        sections.append("\n=== SYMPTOMS ===")
        sections.append(structured.symptoms)

        sections.append("\n=== ROOT CAUSE ===")
        if structured.root_cause:
            sections.append(structured.root_cause)
            if structured.root_cause_evidence:
                sections.append("Evidence for this root cause:")
                sections.extend(
                    _format_evidence_reference("root_cause", ref) for ref in structured.root_cause_evidence
                )
        else:
            sections.append("No root cause has been determined from the supplied evidence.")

        sections.append("\n=== RESOLUTION CANDIDATES ===")
        if structured.resolution_candidates:
            sections.extend(
                _format_resolution_candidate(i, candidate)
                for i, candidate in enumerate(structured.resolution_candidates, start=1)
            )
        else:
            sections.append("No resolution candidates are available from the supplied evidence.")

        sections.append("\n=== VALIDATION STEPS ===")
        if structured.validation_steps:
            for step in structured.validation_steps:
                line = f"  - {step.instruction} (source: {step.source_title})"
                sections.append(line)
        else:
            sections.append("No validation steps are available from the supplied evidence.")

        sections.append("\n=== APPLICABILITY ===")
        applicability = structured.applicability
        applicability_parts = []
        if applicability.customer_names:
            applicability_parts.append(f"Customers: {', '.join(applicability.customer_names)}")
        if applicability.region_names:
            applicability_parts.append(f"Regions: {', '.join(applicability.region_names)}")
        if applicability.component_names:
            applicability_parts.append(f"Components: {', '.join(applicability.component_names)}")
        if applicability.technology_name:
            applicability_parts.append(f"Technology: {applicability.technology_name}")
        sections.append("; ".join(applicability_parts) if applicability_parts else "Not tagged to any known dimension.")

        sections.append("\n=== CONFIDENCE ===")
        sections.append(f"Tier: {structured.confidence.value}")
        if structured.confidence_rationale:
            sections.append(f"Rationale: {structured.confidence_rationale}")

        user_prompt = "\n".join(sections)
        return _SYSTEM_PROMPT, user_prompt
