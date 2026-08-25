"""``PromptBuilder`` -- converts an already-computed
``StructuredResolution`` (plus the user's own question) into an LLM
prompt (Chat Assistant Phase 1 -- Qwen 4B/Ollama integration; reshaped
Chat Assistant Phase 9 -- see "MINIMAL DETERMINISTIC-FACTS PROMPT"
below).

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

=== MINIMAL DETERMINISTIC-FACTS PROMPT (Chat Assistant Phase 9) ===
Phase 8's real-Qwen investigation (read-only) proved that presenting
Qwen with the *full* evidence structure -- every resolution candidate
(not just the selected one), every root-cause evidence block, the raw
applicability structure -- measurably invites it to re-do work
ResolveIQ has already finished (ranking, candidate selection,
confidence calculation), which was the dominant driver of the
reasoning-length/reliability problem documented across Phases 3-7.
Phase 8's real-Qwen A/B/C comparison on the identical fixture used
throughout this project:

    Variant A (old shape, full evidence)   -- 33% success,  ~122s
    Variant B (constrained, kept rationale) -- 100% success, ~226s
    Variant C (minimal, dropped rationale)  -- 100% success, ~38s,
        but ONE observed failure: incorrectly claimed the issue "has
        not been documented previously" -- traced specifically to
        Variant C omitting ``confidence_rationale``, the one field
        that (in this project's real data) carries the "this matches
        a prior historical occurrence" fact.

This module now emits Variant B's proven-safe field set (kept
``confidence_rationale`` -- zero failures across every Phase 8 run
that included it) shaped as compactly as Variant C's system prompt and
section structure, rather than either extreme alone. Concretely, this
means:

    INCLUDED (every field, with its reason):
    - user question       -- what's actually being answered.
    - problem              -- short subject anchor; a standalone
                               question ("has this happened before?")
                               is otherwise ungrounded.
    - selected root cause  -- the one and only root cause; there is no
                               second one to rank against.
    - selected PRIMARY resolution ONLY -- the one and only resolution;
                               alternates are deliberately never shown
                               (see "excluded" below).
    - validation steps     -- directly answers the common "what should
                               I check" class of question tested in
                               every phase since Phase 3.
    - confidence tier       -- must be echoed verbatim, never derived.
    - confidence_rationale  -- see the Phase 8 finding above: this is
                               the deterministic fact that establishes
                               whether the match is a real prior
                               occurrence, not raw evidence to weigh.
    - a compact one-line applicability summary -- answers "does this
                               apply to my customer/technology" without
                               the raw, multi-line structure Variant A
                               sent (Phase 8 Step 6/Test 6: represent
                               it, but never duplicate the raw shape).

    EXCLUDED (every field, with its reason):
    - symptoms              -- never needed to answer any question
                               tested across Phases 3-8; the root cause
                               and resolution already describe the
                               operative facts, so restating symptoms
                               is pure duplication with no observed
                               benefit.
    - non-primary resolution candidates -- Phase 8's core finding: this
                               is exactly what invited Qwen to re-rank
                               and re-select a resolution it has no
                               business choosing. ``is_primary`` is
                               ResolveIQ's own already-made decision.
    - root_cause_evidence / per-candidate EvidenceReference blocks
                               (title/source_id/score/reason per item)
                               -- the *fact* those blocks exist is what
                               ``confidence_rationale`` already
                               communicates in prose; the raw,
                               UI-oriented reference structure adds
                               tokens without adding anything Qwen
                               needs to phrase an answer.
    - log_evidence / sql_evidence / related / superseded_by /
      also_seen              -- UI/Relationship-Explorer-oriented
                               fields with no bearing on phrasing a
                               direct answer to the user's question;
                               never referenced by any tested question.

The system prompt now explicitly states retrieval/ranking/confidence/
resolution-selection are ALREADY DONE and forbids re-deriving them --
Phase 8 Step 9's finding was that this framing (present in Variant B/C,
absent from Variant A) is what actually shortens Qwen's reasoning, not
prompt size alone.

=== SYSTEM PROMPT COMPACTION (Chat Assistant Phase 10) ===
Phase 9's system prompt (1,378 chars) deliberately kept every Phase-1
safety phrase verbatim alongside Phase 9's new "already done" framing,
which left real, measured redundancy: the intro paragraph separately
enumerated "retrieval, ranking, root-cause selection, resolution
selection, applicability, and confidence calculation" AND rule 2
separately re-forbade re-ranking/re-selecting a root cause/resolution
AND rule 3 separately re-forbade calculating confidence -- three
sentences independently guaranteeing overlapping subsets of the same
handful of prohibitions, plus a "You have no other knowledge of this
organization's..." sentence that duplicated rule 1's "answer ONLY from
the facts below."

Phase 10 removes that redundancy -- never a safety guarantee -- taking
the system prompt from 1,378 to 875 chars (~37% smaller) while keeping
all 14 semantic guarantees Phase 10 Step 3 enumerated: use only
supplied facts; never invent; never upgrade confidence; preserve the
exact tier; never say confirmed/verified below Confirmed; and the
"already done" framing for retrieval/ranking/root-cause/resolution/
confidence, phrased once instead of three times. This deliberately
stops well short of Phase 8 Variant C's 326-char system prompt --
Phase 10's explicit priority order is SAFETY > RELIABILITY > BREVITY,
and Variant C's extreme compactness is exactly what was traded away
for its one observed historical-match failure in Phase 8.

=== MULTI-PART QUESTION COMPLETENESS (Chat Assistant Phase 16) ===
An alternative-model investigation (Phases 12-15, qwen2.5:3b as a
candidate faster/more-reliable local model -- not the configured
``ollama_model``, only ever exercised via a directly-constructed
``OllamaProvider`` in those phases' throwaway validation scripts) found
that a smaller model answering a two-part question ("has this happened
before, and what should I check first?") would reliably answer only
the second part, silently dropping the first even though the supplied
``confidence_rationale`` fact answered it. Phase 14 proved this was a
prompt *structure* problem, not a missing-fact problem (hardcoding an
explicit fact line worked, but so did a purely structural instruction
with no injected fact -- and the fact-injection approach independently
turned out to be the less safe of the two, see below). Phase 15 proved
a fully generic version of that structural instruction (no fixture-
specific wording, no injected fact, no numbered-answer mandate) fixes
the omission -- validated on the original two-part question, three
independently-constructed multi-part questions of different shapes,
and the thin-evidence fixture -- with zero hallucination across 29
real-model calls. Rule 8 below is that validated instruction, added
here because the same completeness gap has no reason to be specific to
any one alternative model -- it is a property of "answer every part of
a multi-part question," equally worth guaranteeing for whichever LLM
this prompt is ever sent to, including the currently-configured
``qwen3:4b`` (Phase 3-11's real-Qwen testing never happened to probe a
multi-part question, so this was simply never checked for it before).

Phase 15 explicitly rejected two stronger-looking variants precisely
because they were more failure-prone, not less: an explicit numbered-
answer mandate ("Number the answers...") once produced a fabricated
numeric confidence figure ("80% confident") that no fixture ever
supplied, and a post-hoc "verify every part was answered" instruction
once produced a fabricated applicability claim ("affecting all
customers") on a fixture with zero customer data -- in both cases the
model appears to have manufactured a specific-sounding fact under
implicit pressure to look complete. Rule 8's wording is deliberately
the plainer of the tested alternatives: it explicitly licenses "unknown
or cannot be determined" as a complete, acceptable answer, which is
exactly what avoided that failure mode in every one of Phase 15's 29
validation calls.
"""

from __future__ import annotations

from app.domain.structured_resolution import StructuredResolution

_SYSTEM_PROMPT = """\
You are ResolveIQ's chat assistant. Retrieval, ranking, root-cause \
selection, resolution selection, and confidence calculation are \
ALREADY DONE -- everything below is final. Your ONLY job is to write \
the final answer from these facts, not investigate them.

1. Answer ONLY from the facts below. Never invent facts, details, or \
context not given to you.
2. Do not re-rank, or choose a different root cause or resolution -- \
these are already final.
3. Preserve the exact confidence tier given: Confirmed, Likely, \
Possible, or Unknown. Never upgrade it or calculate a new one.
4. Never use the words "confirmed" or "verified" (or equivalent) \
below the Confirmed tier, even to negate it.
5. If evidence is thin, or confidence is Unknown, say so explicitly.
6. Answer the user's actual question directly and concisely.
7. Do not show your reasoning -- give the final answer only.
8. Identify each distinct part of the user's question and answer every \
part explicitly. Answer the parts in the same order they were asked. \
If the evidence does not establish an answer, say that it is unknown \
or cannot be determined rather than guessing.
"""


def _primary_candidate(structured: StructuredResolution):
    return next((c for c in structured.resolution_candidates if c.is_primary), None)


def _applicability_line(structured: StructuredResolution) -> str:
    applicability = structured.applicability
    parts = []
    if applicability.customer_names:
        parts.append(f"Customers: {', '.join(applicability.customer_names)}")
    if applicability.region_names:
        parts.append(f"Regions: {', '.join(applicability.region_names)}")
    if applicability.component_names:
        parts.append(f"Components: {', '.join(applicability.component_names)}")
    if applicability.technology_name:
        parts.append(f"Technology: {applicability.technology_name}")
    return "; ".join(parts) if parts else "Not tagged to any known dimension."


class PromptBuilder:
    """See module docstring. Stateless -- safe to construct once and
    reuse (``ChatOrchestrator`` does exactly that)."""

    def build(self, question: str, structured: StructuredResolution) -> tuple[str, str]:
        """Returns ``(system_prompt, user_prompt)``. Deterministic:
        the same ``(question, structured)`` pair always produces the
        exact same two strings. See the module docstring's "MINIMAL
        DETERMINISTIC-FACTS PROMPT" section for exactly which fields
        are included/excluded and why."""
        sections: list[str] = []

        sections.append("=== USER QUESTION ===")
        sections.append(question.strip())

        sections.append("\n=== PROBLEM ===")
        sections.append(structured.problem)

        sections.append("\n=== ROOT CAUSE (already selected -- final) ===")
        sections.append(structured.root_cause or "No root cause has been determined from the supplied evidence.")

        sections.append("\n=== RESOLUTION (already selected -- final) ===")
        primary = _primary_candidate(structured)
        sections.append(primary.text if primary is not None else "No resolution candidate is available from the supplied evidence.")

        sections.append("\n=== VALIDATION STEPS ===")
        if structured.validation_steps:
            for step in structured.validation_steps:
                sections.append(f"  - {step.instruction}")
        else:
            sections.append("No validation steps are available from the supplied evidence.")

        sections.append("\n=== APPLICABILITY (already determined) ===")
        sections.append(_applicability_line(structured))

        sections.append("\n=== CONFIDENCE (already calculated -- use exactly) ===")
        sections.append(f"Tier: {structured.confidence.value}")
        sections.append(
            f"Rationale: {structured.confidence_rationale}"
            if structured.confidence_rationale
            else "Rationale: No rationale was supplied."
        )

        user_prompt = "\n".join(sections)
        return _SYSTEM_PROMPT, user_prompt
