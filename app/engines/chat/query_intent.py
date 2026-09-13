"""Deterministic query-intent classification and lightweight entity/
state extraction (Knowledge Answering & Evidence Synthesis phase).

Real problem this addresses: retrieval alone cannot tell "What is
AxeI meter?" (a request for a product/concept DEFINITION) apart from
"AxeI meters are stuck in discovered, how do I make it normal?" (a
TROUBLESHOOTING request) -- both can retrieve the exact same
underlying evidence (a historical case whose title happens to contain
"AxeI meter"), and letting retrieval-similarity alone decide what to
say produces the exact reported failure: a historical incident
presented as if it were a product definition. This module is the
deterministic layer that decides WHAT KIND of question this is,
BEFORE any evidence is selected or rendered -- retrieval finds
evidence, this decides what the evidence needs to answer.

Same "closed list/regex, never an LLM, conservative by construction"
idiom every other classifier in this codebase already uses --
deliberately does NOT replace ``contains_knowledge_question``/
``contains_troubleshooting_synthesis_question``/``contains_log_
analysis_question``/``contains_l2_task_note_question``/``contains_l3_
escalation_question`` (all of them already real, already tested, and
already correctly gate their own composers) -- ``classify_intent``
below is a thin layer ON TOP of those existing functions, mapping
their existing yes/no decisions onto a finer-grained, user-facing
taxonomy for diagnostics (Step 18's debug representation) and for the
one real behavioral distinction this phase needs (ENTITY_DEFINITION/
PRODUCT_EXPLANATION/CONCEPT_EXPLANATION vs. everything else, used by
``ChatOrchestrator._compose_knowledge_synthesis`` to decide when a
historical-case-as-primary answer needs the "not a definition" caveat).
It never makes its own routing decision that the existing composers
don't already make -- see ``AnswerIntent``'s own docstring for exactly
which of these 12 values are "real" (behavior-affecting) vs. purely
descriptive/diagnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from app.engines.chat.knowledge_question import (
    contains_knowledge_question,
    extract_concept_words,
    is_definitional_question,
)
from app.engines.chat.log_question import (
    contains_l2_task_note_question,
    contains_l3_escalation_question,
    contains_log_analysis_question,
    contains_log_comparison_question,
)
from app.engines.chat.troubleshooting_question import contains_troubleshooting_question
from app.engines.chat.troubleshooting_synthesis_question import contains_troubleshooting_synthesis_question


class AnswerIntent(str, Enum):
    """The 12 intents this phase's own brief names, plus UNKNOWN.
    Every value is deterministically derived from one of this
    codebase's existing closed-phrase classifiers, or (ENTITY_
    DEFINITION/PRODUCT_EXPLANATION/CONCEPT_EXPLANATION/CONFIGURATION/
    HOW_TO/ROOT_CAUSE) a further, still-closed-list split of
    ``contains_knowledge_question``'s/``contains_troubleshooting_
    synthesis_question``'s own "yes" case. LOG_ANALYSIS/L2_TASK_NOTES/
    L3_ESCALATION/COMPARISON/TROUBLESHOOTING/HISTORICAL_LOOKUP are the
    ones that actually change which composer runs (unchanged from
    before this phase -- this enum documents that routing, it doesn't
    alter it). ENTITY_DEFINITION/PRODUCT_EXPLANATION/CONCEPT_
    EXPLANATION/CONFIGURATION/HOW_TO/ROOT_CAUSE all still route to the
    SAME knowledge-synthesis or troubleshooting-synthesis composer as
    their parent bucket -- the finer label is diagnostic (Step 18) and,
    for the DEFINITION-shaped three, drives the "historical case is
    not a definition" caveat (see ``is_definitional_question``) -- it
    is not a claim that six separate answer-generation code paths
    exist for six separate labels, which would be new complexity this
    phase's own instruction ("do not invent metadata when it cannot be
    established") argues directly against."""

    ENTITY_DEFINITION = "entity_definition"
    PRODUCT_EXPLANATION = "product_explanation"
    CONCEPT_EXPLANATION = "concept_explanation"
    CONFIGURATION = "configuration"
    HOW_TO = "how_to"
    TROUBLESHOOTING = "troubleshooting"
    ROOT_CAUSE = "root_cause"
    HISTORICAL_LOOKUP = "historical_lookup"
    COMPARISON = "comparison"
    LOG_ANALYSIS = "log_analysis"
    L2_TASK_NOTES = "l2_task_notes"
    L3_ESCALATION = "l3_escalation"
    FOLLOW_UP = "follow_up"
    UNKNOWN = "unknown"


_CONFIGURATION_PHRASES: tuple[str, ...] = (
    "how do i configure",
    "how to configure",
    "how do i set up",
    "how do i setup",
    "how to set up",
    "how to setup",
    "where do i configure",
    "where do i set",
    "where is this configured",
    "where is it configured",
)
_HOW_TO_RE = re.compile(r"\bhow do i\b|\bhow to\b|\bhow can i\b", re.IGNORECASE)
_ROOT_CAUSE_PHRASES: tuple[str, ...] = ("root cause", "what caused")
_WHY_DID_X_FAIL_RE = re.compile(
    r"\bwhy (?:did|does|is|isn'?t) (?:this|it|that|the)\b.{0,40}?\b(?:fail|failing|failed|not working)\b",
    re.IGNORECASE,
)
"""Variable-middle "why did this/it/that <noun> fail" construct (e.g.
"Why did this meter fail?") -- ``troubleshooting_synthesis_question``'s
own closed phrase list only covers the bare "why did this/it fail"
form with nothing inserted between "this"/"it" and "fail"; a real
noun in between (a genuinely common, ordinary phrasing) matched
neither that list nor this module's own ``_ROOT_CAUSE_PHRASES``. Still
a closed-form regex anchored on the same fixed "why did"/"fail"
scaffolding, never a free-form heuristic."""
_COMPARISON_PHRASES: tuple[str, ...] = (
    "difference between", "compare", "versus", " vs ", "which is better", "what's the difference",
)

_KNOWN_STATES: tuple[str, ...] = (
    "discovered", "commissioned", "registered", "provisioning", "installed", "removed", "disabled", "in process",
    "inprocess", "pending", "offline", "unreachable",
)
_STATE_RE = re.compile(r"\b(" + "|".join(re.escape(s) for s in _KNOWN_STATES) + r")\b", re.IGNORECASE)


@dataclass
class QueryContext:
    """The deterministic "what is this question actually about"
    object Step 3 asks for -- every field is either directly extracted
    from the question's own text via a closed pattern, or ``None``
    when it cannot be established (never guessed/invented, per this
    phase's own explicit instruction)."""

    intent: AnswerIntent
    subject: str | None
    """The question's own real content words, space-joined (reuses
    ``extract_concept_words`` -- never a new extraction rule)."""
    state: str | None = None
    """A recognized device/meter state (e.g. "Discovered") when the
    question's own text names one from the closed ``_KNOWN_STATES``
    list -- ``None`` otherwise, never inferred."""
    is_definitional: bool = False
    requested_information: str = ""

    def as_debug_dict(self) -> dict:
        """Step 18's own debug representation -- plain data, never
        rendered by the UI by default (see ``ChatResponse.debug``'s own
        docstring)."""
        return {
            "intent": self.intent.value,
            "subject": self.subject,
            "state": self.state,
            "is_definitional": self.is_definitional,
            "requested_information": self.requested_information,
        }


def classify_intent(
    question: str,
    *,
    has_log_evidence: bool = False,
) -> AnswerIntent:
    """Deterministic, closed-list intent classification -- see module
    docstring for what each value means and which ones actually change
    routing (unchanged from before this phase) versus which are purely
    descriptive. Order matters: more specific intents (log/L2/L3) are
    checked before the general knowledge/troubleshooting buckets,
    mirroring ``ChatOrchestrator._compose_answer``'s own existing
    check order exactly (never a second, potentially-diverging
    ordering)."""
    if has_log_evidence:
        if contains_l2_task_note_question(question):
            return AnswerIntent.L2_TASK_NOTES
        if contains_l3_escalation_question(question):
            return AnswerIntent.L3_ESCALATION
        if contains_log_comparison_question(question) or contains_log_analysis_question(question):
            return AnswerIntent.LOG_ANALYSIS

    lowered = question.lower()

    # Configuration/how-to checked BEFORE the troubleshooting/knowledge
    # buckets: "how do I configure X" is a request for a procedure, not
    # a request to diagnose a failure or explain a concept, and must
    # never be shadowed by either.
    if any(p in lowered for p in _CONFIGURATION_PHRASES):
        return AnswerIntent.CONFIGURATION

    if (
        contains_troubleshooting_synthesis_question(question)
        or contains_troubleshooting_question(question)
        or _WHY_DID_X_FAIL_RE.search(question)
    ):
        if any(p in lowered for p in _ROOT_CAUSE_PHRASES) or _WHY_DID_X_FAIL_RE.search(question):
            return AnswerIntent.ROOT_CAUSE
        return AnswerIntent.TROUBLESHOOTING

    if any(p in lowered for p in _COMPARISON_PHRASES):
        return AnswerIntent.COMPARISON

    # Everything below is gated behind contains_knowledge_question,
    # which is where the actual safety guard lives (Rule 3/4's
    # tier-preservation contract: "what is the root cause"/"the
    # resolution"/"the confidence"/etc. must NEVER be classified as a
    # knowledge question at all -- see _RESERVED_INVESTIGATION_LEAD_
    # INS in knowledge_question.py). is_definitional_question re-checks
    # the SAME underlying regexes that function already validated, so
    # this can never diverge from that guard, but must never be
    # consulted on its own before this check.
    if not contains_knowledge_question(question):
        if _HOW_TO_RE.search(question):
            return AnswerIntent.HOW_TO
        return AnswerIntent.UNKNOWN

    if "has this happened before" in lowered or "seen this before" in lowered or "seen before" in lowered:
        return AnswerIntent.HISTORICAL_LOOKUP

    if is_definitional_question(question):
        if "how does" in lowered and "work" in lowered:
            return AnswerIntent.PRODUCT_EXPLANATION
        # Singular "what is/what's X" names a specific entity/product;
        # plural "what are X" names a category/concept (e.g. "What ARE
        # process settings in CC?") -- a real, closed distinction, not
        # a guess: English singular/plural "to be" already encodes it.
        if re.search(r"\bwhat are\b", question, re.IGNORECASE):
            return AnswerIntent.CONCEPT_EXPLANATION
        return AnswerIntent.ENTITY_DEFINITION

    if _HOW_TO_RE.search(question):
        return AnswerIntent.HOW_TO

    return AnswerIntent.CONCEPT_EXPLANATION


def extract_state(question: str) -> str | None:
    """A recognized device/meter state named in the question's own
    text (e.g. "Discovered") -- closed list (``_KNOWN_STATES``), never
    inferred from context. Returns the recognized word Title-Cased for
    display (e.g. "discovered" -> "Discovered")."""
    match = _STATE_RE.search(question)
    return match.group(1).title() if match else None


def build_query_context(question: str, *, has_log_evidence: bool = False) -> QueryContext:
    """The single entry point ``ChatOrchestrator`` calls: classifies
    intent and extracts what can be established, in one deterministic
    pass, never a second, independently-computed classification."""
    intent = classify_intent(question, has_log_evidence=has_log_evidence)
    concept_words = extract_concept_words(question)
    subject = " ".join(concept_words) if concept_words else None
    state = extract_state(question)
    is_def = intent in (AnswerIntent.ENTITY_DEFINITION, AnswerIntent.PRODUCT_EXPLANATION)
    requested_information = {
        AnswerIntent.ENTITY_DEFINITION: "definition/purpose",
        AnswerIntent.PRODUCT_EXPLANATION: "how it works",
        AnswerIntent.CONCEPT_EXPLANATION: "explanation",
        AnswerIntent.CONFIGURATION: "configuration steps",
        AnswerIntent.HOW_TO: "procedure",
        AnswerIntent.TROUBLESHOOTING: "recovery/troubleshooting",
        AnswerIntent.ROOT_CAUSE: "root cause",
        AnswerIntent.HISTORICAL_LOOKUP: "prior occurrence",
        AnswerIntent.COMPARISON: "comparison",
        AnswerIntent.LOG_ANALYSIS: "log analysis",
        AnswerIntent.L2_TASK_NOTES: "L2 task notes",
        AnswerIntent.L3_ESCALATION: "L3 escalation summary",
        AnswerIntent.FOLLOW_UP: "follow-up on prior topic",
        AnswerIntent.UNKNOWN: "",
    }.get(intent, "")
    return QueryContext(
        intent=intent, subject=subject, state=state, is_definitional=is_def, requested_information=requested_information
    )
