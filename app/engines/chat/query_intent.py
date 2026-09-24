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
    contains_l2_guidance_question,
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
    L2_GUIDANCE = "l2_guidance"
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

_ANAPHORA_RE = re.compile(r"\b(it|this|that|them|these|those)\b", re.IGNORECASE)
"""Closed, narrow set of bare anaphoric references -- Step 17's own
signal that a follow-up question is standing in for a subject named in
an earlier turn, rather than a genuinely new one. Used only by
``build_query_context``'s ``context_text`` fallback/enrichment; never
consulted anywhere the question already fully identifies its own
subject without a pronoun."""


@dataclass
class QueryContext:
    """The deterministic "what is this question actually about"
    object Step 3/5 asks for -- every field is either directly
    extracted from the question's own text via a closed pattern, a
    real EXACT-confidence match against this codebase's own governed
    Customer/Region/Technology/Product/Component/Version tables (reused
    from ``QueryUnderstandingEngine``/``ParsedQuery`` -- never a second,
    independently-maintained entity matcher), or carried forward from
    already-established conversation context -- never invented, never
    guessed, per this phase's own explicit instruction."""

    intent: AnswerIntent
    subject: str | None
    """The question's own real content words, space-joined (reuses
    ``extract_concept_words`` -- never a new extraction rule). Falls
    back to the accumulated conversation's own context text when the
    CURRENT turn's question has no extractable subject words of its
    own (e.g. "Where do I configure it?") -- see ``build_query_
    context``'s ``context_text`` parameter -- so a follow-up question
    inherits the real prior subject instead of losing it."""
    state: str | None = None
    """A recognized device/meter state (e.g. "Discovered") when the
    question's own text names one from the closed ``_KNOWN_STATES``
    list -- ``None`` otherwise, never inferred."""
    is_definitional: bool = False
    requested_information: str = ""
    product: str | None = None
    technology: str | None = None
    version: str | None = None
    component: str | None = None
    customer: str | None = None
    region: str | None = None

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
            "product": self.product,
            "technology": self.technology,
            "version": self.version,
            "component": self.component,
            "customer": self.customer,
            "region": self.region,
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

    # Final Support-Quality Pass, §2 -- L2_GUIDANCE checked before
    # CONFIGURATION/troubleshooting: "What should L2 check?" is a
    # request for a support-team recommendation, never a configuration
    # procedure or a bare troubleshooting request. Deliberately its own
    # closed phrase list (contains_l2_guidance_question, anchored on
    # "L2"/"the support team"), never a widening of the first-person
    # "what should I check" family (troubleshooting_question.py), which
    # must keep resolving to plain TROUBLESHOOTING -- the two never
    # collide since they're anchored on different fixed subjects. Not
    # gated behind has_log_evidence: unlike L2_TASK_NOTES/L3_ESCALATION
    # (which summarize log evidence specifically), an L2 guidance
    # question can be answered from historical/known-bug/documentation
    # evidence alone, with or without an uploaded log.
    if contains_l2_guidance_question(question):
        return AnswerIntent.L2_GUIDANCE

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

    if (
        "has this happened before" in lowered
        or "seen this before" in lowered
        or "seen before" in lowered
        or "in similar cases" in lowered
    ):
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


def _exact_slot_name(slot: object | None) -> str | None:
    """Reuses ``ParsedQuery``'s own ``ExtractedSlot``/``SlotConfidence``
    contract: only an EXACT match populates ``QueryContext`` -- an
    AMBIGUOUS or absent slot stays ``None`` here too, exactly mirroring
    ``QueryUnderstandingEngine._build_retrieval_context``'s own
    "existing-context-wins, EXACT-only" rule (see that method's
    docstring) rather than a second, independently-invented threshold."""
    if slot is None:
        return None
    from app.domain.query_understanding import SlotConfidence

    if getattr(slot, "confidence", None) == SlotConfidence.EXACT:
        return getattr(slot, "value_name", None)
    return None


def build_query_context(
    question: str,
    *,
    has_log_evidence: bool = False,
    parsed_query: object | None = None,
    context_text: str = "",
) -> QueryContext:
    """The single entry point ``ChatOrchestrator`` calls: classifies
    intent and extracts what can be established, in one deterministic
    pass, never a second, independently-computed classification.

    ``parsed_query`` (Step 5) -- when given, is this codebase's own
    real ``app.domain.query_understanding.ParsedQuery`` (already
    computed once per turn by ``QueryUnderstandingEngine``, never
    recomputed here) -- its EXACT-confidence Customer/Region/
    Technology/Product/Component/Version slots populate the matching
    ``QueryContext`` fields; an AMBIGUOUS or absent slot leaves the
    field ``None``, never a guess among tied candidates.

    ``context_text`` (Step 17) -- the conversation's own accumulated
    raw text (``InvestigationSession.context_text``, already computed,
    never a new field) -- used as a subject fallback/enrichment in two
    cases: the current question has NO extractable concept words at
    all, or it contains a bare anaphoric reference ("it"/"this"/
    "that"/...) with nothing else specific enough to identify a
    subject on its own (e.g. "How do I configure it?" -> its own real
    concept words are just ``["configure"]`` -- real, but not a
    SUBJECT; "it" is exactly the word standing in for one). In both
    cases the prior turn's real concept words are merged in (prepended,
    de-duplicated) rather than replacing whatever the current question
    does contribute -- so "How do I configure it?" after "What is
    process settings in CC?" carries the real subject forward as
    "process settings configure", not just "configure". Never
    consulted when ``context_text`` is empty (a fresh, first-turn
    question is never altered by this)."""
    intent = classify_intent(question, has_log_evidence=has_log_evidence)
    concept_words = extract_concept_words(question)
    if context_text and (not concept_words or _ANAPHORA_RE.search(question)):
        context_words = extract_concept_words(context_text)
        merged = list(context_words)
        for word in concept_words:
            if word not in merged:
                merged.append(word)
        concept_words = merged
    subject = " ".join(concept_words) if concept_words else None
    state = extract_state(question) or (extract_state(context_text) if context_text else None)
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

    product = technology = version = component = customer = region = None
    if parsed_query is not None:
        product = _exact_slot_name(getattr(parsed_query, "product", None))
        technology_slot = getattr(parsed_query, "technology", None)
        # Technology's own EXACT-or-PARTIAL rule (see ParsedQuery/
        # QueryUnderstandingEngine._match_technology's docstring) is
        # reused verbatim rather than re-deriving a new threshold.
        if technology_slot is not None:
            from app.domain.query_understanding import SlotConfidence as _SC

            if getattr(technology_slot, "confidence", None) in (_SC.EXACT, _SC.PARTIAL):
                technology = getattr(technology_slot, "value_name", None)
        version = _exact_slot_name(getattr(parsed_query, "version", None))
        component = _exact_slot_name(getattr(parsed_query, "component", None))
        customer = _exact_slot_name(getattr(parsed_query, "customer", None))
        region = _exact_slot_name(getattr(parsed_query, "region", None))

    return QueryContext(
        intent=intent,
        subject=subject,
        state=state,
        is_definitional=is_def,
        requested_information=requested_information,
        product=product,
        technology=technology,
        version=version,
        component=component,
        customer=customer,
        region=region,
    )
