"""Query Understanding -- deterministic, zero-LLM parsing of free-form
user text (a chat question, a pasted ticket snippet) into the same
governed dimensions an Investigation already carries, so a future Chat
Assistant can hand a question to the *existing* applicability-aware
retrieval and Resolution Provenance machinery instead of a second,
independent one (2026-08-14, Phase 2 -- RESOLVEIQ_CHAT_AND_RESOLUTION_
ARCHITECTURE.md, "Phase 2 -- Query Understanding").

There is NO LLM anywhere in this module or its engine
(``app.engines.query_understanding.engine.QueryUnderstandingEngine``).
Every extraction is either a fixed, data-driven pattern table (intent)
or a real governed-entity text match reusing the exact same matching
primitives classification and technology inference already use (see
``app.engines.shared.governed_text_matching`` and
``app.engines.shared.text_matching.evidence_coverage_match_score``).

Confidence model -- deliberately only three states, per explicit
product decision (2026-08-14): "absence of evidence must remain
absence of evidence. Do not manufacture structured values simply to
make the object uniform." A slot with nothing extracted is ``None`` on
``ParsedQuery`` -- there is no ``ExtractedSlot`` instance for it at
all, and no fourth ``UNKNOWN`` confidence state exists to put on one.
See ``SlotConfidence`` for what each of the three real states means.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# Deliberate, narrow exception to this codebase's usual domain-layer/
# engine-layer boundary (no other app/domain module imports from
# app/engines) -- made explicitly per product decision (2026-08-14):
# "ParsedQuery.retrieval_context must use the EXISTING RetrievalContext
# -- do NOT create a second retrieval-context structure." RetrievalContext
# already lives in app.engines.knowledge.applicability (built once per
# Analyze call by RecommendationEngine._resolve_retrieval_context, reused
# by ApplicabilityRanker); duplicating its shape here just to keep a
# cleaner import direction would be exactly the "second, independent
# retrieval-context structure" that decision rules out. No circular
# import results: applicability.py does not import this module.
from app.engines.knowledge.applicability import RetrievalContext


class SlotConfidence(str, Enum):
    """How an ``ExtractedSlot``'s value was determined -- deliberately
    only three states (2026-08-14 product decision: "do not introduce
    any additional confidence states"). The fourth conceptual state,
    "nothing extracted," is represented by the slot itself being
    ``None`` on ``ParsedQuery``, never by a member here.

    - EXACT: the governed value's canonical name or a known alias was
      found as a full-phrase match in the text (the same "title match"
      strength classification.py's HIGH tier and
      ``_match_single_technology``'s full-score match already require).
    - PARTIAL: supported partial/family evidence where the existing
      matching rules for that dimension permit it (today: only
      Technology, via ``evidence_coverage_match_score``'s single-
      significant-word rule for a technology whose full name is one
      significant word, e.g. "RF Mesh"). Customer/Region/Product/
      Component never produce PARTIAL -- their existing matching rules
      (alias-or-nothing; full-name-or-nothing) have no partial tier to
      begin with, and Query Understanding does not invent one for them.
    - AMBIGUOUS: more than one real, distinct governed candidate
      matched with no existing rule to prefer one over the others (no
      hierarchy relationship to resolve the tie via
      ``most_specific_candidates``) -- a genuine "can't safely pick,"
      not a weaker match. ``ambiguous_candidates`` names every tied
      candidate so the caller can see why."""

    EXACT = "exact"
    PARTIAL = "partial"
    AMBIGUOUS = "ambiguous"


class ExtractedSlot(BaseModel):
    """One governed value pulled out of free text, with enough
    information for the caller to see why it was extracted and how
    much to trust it -- never just a bare string. Only ever constructed
    when *something* was found; a dimension with no evidence at all is
    represented by ``None`` on ``ParsedQuery``, not by an
    ``ExtractedSlot`` in some empty/unknown state."""

    value_id: str | None = None
    """The real governed row id, when a single definite value was
    identified (EXACT or PARTIAL). ``None`` for AMBIGUOUS -- there is
    no single winner to attach an id to."""
    value_name: str | None = None
    """The candidate's real canonical name (never the raw alias/phrase
    that matched, when they differ) for EXACT/PARTIAL. ``None`` for
    AMBIGUOUS."""
    confidence: SlotConfidence
    evidence_snippet: str
    """The real text this slot was extracted from -- the matched
    phrase itself, or (for a body-style match) a short window of
    surrounding context. Never fabricated."""
    ambiguous_candidates: list[str] = Field(default_factory=list)
    """Canonical names of every real, distinct candidate that tied,
    populated only when ``confidence == AMBIGUOUS``. Empty for
    EXACT/PARTIAL."""


class QueryIntent(str, Enum):
    """A fixed, closed set of intents a free-form question can be
    classified into -- deterministic pattern matching only (see
    ``QueryUnderstandingEngine``'s intent classifier), never an LLM.
    ``UNKNOWN`` is the conservative default: comparable evidence for
    more than one intent, or no evidence for any, yields UNKNOWN rather
    than a guess."""

    TROUBLESHOOTING = "troubleshooting"
    HISTORICAL_LOOKUP = "historical_lookup"
    KNOWN_BUG_LOOKUP = "known_bug_lookup"
    LOG_GUIDANCE = "log_guidance"
    SQL_GUIDANCE = "sql_guidance"
    DOCUMENTATION_LOOKUP = "documentation_lookup"
    RELEASE_NOTE_LOOKUP = "release_note_lookup"
    UNKNOWN = "unknown"


class ParsedQuery(BaseModel):
    """The structured result of parsing one piece of free text --
    everything a future Chat layer needs to hand off to the existing,
    unchanged retrieval/recommendation/provenance machinery. Query
    Understanding stops here: it produces structured intent and slots,
    it never itself decides a resolution (that stays
    ``RecommendationEngine``/``StructuredResolutionEngine``'s job,
    downstream and untouched by this module)."""

    raw_text: str
    intent: QueryIntent
    intent_confidence: float
    """0.0-1.0, deterministic -- see
    ``QueryUnderstandingEngine``'s intent classifier for exactly how
    it's computed. 0.0 whenever ``intent == UNKNOWN``."""

    customer: ExtractedSlot | None = None
    region: ExtractedSlot | None = None
    technology: ExtractedSlot | None = None
    product: ExtractedSlot | None = None
    component: ExtractedSlot | None = None
    version: ExtractedSlot | None = None
    exception_type: ExtractedSlot | None = None

    ticket_references: list[str] = Field(default_factory=list)
    """Real ticket/CRM identifiers found in the text (e.g. "CS0122697",
    "INC0045821") via the same regex External Knowledge already uses to
    correlate a local record to a live TFS case -- see
    ``app.engines.external_knowledge.service._TICKET_NUMBER_RE``. Not a
    governed entity, so no ``ExtractedSlot``/confidence wrapper -- a
    ticket number either matches the pattern or it doesn't."""

    retrieval_context: RetrievalContext
    """The existing, unchanged ``RetrievalContext`` shape (customer/
    region/technology only -- see that class's own docstring), reused
    verbatim rather than duplicated. Product/Component/Version have no
    field on ``RetrievalContext`` today (inspection confirmed: it was
    never expanded past what ``ApplicabilityRanker`` actually
    consumes), so those three extracted slots live only on this model,
    not on ``retrieval_context`` -- a real, documented limitation, not
    a silent gap: a caller that also needs Product/Component/Version
    for applicability must read ``self.product``/``self.component``/
    ``self.version`` directly. Expanding ``RetrievalContext`` itself
    was explicitly out of scope for this phase absent proof it's
    needed, and no such need was identified."""
