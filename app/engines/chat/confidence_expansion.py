"""Unsupported confidence-upgrade detection (Chat Assistant Phase 49).

Rule 4 (``app/engines/llm/prompt_builder.py``) already tells the model:
"Never use the words 'confirmed' or 'verified' (or equivalent) below the
Confirmed tier, even to negate it." Phase 48's real qwen2.5:3b
benchmark (140 real calls through the actual ``ChatOrchestrator`` path)
found this prompt-only instruction is not always followed: one call
answered *"Has this happened before? Confirmed."* for a fixture whose
authoritative confidence tier was LIKELY, not CONFIRMED -- reaching the
simulated user with no deterministic protection, since (unlike Rule 9
and Rule 11) no output-side gate existed for this rule.

This module is that gate -- the Rule 4 analogue of
``app.engines.chat.scope_expansion.contains_unsupported_scope_
expansion`` and ``app.engines.chat.troubleshooting_expansion.
contains_unsupported_troubleshooting_action``, built on the identical
idiom: closed phrase lists, whole-phrase matching, never an LLM, never
embeddings/similarity, conservative by construction.

Why this gate is shaped differently from its two predecessors: Rule 9
and Rule 11's phrase lists are pure blocklists -- every entry is
*always* unsupported in the gated context, because a legitimate
evidence-backed answer never has any reason to use that exact generic
phrasing. Rule 4 is different: the words "confirmed"/"verified" are
also used SAFELY and correctly in a hedge/negation ("this cannot be
confirmed from the available evidence", "not yet independently
verified" -- the latter being this codebase's own real, existing
``ChatOrchestrator._compose_answer`` LIKELY/POSSIBLE-tier wording). A
pure blocklist on the bare words would therefore reject safe, correct,
already-shipped deterministic phrasing. This gate instead: (1) never
fires at all when the authoritative ``structured.confidence`` already
IS ``ResolutionProvenance.CONFIRMED`` (the claim would be true), then
(2) strips out any recognized SAFE hedge/negation phrase from the text
before (3) checking whether a genuine, unsupported POSITIVE assertion
phrase remains. Every positive-assertion entry is itself a multi-word
phrase (or the bare word immediately followed by sentence-ending
punctuation) chosen so it can never appear as a substring of one of the
safe hedge phrases -- "is confirmed" never occurs inside "cannot be
confirmed", so no ordering ambiguity exists between the two lists.

Known, disclosed limitation (identical in kind to Rule 9/11's own): a
single answer that mixes a genuine hedge phrase not in the closed
SAFE list with a genuine separate overclaim elsewhere in the same text
could, in principle, have its overclaim missed if it also happens to
strip cleanly -- in practice this has not been observed in any real
call examined across Phases 44-48, and the same "closed list, not a
general parser" trade-off already governs every other deterministic
gate in this codebase.
"""

from __future__ import annotations

import re

from app.domain.provenance import ResolutionProvenance
from app.engines.shared.text_matching import phrase_present

SAFE_CONFIDENCE_HEDGE_PHRASES: list[str] = [
    "cannot be confirmed",
    "can't be confirmed",
    "could not be confirmed",
    "not confirmed",
    "not been confirmed",
    "not yet confirmed",
    "not independently confirmed",
    "not yet independently confirmed",
    "not been independently confirmed",
    "unable to confirm",
    "not able to confirm",
    "insufficient evidence to confirm",
    "not enough evidence to confirm",
    "don't have enough evidence to confirm",
    "do not have enough evidence to confirm",
    "doesn't have enough evidence to confirm",
    "does not have enough evidence to confirm",
    "cannot be verified",
    "can't be verified",
    "could not be verified",
    "not verified",
    "not been verified",
    "not yet verified",
    "not independently verified",
    "not yet independently verified",
    "not been independently verified",
    "unable to verify",
    "not able to verify",
    "insufficient evidence to verify",
    "not enough evidence to verify",
    "don't have enough evidence to verify",
    "do not have enough evidence to verify",
    "doesn't have enough evidence to verify",
    "does not have enough evidence to verify",
]
"""Closed list of safe negation/hedge phrases -- see module docstring.
The last three entries of each half ("not independently
confirmed"/"verified" and its "yet"/"been" variants) are this
codebase's OWN existing ``_compose_answer`` deterministic wording
(``app/engines/chat/orchestrator.py``'s LIKELY/POSSIBLE branches), not
invented here -- this gate must never flag a raw LLM answer that
happens to echo the same, already-approved safe phrasing."""

UNSUPPORTED_CONFIDENCE_PHRASES: list[str] = [
    "is confirmed",
    "has been confirmed",
    "confirmed based on",
    "confirmed from the",
    "confirmed by the",
    "this is verified",
    "is verified",
    "has been verified",
    "verified based on",
    "verified from the",
    "verified by the",
]
"""Closed list of genuine positive-assertion phrases -- each is a
multi-word phrase that never occurs as a substring of any
``SAFE_CONFIDENCE_HEDGE_PHRASES`` entry (e.g. "is confirmed" is not a
substring of "cannot be confirmed"), so stripping the safe phrases
first (see ``contains_unsupported_confidence_claim``) never
accidentally removes part of a genuine overclaim."""

_STANDALONE_CONFIRMED_RE = re.compile(r"(?<!\w)confirmed(?=[.,!?;:]|\s*$)", re.IGNORECASE)
_STANDALONE_VERIFIED_RE = re.compile(r"(?<!\w)verified(?=[.,!?;:]|\s*$)", re.IGNORECASE)
"""The exact real Phase 48 failure shape -- a bare "Confirmed." (or
"Verified.") used as a terse, standalone affirmative answer, with no
surrounding sentence for a multi-word phrase to match against. Matched
only when the word is followed by sentence-ending punctuation or the
end of the text (never mid-sentence, where it would already be covered
by one of the two phrase lists above, positive or safe)."""


def contains_unsupported_confidence_claim(text: str, confidence: ResolutionProvenance) -> bool:
    """True if ``text`` (an LLM-generated final answer) asserts
    "confirmed"/"verified" language that ``confidence`` -- the
    authoritative, already-computed ``StructuredResolution.confidence``
    tier, never re-derived from ``text`` itself -- does not support.

    Always ``False`` when ``confidence`` genuinely is
    ``ResolutionProvenance.CONFIRMED`` (the claim would be true).
    Otherwise: strips every recognized safe hedge/negation phrase from
    a working copy of the text, then reports ``True`` if either a
    genuine positive-assertion phrase or a standalone "Confirmed."/
    "Verified." remains.

    IMPORTANT (integration-order requirement, mirroring
    ``contains_unsupported_scope_expansion``/``contains_unsupported_
    troubleshooting_action`` exactly): callers must apply this ONLY to
    the raw text ``_generate_llm_answer`` returns, BEFORE any
    deterministic statement is appended -- see
    ``ChatOrchestrator._attempt_llm_answer``'s call site."""
    if confidence == ResolutionProvenance.CONFIRMED:
        return False

    stripped = text
    for hedge in SAFE_CONFIDENCE_HEDGE_PHRASES:
        stripped = re.sub(re.escape(hedge), " ", stripped, flags=re.IGNORECASE)

    if any(phrase_present(phrase, stripped) for phrase in UNSUPPORTED_CONFIDENCE_PHRASES):
        return True
    return bool(_STANDALONE_CONFIRMED_RE.search(stripped) or _STANDALONE_VERIFIED_RE.search(stripped))
