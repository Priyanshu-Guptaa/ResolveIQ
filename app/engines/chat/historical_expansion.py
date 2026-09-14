"""Unsupported historical-recommendation-to-current-resolution upgrade
detection (Evidence-Centered Knowledge Retrieval & Synthesis phase,
§16 -- "reject if the LLM... converts a historical recommendation
into a current resolution").

Same idiom as its three siblings (``app.engines.chat.confidence_
expansion``, ``scope_expansion``, ``troubleshooting_expansion``):
closed phrase list, whole-phrase matching, never an LLM, never
embeddings/similarity, conservative by construction.

Why this reuses ``structured.confidence`` exactly like ``confidence_
expansion`` does, rather than inspecting the EvidenceBundle's claims
directly: a POSSIBLE/UNKNOWN-tier ``StructuredResolution`` is, by this
codebase's own existing, hardened confidence-tier machinery
(``RecommendationEngine._resolve_provenance_tier``), never backed by a
CURRENT, confirmed resolution -- at those two tiers, any evidence this
turn cites for "the fix" is, by construction, at best a historical
case's own recorded action (a HISTORICAL_RECOMMENDATION-authority
claim, never promoted higher -- see ``app.domain.evidence_bundle.
SourceAuthority``). So a genuine "the solution is X"/"this resolves
it" assertion at POSSIBLE/UNKNOWN tier is unsupported REGARDLESS of
which specific historical case it paraphrases, the same way ``contains_
unsupported_confidence_claim`` needs no per-claim bookkeeping to know
"confirmed" is unsupported below the Confirmed tier. At CONFIRMED/
LIKELY tier, a real current resolution genuinely exists, so this gate
never fires there -- identical structure to its sibling gate."""

from __future__ import annotations

from app.domain.provenance import ResolutionProvenance
from app.engines.shared.text_matching import phrase_present

UNSUPPORTED_RESOLUTION_PHRASES: list[str] = [
    "the solution is",
    "the fix is",
    "this resolves the issue",
    "this resolves it",
    "is now resolved",
    "has been resolved",
    "this will fix",
    "this fixes the issue",
    "this fixes it",
    "the resolution is to",
]
"""Closed list of genuine positive-resolution-assertion phrases. Each
is a multi-word phrase describing a SETTLED fix -- never a hedge (a
POSSIBLE/UNKNOWN-tier deterministic answer's own real wording, e.g. "a
possible explanation is"/"not yet independently verified", never
contains any of these), so no stripping step is needed before checking
for them (unlike ``confidence_expansion``'s hedge/positive split, whose
words ("confirmed"/"verified") ARE also used safely in negation)."""


def contains_unsupported_resolution_claim(text: str, confidence: ResolutionProvenance) -> bool:
    """True if ``text`` (an LLM-generated final answer) asserts settled-
    fix language that ``confidence`` -- the authoritative, already-
    computed ``StructuredResolution.confidence`` tier, never re-derived
    from ``text`` itself -- does not support. Always ``False`` when
    ``confidence`` is CONFIRMED or LIKELY (a real current resolution
    genuinely exists at both tiers)."""
    if confidence in (ResolutionProvenance.CONFIRMED, ResolutionProvenance.LIKELY):
        return False
    return any(phrase_present(phrase, text) for phrase in UNSUPPORTED_RESOLUTION_PHRASES)
