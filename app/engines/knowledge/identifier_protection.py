"""Exact-Identifier Protection (Chat Assistant Phase 2D).

Phase 2C/2D's read-only investigations found the RRF/score-fusion
layer can mis-rank a candidate that contains the EXACT identifier the
user searched for -- a ticket ID, a version string, or a CamelCase
technical term -- below a candidate with weaker, more generic evidence.
Every real regression traced (WiSun, CSTASK0078039, CSTASK0064295) was
a candidate the fused pool already contained at a good depth, just
mis-scored, not a recall/truncation problem -- see PHASE 2D — EXACT
IDENTIFIER PROTECTION DESIGN, Sections 2 and 6.

This module is a pure, standalone function operating only on
``KnowledgeMatch`` objects and the raw query text -- no dependency on
``RecommendationEngine``, ``KnowledgeEngine``, ``ApplicabilityRanker``,
``QueryUnderstandingEngine``, ChromaDB, BM25, RRF, the database, or
configuration. Called by ``HybridKnowledgeStore.query()`` AFTER fusion
(either mode) and BEFORE the results reach ``ApplicabilityRanker`` --
see that design document's Section 23 architecture diagram.

=== WHY THIS IS NOT CUSTOMER-AWARE RANKING ===
Phase 2D Step 1 found the TEPCO-style customer-context regression is
ALREADY solved by the existing, unmodified ``ApplicabilityRanker`` once
a real ``customer_id`` reaches it (confirmed via the real relationship
graph and the real Chat wiring -- no gap existed there). This module
never looks at customer/region/technology at all; it only recognizes
three narrow, generalized identifier SHAPES (ticket, version,
CamelCase) and rewards an EXACT, whole-token textual match -- a
completely orthogonal signal to governed relationship data.

=== KNOWN LIMITATION UNDER hybrid_fusion_mode="rrf" (Phase 2E) ===
Protection is applied to the FINAL fused candidates, after retrieval
fusion -- it never re-runs retrieval, never re-ranks by itself beyond
the bounded boost below, and never inspects anything but the already-
fused ``KnowledgeMatch`` list. Under ``hybrid_fusion_mode="score"``
(Phase 2C's recommended calibration) this reliably reinforces an exact
identifier, because a lexical-only candidate's ``.score`` there is a
real, meaningful blended value. Under ``hybrid_fusion_mode="rrf"``,
however, a lexical-only candidate's ``.score`` is intentionally
materialized as ``0.0`` by ``HybridKnowledgeStore``'s own CRITICAL
SCORE CONTRACT (see that module) BEFORE this function ever runs --
Phase 2E's real-corpus investigation confirmed a fixed ``+0.30`` boost
on top of ``0.0`` is frequently still well below a genuinely unrelated
but vector-similar competitor's real cosine score (real examples
measured at ~0.6-0.73). This is a known, disclosed limitation, not an
implementation failure -- see PHASE 2E — RRF / IDENTIFIER PROTECTION
COMPATIBILITY REVIEW. ``IDENTIFIER_PROTECTION_BOOST`` is deliberately
NOT raised to compensate (that was explicitly evaluated and rejected --
it would amount to overriding the score contract through an
oversized boost rather than respecting it). ``hybrid_fusion_mode="score"``
is the recommended pairing whenever ``identifier_protection_enabled=True``;
``app.config.Settings`` logs a warning if the two are combined.

=== WHY THE REGEXES BELOW ARE DUPLICATED, NOT IMPORTED ===
The ticket pattern deliberately generalizes beyond
``app.engines.external_knowledge.service._TICKET_NUMBER_RE`` (which has
a real, confirmed gap: it does not match ``CTASK``-prefixed tickets --
see the Phase 2D design report, Section 5) rather than reusing that
regex as-is. The version and CamelCase patterns intentionally MIRROR
Phase 2A's own ``_VERSION_RE``/``_CAMEL_BOUNDARY_RE``
(``app.engines.knowledge.lexical_store``) -- same shape, same bounds --
but are redefined here as independent constants rather than imported,
so this query-time protection module is never silently coupled to a
future tokenizer-internal change never intended to affect it. This is
the same "duplicate, don't couple" precedent Phase 2A itself set with
``BM25_STOPWORDS`` (independently authored, not
``RecommendationEngine._ENGLISH_STOPWORDS``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.domain.recommendation import KnowledgeMatch

IDENTIFIER_PROTECTION_BOOST = 0.30
"""Fixed, bounded additive boost -- not a configurable weight (Phase 2D
design Section 8/16: a fixed, evidence-derived rule, not a new tuning
knob). Deliberately stronger than any single ``ApplicabilityRanker``
adjustment (same customer +0.20, different customer -0.25, technology
exact +0.20/family +0.08) -- an exact ticket/version/CamelCase match is
rarer, more certain evidence than governed relationship context, and
Phase 2C measured real margins as thin as ~0.005 between a correct
exact-identifier record and its competitor, too fragile to trust
without real headroom. Chosen so a genuine match survives even a
simultaneous worst-case "different customer" applicability penalty
(net +0.05) -- applicability still gets the final, composable word;
this boost never forces rank 1 by itself. Applied additively, once per
candidate regardless of how many identifiers it matches (no stacking),
then the final score is clamped to [0, 1]."""

# --- Ticket-shaped identifiers ----------------------------------------------

_TICKET_RE = re.compile(r"\b[A-Z]{2,6}\d{5,}\b")
"""A 2-6 uppercase-letter prefix immediately followed by 5-or-more
digits, word-bounded. Deliberately generalized on SHAPE rather than an
enumerated prefix list (CSTASK/CS/TASK/INC/...) -- covers every known
real prefix, including CTASK (the gap left by the older
``_TICKET_NUMBER_RE``), as a natural consequence of the shape, not a
hardcoded fix, and works for any future prefix convention without a
code change. Verified against real corpus noise during design: does
NOT match version strings like ``SS8.6.1.463`` (only 1 digit before
the first dot -- needs 5+ consecutive), and does NOT match long system
codes like ``USIADOCPA124238`` (9-letter prefix, outside the 2-6
bound)."""

# --- Version-shaped identifiers ----------------------------------------------

_VERSION_RE = re.compile(r"\b[a-z]{0,4}\d{1,4}(?:\.\d{1,4}){1,3}\b")
"""Intentionally mirrors Phase 2A's ``lexical_store._VERSION_RE`` exactly
(same bounds, same rationale) but is a SEPARATE, duplicated constant --
see module docstring's "WHY THE REGEXES BELOW ARE DUPLICATED" section.
Applied against lowercased text, same as the tokenizer does."""

# --- CamelCase / technical identifiers ---------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
"""Whole "word" spans in the original, case-preserved text -- mirrors
Phase 2A's own ``_WORD_RE``, duplicated for the same coupling-avoidance
reason."""
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
"""Zero-width split point at a lowercase/digit -> uppercase transition
-- mirrors Phase 2A's ``_CAMEL_BOUNDARY_RE`` exactly. A word is
CamelCase-eligible iff this pattern finds at least one real transition
inside it. This is what correctly, automatically excludes
acronym-led/all-caps forms (``APIResponse``, ``HTTPClient``,
``SQLServer``, ``AXEi`` -- no lowercase-then-uppercase transition
exists in any of them) while including genuine compounds
(``CommandTimeout``, ``KafkaConsumerGroup``, ``WiSun``) -- the exact
same conservative rule Phase 2A already validated against real corpus
evidence; no new heuristic is invented here."""


def _detect_identifiers(text: str) -> set[str]:
    """Returns the set of distinct, lowercased identifier tokens found
    in ``text``, across all three categories. The CamelCase category
    uses the ORIGINAL, UNSPLIT word (lowercased) -- never the split
    parts -- this is the single most important rule in this module:
    matching on ``"wisun"`` rather than ``"wi"``/``"sun"`` is exactly
    what avoids repeating Phase 2A's own documented short-token
    dilution problem for a query-time protection signal."""
    if not text or not text.strip():
        return set()

    identifiers: set[str] = set()

    identifiers.update(match.upper() for match in _TICKET_RE.findall(text))
    identifiers.update(match for match in _VERSION_RE.findall(text.lower()))

    for word_match in _WORD_RE.finditer(text):
        word = word_match.group(0)
        if _CAMEL_BOUNDARY_RE.search(word):
            identifiers.add(word.lower())

    return identifiers


def _candidate_identifiers(match: "KnowledgeMatch") -> set[str]:
    """Same three-category extraction, applied to a candidate's own
    searchable identity: title + snippet + metadata tags only --
    deliberately not the whole metadata dict (avoids false positives
    from unrelated internal metadata values) -- exactly the fields real
    ticket/version/term strings were confirmed to live in during the
    Phase 2C/2D investigations."""
    tags = match.metadata.get("tags", "") if match.metadata else ""
    combined = f"{match.title}\n{match.snippet}\n{tags}"
    return _detect_identifiers(combined)


def protect_exact_identifiers(query: str, matches: list["KnowledgeMatch"]) -> list["KnowledgeMatch"]:
    """Applies Phase 2D's bounded exact-identifier boost to ``matches``
    (an already-fused candidate list, in either fusion mode) based on
    ``query``.

    For each candidate: if its own identifier-token set (from
    title/snippet/tags) shares at least one EXACT, case-insensitive,
    whole-token match with the identifiers detected in ``query``, its
    ``.score`` is increased by ``IDENTIFIER_PROTECTION_BOOST`` (once,
    regardless of how many identifiers matched -- no stacking) and
    clamped to ``[0, 1]``. A candidate with no match is returned
    completely unchanged (same object, not even copied) -- this
    function is a strict no-op for the overwhelming majority of
    real-world natural-language queries (Phase 2D design Section 15).

    Matching is EXACT WHOLE-TOKEN equality, never substring --
    ``"CSTASK0078039"`` does not match ``"CSTASK00780390"``, ``"1.8.2"``
    does not match ``"11.8.20"``, and a candidate containing only the
    split words ``"Wi"``/``"Sun"`` (never the compound ``"WiSun"``)
    does not match a ``"WiSun"`` query -- see the module-level detector
    docstrings for exactly why.

    Source-independent by design: this never inspects
    ``metadata["match_source"]`` -- a vector-only, lexical-only, or
    both-source candidate is protected identically, based purely on its
    own text content (see Phase 2D design Section 10: every real
    regression case traced was a lexical-only candidate, so this must
    not special-case retriever origin).

    Ordering: candidates are re-sorted descending by the (possibly
    boosted) score, ties broken by ``record_id`` ascending -- the exact
    same deterministic tie-break convention ``reciprocal_rank_fusion``/
    ``weighted_score_fusion`` already use, so this stays consistent with
    the existing retrieval contract rather than introducing a new
    sorting policy. Returns ``[]`` unchanged for empty input.
    """
    if not matches:
        return matches

    query_identifiers = _detect_identifiers(query)
    if not query_identifiers:
        return matches

    boosted: list["KnowledgeMatch"] = []
    changed = False
    for match in matches:
        candidate_identifiers = _candidate_identifiers(match)
        if candidate_identifiers & query_identifiers:
            changed = True
            new_score = min(match.score + IDENTIFIER_PROTECTION_BOOST, 1.0)
            metadata = dict(match.metadata)
            metadata["identifier_protected"] = True
            boosted.append(match.model_copy(update={"score": new_score, "metadata": metadata}))
        else:
            boosted.append(match)

    if not changed:
        return matches

    return sorted(boosted, key=lambda m: (-m.score, m.record_id))
