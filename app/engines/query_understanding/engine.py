"""Query Understanding Engine -- deterministic, zero-LLM parsing of
free-form text into the governed ``ParsedQuery`` shape (2026-08-14,
Phase 2 -- RESOLVEIQ_CHAT_AND_RESOLUTION_ARCHITECTURE.md, "Phase 2 --
Query Understanding"). See ``app.domain.query_understanding`` for the
full contract and confidence model this engine populates.

Reuses, rather than reimplements, every matching primitive that
already exists for this exact purpose elsewhere in the codebase:

- Customer/Region/Product/Component -- full-phrase-or-alias matching
  via ``app.engines.shared.governed_text_matching`` (the same
  primitives ``app.engines.knowledge.classification`` now also uses,
  extracted 2026-08-14 specifically so both engines share one
  implementation).
- Technology -- the same hierarchy-aware evidence-coverage scoring
  ``RecommendationEngine._match_single_technology`` uses
  (``app.engines.shared.text_matching.evidence_coverage_match_score``
  plus ``app.engines.shared.hierarchy.most_specific``), reimplemented
  here (not called through ``RecommendationEngine``) only because that
  method collapses "genuinely ambiguous" and "no evidence" into the
  same ``None`` return -- Query Understanding needs to tell those two
  apart (``SlotConfidence.AMBIGUOUS`` with named candidates vs. the
  slot being absent) for a caller to know why nothing definite came
  back. The scoring and specificity-resolution logic itself is
  identical, not a second implementation of the *rule*.
- Exception type -- ``app.engines.log_intelligence.entity_extractor.
  RegexEntityExtractor`` (``EntityType.EXCEPTION_TYPE`` only).
- Ticket references -- ``app.engines.external_knowledge.service.
  _TICKET_NUMBER_RE``, the same pattern External Knowledge already
  uses to correlate a local record to a live TFS case.

Intent classification is the one genuinely new mechanism this engine
adds -- a fixed, data-driven phrase table
(``_INTENT_PATTERNS``, mirroring the "registry of rules, not
hardcoded control flow" idiom ``entity_extractor.py``'s
``_PATTERN_REGISTRY`` already established), never an LLM, never a
learned model. See ``_classify_intent`` for the exact, conservative
scoring rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.domain.entities import ExtractedEntity
from app.domain.enums import EntityType
from app.domain.query_understanding import (
    ExtractedSlot,
    ParsedQuery,
    QueryIntent,
    SlotConfidence,
)
from app.engines.external_knowledge.service import _TICKET_NUMBER_RE
from app.engines.knowledge.applicability import RetrievalContext
from app.engines.log_intelligence.entity_extractor import EntityExtractor, RegexEntityExtractor
from app.engines.shared.governed_text_matching import (
    GovernedCandidate,
    full_phrase_match,
    most_specific_candidates,
)
from app.engines.shared.hierarchy import most_specific
from app.engines.shared.text_matching import FULL_MATCH_SCORE, evidence_coverage_match_score, phrase_present

if TYPE_CHECKING:
    from app.infrastructure.db.component_repository import ComponentProfileRepository
    from app.infrastructure.db.lookup_repository import LookupRepository

_EVIDENCE_SNIPPET_CAP = 120
"""Caps how much raw text is echoed back on an AMBIGUOUS/exception-type
slot's ``evidence_snippet`` when there's no single matched phrase to
quote instead -- same discipline as every other snippet cap in this
codebase (``classification.py``'s ``_SNIPPET_WINDOW``,
``InvestigationSession.context_text``'s cap)."""


@dataclass(frozen=True)
class _IntentPattern:
    phrase: str
    """A literal phrase, checked as a whole-phrase, word-boundary-safe
    match (see ``app.engines.shared.text_matching.phrase_present``) --
    deliberately not a single keyword, so a generic word alone (e.g.
    "issue") never counts as evidence for the more specific intents
    below it."""


# Fixed, data-driven phrase table -- one list of literal cue phrases
# per intent. Not exhaustive by design: a conservative classifier that
# sometimes returns UNKNOWN is the explicit requirement here, not a
# maximal-recall one. TROUBLESHOOTING is deliberately the broadest/most
# generic list (a free-form "something's wrong" question has no more
# specific cue), which is why it only wins on a real margin over every
# other intent's score -- see ``_classify_intent``.
_INTENT_PATTERNS: dict[QueryIntent, list[_IntentPattern]] = {
    QueryIntent.KNOWN_BUG_LOOKUP: [
        _IntentPattern("known bug"),
        _IntentPattern("known issue"),
        _IntentPattern("known defect"),
        _IntentPattern("is this a known"),
        _IntentPattern("already a bug for this"),
        _IntentPattern("bug id"),
        _IntentPattern("defect id"),
    ],
    QueryIntent.HISTORICAL_LOOKUP: [
        _IntentPattern("seen this before"),
        _IntentPattern("seen this issue before"),
        _IntentPattern("similar case"),
        _IntentPattern("similar issue"),
        _IntentPattern("similar ticket"),
        _IntentPattern("past investigation"),
        _IntentPattern("previous ticket"),
        _IntentPattern("prior occurrence"),
        _IntentPattern("history of this"),
        _IntentPattern("any past cases"),
        _IntentPattern("any similar cases"),
    ],
    QueryIntent.LOG_GUIDANCE: [
        _IntentPattern("which logs"),
        _IntentPattern("what logs"),
        _IntentPattern("which log"),
        _IntentPattern("what log"),
        _IntentPattern("log collection"),
        _IntentPattern("where to find logs"),
        _IntentPattern("log location"),
        _IntentPattern("collect logs"),
        _IntentPattern("logs should i collect"),
        _IntentPattern("logs do i need"),
    ],
    QueryIntent.SQL_GUIDANCE: [
        _IntentPattern("sql query"),
        _IntentPattern("sql template"),
        _IntentPattern("database query"),
        _IntentPattern("query to check"),
        _IntentPattern("select statement"),
        _IntentPattern("run a query"),
        _IntentPattern("sql to check"),
    ],
    QueryIntent.RELEASE_NOTE_LOOKUP: [
        _IntentPattern("release note"),
        _IntentPattern("release notes"),
        _IntentPattern("what changed in"),
        _IntentPattern("changelog"),
        _IntentPattern("fixed in version"),
        _IntentPattern("fixed in release"),
        _IntentPattern("what's new in"),
        _IntentPattern("whats new in"),
    ],
    QueryIntent.DOCUMENTATION_LOOKUP: [
        _IntentPattern("documentation for"),
        _IntentPattern("doc for"),
        _IntentPattern("wiki page"),
        _IntentPattern("how do i configure"),
        _IntentPattern("how do i install"),
        _IntentPattern("how to configure"),
        _IntentPattern("how to install"),
        _IntentPattern("user guide"),
        _IntentPattern("setup guide"),
        _IntentPattern("configuration guide"),
    ],
    QueryIntent.TROUBLESHOOTING: [
        _IntentPattern("not working"),
        _IntentPattern("isn't working"),
        _IntentPattern("is not working"),
        _IntentPattern("failing"),
        _IntentPattern("fails to"),
        _IntentPattern("unable to"),
        _IntentPattern("stuck"),
        _IntentPattern("timeout"),
        _IntentPattern("timed out"),
        _IntentPattern("crash"),
        _IntentPattern("crashing"),
        _IntentPattern("not responding"),
        _IntentPattern("doesn't respond"),
        _IntentPattern("error"),
        _IntentPattern("issue"),
        _IntentPattern("problem"),
        _IntentPattern("broken"),
    ],
}


class QueryUnderstandingEngine:
    """See module docstring. Deterministic, no LLM, no new retrieval,
    no DB writes -- ``parse()`` only reads already-governed lookup
    tables and matches them against the text it's given."""

    def __init__(
        self,
        lookup_repo: "LookupRepository",
        component_repo: "ComponentProfileRepository | None" = None,
        entity_extractor: EntityExtractor | None = None,
    ) -> None:
        self._lookup = lookup_repo
        self._components = component_repo
        self._entities = entity_extractor or RegexEntityExtractor()

    def parse(self, text: str, existing_context: RetrievalContext | None = None) -> ParsedQuery:
        raw_text = text or ""

        intent, intent_confidence = self._classify_intent(raw_text)

        customer_slot = self._match_governed_dimension(self._customer_candidates(), raw_text)
        region_slot = self._match_governed_dimension(self._region_candidates(), raw_text)
        product_slot = self._match_governed_dimension(self._product_candidates(), raw_text)
        component_slot = self._match_governed_dimension(self._component_candidates(), raw_text)
        version_slot = self._match_governed_dimension(self._version_candidates(), raw_text)
        technology_slot = self._match_technology(raw_text)

        entities = self._entities.extract(raw_text) if raw_text.strip() else []
        exception_slot = self._exception_type_slot(entities)
        ticket_references = sorted({match.upper() for match in _TICKET_NUMBER_RE.findall(raw_text)})

        retrieval_context = self._build_retrieval_context(existing_context, customer_slot, region_slot, technology_slot)

        return ParsedQuery(
            raw_text=raw_text,
            intent=intent,
            intent_confidence=intent_confidence,
            customer=customer_slot,
            region=region_slot,
            technology=technology_slot,
            product=product_slot,
            component=component_slot,
            version=version_slot,
            exception_type=exception_slot,
            ticket_references=ticket_references,
            retrieval_context=retrieval_context,
        )

    # --- Intent classification ----------------------------------------

    def _classify_intent(self, text: str) -> tuple[QueryIntent, float]:
        """Conservative, deterministic: scores every intent by how many
        of its own cue phrases appear in ``text``; the intent needs a
        strictly higher score than every other intent to win at all --
        a tie at the max (including a 0-0 tie, i.e. nothing matched
        anything) returns UNKNOWN with 0.0 confidence rather than
        guessing. Confidence for a genuine winner is
        ``min(0.9, 0.5 + 0.1 * (score - 1))`` -- 0.5 for a single
        matched cue phrase, +0.1 per additional one, capped at 0.9 (a
        fixed pattern table, however many phrases matched, is never
        asserted as 100% certain the way a real cross-source
        correlation is)."""
        if not text.strip():
            return QueryIntent.UNKNOWN, 0.0

        scores: dict[QueryIntent, int] = {
            intent: sum(1 for pattern in patterns if phrase_present(pattern.phrase, text))
            for intent, patterns in _INTENT_PATTERNS.items()
        }
        best_score = max(scores.values())
        if best_score == 0:
            return QueryIntent.UNKNOWN, 0.0

        winners = [intent for intent, score in scores.items() if score == best_score]
        if len(winners) != 1:
            return QueryIntent.UNKNOWN, 0.0

        confidence = min(0.9, 0.5 + 0.1 * (best_score - 1))
        return winners[0], round(confidence, 2)

    # --- Governed-dimension candidates ---------------------------------
    # Customer/Region/Product/Component share the exact same matching
    # mechanic (full-phrase-or-alias, no hierarchy) -- see
    # ``_match_governed_dimension``. Only the candidate list differs per
    # dimension, and each of those lists is built straight from the
    # real governed table, exactly as classification.py's own ``run()``
    # builds its own per-dimension candidate lists.

    def _customer_candidates(self) -> list[GovernedCandidate]:
        return [
            GovernedCandidate(c.id, c.name, (c.name, *c.aliases))
            for c in self._lookup.list_customers()
        ]

    def _region_candidates(self) -> list[GovernedCandidate]:
        return [
            GovernedCandidate(r.id, r.name, (r.name, *r.aliases))
            for r in self._lookup.list_regions()
        ]

    def _product_candidates(self) -> list[GovernedCandidate]:
        return [GovernedCandidate(p.id, p.name, (p.name,)) for p in self._lookup.list_products()]

    def _component_candidates(self) -> list[GovernedCandidate]:
        if self._components is None:
            return []
        return [GovernedCandidate(c.id, c.name, (c.name,)) for c in self._components.list_all()]

    def _version_candidates(self) -> list[GovernedCandidate]:
        return [GovernedCandidate(v.id, v.name, (v.name,)) for v in self._lookup.list_versions()]

    def _match_governed_dimension(self, candidates: list[GovernedCandidate], text: str) -> ExtractedSlot | None:
        """EXACT on a single full-phrase-or-alias match; AMBIGUOUS when
        more than one real, distinct candidate matches with no
        hierarchy relation to prefer one (every dimension this is used
        for -- Customer/Region/Product/Component/Version -- has no
        parent/child concept at all, so ``most_specific_candidates`` is
        always a structural no-op here; it's still called, rather than
        skipped, so this stays the exact same code path Technology
        matching uses and a future hierarchical dimension needs no
        change here); ``None`` when nothing matched at all. Never
        PARTIAL -- these dimensions' own matching rule (reused verbatim
        from ``classification.py`` via ``full_phrase_match``) has no
        partial tier."""
        if not text.strip() or not candidates:
            return None
        matched = [c for c in candidates if full_phrase_match(c, text) is not None]
        if not matched:
            return None

        specific = most_specific_candidates(matched, candidates)
        if len(specific) == 1:
            winner = specific[0]
            snippet = full_phrase_match(winner, text) or winner.canonical_name
            return ExtractedSlot(
                value_id=winner.id, value_name=winner.canonical_name, confidence=SlotConfidence.EXACT, evidence_snippet=snippet
            )

        names = sorted({c.canonical_name for c in specific})
        return ExtractedSlot(
            confidence=SlotConfidence.AMBIGUOUS,
            evidence_snippet=text.strip()[:_EVIDENCE_SNIPPET_CAP],
            ambiguous_candidates=names,
        )

    # --- Technology ------------------------------------------------------

    def _match_technology(self, text: str) -> ExtractedSlot | None:
        """Mirrors ``RecommendationEngine._match_single_technology``'s
        exact algorithm (see that method's docstring for the full
        rationale and the real bugs its rules fix): score every real
        governed Technology via ``evidence_coverage_match_score``,
        drop anything at 0.0 (no evidence), take the candidates tied at
        the single highest positive score, and prefer the most specific
        one via the real ``Technology.parent_technology_id`` hierarchy.
        Differs from that method only in what it returns once more than
        one candidate survives specificity filtering: that method
        returns ``None`` either way (no evidence, or genuine
        ambiguity); this one distinguishes them, returning
        SlotConfidence.AMBIGUOUS with the tied names instead of
        silently returning nothing."""
        if not text.strip():
            return None
        technologies = self._lookup.list_technologies()
        if not technologies:
            return None

        candidates = [
            GovernedCandidate(t.id, t.name, (t.name,), parent_id=t.parent_technology_id) for t in technologies
        ]
        scores = {c.id: evidence_coverage_match_score(c.canonical_name, text) for c in candidates}
        positive = [c for c in candidates if scores[c.id] > 0]
        if not positive:
            return None

        best_score = max(scores[c.id] for c in positive)
        tied = [c for c in positive if scores[c.id] == best_score]
        specific = most_specific_candidates(tied, candidates) if len(tied) > 1 else tied

        confidence = SlotConfidence.EXACT if best_score >= FULL_MATCH_SCORE else SlotConfidence.PARTIAL
        if len(specific) == 1:
            winner = specific[0]
            return ExtractedSlot(
                value_id=winner.id, value_name=winner.canonical_name, confidence=confidence, evidence_snippet=winner.canonical_name
            )

        names = sorted({c.canonical_name for c in specific})
        return ExtractedSlot(
            confidence=SlotConfidence.AMBIGUOUS,
            evidence_snippet=text.strip()[:_EVIDENCE_SNIPPET_CAP],
            ambiguous_candidates=names,
        )

    # --- Exception type / ticket references -----------------------------

    def _exception_type_slot(self, entities: list[ExtractedEntity]) -> ExtractedSlot | None:
        """Not a governed dimension -- there is no ``exception_types``
        table to match against, only ``RegexEntityExtractor``'s pattern
        recognition (see module docstring). A single distinct extracted
        value is EXACT (the pattern matched real text, unambiguously);
        more than one distinct value in the same text is AMBIGUOUS
        (which one is *the* relevant exception is genuinely unclear
        from text alone) rather than arbitrarily picking the first."""
        values = list(dict.fromkeys(e.value for e in entities if e.entity_type == EntityType.EXCEPTION_TYPE))
        if not values:
            return None
        if len(values) == 1:
            snippet = next(e.context_snippet for e in entities if e.entity_type == EntityType.EXCEPTION_TYPE)
            return ExtractedSlot(value_name=values[0], confidence=SlotConfidence.EXACT, evidence_snippet=snippet)
        return ExtractedSlot(
            confidence=SlotConfidence.AMBIGUOUS,
            evidence_snippet=f"{len(values)} distinct exception/error names found in text",
            ambiguous_candidates=sorted(values),
        )

    # --- RetrievalContext (existing-context-wins) ------------------------

    def _build_retrieval_context(
        self,
        existing_context: RetrievalContext | None,
        customer_slot: ExtractedSlot | None,
        region_slot: ExtractedSlot | None,
        technology_slot: ExtractedSlot | None,
    ) -> RetrievalContext:
        """Populates the existing, unchanged ``RetrievalContext`` shape.
        Existing context always wins: a field already populated on
        ``existing_context`` is carried through untouched, never
        replaced by a same-turn inference from free text -- e.g.
        ``existing_context.customer_name == "TEPCO"`` stays "TEPCO"
        even when the text says "This looks like a CLECO issue," no
        matter how confidently CLECO was extracted. A field only gets
        filled from an extracted slot when ``existing_context`` (or its
        absence) leaves it genuinely unknown, and only from an
        EXACT slot for Customer/Region (matching
        ``RecommendationEngine._resolve_retrieval_context``'s own
        exact-name-match-only rule) or an EXACT-or-PARTIAL Technology
        slot (matching ``_match_single_technology``'s own behavior,
        which never distinguishes a partial evidence-coverage match
        from a full one once it's the unique winner). An AMBIGUOUS slot
        never populates ``RetrievalContext`` -- unknown must stay
        unknown, never a guess among the tied candidates.

        Note what this does NOT hide: the real extracted CLECO
        evidence is still visible on ``ParsedQuery.customer`` (the
        slot itself, populated straight from the text, independent of
        ``existing_context``) -- only *retrieval* prefers the
        already-established TEPCO. Nothing about the disagreement is
        silently discarded, the same "preserve every real signal,
        never silently hide one" discipline Phase 1's conflict handling
        already established for resolution evidence."""
        if existing_context is not None:
            customer_id, customer_name = existing_context.customer_id, existing_context.customer_name
            region_id, region_name = existing_context.region_id, existing_context.region_name
            technology_name = existing_context.technology_name
        else:
            customer_id = customer_name = region_id = region_name = technology_name = None

        if customer_id is None and customer_slot is not None and customer_slot.confidence == SlotConfidence.EXACT:
            customer_id, customer_name = customer_slot.value_id, customer_slot.value_name
        if region_id is None and region_slot is not None and region_slot.confidence == SlotConfidence.EXACT:
            region_id, region_name = region_slot.value_id, region_slot.value_name
        if technology_name is None and technology_slot is not None and technology_slot.confidence in (
            SlotConfidence.EXACT,
            SlotConfidence.PARTIAL,
        ):
            technology_name = technology_slot.value_name

        return RetrievalContext(
            customer_id=customer_id,
            customer_name=customer_name,
            region_id=region_id,
            region_name=region_name,
            technology_name=technology_name,
        )
