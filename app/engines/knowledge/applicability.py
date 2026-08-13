"""Applicability-aware re-ranking for local semantic search (Context
Dimensions phase, 2026-08-12 design assessment Part D; approved product
decision #5: "retrieval/ranking so applicability/context is considered
before pure semantic similarity").

``ChromaKnowledgeStore.query()`` (app/engines/knowledge/knowledge_store.py)
is untouched -- pure vector cosine similarity, exactly as before. This
module is the second pass Part D describes: given an over-fetched
candidate pool, adjust each candidate's score using real,
KnowledgeRelationship-derived Customer/Region tags (see
``app.engines.knowledge.classification``, which is what actually
creates those tags) and the Technology hierarchy
(``Technology.parent_technology_id``), then re-sort -- so a same-
customer match is preferred, a different-*known*-customer match is
actively penalized (never simply left to be outranked, or not, by raw
text similarity alone), and a technology mismatch across a real
parent/child family (RF Mesh vs RF Mesh IP) is treated as "related, not
identical" rather than either a full match or a total miss.

Same fixed-weight, explainable-reasons discipline as
``app.engines.external_knowledge.ranking`` -- deliberately the same
idiom, not a second one; the two differ only in *where* their input
comes from (a live TFS/Wiki candidate vs. an already-indexed local
one). Every adjustment is recorded in the returned match's
``metadata["applicability_reasons"]`` alongside the original
``metadata["semantic_score"]`` -- so a re-ranked result is always
explainable, never a silent reshuffle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.domain.enums import KnowledgeCollection
from app.domain.knowledge_relationships import KnowledgeObjectType, RelationshipType
from app.domain.recommendation import KnowledgeMatch

if TYPE_CHECKING:
    from app.engines.knowledge_relationships.engine import KnowledgeRelationshipEngine
    from app.infrastructure.db.lookup_repository import LookupRepository

_WEIGHT_SAME_CUSTOMER = 0.20
_WEIGHT_DIFFERENT_CUSTOMER = -0.25
"""A real, *known* different customer -- never applied merely because
the candidate has no customer tag at all (absence of evidence is not
evidence of a mismatch). See ``_apply_relationship_signals``."""
_WEIGHT_SAME_REGION = 0.10
_WEIGHT_DIFFERENT_REGION = -0.05
_WEIGHT_TECHNOLOGY_EXACT = 0.20
_WEIGHT_TECHNOLOGY_FAMILY = 0.08
"""Related-but-not-identical technology (parent/child via
``Technology.parent_technology_id``) -- e.g. investigation says "RF
Mesh IP", candidate is tagged bare "RF Mesh": a real, genuinely weaker
signal than an exact match, never collapsed into one. The RF-Mesh-vs-
Mesh-IP fix, applied here at the retrieval layer too, not just Log
Intelligence's log-collection matching
(``RecommendationEngine._recommend_logs``)."""

_COLLECTION_TO_OBJECT_TYPE = {
    KnowledgeCollection.DOCUMENTATION: KnowledgeObjectType.DOCUMENT,
    KnowledgeCollection.HISTORICAL_INVESTIGATIONS: KnowledgeObjectType.HISTORICAL_INVESTIGATION,
    KnowledgeCollection.KNOWN_BUGS: KnowledgeObjectType.KNOWN_BUG,
}


@dataclass(frozen=True)
class RetrievalContext:
    """What's known about the current investigation, resolved to real
    governed rows where possible -- built once per Analyze call
    (``RecommendationEngine._resolve_retrieval_context``), reused
    across every collection re-ranked against it. Any field left
    ``None`` means genuinely unknown, not "assume generic" -- the
    ranker treats unknown as neutral, never as a mismatch."""

    customer_id: str | None = None
    customer_name: str | None = None
    region_id: str | None = None
    region_name: str | None = None
    technology_name: str | None = None
    """Already resolved by ``RecommendationEngine._infer_technology`` --
    reused here, never re-derived a second way."""


@dataclass
class _ScoredCandidate:
    match: KnowledgeMatch
    adjustment: float = 0.0
    reasons: list[str] = field(default_factory=list)


class ApplicabilityRanker:
    """Wraps an already-fetched candidate pool (over-fetched beyond the
    display ``top_k`` by the caller) and re-ranks it before truncation.
    Never issues the vector search itself -- that stays
    ``KnowledgeEngine``/``ChromaKnowledgeStore``'s job unchanged."""

    def __init__(
        self, relationship_engine: "KnowledgeRelationshipEngine | None", lookup_repo: "LookupRepository | None"
    ) -> None:
        self._relationships = relationship_engine
        self._lookup = lookup_repo

    def rerank(self, matches: list[KnowledgeMatch], context: RetrievalContext, *, top_k: int) -> list[KnowledgeMatch]:
        """Adjusts each candidate's score using real applicability
        signals, re-sorts descending by (semantic score + adjustment),
        and truncates to ``top_k``. A no-op beyond truncation when
        neither a relationship engine nor a lookup repository is
        wired, or when ``context`` carries no known dimension at all --
        never fabricates applicability where there's no evidence
        either way, and never changes result order for a caller that
        genuinely has no context yet (e.g. a chat question with no
        investigation open)."""
        if not matches:
            return matches
        if self._relationships is None and self._lookup is None:
            return matches[:top_k]
        if not any((context.customer_id, context.region_id, context.technology_name)):
            return matches[:top_k]

        technology_family = self._technology_family(context.technology_name) if context.technology_name else set()

        scored: list[_ScoredCandidate] = []
        for match in matches:
            candidate = _ScoredCandidate(match=match)
            object_type = _COLLECTION_TO_OBJECT_TYPE.get(match.collection)
            if object_type is not None and self._relationships is not None and (context.customer_id or context.region_id):
                self._apply_relationship_signals(candidate, object_type, context)
            self._apply_technology_signal(candidate, context, technology_family)
            scored.append(candidate)

        scored.sort(key=lambda c: c.match.score + c.adjustment, reverse=True)

        reranked: list[KnowledgeMatch] = []
        for candidate in scored[:top_k]:
            if candidate.adjustment == 0.0:
                reranked.append(candidate.match)
                continue
            adjusted_score = max(0.0, min(1.0, candidate.match.score + candidate.adjustment))
            metadata = dict(candidate.match.metadata)
            metadata["applicability_reasons"] = candidate.reasons
            metadata["semantic_score"] = candidate.match.score
            reranked.append(candidate.match.model_copy(update={"score": adjusted_score, "metadata": metadata}))
        return reranked

    def _apply_relationship_signals(
        self, candidate: "_ScoredCandidate", object_type: KnowledgeObjectType, context: RetrievalContext
    ) -> None:
        relationships = self._relationships.list_relationships(object_type, candidate.match.record_id)
        tagged_customer_ids = {
            (r.to_object.id if r.relationship.from_type == object_type else r.from_object.id)
            for r in relationships
            if r.relationship.relationship_type == RelationshipType.APPLIES_TO
            and KnowledgeObjectType.CUSTOMER in (r.to_object.type, r.from_object.type)
        }
        tagged_region_ids = {
            (r.to_object.id if r.relationship.from_type == object_type else r.from_object.id)
            for r in relationships
            if r.relationship.relationship_type == RelationshipType.APPLIES_TO
            and KnowledgeObjectType.REGION in (r.to_object.type, r.from_object.type)
        }

        if context.customer_id and tagged_customer_ids:
            if context.customer_id in tagged_customer_ids:
                candidate.adjustment += _WEIGHT_SAME_CUSTOMER
                candidate.reasons.append(f"Same customer: {context.customer_name}")
            else:
                candidate.adjustment += _WEIGHT_DIFFERENT_CUSTOMER
                candidate.reasons.append("Tagged for a different customer")

        if context.region_id and tagged_region_ids:
            if context.region_id in tagged_region_ids:
                candidate.adjustment += _WEIGHT_SAME_REGION
                candidate.reasons.append(f"Same region: {context.region_name}")
            else:
                candidate.adjustment += _WEIGHT_DIFFERENT_REGION
                candidate.reasons.append("Tagged for a different region")

    def _apply_technology_signal(
        self, candidate: "_ScoredCandidate", context: RetrievalContext, technology_family: set[str]
    ) -> None:
        if not context.technology_name:
            return
        candidate_technology = str(candidate.match.metadata.get("technology") or "").strip()
        if not candidate_technology:
            return
        if candidate_technology.lower() == context.technology_name.lower():
            candidate.adjustment += _WEIGHT_TECHNOLOGY_EXACT
            candidate.reasons.append(f"Same technology: {context.technology_name}")
        elif candidate_technology in technology_family:
            candidate.adjustment += _WEIGHT_TECHNOLOGY_FAMILY
            candidate.reasons.append(
                f"Related technology family: {candidate_technology} (investigation mentions {context.technology_name})"
            )

    def _technology_family(self, technology_name: str) -> set[str]:
        """Every technology name in the same parent/child family as
        ``technology_name`` (its parent and every sibling/child sharing
        that parent) -- the related-but-not-identical set for
        ``_WEIGHT_TECHNOLOGY_FAMILY``. Empty when no ``LookupRepository``
        is wired or the technology has no populated hierarchy."""
        if self._lookup is None:
            return set()
        technology = self._lookup.get_technology_by_name(technology_name)
        if technology is None:
            return set()
        all_technologies = self._lookup.list_technologies()
        root_id = technology.parent_technology_id or technology.id
        family = {t.name for t in all_technologies if t.id == root_id or t.parent_technology_id == root_id}
        family.discard(technology_name)
        return family
