"""Unit tests for ApplicabilityRanker (Context Dimensions phase,
2026-08-12 -- approved product decision #5: "retrieval/ranking so
applicability/context is considered before pure semantic similarity").

Uses fakes for KnowledgeRelationshipEngine/LookupRepository so these run
fully offline, same discipline as test_recommendation_engine.py's
FakeKnowledgeStore.
"""

from __future__ import annotations

from app.domain.enums import KnowledgeCollection
from app.domain.knowledge_relationships import (
    KnowledgeObjectRef,
    KnowledgeObjectType,
    KnowledgeRelationship,
    RelationshipType,
    ResolvedRelationship,
)
from app.domain.lookup_entities import Technology
from app.domain.recommendation import KnowledgeMatch
from app.engines.knowledge.applicability import ApplicabilityRanker, RetrievalContext


class FakeRelationshipEngine:
    """Structurally satisfies the parts of KnowledgeRelationshipEngine
    ApplicabilityRanker uses -- list_relationships(object_type, id)."""

    def __init__(self, relationships: dict[tuple[str, str], list[ResolvedRelationship]] | None = None) -> None:
        self._relationships = relationships or {}

    def list_relationships(self, object_type: KnowledgeObjectType, object_id: str) -> list[ResolvedRelationship]:
        return self._relationships.get((object_type, object_id), [])


class FakeLookupRepo:
    def __init__(self, technologies: list[Technology] | None = None) -> None:
        self._technologies = technologies or []

    def get_technology_by_name(self, name: str) -> Technology | None:
        return next((t for t in self._technologies if t.name == name), None)

    def list_technologies(self) -> list[Technology]:
        return list(self._technologies)


def _doc_match(record_id: str, score: float, technology: str | None = None) -> KnowledgeMatch:
    metadata = {"technology": technology} if technology else {}
    return KnowledgeMatch(
        collection=KnowledgeCollection.DOCUMENTATION, record_id=record_id, title=f"Doc {record_id}", snippet="", score=score, metadata=metadata
    )


def _applies_to_customer(doc_id: str, customer_id: str) -> ResolvedRelationship:
    return ResolvedRelationship(
        relationship=KnowledgeRelationship(
            from_type=KnowledgeObjectType.DOCUMENT, from_id=doc_id, to_type=KnowledgeObjectType.CUSTOMER, to_id=customer_id,
            relationship_type=RelationshipType.APPLIES_TO,
        ),
        from_object=KnowledgeObjectRef(type=KnowledgeObjectType.DOCUMENT, id=doc_id, title="Doc"),
        to_object=KnowledgeObjectRef(type=KnowledgeObjectType.CUSTOMER, id=customer_id, title="Cust"),
    )


def test_no_context_leaves_order_and_scores_unchanged():
    matches = [_doc_match("a", 0.9), _doc_match("b", 0.8)]
    ranker = ApplicabilityRanker(FakeRelationshipEngine(), FakeLookupRepo())
    result = ranker.rerank(matches, RetrievalContext(), top_k=5)
    assert [m.record_id for m in result] == ["a", "b"]
    assert result[0].score == 0.9


def test_same_customer_boosts_a_lower_scoring_candidate_above_a_higher_one():
    relationships = {
        (KnowledgeObjectType.DOCUMENT, "generic"): [],
        (KnowledgeObjectType.DOCUMENT, "tepco-doc"): [_applies_to_customer("tepco-doc", "cust-tepco")],
    }
    matches = [_doc_match("generic", 0.85), _doc_match("tepco-doc", 0.70)]
    ranker = ApplicabilityRanker(FakeRelationshipEngine(relationships), FakeLookupRepo())
    context = RetrievalContext(customer_id="cust-tepco", customer_name="TEPCO")

    result = ranker.rerank(matches, context, top_k=5)

    assert result[0].record_id == "tepco-doc"
    assert result[0].score > 0.70
    assert "Same customer: TEPCO" in result[0].metadata["applicability_reasons"]
    assert result[0].metadata["semantic_score"] == 0.70


def test_different_known_customer_is_penalized_not_just_left_alone():
    relationships = {
        (KnowledgeObjectType.DOCUMENT, "other-customer-doc"): [_applies_to_customer("other-customer-doc", "cust-atco")],
    }
    matches = [_doc_match("other-customer-doc", 0.90)]
    ranker = ApplicabilityRanker(FakeRelationshipEngine(relationships), FakeLookupRepo())
    context = RetrievalContext(customer_id="cust-tepco", customer_name="TEPCO")

    result = ranker.rerank(matches, context, top_k=5)

    assert result[0].score < 0.90
    assert "Tagged for a different customer" in result[0].metadata["applicability_reasons"]


def test_candidate_with_no_customer_tag_at_all_is_never_penalized():
    """Absence of a customer tag is not evidence of a mismatch -- only
    an explicit, real, different customer tag triggers the penalty."""
    relationships = {(KnowledgeObjectType.DOCUMENT, "untagged"): []}
    matches = [_doc_match("untagged", 0.90)]
    ranker = ApplicabilityRanker(FakeRelationshipEngine(relationships), FakeLookupRepo())
    context = RetrievalContext(customer_id="cust-tepco", customer_name="TEPCO")

    result = ranker.rerank(matches, context, top_k=5)
    assert result[0].score == 0.90


def test_exact_technology_match_outscores_family_match_which_outscores_no_signal():
    rf_mesh = Technology(id="tech-rf-mesh", name="RF Mesh")
    rf_mesh_ip = Technology(id="tech-rf-mesh-ip", name="RF Mesh IP", parent_technology_id="tech-rf-mesh")
    lookup = FakeLookupRepo([rf_mesh, rf_mesh_ip])
    relationships = FakeRelationshipEngine()
    ranker = ApplicabilityRanker(relationships, lookup)
    context = RetrievalContext(technology_name="RF Mesh IP")

    matches = [
        _doc_match("exact", 0.70, technology="RF Mesh IP"),
        _doc_match("family", 0.70, technology="RF Mesh"),
        _doc_match("none", 0.70, technology=None),
    ]
    result = ranker.rerank(matches, context, top_k=5)
    scores = {m.record_id: m.score for m in result}
    assert scores["exact"] > scores["family"] > scores["none"]


def test_no_relationship_engine_or_lookup_wired_is_a_safe_noop():
    """Plain truncation of the input in the order given -- callers
    always pass already similarity-ordered results, but this ranker
    must not re-sort them itself when it has no real signal to add."""
    matches = [_doc_match("b", 0.95), _doc_match("a", 0.9)]
    ranker = ApplicabilityRanker(None, None)
    result = ranker.rerank(matches, RetrievalContext(customer_id="x"), top_k=1)
    assert len(result) == 1
    assert result[0].record_id == "b"
