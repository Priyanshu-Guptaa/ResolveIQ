"""Unit tests for the Recommendation Engine.

Uses a fake KnowledgeStore (structurally satisfying the KnowledgeStore
Protocol) so these tests run with no ChromaDB / embedding model
dependency -- fast and fully offline.
"""

from __future__ import annotations

from app.config import Settings
from app.domain.enums import EvidenceType, KnowledgeCollection
from app.domain.evidence import Evidence
from app.domain.investigation import InvestigationSession
from app.domain.recommendation import KnowledgeMatch
from app.engines.knowledge.engine import KnowledgeEngine
from app.engines.log_intelligence.entity_extractor import RegexEntityExtractor
from app.engines.recommendation.engine import RecommendationEngine


class FakeKnowledgeStore:
    """Structurally satisfies the KnowledgeStore Protocol with canned results."""

    def __init__(self, canned: dict[KnowledgeCollection, list[KnowledgeMatch]] | None = None) -> None:
        self._canned = canned or {}

    def upsert(self, collection, record_id, text, title, metadata) -> None:  # pragma: no cover
        pass

    def query(self, collection: KnowledgeCollection, text: str, top_k: int = 5) -> list[KnowledgeMatch]:
        return self._canned.get(collection, [])[:top_k]

    def count(self, collection: KnowledgeCollection) -> int:
        return len(self._canned.get(collection, []))


def _investigation_with_evidence(text: str) -> InvestigationSession:
    investigation = InvestigationSession(title="Checkout failing for some customers")
    extractor = RegexEntityExtractor()
    evidence = Evidence(
        investigation_id=investigation.id,
        evidence_type=EvidenceType.LOG_FILE,
        raw_content=text,
        extracted_entities=extractor.extract(text),
    )
    investigation.add_evidence(evidence)
    return investigation


def test_strong_historical_match_drives_root_cause_and_next_step():
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hist-007",
        title="NullPointerException in order-service during checkout",
        snippet="...",
        score=0.82,
        metadata={
            "root_cause": "Missing null check on customer shipping address.",
            "next_step": "Search for the exception across all order-service instances.",
        },
    )
    store = FakeKnowledgeStore({KnowledgeCollection.HISTORICAL_INVESTIGATIONS: [match]})
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = _investigation_with_evidence(
        "java.lang.NullPointerException: shippingAddress is null"
    )
    recommendation = engine.generate(investigation)

    assert recommendation.overall_confidence == 0.82
    assert recommendation.root_causes[0].description == "Missing null check on customer shipping address."
    assert recommendation.next_best_step == "Search for the exception across all order-service instances."
    assert recommendation.similar_investigations[0].title == match.title


def test_weak_match_falls_back_to_entity_heuristic():
    weak_match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hist-999",
        title="Unrelated investigation",
        snippet="...",
        score=0.1,  # below min_similarity_for_root_cause
        metadata={"root_cause": "Something unrelated."},
    )
    store = FakeKnowledgeStore({KnowledgeCollection.HISTORICAL_INVESTIGATIONS: [weak_match]})
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = _investigation_with_evidence(
        "java.lang.NullPointerException: shippingAddress is null"
    )
    recommendation = engine.generate(investigation)

    descriptions = [rc.description for rc in recommendation.root_causes]
    assert any("NullPointerException" in d for d in descriptions)
    assert "Something unrelated." not in descriptions


def test_no_evidence_prompts_for_more_information():
    store = FakeKnowledgeStore()
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = InvestigationSession(title="Empty investigation")
    recommendation = engine.generate(investigation)

    assert recommendation.root_causes == []
    assert "task description" in recommendation.next_best_step.lower()


def test_sql_session_entity_yields_suggested_sql():
    store = FakeKnowledgeStore()
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = _investigation_with_evidence("Blocking detected, spid=61 is head blocker")
    recommendation = engine.generate(investigation)

    assert any("61" in sql for sql in recommendation.suggested_sql)
