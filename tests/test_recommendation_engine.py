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


def test_matches_with_no_captured_root_cause_are_not_surfaced_as_a_finding():
    """Regression guard for a real bug: bulk-imported ServiceNow tickets
    (app/engines/task_import) have no distinct root-cause field in their
    source data, so their `root_cause` is deliberately left empty rather
    than filled with a boilerplate placeholder -- an earlier version did
    use a placeholder sentence, and when two such tickets both ranked as
    top matches, the Investigation Workspace showed the identical
    boilerplate sentence twice as if it were two distinct diagnosed root
    causes. A high-scoring match with an empty root_cause must be
    skipped entirely, not surfaced as an empty/fabricated finding."""
    matches = [
        KnowledgeMatch(
            collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
            record_id="hist-imported-1",
            title="CC - Check Kafka Consumer Group Lag",
            snippet="...",
            score=0.85,
            metadata={"root_cause": ""},
        ),
        KnowledgeMatch(
            collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
            record_id="hist-imported-2",
            title="CC - Check Drive Usage",
            snippet="...",
            score=0.80,
            metadata={"root_cause": ""},
        ),
    ]
    store = FakeKnowledgeStore({KnowledgeCollection.HISTORICAL_INVESTIGATIONS: matches})
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = _investigation_with_evidence("Kafka consumer group lag spike on billing-events")
    recommendation = engine.generate(investigation)

    assert recommendation.root_causes == []
    # The matches themselves are still useful precedent -- they must
    # still appear in Historical Matches even though they contribute no
    # root-cause hypothesis.
    assert [m.title for m in recommendation.similar_investigations] == [
        "CC - Check Kafka Consumer Group Lag",
        "CC - Check Drive Usage",
    ]


def test_next_best_step_truncates_a_long_historical_resolution():
    """Regression guard for a real bug: `resolution` on a bulk-imported
    ServiceNow ticket is the ticket's entire "Comments and Work notes"
    column -- every work-note entry ever added, potentially thousands
    of characters. Dumping that verbatim into "Next best step" (meant
    to be a short, scannable hint) made a real match read as an
    unreadable wall of text, easily mistaken for "unrelated task
    information" leaking in. The full text must be capped and marked
    truncated, not echoed in full."""
    long_resolution = "Work note entry. " * 200  # 3,600 chars, well over the cap
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hist-long",
        title="A resolved ticket with a long work-note history",
        snippet="...",
        score=0.9,
        metadata={"resolution": long_resolution, "root_cause": ""},
    )
    store = FakeKnowledgeStore({KnowledgeCollection.HISTORICAL_INVESTIGATIONS: [match]})
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = _investigation_with_evidence("some log content")
    recommendation = engine.generate(investigation)

    assert len(recommendation.next_best_step) < len(long_resolution)
    assert recommendation.next_best_step.endswith("…")


def test_root_cause_description_is_truncated_for_a_long_historical_root_cause():
    long_root_cause = "Detailed root cause narrative. " * 100  # well over the cap
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hist-long-rc",
        title="A resolved ticket with a long root cause writeup",
        snippet="...",
        score=0.9,
        metadata={"root_cause": long_root_cause},
    )
    store = FakeKnowledgeStore({KnowledgeCollection.HISTORICAL_INVESTIGATIONS: [match]})
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = _investigation_with_evidence("some log content")
    recommendation = engine.generate(investigation)

    assert len(recommendation.root_causes[0].description) < len(long_root_cause)
    assert recommendation.root_causes[0].description.endswith("…")


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
