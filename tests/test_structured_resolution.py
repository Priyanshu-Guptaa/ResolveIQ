"""Tests for Structured Resolution Knowledge (2026-08-14, Phase 1 of
RESOLVEIQ_CHAT_AND_RESOLUTION_ARCHITECTURE.md): the read-model
assembly (``RecommendationEngine._build_structured_resolution`` /
``StructuredResolutionEngine.from_historical_investigation`` /
``.from_known_bug``), ``ValidationStep``, and the two conflict-
resolution ordering functions.

Integration-style against a real temp-file SQLite database (same
pattern as test_classification.py/test_resolution_verification.py) --
this genuinely depends on ``KnowledgeRelationshipEngine`` (real
Customer/Region tagging), so a fake would just re-assert the mock. The
Chroma-facing side uses the same ``FakeKnowledgeStore`` pattern as
test_provenance.py, since this phase adds zero new retrieval and the
existing tests already cover that layer.
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.config import Settings
from app.domain.enums import KnowledgeCollection
from app.domain.evidence import HistoricalInvestigationRecord, KnownBugRecord
from app.domain.external_knowledge import ExternalKnowledgeResult, ExternalMatch, ExternalSource, TfsCase, WikiPage
from app.domain.knowledge_relationships import KnowledgeObjectType, RelationshipType
from app.domain.lookup_entities import Customer, Region
from app.domain.provenance import EvidenceKind, ResolutionProvenance, ValidationStep
from app.domain.recommendation import KnowledgeMatch, RootCauseHypothesis
from app.domain.structured_resolution import ApplicabilitySummary, ResolutionCandidate, StructuredResolution
from app.engines.knowledge.engine import KnowledgeEngine
from app.engines.knowledge_relationships.engine import KnowledgeRelationshipEngine
from app.engines.recommendation.engine import RecommendationEngine
from app.engines.structured_resolution.engine import (
    StructuredResolutionEngine,
    order_resolution_candidates,
    order_resolutions,
)
from app.infrastructure.db.component_repository import SqlAlchemyComponentProfileRepository
from app.infrastructure.db.knowledge_repository import SqlAlchemyKnowledgeRepository
from app.infrastructure.db.log_knowledge_repository import SqlAlchemyLogKnowledgeRepository
from app.infrastructure.db.lookup_repository import SqlAlchemyLookupRepository
from app.infrastructure.db.playbook_repository import SqlAlchemyPlaybookRepository
from app.infrastructure.db.relationship_repository import SqlAlchemyRelationshipRepository
from app.infrastructure.db.session import get_engine, get_session_factory
from app.infrastructure.db.sql_template_repository import SqlAlchemySqlTemplateRepository


class FakeKnowledgeStore:
    def upsert(self, collection, record_id, text, title, metadata) -> None:  # pragma: no cover
        pass

    def query(self, collection, text, top_k: int = 5):  # pragma: no cover
        return []

    def count(self, collection) -> int:
        return 0

    def list_recent(self, collection, limit: int = 5):  # pragma: no cover
        return []

    def delete(self, collection, record_id: str) -> None:  # pragma: no cover
        pass


@pytest.fixture
def bundle():
    with tempfile.TemporaryDirectory() as tmp:
        sqlite_url = f"sqlite:///{(Path(tmp) / 'test.db').as_posix()}"
        session_factory = get_session_factory(sqlite_url)

        component_repo = SqlAlchemyComponentProfileRepository(session_factory)
        knowledge_repo = SqlAlchemyKnowledgeRepository(session_factory)
        sql_repo = SqlAlchemySqlTemplateRepository(session_factory)
        lookup_repo = SqlAlchemyLookupRepository(session_factory)
        log_knowledge_repo = SqlAlchemyLogKnowledgeRepository(session_factory)
        playbook_repo = SqlAlchemyPlaybookRepository(session_factory)
        relationship_repo = SqlAlchemyRelationshipRepository(session_factory)

        relationship_engine = KnowledgeRelationshipEngine(
            relationship_repo, component_repo, knowledge_repo, sql_repo, playbook_repo, lookup_repo, log_knowledge_repo
        )
        knowledge_engine = KnowledgeEngine(FakeKnowledgeStore(), knowledge_repo)
        rec_engine = RecommendationEngine(
            knowledge_engine, Settings(), component_repo=component_repo, relationship_engine=relationship_engine, lookup_repo=lookup_repo
        )
        structured_engine = StructuredResolutionEngine(knowledge_repo, relationship_engine)

        yield dict(
            knowledge_repo=knowledge_repo,
            lookup_repo=lookup_repo,
            relationship_engine=relationship_engine,
            rec_engine=rec_engine,
            structured_engine=structured_engine,
        )
        get_engine(sqlite_url).dispose()
        get_engine.cache_clear()


def _tfs_case(tfs_id=1, title="CommandProcessorHost issue", state="Closed", resolution="Restart the service", crm_id=None):
    return TfsCase(
        tfs_id=tfs_id, work_item_type="Bug", title=title, state=state, area_path="Command Center",
        team_project="Command Center", changed_date=datetime(2026, 1, 1), description_text="x",
        resolution_text=resolution, crm_id=crm_id, url=f"https://tfs/edit/{tfs_id}",
    )


def _tfs_result(matches=None, confidence="High"):
    if matches is None:
        return ExternalKnowledgeResult(source=ExternalSource.TFS, available=True, matches=[])
    return ExternalKnowledgeResult(
        source=ExternalSource.TFS, available=True,
        matches=[ExternalMatch(source=ExternalSource.TFS, tfs_case=c, score=s, confidence=confidence, match_reasons=["Same component: X"]) for c, s in matches],
    )


def _save_hi(knowledge_repo, **overrides) -> HistoricalInvestigationRecord:
    defaults = dict(
        id=str(uuid.uuid4()), title="Commands stuck in Pending", description="Queue backlog.",
        root_cause="Stale queue processor", resolution="Restart CommandProcessorHost", next_step="Confirm command state transitions from Pending to Sent to Response.",
    )
    defaults.update(overrides)
    record = HistoricalInvestigationRecord(**defaults)
    knowledge_repo.save_historical_investigation(record)
    return record


def _save_bug(knowledge_repo, **overrides) -> KnownBugRecord:
    defaults = dict(id=str(uuid.uuid4()), title="Known queue backup bug", description="x", workaround="Increase timeout")
    defaults.update(overrides)
    record = KnownBugRecord(**defaults)
    knowledge_repo.save_known_bug(record)
    return record


def _match_for(record: HistoricalInvestigationRecord, score: float, tags: str = "") -> KnowledgeMatch:
    return KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS, record_id=record.id, title=record.title,
        snippet="x", score=score,
        metadata={"root_cause": record.root_cause, "resolution": record.resolution, "next_step": record.next_step, "tags": tags},
    )


# --- 1. A resolution with a single strong source ------------------------


def test_single_strong_source_produces_one_primary_candidate(bundle):
    record = _save_hi(bundle["knowledge_repo"])
    match = _match_for(record, 0.8)
    root_causes = [RootCauseHypothesis(description=record.root_cause, confidence=0.8, rationale=f"Matches historical investigation '{record.title}' (80% similarity).")]
    engine = bundle["rec_engine"]

    solution, sources = engine._synthesize_recommendation(root_causes, [match], _tfs_result(matches=[]), None, [])
    provenance = engine._build_provenance_record(
        root_causes=root_causes, similar_investigations=[match], recommended_logs=[], suggested_sql=[],
        recommended_solution=solution, sources=sources,
    )
    from app.domain.investigation import InvestigationSession
    inv = InvestigationSession(title=record.title)
    result = engine._build_structured_resolution(
        investigation=inv, root_causes=root_causes, similar_investigations=[match], matched_component=None,
        technology=None, recommended_solution=solution, sources=sources, provenance=provenance,
    )

    assert result is not None
    assert len(result.resolution_candidates) == 1
    assert result.resolution_candidates[0].is_primary is True
    assert result.confidence == ResolutionProvenance.LIKELY
    assert result.root_cause == record.root_cause


# --- 2/6. Multiple sources, conflict preserved, not silently hidden -----


def test_multiple_sources_are_all_represented_not_silently_dropped(bundle):
    """Reproduces the real finding from this phase's inspection: Local
    and TFS both have real resolution content. Today's
    _synthesize_recommendation lets TFS unconditionally overwrite
    Local's text in RecommendedSolution.recommended_resolution -- this
    test proves the structured result still carries BOTH, correctly
    attributed, with is_primary accurately reflecting which one won."""
    record = _save_hi(bundle["knowledge_repo"], resolution="Restart CommandProcessorHost after confirming queue depth.")
    match = _match_for(record, 0.8, tags="ticket:CS999999")
    root_causes = [RootCauseHypothesis(description=record.root_cause, confidence=0.8, rationale=f"Matches historical investigation '{record.title}' (80% similarity).")]
    case = _tfs_case(resolution="Do NOT restart the service -- apply configuration change Y instead.", crm_id="CS999999")
    engine = bundle["rec_engine"]

    solution, sources = engine._synthesize_recommendation(root_causes, [match], _tfs_result(matches=[(case, 0.9)]), None, [])
    provenance = engine._build_provenance_record(
        root_causes=root_causes, similar_investigations=[match], recommended_logs=[], suggested_sql=[],
        recommended_solution=solution, sources=sources,
    )
    from app.domain.investigation import InvestigationSession
    inv = InvestigationSession(title=record.title)
    result = engine._build_structured_resolution(
        investigation=inv, root_causes=root_causes, similar_investigations=[match], matched_component=None,
        technology=None, recommended_solution=solution, sources=sources, provenance=provenance,
    )

    assert result is not None
    assert len(result.resolution_candidates) == 2
    texts = {c.text for c in result.resolution_candidates}
    assert any("Restart CommandProcessorHost" in t for t in texts)
    assert any("Do NOT restart the service" in t for t in texts)
    # TFS is what _synthesize_recommendation's existing (unchanged)
    # logic actually picked -- the structured result must say so
    # truthfully, not invent an independent ranking.
    primary = [c for c in result.resolution_candidates if c.is_primary]
    assert len(primary) == 1
    assert "Do NOT restart the service" in primary[0].text
    assert solution.recommended_resolution == primary[0].text


# --- 3. CONFIRMED via cross-source correlation ---------------------------


def test_confirmed_via_cross_source_correlation(bundle):
    record = _save_hi(bundle["knowledge_repo"])
    match = _match_for(record, 0.8, tags="ticket:CSTASK0087353")
    root_causes = [RootCauseHypothesis(description=record.root_cause, confidence=0.8, rationale=f"Matches historical investigation '{record.title}' (80% similarity).")]
    case = _tfs_case(resolution="Reissued GEI, confirmed fixed", crm_id="CS0122697/CSTASK0087353")
    engine = bundle["rec_engine"]

    solution, sources = engine._synthesize_recommendation(root_causes, [match], _tfs_result(matches=[(case, 0.9)]), None, [])
    provenance = engine._build_provenance_record(
        root_causes=root_causes, similar_investigations=[match], recommended_logs=[], suggested_sql=[],
        recommended_solution=solution, sources=sources,
    )
    from app.domain.investigation import InvestigationSession
    inv = InvestigationSession(title=record.title)
    result = engine._build_structured_resolution(
        investigation=inv, root_causes=root_causes, similar_investigations=[match], matched_component=None,
        technology=None, recommended_solution=solution, sources=sources, provenance=provenance,
    )

    assert result.confidence == ResolutionProvenance.CONFIRMED
    assert "Cross-source correlation" in result.confidence_rationale


def test_confirmed_remains_confirmed_with_an_explicitly_contradictory_alternate(bundle):
    """Approved design decision (2026-08-14, closing Phase 1's one open
    question): conflicting resolution evidence must never trigger an
    automatic Confirmed -> downgrade, and must never be silently
    discarded -- Phase 1 does not attempt semantic contradiction
    detection (no keyword/heuristic logic decides "these two texts
    disagree"); it only ever guarantees every real candidate stays
    attributed and visible. Reproduces the literal example from the
    original brief: Source A says "restart the service", Source B says
    "do NOT restart -- apply a configuration change instead." Both are
    real; the Confirmed tier came from cross-source correlation only,
    independent of B's existence."""
    record = _save_hi(bundle["knowledge_repo"], resolution="Restart CommandProcessorHost immediately to clear the backlog.")
    match = _match_for(record, 0.8, tags="ticket:CSTASK0087353")
    root_causes = [RootCauseHypothesis(description=record.root_cause, confidence=0.8, rationale=f"Matches historical investigation '{record.title}' (80% similarity).")]
    case = _tfs_case(resolution="Reissued GEI, confirmed fixed", crm_id="CS0122697/CSTASK0087353")
    bug_match = KnowledgeMatch(
        collection=KnowledgeCollection.KNOWN_BUGS, record_id="bug-conflict", title="Command queue backup known issue",
        snippet="x",
        score=0.6,
        metadata={"workaround": "Do NOT restart CommandProcessorHost -- apply configuration change Y instead; restarting has been observed to make the backlog worse."},
    )
    engine = bundle["rec_engine"]

    solution, sources = engine._synthesize_recommendation(
        root_causes, [match], _tfs_result(matches=[(case, 0.9)]), None, [], known_bugs=[bug_match],
        investigation_context="Command queue backup, restart CommandProcessorHost, reissue GEI, configuration change Y.",
    )
    provenance = engine._build_provenance_record(
        root_causes=root_causes, similar_investigations=[match], recommended_logs=[], suggested_sql=[],
        recommended_solution=solution, sources=sources,
    )
    from app.domain.investigation import InvestigationSession
    inv = InvestigationSession(title=record.title)
    result = engine._build_structured_resolution(
        investigation=inv, root_causes=root_causes, similar_investigations=[match], matched_component=None,
        technology=None, recommended_solution=solution, sources=sources, provenance=provenance,
    )

    # 1. CONFIRMED + conflicting alternate remains CONFIRMED.
    assert result.confidence == ResolutionProvenance.CONFIRMED
    assert "Cross-source correlation" in result.confidence_rationale
    # The rationale names only the two correlated sources (TFS + the
    # correlated local match) -- proving the tier decision itself never
    # looked at the Known Bug candidate at all, let alone downgraded
    # because of it.
    assert "known bug" not in result.confidence_rationale.lower()

    # 2. The alternate evidence remains attributed and visible -- real
    # kind, real source_id, real score, not a placeholder.
    conflicting = next(c for c in result.resolution_candidates if c.evidence.kind == EvidenceKind.KNOWN_BUG)
    assert conflicting.evidence.source_id == "bug-conflict"
    assert conflicting.evidence.score == 0.6
    assert "Do NOT restart" in conflicting.text

    # 3. No source silently discarded because it disagrees with the
    # primary -- all 3 real sources (local, TFS-primary, known bug) are
    # present, explicitly contradictory language and all.
    assert len(result.resolution_candidates) == 3
    primary = next(c for c in result.resolution_candidates if c.is_primary)
    assert "Reissued GEI" in primary.text
    assert primary.text != conflicting.text  # genuinely different, both kept

    # 4. No new threshold was introduced -- this is the exact same
    # CONFIRMED rule as every other cross-source-correlation test.
    assert solution.source_known_bug is True  # the conflicting source DID contribute/qualify...
    assert result.confidence == ResolutionProvenance.CONFIRMED  # ...and still didn't move the tier


def test_precedence_local_beats_wiki_beats_known_bug_when_no_tfs(bundle):
    """Explicit precedence demonstration (all real, non-TFS sources at
    once): with no TFS match, Local (correlated) wins over Wiki, which
    in turn is a real, distinct alternate alongside Known Bug -- the
    exact fixed order _synthesize_recommendation's own code evaluates
    sources in, now visible as ordered, attributed candidates rather
    than silently collapsed to one string."""
    from app.domain.external_knowledge import ExternalMatch, ExternalSource, WikiPage

    record = _save_hi(bundle["knowledge_repo"], resolution="Local resolution text.")
    match = _match_for(record, 0.8)
    root_causes = [RootCauseHypothesis(description=record.root_cause, confidence=0.8, rationale=f"Matches historical investigation '{record.title}' (80% similarity).")]
    wiki_result = ExternalKnowledgeResult(
        source=ExternalSource.WIKI, available=True,
        matches=[ExternalMatch(source=ExternalSource.WIKI, wiki_page=WikiPage(page_id="1", title="Wiki guidance", space_key="CC", excerpt="Wiki resolution text.", url="https://wiki/1"), score=0.6, confidence="Medium", match_reasons=["x"])],
    )
    bug_match = KnowledgeMatch(
        collection=KnowledgeCollection.KNOWN_BUGS, record_id="bug-y", title="Some known bug",
        snippet="x", score=0.5, metadata={"workaround": "Known bug workaround text."},
    )
    engine = bundle["rec_engine"]

    solution, sources = engine._synthesize_recommendation(
        root_causes, [match], _tfs_result(matches=[]), wiki_result, [], known_bugs=[bug_match],
        investigation_context=f"{record.title} Some known bug workaround",
    )
    provenance = engine._build_provenance_record(
        root_causes=root_causes, similar_investigations=[match], recommended_logs=[], suggested_sql=[],
        recommended_solution=solution, sources=sources,
    )
    from app.domain.investigation import InvestigationSession
    inv = InvestigationSession(title=record.title)
    result = engine._build_structured_resolution(
        investigation=inv, root_causes=root_causes, similar_investigations=[match], matched_component=None,
        technology=None, recommended_solution=solution, sources=sources, provenance=provenance,
    )

    assert len(result.resolution_candidates) == 3
    assert result.resolution_candidates[0].is_primary is True
    assert "Local resolution text." in result.resolution_candidates[0].text
    assert result.resolution_candidates[0].evidence.kind == EvidenceKind.HISTORICAL_INVESTIGATION
    remaining_kinds = {c.evidence.kind for c in result.resolution_candidates[1:]}
    assert remaining_kinds == {EvidenceKind.WIKI_PAGE, EvidenceKind.KNOWN_BUG}


# --- 4. CONFIRMED via human verification ---------------------------------


def test_confirmed_via_human_verification_investigation_scoped(bundle):
    record = _save_hi(bundle["knowledge_repo"])
    bundle["knowledge_repo"].save_historical_investigation(
        record.model_copy(update={
            "resolution_verified": True, "resolution_verified_by": "jsmith",
            "resolution_verified_at": datetime(2026, 8, 1, tzinfo=timezone.utc),
            "resolution_verification_note": "Confirmed with the customer.",
        })
    )
    match = _match_for(record, 0.8)
    root_causes = [RootCauseHypothesis(description=record.root_cause, confidence=0.8, rationale=f"Matches historical investigation '{record.title}' (80% similarity).")]
    engine = bundle["rec_engine"]

    solution, sources = engine._synthesize_recommendation(root_causes, [match], _tfs_result(matches=[]), None, [])
    provenance = engine._build_provenance_record(
        root_causes=root_causes, similar_investigations=[match], recommended_logs=[], suggested_sql=[],
        recommended_solution=solution, sources=sources,
    )
    from app.domain.investigation import InvestigationSession
    inv = InvestigationSession(title=record.title)
    result = engine._build_structured_resolution(
        investigation=inv, root_causes=root_causes, similar_investigations=[match], matched_component=None,
        technology=None, recommended_solution=solution, sources=sources, provenance=provenance,
    )

    assert result.confidence == ResolutionProvenance.CONFIRMED
    assert "jsmith" in result.confidence_rationale


def test_confirmed_via_human_verification_standalone(bundle):
    record = _save_hi(bundle["knowledge_repo"])
    bundle["knowledge_repo"].save_historical_investigation(
        record.model_copy(update={
            "resolution_verified": True, "resolution_verified_by": "admin",
            "resolution_verified_at": datetime(2026, 8, 1, tzinfo=timezone.utc),
        })
    )
    result = bundle["structured_engine"].from_historical_investigation(record.id)
    assert result.confidence == ResolutionProvenance.CONFIRMED
    assert "admin" in result.confidence_rationale


# --- 5/13. High similarity without evidence stays below CONFIRMED; UNKNOWN possible ---


def test_high_similarity_without_evidence_never_confirmed(bundle):
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS, record_id="probe", title="Extremely similar but undocumented",
        snippet="x", score=0.97, metadata={},
    )
    engine = bundle["rec_engine"]
    solution, sources = engine._synthesize_recommendation([], [match], _tfs_result(matches=[]), None, [])
    provenance = engine._build_provenance_record(
        root_causes=[], similar_investigations=[match], recommended_logs=[], suggested_sql=[],
        recommended_solution=solution, sources=sources,
    )
    from app.domain.investigation import InvestigationSession
    inv = InvestigationSession(title="probe")
    result = engine._build_structured_resolution(
        investigation=inv, root_causes=[], similar_investigations=[match], matched_component=None,
        technology=None, recommended_solution=solution, sources=sources, provenance=provenance,
    )
    assert result.confidence != ResolutionProvenance.CONFIRMED


def test_unknown_when_evidence_insufficient(bundle):
    engine = bundle["rec_engine"]
    solution, sources = engine._synthesize_recommendation([], [], _tfs_result(matches=[]), None, [])
    provenance = engine._build_provenance_record(
        root_causes=[], similar_investigations=[], recommended_logs=[], suggested_sql=[],
        recommended_solution=solution, sources=sources,
    )
    from app.domain.investigation import InvestigationSession
    inv = InvestigationSession(title="Nothing yet")
    result = engine._build_structured_resolution(
        investigation=inv, root_causes=[], similar_investigations=[], matched_component=None,
        technology=None, recommended_solution=solution, sources=sources, provenance=provenance,
    )
    assert result is not None  # still a real object, never None just because evidence is thin
    assert result.confidence == ResolutionProvenance.UNKNOWN
    assert result.resolution_candidates == []
    assert result.root_cause is None


# --- 7. Conflict ordering is deterministic --------------------------------


def test_order_resolution_candidates_is_deterministic():
    from app.domain.provenance import EvidenceReference

    a = ResolutionCandidate(text="A", evidence=EvidenceReference(kind=EvidenceKind.HISTORICAL_INVESTIGATION, source_id="1", title="A", score=0.5, reason="x", contributes_to=["resolution"]), is_primary=False)
    b = ResolutionCandidate(text="B", evidence=EvidenceReference(kind=EvidenceKind.TFS_CASE, source_id="2", title="B", score=0.9, reason="x", contributes_to=["resolution"]), is_primary=True)
    c = ResolutionCandidate(text="C", evidence=EvidenceReference(kind=EvidenceKind.WIKI_PAGE, source_id="3", title="C", score=0.9, reason="x", contributes_to=["resolution"]), is_primary=False)

    result1 = order_resolution_candidates([a, b, c])
    result2 = order_resolution_candidates([c, a, b])  # different input order

    assert [r.text for r in result1] == [r.text for r in result2]
    assert result1[0].text == "B"  # is_primary always sorts first
    # Among non-primary, real score breaks the tie deterministically by
    # the fixed source priority (TFS > Wiki here would not apply since
    # b is primary and already placed -- among a/c, c's score (0.9) beats a's (0.5))
    assert result1[1].text == "C"
    assert result1[2].text == "A"


def test_order_resolutions_supersedes_beats_tier(bundle):
    from app.domain.knowledge_relationships import KnowledgeObjectRef

    older = StructuredResolution(
        source_kind="known_bug", source_id="bug-old", problem="X", symptoms="x",
        confidence=ResolutionProvenance.CONFIRMED, updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    newer = StructuredResolution(
        source_kind="known_bug", source_id="bug-new", problem="X (updated)", symptoms="x",
        confidence=ResolutionProvenance.POSSIBLE, updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        superseded_by=None,
    )
    # older is explicitly superseded by newer, even though older's tier is higher
    older = older.model_copy(update={"superseded_by": KnowledgeObjectRef(type=KnowledgeObjectType.KNOWN_BUG, id="bug-new", title="X (updated)")})

    result = order_resolutions([older, newer])
    assert result.source_id == "bug-new"
    assert len(result.also_seen) == 1
    assert result.also_seen[0].source_id == "bug-old"


def test_order_resolutions_tier_breaks_tie_absent_supersedes():
    likely = StructuredResolution(source_kind="known_bug", source_id="b1", problem="X", symptoms="x", confidence=ResolutionProvenance.LIKELY)
    possible = StructuredResolution(source_kind="known_bug", source_id="b2", problem="X", symptoms="x", confidence=ResolutionProvenance.POSSIBLE)
    result = order_resolutions([possible, likely])
    assert result.source_id == "b1"
    assert result.also_seen[0].source_id == "b2"


def test_order_resolutions_empty_returns_none():
    assert order_resolutions([]) is None


# --- 8/9. ValidationStep: represented correctly, never "already done" ---


def test_validation_step_populated_from_next_step(bundle):
    record = _save_hi(bundle["knowledge_repo"])
    match = _match_for(record, 0.8)
    root_causes = [RootCauseHypothesis(description=record.root_cause, confidence=0.8, rationale=f"Matches historical investigation '{record.title}' (80% similarity).")]
    engine = bundle["rec_engine"]
    solution, sources = engine._synthesize_recommendation(root_causes, [match], _tfs_result(matches=[]), None, [])

    assert len(solution.validation_steps) == 1
    step = solution.validation_steps[0]
    assert step.instruction == record.next_step
    assert step.source == EvidenceKind.HISTORICAL_INVESTIGATION
    assert step.source_id == record.id


def test_validation_step_never_claims_completion():
    """Structural guarantee: ValidationStep has no field that could ever
    assert a check was already performed -- the only fields are
    instruction/expected_result/source attribution."""
    field_names = set(ValidationStep.model_fields.keys())
    for forbidden in ("completed", "executed", "verified", "done", "passed", "result_status"):
        assert forbidden not in field_names
    assert field_names == {"instruction", "expected_result", "disproving_result", "source", "source_id", "source_title"}


def test_known_bug_has_no_validation_steps(bundle):
    """Real, documented Phase 1 limitation: KnownBugRecord has no
    next_step field, so Known-Bug-sourced validation_steps are always
    empty -- never fabricated."""
    bug = _save_bug(bundle["knowledge_repo"])
    result = bundle["structured_engine"].from_known_bug(bug.id)
    assert result.validation_steps == []


# --- 10/11. Applicability: survives, and customer never becomes component/technology ---


def test_applicability_survives_into_structured_result(bundle):
    customer = Customer(id=str(uuid.uuid4()), name="TEPCO")
    bundle["lookup_repo"].save_customer(customer)
    record = _save_hi(bundle["knowledge_repo"], title="TEPCO Meter Communication Failure")
    bundle["relationship_engine"].add_relationship(
        KnowledgeObjectType.HISTORICAL_INVESTIGATION, record.id, KnowledgeObjectType.CUSTOMER, customer.id, RelationshipType.APPLIES_TO
    )

    result = bundle["structured_engine"].from_historical_investigation(record.id)

    assert "TEPCO" in result.applicability.customer_names
    assert "TEPCO" not in result.applicability.component_names
    assert "TEPCO" not in (result.applicability.technology_name or "")


def test_customer_never_becomes_technology_or_component(bundle):
    """Structural guarantee, unchanged by this phase: applicability_for_object
    only ever reads APPLIES_TO edges to CUSTOMER/REGION typed objects --
    there is no code path here that could populate component_names or
    technology_name from a Customer relationship."""
    customer = Customer(id=str(uuid.uuid4()), name="CLECO")
    bundle["lookup_repo"].save_customer(customer)
    bug = _save_bug(bundle["knowledge_repo"], title="CLECO Meter Program Change Failure")
    bundle["relationship_engine"].add_relationship(
        KnowledgeObjectType.KNOWN_BUG, bug.id, KnowledgeObjectType.CUSTOMER, customer.id, RelationshipType.APPLIES_TO
    )

    result = bundle["structured_engine"].from_known_bug(bug.id)

    assert result.applicability.customer_names == ["CLECO"]
    assert result.applicability.component_names == []
    assert result.applicability.technology_name is None


def test_region_applicability_also_survives(bundle):
    region = Region(id=str(uuid.uuid4()), name="Guam")
    bundle["lookup_repo"].save_region(region)
    record = _save_hi(bundle["knowledge_repo"], title="Guam Power Authority Reads Dropped")
    bundle["relationship_engine"].add_relationship(
        KnowledgeObjectType.HISTORICAL_INVESTIGATION, record.id, KnowledgeObjectType.REGION, region.id, RelationshipType.APPLIES_TO
    )
    result = bundle["structured_engine"].from_historical_investigation(record.id)
    assert result.applicability.region_names == ["Guam"]


# --- 12. RF Mesh / RF Mesh IP hierarchy remains correct (regression) ----


def test_rf_mesh_ip_technology_applicability_end_to_end(bundle):
    from app.domain.lookup_entities import Technology

    lookup_repo = bundle["lookup_repo"]
    rf_mesh = Technology(id=str(uuid.uuid4()), name="RF Mesh")
    lookup_repo.save_technology(rf_mesh)
    rf_mesh_ip = Technology(id=str(uuid.uuid4()), name="RF Mesh IP", parent_technology_id=rf_mesh.id)
    lookup_repo.save_technology(rf_mesh_ip)

    engine = bundle["rec_engine"]
    matched = engine._match_single_technology("RF Mesh IP communication failure between collector and endpoint")
    assert matched == "RF Mesh IP"  # unchanged by this phase

    from app.domain.investigation import InvestigationSession
    inv = InvestigationSession(title="RF Mesh IP communication failure")
    solution, sources = engine._synthesize_recommendation([], [], _tfs_result(matches=[]), None, [])
    provenance = engine._build_provenance_record(root_causes=[], similar_investigations=[], recommended_logs=[], suggested_sql=[], recommended_solution=solution, sources=sources)
    result = engine._build_structured_resolution(
        investigation=inv, root_causes=[], similar_investigations=[], matched_component=None,
        technology=matched, recommended_solution=solution, sources=sources, provenance=provenance,
    )
    assert result.applicability.technology_name == "RF Mesh IP"


# --- Known Bug topical-overlap protection unaffected (Phase 0 regression) ---


def test_known_bug_topical_overlap_protection_unaffected(bundle):
    bug_match = KnowledgeMatch(
        collection=KnowledgeCollection.KNOWN_BUGS, record_id="bug-x", title="IIS worker process crash on multipart uploads over 50MB",
        snippet="x", score=0.56, metadata={"workaround": "Enforce a 50MB client-side upload limit."},
    )
    engine = bundle["rec_engine"]
    solution, sources = engine._synthesize_recommendation(
        [], [], _tfs_result(matches=[]), None, [], known_bugs=[bug_match],
        investigation_context="Investigation into network connectivity between two application servers.",
    )
    assert sources.best_known_bug is None
    assert solution.source_known_bug is False


# --- Standalone-record confidence tier rule (POSSIBLE, placeholder handling) ---


def test_standalone_hi_with_only_description_is_possible(bundle):
    record = _save_hi(bundle["knowledge_repo"], root_cause="-", resolution="-", next_step="")
    result = bundle["structured_engine"].from_historical_investigation(record.id)
    assert result.confidence == ResolutionProvenance.POSSIBLE
    assert result.root_cause is None  # placeholder "-" never surfaced as real content
    assert result.resolution_candidates == []


def test_standalone_known_bug_without_workaround_is_possible(bundle):
    bug = _save_bug(bundle["knowledge_repo"], workaround=None)
    result = bundle["structured_engine"].from_known_bug(bug.id)
    assert result.confidence == ResolutionProvenance.POSSIBLE


def test_from_record_returns_none_for_unknown_id(bundle):
    assert bundle["structured_engine"].from_historical_investigation("nonexistent") is None
    assert bundle["structured_engine"].from_known_bug("nonexistent") is None


# --- 14/15: existing Phase 0 behavior unchanged -- see the full suite run
# (tests/test_recommendation_engine.py, tests/test_provenance.py, all
# other Phase 0 files) reported alongside this file's own results,
# rather than re-asserted redundantly here.
