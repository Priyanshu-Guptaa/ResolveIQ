"""Tests for Resolution Provenance (approved design, 2026-08-13):
- the four ResolutionProvenance tiers, with the Confirmed rule
  specifically proven to never come from a similarity score alone;
- the two closed gaps (KnowledgeMatch.reason, SuggestedSqlItem.match_reason);
- the resolution_verified* field round-trip through the real repository.

Uses fakes for the engine-level tests (same discipline as
test_recommendation_engine.py's FakeKnowledgeStore) so the tier logic
is tested fully offline; the repository round-trip test is
integration-style against a real temp SQLite database (same pattern as
test_context_dimensions.py), since that's genuinely persistence
behavior, not pure logic.
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
from app.domain.provenance import EvidenceKind, ResolutionProvenance
from app.domain.recommendation import KnowledgeMatch, RootCauseHypothesis, SuggestedSqlItem
from app.engines.knowledge.engine import KnowledgeEngine
from app.engines.recommendation.engine import RecommendationEngine


class FakeKnowledgeStore:
    """Structurally satisfies the KnowledgeStore Protocol -- same fake
    used throughout test_recommendation_engine.py."""

    def __init__(self, canned=None) -> None:
        self._canned = canned or {}

    def upsert(self, collection, record_id, text, title, metadata) -> None:  # pragma: no cover
        pass

    def query(self, collection, text, top_k: int = 5):
        return self._canned.get(collection, [])[:top_k]

    def count(self, collection) -> int:
        return len(self._canned.get(collection, []))

    def list_recent(self, collection, limit: int = 5):  # pragma: no cover
        return []

    def delete(self, collection, record_id: str) -> None:  # pragma: no cover
        pass


class FakeVerificationRepo:
    """Structurally satisfies the parts of KnowledgeRepository
    _verified_local_record/_verified_known_bug_record use
    (KnowledgeEngine.get_historical_investigation/get_known_bug delegate
    to these two methods)."""

    def __init__(
        self,
        records: dict[str, HistoricalInvestigationRecord] | None = None,
        known_bugs: dict[str, KnownBugRecord] | None = None,
    ) -> None:
        self._records = records or {}
        self._known_bugs = known_bugs or {}

    def get_historical_investigation(self, record_id: str):
        return self._records.get(record_id)

    def get_known_bug(self, bug_id: str):
        return self._known_bugs.get(bug_id)


def _tfs_case(tfs_id=1, title="CommandProcessorHost issue", state="Closed", resolution="Restart the service", crm_id=None):
    return TfsCase(
        tfs_id=tfs_id,
        work_item_type="Bug",
        title=title,
        state=state,
        area_path="Command Center",
        team_project="Command Center",
        changed_date=datetime(2026, 1, 1),
        description_text="x",
        resolution_text=resolution,
        crm_id=crm_id,
        url=f"https://tfs/edit/{tfs_id}",
    )


def _tfs_result(matches=None, available=True, confidence="Medium"):
    if matches is None:
        return ExternalKnowledgeResult(source=ExternalSource.TFS, available=available, matches=[])
    external_matches = [
        ExternalMatch(source=ExternalSource.TFS, tfs_case=case, score=score, confidence=confidence, match_reasons=["Same component: X"])
        for case, score in matches
    ]
    return ExternalKnowledgeResult(source=ExternalSource.TFS, available=available, matches=external_matches)


def _wiki_result(matches=None, available=True):
    if matches is None:
        return ExternalKnowledgeResult(source=ExternalSource.WIKI, available=available, matches=[])
    external_matches = [
        ExternalMatch(
            source=ExternalSource.WIKI,
            wiki_page=WikiPage(page_id="1", title=title, space_key="CC", excerpt=excerpt, url="https://wiki/1"),
            score=score,
            confidence="Medium",
            match_reasons=["x"],
        )
        for title, excerpt, score in matches
    ]
    return ExternalKnowledgeResult(source=ExternalSource.WIKI, available=available, matches=external_matches)


def _engine(repository=None) -> RecommendationEngine:
    return RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore(), repository), Settings())


# --- The four tiers, per approved item 10 -----------------------------


def test_confirmed_never_comes_from_similarity_score_alone():
    """The central rule: a near-perfect similarity score with NO
    recorded root cause/resolution on file must never reach CONFIRMED
    -- it must not even reach LIKELY, since there's no real resolution
    content to trust. (It still legitimately reaches POSSIBLE: a real
    local match exists, just without a recorded root cause -- see
    test_title_only_local_match_is_possible for that rule in
    isolation.)"""
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hi-1",
        title="Extremely similar but undocumented case",
        snippet="x",
        score=0.97,
        metadata={},  # no root_cause, no resolution -- pure similarity
    )
    engine = _engine()
    solution, sources = engine._synthesize_recommendation([], [match], _tfs_result(matches=[]), _wiki_result(matches=[]), [])

    tier, rationale = engine._resolve_provenance_tier(
        recommended_solution=solution, sources=sources, root_causes=[], similar_investigations=[match]
    )

    assert tier != ResolutionProvenance.CONFIRMED
    assert tier != ResolutionProvenance.LIKELY
    assert tier == ResolutionProvenance.POSSIBLE


def test_cross_source_correlation_is_confirmed():
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hi-1",
        title="ATCO - RF Mesh IP - similar case",
        snippet="x",
        score=0.5,
        metadata={"tags": "ticket:CSTASK0087353, priority:High", "root_cause": "Stale DCW", "resolution": "Reissued GEI"},
    )
    root_causes = [
        RootCauseHypothesis(
            description="Stale DCW",
            confidence=0.5,
            rationale="Matches historical investigation 'ATCO - RF Mesh IP - similar case' (50% similarity).",
        )
    ]
    case = _tfs_case(resolution="Reissued GEI, confirmed fixed", crm_id="CS0122697/CSTASK0087353")
    engine = _engine()
    solution, sources = engine._synthesize_recommendation(
        root_causes, [match], _tfs_result(matches=[(case, 0.6)]), _wiki_result(matches=[]), []
    )

    tier, rationale = engine._resolve_provenance_tier(
        recommended_solution=solution, sources=sources, root_causes=root_causes, similar_investigations=[match]
    )

    assert tier == ResolutionProvenance.CONFIRMED
    assert "Cross-source correlation" in rationale
    assert "TFS-1" in rationale


def test_human_verification_is_confirmed():
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hi-verified",
        title="Verified past case",
        snippet="x",
        score=0.6,
        metadata={"root_cause": "Firmware mismatch", "resolution": "Reissued firmware"},
    )
    root_causes = [
        RootCauseHypothesis(
            description="Firmware mismatch",
            confidence=0.6,
            rationale="Matches historical investigation 'Verified past case' (60% similarity).",
        )
    ]
    verified_record = HistoricalInvestigationRecord(
        id="hi-verified",
        title="Verified past case",
        description="A meter firmware mismatch case.",
        root_cause="Firmware mismatch",
        resolution="Reissued firmware",
        resolution_verified=True,
        resolution_verified_by="jsmith",
        resolution_verified_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        resolution_verification_note="Confirmed with the customer after applying the fix; issue did not recur.",
    )
    engine = _engine(FakeVerificationRepo({"hi-verified": verified_record}))
    solution, sources = engine._synthesize_recommendation(root_causes, [match], _tfs_result(matches=[]), _wiki_result(matches=[]), [])

    tier, rationale = engine._resolve_provenance_tier(
        recommended_solution=solution, sources=sources, root_causes=root_causes, similar_investigations=[match]
    )

    assert tier == ResolutionProvenance.CONFIRMED
    assert "jsmith" in rationale
    assert "Confirmed with the customer" in rationale


def test_unverified_record_never_reaches_confirmed_via_verification_path():
    """The mirror of the previous test: a record that exists but has
    resolution_verified=False (the default) must not be treated as
    verified just because a repository lookup succeeded."""
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hi-unverified",
        title="Unverified past case",
        snippet="x",
        score=0.6,
        metadata={"root_cause": "Firmware mismatch", "resolution": "Reissued firmware"},
    )
    root_causes = [
        RootCauseHypothesis(
            description="Firmware mismatch",
            confidence=0.6,
            rationale="Matches historical investigation 'Unverified past case' (60% similarity).",
        )
    ]
    unverified_record = HistoricalInvestigationRecord(
        id="hi-unverified", title="Unverified past case", description="x", root_cause="Firmware mismatch", resolution="Reissued firmware"
    )
    engine = _engine(FakeVerificationRepo({"hi-unverified": unverified_record}))
    solution, sources = engine._synthesize_recommendation(root_causes, [match], _tfs_result(matches=[]), _wiki_result(matches=[]), [])

    tier, _rationale = engine._resolve_provenance_tier(
        recommended_solution=solution, sources=sources, root_causes=root_causes, similar_investigations=[match]
    )

    assert tier != ResolutionProvenance.CONFIRMED
    assert tier == ResolutionProvenance.LIKELY  # strong local match, recorded root cause, no corroboration


def test_strong_resolved_local_match_without_corroboration_is_likely():
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hi-1",
        title="Similar resolved case",
        snippet="x",
        score=0.8,
        metadata={"root_cause": "Stale DCW version", "resolution": "Reissued GEI"},
    )
    root_causes = [
        RootCauseHypothesis(
            description="Stale DCW version", confidence=0.8, rationale="Matches historical investigation 'Similar resolved case' (80% similarity)."
        )
    ]
    engine = _engine()
    solution, sources = engine._synthesize_recommendation(root_causes, [match], _tfs_result(matches=[]), _wiki_result(matches=[]), [])

    tier, rationale = engine._resolve_provenance_tier(
        recommended_solution=solution, sources=sources, root_causes=root_causes, similar_investigations=[match]
    )

    assert tier == ResolutionProvenance.LIKELY
    assert "no independent corroborating source" in rationale


def test_high_confidence_tfs_match_without_corroboration_is_likely():
    case = _tfs_case(resolution="Restart CommandProcessorHost and verify init messages")
    engine = _engine()
    solution, sources = engine._synthesize_recommendation([], [], _tfs_result(matches=[(case, 0.9)], confidence="High"), _wiki_result(matches=[]), [])

    tier, _rationale = engine._resolve_provenance_tier(
        recommended_solution=solution, sources=sources, root_causes=[], similar_investigations=[]
    )

    assert tier == ResolutionProvenance.LIKELY


def test_title_only_local_match_is_possible():
    """Weak/title-only evidence (no recorded root cause) -- a real
    local match contributed, but not enough for LIKELY."""
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hi-1",
        title="Similar-looking but undocumented case",
        snippet="x",
        score=0.5,
        metadata={},
    )
    engine = _engine()
    solution, sources = engine._synthesize_recommendation([], [match], _tfs_result(matches=[]), _wiki_result(matches=[]), [])

    tier, rationale = engine._resolve_provenance_tier(
        recommended_solution=solution, sources=sources, root_causes=[], similar_investigations=[match]
    )

    assert tier == ResolutionProvenance.POSSIBLE
    assert "hypothesis to verify" in rationale


def test_low_confidence_tfs_match_with_resolution_is_possible():
    case = _tfs_case(resolution="Restart the service")
    engine = _engine()
    solution, sources = engine._synthesize_recommendation([], [], _tfs_result(matches=[(case, 0.45)], confidence="Medium"), _wiki_result(matches=[]), [])

    tier, _rationale = engine._resolve_provenance_tier(
        recommended_solution=solution, sources=sources, root_causes=[], similar_investigations=[]
    )

    assert tier == ResolutionProvenance.POSSIBLE


def test_no_evidence_at_all_is_unknown():
    engine = _engine()
    solution, sources = engine._synthesize_recommendation([], [], _tfs_result(matches=[]), _wiki_result(matches=[]), [])

    tier, rationale = engine._resolve_provenance_tier(
        recommended_solution=solution, sources=sources, root_causes=[], similar_investigations=[]
    )

    assert tier == ResolutionProvenance.UNKNOWN
    assert rationale  # never blank, even for UNKNOWN


def test_provenance_record_identifies_every_underlying_source(monkeypatch=None):
    """Approved item 9: every provenance record must identify its
    underlying source(s) -- proves EvidenceReference.source_id/kind are
    populated for root cause, resolution, logs, and SQL all at once."""
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hi-1",
        title="Similar resolved case",
        snippet="x",
        score=0.8,
        metadata={"root_cause": "Stale DCW version", "resolution": "Reissued GEI"},
    )
    root_causes = [
        RootCauseHypothesis(
            description="Stale DCW version", confidence=0.8, rationale="Matches historical investigation 'Similar resolved case' (80% similarity)."
        )
    ]
    sql_items = [
        SuggestedSqlItem(
            title="Command log lookup",
            sql_text="SELECT 1;",
            explanation="static description",
            source="sql_library",
            template_id="qt-1",
            match_reason='Linked to the matched component "X" via this SQL template\'s related_components.',
        )
    ]
    engine = _engine()
    solution, sources = engine._synthesize_recommendation(root_causes, [match], _tfs_result(matches=[]), _wiki_result(matches=[]), [])

    record = engine._build_provenance_record(
        root_causes=root_causes,
        similar_investigations=[match],
        recommended_logs=[],
        suggested_sql=sql_items,
        recommended_solution=solution,
        sources=sources,
    )

    assert record.root_cause_evidence[0].kind == EvidenceKind.HISTORICAL_INVESTIGATION
    assert record.root_cause_evidence[0].source_id == "hi-1"
    assert record.resolution_evidence[0].source_id == "hi-1"
    assert record.sql_recommendation_evidence[0].kind == EvidenceKind.SQL_TEMPLATE
    assert record.sql_recommendation_evidence[0].source_id == "qt-1"
    assert record.sql_recommendation_evidence[0].reason == sql_items[0].match_reason
    assert record.resolution_provenance == ResolutionProvenance.LIKELY


# --- Known Bugs as a resolution source (2026-08-13, Phase 0 --
# Chat/Structured Resolution Knowledge architecture, closing the gap
# flagged in RESOLVEIQ_CHAT_AND_RESOLUTION_ARCHITECTURE.md Section 17/
# Section 6: Known Bugs were structurally present but never actually
# consulted by _synthesize_recommendation/_resolve_provenance_tier). ---


def test_known_bug_human_verification_is_confirmed():
    bug_match = KnowledgeMatch(
        collection=KnowledgeCollection.KNOWN_BUGS,
        record_id="bug-1",
        title="CommandProcessorHost stops processing after GC pause",
        snippet="x",
        score=0.6,
        metadata={"workaround": "Restart CommandProcessorHost", "status": "fixed-in-9.0.4"},
    )
    verified_bug = KnownBugRecord(
        id="bug-1",
        title="CommandProcessorHost stops processing after GC pause",
        description="x",
        workaround="Restart CommandProcessorHost",
        resolution_verified=True,
        resolution_verified_by="jsmith",
        resolution_verified_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        resolution_verification_note="Confirmed fixed after the 9.0.4 upgrade.",
    )
    engine = _engine(FakeVerificationRepo(known_bugs={"bug-1": verified_bug}))
    solution, sources = engine._synthesize_recommendation(
        [], [], _tfs_result(matches=[]), _wiki_result(matches=[]), [], known_bugs=[bug_match],
        investigation_context="CommandProcessorHost appears to stop processing commands, possibly related to a GC pause.",
    )

    tier, rationale = engine._resolve_provenance_tier(
        recommended_solution=solution, sources=sources, root_causes=[], similar_investigations=[]
    )

    assert solution.source_known_bug is True
    assert solution.supporting_known_bug_id == "bug-1"
    assert tier == ResolutionProvenance.CONFIRMED
    assert "jsmith" in rationale
    assert "known bug" in rationale


def test_known_bug_unverified_never_reaches_confirmed():
    bug_match = KnowledgeMatch(
        collection=KnowledgeCollection.KNOWN_BUGS,
        record_id="bug-2",
        title="Unverified known bug",
        snippet="x",
        score=0.6,
        metadata={"workaround": "Apply patch Y"},
    )
    unverified_bug = KnownBugRecord(id="bug-2", title="Unverified known bug", description="x", workaround="Apply patch Y")
    engine = _engine(FakeVerificationRepo(known_bugs={"bug-2": unverified_bug}))
    solution, sources = engine._synthesize_recommendation(
        [], [], _tfs_result(matches=[]), _wiki_result(matches=[]), [], known_bugs=[bug_match],
        investigation_context="This looks like a known, previously unverified issue.",
    )

    tier, _rationale = engine._resolve_provenance_tier(
        recommended_solution=solution, sources=sources, root_causes=[], similar_investigations=[]
    )

    assert tier != ResolutionProvenance.CONFIRMED
    assert tier == ResolutionProvenance.LIKELY  # real workaround, no corroboration


def test_known_bug_with_workaround_and_no_verification_is_likely():
    bug_match = KnowledgeMatch(
        collection=KnowledgeCollection.KNOWN_BUGS,
        record_id="bug-3",
        title="Command queue backs up under load",
        snippet="x",
        score=0.5,
        metadata={"workaround": "Increase queue timeout to 60s"},
    )
    engine = _engine()
    solution, sources = engine._synthesize_recommendation(
        [], [], _tfs_result(matches=[]), _wiki_result(matches=[]), [], known_bugs=[bug_match],
        investigation_context="Commands are queued but the queue keeps backing up during peak load.",
    )

    tier, rationale = engine._resolve_provenance_tier(
        recommended_solution=solution, sources=sources, root_causes=[], similar_investigations=[]
    )

    assert tier == ResolutionProvenance.LIKELY
    assert "known-bug match" in rationale
    assert solution.recommended_resolution is not None
    assert "Increase queue timeout to 60s" in solution.recommended_resolution


def test_known_bug_without_workaround_is_possible_not_likely():
    bug_match = KnowledgeMatch(
        collection=KnowledgeCollection.KNOWN_BUGS,
        record_id="bug-4",
        title="Title-only known bug match",
        snippet="x",
        score=0.5,
        metadata={},  # no workaround on file
    )
    engine = _engine()
    solution, sources = engine._synthesize_recommendation(
        [], [], _tfs_result(matches=[]), _wiki_result(matches=[]), [], known_bugs=[bug_match],
        investigation_context="This is a title-only match against a known bug record with no further detail.",
    )

    tier, _rationale = engine._resolve_provenance_tier(
        recommended_solution=solution, sources=sources, root_causes=[], similar_investigations=[]
    )

    # Genuine topical overlap is present (title/match/known all shared) --
    # this isolates that the workaround-emptiness, not the overlap gate,
    # is what caps this at POSSIBLE rather than LIKELY.
    assert tier == ResolutionProvenance.POSSIBLE


def test_known_bug_backward_compatible_when_omitted():
    """The new known_bugs parameter is additive -- calling without it
    (every pre-existing call site) must behave exactly as before."""
    engine = _engine()
    solution, sources = engine._synthesize_recommendation([], [], _tfs_result(matches=[]), _wiki_result(matches=[]), [])

    assert solution.source_known_bug is False
    assert sources.best_known_bug is None


def test_known_bug_resolution_evidence_is_recorded_in_provenance_record():
    bug_match = KnowledgeMatch(
        collection=KnowledgeCollection.KNOWN_BUGS,
        record_id="bug-5",
        title="Known issue with GLP mismatch",
        snippet="x",
        score=0.6,
        metadata={"workaround": "Reissue GEI"},
    )
    engine = _engine()
    solution, sources = engine._synthesize_recommendation(
        [], [], _tfs_result(matches=[]), _wiki_result(matches=[]), [], known_bugs=[bug_match],
        investigation_context="Investigating a known issue where the GLP configuration causes a mismatch after firmware push.",
    )

    record = engine._build_provenance_record(
        root_causes=[], similar_investigations=[], recommended_logs=[], suggested_sql=[],
        recommended_solution=solution, sources=sources,
    )

    assert len(record.resolution_evidence) == 1
    assert record.resolution_evidence[0].kind == EvidenceKind.KNOWN_BUG
    assert record.resolution_evidence[0].source_id == "bug-5"
    assert record.resolution_provenance == ResolutionProvenance.LIKELY


# --- Known Bug topical-overlap gate (2026-08-13, Phase 0 acceptance
# review, defect #1 fix): a Known Bug can no longer drive
# source_known_bug/a resolution tier from raw semantic similarity alone
# -- it must also share at least one real, distinguishing word with the
# investigation's own text. See
# RecommendationEngine._known_bug_has_topical_overlap's docstring for
# the full rationale and the real, live-demonstrated defect this closes. ---


def test_original_failing_scenario_irrelevant_known_bug_no_longer_drives_a_tier():
    """The exact defect the acceptance review found, reproduced with
    the real fictional bug's real title and the real observed score:
    'IIS worker process crash on multipart uploads over 50MB' scored
    56% similarity against a generic 'network connectivity' investigation
    with zero actual relevance, and was being quoted as the recommended
    resolution at LIKELY tier. Must no longer happen."""
    bug_match = KnowledgeMatch(
        collection=KnowledgeCollection.KNOWN_BUGS,
        record_id="bug-103",
        title="IIS worker process crash on multipart uploads over 50MB",
        snippet="x",
        score=0.56,
        metadata={"workaround": "Enforce a 50MB client-side upload limit until the AspNetCoreModuleV2 patch is applied."},
    )
    engine = _engine()
    solution, sources = engine._synthesize_recommendation(
        [], [], _tfs_result(matches=[]), _wiki_result(matches=[]), [], known_bugs=[bug_match],
        investigation_context="Investigation into network connectivity between two application servers.",
    )

    tier, rationale = engine._resolve_provenance_tier(
        recommended_solution=solution, sources=sources, root_causes=[], similar_investigations=[]
    )

    assert sources.best_known_bug is None
    assert solution.source_known_bug is False
    assert solution.supporting_known_bug_id is None
    assert solution.recommended_resolution is None
    assert solution.insufficient_evidence is True
    assert tier == ResolutionProvenance.UNKNOWN
    assert "IIS worker process crash" not in rationale


def test_common_stopword_alone_is_not_topical_overlap():
    """A second, real live-verification finding on the same defect
    (found while re-verifying the first fix, GPA scenario): a bare
    English function word ("after") shared by two otherwise unrelated
    texts must not itself count as topical overlap -- reproduced with
    the exact real titles/scores that exposed it: 'GPA Authority
    (Guam)... commands failed to respond after patching' against
    'Collector command queue stalls after mesh router firmware
    v3.4.0', which share only 'after' and nothing else."""
    bug_match = KnowledgeMatch(
        collection=KnowledgeCollection.KNOWN_BUGS,
        record_id="bug-104",
        title="Collector command queue stalls after mesh router firmware v3.4.0",
        snippet="x",
        score=0.62,
        metadata={"workaround": "Hold firmware at v3.3.x; do not roll out v3.4.0 further until vendor patch is available."},
    )
    engine = _engine()
    solution, sources = engine._synthesize_recommendation(
        [], [], _tfs_result(matches=[]), _wiki_result(matches=[]), [], known_bugs=[bug_match],
        investigation_context="GPA Authority (Guam), commands failed to respond after patching.",
    )

    assert sources.best_known_bug is None
    assert solution.source_known_bug is False
    assert solution.recommended_resolution is None


def test_corrected_behavior_topically_relevant_known_bug_still_reaches_likely():
    """The real, legitimate case from the same review round (scenario
    D): a known bug genuinely about command-queue backlog scored 70%
    against a real 'Commands stuck in Pending... queue increasing'
    investigation -- sharing the real word 'queue'. This one SHOULD
    still drive a LIKELY tier; the fix must not overcorrect into
    rejecting genuine matches."""
    bug_match = KnowledgeMatch(
        collection=KnowledgeCollection.KNOWN_BUGS,
        record_id="bug-104",
        title="Collector command queue stalls after mesh router firmware v3.4.0",
        snippet="x",
        score=0.70,
        metadata={"workaround": "Hold firmware at v3.3.x; do not roll out v3.4.0 further until vendor patch is available."},
    )
    engine = _engine()
    solution, sources = engine._synthesize_recommendation(
        [], [], _tfs_result(matches=[]), _wiki_result(matches=[]), [], known_bugs=[bug_match],
        investigation_context="Commands created successfully but no CommandResponse received, queue increasing.",
    )

    tier, rationale = engine._resolve_provenance_tier(
        recommended_solution=solution, sources=sources, root_causes=[], similar_investigations=[]
    )

    assert sources.best_known_bug is not None
    assert solution.source_known_bug is True
    assert solution.recommended_resolution is not None
    assert "Hold firmware at v3.3.x" in solution.recommended_resolution
    assert tier == ResolutionProvenance.LIKELY
    assert "known-bug match" in rationale


def test_non_regression_local_and_cross_source_confirmed_unaffected_by_an_irrelevant_known_bug():
    """Negative/non-regression case: an irrelevant Known Bug sitting
    alongside a real, strong local match + a real TFS cross-source
    correlation must not interfere with (block, downgrade, or distract
    from) that CONFIRMED result -- the fix is scoped to Known Bug
    selection only, nothing else in the tier hierarchy changes."""
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hi-1",
        title="ATCO - RF Mesh IP - similar case",
        snippet="x",
        score=0.5,
        metadata={"tags": "ticket:CSTASK0087353, priority:High", "root_cause": "Stale DCW", "resolution": "Reissued GEI"},
    )
    root_causes = [
        RootCauseHypothesis(
            description="Stale DCW", confidence=0.5,
            rationale="Matches historical investigation 'ATCO - RF Mesh IP - similar case' (50% similarity).",
        )
    ]
    case = _tfs_case(resolution="Reissued GEI, confirmed fixed", crm_id="CS0122697/CSTASK0087353")
    irrelevant_bug = KnowledgeMatch(
        collection=KnowledgeCollection.KNOWN_BUGS,
        record_id="bug-101",
        title="NullPointerException in OrderProcessor.calculateTotal for addressless customers",
        snippet="x",
        score=0.60,
        metadata={"workaround": "Have the customer add any shipping address before checkout; fix is deployed in v2.3.2."},
    )
    engine = _engine()
    solution, sources = engine._synthesize_recommendation(
        root_causes, [match], _tfs_result(matches=[(case, 0.6)]), _wiki_result(matches=[]), [],
        known_bugs=[irrelevant_bug], investigation_context="ATCO RF Mesh IP DCW mismatch investigation.",
    )

    tier, rationale = engine._resolve_provenance_tier(
        recommended_solution=solution, sources=sources, root_causes=root_causes, similar_investigations=[match]
    )

    assert sources.best_known_bug is None  # correctly excluded -- no real overlap with the ATCO/DCW text
    assert tier == ResolutionProvenance.CONFIRMED
    assert "Cross-source correlation" in rationale
    assert "OrderProcessor" not in rationale


# --- Closing the two real gaps (approved items 6/7) ---------------------


def test_historical_investigation_match_gets_a_reason_even_without_applicability():
    """Approved item 6: must have an explicit explanation even when
    applicability did not contribute (no customer/region/technology
    context at all -- the common case for an unclassified corpus)."""
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hi-1",
        title="Some case",
        snippet="x",
        score=0.71,
        metadata={"root_cause": "X", "resolution": "Y"},  # no applicability_reasons key at all
    )
    engine = _engine()

    annotated = engine._annotate_match_reasons([match])

    assert annotated[0].reason != ""
    assert "71%" in annotated[0].reason
    assert "recorded root cause" in annotated[0].reason
    assert "recorded resolution" in annotated[0].reason


def test_match_with_applicability_reasons_uses_them_in_its_reason():
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hi-1",
        title="Some case",
        snippet="x",
        score=0.9,
        metadata={"applicability_reasons": ["Same customer: TEPCO"]},
    )
    engine = _engine()

    annotated = engine._annotate_match_reasons([match])

    assert "Same customer: TEPCO" in annotated[0].reason


def test_sql_library_suggestion_has_investigation_specific_match_reason():
    """Approved item 7: the static template description is not
    sufficient -- match_reason must name the real matched component."""
    from app.domain.product_intelligence import ComponentProfile
    from app.domain.recommendation import MatchedComponent
    from app.domain.sql_studio import QueryTemplate

    class FakeSqlLibrary:
        def __init__(self, templates) -> None:
            self._templates = templates

        def list_templates(self):
            return self._templates

    template = QueryTemplate(
        id="qt-1",
        title="Command log lookup",
        category="ami",
        sql_text="SELECT 1;",
        explanation="Static description of what the query does.",
        related_components=["CommandProcessorHost"],
    )
    matched_component = MatchedComponent(component_id="c-1", component_name="CommandProcessorHost", confidence=1.0, match_reason="exact match")
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), sql_library=FakeSqlLibrary([template])
    )

    items = engine._suggested_sql_items(matched_component, [])

    assert len(items) == 1
    assert items[0].match_reason != ""
    assert items[0].match_reason != items[0].explanation
    assert "CommandProcessorHost" in items[0].match_reason


def test_entity_heuristic_sql_suggestion_has_match_reason():
    from app.domain.entities import ExtractedEntity
    from app.domain.enums import EntityType

    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings())
    entities = [ExtractedEntity(entity_type=EntityType.SQL_SESSION, value="42")]

    items = engine._suggested_sql_items(None, entities)

    assert len(items) == 1
    assert items[0].source == "entity_heuristic"
    assert "42" in items[0].match_reason


# --- resolution_verified* round-trip (approved items 4/5) --------------


@pytest.fixture
def session_factory():
    from app.infrastructure.db.session import get_engine, get_session_factory

    with tempfile.TemporaryDirectory() as tmp:
        sqlite_url = f"sqlite:///{(Path(tmp) / 'test.db').as_posix()}"
        yield get_session_factory(sqlite_url)
        get_engine(sqlite_url).dispose()
        get_engine.cache_clear()


def test_historical_investigation_resolution_verified_round_trips(session_factory):
    from app.infrastructure.db.knowledge_repository import SqlAlchemyKnowledgeRepository

    repo = SqlAlchemyKnowledgeRepository(session_factory)
    record = HistoricalInvestigationRecord(
        id=str(uuid.uuid4()),
        title="A verified case",
        description="x",
        root_cause="Firmware mismatch",
        resolution="Reissued firmware",
        resolution_verified=True,
        resolution_verified_by="jsmith",
        resolution_verified_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        resolution_verification_note="Confirmed with the customer.",
    )

    repo.save_historical_investigation(record)
    fetched = repo.get_historical_investigation(record.id)

    assert fetched.resolution_verified is True
    assert fetched.resolution_verified_by == "jsmith"
    assert fetched.resolution_verification_note == "Confirmed with the customer."
    assert fetched.resolution_verified_at is not None


def test_historical_investigation_resolution_verified_defaults_to_false(session_factory):
    from app.infrastructure.db.knowledge_repository import SqlAlchemyKnowledgeRepository

    repo = SqlAlchemyKnowledgeRepository(session_factory)
    record = HistoricalInvestigationRecord(id=str(uuid.uuid4()), title="An unverified case", description="x", root_cause="X", resolution="Y")

    repo.save_historical_investigation(record)
    fetched = repo.get_historical_investigation(record.id)

    assert fetched.resolution_verified is False
    assert fetched.resolution_verified_by is None


def test_known_bug_resolution_verified_round_trips(session_factory):
    from app.domain.evidence import KnownBugRecord
    from app.infrastructure.db.knowledge_repository import SqlAlchemyKnowledgeRepository

    repo = SqlAlchemyKnowledgeRepository(session_factory)
    record = KnownBugRecord(
        id=str(uuid.uuid4()),
        title="A verified bug",
        description="x",
        resolution_verified=True,
        resolution_verified_by="admin",
        resolution_verified_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        resolution_verification_note="Fix confirmed in 9.0.3.",
    )

    repo.save_known_bug(record)
    fetched = repo.get_known_bug(record.id)

    assert fetched.resolution_verified is True
    assert fetched.resolution_verified_by == "admin"
    assert fetched.resolution_verification_note == "Fix confirmed in 9.0.3."
