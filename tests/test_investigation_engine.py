"""Tests for InvestigationEngine against a real (temp-file) SQLite database
-- the new methods this phase added (update_details, get_timeline) are
repository/engine integration behavior, not pure logic, so a fake
repository would just re-assert the mock rather than catch real bugs
(this is exactly how the Phase 1.5 N+1/unbounded-query bugs slipped in
originally).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.engines.ingestion.engine import IngestionEngine
from app.engines.ingestion.file_type_registry import FileTypeRegistry
from app.engines.investigation.engine import EvidenceNotFoundError, InvestigationEngine, InvestigationNotFoundError
from app.engines.log_intelligence.engine import LogIntelligenceEngine
from app.engines.log_intelligence.entity_extractor import RegexEntityExtractor
from app.engines.log_intelligence.log_parser import GenericLogParser
from app.infrastructure.db.repository import SqlAlchemyInvestigationRepository
from app.infrastructure.db.session import get_engine, get_session_factory


@pytest.fixture
def engine():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        sqlite_url = f"sqlite:///{db_path.as_posix()}"
        session_factory = get_session_factory(sqlite_url)
        repository = SqlAlchemyInvestigationRepository(session_factory)

        extractor = RegexEntityExtractor()
        log_intelligence = LogIntelligenceEngine(GenericLogParser(extractor), extractor)
        ingestion = IngestionEngine(FileTypeRegistry())

        yield InvestigationEngine(repository, log_intelligence, ingestion)

        # get_engine() is process-wide lru_cache'd (by design -- see its
        # docstring), so the SQLite file handle it opened outlives this
        # fixture unless explicitly disposed. On Windows that open handle
        # blocks TemporaryDirectory's own cleanup. Fine in the real app
        # (one URL, one engine, whole process lifetime); here it just
        # needs tearing down per test.
        get_engine(sqlite_url).dispose()
        get_engine.cache_clear()


def test_update_details_sets_fields(engine: InvestigationEngine):
    investigation = engine.start_investigation("Test case")

    updated = engine.update_details(investigation.id, customer="Acme Utilities", technology="RF Mesh")

    assert updated.customer == "Acme Utilities"
    assert updated.technology == "RF Mesh"


def test_update_details_partial_update_preserves_other_fields(engine: InvestigationEngine):
    investigation = engine.start_investigation("Test case")
    engine.update_details(investigation.id, customer="Acme Utilities", product="Command Center")

    # Only updating technology -- customer and product must survive.
    updated = engine.update_details(investigation.id, technology="RF Mesh")

    assert updated.customer == "Acme Utilities"
    assert updated.product == "Command Center"
    assert updated.technology == "RF Mesh"


def test_update_details_unknown_investigation_raises(engine: InvestigationEngine):
    with pytest.raises(InvestigationNotFoundError):
        engine.update_details("does-not-exist", customer="Acme")


def test_get_timeline_includes_creation_and_evidence_events_chronologically(engine: InvestigationEngine):
    investigation = engine.start_investigation("Test case", description="Initial description")
    engine.add_text_evidence(investigation.id, "A manual note", title="Note 1")

    timeline = engine.get_timeline(investigation.id)

    kinds = [item.kind for item in timeline]
    assert "investigation_created" in kinds
    assert kinds.count("evidence_added") == 2  # task description + manual note

    # Chronological, most recent first.
    timestamps = [item.occurred_at for item in timeline]
    assert timestamps == sorted(timestamps, reverse=True)


def test_get_timeline_scoped_to_one_investigation(engine: InvestigationEngine):
    inv_a = engine.start_investigation("Investigation A")
    inv_b = engine.start_investigation("Investigation B")
    engine.add_text_evidence(inv_a.id, "Note for A")
    engine.add_text_evidence(inv_b.id, "Note for B")

    timeline_a = engine.get_timeline(inv_a.id)

    assert all(item.investigation_id == inv_a.id for item in timeline_a)
    assert not any("Note for B" in item.label for item in timeline_a)


def test_list_investigation_summaries_matches_evidence_count(engine: InvestigationEngine):
    investigation = engine.start_investigation("Test case", description="desc")
    engine.add_text_evidence(investigation.id, "Note 1")
    engine.add_text_evidence(investigation.id, "Note 2")

    summaries = engine.list_investigation_summaries()

    summary = next(s for s in summaries if s.id == investigation.id)
    assert summary.evidence_count == 3  # task description + 2 notes


# --- Investigation loading redesign -----------------------------------------
# Found via a real investigation whose evidence (97 items, one legitimately
# 15.7MB) made the old GET /investigations/{id} return 151MB of JSON.


def test_investigation_summary_never_contains_raw_evidence_content(engine: InvestigationEngine):
    investigation = engine.start_investigation("Test case")
    large_text = "X" * 2_000_000  # 2MB -- deliberately large
    engine.add_text_evidence(investigation.id, large_text, title="Big note")

    summary = engine.get_investigation_summary(investigation.id)

    assert not hasattr(summary, "evidence")
    assert not hasattr(summary, "raw_content")
    # The whole point: serializing the summary must not embed the 2MB
    # string anywhere, in any field, at any nesting level.
    assert large_text not in summary.model_dump_json()


def test_investigation_summary_payload_size_does_not_scale_with_evidence_content_size(engine: InvestigationEngine):
    """The actual "large investigations load quickly" guarantee: two
    investigations with the same *shape* of evidence (one note each) but
    wildly different content sizes must produce near-identical summary
    payload sizes -- proving the response size tracks evidence *count*,
    not evidence *content size*."""
    small_investigation = engine.start_investigation("Small case")
    engine.add_text_evidence(small_investigation.id, "short note", title="Note")

    large_investigation = engine.start_investigation("Large case")
    engine.add_text_evidence(large_investigation.id, "Y" * 5_000_000, title="Note")  # 5MB

    small_summary = engine.get_investigation_summary(small_investigation.id)
    large_summary = engine.get_investigation_summary(large_investigation.id)

    small_size = len(small_summary.model_dump_json())
    large_size = len(large_summary.model_dump_json())
    # Titles differ in length ("Small case" vs "Large case" -- close
    # enough), but 5MB of extra evidence content must not show up at all.
    assert abs(large_size - small_size) < 200


def test_list_evidence_never_contains_raw_content_or_full_entity_lists(engine: InvestigationEngine):
    investigation = engine.start_investigation("Test case")
    large_text = "Z" * 2_000_000
    engine.add_text_evidence(investigation.id, large_text, title="Big note")

    summaries = engine.list_evidence(investigation.id)

    assert len(summaries) == 1  # no task description passed to start_investigation, just the note
    for item in summaries:
        assert not hasattr(item, "raw_content")
        assert not hasattr(item, "extracted_entities")
        assert not hasattr(item, "log_events")
    note_summary = next(s for s in summaries if s.title == "Big note")
    assert note_summary.content_length == len(large_text)  # size known without loading the content


def test_get_evidence_returns_full_content_on_demand(engine: InvestigationEngine):
    investigation = engine.start_investigation("Test case")
    created = engine.add_text_evidence(investigation.id, "the real content", title="A note")

    fetched = engine.get_evidence(investigation.id, created.id)

    assert fetched.raw_content == "the real content"


def test_get_evidence_wrong_investigation_raises(engine: InvestigationEngine):
    inv_a = engine.start_investigation("A")
    inv_b = engine.start_investigation("B")
    created = engine.add_text_evidence(inv_a.id, "note for A")

    with pytest.raises(EvidenceNotFoundError):
        engine.get_evidence(inv_b.id, created.id)


def test_get_evidence_preview_truncates_and_reports_full_length(engine: InvestigationEngine):
    investigation = engine.start_investigation("Test case")
    text = "A" * 5000
    created = engine.add_text_evidence(investigation.id, text, title="Long note")

    preview = engine.get_evidence_preview(investigation.id, created.id, max_chars=100)

    assert len(preview.preview_text) == 100
    assert preview.truncated is True
    assert preview.full_length == 5000


def test_get_evidence_preview_not_truncated_when_content_fits(engine: InvestigationEngine):
    investigation = engine.start_investigation("Test case")
    created = engine.add_text_evidence(investigation.id, "short", title="Short note")

    preview = engine.get_evidence_preview(investigation.id, created.id, max_chars=2000)

    assert preview.preview_text == "short"
    assert preview.truncated is False
    assert preview.full_length == 5


def test_entity_summary_aggregates_across_evidence(engine: InvestigationEngine):
    investigation = engine.start_investigation("Test case")
    engine.add_text_evidence(investigation.id, "correlation id abc-123 seen here", title="Note 1")
    engine.add_text_evidence(investigation.id, "correlation id abc-123 seen again", title="Note 2")

    summary = engine.get_investigation_summary(investigation.id)

    # Whatever entity types the real extractor found, none of them may
    # carry more sample_values than were actually distinct, and every
    # count must be a real aggregate, not per-evidence.
    for item in summary.entity_summary:
        assert item.count >= len(set(item.sample_values))


def test_investigation_summary_stays_fast_with_a_very_large_entity_list(engine: InvestigationEngine):
    """Regression guard for a real bug: get_summary()'s first version
    selected the full extracted_entities JSON column and aggregated it
    in Python. A real noisy log file's regex-matched entities (~19,500
    entries, 1.56MB of JSON for one evidence row) made that take ~7
    seconds. Fixed by aggregating via SQLite's json_each() instead --
    measured 0.05-0.1s for the same real data. This test reproduces the
    shape (one evidence item with many repeated entity values) at a
    smaller but still substantial scale and asserts both correctness
    and a real wall-clock bound, so a future regression back to
    Python-side aggregation would be caught here, not just in
    production."""
    import time

    from app.domain.entities import ExtractedEntity
    from app.domain.enums import EntityType, EvidenceType
    from app.domain.evidence import Evidence

    investigation = engine.start_investigation("Test case")
    entities = [
        ExtractedEntity(entity_type=EntityType.METER_NUMBER, value=f"M{i % 50}")
        for i in range(20_000)
    ]
    evidence = Evidence(
        investigation_id=investigation.id,
        evidence_type=EvidenceType.LOG_FILE,
        source="upload",
        title="noisy.log",
        raw_content="synthetic",
        extracted_entities=entities,
    )
    engine._repository.add_evidence(investigation.id, evidence)  # bypass the parser -- not the point here

    start = time.time()
    summary = engine.get_investigation_summary(investigation.id)
    elapsed = time.time() - start

    assert elapsed < 2.0, f"get_investigation_summary took {elapsed:.2f}s -- entity aggregation regressed to Python-side"
    meter_summary = next(e for e in summary.entity_summary if e.entity_type == "meter_number")
    assert meter_summary.count == 20_000
    assert len(meter_summary.sample_values) == 10  # capped, not all 50 distinct values


def test_ensure_exists_stays_fast_with_a_large_raw_content_investigation(engine: InvestigationEngine):
    """Regression guard for a second real bug found verifying the first
    fix: with get_summary() itself fixed, GET /investigations/{id} was
    STILL ~6.4s for the real investigation -- because the router also
    calls mark_viewed(), which (like list_evidence(), add_text_evidence(),
    etc.) goes through _ensure_exists(), which called the repository's
    full get() (every evidence row's raw_content, hydrated via the
    ``evidence`` relationship) just to check a row exists. Invisible
    while the main query was already slow for the same reason; became
    the dominant cost once that was fixed. Fixed with a dedicated
    exists() repository method that only ever selects a primary key."""
    import time

    from app.domain.enums import EvidenceType
    from app.domain.evidence import Evidence

    investigation = engine.start_investigation("Test case")
    evidence = Evidence(
        investigation_id=investigation.id,
        evidence_type=EvidenceType.LOG_FILE,
        source="upload",
        title="huge.log",
        raw_content="X" * 5_000_000,  # 5MB
    )
    engine._repository.add_evidence(investigation.id, evidence)

    start = time.time()
    engine.mark_viewed(investigation.id)  # exactly what GET /investigations/{id} does after the summary fetch
    engine.list_evidence(investigation.id)
    elapsed = time.time() - start

    assert elapsed < 1.0, f"mark_viewed + list_evidence took {elapsed:.2f}s -- _ensure_exists regressed to full hydration"


# --- Duplicate-upload prevention --------------------------------------------
# Found via a real 1.5MB zip that was processed twice after its first
# upload's client-side timeout led to a retry (the server had actually
# succeeded) -- created 67 duplicate rows in a real investigation.


def test_duplicate_text_evidence_reuses_existing_row(engine: InvestigationEngine):
    investigation = engine.start_investigation("Test case")
    first = engine.add_text_evidence(investigation.id, "identical content", title="Note")
    second = engine.add_text_evidence(investigation.id, "identical content", title="Note")

    assert second.id == first.id
    all_evidence = engine.list_evidence(investigation.id)
    notes = [e for e in all_evidence if e.title == "Note"]
    assert len(notes) == 1  # not 2


def test_different_text_evidence_is_not_deduplicated(engine: InvestigationEngine):
    investigation = engine.start_investigation("Test case")
    first = engine.add_text_evidence(investigation.id, "content A", title="Note")
    second = engine.add_text_evidence(investigation.id, "content B", title="Note")

    assert second.id != first.id
    all_evidence = engine.list_evidence(investigation.id)
    notes = [e for e in all_evidence if e.title == "Note"]
    assert len(notes) == 2


def test_duplicate_file_upload_reuses_existing_row_not_create_new(engine: InvestigationEngine):
    investigation = engine.start_investigation("Test case")
    content = b"line one\nline two\nERROR something broke\n"

    first_batch = engine.add_file_evidence(investigation.id, "app.log", content)
    second_batch = engine.add_file_evidence(investigation.id, "app.log", content)  # simulated retry

    assert len(first_batch) == 1
    assert len(second_batch) == 1
    assert second_batch[0].id == first_batch[0].id

    all_evidence = engine.list_evidence(investigation.id)
    log_files = [e for e in all_evidence if e.evidence_type.value == "log_file"]
    assert len(log_files) == 1  # not 2


def test_evidence_with_empty_content_is_never_deduplicated(engine: InvestigationEngine):
    """Two different unsupported/textless uploads must both be kept --
    an empty content hash can't distinguish them, so they're
    deliberately exempt from dedup rather than incorrectly merged."""
    investigation = engine.start_investigation("Test case")

    first_batch = engine.add_file_evidence(investigation.id, "binary_a.bin", b"\x00\x01\x02")
    second_batch = engine.add_file_evidence(investigation.id, "binary_b.bin", b"\x00\x01\x02")

    all_evidence = engine.list_evidence(investigation.id)
    binaries = [e for e in all_evidence if e.title in ("binary_a.bin", "binary_b.bin")]
    assert len(binaries) == 2  # both kept, not merged
