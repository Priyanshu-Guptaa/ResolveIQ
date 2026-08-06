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
from app.engines.investigation.engine import InvestigationEngine, InvestigationNotFoundError
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
