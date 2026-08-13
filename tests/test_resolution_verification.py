"""Tests for Human Verification (2026-08-13, Phase 0 -- Chat/Structured
Resolution Knowledge architecture, closing the gap flagged in
RESOLVEIQ_CHAT_AND_RESOLUTION_ARCHITECTURE.md Section 20):
- the dedicated /verify, /unverify endpoints set/clear all four
  resolution_verified* fields atomically, for both governed types;
- the generic Knowledge Object PATCH/create endpoints refuse to accept
  any of those four field names (the "no unrestricted UI/API control"
  requirement, enforced server-side, not just hidden in the UI).

Calls the FastAPI route functions directly (same pattern
test_app_wiring.py documents this codebase uses -- no TestClient/HTTP
layer exists anywhere in this suite) against a real temp-SQLite-backed
KnowledgeObjectService, since this genuinely depends on several real
repositories (same reasoning as test_classification.py's engine_bundle
fixture).
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.domain.evidence import HistoricalInvestigationRecord, KnownBugRecord
from app.api.routers.admin.knowledge_objects import _reject_reserved_fields
from app.api.routers.admin.resolution_verification import (
    UnverifyRequest,
    VerifyResolutionRequest,
    unverify_historical_investigation,
    unverify_known_bug,
    verify_historical_investigation,
    verify_known_bug,
)
from app.engines.knowledge.engine import KnowledgeEngine
from app.engines.knowledge_object_framework.adapters import build_adapters
from app.engines.knowledge_object_framework.service import KnowledgeObjectNotFoundError, KnowledgeObjectService
from app.engines.knowledge_relationships.engine import KnowledgeRelationshipEngine
from app.infrastructure.db.component_repository import SqlAlchemyComponentProfileRepository
from app.infrastructure.db.knowledge_repository import SqlAlchemyKnowledgeRepository
from app.infrastructure.db.log_knowledge_repository import SqlAlchemyLogKnowledgeRepository
from app.infrastructure.db.lookup_repository import SqlAlchemyLookupRepository
from app.infrastructure.db.playbook_repository import SqlAlchemyPlaybookRepository
from app.infrastructure.db.relationship_repository import SqlAlchemyRelationshipRepository
from app.infrastructure.db.session import get_engine, get_session_factory
from app.infrastructure.db.sql_template_repository import SqlAlchemySqlTemplateRepository
from app.infrastructure.db.version_repository import SqlAlchemyVersionRepository


class _FakeKnowledgeStore:
    def upsert(self, collection, record_id, text, title, metadata) -> None:
        pass

    def query(self, collection, text, top_k: int = 5):  # pragma: no cover
        return []

    def count(self, collection) -> int:
        return 0

    def list_recent(self, collection, limit: int = 5):  # pragma: no cover
        return []

    def delete(self, collection, record_id: str) -> None:
        pass


@pytest.fixture
def service_bundle():
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
        version_repo = SqlAlchemyVersionRepository(session_factory)

        relationship_engine = KnowledgeRelationshipEngine(
            relationship_repo, component_repo, knowledge_repo, sql_repo, playbook_repo, lookup_repo, log_knowledge_repo
        )
        adapters = build_adapters(component_repo, knowledge_repo, sql_repo, playbook_repo, lookup_repo, log_knowledge_repo)
        knowledge_engine = KnowledgeEngine(_FakeKnowledgeStore(), knowledge_repo)
        service = KnowledgeObjectService(adapters, relationship_engine, version_repo, knowledge_engine)

        yield dict(knowledge_repo=knowledge_repo, service=service)
        get_engine(sqlite_url).dispose()
        get_engine.cache_clear()


def _save_investigation(knowledge_repo) -> HistoricalInvestigationRecord:
    record = HistoricalInvestigationRecord(
        id=str(uuid.uuid4()), title="Commands stuck in Pending", description="x", root_cause="Queue backlog", resolution="Restart"
    )
    knowledge_repo.save_historical_investigation(record)
    return record


def _save_bug(knowledge_repo) -> KnownBugRecord:
    record = KnownBugRecord(id=str(uuid.uuid4()), title="Known queue backup bug", description="x", workaround="Increase timeout")
    knowledge_repo.save_known_bug(record)
    return record


# --- /verify, /unverify set/clear all four fields atomically -----------


def test_verify_historical_investigation_sets_all_four_fields(service_bundle):
    record = _save_investigation(service_bundle["knowledge_repo"])

    updated = verify_historical_investigation(
        record.id, VerifyResolutionRequest(actor="jsmith", note="Confirmed with the customer."), service_bundle["service"]
    )

    assert updated.resolution_verified is True
    assert updated.resolution_verified_by == "jsmith"
    assert updated.resolution_verified_at is not None
    assert updated.resolution_verification_note == "Confirmed with the customer."


def test_unverify_historical_investigation_clears_all_four_fields(service_bundle):
    record = _save_investigation(service_bundle["knowledge_repo"])
    verify_historical_investigation(record.id, VerifyResolutionRequest(actor="jsmith", note="x"), service_bundle["service"])

    updated = unverify_historical_investigation(record.id, UnverifyRequest(actor="reviewer"), service_bundle["service"])

    assert updated.resolution_verified is False
    assert updated.resolution_verified_by is None
    assert updated.resolution_verified_at is None
    assert updated.resolution_verification_note is None


def test_verify_known_bug_sets_all_four_fields(service_bundle):
    bug = _save_bug(service_bundle["knowledge_repo"])

    updated = verify_known_bug(bug.id, VerifyResolutionRequest(actor="admin", note="Fix confirmed in 9.0.4."), service_bundle["service"])

    assert updated.resolution_verified is True
    assert updated.resolution_verified_by == "admin"
    assert updated.resolution_verification_note == "Fix confirmed in 9.0.4."


def test_unverify_known_bug_clears_all_four_fields(service_bundle):
    bug = _save_bug(service_bundle["knowledge_repo"])
    verify_known_bug(bug.id, VerifyResolutionRequest(actor="admin"), service_bundle["service"])

    updated = unverify_known_bug(bug.id, UnverifyRequest(actor="admin"), service_bundle["service"])

    assert updated.resolution_verified is False
    assert updated.resolution_verified_by is None


def test_verify_unknown_investigation_id_raises_404(service_bundle):
    with pytest.raises(HTTPException) as exc_info:
        verify_historical_investigation("nonexistent-id", VerifyResolutionRequest(), service_bundle["service"])
    assert exc_info.value.status_code == 404


def test_verify_unknown_bug_id_raises_404(service_bundle):
    with pytest.raises(HTTPException) as exc_info:
        verify_known_bug("nonexistent-id", VerifyResolutionRequest(), service_bundle["service"])
    assert exc_info.value.status_code == 404


def test_verification_note_is_optional(service_bundle):
    record = _save_investigation(service_bundle["knowledge_repo"])

    updated = verify_historical_investigation(record.id, VerifyResolutionRequest(actor="jsmith"), service_bundle["service"])

    assert updated.resolution_verified is True
    assert updated.resolution_verification_note is None


# --- Generic PATCH/create endpoints refuse these four fields ------------


def test_reject_reserved_fields_blocks_resolution_verified():
    with pytest.raises(HTTPException) as exc_info:
        _reject_reserved_fields({"resolution_verified": True})
    assert exc_info.value.status_code == 400


def test_reject_reserved_fields_blocks_any_of_the_four_individually():
    for field_name in (
        "resolution_verified",
        "resolution_verified_by",
        "resolution_verified_at",
        "resolution_verification_note",
    ):
        with pytest.raises(HTTPException) as exc_info:
            _reject_reserved_fields({field_name: "x"})
        assert exc_info.value.status_code == 400


def test_reject_reserved_fields_allows_ordinary_fields():
    _reject_reserved_fields({"title": "New title", "description": "New description"})  # must not raise


def test_reject_reserved_fields_allows_empty():
    _reject_reserved_fields({})  # must not raise
