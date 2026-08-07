"""Tests for the Knowledge Object Framework (Sprint 3, Phase 3.4).

Same real temp-file SQLite pattern as ``test_knowledge_relationships.py``
-- this is persistence/lifecycle behavior, not pure logic. The point of
this suite is specifically to prove the framework supports every one of
the nine ``KnowledgeObjectType`` values through the SAME service code,
with zero per-type branching added here or in the service itself.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest

from app.config import get_settings
from app.domain.knowledge_relationships import KnowledgeObjectType, RelationshipType
from app.engines.knowledge_object_framework.adapters import build_adapters
from app.engines.knowledge_object_framework.service import (
    KnowledgeObjectNotFoundError,
    KnowledgeObjectService,
    ObjectHasDependentsError,
)
from app.engines.knowledge_relationships.engine import KnowledgeRelationshipEngine
from app.engines.knowledge.engine import KnowledgeEngine
from app.infrastructure.db.component_repository import SqlAlchemyComponentProfileRepository
from app.infrastructure.db.knowledge_repository import SqlAlchemyKnowledgeRepository
from app.infrastructure.db.lookup_repository import SqlAlchemyLookupRepository
from app.infrastructure.db.playbook_repository import SqlAlchemyPlaybookRepository
from app.infrastructure.db.relationship_repository import SqlAlchemyRelationshipRepository
from app.infrastructure.db.seed_migration import migrate_all
from app.infrastructure.db.session import get_engine, get_session_factory
from app.infrastructure.db.sql_template_repository import SqlAlchemySqlTemplateRepository
from app.infrastructure.db.version_repository import SqlAlchemyVersionRepository

SAMPLE_DIR = get_settings().sample_knowledge_dir

T = KnowledgeObjectType


class _StubKnowledgeStore:
    """Avoids booting a real embedding model / Chroma client for tests
    that only exercise lifecycle, not search -- KnowledgeEngine only
    needs ``upsert``/``delete`` (the ``KnowledgeStore`` Protocol) to be
    callable."""

    def __init__(self) -> None:
        self.upserts: list[str] = []
        self.deletes: list[str] = []

    def upsert(self, collection, record_id: str, text: str, title: str, *, metadata: dict | None = None) -> None:
        self.upserts.append(record_id)

    def delete(self, collection, record_id: str) -> None:
        self.deletes.append(record_id)

    def count(self, collection) -> int:
        return 0


@pytest.fixture
def setup():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        sqlite_url = f"sqlite:///{db_path.as_posix()}"
        session_factory = get_session_factory(sqlite_url)

        component_repo = SqlAlchemyComponentProfileRepository(session_factory)
        knowledge_repo = SqlAlchemyKnowledgeRepository(session_factory)
        sql_repo = SqlAlchemySqlTemplateRepository(session_factory)
        playbook_repo = SqlAlchemyPlaybookRepository(session_factory)
        lookup_repo = SqlAlchemyLookupRepository(session_factory)
        relationship_repo = SqlAlchemyRelationshipRepository(session_factory)
        version_repo = SqlAlchemyVersionRepository(session_factory)

        migrate_all(
            component_repo=component_repo,
            knowledge_repo=knowledge_repo,
            sql_repo=sql_repo,
            lookup_repo=lookup_repo,
            sample_knowledge_dir=SAMPLE_DIR,
        )

        relationship_engine = KnowledgeRelationshipEngine(
            relationship_repo, component_repo, knowledge_repo, sql_repo, playbook_repo, lookup_repo
        )
        adapters = build_adapters(component_repo, knowledge_repo, sql_repo, playbook_repo, lookup_repo)
        knowledge_engine = KnowledgeEngine(_StubKnowledgeStore(), knowledge_repo)
        service = KnowledgeObjectService(adapters, relationship_engine, version_repo, knowledge_engine)

        yield service, {
            "components": component_repo,
            "knowledge": knowledge_repo,
            "sql": sql_repo,
            "playbooks": playbook_repo,
            "lookups": lookup_repo,
            "relationships": relationship_repo,
            "relationship_engine": relationship_engine,
        }

        get_engine(sqlite_url).dispose()
        get_engine.cache_clear()


def _component_id(repos) -> str:
    return repos["components"].get_by_name("CommandProcessorHost").id


# --- Create --------------------------------------------------------------


@pytest.mark.parametrize(
    "object_type,fields",
    [
        (T.KNOWN_BUG, {"title": "New bug", "description": "desc"}),
        (
            T.SQL_TEMPLATE,
            {"title": "New template", "category": "diagnostic", "sql_text": "SELECT 1", "explanation": "e"},
        ),
        (T.PLAYBOOK, {"title": "New playbook"}),
        (T.PRODUCT, {"name": "New Product"}),
        (T.TECHNOLOGY, {"name": "New Tech"}),
        (T.VERSION, {"name": "9.9.9"}),
        (
            T.HISTORICAL_INVESTIGATION,
            {
                "title": "New investigation",
                "domain": "billing",
                "description": "d",
                "root_cause": "rc",
                "resolution": "res",
            },
        ),
    ],
)
def test_create_works_for_every_type_without_per_type_code(setup, object_type, fields):
    """The whole point of the framework: the SAME ``service.create``
    call, with no branching, produces a valid, persisted, retrievable
    object for every one of these types."""
    service, _ = setup
    created = service.create(object_type, created_by="tester", **fields)
    assert created.id
    fetched = service.get(object_type, created.id)
    assert fetched is not None
    assert fetched.id == created.id


def test_create_component(setup):
    service, _ = setup
    created = service.create(T.COMPONENT, name="NewComponent", product="Command Center")
    assert service.get(T.COMPONENT, created.id) is not None


def test_create_document(setup):
    service, _ = setup
    created = service.create(T.DOCUMENT, title="New Doc", content="body text")
    assert service.get(T.DOCUMENT, created.id) is not None


def test_create_surfaces_pydantic_validation_error(setup):
    """Missing a genuinely required field (Known Bug needs a title) --
    the service does not hardcode this rule; Pydantic does."""
    service, _ = setup
    with pytest.raises(Exception):
        service.create(T.KNOWN_BUG, description="no title given")


# --- Edit metadata ---------------------------------------------------------


def test_edit_metadata_updates_fields_and_records_history(setup):
    service, _ = setup
    created = service.create(T.KNOWN_BUG, title="Bug A", description="d")
    updated = service.edit_metadata(T.KNOWN_BUG, created.id, updated_by="editor", title="Bug A Renamed")
    assert updated.title == "Bug A Renamed"
    assert updated.updated_by == "editor"

    history = service.get_history(T.KNOWN_BUG, created.id)
    assert len(history) == 2  # created + edited
    assert history[0].version_number == 2
    assert history[0].change_summary == "edited"
    assert history[1].change_summary == "created"


def test_edit_metadata_raises_not_found(setup):
    service, _ = setup
    with pytest.raises(KnowledgeObjectNotFoundError):
        service.edit_metadata(T.KNOWN_BUG, "does-not-exist", title="x")


# --- Lifecycle ---------------------------------------------------------------


@pytest.mark.parametrize(
    "transition,expected_status",
    [
        ("publish", "published"),
        ("archive", "archived"),
        ("restore", "draft"),
        ("deprecate", "deprecated"),
    ],
)
def test_lifecycle_transitions(setup, transition, expected_status):
    service, _ = setup
    created = service.create(T.SQL_TEMPLATE, title="T", category="c", sql_text="SELECT 1", explanation="e")
    method = getattr(service, transition)
    updated = method(T.SQL_TEMPLATE, created.id, updated_by="ops")
    assert updated.status.value == expected_status


def test_full_lifecycle_round_trip_records_history_each_step(setup):
    service, _ = setup
    created = service.create(T.PLAYBOOK, title="P")
    service.publish(T.PLAYBOOK, created.id)
    service.archive(T.PLAYBOOK, created.id)
    service.restore(T.PLAYBOOK, created.id)
    service.deprecate(T.PLAYBOOK, created.id)

    history = service.get_history(T.PLAYBOOK, created.id)
    assert [h.change_summary for h in history] == [
        "deprecated",
        "restored to draft",
        "archived",
        "published",
        "created",
    ]
    assert [h.version_number for h in history] == [5, 4, 3, 2, 1]


def test_publishing_document_indexes_it_and_archiving_unindexes(setup):
    service, repos = setup
    created = service.create(T.DOCUMENT, title="Doc", content="body")
    knowledge_store: _StubKnowledgeStore = service._knowledge._store  # type: ignore[attr-defined]

    service.publish(T.DOCUMENT, created.id)
    assert created.id in knowledge_store.upserts

    service.archive(T.DOCUMENT, created.id)
    assert created.id in knowledge_store.deletes


# --- Delete / Impact gate -----------------------------------------------------


def test_delete_succeeds_when_no_dependents(setup):
    service, _ = setup
    created = service.create(T.PLAYBOOK, title="Orphan playbook")
    service.delete(T.PLAYBOOK, created.id)
    assert service.get(T.PLAYBOOK, created.id) is None


def test_delete_raises_not_found(setup):
    service, _ = setup
    with pytest.raises(KnowledgeObjectNotFoundError):
        service.delete(T.PLAYBOOK, "does-not-exist")


def test_delete_blocked_when_object_has_dependents(setup):
    service, repos = setup
    component_id = _component_id(repos)
    bug = service.create(T.KNOWN_BUG, title="Linked bug", description="d")

    repos["relationship_engine"].add_relationship(
        T.KNOWN_BUG, bug.id, T.COMPONENT, component_id, RelationshipType.RELATED_TO, created_by="tester"
    )

    with pytest.raises(ObjectHasDependentsError) as excinfo:
        service.delete(T.COMPONENT, component_id)
    assert excinfo.value.impact.total_dependents >= 1
    # Component must still exist -- delete was refused, not partially applied.
    assert service.get(T.COMPONENT, component_id) is not None


# --- History / versioning -----------------------------------------------------


def test_history_numbers_increment_monotonically(setup):
    service, _ = setup
    created = service.create(T.TECHNOLOGY, name="Tech A")
    service.edit_metadata(T.TECHNOLOGY, created.id, name="Tech A2")
    service.edit_metadata(T.TECHNOLOGY, created.id, name="Tech A3")
    history = service.get_history(T.TECHNOLOGY, created.id)
    assert [h.version_number for h in history] == [3, 2, 1]
    assert history[0].snapshot["name"] == "Tech A3"


# --- Validation ----------------------------------------------------------------


def test_validate_flags_duplicate_titles(setup):
    """Playbook titles have no DB-level uniqueness constraint (unlike
    Product/Technology/Version names) -- exactly the case ``validate``'s
    duplicate check exists for."""
    service, _ = setup
    service.create(T.PLAYBOOK, title="Duplicate Name")
    second = service.create(T.PLAYBOOK, title="Duplicate Name")
    warnings = service.validate(T.PLAYBOOK, second.id)
    assert any("possible duplicate" in w for w in warnings)


def test_validate_flags_blank_title(setup):
    service, _ = setup
    created = service.create(T.PLAYBOOK, title=" ")
    warnings = service.validate(T.PLAYBOOK, created.id)
    assert any("title" in w.lower() for w in warnings)


def test_validate_clean_object_has_no_warnings(setup):
    service, _ = setup
    created = service.create(T.TECHNOLOGY, name="Unique Tech Name")
    warnings = service.validate(T.TECHNOLOGY, created.id)
    assert warnings == []


def test_validate_raises_not_found(setup):
    service, _ = setup
    with pytest.raises(KnowledgeObjectNotFoundError):
        service.validate(T.TECHNOLOGY, "does-not-exist")


# --- Relationship/Impact delegation (not reimplemented) ------------------------


def test_service_delegates_relationships_to_relationship_engine(setup):
    service, repos = setup
    component_id = _component_id(repos)
    bug = service.create(T.KNOWN_BUG, title="Linked bug 2", description="d")
    repos["relationship_engine"].add_relationship(
        T.KNOWN_BUG, bug.id, T.COMPONENT, component_id, RelationshipType.RELATED_TO, created_by="tester"
    )

    via_service = service.list_relationships(T.COMPONENT, component_id)
    via_engine = repos["relationship_engine"].list_relationships(T.COMPONENT, component_id)
    assert {r.relationship.id for r in via_service} == {r.relationship.id for r in via_engine}
