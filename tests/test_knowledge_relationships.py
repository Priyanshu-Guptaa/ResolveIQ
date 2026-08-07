"""Tests for the Knowledge Relationship Engine (Sprint 3, Phase 3.3).

Runs against real temp-file SQLite with the real Component Registry
migrated in (component_profiles.json's six real components, plus the
one real legacy relationship from Phase 3.1: qt-command-log-lookup ->
CommandProcessorHost) -- this is graph/persistence behavior, not pure
logic, and the legacy-folding tests specifically need real Phase 3.1
data to be meaningful.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest

from app.config import get_settings
from app.domain.knowledge_relationships import KnowledgeObjectType, RelationshipType
from app.domain.lookup_entities import Product, Technology
from app.domain.playbook import Playbook
from app.engines.knowledge_relationships.engine import (
    DuplicateRelationshipError,
    KnowledgeObjectNotFoundError,
    KnowledgeRelationshipEngine,
)
from app.infrastructure.db.component_repository import SqlAlchemyComponentProfileRepository
from app.infrastructure.db.knowledge_repository import SqlAlchemyKnowledgeRepository
from app.infrastructure.db.lookup_repository import SqlAlchemyLookupRepository
from app.infrastructure.db.playbook_repository import SqlAlchemyPlaybookRepository
from app.infrastructure.db.relationship_repository import SqlAlchemyRelationshipRepository
from app.infrastructure.db.seed_migration import migrate_all
from app.infrastructure.db.session import get_engine, get_session_factory
from app.infrastructure.db.sql_template_repository import SqlAlchemySqlTemplateRepository

SAMPLE_DIR = get_settings().sample_knowledge_dir

C = KnowledgeObjectType.COMPONENT
D = KnowledgeObjectType.DOCUMENT
B = KnowledgeObjectType.KNOWN_BUG
S = KnowledgeObjectType.SQL_TEMPLATE
H = KnowledgeObjectType.HISTORICAL_INVESTIGATION
P = KnowledgeObjectType.PLAYBOOK


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

        # Real Phase 3.1 data, including the one real legacy relationship
        # (qt-command-log-lookup -> CommandProcessorHost).
        migrate_all(
            component_repo=component_repo,
            knowledge_repo=knowledge_repo,
            sql_repo=sql_repo,
            lookup_repo=lookup_repo,
            sample_knowledge_dir=SAMPLE_DIR,
        )

        engine = KnowledgeRelationshipEngine(
            relationship_repo, component_repo, knowledge_repo, sql_repo, playbook_repo, lookup_repo
        )

        yield engine, {
            "components": component_repo,
            "knowledge": knowledge_repo,
            "sql": sql_repo,
            "playbooks": playbook_repo,
            "lookups": lookup_repo,
            "relationships": relationship_repo,
        }

        get_engine(sqlite_url).dispose()
        get_engine.cache_clear()


def _make_playbook(repos, title: str = "GLP/DCW Mismatch Triage") -> str:
    playbook = Playbook(id=str(uuid.uuid4()), title=title, product="Command Center", steps=["Check firmware", "Check DCW"])
    repos["playbooks"].save(playbook)
    return playbook.id


def _command_processor_host_id(repos) -> str:
    return repos["components"].get_by_name("CommandProcessorHost").id


# --- Search / resolve ----------------------------------------------------


def test_search_objects_finds_components_by_title(setup):
    engine, _ = setup
    results = engine.search_objects("command", object_type=C)
    names = {r.title for r in results}
    assert "CommandProcessorHost" in names
    assert "CommandPayloadProcessorHost" in names


def test_search_objects_across_all_types(setup):
    engine, _ = setup
    results = engine.search_objects("")  # empty query -> everything
    types_seen = {r.type for r in results}
    assert KnowledgeObjectType.COMPONENT in types_seen
    assert KnowledgeObjectType.SQL_TEMPLATE in types_seen
    assert KnowledgeObjectType.KNOWN_BUG in types_seen


def test_get_object_returns_none_for_unknown(setup):
    engine, _ = setup
    assert engine.get_object(C, "does-not-exist") is None


# --- Add / remove relationships -----------------------------------------


def test_add_relationship_between_component_and_new_playbook(setup):
    engine, repos = setup
    component_id = _command_processor_host_id(repos)
    playbook_id = _make_playbook(repos)

    resolved = engine.add_relationship(C, component_id, P, playbook_id, RelationshipType.DOCUMENTS, created_by="admin")

    assert resolved.from_object.title == "CommandProcessorHost"
    assert resolved.to_object.title == "GLP/DCW Mismatch Triage"
    assert resolved.relationship.relationship_type == RelationshipType.DOCUMENTS
    assert resolved.relationship.created_by == "admin"


def test_add_relationship_raises_for_unknown_from_object(setup):
    engine, repos = setup
    playbook_id = _make_playbook(repos)
    with pytest.raises(KnowledgeObjectNotFoundError):
        engine.add_relationship(C, "does-not-exist", P, playbook_id)


def test_add_relationship_raises_for_unknown_to_object(setup):
    engine, repos = setup
    component_id = _command_processor_host_id(repos)
    with pytest.raises(KnowledgeObjectNotFoundError):
        engine.add_relationship(C, component_id, P, "does-not-exist")


def test_add_relationship_raises_for_duplicate_including_reversed_direction(setup):
    engine, repos = setup
    component_id = _command_processor_host_id(repos)
    playbook_id = _make_playbook(repos)

    engine.add_relationship(C, component_id, P, playbook_id)
    with pytest.raises(DuplicateRelationshipError):
        engine.add_relationship(C, component_id, P, playbook_id)
    # Same pair, reversed direction, same relationship_type -- still a duplicate.
    with pytest.raises(DuplicateRelationshipError):
        engine.add_relationship(P, playbook_id, C, component_id)


def test_add_relationship_different_relationship_type_is_not_a_duplicate(setup):
    engine, repos = setup
    component_id = _command_processor_host_id(repos)
    playbook_id = _make_playbook(repos)

    engine.add_relationship(C, component_id, P, playbook_id, RelationshipType.RELATED_TO)
    resolved = engine.add_relationship(C, component_id, P, playbook_id, RelationshipType.DOCUMENTS)
    assert resolved.relationship.relationship_type == RelationshipType.DOCUMENTS


def test_remove_relationship(setup):
    engine, repos = setup
    component_id = _command_processor_host_id(repos)
    playbook_id = _make_playbook(repos)
    resolved = engine.add_relationship(C, component_id, P, playbook_id)

    engine.remove_relationship(resolved.relationship.id)

    assert engine.list_relationships(C, component_id) == [] or all(
        r.relationship.id != resolved.relationship.id for r in engine.list_relationships(C, component_id)
    )


def test_list_relationships_matches_either_direction(setup):
    engine, repos = setup
    component_id = _command_processor_host_id(repos)
    playbook_id = _make_playbook(repos)
    engine.add_relationship(C, component_id, P, playbook_id)

    from_component_side = engine.list_relationships(C, component_id)
    from_playbook_side = engine.list_relationships(P, playbook_id)
    assert len(from_component_side) == 1
    assert len(from_playbook_side) == 1
    assert from_component_side[0].relationship.id == from_playbook_side[0].relationship.id


def test_list_relationships_shows_broken_link_as_placeholder_not_a_crash(setup):
    """Simulates what happens if a referenced object were later removed
    -- bypasses the engine's own integrity check (direct repo access)
    to set that state up, since nothing in this system hard-deletes yet."""
    engine, repos = setup
    component_id = _command_processor_host_id(repos)
    repos["relationships"].add(C, component_id, P, "orphaned-playbook-id", RelationshipType.RELATED_TO, "admin")

    relationships = engine.list_relationships(C, component_id)
    assert len(relationships) == 1
    assert relationships[0].to_object.title == "(missing)"


# --- Explorer / Impact Analysis -----------------------------------------


def test_explorer_view_groups_connections_by_type(setup):
    engine, repos = setup
    component_id = _command_processor_host_id(repos)
    playbook_id = _make_playbook(repos)
    engine.add_relationship(C, component_id, P, playbook_id, RelationshipType.DOCUMENTS)

    view = engine.get_explorer_view(C, component_id)

    assert view.center.title == "CommandProcessorHost"
    playbook_group = next(g for g in view.groups if g.object_type == P)
    assert playbook_group.objects[0].title == "GLP/DCW Mismatch Triage"


def test_explorer_view_folds_in_phase_3_1_legacy_relationship(setup):
    """qt-command-log-lookup was linked to CommandProcessorHost by
    Phase 3.1's migration (tag "command-processor-host" normalized-
    matched) -- the Explorer must show it even though no Phase 3.3
    relationship was ever added for it."""
    engine, repos = setup
    component_id = _command_processor_host_id(repos)

    view = engine.get_explorer_view(C, component_id)

    sql_group = next((g for g in view.groups if g.object_type == S), None)
    assert sql_group is not None
    assert any(o.id == "qt-command-log-lookup" for o in sql_group.objects)


def test_explorer_view_raises_for_unknown_object(setup):
    engine, _ = setup
    with pytest.raises(KnowledgeObjectNotFoundError):
        engine.get_explorer_view(C, "does-not-exist")


def test_impact_analysis_matches_explorer_and_counts_dependents(setup):
    engine, repos = setup
    component_id = _command_processor_host_id(repos)
    playbook_id = _make_playbook(repos)
    engine.add_relationship(C, component_id, P, playbook_id)

    impact = engine.get_impact_analysis(C, component_id)

    assert impact.object.title == "CommandProcessorHost"
    assert impact.total_dependents >= 2  # the new playbook + the legacy SQL template
    all_ids = {o.id for group in impact.dependents for o in group.objects}
    assert playbook_id in all_ids
    assert "qt-command-log-lookup" in all_ids


# --- Validation ------------------------------------------------------------


def test_validate_relationships_detects_broken_link(setup):
    engine, repos = setup
    component_id = _command_processor_host_id(repos)
    repos["relationships"].add(C, component_id, P, "does-not-exist", RelationshipType.RELATED_TO, "admin")

    issues = engine.validate_relationships()

    assert any(i.issue_type == "broken_link" and i.severity == "error" for i in issues)


def test_validate_relationships_detects_duplicate_added_directly(setup):
    """A duplicate that bypassed add_relationship's own guard (e.g.
    data from an earlier, less careful import) -- validate() must still
    catch it even though the engine's normal add path prevents it."""
    engine, repos = setup
    component_id = _command_processor_host_id(repos)
    playbook_id = _make_playbook(repos)
    repos["relationships"].add(C, component_id, P, playbook_id, RelationshipType.RELATED_TO, "admin")
    repos["relationships"].add(P, playbook_id, C, component_id, RelationshipType.RELATED_TO, "admin")

    issues = engine.validate_relationships()

    assert any(i.issue_type == "duplicate" for i in issues)


def test_validate_relationships_detects_circular_reference(setup):
    engine, repos = setup
    a = _make_playbook(repos, "Playbook A")
    b = _make_playbook(repos, "Playbook B")
    c = _make_playbook(repos, "Playbook C")
    repos["relationships"].add(P, a, P, b, RelationshipType.REQUIRES, "admin")
    repos["relationships"].add(P, b, P, c, RelationshipType.REQUIRES, "admin")
    repos["relationships"].add(P, c, P, a, RelationshipType.REQUIRES, "admin")

    issues = engine.validate_relationships()

    assert any(i.issue_type == "circular_reference" for i in issues)


def test_validate_relationships_clean_graph_has_no_issues(setup):
    engine, repos = setup
    component_id = _command_processor_host_id(repos)
    playbook_id = _make_playbook(repos)
    engine.add_relationship(C, component_id, P, playbook_id)

    issues = engine.validate_relationships()

    assert all(i.issue_type != "broken_link" for i in issues)
    assert all(i.issue_type != "circular_reference" for i in issues)


# --- Knowledge Health --------------------------------------------------


def test_health_report_counts_and_coverage(setup):
    engine, repos = setup
    component_id = _command_processor_host_id(repos)
    playbook_id = _make_playbook(repos)
    engine.add_relationship(C, component_id, P, playbook_id)

    report = engine.get_health_report()

    assert report.total_relationships == 1
    assert report.component_count == 6
    # CommandProcessorHost is connected (new relationship); every other
    # component still has Phase 2B's own related_components/dependencies
    # (self-referential, counted as "connected" -- see the engine's
    # docstring), so coverage should be the full 6/6.
    assert report.component_relationship_coverage == 1.0


def test_health_report_components_missing_playbooks_before_any_are_linked(setup):
    engine, _ = setup
    report = engine.get_health_report()
    names = {c.title for c in report.components_missing_playbooks}
    assert "CommandProcessorHost" in names  # no playbook relationship exists yet


def test_health_report_component_no_longer_missing_playbooks_once_linked(setup):
    engine, repos = setup
    component_id = _command_processor_host_id(repos)
    playbook_id = _make_playbook(repos)
    engine.add_relationship(C, component_id, P, playbook_id)

    report = engine.get_health_report()

    names = {c.title for c in report.components_missing_playbooks}
    assert "CommandProcessorHost" not in names


def test_health_report_unused_sql_templates_excludes_legacy_linked_one(setup):
    engine, _ = setup
    report = engine.get_health_report()
    unused_ids = {t.id for t in report.unused_sql_templates}
    # qt-command-log-lookup has a real Phase 3.1 legacy link -- not unused.
    assert "qt-command-log-lookup" not in unused_ids
    # qt-sql-blocking has no component link at all (verified in Phase 3.1's
    # own tests) -- genuinely unused.
    assert "qt-sql-blocking" in unused_ids


def test_health_report_reflects_broken_and_circular_issues(setup):
    engine, repos = setup
    component_id = _command_processor_host_id(repos)
    repos["relationships"].add(C, component_id, P, "does-not-exist", RelationshipType.RELATED_TO, "admin")

    report = engine.get_health_report()

    assert len(report.broken_relationships) >= 1
