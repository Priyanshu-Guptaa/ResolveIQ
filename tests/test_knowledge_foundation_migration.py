"""Tests for Sprint 3 Phase 3.1 -- Knowledge Foundation & Data Model
Migration.

Runs the real migration against the real sample seed files (the same
ones the app ships with) into a real temp-file SQLite database -- this
is persistence/integration behavior, not pure logic, so fakes would just
re-assert the mock (see test_investigation_engine.py's docstring for the
same reasoning).

Relationship-linking tests document the *honest* outcome, not a wished-
for one: today's sample data was authored before the Component Registry
existed, so several association tables legitimately import empty. That
is asserted explicitly below, not glossed over.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.config import get_settings
from app.domain.recommendation import KnowledgeMatch
from app.engines.knowledge.engine import KnowledgeEngine
from app.engines.product_intelligence.component_registry import ComponentRegistry
from app.engines.sql_library.engine import SqlLibraryEngine
from app.infrastructure.db.component_repository import SqlAlchemyComponentProfileRepository
from app.infrastructure.db.knowledge_repository import SqlAlchemyKnowledgeRepository
from app.infrastructure.db.seed_migration import migrate_all
from app.infrastructure.db.session import get_engine, get_session_factory
from app.infrastructure.db.sql_template_repository import SqlAlchemySqlTemplateRepository

SAMPLE_DIR = get_settings().sample_knowledge_dir


class FakeKnowledgeStore:
    """Structurally satisfies the KnowledgeStore Protocol -- records every
    upsert instead of hitting a real ChromaDB/embedding model, exactly
    like test_recommendation_engine.py's fake."""

    def __init__(self) -> None:
        self.upserts: list[tuple] = []

    def upsert(self, collection, record_id, text, title, metadata) -> None:
        self.upserts.append((collection, record_id, text, title, metadata))

    def query(self, collection, text, top_k: int = 5) -> list[KnowledgeMatch]:  # pragma: no cover
        return []

    def count(self, collection) -> int:
        return len([u for u in self.upserts if u[0] == collection])

    def list_recent(self, collection, limit: int = 5) -> list[KnowledgeMatch]:  # pragma: no cover
        return []

    def delete(self, collection, record_id: str) -> None:
        self.upserts = [u for u in self.upserts if not (u[0] == collection and u[1] == record_id)]


@pytest.fixture
def repos():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        sqlite_url = f"sqlite:///{db_path.as_posix()}"
        session_factory = get_session_factory(sqlite_url)

        component_repo = SqlAlchemyComponentProfileRepository(session_factory)
        knowledge_repo = SqlAlchemyKnowledgeRepository(session_factory)
        sql_repo = SqlAlchemySqlTemplateRepository(session_factory)

        yield component_repo, knowledge_repo, sql_repo

        get_engine(sqlite_url).dispose()
        get_engine.cache_clear()


def _migrate(repos):
    component_repo, knowledge_repo, sql_repo = repos
    return migrate_all(
        component_repo=component_repo,
        knowledge_repo=knowledge_repo,
        sql_repo=sql_repo,
        sample_knowledge_dir=SAMPLE_DIR,
    )


# --- Import correctness ------------------------------------------------


def test_migration_imports_expected_counts(repos):
    report = _migrate(repos)
    assert report == {
        "component_profiles": 6,
        "known_bugs": 4,
        "historical_investigations": 8,
        "documentation": 7,
        "sql_templates": 5,
    }


def test_migration_is_idempotent(repos):
    first = _migrate(repos)
    second = _migrate(repos)

    assert second == {
        "component_profiles": 0,
        "known_bugs": 0,
        "historical_investigations": 0,
        "documentation": 0,
        "sql_templates": 0,
    }

    component_repo, knowledge_repo, sql_repo = repos
    assert component_repo.count() == first["component_profiles"]
    assert knowledge_repo.count_known_bugs() == first["known_bugs"]
    assert knowledge_repo.count_historical_investigations() == first["historical_investigations"]
    assert knowledge_repo.count_documentation() == first["documentation"]
    assert sql_repo.count() == first["sql_templates"]


def test_migration_running_three_times_never_duplicates(repos):
    _migrate(repos)
    _migrate(repos)
    _migrate(repos)
    component_repo, knowledge_repo, sql_repo = repos
    assert component_repo.count() == 6
    assert knowledge_repo.count_known_bugs() == 4
    assert knowledge_repo.count_historical_investigations() == 8
    assert sql_repo.count() == 5


# --- Domain Intelligence fidelity (Phase 2B fields survive the move) ---


def test_component_profile_migrated_with_full_domain_intelligence_fields(repos):
    _migrate(repos)
    component_repo, _, _ = repos
    profile = component_repo.get_by_name("CommandProcessorHost")

    assert profile is not None
    assert profile.product == "Command Center"
    assert "CommandPayloadProcessorHost" in profile.related_components
    assert profile.dependencies == ["Scheduler", "MeterDeviceRegistry"]
    assert profile.version_differences[0].version == "9.1"
    assert profile.version_differences[1].version == "9.2"
    assert any("stale" in failure.lower() for failure in profile.typical_failures)
    assert profile.common_sql
    assert profile.playbooks
    # Governance fields present and stamped by the migration.
    assert profile.created_by == "system:migration"
    assert profile.is_active is True


# --- Relationship integrity (foreign keys, not free text) ---------------


def test_known_bug_related_components_link_is_real_but_empty_for_todays_sample_data(repos):
    """Honest result, not a bug: known_bugs.json's affected_components
    (order-service, billing-events-consumer-group, checkout-api,
    collector-service, mesh-routers, ...) were authored before the
    Component Registry existed and none of them names one of its six
    Command Center components -- so the known_bug_components FK table
    correctly imports empty. affected_components itself is untouched."""
    _migrate(repos)
    _, knowledge_repo, _ = repos
    bugs = knowledge_repo.list_known_bugs()

    assert len(bugs) == 4
    for bug in bugs:
        assert bug.related_components == []
        assert bug.affected_components  # original free text preserved


def test_sql_template_related_components_resolves_the_one_real_match(repos):
    """qt-command-log-lookup is tagged "command-processor-host" --
    normalizing both sides (strip hyphens, lowercase) matches it to the
    real "CommandProcessorHost" component. No other seeded template's
    tags happen to name a component."""
    _migrate(repos)
    _, _, sql_repo = repos
    templates = {t.id: t for t in sql_repo.list_all()}

    assert templates["qt-command-log-lookup"].related_components == ["CommandProcessorHost"]
    assert templates["qt-sql-blocking"].related_components == []
    assert templates["qt-meter-comm-history"].related_components == []
    assert templates["qt-investigation-by-correlation"].related_components == []
    assert templates["qt-firmware-dcw-mismatch"].related_components == []


def test_historical_investigation_related_components_empty_for_todays_sample_data(repos):
    """None of the eight seeded historical investigations' titles or
    descriptions literally mention a Command Center component name
    (hist-004 is meter/collector-related but never names GapRecon, NMS,
    Scheduler, etc. specifically) -- so this link table correctly
    imports empty too. The matching mechanism itself is exercised
    positively by the SQL template test above."""
    _migrate(repos)
    _, knowledge_repo, _ = repos
    investigations = knowledge_repo.list_historical_investigations()

    assert len(investigations) == 8
    assert all(inv.related_components == [] for inv in investigations)


def test_component_self_referential_related_components_graph_is_consistent(repos):
    """ComponentProfile.related_components (Phase 2B) is a mix of real
    component references and, for NMS, one deliberate external
    reference ("Collectors", not a registered component). This test
    locks in that exact, known shape -- catching a typo introduced
    later, without falsely demanding every relationship resolve to a
    component when some are intentionally external (see the RFC-003
    "Assumptions Challenged" note on why this isn't a hard FK
    constraint)."""
    _migrate(repos)
    component_repo, _, _ = repos
    profiles = {p.name: p for p in component_repo.list_all()}
    known_names = set(profiles.keys())

    fully_internal = ["CommandProcessorHost", "CommandPayloadProcessorHost", "MsgProcCmdRspHost", "Scheduler", "GapRecon"]
    for name in fully_internal:
        unresolved = [r for r in profiles[name].related_components if r not in known_names]
        assert unresolved == [], f"{name} has an unexpected unresolved relationship: {unresolved}"

    nms_unresolved = [r for r in profiles["NMS"].related_components if r not in known_names]
    assert nms_unresolved == ["Collectors"], "NMS's known external reference changed unexpectedly"


# --- Downstream consumers read transparently from the database ---------


def test_component_registry_loads_from_repository_with_same_shape_as_json(repos):
    _migrate(repos)
    component_repo, _, _ = repos
    registry = ComponentRegistry.load_from_repository(component_repo)

    names = {p.name for p in registry.list_all()}
    assert names == {
        "CommandProcessorHost",
        "CommandPayloadProcessorHost",
        "MsgProcCmdRspHost",
        "Scheduler",
        "GapRecon",
        "NMS",
    }
    assert registry.get("CommandProcessorHost").product == "Command Center"
    assert registry.search("command")[0].name in ("CommandProcessorHost", "CommandPayloadProcessorHost")


def test_sql_library_engine_reads_migrated_templates(repos):
    _migrate(repos)
    _, _, sql_repo = repos
    engine = SqlLibraryEngine(sql_repo)
    templates = engine.list_templates()

    assert len(templates) == 5
    assert {t.id for t in templates} == {
        "qt-sql-blocking",
        "qt-command-log-lookup",
        "qt-meter-comm-history",
        "qt-investigation-by-correlation",
        "qt-firmware-dcw-mismatch",
    }


def test_knowledge_engine_seeds_chroma_from_the_database_not_json(repos):
    """The Recommendation Engine's search still works after migration
    because KnowledgeEngine.seed_from_directory() sources known bugs,
    historical investigations, *and* (since Phase 3.2) documentation
    from the repository -- this verifies that sourcing, independent of
    RecommendationEngine's own scoring logic (unchanged, covered by
    test_recommendation_engine.py)."""
    _migrate(repos)
    component_repo, knowledge_repo, _ = repos

    fake_store = FakeKnowledgeStore()
    engine = KnowledgeEngine(fake_store, knowledge_repo)
    loaded = engine.seed_from_directory(SAMPLE_DIR)

    assert loaded["known_bugs"] == 4
    assert loaded["historical_investigations"] == 8
    assert loaded["documentation"] == 7  # all 7 sample docs are Published post-migration

    # Spot-check one upserted record's content actually came from the DB row.
    from app.domain.enums import KnowledgeCollection

    known_bug_upserts = [u for u in fake_store.upserts if u[0] == KnowledgeCollection.KNOWN_BUGS]
    assert any("order-service" in text for _, _, text, _, _ in known_bug_upserts)

    doc_upserts = [u for u in fake_store.upserts if u[0] == KnowledgeCollection.DOCUMENTATION]
    assert len(doc_upserts) == 7
    assert any("blocking" in text.lower() for _, _, text, _, _ in doc_upserts)


def test_knowledge_engine_constructed_without_repository_still_works():
    """Backward compatibility: existing callers/tests that construct
    KnowledgeEngine(store) with no repository (e.g.
    test_recommendation_engine.py) must keep working unchanged --
    search never touches the repository."""
    store = FakeKnowledgeStore()
    engine = KnowledgeEngine(store)  # no repository argument
    assert engine.list_known_bugs() == []
    assert engine.search_known_bugs("anything") == []
