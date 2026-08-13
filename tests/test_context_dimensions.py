"""Tests for the Context Dimensions phase (2026-08-12): Customer/Region
governed lookup entities, the Technology.parent_technology_id hierarchy,
and the real-data technology/region seeding fix.

Runs against a real temp-file SQLite database (same pattern as
test_knowledge_foundation_migration.py) -- this is persistence/wiring
behavior, not pure logic.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest

from app.domain.knowledge_relationships import KnowledgeObjectType
from app.domain.log_intelligence_kb import LogCollectionScenario, LogRepositoryLocation, LogSourceApplication
from app.domain.lookup_entities import Customer, Region, Technology
from app.engines.knowledge_object_framework.adapters import build_adapters
from app.infrastructure.db.component_repository import SqlAlchemyComponentProfileRepository
from app.infrastructure.db.knowledge_repository import SqlAlchemyKnowledgeRepository
from app.infrastructure.db.log_knowledge_repository import SqlAlchemyLogKnowledgeRepository
from app.infrastructure.db.lookup_repository import SqlAlchemyLookupRepository
from app.infrastructure.db.playbook_repository import SqlAlchemyPlaybookRepository
from app.infrastructure.db.seed_migration import migrate_lookup_entities
from app.infrastructure.db.session import get_engine, get_session_factory


@pytest.fixture
def session_factory():
    with tempfile.TemporaryDirectory() as tmp:
        sqlite_url = f"sqlite:///{(Path(tmp) / 'test.db').as_posix()}"
        yield get_session_factory(sqlite_url)
        get_engine(sqlite_url).dispose()
        get_engine.cache_clear()


@pytest.fixture
def lookup_repo(session_factory):
    return SqlAlchemyLookupRepository(session_factory)


# --- Customer CRUD -----------------------------------------------------


def test_customer_round_trips_with_provenance(lookup_repo):
    customer = Customer(
        id=str(uuid.uuid4()),
        name="TEPCO",
        aliases=["Tepco", "Tokyo Electric Power Company"],
        verified=True,
        source_type="document_title_corpus",
        source_reference="5 real documents in Knowledge Center",
    )
    lookup_repo.save_customer(customer)

    fetched = lookup_repo.get_customer(customer.id)
    assert fetched is not None
    assert fetched.name == "TEPCO"
    assert fetched.aliases == ["Tepco", "Tokyo Electric Power Company"]
    assert fetched.verified is True
    assert fetched.source_type == "document_title_corpus"


def test_get_customer_by_name_matches_alias(lookup_repo):
    customer = Customer(id=str(uuid.uuid4()), name="TEPCO", aliases=["Tokyo Electric Power Company"])
    lookup_repo.save_customer(customer)

    by_alias = lookup_repo.get_customer_by_name("Tokyo Electric Power Company")
    assert by_alias is not None
    assert by_alias.id == customer.id

    by_alias_case_insensitive = lookup_repo.get_customer_by_name("tokyo electric power company")
    assert by_alias_case_insensitive is not None
    assert by_alias_case_insensitive.id == customer.id


def test_get_customer_by_name_no_match_returns_none(lookup_repo):
    assert lookup_repo.get_customer_by_name("Nonexistent Customer") is None


# --- Region CRUD ---------------------------------------------------------


def test_region_round_trips(lookup_repo):
    region = Region(id=str(uuid.uuid4()), name="Guam", aliases=[])
    lookup_repo.save_region(region)
    fetched = lookup_repo.get_region_by_name("Guam")
    assert fetched is not None
    assert fetched.name == "Guam"


# --- Technology hierarchy -------------------------------------------------


def test_technology_parent_technology_id_round_trips(lookup_repo):
    parent = Technology(id=str(uuid.uuid4()), name="RF Mesh")
    lookup_repo.save_technology(parent)
    child = Technology(id=str(uuid.uuid4()), name="RF Mesh IP", parent_technology_id=parent.id)
    lookup_repo.save_technology(child)

    fetched_child = lookup_repo.get_technology_by_name("RF Mesh IP")
    assert fetched_child is not None
    assert fetched_child.parent_technology_id == parent.id


# --- Knowledge Object Framework wiring (Customer/Region as governed types) -


def test_customer_and_region_are_wired_into_adapters(session_factory, lookup_repo):
    component_repo = SqlAlchemyComponentProfileRepository(session_factory)
    knowledge_repo = SqlAlchemyKnowledgeRepository(session_factory)
    from app.infrastructure.db.sql_template_repository import SqlAlchemySqlTemplateRepository

    sql_repo = SqlAlchemySqlTemplateRepository(session_factory)
    playbook_repo = SqlAlchemyPlaybookRepository(session_factory)

    adapters = build_adapters(component_repo, knowledge_repo, sql_repo, playbook_repo, lookup_repo)
    assert KnowledgeObjectType.CUSTOMER in adapters
    assert KnowledgeObjectType.REGION in adapters

    customer = Customer(id=str(uuid.uuid4()), name="ATCO")
    adapters[KnowledgeObjectType.CUSTOMER].save(customer)
    ref = adapters[KnowledgeObjectType.CUSTOMER].to_ref(customer)
    assert ref.title == "ATCO"
    assert ref.type == KnowledgeObjectType.CUSTOMER


# --- Real-data technology/region seeding fix ------------------------------


def test_migrate_lookup_entities_seeds_technology_from_log_knowledge_not_only_blank_documentation(
    session_factory, lookup_repo
):
    """Real bug fix: before this phase, Technology was only seeded from
    documentation.technology (a column that's almost always blank), so
    a real, populated Log Intelligence Knowledge Base produced zero
    governed Technology rows. This must no longer be true."""
    component_repo = SqlAlchemyComponentProfileRepository(session_factory)
    knowledge_repo = SqlAlchemyKnowledgeRepository(session_factory)
    log_knowledge_repo = SqlAlchemyLogKnowledgeRepository(session_factory)

    source = LogSourceApplication(
        id="log-src-test",
        name="TestComponent",
        location=LogRepositoryLocation(platform="windows", root_path="~\\Logs"),
        technology=["RF Mesh IP"],
    )
    log_knowledge_repo.save_log_source(source)
    scenario = LogCollectionScenario(
        id="log-scenario-test",
        product="Command Center",
        technology="RF Mesh",
        scenario_type="Command Request (Outbound)",
        region="NAM",
        source_wiki_page="Sample",
    )
    log_knowledge_repo.save_scenario(scenario)

    # Before the fix: documentation.technology is blank everywhere, so
    # this would have created 0 Technology rows and 0 Region rows.
    created = migrate_lookup_entities(lookup_repo, component_repo, knowledge_repo, log_knowledge_repo)
    assert created > 0

    technology_names = {t.name for t in lookup_repo.list_technologies()}
    assert "RF Mesh" in technology_names
    assert "RF Mesh IP" in technology_names
    region_names = {r.name for r in lookup_repo.list_regions()}
    assert "NAM" in region_names


def test_migrate_lookup_entities_applies_real_technology_hierarchy(session_factory, lookup_repo):
    """RF Mesh IP / RF Mesh (DAS implementation) must be linked to their
    real parent 'RF Mesh' once both rows exist -- the direct data-level
    fix for 'RF Mesh and Mesh IP are not automatically the same thing'."""
    component_repo = SqlAlchemyComponentProfileRepository(session_factory)
    knowledge_repo = SqlAlchemyKnowledgeRepository(session_factory)
    log_knowledge_repo = SqlAlchemyLogKnowledgeRepository(session_factory)

    for name in ["RF Mesh", "RF Mesh IP", "RF Mesh (DAS implementation)", "Wi-Sun"]:
        log_knowledge_repo.save_log_source(
            LogSourceApplication(
                id=f"log-src-{name}",
                name=f"Component-{name}",
                location=LogRepositoryLocation(platform="windows", root_path="~\\Logs"),
                technology=[name],
            )
        )

    migrate_lookup_entities(lookup_repo, component_repo, knowledge_repo, log_knowledge_repo)

    rf_mesh = lookup_repo.get_technology_by_name("RF Mesh")
    rf_mesh_ip = lookup_repo.get_technology_by_name("RF Mesh IP")
    rf_mesh_das = lookup_repo.get_technology_by_name("RF Mesh (DAS implementation)")
    wi_sun = lookup_repo.get_technology_by_name("Wi-Sun")

    assert rf_mesh.parent_technology_id is None
    assert rf_mesh_ip.parent_technology_id == rf_mesh.id
    assert rf_mesh_das.parent_technology_id == rf_mesh.id
    assert wi_sun.parent_technology_id is None  # unrelated technology, never linked to anything


def test_migrate_lookup_entities_is_idempotent(session_factory, lookup_repo):
    component_repo = SqlAlchemyComponentProfileRepository(session_factory)
    knowledge_repo = SqlAlchemyKnowledgeRepository(session_factory)
    log_knowledge_repo = SqlAlchemyLogKnowledgeRepository(session_factory)
    log_knowledge_repo.save_log_source(
        LogSourceApplication(
            id="log-src-idem",
            name="Comp",
            location=LogRepositoryLocation(platform="windows", root_path="~\\Logs"),
            technology=["RF Mesh"],
        )
    )

    first = migrate_lookup_entities(lookup_repo, component_repo, knowledge_repo, log_knowledge_repo)
    second = migrate_lookup_entities(lookup_repo, component_repo, knowledge_repo, log_knowledge_repo)
    assert first > 0
    assert second == 0
    assert len(lookup_repo.list_technologies()) == 1
