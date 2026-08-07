"""Tests for the Component Registry (Phase 2B, incremental).

Covers both the real seed file (data/sample_knowledge/component_profiles.json
-- catches a malformed JSON edit before it reaches the app) and the
registry's lookup logic against small in-memory fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import get_settings
from app.domain.product_intelligence import ComponentProfile, VersionDifference
from app.engines.product_intelligence.component_registry import ComponentRegistry
from app.engines.product_intelligence.engine import ProductIntelligenceEngine

REQUIRED_COMPONENTS = {
    "CommandProcessorHost",
    "CommandPayloadProcessorHost",
    "MsgProcCmdRspHost",
    "Scheduler",
    "GapRecon",
    "NMS",
}


def test_seed_file_loads_all_required_components():
    settings = get_settings()
    registry = ComponentRegistry.load_from_file(settings.sample_knowledge_dir / "component_profiles.json")

    names = {p.name for p in registry.list_all()}
    assert REQUIRED_COMPONENTS <= names


def test_seed_file_every_profile_has_the_full_field_set():
    settings = get_settings()
    registry = ComponentRegistry.load_from_file(settings.sample_knowledge_dir / "component_profiles.json")

    for profile in registry.list_all():
        # Not every field must be non-empty, but every field must exist
        # and be the right shape -- a malformed entry shouldn't silently
        # lose a field. Covers both the original documentation fields
        # and the Domain Intelligence relationship/operational fields.
        assert isinstance(profile.responsibilities, list)
        assert isinstance(profile.related_components, list)
        assert isinstance(profile.dependencies, list)
        assert isinstance(profile.consumes, list)
        assert isinstance(profile.produces, list)
        assert isinstance(profile.related_services, list)
        assert isinstance(profile.related_queues, list)
        assert isinstance(profile.database_tables, list)
        assert isinstance(profile.configuration, list)
        assert isinstance(profile.events, list)
        assert isinstance(profile.message_flows, list)
        assert isinstance(profile.data_flows, list)
        assert isinstance(profile.version_differences, list)
        assert all(isinstance(vd, VersionDifference) for vd in profile.version_differences)
        assert isinstance(profile.typical_failures, list)
        assert isinstance(profile.required_logs, list)
        assert isinstance(profile.common_sql, list)
        assert isinstance(profile.known_bugs, list)
        assert isinstance(profile.documentation_links, list)
        assert isinstance(profile.playbooks, list)
        assert isinstance(profile.historical_investigations, list)
        assert profile.product


def test_seed_file_relationship_graph_is_internally_consistent():
    """Every related_components/dependencies reference that names another
    seeded component must match a real profile name -- catches a typo
    that would silently break the Architecture Explorer's click-through
    navigation (which matches on exact name equality)."""
    settings = get_settings()
    registry = ComponentRegistry.load_from_file(settings.sample_knowledge_dir / "component_profiles.json")
    known_names = {p.name for p in registry.list_all()}

    # Not every relationship target is a component in this registry (some
    # are external services) -- so this only checks the well-known
    # component-to-component edges that were part of the original seed.
    profile = registry.get("CommandProcessorHost")
    assert profile is not None
    assert set(profile.related_components) <= known_names | {"Collectors"}


def test_version_difference_requires_version_and_change():
    vd = VersionDifference(version="9.2", change="DCW validation moved upstream")
    assert vd.version == "9.2"
    assert vd.change == "DCW validation moved upstream"


def test_missing_seed_file_returns_empty_registry_not_an_error(tmp_path: Path):
    registry = ComponentRegistry.load_from_file(tmp_path / "does_not_exist.json")
    assert registry.list_all() == []


@pytest.fixture
def sample_registry() -> ComponentRegistry:
    return ComponentRegistry(
        [
            ComponentProfile(id="a", name="CommandProcessorHost", product="Command Center", responsibilities=["x"]),
            ComponentProfile(id="b", name="Scheduler", product="Command Center"),
        ]
    )


def test_get_returns_exact_match(sample_registry: ComponentRegistry):
    profile = sample_registry.get("CommandProcessorHost")
    assert profile is not None
    assert profile.responsibilities == ["x"]


def test_get_unknown_component_returns_none(sample_registry: ComponentRegistry):
    assert sample_registry.get("NotARealComponent") is None


def test_search_is_case_insensitive_substring(sample_registry: ComponentRegistry):
    results = sample_registry.search("command")
    assert [p.name for p in results] == ["CommandProcessorHost"]


def test_search_empty_query_returns_everything(sample_registry: ComponentRegistry):
    assert len(sample_registry.search("")) == 2


def test_engine_delegates_to_registry(sample_registry: ComponentRegistry):
    engine = ProductIntelligenceEngine(sample_registry)

    assert len(engine.list_components()) == 2
    assert engine.get_component("Scheduler") is not None
    assert engine.get_component("Scheduler").name == "Scheduler"
    assert len(engine.search_components("sched")) == 1
