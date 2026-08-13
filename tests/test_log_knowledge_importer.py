"""Tests for the Log Intelligence wiki extractor (deterministic,
no-LLM parsing) and ``LogWikiImporter`` (idempotent, name-keyed upsert
+ deterministic Component Registry matching).

Same real temp-file SQLite pattern as test_task_import.py -- upsert
idempotency and relationship creation are persistence-backed behavior,
not pure logic.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.domain.knowledge_relationships import KnowledgeObjectType, RelationshipType
from app.domain.log_intelligence_kb import LogRepositoryLocation, LogSourceApplication
from app.domain.product_intelligence import ComponentProfile
from app.engines.knowledge.engine import KnowledgeEngine
from app.engines.knowledge_object_framework.adapters import build_adapters
from app.engines.knowledge_object_framework.service import KnowledgeObjectService
from app.engines.knowledge_relationships.engine import KnowledgeRelationshipEngine
from app.engines.log_knowledge.extractor import _split_path, extract, extract_log_entries
from app.engines.log_knowledge.importer import LogWikiImporter
from app.infrastructure.db.component_repository import SqlAlchemyComponentProfileRepository
from app.infrastructure.db.knowledge_repository import SqlAlchemyKnowledgeRepository
from app.infrastructure.db.log_knowledge_repository import SqlAlchemyLogKnowledgeRepository
from app.infrastructure.db.lookup_repository import SqlAlchemyLookupRepository
from app.infrastructure.db.playbook_repository import SqlAlchemyPlaybookRepository
from app.infrastructure.db.relationship_repository import SqlAlchemyRelationshipRepository
from app.infrastructure.db.session import get_engine, get_session_factory
from app.infrastructure.db.sql_template_repository import SqlAlchemySqlTemplateRepository
from app.infrastructure.db.version_repository import SqlAlchemyVersionRepository

T = KnowledgeObjectType

# A small, self-contained wiki-text sample that reproduces the real
# document's structural shapes: a technology-header scenario table
# (with a region switch mid-table), a standalone reference section with
# bare filenames, and a bare filename sitting directly under a generic
# "/var/log/" root (the exact shape that used to be mis-parsed as a
# fake "log" component -- see _split_path's _GENERIC_DIR_NAMES).
_SAMPLE_WIKI_TEXT = """\
RF Mesh workflow and log location

Command
Request
(Outbound)
Logs:
~\\Logs\\CommandProcessor\\logfile.log
~\\Logs\\NMS\\NMS_Listener.log - captures inbound events

NAM Region
Command Response
Logs:
~\\Logs\\CommandProcessor\\logfile.log
/var/log/landisgyr/NMS/NMS_Manager.log

NMS logs:
NMS_Listener.log
NMS_Manager.log

EIC workflow and logs

General
Logs:
/var/log/appos/Kernel/Kernel.log
/var/log/RemoteAccess.log
"""


class _StubKnowledgeStore:
    def upsert(self, collection, record_id, text, title, *, metadata=None):
        pass

    def delete(self, collection, record_id):
        pass

    def count(self, collection):
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
        log_repo = SqlAlchemyLogKnowledgeRepository(session_factory)
        version_repo = SqlAlchemyVersionRepository(session_factory)

        for name in ("CommandProcessorHost", "NMS", "Scheduler"):
            component_repo.save(ComponentProfile(id=f"comp-{name.lower()}", name=name, product="Command Center"))

        relationship_engine = KnowledgeRelationshipEngine(
            relationship_repo, component_repo, knowledge_repo, sql_repo, playbook_repo, lookup_repo, log_repo
        )
        adapters = build_adapters(component_repo, knowledge_repo, sql_repo, playbook_repo, lookup_repo, log_repo)
        knowledge_engine = KnowledgeEngine(_StubKnowledgeStore(), knowledge_repo)
        service = KnowledgeObjectService(adapters, relationship_engine, version_repo, knowledge_engine)
        importer = LogWikiImporter(service, relationship_engine, component_repo)

        yield importer, {
            "service": service,
            "log_repo": log_repo,
            "relationship_engine": relationship_engine,
            "component_repo": component_repo,
        }

        get_engine(sqlite_url).dispose()
        get_engine.cache_clear()


# --- Extractor: path parsing --------------------------------------------


def test_split_path_windows_style():
    root, component, filename = _split_path("~\\Logs\\CommandProcessor\\logfile.log")
    assert component == "CommandProcessor"
    assert filename == "logfile.log"


def test_split_path_generic_log_dir_falls_back_to_none():
    """Regression: '/var/log/RemoteAccess.log' must NOT produce a
    component literally named 'log' -- 'log'/'logs' is a generic OS
    root, not a component-identifying folder."""
    root, component, filename = _split_path("/var/log/RemoteAccess.log")
    assert component is None
    assert filename == "RemoteAccess.log"
    assert root == "var/log"


def test_split_path_bare_filename_has_no_component():
    root, component, filename = _split_path("NMS_Listener.log")
    assert component is None
    assert filename == "NMS_Listener.log"


# --- Extractor: structural parsing over the sample text -----------------


def test_extract_log_entries_tags_technology_region_and_scenario():
    entries = extract_log_entries(_SAMPLE_WIKI_TEXT)
    outbound = [e for e in entries if e.scenario_type == "Command Request (Outbound)"]
    assert len(outbound) == 2
    assert all(e.technology == "RF Mesh" and e.region is None for e in outbound)

    response = [e for e in entries if e.scenario_type == "Command Response"]
    assert len(response) == 2
    assert all(e.region == "NAM" for e in response)


def test_extract_log_entries_captures_inline_note():
    entries = extract_log_entries(_SAMPLE_WIKI_TEXT)
    noted = [e for e in entries if e.note]
    assert any("captures inbound events" in e.note for e in noted)


_SAMPLE_WIKI_TEXT_WITH_FLOW_SUMMARY = """\
RF Mesh workflow and log location

Command
Request
(Outbound)
Command Flow: Web UI  CommandProcessor  CommandPayloadProcessor  EventListener/AUTD  Service
Logs:
~\\Logs\\CommandProcessor\\logfile.log

Command
Response
(Inbound)
Response Flow: ReadingProcessorHost  MsgProcCmdRspHost  EventListener  Network
Logs:
~\\Logs\\MsgProcCmdRspHost\\logfile.log
"""


def test_extract_log_entries_scenario_label_survives_flow_summary_line():
    """Real-document shape: short label fragments ("Command" / "Request"
    / "(Outbound)"), then the scenario's arrow-chain summary line
    ("Command Flow: ..."), then "Logs:". The summary line is always
    longer than the label-fragment threshold and must not be treated as
    ordinary overflow text that wipes the fragments collected just
    before it -- every real scenario in the wiki has exactly this shape,
    and before this fix every one of them silently collapsed to
    "General"."""
    entries = extract_log_entries(_SAMPLE_WIKI_TEXT_WITH_FLOW_SUMMARY)
    outbound = [e for e in entries if e.scenario_type == "Command Request (Outbound)"]
    assert len(outbound) == 1
    assert outbound[0].path_text == "~\\Logs\\CommandProcessor\\logfile.log"

    inbound = [e for e in entries if e.scenario_type == "Command Response (Inbound)"]
    assert len(inbound) == 1
    assert inbound[0].path_text == "~\\Logs\\MsgProcCmdRspHost\\logfile.log"

    assert not any(e.scenario_type == "General" for e in entries)


def test_extract_generic_log_dir_bug_is_fixed_end_to_end():
    """The EIC section's '/var/log/RemoteAccess.log' line must group
    under the EIC technology (its section fallback), never under a
    fake 'log' source."""
    sources, _ = extract(_SAMPLE_WIKI_TEXT, product="Command Center", source_wiki_page="Sample")
    names = {s.name for s in sources}
    assert "log" not in {n.lower() for n in names}
    eic = next(s for s in sources if "RemoteAccess.log" in s.location.filename_patterns)
    assert eic.name == "EIC"


def test_extract_reference_section_groups_bare_filenames_under_one_source():
    sources, _ = extract(_SAMPLE_WIKI_TEXT, product="Command Center", source_wiki_page="Sample")
    nms = next(s for s in sources if s.name == "NMS")
    assert "NMS_Listener.log" in nms.location.filename_patterns
    assert "NMS_Manager.log" in nms.location.filename_patterns


def test_extract_priority_preserves_wiki_order():
    _, scenarios = extract(_SAMPLE_WIKI_TEXT, product="Command Center", source_wiki_page="Sample")
    outbound = next(s for s in scenarios if s.scenario_type == "Command Request (Outbound)")
    priorities = [step.priority for step in outbound.steps]
    assert priorities == sorted(priorities)
    assert priorities[0] == 1
    # CommandProcessor was listed before NMS in the sample -> step 1.
    assert outbound.steps[0].component_name == "CommandProcessor"


def test_extract_explanation_describes_real_extracted_position():
    _, scenarios = extract(_SAMPLE_WIKI_TEXT, product="Command Center", source_wiki_page="Sample")
    outbound = next(s for s in scenarios if s.scenario_type == "Command Request (Outbound)")
    step = outbound.steps[0]
    assert "CommandProcessor" in step.explanation
    assert "Command Request (Outbound)" in step.explanation
    assert "RF Mesh" in step.explanation
    assert "step 1 of" in step.explanation


# --- Context Dimensions phase (2026-08-12): a customer heading must never
# become a component name (approved product decision #6) ------------------


def test_customer_named_section_never_becomes_a_component_name():
    """A wiki heading naming a real customer (here: 'Tepco workflow and
    log location') must not produce a LogSourceApplication/component
    literally named after that customer -- see
    _resolve_component's known_customer_names guard."""
    text = (
        "Tepco workflow and log location\n"
        "Command\n"
        "Request\n"
        "(Outbound)\n"
        "Logs:\n"
        "~\\Logs\\logfile.log\n"
    )
    sources, _ = extract(
        text, product="Command Center", source_wiki_page="Sample", known_customer_names=frozenset({"tepco"})
    )
    names = {s.name for s in sources}
    assert "Tepco" not in names
    assert "Unspecified" in names


def test_customer_named_section_unaffected_when_customer_list_not_supplied():
    """Backward compatible: an empty (default) known_customer_names
    guards nothing, same behavior as before this phase."""
    text = (
        "Tepco workflow and log location\n"
        "Command\n"
        "Request\n"
        "(Outbound)\n"
        "Logs:\n"
        "~\\Logs\\logfile.log\n"
    )
    sources, _ = extract(text, product="Command Center", source_wiki_page="Sample")
    names = {s.name for s in sources}
    assert "Tepco" in names


def test_extract_region_scoped_scenarios_are_distinct_records():
    _, scenarios = extract(_SAMPLE_WIKI_TEXT, product="Command Center", source_wiki_page="Sample")
    keys = {(s.technology, s.scenario_type, s.region) for s in scenarios}
    assert ("RF Mesh", "Command Response", "NAM") in keys
    assert ("RF Mesh", "Command Request (Outbound)", None) in keys


def test_extract_no_duplicate_source_names():
    sources, _ = extract(_SAMPLE_WIKI_TEXT, product="Command Center", source_wiki_page="Sample")
    names = [s.name for s in sources]
    assert len(names) == len(set(names))


# --- Importer: fresh import -----------------------------------------------


def test_import_creates_sources_and_scenarios(setup):
    importer, repos = setup
    sources, scenarios = extract(_SAMPLE_WIKI_TEXT, product="Command Center", source_wiki_page="Sample")
    summary = importer.import_page(sources, scenarios, source_wiki_page="Sample", actor="tester")

    assert summary.sources_created == len(sources)
    assert summary.scenarios_created == len(scenarios)
    assert summary.sources_enriched == 0
    assert summary.scenarios_updated == 0
    assert summary.errors == []
    assert len(repos["log_repo"].list_log_sources()) == len(sources)
    assert len(repos["log_repo"].list_scenarios()) == len(scenarios)


def test_import_matches_deterministic_component_registry(setup):
    """CommandProcessor -> CommandProcessorHost and NMS -> NMS are real,
    deterministic matches against the seeded registry; EIC has no
    registry counterpart and must not get a relationship."""
    importer, repos = setup
    sources, scenarios = extract(_SAMPLE_WIKI_TEXT, product="Command Center", source_wiki_page="Sample")
    summary = importer.import_page(sources, scenarios, source_wiki_page="Sample", actor="tester")

    assert summary.component_relationships_created == 2

    cmd_processor = repos["log_repo"].find_log_source_by_name("CommandProcessor")
    rels = repos["relationship_engine"].list_relationships(T.LOG_SOURCE_APPLICATION, cmd_processor.id)
    assert any(r.relationship.relationship_type == RelationshipType.IMPLEMENTS_LOGGING_FOR for r in rels)

    nms = repos["log_repo"].find_log_source_by_name("NMS")
    rels = repos["relationship_engine"].list_relationships(T.LOG_SOURCE_APPLICATION, nms.id)
    assert any(r.relationship.relationship_type == RelationshipType.IMPLEMENTS_LOGGING_FOR for r in rels)

    eic = repos["log_repo"].find_log_source_by_name("EIC")
    rels = repos["relationship_engine"].list_relationships(T.LOG_SOURCE_APPLICATION, eic.id)
    assert not any(r.relationship.relationship_type == RelationshipType.IMPLEMENTS_LOGGING_FOR for r in rels)


def test_import_scales_to_registry_additions_without_reimport(setup):
    """Refinement #1: the importer queries the live registry fresh on
    every call -- a component registered *after* the wiki import still
    gets linked the next time import_page runs, no importer change
    needed."""
    importer, repos = setup
    sources, scenarios = extract(_SAMPLE_WIKI_TEXT, product="Command Center", source_wiki_page="Sample")
    importer.import_page(sources, scenarios, source_wiki_page="Sample", actor="tester")

    eic = repos["log_repo"].find_log_source_by_name("EIC")
    assert not repos["relationship_engine"].list_relationships(T.LOG_SOURCE_APPLICATION, eic.id)

    repos["component_repo"].save(ComponentProfile(id="comp-eic", name="EIC", product="Command Center"))
    summary2 = importer.import_page(sources, scenarios, source_wiki_page="Sample", actor="tester")

    assert summary2.component_relationships_created == 1
    rels = repos["relationship_engine"].list_relationships(T.LOG_SOURCE_APPLICATION, eic.id)
    assert any(r.relationship.relationship_type == RelationshipType.IMPLEMENTS_LOGGING_FOR for r in rels)


def test_scenario_steps_resolve_to_real_persisted_source_ids(setup):
    """The extractor assigns deterministic ids purely so steps can
    reference their source before persistence -- the importer must
    remap them to the actual service-issued ids."""
    importer, repos = setup
    sources, scenarios = extract(_SAMPLE_WIKI_TEXT, product="Command Center", source_wiki_page="Sample")
    importer.import_page(sources, scenarios, source_wiki_page="Sample", actor="tester")

    for scenario in repos["log_repo"].list_scenarios():
        for step in scenario.steps:
            resolved = repos["log_repo"].get_log_source(step.log_source_id)
            assert resolved is not None
            assert resolved.name == step.component_name


# --- Importer: idempotent re-import / enrichment ---------------------------


def test_reimport_same_page_is_idempotent(setup):
    importer, repos = setup
    sources, scenarios = extract(_SAMPLE_WIKI_TEXT, product="Command Center", source_wiki_page="Sample")
    first = importer.import_page(sources, scenarios, source_wiki_page="Sample", actor="tester")
    second = importer.import_page(sources, scenarios, source_wiki_page="Sample", actor="tester")

    assert second.sources_created == 0
    assert second.sources_enriched == first.sources_created
    assert second.scenarios_created == 0
    assert second.scenarios_updated == first.scenarios_created
    assert second.component_relationships_created == 0  # DuplicateRelationshipError swallowed
    assert len(repos["log_repo"].list_log_sources()) == len(sources)
    assert len(repos["log_repo"].list_scenarios()) == len(scenarios)


def test_reimport_from_different_page_enriches_without_clobbering(setup):
    """A future, more specific wiki page describing NMS's purpose must
    fill the previously-empty field and add its own provenance entry,
    without erasing what the first import already established."""
    importer, repos = setup
    sources, scenarios = extract(_SAMPLE_WIKI_TEXT, product="Command Center", source_wiki_page="Sample")
    importer.import_page(sources, scenarios, source_wiki_page="Sample", actor="tester")

    before = repos["log_repo"].find_log_source_by_name("NMS")
    assert before.purpose is None
    assert before.source_wiki_pages == ["Sample"]
    assert set(before.location.filename_patterns) == {"NMS_Listener.log", "NMS_Manager.log"}

    enrichment = LogSourceApplication(
        id="ignored-by-importer",
        name="NMS",
        location=LogRepositoryLocation(platform="linux", root_path="var/log/landisgyr", raw_paths=[]),
        product="Command Center",
        purpose="Network Management System service",
        source_wiki_pages=["NMS Application Overview"],
    )
    importer.import_page([enrichment], [], source_wiki_page="NMS Application Overview", actor="tester")

    after = repos["log_repo"].find_log_source_by_name("NMS")
    assert after.purpose == "Network Management System service"
    assert set(after.source_wiki_pages) == {"Sample", "NMS Application Overview"}
    # The original import's filenames survive the enrichment merge.
    assert set(after.location.filename_patterns) == {"NMS_Listener.log", "NMS_Manager.log"}


def test_enrichment_does_not_overwrite_existing_non_empty_purpose(setup):
    """A hand-edited/previously-set purpose must not be clobbered by a
    later import that (re-)supplies a different value for the same
    field -- 'enrich, don't clobber'."""
    importer, repos = setup
    first = LogSourceApplication(
        id="ignored",
        name="NMS",
        location=LogRepositoryLocation(platform="linux", root_path="var/log", raw_paths=[]),
        product="Command Center",
        purpose="Original purpose text",
        source_wiki_pages=["Page A"],
    )
    importer.import_page([first], [], source_wiki_page="Page A", actor="tester")

    second = LogSourceApplication(
        id="ignored",
        name="NMS",
        location=LogRepositoryLocation(platform="linux", root_path="var/log", raw_paths=[]),
        product="Command Center",
        purpose="A different purpose text",
        source_wiki_pages=["Page B"],
    )
    importer.import_page([second], [], source_wiki_page="Page B", actor="tester")

    final = repos["log_repo"].find_log_source_by_name("NMS")
    assert final.purpose == "Original purpose text"


# --- Importer: graceful handling of missing fields --------------------------


def test_import_handles_scenario_with_no_region_or_notes(setup):
    importer, repos = setup
    sources, scenarios = extract(_SAMPLE_WIKI_TEXT, product="Command Center", source_wiki_page="Sample")
    importer.import_page(sources, scenarios, source_wiki_page="Sample", actor="tester")

    outbound = repos["log_repo"].find_scenario(
        product="Command Center", technology="RF Mesh", scenario_type="Command Request (Outbound)", region=None
    )
    assert outbound is not None
    assert outbound.region is None
    assert outbound.notes is None


def test_import_one_bad_record_does_not_abort_the_batch(setup, monkeypatch):
    importer, repos = setup
    sources, scenarios = extract(_SAMPLE_WIKI_TEXT, product="Command Center", source_wiki_page="Sample")

    original_create = repos["service"].create
    calls = {"n": 0}

    def flaky_create(object_type, **kwargs):
        if object_type == T.LOG_SOURCE_APPLICATION:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated failure")
        return original_create(object_type, **kwargs)

    monkeypatch.setattr(repos["service"], "create", flaky_create)
    summary = importer.import_page(sources, scenarios, source_wiki_page="Sample", actor="tester")

    assert len(summary.errors) == 1
    assert summary.sources_created == len(sources) - 1
