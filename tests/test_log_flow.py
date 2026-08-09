"""Tests for Log Intelligence's cross-file search and Command Flow
reconstruction (app/engines/log_intelligence/flow.py) -- pure logic,
no DB needed, since both operate purely on an already-hydrated
InvestigationSession plus canned Log Intelligence Knowledge Base data.
"""

from __future__ import annotations

from app.domain.entities import ExtractedEntity, LogEvent
from app.domain.enums import EntityType, EvidenceType, LogLevel
from app.domain.evidence import Evidence
from app.domain.investigation import InvestigationSession
from app.domain.log_intelligence_kb import (
    LogCollectionScenario,
    LogCollectionStep,
    LogRepositoryLocation,
    LogSourceApplication,
)
from app.engines.log_intelligence.flow import reconstruct_flow, search_logs


def _entity(entity_type: EntityType, value: str) -> ExtractedEntity:
    return ExtractedEntity(entity_type=entity_type, value=value)


def _event(message: str, *, level: LogLevel = LogLevel.INFO, entities: list[ExtractedEntity] | None = None) -> LogEvent:
    return LogEvent(raw_line=message, message=message, level=level, entities=entities or [])


def _log_evidence(title: str, events: list[LogEvent]) -> Evidence:
    investigation_id = "inv-1"
    return Evidence(investigation_id=investigation_id, evidence_type=EvidenceType.LOG_FILE, title=title, log_events=events)


def _investigation(evidence: list[Evidence]) -> InvestigationSession:
    investigation = InvestigationSession(title="Test investigation")
    for e in evidence:
        investigation.add_evidence(e)
    return investigation


class FakeLogKnowledgeRepo:
    def __init__(self, sources: list[LogSourceApplication], scenarios: list[LogCollectionScenario]) -> None:
        self._sources = sources
        self._scenarios = scenarios

    def list_log_sources(self, *, active_only: bool = True):
        return self._sources

    def list_scenarios(self, *, active_only: bool = True):
        return self._scenarios


def _make_source(name: str, filename_patterns: list[str] | None = None) -> LogSourceApplication:
    return LogSourceApplication(
        id=f"src-{name.lower()}",
        name=name,
        location=LogRepositoryLocation(platform="windows", root_path="~/Logs", filename_patterns=filename_patterns or ["logfile.log"]),
        product="Command Center",
    )


def _make_step(component_name: str, priority: int) -> LogCollectionStep:
    return LogCollectionStep(
        log_source_id=f"src-{component_name.lower()}", component_name=component_name, priority=priority,
        explanation=f"Produced by {component_name} -- step {priority}.",
    )


def _make_scenario(scenario_type: str, steps: list[LogCollectionStep]) -> LogCollectionScenario:
    return LogCollectionScenario(
        id=f"sc-{scenario_type.lower().replace(' ', '-')}", product="Command Center", technology="RF Mesh",
        scenario_type=scenario_type, steps=steps, source_wiki_page="Test",
    )


# --- search_logs --------------------------------------------------------


def test_search_by_entity_matches_across_files():
    e1 = _log_evidence("A.log", [_event("cmd sent", entities=[_entity(EntityType.COMMAND_LOG_ID, "CL-1")])])
    e2 = _log_evidence("B.log", [_event("response received", entities=[_entity(EntityType.COMMAND_LOG_ID, "CL-1")])])
    e3 = _log_evidence("C.log", [_event("unrelated", entities=[_entity(EntityType.COMMAND_LOG_ID, "CL-2")])])
    investigation = _investigation([e1, e2, e3])

    result = search_logs(investigation, entity_type="command_log_id", entity_value="CL-1")
    assert result.total_matches == 2
    assert {h.evidence_title for h in result.hits} == {"A.log", "B.log"}


def test_search_by_keyword_is_case_insensitive():
    e1 = _log_evidence("A.log", [_event("Timeout waiting for ACK")])
    investigation = _investigation([e1])
    result = search_logs(investigation, keyword="timeout")
    assert result.total_matches == 1


def test_search_by_level_filters():
    e1 = _log_evidence("A.log", [_event("info line", level=LogLevel.INFO), _event("error line", level=LogLevel.ERROR)])
    investigation = _investigation([e1])
    result = search_logs(investigation, level="ERROR")
    assert result.total_matches == 1
    assert result.hits[0].message == "error line"


def test_search_filters_are_anded_together():
    e1 = _log_evidence(
        "A.log",
        [
            _event("Timeout on command", level=LogLevel.ERROR, entities=[_entity(EntityType.METER_NUMBER, "M1")]),
            _event("Timeout on command", level=LogLevel.INFO, entities=[_entity(EntityType.METER_NUMBER, "M1")]),
        ],
    )
    investigation = _investigation([e1])
    result = search_logs(investigation, keyword="timeout", level="ERROR", entity_type="meter_number", entity_value="M1")
    assert result.total_matches == 1


def test_search_ignores_non_log_file_evidence():
    note = Evidence(investigation_id="inv-1", evidence_type=EvidenceType.MANUAL_NOTE, title="note", raw_content="cmd sent")
    investigation = _investigation([note])
    result = search_logs(investigation, keyword="cmd")
    assert result.total_matches == 0


def test_search_with_no_filters_returns_query_summary_placeholder():
    e1 = _log_evidence("A.log", [_event("line")])
    investigation = _investigation([e1])
    result = search_logs(investigation)
    assert result.query_summary == "(no filter)"
    assert result.total_matches == 1


# --- reconstruct_flow ----------------------------------------------------


def test_flow_orders_outbound_scenario_by_wiki_step_priority():
    sources = [_make_source("CommandProcessor"), _make_source("CommandPayloadProcessor")]
    scenario = _make_scenario(
        "Command Request (Outbound)",
        [_make_step("CommandProcessor", 1), _make_step("CommandPayloadProcessor", 2)],
    )
    repo = FakeLogKnowledgeRepo(sources, [scenario])

    e1 = _log_evidence("CommandProcessor/logfile.log", [_event("sent", entities=[_entity(EntityType.COMMAND_LOG_ID, "CL-1")])])
    e2 = _log_evidence(
        "CommandPayloadProcessor/logfile.log", [_event("forwarded", entities=[_entity(EntityType.COMMAND_LOG_ID, "CL-1")])]
    )
    investigation = _investigation([e1, e2])

    flow = reconstruct_flow(investigation, entity_type="command_log_id", entity_value="CL-1", log_knowledge_repo=repo)

    assert flow.scenario_type == "Command Request (Outbound)"
    assert [s.component_name for s in flow.outbound_steps] == ["CommandProcessor", "CommandPayloadProcessor"]
    assert flow.inbound_steps == []
    assert all(s.has_log_entry for s in flow.outbound_steps)
    assert flow.matched_component_count == 2


def test_flow_labels_inbound_scenario_correctly():
    sources = [_make_source("MsgProcCmdRspHost")]
    scenario = _make_scenario("Command Response (Inbound)", [_make_step("MsgProcCmdRspHost", 1)])
    repo = FakeLogKnowledgeRepo(sources, [scenario])

    e1 = _log_evidence(
        "MsgProcCmdRspHost/logfile.log", [_event("response", entities=[_entity(EntityType.COMMAND_LOG_ID, "CL-9")])]
    )
    investigation = _investigation([e1])

    flow = reconstruct_flow(investigation, entity_type="command_log_id", entity_value="CL-9", log_knowledge_repo=repo)

    assert flow.inbound_steps[0].direction == "inbound"
    assert flow.outbound_steps == []


def test_flow_flags_gap_when_wiki_step_has_no_matching_log_entry():
    sources = [_make_source("CommandProcessor"), _make_source("MsgProcCmdRspHost")]
    scenario = _make_scenario(
        "Command Request (Outbound)", [_make_step("CommandProcessor", 1), _make_step("MsgProcCmdRspHost", 2)]
    )
    repo = FakeLogKnowledgeRepo(sources, [scenario])

    # Only CommandProcessor's log was uploaded -- MsgProcCmdRspHost never
    # got a matching entry, even though the wiki says it should.
    e1 = _log_evidence(
        "CommandProcessor/logfile.log", [_event("sent", entities=[_entity(EntityType.COMMAND_LOG_ID, "CL-1")])]
    )
    investigation = _investigation([e1])

    flow = reconstruct_flow(investigation, entity_type="command_log_id", entity_value="CL-1", log_knowledge_repo=repo)

    steps_by_name = {s.component_name: s for s in flow.outbound_steps}
    assert steps_by_name["CommandProcessor"].has_log_entry is True
    assert steps_by_name["MsgProcCmdRspHost"].has_log_entry is False
    assert steps_by_name["MsgProcCmdRspHost"].events == []


def test_flow_unresolved_events_when_source_file_not_recognized():
    repo = FakeLogKnowledgeRepo([], [])
    e1 = _log_evidence("mystery.log", [_event("something", entities=[_entity(EntityType.COMMAND_LOG_ID, "CL-1")])])
    investigation = _investigation([e1])

    flow = reconstruct_flow(investigation, entity_type="command_log_id", entity_value="CL-1", log_knowledge_repo=repo)

    assert flow.matched_component_count == 0
    assert flow.scenario_type is None
    assert len(flow.unresolved_events) == 1


def test_flow_component_matched_but_not_in_any_scenario_is_unresolved():
    sources = [_make_source("Orphan")]
    repo = FakeLogKnowledgeRepo(sources, [])  # no scenarios at all
    e1 = _log_evidence("Orphan/logfile.log", [_event("x", entities=[_entity(EntityType.COMMAND_LOG_ID, "CL-1")])])
    investigation = _investigation([e1])

    flow = reconstruct_flow(investigation, entity_type="command_log_id", entity_value="CL-1", log_knowledge_repo=repo)

    assert flow.scenario_id is None
    assert len(flow.unresolved_events) == 1
    assert flow.matched_component_count == 1


def test_flow_no_matches_returns_empty_result():
    repo = FakeLogKnowledgeRepo([], [])
    e1 = _log_evidence("A.log", [_event("no correlating id here")])
    investigation = _investigation([e1])

    flow = reconstruct_flow(investigation, entity_type="command_log_id", entity_value="CL-999", log_knowledge_repo=repo)

    assert flow.matched_component_count == 0
    assert flow.outbound_steps == []
    assert flow.inbound_steps == []
    assert flow.unresolved_events == []


def test_flow_generic_filename_alone_does_not_identify_component():
    """Same conservative discipline as the Recommendation Engine's
    already-collected check -- a bare "logfile.log" can't identify
    which of several sources produced it."""
    sources = [_make_source("CommandProcessor", filename_patterns=["logfile.log"])]
    repo = FakeLogKnowledgeRepo(sources, [])
    e1 = _log_evidence("logfile.log", [_event("x", entities=[_entity(EntityType.COMMAND_LOG_ID, "CL-1")])])
    investigation = _investigation([e1])

    flow = reconstruct_flow(investigation, entity_type="command_log_id", entity_value="CL-1", log_knowledge_repo=repo)

    assert flow.matched_component_count == 0
    assert len(flow.unresolved_events) == 1


def test_flow_prefers_scenario_that_explains_the_most_matched_components():
    sources = [_make_source("CommandProcessor"), _make_source("MsgProcCmdRspHost")]
    scenario_both = _make_scenario(
        "Command Request (Outbound)", [_make_step("CommandProcessor", 1), _make_step("MsgProcCmdRspHost", 2)]
    )
    scenario_one_only = LogCollectionScenario(
        id="sc-other", product="Command Center", technology="RF Mesh IP",
        scenario_type="Firmware Download", steps=[_make_step("CommandProcessor", 1)], source_wiki_page="Test",
    )
    repo = FakeLogKnowledgeRepo(sources, [scenario_one_only, scenario_both])

    e_a = _log_evidence(
        "CommandProcessor/logfile.log", [_event("x", entities=[_entity(EntityType.COMMAND_LOG_ID, "CL-1")])]
    )
    e_b = _log_evidence(
        "MsgProcCmdRspHost/logfile.log", [_event("y", entities=[_entity(EntityType.COMMAND_LOG_ID, "CL-1")])]
    )
    investigation = _investigation([e_a, e_b])

    flow = reconstruct_flow(investigation, entity_type="command_log_id", entity_value="CL-1", log_knowledge_repo=repo)

    # scenario_both explains both matched components (score 2) vs
    # scenario_one_only (score 1).
    assert flow.scenario_id == scenario_both.id
