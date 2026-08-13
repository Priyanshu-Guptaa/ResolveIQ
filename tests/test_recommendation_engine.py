"""Unit tests for the Recommendation Engine.

Uses a fake KnowledgeStore (structurally satisfying the KnowledgeStore
Protocol) so these tests run with no ChromaDB / embedding model
dependency -- fast and fully offline.
"""

from __future__ import annotations

from app.config import Settings
from app.domain.enums import EvidenceType, KnowledgeCollection
from app.domain.evidence import Evidence
from app.domain.investigation import InvestigationSession
from app.domain.recommendation import KnowledgeMatch, RootCauseHypothesis
from app.engines.knowledge.engine import KnowledgeEngine
from app.engines.log_intelligence.entity_extractor import RegexEntityExtractor
from app.engines.recommendation.engine import RecommendationEngine


class FakeKnowledgeStore:
    """Structurally satisfies the KnowledgeStore Protocol with canned results."""

    def __init__(self, canned: dict[KnowledgeCollection, list[KnowledgeMatch]] | None = None) -> None:
        self._canned = canned or {}

    def upsert(self, collection, record_id, text, title, metadata) -> None:  # pragma: no cover
        pass

    def query(self, collection: KnowledgeCollection, text: str, top_k: int = 5) -> list[KnowledgeMatch]:
        return self._canned.get(collection, [])[:top_k]

    def count(self, collection: KnowledgeCollection) -> int:
        return len(self._canned.get(collection, []))


def _investigation_with_evidence(text: str) -> InvestigationSession:
    investigation = InvestigationSession(title="Checkout failing for some customers")
    extractor = RegexEntityExtractor()
    evidence = Evidence(
        investigation_id=investigation.id,
        evidence_type=EvidenceType.LOG_FILE,
        raw_content=text,
        extracted_entities=extractor.extract(text),
    )
    investigation.add_evidence(evidence)
    return investigation


def test_strong_historical_match_drives_root_cause_and_next_step():
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hist-007",
        title="NullPointerException in order-service during checkout",
        snippet="...",
        score=0.82,
        metadata={
            "root_cause": "Missing null check on customer shipping address.",
            "next_step": "Search for the exception across all order-service instances.",
        },
    )
    store = FakeKnowledgeStore({KnowledgeCollection.HISTORICAL_INVESTIGATIONS: [match]})
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = _investigation_with_evidence(
        "java.lang.NullPointerException: shippingAddress is null"
    )
    recommendation = engine.generate(investigation)

    assert recommendation.overall_confidence == 0.82
    assert recommendation.root_causes[0].description == "Missing null check on customer shipping address."
    assert recommendation.next_best_step == "Search for the exception across all order-service instances."
    assert recommendation.similar_investigations[0].title == match.title


def test_weak_match_falls_back_to_entity_heuristic():
    weak_match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hist-999",
        title="Unrelated investigation",
        snippet="...",
        score=0.1,  # below min_similarity_for_root_cause
        metadata={"root_cause": "Something unrelated."},
    )
    store = FakeKnowledgeStore({KnowledgeCollection.HISTORICAL_INVESTIGATIONS: [weak_match]})
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = _investigation_with_evidence(
        "java.lang.NullPointerException: shippingAddress is null"
    )
    recommendation = engine.generate(investigation)

    descriptions = [rc.description for rc in recommendation.root_causes]
    assert any("NullPointerException" in d for d in descriptions)
    assert "Something unrelated." not in descriptions


def test_matches_with_no_captured_root_cause_are_not_surfaced_as_a_finding():
    """Regression guard for a real bug: bulk-imported ServiceNow tickets
    (app/engines/task_import) have no distinct root-cause field in their
    source data, so their `root_cause` is deliberately left empty rather
    than filled with a boilerplate placeholder -- an earlier version did
    use a placeholder sentence, and when two such tickets both ranked as
    top matches, the Investigation Workspace showed the identical
    boilerplate sentence twice as if it were two distinct diagnosed root
    causes. A high-scoring match with an empty root_cause must be
    skipped entirely, not surfaced as an empty/fabricated finding."""
    matches = [
        KnowledgeMatch(
            collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
            record_id="hist-imported-1",
            title="CC - Check Kafka Consumer Group Lag",
            snippet="...",
            score=0.85,
            metadata={"root_cause": ""},
        ),
        KnowledgeMatch(
            collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
            record_id="hist-imported-2",
            title="CC - Check Drive Usage",
            snippet="...",
            score=0.80,
            metadata={"root_cause": ""},
        ),
    ]
    store = FakeKnowledgeStore({KnowledgeCollection.HISTORICAL_INVESTIGATIONS: matches})
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = _investigation_with_evidence("Kafka consumer group lag spike on billing-events")
    recommendation = engine.generate(investigation)

    assert recommendation.root_causes == []
    # The matches themselves are still useful precedent -- they must
    # still appear in Historical Matches even though they contribute no
    # root-cause hypothesis.
    assert [m.title for m in recommendation.similar_investigations] == [
        "CC - Check Kafka Consumer Group Lag",
        "CC - Check Drive Usage",
    ]


def test_next_best_step_truncates_a_long_historical_resolution():
    """Regression guard for a real bug: `resolution` on a bulk-imported
    ServiceNow ticket is the ticket's entire "Comments and Work notes"
    column -- every work-note entry ever added, potentially thousands
    of characters. Dumping that verbatim into "Next best step" (meant
    to be a short, scannable hint) made a real match read as an
    unreadable wall of text, easily mistaken for "unrelated task
    information" leaking in. The full text must be capped and marked
    truncated, not echoed in full."""
    long_resolution = "Work note entry. " * 200  # 3,600 chars, well over the cap
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hist-long",
        title="A resolved ticket with a long work-note history",
        snippet="...",
        score=0.9,
        metadata={"resolution": long_resolution, "root_cause": ""},
    )
    store = FakeKnowledgeStore({KnowledgeCollection.HISTORICAL_INVESTIGATIONS: [match]})
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = _investigation_with_evidence("some log content")
    recommendation = engine.generate(investigation)

    assert len(recommendation.next_best_step) < len(long_resolution)
    assert recommendation.next_best_step.endswith("…")


def test_root_cause_description_is_truncated_for_a_long_historical_root_cause():
    long_root_cause = "Detailed root cause narrative. " * 100  # well over the cap
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hist-long-rc",
        title="A resolved ticket with a long root cause writeup",
        snippet="...",
        score=0.9,
        metadata={"root_cause": long_root_cause},
    )
    store = FakeKnowledgeStore({KnowledgeCollection.HISTORICAL_INVESTIGATIONS: [match]})
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = _investigation_with_evidence("some log content")
    recommendation = engine.generate(investigation)

    assert len(recommendation.root_causes[0].description) < len(long_root_cause)
    assert recommendation.root_causes[0].description.endswith("…")


def test_no_evidence_prompts_for_more_information():
    store = FakeKnowledgeStore()
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = InvestigationSession(title="Empty investigation")
    recommendation = engine.generate(investigation)

    assert recommendation.root_causes == []
    assert "task description" in recommendation.next_best_step.lower()


def test_sql_session_entity_yields_suggested_sql():
    store = FakeKnowledgeStore()
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = _investigation_with_evidence("Blocking detected, spid=61 is head blocker")
    recommendation = engine.generate(investigation)

    assert any("61" in sql for sql in recommendation.suggested_sql)


# --- Log Intelligence: recommended log collection ---------------------------


class FakeLogKnowledgeRepo:
    """Structurally satisfies the LogKnowledgeRepository Protocol's
    read side with canned scenarios/sources -- no DB needed for these
    pure matching/labeling tests."""

    def __init__(self, scenarios, sources) -> None:
        self._scenarios = scenarios
        self._sources = {s.id: s for s in sources}

    def list_scenarios(self, *, active_only: bool = True):
        return self._scenarios

    def get_log_source(self, source_id: str):
        return self._sources.get(source_id)


def _make_scenario(id_, technology, scenario_type, steps):
    from app.domain.log_intelligence_kb import LogCollectionScenario

    return LogCollectionScenario(
        id=id_, product="Command Center", technology=technology, scenario_type=scenario_type,
        steps=steps, source_wiki_page="Test Page",
    )


def _make_step(log_source_id, component_name, priority):
    from app.domain.log_intelligence_kb import LogCollectionStep

    return LogCollectionStep(
        log_source_id=log_source_id, component_name=component_name, priority=priority,
        explanation=f"Produced by {component_name} -- step {priority}.",
    )


def _make_source(id_, name):
    from app.domain.log_intelligence_kb import LogRepositoryLocation, LogSourceApplication

    return LogSourceApplication(
        id=id_, name=name,
        location=LogRepositoryLocation(platform="linux", root_path="/var/log/landisgyr", filename_patterns=[f"{name}.log"]),
        product="Command Center",
    )


def _investigation_with_log_evidence(context_text: str, log_filenames: list[str] | None = None) -> InvestigationSession:
    """Like _investigation_with_evidence, but can also attach LOG_FILE
    evidence items with specific *filenames* (titles) -- the signal
    "already collected" detection cross-references against. The task
    description alone (no title) still drives technology/issue-type
    text matching."""
    investigation = _investigation_with_evidence(context_text)
    for filename in log_filenames or []:
        investigation.add_evidence(
            Evidence(
                investigation_id=investigation.id,
                evidence_type=EvidenceType.LOG_FILE,
                source="upload",
                title=filename,
                raw_content="(log content not relevant to this test)",
            )
        )
    return investigation


def test_full_technology_and_issue_type_match_yields_critical_ordered_items():
    scenario = _make_scenario(
        "sc-1", "RF Mesh", "Command Request (Outbound)",
        [_make_step("src-nms", "NMS", 2), _make_step("src-cp", "CommandProcessor", 1)],
    )
    sources = [_make_source("src-nms", "NMS"), _make_source("src-cp", "CommandProcessor")]
    log_repo = FakeLogKnowledgeRepo([scenario], sources)
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    # Mentions the full technology name AND the full issue-type phrase.
    investigation = _investigation_with_evidence("Command Request (Outbound) failed during RF Mesh delivery")
    recommendation = engine.generate(investigation)

    assert len(recommendation.recommended_logs) == 2
    assert all(item.priority_label == "Critical" for item in recommendation.recommended_logs)
    # Wiki collection order preserved regardless of the steps list's own order.
    assert [item.order for item in recommendation.recommended_logs] == [1, 2]
    assert recommendation.recommended_logs[0].component_name == "CommandProcessor"
    assert recommendation.recommended_logs[0].repository_root_path == "/var/log/landisgyr"


def test_technology_only_match_yields_recommended_not_critical():
    """Issue-aware matching (polishing phase): technology alone is no
    longer enough for Critical -- the specific issue type must match
    too, otherwise this is just one of several possible operations for
    that technology."""
    scenario = _make_scenario("sc-1", "RF Mesh", "Firmware Download", [_make_step("src-a", "A", 1)])
    log_repo = FakeLogKnowledgeRepo([scenario], [_make_source("src-a", "A")])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    investigation = _investigation_with_evidence("RF Mesh meter is unreachable, need to investigate")
    recommendation = engine.generate(investigation)

    assert len(recommendation.recommended_logs) == 1
    assert recommendation.recommended_logs[0].priority_label == "Recommended"


def test_unrelated_technology_is_not_recommended():
    scenario = _make_scenario("sc-1", "Kafka broker", "General", [_make_step("src-k", "Broker", 1)])
    log_repo = FakeLogKnowledgeRepo([scenario], [_make_source("src-k", "Broker")])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    investigation = _investigation_with_evidence("Command failed during RF Mesh outbound delivery")
    recommendation = engine.generate(investigation)

    assert recommendation.recommended_logs == []


def test_partial_keyword_match_yields_optional_label():
    scenario = _make_scenario("sc-1", "RF Mesh IP", "General", [_make_step("src-x", "X", 1)])
    log_repo = FakeLogKnowledgeRepo([scenario], [_make_source("src-x", "X")])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    # Mentions "Mesh" but not the full "RF Mesh IP" phrase.
    investigation = _investigation_with_evidence("Mesh device unreachable after firmware update")
    recommendation = engine.generate(investigation)

    assert len(recommendation.recommended_logs) == 1
    assert recommendation.recommended_logs[0].priority_label == "Optional"


def test_best_matching_issue_type_is_critical_sibling_scenario_is_recommended():
    outbound = _make_scenario("sc-1", "RF Mesh", "Command Request (Outbound)", [_make_step("src-a", "A", 1)])
    response = _make_scenario("sc-2", "RF Mesh", "Command Response", [_make_step("src-b", "B", 1)])
    log_repo = FakeLogKnowledgeRepo([outbound, response], [_make_source("src-a", "A"), _make_source("src-b", "B")])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    # Full "Command Response" phrase present; "Command Request (Outbound)"
    # is not (its words overlap, but not as one contiguous phrase).
    investigation = _investigation_with_evidence("RF Mesh Command Response never arrived after the outbound request")
    recommendation = engine.generate(investigation)

    labels = {item.scenario_type: item.priority_label for item in recommendation.recommended_logs}
    assert labels["Command Response"] == "Critical"
    assert labels["Command Request (Outbound)"] == "Recommended"


def test_no_log_knowledge_repo_yields_no_recommended_logs():
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings())
    investigation = _investigation_with_evidence("RF Mesh command timed out")
    recommendation = engine.generate(investigation)
    assert recommendation.recommended_logs == []


def test_empty_investigation_yields_no_recommended_logs():
    scenario = _make_scenario("sc-1", "RF Mesh", "General", [_make_step("src-x", "X", 1)])
    log_repo = FakeLogKnowledgeRepo([scenario], [_make_source("src-x", "X")])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    recommendation = engine.generate(InvestigationSession(title="Empty investigation"))
    assert recommendation.recommended_logs == []


# --- Log Intelligence: match_reason ------------------------------------------


def test_match_reason_names_both_technology_and_issue_type_when_both_match():
    scenario = _make_scenario("sc-1", "RF Mesh", "Command Response", [_make_step("src-a", "A", 1)])
    log_repo = FakeLogKnowledgeRepo([scenario], [_make_source("src-a", "A")])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    investigation = _investigation_with_evidence("RF Mesh Command Response is failing")
    recommendation = engine.generate(investigation)

    reason = recommendation.recommended_logs[0].match_reason
    assert "RF Mesh" in reason
    assert "Command Response" in reason


def test_match_reason_notes_unidentified_issue_type_when_only_technology_matches():
    scenario = _make_scenario("sc-1", "RF Mesh", "Firmware Download", [_make_step("src-a", "A", 1)])
    log_repo = FakeLogKnowledgeRepo([scenario], [_make_source("src-a", "A")])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    investigation = _investigation_with_evidence("RF Mesh meter is unreachable")
    recommendation = engine.generate(investigation)

    reason = recommendation.recommended_logs[0].match_reason
    assert "RF Mesh" in reason
    assert "wasn't identified" in reason


# --- Log Intelligence: already-collected detection ---------------------------


def test_already_collected_true_when_component_name_appears_in_uploaded_filename():
    scenario = _make_scenario("sc-1", "RF Mesh", "General", [_make_step("src-cp", "CommandProcessor", 1)])
    log_repo = FakeLogKnowledgeRepo([scenario], [_make_source("src-cp", "CommandProcessor")])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    investigation = _investigation_with_log_evidence(
        "RF Mesh command issue", log_filenames=["CommandProcessor/logfile.log"]
    )
    recommendation = engine.generate(investigation)

    assert recommendation.recommended_logs[0].already_collected is True


def test_already_collected_false_when_no_matching_evidence_uploaded():
    scenario = _make_scenario("sc-1", "RF Mesh", "General", [_make_step("src-cp", "CommandProcessor", 1)])
    log_repo = FakeLogKnowledgeRepo([scenario], [_make_source("src-cp", "CommandProcessor")])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    investigation = _investigation_with_log_evidence("RF Mesh command issue", log_filenames=["NMS_Listener.log"])
    recommendation = engine.generate(investigation)

    assert recommendation.recommended_logs[0].already_collected is False


def test_already_collected_not_inferred_from_a_generic_filename_shared_by_many_sources():
    """A bare 'logfile.log' upload can't distinguish CommandProcessor
    from any of the dozens of other sources that share that exact
    filename in the wiki -- must not be marked collected on that alone."""
    scenario = _make_scenario("sc-1", "RF Mesh", "General", [_make_step("src-cp", "CommandProcessor", 1)])
    source = _make_source("src-cp", "CommandProcessor")
    source.location.filename_patterns = ["logfile.log"]
    log_repo = FakeLogKnowledgeRepo([scenario], [source])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    investigation = _investigation_with_log_evidence("RF Mesh command issue", log_filenames=["logfile.log"])
    recommendation = engine.generate(investigation)

    assert recommendation.recommended_logs[0].already_collected is False


# --- Log Intelligence: manual technology browse ------------------------------


def test_logs_for_technology_returns_every_scenario_for_that_technology_regardless_of_issue_text():
    """The manual counterpart to _recommend_logs(): issue-type text is
    irrelevant here on purpose -- covers the real gap automatic
    matching can't (e.g. a "Data Extract" issue when the Knowledge Base
    has no Data Extract scenario_type at all)."""
    outbound = _make_scenario("sc-1", "RF Mesh IP", "Command Request (Outbound)", [_make_step("src-a", "A", 1)])
    inbound = _make_scenario("sc-2", "RF Mesh IP", "Command Response (Inbound)", [_make_step("src-b", "B", 1)])
    other_tech = _make_scenario("sc-3", "Wi-Sun", "Events", [_make_step("src-c", "C", 1)])
    log_repo = FakeLogKnowledgeRepo(
        [outbound, inbound, other_tech], [_make_source("src-a", "A"), _make_source("src-b", "B"), _make_source("src-c", "C")]
    )
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    investigation = _investigation_with_evidence("Interval Data Extract for all meters is empty")
    items = engine.logs_for_technology(investigation, "RF Mesh IP")

    assert {item.component_name for item in items} == {"A", "B"}
    assert all(item.priority_label == "Browsed" for item in items)


def test_logs_for_technology_matches_case_insensitively():
    scenario = _make_scenario("sc-1", "RF Mesh IP", "General", [_make_step("src-a", "A", 1)])
    log_repo = FakeLogKnowledgeRepo([scenario], [_make_source("src-a", "A")])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    items = engine.logs_for_technology(_investigation_with_evidence("x"), "rf mesh ip")

    assert len(items) == 1


def test_logs_for_technology_empty_for_unknown_technology():
    scenario = _make_scenario("sc-1", "RF Mesh IP", "General", [_make_step("src-a", "A", 1)])
    log_repo = FakeLogKnowledgeRepo([scenario], [_make_source("src-a", "A")])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    items = engine.logs_for_technology(_investigation_with_evidence("x"), "Gas Meter")

    assert items == []


def test_logs_for_technology_empty_when_no_log_knowledge_repo():
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), None)

    items = engine.logs_for_technology(_investigation_with_evidence("x"), "RF Mesh IP")

    assert items == []


def test_available_log_technologies_lists_distinct_real_technologies():
    scenarios = [
        _make_scenario("sc-1", "RF Mesh IP", "Command Request (Outbound)", [_make_step("src-a", "A", 1)]),
        _make_scenario("sc-2", "RF Mesh IP", "Command Response (Inbound)", [_make_step("src-a", "A", 1)]),
        _make_scenario("sc-3", "Wi-Sun", "General", [_make_step("src-a", "A", 1)]),
    ]
    log_repo = FakeLogKnowledgeRepo(scenarios, [_make_source("src-a", "A")])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    assert engine.available_log_technologies() == ["RF Mesh IP", "Wi-Sun"]


def test_available_log_technologies_empty_when_no_log_knowledge_repo():
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), None)

    assert engine.available_log_technologies() == []


def test_infer_technology_prefers_the_more_specific_tied_match():
    """Regression test for a real reported bug: "RF Mesh" and "RF Mesh
    IP" both score a full 1.0 keyword_match_score against text
    containing "Technology: RF Mesh IP" (the word-boundary check after
    "Mesh" is satisfied by the space before "IP"), so a real ATCO RF
    Mesh IP investigation had its live TFS/Wiki search anchored on the
    vaguer "RF Mesh" purely because it happened to be listed first --
    pulling in genuinely unrelated generic RF Mesh tickets instead of
    RF-Mesh-IP-specific ones. On a tied score, the more specific
    technology must win, regardless of list order -- resolved via the
    real Technology.parent_technology_id hierarchy (see
    _match_single_technology, 2026-08-13), not technology-name length;
    production always wires a LookupRepository alongside the log
    knowledge repository (app/api/dependencies.py), so that's the
    configuration under test here."""
    scenarios = [
        _make_scenario("sc-1", "RF Mesh", "Command Request (Outbound)", [_make_step("src-a", "A", 1)]),
        _make_scenario("sc-2", "RF Mesh IP", "Command Request (Outbound)", [_make_step("src-a", "A", 1)]),
    ]
    log_repo = FakeLogKnowledgeRepo(scenarios, [_make_source("src-a", "A")])
    lookup_repo = FakeLookupRepoForTechnology(_rf_mesh_family())
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo, lookup_repo=lookup_repo)
    investigation = _investigation_with_evidence(
        "Organization Name: ATCO. Technology: RF Mesh IP. Defect: Interval Data Extract for All meters is Empty."
    )

    assert engine._infer_technology(investigation) == "RF Mesh IP"


def test_infer_technology_list_order_reversed_still_prefers_specific_match():
    """Same as above with the two technologies registered in the
    opposite order, to prove the fix isn't accidentally still
    depending on iteration order."""
    scenarios = [
        _make_scenario("sc-1", "RF Mesh IP", "Command Request (Outbound)", [_make_step("src-a", "A", 1)]),
        _make_scenario("sc-2", "RF Mesh", "Command Request (Outbound)", [_make_step("src-a", "A", 1)]),
    ]
    log_repo = FakeLogKnowledgeRepo(scenarios, [_make_source("src-a", "A")])
    lookup_repo = FakeLookupRepoForTechnology(_rf_mesh_family())
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo, lookup_repo=lookup_repo)
    investigation = _investigation_with_evidence("Technology: RF Mesh IP.")

    assert engine._infer_technology(investigation) == "RF Mesh IP"


def test_infer_technology_no_lookup_repo_cannot_resolve_tie_returns_none():
    """Without a LookupRepository (no hierarchy data available at
    all), a genuine tie between "RF Mesh" and "RF Mesh IP" can no
    longer be resolved by guessing (e.g. by string length) -- the
    fixed rule returns None rather than a possibly-wrong answer. This
    differs from the two tests above only in omitting lookup_repo;
    production never omits it (both are always wired together, see
    app/api/dependencies.py), so this documents the honest fallback
    behavior for the incomplete configuration, not the production
    path."""
    scenarios = [
        _make_scenario("sc-1", "RF Mesh", "Command Request (Outbound)", [_make_step("src-a", "A", 1)]),
        _make_scenario("sc-2", "RF Mesh IP", "Command Request (Outbound)", [_make_step("src-a", "A", 1)]),
    ]
    log_repo = FakeLogKnowledgeRepo(scenarios, [_make_source("src-a", "A")])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)
    investigation = _investigation_with_evidence("Technology: RF Mesh IP.")

    assert engine._infer_technology(investigation) is None


# --- Log Intelligence: Product Intelligence integration ----------------------


class FakeRelationshipEngine:
    def __init__(self, relationships_by_source_id: dict) -> None:
        self._rels = relationships_by_source_id

    def list_relationships(self, object_type, object_id):
        return self._rels.get(object_id, [])


def _resolved_implements_logging_for(source_id: str, component_id: str, component_name: str):
    from app.domain.knowledge_relationships import (
        KnowledgeObjectRef,
        KnowledgeObjectType,
        KnowledgeRelationship,
        RelationshipType,
        ResolvedRelationship,
    )

    return ResolvedRelationship(
        relationship=KnowledgeRelationship(
            id="rel-1",
            from_type=KnowledgeObjectType.LOG_SOURCE_APPLICATION,
            from_id=source_id,
            to_type=KnowledgeObjectType.COMPONENT,
            to_id=component_id,
            relationship_type=RelationshipType.IMPLEMENTS_LOGGING_FOR,
        ),
        from_object=KnowledgeObjectRef(type=KnowledgeObjectType.LOG_SOURCE_APPLICATION, id=source_id, title="CommandProcessor"),
        to_object=KnowledgeObjectRef(type=KnowledgeObjectType.COMPONENT, id=component_id, title=component_name),
    )


def test_linked_component_surfaced_when_relationship_exists():
    scenario = _make_scenario("sc-1", "RF Mesh", "General", [_make_step("src-cp", "CommandProcessor", 1)])
    log_repo = FakeLogKnowledgeRepo([scenario], [_make_source("src-cp", "CommandProcessor")])
    rel_engine = FakeRelationshipEngine(
        {"src-cp": [_resolved_implements_logging_for("src-cp", "comp-1", "CommandProcessorHost")]}
    )
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo, component_repo=None, relationship_engine=rel_engine
    )

    investigation = _investigation_with_evidence("RF Mesh command issue")
    recommendation = engine.generate(investigation)

    item = recommendation.recommended_logs[0]
    assert item.linked_component_id == "comp-1"
    assert item.linked_component_name == "CommandProcessorHost"


def test_linked_component_none_when_no_relationship_exists():
    scenario = _make_scenario("sc-1", "RF Mesh", "General", [_make_step("src-cp", "CommandProcessor", 1)])
    log_repo = FakeLogKnowledgeRepo([scenario], [_make_source("src-cp", "CommandProcessor")])
    rel_engine = FakeRelationshipEngine({})
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo, component_repo=None, relationship_engine=rel_engine
    )

    investigation = _investigation_with_evidence("RF Mesh command issue")
    recommendation = engine.generate(investigation)

    item = recommendation.recommended_logs[0]
    assert item.linked_component_id is None
    assert item.linked_component_name is None


def test_linked_component_none_when_no_relationship_engine_provided():
    scenario = _make_scenario("sc-1", "RF Mesh", "General", [_make_step("src-cp", "CommandProcessor", 1)])
    log_repo = FakeLogKnowledgeRepo([scenario], [_make_source("src-cp", "CommandProcessor")])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    investigation = _investigation_with_evidence("RF Mesh command issue")
    recommendation = engine.generate(investigation)

    assert recommendation.recommended_logs[0].linked_component_id is None


# --- Investigation Strategy (Recommendation Engine V2) ----------------------


class FakeComponentRepo:
    def __init__(self, components: list) -> None:
        self._components = components

    def list_all(self, *, active_only: bool = True):
        return self._components


class FakeSqlLibrary:
    def __init__(self, templates: list) -> None:
        self._templates = templates

    def list_templates(self):
        return self._templates


def _make_component(id_, name, product="Command Center"):
    from app.domain.product_intelligence import ComponentProfile

    return ComponentProfile(id=id_, name=name, product=product)


def _make_query_template(id_, title, related_components, sql_text="SELECT 1;", explanation="test"):
    from app.domain.sql_studio import QueryTemplate

    return QueryTemplate(
        id=id_, title=title, category="test", sql_text=sql_text, explanation=explanation,
        related_components=related_components,
    )


def test_strategy_is_always_populated_even_for_empty_investigation():
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings())
    recommendation = engine.generate(InvestigationSession(title="Empty investigation"))

    assert recommendation.strategy is not None
    assert recommendation.strategy.current_stage.value == "triage"
    assert recommendation.strategy.progress == 0.0


def test_strategy_tfs_and_wiki_matches_stay_none_without_external_knowledge_service():
    """Backward compatibility: RecommendationEngine still works exactly
    as before when no ExternalKnowledgeService is injected -- the
    field stays None, not an empty/failed result."""
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings())
    recommendation = engine.generate(InvestigationSession(title="Some investigation"))

    assert recommendation.strategy.tfs_matches is None
    assert recommendation.strategy.wiki_matches is None


def test_strategy_populates_tfs_and_wiki_matches_when_service_provided():
    from app.domain.external_knowledge import ExternalKnowledgeResult, ExternalSource

    class FakeExternalKnowledgeService:
        def gather(self, investigation, entities, matched_component, technology):
            return (
                ExternalKnowledgeResult(source=ExternalSource.TFS, available=True, query_summary="test"),
                ExternalKnowledgeResult(source=ExternalSource.WIKI, available=False, error="Wiki is not configured."),
            )

    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), external_knowledge=FakeExternalKnowledgeService()
    )
    recommendation = engine.generate(InvestigationSession(title="Some investigation"))

    assert recommendation.strategy.tfs_matches.available is True
    assert recommendation.strategy.wiki_matches.available is False
    assert recommendation.strategy.wiki_matches.error == "Wiki is not configured."


def test_strategy_stage_is_evidence_collection_when_logs_missing():
    from app.domain.log_intelligence_kb import LogCollectionScenario, LogCollectionStep, LogRepositoryLocation, LogSourceApplication

    scenario = LogCollectionScenario(
        id="sc-1", product="Command Center", technology="RF Mesh", scenario_type="General",
        steps=[LogCollectionStep(log_source_id="src-a", component_name="A", priority=1, explanation="x")],
        source_wiki_page="Test",
    )
    source = LogSourceApplication(
        id="src-a", name="A", location=LogRepositoryLocation(platform="linux", root_path="/var/log", filename_patterns=["a.log"]),
        product="Command Center",
    )
    log_repo = FakeLogKnowledgeRepo([scenario], [source])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    investigation = _investigation_with_evidence("RF Mesh issue here")
    recommendation = engine.generate(investigation)

    assert recommendation.strategy.current_stage.value == "evidence_collection"
    assert len(recommendation.strategy.missing_evidence) >= 1
    assert 0.0 <= recommendation.strategy.progress < 0.5


def test_strategy_stage_is_root_cause_identified_on_strong_historical_match():
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS, record_id="hist-1", title="Past issue", snippet="...",
        score=0.9, metadata={"root_cause": "Known firmware bug."},
    )
    store = FakeKnowledgeStore({KnowledgeCollection.HISTORICAL_INVESTIGATIONS: [match]})
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = _investigation_with_evidence("Something broke")
    recommendation = engine.generate(investigation)

    assert recommendation.strategy.current_stage.value == "root_cause_identified"
    assert recommendation.strategy.progress == 1.0


def test_strategy_reuses_same_objects_as_legacy_fields_not_copies():
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS, record_id="hist-1", title="Past issue", snippet="...",
        score=0.9, metadata={"root_cause": "Known firmware bug."},
    )
    store = FakeKnowledgeStore({KnowledgeCollection.HISTORICAL_INVESTIGATIONS: [match]})
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = _investigation_with_evidence("Something broke")
    recommendation = engine.generate(investigation)

    # Pydantic wraps each list field in a fresh list on construction, but
    # reuses already-valid sub-model instances without revalidating them --
    # so item-level identity is the correct "not recomputed" check.
    assert recommendation.strategy.historical_investigations[0] is recommendation.similar_investigations[0]
    assert recommendation.strategy.recommended_next_action == recommendation.next_best_step


def test_decision_checkpoint_none_with_no_root_cause_candidates():
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings())
    investigation = _investigation_with_evidence("Nothing recognizable here")
    recommendation = engine.generate(investigation)
    assert recommendation.strategy.decision_checkpoint is None


def test_decision_checkpoint_confirms_single_candidate():
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS, record_id="hist-1", title="Past issue", snippet="...",
        score=0.9, metadata={"root_cause": "Known firmware bug."},
    )
    store = FakeKnowledgeStore({KnowledgeCollection.HISTORICAL_INVESTIGATIONS: [match]})
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = _investigation_with_evidence("Something broke")
    recommendation = engine.generate(investigation)

    assert recommendation.strategy.decision_checkpoint is not None
    assert "Confirm" in recommendation.strategy.decision_checkpoint


def test_decision_checkpoint_distinguishes_two_candidates():
    match1 = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS, record_id="hist-1", title="Issue A", snippet="...",
        score=0.9, metadata={"root_cause": "Cause A."},
    )
    match2 = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS, record_id="hist-2", title="Issue B", snippet="...",
        score=0.85, metadata={"root_cause": "Cause B."},
    )
    store = FakeKnowledgeStore({KnowledgeCollection.HISTORICAL_INVESTIGATIONS: [match1, match2]})
    engine = RecommendationEngine(KnowledgeEngine(store), Settings())

    investigation = _investigation_with_evidence("Something broke")
    recommendation = engine.generate(investigation)

    assert "Two plausible causes" in recommendation.strategy.decision_checkpoint
    assert "Cause A" in recommendation.strategy.decision_checkpoint
    assert "Cause B" in recommendation.strategy.decision_checkpoint


# --- Component matching -------------------------------------------------


def test_component_matched_by_exact_name_in_text():
    components = [_make_component("comp-1", "CommandProcessorHost")]
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), component_repo=FakeComponentRepo(components)
    )
    investigation = _investigation_with_evidence("CommandProcessorHost is timing out on every request")
    recommendation = engine.generate(investigation)

    matched = recommendation.strategy.matched_component
    assert matched is not None
    assert matched.component_id == "comp-1"
    assert matched.confidence == 1.0


def test_component_matched_by_exact_entity_value():
    components = [_make_component("comp-1", "order-service")]
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), component_repo=FakeComponentRepo(components)
    )
    # RegexEntityExtractor picks up "order-service" as a SERVICE_NAME entity from this text.
    investigation = _investigation_with_evidence("Error in service=order-service during checkout")
    recommendation = engine.generate(investigation)

    matched = recommendation.strategy.matched_component
    if matched is not None:
        assert matched.component_name == "order-service"
        assert matched.confidence == 1.0


def test_no_component_match_when_nothing_relevant_mentioned():
    components = [_make_component("comp-1", "CommandProcessorHost")]
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), component_repo=FakeComponentRepo(components)
    )
    investigation = _investigation_with_evidence("Totally unrelated text about weather")
    recommendation = engine.generate(investigation)
    assert recommendation.strategy.matched_component is None


def test_no_component_match_when_no_component_repo_provided():
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings())
    investigation = _investigation_with_evidence("CommandProcessorHost is broken")
    recommendation = engine.generate(investigation)
    assert recommendation.strategy.matched_component is None


# --- SQL Library integration ----------------------------------------------


def test_suggested_sql_prefers_component_linked_query_template():
    components = [_make_component("comp-1", "CommandProcessorHost")]
    templates = [
        _make_query_template("qt-1", "Command log lookup", related_components=["CommandProcessorHost"]),
        _make_query_template("qt-2", "Unrelated template", related_components=["SomeOtherComponent"]),
    ]
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(),
        component_repo=FakeComponentRepo(components), sql_library=FakeSqlLibrary(templates),
    )
    investigation = _investigation_with_evidence("CommandProcessorHost command is failing")
    recommendation = engine.generate(investigation)

    sql_items = recommendation.strategy.suggested_sql
    assert len(sql_items) == 1
    assert sql_items[0].source == "sql_library"
    assert sql_items[0].template_id == "qt-1"


def test_suggested_sql_falls_back_to_entity_heuristic_without_component_match():
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings())
    investigation = _investigation_with_evidence("Blocking detected, spid=61 is head blocker")
    recommendation = engine.generate(investigation)

    sql_items = recommendation.strategy.suggested_sql
    assert len(sql_items) >= 1
    assert all(item.source == "entity_heuristic" for item in sql_items)


def test_suggested_sql_empty_when_component_matches_but_no_template_linked():
    components = [_make_component("comp-1", "CommandProcessorHost")]
    templates = [_make_query_template("qt-1", "Unrelated", related_components=["SomeOtherComponent"])]
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(),
        component_repo=FakeComponentRepo(components), sql_library=FakeSqlLibrary(templates),
    )
    investigation = _investigation_with_evidence("CommandProcessorHost issue, no other clues")
    recommendation = engine.generate(investigation)

    # No component-linked template and no entity-heuristic signal either.
    assert recommendation.strategy.suggested_sql == []


# --- Required / missing evidence --------------------------------------------


def test_required_evidence_deduplicates_by_component_across_scenarios():
    from app.domain.log_intelligence_kb import LogCollectionScenario, LogCollectionStep, LogRepositoryLocation, LogSourceApplication

    scenario_a = LogCollectionScenario(
        id="sc-1", product="Command Center", technology="RF Mesh", scenario_type="General",
        steps=[LogCollectionStep(log_source_id="src-a", component_name="A", priority=1, explanation="x")],
        source_wiki_page="Test",
    )
    scenario_b = LogCollectionScenario(
        id="sc-2", product="Command Center", technology="RF Mesh IP", scenario_type="General",
        steps=[LogCollectionStep(log_source_id="src-a", component_name="A", priority=1, explanation="y")],
        source_wiki_page="Test",
    )
    source = LogSourceApplication(
        id="src-a", name="A", location=LogRepositoryLocation(platform="linux", root_path="/var/log", filename_patterns=["a.log"]),
        product="Command Center",
    )
    log_repo = FakeLogKnowledgeRepo([scenario_a, scenario_b], [source])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    investigation = _investigation_with_evidence("RF Mesh issue")
    recommendation = engine.generate(investigation)

    component_a_items = [i for i in recommendation.strategy.required_evidence if i.description.startswith("A logs")]
    assert len(component_a_items) == 1


def test_missing_evidence_is_subset_of_required_evidence_where_unsatisfied():
    from app.domain.log_intelligence_kb import LogCollectionScenario, LogCollectionStep, LogRepositoryLocation, LogSourceApplication

    scenario = LogCollectionScenario(
        id="sc-1", product="Command Center", technology="RF Mesh", scenario_type="General",
        steps=[LogCollectionStep(log_source_id="src-a", component_name="A", priority=1, explanation="x")],
        source_wiki_page="Test",
    )
    source = LogSourceApplication(
        id="src-a", name="A", location=LogRepositoryLocation(platform="linux", root_path="/var/log", filename_patterns=["a.log"]),
        product="Command Center",
    )
    log_repo = FakeLogKnowledgeRepo([scenario], [source])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)

    investigation = _investigation_with_log_evidence("RF Mesh issue", log_filenames=["A_log.log"])
    recommendation = engine.generate(investigation)

    for item in recommendation.strategy.missing_evidence:
        assert item.satisfied is False
    assert all(item in recommendation.strategy.required_evidence for item in recommendation.strategy.missing_evidence)


# --- RecommendationEngine._synthesize_recommendation --------------------------


def _tfs_case(tfs_id=1, title="CommandProcessorHost issue", state="Closed", resolution="Restart the service", crm_id=None):
    from datetime import datetime

    from app.domain.external_knowledge import TfsCase

    return TfsCase(
        tfs_id=tfs_id,
        work_item_type="Bug",
        title=title,
        state=state,
        area_path="Command Center",
        team_project="Command Center",
        changed_date=datetime(2026, 1, 1),
        description_text="x",
        resolution_text=resolution,
        crm_id=crm_id,
        url=f"https://am.tfs.landisgyr.net/tfs/DefaultCollection/Command%20Center/_workitems/edit/{tfs_id}",
    )


def _tfs_result(matches=None, available=True):
    from app.domain.external_knowledge import ExternalKnowledgeResult, ExternalMatch, ExternalSource

    if matches is None:
        return ExternalKnowledgeResult(source=ExternalSource.TFS, available=available, matches=[])
    external_matches = [
        ExternalMatch(source=ExternalSource.TFS, tfs_case=case, score=score, confidence="Medium", match_reasons=["x"])
        for case, score in matches
    ]
    return ExternalKnowledgeResult(source=ExternalSource.TFS, available=available, matches=external_matches)


def _wiki_result(matches=None, available=True):
    from app.domain.external_knowledge import ExternalKnowledgeResult, ExternalMatch, ExternalSource, WikiPage

    if matches is None:
        return ExternalKnowledgeResult(source=ExternalSource.WIKI, available=available, matches=[])
    external_matches = [
        ExternalMatch(source=ExternalSource.WIKI, wiki_page=WikiPage(page_id="1", title=title, space_key="CC", excerpt=excerpt, url="https://wiki.landisgyr.net/1"), score=score, confidence="Medium", match_reasons=["x"])
        for title, excerpt, score in matches
    ]
    return ExternalKnowledgeResult(source=ExternalSource.WIKI, available=available, matches=external_matches)


def test_synthesize_recommendation_insufficient_evidence_when_nothing_clears_bar():
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings())
    solution, sources = engine._synthesize_recommendation([], [], _tfs_result(matches=[]), _wiki_result(matches=[]), [])

    assert solution.insufficient_evidence is True
    assert solution.recommended_resolution is None
    assert solution.confidence == "Insufficient"


def test_synthesize_recommendation_never_fabricates_when_tfs_match_has_no_resolution_text():
    """Real requirement: a TFS match can clear the ranking confidence
    bar (component/technology matched) without TFS itself having ever
    recorded a resolution -- must not fabricate one."""
    case = _tfs_case(resolution=None)
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings())
    solution, sources = engine._synthesize_recommendation([], [], _tfs_result(matches=[(case, 0.6)]), _wiki_result(matches=[]), [])

    assert solution.source_tfs is True
    assert solution.recommended_resolution is None
    assert solution.insufficient_evidence is True


def test_synthesize_recommendation_uses_tfs_resolution_when_available():
    case = _tfs_case(resolution="Restart CommandProcessorHost and verify init messages")
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings())
    solution, sources = engine._synthesize_recommendation([], [], _tfs_result(matches=[(case, 0.6)]), _wiki_result(matches=[]), [])

    assert solution.insufficient_evidence is False
    assert "Restart CommandProcessorHost" in solution.recommended_resolution
    assert f"TFS-{case.tfs_id}" in solution.recommended_resolution
    assert solution.supporting_tfs_id == case.tfs_id
    assert solution.supporting_tfs_url == case.url


def test_synthesize_recommendation_below_floor_tfs_match_does_not_drive_solution():
    case = _tfs_case(resolution="Restart the service")
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings())
    solution, sources = engine._synthesize_recommendation([], [], _tfs_result(matches=[(case, 0.1)]), _wiki_result(matches=[]), [])

    assert solution.source_tfs is False
    assert solution.insufficient_evidence is True


def test_synthesize_recommendation_correlates_tfs_crm_id_with_local_ticket():
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hi-1",
        title="ATCO - RF Mesh IP - similar case",
        snippet="x",
        score=0.5,
        metadata={"tags": "ticket:CSTASK0087353, priority:High", "root_cause": "Stale DCW", "resolution": "Reissued GEI"},
    )
    case = _tfs_case(resolution="Reissued GEI, confirmed fixed", crm_id="CS0122697/CSTASK0087353")
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings())
    solution, sources = engine._synthesize_recommendation([], [match], _tfs_result(matches=[(case, 0.6)]), _wiki_result(matches=[]), [])

    assert solution.source_local is True
    assert solution.source_tfs is True
    assert "same real-world case" in solution.rationale


def test_synthesize_recommendation_what_to_check_falls_back_to_generic_when_no_missing_evidence():
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings())
    case = _tfs_case()
    solution, sources = engine._synthesize_recommendation([], [], _tfs_result(matches=[(case, 0.6)]), _wiki_result(matches=[]), [])

    assert len(solution.what_to_check) >= 1


def test_synthesize_recommendation_wiki_fills_resolution_when_tfs_has_none():
    case = _tfs_case(resolution=None)
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings())
    solution, sources = engine._synthesize_recommendation(
        [], [], _tfs_result(matches=[(case, 0.6)]), _wiki_result(matches=[("Troubleshooting CommandProcessorHost", "Restart the service and check logs", 0.6)]), []
    )

    assert solution.source_wiki is True
    assert solution.insufficient_evidence is False
    assert "Troubleshooting CommandProcessorHost" in solution.recommended_resolution
    assert solution.supporting_wiki_title == "Troubleshooting CommandProcessorHost"


def test_synthesize_recommendation_prefers_root_cause_over_raw_local_match_title():
    """Regression test for a real reported bug: a local historical match
    with no recorded root_cause used to fall back to its raw ticket
    title, which could be an unrelated customer's ticket title presented
    as if it diagnosed the current case. root_causes (already filtered
    by _build_root_causes) must win when present."""
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hi-1",
        title="CLP | Prod | CC 9.0 | Meter load profile found with frozen value",
        snippet="x",
        score=0.55,
        metadata={},  # no root_cause on file -- the exact real-world case
    )
    root_causes = [
        RootCauseHypothesis(
            description="Unhandled application exception: DCWErr_Invalid_Response_Length",
            confidence=0.6,
            rationale="Investigation evidence contains a exception_type ('DCWErr_Invalid_Response_Length').",
        )
    ]
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings())
    solution, sources = engine._synthesize_recommendation(root_causes, [match], _tfs_result(matches=[]), _wiki_result(matches=[]), [])

    assert solution.likely_issue == root_causes[0].description
    assert "CLP | Prod | CC 9.0" not in solution.likely_issue


def test_synthesize_recommendation_local_title_fallback_is_labeled_unconfirmed():
    """When there's no root-cause hypothesis at all, a title-only local
    match may still be surfaced, but must be clearly caveated as
    unconfirmed rather than presented as a diagnosis."""
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hi-1",
        title="CLP | Prod | CC 9.0 | Meter load profile found with frozen value",
        snippet="x",
        score=0.55,
        metadata={},
    )
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings())
    solution, sources = engine._synthesize_recommendation([], [match], _tfs_result(matches=[]), _wiki_result(matches=[]), [])

    assert "no confirmed root cause" in solution.likely_issue.lower()
    assert match.title in solution.likely_issue


def test_synthesize_recommendation_title_fallback_never_quotes_resolution_of_untrusted_match():
    """Regression test for a second real reported bug (same CLECO case):
    when the top local match has no recorded root cause, we already
    caveat likely_issue as unconfirmed -- but the old code still quoted
    that same match's *resolution* as if it were a trustworthy fix,
    even though it may describe a genuinely different defect (same
    customer/component, different root cause). A resolution must never
    be attached to a match we've just said we don't trust the root
    cause of."""
    match = KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id="hi-1",
        title="CLECO: RF Enhanced Focus AX: Request for Full Meter Read Analysis",
        snippet="x",
        score=0.86,
        metadata={"resolution": "Closure Summary: unrelated ST-03 advisory investigation, not this defect."},
    )
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings())
    solution, sources = engine._synthesize_recommendation([], [match], _tfs_result(matches=[]), _wiki_result(matches=[]), [])

    assert solution.recommended_resolution is None
    assert solution.insufficient_evidence is True
    # Local KB still genuinely shaped the (caveated) likely_issue, so it's
    # still an honest source to disclose -- just not for the resolution.
    assert solution.source_local is True


# --- RF Mesh vs Mesh IP specificity fix (Context Dimensions phase, 2026-08-12,
# approved product decision #6/#7) -------------------------------------------


class FakeLookupRepoForTechnology:
    """Structurally satisfies the parts of LookupRepository
    _specificity_demoted_technologies/_resolve_retrieval_context use."""

    def __init__(self, technologies) -> None:
        self._technologies = technologies

    def list_technologies(self, *, active_only: bool = True):
        return self._technologies

    def get_technology_by_name(self, name: str):
        return next((t for t in self._technologies if t.name == name), None)

    def get_customer_by_name(self, name: str):  # pragma: no cover -- unused in these tests
        return None


def _rf_mesh_family():
    from app.domain.lookup_entities import Technology

    parent = Technology(id="tech-rf-mesh", name="RF Mesh")
    child_ip = Technology(id="tech-rf-mesh-ip", name="RF Mesh IP", parent_technology_id="tech-rf-mesh")
    child_das = Technology(id="tech-rf-mesh-das", name="RF Mesh (DAS implementation)", parent_technology_id="tech-rf-mesh")
    unrelated = Technology(id="tech-wi-sun", name="Wi-Sun")
    return [parent, child_ip, child_das, unrelated]


def test_bare_technology_is_demoted_when_specific_variant_also_mentioned():
    """The real, reported bug: 'RF Mesh' used to score a full match
    against text that actually said the more specific 'RF Mesh IP' --
    both scenarios then looked equally 'Critical'."""
    bare = _make_scenario("sc-bare", "RF Mesh", "Command Request (Outbound)", [_make_step("src-a", "A", 1)])
    specific = _make_scenario("sc-specific", "RF Mesh IP", "Command Request (Outbound)", [_make_step("src-b", "B", 1)])
    log_repo = FakeLogKnowledgeRepo([bare, specific], [_make_source("src-a", "A"), _make_source("src-b", "B")])
    lookup_repo = FakeLookupRepoForTechnology(_rf_mesh_family())
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo, lookup_repo=lookup_repo
    )
    investigation = _investigation_with_evidence(
        "Meter stuck in Discovered state. Technology: RF Mesh IP. Command Request is failing."
    )

    items = engine._recommend_logs(investigation)

    specific_items = [i for i in items if i.scenario_technology == "RF Mesh IP"]
    bare_items = [i for i in items if i.scenario_technology == "RF Mesh"]
    assert specific_items, "the specifically-mentioned technology must still match"
    assert all(i.priority_label != "Critical" or bare_items == [] for i in bare_items)
    if bare_items:
        assert bare_items[0].priority_label != "Critical"
        assert "more specific" in bare_items[0].match_reason
        assert "RF Mesh IP" in bare_items[0].match_reason


def test_bare_technology_still_scores_critical_when_no_specific_variant_is_mentioned():
    """No over-correction: mentioning bare 'RF Mesh' (with no 'IP')
    must not demote the bare-technology scenario at all."""
    bare = _make_scenario("sc-bare", "RF Mesh", "Command Request (Outbound)", [_make_step("src-a", "A", 1)])
    log_repo = FakeLogKnowledgeRepo([bare], [_make_source("src-a", "A")])
    lookup_repo = FakeLookupRepoForTechnology(_rf_mesh_family())
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo, lookup_repo=lookup_repo
    )
    investigation = _investigation_with_evidence(
        "Meter stuck in Discovered state. Technology: RF Mesh. Command Request (Outbound) is failing."
    )

    items = engine._recommend_logs(investigation)

    assert items
    assert items[0].priority_label == "Critical"
    assert items[0].scenario_technology == "RF Mesh"


def test_no_lookup_repo_wired_never_demotes_anything():
    """Graceful degradation: older call sites/tests with no
    LookupRepository keep their exact pre-fix behavior."""
    bare = _make_scenario("sc-bare", "RF Mesh", "Command Request (Outbound)", [_make_step("src-a", "A", 1)])
    log_repo = FakeLogKnowledgeRepo([bare], [_make_source("src-a", "A")])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)
    investigation = _investigation_with_evidence(
        "Meter stuck in Discovered state. Technology: RF Mesh IP. Command Request is failing."
    )

    demoted = engine._specificity_demoted_technologies(investigation.context_text)
    assert demoted == {}


def test_specificity_demoted_technologies_direct():
    lookup_repo = FakeLookupRepoForTechnology(_rf_mesh_family())
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), FakeLogKnowledgeRepo([], []), lookup_repo=lookup_repo
    )

    demoted = engine._specificity_demoted_technologies("Investigation mentions RF Mesh IP explicitly.")
    assert demoted == {"RF Mesh": "RF Mesh IP"}

    not_demoted = engine._specificity_demoted_technologies("Investigation mentions RF Mesh only.")
    assert not_demoted == {}


# --- Fabricated technology inference fix (2026-08-13, Final Knowledge-
# Quality Acceptance Test, Findings #1/#2) -----------------------------


class FakeLookupRepoWithList:
    """Structurally satisfies the parts of LookupRepository
    _match_single_technology uses -- unlike FakeLookupRepoForTechnology,
    exposes list_technologies() with the real (id, parent_technology_id)
    shape needed to test the hierarchy-aware tie-break directly."""

    def __init__(self, technologies) -> None:
        self._technologies = technologies

    def list_technologies(self, *, active_only: bool = True):
        return self._technologies

    def get_technology_by_name(self, name: str):
        return next((t for t in self._technologies if t.name == name), None)

    def get_customer_by_name(self, name: str):  # pragma: no cover -- unused in these tests
        return None


def test_no_technology_mentioned_returns_none():
    """Finding #1: a zero-score candidate must never win merely for
    lacking competition."""
    lookup_repo = FakeLookupRepoWithList(_rf_mesh_family())
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), FakeLogKnowledgeRepo([], []), lookup_repo=lookup_repo
    )
    investigation = _investigation_with_evidence("TEPCO customer reporting HES issue -- meters not communicating via DLMS.")

    assert engine._infer_technology(investigation) is None


def test_random_generic_text_returns_none():
    """Finding #1, the literal reported case: fully generic text with
    zero technology vocabulary must never resolve to the single
    longest-named governed technology (or anything else)."""
    lookup_repo = FakeLookupRepoWithList(_rf_mesh_family())
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), FakeLogKnowledgeRepo([], []), lookup_repo=lookup_repo
    )
    investigation = _investigation_with_evidence("Command processing appears delayed. Need to check command lifecycle and response status.")

    assert engine._infer_technology(investigation) is None


def test_bare_rf_mesh_infers_rf_mesh():
    lookup_repo = FakeLookupRepoWithList(_rf_mesh_family())
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), FakeLogKnowledgeRepo([], []), lookup_repo=lookup_repo
    )
    investigation = _investigation_with_evidence("Commands are failing on the RF Mesh network. Technology: RF Mesh.")

    assert engine._infer_technology(investigation) == "RF Mesh"


def test_explicit_rf_mesh_ip_infers_rf_mesh_ip():
    lookup_repo = FakeLookupRepoWithList(_rf_mesh_family())
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), FakeLogKnowledgeRepo([], []), lookup_repo=lookup_repo
    )
    investigation = _investigation_with_evidence("Meters showing Discovered state. Technology: RF Mesh IP.")

    assert engine._infer_technology(investigation) == "RF Mesh IP"


def test_mesh_ip_without_rf_prefix_resolves_to_rf_mesh_ip_not_das_implementation():
    """Finding #2, the exact reported case, resolved by the generic
    evidence-coverage rule (2026-08-13): 'Mesh IP' text contains the
    single significant word "Mesh", which is 100% coverage of both
    "RF Mesh"'s and "RF Mesh IP"'s own (single-word) significant
    vocabulary -- both remain valid partial matches. "RF Mesh (DAS
    implementation)" needs BOTH "Mesh" and "implementation" (its own
    two significant words) to count as valid evidence; only "Mesh" is
    present, so it's excluded outright (0.0, not merely "weaker").
    That leaves "RF Mesh" (ancestor) and "RF Mesh IP" (its real,
    governed child) tied at the top score -- specificity resolves this
    exactly as it does for the full-phrase "RF Mesh IP" case, so the
    real technology inference is "RF Mesh IP", not an unresolved
    ambiguity and never the wrong sibling."""
    lookup_repo = FakeLookupRepoWithList(_rf_mesh_family())
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), FakeLogKnowledgeRepo([], []), lookup_repo=lookup_repo
    )
    investigation = _investigation_with_evidence("Interval extract files are breaking on the Mesh IP network.")

    result = engine._infer_technology(investigation)
    assert result != "RF Mesh (DAS implementation)"
    assert result == "RF Mesh IP"


def test_explicit_das_implementation_infers_that_technology():
    """Sanity check: the third sibling, when explicitly and uniquely
    named, still resolves correctly -- proves the fix doesn't just
    suppress the family, it correctly resolves an unambiguous case."""
    lookup_repo = FakeLookupRepoWithList(_rf_mesh_family())
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), FakeLogKnowledgeRepo([], []), lookup_repo=lookup_repo
    )
    investigation = _investigation_with_evidence("Technology: RF Mesh (DAS implementation). Water module deployment.")

    assert engine._infer_technology(investigation) == "RF Mesh (DAS implementation)"


def test_no_lookup_repo_wired_still_never_fabricates_from_zero_score():
    """Graceful degradation path (available_log_technologies() fallback,
    no hierarchy) must obey the same 'zero score never wins' rule."""
    scenario = _make_scenario("sc-1", "RF Mesh", "Command Request (Outbound)", [_make_step("src-a", "A", 1)])
    log_repo = FakeLogKnowledgeRepo([scenario], [_make_source("src-a", "A")])
    engine = RecommendationEngine(KnowledgeEngine(FakeKnowledgeStore()), Settings(), log_repo)
    investigation = _investigation_with_evidence("Totally unrelated text about weather and lunch.")

    assert engine._infer_technology(investigation) is None


# --- Generic single-word-overlap evidence threshold (2026-08-13) -------


def _multi_word_technology_family():
    """A real-shaped multi-significant-word technology name plus an
    unrelated single-significant-word one -- mirrors the real
    'Tool Data (BCS / HHU) processing' vs 'RF Mesh' shapes without
    hardcoding those exact real names into the test helper itself."""
    from app.domain.lookup_entities import Technology

    multi_word = Technology(id="tech-tool-data", name="Tool Data (BCS / HHU) processing")
    unrelated = Technology(id="tech-wi-sun", name="Wi-Sun")
    return [multi_word, unrelated]


def test_single_generic_word_overlap_does_not_match_multi_word_technology():
    """The exact reported case: text sharing only ONE of a multi-word
    technology's several significant words ("processing" out of
    {Tool, Data, processing}) must not be treated as evidence for that
    technology at all."""
    lookup_repo = FakeLookupRepoWithList(_multi_word_technology_family())
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), FakeLogKnowledgeRepo([], []), lookup_repo=lookup_repo
    )
    investigation = _investigation_with_evidence(
        "Command processing appears delayed. Need to check command lifecycle and response status."
    )

    assert engine._infer_technology(investigation) is None


def test_technology_evidence_score_requires_full_coverage_of_significant_words():
    """Direct unit test of the scoring rule itself: one of two
    significant words present scores 0.0 (insufficient), not a
    partial/weak score."""
    score_one_of_two = RecommendationEngine._technology_evidence_score(
        "Tool Data (BCS / HHU) processing", "Something about processing only."
    )
    assert score_one_of_two == 0.0

    score_all_present = RecommendationEngine._technology_evidence_score(
        "Tool Data (BCS / HHU) processing", "Tool Data processing needs review."
    )
    assert score_all_present == 0.5

    score_single_word_name = RecommendationEngine._technology_evidence_score("RF Mesh", "Something about the Mesh network.")
    assert score_single_word_name == 0.5

    score_exact_phrase = RecommendationEngine._technology_evidence_score(
        "Tool Data (BCS / HHU) processing", "Uses Tool Data (BCS / HHU) processing directly."
    )
    assert score_exact_phrase == 1.0


def test_explicit_full_technology_name_still_matches_despite_multiple_significant_words():
    """Positive control: the full, exact technology name (all
    significant words present, in the exact real phrase) must still
    resolve correctly -- the fix tightens partial matching, it does
    not break exact matching."""
    lookup_repo = FakeLookupRepoWithList(_multi_word_technology_family())
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), FakeLogKnowledgeRepo([], []), lookup_repo=lookup_repo
    )
    investigation = _investigation_with_evidence(
        "Investigating an issue in Tool Data (BCS / HHU) processing for a batch of endpoints."
    )

    assert engine._infer_technology(investigation) == "Tool Data (BCS / HHU) processing"


def test_all_significant_words_present_but_not_as_exact_phrase_still_matches():
    """Full coverage of a multi-word candidate's significant
    vocabulary counts as valid (rule 2, "strong multi-word evidence"),
    even when the words aren't adjacent/in the exact original phrase
    order -- distinct from the single-word-only case above."""
    lookup_repo = FakeLookupRepoWithList(_multi_word_technology_family())
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), FakeLogKnowledgeRepo([], []), lookup_repo=lookup_repo
    )
    investigation = _investigation_with_evidence("Tool support ticket: Data processing pipeline is delayed for this batch.")

    assert engine._infer_technology(investigation) == "Tool Data (BCS / HHU) processing"


# --- Generic component matching fix (2026-08-13, Finding #3) -----------


def test_generic_network_word_does_not_match_network_hub():
    components = [_make_component("comp-1", "Network Hub")]
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), component_repo=FakeComponentRepo(components)
    )
    investigation = _investigation_with_evidence("Commands are failing on the RF Mesh network.")

    recommendation = engine.generate(investigation)
    assert recommendation.strategy.matched_component is None


def test_explicit_network_hub_name_still_matches():
    components = [_make_component("comp-1", "Network Hub")]
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), component_repo=FakeComponentRepo(components)
    )
    investigation = _investigation_with_evidence("Network Hub is failing to route endpoint traffic.")

    recommendation = engine.generate(investigation)
    matched = recommendation.strategy.matched_component
    assert matched is not None
    assert matched.component_name == "Network Hub"
    assert matched.confidence == 1.0


# --- TEPCO: customer only, never technology/component (Finding audit) --


def test_tepco_customer_never_matched_as_technology_or_component():
    """TEPCO exists only in the (separate) Customer table -- never in
    Technology or Component candidates -- so it structurally cannot
    ever be returned by either matcher, regardless of the fixes above."""
    lookup_repo = FakeLookupRepoWithList(_rf_mesh_family())
    components = [_make_component("comp-1", "Device Hub")]
    engine = RecommendationEngine(
        KnowledgeEngine(FakeKnowledgeStore()), Settings(), component_repo=FakeComponentRepo(components), lookup_repo=lookup_repo
    )
    investigation = _investigation_with_evidence("TEPCO customer reporting HES issue -- meters not communicating via DLMS.")

    assert engine._infer_technology(investigation) is None
    matched_component = engine._match_component(investigation, investigation.merged_entities)
    assert matched_component is None
