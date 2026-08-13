"""Tests for DocumentClassificationEngine (Context Dimensions phase,
2026-08-12 -- approved product decisions #2/#3/#4: confidence-tiered
classification, never overwriting existing metadata, extractive-only
matching against already-governed values).

Integration-style against a real temp-file SQLite database (same
pattern as test_log_knowledge_importer.py) -- classification genuinely
depends on several real repositories/the Knowledge Object Framework,
so a fake would just re-assert the mock.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest

from app.domain.classification import ClassificationDimension, ConfidenceTier, SuggestionStatus
from app.domain.evidence import DocumentationRecord, HistoricalInvestigationRecord, KnownBugRecord
from app.domain.knowledge_relationships import KnowledgeObjectType, RelationshipType
from app.domain.lookup_entities import Customer, Region, Technology
from app.domain.product_intelligence import ComponentProfile
from app.engines.knowledge.classification import DocumentClassificationEngine
from app.engines.knowledge_object_framework.adapters import build_adapters
from app.engines.knowledge_object_framework.service import KnowledgeObjectService
from app.engines.knowledge_relationships.engine import KnowledgeRelationshipEngine
from app.infrastructure.db.classification_repository import SqlAlchemyClassificationRepository
from app.infrastructure.db.component_repository import SqlAlchemyComponentProfileRepository
from app.infrastructure.db.knowledge_repository import SqlAlchemyKnowledgeRepository
from app.infrastructure.db.log_knowledge_repository import SqlAlchemyLogKnowledgeRepository
from app.infrastructure.db.lookup_repository import SqlAlchemyLookupRepository
from app.infrastructure.db.playbook_repository import SqlAlchemyPlaybookRepository
from app.infrastructure.db.relationship_repository import SqlAlchemyRelationshipRepository
from app.engines.knowledge.engine import KnowledgeEngine
from app.infrastructure.db.session import get_engine, get_session_factory
from app.infrastructure.db.sql_template_repository import SqlAlchemySqlTemplateRepository
from app.infrastructure.db.version_repository import SqlAlchemyVersionRepository


class _FakeKnowledgeStore:
    """Structurally satisfies KnowledgeStore -- classification indexes
    edited Documents through KnowledgeObjectService.edit_metadata, which
    always calls KnowledgeEngine.index_documentation; a real ChromaDB/
    embedding model is unnecessary for these tests, same discipline as
    every other fake store in this test suite."""

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
def engine_bundle():
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
        classification_repo = SqlAlchemyClassificationRepository(session_factory)

        relationship_engine = KnowledgeRelationshipEngine(
            relationship_repo, component_repo, knowledge_repo, sql_repo, playbook_repo, lookup_repo, log_knowledge_repo
        )
        adapters = build_adapters(component_repo, knowledge_repo, sql_repo, playbook_repo, lookup_repo, log_knowledge_repo)
        knowledge_engine = KnowledgeEngine(_FakeKnowledgeStore(), knowledge_repo)
        knowledge_object_service = KnowledgeObjectService(adapters, relationship_engine, version_repo, knowledge_engine)

        classification_engine = DocumentClassificationEngine(
            knowledge_repo, lookup_repo, component_repo, classification_repo, knowledge_object_service, relationship_engine
        )

        yield dict(
            knowledge_repo=knowledge_repo,
            lookup_repo=lookup_repo,
            component_repo=component_repo,
            classification_repo=classification_repo,
            classification_engine=classification_engine,
            relationship_engine=relationship_engine,
        )
        get_engine(sqlite_url).dispose()
        get_engine.cache_clear()


def _save_document(knowledge_repo, *, title: str, content: str, product=None, technology=None) -> DocumentationRecord:
    record = DocumentationRecord(id=str(uuid.uuid4()), title=title, content=content, product=product, technology=technology, status="published")
    knowledge_repo.save_documentation(record)
    return record


def _save_historical_investigation(knowledge_repo, *, title: str, description: str) -> HistoricalInvestigationRecord:
    record = HistoricalInvestigationRecord(id=str(uuid.uuid4()), title=title, description=description, root_cause="-", resolution="-")
    knowledge_repo.save_historical_investigation(record)
    return record


def _save_known_bug(knowledge_repo, *, title: str, description: str) -> KnownBugRecord:
    record = KnownBugRecord(id=str(uuid.uuid4()), title=title, description=description)
    knowledge_repo.save_known_bug(record)
    return record


def test_title_match_is_high_confidence_and_auto_applies(engine_bundle):
    lookup_repo = engine_bundle["lookup_repo"]
    lookup_repo.save_technology(Technology(id=str(uuid.uuid4()), name="RF Mesh IP"))
    doc = _save_document(engine_bundle["knowledge_repo"], title="RF Mesh IP Troubleshooting Guide", content="Generic content.")

    summary = engine_bundle["classification_engine"].run(actor="test")

    assert summary.auto_accepted >= 1
    updated = engine_bundle["knowledge_repo"].get_documentation(doc.id)
    assert updated.technology == "RF Mesh IP"


def test_existing_metadata_is_never_overwritten(engine_bundle):
    """Approved product decision #4: 'without destroying or
    overwriting existing knowledge'."""
    lookup_repo = engine_bundle["lookup_repo"]
    lookup_repo.save_technology(Technology(id=str(uuid.uuid4()), name="RF Mesh IP"))
    doc = _save_document(
        engine_bundle["knowledge_repo"],
        title="RF Mesh IP Troubleshooting Guide",
        content="Generic content.",
        technology="Wi-Sun",  # already set to something else -- must survive untouched
    )

    engine_bundle["classification_engine"].run(actor="test")

    updated = engine_bundle["knowledge_repo"].get_documentation(doc.id)
    assert updated.technology == "Wi-Sun"


def test_body_repeated_mention_is_medium_confidence_pending_not_applied(engine_bundle):
    lookup_repo = engine_bundle["lookup_repo"]
    lookup_repo.save_technology(Technology(id=str(uuid.uuid4()), name="PANA"))
    doc = _save_document(
        engine_bundle["knowledge_repo"],
        title="Security Components Overview",
        content="This document discusses PANA encryption. PANA is used for Mesh IP security. See PANA config.",
    )

    summary = engine_bundle["classification_engine"].run(actor="test")

    assert summary.pending_review >= 1
    updated = engine_bundle["knowledge_repo"].get_documentation(doc.id)
    assert updated.technology is None  # never auto-applied at Medium confidence

    pending = engine_bundle["classification_repo"].list_by_status(SuggestionStatus.PENDING)
    matching = [s for s in pending if s.object_id == doc.id and s.suggested_value_text == "PANA"]
    assert len(matching) == 1
    assert matching[0].confidence_tier == ConfidenceTier.MEDIUM


def test_single_body_mention_is_low_confidence_never_applied_or_queued(engine_bundle):
    lookup_repo = engine_bundle["lookup_repo"]
    lookup_repo.save_technology(Technology(id=str(uuid.uuid4()), name="Wi-Sun"))
    doc = _save_document(
        engine_bundle["knowledge_repo"], title="General Architecture Overview", content="This system also touches Wi-Sun in one place."
    )

    summary = engine_bundle["classification_engine"].run(actor="test")

    assert summary.recorded_low_confidence >= 1
    pending = engine_bundle["classification_repo"].list_by_status(SuggestionStatus.PENDING)
    assert not any(s.object_id == doc.id for s in pending)
    updated = engine_bundle["knowledge_repo"].get_documentation(doc.id)
    assert updated.technology is None


def test_customer_high_confidence_creates_relationship_not_a_free_text_field(engine_bundle):
    """There is no scalar 'customer' field on DocumentationRecord --
    Customer applies exclusively via a real KnowledgeRelationship
    (APPLIES_TO), never invented as free text."""
    lookup_repo = engine_bundle["lookup_repo"]
    customer = Customer(id=str(uuid.uuid4()), name="TEPCO", aliases=["Tepco"])
    lookup_repo.save_customer(customer)
    doc = _save_document(engine_bundle["knowledge_repo"], title="TEPCO HES Issue Troubleshooting", content="Generic body.")

    summary = engine_bundle["classification_engine"].run(actor="test")

    assert summary.relationships_created >= 1
    relationships = engine_bundle["relationship_engine"].list_relationships(KnowledgeObjectType.DOCUMENT, doc.id)
    customer_edges = [r for r in relationships if r.relationship.relationship_type == RelationshipType.APPLIES_TO and r.to_object.type == KnowledgeObjectType.CUSTOMER]
    assert len(customer_edges) == 1
    assert customer_edges[0].to_object.title == "TEPCO"


def test_never_proposes_a_customer_as_a_technology_or_component(engine_bundle):
    """Structural guarantee (approved product decision #6): the
    classification engine only ever matches against already-governed
    Technology/Component/Product rows, never against the Customer
    list, and vice versa -- so a customer-named document can score a
    HIGH customer suggestion without ever also producing a spurious
    Technology/Component suggestion for the same name."""
    lookup_repo = engine_bundle["lookup_repo"]
    customer = Customer(id=str(uuid.uuid4()), name="TEPCO")
    lookup_repo.save_customer(customer)
    # Deliberately do NOT register "TEPCO" as a Technology or Component --
    # confirms nothing in this engine could invent one from the title alone.
    doc = _save_document(engine_bundle["knowledge_repo"], title="TEPCO Solution Architecture Diagram", content="Generic body.")

    engine_bundle["classification_engine"].run(actor="test")

    updated = engine_bundle["knowledge_repo"].get_documentation(doc.id)
    assert updated.technology is None
    assert updated.product is None
    assert "TEPCO" not in updated.related_components


def test_idempotent_rerun_does_not_duplicate_suggestions(engine_bundle):
    lookup_repo = engine_bundle["lookup_repo"]
    lookup_repo.save_technology(Technology(id=str(uuid.uuid4()), name="Wi-Sun"))
    _save_document(engine_bundle["knowledge_repo"], title="General notes", content="Mentions Wi-Sun once here.")

    first = engine_bundle["classification_engine"].run(actor="test")
    second = engine_bundle["classification_engine"].run(actor="test")

    assert first.suggestions_created >= 1
    assert second.suggestions_created == 0


def test_accept_pending_suggestion_applies_it(engine_bundle):
    lookup_repo = engine_bundle["lookup_repo"]
    lookup_repo.save_technology(Technology(id=str(uuid.uuid4()), name="PANA"))
    doc = _save_document(
        engine_bundle["knowledge_repo"],
        title="Security Components Overview",
        content="This document discusses PANA. PANA is used here. See PANA again.",
    )
    engine_bundle["classification_engine"].run(actor="test")
    pending = [s for s in engine_bundle["classification_repo"].list_by_status(SuggestionStatus.PENDING) if s.object_id == doc.id]
    assert len(pending) == 1

    engine_bundle["classification_engine"].accept(pending[0].id, actor="reviewer")

    updated = engine_bundle["knowledge_repo"].get_documentation(doc.id)
    assert updated.technology == "PANA"
    accepted = engine_bundle["classification_repo"].get(pending[0].id)
    assert accepted.status == SuggestionStatus.ACCEPTED
    assert accepted.reviewed_by == "reviewer"


def test_reject_pending_suggestion_does_not_apply_it(engine_bundle):
    lookup_repo = engine_bundle["lookup_repo"]
    lookup_repo.save_technology(Technology(id=str(uuid.uuid4()), name="PANA"))
    doc = _save_document(
        engine_bundle["knowledge_repo"],
        title="Security Components Overview",
        content="This document discusses PANA. PANA is used here. See PANA again.",
    )
    engine_bundle["classification_engine"].run(actor="test")
    pending = [s for s in engine_bundle["classification_repo"].list_by_status(SuggestionStatus.PENDING) if s.object_id == doc.id]

    engine_bundle["classification_engine"].reject(pending[0].id, actor="reviewer")

    updated = engine_bundle["knowledge_repo"].get_documentation(doc.id)
    assert updated.technology is None


# --- Specificity fix (2026-08-13): classification must prefer the most
# specific governed technology, generically, via parent_technology_id
# (real bug found by the Phase 1 acceptance test: "RF Mesh IP doc" was
# tagged "RF Mesh", the exact conflation Phase 1 was supposed to fix,
# reintroduced in this engine's own, independent matching code). -------


def _seed_rf_mesh_family(lookup_repo) -> dict[str, str]:
    rf_mesh = Technology(id=str(uuid.uuid4()), name="RF Mesh")
    lookup_repo.save_technology(rf_mesh)
    rf_mesh_ip = Technology(id=str(uuid.uuid4()), name="RF Mesh IP", parent_technology_id=rf_mesh.id)
    lookup_repo.save_technology(rf_mesh_ip)
    wi_sun = Technology(id=str(uuid.uuid4()), name="Wi-Sun")
    lookup_repo.save_technology(wi_sun)
    return {"RF Mesh": rf_mesh.id, "RF Mesh IP": rf_mesh_ip.id, "Wi-Sun": wi_sun.id}


def test_rf_mesh_ip_overview_title_classifies_as_rf_mesh_ip(engine_bundle):
    lookup_repo = engine_bundle["lookup_repo"]
    _seed_rf_mesh_family(lookup_repo)
    doc = _save_document(engine_bundle["knowledge_repo"], title="RF Mesh IP Overview", content="Generic content.")

    engine_bundle["classification_engine"].run(actor="test")

    updated = engine_bundle["knowledge_repo"].get_documentation(doc.id)
    assert updated.technology == "RF Mesh IP"


def test_rf_mesh_ip_network_monitoring_title_classifies_as_rf_mesh_ip(engine_bundle):
    lookup_repo = engine_bundle["lookup_repo"]
    _seed_rf_mesh_family(lookup_repo)
    doc = _save_document(
        engine_bundle["knowledge_repo"], title="RF Mesh IP Network Monitoring", content="Generic content."
    )

    engine_bundle["classification_engine"].run(actor="test")

    updated = engine_bundle["knowledge_repo"].get_documentation(doc.id)
    assert updated.technology == "RF Mesh IP"


def test_bare_rf_mesh_troubleshooting_classifies_as_rf_mesh_not_rf_mesh_ip(engine_bundle):
    """The parent technology must still win cleanly when the more
    specific child genuinely isn't mentioned at all -- no over-
    correction."""
    lookup_repo = engine_bundle["lookup_repo"]
    _seed_rf_mesh_family(lookup_repo)
    doc = _save_document(engine_bundle["knowledge_repo"], title="RF Mesh troubleshooting", content="Generic content.")

    engine_bundle["classification_engine"].run(actor="test")

    updated = engine_bundle["knowledge_repo"].get_documentation(doc.id)
    assert updated.technology == "RF Mesh"


def test_title_mentioning_both_deterministically_resolves_to_the_more_specific_child(engine_bundle):
    """The classification engine's live matching rule (distinct from
    the extra, more cautious safety check applied only to the one-time
    retroactive correction of pre-existing data, see
    scratchpad/fix_rf_mesh_ip_classifications.py): whenever both a
    technology and one of its real, governed children fully match the
    same title, the child wins -- unconditionally, deterministically,
    every time. This is a deliberate, simple, generic rule (requirement:
    'if a child technology matches, it must win over its parent'), not
    a special case for RF Mesh specifically -- see
    test_future_child_technology_wins_without_hardcoding below."""
    lookup_repo = engine_bundle["lookup_repo"]
    _seed_rf_mesh_family(lookup_repo)
    doc = _save_document(
        engine_bundle["knowledge_repo"], title="RF Mesh and RF Mesh IP Overview", content="Generic content."
    )

    engine_bundle["classification_engine"].run(actor="test")

    updated = engine_bundle["knowledge_repo"].get_documentation(doc.id)
    assert updated.technology == "RF Mesh IP"


def test_unrelated_technology_never_false_matches_the_rf_mesh_family(engine_bundle):
    lookup_repo = engine_bundle["lookup_repo"]
    _seed_rf_mesh_family(lookup_repo)
    doc = _save_document(engine_bundle["knowledge_repo"], title="Wi-Sun Deployment Guide", content="Generic content.")

    engine_bundle["classification_engine"].run(actor="test")

    updated = engine_bundle["knowledge_repo"].get_documentation(doc.id)
    assert updated.technology == "Wi-Sun"


def test_future_child_technology_wins_without_hardcoding(engine_bundle):
    """A hierarchy the fix has never seen (a hypothetical 'DLMS COSEM'
    child of 'DLMS') must resolve correctly with zero code changes --
    proves the fix is generic, not hardcoded to RF Mesh."""
    lookup_repo = engine_bundle["lookup_repo"]
    dlms = Technology(id=str(uuid.uuid4()), name="DLMS")
    lookup_repo.save_technology(dlms)
    dlms_cosem = Technology(id=str(uuid.uuid4()), name="DLMS COSEM", parent_technology_id=dlms.id)
    lookup_repo.save_technology(dlms_cosem)
    doc = _save_document(engine_bundle["knowledge_repo"], title="DLMS COSEM Implementation Notes", content="Generic content.")

    engine_bundle["classification_engine"].run(actor="test")

    updated = engine_bundle["knowledge_repo"].get_documentation(doc.id)
    assert updated.technology == "DLMS COSEM"


def test_existing_explicit_rf_mesh_ip_classification_is_never_overwritten(engine_bundle):
    lookup_repo = engine_bundle["lookup_repo"]
    _seed_rf_mesh_family(lookup_repo)
    doc = _save_document(
        engine_bundle["knowledge_repo"], title="RF Mesh IP Overview", content="Generic content.", technology="Wi-Sun"
    )

    engine_bundle["classification_engine"].run(actor="test")

    updated = engine_bundle["knowledge_repo"].get_documentation(doc.id)
    assert updated.technology == "Wi-Sun"


def test_rf_mesh_ip_classification_rerun_is_idempotent(engine_bundle):
    lookup_repo = engine_bundle["lookup_repo"]
    _seed_rf_mesh_family(lookup_repo)
    _save_document(engine_bundle["knowledge_repo"], title="RF Mesh IP Overview", content="Generic content.")

    first = engine_bundle["classification_engine"].run(actor="test")
    second = engine_bundle["classification_engine"].run(actor="test")

    assert first.auto_accepted >= 1
    assert second.suggestions_created == 0
    assert second.auto_accepted == 0


# --- Scan-target extension (2026-08-13, Phase 0 -- Chat/Structured
# Resolution Knowledge architecture): Historical Investigations and
# Known Bugs now also get Customer/Region/Component classification,
# closing the gap flagged in
# RESOLVEIQ_CHAT_AND_RESOLUTION_ARCHITECTURE.md's Section 9 correction.
# Deliberately NOT Technology/Product (those record types have no
# scalar field to set) -- covered explicitly below. ------------------


def test_historical_investigation_gets_customer_relationship_from_title(engine_bundle):
    lookup_repo = engine_bundle["lookup_repo"]
    customer = Customer(id=str(uuid.uuid4()), name="TEPCO", aliases=["Tepco"])
    lookup_repo.save_customer(customer)
    investigation = _save_historical_investigation(
        engine_bundle["knowledge_repo"], title="TEPCO Meter Communication Failure", description="Generic body."
    )

    summary = engine_bundle["classification_engine"].run(actor="test")

    assert summary.relationships_created >= 1
    relationships = engine_bundle["relationship_engine"].list_relationships(
        KnowledgeObjectType.HISTORICAL_INVESTIGATION, investigation.id
    )
    customer_edges = [
        r for r in relationships
        if r.relationship.relationship_type == RelationshipType.APPLIES_TO and r.to_object.type == KnowledgeObjectType.CUSTOMER
    ]
    assert len(customer_edges) == 1
    assert customer_edges[0].to_object.title == "TEPCO"


def test_known_bug_gets_related_component_from_title(engine_bundle):
    component_repo = engine_bundle["component_repo"]
    component_repo.save(ComponentProfile(id=str(uuid.uuid4()), name="CommandProcessorHost", product="Command Center"))
    bug = _save_known_bug(
        engine_bundle["knowledge_repo"], title="CommandProcessorHost Queue Backup", description="Generic body."
    )

    summary = engine_bundle["classification_engine"].run(actor="test")

    assert summary.fields_set >= 1
    updated = engine_bundle["knowledge_repo"].get_known_bug(bug.id)
    assert "CommandProcessorHost" in updated.related_components


def test_historical_investigation_never_gets_a_scalar_technology_or_product_suggestion(engine_bundle):
    """HistoricalInvestigationRecord has no scalar technology/product
    field to set (unlike DocumentationRecord) -- confirms the scan
    extension only ever runs _classify_relationship/
    _classify_related_components against it, never
    _classify_scalar_field, which would otherwise raise via getattr on
    a field that doesn't exist."""
    lookup_repo = engine_bundle["lookup_repo"]
    lookup_repo.save_technology(Technology(id=str(uuid.uuid4()), name="RF Mesh IP"))
    investigation = _save_historical_investigation(
        engine_bundle["knowledge_repo"], title="RF Mesh IP Overview", description="Generic body."
    )

    # Must not raise (AttributeError from getattr on a missing field
    # would be the real symptom of a regression here).
    engine_bundle["classification_engine"].run(actor="test")

    all_suggestions = engine_bundle["classification_repo"].list_by_status(SuggestionStatus.AUTO_ACCEPTED)
    for suggestion in all_suggestions:
        if suggestion.object_id == investigation.id:
            assert suggestion.dimension not in (ClassificationDimension.TECHNOLOGY, ClassificationDimension.PRODUCT)


def test_known_bug_customer_relationship_is_idempotent_on_rerun(engine_bundle):
    lookup_repo = engine_bundle["lookup_repo"]
    customer = Customer(id=str(uuid.uuid4()), name="CLECO")
    lookup_repo.save_customer(customer)
    _save_known_bug(engine_bundle["knowledge_repo"], title="CLECO Meter Program Change Failure", description="Generic body.")

    first = engine_bundle["classification_engine"].run(actor="test")
    second = engine_bundle["classification_engine"].run(actor="test")

    assert first.relationships_created >= 1
    assert second.suggestions_created == 0


def test_classification_scan_counts_investigations_and_known_bugs(engine_bundle):
    _save_historical_investigation(engine_bundle["knowledge_repo"], title="Some investigation", description="x")
    _save_known_bug(engine_bundle["knowledge_repo"], title="Some bug", description="x")

    summary = engine_bundle["classification_engine"].run(actor="test")

    # At least the 2 new records, on top of whatever documents exist
    # (none, in this fixture's fresh DB) -- proves both new loops in
    # run() actually execute and advance the shared counter.
    assert summary.documents_scanned >= 2
