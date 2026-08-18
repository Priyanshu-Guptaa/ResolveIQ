"""Tests for QueryUnderstandingEngine (2026-08-14, Phase 2 -- Query
Understanding). Integration-style against a real temp-file SQLite
LookupRepository/ComponentProfileRepository, same pattern
``test_classification.py`` already established -- Customer/Region/
Product/Component/Version/Technology matching genuinely depends on the
real governed tables, so a fake would just re-assert the mock.

No LLM, no network, no DB writes performed by the engine itself
(``parse()`` only reads).
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest

from app.domain.lookup_entities import Customer, Product, Region, Technology, Version
from app.domain.product_intelligence import ComponentProfile
from app.domain.query_understanding import QueryIntent, SlotConfidence
from app.engines.knowledge.applicability import RetrievalContext
from app.engines.query_understanding.engine import QueryUnderstandingEngine
from app.infrastructure.db.component_repository import SqlAlchemyComponentProfileRepository
from app.infrastructure.db.lookup_repository import SqlAlchemyLookupRepository
from app.infrastructure.db.session import get_engine, get_session_factory


@pytest.fixture
def bundle():
    with tempfile.TemporaryDirectory() as tmp:
        sqlite_url = f"sqlite:///{(Path(tmp) / 'test.db').as_posix()}"
        session_factory = get_session_factory(sqlite_url)

        lookup_repo = SqlAlchemyLookupRepository(session_factory)
        component_repo = SqlAlchemyComponentProfileRepository(session_factory)

        # Real, representative governed data -- same TEPCO/RF-Mesh-family/
        # Network-Hub fixtures the rest of this session's live
        # verification has used throughout.
        lookup_repo.save_customer(Customer(id=str(uuid.uuid4()), name="TEPCO", aliases=["Tokyo Electric Power"]))
        lookup_repo.save_customer(Customer(id=str(uuid.uuid4()), name="CLECO"))
        lookup_repo.save_region(Region(id=str(uuid.uuid4()), name="APAC", aliases=["Asia Pacific"]))
        lookup_repo.save_region(Region(id=str(uuid.uuid4()), name="NAM"))
        lookup_repo.save_product(Product(id=str(uuid.uuid4()), name="Command Center"))
        lookup_repo.save_version(Version(id=str(uuid.uuid4()), name="8.6.1.142"))

        rf_mesh_id = str(uuid.uuid4())
        rf_mesh_ip_id = str(uuid.uuid4())
        lookup_repo.save_technology(Technology(id=rf_mesh_id, name="RF Mesh"))
        lookup_repo.save_technology(Technology(id=rf_mesh_ip_id, name="RF Mesh IP", parent_technology_id=rf_mesh_id))
        lookup_repo.save_technology(Technology(id=str(uuid.uuid4()), name="Cellular"))

        component_repo.save(ComponentProfile(id=str(uuid.uuid4()), name="Network Hub", product="Command Center"))
        component_repo.save(ComponentProfile(id=str(uuid.uuid4()), name="Device Hub", product="Command Center"))

        engine = QueryUnderstandingEngine(lookup_repo, component_repo)
        yield dict(engine=engine, lookup_repo=lookup_repo, component_repo=component_repo)
        get_engine(sqlite_url).dispose()
        get_engine.cache_clear()


# --- Intent classification -----------------------------------------------


def test_intent_known_bug_lookup(bundle):
    parsed = bundle["engine"].parse("Is this a known bug in the collector?")
    assert parsed.intent == QueryIntent.KNOWN_BUG_LOOKUP
    assert parsed.intent_confidence > 0.0


def test_intent_historical_lookup(bundle):
    parsed = bundle["engine"].parse("Have we seen this before? Looking for a similar case.")
    assert parsed.intent == QueryIntent.HISTORICAL_LOOKUP


def test_intent_log_guidance(bundle):
    parsed = bundle["engine"].parse("Which logs should I collect for this issue?")
    assert parsed.intent == QueryIntent.LOG_GUIDANCE


def test_intent_troubleshooting_generic(bundle):
    parsed = bundle["engine"].parse("Commands are failing to reach the endpoint, timeout after 30s.")
    assert parsed.intent == QueryIntent.TROUBLESHOOTING


def test_intent_ambiguous_evidence_returns_unknown(bundle):
    """Deliberately constructed so LOG_GUIDANCE and TROUBLESHOOTING each
    score exactly one matched cue phrase -- a genuine tie, no safe
    winner. Must be UNKNOWN with 0.0 confidence, never a guess."""
    text = "issue: which log is relevant here"
    parsed = bundle["engine"].parse(text)
    assert parsed.intent == QueryIntent.UNKNOWN
    assert parsed.intent_confidence == 0.0


def test_intent_empty_text_is_unknown(bundle):
    parsed = bundle["engine"].parse("")
    assert parsed.intent == QueryIntent.UNKNOWN
    assert parsed.intent_confidence == 0.0


# --- Customer / Region --------------------------------------------------


def test_customer_exact_match_tepco(bundle):
    parsed = bundle["engine"].parse("TEPCO reported an issue with command delivery.")
    assert parsed.customer is not None
    assert parsed.customer.value_name == "TEPCO"
    assert parsed.customer.confidence == SlotConfidence.EXACT
    assert parsed.retrieval_context.customer_name == "TEPCO"


def test_customer_no_match_is_none(bundle):
    parsed = bundle["engine"].parse("Generic connectivity issue with no customer mentioned.")
    assert parsed.customer is None


def test_customer_name_never_becomes_technology_or_component(bundle):
    parsed = bundle["engine"].parse("TEPCO reported an issue.")
    assert parsed.technology is None
    assert parsed.component is None


def test_region_match(bundle):
    parsed = bundle["engine"].parse("This is affecting our APAC deployment.")
    assert parsed.region is not None
    assert parsed.region.value_name == "APAC"
    assert parsed.region.confidence == SlotConfidence.EXACT


# --- Technology (RF Mesh / RF Mesh IP / Mesh IP regression) --------------


def test_technology_rf_mesh_exact(bundle):
    parsed = bundle["engine"].parse("RF Mesh command timeout across endpoints.")
    assert parsed.technology is not None
    assert parsed.technology.value_name == "RF Mesh"
    assert parsed.technology.confidence == SlotConfidence.EXACT


def test_technology_rf_mesh_ip_not_downgraded_to_rf_mesh(bundle):
    parsed = bundle["engine"].parse("RF Mesh IP communication failure, collector unreachable.")
    assert parsed.technology is not None
    assert parsed.technology.value_name == "RF Mesh IP"
    assert parsed.technology.confidence == SlotConfidence.EXACT


def test_technology_bare_mesh_ip_resolves_to_rf_mesh_ip_partial(bundle):
    parsed = bundle["engine"].parse("Mesh IP collector unable to reach endpoints.")
    assert parsed.technology is not None
    assert parsed.technology.value_name == "RF Mesh IP"
    assert parsed.technology.confidence == SlotConfidence.PARTIAL


def test_technology_generic_text_no_match(bundle):
    parsed = bundle["engine"].parse("The network is generally slow today.")
    assert parsed.technology is None


# --- Component (strict full-name match) -----------------------------------


def test_component_full_name_match(bundle):
    parsed = bundle["engine"].parse("Network Hub is not routing messages to Device Hub.")
    assert parsed.component is not None
    assert parsed.component.confidence == SlotConfidence.AMBIGUOUS
    assert set(parsed.component.ambiguous_candidates) == {"Network Hub", "Device Hub"}


def test_component_single_full_name_match(bundle):
    parsed = bundle["engine"].parse("Network Hub is not routing messages downstream.")
    assert parsed.component is not None
    assert parsed.component.value_name == "Network Hub"
    assert parsed.component.confidence == SlotConfidence.EXACT


def test_component_generic_word_is_not_a_match(bundle):
    """Real Phase 0 regression: a single generic word ('network') must
    never be enough to create a component match."""
    parsed = bundle["engine"].parse("We have a network issue affecting several sites.")
    assert parsed.component is None


# --- Product / Version -----------------------------------------------------


def test_product_match(bundle):
    parsed = bundle["engine"].parse("This is a Command Center deployment question.")
    assert parsed.product is not None
    assert parsed.product.value_name == "Command Center"
    assert parsed.product.confidence == SlotConfidence.EXACT


def test_product_no_match_returns_none(bundle):
    parsed = bundle["engine"].parse("No product mentioned here at all.")
    assert parsed.product is None


def test_version_match(bundle):
    parsed = bundle["engine"].parse("Running CC 8.6.1.142 in the KTST environment.")
    assert parsed.version is not None
    assert parsed.version.value_name == "8.6.1.142"
    assert parsed.version.confidence == SlotConfidence.EXACT


def test_version_no_match_returns_none(bundle):
    parsed = bundle["engine"].parse("No version string anywhere in this text.")
    assert parsed.version is None


# --- Exception type / ticket references -------------------------------------


def test_exception_type_extraction(bundle):
    parsed = bundle["engine"].parse("The service threw a NullPointerException during processing.")
    assert parsed.exception_type is not None
    assert parsed.exception_type.value_name == "NullPointerException"
    assert parsed.exception_type.confidence == SlotConfidence.EXACT


def test_ticket_reference_extraction(bundle):
    parsed = bundle["engine"].parse("Related to CS0122697 and INC0045821.")
    assert parsed.ticket_references == ["CS0122697", "INC0045821"]


def test_no_ticket_reference_is_empty_list(bundle):
    parsed = bundle["engine"].parse("No ticket numbers mentioned.")
    assert parsed.ticket_references == []


# --- existing_context wins --------------------------------------------------


def test_existing_context_customer_is_never_overwritten(bundle):
    """The TEPCO/CLECO example from the approved Phase 2 spec: existing
    context already says TEPCO; free text mentions CLECO. Retrieval
    must keep TEPCO. The real CLECO evidence extracted from the text
    itself is still visible on the slot -- nothing is hidden, only
    retrieval prefers the already-established value."""
    existing = RetrievalContext(customer_id="tepco-id", customer_name="TEPCO")
    parsed = bundle["engine"].parse("This looks like a CLECO issue based on the symptoms.", existing_context=existing)

    assert parsed.retrieval_context.customer_name == "TEPCO"
    assert parsed.retrieval_context.customer_id == "tepco-id"
    # The real extraction from the text itself is not suppressed.
    assert parsed.customer is not None
    assert parsed.customer.value_name == "CLECO"


def test_existing_context_technology_is_never_overwritten(bundle):
    existing = RetrievalContext(technology_name="Cellular")
    parsed = bundle["engine"].parse("RF Mesh command timeout across endpoints.", existing_context=existing)
    assert parsed.retrieval_context.technology_name == "Cellular"
    assert parsed.technology is not None
    assert parsed.technology.value_name == "RF Mesh"


def test_no_existing_context_populates_from_exact_slots(bundle):
    parsed = bundle["engine"].parse("TEPCO reported an RF Mesh IP failure.")
    assert parsed.retrieval_context.customer_name == "TEPCO"
    assert parsed.retrieval_context.technology_name == "RF Mesh IP"


def test_ambiguous_slot_never_populates_retrieval_context(bundle):
    parsed = bundle["engine"].parse("Network Hub is not routing messages to Device Hub.")
    # Component has no RetrievalContext field at all (documented
    # limitation), but confirm the ambiguous match itself never
    # silently becomes a guessed technology/customer either.
    assert parsed.retrieval_context.customer_name is None
    assert parsed.retrieval_context.technology_name is None
