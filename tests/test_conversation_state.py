"""Tests for ConversationStateEngine (2026-08-14, Phase 3 -- Conversation
State + Persistence). Integration-style against a real temp-file SQLite
database, same pattern as test_classification.py/test_query_understanding.py
-- chat_sessions/chat_messages persistence, slot merge, and reference
resolution all genuinely depend on real repositories.

No LLM, no network. Every conversation transcript below mirrors one of
the 10 required examples from the approved Phase 3 request.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest

from app.domain.chat import ReferenceState
from app.domain.investigation import InvestigationSession
from app.domain.lookup_entities import Customer, Product, Region, Technology, Version
from app.domain.product_intelligence import ComponentProfile
from app.domain.query_understanding import SlotConfidence
from app.engines.chat.conversation_state import ConversationStateEngine
from app.engines.query_understanding.engine import QueryUnderstandingEngine
from app.infrastructure.db.chat_repository import SqlAlchemyChatRepository
from app.infrastructure.db.component_repository import SqlAlchemyComponentProfileRepository
from app.infrastructure.db.lookup_repository import SqlAlchemyLookupRepository
from app.infrastructure.db.repository import SqlAlchemyInvestigationRepository
from app.infrastructure.db.session import get_engine, get_session_factory


@pytest.fixture
def bundle():
    with tempfile.TemporaryDirectory() as tmp:
        sqlite_url = f"sqlite:///{(Path(tmp) / 'test.db').as_posix()}"
        session_factory = get_session_factory(sqlite_url)

        lookup_repo = SqlAlchemyLookupRepository(session_factory)
        component_repo = SqlAlchemyComponentProfileRepository(session_factory)
        chat_repo = SqlAlchemyChatRepository(session_factory)
        investigation_repo = SqlAlchemyInvestigationRepository(session_factory)

        lookup_repo.save_customer(Customer(id=str(uuid.uuid4()), name="TEPCO", aliases=["Tokyo Electric Power"]))
        lookup_repo.save_customer(Customer(id=str(uuid.uuid4()), name="CLECO"))
        lookup_repo.save_region(Region(id=str(uuid.uuid4()), name="APAC", aliases=["Asia Pacific"]))
        lookup_repo.save_product(Product(id=str(uuid.uuid4()), name="Command Center"))
        lookup_repo.save_version(Version(id=str(uuid.uuid4()), name="8.6.1.142"))

        rf_mesh_id = str(uuid.uuid4())
        rf_mesh_ip_id = str(uuid.uuid4())
        cellular_id = str(uuid.uuid4())
        lookup_repo.save_technology(Technology(id=rf_mesh_id, name="RF Mesh"))
        lookup_repo.save_technology(Technology(id=rf_mesh_ip_id, name="RF Mesh IP", parent_technology_id=rf_mesh_id))
        lookup_repo.save_technology(Technology(id=cellular_id, name="Cellular"))

        component_repo.save(ComponentProfile(id=str(uuid.uuid4()), name="Network Hub", product="Command Center"))
        component_repo.save(ComponentProfile(id=str(uuid.uuid4()), name="Device Hub", product="Command Center"))

        qu_engine = QueryUnderstandingEngine(lookup_repo, component_repo)
        state_engine = ConversationStateEngine(chat_repo, qu_engine, lookup_repo, investigation_repo)

        yield dict(
            state_engine=state_engine,
            chat_repo=chat_repo,
            lookup_repo=lookup_repo,
            investigation_repo=investigation_repo,
            rf_mesh_id=rf_mesh_id,
            rf_mesh_ip_id=rf_mesh_ip_id,
            cellular_id=cellular_id,
        )
        get_engine(sqlite_url).dispose()
        get_engine.cache_clear()


# --- Session creation / message persistence / chronological reconstruction --


def test_create_standalone_session(bundle):
    session = bundle["state_engine"].create_session()
    assert session.investigation_id is None
    assert session.slots.customer.value_id is None
    fetched = bundle["state_engine"].get_session(session.id)
    assert fetched is not None
    assert fetched.id == session.id


def test_create_investigation_scoped_session_seeds_slots(bundle):
    investigation = InvestigationSession(title="TEPCO RF Mesh issue", customer="TEPCO", technology="RF Mesh IP")
    bundle["investigation_repo"].save(investigation)

    session = bundle["state_engine"].create_session(investigation_id=investigation.id)
    assert session.investigation_id == investigation.id
    assert session.slots.customer.value_name == "TEPCO"
    assert session.slots.technology.value_name == "RF Mesh IP"
    assert session.slots.technology.value_id == bundle["rf_mesh_ip_id"]


def test_message_persistence_and_chronological_reconstruction(bundle):
    engine = bundle["state_engine"]
    session = engine.create_session()
    engine.add_user_message(session.id, "TEPCO RF Mesh IP command timeout.")
    engine.record_assistant_turn(session.id, "Found a real local match.", focus="historical_match")
    engine.add_user_message(session.id, "Has this happened before?")

    messages = engine.list_messages(session.id)
    assert [m.sequence for m in messages] == [1, 2, 3]
    assert [m.role.value for m in messages] == ["user", "assistant", "user"]
    assert messages[0].content == "TEPCO RF Mesh IP command timeout."
    assert messages[1].parsed_query is None  # assistant turns never run Query Understanding


def test_no_duplicated_knowledge_stored_in_chat_tables(bundle):
    """ChatMessage never stores a copy of matched-record content --
    only the real ParsedQuery extraction and (for assistant turns) the
    caller-supplied text. Assert the persisted message has no field
    holding investigation/document/TFS body content."""
    engine = bundle["state_engine"]
    session = engine.create_session()
    message = engine.add_user_message(session.id, "TEPCO RF Mesh IP command timeout.")
    dumped = message.model_dump(mode="json")
    assert set(dumped.keys()) == {"id", "session_id", "sequence", "role", "content", "parsed_query", "reference_resolution", "created_at"}
    # parsed_query holds only extraction slots/ids/names, never raw evidence bodies
    assert "raw_content" not in str(dumped["parsed_query"])


def test_idempotent_save_message(bundle):
    engine = bundle["state_engine"]
    session = engine.create_session()
    message = engine.add_user_message(session.id, "TEPCO issue.")
    bundle["chat_repo"].save_message(message)  # re-save the identical message
    messages = engine.list_messages(session.id)
    assert len(messages) == 1  # no duplicate row


# --- State persistence / reload / restart -----------------------------------


def test_state_persists_across_reload(bundle):
    engine = bundle["state_engine"]
    session = engine.create_session()
    engine.add_user_message(session.id, "TEPCO RF Mesh IP command timeout.")

    reloaded = engine.get_session(session.id)
    assert reloaded.slots.customer.value_name == "TEPCO"
    assert reloaded.slots.technology.value_name == "RF Mesh IP"


def test_restart_new_engine_instance_sees_same_state(bundle):
    """Simulates a process restart: a brand-new ConversationStateEngine
    (fresh QueryUnderstandingEngine too) backed by the same repository
    must see identical persisted state -- nothing lives only in memory."""
    engine = bundle["state_engine"]
    session = engine.create_session()
    engine.add_user_message(session.id, "TEPCO RF Mesh IP command timeout.")

    fresh_qu = QueryUnderstandingEngine(bundle["lookup_repo"])
    fresh_engine = ConversationStateEngine(bundle["chat_repo"], fresh_qu, bundle["lookup_repo"])
    reloaded = fresh_engine.get_session(session.id)
    assert reloaded.slots.customer.value_name == "TEPCO"
    messages = fresh_engine.list_messages(session.id)
    assert len(messages) == 1


# --- Conversation examples (verbatim from the approved Phase 3 request) -----


def test_example_1_and_2_and_3_context_carry_forward_and_reference_resolution(bundle):
    engine = bundle["state_engine"]
    session = engine.create_session()

    # Turn 1: "Show me similar cases for TEPCO RF Mesh IP."
    m1 = engine.add_user_message(session.id, "Show me similar cases for TEPCO RF Mesh IP.")
    assert m1.parsed_query.customer.value_name == "TEPCO"
    assert m1.parsed_query.technology.value_name == "RF Mesh IP"
    slots = engine.get_session(session.id).slots
    assert slots.customer.value_name == "TEPCO"
    assert slots.technology.value_name == "RF Mesh IP"
    assert slots.customer.origin == "stated"

    engine.record_assistant_turn(
        session.id, "Real local match found.", focus="historical_match", referenced_investigation_id="hist-1"
    )

    # Turn 2: "Has this happened before?" -- preserves TEPCO + RF Mesh IP,
    # resolves the reference to the current problem.
    m2 = engine.add_user_message(session.id, "Has this happened before?")
    assert m2.reference_resolution.state == ReferenceState.RESOLVED
    assert m2.reference_resolution.resolved_focus == "problem"
    slots = engine.get_session(session.id).slots
    assert slots.customer.value_name == "TEPCO"
    assert slots.technology.value_name == "RF Mesh IP"
    assert slots.customer.origin == "carried_forward"  # nothing in turn 2's text restated it

    engine.record_assistant_turn(session.id, "Yes, real local match hist-1.", focus="resolution")

    # Turn 3 (example 2): "What was the resolution?" -- preserves context.
    m3 = engine.add_user_message(session.id, "What was the resolution?")
    assert m3.reference_resolution.state == ReferenceState.RESOLVED
    assert m3.reference_resolution.resolved_focus == "resolution"
    slots = engine.get_session(session.id).slots
    assert slots.customer.value_name == "TEPCO"
    assert slots.technology.value_name == "RF Mesh IP"

    # Turn 4 (example 3): "Was that confirmed?" -- resolves against the
    # previous resolution/evidence subject, no new investigation invented.
    m4 = engine.add_user_message(session.id, "Was that confirmed?")
    assert m4.reference_resolution.state == ReferenceState.RESOLVED
    assert m4.reference_resolution.resolved_focus == "resolution"
    assert m4.reference_resolution.resolved_investigation_id == "hist-1"


def test_example_4_applicability_check_is_not_silently_merged(bundle):
    """'Does this apply to CLECO?' -- CLECO is visible as a real
    extraction on this turn, but must not silently replace or merge
    with the already-established TEPCO context."""
    engine = bundle["state_engine"]
    session = engine.create_session()
    engine.add_user_message(session.id, "Show me similar cases for TEPCO RF Mesh IP.")

    message = engine.add_user_message(session.id, "Does this apply to CLECO?")
    assert message.parsed_query.customer.value_name == "CLECO"  # real, visible extraction

    slots = engine.get_session(session.id).slots
    assert slots.customer.value_name == "TEPCO"  # active context unchanged
    assert slots.customer.origin == "carried_forward"


def test_example_5_rf_mesh_parent_mention_does_not_demote_rf_mesh_ip(bundle):
    engine = bundle["state_engine"]
    session = engine.create_session()
    engine.add_user_message(session.id, "TEPCO RF Mesh IP command timeout.")

    message = engine.add_user_message(session.id, "What about RF Mesh?")
    assert message.parsed_query.technology.value_name == "RF Mesh"  # real, visible extraction

    slots = engine.get_session(session.id).slots
    assert slots.technology.value_name == "RF Mesh IP"  # never demoted
    assert slots.technology.value_id == bundle["rf_mesh_ip_id"]


def test_example_6_bare_mesh_ip_preserves_established_rf_mesh_ip(bundle):
    engine = bundle["state_engine"]
    session = engine.create_session()
    engine.add_user_message(session.id, "TEPCO RF Mesh IP command timeout.")

    message = engine.add_user_message(session.id, "What about Mesh IP?")
    assert message.parsed_query.technology.value_name == "RF Mesh IP"
    assert message.parsed_query.technology.confidence == SlotConfidence.PARTIAL

    slots = engine.get_session(session.id).slots
    assert slots.technology.value_name == "RF Mesh IP"
    assert slots.technology.confidence == SlotConfidence.EXACT  # the stronger evidence is kept, not downgraded


def test_example_7_ticket_reference_persistence(bundle):
    engine = bundle["state_engine"]
    session = engine.create_session()
    engine.add_user_message(session.id, "TEPCO RF Mesh IP command timeout.")
    engine.add_user_message(session.id, "Use ticket INC0045821.")

    slots = engine.get_session(session.id).slots
    assert slots.ticket_references == ["INC0045821"]

    # A second, different ticket accumulates rather than replacing.
    engine.add_user_message(session.id, "Also related to CS0122697.")
    slots = engine.get_session(session.id).slots
    assert slots.ticket_references == ["CS0122697", "INC0045821"]


def test_example_8_explicit_correction_replaces_customer(bundle):
    engine = bundle["state_engine"]
    session = engine.create_session()
    engine.add_user_message(session.id, "Show me similar cases for TEPCO RF Mesh IP.")

    message = engine.add_user_message(session.id, "Actually, this is for CLECO.")
    assert message.parsed_query.customer.value_name == "CLECO"

    slots = engine.get_session(session.id).slots
    assert slots.customer.value_name == "CLECO"  # explicit correction replaces TEPCO
    assert slots.customer.origin == "stated"
    assert slots.customer.set_by_message_id == message.id


def test_example_9_ambiguous_reference_multiple_candidates(bundle):
    engine = bundle["state_engine"]
    session = engine.create_session()
    engine.add_user_message(session.id, "TEPCO RF Mesh IP command timeout.")
    engine.record_assistant_turn(
        session.id,
        "Found both a local match and a TFS case.",
        focus="historical_match",
        referenced_investigation_id="hist-1",
        referenced_tfs_id=2051535,
    )

    message = engine.add_user_message(session.id, "What about that one?")
    assert message.reference_resolution.state == ReferenceState.AMBIGUOUS
    assert len(message.reference_resolution.candidates) == 2


def test_example_9_ambiguous_reference_zero_candidates(bundle):
    """A reference cue with nothing at all established yet is also
    AMBIGUOUS -- never fabricated."""
    engine = bundle["state_engine"]
    session = engine.create_session()
    message = engine.add_user_message(session.id, "What about that one?")
    assert message.reference_resolution.state == ReferenceState.AMBIGUOUS


def test_example_10_existing_retrieval_context_precedence(bundle):
    """A new turn with an explicitly populated existing context (here:
    investigation-scoped session, seeded before any chat turn) must not
    be overwritten by weaker same-turn inference -- Phase 2 behavior,
    preserved end-to-end through Phase 3's persisted state."""
    investigation = InvestigationSession(title="TEPCO case", customer="TEPCO")
    bundle["investigation_repo"].save(investigation)
    engine = bundle["state_engine"]
    session = engine.create_session(investigation_id=investigation.id)

    message = engine.add_user_message(session.id, "This looks like a CLECO issue based on the symptoms.")
    assert message.parsed_query.customer.value_name == "CLECO"  # real extraction, not hidden

    slots = engine.get_session(session.id).slots
    assert slots.customer.value_name == "TEPCO"  # established context wins


# --- Ambiguous slot handling (component, generic dimension) -----------------


def test_ambiguous_component_slot_never_becomes_active(bundle):
    engine = bundle["state_engine"]
    session = engine.create_session()
    message = engine.add_user_message(session.id, "Network Hub is not routing messages to Device Hub.")
    assert message.parsed_query.component.confidence == SlotConfidence.AMBIGUOUS

    slots = engine.get_session(session.id).slots
    assert slots.component.value_id is None
