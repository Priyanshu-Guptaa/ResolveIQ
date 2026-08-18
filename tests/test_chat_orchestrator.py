"""Tests for ChatOrchestrator (2026-08-14, Phase 4 -- Chat Orchestrator +
API + UI). Integration-style against a real temp-file SQLite database,
same pattern as test_structured_resolution.py -- a configurable fake
KnowledgeStore (so local-match provenance tiers are reachable
end-to-end through the real, public RecommendationEngine.generate(),
never bypassed) stands in for ChromaDB/the embedding model, exactly the
discipline test_structured_resolution.py already established.

No LLM, no network, no TFS/Wiki live calls (External Knowledge is left
unwired in this bundle -- tfs_matches/wiki_matches stay None, which is
itself asserted as the correct "not configured" contract).
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.domain.chat import ChatAmbiguityKind, MessageRole
from app.domain.enums import KnowledgeCollection
from app.domain.evidence import HistoricalInvestigationRecord, KnownBugRecord
from app.domain.lookup_entities import Customer, Product, Region, Technology, Version
from app.domain.product_intelligence import ComponentProfile
from app.domain.provenance import ResolutionProvenance
from app.domain.query_understanding import SlotConfidence
from app.domain.recommendation import KnowledgeMatch
from app.engines.chat.conversation_state import ConversationStateEngine
from app.engines.chat.orchestrator import ChatOrchestrator, ChatSessionNotFoundError, EmptyMessageError
from app.engines.external_knowledge.service import ExternalKnowledgeService
from app.engines.investigation.engine import InvestigationEngine, InvestigationNotFoundError
from app.engines.knowledge.engine import KnowledgeEngine
from app.engines.knowledge_relationships.engine import KnowledgeRelationshipEngine
from app.engines.query_understanding.engine import QueryUnderstandingEngine
from app.engines.recommendation.engine import RecommendationEngine
from app.infrastructure.db.chat_repository import SqlAlchemyChatRepository
from app.infrastructure.db.component_repository import SqlAlchemyComponentProfileRepository
from app.infrastructure.db.knowledge_repository import SqlAlchemyKnowledgeRepository
from app.infrastructure.db.log_knowledge_repository import SqlAlchemyLogKnowledgeRepository
from app.infrastructure.db.lookup_repository import SqlAlchemyLookupRepository
from app.infrastructure.db.playbook_repository import SqlAlchemyPlaybookRepository
from app.infrastructure.db.relationship_repository import SqlAlchemyRelationshipRepository
from app.infrastructure.db.repository import SqlAlchemyInvestigationRepository
from app.infrastructure.db.session import get_engine, get_session_factory
from app.infrastructure.db.sql_template_repository import SqlAlchemySqlTemplateRepository


class _NoOpTfsConnector:
    """Reports 'not configured' -- ExternalKnowledgeService's own
    is_configured() gate then returns available=False immediately, no
    network I/O, no real TFS call -- see service.py's own gather()/
    _run_tfs(). Wired (rather than leaving external_knowledge=None
    entirely) so RecommendationEngine.generate() populates
    ``recommended_solution``/``structured_resolution`` end-to-end
    exactly as it does in the real, deployed app (where External
    Knowledge is always wired) -- see
    InvestigationStrategy.recommended_solution's own docstring: "None
    only when External Knowledge isn't wired in this build.\""""

    def is_configured(self) -> bool:
        return False

    def search_work_items(self, **kwargs):  # pragma: no cover -- never reached, not configured
        return []

    def get_work_item(self, tfs_id: int):  # pragma: no cover
        return None

    def get_work_item_history(self, tfs_id: int):  # pragma: no cover
        return None


class _NoOpWikiConnector:
    def is_configured(self) -> bool:
        return False

    def search(self, **kwargs):  # pragma: no cover -- never reached, not configured
        return []


class ConfigurableFakeKnowledgeStore:
    """Same Protocol-satisfying shape as every other fake store in this
    suite (test_provenance.py, test_structured_resolution.py) --
    ``query()`` returns pre-seeded ``KnowledgeMatch`` lists per
    collection instead of always ``[]``, so a real end-to-end
    ``RecommendationEngine.generate()`` call can reach a real LIKELY/
    POSSIBLE/CONFIRMED tier without a real embedding model."""

    def __init__(self) -> None:
        self.matches: dict[KnowledgeCollection, list[KnowledgeMatch]] = {}

    def upsert(self, collection, record_id, text, title, metadata) -> None:  # pragma: no cover
        pass

    def query(self, collection, text, top_k: int = 5):
        return list(self.matches.get(collection, []))[:top_k]

    def count(self, collection) -> int:  # pragma: no cover
        return 0

    def list_recent(self, collection, limit: int = 5):  # pragma: no cover
        return []

    def delete(self, collection, record_id: str) -> None:  # pragma: no cover
        pass


@pytest.fixture
def bundle():
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
        investigation_repo = SqlAlchemyInvestigationRepository(session_factory)
        chat_repo = SqlAlchemyChatRepository(session_factory)

        relationship_engine = KnowledgeRelationshipEngine(
            relationship_repo, component_repo, knowledge_repo, sql_repo, playbook_repo, lookup_repo, log_knowledge_repo
        )
        store = ConfigurableFakeKnowledgeStore()
        knowledge_engine = KnowledgeEngine(store, knowledge_repo)
        external_knowledge = ExternalKnowledgeService(
            tfs_connector=_NoOpTfsConnector(), wiki_connector=_NoOpWikiConnector(), settings=Settings()
        )
        rec_engine = RecommendationEngine(
            knowledge_engine, Settings(), component_repo=component_repo, relationship_engine=relationship_engine,
            lookup_repo=lookup_repo, external_knowledge=external_knowledge,
        )
        investigation_engine = InvestigationEngine(investigation_repo, None, None)  # log_intelligence/ingestion unused by these tests
        qu_engine = QueryUnderstandingEngine(lookup_repo, component_repo)
        state_engine = ConversationStateEngine(chat_repo, qu_engine, lookup_repo, investigation_repo)
        orchestrator = ChatOrchestrator(state_engine, rec_engine, investigation_engine)

        lookup_repo.save_customer(Customer(id=str(uuid.uuid4()), name="TEPCO"))
        lookup_repo.save_customer(Customer(id=str(uuid.uuid4()), name="CLECO"))
        lookup_repo.save_region(Region(id=str(uuid.uuid4()), name="APAC"))
        lookup_repo.save_product(Product(id=str(uuid.uuid4()), name="Command Center"))
        lookup_repo.save_version(Version(id=str(uuid.uuid4()), name="8.6.1.142"))
        rf_mesh_id = str(uuid.uuid4())
        rf_mesh_ip_id = str(uuid.uuid4())
        lookup_repo.save_technology(Technology(id=rf_mesh_id, name="RF Mesh"))
        lookup_repo.save_technology(Technology(id=rf_mesh_ip_id, name="RF Mesh IP", parent_technology_id=rf_mesh_id))
        component_repo.save(ComponentProfile(id=str(uuid.uuid4()), name="Network Hub", product="Command Center"))
        component_repo.save(ComponentProfile(id=str(uuid.uuid4()), name="Device Hub", product="Command Center"))

        yield dict(
            orchestrator=orchestrator,
            state_engine=state_engine,
            rec_engine=rec_engine,
            investigation_engine=investigation_engine,
            knowledge_repo=knowledge_repo,
            lookup_repo=lookup_repo,
            store=store,
            rf_mesh_ip_id=rf_mesh_ip_id,
        )
        get_engine(sqlite_url).dispose()
        get_engine.cache_clear()


def _save_hi(knowledge_repo, **overrides) -> HistoricalInvestigationRecord:
    defaults = dict(
        id=str(uuid.uuid4()), title="RF Mesh IP command timeout", description="Meters stopped responding.",
        root_cause="Collector lost network route to the mesh gateway.", resolution="Restart the collector service.",
        next_step="Confirm the collector's route table is restored.",
    )
    defaults.update(overrides)
    record = HistoricalInvestigationRecord(**defaults)
    knowledge_repo.save_historical_investigation(record)
    return record


def _save_bug(knowledge_repo, **overrides) -> KnownBugRecord:
    defaults = dict(
        id=str(uuid.uuid4()), title="Known RF Mesh IP collector bug", description="Collector loses route after patch.",
        workaround="Do NOT restart the collector service; apply configuration change Y instead.",
    )
    defaults.update(overrides)
    record = KnownBugRecord(**defaults)
    knowledge_repo.save_known_bug(record)
    return record


def _bug_match_for(record, score) -> KnowledgeMatch:
    return KnowledgeMatch(
        collection=KnowledgeCollection.KNOWN_BUGS, record_id=record.id, title=record.title,
        snippet=record.description, score=score,
        metadata={"workaround": record.workaround, "tags": ""},
    )


def _match_for(record, score, tags: str = "") -> KnowledgeMatch:
    return KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS, record_id=record.id, title=record.title,
        snippet=record.description, score=score,
        metadata={"root_cause": record.root_cause, "resolution": record.resolution, "next_step": record.next_step, "tags": tags},
    )


# --- A. Session --------------------------------------------------------


def test_create_standalone_session(bundle):
    session = bundle["orchestrator"].create_session()
    assert session.investigation_id is None


def test_create_investigation_scoped_session(bundle):
    investigation = bundle["investigation_engine"].start_investigation("RF Mesh IP meters stopped sending reads")
    session = bundle["orchestrator"].create_session(investigation.id)
    assert session.investigation_id == investigation.id


def test_create_session_invalid_investigation_raises(bundle):
    with pytest.raises(InvestigationNotFoundError):
        bundle["orchestrator"].create_session("does-not-exist")


# --- B. Message flow -----------------------------------------------------


def test_user_message_and_assistant_response_persist_and_reload(bundle):
    orchestrator = bundle["orchestrator"]
    session = orchestrator.create_session()
    response = orchestrator.handle_message(session.id, "Commands stuck in Pending status.")
    assert response.answer_text

    messages = orchestrator.list_messages(session.id)
    assert [m.sequence for m in messages] == [1, 2]
    assert messages[0].role == MessageRole.USER
    assert messages[1].role == MessageRole.ASSISTANT
    assert messages[1].content == response.answer_text


def test_chronological_ordering_across_multiple_turns(bundle):
    orchestrator = bundle["orchestrator"]
    session = orchestrator.create_session()
    orchestrator.handle_message(session.id, "Commands stuck in Pending status.")
    orchestrator.handle_message(session.id, "Has this happened before?")
    messages = orchestrator.list_messages(session.id)
    assert [m.sequence for m in messages] == [1, 2, 3, 4]


def test_empty_message_raises(bundle):
    session = bundle["orchestrator"].create_session()
    with pytest.raises(EmptyMessageError):
        bundle["orchestrator"].handle_message(session.id, "   ")


def test_invalid_session_raises(bundle):
    with pytest.raises(ChatSessionNotFoundError):
        bundle["orchestrator"].handle_message("does-not-exist", "hello")


# --- C. Query Understanding integration through chat ----------------------


def test_tepco_customer_recognized_in_chat(bundle):
    response = bundle["orchestrator"].handle_message(bundle["orchestrator"].create_session().id, "TEPCO reported an RF Mesh IP issue.")
    assert response.parsed_query.customer.value_name == "TEPCO"
    assert response.active_context.customer_name == "TEPCO"


def test_cleco_customer_recognized_in_chat(bundle):
    response = bundle["orchestrator"].handle_message(bundle["orchestrator"].create_session().id, "CLECO reported a meter issue.")
    assert response.parsed_query.customer.value_name == "CLECO"


def test_rf_mesh_recognized_in_chat(bundle):
    response = bundle["orchestrator"].handle_message(bundle["orchestrator"].create_session().id, "RF Mesh command timeout.")
    assert response.parsed_query.technology.value_name == "RF Mesh"
    assert response.active_context.technology_name == "RF Mesh"


def test_rf_mesh_ip_not_demoted_to_rf_mesh_in_chat(bundle):
    response = bundle["orchestrator"].handle_message(bundle["orchestrator"].create_session().id, "RF Mesh IP command timeout.")
    assert response.parsed_query.technology.value_name == "RF Mesh IP"
    assert response.active_context.technology_name == "RF Mesh IP"


def test_bare_mesh_ip_recognized_in_chat(bundle):
    response = bundle["orchestrator"].handle_message(bundle["orchestrator"].create_session().id, "Mesh IP collector unreachable.")
    assert response.parsed_query.technology.value_name == "RF Mesh IP"
    assert response.parsed_query.technology.confidence == SlotConfidence.PARTIAL


def test_ticket_reference_recognized_in_chat(bundle):
    response = bundle["orchestrator"].handle_message(bundle["orchestrator"].create_session().id, "Related to CS0122697.")
    assert response.parsed_query.ticket_references == ["CS0122697"]


def test_exception_type_recognized_in_chat(bundle):
    response = bundle["orchestrator"].handle_message(
        bundle["orchestrator"].create_session().id, "Service threw a NullPointerException during startup."
    )
    assert response.parsed_query.exception_type.value_name == "NullPointerException"


# --- D. Conversation behavior (Phase 3 reused exactly) ---------------------


def test_has_this_happened_before_preserves_context(bundle):
    orchestrator = bundle["orchestrator"]
    session = orchestrator.create_session()
    orchestrator.handle_message(session.id, "Show me similar cases for TEPCO RF Mesh IP.")
    response = orchestrator.handle_message(session.id, "Has this happened before?")
    assert response.active_context.customer_name == "TEPCO"
    assert response.active_context.technology_name == "RF Mesh IP"


def test_was_that_confirmed_resolves_against_prior_focus(bundle):
    orchestrator = bundle["orchestrator"]
    knowledge_repo = bundle["knowledge_repo"]
    record = _save_hi(knowledge_repo)
    bundle["store"].matches[KnowledgeCollection.HISTORICAL_INVESTIGATIONS] = [_match_for(record, 0.82)]

    session = orchestrator.create_session()
    orchestrator.handle_message(session.id, "RF Mesh IP command timeout.")
    response = orchestrator.handle_message(session.id, "Was that confirmed?")
    # The turn itself must not fabricate a new investigation -- answer text
    # is still driven by real, already-computed provenance, and the real
    # matched record from the previous turn is what backs it.
    assert response.resolution_provenance == ResolutionProvenance.LIKELY
    assert response.structured_resolution is not None
    assert response.structured_resolution.resolution_candidates[0].evidence.source_id == record.id


def test_what_was_the_resolution_preserves_context(bundle):
    orchestrator = bundle["orchestrator"]
    session = orchestrator.create_session()
    orchestrator.handle_message(session.id, "TEPCO RF Mesh IP command timeout.")
    response = orchestrator.handle_message(session.id, "What was the resolution?")
    assert response.active_context.customer_name == "TEPCO"


def test_does_this_apply_to_cleco_does_not_overwrite_tepco(bundle):
    orchestrator = bundle["orchestrator"]
    session = orchestrator.create_session()
    orchestrator.handle_message(session.id, "Show me similar cases for TEPCO RF Mesh IP.")
    response = orchestrator.handle_message(session.id, "Does this apply to CLECO?")
    assert response.parsed_query.customer.value_name == "CLECO"  # real, visible extraction
    assert response.active_context.customer_name == "TEPCO"  # active context unchanged


def test_actually_this_is_for_cleco_replaces_tepco(bundle):
    orchestrator = bundle["orchestrator"]
    session = orchestrator.create_session()
    orchestrator.handle_message(session.id, "Show me similar cases for TEPCO RF Mesh IP.")
    response = orchestrator.handle_message(session.id, "Actually, this is for CLECO.")
    assert response.active_context.customer_name == "CLECO"


def test_what_about_rf_mesh_does_not_demote_established_rf_mesh_ip(bundle):
    orchestrator = bundle["orchestrator"]
    session = orchestrator.create_session()
    orchestrator.handle_message(session.id, "RF Mesh IP command timeout.")
    response = orchestrator.handle_message(session.id, "What about RF Mesh?")
    assert response.active_context.technology_name == "RF Mesh IP"


def test_what_about_mesh_ip_preserves_established_rf_mesh_ip(bundle):
    orchestrator = bundle["orchestrator"]
    session = orchestrator.create_session()
    orchestrator.handle_message(session.id, "RF Mesh IP command timeout.")
    response = orchestrator.handle_message(session.id, "What about Mesh IP?")
    assert response.active_context.technology_name == "RF Mesh IP"


def test_ambiguous_that_one_asks_for_clarification_without_guessing(bundle):
    orchestrator = bundle["orchestrator"]
    state_engine = bundle["state_engine"]
    session = orchestrator.create_session()
    orchestrator.handle_message(session.id, "RF Mesh IP command timeout.")
    # Simulate two plausible prior anchors, as a real answer would set.
    state_engine.record_assistant_turn(
        session.id, "Found both a local match and a TFS case.", focus="historical_match",
        referenced_investigation_id="hist-1", referenced_tfs_id=2051535,
    )
    response = orchestrator.handle_message(session.id, "What about that one?")
    assert response.ambiguity.kind == ChatAmbiguityKind.REFERENCE_AMBIGUOUS
    assert response.follow_up_question is not None
    # No retrieval was invented for this turn -- no historical matches attached.
    assert response.historical_investigations == []
    assert response.structured_resolution is None


# --- E. Evidence -----------------------------------------------------------


def test_evidence_is_attributed_with_real_source(bundle):
    orchestrator = bundle["orchestrator"]
    record = _save_hi(bundle["knowledge_repo"])
    bundle["store"].matches[KnowledgeCollection.HISTORICAL_INVESTIGATIONS] = [_match_for(record, 0.82)]
    response = orchestrator.handle_message(orchestrator.create_session().id, "RF Mesh IP command timeout, meters not responding.")
    assert response.structured_resolution is not None
    for candidate in response.structured_resolution.resolution_candidates:
        assert candidate.evidence.source_id  # every candidate traces to a real id
        assert candidate.evidence.title


def test_no_fabricated_evidence_when_nothing_matches(bundle):
    orchestrator = bundle["orchestrator"]
    response = orchestrator.handle_message(orchestrator.create_session().id, "zzz nonsense placeholder xyz qwqwqw gibberish")
    assert response.historical_investigations == []
    assert response.known_bugs == []
    if response.structured_resolution is not None:
        assert response.structured_resolution.resolution_candidates == []


def test_conflicting_evidence_is_preserved_not_hidden(bundle):
    """Two DIFFERENT source types (local historical investigation vs.
    Known Bug) disagreeing -- the real "restart X" vs. "do NOT restart
    X" example from Phase 1's own conflict-preservation guarantee.
    Within a single source type only the single best-scored match
    becomes a candidate (existing, correct behavior from Phase 0/1,
    unrelated to Phase 4); across source types, every real candidate
    with resolution content survives."""
    orchestrator = bundle["orchestrator"]
    knowledge_repo = bundle["knowledge_repo"]
    r1 = _save_hi(knowledge_repo, title="RF Mesh IP collector restart procedure", resolution="Restart the collector service.")
    bug = _save_bug(knowledge_repo)
    bundle["store"].matches[KnowledgeCollection.HISTORICAL_INVESTIGATIONS] = [_match_for(r1, 0.85)]
    bundle["store"].matches[KnowledgeCollection.KNOWN_BUGS] = [_bug_match_for(bug, 0.7)]
    response = orchestrator.handle_message(orchestrator.create_session().id, "RF Mesh IP command timeout, meters not responding.")
    assert response.structured_resolution is not None
    kinds = {c.evidence.kind for c in response.structured_resolution.resolution_candidates}
    resolution_texts = " | ".join(c.text for c in response.structured_resolution.resolution_candidates)
    assert len(response.structured_resolution.resolution_candidates) >= 2
    assert "Restart the collector service" in resolution_texts
    assert "Do NOT restart the collector service" in resolution_texts  # the conflicting alternate is never silently dropped


# --- F. Trust language -------------------------------------------------------


def test_confirmed_never_from_similarity_alone(bundle):
    orchestrator = bundle["orchestrator"]
    record = _save_hi(bundle["knowledge_repo"])  # not resolution_verified, no TFS correlation
    bundle["store"].matches[KnowledgeCollection.HISTORICAL_INVESTIGATIONS] = [_match_for(record, 0.99)]  # very high score alone
    response = orchestrator.handle_message(orchestrator.create_session().id, "RF Mesh IP command timeout, meters not responding.")
    assert response.resolution_provenance != ResolutionProvenance.CONFIRMED
    assert "confirm" not in response.answer_text.lower()


def test_confirmed_reachable_via_human_verification_and_uses_confirmed_language(bundle):
    orchestrator = bundle["orchestrator"]
    record = _save_hi(
        bundle["knowledge_repo"], resolution_verified=True, resolution_verified_by="admin",
    )
    bundle["store"].matches[KnowledgeCollection.HISTORICAL_INVESTIGATIONS] = [_match_for(record, 0.82)]
    response = orchestrator.handle_message(orchestrator.create_session().id, "RF Mesh IP command timeout, meters not responding.")
    assert response.resolution_provenance == ResolutionProvenance.CONFIRMED
    assert "confirmed" in response.answer_text.lower()


def test_likely_never_says_confirmed(bundle):
    orchestrator = bundle["orchestrator"]
    record = _save_hi(bundle["knowledge_repo"])
    bundle["store"].matches[KnowledgeCollection.HISTORICAL_INVESTIGATIONS] = [_match_for(record, 0.82)]
    response = orchestrator.handle_message(orchestrator.create_session().id, "RF Mesh IP command timeout, meters not responding.")
    assert response.resolution_provenance == ResolutionProvenance.LIKELY
    assert "confirm" not in response.answer_text.lower()
    assert "likely" in response.answer_text.lower()


def test_possible_uses_hedged_language(bundle):
    orchestrator = bundle["orchestrator"]
    record = _save_hi(bundle["knowledge_repo"], root_cause="")
    bundle["store"].matches[KnowledgeCollection.HISTORICAL_INVESTIGATIONS] = [_match_for(record, 0.5)]
    response = orchestrator.handle_message(orchestrator.create_session().id, "RF Mesh IP command timeout, meters not responding.")
    if response.resolution_provenance == ResolutionProvenance.POSSIBLE:
        assert "possible" in response.answer_text.lower()
        assert "confirm" not in response.answer_text.lower()


def test_unknown_does_not_invent_a_resolution(bundle):
    orchestrator = bundle["orchestrator"]
    response = orchestrator.handle_message(orchestrator.create_session().id, "zzz nonsense placeholder xyz qwqwqw gibberish")
    if response.resolution_provenance in (None, ResolutionProvenance.UNKNOWN):
        assert "don't have enough evidence" in response.answer_text.lower()


# --- G/H. API + response contract -------------------------------------------


@pytest.fixture
def api_client(bundle):
    from app.api.dependencies import get_chat_orchestrator
    from app.api.main import app

    app.dependency_overrides[get_chat_orchestrator] = lambda: bundle["orchestrator"]
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.pop(get_chat_orchestrator, None)


def test_api_create_session_and_send_message(api_client):
    created = api_client.post("/chat/sessions", json={"investigation_id": None})
    assert created.status_code == 201
    session_id = created.json()["id"]

    posted = api_client.post(f"/chat/sessions/{session_id}/messages", json={"message": "RF Mesh IP command timeout."})
    assert posted.status_code == 200
    body = posted.json()
    # UI contract: every field the Streamlit page reads must be present.
    for field in (
        "answer_text", "intent", "active_context", "parsed_query", "ambiguity", "follow_up_question",
        "structured_resolution", "resolution_provenance", "historical_investigations", "known_bugs",
        "documentation", "recommended_logs", "suggested_sql", "tfs_matches", "wiki_matches", "investigation_id",
    ):
        assert field in body

    history = api_client.get(f"/chat/sessions/{session_id}/messages")
    assert history.status_code == 200
    assert len(history.json()) == 2


def test_api_invalid_session_returns_404(api_client):
    response = api_client.post("/chat/sessions/does-not-exist/messages", json={"message": "hello"})
    assert response.status_code == 404
    assert "Traceback" not in response.text  # no stack trace leaked to the client


def test_api_invalid_investigation_returns_404(api_client):
    response = api_client.post("/chat/sessions", json={"investigation_id": "does-not-exist"})
    assert response.status_code == 404


def test_api_empty_message_returns_400(api_client):
    created = api_client.post("/chat/sessions", json={"investigation_id": None})
    session_id = created.json()["id"]
    response = api_client.post(f"/chat/sessions/{session_id}/messages", json={"message": "   "})
    assert response.status_code == 400


def test_api_get_nonexistent_session_returns_404(api_client):
    response = api_client.get("/chat/sessions/does-not-exist")
    assert response.status_code == 404
