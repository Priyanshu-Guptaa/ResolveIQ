"""Tests for the External Knowledge (TFS + Wiki live connector)
feature -- fully offline, fake connectors implementing the
TfsConnector/WikiConnector Protocols (same discipline as
FakeLogKnowledgeRepo/FakeKnowledgeStore elsewhere in this suite). No
test here ever makes a real network call.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from app.config import Settings
from app.domain.entities import ExtractedEntity
from app.domain.enums import EntityType
from app.domain.external_knowledge import TfsCase, WikiPage
from app.domain.recommendation import MatchedComponent
from app.engines.external_knowledge.cache import TtlCache
from app.engines.external_knowledge.extraction import extract_and_truncate, strip_html, truncate
from app.engines.external_knowledge.ranking import confidence_for_score, score_candidate
from app.engines.external_knowledge.service import ExternalKnowledgeService, build_search_terms
from app.engines.external_knowledge.tfs_rest_client import TfsConnectorError, _build_wiql
from app.engines.external_knowledge.wiki_rest_client import WikiConnectorError, _build_cql

# --- extraction.py -----------------------------------------------------


def test_strip_html_removes_tags_and_decodes_common_entities():
    raw = "<div>Command &amp; Response<br/>failed &nbsp;here</div>"
    assert strip_html(raw) == "Command & Response failed here"


def test_strip_html_handles_none_and_empty():
    assert strip_html(None) == ""
    assert strip_html("") == ""


def test_truncate_word_boundary_with_ellipsis():
    text = "a" * 50 + " " + "b" * 50
    result = truncate(text, max_len=55)
    assert result.endswith("…")
    assert not result[:-1].endswith(" ")


def test_extract_and_truncate_combines_both():
    raw = "<p>" + ("word " * 300) + "</p>"
    result = extract_and_truncate(raw, max_len=100)
    assert "<p>" not in result
    assert len(result) <= 101


# --- ranking.py ----------------------------------------------------------


def test_score_candidate_rewards_component_technology_and_state():
    score, reasons = score_candidate(
        investigation_title="Meter stuck in Discovered state",
        candidate_title="Meter stuck in Discovered state after deployment",
        candidate_body="CommandProcessorHost failed to process init message",
        state="Resolved",
        matched_component_name="CommandProcessorHost",
        technology="RF Mesh",
        entity_values=[],
    )
    assert score > 0
    assert any("component" in r.lower() for r in reasons)
    assert any("resolved" in r.lower() for r in reasons)


def test_score_candidate_zero_when_nothing_matches():
    score, reasons = score_candidate(
        investigation_title="Unrelated topic entirely",
        candidate_title="Completely different subject",
        candidate_body="No overlap at all",
        state="New",
        matched_component_name="SomeOtherComponent",
        technology="Wi-Sun",
        entity_values=["SomeException"],
    )
    assert score == 0.0
    assert reasons == []


def test_score_candidate_clamped_to_one():
    score, _ = score_candidate(
        investigation_title="CommandProcessorHost RF Mesh error error error error",
        candidate_title="CommandProcessorHost RF Mesh error error error error",
        candidate_body="CommandProcessorHost RF Mesh",
        state="Closed",
        matched_component_name="CommandProcessorHost",
        technology="RF Mesh",
        entity_values=["error", "error2", "error3", "error4"],
    )
    assert score <= 1.0


def test_confidence_bands():
    assert confidence_for_score(0.9) == "High"
    assert confidence_for_score(0.65) == "High"
    assert confidence_for_score(0.5) == "Medium"
    assert confidence_for_score(0.1) == "Low"


# --- cache.py --------------------------------------------------------------


def test_cache_returns_none_when_missing():
    cache: TtlCache[str] = TtlCache(max_entries=5, ttl_seconds=60)
    assert cache.get("x") is None


def test_cache_set_and_get_roundtrip():
    cache: TtlCache[str] = TtlCache(max_entries=5, ttl_seconds=60)
    cache.set("x", "value")
    assert cache.get("x") == "value"


def test_cache_expires_after_ttl(monkeypatch):
    import app.engines.external_knowledge.cache as cache_module

    fake_time = [1000.0]
    monkeypatch.setattr(cache_module.time, "monotonic", lambda: fake_time[0])
    cache: TtlCache[str] = TtlCache(max_entries=5, ttl_seconds=10)
    cache.set("x", "value")
    fake_time[0] += 11
    assert cache.get("x") is None


def test_cache_evicts_oldest_beyond_max_entries():
    cache: TtlCache[str] = TtlCache(max_entries=2, ttl_seconds=60)
    cache.set("a", "1")
    cache.set("b", "2")
    cache.set("c", "3")
    assert cache.get("a") is None
    assert cache.get("b") == "2"
    assert cache.get("c") == "3"


# --- query builders ----------------------------------------------------------


def test_build_wiql_escapes_quotes_and_scopes_project():
    wiql = _build_wiql(team_project="Command Center", work_item_types=["Bug", "Issue"], anchor_terms=["O'Brien"], supporting_terms=[])
    assert "Command Center" in wiql
    assert "O''Brien" in wiql
    assert "'Bug'" in wiql and "'Issue'" in wiql


def test_build_wiql_empty_terms_falls_back_to_tautology():
    wiql = _build_wiql(team_project="Command Center", work_item_types=["Bug"], anchor_terms=[], supporting_terms=[])
    assert "1 = 1" in wiql


def test_build_wiql_anchor_scopes_query_supporting_terms_excluded():
    """Real bug this guards against: OR-ing every term (including
    generic title words) together let noise flood the recency-ordered
    candidate window and bury genuinely relevant matches (confirmed
    against the real TFS server)."""
    wiql = _build_wiql(
        team_project="Command Center",
        work_item_types=["Bug"],
        anchor_terms=["CommandProcessorHost"],
        supporting_terms=["Meter", "stuck", "state"],
    )
    assert "CommandProcessorHost" in wiql
    assert "Meter" not in wiql
    assert "stuck" not in wiql


def test_build_wiql_falls_back_to_supporting_terms_without_anchor():
    wiql = _build_wiql(team_project="Command Center", work_item_types=["Bug"], anchor_terms=[], supporting_terms=["Meter"])
    assert "Meter" in wiql


def test_build_cql_scopes_space_when_provided():
    cql = _build_cql(anchor_terms=["CommandProcessor"], supporting_terms=[], space_key="CC")
    assert 'space = "CC"' in cql
    assert "CommandProcessor" in cql


def test_build_cql_escapes_quotes():
    cql = _build_cql(anchor_terms=['say "hi"'], supporting_terms=[], space_key=None)
    assert '\\"hi\\"' in cql


def test_build_cql_anchor_excludes_supporting_terms():
    cql = _build_cql(anchor_terms=["CommandProcessorHost"], supporting_terms=["Meter"], space_key=None)
    assert "CommandProcessorHost" in cql
    assert "Meter" not in cql


# --- service.py: build_search_terms -----------------------------------------


def _Investigation(title: str):
    from app.domain.investigation import InvestigationSession

    return InvestigationSession(title=title)


def test_build_search_terms_component_and_technology_are_anchors():
    inv = _Investigation("Meter stuck in Discovered state after deployment")
    component = MatchedComponent(component_id="c1", component_name="CommandProcessorHost", confidence=1.0, match_reason="x")
    terms = build_search_terms(inv, [], component, "RF Mesh")
    assert terms.anchor == ["RF Mesh", "CommandProcessorHost"]
    assert "RF Mesh" not in terms.supporting
    assert "CommandProcessorHost" not in terms.supporting


def test_build_search_terms_title_words_and_entities_are_supporting():
    inv = _Investigation("Meter stuck in Discovered state")
    terms = build_search_terms(inv, [], None, None)
    assert terms.anchor == []
    assert "Meter" in terms.supporting


def test_build_search_terms_deduplicates_case_insensitively():
    inv = _Investigation("Mesh Mesh Mesh issue")
    terms = build_search_terms(inv, [], None, "Mesh")
    assert terms.all.count("Mesh") == 1


def test_build_search_terms_only_includes_exception_type_entities():
    inv = _Investigation("short")
    entities = [
        ExtractedEntity(entity_type=EntityType.EXCEPTION_TYPE, value="NullPointerException"),
        ExtractedEntity(entity_type=EntityType.THREAD_ID, value="12345"),
    ]
    terms = build_search_terms(inv, entities, None, None)
    assert "NullPointerException" in terms.supporting
    assert "12345" not in terms.all


def test_build_search_terms_caps_total_count():
    inv = _Investigation("alpha bravo charlie delta echo foxtrot golf hotel india juliet")
    terms = build_search_terms(inv, [], None, None)
    assert len(terms.all) <= 8


def test_build_search_terms_includes_customer_product_version_as_anchors():
    inv = _Investigation("Meter stuck")
    inv.customer = "Salt River Project"
    inv.product = "Command Center"
    inv.version = "9.0.5"
    terms = build_search_terms(inv, [], None, None)
    assert "Salt River Project" in terms.anchor
    assert "Command Center" in terms.anchor
    assert "9.0.5" in terms.anchor


def test_build_search_terms_extracts_ticket_number_from_title_as_anchor():
    inv = _Investigation("CS0122697 Meter stuck in Discovered state")
    terms = build_search_terms(inv, [], None, None)
    assert "CS0122697" in terms.anchor


def test_build_search_terms_extracts_ticket_number_from_task_description():
    from app.domain.evidence import Evidence
    from app.domain.enums import EvidenceType

    inv = _Investigation("Meter stuck")
    inv.add_evidence(Evidence(investigation_id=inv.id, evidence_type=EvidenceType.MANUAL_NOTE, raw_content="See CSTASK0087353 for background."))
    terms = build_search_terms(inv, [], None, None)
    assert "CSTASK0087353" in terms.anchor


# --- service.py: ExternalKnowledgeService -----------------------------------


class FakeTfsConnector:
    def __init__(self, *, configured=True, cases=None, raise_error=False):
        self._configured = configured
        self._cases = cases or []
        self._raise_error = raise_error

    def is_configured(self):
        return self._configured

    def search_work_items(self, *, anchor_terms, supporting_terms, work_item_types, max_results):
        if self._raise_error:
            raise TfsConnectorError("simulated failure")
        return self._cases

    def get_work_item(self, tfs_id):
        return next((c for c in self._cases if c.tfs_id == tfs_id), None)

    def get_work_item_history(self, tfs_id):
        return None


class FakeWikiConnector:
    def __init__(self, *, configured=True, pages=None, raise_error=False):
        self._configured = configured
        self._pages = pages or []
        self._raise_error = raise_error

    def is_configured(self):
        return self._configured

    def search(self, *, anchor_terms, supporting_terms, max_results):
        if self._raise_error:
            raise WikiConnectorError("simulated failure")
        return self._pages

    def get_page(self, page_id):
        return next((p for p in self._pages if p.page_id == page_id), None)


def _make_case(tfs_id=1, title="Meter stuck in Discovered state", state="Resolved", resolution="Restart CommandProcessorHost"):
    return TfsCase(
        tfs_id=tfs_id,
        work_item_type="Bug",
        title=title,
        state=state,
        area_path="Command Center",
        team_project="Command Center",
        changed_date=datetime(2026, 1, 1),
        description_text="CommandProcessorHost failed to process init message",
        resolution_text=resolution,
        url=f"https://am.tfs.landisgyr.net/tfs/DefaultCollection/Command%20Center/_workitems/edit/{tfs_id}",
    )


def _make_page(page_id="1", title="Troubleshooting CommandProcessorHost"):
    return WikiPage(
        page_id=page_id,
        title=title,
        space_key="CC",
        excerpt="Check the CommandProcessorHost log for init errors",
        url="https://wiki.landisgyr.net/pages/1",
    )


def _investigation(title="Meter stuck in Discovered state after deployment"):
    from app.domain.investigation import InvestigationSession

    return InvestigationSession(title=title)


def test_gather_returns_available_true_with_ranked_matches():
    tfs = FakeTfsConnector(cases=[_make_case()])
    wiki = FakeWikiConnector(pages=[_make_page()])
    service = ExternalKnowledgeService(tfs_connector=tfs, wiki_connector=wiki, settings=Settings())
    component = MatchedComponent(component_id="c1", component_name="CommandProcessorHost", confidence=1.0, match_reason="x")

    tfs_result, wiki_result = service.gather(_investigation(), [], component, None)

    assert tfs_result.available is True
    assert len(tfs_result.matches) == 1
    assert tfs_result.matches[0].tfs_case.tfs_id == 1
    assert wiki_result.available is True
    assert len(wiki_result.matches) == 1


def test_gather_degrades_gracefully_on_connector_error():
    tfs = FakeTfsConnector(raise_error=True)
    wiki = FakeWikiConnector(raise_error=True)
    service = ExternalKnowledgeService(tfs_connector=tfs, wiki_connector=wiki, settings=Settings())

    tfs_result, wiki_result = service.gather(_investigation(), [], None, None)

    assert tfs_result.available is False
    assert "could not be reached" in tfs_result.error
    assert wiki_result.available is False


def test_gather_reports_not_configured_distinctly():
    tfs = FakeTfsConnector(configured=False)
    wiki = FakeWikiConnector(configured=False)
    service = ExternalKnowledgeService(tfs_connector=tfs, wiki_connector=wiki, settings=Settings())

    tfs_result, wiki_result = service.gather(_investigation(), [], None, None)

    assert tfs_result.available is False
    assert "not configured" in tfs_result.error
    assert wiki_result.available is False


def test_gather_respects_disabled_kill_switch():
    tfs = FakeTfsConnector(cases=[_make_case()])
    wiki = FakeWikiConnector(pages=[_make_page()])
    settings = Settings(external_knowledge_enabled=False)
    service = ExternalKnowledgeService(tfs_connector=tfs, wiki_connector=wiki, settings=settings)

    tfs_result, wiki_result = service.gather(_investigation(), [], None, None)

    assert tfs_result.available is False
    assert wiki_result.available is False


def test_gather_drops_low_scoring_matches():
    unrelated_case = _make_case(title="Completely unrelated billing extract issue")
    tfs = FakeTfsConnector(cases=[unrelated_case])
    wiki = FakeWikiConnector(pages=[])
    service = ExternalKnowledgeService(tfs_connector=tfs, wiki_connector=wiki, settings=Settings())

    tfs_result, _ = service.gather(_investigation("Meter stuck Discovered"), [], None, None)

    assert tfs_result.matches == []


def test_gather_caches_repeat_queries():
    call_count = {"n": 0}

    class CountingTfsConnector(FakeTfsConnector):
        def search_work_items(self, *, anchor_terms, supporting_terms, work_item_types, max_results):
            call_count["n"] += 1
            return super().search_work_items(
                anchor_terms=anchor_terms, supporting_terms=supporting_terms, work_item_types=work_item_types, max_results=max_results
            )

    tfs = CountingTfsConnector(cases=[_make_case()])
    wiki = FakeWikiConnector(pages=[])
    service = ExternalKnowledgeService(tfs_connector=tfs, wiki_connector=wiki, settings=Settings())

    investigation = _investigation()
    service.gather(investigation, [], None, None)
    result2, _ = service.gather(investigation, [], None, None)

    assert call_count["n"] == 1
    assert result2.from_cache is True


def test_tfs_recommended_action_quotes_resolution_never_invents():
    tfs = FakeTfsConnector(cases=[_make_case(resolution="Restart CommandProcessorHost and verify init messages")])
    wiki = FakeWikiConnector(pages=[])
    service = ExternalKnowledgeService(tfs_connector=tfs, wiki_connector=wiki, settings=Settings())

    tfs_result, _ = service.gather(_investigation(), [], None, None)

    assert "Restart CommandProcessorHost and verify init messages" in tfs_result.matches[0].recommended_action


def test_tfs_recommended_action_none_when_no_resolution_text():
    case = _make_case(resolution=None)
    tfs = FakeTfsConnector(cases=[case])
    wiki = FakeWikiConnector(pages=[])
    service = ExternalKnowledgeService(tfs_connector=tfs, wiki_connector=wiki, settings=Settings())

    tfs_result, _ = service.gather(_investigation(), [], None, None)

    if tfs_result.matches:
        assert tfs_result.matches[0].recommended_action is None
