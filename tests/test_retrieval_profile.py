"""Unit tests for app.engines.chat.retrieval_profile (Evidence-Centered
Knowledge Retrieval & Synthesis phase).

These are pure, direct tests of ``build_evidence_bundle`` and its
helpers -- constructed straight from ``KnowledgeMatch``/
``InvestigationStrategy``/``ExternalMatch`` fixtures, never going
through the full ``ChatOrchestrator``/DB/Chroma pipeline (that
end-to-end coverage already lives in tests/test_chat_orchestrator.py
and is asserted, unmodified, by the two regression tests this module's
docstring names)."""

from __future__ import annotations

from app.domain.evidence_bundle import SourceAuthority, SufficiencyLevel
from app.domain.external_knowledge import ExternalKnowledgeResult, ExternalMatch, ExternalSource, TfsCase, WikiPage
from app.domain.recommendation import InvestigationStage, InvestigationStrategy, KnowledgeMatch
from app.engines.chat.query_intent import AnswerIntent, build_query_context
from app.engines.chat.retrieval_profile import (
    RETRIEVAL_PROFILES,
    build_evidence_bundle,
    detect_contradictions,
    kind_priority,
)
from datetime import datetime, timezone


def _strategy(**overrides) -> InvestigationStrategy:
    defaults = dict(
        current_stage=InvestigationStage.TRIAGE,
        stage_rationale="r",
        progress=0.0,
        progress_summary="s",
        recommended_next_action="n",
        next_action_rationale="r",
    )
    defaults.update(overrides)
    return InvestigationStrategy(**defaults)


def _doc(title: str, snippet: str, score: float, record_id: str = "doc-1") -> KnowledgeMatch:
    from app.domain.enums import KnowledgeCollection

    return KnowledgeMatch(collection=KnowledgeCollection.DOCUMENTATION, record_id=record_id, title=title, snippet=snippet, score=score)


def _hist(title: str, snippet: str, score: float, record_id: str = "hi-1", **metadata) -> KnowledgeMatch:
    from app.domain.enums import KnowledgeCollection

    return KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS, record_id=record_id, title=title, snippet=snippet, score=score, metadata=metadata
    )


def _bug(title: str, snippet: str, score: float, record_id: str = "bug-1", **metadata) -> KnowledgeMatch:
    from app.domain.enums import KnowledgeCollection

    return KnowledgeMatch(collection=KnowledgeCollection.KNOWN_BUGS, record_id=record_id, title=title, snippet=snippet, score=score, metadata=metadata)


# --- A. RETRIEVAL_PROFILES / kind_priority -----------------------------------


def test_every_answer_intent_has_a_retrieval_profile():
    for intent in AnswerIntent:
        assert intent in RETRIEVAL_PROFILES, f"{intent} has no retrieval profile"


def test_definitional_profiles_rank_documentation_above_historical():
    for intent in (AnswerIntent.ENTITY_DEFINITION, AnswerIntent.PRODUCT_EXPLANATION, AnswerIntent.CONCEPT_EXPLANATION):
        priority = kind_priority(RETRIEVAL_PROFILES[intent])
        assert priority["documentation"] > priority["historical"]


def test_troubleshooting_profile_ranks_known_bug_above_documentation_above_historical():
    priority = kind_priority(RETRIEVAL_PROFILES[AnswerIntent.TROUBLESHOOTING])
    assert priority["known_bug"] > priority["documentation"] > priority["historical"]


def test_historical_lookup_profile_ranks_historical_first():
    profile = RETRIEVAL_PROFILES[AnswerIntent.HISTORICAL_LOOKUP]
    assert profile[0] == "historical"


# --- B. Source authority classification (§3) ---------------------------------


def test_documentation_is_authoritative_definition_for_entity_definition_question():
    context = build_query_context("What is AxeI meter?")
    strategy = _strategy(documentation=[_doc("AxeI Meter Overview", "AxeI meter is a Landis+Gyr RF mesh endpoint.", 0.8)])
    bundle = build_evidence_bundle("What is AxeI meter?", context, strategy)

    assert bundle.authoritative_documentation, "a real definitional doc must be classified authoritative"
    assert bundle.authoritative_documentation[0].authority == SourceAuthority.AUTHORITATIVE_DEFINITION
    assert bundle.has_authoritative_evidence()


def test_documentation_is_only_documented_behavior_for_a_historical_lookup_question():
    context = build_query_context("Has this happened before with AxeI meter?")
    strategy = _strategy(documentation=[_doc("AxeI Meter Overview", "AxeI meter is a Landis+Gyr RF mesh endpoint.", 0.8)])
    bundle = build_evidence_bundle("Has this happened before with AxeI meter?", context, strategy)

    assert not bundle.authoritative_documentation
    assert bundle.documentation
    assert bundle.documentation[0].authority == SourceAuthority.DOCUMENTED_BEHAVIOR


def test_historical_case_is_never_authoritative_regardless_of_score():
    context = build_query_context("What is AxeI meter?")
    strategy = _strategy(
        historical_investigations=[_hist("Empresa Electrica de Guatemala | Self Hosted | Focus AxeI meter discovered", "AxeI meter discovered.", 0.95)]
    )
    bundle = build_evidence_bundle("What is AxeI meter?", context, strategy)

    assert not bundle.authoritative_documentation
    assert bundle.historical_case_evidence
    assert bundle.historical_case_evidence[0].authority == SourceAuthority.HISTORICAL_OBSERVATION
    assert not bundle.has_authoritative_evidence()


def test_known_bug_and_tfs_are_never_authoritative():
    context = build_query_context("Why did this meter fail?")
    tfs_case = TfsCase(
        tfs_id=1, work_item_type="Bug", title="AxeI meter comm failure", state="Active",
        area_path="a", team_project="p", changed_date=datetime.now(timezone.utc), description_text="AxeI meter comm failure known issue.",
        url="http://tfs.example/1",
    )
    strategy = _strategy(
        known_bugs=[_bug("AxeI meter comm failure", "Known collector timeout with AxeI meter.", 0.8)],
        tfs_matches=ExternalKnowledgeResult(
            source=ExternalSource.TFS, available=True,
            matches=[ExternalMatch(source=ExternalSource.TFS, tfs_case=tfs_case, score=0.7, confidence="High")],
        ),
    )
    bundle = build_evidence_bundle("Why did this meter fail?", context, strategy)

    assert all(item.authority == SourceAuthority.KNOWN_BUG for item in bundle.known_bug_evidence + bundle.tfs_evidence)
    assert not bundle.has_authoritative_evidence()


# --- C. Claims (§6) -----------------------------------------------------------


def test_historical_case_with_no_recorded_resolution_produces_only_an_observation_claim():
    context = build_query_context("What is AxeI meter?")
    strategy = _strategy(historical_investigations=[_hist("AxeI meter discovered case", "AxeI meter discovered.", 0.9, root_cause="", resolution="", next_step="")])
    bundle = build_evidence_bundle("What is AxeI meter?", context, strategy)

    categories = [c.category for c in bundle.claims]
    assert "observation" in categories
    assert "recommendation" not in categories


def test_historical_case_with_a_recorded_resolution_produces_a_separate_recommendation_claim_never_a_current_fact():
    context = build_query_context("What is AxeI meter?")
    strategy = _strategy(
        historical_investigations=[
            _hist("AxeI meter discovered case", "AxeI meter discovered.", 0.9, resolution="Reset the collector queue.", next_step="")
        ]
    )
    bundle = build_evidence_bundle("What is AxeI meter?", context, strategy)

    recommendation_claims = [c for c in bundle.claims if c.category == "recommendation"]
    assert len(recommendation_claims) == 1
    rec = recommendation_claims[0]
    assert rec.authority == SourceAuthority.HISTORICAL_RECOMMENDATION
    assert rec.is_current is False
    # The explicit anti-pattern this phase forbids:
    assert "is the solution" not in rec.text.lower()
    assert "historical recommendation" in rec.text.lower() or "past case" in rec.text.lower()


def test_current_log_evidence_claim_is_marked_current():
    from app.domain.evidence import Evidence
    from app.domain.enums import EvidenceType, LogLevel
    from app.domain.entities import LogEvent

    evidence = Evidence(
        investigation_id="i1", evidence_type=EvidenceType.LOG_FILE, title="collector.log",
        log_events=[LogEvent(timestamp=datetime.now(timezone.utc), level=LogLevel.ERROR, message="Connection reset", raw_line="Connection reset")],
    )
    context = build_query_context("Why did it fail?")
    strategy = _strategy()
    bundle = build_evidence_bundle("Why did it fail?", context, strategy, log_evidence=[evidence])

    assert bundle.current_log_evidence
    assert bundle.current_log_evidence[0].authority == SourceAuthority.CURRENT_OBSERVATION
    current_claims = [c for c in bundle.claims if c.is_current]
    assert current_claims


# --- D. Sufficiency (§7) -------------------------------------------------------


def test_sufficiency_is_insufficient_when_nothing_retrieved():
    context = build_query_context("What is AxeI meter?")
    bundle = build_evidence_bundle("What is AxeI meter?", context, _strategy())
    assert bundle.sufficiency == SufficiencyLevel.INSUFFICIENT


def test_sufficiency_is_authoritative_for_a_real_definitional_document_match():
    context = build_query_context("What is AxeI meter?")
    strategy = _strategy(documentation=[_doc("AxeI Meter Overview", "AxeI meter is a Landis+Gyr RF mesh endpoint.", 0.8)])
    bundle = build_evidence_bundle("What is AxeI meter?", context, strategy)
    assert bundle.sufficiency == SufficiencyLevel.AUTHORITATIVE


def test_sufficiency_is_moderate_not_strong_for_historical_only_evidence():
    context = build_query_context("What is AxeI meter?")
    strategy = _strategy(historical_investigations=[_hist("AxeI meter discovered case", "AxeI meter discovered.", 0.9)])
    bundle = build_evidence_bundle("What is AxeI meter?", context, strategy)
    # Anti-pattern (§7): a single, even highly-scored, historical match
    # must never be treated as STRONG/AUTHORITATIVE.
    assert bundle.sufficiency == SufficiencyLevel.MODERATE


def test_sufficiency_is_weak_when_nothing_shares_the_questions_subject():
    context = build_query_context("what is process setting in emerge")
    strategy = _strategy(documentation=[_doc("task", "Task List. Assigned to = Arun Bhukker AND Active = false.", 0.693)])
    bundle = build_evidence_bundle("what is process setting in emerge", context, strategy)
    assert bundle.sufficiency == SufficiencyLevel.WEAK


def test_many_weakly_relevant_documents_do_not_upgrade_sufficiency_past_moderate():
    """Anti-pattern (§7): document COUNT is never conflated with
    authority -- ten historical matches must not outrank the single-
    document AUTHORITATIVE case above, nor jump past MODERATE."""
    context = build_query_context("What is AxeI meter?")
    strategy = _strategy(historical_investigations=[_hist(f"Case {i}", "AxeI meter discovered.", 0.9, record_id=f"hi-{i}") for i in range(10)])
    bundle = build_evidence_bundle("What is AxeI meter?", context, strategy)
    assert bundle.sufficiency == SufficiencyLevel.MODERATE


# --- E. Contradictions (§8) ----------------------------------------------------


def test_detect_contradictions_flags_differing_version_tokens():
    context = build_query_context("What version is supported?")
    strategy = _strategy(
        documentation=[
            _doc("Install Guide A", "This feature requires version 8.4 or later.", 0.8, record_id="d1"),
            _doc("Install Guide B", "This feature requires version 9.1 or later.", 0.8, record_id="d2"),
        ]
    )
    bundle = build_evidence_bundle("What version is supported?", context, strategy)
    assert bundle.contradictions
    assert bundle.sufficiency == SufficiencyLevel.CONTRADICTORY


def test_detect_contradictions_flags_differing_key_value_statements():
    context = build_query_context("How do I configure the timeout?")
    strategy = _strategy(
        documentation=[
            _doc("Config Guide A", "timeout is set to 30 seconds by default.", 0.8, record_id="d1"),
            _doc("Config Guide B", "timeout is set to 60 seconds by default.", 0.8, record_id="d2"),
        ]
    )
    bundle = build_evidence_bundle("How do I configure the timeout?", context, strategy)
    assert bundle.contradictions
    assert any("timeout" in c.description.lower() for c in bundle.contradictions)


def test_no_contradiction_flagged_for_agreeing_documentation():
    context = build_query_context("What version is supported?")
    strategy = _strategy(
        documentation=[
            _doc("Install Guide A", "This feature requires version 8.4 or later.", 0.8, record_id="d1"),
            _doc("Install Guide B", "This feature requires version 8.4 or later, per release notes.", 0.8, record_id="d2"),
        ]
    )
    bundle = build_evidence_bundle("What version is supported?", context, strategy)
    assert not bundle.contradictions


def test_contradiction_never_checked_between_historical_cases_only():
    """§8 scope: contradiction detection is bounded to AUTHORITATIVE_*
    documentation sources -- two historical cases disagreeing is not
    (yet) flagged, by design (never unrestricted NLP)."""
    context = build_query_context("Has this happened before?")
    strategy = _strategy(
        historical_investigations=[
            _hist("Case A", "Fixed by setting timeout to 30 seconds.", 0.8, record_id="h1"),
            _hist("Case B", "Fixed by setting timeout to 60 seconds.", 0.8, record_id="h2"),
        ]
    )
    bundle = build_evidence_bundle("Has this happened before?", context, strategy)
    assert not bundle.contradictions


# --- F. rejected_evidence / retrieval_profile debug surface --------------------


def test_rejected_evidence_records_why_a_below_bar_match_was_excluded():
    context = build_query_context("What is AxeI meter?")
    strategy = _strategy(documentation=[_doc("Unrelated low-score doc", "Something else entirely.", 0.1)])
    bundle = build_evidence_bundle("What is AxeI meter?", context, strategy)
    assert bundle.all_evidence() == []
    assert any("below relevance bar" in reason for reason in bundle.rejected_evidence)


def test_retrieval_profile_recorded_on_bundle_matches_the_table():
    context = build_query_context("What is AxeI meter?")
    bundle = build_evidence_bundle("What is AxeI meter?", context, _strategy())
    assert bundle.retrieval_profile == list(RETRIEVAL_PROFILES[AnswerIntent.ENTITY_DEFINITION])


# --- Final Support-Quality Pass, §5 -- claim traceability -------------------


def test_claims_get_stable_sequential_ids():
    context = build_query_context("What is AxeI meter?")
    strategy = _strategy(
        historical_investigations=[_hist("AxeI meter discovered case", "AxeI meter discovered.", 0.9, resolution="Reset the queue.")]
    )
    bundle = build_evidence_bundle("What is AxeI meter?", context, strategy)
    assert len(bundle.claims) >= 2
    ids = [c.claim_id for c in bundle.claims]
    assert ids == [f"claim-{i+1:03d}" for i in range(len(ids))]
    assert len(set(ids)) == len(ids)  # all unique


def test_claim_carries_source_ids_and_confidence():
    context = build_query_context("What is AxeI meter?")
    strategy = _strategy(documentation=[_doc("AxeI Meter Overview", "AxeI meter is an RF mesh endpoint.", 0.8, record_id="doc-42")])
    bundle = build_evidence_bundle("What is AxeI meter?", context, strategy)
    assert bundle.claims
    claim = bundle.claims[0]
    assert claim.source_ids == ["doc-42"]
    assert claim.confidence == 0.8
    assert claim.supported is True
