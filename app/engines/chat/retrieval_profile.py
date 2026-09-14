"""Intent-aware evidence selection, provenance classification, claim
construction, sufficiency assessment, and bounded contradiction
detection (Evidence-Centered Knowledge Retrieval & Synthesis phase).

This is deliberately NOT a new retrieval call. ``build_evidence_bundle``
below operates entirely on the ``InvestigationStrategy`` the existing,
untouched ``RecommendationEngine.generate()`` already produced (its
``documentation``/``historical_investigations``/``known_bugs``/
``tfs_matches``/``wiki_matches`` pools) plus already-parsed log
evidence -- it never queries Chroma/SQLite/TFS/Wiki a second time and
never changes what those systems return.

What IS new, and genuinely goes beyond "composer tie-breaking" (the
prior phase's ``_KIND_PRIORITY`` dicts inline in ``orchestrator.py``):

  * ``RETRIEVAL_PROFILES`` -- a real, named, per-``AnswerIntent``
    source-type priority table, used to decide which EVIDENCE CATEGORY
    should lead the answer when overlap ties, not just a two-entry
    dict local to one composer.
  * ``_authority_for_source`` -- a closed, source-type + intent ->
    ``SourceAuthority`` mapping (Step 3), so every piece of evidence
    carries an explicit, auditable claim about what KIND of fact it
    can support, before any text is rendered.
  * Claim construction (Step 6) -- a real intermediate layer between
    "here is a retrieved record" and "here is what to tell the user."
  * ``assess_sufficiency`` (Step 7) -- a six-level scale that
    explicitly refuses to equate document count or lexical similarity
    with authority.
  * ``detect_contradictions`` (Step 8) -- narrow, practical checks for
    version-token and "key set to value" conflicts between two
    AUTHORITATIVE_* sources, never silently reconciled.

Ranking discipline (unchanged from the prior phase, reused verbatim,
never re-derived): within one source-type pool, real lexical overlap
with the question's own subject words (``extract_concept_words``/
``lexical_overlap``) decides first, raw semantic score second. The
``RETRIEVAL_PROFILES`` table only breaks a genuine TIE in overlap
*across* source types when picking which type leads -- exactly the
role the old inline ``_KIND_PRIORITY`` dicts played, generalized to
every intent and every source type instead of two ad-hoc two-entry
dicts. The two pre-existing regression tests this rule exists to keep
passing (``test_process_setting_emerge_regression_...``,
``test_dashboard_cc_regression_...``) still pass unmodified against
this module -- see ``tests/test_retrieval_profile.py``.

Pipeline position (see ``app/domain/evidence_bundle.py``'s own
docstring): question -> intent/context (query_intent.py) -> THIS
MODULE -> EvidenceBundle -> orchestrator.py's composers -> optional LLM
wording -> grounding/completeness -> answer.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.domain.enums import LogLevel
from app.domain.evidence_bundle import (
    Claim,
    Contradiction,
    EvidenceBundle,
    EvidenceItem,
    SourceAuthority,
    SufficiencyLevel,
)
from app.engines.chat.knowledge_question import extract_concept_words, lexical_overlap
from app.engines.chat.query_intent import AnswerIntent, QueryContext
from app.engines.external_knowledge.extraction import truncate as truncate_extract

if TYPE_CHECKING:
    from app.domain.evidence import Evidence
    from app.domain.recommendation import InvestigationStrategy, KnowledgeMatch

# ---------------------------------------------------------------------------
# Step 4: intent-aware retrieval profiles
# ---------------------------------------------------------------------------

_DEFINITION_PROFILE: tuple[str, ...] = ("documentation", "wiki", "historical", "known_bug", "tfs")
_CONFIGURATION_PROFILE: tuple[str, ...] = ("documentation", "wiki", "tfs", "historical", "known_bug")
_TROUBLESHOOTING_PROFILE: tuple[str, ...] = ("known_bug", "documentation", "historical", "tfs", "wiki")
_ROOT_CAUSE_PROFILE: tuple[str, ...] = ("known_bug", "historical", "documentation", "tfs", "wiki")
_HISTORICAL_LOOKUP_PROFILE: tuple[str, ...] = ("historical", "tfs", "known_bug", "documentation", "wiki")
_LOG_ANALYSIS_PROFILE: tuple[str, ...] = ("log", "documentation", "known_bug", "historical")
_L2_PROFILE: tuple[str, ...] = ("log", "known_bug", "documentation", "historical")
_L3_PROFILE: tuple[str, ...] = ("log", "known_bug", "historical", "documentation", "tfs")
_COMPARISON_PROFILE: tuple[str, ...] = ("documentation", "historical", "known_bug", "tfs", "wiki")
_DEFAULT_PROFILE: tuple[str, ...] = ("documentation", "historical", "known_bug", "tfs", "wiki")

RETRIEVAL_PROFILES: dict[AnswerIntent, tuple[str, ...]] = {
    AnswerIntent.ENTITY_DEFINITION: _DEFINITION_PROFILE,
    AnswerIntent.PRODUCT_EXPLANATION: _DEFINITION_PROFILE,
    AnswerIntent.CONCEPT_EXPLANATION: _DEFINITION_PROFILE,
    AnswerIntent.CONFIGURATION: _CONFIGURATION_PROFILE,
    AnswerIntent.HOW_TO: _CONFIGURATION_PROFILE,
    AnswerIntent.TROUBLESHOOTING: _TROUBLESHOOTING_PROFILE,
    AnswerIntent.ROOT_CAUSE: _ROOT_CAUSE_PROFILE,
    AnswerIntent.HISTORICAL_LOOKUP: _HISTORICAL_LOOKUP_PROFILE,
    AnswerIntent.LOG_ANALYSIS: _LOG_ANALYSIS_PROFILE,
    AnswerIntent.L2_TASK_NOTES: _L2_PROFILE,
    AnswerIntent.L3_ESCALATION: _L3_PROFILE,
    AnswerIntent.COMPARISON: _COMPARISON_PROFILE,
    AnswerIntent.FOLLOW_UP: _DEFAULT_PROFILE,
    AnswerIntent.UNKNOWN: _DEFAULT_PROFILE,
}
"""Per-intent source-type priority order, highest-priority first.
Deliberately a plain, inspectable table (not a formula) so any future
change to one intent's evidence priority is a one-line, reviewable
diff -- exactly this codebase's own standing convention for every
other closed-list classifier (``_CONFIGURATION_PHRASES``, ``_KNOWN_
STATES``, etc.)."""


def kind_priority(profile: tuple[str, ...]) -> dict[str, int]:
    """Turns a profile ordering into a tiebreak score -- first-listed
    source type gets the highest number. Used ONLY to break a genuine
    tie in real lexical overlap (see module docstring); never consulted
    before overlap."""
    n = len(profile)
    return {source_type: n - i for i, source_type in enumerate(profile)}


# ---------------------------------------------------------------------------
# Step 3: source authority / provenance
# ---------------------------------------------------------------------------

_DEFINITIONAL_INTENTS = (
    AnswerIntent.ENTITY_DEFINITION,
    AnswerIntent.PRODUCT_EXPLANATION,
    AnswerIntent.CONCEPT_EXPLANATION,
)
_CONFIGURATION_INTENTS = (AnswerIntent.CONFIGURATION, AnswerIntent.HOW_TO)
_TROUBLESHOOTING_INTENTS = (AnswerIntent.TROUBLESHOOTING, AnswerIntent.ROOT_CAUSE)


def _authority_for_source(source_type: str, intent: AnswerIntent, has_majority_overlap: bool = True) -> SourceAuthority:
    """The single, closed mapping assigning ``SourceAuthority`` --
    never re-derived elsewhere. Real content from Documentation/Wiki
    can be AUTHORITATIVE for a definitional or configuration question
    (it directly states what something is / how it is configured);
    for any other intent the same source is only DOCUMENTED_BEHAVIOR
    or DOCUMENTED_TROUBLESHOOTING -- still real, still citable, never
    upgraded to "authoritative" for a question it doesn't actually
    settle. Known Bugs and TFS work items both record real, tracked
    defects -- never a definition, never automatically the cause of
    THIS incident. Historical Investigations are always
    HISTORICAL_OBSERVATION here (their resolution/next_step becomes a
    separate HISTORICAL_RECOMMENDATION claim -- see ``_claims_for_
    item``); current log evidence is always CURRENT_OBSERVATION.

    ``has_majority_overlap`` (Real-Corpus Answer Quality & Final Chat
    Hardening phase) -- a real, live finding against the actual
    ResolveIQ corpus: "What is AxeI meter?" against the real corpus
    retrieved "SM Registration,Removal and Disposal for Japanese
    Meters" as a Documentation candidate, sharing only the single,
    generic word "meter" with the question's two concept words ("axei",
    "meter") -- genuinely unrelated to AxeI, yet the plain overlap>0
    filter (``build_evidence_bundle``'s own ranking gate) let it
    through, and this function then unconditionally labeled it
    AUTHORITATIVE_DEFINITION purely because it is Documentation and the
    question is definitional. The exact same "task"/69%-similarity
    failure shape this codebase has already fixed twice at the
    composer-ranking level, discovered here at the AUTHORITY-
    ASSIGNMENT level: a document sharing only the generic word, not the
    distinctive one, must never be presented as if it AUTHORITATIVELY
    defines/configures the subject. When ``has_majority_overlap`` is
    False (the caller found the match shares FEWER than half the
    question's real concept words), a would-be AUTHORITATIVE_DEFINITION/
    AUTHORITATIVE_CONFIGURATION is downgraded one tier, to
    DOCUMENTED_BEHAVIOR -- still real, still citable as related
    documentation, just never claimed to settle the question. Every
    other authority in this mapping is unaffected (never applies to
    Known Bug/TFS/Historical/current-log at all, and this parameter's
    own default is True so every existing caller that doesn't pass it
    keeps its prior behavior unchanged)."""
    if source_type in ("documentation", "wiki"):
        if intent in _DEFINITIONAL_INTENTS:
            return SourceAuthority.AUTHORITATIVE_DEFINITION if has_majority_overlap else SourceAuthority.DOCUMENTED_BEHAVIOR
        if intent in _CONFIGURATION_INTENTS:
            return SourceAuthority.AUTHORITATIVE_CONFIGURATION if has_majority_overlap else SourceAuthority.DOCUMENTED_BEHAVIOR
        if intent in _TROUBLESHOOTING_INTENTS:
            return SourceAuthority.DOCUMENTED_TROUBLESHOOTING
        return SourceAuthority.DOCUMENTED_BEHAVIOR
    if source_type in ("known_bug", "tfs"):
        return SourceAuthority.KNOWN_BUG
    if source_type == "historical":
        return SourceAuthority.HISTORICAL_OBSERVATION
    if source_type == "log":
        return SourceAuthority.CURRENT_OBSERVATION
    return SourceAuthority.INFERENCE


def has_majority_overlap(overlap: int, concept_word_count: int) -> bool:
    """True when ``overlap`` covers STRICTLY MORE THAN half of the
    question's real concept words, or when the question has no
    extractable concept words at all (nothing to compare against, so
    never penalized -- matches this module's existing "no concept
    words means never rejected" rule elsewhere). 1-of-1 and 2-of-2 and
    2-of-3 all count as majority; 1-of-2 and 1-of-3 do NOT -- the exact
    real corpus finding this fixes: "axei meter" (2 concept words)
    sharing only the generic "meter" (1-of-2) must never count as a
    majority, since that is exactly the unrelated-document case this
    function exists to reject. Integer comparison (``overlap * 2 >
    concept_word_count``), never floating-point division."""
    if concept_word_count <= 0:
        return True
    return overlap * 2 > concept_word_count


_ESTABLISHES_TEXT: dict[SourceAuthority, str] = {
    SourceAuthority.AUTHORITATIVE_DEFINITION: "defines what this is, per ResolveIQ's own documentation",
    SourceAuthority.AUTHORITATIVE_CONFIGURATION: "states a documented configuration/setting",
    SourceAuthority.DOCUMENTED_BEHAVIOR: "describes documented behavior related to this question",
    SourceAuthority.DOCUMENTED_TROUBLESHOOTING: "documents a troubleshooting procedure or check",
    SourceAuthority.KNOWN_BUG: "records a real, tracked defect",
    SourceAuthority.CURRENT_OBSERVATION: "is a fact read directly from this investigation's own current evidence",
    SourceAuthority.HISTORICAL_OBSERVATION: "shows this subject was previously observed/investigated",
    SourceAuthority.HISTORICAL_RECOMMENDATION: "records a resolution/next step a past case took",
    SourceAuthority.INFERENCE: "is a reasoned inference, not a directly sourced fact",
}

_LIMITATION_TEXT: dict[SourceAuthority, str | None] = {
    SourceAuthority.AUTHORITATIVE_DEFINITION: None,
    SourceAuthority.AUTHORITATIVE_CONFIGURATION: None,
    SourceAuthority.DOCUMENTED_BEHAVIOR: None,
    SourceAuthority.DOCUMENTED_TROUBLESHOOTING: None,
    SourceAuthority.KNOWN_BUG: "does not by itself establish that this defect is the cause of the current issue",
    SourceAuthority.CURRENT_OBSERVATION: None,
    SourceAuthority.HISTORICAL_OBSERVATION: (
        "is a past case, not a product/concept definition, and does not by itself confirm the current issue "
        "has the same cause"
    ),
    SourceAuthority.HISTORICAL_RECOMMENDATION: (
        "is what a PAST case did, not a confirmed current resolution -- it has not been independently verified "
        "for this situation"
    ),
    SourceAuthority.INFERENCE: "has not been independently confirmed",
}


# ---------------------------------------------------------------------------
# Ranking (reused, not re-derived, from the prior phase's composers)
# ---------------------------------------------------------------------------


def _rank_by_overlap(matches: list, concept_words: list[str]) -> list[tuple[object, int]]:
    """Real subject overlap first, raw score second -- the exact rule
    ``_compose_knowledge_synthesis``/``_compose_troubleshooting_
    synthesis`` already established (the "task", 69%-similarity
    regression). Reused here as the one ranking primitive every
    source-type pool below uses, so this module can never silently
    diverge from those composers' own relevance judgment."""
    return sorted(
        ((m, lexical_overlap(concept_words, f"{m.title} {m.snippet[:500]}")) for m in matches),
        key=lambda pair: (pair[1], pair[0].score),
        reverse=True,
    )


_EXCERPT_WINDOW = 500
"""Matches ``_rank_by_overlap``'s own ranking window (``m.snippet[:500]``)
exactly, deliberately -- an ``EvidenceItem.excerpt`` shorter than the
window actually used to rank it would let a composer's final "does
this really overlap" recheck disagree with the ranking that already
happened, for no reason. Composers still apply their own, shorter
DISPLAY truncation (``truncate_extract(item.excerpt, 240)`` etc.) at
render time, exactly as before this phase -- this is the shared
upstream cap both ranking and rendering draw from, not a raw/unbounded
snippet."""


def _knowledge_match_item(
    match: "KnowledgeMatch", source_type: str, intent: AnswerIntent, overlap: int, concept_word_count: int = 0
) -> EvidenceItem:
    authority = _authority_for_source(source_type, intent, has_majority_overlap(overlap, concept_word_count))
    excerpt = truncate_extract(match.snippet, _EXCERPT_WINDOW)
    limitation = _LIMITATION_TEXT[authority]
    return EvidenceItem(
        source_type=source_type,
        title=match.title,
        source_id=match.record_id,
        excerpt=excerpt,
        relevance_score=match.score,
        authority=authority,
        establishes=f'"{match.title}" {_ESTABLISHES_TEXT[authority]}.',
        limitation=f'"{match.title}" {limitation}.' if limitation else None,
    )


def _external_match_item(match, source_type: str, intent: AnswerIntent, concept_words: list[str] | None = None) -> EvidenceItem | None:
    """``ExternalMatch`` (TFS/Wiki) -> ``EvidenceItem``. Returns
    ``None`` when the match carries neither ``tfs_case`` nor
    ``wiki_page`` (should not happen per ``ExternalMatch``'s own
    invariant, but this module never assumes an invariant it did not
    itself enforce -- see ``EvidenceBundle``'s "no arbitrary raw DB
    objects" rule). ``concept_words`` (Real-Corpus Answer Quality
    phase) -- when given, a live Wiki page's authority gets the exact
    same majority-overlap downgrade ``_knowledge_match_item`` applies
    to Documentation (a Wiki page sharing only one generic word out of
    several is never AUTHORITATIVE_DEFINITION/CONFIGURATION either);
    TFS is unaffected since ``_authority_for_source`` never assigns
    either authority to it regardless."""
    concept_words = concept_words or []
    if source_type == "tfs" and match.tfs_case is not None:
        case = match.tfs_case
        authority = _authority_for_source(source_type, intent)
        limitation = _LIMITATION_TEXT[authority]
        text = case.resolution_text or case.description_text or ""
        return EvidenceItem(
            source_type="tfs",
            title=case.title,
            source_id=str(case.tfs_id),
            excerpt=truncate_extract(text, _EXCERPT_WINDOW),
            relevance_score=match.score,
            authority=authority,
            establishes=f'TFS {case.work_item_type} "{case.title}" (state: {case.state}) {_ESTABLISHES_TEXT[authority]}.',
            limitation=f'"{case.title}" {limitation}.' if limitation else None,
        )
    if source_type == "wiki" and match.wiki_page is not None:
        page = match.wiki_page
        overlap = lexical_overlap(concept_words, f"{page.title} {page.excerpt}")
        authority = _authority_for_source(source_type, intent, has_majority_overlap(overlap, len(concept_words)))
        limitation = _LIMITATION_TEXT[authority]
        return EvidenceItem(
            source_type="wiki",
            title=page.title,
            source_id=page.page_id,
            excerpt=truncate_extract(page.excerpt, _EXCERPT_WINDOW),
            relevance_score=match.score,
            authority=authority,
            establishes=f'Wiki page "{page.title}" {_ESTABLISHES_TEXT[authority]}.',
            limitation=f'"{page.title}" {limitation}.' if limitation else None,
            url=page.url,
        )
    return None


def _log_evidence_items(log_evidence: "list[Evidence]") -> list[EvidenceItem]:
    """A bounded, deterministic summary of THIS investigation's own
    current log evidence (Step 14) -- reuses the exact ``LogEvent``
    data ``LogIntelligenceEngine`` already parsed at upload time; never
    a new parser, never full raw-log content (only the first error-or-
    fatal message plus an event count, matching the existing
    ``_compose_log_analysis_answer``'s own "quote real events, cap the
    count" discipline). Always ``CURRENT_OBSERVATION`` authority --
    the one authority level reserved for this investigation's own
    current evidence."""
    items: list[EvidenceItem] = []
    for evidence in log_evidence:
        events = evidence.log_events
        if not events:
            continue
        error_events = [e for e in events if e.level in (LogLevel.ERROR, LogLevel.FATAL)]
        first_error = error_events[0] if error_events else None
        summary = f"{len(events)} parsed event(s)"
        if first_error is not None:
            summary += f'; first error/fatal event: "{(first_error.message or first_error.raw_line)[:200]}"'
        items.append(
            EvidenceItem(
                source_type="log",
                title=evidence.title,
                source_id=evidence.id if hasattr(evidence, "id") else None,
                excerpt=summary,
                relevance_score=1.0,
                authority=SourceAuthority.CURRENT_OBSERVATION,
                establishes=f'"{evidence.title}" {_ESTABLISHES_TEXT[SourceAuthority.CURRENT_OBSERVATION]}.',
                limitation=None,
            )
        )
    return items


# ---------------------------------------------------------------------------
# Step 6: claims
# ---------------------------------------------------------------------------


def _claims_for_item(item: EvidenceItem, intent: AnswerIntent, has_recorded_recommendation: bool = False) -> list[Claim]:
    """One ``EvidenceItem`` -> one or more ``Claim``s. A Historical
    Investigation item can support TWO distinct claims (Step 6's own
    worked examples): an OBSERVATION claim ("this was investigated")
    and, only when the underlying match actually recorded a
    resolution/next_step (``has_recorded_recommendation``, checked by
    the caller against the real ``KnowledgeMatch.metadata`` -- never
    guessed from the item alone), a separate HISTORICAL_RECOMMENDATION
    claim ("a past case took this action") -- never collapsed into one
    claim that reads like a current fix, and never added when the case
    recorded no resolution/next_step at all (the two protected
    orchestrator regression fixtures deliberately pass empty
    resolution/next_step to stay at POSSIBLE/UNKNOWN tier -- this must
    not spuriously add a recommendation claim for them). The anti-
    pattern this explicitly avoids: a historical recommendation must
    never be rendered as "Changing X is the solution" -- see the
    ``category`` field and ``authority`` on the recommendation claim,
    both of which the composers must check before wording anything as
    settled."""
    claims: list[Claim] = []
    if item.authority in (SourceAuthority.AUTHORITATIVE_DEFINITION,):
        category = "definition"
    elif item.authority == SourceAuthority.AUTHORITATIVE_CONFIGURATION:
        category = "configuration"
    elif item.authority == SourceAuthority.DOCUMENTED_TROUBLESHOOTING:
        category = "troubleshooting"
    elif item.authority == SourceAuthority.DOCUMENTED_BEHAVIOR:
        category = "behavior"
    elif item.authority == SourceAuthority.KNOWN_BUG:
        category = "root_cause" if intent == AnswerIntent.ROOT_CAUSE else "troubleshooting"
    elif item.authority == SourceAuthority.CURRENT_OBSERVATION:
        category = "observation"
    else:
        category = "observation"

    claims.append(
        Claim(
            text=item.establishes,
            supported_by=[item.title],
            authority=item.authority,
            category=category,
            is_current=item.authority == SourceAuthority.CURRENT_OBSERVATION,
        )
    )

    if item.source_type == "historical" and has_recorded_recommendation:
        # The historical-recommendation split: a second, separate,
        # explicitly-lower-authority claim, never merged into the
        # observation claim above.
        claims.append(
            Claim(
                text=(
                    f'A past case ("{item.title}") recorded taking an action -- this is a historical '
                    f"recommendation, not a confirmed current resolution."
                ),
                supported_by=[item.title],
                authority=SourceAuthority.HISTORICAL_RECOMMENDATION,
                category="recommendation",
                is_current=False,
            )
        )
    return claims


# ---------------------------------------------------------------------------
# Step 7: evidence sufficiency
# ---------------------------------------------------------------------------


def assess_sufficiency(
    bundle_items: list[EvidenceItem],
    *,
    intent: AnswerIntent,
    has_overlap: bool,
    contradictions: list[Contradiction],
) -> SufficiencyLevel:
    """Step 7's own explicit anti-patterns, enforced structurally
    rather than by convention: document COUNT never appears in this
    function at all (only ``len() == 0`` as the floor case); a bare
    semantic score never appears either -- ``has_overlap`` (real
    subject-word overlap, computed by the caller exactly like the
    composers already do) is the only relevance signal consulted, and
    a HISTORICAL_RECOMMENDATION/HISTORICAL_OBSERVATION item is never,
    by itself, enough to reach STRONG or AUTHORITATIVE regardless of
    its score."""
    if contradictions:
        return SufficiencyLevel.CONTRADICTORY
    if not bundle_items:
        return SufficiencyLevel.INSUFFICIENT
    if not has_overlap:
        return SufficiencyLevel.WEAK

    authorities = {item.authority for item in bundle_items}
    if intent in _DEFINITIONAL_INTENTS and SourceAuthority.AUTHORITATIVE_DEFINITION in authorities:
        return SufficiencyLevel.AUTHORITATIVE
    if intent in _CONFIGURATION_INTENTS and SourceAuthority.AUTHORITATIVE_CONFIGURATION in authorities:
        return SufficiencyLevel.AUTHORITATIVE
    if authorities & {
        SourceAuthority.KNOWN_BUG,
        SourceAuthority.CURRENT_OBSERVATION,
        SourceAuthority.AUTHORITATIVE_DEFINITION,
        SourceAuthority.AUTHORITATIVE_CONFIGURATION,
        SourceAuthority.DOCUMENTED_TROUBLESHOOTING,
        SourceAuthority.DOCUMENTED_BEHAVIOR,
    }:
        return SufficiencyLevel.STRONG
    if authorities & {SourceAuthority.HISTORICAL_OBSERVATION, SourceAuthority.HISTORICAL_RECOMMENDATION}:
        return SufficiencyLevel.MODERATE
    return SufficiencyLevel.WEAK


# ---------------------------------------------------------------------------
# Step 8: bounded contradiction detection
# ---------------------------------------------------------------------------

_VERSION_TOKEN_RE = re.compile(r"\bv?(\d+\.\d+(?:\.\d+)?)\b", re.IGNORECASE)
_KEY_VALUE_RE = re.compile(
    r"\b([a-zA-Z][a-zA-Z0-9_ ]{2,30}?)\s*(?:=|:|is set to|set to)\s*([\w.\-]+)\b", re.IGNORECASE
)
"""Deliberately narrow (Step 8's own "bounded, not unrestricted NLP"
instruction): only two concrete patterns are checked --
version-token conflicts and explicit "key = value"/"key set to value"
statements -- across pairs of AUTHORITATIVE_DEFINITION/AUTHORITATIVE_
CONFIGURATION documentation items only (the only sources allowed to
make a configuration/version claim at all per ``_authority_for_
source``). Never attempts general semantic contradiction detection."""


def detect_contradictions(authoritative_items: list[EvidenceItem]) -> list[Contradiction]:
    contradictions: list[Contradiction] = []
    for i, item_a in enumerate(authoritative_items):
        for item_b in authoritative_items[i + 1 :]:
            versions_a = set(_VERSION_TOKEN_RE.findall(item_a.excerpt))
            versions_b = set(_VERSION_TOKEN_RE.findall(item_b.excerpt))
            if versions_a and versions_b and versions_a.isdisjoint(versions_b):
                contradictions.append(
                    Contradiction(
                        description="Documented version information differs between two sources.",
                        source_a=item_a.title,
                        claim_a=f"Version(s) mentioned: {', '.join(sorted(versions_a))}",
                        source_b=item_b.title,
                        claim_b=f"Version(s) mentioned: {', '.join(sorted(versions_b))}",
                    )
                )
                continue  # one contradiction per pair is enough signal

            kv_a = {k.strip().lower(): v for k, v in _KEY_VALUE_RE.findall(item_a.excerpt)}
            kv_b = {k.strip().lower(): v for k, v in _KEY_VALUE_RE.findall(item_b.excerpt)}
            for key, value_a in kv_a.items():
                value_b = kv_b.get(key)
                if value_b is not None and value_b.lower() != value_a.lower():
                    contradictions.append(
                        Contradiction(
                            description=f'Documented value for "{key}" differs between two sources.',
                            source_a=item_a.title,
                            claim_a=f"{key} = {value_a}",
                            source_b=item_b.title,
                            claim_b=f"{key} = {value_b}",
                        )
                    )
    return contradictions


def rank_items(
    items: list[EvidenceItem], concept_words: list[str], profile: tuple[str, ...]
) -> list[tuple[EvidenceItem, int]]:
    """Rank already-classified ``EvidenceItem``s the SAME way ``_rank_
    by_overlap`` ranks raw ``KnowledgeMatch``es: real subject overlap
    first, this intent's own ``RETRIEVAL_PROFILES`` priority second
    (breaking a tie between source TYPES -- e.g. documentation vs.
    historical -- exactly as the old inline ``_KIND_PRIORITY`` dicts
    did, generalized to every source type), raw relevance score third.
    This is the function ``ChatOrchestrator``'s composers call to pick
    the PRIMARY evidence for an answer -- the bundle's own evidence
    lists genuinely drive the answer, not just debug metadata (§29)."""
    priority = kind_priority(profile)
    return sorted(
        (
            (item, lexical_overlap(concept_words, f"{item.title} {item.excerpt}"))
            for item in items
        ),
        key=lambda pair: (pair[1], priority.get(pair[0].source_type, 0), pair[0].relevance_score),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_MAX_REJECTED_TRACKED = 20
"""Diagnostic-only cap (Step 18/25's own "do not add unnecessary
processing/data" discipline) -- ``rejected_evidence`` is for
``ChatResponse.debug``, never for the end user; nothing needs an
unbounded list of every discarded candidate."""


def build_evidence_bundle(
    question: str,
    context: QueryContext,
    strategy: "InvestigationStrategy",
    *,
    log_evidence: "list[Evidence] | None" = None,
    min_score: float = 0.35,
    min_score_secondary: float = 0.6,
    next_actions: list[str] | None = None,
) -> EvidenceBundle:
    """The one function that assembles an ``EvidenceBundle`` from an
    already-computed ``InvestigationStrategy`` -- never a new
    retrieval call (module docstring). ``min_score``/``min_score_
    secondary`` default to the exact values ``ChatOrchestrator`` has
    used since the Knowledge Answering & Evidence Synthesis phase
    (``_KNOWLEDGE_SYNTHESIS_MIN_SCORE``/``_SECONDARY``) -- callers
    should pass those class constants explicitly rather than relying
    on the default staying in sync, but the default exists so this
    module is independently testable/usable without an orchestrator
    instance."""
    intent = context.intent
    profile = RETRIEVAL_PROFILES.get(intent, _DEFAULT_PROFILE)
    concept_words = extract_concept_words(question)
    if not concept_words and context.subject:
        concept_words = context.subject.split()

    rejected: list[str] = []

    def _track_rejected(title: str, reason: str) -> None:
        if len(rejected) < _MAX_REJECTED_TRACKED:
            rejected.append(f"{title} -- {reason}")

    doc_pool = [m for m in strategy.documentation if m.score >= min_score]
    for m in strategy.documentation:
        if m.score < min_score:
            _track_rejected(m.title, f"documentation score {m.score:.2f} below relevance bar {min_score}")
    hist_pool = [m for m in strategy.historical_investigations if m.score >= min_score]
    for m in strategy.historical_investigations:
        if m.score < min_score:
            _track_rejected(m.title, f"historical score {m.score:.2f} below relevance bar {min_score}")
    bug_pool = [m for m in strategy.known_bugs if m.score >= min_score_secondary]
    for m in strategy.known_bugs:
        if m.score < min_score_secondary:
            _track_rejected(m.title, f"known bug score {m.score:.2f} below relevance bar {min_score_secondary}")
    tfs_result = strategy.tfs_matches
    wiki_result = strategy.wiki_matches
    tfs_pool = [m for m in (tfs_result.matches if tfs_result is not None and tfs_result.available else []) if m.score >= min_score_secondary]
    wiki_pool = [m for m in (wiki_result.matches if wiki_result is not None and wiki_result.available else []) if m.score >= min_score_secondary]

    doc_ranked = _rank_by_overlap(doc_pool, concept_words)
    hist_ranked = _rank_by_overlap(hist_pool, concept_words)
    bug_ranked = _rank_by_overlap(bug_pool, concept_words)
    # Whether ANYTHING cleared the score bar at all, before the overlap
    # filter below removes zero-overlap candidates -- the distinction
    # between WEAK ("real evidence exists, but none of it shares this
    # question's subject") and INSUFFICIENT ("nothing cleared the
    # relevance bar in the first place") depends on this, and is lost
    # once the overlap filter has run.
    any_score_bar_match = bool(doc_pool or hist_pool or bug_pool or tfs_pool or wiki_pool)

    # A real word-overlap match is required whenever the question has
    # extractable subject words at all -- same "task" regression
    # discipline the prior phase's composers already enforce, applied
    # uniformly to every KnowledgeMatch-backed pool here.
    def _filter_overlap(ranked: list[tuple[object, int]]) -> list[tuple[object, int]]:
        if not concept_words:
            return ranked
        kept = [pair for pair in ranked if pair[1] > 0]
        for match, overlap in ranked:
            if overlap == 0:
                _track_rejected(match.title, "zero real subject-word overlap with the question")
        return kept

    doc_ranked = _filter_overlap(doc_ranked)
    hist_ranked = _filter_overlap(hist_ranked)
    bug_ranked = _filter_overlap(bug_ranked)

    word_count = len(concept_words)
    documentation_items = [_knowledge_match_item(m, "documentation", intent, ov, word_count) for m, ov in doc_ranked]
    historical_items = [_knowledge_match_item(m, "historical", intent, ov, word_count) for m, ov in hist_ranked]
    known_bug_items = [_knowledge_match_item(m, "known_bug", intent, ov, word_count) for m, ov in bug_ranked]
    # Which historical items actually recorded a resolution/next_step --
    # keyed by title (EvidenceItem carries no raw metadata by design;
    # see EvidenceItem's own "no arbitrary raw DB objects" rule), so
    # claim-construction below can tell a bare observation apart from a
    # genuine historical recommendation without guessing.
    hist_has_recommendation = {
        m.title: bool(m.metadata.get("resolution") or m.metadata.get("next_step")) for m, _ov in hist_ranked
    }
    tfs_items = [item for m in tfs_pool if (item := _external_match_item(m, "tfs", intent)) is not None]
    wiki_items = [item for m in wiki_pool if (item := _external_match_item(m, "wiki", intent, concept_words)) is not None]
    log_items = _log_evidence_items(log_evidence or [])

    # Documentation/Wiki items that are AUTHORITATIVE_DEFINITION/
    # AUTHORITATIVE_CONFIGURATION are surfaced separately (Step 2's
    # ``authoritative_documentation`` field) so a caller/composer can
    # tell "this genuinely defines/configures the subject" apart from
    # merely-related documentation without re-deriving the authority
    # mapping itself.
    authoritative_documentation = [
        item
        for item in documentation_items + wiki_items
        if item.authority in (SourceAuthority.AUTHORITATIVE_DEFINITION, SourceAuthority.AUTHORITATIVE_CONFIGURATION)
    ]

    has_overlap = bool(doc_ranked and doc_ranked[0][1] > 0) or bool(hist_ranked and hist_ranked[0][1] > 0) or bool(
        bug_ranked and bug_ranked[0][1] > 0
    ) or bool(log_items) or not concept_words

    # §8: checked over ALL real Documentation/Wiki content (not only
    # the subset this question's own intent happens to grant
    # AUTHORITATIVE_* status) -- both source types are real, authored
    # content capable of stating a version/config value regardless of
    # what THIS question is asking, so a genuine conflict between them
    # is worth surfacing even when neither is being cited as "the"
    # authoritative answer to the current question.
    contradictions = detect_contradictions(documentation_items + wiki_items)

    claims: list[Claim] = []
    for item in documentation_items:
        claims.extend(_claims_for_item(item, intent))
    for item in historical_items:
        claims.extend(_claims_for_item(item, intent, has_recorded_recommendation=hist_has_recommendation.get(item.title, False)))
    for item in known_bug_items + tfs_items + wiki_items + log_items:
        claims.extend(_claims_for_item(item, intent))

    all_items = documentation_items + historical_items + known_bug_items + tfs_items + wiki_items + log_items
    if not all_items:
        sufficiency = SufficiencyLevel.CONTRADICTORY if contradictions else (
            SufficiencyLevel.WEAK if any_score_bar_match else SufficiencyLevel.INSUFFICIENT
        )
    else:
        sufficiency = assess_sufficiency(all_items, intent=intent, has_overlap=has_overlap, contradictions=contradictions)

    return EvidenceBundle(
        question=question,
        intent=intent.value,
        subject=context.subject,
        product=context.product,
        technology=context.technology,
        version=context.version,
        component=context.component,
        customer=context.customer,
        region=context.region,
        state=context.state,
        authoritative_documentation=authoritative_documentation,
        documentation=[item for item in documentation_items if item not in authoritative_documentation],
        wiki_evidence=[item for item in wiki_items if item not in authoritative_documentation],
        tfs_evidence=tfs_items,
        known_bug_evidence=known_bug_items,
        historical_case_evidence=historical_items,
        current_log_evidence=log_items,
        claims=claims,
        contradictions=contradictions,
        next_actions=list(next_actions) if next_actions else [],
        sufficiency=sufficiency,
        rejected_evidence=rejected,
        retrieval_profile=list(profile),
    )
