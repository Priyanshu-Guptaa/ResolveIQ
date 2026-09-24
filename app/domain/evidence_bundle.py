"""The formal evidence model for chat answer synthesis (Evidence-
Centered Knowledge Retrieval & Synthesis phase).

Where this sits in the pipeline (see ``app.engines.chat.
retrieval_profile.build_evidence_bundle``, the one function that
constructs this from an already-computed ``InvestigationStrategy`` --
never a new retrieval call):

    question -> intent/context (query_intent.py)
             -> intent-aware evidence selection (retrieval_profile.py)
             -> EvidenceBundle (this module)
             -> claims + sufficiency + contradictions (also this module,
                populated by retrieval_profile.py)
             -> deterministic answer synthesis (orchestrator.py's
                composers, now driven by the bundle instead of raw
                KnowledgeMatch/ExternalMatch pools)
             -> optional LLM wording -> grounding/completeness -> answer

Every field here is either copied from data ``RecommendationEngine.
generate()``/``LogIntelligenceEngine`` already computed, or a plain,
deterministic classification of that data (an authority label, a
sufficiency level) -- nothing here performs a new retrieval, and
nothing here is LLM-derived. Pydantic models, matching this codebase's
own existing convention for computed-fresh, never-persisted shapes
(``StructuredResolution``, ``ProvenanceRecord``, ``LogObservationSummary``)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SourceAuthority(str, Enum):
    """What KIND of claim a piece of evidence can support -- the
    central rule this whole phase exists to enforce: a historical case
    can prove something HAPPENED; it can never, by construction, carry
    ``AUTHORITATIVE_DEFINITION`` authority, no matter how highly it
    scores or how closely its title matches the question. Never
    silently upgraded (a HISTORICAL_RECOMMENDATION is never treated as
    a CURRENT confirmed resolution; an INFERENCE is never treated as a
    directly-sourced fact) -- see ``retrieval_profile.
    _authority_for_source``, the single function that assigns this,
    for the exact, closed mapping."""

    AUTHORITATIVE_DEFINITION = "authoritative_definition"
    """Real product/documentation content that defines what a product,
    component, or concept IS. Only ever assigned to a Documentation or
    live-Wiki match -- never Historical Investigations/Known Bugs,
    which record incidents and defects, not definitions."""
    AUTHORITATIVE_CONFIGURATION = "authoritative_configuration"
    """Real documentation that states a supported configuration/
    setting explicitly. Same source restriction as above."""
    DOCUMENTED_BEHAVIOR = "documented_behavior"
    """Real documentation describing how something behaves/works,
    short of a bare definition or a configuration value."""
    DOCUMENTED_TROUBLESHOOTING = "documented_troubleshooting"
    """Real documentation/Wiki content that documents a
    troubleshooting procedure or check."""
    KNOWN_BUG = "known_bug"
    """A real, recorded defect. Establishes that a known defect
    exists; never automatically establishes that it is the cause of
    the CURRENT incident -- see ``Claim.category``."""
    CURRENT_OBSERVATION = "current_observation"
    """A fact read directly off the current investigation's own
    uploaded log/evidence -- the only authority level for anything
    happening in the log(s) attached to THIS conversation."""
    HISTORICAL_OBSERVATION = "historical_observation"
    """What a real, PAST historical case recorded as having happened.
    Never promoted to a current fact or a definition."""
    HISTORICAL_RECOMMENDATION = "historical_recommendation"
    """A resolution/next-step a real PAST case recorded taking. Never
    promoted to "the current confirmed resolution" -- see
    ``Claim.category`` == "recommendation", always rendered with an
    explicit historical-not-current caveat."""
    INFERENCE = "inference"
    """A reasoned conclusion this system itself derived from evidence
    (e.g. "this MAY indicate recovery"), never presented as a directly
    sourced fact."""


class SufficiencyLevel(str, Enum):
    """How well the assembled evidence actually supports answering the
    question -- computed by ``retrieval_profile.assess_sufficiency``,
    never guessed and never equated with retrieval volume (Step 7's
    own explicit rule: many documents, or one highly-lexically-similar
    document, is not automatically strong evidence)."""

    AUTHORITATIVE = "authoritative"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    INSUFFICIENT = "insufficient"
    CONTRADICTORY = "contradictory"


class EvidenceItem(BaseModel):
    """One real, already-retrieved piece of evidence, classified. Never
    an arbitrary raw database row -- only the fields actually needed to
    cite and reason about this item are kept, and ``excerpt`` is always
    already-truncated (this module never grows a snippet beyond what
    the existing composers already capped it to)."""

    source_type: str
    """"documentation" | "wiki" | "tfs" | "known_bug" | "historical" |
    "log" -- which of ``EvidenceBundle``'s own list fields this
    belongs to; kept as a plain string (matching ``StructuredResolution.
    source_kind``'s own established "plain string, not a cross-layer
    enum" convention) since some of these (tfs/wiki) aren't
    ``KnowledgeCollection`` members at all."""
    title: str
    source_id: str | None = None
    excerpt: str
    relevance_score: float
    authority: SourceAuthority
    establishes: str
    """Plain-language statement of what this source actually
    establishes -- never blank; every EvidenceItem must be able to
    answer "why is this here"."""
    limitation: str | None = None
    """What this source does NOT establish, stated explicitly whenever
    that gap matters (e.g. a historical case's own limitation is always
    "this does not establish a current confirmed resolution")."""
    url: str | None = None


class Claim(BaseModel):
    """One candidate statement the assembled evidence can support --
    the layer between "here is a retrieved document" and "here is what
    I am telling the user," per Step 6's own explicit requirement.
    Never rendered to the user as more certain than ``authority``
    allows (a HISTORICAL_RECOMMENDATION-authority claim is never
    phrased as "the fix is..."; see the composers in
    ``ChatOrchestrator`` for the actual wording rules this enforces)."""

    claim_id: str = ""
    """Final Support-Quality Pass, §5 -- a stable, per-response
    identifier ("claim-001", "claim-002", ...), assigned once by
    ``retrieval_profile.build_evidence_bundle`` in the order claims are
    constructed -- never reused across a different response, never
    exposed to normal users by default (see ``ChatResponse.debug``'s
    own docstring), but present so a specific final-answer statement
    can be traced back to exactly which claim, which ``EvidenceItem``,
    and which real source produced it. Defaults to ``""`` only for a
    ``Claim`` constructed directly (e.g. in a unit test) before that
    numbering pass runs."""
    text: str
    supported_by: list[str] = Field(default_factory=list)
    """Titles of the ``EvidenceItem``(s) that support this claim."""
    source_ids: list[str] = Field(default_factory=list)
    """The same ``EvidenceItem``(s)' own ``source_id`` values, parallel
    to ``supported_by`` -- title is for display, ``source_ids`` is for
    a debugging tool or a future validator to look up the exact
    underlying record without a title-string match."""
    authority: SourceAuthority
    category: str
    """"definition" | "configuration" | "behavior" | "troubleshooting"
    | "root_cause" | "recommendation" | "observation" -- what KIND of
    claim this is, independent of confidence."""
    confidence: float | None = None
    """The supporting ``EvidenceItem``'s own ``relevance_score``, when
    exactly one item supports this claim -- carried through for
    traceability, never a new, independently-computed confidence
    value. ``None`` when no single score applies."""
    supported: bool = True
    """True for every claim this module actually constructs (a Claim
    is only ever built FROM a real, already-selected ``EvidenceItem``
    -- there is no code path that builds an unsupported one today).
    Kept as an explicit field, not merely implied, so a future
    validator can check it without assuming the invariant holds
    forever."""
    is_current: bool = False
    """True only for a claim about THIS investigation's own current
    evidence (CURRENT_OBSERVATION authority); False for anything
    historical/documented/inferred."""


class Contradiction(BaseModel):
    """A real, structured disagreement between two sources -- never
    silently reconciled (Step 8's own explicit rule). Deliberately
    narrow/bounded (configuration values, supported versions) rather
    than an attempt at unrestricted contradiction NLP -- see
    ``retrieval_profile.detect_contradictions`` for exactly what is,
    and is not, checked."""

    description: str
    source_a: str
    claim_a: str
    source_b: str
    claim_b: str
    resolved_by_context: str | None = None
    """Set when the apparent conflict is explained by differing
    context (e.g. two different documented versions) -- never invented,
    only set when both sources themselves name the differing context."""


class EvidenceBundle(BaseModel):
    """The complete, structured evidence picture for one turn -- see
    module docstring for the pipeline this sits in. Constructed fresh
    every turn by ``retrieval_profile.build_evidence_bundle``, never
    persisted, never a second source of truth for anything
    ``InvestigationStrategy``/``LogObservationSummary`` already is."""

    question: str
    intent: str
    subject: str | None = None
    product: str | None = None
    technology: str | None = None
    version: str | None = None
    component: str | None = None
    customer: str | None = None
    region: str | None = None
    state: str | None = None
    identifiers: list[str] = Field(default_factory=list)

    authoritative_documentation: list[EvidenceItem] = Field(default_factory=list)
    documentation: list[EvidenceItem] = Field(default_factory=list)
    wiki_evidence: list[EvidenceItem] = Field(default_factory=list)
    tfs_evidence: list[EvidenceItem] = Field(default_factory=list)
    known_bug_evidence: list[EvidenceItem] = Field(default_factory=list)
    historical_case_evidence: list[EvidenceItem] = Field(default_factory=list)
    current_log_evidence: list[EvidenceItem] = Field(default_factory=list)

    claims: list[Claim] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    sufficiency: SufficiencyLevel = SufficiencyLevel.INSUFFICIENT
    rejected_evidence: list[str] = Field(default_factory=list)
    """Human-readable "title -- why excluded" entries, kept purely for
    diagnosis (``ChatResponse.debug``) -- never shown to the end user
    by default."""
    retrieval_profile: list[str] = Field(default_factory=list)
    """The intent's own source-type priority order actually used (see
    ``retrieval_profile.RETRIEVAL_PROFILES``), recorded here so a
    caller (and ``ChatResponse.debug``) can see exactly which profile
    governed this turn's evidence selection."""

    def all_evidence(self) -> list[EvidenceItem]:
        return (
            self.authoritative_documentation
            + self.documentation
            + self.wiki_evidence
            + self.tfs_evidence
            + self.known_bug_evidence
            + self.historical_case_evidence
            + self.current_log_evidence
        )

    def has_authoritative_evidence(self) -> bool:
        return any(
            item.authority in (SourceAuthority.AUTHORITATIVE_DEFINITION, SourceAuthority.AUTHORITATIVE_CONFIGURATION)
            for item in self.all_evidence()
        )
