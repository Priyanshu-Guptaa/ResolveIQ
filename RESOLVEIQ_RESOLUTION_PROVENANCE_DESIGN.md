# Resolution Provenance — Design

**Status: design only. No code written. Deterministic, zero-LLM — consistent with every other engine in this codebase.**

---

## 1. Principle

Every recommendation ResolveIQ already produces traces back to something real — a score, a matched entity, a rule that fired. The gap isn't that evidence doesn't exist; it's that **it's scattered across five different shapes** (`KnowledgeMatch.metadata`, `ExternalMatch.match_reasons`, `RecommendedLogCollectionItem.match_reason`, `RootCauseHypothesis.rationale`, and — nowhere at all for SQL) and **there's no single tier that says how much any of it is worth trusting**. This design does two things, both additive, both reusing what already exists rather than replacing it:

1. A uniform `EvidenceReference` shape so every one of the 8 questions you asked can be answered the same way, regardless of source type.
2. A `ResolutionProvenance` tier (`Confirmed`/`Likely`/`Possible`/`Unknown`), computed by a deterministic rule that never lets a similarity score alone reach `Confirmed`.

No LLM, no new retrieval, no new matching algorithm. This phase is entirely "take what `RecommendationEngine` already computed and make its reasoning legible and auditable."

---

## 2. Current state, per question — what already exists, and the real gaps

| # | Question | What already answers it | Gap |
|---|---|---|---|
| 1 | Why was this historical investigation recommended? | `KnowledgeMatch.score` (cosine similarity) + `metadata["applicability_reasons"]` when the Phase 1 applicability ranker fired | **No synthesized reason string at all when applicability didn't fire** — a bare "score=0.82" is not an answer to "why" |
| 2 | Why was this TFS bug recommended? | `ExternalMatch.match_reasons: list[str]` — already real, deterministic, built by `ranking.py`'s `score_candidate()` ("Same component: X", "Title overlap: 40%", ...) | None — this is the best-instrumented reason in the system today |
| 3 | Why was this Wiki document recommended? | Same `ExternalMatch.match_reasons` (identical mechanism) | None |
| 4 | Why was this log recommended? | `RecommendedLogCollectionItem.match_reason` (why the scenario matched) + `.explanation` (why this step, from the wiki's own text) | None |
| 5 | Why was this SQL query recommended? | `SuggestedSqlItem.explanation` — but this is the **template's own static description of what the query does**, not why it was matched to *this* investigation | **No per-item "why this template, for this investigation" reason at all** — the only real signal (`matched_component.component_name in template.related_components`) is never surfaced as text |
| 6 | What evidence supports the proposed root cause? | `RootCauseHypothesis.rationale` — a real sentence ("Matches historical investigation 'X' (82% similarity)") | Text only — no structured, clickable reference back to the specific `KnowledgeMatch`/entity that produced it |
| 7 | What evidence supports the proposed resolution? | `RecommendedSolution.rationale` + `source_local`/`source_tfs`/`source_wiki` + `supporting_tfs_id`/`url` + `supporting_wiki_title`/`url` | **Asymmetric**: TFS and Wiki get structured, clickable supporting-reference fields; local Historical Investigations do not — `source_local=True` has no equivalent `supporting_historical_investigation_id` |
| 8 | Confirmed / Likely / Possible / Unknown? | `RecommendedSolution.confidence: "High"/"Medium"/"Low"/"Insufficient"` | **This is a score-derived label, not the provenance tier you're asking for.** It's exactly the thing "don't let similarity alone establish Confirmed" is warning about — today, nothing stops a single very-high-similarity match from showing as "High" confidence with no cross-source corroboration at all. |

**Bottom line**: five of eight questions are already well-answered by existing code. Three real gaps: historical investigations lack a per-item reason, SQL lacks one entirely, and the confidence label conflates "the match was strong" with "this is safe to call resolved" — which is exactly what you're asking this phase to separate.

---

## 3. Proposed data model

New module, `app/domain/provenance.py` (mirrors the shape already sketched for the — separately gated, not-yet-approved — LLM Gateway's `EvidenceItem`, deliberately: this is the correct shape either way, and building it now means that future phase reuses it instead of inventing a second one).

```python
class ResolutionProvenance(str, Enum):
    CONFIRMED = "confirmed"
    LIKELY = "likely"
    POSSIBLE = "possible"
    UNKNOWN = "unknown"

class EvidenceKind(str, Enum):
    HISTORICAL_INVESTIGATION = "historical_investigation"
    KNOWN_BUG = "known_bug"
    DOCUMENTATION = "documentation"
    TFS_CASE = "tfs_case"
    WIKI_PAGE = "wiki_page"
    SQL_TEMPLATE = "sql_template"
    LOG_COLLECTION_STEP = "log_collection_step"
    ENTITY_HEURISTIC = "entity_heuristic"
    COMPONENT_MATCH = "component_match"

class EvidenceReference(BaseModel):
    kind: EvidenceKind
    source_id: str          # the REAL id already on the underlying object --
                             # KnowledgeMatch.record_id, TfsCase.tfs_id, WikiPage.page_id,
                             # QueryTemplate.id, LogCollectionScenario.id, ...
    title: str
    url: str | None = None  # already exists on TFS/Wiki matches; None for local-only sources
    score: float | None = None      # the real, already-computed score -- never re-derived
    reason: str                     # deterministic explanation -- reuses match_reasons/
                                     # match_reason/rationale wherever they already exist;
                                     # newly synthesized only for the two real gaps (§2)
    contributes_to: list[str]       # "root_cause" | "resolution" | "log_recommendation" |
                                     # "sql_recommendation" -- an item can contribute to more
                                     # than one (e.g. the same historical investigation can be
                                     # both a root-cause source and cited in the resolution)

class ProvenanceRecord(BaseModel):
    """Answers all 8 questions for one InvestigationStrategy, in one
    traceable object. Attached as InvestigationStrategy.provenance --
    additive, nothing existing changes shape."""
    root_cause_evidence: list[EvidenceReference]
    resolution_evidence: list[EvidenceReference]
    log_recommendation_evidence: list[EvidenceReference]
    sql_recommendation_evidence: list[EvidenceReference]
    resolution_provenance: ResolutionProvenance
    provenance_rationale: str   # deterministic, e.g. "Confirmed: TFS-2467475 and local
                                 # investigation HI-a19f... share ticket CS0122697" or
                                 # "Likely: single local match at 82% similarity with a
                                 # recorded root cause, no cross-source corroboration"
```

Documentation matches are deliberately **excluded** from `resolution_provenance` entirely — a wiki/doc match is "troubleshooting guidance," never itself a resolution claim, so it's never tiered Confirmed/Likely/Possible. It still gets an `EvidenceReference` (kind=`DOCUMENTATION`), just with `contributes_to` empty or limited to whatever it genuinely supports.

---

## 4. The Confirmed rule — the actual answer to "don't let similarity alone establish Confirmed"

```
UNKNOWN
    if insufficient_evidence (unchanged existing signal)

CONFIRMED
    if a real cross-source correlation exists
       -- REUSES the already-implemented correlated_local logic in
          _synthesize_recommendation(): a TFS Bug's own LandisGyr.CRMID
          field matches a local Historical Investigation's ticket tag.
          Two independent systems agreeing on the same real-world
          incident, not one high score.
    OR an explicit human verification flag is set
       -- NEW: KnownBugRecord.resolution_verified / HistoricalInvestigationRecord.
          resolution_verified (see §5) -- an admin explicitly marking a
          specific fix as confirmed. Never set by any automated process.

LIKELY
    if the single best source clears an existing, already-used bar AND
    carries real resolution content:
       - a local Historical Investigation at/above min_similarity_for_root_cause
         WITH a recorded root_cause (not the title-only fallback --
         _synthesize_recommendation already distinguishes these two cases
         today, just never exposes the distinction as a tier), OR
       - a TFS/Wiki match at "High" confidence (ranking.py's own existing
         band, score >= 0.65) WITH real resolution_text/excerpt

POSSIBLE
    any real evidence exists at all but doesn't clear the LIKELY bar --
    entity heuristics, Medium/Low external matches, a local match with
    no recorded root cause (title-only, already-existing fallback path)

UNKNOWN
    otherwise
```

Every threshold here (`min_similarity_for_root_cause`, the 0.65 "High" band, the correlated-ticket logic) already exists and is already tested — this rule doesn't invent new numbers, it names tiers around numbers this system already trusted implicitly.

---

## 5. Two small, additive changes needed (both schema, both proposed here, neither implemented)

**5.1 — `resolution_verified` fields**, additive columns via the existing `_add_missing_columns` migration mechanism (the same safe path used for every column added this project, including all of Phase 1's):

```
KnownBugRecord:               HistoricalInvestigationRecord:
  resolution_verified: bool = False    resolution_verified: bool = False
  resolution_verified_by: str | None   resolution_verified_by: str | None
  resolution_verified_at: datetime | None   resolution_verified_at: datetime | None
```

No admin UI to *set* this is proposed as part of this phase — the field existing and being readable is what `ResolutionProvenance.CONFIRMED` needs; a small "mark verified" action is a natural, separate follow-on once you've reviewed this design (same discipline as the review-queue accept/reject action built for classification).

**5.2 — Two field additions to close the real gaps in §2**:
- `KnowledgeMatch.reason: str = ""` — a deterministically synthesized sentence for historical investigation / documentation / known-bug matches when nothing more specific (applicability reasons) already exists. Built from data already on the match: score band, whether it carries a recorded root cause/resolution, applicability reasons when present. Same discipline as every other `_match_reason`-style function in this codebase — never invents anything not already computed.
- `SuggestedSqlItem.match_reason: str = ""` — closes gap #5 directly: e.g. `"Linked to matched component 'CommandProcessorHost' via this SQL template's related_components"` for `source="sql_library"`, or `"Investigation evidence contains a sql_session entity"` for `source="entity_heuristic"` — both facts the engine already has, just never stated.

---

## 6. Where this plugs into `RecommendationEngine`

One new method, `_build_provenance_record()`, called from `_build_strategy()` alongside the existing `_synthesize_recommendation()` call (same inputs it already receives: `root_causes`, `similar_investigations`, `tfs_matches`, `wiki_matches`, `recommended_logs`, the `suggested_sql` list, `matched_component`) — it does not recompute anything, it wraps what those calls already produced into `EvidenceReference`s and applies §4's rule. `InvestigationStrategy` gets one new field, `provenance: ProvenanceRecord`, additive, same pattern as every field added to that model since Phase 2C.

---

## 7. Reuse inventory (what you asked for explicitly)

| Existing piece | Role in this design |
|---|---|
| `KnowledgeMatch` (`app/domain/recommendation.py`) | Source of historical investigation / documentation / known-bug evidence — reused as-is, `+reason` field added |
| `ExternalMatch` / `TfsCase` / `WikiPage` (`app/domain/external_knowledge.py`) | Source of TFS/Wiki evidence — `match_reasons`, `score`, `url` all reused unchanged |
| `RecommendedLogCollectionItem` | Source of log evidence — `match_reason`/`explanation` reused unchanged |
| `SuggestedSqlItem` | Source of SQL evidence — `+match_reason` field added |
| `RootCauseHypothesis` | Source of root-cause evidence — `rationale`/`confidence` reused unchanged |
| `RecommendedSolution` | Source of resolution evidence and the existing `correlated_local` cross-source logic — reused unchanged, `ProvenanceRecord` is built alongside it, not instead of it |
| `MatchedComponent` | Source of component-match evidence |
| `KnowledgeRepository` (`app/infrastructure/db/knowledge_repository.py`) | Gets the two new `resolution_verified*` columns on `known_bugs`/`historical_investigations` — same repository, no new one |
| `RecommendationEngine._synthesize_recommendation`, `._build_root_causes`, `._recommend_logs`, `._suggested_sql_items`, `._match_component` | All called exactly as today; `_build_provenance_record` only wraps their outputs |
| `ranking.py`'s `confidence_for_score` / `_CONFIDENCE_HIGH` band | Reused directly for the LIKELY-tier TFS/Wiki check in §4 |
| `_add_missing_columns` migration mechanism | Reused for §5.1's new columns — no new migration machinery |

**No new database tables.** `ProvenanceRecord` is computed fresh on every `generate()` call, exactly like `RecommendedSolution` already is — never persisted, never a second source of truth to keep in sync.

---

## 8. What this phase explicitly does not include

- The LLM Gateway (separately designed, not approved, not started).
- An admin UI/API action to set `resolution_verified` — the field is designed here; the action to set it is a natural next increment once you've reviewed this.
- Persisting/auditing `ProvenanceRecord` over time (e.g. "show me how confidence changed as evidence accumulated") — out of scope, would need real storage design of its own.
- Any change to retrieval, ranking, or classification — this phase only adds a reasoning/explanation layer on top of what those already decided.

---

## 9. Phased plan

1. `app/domain/provenance.py` (new, pure data model) + tests for the §4 rule in isolation (given canned inputs, assert the right tier).
2. The two additive schema/field changes (§5) + migration verification.
3. `_build_provenance_record()` in `RecommendationEngine`, wired into `_build_strategy()`, `InvestigationStrategy.provenance` added.
4. Regression tests: one per question in §2 (does the resulting `ProvenanceRecord` actually answer it, using real fixture data patterned on this session's real corpus cases — TEPCO, ATCO RF Mesh IP, CLECO).
5. Live verification against the real corpus (same discipline as every phase this session), full suite run, report before commit.

---

## 10. Open questions

1. Who should be allowed to set `resolution_verified` — any authenticated actor (there's no role system yet, per `GovernanceFields`' own docstring), or is this the first thing in ResolveIQ that needs a real permission check? I'd lean "no role system yet, so anyone using the admin API can, same as every other admin action today" — but flagging since it's a "this fact becomes CONFIRMED for everyone" action, a step up in consequence from editing a document's tags.
2. Should `POSSIBLE` ever be shown to an engineer as "here's a hypothesis" in the UI today, or is this phase purely building the data model or wiring it into the existing Investigation Strategy panels? I've scoped this design to the engine/data layer per your request; UI surfacing (a visible Confirmed/Likely/Possible/Unknown badge) would be a small, separate follow-on I can size once this lands.
3. §5.2's new `reason`/`match_reason` fields touch `KnowledgeMatch` and `SuggestedSqlItem` — both are read by existing UI code (`ui/components/`). Adding an optional field with a default is additive and shouldn't break anything, but I'll confirm no UI file does strict schema validation that would reject an unexpected field before I touch either model.

---

*No code has been written. Waiting for your review before implementation begins.*
