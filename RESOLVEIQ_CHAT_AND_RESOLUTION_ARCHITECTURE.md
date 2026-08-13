# ResolveIQ — Structured Resolution Knowledge & Chat Assistant Architecture

**Status: design only. No code written, no database changed, no dependencies added. Every claim below was verified against the actual repository (file references given) or the live database on 2026-08-13 — nothing here is assumed or illustrative.**

---

## 1. Executive Summary

ResolveIQ already has almost every load-bearing piece this phase needs — it just doesn't have a conversational front door yet. Concretely, as of today:

- **Resolution Provenance** (`app/domain/provenance.py`) shipped this session: every recommendation carries a `CONFIRMED`/`LIKELY`/`POSSIBLE`/`UNKNOWN` tier, computed by a deterministic rule that can never reach `CONFIRMED` from a similarity score alone, plus a full `EvidenceReference` trail naming the real source behind every root cause, resolution, log recommendation, and SQL suggestion.
- **Applicability-aware retrieval** (`app/engines/knowledge/applicability.py`) shipped in Phase 1: local semantic search is re-ranked by real Customer/Region/Technology signals before truncation, not left as raw cosine similarity.
- **The RF Mesh vs RF Mesh IP conflation** — the specific defect that damaged trust in the team demo — is fixed at three independent layers (log-collection matching, technology inference, metadata classification), all via one generic mechanism (`Technology.parent_technology_id` + `app.engines.shared.hierarchy.most_specific`), not a hardcoded special case.
- **TEPCO-as-customer** is now structurally guaranteed, not just avoided by luck: `Customer` is its own governed `KnowledgeObjectType`, seeded only from real evidence, and the classification engine's own architecture makes a customer name mechanically unable to become a Technology/Component/Product suggestion (`app/engines/knowledge/classification.py`'s module docstring explains why).
- **What's genuinely missing**: a query-understanding layer that turns a free-text question into the same `RetrievalContext`/`InvestigationStrategy` shape an open investigation already produces; conversation state; a real Structured Resolution Knowledge *view* (the data already exists, scattered correctly across governed models — it has never been assembled into the "Problem → Symptoms → Applicability → Root Cause → Evidence → Resolution → Validation" shape end-to-end); and the LLM/Copilot answer-synthesis layer itself (designed in a prior document, never built).

**The central architectural finding of this document**: the Chat Assistant is not a new reasoning system. It is a new *entry point* — query understanding plus conversation state — sitting in front of the exact same `RecommendationEngine`/`ApplicabilityRanker`/`ProvenanceRecord` pipeline the Investigation Workspace already calls. Everything downstream of "what does this question mean, and what `RetrievalContext` does it imply" is REUSE, not NEW. This is precisely the discipline the user's own decision log established in the prior Context Dimensions phase ("Investigation Workspace and Chat share one knowledge/retrieval/reasoning layer — no forked logic") and it holds up under this deeper inspection.

**Recommended sequencing** (detailed in §35): (1) close two small, real gaps in what already exists — Known Bugs aren't yet a resolution-provenance source, and the 409-item classification review queue is unworked; (2) build Structured Resolution Knowledge as a *read model* over existing data, no new source-of-truth table; (3) build Query Understanding + Conversation State (the two genuinely new subsystems); (4) wire Chat to the existing engines; (5) the LLM/Copilot Gateway, reusing the design already produced and approved-in-direction last session, now with a real `ProvenanceRecord` to consume instead of a hypothetical one.

---

## 2. Current Architecture Assessment

Layering is consistent and has held up across nine phases of work without a forced exception:

```
app/domain/*            -- pydantic models, framework-agnostic, zero I/O
app/engines/*            -- business logic, one package per capability
app/infrastructure/db/*  -- SQLAlchemy repositories (one per governed table/table-family)
app/infrastructure/vectorstore/* -- ChromaDB client wrapper
app/api/routers/*        -- thin FastAPI routers, no logic of their own
app/api/dependencies.py  -- the one place every engine is constructed and wired
ui/views/*                -- Streamlit pages, talk to the API only via ui/api_client.py
```

Zero LLM anywhere in the runtime today — re-confirmed this pass (grep across `app/` for any LLM/API-provider client turns up only comments stating "no LLM reasoning" as a design constraint). Every value a user sees is a database row, a deterministic keyword/regex/hierarchy match, or a cosine-similarity vector search, each with its scoring function inspectable in plain Python. This discipline is the reason the RF Mesh/TEPCO defects found in the team demo were **fixable** at all — every wrong answer traced to one identifiable function, not an opaque model weight.

The **Knowledge Object Framework** (`app/domain/knowledge_relationships.py`, `app/engines/knowledge_object_framework/`) is the standing generalization pattern this whole codebase already uses for "new knowledge type shows up, give it identity, versioning, and relationships for near-free": eleven `KnowledgeObjectType`s today (Component, Document, Known Bug, SQL Template, Historical Investigation, Playbook, Technology, Product, Version, Log Source Application, Log Collection Scenario, Customer, Region), one generic edge table (`KnowledgeRelationship`), one generic version-snapshot table (`EntityVersion`, `app/domain/entity_version.py`), one adapter registry (`build_adapters()`) that dispatches by type to each object's real repository. This pattern is directly reusable for anything new this phase needs (Release Notes, a `SUPERSEDES` relationship, Chat's own session/message persistence does *not* need this pattern — see §11).

---

## 3. Existing Knowledge Sources

Live counts, `data/resolveiq.db`, verified today (2026-08-13):

| Source | Table | Rows | Notes |
|---|---|---|---|
| Documentation | `documentation` | 496 | 474 real L+G wiki docs bulk-imported 2026-08-12, rest earlier seed/upload |
| Historical Investigations | `historical_investigations` | 728 | ServiceNow `task.xlsx` import |
| Known Bugs | `known_bugs` | 4 | Original seed data — never grown, never re-imported |
| Component Profiles | `component_profiles` | 11 | 6 fictional Sprint-1 samples + 5 real |
| SQL Templates | `sql_templates` | 12 | 5 seed + 7 real, added this session |
| Log Source Applications | `log_source_applications` | 85 | One wiki page import, frozen "bug fixes only" |
| Log Collection Scenarios | `log_collection_scenarios` | 40 | Same import |
| Playbooks | `playbooks` | **0** | Schema and framework exist, never populated |
| Products (governed) | `products` | 1 | Schema exists, minimally used |
| Technologies (governed) | `technologies` | **11** | Was 0 before this session; now real, hierarchy-populated (RF Mesh family) |
| Versions (governed) | `versions` | **0** | Schema exists, empty |
| Customers (governed) | `customers` | 5 | TEPCO, CLECO, ATCO, SRP, GPA — all `verified=True`, `source_type="document_title_corpus"` |
| Regions (governed) | `regions` | 3 | APAC, NAM, Guam |
| Knowledge Relationships (graph edges) | `knowledge_relationships` | 59 | 35 `applies_to` (Customer/Region tags), 8 `implements_logging_for`, 16 `related_to` |
| Metadata Classification Suggestions | `metadata_classification_suggestions` | 649 | **79 auto-accepted, 1 accepted, 39 rejected, 121 recorded-low, 409 still pending review** — a real, unworked backlog (see §8) |
| Investigations (live workspace) | `investigations` | 31 | Real engineer-created investigations from this project's own testing/verification |
| TFS Bugs/Issues | *(not stored)* | n/a | Live query only, confirmed reachable this session (real TFS-2051535/2462358/304438 returned in live verification) |
| Confluence Wiki | *(not stored)* | n/a | Live connector built, **still not live-verified against the real instance** — needs a PAT from a Wiki admin |
| Release Notes | *(no table)* | n/a | No domain model exists yet — blocked on you providing real files (§16) |

Only **15 of 496** documents have `technology` set and **28 of 496** have `product` set as scalar fields — the classification engine's High-confidence auto-accept tier is deliberately conservative (title-exact-match only), so most of the corpus's real applicability signal is sitting in the 409-item pending queue, not yet asserted as fact. This is the single most consequential fact for this design: **Chat cannot answer "does this apply to TEPCO" more accurately than the underlying classification data allows**, and today, most of that data hasn't been reviewed. §8 and §35 treat clearing this backlog as a prerequisite, not a nice-to-have.

---

## 4. Current Retrieval Flow

```
1. RecommendationEngine._resolve_retrieval_context() builds a RetrievalContext
   from investigation.customer (governed-list exact/alias match only,
   never inferred) and _infer_technology() (hierarchy-aware, specificity-
   resolving, positive-evidence-only -- see engine.py's own extensive
   docstrings on this from this session's defect fixes).

2. ChromaKnowledgeStore.query() -- pure cosine similarity, over-fetched to
   4x top_k (_APPLICABILITY_OVERFETCH_MULTIPLIER), against
   historical_investigations / documentation / known_bugs collections.

3. ApplicabilityRanker.rerank() -- adjusts each candidate's score using
   real KnowledgeRelationship-derived Customer/Region tags and the
   Technology parent/child hierarchy, THEN truncates to top_k. A same-
   customer match is boosted (+0.20); a real, KNOWN different customer is
   penalized (-0.25), never merely left to raw similarity; a technology
   family match (RF Mesh IP vs RF Mesh) gets a partial boost (+0.08),
   never collapsed into a full match. No signal fabricated when a
   dimension is genuinely unknown on either side.

4. _annotate_match_reasons() (Resolution Provenance phase) -- every
   surviving match gets a real "why" string, whether or not applicability
   fired.

5. External Knowledge (TFS/Wiki), independently, in parallel: anchor/
   supporting term split (never one flat OR'd bag -- this was a real,
   live-observed fix; see external_knowledge/service.py), deterministic
   fixed-weight re-ranking (ranking.py: component 0.30, technology 0.20,
   title overlap <=0.25, entity match <=0.30, resolved-state 0.10,
   customer 0.15 -- customer is ranking-only, never a query anchor, for
   the same reason it doesn't dilate local search either).
```

This is a genuinely mature retrieval pipeline for a *known investigation with an open context*. What it does not yet have: an entry point that accepts a bare question with no `InvestigationSession` behind it and produces the same `RetrievalContext`. That gap is Query Understanding (§12), not a retrieval redesign.

---

## 5. Current Recommendation Flow

`RecommendationEngine.generate(investigation) -> Recommendation` (`app/engines/recommendation/engine.py`, ~1690 lines) is the single orchestration point:

```
generate()
 -> _infer_technology()              -- hierarchy-aware, evidence-gated
 -> _resolve_retrieval_context()     -- RetrievalContext (customer/region/technology)
 -> local search x3 (historical/docs/bugs) -> ApplicabilityRanker.rerank() -> _annotate_match_reasons()
 -> _build_root_causes()             -- historical-match-derived + entity-heuristic hypotheses
 -> _recommend_logs()                -- Log Intelligence KB scenario matching, specificity-demoted
 -> _next_action()
 -> _build_strategy()
     -> _match_component()           -- full-name-only Component Registry match
     -> _required_evidence() / _current_stage() / _progress()
     -> _suggested_sql_items()       -- SQL Library via matched_component.related_components, else entity heuristics
     -> ExternalKnowledgeService.gather() -- live TFS + Wiki, parallel, timeout-bounded
     -> _synthesize_recommendation() -- cross-source correlation, single RecommendedSolution
     -> _build_provenance_record()   -- ProvenanceRecord (Resolution Provenance phase)
     -> InvestigationStrategy        -- the one orchestrated response object
```

`InvestigationStrategy` is already, structurally, almost exactly the "Structured Resolution Knowledge" object §9 is asked to design — it just isn't yet assembled into that explicit conceptual shape or persisted/citable as a standalone artifact independent of one investigation's live state. §9 makes this precise.

---

## 6. Resolution Provenance Assessment

Shipped this session (`app/domain/provenance.py`, wired into `RecommendationEngine`, 17 dedicated tests + full-suite regression, live-verified against 10 real investigations from the live corpus — see the prior session report). Current state:

- **Working correctly against real data**: 10 live investigations tested, none falsely reached `CONFIRMED` (correct — no real cross-source ticket correlation or human verification exists in the corpus yet), one reached `LIKELY` on real evidence (`INC0045821`, a single strong local match with a recorded root cause), the rest correctly landed at `POSSIBLE`.
- **Real, acknowledged gap #1**: `KnownBugRecord.resolution_verified*` fields exist and round-trip through the repository, but `_resolve_provenance_tier`/`_synthesize_recommendation` never actually consume a Known Bug as a resolution source at all — this is a pre-existing structural gap (Known Bugs were never wired into `_synthesize_recommendation`, this phase didn't expand that), not a regression. Relevant here because a user asking "is there a known bug for this?" (explicitly listed in §1's example questions) deserves the same provenance treatment TFS/local/Wiki already get.
- **Real, acknowledged gap #2**: no admin UI/API action exists to *set* `resolution_verified` — by explicit prior-phase decision, deferred pending an access-control decision (there is no role/permission system anywhere in ResolveIQ yet — `GovernanceFields`' own docstring notes `created_by`/`updated_by` are plain optional strings, not a real user FK).
- **Documentation matches are deliberately never tiered** — a wiki/doc match is guidance, never itself a resolution claim; this is correct and should be preserved unchanged.

---

## 7. Problems Observed During Team Demo — Current Status

| Problem | Status | Where it was fixed |
|---|---|---|
| RF Mesh / RF Mesh IP conflation | **Fixed**, three layers | `_recommend_logs` specificity demotion, `_match_single_technology`/`_technology_evidence_score`, classification `_best_match` — all via `Technology.parent_technology_id` + `most_specific()` |
| TEPCO treated as a module/component | **Structurally prevented**, not just fixed once | `Customer` is its own `KnowledgeObjectType`; classification candidate lists are curated from governed tables, never derived from arbitrary document titles for non-Customer dimensions |
| Customer/region info mixed with product info | **Mechanism built, data incomplete** | `APPLIES_TO` relationships + `ApplicabilityRanker` work correctly; only 35 `applies_to` edges exist against 496 documents — most of the corpus is not yet tagged (§3, §8) |
| Could not reliably determine applicability | **Fixed for local + TFS/Wiki search** | `ApplicabilityRanker` (local), `ranking.py`'s customer/technology weights (external) |
| Returned unrelated info via pure semantic similarity | **Mitigated, not eliminated** | Applicability re-ranking demotes/promotes; it does not *filter out* a low-relevance candidate that has no applicability signal either way — a genuinely irrelevant match with no known customer/technology tag still surfaces at raw similarity. Query Understanding (§12) closes more of this by giving retrieval a real structured query instead of raw free text. |
| No conversational follow-up | **Not built** | This document's primary subject (§10–§13) |

---

## 8. Root Causes

1. **No query-understanding layer.** Every existing entry point (`InvestigationSession.context_text`, `Documentation.title/content`) assumes the caller already assembled a coherent text blob; nothing today parses a single natural-language question into `{customer, region, technology, component, symptom, intent}`.
2. **No conversation state.** `InvestigationSession` is the only stateful container that persists across turns, and it's built for evidence accumulation (logs, notes), not a dialogue.
3. **Classification backlog.** 409 pending-review suggestions mean most of the corpus's real Customer/Region/Technology/Component signal is sitting unused. This is not a code gap — the pipeline that would apply them (`DocumentClassificationEngine.accept()`) already exists and works (verified: `accepted=1` in the live DB proves the code path is exercised) — it's an unworked queue.
4. **No Release Notes model.** Version-scoped "what changed" questions (§16, explicitly requested: *"Could this issue be caused by something introduced in CC 9.0.3?"*) have literally no data to answer from yet.
5. **Known Bugs are a second-class resolution source.** Structurally present, never wired into synthesis (§6).
6. **No persisted, standalone-citable resolution record.** `InvestigationStrategy` is real but ephemeral (recomputed on every `generate()` call, tied to one `InvestigationSession`) — there is no "this is the canonical answer to this class of problem" object a chat answer or a future investigation could cite directly without re-running the whole pipeline. §9 addresses this as a read model, not a new source of truth.

---

## 9. Structured Resolution Knowledge Model

The user's proposed shape — Problem → Symptoms → Applicability → Root Cause → Evidence → Resolution → Validation → Related Knowledge → Confidence — maps almost entirely onto data that already exists, scattered across governed models exactly because this codebase never modeled "a resolved issue" as a single monolithic object (a decision that has held up well: it's why Customer/Region/Provenance could each be added additively without a schema rewrite). The mapping:

| Proposed field | Existing model / field | Disposition |
|---|---|---|
| Problem | `HistoricalInvestigationRecord.title` / `InvestigationSession.title` | REUSE |
| Symptoms | `HistoricalInvestigationRecord.description`, `.tags`; `InvestigationStrategy.required_evidence`/`.matched_component` | REUSE |
| Applicability (Product/Component/Technology/Customer/Region/Version) | `RetrievalContext` (Customer/Region/Technology — built per-request, not persisted); `related_components` (real FK); `Product`/`Version` (governed, sparsely populated) | **REUSE the mechanism, EXTEND the data** — `RetrievalContext` needs a persisted counterpart for a *specific resolved case*, not just an ephemeral per-query object (see below) |
| Root Cause | `RootCauseHypothesis` | REUSE |
| Evidence | `ProvenanceRecord.root_cause_evidence` / `.resolution_evidence` (`EvidenceReference`) | REUSE, wholesale |
| Resolution | `RecommendedSolution.recommended_resolution` | REUSE |
| Validation | **Does not exist anywhere** | **NEW** — no domain model today captures "how do you confirm the fix worked" as structured data; `RecommendedSolution` has no validation-steps field |
| Related Knowledge | `KnowledgeRelationship` graph (`ExplorerView`) | REUSE |
| Confidence | `ResolutionProvenance` | REUSE |

**What this means concretely**: do **not** build a new `ResolvedIssue` or `StructuredKnowledge` table that duplicates `HistoricalInvestigationRecord`/`RecommendedSolution`/`ProvenanceRecord`. That would immediately create the exact two-sources-of-truth problem this codebase has consistently avoided (see `KnowledgeRelationship` superseding four legacy per-pair tables, `EntityVersion` superseding per-type version tables). Instead:

- **NEW, small**: `ValidationStep` — a short, structured field on `RecommendedSolution` (`validation_steps: list[str]`, or a slightly richer `ValidationStep(description, expected_result)`), populated the same deterministic way every other field on that model is (quoted from the winning source's own text when available — a TFS `Microsoft.VSTS.Common.ResolvedReason`/closure note, a Wiki page's own "how to verify" section — never fabricated; empty list when no source states one, which is the honest common case today).
- **EXTEND, not NEW**: a persisted `ApplicabilityProfile` (small, new table) that captures, for one *specific* Historical Investigation or Known Bug, the resolved Customer/Region/Technology/Version/Component it applies to — today this only exists as `related_components` (FK, real) plus free-text `tags` (real, but not structured) on `HistoricalInvestigationRecord`. This is the one place a genuinely new table is warranted, because "which governed dimension values does this specific resolved case apply to" is not currently expressible as anything but a relationship-graph walk (`APPLIES_TO`, currently only wired for Documents, not Historical Investigations/Known Bugs) — extending `APPLIES_TO`'s existing target set to include `HISTORICAL_INVESTIGATION`/`KNOWN_BUG` as *from*-types is the mechanism; no new relationship type, no new table beyond what `KnowledgeRelationship` already is. **Correction to the above**: on inspection, this needs zero new tables — `KnowledgeRelationship` already supports any `from_type`, so this is a **data-population gap** (classification never runs against Historical Investigations/Known Bugs today, only Documents — `DocumentClassificationEngine.run()` iterates `self._knowledge.list_documentation_summaries(...)` exclusively), not a schema gap. **EXTEND the classification engine's scan target, not the schema.**
- **NEW (read-only)**: a `StructuredResolution` *view* — a Pydantic model, computed on demand (like `InvestigationStrategy`, never a second source of truth), that assembles Problem/Symptoms/Applicability/Root Cause/Evidence/Resolution/Validation/Related/Confidence from a `HistoricalInvestigationRecord` or `KnownBugRecord` plus its `KnowledgeRelationship` edges plus a synthesized `ProvenanceRecord`. This is what Chat cites when it says "here's a structured resolution." Computed fresh, same discipline as everything else in this codebase — never persisted separately, never drifts from its sources.

---

## 10. Chat Assistant Architecture

```
                    USER
                     |
              ChatOrchestrator (NEW, app/engines/chat/)
                     |
        +------------+-------------+
        |                          |
  QueryUnderstanding (NEW)   ConversationState (NEW)
        |                          |
        +------------+-------------+
                     |
         RetrievalContext (REUSE, RecommendationEngine._resolve_retrieval_context
                            generalized to accept chat-derived signals too)
                     |
   RecommendationEngine.generate()-equivalent (REUSE almost entirely --
   see "Standalone Question" note below)
                     |
         InvestigationStrategy / StructuredResolution (REUSE / NEW read model, §9)
                     |
              ChatResponse (NEW, deterministic template -- Chat V1, no LLM)
```

**Standalone Question note**: `RecommendationEngine.generate()` takes a full `InvestigationSession`. A bare chat question has no evidence, no uploaded logs, most of the Summary Card fields blank. Two options were considered:

- (A) Synthesize a throwaway `InvestigationSession` from the chat question (`title=question`, one `Evidence` item holding the question text) and call `generate()` unchanged.
- (B) Add a second, narrower entry point that takes a `RetrievalContext` + raw text directly, skipping the log-collection/required-evidence/stage-progression logic that only makes sense for a real, evolving investigation.

**Recommendation: (A) for V1.** It requires zero new engine code — `generate()` already degrades gracefully when there's no `technology`/`customer` set (falls back to inference) and when there's no log evidence (`_recommend_logs` still runs against whatever text exists). `_current_stage`/`_progress`/`_required_evidence` will report values that don't mean much for a standalone question (e.g. "Triage" stage), but the *response renderer* (not the engine) simply doesn't surface those fields in Chat's answer — same InvestigationStrategy, different UI-layer projection. This avoids a second, parallel recommendation pathway that could drift from the first, which is exactly the risk the user's own decision log flags. Revisit (B) only if V1's real usage shows the synthesized-session approach producing confusing intermediate state (e.g. accidentally polluting `investigations` table with throwaway rows — mitigated by simply not persisting the synthesized session at all; `generate()` doesn't require its input to be saved).

**REUSE / EXTEND / NEW for this section**: `RecommendationEngine` REUSE (unchanged signature and logic); `ApplicabilityRanker` REUSE; `ProvenanceRecord` REUSE; `ChatOrchestrator` NEW; `QueryUnderstanding` NEW; `ConversationState` NEW; `ChatResponse` rendering NEW (templated, same deterministic discipline as `RecommendedSolution`'s own templates).

---

## 11. Conversation Context Model

New domain module `app/domain/chat.py` (mirrors the shape sketched in the (unapproved, unbuilt) LLM Gateway design, kept consistent so that phase reuses it):

```python
class ChatSession(BaseModel):
    id: str
    investigation_id: str | None   # set when opened from inside an
                                     # Investigation Workspace -- shares
                                     # that investigation's real context
                                     # instead of re-deriving one
    slots: ConversationSlots        # see below -- the accumulated,
                                     # resolved context across turns
    created_at: datetime
    updated_at: datetime

class ConversationSlots(BaseModel):
    """What the conversation currently believes about the problem --
    the chat-native equivalent of RetrievalContext, but persisted and
    incrementally updated turn over turn, and carrying provenance for
    each slot (was this stated by the user, or inferred?)."""
    customer_id: str | None = None
    customer_source: Literal["stated","inferred","unset"] = "unset"
    region_id: str | None = None
    technology_name: str | None = None
    technology_source: Literal["stated","inferred","unset"] = "unset"
    component_id: str | None = None
    version_text: str | None = None
    last_focus: Literal["problem","historical_match","resolution","logs","sql"] | None = None
    """What "this"/"that"/"it" currently refers to -- see the pronoun-
    resolution note below."""
    last_referenced_investigation_id: str | None = None
    last_referenced_tfs_id: int | None = None

class ChatMessage(BaseModel):
    id: str
    session_id: str
    role: Literal["user","assistant"]
    content: str
    structured_response: ChatResponse | None = None  # assistant turns only
    created_at: datetime

class ChatResponse(BaseModel):
    intent: ChatIntent
    answer_text: str                 # deterministic template output, V1
    strategy: InvestigationStrategy | None   # the real, full evidence --
                                               # answer_text is a rendering
                                               # of this, never independent
    clarifying_question: str | None = None
    missing_slots: list[str] = []
```

**Pronoun/reference resolution** (the user's explicit "has this happened before" → "what was the fix" → "was that fix confirmed" → "does this apply to TEPCO" chain): `last_focus` + `last_referenced_investigation_id`/`last_referenced_tfs_id` is the whole mechanism needed. This is deliberately **not** an NLP coreference model — it's a simple last-write-wins pointer, updated by `QueryUnderstanding` whenever a turn's answer centers on one specific historical match, TFS case, or resolution. "What was the fix?" with no new customer/technology mentioned resolves against `last_referenced_investigation_id` before falling back to a fresh search. This is a deterministic rule, testable with fixed conversation transcripts (§31), not a probabilistic guess.

**Persistence**: two new tables, `chat_sessions`/`chat_messages`, same repository pattern as `investigations`/`evidence` (`SqlAlchemyChatRepository`, following `SqlAlchemyInvestigationRepository`'s exact shape). **NEW.**

---

## 12. Query Understanding

This is the one subsystem doing genuinely new work, and it must hold to the same "extractive, never inferred-from-a-pattern-alone for Customer" discipline §1's binding decisions require. Concretely, **not** a general NLP intent classifier — a deterministic, layered match against already-governed vocabulary, directly reusing the exact extractive matching `DocumentClassificationEngine`/`_match_single_technology` already implement:

```
QueryUnderstanding.parse(question: str, slots: ConversationSlots) -> ParsedQuery

1. Intent classification -- keyword/pattern rules against a fixed
   vocabulary (REUSE the discipline of _ENTITY_HEURISTICS'
   data-not-control-flow design):
     "what logs"          -> LOG_GUIDANCE
     "what sql" / "query"  -> SQL_GUIDANCE
     "has this happened"/"seen this before" -> HISTORICAL_LOOKUP
     "known bug"/"is there a bug"           -> KNOWN_BUG_LOOKUP
     "what changed"/"introduced in"          -> RELEASE_NOTE_LOOKUP
     "what should I do"/"how do I troubleshoot"/(no other pattern matched)
                                              -> TROUBLESHOOTING (default)
     "was that confirmed"/"is this confirmed" -> PROVENANCE_EXPLANATION (NEW
                                                  intent -- see §19)
     "is this X or Y" (two known Technology names in one question)
                                              -> DISAMBIGUATION_QUESTION

2. Slot extraction -- runs the SAME candidate-matching machinery
   classification.py already has (_Candidate, _title_match-equivalent
   full-phrase matching against governed Customer/Region/Technology/
   Component names+aliases) against the question text instead of a
   document title/body. REUSE, not reimplemented: extract this into a
   shared function both classification.py and QueryUnderstanding call
   (see the note on _match_single_technology's generic hierarchy logic,
   which is ALREADY reusable as-is).

   - Customer: extracted ONLY as an exact/alias match against the
     governed customers table -- "TEPCO" in "this is a TEPCO issue"
     resolves because TEPCO is a real Customer row, never because the
     word looks like an org name. An unrecognized customer name is
     surfaced as a clarifying question ("I don't have <X> as a known
     customer yet"), never silently dropped or guessed.
   - Technology: reuses _match_single_technology's exact evidence-
     coverage + hierarchy-specificity rule verbatim -- "Is this RF Mesh
     or RF Mesh IP?" is handled natively by this existing logic (both
     names present -> ambiguous tie -> the DISAMBIGUATION_QUESTION
     intent, which surfaces the SAME clarifying-question mechanism
     _match_single_technology already triggers by returning None on a
     genuine tie).
   - Component: full-name match against the live Component Registry,
     same rule as _match_component (no single-word partial match).

3. Slot merge -- new slots from this turn overlay ConversationSlots;
   nothing already-stated is silently overwritten by a weaker inferred
   signal (a stated customer from turn 1 survives even if turn 3's text
   doesn't repeat it).

4. Missing-context check (§ "Follow-up Questions" in the user's request)
   -- for TROUBLESHOOTING intent specifically, if technology is
   genuinely ambiguous (tied, not just absent) or a matched Technology
   has more than one plausible child variant with no way to disambiguate
   from the text, do not guess: return a ChatResponse with
   clarifying_question set and NO strategy computed yet. This is a
   direct, explicit implementation of the user's example ("I need two
   details before narrowing this down: 1. RF Mesh or RF Mesh IP? 2.
   Which Command Center version?") -- built from the SAME positive-
   evidence-required rule that already prevents a fabricated single
   answer (_technology_evidence_score's "ambiguous competing
   technologies -> None" rule, generalized to also ask a clarifying
   question instead of silently returning no technology at all, which
   is what happens inside an Investigation Workspace today).
```

**REUSE / EXTEND / NEW**: intent-pattern matching NEW (small, data-driven); Customer/Technology/Component slot extraction **EXTEND** (factor the existing `_Candidate`/matching logic in `classification.py` and `_match_single_technology` in `engine.py` into a shared, reusable function both call — currently near-duplicated logic, this phase is the forcing function to de-duplicate it properly); clarifying-question mechanism NEW but built directly on an existing "return None rather than guess" rule.

---

## 13. Applicability Engine

**REUSE, wholesale.** `ApplicabilityRanker` takes a `RetrievalContext`; Query Understanding's job is to produce one from conversation slots (`RetrievalContext(customer_id=slots.customer_id, ..., technology_name=slots.technology_name)`) — a pure mapping, no new ranking logic. The only real extension: `RetrievalContext` currently has `region_id: None` hardcoded at its one call site (`_resolve_retrieval_context`, since `InvestigationSession` has no region field) — Chat's `ConversationSlots.region_id` is real data the ranker can already consume once populated; this is a one-line change to stop hardcoding `None`, not new ranking logic. **EXTEND** (trivial).

---

## 14. Retrieval Orchestration

Already fully centralized in `RecommendationEngine._build_strategy()` (§5). Chat's orchestrator calls this exactly as the Investigation Workspace does (via the synthesized-session approach, §10). No new orchestration logic — **REUSE**. The only new orchestration-adjacent responsibility is *when* to call it: Query Understanding's missing-context check (§12) can short-circuit before retrieval ever runs, which is itself new control flow in `ChatOrchestrator`, not in `RecommendationEngine`.

---

## 15. TFS/Wiki Integration — Bulk Import vs. Live Query

Evaluated against every dimension requested:

| Dimension | Bulk import | Live query (current architecture) |
|---|---|---|
| Freshness | Stale between re-imports; a fix landed in TFS an hour ago is invisible until the next sync | Always current — confirmed live-verified this session against real TFS-2051535/2462358/304438 |
| Scale | TFS alone has **80,000+** Bug work items in the Command Center project (documented constraint from the original design assessment, not re-verified this pass but structurally unchanged) — bulk-importing and re-embedding all of it was already evaluated and rejected in the prior design phase | Bounded: one Analyze/chat turn issues one scoped WIQL query + a capped detail fetch (`external_knowledge_max_results=8`) |
| Performance | Local query is fast (ChromaDB, in-process) | Live query is genuinely slow — measured 8-20s end-to-end against the real instance (`config.py`'s own documented measurement) even after connection-reuse optimization; this is the single biggest latency cost in the whole pipeline today |
| Security | A local copy of 80k+ TFS work items (some containing customer-identifying detail) becomes a second system to secure, audit, and keep in sync with TFS's own access controls | TFS's own auth (NTLM/SSPI) is the access boundary; ResolveIQ never becomes a shadow copy of data whose access TFS itself governs |
| Applicability | A bulk-imported copy still needs the same Customer/Region/Technology classification pass as local Documentation — doesn't remove that cost, just moves 80k rows through it | N/A — live results are ranked at query time (`ranking.py`), no separate classification pass needed for TFS's own content |
| Traceability | A copy's `url` still needs to point back to the real TFS record — no benefit over live | `TfsCase.url` is already the deep link, always current |
| Resolution accuracy | Only as fresh as the last sync | Always reflects the real current state (`state`, `resolution_text`) |
| Maintenance | A sync job, conflict handling, dedup against re-imports, embedding drift | Already built, already working: connector Protocol + cache + timeout + graceful degradation |
| Offline/on-prem | A local copy works with TFS unreachable — the one real advantage of bulk import | Fails closed to "TFS unavailable" (already handled gracefully — `available=False`, never blocks the rest of the answer) |
| Data duplication | Real, ongoing duplication of a system TFS already owns | None |

**Recommendation: keep live query as the primary architecture, unchanged from the prior design phase's conclusion.** The one dimension bulk import wins on — offline availability — does not currently matter (no stated on-prem/air-gapped requirement) and is outweighed by every other dimension, especially scale (80k+ items) and the fact that live query is already built, tested, and working. If an offline requirement emerges later, the correct mitigation is a *narrow, applicability-scoped* cache of specifically the TFS cases already surfaced in real ResolveIQ investigations (a natural extension of the existing 15-minute `TtlCache`, not a bulk import), not full replication.

**The one real gap**: latency. 8-20s per External Knowledge call is tolerable for a one-shot Investigation Strategy generation (the engineer is reading a full page of results) but is a poor fit for a conversational turn-by-turn chat UI where the user expects a fast reply. §34 addresses this directly (streaming/progressive rendering: answer with local KB results immediately, TFS/Wiki results arrive and update the same turn a few seconds later — the exact pattern `ExternalKnowledgeResult.available`/`from_cache` already supports at the data-model level).

**Wiki-specific**: still not live-verified against the real Confluence instance (needs a PAT). This blocks Wiki-sourced chat answers specifically, not the rest of this design.

---

## 16. Release Notes Integration

No domain model exists. Proposed (unchanged in substance from the prior design assessment, restated here for completeness — blocked on you providing real files):

```python
class ReleaseNote(GovernanceFields):
    id: str
    product_id: str | None
    version_id: str | None
    title: str
    release_date: datetime
    entries: list[ReleaseNoteEntry]

class ReleaseNoteEntry(BaseModel):
    category: Literal["fix","known_issue","behavior_change","new_feature"]
    component_id: str | None
    technology_id: str | None
    description: str                # verbatim
    related_bug_id: str | None      # FK to known_bugs, when a bug number is cited
```

**How the example question resolves**: *"Could this issue be caused by something introduced in CC 9.0.3?"* — `ReleaseNoteEntry.related_bug_id` gives a direct join to `known_bugs`; `component_id`/`technology_id` let it participate in the exact same `ApplicabilityRanker`/`KnowledgeRelationship` machinery every other knowledge type already uses (`ReleaseNote` as a twelfth `KnowledgeObjectType`, REUSE the whole framework). Query Understanding's `RELEASE_NOTE_LOOKUP` intent (§12) filters by the version mentioned in the question (a real, governed `Version` row) and returns entries plus any linked Known Bug/TFS/Historical Investigation — evidence-backed by construction, since every field traces to a real imported document. **NEW domain model, REUSE everything else** (Knowledge Object Framework, `KnowledgeRelationship`, `ApplicabilityRanker`).

---

## 17. Solution Recommendation Engine

**REUSE, near-total.** `_synthesize_recommendation()` + `ResolutionProvenance` already do exactly what §8 (of the user's original request numbering — "what should I do to fix this") asks for: a distinguished, never-fabricated `RecommendedSolution` with a real tier. The one required extension: wire Known Bugs into `_synthesize_recommendation` as a fourth candidate source alongside local/TFS/Wiki (§6's gap #1) — same shape as the existing three (`best_local`/`best_tfs`/`best_wiki` gets a `best_known_bug`), same cross-source-correlation opportunity (a Known Bug's own tracked TFS/ticket reference, if it has one, could independently corroborate a local match exactly like TFS does today). **EXTEND**, scoped and small.

---

## 18. Evidence & Provenance

**REUSE, wholesale.** `ProvenanceRecord`/`EvidenceReference` (§6) already answer every one of the 8 "why" questions this design's predecessor was built to answer, generically across source type. Chat's job is to *render* this structure conversationally (§10's `ChatResponse.strategy.provenance`), never to recompute or duplicate it.

---

## 19. Confidence Model

**REUSE, exactly as-is**, with one new consumer-facing rule made explicit for the future LLM layer (§21): the four tiers and their computation (`_resolve_provenance_tier`) do not change for Chat. The one new *behavior* this phase adds: a `PROVENANCE_EXPLANATION` chat intent (§12) — "was that fix confirmed?" — that answers directly from `ProvenanceRecord.provenance_rationale`, which already exists and is already human-readable by design (`f'Cross-source correlation: TFS-{...} and local historical investigation "{...}" ... -- not a similarity score, two independent systems agreeing.'` — this sentence, verbatim, already answers "why do you trust this," it just has no conversational entry point pointing at it today). **NEW intent wiring, REUSE the underlying explanation text.**

---

## 20. Human Verification

**EXTEND** — the acknowledged gap from §6. Needed: (1) an API action, `POST /admin/historical-investigations/{id}/verify` (and the Known Bug equivalent), setting `resolution_verified=True`/`resolution_verified_by`/`resolution_verified_at`/`resolution_verification_note` via the existing repository save path (`SqlAlchemyKnowledgeRepository.save_historical_investigation`, unchanged); (2) a small UI affordance (a "Mark resolution verified" button on the Historical Investigation/Known Bug detail view, with a required note field — same pattern as the classification review queue's accept/reject action). **Access control open question, carried over unresolved from the Resolution Provenance design**: there is no role/permission system in ResolveIQ at all yet. Recommendation unchanged from before: any authenticated admin-API caller can verify, same as every other admin action today, but this is explicitly flagged as a step up in consequence (a `CONFIRMED` tier becomes visible to every future user/chat answer on that record) worth a real decision before building.

---

## 21. LLM/Copilot Gateway

**REUSE the design already produced** (`RESOLVEIQ_LLM_GATEWAY_DESIGN.md`, still unapproved-for-implementation, unchanged in substance). The one material update this inspection surfaces: that design's `GroundedContextPackage`/`EvidenceItem` shape was deliberately sketched to match what `ProvenanceRecord`/`EvidenceReference` would eventually look like — they now exist for real, so the Gateway's context-package builder becomes a thin, almost mechanical projection (`ProvenanceRecord.root_cause_evidence + .resolution_evidence + ... -> list[EvidenceItem]`) rather than a new evidence-selection algorithm. Nothing else in that design changes: the LLM still never retrieves independently, still never decides applicability/customer/technology, still never upgrades a tier, still only receives the bounded, capped context package plus the question. **This document does not re-litigate that design — it confirms Resolution Provenance (a prerequisite that design explicitly called out, §8 of that document) is now real, closing the one open dependency it had.**

---

## 22. Security & Data Boundary

**REUSE the analysis already produced** in the LLM Gateway design (§7 of that document) verbatim — it was already thorough and grounded, and nothing in this inspection changes its conclusions. Restated briefly for this document's completeness: what would leave the network (a bounded, capped context package + the question, ≤~6,000 characters, never raw DB rows/embeddings/credentials), what never leaves (SQLite, ChromaDB, TFS/Wiki credentials, full document text), the specific new risks (sensitive content in a snippet, provider data retention, prompt injection via retrieved evidence) and mitigations already proposed (secret/PII pattern scan before the network call, citation verification, audit logging). **Explicitly unconfirmed, still, and this document repeats the same caveat rather than assuming it away**: your organization's actual Azure/Copilot contractual data-retention and training-usage terms — that is a procurement/legal fact this document cannot determine, only flag as a hard precondition before any LLM Gateway implementation.

---

## 23. RAG vs. Fine-Tuning vs. Structured Knowledge + Retrieval + LLM Synthesis

**Recommendation: structured knowledge + retrieval + LLM synthesis (what this whole document already describes) — not fine-tuning, not undifferentiated RAG.** Verified against this specific system's actual constraints, not asserted generically:

- **Fine-tuning is wrong for this domain, concretely**: (1) the corpus changes shape constantly (new investigations daily, real-time TFS state, a review queue that's still 63% unworked) — a fine-tuned model needs re-training to reflect any of that, while retrieval reflects it instantly; (2) applicability (customer/region/technology scoping) is a *per-query filter*, not a fact a model can bake in at training time — a model fine-tuned on TEPCO's RF Mesh IP cases would need to somehow "know" not to apply that knowledge to a CLECO RF Mesh question, which is exactly the class of conflation error the team demo already exposed; a filter enforced in code (as `ApplicabilityRanker` does today) is the correct place for that boundary, not model weights; (3) traceable citation — every requirement in this document (provenance, evidence references, "why was this recommended") — is structurally impossible to get reliably from a fine-tuned model's generated text, which has no mechanism to point back to which specific training example produced which specific claim; retrieval-then-cite does this natively; (4) this system's real corpus (728 historical investigations, 496 documents) is far below the volume where fine-tuning would even plausibly outperform retrieval for a knowledge-recall task, and would still need the same classification/applicability work done first regardless.
- **Plain/undifferentiated RAG (embed everything, retrieve by similarity, hand it to an LLM) is also wrong, and is specifically what the team demo already proved fails**: it is architecturally identical to what local semantic search did *before* `ApplicabilityRanker` existed — no applicability filtering, no provenance tiering, no hierarchy-aware technology matching. Adding an LLM on top of that retrieval would generate fluent, confident-sounding text about *wrong* evidence, which is a worse failure mode than the current system's honest "insufficient evidence" — a plausible-sounding wrong answer is more dangerous than a correctly-flagged unknown.
- **What this document calls "structured knowledge + retrieval + LLM synthesis"** is precisely: retrieval that's already applicability-aware and hierarchy-aware (built), evidence that's already structured and provenance-tiered (built), and an LLM that only ever explains/synthesizes natural language *from* that pre-filtered, pre-tiered package (designed, not built) — never a second retrieval pass, never the arbiter of confidence. This is the only one of the three options that is consistent with the explicit, repeated constraint across this whole project: ResolveIQ remains the source of truth for retrieval, applicability, evidence, and confidence.

---

## 24. Knowledge Lifecycle

**REUSE, largely already built.** `ObjectLifecycleStatus` (Draft/Under Review/Published/Archived/Deprecated) already applies to every governed object type via `GovernanceFields` (`app/domain/enums.py`) — only Documentation's publish gate is actually enforced today (only `PUBLISHED` documents are indexed), the other object types carry the field but nothing transitions them yet. `EntityVersion` (§2) already gives every governed object edit history for free. **What's missing, concretely NEW**: nothing transitions anything to `DEPRECATED`/`ARCHIVED` automatically — "stale knowledge" detection (§13 of the user's request) doesn't exist as a mechanism. Proposed, small: a `last_confirmed_relevant_at` field (or simply: age since `updated_at` past a configurable threshold, e.g. 18 months with zero re-citation in any recent `ProvenanceRecord`) surfaced as a Knowledge Health signal (`KnowledgeHealthReport` already exists and already reports `unused_documents` — extending it with `stale_documents` is the same pattern, not a new report mechanism). **EXTEND `KnowledgeHealthReport`, NEW staleness rule.**

---

## 25. Versioning / Superseded Knowledge

`EntityVersion` (REUSE) already answers "how has this record changed over time." What's missing is the *cross-object* relationship the user's example describes ("Wiki says Fix A; a later release note says Fix A is obsolete, use Fix B") — this is not a version of the *same* record, it's two different records where one supersedes the other. **NEW**: add `SUPERSEDES` to `RelationshipType` (`app/domain/knowledge_relationships.py` — a one-line enum addition, same pattern as `APPLIES_TO`/`IMPLEMENTS_LOGGING_FOR` before it), populated (a) automatically where a `ReleaseNoteEntry.related_bug_id` and an existing Known Bug's workaround overlap in a detectable way (a deterministic rule, not guessed), and more commonly (b) by an admin action, same discipline as `resolution_verified` (§20) — marking one document/bug/resolution as explicitly superseded by another is a human judgment call, not something to infer from text similarity.

---

## 26. Conflict Resolution

**NEW, small, rule-based** (not a new subsystem): when `StructuredResolution` (§9) or a chat answer would otherwise surface two resolutions that materially disagree, the ordering rule is:

1. A `SUPERSEDES` edge, if one exists — the newer record always wins, older one still shown but explicitly labeled "superseded by X" rather than hidden (auditability, same discipline as everywhere else in this codebase — nothing is silently dropped).
2. Absent a `SUPERSEDES` edge, higher `ResolutionProvenance` tier wins as the *primary* answer; the lower-tier one is still shown as "also seen" rather than discarded.
3. Absent a tier difference, more recent `updated_at`/`imported_at` wins as primary, same "still shown, not hidden" rule.

This is a deterministic ranking function over already-existing fields (`SUPERSEDES` edges from §25, `ResolutionProvenance` from §6/§18, timestamps already on every `GovernanceFields` record) — no new data collection required, only a new small ordering function in the `StructuredResolution` assembly step (§9).

---

## 27. Example User Conversations

Grounded in real data actually in the corpus today (verified live, §3/§8's prior-session run), not invented:

**Conversation A — the user's own example, using a real close analog from the live corpus:**
> **User**: Commands are stuck in Pending. What is the likely cause?
> **ResolveIQ**: I found a real local match — "CC - CAMEPWCUMFLEP01_Emerge_App_flec - Windows NCPA Check MSMQ Queues Status - Queues are stuck" — and a TFS case, TFS-2051535 ("Unable to send commands; CommandProcessorHost could not be activated"), already resolved. The matched component is **CommandProcessorHost**.
> Confidence: **POSSIBLE** — real evidence exists, but no single source is strong enough yet to call this Likely or Confirmed (no corroboration between the two).
> Recommended SQL: 3 templates linked to CommandProcessorHost (Command lifecycle lookup, Command log lookup, Commands stuck in Pending/Sent lookup) — each with a stated reason, not just a static description.
> **User**: Has this happened before?
> **ResolveIQ**: Yes — the local match above, and TFS-2051535 (Closed). *(`last_referenced_investigation_id`/`last_referenced_tfs_id` now set.)*
> **User**: Was that fix confirmed?
> **ResolveIQ**: No — TFS-2051535 is Closed with a resolution on file, but there's no cross-source correlation or human verification linking it to *your* specific case, so this stays at POSSIBLE, not Confirmed. *(Directly answers from `ProvenanceRecord.provenance_rationale` — §19.)*
> **User**: Does this apply to TEPCO?
> **ResolveIQ**: I don't have a TEPCO-tagged record for this specific case. TEPCO is a known customer in ResolveIQ, but neither the local match nor TFS-2051535 is tagged to any customer — I can't confirm or rule out applicability here.

**Conversation B — the disambiguation mechanism (§12), using the real RF Mesh/RF Mesh IP hierarchy:**
> **User**: Meter communication is failing. What should I do?
> **ResolveIQ**: I need one detail before narrowing this down: is this **RF Mesh** or **RF Mesh IP**? (I found technologies matching both, and they have different log collection guidance.)
> **User**: RF Mesh IP.
> **ResolveIQ**: *(re-runs retrieval with `technology_name="RF Mesh IP"` now resolved unambiguously; proceeds normally.)*

**Conversation C — a known-bug question, exposing the real current gap (§6/§17):**
> **User**: Is there a known bug for this?
> **ResolveIQ**: *(today: known bugs ARE searched and shown as a `known_bugs` list on `InvestigationStrategy`, but never contribute to `ProvenanceRecord`/`RecommendedSolution` even when one is a strong match — this conversation is exactly why §17's extension is scoped as a real requirement, not speculative.)*

---

## 28. Example "Problem → Cause → Fix → Validation" Flow

Using the real, live-verified `LIKELY`-tier example already produced this session (from the completed Resolution Provenance work — not hypothetical):

```
Problem:      INC0045821 - Checkout failing with 500 for some customers
Symptoms:     (from the matched historical investigation's description)
Applicability: (not yet tagged for this record -- real gap, §8)
Root Cause:   The inventory-service was under-provisioned and its P99
              latency exceeded the orders API's HTTP client timeout,
              causing the orders API to fail the whole request.
Evidence:     hist-008 "Intermittent 500 errors on /api/v1/orders due to
              downstream timeout" (77% similarity, real recorded root
              cause) + an IllegalStateException entity heuristic.
Resolution:   (quoted from hist-008's own resolution field)
Validation:   NOT CAPTURED TODAY -- the real gap this document's §9
              identifies (RecommendedSolution has no validation-steps
              field yet).
Confidence:   LIKELY -- "single local historical match with a recorded
              root cause -- no independent corroborating source."
```

This is not a mockup — it is the actual output of the actual, already-implemented pipeline against real data, minus the one genuinely missing field (Validation).

---

## 29. API/Interface Proposal

New router, `app/api/routers/chat.py` (REUSE the thin-router discipline every other router already follows):

```
POST   /chat/sessions                          -> ChatSession (optionally investigation_id)
POST   /chat/sessions/{id}/messages             -> ChatMessage (assistant turn, includes ChatResponse)
GET    /chat/sessions/{id}/messages             -> list[ChatMessage]  (history)
GET    /chat/sessions/{id}                      -> ChatSession
POST   /admin/historical-investigations/{id}/verify   -> HistoricalInvestigationRecord (§20)
POST   /admin/known-bugs/{id}/verify                  -> KnownBugRecord (§20)
```

No changes to any existing endpoint. `get_chat_engine()` follows the exact `get_recommendation_engine()`-style wiring in `dependencies.py` — **NEW router, REUSE wiring pattern.**

---

## 30. UI Proposal

**Replace the body of `ui/views/7_AI_Assistant.py`, keep its nav slot** (currently 35 lines, static Recommendation display only — confirmed by direct inspection this pass). New chat UI: a message list (user/assistant turns), a text input, and — critically — every assistant turn renders its real `strategy`/`provenance` inline (collapsible "Evidence" / "Why this answer" sections, reusing whatever existing Investigation Workspace components already render `KnowledgeMatch`/`RecommendedSolution`/`ExternalMatch` — these Streamlit components already exist and format this exact data today; Chat should call them, not reimplement rendering). A visible Confirmed/Likely/Possible/Unknown badge (open question #2 from the original Resolution Provenance design, now answered: yes, build it, since Chat is precisely the surface where an engineer needs that signal at a glance without opening a detail view). **REUSE existing render components, NEW page body.**

---

## 31. Testing Strategy

Same discipline as every phase this session: fakes for engine-level tests (no real ChromaDB/SQLite needed for `QueryUnderstanding`/`ChatOrchestrator` logic), real-temp-SQLite integration tests for the two new tables (`chat_sessions`/`chat_messages`), following `test_context_dimensions.py`'s established `session_factory` fixture pattern exactly.

- `tests/test_query_understanding.py` — intent classification golden set (one test per intent pattern); slot extraction against real governed Customer/Technology/Component names (reusing this session's real fixture data — TEPCO, RF Mesh/RF Mesh IP); the disambiguation-triggers-clarifying-question case (direct regression test mirroring the RF-Mesh-IP defect this whole session was built around).
- `tests/test_conversation_state.py` — multi-turn fixed transcripts (`ConversationSlots` accumulation across turns; `last_focus`/`last_referenced_investigation_id` pronoun resolution — literally the four-turn example in §1 of the user's request, as an executable test).
- `tests/test_chat_orchestrator.py` — end-to-end with fakes: confirms Chat's `ChatResponse.strategy` is bit-for-bit the same shape `RecommendationEngine.generate()` already produces for an equivalent `InvestigationSession` — the direct, automatable proof that Chat never forks retrieval/reasoning logic (the user's own binding constraint from the prior phase).
- `tests/test_structured_resolution.py` — the §9 read-model assembly, including the Conflict Resolution ordering rule (§26) with two deliberately-conflicting fixture records.

---

## 32. Evaluation Strategy

Formalize what this session already did manually (the 8-case Final Knowledge-Quality Acceptance Test, the 10-investigation live-verification pass) into a **repeatable eval harness**, `scripts/eval_chat.py` (or a pytest-marked "slow" suite, run on demand, not in CI by default given TFS/Wiki live-call latency):

- A fixed set of real questions (the user's own §1 list is a ready-made starting set) run against the real corpus.
- Per question, assert: an intent was recognized (not silently defaulted), no fabricated Customer/Technology appears in the response, `ResolutionProvenance` is never `CONFIRMED` unless a real correlated/verified source backs it (the exact invariant §19/§6 already test at the unit level — this asserts it holds end-to-end, live), and the RF Mesh vs RF Mesh IP / TEPCO-as-customer invariants specifically (regression coverage for the exact defects that motivated this whole design).
- Track results over time as the classification backlog (§8) shrinks and Release Notes (§16) come online — the harness should show *measurable* improvement in "did it find real applicable evidence," not just "did it not crash."

---

## 33. Failure Modes

| Failure | Current handling | Gap |
|---|---|---|
| TFS/Wiki unreachable | `available=False`, graceful, already tested | None — reuse as-is |
| Ambiguous technology/customer | `_match_single_technology` returns None; Chat should turn this into a clarifying question (§12) | NEW: the clarifying-question wrapper itself |
| Zero evidence at all | `insufficient_evidence=True`, `UNKNOWN` tier, honest | None — reuse as-is |
| Contradicting evidence (two resolutions disagree) | Not handled today (no conflict concept exists) | NEW — §26 |
| A chat question tries to make the system invent a resolution ("just tell me the fix") | N/A — no LLM yet to be pressured; deterministic templates cannot be talked into fabricating | Becomes relevant only once §21 (LLM Gateway) exists — that design's response validation (§4 of that document) is the mitigation, already designed |
| Sensitive content (a real secret/PII) surfaces in a chat answer's evidence snippet | Same risk as any retrieval today — this session already found and excluded one real credential file at import time by hand | Not chat-specific; a real, standing gap in ingestion validation, out of this document's scope but worth flagging again |
| Conversation drifts across unrelated topics in one session | `ConversationSlots`' last-write-wins model could carry stale context forward incorrectly (e.g. customer from turn 1 wrongly applied to an unrelated turn 5 question) | NEW: a simple heuristic — if a new turn's extracted signals conflict with (not merely omit) a stored slot, treat it as a topic change and clear stale slots rather than silently merging; needs explicit test coverage (§31) |

---

## 34. Performance Considerations

- **Local retrieval + applicability**: fast today (ChromaDB in-process, dozens-to-low-hundreds-of-rows relationship graph walks) — no expected regression from Chat's added load at this corpus scale.
- **TFS/Wiki live latency (8-20s)** is the dominant cost and the one real UX risk for a conversational surface (§15). Mitigation, in order of preference: (1) progressive/streaming response — render the local-KB-derived answer immediately, patch in TFS/Wiki results a few seconds later in the same chat turn (the data model already separates these — `ExternalKnowledgeResult.available`/`from_cache` — this is a rendering/API-streaming change, not a data model change); (2) the existing 15-minute TTL cache already helps repeat/similar questions within a session; (3) for chat specifically, consider skipping the live TFS/Wiki call entirely for intents where it adds little (e.g. `LOG_GUIDANCE`, `SQL_GUIDANCE` — these never consult TFS/Wiki today regardless) rather than paying the latency on every single turn.
- **Multi-turn conversation state** is cheap (a handful of small rows per turn) — no concern.
- **The classification backlog (§8, 409 pending)**: not a runtime performance concern, but worth stating plainly — reviewing it is manual admin work, not something this architecture can make faster; it's a process/staffing dependency for how *good* Chat's applicability answers are, independent of code.

---

## 35. Implementation Phases

Ordered by dependency; "Scope" is relative size/judgment-weight, not calendar time.

| Phase | Scope | What ships | Depends on |
|---|---|---|---|
| **0. Close real gaps in what already exists** | S | Known Bugs wired into `_synthesize_recommendation`/provenance (§17); classification engine's scan target extended to Historical Investigations/Known Bugs (§9's correction); `resolution_verified` admin action + minimal UI (§20) | none |
| **1. Structured Resolution Knowledge read model** | M | `StructuredResolution` assembly (§9), `ValidationStep` field on `RecommendedSolution`, `SUPERSEDES` relationship type (§25), Conflict Resolution ordering (§26) | Phase 0 |
| **2. Query Understanding** | L (judgment-heavy) | Intent classification, slot extraction (factored out of/shared with `classification.py`/`_match_single_technology`), disambiguation → clarifying-question mechanism (§12) | Phase 1 (needs `StructuredResolution` to cite) |
| **3. Conversation State + Chat persistence** | M | `chat_sessions`/`chat_messages` tables + repository, `ConversationSlots` accumulation, pronoun/reference resolution (§11) | Phase 2 |
| **4. Chat Orchestrator + API + UI (V1, no LLM)** | L | Deterministic-template chat answers wired to the real `RecommendationEngine`/`ProvenanceRecord` pipeline via the synthesized-session approach (§10); new router (§29); replace `7_AI_Assistant.py` body (§30) | Phase 3 |
| **5. Classification backlog clearance** | Ongoing, process not code | Work down the 409-item pending queue (or build a faster review UI if that's the actual bottleneck — needs your input on which) | Can run in parallel with Phases 1–4; improves their real-world answer quality but doesn't block shipping them |
| **6. Release Notes** | M | Domain model (§16) + import path, once real files are provided | Blocked on you |
| **7. LLM/Copilot Gateway** | XL | The already-designed `GroundedContextPackage`/`AnswerEngine`/`LLMProvider` (§21), now consuming real `ProvenanceRecord`/`StructuredResolution` | Phases 1–4 (needs a real Chat surface and real structured evidence to wrap); explicit security/procurement sign-off (§22) |

**Explicitly not scheduled without further input**: bulk TFS/Wiki import (§15 — recommend against it structurally, not just "not yet"); a role/permission system (§20's open question — needed before Human Verification can be more than "any admin caller"); Environment as a governed dimension (no decision calls for it yet, same as the prior phase's conclusion).

---

## 36. REUSE / EXTEND / NEW Matrix

| Component | Disposition |
|---|---|
| `RecommendationEngine` | REUSE (Chat calls it via synthesized session, §10) |
| `InvestigationStrategy` | REUSE |
| `ResolutionProvenance` / `ProvenanceRecord` / `EvidenceReference` | REUSE |
| `ApplicabilityRanker` | REUSE (EXTEND: stop hardcoding `region_id=None`, §13) |
| `KnowledgeRelationship` / `KnowledgeRelationshipEngine` | REUSE |
| `Technology.parent_technology_id` + `most_specific()` | REUSE |
| `DocumentClassificationEngine` | EXTEND (scan target: add Historical Investigations/Known Bugs, §9) |
| TFS connector | REUSE (unchanged architecture, §15) |
| Wiki connector | REUSE (still needs live PAT verification — unchanged blocker) |
| `_synthesize_recommendation` | EXTEND (add Known Bugs as a fourth source, §17) |
| `RecommendedSolution` | EXTEND (`validation_steps` field, §9) |
| `RelationshipType` enum | EXTEND (`SUPERSEDES`, §25) |
| `KnowledgeHealthReport` | EXTEND (`stale_documents`, §24) |
| SQL Library, Product Intelligence, Log Intelligence KB | REUSE, unchanged |
| `StructuredResolution` (read model) | NEW (§9) |
| Query Understanding | NEW (§12) |
| Conversation State (`ChatSession`/`ChatMessage`/`ConversationSlots`) | NEW (§11) |
| Chat Orchestrator | NEW (§10) |
| Chat API router | NEW (§29) |
| Chat UI | NEW body, REUSE nav slot + existing render components (§30) |
| Human Verification action/UI | NEW (§20) |
| Release Notes domain model + import | NEW, blocked on real files (§16) |
| LLM Gateway (`GroundedContextPackage`/`AnswerEngine`/`LLMProvider`) | NEW, already designed, unapproved (§21) |
| Conflict Resolution ordering | NEW, small (§26) |
| Eval harness | NEW (§32) |

---

## 37. Risks

1. **Classification backlog is the real ceiling on answer quality**, not code. Chat can be architecturally perfect and still answer "I don't have a customer tag for this" most of the time until the 409-item queue is worked. This is a process risk this document can flag but not resolve.
2. **Synthesized-session approach (§10) could leak throwaway state** if a future change accidentally persists it — needs an explicit test asserting a chat-only question never creates a row in `investigations`.
3. **TFS/Wiki latency (§34)** is a real, measured UX risk for a conversational surface; the mitigation (streaming) is a genuine engineering task, not a formality.
4. **Human Verification's access-control gap (§20)** — shipping it with "any admin caller" is a real, if bounded, risk (a `CONFIRMED` tier is a strong trust signal); worth a deliberate decision, not a default.
5. **LLM Gateway data-retention confirmation (§22)** remains a hard precondition this document cannot close — a real risk if Phase 7 proceeds without it.
6. **Query Understanding's intent/slot rules will need real usage data to tune** — the patterns in §12 are a reasonable first cut grounded in the user's own example questions, not validated against real engineer phrasing at volume yet; expect iteration after V1 ships, same as every other pattern-matching rule in this codebase has needed real-data tuning (the RF Mesh/generic-word fixes this whole session are the precedent).

---

## 38. Open Questions

1. Should the classification review-queue backlog (409 items) be worked manually, or is a faster bulk-review UI (e.g. "accept all Medium-confidence Technology suggestions above X mentions") worth building first? Affects how much real applicability data Chat has to work with at launch.
2. Access control for Human Verification (§20, §37.4) — any authenticated admin caller, or does this need to be the first feature that requires a real permission check?
3. Standalone-question chat vs. investigation-scoped-only for V1 — this design supports both identically (§10), but do you want V1 scoped to *inside an open investigation* only, with a bare "ask anything" chat as a deliberate V2?
4. Confirm the Confirmed/Likely/Possible/Unknown badge (§30) is wanted directly in the Chat UI, not just in an expandable detail panel.
5. Same three LLM Gateway open questions carried over unchanged from that design (§8 of `RESOLVEIQ_LLM_GATEWAY_DESIGN.md`): which specific Azure/Copilot product, confirmed data-retention terms, and context-package size budget.
6. Release Notes: still waiting on real files.
7. Confluence Wiki PAT: still needed for live verification.

---

## 39. Recommended Next Steps

1. Review this document; approve, adjust, or reject the Phase 0–7 sequencing (§35).
2. If approved, I'd recommend starting with **Phase 0** specifically (closing the Known Bug provenance gap and the classification scan-target gap) — both are small, deterministic, directly improve the existing Investigation Workspace (not chat-dependent), and remove two real caveats from this very document before anything conversational is built on top of them.
3. Decide the two access-control-shaped open questions (§38.2, §38.3) before Phase 3/4 — they affect the shape of the Chat session/message model and the verification endpoint, not just later polish.
4. No code will be written until you confirm.

---

*No application code has been modified, no database has been changed, and no dependencies have been added to produce this document. Waiting for your review and approval before any implementation begins.*
