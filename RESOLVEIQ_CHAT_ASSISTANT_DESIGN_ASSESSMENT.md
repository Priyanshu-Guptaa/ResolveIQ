# ResolveIQ → Engineering Knowledge Assistant: Design Assessment

**Status: Revision 2 — architecture direction approved 2026-08-12. Decisions below are binding. No implementation yet — still awaiting sign-off on this revision (data model, retrieval flow, phased scope) before any code is written.**

Every claim in Part A was verified against the actual repository (file/line references given) or the running database on 2026-08-12, not assumed. Part B onward incorporates the user's approved decisions and adds the concrete data model, retrieval flow, reuse plan, and phased scope requested for this revision.

---

# Part A — Current-State Assessment (unchanged from Revision 1, still accurate)

## 1. Current ResolveIQ architecture relevant to this goal

ResolveIQ is a FastAPI backend (`app/`) + Streamlit UI (`ui/`) over a local SQLite database (`data/resolveiq.db`) and a local ChromaDB vector store (`data/chroma/`). There is no LLM anywhere in the runtime — confirmed by grep across `app/` for any LLM/API-provider reference; the only hits are comments stating "no LLM reasoning" as a design constraint. Everything a user sees is either a database row, a deterministic keyword/regex match, or a cosine-similarity vector search — never generated text.

Layering is consistent throughout:
- `app/domain/*` — pydantic models, framework-agnostic.
- `app/engines/*` — business logic, one package per capability (`recommendation`, `log_intelligence`, `product_intelligence`, `knowledge`, `external_knowledge`, `knowledge_object_framework`, ...).
- `app/infrastructure/db/*` and `app/infrastructure/vectorstore/*` — SQLAlchemy repositories and the ChromaDB client.
- `app/api/routers/*` — thin FastAPI routers.
- `ui/views/*` — Streamlit pages, one per module, talking to the API only via `ui/api_client.py`.

This is a genuinely reusable layering for a Chat Assistant: a chat layer can sit beside `app/api/routers`, call the exact same engines, and never touch SQL or Chroma directly.

## 2. Current knowledge sources

Verified against `data/resolveiq.db`:

| Source | Table | Row count | How it got there |
|---|---|---|---|
| Documentation (wiki exports, guides, release notes, tickets) | `documentation` | 496 | Admin upload / bulk import script (474 this session) |
| Historical Investigations | `historical_investigations` | 728 | ServiceNow task.xlsx import |
| Known Bugs | `known_bugs` | 4 | Original Sprint-1 seed data |
| Product Intelligence Components | `component_profiles` | 11 | 6 fictional Sprint-1 samples + 5 real, added this session |
| SQL Library templates | `sql_templates` | 12 | 5 seed + 7 added this session |
| Log Source Applications | `log_source_applications` | 85 | One wiki page import, frozen v1 |
| Log Collection Scenarios | `log_collection_scenarios` | 40 | Same one wiki import |
| Playbooks | `playbooks` | 0 | Not populated |
| Products (governed lookup) | `products` | 1 | Schema exists, minimally used |
| Technologies (governed lookup) | `technologies` | 0 | Schema exists, **empty** |
| Versions (governed lookup) | `versions` | 0 | Schema exists, **empty** |
| Knowledge relationships (graph edges) | `knowledge_relationships` | 24 | IMPLEMENTS_LOGGING_FOR only |
| TFS Bugs/Issues | *(not stored)* | n/a | Live query only |
| Confluence Wiki | *(not stored)* | n/a | Live query only, unverified against the real instance |

Release Notes have no representation anywhere yet — no table, no domain model, no import path, and none provided yet.

## 3. Current knowledge model — what exists and what's missing

Governed entity types today (`app/domain/knowledge_relationships.py:32`, `KnowledgeObjectType`): Component, Document, Known Bug, SQL Template, Historical Investigation, Playbook, Technology, Product, Version, Log Source Application, Log Collection Scenario. No Customer, Region, Environment, Protocol, Communication Method, Symptom, or Issue Type exists anywhere.

- `InvestigationSession` (`app/domain/investigation.py:64-68`) has five manually-typed, unvalidated free-text fields: `customer`, `product`, `version`, `technology`, `assigned_engineer` — almost always blank.
- `DocumentationRecord` (`app/domain/evidence.py:110`) has `product`/`version`/`technology`/`related_components` with a real FK association table for the last one — but never auto-populated. Verified live: **0 of the 474 documents bulk-imported this session have any of these set.**
- `HistoricalInvestigationRecord` has no customer/region field at all.
- `LogSourceApplication.technology` (`app/domain/log_intelligence_kb.py:78`) is a flat `list[str]` — `["RF Mesh", "RF Mesh IP"]` are unrelated strings with no hierarchy.
- `Product`/`Technology`/`Version` governed lookup tables (Sprint 3 Phase 3.3) exist specifically to solve this and are **empty or nearly so** (`technologies`: 0, `versions`: 0, `products`: 1), and **nothing in the Recommendation Engine, ranking, or search code queries them at all** — dead schema, not a working part of retrieval.

## 4. Current retrieval architecture

**A. Local semantic search** (`app/engines/knowledge/knowledge_store.py:90-124`) — pure vector cosine similarity, top-k, zero metadata filter or boost. A query about generic Command Center RF Mesh and a query about TEPCO's RF Mesh deployment retrieve from the exact same undifferentiated pool with the exact same scoring.

**B. Live external search** (`app/engines/external_knowledge/`) — TFS/Confluence, queried live, never imported. Has real structure worth generalizing: anchor/supporting term split (`service.py:78-121`), a deterministic re-ranking pass with named fixed-weight signals (`ranking.py`) — component 0.30, technology 0.20, title overlap ≤0.25, entity match ≤0.30, resolved-state 0.10, customer 0.15 (ranking-only, never a query anchor).

## 5. Current Recommendation Engine architecture

`app/engines/recommendation/engine.py` (1049 lines) orchestrates local semantic search → component match → log recommendations → SQL suggestions → external knowledge → one synthesized `RecommendedSolution`. Deterministic throughout — no free-text generation. `_synthesize_recommendation()` (`engine.py:713-879`) already distinguishes a correlated CRM/ticket match, a similar case with no recorded root cause, and insufficient evidence — the working precursor to the Confirmed/Likely/Possible/Unknown tiering now required (§ Decision 5, Part C).

Confirmed, still-unfixed weakness: `_recommend_logs()` (`engine.py:447-494`) scores technology names with the same `keyword_match_score` that gives "RF Mesh" a full match against text that says "RF Mesh IP" (the shorter name is a prefix of the longer one). `_infer_technology()` was fixed for its own narrower purpose this session; its own docstring states the fix is *deliberately independent* of `_recommend_logs()`, which remains frozen "bug fixes only" and still has this bug live. This is Decision 6's Phase 0 (Part F).

## 6. Current Log Intelligence architecture

Two subsystems: the parsing **Engine** (mechanical entity extraction from uploaded logs, no product/technology awareness needed) and the **Knowledge Base** (`log_intelligence_kb.py`, `log_knowledge/`) — the "collect exactly these logs, in order, and why" guidance, imported once from one wiki page via a state-machine parser. `LogCollectionScenario.technology` is a bare extracted string; matching is keyword-only, with no concept of version-scoping or customer-scoping. The extractor's fallback rule (`extractor.py:318-323`, `_resolve_component`) could fold a customer-named section heading straight into a component/log-source name for any log line without an explicit folder segment — no literal "TEPCO" component exists in the DB today (checked directly), but the code path that could produce one is real and unguarded. This module is frozen "bug fixes only"; any structural change here is a deliberate, scoped unfreeze.

## 7. Current Product Intelligence architecture

`ComponentRegistry` (`component_registry.py`) is a thin in-memory index, substring-search only, "no ranking, no semantics" by its own docstring. 6 of the 11 current components are fictional Sprint-1 sample data, left untouched rather than "fixed" without real grounding, and currently indistinguishable in the UI from the 5 real ones added this session.

## 8. Current Wiki/TFS integration status

**TFS**: live, working, verified against `am.tfs.landisgyr.net` this session, NTLM/SSPI, no password/token in code. **Confluence Wiki**: implemented but explicitly **not yet live-verified** — built against the documented REST contract, not confirmed against the real instance; needs a real PAT from a Wiki admin.

## 9. Current problems that explain the poor team-demo results

- **Fails to find relevant knowledge** — no metadata pre-filter anywhere in local search; a ~256-token embedding window with nothing else compensating.
- **Mixes RF Mesh with Mesh IP** — confirmed, unfixed, in `_recommend_logs()`.
- **Mixes customer/region with product/component info** — no Customer/Region field exists anywhere in the indexed schema.
- **Treats TEPCO as a module** — no literal fake component exists, but dozens of real imported TEPCO-titled documents carry zero structured signal marking them customer-specific, competing undifferentiated with generic content.
- **Recommends logs without explaining why** — partially untrue: real deterministic explanations already exist (`RecommendedLogCollectionItem.explanation`/`match_reason`); what's missing is confirm/reject-hypothesis reasoning, which needs an Issue Type dimension that doesn't exist yet.
- **Retrieves semantically similar text without understanding applicability** — exactly the local-retrieval filtering gap.
- **Recommendations can look plausible without being grounded** — pushback: text is never invented (every field traces to a real quoted source); the real risk is correctly-quoted content from an *insufficiently qualified* match, i.e. the same retrieval-filtering gap, not fabrication.

## 10. Where the distinctions are lost

At ingestion (no metadata extraction) → at the domain-model level (no fields exist) → at indexing (only `product`/`technology`, usually blank, get copied to Chroma metadata) → at query time (metadata never read before scoring) → in free text (`LogSourceApplication.technology` has no hierarchy) → in the governed-lookup layer (Product/Technology/Version tables are unused dead schema).

---

# Part B — Approved Decisions (2026-08-12, binding)

1. **Customer/organization**: a controlled, verified list, seeded from reliable Wiki/TFS evidence. Never inferred from text patterns alone.
2. **Existing-document backfill**: automatic extraction, confidence-tiered. High → auto-accept. Medium → review queue. Low → stays unclassified, never guessed.
3. **Chat V1 scope**: troubleshooting/knowledge questions, historical investigations, known bugs, logs, SQL, documentation, version/release-note questions, and conversational clarification of missing context, with follow-up questions using current conversation/investigation context.
4. **Environment**: not a first-class dimension yet; the mechanism must make adding it later cheap.
5. **Resolution provenance**: Confirmed / Likely / Possible / Unknown. Never "Confirmed" merely because a score is high. Always preserve the supporting source.
6. **RF Mesh vs Mesh IP fix**: lands *before* Chat V1, not alongside it.
7. **No further bulk document import** until contextual metadata, applicability-aware retrieval/ranking, and source relationships are in place. 496 documents already in the system is enough to build and validate against.
8. **Investigation Workspace and Chat share one knowledge/retrieval/reasoning layer** — no forked logic.

Everything below is designed to satisfy these eight constraints directly, citing which existing file/table each piece reuses.

---

# Part C — Proposed Data Model (concrete)

A key finding while designing this revision: the exact repository/adapter pattern needed for Customer and Region **already exists and is trivially extensible** — `SqlAlchemyLookupRepository` (`app/infrastructure/db/lookup_repository.py`) already implements this identical shape for Product/Technology/Version (`save_x`/`get_x`/`get_x_by_name`/`list_x`/`delete_x`, backed by a governed table with `id`/`name`/`GovernanceFields`), and `build_adapters()` (`app/engines/knowledge_object_framework/adapters.py:70-181`) already shows the exact 10-line pattern for wiring a new lookup type into the Knowledge Object Framework (create/edit/version/history all for free). Customer and Region are **new rows in this same pattern, not a new mechanism.**

### C.1 New/extended tables

**`customers`** (new, mirrors `ProductModel` exactly) — `id`, `name`, `aliases: list[str]` (e.g. `["TEPCO", "Tepco Japan", "Tokyo Electric Power"]` — so near-duplicate spellings resolve to one governed record instead of silently becoming two), `verified: bool` (always `True` for anything actually seeded — see C.3), `source_type: str` (`"tfs_area_path" | "wiki_space" | "admin"`), `source_reference: str` (the literal Area Path / space key / admin note that justified creating this row — provenance, same discipline as `LogSourceApplication.raw_paths`), plus `GovernanceFields`.

**`regions`** (new, same shape, smaller) — `id`, `name`, `aliases`, `GovernanceFields`. Seeded from the one real precedent already in the schema: `LogCollectionScenario.region` (`"NAM"`/`"APAC"`, real extracted values, not invented).

**`technologies`** (existing table, extended) — add one nullable column: `parent_technology_id: str | None` (self-referential FK). This is the entire fix for "RF Mesh and Mesh IP are not automatically the same thing": `RF Mesh IP` row gets `parent_technology_id = <RF Mesh row id>`, expressing "is a more specific variant of," never "is the same as." A matching/ranking function can then ask "is this an exact match, a parent/child match, or unrelated?" instead of colliding on prefix overlap. No new table, no new object type — a one-column migration on a table that already exists and is already empty (so there's no backfill-migration risk).

**`context_dimension_values`** — explicitly **not** introduced as a new generic table. Considered it (one polymorphic table for Customer/Region/Protocol/CommunicationMethod/Environment) and rejected it: it would break from this codebase's own established convention of one small typed table per lookup concept (Product/Technology/Version are three near-identical tables, not one polymorphic one) for no real benefit here. Customer and Region get their own tables now; Protocol gets the identical treatment (`protocols` table, same shape) **when real protocol-distinguishing data actually requires it** — not built in this phase, since no decision above calls for it yet, but the pattern is proven and the addition is small (one migration, one repository extension, one adapter entry) whenever it's needed. This is exactly how Decision 4 (Environment "designed so it can be introduced later") is satisfied structurally: adding `environments` later is the same ~40-line pattern as adding `regions` was, not a redesign.

**Tagging content with a dimension — reusing the existing relationship graph, not inventing a parallel one.** `KnowledgeRelationship`/`KnowledgeRelationshipEngine` (`app/domain/knowledge_relationships.py`, `app/engines/knowledge_relationships/engine.py`) already provide a typed, duplicate-checked, generic edge table between any two `KnowledgeObjectType`s, and the enum already has an unused `RelationshipType.APPLIES_TO` (`knowledge_relationships.py:64`) — evidently anticipated for exactly this. Adding `KnowledgeObjectType.CUSTOMER`/`.REGION` and reusing `APPLIES_TO` means "this Document applies to Customer TEPCO" is a normal, already-implemented relationship (`Document --APPLIES_TO--> Customer(TEPCO)`), automatically visible in the existing Relationship Explorer/Impact Analysis UI, with zero new tagging infrastructure.

**`metadata_classification_suggestions`** (new — the one genuinely new table this revision requires) — implements Decision 2's confidence tiering as a review workflow, kept separate from the relationship graph so a suggestion is provisional data and a `KnowledgeRelationship` is asserted fact, matching the codebase's existing Draft-vs-Published governance discipline elsewhere:
```
id, object_type, object_id,
dimension_type          -- "customer" | "region" | "technology" | "component"
suggested_value_id       -- FK into customers/regions/technologies/component_profiles, when the value is already governed
suggested_value_text     -- the raw candidate string, always kept even when suggested_value_id is set (traceability)
confidence_tier          -- "high" | "medium" | "low"
evidence_snippet         -- the literal source text that triggered this suggestion
evidence_rule            -- which extraction rule fired (traceability/debuggability)
status                   -- "pending" | "auto_accepted" | "accepted" | "rejected"
reviewed_by, reviewed_at, created_at
```
- **High** → the corresponding `KnowledgeRelationship` (or field update) is created immediately, row logged as `auto_accepted` (auditable/reversible — deleting the relationship is enough to undo it, nothing else references this table).
- **Medium** → stays `pending`; a small review-queue UI (list + accept/reject, reusing existing `KnowledgeObjectAdapter`/Streamlit patterns) is the human gate. Accepting creates the same relationship as the high-confidence path; rejecting just marks the row `rejected` and creates nothing.
- **Low** → not written into the relationship graph at all. (Whether to even log a `low` row for future re-evaluation, or silently drop it, is one of the open items in Part G — leaning toward logging it, since it's free provenance and might matter once more corroborating evidence exists later.)

**Resolution provenance (Decision 5)** — no new table. Extends `RecommendedSolution` (`app/domain/recommendation.py:163`) with `resolution_provenance: ResolutionProvenance` (new enum: `CONFIRMED | LIKELY | POSSIBLE | UNKNOWN`), and adds two optional fields to `KnownBugRecord` (and, for symmetry, `HistoricalInvestigationRecord`): `resolution_verified_by: str | None`, `resolution_verified_at: datetime | None` — the explicit, human-set flag that lets a single-source resolution ever reach `CONFIRMED` (see the deterministic mapping rule in Part D, step 6). `supporting_tfs_id`/`supporting_wiki_title`/the correlated-match logic already on this model *are* "preserve the source" — reused as-is, not rebuilt.

**Chat (Decision 3/8)** — new `app/domain/chat.py`: `ChatSession` (`id`, `investigation_id: str | None`, `context_profile: ContextProfile`, `created_at`, `updated_at`), `ChatMessage` (`id`, `session_id`, `role`, `content`, `structured_response: ChatResponse | None`, `created_at`), `ChatIntent` enum (`TROUBLESHOOTING | HISTORICAL_LOOKUP | KNOWN_BUG_LOOKUP | LOG_GUIDANCE | SQL_GUIDANCE | DOCUMENTATION_LOOKUP | RELEASE_NOTE_LOOKUP | CLARIFICATION_NEEDED | WHY_EXPLANATION`), `ChatResponse` (`answer`, `resolution_provenance`, `evidence: list[KnowledgeMatch]`, `related_knowledge` grouped by type, `recommended_next_step`, `clarifying_question: str | None`, `missing_context: list[str]`). New tables `chat_sessions`/`chat_messages`, same repository pattern as `investigations`/`evidence`.

**`ContextProfile`** (new, shared) — replaces/extends `InvestigationSession`'s five loose strings: `customer_id`/`region_id`/`product_id`/`version_id`/`technology_id` (real FK references, resolved from free text where possible) plus `*_text` free-text fallbacks for anything not yet in a governed table, so nobody is ever blocked because a real value isn't in a dropdown yet. Used by both `InvestigationSession` and `ChatSession` — literally the mechanism behind Decision 8.

### C.2 Release Notes (unchanged from Revision 1, still a proposed schema — no source material yet)

```
ReleaseNote(GovernanceFields): id, product_id, version_id, title, release_date,
  entries: list[ReleaseNoteEntry]
ReleaseNoteEntry:
  category: "fix" | "known_issue" | "behavior_change" | "new_feature"
  component_id: str | None
  technology_id: str | None
  description: str            -- verbatim
  related_bug_id: str | None  -- FK to known_bugs, when a bug number is cited
```
Blocked on you providing real release-note files (Part G).

### C.3 How the "controlled, verified customer list" actually gets built (Decision 1)

Decision 1 forbids inferring customers from text patterns alone — so the seeding step itself must not be "run a classifier and accept its output." Concretely, phase 1's deliverable is a **candidate report, not an automatic import**: a script scans real, structural evidence only —
- TFS `System.AreaPath` values already fetched by the live connector (per-work-item, not yet examined for customer segments — needs a short live check against real TFS data to confirm the Area Path taxonomy actually encodes customer, which I have not yet verified and won't claim without checking),
- Confluence space keys/titles (once the Wiki PAT is available, per Revision 1 §8),
- recurring, capitalized tokens across historical-investigation and documentation titles (the same kind of signal that surfaced "TEPCO"/"ATCO"/"CLECO" in this session's real imported content),

and produces a de-duplicated candidate list with source evidence attached for you to confirm, edit, merge, or reject line-by-line — the human is the verification step Decision 1 requires, not a second, hidden classifier.

---

# Part D — Retrieval Flow (updated)

```
1. Build/refresh ContextProfile
   - InvestigationSession fields (now real FK references, not loose strings)
     or chat-inferred slots, or explicit engineer/chat-user input.
   - Customer: NEVER auto-filled from free text (Decision 1) -- only from an
     explicit selection or an already-governed relationship.
   - Technology: confidence-gated. An exact/full match against a governed
     Technology name is accepted; an ambiguous case (matches both a
     technology and its parent, per the new parent_technology_id hierarchy)
     triggers a clarifying question instead of a silent pick -- this is the
     Phase 0 fix (Decision 6), reused here rather than re-implemented.

2. Query text prep -- unchanged (strip_low_signal_boilerplate).

3. Vector search over-fetch -- unchanged mechanism (ChromaKnowledgeStore),
   widened from an exact top-k to a larger candidate pool (e.g. top-20) so
   step 4 has real candidates to differentiate, not just the 5 the old
   flow already committed to.

4. NEW -- applicability scoring pass, one new module
   (app/engines/knowledge/applicability.py), for each candidate:
   - Look up its real KnowledgeRelationship edges (APPLIES_TO Customer/
     Region, USES/RELATED_TO Technology, RELATED_TO Component) via the
     existing KnowledgeRelationshipEngine.list_relationships -- no new
     graph-read mechanism.
   - Same customer as ContextProfile.customer_id -> boost (reuses
     ranking.py's exact weight-and-reason-string discipline, e.g. +0.15,
     matching the customer signal already proven for TFS/Wiki).
   - Different, EXPLICITLY KNOWN customer (not merely absent) -> penalize
     and flag in match_reasons -- new; doesn't exist even in the TFS/Wiki
     ranker today, added here and back-ported there for consistency.
   - Technology: exact governed-name match -> full weight; parent/child
     match via parent_technology_id -> partial weight with an explicit
     "related but not identical" reason string (never silently collapsed
     into a full match -- this is the direct RF-Mesh/Mesh-IP fix applied
     at the retrieval layer, not just the log-collection layer).
   - No dimension known on either side -> neutral, unchanged from today
     (never fabricate applicability where there's no evidence either way).

5. Re-rank: vector similarity + applicability score, fixed-weight sum,
   same explainable-by-construction discipline as every other scoring
   function in this codebase.

6. Truncate to top_k for display. Each result now carries which context
   dimensions matched/mismatched, feeding both the Investigation
   Workspace's existing "why" fields and the Chat Assistant's "why are
   you recommending this" answers (Decision 3) from the same computed
   data -- not a second explanation generator.
```

This changes `ChromaKnowledgeStore.query()`'s signature additively (an optional `context: ContextProfile | None` parameter, default `None` = today's exact unfiltered behavior, so every existing caller keeps working unchanged until explicitly updated) and adds one new scoring module — it does not replace or fork the vector search itself.

---

# Part E — Reuse Matrix

| Existing piece | Disposition | Notes |
|---|---|---|
| `KnowledgeRelationship` + `KnowledgeRelationshipEngine` | **Reuse as-is** | Becomes the Customer/Region tagging mechanism via the already-unused `APPLIES_TO` relationship type |
| `SqlAlchemyLookupRepository` (`lookup_repository.py`) | **Extend** | Add `save_customer`/`get_customer`/... and `save_region`/... — identical shape to existing Product/Technology methods |
| `build_adapters()` (`adapters.py`) | **Extend** | Two new dict entries, ~10 lines each, same pattern as `T.PRODUCT`/`T.TECHNOLOGY` |
| `ranking.py` scoring pattern | **Port** | Same named-weight, explainable-reasons idiom, applied to local retrieval in Part D step 4 |
| `ChromaKnowledgeStore.query()` | **Extend, not replace** | New optional parameter, backward compatible |
| `ComponentRegistry`/`ComponentProfile` | **Reuse as-is** | No changes needed |
| `RecommendationEngine._synthesize_recommendation` | **Extend** | Add `ResolutionProvenance` mapping (Part D isn't the only consumer — this is engine-side) |
| `KnowledgeObjectService`/versioning/lifecycle | **Reuse as-is** | Customer/Region are new `KnowledgeObjectType` members, get history/versioning for free |
| Investigation Workspace, Analyze endpoint | **Reuse as-is** | Chat calls the same public engine methods, never forks logic (Decision 8) |
| `ui/views/7_AI_Assistant.py` | **Replace body, keep nav slot** | Currently static Recommendation display only |
| `ui/views/11_Knowledge_Objects.py` (generic object CRUD) | **Reuse as-is, likely** | Already handles arbitrary `KnowledgeObjectType`s generically — needs verification once Customer/Region adapters exist, not a rebuild |
| Log-wiki extractor `_resolve_component` | **Targeted fix** | Guard against a customer-named heading becoming a component name (Part F, Phase 0/6 boundary) |
| `_recommend_logs()` / `keyword_match_score` | **Targeted fix** | Use the new `parent_technology_id` hierarchy instead of raw prefix matching |
| The 496 documents / 728 investigations / 11 components already imported | **Reuse as data** | No re-import — Decision 7 explicitly defers further volume; this is the real corpus classification runs against |
| `_TICKET_NUMBER_RE` / CRM-correlation logic | **Reuse as-is** | Directly applicable to Release Note ↔ Known Bug correlation later |

---

# Part F — Implementation Phases and Estimated Scope

Ordered by dependency, honoring Decision 6 (RF Mesh fix before Chat) and Decision 7 (no import phase). "Scope" is relative size (S/M/L/XL) in terms of new files/tables and the judgment-heavy vs. mechanical nature of the work — not calendar time, since that depends on review/iteration cadence with you, not raw coding effort.

| Phase | Scope | What ships | New DB objects | Key files | Depends on |
|---|---|---|---|---|---|
| **0. RF Mesh / Mesh IP fix** | **S** | `parent_technology_id` hierarchy; `_recommend_logs()` and any other affected matcher stop treating a prefix as a full match | 1 column + migration | `lookup_entities.py`, `models.py`, `lookup_repository.py`, `recommendation/engine.py`, tests | none |
| **1. Context Dimensions foundation** | **M–L** | `customers`/`regions` tables + repository + adapters; `technologies`/`versions` actually populated from real content; candidate-customer report delivered to you for confirmation (Part C.3) | 2 tables + migration | `lookup_entities.py`, `models.py`, `lookup_repository.py`, `adapters.py`, `knowledge_relationships.py` (enum members), new seeding script | Phase 0 (technology hierarchy should exist first, avoids two migrations touching the same table) |
| **2. Metadata classification pipeline** | **L** (largest, judgment-heavy) | Confidence-tiered extraction (Decision 2) over the 496 existing documents + 728 investigations; auto-accept High into the relationship graph; review-queue UI for Medium; Low left alone; wired into future ingestion | 1 table (`metadata_classification_suggestions`) | new `app/engines/knowledge/classification.py`, new review-queue router + UI page, `knowledge_management/engine.py` (hook into `upload_document`) | Phase 1 |
| **3. Applicability-aware retrieval & ranking** | **M** | The Part D pipeline: metadata pre-filter/boost on local search, symmetric "different customer" penalty added to TFS/Wiki ranking too | none (uses Phase 1/2 data) | new `applicability.py`, `knowledge_store.py`, `ranking.py`, `recommendation/engine.py` call sites | Phases 1–2 (needs real tagged data to filter on, though can be developed against a partial set) |
| **4. Resolution provenance** | **S–M** | `ResolutionProvenance` enum + field; `resolution_verified_by/at` on Known Bug (and Historical Investigation); admin "mark verified" action; deterministic mapping rule (Part G has the exact rule for review) | 2 nullable columns | `recommendation.py`, `evidence.py`, `recommendation/engine.py`, small admin UI addition | Can land independently, any time after Phase 0 |
| **5. Chat Assistant v1** | **XL** (largest overall) | Intent classifier + slot-filling + answer templates for the question set in Decision 3, with follow-up/conversation-context support; release-note questions specifically gated on you providing real release-note files | 2 tables (`chat_sessions`/`chat_messages`) | new `app/domain/chat.py`, `app/engines/chat/*`, new API router, new/replaced Streamlit chat UI | Phases 0–4 (the entire point is Chat calls the now-applicability-aware, provenance-aware engines, not a parallel implementation) |
| **6. Investigation Workspace ↔ Chat shared context** | **S** | Embedded/linked chat panel in the Workspace reading/writing the same live `ContextProfile`; chat history visible on the investigation | none | Workspace UI, `ContextProfile` wiring | Phase 5 (or folded into its tail end rather than fully separate) |

**Explicitly not scheduled**: any further bulk document import (Decision 7 — the current 496 is the working set until Phases 1–3 land), a Protocol dimension table (no decision calls for it yet; Part C.1 confirms the pattern is ready whenever needed), Environment as a real table (Decision 4 — same).

---

# Part G — Open items for you before implementation starts

1. **Low-confidence classification suggestions** (Part C.1) — log them (status stays visible, just never promoted to a relationship) for future re-evaluation, or don't write them at all? Leaning toward "log them," but this is genuinely your call.
2. **Resolution provenance mapping rule** — proposed concretely: `CONFIRMED` requires either a correlated CRM/ticket match across two independent sources (already-implemented `correlated_local` logic) **or** an explicit `resolution_verified_by` admin flag; a single-source high-similarity match, however high the score, caps at `LIKELY`; anything below the existing Medium confidence threshold is `POSSIBLE`; no qualifying source at all is `UNKNOWN`. Confirming this matches your intent before I build it, since it's the exact rule that prevents "confirmed merely because the score is high."
3. **TFS Area Path as a customer signal** — I have not yet live-verified whether `System.AreaPath` on real Command Center work items actually encodes customer (vs. just product/team structure). This needs a short real-data check at the start of Phase 1 before it can be relied on as one of the "reliable Wiki/TFS evidence" sources Decision 1 requires — flagging now so it isn't a silent assumption later.
4. **Release Notes** — still blocked on you providing real files (unchanged ask from Revision 1).
5. **Confluence PAT** — still needed to verify the Wiki live connector (unchanged ask from Revision 1); not required for Phases 0–4, but relevant to Phase 5's Wiki-sourced chat answers.

---

*No application code has been modified to produce this revision. Waiting for your review/approval of Part B–F before any implementation begins.*
