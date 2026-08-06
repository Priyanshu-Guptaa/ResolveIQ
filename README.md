# ResolveIQ

**Investigation Intelligence Platform for L2/L3 Product Support Engineers.**

ResolveIQ is not a chatbot -- it's a system that reasons over Evidence
(logs, task descriptions, historical investigations, documentation, known
bugs) and answers one question on every screen: **"What should I do
next?"**

This is **Sprint 1**: an end-to-end vertical slice proving the core
investigation workflow, using sample data in place of real connectors.

## What Sprint 1 proves

1. An engineer starts an investigation by pasting a task description.
2. Uploads log files -- the Log Intelligence Engine parses them into a
   common event model and extracts entities (IDs, hosts, exceptions,
   Kafka topics, meter numbers, ...) regardless of log format.
3. The Knowledge Engine semantically searches sample historical
   investigations, documentation, and known bugs.
4. The Recommendation Engine combines both signals into root-cause
   hypotheses, similar past investigations, and a concrete next best step.

**Explicitly out of scope for Sprint 1:** ServiceNow/Wiki/Azure DevOps/
Copilot connectors, PDF/PPT ingestion, authentication. Sample JSON
knowledge (`data/sample_knowledge/`) and sample logs (`data/sample_logs/`)
stand in for those sources.

## Architecture

```
app/
├── domain/            Framework-agnostic models: Evidence, InvestigationSession,
│                       ExtractedEntity, LogEvent, Recommendation, ...
├── engines/
│   ├── log_intelligence/   Parses any log format -> LogEvent; extracts entities
│   ├── knowledge/          Imports + semantically searches historical
│   │                       investigations / docs / bugs (ChromaDB)
│   ├── investigation/      Owns the investigation lifecycle + shared context
│   └── recommendation/     Root causes, similar investigations, next best step
├── infrastructure/
│   ├── db/             SQLite persistence (SQLAlchemy) via the repository pattern
│   └── vectorstore/    ChromaDB client wrapper
└── api/                FastAPI app: routers, schemas, DI wiring
ui/streamlit_app.py     Thin UI client over the API
scripts/seed_knowledge.py   Standalone knowledge (re)seed CLI
data/sample_knowledge/  Sample historical investigations / docs / known bugs
data/sample_logs/       Sample log files (Java, IIS, Kafka, meter/collector)
tests/                  pytest unit tests for the engines
```

Every engine sits behind a Protocol interface (`EntityExtractor`,
`LogParser`, `KnowledgeStore`, `EmbeddingProvider`, `InvestigationRepository`)
so implementations are swappable and testable without a DI framework --
FastAPI's `Depends` plus plain Python constructor injection is enough
(see `app/api/dependencies.py`).

## Stack

- **FastAPI** + Uvicorn (API)
- **SQLite** via SQLAlchemy 2.0 (structured data: investigations, evidence)
- **ChromaDB** (persistent, local) for semantic search
- **sentence-transformers** (`all-MiniLM-L6-v2`) for local, on-prem-friendly
  embeddings -- no evidence content leaves the network
- **Streamlit** for the Sprint 1 UI
- **pytest** for tests

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Terminal 1: API (auto-seeds sample knowledge into Chroma on first run)
uvicorn app.api.main:app --reload

# Terminal 2: UI
streamlit run ui/streamlit_app.py
```

Then open the Streamlit URL, start an investigation, paste a description
(or use one that mentions e.g. a `NullPointerException` or `spid=61`),
upload one of the files in `data/sample_logs/`, and click **Analyze**.

### Tests

```bash
pytest
```

### Re-seeding knowledge manually

```bash
python -m scripts.seed_knowledge --force
```

## Design notes / deliberate Sprint 1 simplifications

- Extracted entities and log events are stored as JSON columns on the
  `evidence` table rather than normalized tables -- there's no
  cross-investigation entity search yet. Promoting them to their own
  table is a contained migration when that's needed.
- Recommendations are rule-based (semantic similarity + entity
  heuristics), not LLM-generated -- keeps Sprint 1 fully local/offline per
  the embeddings decision. A generative explanation layer on top of this
  same output shape is a natural additive enhancement later.
- API responses mostly re-export domain (Pydantic) models directly rather
  than a parallel DTO layer -- reconsider once the domain model needs to
  diverge from the wire format.

## Next sprints (not yet built)

- Connectors: ServiceNow, Internal Wiki, Azure DevOps, PDF/PPT ingestion
- Real Knowledge Management import UI
- Investigation Engine enhancements: multi-user, status transitions, audit trail
- Authentication / authorization
- Production database (PostgreSQL or SQL Server) migration path
