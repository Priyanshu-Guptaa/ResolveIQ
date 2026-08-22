"""Centralized application configuration.

All environment-dependent values live here, loaded via pydantic-settings
from environment variables / a ``.env`` file. Nothing else in the codebase
should call ``os.environ`` directly -- this keeps configuration
discoverable and testable (swap in a different ``Settings`` instance via
dependency injection instead of monkeypatching env vars).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RESOLVEIQ_", extra="ignore")

    app_name: str = "ResolveIQ"
    log_level: str = "INFO"

    # --- Persistence -----------------------------------------------------
    sqlite_path: Path = BASE_DIR / "data" / "resolveiq.db"
    chroma_persist_dir: Path = BASE_DIR / "data" / "chroma"
    knowledge_upload_dir: Path = BASE_DIR / "data" / "uploads" / "knowledge"
    """Where Knowledge Management (Sprint 3, Phase 3.2) keeps the
    original bytes of an admin's uploaded document, for provenance --
    search only ever uses the extracted ``content`` stored in SQLite,
    never reads this back."""

    # --- Knowledge / embeddings -------------------------------------------
    embedding_model_name: str = "all-MiniLM-L6-v2"
    sample_knowledge_dir: Path = BASE_DIR / "data" / "sample_knowledge"
    auto_seed_knowledge: bool = True
    """If True, the API seeds sample knowledge into Chroma on startup when
    the collections are empty. Disable once real connectors exist."""

    # --- Recommendation Engine ---------------------------------------------
    similarity_top_k: int = 5
    min_similarity_for_root_cause: float = 0.35
    """Below this similarity score, the Recommendation Engine falls back to
    entity-based heuristics instead of borrowing a historical root cause."""

    # --- External Knowledge: TFS + Wiki live connectors ---------------------
    # Deliberately not imported/embedded (see app/domain/external_knowledge.py's
    # module docstring) -- queried live, per Analyze click, with a short-TTL
    # cache. Every value here is optional and defaults to "disabled/absent"
    # so a fresh checkout with no .env still runs -- Analyze must never
    # require these to be configured.
    external_knowledge_enabled: bool = True
    """Kill switch -- False skips both connectors entirely (no network
    calls), independent of whether individual credentials are set."""
    external_knowledge_timeout_seconds: float = 20.0
    """Hard per-connector timeout for a live Analyze-time query -- a
    slow/unreachable TFS or Wiki must never make an engineer wait
    indefinitely for the rest of the Investigation Strategy.

    Live-measured against the real am.tfs.landisgyr.net (not a guess):
    one WIQL search + one batched work-item detail fetch (the two
    real requests search_work_items() makes) took 8-20s end to end
    across repeated real calls, even after fixing the connector to
    reuse a persistent NTLM-authenticated session instead of
    renegotiating per request (that fix alone cut a 42s worst case
    roughly in half). The remaining latency appears to be genuine
    server-side/network cost on this instance, not something this
    client can optimize further. 20s is set to comfortably cover the
    slowest real case observed, not an arbitrary round number --
    lower it only after re-measuring against your own environment."""
    external_knowledge_cache_ttl_seconds: int = 900
    """15 minutes. Cache is in-memory only (see
    app/engines/external_knowledge/cache.py) -- cleared on process
    restart, never persisted; this bounds repeat-query cost, not a
    data store."""
    external_knowledge_max_results: int = 8
    """Per connector, per query -- keeps the UI section scannable and
    bounds how many work-item/page detail fetches one Analyze click
    can trigger."""

    tfs_base_url: str | None = None
    """e.g. "https://am.tfs.landisgyr.net/tfs/DefaultCollection" --
    unset disables the TFS connector (available=False, not an error)."""
    tfs_project: str | None = None
    """e.g. "Command Center" -- the TFS project to scope every query
    to; required alongside tfs_base_url."""

    wiki_base_url: str | None = None
    """e.g. "https://wiki.landisgyr.net" -- unset disables the Wiki
    connector (available=False, not an error)."""
    wiki_api_username: str | None = None
    wiki_api_token: str | None = None
    """Confluence Personal Access Token, used as the Basic-auth
    password (see wiki_rest_client.py). Never logged, never returned
    in any API response, never stored anywhere but this env-loaded
    field -- set via RESOLVEIQ_WIKI_API_TOKEN in a local .env
    (git-ignored) or a real environment variable, never in source or
    the database."""
    wiki_space_key: str | None = None
    """Optional -- scopes Wiki search to one space (e.g. "CC") when
    set; searches all spaces the token can read when unset."""

    # --- LLM Gateway: local Ollama/Qwen provider (Chat Assistant Phase 1) ---
    # Kill switch defaults to False, same idiom as external_knowledge_enabled
    # -- a fresh checkout or a build with no Ollama installed runs with zero
    # behavior change: ChatOrchestrator falls back to its existing
    # deterministic _compose_answer() unconditionally.
    llm_enabled: bool = False
    """False (default): ChatOrchestrator never attempts LLM generation --
    behavior is byte-identical to the pre-Phase-1 deterministic path."""
    ollama_base_url: str = "http://localhost:11434"
    """Where the local Ollama daemon is reachable. No environment-specific
    value hardcoded beyond this documented local default."""
    ollama_model: str = "qwen3:4b"
    """PROVISIONAL / UNVERIFIED LOCALLY -- Ollama is not installed on the
    machine this was implemented on, so this tag has not been checked
    against a real `ollama list`. Fully overridable via
    RESOLVEIQ_OLLAMA_MODEL; must be verified (or corrected) the first time
    this runs against a real Ollama instance."""
    ollama_timeout_seconds: float = 300.0
    """Empirically measured (Phase 3/3A real-Ollama validation, real
    qwen3:4b on this hardware, default/thinking-enabled generation --
    the only viable mode, see Phase 3A's own report for why `think:
    false` was tested and rejected): prompt evaluation and model
    loading are negligible (<0.1s each, confirmed via Ollama's own
    `prompt_eval_duration`/`load_duration`); the real cost is token
    generation at a measured ~7-9 tokens/sec CPU-bound throughput.
    Real structured-resolution prompts completed successfully in
    124-177s across multiple runs, but also produced multiple
    timeouts at 180s and one run exceeding 300s outright -- reasoning
    length is stochastic, not fixed, so no timeout eliminates fallback
    entirely. 300s was chosen to cover the observed successful range
    with real margin while still bounding worst-case wait time;
    occasional fallback for outlier-length reasoning is expected,
    graceful degradation (see ChatOrchestrator._generate_answer()),
    not a bug to chase away by raising this further."""
    ollama_num_predict: int = 2048
    """Empirically measured (Phase 3D real-Ollama investigation): Ollama's
    ``options.num_predict`` bounds TOTAL generated tokens INCLUDING Qwen's
    reasoning, not just the visible answer -- directly proven by a real
    ``num_predict=50`` probe that produced ``done_reason="length"`` with
    an EMPTY final answer, cut off mid-reasoning, never an abbreviated-
    but-complete one. Real structured-resolution prompts were observed
    generating 1209-1934 total tokens to completion; 2048 is the
    smallest value from that investigation's evaluated candidate set
    that clears every one of those real, completed samples. This does
    NOT guarantee completion before ``ollama_timeout_seconds`` -- at the
    measured worst-case ~7 tokens/sec throughput, 2048 tokens alone is
    ~293s, leaving only slim margin under the current 300s timeout, and
    at least one real observed generation exceeded even 1934 tokens
    without a captured upper bound. ``0`` disables the cap entirely
    (no ``options`` key is sent at all -- see ``OllamaProvider``),
    restoring the original, pre-Phase-3D unbounded-generation behavior."""

    @model_validator(mode="after")
    def _validate_ollama_num_predict(self) -> "Settings":
        """Fail fast, same idiom as ``_validate_hybrid_fusion_weights`` --
        a negative token budget is nonsensical, never silently clamped."""
        if self.ollama_num_predict < 0:
            raise ValueError(
                f"ollama_num_predict must be >= 0 (0 disables the cap) -- got {self.ollama_num_predict!r}"
            )
        return self

    # --- Hybrid Retrieval: BM25 + RRF foundation (Chat Assistant Phase 2) ---
    # Kill switch defaults to False, same idiom as llm_enabled/
    # external_knowledge_enabled -- a fresh checkout runs with the existing,
    # unchanged ChromaKnowledgeStore (pure vector search) until this is
    # explicitly enabled AND the existing 1,222-record real corpus has been
    # backfilled into the lexical index (operational step: re-run
    # KnowledgeEngine.seed_from_directory(force=True) once HybridKnowledgeStore
    # is wired in -- see app/engines/knowledge/hybrid_store.py).
    rrf_enabled: bool = False
    """False (default): _knowledge_store() returns the existing, unmodified
    ChromaKnowledgeStore exactly as before Phase 2 -- behavior is
    byte-identical to pre-Phase-2 retrieval."""
    rrf_k: int = 60
    """Reciprocal Rank Fusion's own constant -- the literature-standard
    default (Cormack et al.), robust across very different score
    distributions between retrievers. See
    app/engines/knowledge/rank_fusion.py."""

    # --- Hybrid Retrieval calibration (Chat Assistant Phase 2C) -------------
    # Phase 2C (read-only investigation) found current RRF fusion
    # structurally vulnerable to a rank-based tie-break artifact at the
    # real production overfetch depth (20): a record with two mediocre
    # signals can out-accumulate a record with one perfect signal (a
    # ticket ID or exact identifier found only by BM25). Weighted score
    # fusion was the only tested calibration strategy immune to this --
    # see PHASE 2C — RRF CALIBRATION REPORT, Section 14. Both settings
    # below are completely inert unless BOTH ``rrf_enabled=True`` AND
    # ``hybrid_fusion_mode="score"`` -- neither is true by default, so a
    # fresh checkout's behavior is unaffected by this phase.
    hybrid_fusion_mode: Literal["rrf", "score"] = "rrf"
    """Which algorithm HybridKnowledgeStore uses to combine vector +
    lexical candidates when ``rrf_enabled=True``. "rrf" (default)
    preserves the exact Phase 2 behavior. "score" is Phase 2C's
    recommended calibration (weighted linear combination of the real
    vector/lexical scores) -- not yet enabled by default pending the
    natural-language-query validation gap the Phase 2C report explicitly
    left open."""
    hybrid_vector_weight: float = 0.25
    """Weight given to the real vector (cosine similarity) score under
    ``hybrid_fusion_mode="score"``. Phase 2C's recommended value --
    see that report's Section 5 (6/6 ground-truth top-1, MRR 1.00)."""
    hybrid_lexical_weight: float = 0.75
    """Weight given to the real lexical (BM25, already normalized to
    [0,1] by LexicalKnowledgeStore) score under
    ``hybrid_fusion_mode="score"``. Phase 2C's recommended value --
    see that report's Section 5."""

    # --- Exact-Identifier Protection (Chat Assistant Phase 2D) --------------
    # Kill switch defaults to False, same idiom as every other Phase 2/2C
    # flag -- inert unless BOTH rrf_enabled=True AND this is explicitly
    # True, so a fresh checkout's behavior is unaffected. See
    # app/engines/knowledge/identifier_protection.py and PHASE 2D — EXACT
    # IDENTIFIER PROTECTION DESIGN. Deliberately no configurable boost
    # weight -- the +0.30 adjustment is a fixed, evidence-derived
    # constant (IDENTIFIER_PROTECTION_BOOST in that module), not a new
    # tuning knob.
    identifier_protection_enabled: bool = False
    """False (default): HybridKnowledgeStore.query() never applies the
    exact-identifier boost -- behavior is byte-identical to
    pre-Phase-2D retrieval."""

    @model_validator(mode="after")
    def _validate_hybrid_fusion_weights(self) -> "Settings":
        """Fail fast on a nonsensical weight configuration rather than
        silently normalizing it away -- see Phase 2C implementation
        Section 5's explicit "prefer failing fast" instruction. Only
        checked here (not deferred to first query) so a misconfigured
        ``.env`` is caught at startup, not at an engineer's first
        Analyze click."""
        if self.hybrid_vector_weight < 0 or self.hybrid_lexical_weight < 0:
            raise ValueError(
                "hybrid_vector_weight and hybrid_lexical_weight must both be >= 0 "
                f"(got vector={self.hybrid_vector_weight!r}, lexical={self.hybrid_lexical_weight!r})"
            )
        if self.hybrid_vector_weight + self.hybrid_lexical_weight <= 0:
            raise ValueError(
                "hybrid_vector_weight + hybrid_lexical_weight must be > 0 -- "
                "at least one retrieval signal must carry weight"
            )
        return self

    @model_validator(mode="after")
    def _warn_identifier_protection_under_rrf(self) -> "Settings":
        """Observability only (Phase 2E — RRF / Identifier Protection
        Compatibility Review, Option A) -- deliberately a WARNING, never
        a ``raise``: the combination below is not invalid, only
        limited. Never modifies either setting, never blocks
        construction, never changes retrieval behavior.

        Phase 2E's real-corpus investigation found that a lexical-only
        candidate's ``KnowledgeMatch.score`` is intentionally
        materialized as ``0.0`` under ``hybrid_fusion_mode="rrf"`` (see
        ``HybridKnowledgeStore``'s own CRITICAL SCORE CONTRACT) --
        Phase 2D's identifier-protection boost is a small, bounded
        addition on top of whatever score fusion already produced, not
        a substitute for it, so ``0.0 + IDENTIFIER_PROTECTION_BOOST``
        is frequently still far below a genuinely unrelated but
        vector-similar competitor's real cosine score (Phase 2E
        measured real competitors at ~0.6-0.73 against a boosted
        ~0.3-0.4). A ``both``-sourced candidate (e.g. a known bug
        found by both retrievers) is NOT affected by this and still
        benefits from protection under RRF -- which is exactly why
        this is a warning, not a hard failure: the combination has
        real, partial value, just not a reliable fix for lexical-only
        exact identifiers. ``hybrid_fusion_mode="score"`` remains the
        recommended pairing whenever identifier protection is
        enabled."""
        if self.identifier_protection_enabled and self.hybrid_fusion_mode == "rrf":
            logger.warning(
                "identifier_protection_enabled=True with hybrid_fusion_mode='rrf': exact identifier "
                "protection is not reliably effective for lexical-only candidates in this mode, because "
                "RRF intentionally materializes a lexical-only KnowledgeMatch.score as 0.0 before the "
                "protection boost is applied -- a genuinely unrelated vector-similar candidate routinely "
                "outscores it even after boosting. hybrid_fusion_mode='score' is the recommended pairing "
                "when identifier protection is enabled; both-sourced candidates still benefit under 'rrf'."
            )
        return self

    @property
    def sqlite_url(self) -> str:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.sqlite_path.as_posix()}"


@lru_cache
def get_settings() -> Settings:
    """Process-wide singleton. FastAPI routes depend on this via
    ``Depends(get_settings)`` rather than importing a module-level instance,
    so tests can override it with ``app.dependency_overrides``.
    """
    return Settings()
