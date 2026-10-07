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

    # --- Hosted-deployment overrides ---------------------------------------
    # Every value defaults to the local single-process behavior above, so a
    # fresh checkout is unchanged; set these to run against shared services.
    database_url: str | None = None
    """Full SQLAlchemy URL (e.g. ``postgresql+psycopg://user:pw@host/db``).
    When set it replaces ``sqlite_path``; unset keeps the local SQLite file."""
    chroma_host: str | None = None
    """When set, talk to a standalone Chroma server over HTTP instead of
    opening ``chroma_persist_dir`` in-process -- required once more than one
    API replica runs (an embedded persistent client is single-process)."""
    chroma_port: int = 8000
    chroma_ssl: bool = False
    chroma_auth_token: str | None = None
    upload_backend: Literal["local", "s3"] = "local"
    """Where Knowledge Management keeps original uploaded bytes. ``local``
    writes under ``knowledge_upload_dir`` (needs a persistent volume);
    ``s3`` uses any S3-compatible bucket (needs the optional ``boto3``)."""
    upload_s3_bucket: str | None = None
    upload_s3_prefix: str = "knowledge/"
    upload_s3_endpoint_url: str | None = None
    """Set for S3-compatible stores (MinIO, etc.); leave unset for AWS S3."""
    embedding_backend: Literal["sentence-transformers", "onnx"] = "sentence-transformers"
    """``onnx`` runs the same all-MiniLM-L6-v2 weights through onnxruntime
    (no torch) for a much smaller image; vectors match sentence-transformers
    to float tolerance (cosine 1.000000 measured), so existing indexes
    remain valid."""

    # --- Authentication (hosted deployment) --------------------------------
    auth_enabled: bool = False
    """False (default): every route is open, exactly as in local use. True:
    all routes except /health and /auth/login require a bearer token, and
    /admin/* additionally requires the ``admin`` role."""
    auth_secret_key: str | None = None
    auth_token_ttl_minutes: int = 480
    auth_bootstrap_admin_username: str | None = None
    auth_bootstrap_admin_password: str | None = None
    """When the users table is empty at startup, an admin is created from
    these two values (so a fresh deployment can sign in). Ignored once any
    user exists."""

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

    tfs_personal_access_token: str | None = None
    """Optional TFS PAT. When set, the connector authenticates with it
    (Basic, empty username) instead of Windows SSPI -- required on a
    Linux/container host, where SSPI is unavailable. Never logged or
    returned by any API; set via RESOLVEIQ_TFS_PERSONAL_ACCESS_TOKEN."""

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
    ollama_model: str = "qwen2.5:3b"
    """Chat Assistant Phase 21 -- switched from ``qwen3:4b``. Phase 11's
    40-call real statistical baseline measured qwen3:4b at 32.5% success
    (95% CI [20.1%, 48.0%]), with 60% of calls hitting the
    ``ollama_num_predict`` cap empty-handed -- a structural property of
    its "thinking" reasoning phase (Phase 7 proved ``think=false`` does
    not suppress it in this Ollama build), not fixable by prompt
    engineering (Phases 9-10 each tried, neither moved the ceiling).
    ``qwen2.5:3b`` has no documented or observed thinking capability
    (confirmed via ``ollama show`` -- capabilities are exactly
    ``completion``, ``tools``, no ``thinking``) and, on the identical
    fixtures, real-tested at 100% completion (Phase 13: 40/40; Phase 19:
    40/40 after Rule 8) with 40-100x lower latency (single-digit seconds
    vs. qwen3:4b's ~208-221s mean/median). Phase 19 additionally found
    (and Phase 22's structured applicability block, see
    ``_APPLICABILITY_HEADER``/``_applicability_block`` in
    ``app/engines/llm/prompt_builder.py``, specifically addresses -- a
    60-call real-model stress test found the rigid per-field format
    reaches 100% unknown-customer safety, vs. 70% for Phase 21's
    free-prose guard) a
    narrow fabrication risk on "which customer is affected"-style
    questions when applicability is empty. Fully overridable via
    RESOLVEIQ_OLLAMA_MODEL. Reverting to ``qwen3:4b`` is a one-line
    change here -- no other file depends on which model tag is
    configured (``OllamaProvider`` is fully model-agnostic)."""
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
    llm_async_enabled: bool = False
    """Chat Assistant Phase 37 -- when True (and only when
    ``llm_enabled`` is ALSO True), ``ChatOrchestrator.handle_message()``
    returns the deterministic answer immediately and schedules LLM
    generation as a background enhancement job (see
    ``app.engines.chat.enhancement``) instead of blocking the request on
    Ollama. Defaults to False so this phase's work is purely additive:
    with the default ``Settings`` (``llm_enabled=False``), behavior is
    byte-identical to every prior phase; even with ``llm_enabled=True``
    and this still False, behavior is the existing Phase 1-35B
    synchronous path, completely unchanged. Deliberately independent of
    ``llm_enabled`` -- production behavior must never depend on this
    flag while ``llm_enabled`` is False, and it is not: this flag alone,
    with ``llm_enabled`` at its own default, changes nothing. A future
    production enablement can turn both flags on together as one
    deliberate step."""
    llm_max_concurrent_jobs: int = 1
    """Chat Assistant Phase 37 -- the maximum number of LLM enhancement
    jobs allowed to run at once when ``llm_async_enabled`` is True.
    qwen2.5:3b was directly measured (Phase 35, a real ``ollama ps``
    during real calls) running at 100% CPU with no GPU/iGPU acceleration
    active on this hardware -- a second concurrent generation would only
    contend for the same CPU cycles Ollama is already using, not add
    real throughput, so 1 is the conservative, evidence-based starting
    value, not an arbitrary guess. Raise this only once a real
    environment (e.g. genuine GPU acceleration, or multiple Ollama
    instances) has been measured to actually benefit from it."""
    llm_enhancement_max_queued: int = 1
    """Chat Assistant Phase 37 -- how many further enhancement jobs may
    wait beyond the ``llm_max_concurrent_jobs`` already running before a
    new one is REJECTED outright (see
    ``app.engines.chat.enhancement.ChatEnhancementService``). Always
    small and finite by design -- the queue must never grow unbounded,
    and a rejected enhancement never blocks or degrades the request:
    the deterministic answer is returned regardless."""

    @model_validator(mode="after")
    def _validate_ollama_num_predict(self) -> "Settings":
        """Fail fast, same idiom as ``_validate_hybrid_fusion_weights`` --
        a negative token budget is nonsensical, never silently clamped."""
        if self.ollama_num_predict < 0:
            raise ValueError(
                f"ollama_num_predict must be >= 0 (0 disables the cap) -- got {self.ollama_num_predict!r}"
            )
        return self

    @model_validator(mode="after")
    def _validate_auth_and_storage(self) -> "Settings":
        """Fail fast at startup rather than boot an open or broken
        deployment: auth needs a real signing key, S3 needs a bucket."""
        if self.auth_enabled:
            if not self.auth_secret_key or len(self.auth_secret_key) < 32:
                raise ValueError("auth_enabled requires auth_secret_key of at least 32 characters")
            if self.auth_token_ttl_minutes <= 0:
                raise ValueError("auth_token_ttl_minutes must be positive")
        if (self.auth_bootstrap_admin_username is None) != (self.auth_bootstrap_admin_password is None):
            raise ValueError("auth_bootstrap_admin_username and auth_bootstrap_admin_password must be set together")
        if self.upload_backend == "s3" and not self.upload_s3_bucket:
            raise ValueError("upload_backend='s3' requires upload_s3_bucket")
        return self

    @property
    def sqlite_url(self) -> str:
        """The database URL actually used. Name kept for existing callers;
        ``database_url`` (hosted) takes precedence over the local file."""
        if self.database_url:
            return self.database_url
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.sqlite_path.as_posix()}"


@lru_cache
def get_settings() -> Settings:
    """Process-wide singleton. FastAPI routes depend on this via
    ``Depends(get_settings)`` rather than importing a module-level instance,
    so tests can override it with ``app.dependency_overrides``.
    """
    return Settings()
