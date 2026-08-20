"""Centralized application configuration.

All environment-dependent values live here, loaded via pydantic-settings
from environment variables / a ``.env`` file. Nothing else in the codebase
should call ``os.environ`` directly -- this keeps configuration
discoverable and testable (swap in a different ``Settings`` instance via
dependency injection instead of monkeypatching env vars).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


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
    ollama_timeout_seconds: float = 60.0
    """Placeholder, not empirically measured (unlike
    external_knowledge_timeout_seconds's own docstring, which cites real
    measured TFS latency) -- local LLM inference latency on the target
    machine is unknown and needs benchmarking once Ollama is installed."""

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
