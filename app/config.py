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
