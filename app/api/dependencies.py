"""Dependency-injection wiring for the FastAPI layer.

Expensive resources (embedding model, Chroma client, DB engine) are
process-wide singletons via ``lru_cache``; engines are lightweight and
constructed per-request from those singletons. Routes depend on the
``get_*_engine`` functions via ``Depends(...)`` -- never on the singletons
directly -- so tests can override any of these with
``app.dependency_overrides``.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.config import Settings, get_settings
from app.engines.investigation.engine import InvestigationEngine
from app.engines.knowledge.embedding_provider import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from app.engines.knowledge.engine import KnowledgeEngine
from app.engines.knowledge.knowledge_store import ChromaKnowledgeStore, KnowledgeStore
from app.engines.log_intelligence.engine import LogIntelligenceEngine
from app.engines.log_intelligence.entity_extractor import EntityExtractor, RegexEntityExtractor
from app.engines.log_intelligence.log_parser import GenericLogParser, LogParser
from app.engines.recommendation.engine import RecommendationEngine
from app.infrastructure.db.repository import InvestigationRepository, SqlAlchemyInvestigationRepository
from app.infrastructure.db.session import get_session_factory


@lru_cache
def _embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    return SentenceTransformerEmbeddingProvider(settings.embedding_model_name)


@lru_cache
def _knowledge_store() -> KnowledgeStore:
    settings = get_settings()
    return ChromaKnowledgeStore(settings.chroma_persist_dir, _embedding_provider())


@lru_cache
def _knowledge_engine_singleton() -> KnowledgeEngine:
    return KnowledgeEngine(_knowledge_store())


@lru_cache
def _db_session_factory() -> sessionmaker[OrmSession]:
    settings = get_settings()
    return get_session_factory(settings.sqlite_url)


@lru_cache
def _investigation_repository() -> InvestigationRepository:
    return SqlAlchemyInvestigationRepository(_db_session_factory())


@lru_cache
def _entity_extractor() -> EntityExtractor:
    return RegexEntityExtractor()


@lru_cache
def _log_parser() -> LogParser:
    return GenericLogParser(_entity_extractor())


@lru_cache
def _log_intelligence_engine() -> LogIntelligenceEngine:
    return LogIntelligenceEngine(_log_parser(), _entity_extractor())


# --- Public dependencies (used via FastAPI's Depends(...)) -----------------


def get_settings_dep() -> Settings:
    return get_settings()


def get_investigation_engine() -> InvestigationEngine:
    return InvestigationEngine(_investigation_repository(), _log_intelligence_engine())


def get_knowledge_engine() -> KnowledgeEngine:
    return _knowledge_engine_singleton()


def get_recommendation_engine() -> RecommendationEngine:
    return RecommendationEngine(_knowledge_engine_singleton(), get_settings())


def reset_singletons() -> None:
    """Test/dev helper: clears every cached singleton so the next
    dependency call rebuilds from current settings. Not used by the app
    itself in normal operation."""
    for fn in (
        _embedding_provider,
        _knowledge_store,
        _knowledge_engine_singleton,
        _db_session_factory,
        _investigation_repository,
        _entity_extractor,
        _log_parser,
        _log_intelligence_engine,
    ):
        fn.cache_clear()
