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
from app.engines.chat.conversation_state import ConversationStateEngine
from app.engines.chat.orchestrator import ChatOrchestrator
from app.engines.ingestion.engine import IngestionEngine
from app.engines.ingestion.file_type_registry import FileTypeRegistry
from app.engines.investigation.engine import InvestigationEngine
from app.engines.knowledge.classification import DocumentClassificationEngine
from app.engines.knowledge.embedding_provider import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from app.engines.knowledge.engine import KnowledgeEngine
from app.engines.knowledge.knowledge_store import ChromaKnowledgeStore, KnowledgeStore
from app.engines.knowledge_management.engine import KnowledgeManagementEngine
from app.engines.knowledge_object_framework.adapters import KnowledgeObjectAdapter, build_adapters
from app.engines.knowledge_object_framework.service import KnowledgeObjectService
from app.engines.knowledge_relationships.engine import KnowledgeRelationshipEngine
from app.engines.llm.ollama_provider import OllamaProvider
from app.engines.llm.provider import LLMProvider
from app.engines.log_intelligence.engine import LogIntelligenceEngine
from app.engines.log_intelligence.entity_extractor import EntityExtractor, RegexEntityExtractor
from app.engines.log_intelligence.log_parser import GenericLogParser, LogParser
from app.engines.log_knowledge.importer import LogWikiImporter
from app.engines.product_intelligence.component_registry import ComponentRegistry
from app.engines.product_intelligence.engine import ProductIntelligenceEngine
from app.engines.query_understanding.engine import QueryUnderstandingEngine
from app.engines.external_knowledge.service import ExternalKnowledgeService
from app.engines.external_knowledge.tfs_connector import TfsConnector
from app.engines.external_knowledge.tfs_rest_client import TfsRestConnector
from app.engines.external_knowledge.wiki_connector import WikiConnector
from app.engines.external_knowledge.wiki_rest_client import WikiRestConnector
from app.engines.recommendation.engine import RecommendationEngine
from app.engines.sql_library.engine import SqlLibraryEngine
from app.engines.task_import.importer import TaskImporter
from app.infrastructure.db.chat_repository import ChatRepository, SqlAlchemyChatRepository
from app.infrastructure.db.classification_repository import ClassificationRepository, SqlAlchemyClassificationRepository
from app.infrastructure.db.component_repository import ComponentProfileRepository, SqlAlchemyComponentProfileRepository
from app.infrastructure.db.knowledge_repository import KnowledgeRepository, SqlAlchemyKnowledgeRepository
from app.infrastructure.db.log_knowledge_repository import LogKnowledgeRepository, SqlAlchemyLogKnowledgeRepository
from app.infrastructure.db.lookup_repository import LookupRepository, SqlAlchemyLookupRepository
from app.infrastructure.db.playbook_repository import PlaybookRepository, SqlAlchemyPlaybookRepository
from app.infrastructure.db.relationship_repository import RelationshipRepository, SqlAlchemyRelationshipRepository
from app.infrastructure.db.repository import InvestigationRepository, SqlAlchemyInvestigationRepository
from app.infrastructure.db.seed_migration import migrate_all
from app.infrastructure.db.session import get_session_factory
from app.infrastructure.db.sql_template_repository import SqlAlchemySqlTemplateRepository, SqlTemplateRepository
from app.infrastructure.db.version_repository import SqlAlchemyVersionRepository, VersionRepository
from app.domain.knowledge_relationships import KnowledgeObjectType


@lru_cache
def _embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    return SentenceTransformerEmbeddingProvider(settings.embedding_model_name)


@lru_cache
def _knowledge_store() -> KnowledgeStore:
    settings = get_settings()
    return ChromaKnowledgeStore(settings.chroma_persist_dir, _embedding_provider())


@lru_cache
def _db_session_factory() -> sessionmaker[OrmSession]:
    settings = get_settings()
    return get_session_factory(settings.sqlite_url)


@lru_cache
def _investigation_repository() -> InvestigationRepository:
    return SqlAlchemyInvestigationRepository(_db_session_factory())


# --- Governed knowledge repositories (Sprint 3, Phase 3.1) ------------------


@lru_cache
def _component_profile_repository() -> ComponentProfileRepository:
    return SqlAlchemyComponentProfileRepository(_db_session_factory())


@lru_cache
def _knowledge_repository() -> KnowledgeRepository:
    return SqlAlchemyKnowledgeRepository(_db_session_factory())


@lru_cache
def _sql_template_repository() -> SqlTemplateRepository:
    return SqlAlchemySqlTemplateRepository(_db_session_factory())


@lru_cache
def _knowledge_engine_singleton() -> KnowledgeEngine:
    return KnowledgeEngine(_knowledge_store(), _knowledge_repository())


@lru_cache
def _sql_library_engine() -> SqlLibraryEngine:
    return SqlLibraryEngine(_sql_template_repository())


@lru_cache
def _knowledge_management_engine() -> KnowledgeManagementEngine:
    settings = get_settings()
    return KnowledgeManagementEngine(
        _knowledge_repository(),
        _component_profile_repository(),
        _ingestion_engine(),
        _knowledge_engine_singleton(),
        settings.knowledge_upload_dir,
    )


# --- Knowledge Relationship Manager (Sprint 3, Phase 3.3) -------------------


@lru_cache
def _playbook_repository() -> PlaybookRepository:
    return SqlAlchemyPlaybookRepository(_db_session_factory())


@lru_cache
def _lookup_repository() -> LookupRepository:
    return SqlAlchemyLookupRepository(_db_session_factory())


@lru_cache
def _relationship_repository() -> RelationshipRepository:
    return SqlAlchemyRelationshipRepository(_db_session_factory())


@lru_cache
def _log_knowledge_repository() -> LogKnowledgeRepository:
    return SqlAlchemyLogKnowledgeRepository(_db_session_factory())


@lru_cache
def _knowledge_relationship_engine() -> KnowledgeRelationshipEngine:
    return KnowledgeRelationshipEngine(
        _relationship_repository(),
        _component_profile_repository(),
        _knowledge_repository(),
        _sql_template_repository(),
        _playbook_repository(),
        _lookup_repository(),
        _log_knowledge_repository(),
    )


@lru_cache
def _version_repository() -> VersionRepository:
    return SqlAlchemyVersionRepository(_db_session_factory())


@lru_cache
def _knowledge_object_adapters() -> dict[KnowledgeObjectType, KnowledgeObjectAdapter]:
    """One shared adapter registry (Sprint 3, Phase 3.4) -- built once
    and reused by both ``KnowledgeRelationshipEngine`` (Phase 3.3) and
    ``KnowledgeObjectService`` (this phase), so the id->repository
    dispatch exists exactly once in the whole app."""
    return build_adapters(
        _component_profile_repository(),
        _knowledge_repository(),
        _sql_template_repository(),
        _playbook_repository(),
        _lookup_repository(),
        _log_knowledge_repository(),
    )


@lru_cache
def _knowledge_object_service() -> KnowledgeObjectService:
    return KnowledgeObjectService(
        _knowledge_object_adapters(),
        _knowledge_relationship_engine(),
        _version_repository(),
        _knowledge_engine_singleton(),
    )


@lru_cache
def _task_importer() -> TaskImporter:
    return TaskImporter(_knowledge_object_service(), _knowledge_relationship_engine())


@lru_cache
def _log_wiki_importer() -> LogWikiImporter:
    return LogWikiImporter(_knowledge_object_service(), _knowledge_relationship_engine(), _component_profile_repository())


# --- Context Dimensions / Metadata Classification (2026-08-12) -------------


@lru_cache
def _classification_repository() -> ClassificationRepository:
    return SqlAlchemyClassificationRepository(_db_session_factory())


@lru_cache
def _document_classification_engine() -> DocumentClassificationEngine:
    return DocumentClassificationEngine(
        _knowledge_repository(),
        _lookup_repository(),
        _component_profile_repository(),
        _classification_repository(),
        _knowledge_object_service(),
        _knowledge_relationship_engine(),
    )


def get_document_classification_engine() -> DocumentClassificationEngine:
    return _document_classification_engine()


def get_lookup_repository() -> LookupRepository:
    return _lookup_repository()


@lru_cache
def _entity_extractor() -> EntityExtractor:
    return RegexEntityExtractor()


@lru_cache
def _log_parser() -> LogParser:
    return GenericLogParser(_entity_extractor())


@lru_cache
def _log_intelligence_engine() -> LogIntelligenceEngine:
    return LogIntelligenceEngine(_log_parser(), _entity_extractor())


@lru_cache
def _file_type_registry() -> FileTypeRegistry:
    return FileTypeRegistry()


@lru_cache
def _ingestion_engine() -> IngestionEngine:
    return IngestionEngine(_file_type_registry())


@lru_cache
def _component_registry() -> ComponentRegistry:
    return ComponentRegistry.load_from_repository(_component_profile_repository())


@lru_cache
def _product_intelligence_engine() -> ProductIntelligenceEngine:
    return ProductIntelligenceEngine(_component_registry())


# --- Public dependencies (used via FastAPI's Depends(...)) -----------------


def get_settings_dep() -> Settings:
    return get_settings()


def get_investigation_engine() -> InvestigationEngine:
    return InvestigationEngine(_investigation_repository(), _log_intelligence_engine(), _ingestion_engine())


def get_knowledge_engine() -> KnowledgeEngine:
    return _knowledge_engine_singleton()


@lru_cache
def _tfs_connector() -> TfsConnector:
    settings = get_settings()
    return TfsRestConnector(
        base_url=settings.tfs_base_url,
        project=settings.tfs_project,
        timeout_seconds=settings.external_knowledge_timeout_seconds,
    )


@lru_cache
def _wiki_connector() -> WikiConnector:
    settings = get_settings()
    return WikiRestConnector(
        base_url=settings.wiki_base_url,
        username=settings.wiki_api_username,
        api_token=settings.wiki_api_token,
        space_key=settings.wiki_space_key,
        timeout_seconds=settings.external_knowledge_timeout_seconds,
    )


@lru_cache
def _external_knowledge_service() -> ExternalKnowledgeService:
    return ExternalKnowledgeService(
        tfs_connector=_tfs_connector(),
        wiki_connector=_wiki_connector(),
        settings=get_settings(),
    )


def get_external_knowledge_service() -> ExternalKnowledgeService:
    return _external_knowledge_service()


def get_recommendation_engine() -> RecommendationEngine:
    return RecommendationEngine(
        _knowledge_engine_singleton(),
        get_settings(),
        _log_knowledge_repository(),
        _component_profile_repository(),
        _knowledge_relationship_engine(),
        _sql_library_engine(),
        _external_knowledge_service(),
        _lookup_repository(),
    )


def get_product_intelligence_engine() -> ProductIntelligenceEngine:
    return _product_intelligence_engine()


def get_sql_library_engine() -> SqlLibraryEngine:
    return _sql_library_engine()


# --- Chat (2026-08-14, Phase 3 Conversation State + Phase 4 Orchestrator) --


@lru_cache
def _chat_repository() -> ChatRepository:
    return SqlAlchemyChatRepository(_db_session_factory())


@lru_cache
def _query_understanding_engine() -> QueryUnderstandingEngine:
    return QueryUnderstandingEngine(_lookup_repository(), _component_profile_repository(), _entity_extractor())


def get_conversation_state_engine() -> ConversationStateEngine:
    return ConversationStateEngine(
        _chat_repository(), _query_understanding_engine(), _lookup_repository(), _investigation_repository()
    )


@lru_cache
def _llm_provider() -> LLMProvider:
    """Reused across requests (not constructed per call) -- same
    reasoning as _tfs_connector()/_wiki_connector(): a persistent
    httpx.Client benefits from connection reuse, same as TFS's own
    persistent NTLM session."""
    settings = get_settings()
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )


def get_chat_orchestrator() -> ChatOrchestrator:
    settings = get_settings()
    llm = _llm_provider() if settings.llm_enabled else None
    return ChatOrchestrator(
        get_conversation_state_engine(), get_recommendation_engine(), get_investigation_engine(), llm
    )


def get_knowledge_management_engine() -> KnowledgeManagementEngine:
    return _knowledge_management_engine()


def get_knowledge_relationship_engine() -> KnowledgeRelationshipEngine:
    return _knowledge_relationship_engine()


def get_log_knowledge_repository() -> LogKnowledgeRepository:
    return _log_knowledge_repository()


def get_knowledge_object_service() -> KnowledgeObjectService:
    return _knowledge_object_service()


def get_task_importer() -> TaskImporter:
    return _task_importer()


def get_log_wiki_importer() -> LogWikiImporter:
    return _log_wiki_importer()


def get_file_type_registry() -> FileTypeRegistry:
    return _file_type_registry()


def run_knowledge_foundation_migration() -> dict[str, int]:
    """Idempotent JSON/constant -> governed-table migration (Sprint 3,
    Phase 3.1, extended in Phase 3.3 with Product/Technology lookup
    seeding). Called once from the FastAPI lifespan, before Chroma
    seeding -- see ``app/api/main.py``. Safe to call on every startup;
    re-running never duplicates rows (see ``seed_migration.migrate_all``).
    """
    settings = get_settings()
    return migrate_all(
        component_repo=_component_profile_repository(),
        knowledge_repo=_knowledge_repository(),
        sql_repo=_sql_template_repository(),
        lookup_repo=_lookup_repository(),
        log_knowledge_repo=_log_knowledge_repository(),
        sample_knowledge_dir=settings.sample_knowledge_dir,
    )


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
        _file_type_registry,
        _ingestion_engine,
        _component_registry,
        _product_intelligence_engine,
        _component_profile_repository,
        _knowledge_repository,
        _sql_template_repository,
        _sql_library_engine,
        _knowledge_management_engine,
        _playbook_repository,
        _lookup_repository,
        _relationship_repository,
        _log_knowledge_repository,
        _knowledge_relationship_engine,
        _version_repository,
        _knowledge_object_adapters,
        _knowledge_object_service,
        _task_importer,
        _log_wiki_importer,
        _classification_repository,
        _document_classification_engine,
        _llm_provider,
    ):
        fn.cache_clear()
