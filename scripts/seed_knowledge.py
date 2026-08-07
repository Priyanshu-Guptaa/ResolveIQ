"""Standalone CLI to (re)seed the Knowledge Engine from sample data.

The API also auto-seeds on startup (see ``app.api.main.lifespan``) when
collections are empty, so this script is mainly for explicitly forcing a
re-seed during development:

    python -m scripts.seed_knowledge --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.engines.knowledge.embedding_provider import SentenceTransformerEmbeddingProvider  # noqa: E402
from app.engines.knowledge.engine import KnowledgeEngine  # noqa: E402
from app.engines.knowledge.knowledge_store import ChromaKnowledgeStore  # noqa: E402
from app.infrastructure.db.component_repository import SqlAlchemyComponentProfileRepository  # noqa: E402
from app.infrastructure.db.knowledge_repository import SqlAlchemyKnowledgeRepository  # noqa: E402
from app.infrastructure.db.seed_migration import migrate_all  # noqa: E402
from app.infrastructure.db.session import get_session_factory  # noqa: E402
from app.infrastructure.db.sql_template_repository import SqlAlchemySqlTemplateRepository  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed ResolveIQ's Knowledge Engine from sample data.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-seed even if a collection already has data (upserts, so it's safe/idempotent per record).",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    session_factory = get_session_factory(settings.sqlite_url)
    component_repo = SqlAlchemyComponentProfileRepository(session_factory)
    knowledge_repo = SqlAlchemyKnowledgeRepository(session_factory)
    sql_repo = SqlAlchemySqlTemplateRepository(session_factory)

    # Sprint 3, Phase 3.1+: the governed tables are the source of truth
    # ChromaDB gets seeded from -- make sure they're populated first.
    migrate_all(
        component_repo=component_repo,
        knowledge_repo=knowledge_repo,
        sql_repo=sql_repo,
        sample_knowledge_dir=settings.sample_knowledge_dir,
    )

    embedding_provider = SentenceTransformerEmbeddingProvider(settings.embedding_model_name)
    store = ChromaKnowledgeStore(settings.chroma_persist_dir, embedding_provider)
    engine = KnowledgeEngine(store, knowledge_repo)

    counts = engine.seed_from_directory(settings.sample_knowledge_dir, force=args.force)
    logger.info("Done. Loaded: %s", counts)


if __name__ == "__main__":
    main()
