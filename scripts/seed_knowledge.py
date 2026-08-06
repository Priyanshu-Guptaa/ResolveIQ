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

    embedding_provider = SentenceTransformerEmbeddingProvider(settings.embedding_model_name)
    store = ChromaKnowledgeStore(settings.chroma_persist_dir, embedding_provider)
    engine = KnowledgeEngine(store)

    counts = engine.seed_from_directory(settings.sample_knowledge_dir, force=args.force)
    logger.info("Done. Loaded: %s", counts)


if __name__ == "__main__":
    main()
