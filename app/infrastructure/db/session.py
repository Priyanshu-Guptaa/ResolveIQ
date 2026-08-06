"""SQLAlchemy engine/session wiring for SQLite.

``create_engine`` is cached per URL (mirroring the Chroma client pattern)
so both the FastAPI app and test suite can create isolated engines without
a shared-state singleton getting in the way.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.infrastructure.db.models import Base

logger = logging.getLogger(__name__)


@lru_cache
def get_engine(sqlite_url: str) -> Engine:
    logger.info("Connecting to database: %s", sqlite_url)
    engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(sqlite_url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(sqlite_url), expire_on_commit=False)
