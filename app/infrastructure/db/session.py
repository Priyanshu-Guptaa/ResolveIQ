"""SQLAlchemy engine/session wiring for SQLite.

``create_engine`` is cached per URL (mirroring the Chroma client pattern)
so both the FastAPI app and test suite can create isolated engines without
a shared-state singleton getting in the way.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sqlalchemy import Engine, create_engine, inspect, text
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
    _add_missing_columns(engine)
    return engine


def _add_missing_columns(engine: Engine) -> None:
    """Additive-only schema safety net: ``create_all`` creates missing
    *tables* but never alters existing ones, so a column added to an ORM
    model (e.g. ``last_viewed_at`` this sprint) is invisible to a SQLite
    file created by an earlier version of the app until something issues
    ``ALTER TABLE ... ADD COLUMN``.

    This sprint's database is explicitly a dev/early-adoption choice with
    a documented future migration to PostgreSQL/SQL Server (see README) --
    pulling in Alembic for it now would be solving a problem the project
    doesn't have yet. This is the smallest safe fix: additive only (never
    drops or alters existing columns), and it runs once at startup.
    """
    inspector = inspect(engine)
    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        existing_columns = {col["name"] for col in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing_columns:
                continue
            ddl_type = column.type.compile(engine.dialect)
            logger.warning(
                "Adding missing column %s.%s (%s) -- existing DB predates this field",
                table.name,
                column.name,
                ddl_type,
            )
            with engine.begin() as conn:
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl_type}'))


def get_session_factory(sqlite_url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(sqlite_url), expire_on_commit=False)
