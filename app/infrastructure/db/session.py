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

    Includes the column's configured scalar ``default`` (if any) in the
    generated DDL as SQLite's own ``DEFAULT`` clause -- found and fixed
    in Phase 3.4: without it, ``ALTER TABLE ... ADD COLUMN`` leaves every
    *existing* row's new column NULL even when the ORM column declares
    e.g. ``default="published"``, silently contradicting what every
    caller of that field reasonably expects. New rows were never
    affected (INSERT always went through the ORM, which applies Python-
    side defaults correctly) -- only pre-existing rows backfilled by this
    exact code path were ever wrong, which is exactly the path Phase
    3.4's new ``status`` column on eight existing tables relies on.
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
            default_clause = _scalar_default_clause(column)
            logger.warning(
                "Adding missing column %s.%s (%s)%s -- existing DB predates this field",
                table.name,
                column.name,
                ddl_type,
                f" {default_clause}" if default_clause else "",
            )
            with engine.begin() as conn:
                conn.execute(
                    text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl_type} {default_clause}'.strip())
                )


def _scalar_default_clause(column) -> str:
    """Renders a SQLAlchemy column's configured Python-side scalar
    default as a SQLite ``DEFAULT ...`` clause, or ``""`` if the column
    has no default (or a non-scalar one, e.g. a callable like
    ``default_factory``-backed columns -- those can't be expressed as a
    static SQL literal and are left NULL, same as before this fix)."""
    default = column.default
    if default is None or not getattr(default, "is_scalar", False):
        return ""
    value = default.arg
    if isinstance(value, bool):
        return f"DEFAULT {1 if value else 0}"
    if isinstance(value, (int, float)):
        return f"DEFAULT {value}"
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"DEFAULT '{escaped}'"
    return ""


def get_session_factory(sqlite_url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(sqlite_url), expire_on_commit=False)
