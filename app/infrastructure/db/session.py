"""SQLAlchemy engine/session wiring (SQLite locally, any SQLAlchemy URL when hosted).

``create_engine`` is cached per URL (mirroring the Chroma client pattern)
so both the FastAPI app and test suite can create isolated engines without
a shared-state singleton getting in the way.
"""

from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.infrastructure.db.models import Base

logger = logging.getLogger(__name__)


@lru_cache
def get_engine(sqlite_url: str) -> Engine:
    logger.info("Connecting to database: %s", make_url(sqlite_url).render_as_string(hide_password=True))
    if sqlite_url.startswith("sqlite"):
        engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
    else:
        # Hosted database (e.g. PostgreSQL): pre-ping so a connection the
        # server/pooler dropped is replaced instead of failing a request.
        engine = create_engine(sqlite_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)
    _reconcile_orphaned_columns(engine)
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
            default_clause = _scalar_default_clause(column, engine.dialect.name)
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
                if column.index:
                    # ADD COLUMN never creates an index -- a gap found
                    # while adding evidence.content_hash (Investigation
                    # loading redesign): index=True on an additively-
                    # migrated column was silently a no-op for any DB
                    # that predates it, unlike a freshly create_all()'d
                    # table, which gets the index correctly. IF NOT
                    # EXISTS makes this safe to run every startup.
                    index_name = f"idx_{table.name}_{column.name}"
                    conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table.name}" ("{column.name}")'))
                    logger.warning("Created index %s on %s.%s (additive column)", index_name, table.name, column.name)


_ORPHANED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("chat_messages", "status"),
)
"""Explicit, narrow allowlist of ``(table, column)`` pairs known to be
physically present in a developer's existing SQLite file but absent
from the current ORM model -- the opposite direction from
``_add_missing_columns``. ``chat_messages.status`` (2026-08-18, Chat
Assistant Phases 1-4 fix): an earlier, never-committed local iteration
of ``ChatMessageModel`` briefly had a ``status`` column
(``index=True``, ``NOT NULL``, no default); the physical table it
created on this developer's ``data/resolveiq.db`` survived that
model edit even after the field was simplified out of
``ChatMessageModel`` before anything was committed -- ``create_all()``
never touches an existing table, and ``_add_missing_columns`` only
ever adds, never removes, so nothing in the existing migration path
could reconcile it. See ``app/domain/chat.py``/``ChatMessageModel``
themselves for the current, correct 8-column shape -- ``status`` has
no meaning anywhere in the current domain model or any caller.

Deliberately a short, explicit, hand-maintained list, not a generic
schema-diff engine: this function only ever considers a column if its
exact ``(table, column)`` pair is named here. Do not turn this into
automatic "drop anything the model doesn't recognize" behavior --
that would risk silently destroying a column added by hand or by a
tool this reconciliation doesn't know about."""


def _reconcile_orphaned_columns(engine: Engine) -> None:
    """The narrow, opposite-direction companion to
    ``_add_missing_columns``: removes a column named in
    ``_ORPHANED_COLUMNS`` when it is safe to do so, and never touches
    anything else. ``_add_missing_columns`` remains additive/general on
    purpose (see its own docstring) -- this function is intentionally
    the only place this codebase ever removes a column, and only for
    the exact, named pairs above.

    No-op, safely, in every case except the one it exists to fix:
    - Fresh database (table doesn't exist yet): no-op.
    - Table exists but the column is already absent (already
      reconciled, or never had it -- e.g. every other developer's/CI's
      database): no-op.
    - Table exists, column exists, but the table has >=1 row: refuses
      to touch it and logs a warning instead of guessing. This
      reconciliation only ever acts on a table it can prove is empty --
      a non-empty legacy table needs a human decision, not an
      automatic column drop, and there is no data in this specific
      case (``chat_messages`` has 0 rows) that this guard would ever
      need to protect *today*, but the guard exists so this function
      can never become a silent data-loss risk if ever reused for a
      future orphaned column that isn't empty.

    When it does act: drops any index that references the orphaned
    column first (SQLite's ``ALTER TABLE ... DROP COLUMN`` errors if
    the column is still indexed), then drops the column itself. Uses
    SQLite's native ``DROP COLUMN`` (available since SQLite 3.35.0;
    this project's runtime is verified >=3.35.0 -- see the version
    guard below) rather than the create-copy-swap dance
    ``test_db_migration_helpers.py`` uses to *simulate* an old schema
    for tests -- that dance is a test fixture technique, not something
    this function needs to reproduce; a real, single ``DROP COLUMN``
    is simpler and exactly as safe once the guarding index is gone.
    """
    if engine.dialect.name != "sqlite":
        # The orphaned column is local SQLite file drift (see
        # ``_ORPHANED_COLUMNS``); a hosted database never had it.
        return
    if sqlite3.sqlite_version_info < (3, 35, 0):
        logger.warning(
            "Skipping orphaned-column reconciliation: SQLite %s predates DROP COLUMN support (3.35.0+) -- "
            "no schema change made, existing tables are unaffected either way.",
            sqlite3.sqlite_version,
        )
        return

    inspector = inspect(engine)
    for table_name, column_name in _ORPHANED_COLUMNS:
        if not inspector.has_table(table_name):
            continue  # fresh database -- create_all() never made this column in the first place

        existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
        if column_name not in existing_columns:
            continue  # already reconciled, or this database never had the orphaned column

        with engine.connect() as conn:
            row_count = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
        if row_count:
            logger.warning(
                "Found orphaned column %s.%s but the table has %d row(s) -- refusing to remove it "
                "automatically. This needs a manual decision, not an automatic column drop.",
                table_name,
                column_name,
                row_count,
            )
            continue

        orphaned_indexes = [
            index["name"] for index in inspector.get_indexes(table_name) if column_name in (index.get("column_names") or [])
        ]
        with engine.begin() as conn:
            for index_name in orphaned_indexes:
                conn.execute(text(f'DROP INDEX IF EXISTS "{index_name}"'))
                logger.warning("Dropped orphaned index %s (on %s.%s)", index_name, table_name, column_name)
            conn.execute(text(f'ALTER TABLE "{table_name}" DROP COLUMN "{column_name}"'))
            logger.warning(
                "Dropped orphaned column %s.%s (table was empty -- safe, no data lost)", table_name, column_name
            )


def _scalar_default_clause(column, dialect_name: str = "sqlite") -> str:
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
        if dialect_name == "postgresql":
            return f"DEFAULT {'TRUE' if value else 'FALSE'}"  # PostgreSQL booleans reject 1/0
        return f"DEFAULT {1 if value else 0}"
    if isinstance(value, (int, float)):
        return f"DEFAULT {value}"
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"DEFAULT '{escaped}'"
    return ""


def get_session_factory(sqlite_url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(sqlite_url), expire_on_commit=False)
