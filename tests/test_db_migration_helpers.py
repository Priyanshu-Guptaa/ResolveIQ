"""Tests for ``_add_missing_columns`` (Sprint 3, Phase 3.4).

Found and fixed a real bug while adding Phase 3.4's ``status`` column
to eight existing tables: the additive-column migration generated
``ALTER TABLE ... ADD COLUMN col TYPE`` with no ``DEFAULT`` clause, so
every *pre-existing* row got NULL instead of the column's configured
default -- silently contradicting what every reader of that field
expects (e.g. "every SQL template defaults to Published"). New rows
were never affected (they go through the ORM, which applies Python-side
defaults correctly); only rows already in the database when a new
defaulted column is added hit this path.

Runs against the real ``sql_templates`` table (real production schema,
not a synthetic stand-in) with the ``status``/``is_active`` columns
stripped out via raw SQL first, simulating exactly what a database that
predates Phase 3.4 looks like.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from sqlalchemy import create_engine, text

from app.infrastructure.db.session import _reconcile_orphaned_columns, get_engine

_LEGACY_CHAT_MESSAGES_DDL = (
    "CREATE TABLE chat_messages ("
    "id VARCHAR(36) NOT NULL, session_id VARCHAR(36) NOT NULL, sequence INTEGER NOT NULL, "
    "role VARCHAR(20) NOT NULL, content TEXT NOT NULL, parsed_query JSON, reference_resolution JSON, "
    "created_at DATETIME NOT NULL, status VARCHAR(20) NOT NULL, PRIMARY KEY (id), "
    "CONSTRAINT uq_chat_messages_session_sequence UNIQUE (session_id, sequence), "
    "FOREIGN KEY(session_id) REFERENCES chat_sessions (id))"
)
"""The real, live shape found on a developer's existing ``data/resolveiq.db``
during the Chat Assistant Phases 1-4 investigation -- an earlier,
never-committed local iteration of ``ChatMessageModel`` briefly had a
``status`` column (indexed, NOT NULL, no default); the physical table
it created survived that model edit even after ``status`` was removed
from the model before anything was committed. Used below to simulate
exactly that drifted database, the same "swap the real table for an
older shape via raw sqlite3" technique the tests above already use."""

_CURRENT_CHAT_MESSAGES_COLUMNS = {
    "id",
    "session_id",
    "sequence",
    "role",
    "content",
    "parsed_query",
    "reference_resolution",
    "created_at",
}
"""The current, correct ``ChatMessageModel`` column set -- no ``status``."""


def test_add_missing_columns_backfills_the_configured_default_for_existing_rows():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        sqlite_url = f"sqlite:///{db_path.as_posix()}"

        # Step 1: build the real schema once (creates every table,
        # including sql_templates with its current, full column set)...
        engine = get_engine(sqlite_url)
        engine.dispose()
        get_engine.cache_clear()

        # ...then simulate a pre-Phase-3.4 database by physically
        # removing the columns this test cares about and inserting a
        # row the old-fashioned way, bypassing the ORM entirely.
        raw_conn = sqlite3.connect(str(db_path))
        raw_conn.execute(
            "CREATE TABLE sql_templates_old AS SELECT id, title, category, sql_text, explanation, tags, "
            "created_at, updated_at, created_by, updated_by FROM sql_templates"
        )
        raw_conn.execute("DROP TABLE sql_templates")
        raw_conn.execute("ALTER TABLE sql_templates_old RENAME TO sql_templates")
        raw_conn.execute(
            "INSERT INTO sql_templates (id, title, category, sql_text, explanation, tags, created_at, "
            "updated_at, created_by, updated_by) VALUES "
            "('t1', 'Old Template', 'general', 'SELECT 1', 'test', '[]', '2020-01-01', '2020-01-01', NULL, NULL)"
        )
        raw_conn.commit()
        raw_conn.close()

        # Step 2: re-run the real migration path -- this is what every
        # app startup does (get_engine -> create_all -> _add_missing_columns).
        engine = get_engine(sqlite_url)
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT status, is_active FROM sql_templates WHERE id = 't1'")
            ).fetchone()

        assert row.status == "published"  # not NULL -- this is the bug that was found and fixed
        assert row.is_active == 1

        engine.dispose()
        get_engine.cache_clear()


def test_add_missing_columns_creates_an_index_for_additively_migrated_indexed_columns():
    """Found while adding evidence.content_hash (Investigation loading
    redesign): ``ALTER TABLE ... ADD COLUMN`` never creates an index --
    a column declared ``index=True`` only actually gets one on a
    freshly ``create_all()``'d table, silently not on any database that
    predates the column. Runs against the real ``evidence`` table with
    ``content_hash`` stripped out, simulating a pre-existing database."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        sqlite_url = f"sqlite:///{db_path.as_posix()}"

        engine = get_engine(sqlite_url)
        engine.dispose()
        get_engine.cache_clear()

        raw_conn = sqlite3.connect(str(db_path))
        raw_conn.execute(
            "CREATE TABLE evidence_old AS SELECT id, investigation_id, evidence_type, source, title, "
            "raw_content, created_at, extracted_entities, log_events, evidence_metadata FROM evidence"
        )
        raw_conn.execute("DROP TABLE evidence")
        raw_conn.execute("ALTER TABLE evidence_old RENAME TO evidence")
        raw_conn.commit()
        raw_conn.close()

        engine = get_engine(sqlite_url)
        with engine.connect() as conn:
            indexes = conn.execute(text('PRAGMA index_list("evidence")')).fetchall()
        index_names = {row[1] for row in indexes}
        assert "idx_evidence_content_hash" in index_names

        engine.dispose()
        get_engine.cache_clear()


# =============================================================================
# _reconcile_orphaned_columns (2026-08-18, Chat Assistant Phases 1-4 fix) --
# the narrow, opposite-direction companion to _add_missing_columns above.
# Removes only the exact, hand-allowlisted (table, column) pairs in
# app.infrastructure.db.session._ORPHANED_COLUMNS, and only when it can
# prove doing so is safe. See that module's own docstrings for the full
# rationale; these tests exercise every branch named in its docstring.
# =============================================================================


def test_reconcile_orphaned_columns_removes_legacy_chat_messages_status_when_empty():
    """A: legacy schema (status column + its index present, table empty)
    -- the exact real-world case found on a developer's existing
    database. Verifies status and its index are both gone afterward,
    and every current ChatMessageModel column survives intact."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        sqlite_url = f"sqlite:///{db_path.as_posix()}"

        # Step 1: build the real, current schema once...
        engine = get_engine(sqlite_url)
        engine.dispose()
        get_engine.cache_clear()

        # ...then simulate the drifted database: swap chat_messages for
        # the legacy shape (status column + its index), no rows.
        raw_conn = sqlite3.connect(str(db_path))
        raw_conn.execute("DROP TABLE chat_messages")
        raw_conn.execute(_LEGACY_CHAT_MESSAGES_DDL)
        raw_conn.execute("CREATE INDEX ix_chat_messages_status ON chat_messages (status)")
        raw_conn.execute("CREATE INDEX ix_chat_messages_session_id ON chat_messages (session_id)")
        raw_conn.commit()
        raw_conn.close()

        # Step 2: re-run the real migration path -- exactly what every
        # app startup does (get_engine -> create_all -> _add_missing_columns
        # -> _reconcile_orphaned_columns).
        engine = get_engine(sqlite_url)
        with engine.connect() as conn:
            columns = {row[1] for row in conn.execute(text('PRAGMA table_info("chat_messages")')).fetchall()}
            indexes = {row[1] for row in conn.execute(text('PRAGMA index_list("chat_messages")')).fetchall()}

        assert "status" not in columns
        assert "ix_chat_messages_status" not in indexes
        assert columns == _CURRENT_CHAT_MESSAGES_COLUMNS

        engine.dispose()
        get_engine.cache_clear()


def test_reconcile_orphaned_columns_is_a_noop_on_already_current_schema():
    """B: a database already on the current schema (no status column)
    -- running the reconciliation must not error and must not change
    anything, whether that's a brand-new database or one already
    reconciled by a prior run."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        sqlite_url = f"sqlite:///{db_path.as_posix()}"

        engine = get_engine(sqlite_url)
        with engine.connect() as conn:
            before = {row[1] for row in conn.execute(text('PRAGMA table_info("chat_messages")')).fetchall()}
        engine.dispose()
        get_engine.cache_clear()

        # Re-run the real migration path again on the same, already-current
        # database -- must be a pure no-op.
        engine = get_engine(sqlite_url)
        with engine.connect() as conn:
            after = {row[1] for row in conn.execute(text('PRAGMA table_info("chat_messages")')).fetchall()}

        assert before == after == _CURRENT_CHAT_MESSAGES_COLUMNS

        engine.dispose()
        get_engine.cache_clear()


def test_reconcile_orphaned_columns_refuses_to_touch_a_non_empty_legacy_table():
    """C: a legacy table that actually has data in it. The reconciliation
    must refuse to remove the column and must leave the existing row
    completely untouched -- fail safe rather than silently destroy
    data, exactly as designed."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        sqlite_url = f"sqlite:///{db_path.as_posix()}"

        engine = get_engine(sqlite_url)
        engine.dispose()
        get_engine.cache_clear()

        raw_conn = sqlite3.connect(str(db_path))
        raw_conn.execute("DROP TABLE chat_messages")
        raw_conn.execute(_LEGACY_CHAT_MESSAGES_DDL)
        raw_conn.execute("CREATE INDEX ix_chat_messages_status ON chat_messages (status)")
        # A real chat_sessions row first, so the FK reference is genuine.
        raw_conn.execute(
            "INSERT INTO chat_sessions (id, investigation_id, slots, created_at, updated_at) VALUES "
            "('s1', NULL, '{}', '2020-01-01', '2020-01-01')"
        )
        raw_conn.execute(
            "INSERT INTO chat_messages (id, session_id, sequence, role, content, created_at, status) "
            "VALUES ('m1', 's1', 1, 'user', 'hello', '2020-01-01', 'sent')"
        )
        raw_conn.commit()
        raw_conn.close()

        engine = get_engine(sqlite_url)
        with engine.connect() as conn:
            columns = {row[1] for row in conn.execute(text('PRAGMA table_info("chat_messages")')).fetchall()}
            row = conn.execute(
                text("SELECT id, content, status FROM chat_messages WHERE id = :id"), {"id": "m1"}
            ).fetchone()

        assert "status" in columns  # refused to remove it -- table wasn't empty
        assert row is not None
        assert row.content == "hello"
        assert row.status == "sent"  # completely untouched

        engine.dispose()
        get_engine.cache_clear()


def test_reconcile_orphaned_columns_is_a_noop_for_a_brand_new_database():
    """D (part 1): the common real-world path -- a database that has
    never existed before. create_all() creates chat_messages in its
    current shape; reconciliation must be a no-op, not an error."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        sqlite_url = f"sqlite:///{db_path.as_posix()}"

        engine = get_engine(sqlite_url)
        with engine.connect() as conn:
            columns = {row[1] for row in conn.execute(text('PRAGMA table_info("chat_messages")')).fetchall()}

        assert columns == _CURRENT_CHAT_MESSAGES_COLUMNS

        engine.dispose()
        get_engine.cache_clear()


def test_reconcile_orphaned_columns_is_a_noop_when_the_table_does_not_exist_yet():
    """D (part 2): exercises the has_table()-False branch directly.
    Through get_engine(), create_all() always creates chat_messages
    immediately before this function ever runs, so the only way to
    observe this branch is to call the helper directly against an
    engine with no tables at all."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        bare_engine = create_engine(f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False})

        _reconcile_orphaned_columns(bare_engine)  # must not raise

        with bare_engine.connect() as conn:
            tables = {row[0] for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()}
        assert "chat_messages" not in tables

        bare_engine.dispose()
