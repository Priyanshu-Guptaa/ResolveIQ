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

from sqlalchemy import text

from app.infrastructure.db.session import get_engine


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
