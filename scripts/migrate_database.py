"""Copy every table from one ResolveIQ database to another (e.g. the local
SQLite file -> a hosted PostgreSQL), so a hosted deployment starts with the
existing knowledge corpus instead of the small seed set.

The source is only ever read. The target must have no rows in any
ResolveIQ table (it is created on the fly); the script refuses otherwise,
so it can never merge into or overwrite live data:

    python -m scripts.migrate_database \\
        --source sqlite:///data/resolveiq.db \\
        --target postgresql+psycopg://user:pw@host/resolveiq

Naive datetimes coming out of SQLite are interpreted as UTC (that is what
the app wrote), since PostgreSQL ``timestamptz`` needs an explicit zone.
"""

from __future__ import annotations

import argparse
import sys
from datetime import timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import DateTime, create_engine, func, insert, inspect, select  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402

from app.infrastructure.db.models import Base  # noqa: E402
from app.infrastructure.db.session import get_engine  # noqa: E402

_BATCH = 500


def migrate(source_url: str, target_url: str) -> dict[str, tuple[int, int]]:
    """Returns ``{table: (source_rows, target_rows)}``."""
    if make_url(source_url) == make_url(target_url):
        raise SystemExit("Source and target are the same database.")

    source_db = make_url(source_url)
    if source_db.get_backend_name() == "sqlite" and source_db.database and not Path(source_db.database).exists():
        raise SystemExit(f"Source database file not found: {source_db.database}")  # create_engine would create it empty
    source = create_engine(source_url)
    # get_engine() creates the schema (and runs the additive-column safety
    # net) on the target exactly as the app would at startup.
    target = get_engine(target_url)

    source_tables = set(inspect(source).get_table_names())
    with target.connect() as conn:
        for table in Base.metadata.sorted_tables:
            if conn.execute(select(func.count()).select_from(table)).scalar():
                raise SystemExit(f"Target table '{table.name}' already has rows -- refusing to merge into existing data.")

    report: dict[str, tuple[int, int]] = {}
    for table in Base.metadata.sorted_tables:  # parents before children (FKs)
        if table.name not in source_tables:
            continue
        datetime_columns = [c.name for c in table.columns if isinstance(c.type, DateTime)]
        with source.connect() as src, target.begin() as dst:
            total = src.execute(select(func.count()).select_from(table)).scalar() or 0
            result = src.execution_options(stream_results=True).execute(select(table))
            while True:
                rows = result.fetchmany(_BATCH)
                if not rows:
                    break
                batch = []
                for row in rows:
                    record = dict(row._mapping)
                    for name in datetime_columns:
                        value = record.get(name)
                        if value is not None and value.tzinfo is None:
                            record[name] = value.replace(tzinfo=timezone.utc)
                    batch.append(record)
                dst.execute(insert(table), batch)
        with target.connect() as conn:
            copied = conn.execute(select(func.count()).select_from(table)).scalar() or 0
        report[table.name] = (total, copied)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a ResolveIQ database to an empty target database.")
    parser.add_argument("--source", required=True, help="SQLAlchemy URL of the database to read")
    parser.add_argument("--target", required=True, help="SQLAlchemy URL of the EMPTY database to write")
    args = parser.parse_args()

    report = migrate(args.source, args.target)
    ok = True
    for name, (before, after) in report.items():
        flag = "ok" if before == after else "MISMATCH"
        ok &= before == after
        print(f"{name}: {before} -> {after} [{flag}]")
    if not ok:
        raise SystemExit("Row counts differ -- do not use the target database.")
    print("Done.")


if __name__ == "__main__":
    main()
