"""SQL Studio domain models.

Sprint 2 scope (RFC rev 3, SQL Studio): a real query library, saved/recent/
favourite query tracking, and query explanations -- everything except
execution against a live database, which stays out of scope until a later
sprint explicitly picks a connection/security model for it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class QueryTemplate(BaseModel):
    """A canned, parameterized diagnostic query. The static seed set below
    is the Query Library; ``QUERY_LIBRARY`` is imported by both the
    Recommendation Engine (Sprint 1's ``suggested_sql``) and SQL Studio
    (Phase 8) so the two surfaces never drift apart."""

    id: str
    title: str
    category: str
    sql_text: str
    explanation: str
    tags: list[str] = Field(default_factory=list)


class SavedQuery(BaseModel):
    """A query an engineer saved, optionally starting from a QueryTemplate.
    Distinct from QueryTemplate: this is per-engineer state, not shared
    library content."""

    id: str = Field(default_factory=_new_id)
    title: str
    sql_text: str
    is_favourite: bool = False
    source_template_id: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    last_used_at: datetime | None = None
    use_count: int = 0


QUERY_LIBRARY: list[QueryTemplate] = [
    QueryTemplate(
        id="qt-sql-blocking",
        title="SQL Server blocking session lookup",
        category="sql-server",
        sql_text=(
            "SELECT * FROM sys.dm_exec_requests\n"
            "WHERE session_id = :session_id;\n"
            "-- check blocking_session_id and wait_type"
        ),
        explanation="Finds the blocking session and its wait type for a given SQL session ID.",
        tags=["sql-server", "blocking", "deadlock"],
    ),
    QueryTemplate(
        id="qt-command-log-lookup",
        title="Command log lookup by Command Log ID",
        category="ami",
        sql_text="SELECT * FROM CommandLog WHERE CommandLogId = :command_log_id;",
        explanation="Retrieves the full command record (type, payload, response) for a "
        "CommandProcessorHost command log entry.",
        tags=["ami", "command-log", "command-processor-host"],
    ),
    QueryTemplate(
        id="qt-meter-comm-history",
        title="Meter communication history",
        category="ami",
        sql_text=(
            "SELECT TOP 100 * FROM MeterCommunicationLog\n"
            "WHERE MeterNumber = :meter_number\n"
            "ORDER BY CommunicationTimestamp DESC;"
        ),
        explanation="Last 100 communication attempts for a meter, most recent first -- "
        "useful for confirming whether commands are reaching the device at all.",
        tags=["ami", "meter", "collector"],
    ),
    QueryTemplate(
        id="qt-investigation-by-correlation",
        title="Find requests by correlation ID",
        category="application",
        sql_text=(
            "SELECT * FROM RequestLog\n"
            "WHERE CorrelationId = :correlation_id\n"
            "ORDER BY Timestamp;"
        ),
        explanation="Reconstructs a request's path across services by correlation ID -- "
        "same technique as the Log Intelligence Engine's correlation_id entity.",
        tags=["correlation-id", "microservices", "tracing"],
    ),
    QueryTemplate(
        id="qt-firmware-dcw-mismatch",
        title="Meters with firmware/DCW version mismatch",
        category="ami",
        sql_text=(
            "SELECT MeterNumber, FirmwareVersion, DcwVersion\n"
            "FROM MeterInventory\n"
            "WHERE FirmwareVersion <> ExpectedFirmwareForDcw(DcwVersion);"
        ),
        explanation="Surfaces meters where a firmware push landed without a matching GEI, "
        "the root cause behind the GLP/DCW mismatch case in RFC rev 2.",
        tags=["ami", "firmware", "dcw", "gei"],
    ),
]
"""Static seed library. Sprint 3 promotes this to a proper table if
engineers need to add their own templates; Sprint 2 keeps it code-defined
since it's small and reviewed alongside the app itself."""
