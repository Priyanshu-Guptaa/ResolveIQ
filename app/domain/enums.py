"""Enumerations shared across the ResolveIQ domain model.

These enums are intentionally framework-agnostic (plain ``enum.Enum``) so the
domain layer has zero dependency on FastAPI, SQLAlchemy, or any other
infrastructure concern.
"""

from __future__ import annotations

from enum import Enum


class EvidenceType(str, Enum):
    """The kind of source material an :class:`~app.domain.evidence.Evidence`
    item was derived from.

    Every artifact ResolveIQ ingests -- a ServiceNow task, a log file, a wiki
    page, a bug report -- is normalized into Evidence. This enum records
    *what it originally was* so engines can apply source-specific handling
    (e.g. only the Log Intelligence Engine parses ``LOG_FILE`` evidence)
    while the Investigation Engine treats all evidence uniformly.
    """

    TASK_DESCRIPTION = "task_description"
    LOG_FILE = "log_file"
    MANUAL_NOTE = "manual_note"
    HISTORICAL_INVESTIGATION = "historical_investigation"
    DOCUMENTATION = "documentation"
    BUG_REPORT = "bug_report"
    SQL_RESULT = "sql_result"


class EntityType(str, Enum):
    """Recognized entity types the Log Intelligence Engine extracts from
    free text / log content.

    This list intentionally spans application, infrastructure, and
    device-communication domains -- ResolveIQ must not assume every
    investigation is about smart meters.
    """

    THREAD_ID = "thread_id"
    PROCESS_ID = "process_id"
    SESSION_ID = "session_id"
    ENDPOINT_ID = "endpoint_id"
    METER_NUMBER = "meter_number"
    SERIAL_NUMBER = "serial_number"
    COMMAND_LOG_ID = "command_log_id"
    REQUEST_ID = "request_id"
    CORRELATION_ID = "correlation_id"
    EXCEPTION_TYPE = "exception_type"
    STACK_TRACE = "stack_trace"
    SQL_SESSION = "sql_session"
    POD_NAME = "pod_name"
    HOST_NAME = "host_name"
    IP_ADDRESS = "ip_address"
    SERVICE_NAME = "service_name"
    KAFKA_TOPIC = "kafka_topic"
    RABBITMQ_QUEUE = "rabbitmq_queue"
    CONSUMER_GROUP = "consumer_group"
    URL = "url"
    API_ENDPOINT = "api_endpoint"
    USER_ID = "user_id"
    EVENT_ID = "event_id"


class LogLevel(str, Enum):
    """Normalized log severity, independent of the source format's spelling
    (e.g. ``WARN`` vs ``WARNING``)."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"
    UNKNOWN = "UNKNOWN"


class InvestigationStatus(str, Enum):
    """Lifecycle state of an :class:`~app.domain.investigation.InvestigationSession`."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class ObjectLifecycleStatus(str, Enum):
    """Lifecycle state shared by every governed knowledge object (Sprint
    3, Phase 3.4 -- Knowledge Object Framework).

    Originated in Phase 3.2 as ``DocumentStatus`` (Documentation only);
    Phase 3.4 generalizes it onto ``GovernanceFields`` itself, so every
    governed entity (Component, Known Bug, SQL Template, Historical
    Investigation, Playbook, Product, Technology, Version -- not just
    Documentation) gets the same four/five-state lifecycle instead of
    each inventing its own. ``UNDER_REVIEW`` and ``DEPRECATED`` are
    defined now (cheap) but nothing transitions an object into either
    yet -- RFC-003's full Draft -> Under Review -> Approved -> Published
    -> Archived approval workflow is still a later phase; this is that
    same "keep the design extensible" discipline, now with the
    generalized shape "compatible with future approval workflows"
    (Phase 3.4's own wording) actually requires.

    Only ``PUBLISHED`` documents are indexed/searchable by the Knowledge
    Engine -- unchanged from Phase 3.2, and true for Documentation only;
    the other eight object types have no search index of their own to
    gate (they're relationship-graph nodes, not full-text search
    targets), so their status only governs the Knowledge Object
    Framework's own visibility/lifecycle UI, not a second index.
    """

    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


DocumentStatus = ObjectLifecycleStatus
"""Backward-compatible alias -- Phase 3.2's ``DocumentStatus`` is the
exact same enum as Phase 3.4's ``ObjectLifecycleStatus``, just named
before its scope generalized. Existing imports of ``DocumentStatus``
(``evidence.py``, ``seed_migration.py``, the Knowledge Management
engine/router) keep working unchanged; new code should prefer
``ObjectLifecycleStatus``, the name that now actually describes it."""


class KnowledgeCollection(str, Enum):
    """Logical ChromaDB collections the Knowledge Engine searches over.

    Keeping this as an enum (rather than raw strings scattered through the
    codebase) means adding a new knowledge source later is a one-line change.
    """

    HISTORICAL_INVESTIGATIONS = "historical_investigations"
    DOCUMENTATION = "documentation"
    KNOWN_BUGS = "known_bugs"
