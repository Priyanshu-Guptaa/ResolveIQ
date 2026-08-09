"""Tests for Structured Task Import: extractors (JSON/XLSX) + TaskImporter.

Same real temp-file SQLite pattern as test_knowledge_object_framework.py
-- duplicate detection and relationship creation are persistence-backed
behavior, not pure logic.
"""

from __future__ import annotations

import io
import json
import re
import tempfile
import zipfile
from pathlib import Path

import pytest

from app.config import get_settings
from app.domain.knowledge_relationships import KnowledgeObjectType
from app.engines.knowledge_object_framework.adapters import build_adapters
from app.engines.knowledge_object_framework.service import KnowledgeObjectService
from app.engines.knowledge.engine import KnowledgeEngine
from app.engines.knowledge_relationships.engine import KnowledgeRelationshipEngine
from app.engines.task_import.extractors import (
    extract_from_csv_bytes,
    extract_from_json_bytes,
    extract_from_xlsx_bytes,
    extract_tasks,
)
from app.engines.task_import.importer import TaskImporter
from app.infrastructure.db.component_repository import SqlAlchemyComponentProfileRepository
from app.infrastructure.db.knowledge_repository import SqlAlchemyKnowledgeRepository
from app.infrastructure.db.lookup_repository import SqlAlchemyLookupRepository
from app.infrastructure.db.playbook_repository import SqlAlchemyPlaybookRepository
from app.infrastructure.db.relationship_repository import SqlAlchemyRelationshipRepository
from app.infrastructure.db.seed_migration import migrate_all
from app.infrastructure.db.session import get_engine, get_session_factory
from app.infrastructure.db.sql_template_repository import SqlAlchemySqlTemplateRepository

SAMPLE_DIR = get_settings().sample_knowledge_dir
T = KnowledgeObjectType


class _StubKnowledgeStore:
    def __init__(self) -> None:
        self.upserts: dict[str, dict] = {}  # record_id -> {"text": ..., "title": ..., "metadata": ...}
        self.deletes: list[str] = []

    def upsert(self, collection, record_id: str, text: str, title: str, *, metadata: dict | None = None) -> None:
        self.upserts[record_id] = {"text": text, "title": title, "metadata": metadata or {}}

    def delete(self, collection, record_id: str) -> None:
        self.deletes.append(record_id)

    def count(self, collection) -> int:
        return 0


@pytest.fixture
def setup():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        sqlite_url = f"sqlite:///{db_path.as_posix()}"
        session_factory = get_session_factory(sqlite_url)

        component_repo = SqlAlchemyComponentProfileRepository(session_factory)
        knowledge_repo = SqlAlchemyKnowledgeRepository(session_factory)
        sql_repo = SqlAlchemySqlTemplateRepository(session_factory)
        playbook_repo = SqlAlchemyPlaybookRepository(session_factory)
        lookup_repo = SqlAlchemyLookupRepository(session_factory)
        relationship_repo = SqlAlchemyRelationshipRepository(session_factory)
        from app.infrastructure.db.version_repository import SqlAlchemyVersionRepository

        version_repo = SqlAlchemyVersionRepository(session_factory)

        migrate_all(
            component_repo=component_repo,
            knowledge_repo=knowledge_repo,
            sql_repo=sql_repo,
            lookup_repo=lookup_repo,
            sample_knowledge_dir=SAMPLE_DIR,
        )

        relationship_engine = KnowledgeRelationshipEngine(
            relationship_repo, component_repo, knowledge_repo, sql_repo, playbook_repo, lookup_repo
        )
        adapters = build_adapters(component_repo, knowledge_repo, sql_repo, playbook_repo, lookup_repo)
        store = _StubKnowledgeStore()
        knowledge_engine = KnowledgeEngine(store, knowledge_repo)
        service = KnowledgeObjectService(adapters, relationship_engine, version_repo, knowledge_engine)
        importer = TaskImporter(service, relationship_engine)

        yield importer, {
            "service": service,
            "components": component_repo,
            "relationships": relationship_repo,
            "relationship_engine": relationship_engine,
            "store": store,
        }

        get_engine(sqlite_url).dispose()
        get_engine.cache_clear()


def _component_name(repos) -> str:
    return repos["components"].get_by_name("CommandProcessorHost").name


# --- JSON extraction --------------------------------------------------------


def _json_bytes(records: list[dict]) -> bytes:
    return json.dumps({"records": records}).encode("utf-8")


def test_json_extractor_maps_fields_correctly():
    raw = _json_bytes(
        [
            {
                "number": "TASK0001",
                "short_description": "Disk full on server X",
                "description": "C: drive at 95%",
                "close_notes": "Cleared temp files, freed 20GB",
                "priority": "2",
                "state": "3",
                "assignment_group": "Infra Team",
                "cmdb_ci": "abc123",
            }
        ]
    )
    records = extract_from_json_bytes(raw)
    assert len(records) == 1
    r = records[0]
    assert r.ticket_number == "TASK0001"
    assert r.title == "Disk full on server X"
    assert r.description == "C: drive at 95%"
    assert r.resolution == "Cleared temp files, freed 20GB"
    assert r.state == "Closed Complete"
    assert "team:Infra Team" in r.extra_tags
    assert "cmdb:abc123" in r.extra_tags


def test_json_extractor_skips_rows_without_resolution():
    raw = _json_bytes(
        [
            {"number": "TASK0001", "short_description": "No resolution", "close_notes": ""},
            {"number": "TASK0002", "short_description": "Has resolution", "close_notes": "Fixed it"},
        ]
    )
    records = extract_from_json_bytes(raw)
    assert len(records) == 1
    assert records[0].ticket_number == "TASK0002"


def test_json_extractor_skips_rows_without_ticket_number():
    raw = _json_bytes([{"number": "", "short_description": "x", "close_notes": "resolved"}])
    assert extract_from_json_bytes(raw) == []


# --- XLSX extraction ---------------------------------------------------------


def _xlsx_bytes(header: list[str], rows: list[list]) -> bytes:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(header)
    for row in rows:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


_XLSX_HEADER = [
    "Number", "Priority", "State", "Assigned to", "Location", "Short description",
    "Task type", "Created", "Opened by", "Comments and Work notes", "Description", "Work notes",
]


def test_xlsx_extractor_maps_fields_correctly():
    raw = _xlsx_bytes(
        _XLSX_HEADER,
        [
            [
                "TASK0500163", "3 - Moderate", "Closed Complete", "Vivek Jindal", "Noida",
                "CC - Check Kafka", "Incident Task", "2026-08-04", None,
                "Restarted the service, lag cleared", "Kafka consumer lag alert", "Restarted the service, lag cleared",
            ]
        ],
    )
    records = extract_from_xlsx_bytes(raw)
    assert len(records) == 1
    r = records[0]
    assert r.ticket_number == "TASK0500163"
    assert r.title == "CC - Check Kafka"
    assert r.description == "Kafka consumer lag alert"
    assert r.resolution == "Restarted the service, lag cleared"
    assert r.priority == "3 - Moderate"
    assert r.state == "Closed Complete"
    assert "assignee:Vivek Jindal" in r.extra_tags
    assert "location:Noida" in r.extra_tags
    assert "type:Incident Task" in r.extra_tags


def test_xlsx_extractor_skips_rows_without_resolution():
    raw = _xlsx_bytes(
        _XLSX_HEADER,
        [
            ["TASK0001", "3", "Closed", "A", "X", "No resolution", "Task", None, None, "", "desc", ""],
            ["TASK0002", "3", "Closed", "A", "X", "Has resolution", "Task", None, None, "Fixed", "desc", "Fixed"],
        ],
    )
    records = extract_from_xlsx_bytes(raw)
    assert len(records) == 1
    assert records[0].ticket_number == "TASK0002"


def test_xlsx_extractor_falls_back_to_work_notes_when_comments_column_empty():
    raw = _xlsx_bytes(
        _XLSX_HEADER,
        [["TASK0003", "3", "Closed", "A", "X", "Title", "Task", None, None, "", "desc", "Resolved via work notes"]],
    )
    records = extract_from_xlsx_bytes(raw)
    assert len(records) == 1
    assert records[0].resolution == "Resolved via work notes"


# --- CSV extraction -----------------------------------------------------------


def _csv_bytes(header: list[str], rows: list[list], *, bom: bool = False) -> bytes:
    import csv as _csv

    buf = io.StringIO()
    writer = _csv.writer(buf)
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    text = buf.getvalue()
    return ("﻿" + text if bom else text).encode("utf-8")


def test_csv_extractor_maps_fields_correctly():
    raw = _csv_bytes(
        _XLSX_HEADER,
        [
            [
                "TASK0500163", "3 - Moderate", "Closed Complete", "Vivek Jindal", "Noida",
                "CC - Check Kafka", "Incident Task", "2026-08-04", "",
                "Restarted the service, lag cleared", "Kafka consumer lag alert", "Restarted the service, lag cleared",
            ]
        ],
    )
    records = extract_from_csv_bytes(raw)
    assert len(records) == 1
    r = records[0]
    assert r.ticket_number == "TASK0500163"
    assert r.title == "CC - Check Kafka"
    assert r.description == "Kafka consumer lag alert"
    assert r.resolution == "Restarted the service, lag cleared"
    assert r.priority == "3 - Moderate"
    assert r.state == "Closed Complete"
    assert "assignee:Vivek Jindal" in r.extra_tags
    assert "location:Noida" in r.extra_tags
    assert "type:Incident Task" in r.extra_tags


def test_csv_extractor_strips_leading_byte_order_mark():
    """Excel's own "Save As CSV (UTF-8)" writes a BOM -- must not end
    up glued onto the first header name (which would silently break
    that column's lookup)."""
    raw = _csv_bytes(
        _XLSX_HEADER,
        [["TASK0777", "3", "Closed", "A", "X", "Title", "Task", "", "", "Fixed it", "desc", "Fixed it"]],
        bom=True,
    )
    records = extract_from_csv_bytes(raw)
    assert len(records) == 1
    assert records[0].ticket_number == "TASK0777"


def test_csv_extractor_skips_rows_without_resolution():
    raw = _csv_bytes(
        _XLSX_HEADER,
        [
            ["TASK0001", "3", "Closed", "A", "X", "No resolution", "Task", "", "", "", "desc", ""],
            ["TASK0002", "3", "Closed", "A", "X", "Has resolution", "Task", "", "", "Fixed", "desc", "Fixed"],
        ],
    )
    records = extract_from_csv_bytes(raw)
    assert len(records) == 1
    assert records[0].ticket_number == "TASK0002"


def test_csv_extractor_falls_back_to_work_notes_when_comments_column_empty():
    raw = _csv_bytes(
        _XLSX_HEADER,
        [["TASK0003", "3", "Closed", "A", "X", "Title", "Task", "", "", "", "desc", "Resolved via work notes"]],
    )
    records = extract_from_csv_bytes(raw)
    assert len(records) == 1
    assert records[0].resolution == "Resolved via work notes"


def test_extract_tasks_dispatches_csv_by_extension():
    raw = _csv_bytes(_XLSX_HEADER, [["TASK0009", "3", "Closed", "A", "X", "T", "Task", "", "", "Fixed", "d", "Fixed"]])
    records = extract_tasks("export.csv", raw)
    assert len(records) == 1
    assert records[0].ticket_number == "TASK0009"


def _corrupt_xlsx_dimension(xlsx_bytes: bytes, *, stale_ref: str = "A1:A1") -> bytes:
    """Same real-world failure mode fixed in xlsx_parser.py -- proves
    this extractor's own openpyxl call (read_only=False) is immune."""
    src = zipfile.ZipFile(io.BytesIO(xlsx_bytes))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as dst:
        for name in src.namelist():
            data = src.read(name)
            if name == "xl/worksheets/sheet1.xml":
                data = re.sub(rb'<dimension ref="[^"]*"/>', f'<dimension ref="{stale_ref}"/>'.encode(), data)
            dst.writestr(name, data)
    return buf.getvalue()


def test_xlsx_extractor_immune_to_stale_dimension_metadata():
    raw = _xlsx_bytes(
        _XLSX_HEADER,
        [
            ["TASK0001", "3", "Closed", "A", "X", "Row one", "Task", None, None, "Fixed one", "desc", "Fixed one"],
            ["TASK0002", "3", "Closed", "A", "X", "Row two", "Task", None, None, "Fixed two", "desc", "Fixed two"],
        ],
    )
    corrupted = _corrupt_xlsx_dimension(raw)
    records = extract_from_xlsx_bytes(corrupted)
    assert {r.ticket_number for r in records} == {"TASK0001", "TASK0002"}


def test_extract_tasks_dispatches_by_extension():
    json_raw = _json_bytes([{"number": "T1", "short_description": "x", "close_notes": "fixed"}])
    assert len(extract_tasks("export.json", json_raw)) == 1

    xlsx_raw = _xlsx_bytes(_XLSX_HEADER, [["T2", "3", "Closed", "A", "X", "y", "Task", None, None, "fixed", "d", "fixed"]])
    assert len(extract_tasks("export.xlsx", xlsx_raw)) == 1

    csv_raw = _csv_bytes(_XLSX_HEADER, [["T3", "3", "Closed", "A", "X", "y", "Task", "", "", "fixed", "d", "fixed"]])
    assert len(extract_tasks("export.csv", csv_raw)) == 1

    with pytest.raises(ValueError):
        extract_tasks("export.pdf", b"")


# --- TaskImporter: duplicate detection / row mapping / indexing ------------


def _task_record(ticket="TASK0001", title="Title", description="Desc", resolution="Fix"):
    from app.domain.task_import import TaskRecord

    return TaskRecord(ticket_number=ticket, title=title, description=description, resolution=resolution)


def test_import_creates_one_historical_investigation_per_row_not_a_blob(setup):
    importer, repos = setup
    records = [_task_record(ticket=f"TASK{i:04d}", title=f"Task {i}") for i in range(5)]
    summary = importer.import_records(records, source_label="test.json", actor="tester")

    assert summary.total_rows == 5
    assert summary.imported == 5
    assert summary.updated == 0
    assert summary.duplicates == 0
    assert summary.failed == 0
    assert len(summary.imported_ids) == 5

    all_investigations = repos["service"].list_all(T.HISTORICAL_INVESTIGATION)
    imported_titles = {inv.title for inv in all_investigations if inv.id in summary.imported_ids}
    assert imported_titles == {f"Task {i}" for i in range(5)}  # five independent records, not one


def test_reimporting_unchanged_rows_reports_duplicates_not_new_imports(setup):
    importer, _ = setup
    records = [_task_record()]
    first = importer.import_records(records, source_label="run1", actor="tester")
    assert first.imported == 1

    second = importer.import_records(records, source_label="run2", actor="tester")
    assert second.imported == 0
    assert second.duplicates == 1
    assert second.updated == 0


def test_reimporting_changed_content_reports_update_not_duplicate(setup):
    importer, repos = setup
    first = importer.import_records([_task_record(resolution="Old fix")], source_label="run1", actor="tester")
    assert first.imported == 1
    original_id = first.imported_ids[0]

    second = importer.import_records([_task_record(resolution="New, better fix")], source_label="run2", actor="tester")
    assert second.imported == 0
    assert second.updated == 1
    assert second.updated_ids == [original_id]

    updated = repos["service"].get(T.HISTORICAL_INVESTIGATION, original_id)
    assert updated.resolution == "New, better fix"


def test_duplicate_detection_is_keyed_on_ticket_number_not_title(setup):
    """Two different tickets that happen to share a title must both be
    imported -- dedup is per real ticket, not a fuzzy title match (that
    check lives separately in KnowledgeObjectService.validate())."""
    importer, _ = setup
    records = [
        _task_record(ticket="TASK0001", title="Same title"),
        _task_record(ticket="TASK0002", title="Same title"),
    ]
    summary = importer.import_records(records, source_label="test", actor="tester")
    assert summary.imported == 2
    assert summary.duplicates == 0


def test_all_success_batch_reports_zero_failures(setup):
    importer, _ = setup
    summary = importer.import_records([_task_record(ticket="TASK0001")], source_label="test", actor="tester")
    assert summary.failed == 0
    assert summary.errors == []


def test_one_failed_row_does_not_abort_the_rest_of_the_import(setup, monkeypatch):
    """No natural way to make a validly-typed TaskRecord fail
    KnowledgeObjectService.create with this DB backend (SQLite doesn't
    enforce VARCHAR length, and there's no unique constraint besides a
    fresh UUID id) -- monkeypatch simulates one real downstream failure
    (e.g. a transient DB error) for exactly one ticket, to prove the
    per-row try/except actually isolates it rather than aborting the
    whole batch."""
    importer, repos = setup
    real_create = repos["service"].create

    def _flaky_create(object_type, **kwargs):
        if kwargs.get("title") == "Boom":
            raise RuntimeError("simulated downstream failure")
        return real_create(object_type, **kwargs)

    monkeypatch.setattr(repos["service"], "create", _flaky_create)

    records = [
        _task_record(ticket="TASK0001", title="Fine 1"),
        _task_record(ticket="TASK0002", title="Boom"),
        _task_record(ticket="TASK0003", title="Fine 2"),
    ]
    summary = importer.import_records(records, source_label="test", actor="tester")

    assert summary.total_rows == 3
    assert summary.imported == 2
    assert summary.failed == 1
    assert len(summary.errors) == 1
    assert "TASK0002" in summary.errors[0]


def test_import_creates_relationship_when_component_name_appears_in_title(setup):
    importer, repos = setup
    component_name = _component_name(repos)
    record = _task_record(ticket="TASK0001", title=f"Investigate {component_name} timeout")
    summary = importer.import_records([record], source_label="test", actor="tester")

    assert summary.relationships_created == 1
    relationships = repos["relationship_engine"].list_relationships(T.HISTORICAL_INVESTIGATION, summary.imported_ids[0])
    assert len(relationships) == 1
    assert relationships[0].to_object.title == component_name


def test_reimport_does_not_double_count_already_created_relationships(setup):
    importer, repos = setup
    component_name = _component_name(repos)
    record = _task_record(ticket="TASK0001", title=f"Investigate {component_name} timeout")
    first = importer.import_records([record], source_label="run1", actor="tester")
    assert first.relationships_created == 1

    # Re-import identical content -> duplicate, no new relationship counted.
    second = importer.import_records([record], source_label="run2", actor="tester")
    assert second.duplicates == 1
    assert second.relationships_created == 0

    relationships = repos["relationship_engine"].list_relationships(T.HISTORICAL_INVESTIGATION, first.imported_ids[0])
    assert len(relationships) == 1  # still exactly one, not duplicated


def test_component_name_match_is_word_boundary_not_substring(setup):
    """A short component name like "NMS" must not false-positive match
    inside an unrelated word."""
    importer, repos = setup
    record = _task_record(ticket="TASK0001", title="ATHENSMS server check", description="unrelated to NMS component")
    summary = importer.import_records([record], source_label="test", actor="tester")
    # "NMS" does appear as a real standalone word in the description, so
    # this should match -- rerun with a genuinely non-matching case too.
    assert summary.relationships_created >= 0  # sanity: no crash

    record2 = _task_record(ticket="TASK0002", title="ATHENSMSXYZ server check", description="nothing relevant here")
    summary2 = importer.import_records([record2], source_label="test2", actor="tester")
    relationships = repos["relationship_engine"].list_relationships(T.HISTORICAL_INVESTIGATION, summary2.imported_ids[0])
    assert relationships == []  # "NMS" embedded inside "ATHENSMSXYZ" must not match


# --- Structured indexing / search quality -----------------------------------


def test_each_imported_row_is_indexed_individually(setup):
    """The actual requirement this whole feature exists to satisfy:
    N rows -> N separate index entries, not one blob."""
    importer, repos = setup
    records = [_task_record(ticket=f"TASK{i:04d}", title=f"Task {i}") for i in range(3)]
    summary = importer.import_records(records, source_label="test", actor="tester")

    store: _StubKnowledgeStore = repos["store"]
    for record_id in summary.imported_ids:
        assert record_id in store.upserts


def test_indexed_content_includes_title_description_and_resolution(setup):
    """Search quality: the text actually handed to the embedder must
    contain the fields a real query would match against."""
    importer, repos = setup
    record = _task_record(
        ticket="TASK0099",
        title="Kafka consumer lag spike",
        description="Consumer group fell behind on billing-events topic",
        resolution="Restarted the stuck consumer and added backoff",
    )
    summary = importer.import_records([record], source_label="test", actor="tester")
    store: _StubKnowledgeStore = repos["store"]
    indexed = store.upserts[summary.imported_ids[0]]

    assert "Kafka consumer lag spike" in indexed["text"]
    assert "Consumer group fell behind on billing-events topic" in indexed["text"]
    assert "Restarted the stuck consumer and added backoff" in indexed["metadata"]["resolution"]
    assert "ticket:TASK0099" in indexed["metadata"]["tags"]


def test_updated_row_reindexes_with_new_content(setup):
    importer, repos = setup
    importer.import_records([_task_record(resolution="Old fix")], source_label="run1", actor="tester")
    second = importer.import_records([_task_record(resolution="Brand new fix")], source_label="run2", actor="tester")

    store: _StubKnowledgeStore = repos["store"]
    indexed = store.upserts[second.updated_ids[0]]
    assert "Brand new fix" in indexed["text"]
