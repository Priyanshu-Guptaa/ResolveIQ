"""Structured Task Import endpoints.

One upload -> extraction (format-detected) -> TaskImporter -> summary.
Deliberately separate from Knowledge Management's document upload
(``/admin/knowledge/documents/upload``): that endpoint always creates
one Document per file (a searchable blob); this endpoint never creates
a Document at all -- it decomposes the file straight into individual
Historical Investigation records. The two are independent operations
that happen to accept similar file types.

See ``app/api/routers/admin/__init__.py`` for why there's no role check
yet.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.api.dependencies import get_task_importer
from app.domain.task_import import TaskImportSummary
from app.engines.task_import.extractors import extract_tasks
from app.engines.task_import.importer import TaskImporter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/task-import", tags=["admin-task-import"])


@router.post("", response_model=TaskImportSummary, status_code=201)
async def import_tasks(
    file: UploadFile,
    actor: str = "admin",
    importer: TaskImporter = Depends(get_task_importer),
) -> TaskImportSummary:
    """Accepts one .json, .xlsx, or .csv ServiceNow task export. Every
    row becomes its own Historical Investigation (never a single blob
    document) -- duplicate detection, versioning, and search indexing
    all reuse existing machinery (see importer.py's docstring)."""
    filename = file.filename or "upload"
    raw_bytes = await file.read()
    try:
        records = extract_tasks(filename, raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 -- malformed JSON/XLSX, not our bug
        raise HTTPException(status_code=422, detail=f"Could not parse {filename}: {exc}") from exc

    return importer.import_records(records, source_label=filename, actor=actor)
