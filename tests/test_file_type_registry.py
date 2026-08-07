"""Tests for the Evidence Ingestion Pipeline.

The core regression test here is `test_binary_formats_never_decoded_as_utf8`
-- this is Problem 1 from the platform RFC ("DOCX files display PK headers,
XLSX files display XML internals, JPG files display binary JFIF data").
Every other test builds real fixture bytes with the same libraries the
parsers use, rather than static binary files checked into the repo.
"""

from __future__ import annotations

import io
import re
import zipfile

import pytest

from app.domain.ingestion import FileKind
from app.engines.ingestion.file_type_registry import FileTypeRegistry


@pytest.fixture
def registry() -> FileTypeRegistry:
    return FileTypeRegistry()


def _make_docx_bytes(text: str) -> bytes:
    from docx import Document

    doc = Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _make_xlsx_bytes(rows: list[list[str]]) -> bytes:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_pdf_bytes(text: str) -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    content = doc.tobytes()
    doc.close()
    return content


def _make_png_bytes() -> bytes:
    from PIL import Image

    image = Image.new("RGB", (100, 30), color="white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# --- The regression test that started this whole RFC ------------------------


def test_binary_formats_never_decoded_as_utf8(registry: FileTypeRegistry):
    """.docx, .xlsx, and .jpg bytes must never end up verbatim (as
    mangled UTF-8) in ParsedFile.text -- that's the exact bug (Problem 1)
    this pipeline exists to fix."""
    docx_bytes = _make_docx_bytes("Command Log ID: CMD-778213, firmware 13.63")
    parsed = registry.parse("Intermittent Results.docx", docx_bytes)[0]
    assert parsed.kind == FileKind.DOCX
    assert "PK" not in parsed.text[:2]  # zip magic bytes, not garbled into text
    assert "CMD-778213" in parsed.text

    xlsx_bytes = _make_xlsx_bytes([["MeterNumber", "Firmware"], ["105550273LG", "13.63"]])
    parsed = registry.parse("GLP Meters - Good DCW.xlsx", xlsx_bytes)[0]
    assert parsed.kind == FileKind.XLSX
    assert "[Content_Types].xml" not in parsed.text
    assert "105550273LG" in parsed.text

    png_bytes = _make_png_bytes()
    parsed = registry.parse("GetLP_Popup_Meter.png", png_bytes)[0]
    assert parsed.kind == FileKind.IMAGE
    assert "JFIF" not in parsed.text
    assert "\x00" not in parsed.text  # no raw binary leaked into text


# --- Per-parser correctness --------------------------------------------------


def test_text_parser_handles_plain_log():
    registry = FileTypeRegistry()
    parsed = registry.parse("app.log", b"2026-08-06 ERROR something broke")[0]
    assert parsed.kind == FileKind.TEXT
    assert "something broke" in parsed.text


def test_text_parser_pretty_prints_valid_json():
    registry = FileTypeRegistry()
    parsed = registry.parse("data.json", b'{"a": 1}')[0]
    assert parsed.kind == FileKind.TEXT
    assert not parsed.warnings
    assert '"a": 1' in parsed.text


def test_text_parser_flags_invalid_json_without_failing():
    registry = FileTypeRegistry()
    parsed = registry.parse("data.json", b"{not valid json")[0]
    assert parsed.kind == FileKind.TEXT
    assert any(w.severity == "warning" for w in parsed.warnings)
    assert "not valid json" in parsed.text  # stored as raw text, not dropped


def test_docx_parser_extracts_table_text():
    from docx import Document

    doc = Document()
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "CommandLogId"
    table.rows[0].cells[1].text = "CMD-99213"
    buf = io.BytesIO()
    doc.save(buf)

    registry = FileTypeRegistry()
    parsed = registry.parse("table.docx", buf.getvalue())[0]
    assert "CMD-99213" in parsed.text


def test_pptx_parser_extracts_slide_and_notes_text():
    """Sprint 3, Phase 3.2: Knowledge Management's PPTX support."""
    from pptx import Presentation

    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "GLP/DCW Mismatch Triage"
    slide.placeholders[1].text = "Verify firmware version first"
    slide.notes_slide.notes_text_frame.text = "Remember to check GEI timing"
    buf = io.BytesIO()
    presentation.save(buf)

    registry = FileTypeRegistry()
    parsed = registry.parse("triage.pptx", buf.getvalue())[0]

    assert parsed.kind == FileKind.PPTX
    assert "GLP/DCW Mismatch Triage" in parsed.text
    assert "Verify firmware version first" in parsed.text
    assert "Remember to check GEI timing" in parsed.text
    assert parsed.metadata["slide_count"] == 1
    assert parsed.metadata["notes_slide_count"] == 1


def test_pptx_parser_never_decoded_as_utf8_garbage():
    """Same Problem-1 regression guard as the .docx test below, applied
    to .pptx (also a zip of XML parts)."""
    from pptx import Presentation

    presentation = Presentation()
    buf = io.BytesIO()
    presentation.save(buf)

    registry = FileTypeRegistry()
    parsed = registry.parse("empty.pptx", buf.getvalue())[0]

    assert parsed.kind == FileKind.PPTX
    assert "PK\x03\x04" not in parsed.text


def test_xlsx_parser_reports_sheet_names():
    registry = FileTypeRegistry()
    parsed = registry.parse("book.xlsx", _make_xlsx_bytes([["x"]]))[0]
    assert parsed.metadata.get("sheet_names")


def _corrupt_xlsx_dimension(xlsx_bytes: bytes, *, stale_ref: str = "A1:A1") -> bytes:
    """Rewrites sheet1.xml's declared ``<dimension ref="...">`` to a
    range smaller than the real data -- reproduces the stale-dimension
    export (seen from a real ServiceNow report export) that made
    openpyxl's ``read_only=True`` mode silently iterate almost nothing.
    """
    src = zipfile.ZipFile(io.BytesIO(xlsx_bytes))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as dst:
        for name in src.namelist():
            data = src.read(name)
            if name == "xl/worksheets/sheet1.xml":
                data = re.sub(rb'<dimension ref="[^"]*"/>', f'<dimension ref="{stale_ref}"/>'.encode(), data)
            dst.writestr(name, data)
    return buf.getvalue()


def test_xlsx_parser_extracts_all_rows_even_with_stale_dimension_metadata():
    """Regression test: a workbook whose declared <dimension> undercounts
    its real data must still be fully extracted, not silently truncated
    to that declared range with no warning."""
    registry = FileTypeRegistry()
    real_bytes = _make_xlsx_bytes(
        [["Number", "Description"], ["TASK001", "First row"], ["TASK002", "Second row"], ["TASK003", "Third row"]]
    )
    corrupted = _corrupt_xlsx_dimension(real_bytes)
    parsed = registry.parse("stale_dimension.xlsx", corrupted)[0]
    assert "TASK001" in parsed.text
    assert "TASK002" in parsed.text
    assert "TASK003" in parsed.text


def test_pdf_parser_extracts_text():
    registry = FileTypeRegistry()
    parsed = registry.parse("report.pdf", _make_pdf_bytes("Firmware 13.66 / DCW 13.63 mismatch"))[0]
    assert parsed.kind == FileKind.PDF
    assert "13.63" in parsed.text
    assert parsed.metadata["page_count"] == 1


def test_image_ocr_degrades_gracefully_without_crashing():
    """Whether or not tesseract is installed on this machine, OCR must
    never raise -- it should return empty text with a warning instead."""
    registry = FileTypeRegistry()
    parsed = registry.parse("blank.png", _make_png_bytes())[0]
    assert parsed.kind == FileKind.IMAGE
    assert isinstance(parsed.text, str)  # didn't crash


def test_unsupported_extension_is_never_decoded():
    registry = FileTypeRegistry()
    binary_content = bytes(range(256))  # guaranteed invalid UTF-8 in places
    parsed = registry.parse("firmware.bin", binary_content)[0]
    assert parsed.kind == FileKind.UNSUPPORTED
    assert parsed.text == ""
    assert any(w.severity == "warning" for w in parsed.warnings)


# --- Zip recursion -----------------------------------------------------------


def test_zip_expands_into_one_parsedfile_per_entry():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("notes.txt", "hello from zip")
        zf.writestr("data.json", '{"x": 1}')
    registry = FileTypeRegistry()

    results = registry.parse("bundle.zip", buf.getvalue())

    assert len(results) == 2
    filenames = {r.filename for r in results}
    assert filenames == {"notes.txt", "data.json"}
    notes = next(r for r in results if r.filename == "notes.txt")
    assert "hello from zip" in notes.text


def test_corrupt_zip_does_not_crash():
    registry = FileTypeRegistry()
    results = registry.parse("bad.zip", b"not actually a zip file")
    assert len(results) == 1
    assert results[0].kind == FileKind.UNSUPPORTED
    assert any(w.severity == "error" for w in results[0].warnings)


def test_nested_zip_depth_is_capped():
    def _zip_of(entries: dict[str, bytes]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, data in entries.items():
                zf.writestr(name, data)
        return buf.getvalue()

    innermost = _zip_of({"leaf.txt": "deep"})
    for _ in range(5):  # deeper than _MAX_ZIP_DEPTH
        innermost = _zip_of({"inner.zip": innermost})

    registry = FileTypeRegistry()
    results = registry.parse("outer.zip", innermost)
    assert len(results) >= 1  # doesn't hang or crash; depth cap kicks in
