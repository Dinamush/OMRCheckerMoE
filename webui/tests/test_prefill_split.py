"""Tests for the optional "split combined PDF into smaller segments" feature.

Coverage:

*   ``_chunk_ranges_for_split`` greedy chunker honours both byte and page caps.
*   ``_format_part_name`` collapses to ``{stem}.pdf`` for single-segment outputs.
*   ``_split_pdf_into_segments`` produces real PyMuPDF documents whose
    concatenated page ranges reproduce the source and whose individual sizes
    sit within the requested cap.
*   ``generate_batch_split_pdf_to_zip`` wraps the segments in a ZIP and keeps
    page numbering continuous across them (no resets to 1).
*   ``generate_batch_grouped_split_zip_to_file`` splits each per-group PDF
    independently while preserving continuous numbering inside each group.
*   ``/api/v1/prefill/batch`` is opt-in: omitting ``split_pdfs`` keeps the
    legacy single-PDF flat path, ``split_pdfs=true`` routes to the new
    splitter and returns ``split_pdfs=True`` in the JSON response.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from webui.services import prefill as prefill_service


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_chunk_ranges_size_bound_splits_at_first_page_over_cap() -> None:
    # Each "page" is 10 bytes. Cap of 25 bytes -> max 2 pages per segment
    # (the third page would push the segment to 30 > 25).
    sizes = [10, 10, 10, 10, 10]
    ranges = prefill_service._chunk_ranges_for_split(
        sizes, max_bytes=25, max_pages=999
    )
    assert ranges == [(0, 1), (2, 3), (4, 4)]


def test_chunk_ranges_page_bound_dominates_when_smaller() -> None:
    # Cap of 1000 bytes is unreachable; max_pages=3 should be the trigger.
    sizes = [10] * 10
    ranges = prefill_service._chunk_ranges_for_split(
        sizes, max_bytes=1000, max_pages=3
    )
    assert ranges == [(0, 2), (3, 5), (6, 8), (9, 9)]


def test_chunk_ranges_single_oversize_page_still_emitted_alone() -> None:
    # A single page bigger than the cap MUST still be emitted (otherwise
    # the splitter would lose data). Our greedy walker handles this by
    # only starting a fresh segment when the current segment has > 1 page.
    sizes = [100, 5, 5]
    ranges = prefill_service._chunk_ranges_for_split(
        sizes, max_bytes=50, max_pages=999
    )
    assert ranges == [(0, 0), (1, 2)]


def test_chunk_ranges_empty_input_returns_empty() -> None:
    assert prefill_service._chunk_ranges_for_split(
        [], max_bytes=10, max_pages=10
    ) == []


def test_format_part_name_collapses_single_segment() -> None:
    assert prefill_service._format_part_name("prefilled_sheets", 1, 1) == (
        "prefilled_sheets.pdf"
    )


def test_format_part_name_uses_two_digit_padded_indices() -> None:
    assert (
        prefill_service._format_part_name("prefilled_sheets", 2, 7)
        == "prefilled_sheets_part_02_of_07.pdf"
    )
    assert (
        prefill_service._format_part_name("prefilled_sheets", 12, 12)
        == "prefilled_sheets_part_12_of_12.pdf"
    )


# ---------------------------------------------------------------------------
# Real PyMuPDF splitter
# ---------------------------------------------------------------------------


def _build_synthetic_pdf(tmp_path: Path, *, pages: int, stamp_numbers: bool = True) -> Path:
    """Render a tiny multi-page PDF with each page labelled by its 1-based number.

    Using a tiny canvas (200 x 200 pt) keeps the file small while still
    being a valid PDF that ``insert_pdf`` can chunk. When ``stamp_numbers``
    is true each page contains its 1-based page number as plain text, so
    the test can later read the segments back and verify the labels are
    continuous (1, 2, 3, ...) across segment boundaries.
    """
    doc = fitz.open()
    for i in range(1, pages + 1):
        page = doc.new_page(width=200, height=200)
        if stamp_numbers:
            page.insert_text((20, 100), f"page {i}", fontsize=20)
    out = tmp_path / "synthetic.pdf"
    doc.save(str(out))
    doc.close()
    return out


def _read_text_per_page(pdf_path: Path) -> list[str]:
    """Return the trimmed text content of each page in ``pdf_path``."""
    doc = fitz.open(str(pdf_path))
    try:
        return [doc.load_page(i).get_text().strip() for i in range(doc.page_count)]
    finally:
        doc.close()


def test_split_pdf_into_segments_passthrough_when_under_caps(tmp_path: Path) -> None:
    src = _build_synthetic_pdf(tmp_path, pages=3)
    segments = prefill_service._split_pdf_into_segments(
        src, max_pdf_mb=100, max_pdf_pages=999
    )
    # Single-segment output returns the source path itself untouched.
    assert len(segments) == 1
    seg_path, first_page, last_page, _ = segments[0]
    assert seg_path == src
    assert (first_page, last_page) == (1, 3)


def test_split_pdf_into_segments_respects_page_cap(tmp_path: Path) -> None:
    src = _build_synthetic_pdf(tmp_path, pages=10)
    segments = prefill_service._split_pdf_into_segments(
        src, max_pdf_mb=100, max_pdf_pages=3
    )
    try:
        page_ranges = [(first, last) for _, first, last, _ in segments]
        assert page_ranges == [(1, 3), (4, 6), (7, 9), (10, 10)]

        # Verify each segment is a valid PDF whose pages contain the
        # ORIGINAL 1-based page label - i.e. numbering stays continuous
        # across the segments rather than resetting to 1 in each one.
        observed_labels: list[str] = []
        for seg_path, _, _, _ in segments:
            assert seg_path != src
            observed_labels.extend(_read_text_per_page(seg_path))
        assert observed_labels == [f"page {i}" for i in range(1, 11)]
    finally:
        for seg_path, _, _, _ in segments:
            if seg_path != src:
                seg_path.unlink(missing_ok=True)


def test_split_pdf_into_segments_writes_independent_files(tmp_path: Path) -> None:
    """Each segment is a standalone temp file that survives the helper exit."""
    src = _build_synthetic_pdf(tmp_path, pages=6)
    segments = prefill_service._split_pdf_into_segments(
        src, max_pdf_mb=100, max_pdf_pages=2
    )
    try:
        # 6 pages / 2 page cap -> 3 segments. Each must exist on disk and
        # be readable by fitz independently of the source PDF.
        assert len(segments) == 3
        for seg_path, _, _, size_bytes in segments:
            assert seg_path.exists()
            assert size_bytes > 0
            standalone = fitz.open(str(seg_path))
            try:
                assert standalone.page_count == 2
            finally:
                standalone.close()
    finally:
        for seg_path, _, _, _ in segments:
            if seg_path != src:
                seg_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# End-to-end batch entry points (use the real renderer)
# ---------------------------------------------------------------------------


def _csv_rows(n: int) -> list[dict[str, str]]:
    return [
        {
            "student_name": f"Student {i}",
            "school_name": "Riverview Primary",
            "exam_name": "Grade 5 Maths",
            "candidate_number": f"{1_000_000_000 + i:010d}",
        }
        for i in range(1, n + 1)
    ]


def test_generate_batch_split_pdf_to_zip_keeps_numbering_continuous(
    tmp_path: Path,
) -> None:
    rows = _csv_rows(5)
    dst = tmp_path / "out.zip"
    meta = prefill_service.generate_batch_split_pdf_to_zip(
        rows,
        dst,
        max_pdf_mb=200,            # big enough that pages, not size, drives the split
        max_pdf_pages=2,           # 5 rows / 2 pages per segment = 3 segments
        include_page_numbers=True,
    )
    assert meta["count"] == 5
    assert meta["successes"] == 5
    assert dst.exists() and dst.stat().st_size > 0

    segments = meta["segments"]
    assert [s["pages"] for s in segments] == [2, 2, 1]
    # first_page / last_page expose continuous numbering across the
    # segments, not a reset-to-1 per segment.
    assert [(s["first_page"], s["last_page"]) for s in segments] == [
        (1, 2),
        (3, 4),
        (5, 5),
    ]

    # The ZIP must contain exactly the segment files the metadata declares.
    with zipfile.ZipFile(dst) as zf:
        names = sorted(zf.namelist())
    expected = sorted(s["name"] for s in segments)
    assert names == expected
    assert all("_part_" in name and name.endswith(".pdf") for name in names)


def test_generate_batch_split_pdf_to_zip_collapses_single_segment_name(
    tmp_path: Path,
) -> None:
    """When the cap is never reached the ZIP holds one cleanly-named PDF."""
    rows = _csv_rows(2)
    dst = tmp_path / "out.zip"
    meta = prefill_service.generate_batch_split_pdf_to_zip(
        rows,
        dst,
        max_pdf_mb=200,
        max_pdf_pages=999,
        include_page_numbers=False,
        stem="prefilled_sheets",
    )
    with zipfile.ZipFile(dst) as zf:
        names = zf.namelist()
    assert names == ["prefilled_sheets.pdf"]
    assert meta["segments"][0]["name"] == "prefilled_sheets.pdf"


def test_generate_batch_grouped_split_zip_to_file_splits_per_group(
    tmp_path: Path,
) -> None:
    # Two regions, 4 rows total. Force every group to split into multi-segment
    # PDFs by capping at 1 page per segment so we can assert the segment
    # naming and per-group numbering rules.
    rows = [
        {
            "student_name": "Alice",
            "school_name": "Riverview Primary",
            "exam_name": "Grade 5 Maths",
            "candidate_number": "0000000001",
            "region": "Demerara",
        },
        {
            "student_name": "Bob",
            "school_name": "Riverview Primary",
            "exam_name": "Grade 5 Maths",
            "candidate_number": "0000000002",
            "region": "Demerara",
        },
        {
            "student_name": "Carol",
            "school_name": "Coastal High",
            "exam_name": "Grade 5 Maths",
            "candidate_number": "0000000003",
            "region": "Berbice",
        },
        {
            "student_name": "Dave",
            "school_name": "Coastal High",
            "exam_name": "Grade 5 Maths",
            "candidate_number": "0000000004",
            "region": "Berbice",
        },
    ]
    dst = tmp_path / "out.zip"
    meta = prefill_service.generate_batch_grouped_split_zip_to_file(
        rows,
        dst,
        group_by="region",
        max_pdf_mb=200,
        max_pdf_pages=1,
        include_page_numbers=True,
    )

    assert meta["count"] == 4
    assert meta["successes"] == 4

    with zipfile.ZipFile(dst) as zf:
        names = sorted(zf.namelist())

    # Each region had 2 rows + a 1-page-per-segment cap => 2 segments per
    # region. Names must include "_part_01_of_02" / "_part_02_of_02".
    assert names == [
        "Berbice_part_01_of_02.pdf",
        "Berbice_part_02_of_02.pdf",
        "Demerara_part_01_of_02.pdf",
        "Demerara_part_02_of_02.pdf",
    ]

    # Per-group page numbering must remain continuous across that group's
    # two segments (1, 2) rather than resetting to (1, 1).
    by_group = {g["name"]: g for g in meta["groups"]}
    for group_name in ("Demerara.pdf", "Berbice.pdf"):
        segs = by_group[group_name]["segments"]
        assert [(s["first_page"], s["last_page"]) for s in segs] == [(1, 1), (2, 2)]


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


_HEADERS = "student_name,school_name,exam_name,candidate_number"


def _csv_text(n: int) -> str:
    rows = [
        f"Student {i},Riverview,Grade 5,{1_000_000_000 + i:010d}"
        for i in range(1, n + 1)
    ]
    return "\n".join([_HEADERS, *rows])


def test_prefill_batch_split_disabled_by_default_uses_flat_pdf_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting ``split_pdfs`` MUST preserve the original single-PDF route."""
    called: dict[str, bool] = {"flat": False, "split": False}

    def fake_pdf(rows, dst_path, **kwargs):
        called["flat"] = True
        dst_path.write_bytes(b"%PDF-1.4\n%%EOF")
        return {
            "count": len(rows),
            "successes": len(rows),
            "errors": [],
            "elapsed_s": 0.01,
            "size_bytes": dst_path.stat().st_size,
        }

    def fake_split(*args, **kwargs):  # pragma: no cover - must not be called
        called["split"] = True
        raise AssertionError("split path should NOT run when split_pdfs is omitted")

    monkeypatch.setattr(prefill_service, "generate_batch_pdf_to_file", fake_pdf)
    monkeypatch.setattr(
        prefill_service, "generate_batch_split_pdf_to_zip", fake_split
    )

    response = client.post(
        "/api/v1/prefill/batch",
        data={"output_mode": "pdf", "csv_text": _csv_text(2)},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert called["flat"] is True
    assert called["split"] is False
    assert payload["split_pdfs"] is False
    assert payload["segments"] == []
    assert payload["filename"].endswith(".pdf")


def test_prefill_batch_split_enabled_routes_to_split_zip(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_split(
        rows,
        dst_path,
        *,
        max_pdf_mb,
        max_pdf_pages,
        realism_preset="none",
        include_page_numbers=False,
        marking_profile="none",
        answers=None,
        stem="prefilled_sheets",
    ):
        captured["max_pdf_mb"] = max_pdf_mb
        captured["max_pdf_pages"] = max_pdf_pages
        captured["include_page_numbers"] = include_page_numbers
        # Write a tiny but valid ZIP so the download token system is happy.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{stem}_part_01_of_02.pdf", b"%PDF-1.4")
            zf.writestr(f"{stem}_part_02_of_02.pdf", b"%PDF-1.4")
        dst_path.write_bytes(buf.getvalue())
        return {
            "count": len(rows),
            "successes": len(rows),
            "errors": [],
            "elapsed_s": 0.02,
            "size_bytes": dst_path.stat().st_size,
            "segments": [
                {
                    "name": f"{stem}_part_01_of_02.pdf",
                    "first_page": 1,
                    "last_page": 1,
                    "pages": 1,
                    "size_bytes": 32,
                },
                {
                    "name": f"{stem}_part_02_of_02.pdf",
                    "first_page": 2,
                    "last_page": 2,
                    "pages": 1,
                    "size_bytes": 32,
                },
            ],
        }

    monkeypatch.setattr(
        prefill_service, "generate_batch_split_pdf_to_zip", fake_split
    )

    response = client.post(
        "/api/v1/prefill/batch",
        data={
            "output_mode": "pdf",
            "csv_text": _csv_text(2),
            "split_pdfs": "true",
            "include_page_numbers": "true",
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["split_pdfs"] is True
    assert payload["filename"].endswith("_split.zip")
    assert payload["split_max_pdf_mb"] == 100          # operator default
    assert payload["split_max_pdf_pages"] == 500       # operator default
    assert captured["max_pdf_mb"] == 100
    assert captured["max_pdf_pages"] == 500
    assert captured["include_page_numbers"] is True
    assert len(payload["segments"]) == 2
    assert [
        (s["first_page"], s["last_page"]) for s in payload["segments"]
    ] == [(1, 1), (2, 2)]


def test_prefill_batch_split_with_flat_zip_output_silently_ignored(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``output_mode=zip`` + no grouping has no PDF to split; flag is dropped."""
    called: dict[str, bool] = {"zip": False, "split": False}

    def fake_zip(rows, dst_path, **kwargs):
        called["zip"] = True
        dst_path.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
        return {
            "count": len(rows),
            "successes": len(rows),
            "errors": [],
            "elapsed_s": 0.01,
            "size_bytes": dst_path.stat().st_size,
        }

    def fake_split(*args, **kwargs):  # pragma: no cover
        called["split"] = True
        raise AssertionError("split must not run for flat PNG ZIP output")

    monkeypatch.setattr(prefill_service, "generate_batch_zip_to_file", fake_zip)
    monkeypatch.setattr(
        prefill_service, "generate_batch_split_pdf_to_zip", fake_split
    )

    response = client.post(
        "/api/v1/prefill/batch",
        data={
            "output_mode": "zip",
            "csv_text": _csv_text(2),
            "split_pdfs": "true",
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert called["zip"] is True
    assert called["split"] is False
    assert payload["split_pdfs"] is False
    assert payload["segments"] == []


def test_prefill_batch_grouped_split_routes_to_grouped_split_zip(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_grouped_split(
        rows,
        dst_path,
        *,
        group_by,
        max_pdf_mb,
        max_pdf_pages,
        realism_preset="none",
        include_page_numbers=False,
        marking_profile="none",
        answers=None,
    ):
        captured["group_by"] = group_by
        captured["max_pdf_mb"] = max_pdf_mb
        captured["max_pdf_pages"] = max_pdf_pages
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("Demerara_part_01_of_02.pdf", b"%PDF-1.4")
            zf.writestr("Demerara_part_02_of_02.pdf", b"%PDF-1.4")
        dst_path.write_bytes(buf.getvalue())
        return {
            "count": len(rows),
            "successes": len(rows),
            "errors": [],
            "elapsed_s": 0.03,
            "size_bytes": dst_path.stat().st_size,
            "groups": [
                {
                    "name": "Demerara.pdf",
                    "count": 2,
                    "successes": 2,
                    "errors": [],
                    "segments": [
                        {
                            "name": "Demerara_part_01_of_02.pdf",
                            "first_page": 1,
                            "last_page": 1,
                            "pages": 1,
                            "size_bytes": 32,
                        },
                        {
                            "name": "Demerara_part_02_of_02.pdf",
                            "first_page": 2,
                            "last_page": 2,
                            "pages": 1,
                            "size_bytes": 32,
                        },
                    ],
                }
            ],
        }

    monkeypatch.setattr(
        prefill_service,
        "generate_batch_grouped_split_zip_to_file",
        fake_grouped_split,
    )

    csv_text = (
        "student_name,school_name,exam_name,candidate_number,region\n"
        "Alice,Riverview,Test,0000000001,Demerara\n"
        "Bob,Riverview,Test,0000000002,Demerara\n"
    )
    response = client.post(
        "/api/v1/prefill/batch",
        data={
            "output_mode": "pdf",  # Overridden by grouping.
            "group_by": "region",
            "split_pdfs": "true",
            "csv_text": csv_text,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["split_pdfs"] is True
    assert payload["group_by"] == "region"
    assert payload["filename"].endswith("_by_region_split.zip")
    assert captured["group_by"] == "region"
    # Grouped+split surfaces segments under each group's metadata, NOT at
    # the top level (the top-level ``segments`` list is reserved for the
    # ungrouped split path).
    assert payload["segments"] == []
    assert payload["groups"][0]["segments"][0]["first_page"] == 1
    assert payload["groups"][0]["segments"][1]["first_page"] == 2
