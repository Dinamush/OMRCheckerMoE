"""Tests for the prefill batch grouping feature.

The ``/api/v1/prefill/batch`` endpoint accepts an optional ``group_by`` form
field with values ``school``, ``region``, or ``region_school``. When set,
the output is always a ZIP containing one PDF per group (regardless of the
``output_mode`` value).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webui.services import prefill as prefill_service


# ---------------------------------------------------------------------------
# Pure unit tests — group key derivation, filename sanitisation, normalisation.
# ---------------------------------------------------------------------------


def test_normalize_group_by_accepts_known_values() -> None:
    assert prefill_service.normalize_group_by(None) == "none"
    assert prefill_service.normalize_group_by("") == "none"
    assert prefill_service.normalize_group_by("None") == "none"
    assert prefill_service.normalize_group_by("school") == "school"
    assert prefill_service.normalize_group_by(" REGION ") == "region"
    assert prefill_service.normalize_group_by("region_school") == "region_school"


def test_normalize_group_by_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unknown group_by"):
        prefill_service.normalize_group_by("bogus")


def test_safe_group_filename_strips_path_traversal_chars() -> None:
    safe = prefill_service._safe_group_filename
    assert safe("Riverview / Primary") == "Riverview _ Primary"
    assert safe("../etc/passwd") == "etc_passwd"
    assert safe(r"C:\Windows\System32") == "C_Windows_System32"
    # NULL byte and control characters are stripped
    assert safe("Foo\x00Bar\x07") == "FooBar"


def test_safe_group_filename_falls_back_for_empty_input() -> None:
    safe = prefill_service._safe_group_filename
    assert safe("") == "_ungrouped"
    assert safe("   ") == "_ungrouped"
    assert safe(None) == "_ungrouped"
    assert safe("___") == "_ungrouped"
    assert safe("__", fallback="_unknown") == "_unknown"


def test_row_group_keys_for_school_uses_school_name() -> None:
    keys = prefill_service._row_group_keys(
        {"school_name": "Riverview Primary"}, "school"
    )
    assert keys == ("Riverview Primary",)


def test_row_group_keys_for_region_uses_region_field() -> None:
    keys = prefill_service._row_group_keys(
        {"school_name": "Ignored", "region": "Demerara"}, "region"
    )
    assert keys == ("Demerara",)


def test_row_group_keys_for_region_school_is_nested() -> None:
    keys = prefill_service._row_group_keys(
        {"school_name": "Riverview", "region": "Demerara"}, "region_school"
    )
    assert keys == ("Demerara", "Riverview")


def test_row_group_keys_missing_region_falls_into_unknown_bucket() -> None:
    keys = prefill_service._row_group_keys(
        {"school_name": "Riverview"}, "region_school"
    )
    assert keys == ("_unknown_region", "Riverview")


def test_group_zip_entry_name_appends_pdf_suffix() -> None:
    assert prefill_service._group_zip_entry_name(("Riverview",)) == "Riverview.pdf"
    assert (
        prefill_service._group_zip_entry_name(("Demerara", "Riverview"))
        == "Demerara/Riverview.pdf"
    )


def test_group_rows_preserves_insertion_order_within_bucket() -> None:
    rows = [
        {"school_name": "A", "candidate_number": "0000000001"},
        {"school_name": "B", "candidate_number": "0000000002"},
        {"school_name": "A", "candidate_number": "0000000003"},
    ]
    grouped = prefill_service._group_rows(rows, "school")
    assert list(grouped.keys()) == [("A",), ("B",)]
    a_indexes = [idx for idx, _ in grouped[("A",)]]
    assert a_indexes == [0, 2]


# ---------------------------------------------------------------------------
# Service-layer behaviour using a fake ``generate_batch_pdf_to_file``
# stub. This is the safest layer to exercise because it stays I/O-free and
# avoids the full Pillow + PDF round-trip per row.
# ---------------------------------------------------------------------------


def test_generate_batch_grouped_zip_writes_one_pdf_per_school(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two schools, three rows. Expect a ZIP with two PDF entries."""

    written: dict[Path, list[dict]] = {}

    def fake_generate(
        rows,
        dst_path,
        realism_preset="none",
        include_page_numbers=False,
        marking_profile="none",
        answers=None,
    ):
        written[dst_path] = list(rows)
        dst_path.write_bytes(b"%PDF-1.4\n%%EOF")
        return {
            "count": len(rows),
            "successes": len(rows),
            "errors": [],
            "elapsed_s": 0.01,
            "size_bytes": dst_path.stat().st_size,
        }

    monkeypatch.setattr(
        prefill_service, "generate_batch_pdf_to_file", fake_generate
    )

    rows = [
        {
            "student_name": "Alice",
            "school_name": "Riverview Primary",
            "exam_name": "Grade 5 Maths",
            "candidate_number": "0000000001",
        },
        {
            "student_name": "Bob",
            "school_name": "Riverview Primary",
            "exam_name": "Grade 5 Maths",
            "candidate_number": "0000000002",
        },
        {
            "student_name": "Carol",
            "school_name": "Hilltop Academy",
            "exam_name": "Grade 5 Maths",
            "candidate_number": "0000000003",
        },
    ]

    dst = tmp_path / "out.zip"
    meta = prefill_service.generate_batch_grouped_zip_to_file(
        rows, dst, group_by="school"
    )

    assert meta["count"] == 3
    assert meta["successes"] == 3
    assert dst.exists()

    with zipfile.ZipFile(dst) as zf:
        names = sorted(zf.namelist())
    assert names == ["Hilltop Academy.pdf", "Riverview Primary.pdf"]

    group_names = sorted(g["name"] for g in meta["groups"])
    assert group_names == ["Hilltop Academy.pdf", "Riverview Primary.pdf"]


def test_generate_batch_grouped_zip_nests_region_school(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_generate(rows, dst_path, **kwargs):
        dst_path.write_bytes(b"%PDF-1.4\n%%EOF")
        return {
            "count": len(rows),
            "successes": len(rows),
            "errors": [],
            "elapsed_s": 0.01,
            "size_bytes": dst_path.stat().st_size,
        }

    monkeypatch.setattr(
        prefill_service, "generate_batch_pdf_to_file", fake_generate
    )

    rows = [
        {
            "student_name": "Alice",
            "school_name": "Riverview Primary",
            "exam_name": "Test",
            "candidate_number": "0000000001",
            "region": "Demerara",
        },
        {
            "student_name": "Bob",
            "school_name": "Coastal School",
            "exam_name": "Test",
            "candidate_number": "0000000002",
            "region": "Berbice",
        },
    ]

    dst = tmp_path / "out.zip"
    meta = prefill_service.generate_batch_grouped_zip_to_file(
        rows, dst, group_by="region_school"
    )

    assert meta["successes"] == 2
    with zipfile.ZipFile(dst) as zf:
        names = sorted(zf.namelist())
    assert names == [
        "Berbice/Coastal School.pdf",
        "Demerara/Riverview Primary.pdf",
    ]


def test_generate_batch_grouped_zip_rejects_group_by_none(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-'none' group_by"):
        prefill_service.generate_batch_grouped_zip_to_file(
            [], tmp_path / "out.zip", group_by="none"
        )


# ---------------------------------------------------------------------------
# API-layer tests: the endpoint correctly dispatches to grouped or flat
# generation, returns the expected payload shape, and validates ``region``
# column presence when needed.
# ---------------------------------------------------------------------------


_DEFAULT_HEADERS = "student_name,school_name,exam_name,candidate_number"


def _csv(rows: list[str], with_region: bool = False) -> str:
    header = (
        f"{_DEFAULT_HEADERS},region" if with_region else _DEFAULT_HEADERS
    )
    return "\n".join([header, *rows])


def test_prefill_batch_dispatches_to_grouped_zip_when_group_by_school(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    def fake_generate(
        rows,
        dst_path,
        *,
        group_by,
        realism_preset="none",
        include_page_numbers=False,
        marking_profile="none",
        answers=None,
    ):
        calls["count"] = len(rows)
        calls["group_by"] = group_by
        calls["include_page_numbers"] = include_page_numbers
        # Write a tiny but valid-looking ZIP so the download token can be created.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("Riverview.pdf", b"%PDF-1.4")
            zf.writestr("Hilltop.pdf", b"%PDF-1.4")
        dst_path.write_bytes(buf.getvalue())
        return {
            "count": len(rows),
            "successes": len(rows),
            "errors": [],
            "elapsed_s": 0.01,
            "size_bytes": dst_path.stat().st_size,
            "groups": [
                {"name": "Riverview.pdf", "count": 2, "successes": 2, "errors": []},
                {"name": "Hilltop.pdf", "count": 1, "successes": 1, "errors": []},
            ],
        }

    monkeypatch.setattr(
        prefill_service, "generate_batch_grouped_zip_to_file", fake_generate
    )

    csv_text = _csv(
        [
            "Alice,Riverview Primary,Grade 5 Maths,0000000001",
            "Bob,Riverview Primary,Grade 5 Maths,0000000002",
            "Carol,Hilltop Academy,Grade 5 Maths,0000000003",
        ]
    )

    response = client.post(
        "/api/v1/prefill/batch",
        data={
            "output_mode": "pdf",  # Should be overridden by grouping.
            "group_by": "school",
            "csv_text": csv_text,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["count"] == 3
    assert payload["successes"] == 3
    assert payload["group_by"] == "school"
    assert payload["filename"].endswith("_by_school.zip")
    assert len(payload["groups"]) == 2
    assert calls["group_by"] == "school"
    assert calls["count"] == 3


def test_prefill_batch_rejects_region_grouping_without_region_column(
    client: TestClient,
) -> None:
    csv_text = _csv(
        [
            "Alice,Riverview Primary,Grade 5 Maths,0000000001",
        ]
    )
    response = client.post(
        "/api/v1/prefill/batch",
        data={"output_mode": "pdf", "group_by": "region", "csv_text": csv_text},
    )
    assert response.status_code == 422
    assert "region" in response.json()["detail"].lower()


def test_prefill_batch_accepts_region_grouping_with_region_column(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_generate(rows, dst_path, *, group_by, **kwargs):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("Demerara.pdf", b"%PDF-1.4")
            zf.writestr("Berbice.pdf", b"%PDF-1.4")
        dst_path.write_bytes(buf.getvalue())
        return {
            "count": len(rows),
            "successes": len(rows),
            "errors": [],
            "elapsed_s": 0.01,
            "size_bytes": dst_path.stat().st_size,
            "groups": [
                {"name": "Demerara.pdf", "count": 1, "successes": 1, "errors": []},
                {"name": "Berbice.pdf", "count": 1, "successes": 1, "errors": []},
            ],
        }

    monkeypatch.setattr(
        prefill_service, "generate_batch_grouped_zip_to_file", fake_generate
    )

    csv_text = _csv(
        [
            "Alice,Riverview,Test,0000000001,Demerara",
            "Bob,Coastal,Test,0000000002,Berbice",
        ],
        with_region=True,
    )

    response = client.post(
        "/api/v1/prefill/batch",
        data={"output_mode": "zip", "group_by": "region", "csv_text": csv_text},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["group_by"] == "region"
    assert payload["filename"].endswith("_by_region.zip")


def test_prefill_batch_rejects_unknown_group_by(client: TestClient) -> None:
    csv_text = _csv(["Alice,Riverview,Test,0000000001"])
    response = client.post(
        "/api/v1/prefill/batch",
        data={"output_mode": "pdf", "group_by": "subject", "csv_text": csv_text},
    )
    assert response.status_code == 422
    assert "group_by" in response.json()["detail"].lower()


def test_prefill_batch_group_by_none_still_uses_flat_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm we did not regress the default flat behaviour."""

    called: dict[str, bool] = {"flat": False, "grouped": False}

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

    def fake_grouped(rows, dst_path, **kwargs):
        called["grouped"] = True
        return {
            "count": 0,
            "successes": 0,
            "errors": [],
            "elapsed_s": 0.0,
            "size_bytes": 0,
            "groups": [],
        }

    monkeypatch.setattr(prefill_service, "generate_batch_pdf_to_file", fake_pdf)
    monkeypatch.setattr(
        prefill_service, "generate_batch_grouped_zip_to_file", fake_grouped
    )

    csv_text = _csv(["Alice,Riverview,Test,0000000001"])
    response = client.post(
        "/api/v1/prefill/batch",
        data={"output_mode": "pdf", "csv_text": csv_text},
    )

    assert response.status_code == 200, response.text
    assert called["flat"] is True
    assert called["grouped"] is False
    payload = response.json()
    assert payload["group_by"] == "none"
    assert payload["groups"] == []
