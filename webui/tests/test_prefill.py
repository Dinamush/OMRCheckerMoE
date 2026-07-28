"""Tests for the prefill answer-sheet API paths."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from webui.services import prefill as prefill_service


def test_prefill_batch_accepts_large_csv_without_buffered_generation(
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_generate(rows, dst_path, realism_preset="none"):
        calls["count"] = len(rows)
        calls["dst_path"] = dst_path
        calls["realism_preset"] = realism_preset
        dst_path.write_bytes(b"fake pdf")
        return {
            "count": len(rows),
            "successes": len(rows),
            "errors": [],
            "elapsed_s": 0.01,
            "size_bytes": dst_path.stat().st_size,
        }

    monkeypatch.setattr(prefill_service, "generate_batch_pdf_to_file", fake_generate)
    csv_lines = ["student_name,school_name,exam_name,candidate_number"]
    csv_lines.extend(
        f"Student {idx},School,Exam,{idx:010d}" for idx in range(4000)
    )

    response = client.post(
        "/api/v1/prefill/batch",
        data={"output_mode": "pdf", "csv_text": "\n".join(csv_lines)},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["count"] == 4000
    assert payload["successes"] == 4000
    assert payload["download_url"].startswith("/api/v1/prefill/batch/download/")
    assert calls["count"] == 4000
    assert calls["realism_preset"] == "none"


def test_normalize_row_keys_maps_centre_aliases() -> None:
    """center_name / centre_name / Centre Name all resolve to school_name."""
    for alias in ("center_name", "centre_name", "Centre Name", "CENTER_NAME"):
        row = {
            "student_name": "Jane Doe",
            "region": "Region 11",
            "candidate_number": "0091110006",
            alias: "Region 11 Examination Centre",
        }
        norm = prefill_service.normalize_row_keys(row)
        assert norm["school_name"] == "Region 11 Examination Centre"
        assert norm["region"] == "Region 11"
        assert norm["candidate_number"] == "0091110006"
        assert prefill_service.REQUIRED_CSV_COLUMNS <= set(norm.keys())


def test_normalize_row_keys_prefers_non_empty_alias_value() -> None:
    """When both school_name and center_name exist, the populated one wins."""
    row = {
        "student_name": "Jane",
        "school_name": "",
        "center_name": "Region 2 Examination Centre",
        "candidate_number": "0090210001",
    }
    norm = prefill_service.normalize_row_keys(row)
    assert norm["school_name"] == "Region 2 Examination Centre"


def test_prefill_batch_accepts_region_center_name_schema(
    client: TestClient,
    monkeypatch,
) -> None:
    """The MoE registry export (student_name, region, candidate_number,
    center_name — no exam_name) must be accepted and normalised."""
    captured: dict[str, object] = {}

    def fake_generate(rows, dst_path, realism_preset="none"):
        captured["rows"] = rows
        dst_path.write_bytes(b"fake pdf")
        return {
            "count": len(rows),
            "successes": len(rows),
            "errors": [],
            "elapsed_s": 0.01,
            "size_bytes": dst_path.stat().st_size,
        }

    monkeypatch.setattr(prefill_service, "generate_batch_pdf_to_file", fake_generate)
    csv_text = (
        "student_name,region,candidate_number,center_name\n"
        "DIMITRI DAVID PHILLIPS,Region 11,0091110006,Region 11 Examination Centre\n"
        "AALIYAH NATOYA BUTTERS,Region 2,0090210001,Region 2 Examination Centre\n"
    )

    response = client.post(
        "/api/v1/prefill/batch",
        data={"output_mode": "pdf", "csv_text": csv_text},
    )

    assert response.status_code == 200, response.text
    rows = captured["rows"]
    assert isinstance(rows, list) and len(rows) == 2
    assert rows[0]["school_name"] == "Region 11 Examination Centre"
    assert rows[0]["region"] == "Region 11"
    assert rows[0]["candidate_number"] == "0091110006"


def test_prefill_batch_missing_centre_column_returns_helpful_error(
    client: TestClient,
) -> None:
    csv_text = (
        "student_name,region,candidate_number\n"
        "Jane Doe,Region 1,0090210001\n"
    )
    response = client.post(
        "/api/v1/prefill/batch",
        data={"output_mode": "pdf", "csv_text": csv_text},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "school_name" in detail
    assert "center_name" in detail

