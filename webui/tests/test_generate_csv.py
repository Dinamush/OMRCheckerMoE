"""Tests for the server-backed Generate Test CSV flow."""

from __future__ import annotations

import csv
import io

from fastapi.testclient import TestClient


def test_generate_csv_returns_download_token_and_file(client: TestClient) -> None:
    response = client.post(
        "/api/v1/generate-csv",
        data={
            "count": "3",
            "school_name": "Test School",
            "exam_name": "National Test",
            "candidate_start": "9000000001",
            "name_style": "random",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["count"] == 3
    assert payload["download_url"].startswith("/api/v1/generate-csv/download/")
    assert payload["filename"].startswith("test_students_3_random_")

    download = client.get(payload["download_url"])
    assert download.status_code == 200, download.text
    assert "text/csv" in download.headers["content-type"]

    text = download.content.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 3
    assert rows[0]["school_name"] == "Test School"
    assert rows[0]["exam_name"] == "National Test"
    assert rows[0]["candidate_number"] == "9000000001"
    assert rows[1]["candidate_number"] == "9000000002"
    assert rows[0]["output_file"].endswith(".png")

    second_download = client.get(payload["download_url"])
    assert second_download.status_code == 404


def test_generate_csv_rejects_candidate_overflow(client: TestClient) -> None:
    response = client.post(
        "/api/v1/generate-csv",
        data={
            "count": "2",
            "school_name": "Test School",
            "exam_name": "National Test",
            "candidate_start": "9999999999",
            "name_style": "numbered",
        },
    )

    assert response.status_code == 422
    assert "exceed 10 digits" in response.json()["detail"]
