"""API-level tests for the student-fill feature.

Covers:
  * GET /api/v1/prefill/marking-profiles
  * POST /api/v1/prefill/single with marking_profile + answers
  * POST /api/v1/prefill/batch with marking_profile + answers
  * Backward compatibility (no new fields → same behaviour as before)
  * Validation errors for unknown profiles / malformed answers
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from webui.services import prefill as prefill_service
from webui.services.scan_simulation import image_difference_score


def _image_from_png(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


# ---------------------------------------------------------------------------
# GET /prefill/marking-profiles
# ---------------------------------------------------------------------------
def test_marking_profiles_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/prefill/marking-profiles")
    assert response.status_code == 200
    payload = response.json()
    assert "profiles" in payload
    ids = {p["id"] for p in payload["profiles"]}
    assert "medium_pencil" in ids
    assert "none" in ids


# ---------------------------------------------------------------------------
# POST /prefill/single — backward compatibility
# ---------------------------------------------------------------------------
def test_prefill_single_without_student_fill_unchanged(client: TestClient) -> None:
  """Existing callers that omit marking_profile/answers must still work."""
  response = client.post(
      "/api/v1/prefill/single",
      data={
          "student_name": "A Student",
          "school_name": "School",
          "exam_name": "Exam",
          "candidate_number": "9010690012",
          "output_format": "png",
          "realism_preset": "none",
      },
  )
  assert response.status_code == 200, response.text
  assert response.headers["content-type"].startswith("image/png")


def test_prefill_single_explicit_none_profile_unchanged(client: TestClient) -> None:
    args = ("A Student", "School", "Exam", "9010690012")
    baseline = prefill_service.generate_single_png(*args, realism_preset="none")
    response = client.post(
        "/api/v1/prefill/single",
        data={
            "student_name": args[0],
            "school_name": args[1],
            "exam_name": args[2],
            "candidate_number": args[3],
            "output_format": "png",
            "realism_preset": "none",
            "marking_profile": "none",
            "answers": "",
        },
    )
    assert response.status_code == 200
    assert response.content == baseline


# ---------------------------------------------------------------------------
# POST /prefill/single — student fill
# ---------------------------------------------------------------------------
def test_prefill_single_with_marking_profile_and_answers(client: TestClient) -> None:
    response = client.post(
        "/api/v1/prefill/single",
        data={
            "student_name": "A Student",
            "school_name": "School",
            "exam_name": "Exam",
            "candidate_number": "9010690012",
            "output_format": "png",
            "realism_preset": "none",
            "marking_profile": "medium_pencil",
            "answers": "all_a",
        },
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("image/png")
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_prefill_single_student_fill_is_visible(client: TestClient) -> None:
    baseline = _image_from_png(
        prefill_service.generate_single_png(
            "A Student", "School", "Exam", "9010690012",
            realism_preset="none", marking_profile="none",
        )
    )
    response = client.post(
        "/api/v1/prefill/single",
        data={
            "student_name": "A Student",
            "school_name": "School",
            "exam_name": "Exam",
            "candidate_number": "9010690012",
            "output_format": "png",
            "realism_preset": "none",
            "marking_profile": "heavy_pencil",
            "answers": "all_a",
        },
    )
    assert response.status_code == 200
    filled = _image_from_png(response.content)
    assert image_difference_score(baseline, filled) > 0.05


def test_prefill_single_student_fill_is_deterministic(client: TestClient) -> None:
    data = {
        "student_name": "A Student",
        "school_name": "School",
        "exam_name": "Exam",
        "candidate_number": "9010690012",
        "output_format": "png",
        "realism_preset": "none",
        "marking_profile": "medium_pencil",
        "answers": "alternating",
    }
    r1 = client.post("/api/v1/prefill/single", data=data)
    r2 = client.post("/api/v1/prefill/single", data=data)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.content == r2.content


def test_prefill_single_rejects_unknown_marking_profile(client: TestClient) -> None:
    response = client.post(
        "/api/v1/prefill/single",
        data={
            "student_name": "A Student",
            "school_name": "School",
            "exam_name": "Exam",
            "candidate_number": "9010690012",
            "output_format": "png",
            "marking_profile": "disco_pencil",
        },
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "profile",
    ["light_pencil", "medium_pencil", "heavy_pencil", "pen_ballpoint",
     "check_mark", "cross_mark", "partial_fill", "careful_student"],
)
def test_prefill_single_accepts_all_profiles(client: TestClient, profile: str) -> None:
    response = client.post(
        "/api/v1/prefill/single",
        data={
            "student_name": "A Student",
            "school_name": "School",
            "exam_name": "Exam",
            "candidate_number": "9010690012",
            "output_format": "png",
            "realism_preset": "none",
            "marking_profile": profile,
            "answers": "all_b",
        },
    )
    assert response.status_code == 200, f"profile={profile}: {response.text}"


def test_prefill_single_filename_includes_profile_suffix(client: TestClient) -> None:
    response = client.post(
        "/api/v1/prefill/single",
        data={
            "student_name": "A Student",
            "school_name": "School",
            "exam_name": "Exam",
            "candidate_number": "9010690012",
            "output_format": "png",
            "marking_profile": "heavy_pencil",
            "answers": "all_a",
        },
    )
    assert response.status_code == 200
    disposition = response.headers.get("content-disposition", "")
    assert "heavy_pencil" in disposition


def test_prefill_single_with_json_answers(client: TestClient) -> None:
    response = client.post(
        "/api/v1/prefill/single",
        data={
            "student_name": "A Student",
            "school_name": "School",
            "exam_name": "Exam",
            "candidate_number": "9010690012",
            "output_format": "png",
            "marking_profile": "medium_pencil",
            "answers": '{"q1": "A", "q2": "B", "q3": "C", "q4": "D", "q5": "A"}',
        },
    )
    assert response.status_code == 200


def test_prefill_single_with_realism_and_student_fill(client: TestClient) -> None:
    """Both realism preset and student fill can be combined."""
    response = client.post(
        "/api/v1/prefill/single",
        data={
            "student_name": "A Student",
            "school_name": "School",
            "exam_name": "Exam",
            "candidate_number": "9010690012",
            "output_format": "png",
            "realism_preset": "subtle",
            "marking_profile": "medium_pencil",
            "answers": "all_a",
        },
    )
    assert response.status_code == 200
    clean = _image_from_png(
        prefill_service.generate_single_png(
            "A Student", "School", "Exam", "9010690012",
            realism_preset="none", marking_profile="none",
        )
    )
    combined = _image_from_png(response.content)
    assert image_difference_score(clean, combined) > 0.5


# ---------------------------------------------------------------------------
# POST /prefill/batch — student fill
# ---------------------------------------------------------------------------
def test_prefill_batch_forwards_marking_profile(
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_generate(rows, dst_path, realism_preset="none", marking_profile="none", answers=None):
        calls["count"] = len(rows)
        calls["marking_profile"] = marking_profile
        calls["answers"] = answers
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
        "student_name,school_name,exam_name,candidate_number\n"
        "Student A,School,Exam,9010690012\n"
        "Student B,School,Exam,9010690013\n"
    )
    response = client.post(
        "/api/v1/prefill/batch",
        data={
            "output_mode": "pdf",
            "csv_text": csv_text,
            "marking_profile": "heavy_pencil",
            "answers": "all_a",
        },
    )
    assert response.status_code == 200, response.text
    assert calls["marking_profile"] == "heavy_pencil"
    assert calls["answers"] == "all_a"
    assert calls["count"] == 2


def test_prefill_batch_rejects_unknown_marking_profile(client: TestClient) -> None:
    csv_text = (
        "student_name,school_name,exam_name,candidate_number\n"
        "Student A,School,Exam,9010690012\n"
    )
    response = client.post(
        "/api/v1/prefill/batch",
        data={
            "output_mode": "pdf",
            "csv_text": csv_text,
            "marking_profile": "disco_pencil",
        },
    )
    assert response.status_code == 422


def test_prefill_batch_per_row_answers_column(
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Per-row ``answers`` column in CSV should be forwarded to the service."""
    captured_rows: list[dict] = []

    def fake_generate(rows, dst_path, realism_preset="none", marking_profile="none", answers=None):
        captured_rows.extend(rows)
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
        "student_name,school_name,exam_name,candidate_number,answers\n"
        "Student A,School,Exam,9010690012,all_a\n"
        "Student B,School,Exam,9010690013,all_b\n"
    )
    response = client.post(
        "/api/v1/prefill/batch",
        data={
            "output_mode": "pdf",
            "csv_text": csv_text,
            "marking_profile": "medium_pencil",
            "answers": "random",
        },
    )
    assert response.status_code == 200, response.text
    assert len(captured_rows) == 2
    # Per-row answers should override the batch default.
    assert captured_rows[0].get("answers") == "all_a"
    assert captured_rows[1].get("answers") == "all_b"


# ---------------------------------------------------------------------------
# GET /prefill/sample with marking profile
# ---------------------------------------------------------------------------
def test_prefill_sample_accepts_marking_profile(client: TestClient) -> None:
    response = client.get(
        "/api/v1/prefill/sample",
        params={
            "preset": "none",
            "marking_profile": "medium_pencil",
            "answers": "all_a",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")


def test_prefill_sample_rejects_unknown_marking_profile(client: TestClient) -> None:
    response = client.get(
        "/api/v1/prefill/sample",
        params={"marking_profile": "disco_pencil"},
    )
    assert response.status_code == 422
