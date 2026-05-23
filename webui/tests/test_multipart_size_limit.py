"""Regression tests for the multipart per-part size limit.

Starlette's ``MultiPartParser`` caps each part at 1 MiB by default, which
caused ``/prefill/batch`` to reject 30 000-row CSVs (~3.5 MiB submitted
as a single ``csv_text`` form field) with *"Part exceeded maximum size
of 1024KB."* even though the application-layer caps allow far larger
payloads.

The ``/prefill/batch`` endpoint parses its form manually with
``settings.max_upload_bytes`` so large multipart text fields and file
parts can reach the route handler. The application-layer caps must still
apply once the body is parsed.
"""

from __future__ import annotations

import io

from fastapi.testclient import TestClient

from webui.services import prefill as prefill_service
from webui.settings import get_settings


def _csv_with_padded_rows(count: int, pad_per_row: int = 0) -> str:
    """Build a valid CSV whose body exceeds ``count * pad_per_row`` bytes."""
    lines = ["student_name,school_name,exam_name,candidate_number"]
    pad = "x" * pad_per_row
    for idx in range(count):
        lines.append(f"Student {pad}{idx},School,Exam,{9010690000 + idx:010d}")
    return "\n".join(lines)


def test_prefill_batch_accepts_csv_text_field_above_1_mib(
    client: TestClient, monkeypatch
) -> None:
    """A ``csv_text`` field above 1 MiB must reach the handler.

    Without the multipart pre-parse, Starlette would reject the request
    with a 400 *"Part exceeded maximum size of 1024KB."* error before
    our endpoint code runs.
    """
    captured: dict[str, object] = {}

    def fake_generate(rows, dst_path, realism_preset="none", include_page_numbers=False):
        captured["count"] = len(rows)
        dst_path.write_bytes(b"%PDF-1.4 stub")
        return {
            "count": len(rows),
            "successes": len(rows),
            "errors": [],
            "elapsed_s": 0.0,
            "size_bytes": dst_path.stat().st_size,
        }

    monkeypatch.setattr(prefill_service, "generate_batch_pdf_to_file", fake_generate)

    csv_text = _csv_with_padded_rows(count=200, pad_per_row=8 * 1024)  # ~1.6 MiB
    assert len(csv_text.encode("utf-8")) > 1 * 1024 * 1024, (
        "csv_text must exceed 1 MiB to exercise the per-part limit"
    )

    response = client.post(
        "/api/v1/prefill/batch",
        files={
            "output_mode": (None, "pdf"),
            "csv_text": (None, csv_text),
        },
    )

    assert response.status_code == 200, response.text
    assert captured["count"] == 200


def test_prefill_batch_accepts_csv_file_above_1_mib(
    client: TestClient, monkeypatch
) -> None:
    """A multipart file part larger than 1 MiB must also be accepted.

    The frontend submits uploaded CSVs as files so browser uploads avoid
    re-posting the full file as one giant text field. Scripted uploads
    should work the same way.
    """
    captured: dict[str, object] = {}

    def fake_generate(rows, dst_path, realism_preset="none", include_page_numbers=False):
        captured["count"] = len(rows)
        dst_path.write_bytes(b"%PDF-1.4 stub")
        return {
            "count": len(rows),
            "successes": len(rows),
            "errors": [],
            "elapsed_s": 0.0,
            "size_bytes": dst_path.stat().st_size,
        }

    monkeypatch.setattr(prefill_service, "generate_batch_pdf_to_file", fake_generate)

    csv_text = _csv_with_padded_rows(count=200, pad_per_row=8 * 1024)
    payload = csv_text.encode("utf-8")
    assert len(payload) > 1 * 1024 * 1024

    response = client.post(
        "/api/v1/prefill/batch",
        data={"output_mode": "pdf"},
        files={"csv_file": ("students.csv", io.BytesIO(payload), "text/csv")},
    )

    assert response.status_code == 200, response.text
    assert captured["count"] == 200


def test_prefill_batch_still_rejects_payloads_above_application_cap(
    client: TestClient,
) -> None:
    """The dependency raises the multipart limit but does NOT bypass the
    application-layer ``prefill_csv_max_bytes`` cap. Payloads above that
    cap must still get a clean 413 from our endpoint instead of a 400
    from the parser."""
    settings = get_settings()
    cap = settings.prefill_csv_max_bytes

    # One byte over the cap.
    pad_size = cap // 32 + 1
    csv_text = _csv_with_padded_rows(count=64, pad_per_row=pad_size)
    assert len(csv_text.encode("utf-8")) > cap

    response = client.post(
        "/api/v1/prefill/batch",
        files={
            "output_mode": (None, "pdf"),
            "csv_text": (None, csv_text),
        },
    )

    assert response.status_code == 413, response.text
