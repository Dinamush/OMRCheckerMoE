"""Tests for the optional page-number stamp on prefill PDF batches.

Coverage:

*   :func:`draw_page_number` actually paints something at the bottom-right
    anchor and leaves the ArUco markers / candidate-number block / answer
    bubble area untouched.
*   :func:`generate_batch_pdf_to_file` threads ``include_page_numbers``
    through to the renderer so each page bakes the right ordinal in CSV
    order.
*   The ``/api/v1/prefill/batch`` endpoint forwards the form flag for PDF
    output and silently drops it for ZIP output.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from prefill_only_package import prefill_answer_sheet_final as prefill_module
from webui.services import prefill as prefill_service


# ---------------------------------------------------------------------------
# draw_page_number — placement + non-interference
# ---------------------------------------------------------------------------

def _diff_mask(before: Image.Image, after: Image.Image) -> np.ndarray:
    """Boolean mask where the two RGB images differ (any channel)."""
    a = np.array(before)
    b = np.array(after)
    return np.any(a != b, axis=-1)


def test_draw_page_number_stamps_in_bottom_right_band() -> None:
    """The stamp must land inside the band between the answer bubbles
    and the bottom edge, and to the LEFT of the BR ArUco marker."""
    base = prefill_module.draw_aruco_corners(
        Image.new("RGB", (666, 515), color="white")
    )
    stamped = prefill_module.draw_page_number(base.copy(), 7)

    diff = _diff_mask(base, stamped)
    assert diff.any(), "draw_page_number should change at least one pixel"

    ys, xs = np.where(diff)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    # Bottom-right quadrant: right of the page midline, below the answer
    # grid which terminates around y ~= 0.66 * h on this canvas.
    h, w = 515, 666
    assert x_min >= w * 0.5, (x_min, x_max)
    assert y_min >= h * 0.66, (y_min, y_max)

    # Must not encroach on the BR ArUco marker box.
    br_box = next(
        b for b in prefill_module.aruco_marker_boxes(w, h) if b["corner"] == 3
    )
    assert x_max < br_box["x0"], (
        f"page number bbox extends to x={x_max} which overlaps the BR "
        f"ArUco marker starting at x={br_box['x0']}"
    )


def test_draw_page_number_does_not_touch_aruco_markers() -> None:
    """The stamp must leave every ArUco fiducial pixel-perfect."""
    base = prefill_module.draw_aruco_corners(
        Image.new("RGB", (666, 515), color="white")
    )
    stamped = prefill_module.draw_page_number(base.copy(), 12)

    base_arr = np.array(base)
    stamped_arr = np.array(stamped)
    for box in prefill_module.aruco_marker_boxes(666, 515):
        a = base_arr[box["y0"]:box["y1"], box["x0"]:box["x1"]]
        b = stamped_arr[box["y0"]:box["y1"], box["x0"]:box["x1"]]
        assert np.array_equal(a, b), (
            f"page number altered marker corner={box['corner']}"
        )


def test_draw_page_number_skips_for_none_or_non_positive() -> None:
    base = prefill_module.draw_aruco_corners(
        Image.new("RGB", (666, 515), color="white")
    )
    for value in (None, 0, -3):
        out = prefill_module.draw_page_number(base.copy(), value)
        assert np.array_equal(np.array(base), np.array(out)), value


def test_draw_page_number_distinguishes_consecutive_indices() -> None:
    """Different ordinals must produce different rendered images so we
    can be sure each PDF page gets its own stamp."""
    base = prefill_module.draw_aruco_corners(
        Image.new("RGB", (666, 515), color="white")
    )
    one = np.array(prefill_module.draw_page_number(base.copy(), 1))
    two = np.array(prefill_module.draw_page_number(base.copy(), 2))
    twelve = np.array(prefill_module.draw_page_number(base.copy(), 12))

    assert not np.array_equal(one, two)
    assert not np.array_equal(two, twelve)
    # 12 has more ink than 1 because it is a two-digit number.
    assert (255 - twelve).sum() > (255 - one).sum()


def test_draw_page_number_scales_to_full_resolution_template() -> None:
    """Anchor logic must work for the 1426x1103 reference template too."""
    base = prefill_module.draw_aruco_corners(
        Image.new("RGB", (1426, 1103), color="white")
    )
    stamped = prefill_module.draw_page_number(base.copy(), 42)

    diff = _diff_mask(base, stamped)
    assert diff.any()

    br_box = next(
        b for b in prefill_module.aruco_marker_boxes(1426, 1103) if b["corner"] == 3
    )
    ys, xs = np.where(diff)
    assert int(xs.max()) < br_box["x0"]
    assert int(ys.max()) < 1103


# ---------------------------------------------------------------------------
# Service-level: batch PDF threading
# ---------------------------------------------------------------------------

def test_generate_batch_pdf_includes_page_numbers_when_requested(
    tmp_path: Path,
) -> None:
    """End-to-end: batch PDF with ``include_page_numbers=True`` differs
    from the same batch without page numbers, and the difference grows
    with row count (because more pages get stamps)."""
    rows = [
        {
            "student_name": f"Student {idx}",
            "school_name": "School",
            "exam_name": "Exam",
            "candidate_number": f"{9010690000 + idx:010d}",
        }
        for idx in range(3)
    ]

    plain_path = tmp_path / "plain.pdf"
    numbered_path = tmp_path / "numbered.pdf"

    prefill_service.generate_batch_pdf_to_file(rows, plain_path)
    prefill_service.generate_batch_pdf_to_file(
        rows, numbered_path, include_page_numbers=True
    )

    assert plain_path.exists() and numbered_path.exists()
    assert plain_path.stat().st_size > 0
    assert numbered_path.stat().st_size > 0
    # Different bytes — at minimum the embedded JPEGs differ because
    # each page has an extra digit drawn before encoding.
    assert plain_path.read_bytes() != numbered_path.read_bytes()


def test_thread_render_stamps_page_number_into_output() -> None:
    """``_thread_render`` must call ``draw_page_number`` whenever the
    payload carries a ``page_number`` key."""
    payload_plain = {
        "student_name": "A Student",
        "school_name": "School",
        "exam_name": "Exam",
        "candidate_number": "9010690012",
        "output_format": "png",
        "realism_preset": "none",
    }
    payload_numbered = dict(payload_plain, page_number=99)

    plain = Image.open(io.BytesIO(prefill_service._thread_render(payload_plain)))
    numbered = Image.open(io.BytesIO(prefill_service._thread_render(payload_numbered)))

    assert plain.size == numbered.size
    diff = _diff_mask(plain.convert("RGB"), numbered.convert("RGB"))
    assert diff.any(), "page_number=99 should leave visible pixel changes"


# ---------------------------------------------------------------------------
# API-level: form flag plumbing
# ---------------------------------------------------------------------------

def _csv_for(count: int) -> str:
    lines = ["student_name,school_name,exam_name,candidate_number"]
    for idx in range(count):
        lines.append(f"Student {idx},School,Exam,{9010690000 + idx:010d}")
    return "\n".join(lines)


def test_prefill_batch_threads_include_page_numbers_for_pdf(
    client: TestClient, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_generate(rows, dst_path, realism_preset="none", include_page_numbers=False):
        captured["count"] = len(rows)
        captured["realism_preset"] = realism_preset
        captured["include_page_numbers"] = include_page_numbers
        dst_path.write_bytes(b"%PDF-1.4 stub")
        return {
            "count": len(rows),
            "successes": len(rows),
            "errors": [],
            "elapsed_s": 0.0,
            "size_bytes": dst_path.stat().st_size,
        }

    monkeypatch.setattr(prefill_service, "generate_batch_pdf_to_file", fake_generate)

    response = client.post(
        "/api/v1/prefill/batch",
        data={
            "output_mode": "pdf",
            "csv_text": _csv_for(2),
            "include_page_numbers": "true",
        },
    )

    assert response.status_code == 200, response.text
    assert captured["include_page_numbers"] is True
    assert response.json()["page_numbers"] is True


def test_prefill_batch_defaults_include_page_numbers_off(
    client: TestClient, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_generate(rows, dst_path, realism_preset="none", include_page_numbers=False):
        captured["include_page_numbers"] = include_page_numbers
        dst_path.write_bytes(b"%PDF-1.4 stub")
        return {
            "count": len(rows),
            "successes": len(rows),
            "errors": [],
            "elapsed_s": 0.0,
            "size_bytes": dst_path.stat().st_size,
        }

    monkeypatch.setattr(prefill_service, "generate_batch_pdf_to_file", fake_generate)

    response = client.post(
        "/api/v1/prefill/batch",
        data={"output_mode": "pdf", "csv_text": _csv_for(1)},
    )

    assert response.status_code == 200, response.text
    assert captured["include_page_numbers"] is False
    assert response.json()["page_numbers"] is False


def test_prefill_batch_zip_silently_drops_page_numbers(
    client: TestClient, monkeypatch
) -> None:
    """ZIP output is multiple single-page PNGs, so a page-number stamp
    is meaningless. The flag must NOT propagate to the ZIP renderer."""
    captured: dict[str, object] = {"called": False}

    def fake_zip(rows, dst_path, realism_preset="none"):
        captured["called"] = True
        captured["zip_count"] = len(rows)
        dst_path.write_bytes(b"PK\x03\x04 stub")
        return {
            "count": len(rows),
            "successes": len(rows),
            "errors": [],
            "elapsed_s": 0.0,
            "size_bytes": dst_path.stat().st_size,
        }

    def fake_pdf(*args, **kwargs):
        captured["pdf_called"] = True
        raise AssertionError("ZIP request must not call generate_batch_pdf_to_file")

    monkeypatch.setattr(prefill_service, "generate_batch_zip_to_file", fake_zip)
    monkeypatch.setattr(prefill_service, "generate_batch_pdf_to_file", fake_pdf)

    response = client.post(
        "/api/v1/prefill/batch",
        data={
            "output_mode": "zip",
            "csv_text": _csv_for(2),
            "include_page_numbers": "true",
        },
    )

    assert response.status_code == 200, response.text
    assert captured["called"] is True
    assert response.json()["page_numbers"] is False
