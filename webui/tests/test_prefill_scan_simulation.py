"""Tests for realistic scan simulation in the prefill pipeline."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from webui.services import prefill as prefill_service
from webui.services.scan_simulation import image_difference_score


def _image_from_png(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def test_realism_none_preserves_existing_single_png_bytes() -> None:
    """The default/no-simulation path must remain byte-identical."""
    args = ("A Student", "School", "Exam", "9010690012")

    default_png = prefill_service.generate_single_png(*args)
    explicit_none_png = prefill_service.generate_single_png(*args, realism_preset="none")

    assert explicit_none_png == default_png


def test_subtle_scan_simulation_is_deterministic_and_visible() -> None:
    args = ("A Student", "School", "Exam", "9010690012")

    clean = _image_from_png(prefill_service.generate_single_png(*args, realism_preset="none"))
    subtle_a = prefill_service.generate_single_png(*args, realism_preset="subtle")
    subtle_b = prefill_service.generate_single_png(*args, realism_preset="subtle")
    subtle = _image_from_png(subtle_a)

    assert subtle_a == subtle_b
    assert subtle.size == clean.size
    assert image_difference_score(clean, subtle) > 0.5


def test_adversarial_scan_simulation_is_stronger_than_subtle() -> None:
    args = ("A Student", "School", "Exam", "9010690012")

    clean = _image_from_png(prefill_service.generate_single_png(*args, realism_preset="none"))
    subtle = _image_from_png(prefill_service.generate_single_png(*args, realism_preset="subtle"))
    adversarial = _image_from_png(
        prefill_service.generate_single_png(*args, realism_preset="adversarial")
    )

    subtle_delta = image_difference_score(clean, subtle)
    adversarial_delta = image_difference_score(clean, adversarial)

    assert adversarial_delta > subtle_delta


def test_prefill_single_accepts_realism_preset(client: TestClient) -> None:
    response = client.post(
        "/api/v1/prefill/single",
        data={
            "student_name": "A Student",
            "school_name": "School",
            "exam_name": "Exam",
            "candidate_number": "9010690012",
            "output_format": "png",
            "realism_preset": "subtle",
        },
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("image/png")
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_prefill_single_rejects_unknown_realism_preset(client: TestClient) -> None:
    response = client.post(
        "/api/v1/prefill/single",
        data={
            "student_name": "A Student",
            "school_name": "School",
            "exam_name": "Exam",
            "candidate_number": "9010690012",
            "output_format": "png",
            "realism_preset": "space-laser",
        },
    )

    assert response.status_code == 422
    assert "realism_preset" in response.json()["detail"]


def test_prefill_batch_forwards_realism_preset(
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_generate(rows, dst_path, realism_preset="none"):
        calls["count"] = len(rows)
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
    csv_text = "\n".join([
        "student_name,school_name,exam_name,candidate_number",
        "A Student,School,Exam,9010690012",
    ])

    response = client.post(
        "/api/v1/prefill/batch",
        data={
            "output_mode": "pdf",
            "realism_preset": "moderate",
            "csv_text": csv_text,
        },
    )

    assert response.status_code == 200, response.text
    assert calls["count"] == 1
    assert calls["realism_preset"] == "moderate"


# ---------------------------------------------------------------------------
# Marker survival regression.
#
# The "moderate" preset previously destroyed ArUco contrast at the corners
# (vignette + JPEG roundtrip + noise) and the OMR engine reported a 68%
# preprocess-failure rate on synthetic moderate sheets. The marker-restore
# pass added in scan_simulation must keep that close to 0%.
# ---------------------------------------------------------------------------


def _detect_aruco_ids(png_bytes: bytes) -> set[int]:
    """Return the ArUco IDs decoded from a single rendered sheet.

    Mirrors the OMR engine's tuned ``ArucoDetector`` parameters (see
    ``src/processors/CropOnMarkers.py``) so this regression test exercises
    the **production detector configuration** rather than OpenCV defaults.
    """
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "PNG decode failed"
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 15
    params.adaptiveThreshWinSizeStep = 4
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 0.5
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    _, ids, _ = detector.detectMarkers(img)
    if ids is None:
        return set()
    return {int(i) for i in ids.flatten()}


# ---------------------------------------------------------------------------
# Candidate-number "printed" preservation regression.
#
# The candidate number is meant to be machine-printed at the top of every
# sheet, never hand-written, so its bubbles and digit headers must remain
# pristine even at moderate/adversarial realism. The test below compares
# scan-simulated output WITH and WITHOUT the ``candidate_region`` argument
# at a fixed seed: protection must change pixels INSIDE the rectangle (it
# restores pristine content) and leave the rest of the page byte-identical.
# ---------------------------------------------------------------------------


def _candidate_geometry_for_test(candidate_number: str = "9010690012"):
    """Return (pil_image, region, bubbles, markers) for a fresh generated sheet."""
    from prefill_only_package import prefill_answer_sheet_final as m
    from webui.services.scan_simulation import BubbleGeometry, MarkerBox

    clean_png = prefill_service.generate_single_png(
        "A Student", "School", "Exam", candidate_number, realism_preset="none"
    )
    pil = _image_from_png(clean_png)
    w, h = pil.size
    bubbles = [
        BubbleGeometry(
            column=int(it["column"]),
            digit=int(it["digit"]),
            cx=int(it["cx"]),
            cy=int(it["cy"]),
            radius=int(it["radius"]),
            filled=bool(it.get("filled", False)),
        )
        for it in m.candidate_bubble_geometry(w, h, candidate_number)
    ]
    markers = [
        MarkerBox(
            corner=int(it["corner"]),
            x0=int(it["x0"]),
            y0=int(it["y0"]),
            x1=int(it["x1"]),
            y1=int(it["y1"]),
        )
        for it in m.aruco_marker_boxes(w, h)
    ]
    region = m.candidate_region_box(w, h)
    return pil, region, bubbles, markers


def _reproduce_geo_transform(
    pil: Image.Image, bubbles, preset: str, candidate_number: str, seed: int
) -> np.ndarray:
    """Re-run the deterministic RNG path to get the geometry transform used inside
    ``apply_scan_simulation``. Tests use this to compute the warped quad."""
    from webui.services.scan_simulation import (
        _to_rgb_array,
        _apply_geometry,
        _draw_imperfect_bubbles,
        _rng_for,
    )

    rng = _rng_for(candidate_number, preset, seed)
    arr = _to_rgb_array(pil)
    arr = _draw_imperfect_bubbles(arr, bubbles, preset=preset, rng=rng)
    _, geo_transform = _apply_geometry(arr, preset=preset, rng=rng)
    return geo_transform


def _warped_polygon_mask(
    region: tuple[int, int, int, int], geo_transform: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Build a mask of the post-warp candidate-region quad."""
    rx0, ry0, rx1, ry1 = region
    src = np.array(
        [[rx0, ry0], [rx1, ry0], [rx1, ry1], [rx0, ry1]], dtype=np.float32
    ).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(src, geo_transform).reshape(-1, 2)
    poly = np.round(warped).astype(np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    return mask


@pytest.mark.parametrize("preset", ["moderate", "adversarial"])
def test_candidate_region_protection_only_changes_candidate_block(preset: str) -> None:
    """Passing ``candidate_region`` must alter ONLY the warped candidate quad."""
    from webui.services.scan_simulation import apply_scan_simulation

    pil, region, bubbles, markers = _candidate_geometry_for_test()
    seed = 4242
    candidate_number = "9010690012"
    common = dict(
        candidate_number=candidate_number,
        bubbles=bubbles,
        markers=markers,
        seed=seed,
        preset=preset,
    )

    without = np.array(apply_scan_simulation(pil, candidate_region=None, **common))
    with_protect = np.array(apply_scan_simulation(pil, candidate_region=region, **common))

    geo_transform = _reproduce_geo_transform(pil, bubbles, preset, candidate_number, seed)
    mask = _warped_polygon_mask(region, geo_transform, with_protect.shape[:2])

    diff = np.abs(without.astype(np.int16) - with_protect.astype(np.int16)).max(axis=2)
    diff_inside = float(diff[mask > 0].mean())
    diff_outside_max = int(diff[mask == 0].max())

    assert diff_inside > 4.0, (
        f"preset={preset!r}: protection changed nothing inside the candidate "
        f"region quad (mean per-pixel max-channel diff {diff_inside:.2f}); "
        "the protect pass may not be running."
    )
    assert diff_outside_max == 0, (
        f"preset={preset!r}: protection changed pixels OUTSIDE the warped "
        f"candidate quad (max abs diff = {diff_outside_max}); the polygon "
        "mask must not bleed."
    )


@pytest.mark.parametrize("preset", ["moderate", "adversarial"])
def test_protect_region_restores_pristine_pixels_in_polygon(preset: str) -> None:
    """Inside the warped quad, ``apply_scan_simulation`` output must equal the
    pristine-pre-bubbles snapshot warped through the same geometry."""
    from webui.services.scan_simulation import _to_rgb_array, apply_scan_simulation

    pil, region, bubbles, markers = _candidate_geometry_for_test()
    seed = 99
    candidate_number = "9010690012"

    geo_transform = _reproduce_geo_transform(pil, bubbles, preset, candidate_number, seed)
    arr_initial = _to_rgb_array(pil)
    pristine_warped = cv2.warpPerspective(
        arr_initial,
        geo_transform,
        (arr_initial.shape[1], arr_initial.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    out = np.array(
        apply_scan_simulation(
            pil,
            preset=preset,
            candidate_number=candidate_number,
            bubbles=bubbles,
            markers=markers,
            candidate_region=region,
            seed=seed,
        )
    )
    mask = _warped_polygon_mask(region, geo_transform, out.shape[:2])

    diff = np.abs(out.astype(np.int16) - pristine_warped.astype(np.int16))
    inside_max = int(diff[mask > 0].max())

    assert inside_max == 0, (
        f"preset={preset!r}: candidate-region pixels diverge from the warped "
        f"pristine snapshot inside the polygon (max abs diff = {inside_max}); "
        "the protect pass is not exact."
    )


@pytest.mark.parametrize("preset", ["none", "subtle", "moderate"])
def test_realism_preset_preserves_all_four_aruco_markers(preset: str) -> None:
    """Each non-adversarial preset must keep all 4 corner markers detectable.

    Without the marker-protection pass, ``moderate`` regularly dropped 1-2
    markers per sheet and the OMR engine reported ~68% preprocess failures
    on 100-page batches. This test is the fast guard against that
    regression.
    """
    candidate_numbers = [
        "9010690012",
        "0123456789",
        "5555555555",
        "1112223333",
        "9999999999",
        "0000000001",
        "8675309000",
        "2718281828",
    ]
    expected = {0, 1, 2, 3}
    failed: list[tuple[str, set[int]]] = []
    for n in candidate_numbers:
        png = prefill_service.generate_single_png(
            "A Student", "School", "Exam", n, realism_preset=preset
        )
        ids = _detect_aruco_ids(png)
        if not expected.issubset(ids):
            failed.append((n, ids))

    assert not failed, (
        f"preset={preset!r} dropped ArUco markers on "
        f"{len(failed)}/{len(candidate_numbers)} sheets: {failed!r}"
    )
