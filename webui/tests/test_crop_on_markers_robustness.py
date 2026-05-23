"""Tests for the ArUco cropper robustness upgrade (Board + RANSAC + sanity)."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.core import _auto_orient_to_template
from src.processors.CropOnMarkers import (
    CropOnMarkers,
    _find_homography_robust,
    _homography_is_sane,
    _similarity_homography_from_pairs,
)

REPO = Path(__file__).resolve().parents[2]
PAGE_W, PAGE_H = 666, 515
REFERENCE_CENTRES = [
    [13.5, 13.2],
    [651.5, 13.2],
    [13.5, 499.0],
    [651.5, 499.0],
]


class _StubImageOps:
    def __init__(self) -> None:
        from dotmap import DotMap

        self.tuning_config = DotMap(
            {
                "outputs": {"show_image_level": 0},
                "dimensions": {
                    "display_width": PAGE_W,
                    "display_height": PAGE_H,
                    "processing_width": PAGE_W,
                    "processing_height": PAGE_H,
                },
            },
            _dynamic=False,
        )

    def append_save_img(self, *_args, **_kwargs) -> None:
        pass


class _StubTemplate:
    def __init__(self, pre_processors: list[CropOnMarkers]) -> None:
        self.pre_processors = pre_processors


def _make_cropper() -> CropOnMarkers:
    return CropOnMarkers(
        options={
            "type": "aruco",
            "arucoDictionary": "DICT_4X4_50",
            "arucoCornerIds": [0, 1, 2, 3],
            "preserveFullImage": True,
            "referenceMarkerCenters": REFERENCE_CENTRES,
        },
        relative_dir=str(REPO),
        image_instance_ops=_StubImageOps(),
    )


def _render_clean_bgr() -> np.ndarray:
    from webui.services import prefill as prefill_service

    png = prefill_service.generate_single_png(
        "A Student", "School", "Exam", "9010690012", realism_preset="none"
    )
    pil = Image.open(io.BytesIO(png)).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (PAGE_W, PAGE_H), interpolation=cv2.INTER_AREA)


def test_homography_sanity_accepts_reasonable_transform() -> None:
    expected_aspect = PAGE_W / PAGE_H
    # Near-identity homography on a clean sheet.
    ok, reason = _homography_is_sane(
        np.eye(3, dtype=np.float32),
        (PAGE_H, PAGE_W, 3),
        expected_aspect,
    )
    assert ok, reason


def test_homography_sanity_rejects_collapsed_quad() -> None:
    expected_aspect = PAGE_W / PAGE_H
    # Scale the entire page down to a 3x3 pixel speck.
    bad = np.array(
        [[0.005, 0.0, 0.0], [0.0, 0.005, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    ok, reason = _homography_is_sane(bad, (PAGE_H, PAGE_W, 3), expected_aspect)
    assert not ok
    assert "area" in reason or "aspect" in reason or "degenerate" in reason


def test_find_homography_robust_handles_four_points() -> None:
    src = np.array(REFERENCE_CENTRES, dtype=np.float32)
    dst = src + np.array([2.0, -1.5], dtype=np.float32)
    h = _find_homography_robust(src, dst)
    assert h is not None
    assert h.shape == (3, 3)


def test_aruco_cropper_processes_clean_generated_sheet() -> None:
    cropper = _make_cropper()
    sheet = _render_clean_bgr()
    result = cropper._apply_aruco_filter(sheet, "clean.png")
    assert result is not None
    assert result.shape == sheet.shape


def test_aruco_cropper_extrapolates_one_missing_marker() -> None:
    """Cover one marker with white; 3 visible markers must still align."""
    cropper = _make_cropper()
    sheet = _render_clean_bgr()
    # Paint over top-left marker region.
    sheet[0:30, 0:30] = 255
    result = cropper._apply_aruco_filter(sheet, "one-missing.png")
    assert result is not None


def test_auto_orient_rotates_portrait_scan_to_template_aspect() -> None:
    cropper = _make_cropper()
    sheet = _render_clean_bgr()
    portrait_scan = cv2.rotate(sheet, cv2.ROTATE_90_CLOCKWISE)

    result = _auto_orient_to_template(
        "portrait-scan.png",
        portrait_scan,
        _StubTemplate([cropper]),
        cropper.tuning_config,
    )

    assert result.shape[:2] == sheet.shape[:2]
    diff = cv2.absdiff(result, sheet)
    assert float(diff.mean()) < 1.0


def test_aruco_board_is_constructed_when_reference_centers_set() -> None:
    cropper = _make_cropper()
    assert cropper.aruco_board is not None
    assert cropper.reference_marker_centers is not None
    assert len(cropper.reference_marker_centers) == 4


def test_similarity_homography_recovers_rotation_and_scale() -> None:
    src = np.array([[10.0, 10.0], [50.0, 30.0]], dtype=np.float32)
    angle = np.radians(15.0)
    scale = 1.5
    rot = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    dst = (src @ rot.T) * scale + np.array([7.0, -3.0])
    homography = _similarity_homography_from_pairs(src, dst.astype(np.float32))
    assert homography is not None
    projected = cv2.perspectiveTransform(
        src.reshape(-1, 1, 2), homography.astype(np.float32)
    ).reshape(-1, 2)
    assert np.allclose(projected, dst, atol=1e-3)


def test_aruco_cropper_recovers_two_decoded_markers() -> None:
    """Cover two markers entirely; the similarity-transform fallback must align."""
    cropper = _make_cropper()
    sheet = _render_clean_bgr()
    # Erase the two top markers (IDs 0 and 1) so only IDs 2 and 3 remain.
    sheet[0:35, 0:35] = 255
    sheet[0:35, sheet.shape[1] - 35 :] = 255
    result = cropper._apply_aruco_filter(sheet, "two-missing.png")
    assert result is not None
    assert result.shape == sheet.shape


def test_aruco_cropper_rejects_one_decoded_marker() -> None:
    cropper = _make_cropper()
    sheet = _render_clean_bgr()
    sheet[0:35, 0:35] = 255
    sheet[0:35, sheet.shape[1] - 35 :] = 255
    sheet[sheet.shape[0] - 35 :, 0:35] = 255
    result = cropper._apply_aruco_filter(sheet, "three-missing.png")
    assert result is None
