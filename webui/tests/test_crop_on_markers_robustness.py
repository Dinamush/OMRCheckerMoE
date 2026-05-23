"""Tests for the ArUco cropper robustness upgrade (Board + RANSAC + sanity)."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.processors.CropOnMarkers import (
    CropOnMarkers,
    _find_homography_robust,
    _homography_is_sane,
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
    """Cover one marker with white — the 3-of-4 affine extrapolation must still succeed."""
    cropper = _make_cropper()
    sheet = _render_clean_bgr()
    # Paint over top-left marker region.
    sheet[0:30, 0:30] = 255
    result = cropper._apply_aruco_filter(sheet, "one-missing.png")
    assert result is not None


def test_aruco_board_is_constructed_when_reference_centers_set() -> None:
    cropper = _make_cropper()
    assert cropper.aruco_board is not None
    assert cropper.reference_marker_centers is not None
    assert len(cropper.reference_marker_centers) == 4
