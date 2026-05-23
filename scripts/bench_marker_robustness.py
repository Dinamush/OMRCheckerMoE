"""Benchmark robustness/accuracy/speed of the ArUco crop preprocessor.

Generates synthetic prefilled answer sheets, optionally occludes 0/1/2/3
corners with white rectangles (dog-ear / hand-occlusion stand-ins), runs
each through `CropOnMarkers._apply_aruco_filter`, and reports:

* **Robustness** — fraction of sheets that produced a non-None warp.
* **Accuracy** — mean absolute reprojection error of the four expected
  reference centres after re-detecting markers in the warped output.
* **Speed** — wall-clock time per crop call.

The point is to measure the same code path before vs. after the
Board + ``refineDetectedMarkers`` upgrade. Run twice (once on each git
revision) and diff the outputs.

Usage:
    python scripts/bench_marker_robustness.py --rows 30 --label baseline
    python scripts/bench_marker_robustness.py --rows 30 --label refine
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from statistics import mean, median, pstdev

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prefill_only_package import prefill_answer_sheet_final as prefill_module
from src.constants.image_processing import QUADRANT_DIVISION  # noqa: F401  (used downstream)
from webui.services import prefill as prefill_service

# Lazy-construct one CropOnMarkers via the same plumbing the OMR engine uses,
# but avoid importing the whole entry pipeline (we only need apply_filter).
from src.processors.CropOnMarkers import CropOnMarkers
from src.utils.parsing import OVERRIDE_MERGER  # noqa: F401  (forces tuning_config init)


# Re-use the project's prefilled-25q processing canvas so we measure the
# exact same parameters production runs at.
PAGE_W, PAGE_H = 666, 515
REFERENCE_CENTRES = [
    [13.5, 13.2],
    [651.5, 13.2],
    [13.5, 499.0],
    [651.5, 499.0],
]
ARUCO_IDS = [0, 1, 2, 3]


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


def make_cropper() -> CropOnMarkers:
    image_ops = _StubImageOps()
    proc = CropOnMarkers(
        options={
            "type": "aruco",
            "arucoDictionary": "DICT_4X4_50",
            "arucoCornerIds": ARUCO_IDS,
            "preserveFullImage": True,
            "referenceMarkerCenters": REFERENCE_CENTRES,
        },
        relative_dir=str(REPO),
        image_instance_ops=image_ops,
    )
    return proc


def render_sheet(seed: int) -> np.ndarray:
    """Generate a clean prefilled sheet, return as resized BGR ndarray."""
    student = f"Student {seed}"
    candidate = f"{9000000000 + seed:010d}"
    png = prefill_service.generate_single_png(
        student, "Test School", "Test Exam", candidate, realism_preset="none"
    )
    pil = Image.open(io.BytesIO(png)).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (PAGE_W, PAGE_H), interpolation=cv2.INTER_AREA)
    return bgr


def _marker_box_on_canvas(corner_index: int) -> tuple[int, int, int, int]:
    """Pixel rect of one ArUco marker on the 666x515 processing canvas."""
    boxes = prefill_module.aruco_marker_boxes(PAGE_W, PAGE_H)
    box = boxes[corner_index]
    return box["x0"], box["y0"], box["x1"], box["y1"]


def _full_dogear(out: np.ndarray, corner_idx: int) -> None:
    """Fully obscure a marker (worst-case dog-ear that covers the corner)."""
    h, w = out.shape[:2]
    x0, y0, x1, y1 = _marker_box_on_canvas(corner_idx)
    pad = max(8, int((x1 - x0) * 1.4))
    if corner_idx == 0:
        pts = np.array(
            [[max(0, x0 - pad), max(0, y0 - pad)],
             [min(w - 1, x1 + pad), max(0, y0 - pad)],
             [max(0, x0 - pad), min(h - 1, y1 + pad)]], dtype=np.int32)
    elif corner_idx == 1:
        pts = np.array(
            [[max(0, x0 - pad), max(0, y0 - pad)],
             [min(w - 1, x1 + pad), max(0, y0 - pad)],
             [min(w - 1, x1 + pad), min(h - 1, y1 + pad)]], dtype=np.int32)
    elif corner_idx == 2:
        pts = np.array(
            [[max(0, x0 - pad), max(0, y0 - pad)],
             [max(0, x0 - pad), min(h - 1, y1 + pad)],
             [min(w - 1, x1 + pad), min(h - 1, y1 + pad)]], dtype=np.int32)
    else:
        pts = np.array(
            [[min(w - 1, x1 + pad), max(0, y0 - pad)],
             [max(0, x0 - pad), min(h - 1, y1 + pad)],
             [min(w - 1, x1 + pad), min(h - 1, y1 + pad)]], dtype=np.int32)
    cv2.fillConvexPoly(out, pts, (235, 235, 235))
    if len(pts) >= 2:
        cv2.line(out, tuple(pts[0]), tuple(pts[-1]), (180, 180, 180), thickness=1, lineType=cv2.LINE_AA)


def _dogear_with_size(out: np.ndarray, corner_idx: int, fraction: float) -> None:
    """Triangular page-corner fold sized as a fraction of the marker side.

    ``fraction`` of 0.25 covers ¼ of the marker (very small dog-ear that
    still kills strict ArUco decode), 0.50 covers half, 1.0+ covers the
    whole marker. Used to map out the recovery curve.
    """
    h, w = out.shape[:2]
    x0, y0, x1, y1 = _marker_box_on_canvas(corner_idx)
    side = x1 - x0
    fold_size = max(3, int(side * fraction))
    if corner_idx == 0:
        cx, cy = x0, y0
        pts = np.array(
            [[max(0, cx - 4), max(0, cy - 4)],
             [cx + fold_size, max(0, cy - 4)],
             [max(0, cx - 4), cy + fold_size]], dtype=np.int32)
    elif corner_idx == 1:
        cx, cy = x1, y0
        pts = np.array(
            [[cx - fold_size, max(0, cy - 4)],
             [min(w - 1, cx + 4), max(0, cy - 4)],
             [min(w - 1, cx + 4), cy + fold_size]], dtype=np.int32)
    elif corner_idx == 2:
        cx, cy = x0, y1
        pts = np.array(
            [[max(0, cx - 4), cy - fold_size],
             [max(0, cx - 4), min(h - 1, cy + 4)],
             [cx + fold_size, min(h - 1, cy + 4)]], dtype=np.int32)
    else:
        cx, cy = x1, y1
        pts = np.array(
            [[min(w - 1, cx + 4), cy - fold_size],
             [cx - fold_size, min(h - 1, cy + 4)],
             [min(w - 1, cx + 4), min(h - 1, cy + 4)]], dtype=np.int32)
    cv2.fillConvexPoly(out, pts, (245, 245, 245))
    cv2.line(out, tuple(pts[0]), tuple(pts[-1]), (170, 170, 170), thickness=1, lineType=cv2.LINE_AA)


def _partial_dogear(out: np.ndarray, corner_idx: int) -> None:
    """Small dog-ear: 25% of marker side. Inside the recovery range."""
    _dogear_with_size(out, corner_idx, 0.25)


def _medium_dogear(out: np.ndarray, corner_idx: int) -> None:
    """Medium dog-ear: 50% of marker side. Marginal."""
    _dogear_with_size(out, corner_idx, 0.50)


def _edge_clip(out: np.ndarray, corner_idx: int) -> None:
    """Scanner clipping: 5 mm ≈ 12 px at the bench's resolution removed
    along one full edge, taking out half a marker. ``refineDetectedMarkers``
    can recover this when the opposite-edge marker is intact."""
    h, w = out.shape[:2]
    clip_px = 12
    if corner_idx in (0, 1):  # top edge
        out[:clip_px, :] = 255
    if corner_idx in (2, 3):  # bottom edge
        out[h - clip_px:, :] = 255
    if corner_idx in (0, 2):  # left edge
        out[:, :clip_px] = 255
    if corner_idx in (1, 3):  # right edge
        out[:, w - clip_px:] = 255


_OCCLUSION_FUNCS = {
    "full": _full_dogear,
    "partial": _partial_dogear,
    "medium": _medium_dogear,
    "clip": _edge_clip,
}


def occlude(
    image: np.ndarray,
    corners_to_occlude: list[int],
    rng: np.random.Generator,
    *,
    severity: str = "full",
) -> np.ndarray:
    out = image.copy()
    fn = _OCCLUSION_FUNCS[severity]
    for corner_idx in corners_to_occlude:
        fn(out, corner_idx)
    # Light speckle so flat patches don't fool the detector with quad-like noise.
    h, w = out.shape[:2]
    for _ in range(int(rng.integers(20, 40))):
        ry = int(rng.integers(0, h))
        rx = int(rng.integers(0, w))
        out[ry, rx] = (190, 190, 190)
    return out


def detect_marker_centres(image_bgr: np.ndarray) -> dict[int, np.ndarray]:
    """Detect ArUco markers and return id → centre (x, y) mapping."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    pad = 60
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 15
    params.adaptiveThreshWinSizeStep = 4
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 0.5
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(padded)
    out: dict[int, np.ndarray] = {}
    if ids is None:
        return out
    for i, marker_id in enumerate(ids.flatten()):
        c = corners[i][0].mean(axis=0) - np.array([pad, pad], dtype=np.float32)
        out[int(marker_id)] = c
    return out


def reprojection_error(warped: np.ndarray) -> float | None:
    """Mean abs distance between detected marker centres and the references.

    Returns None if fewer than 3 markers detected on the warp.
    """
    found = detect_marker_centres(warped)
    diffs: list[float] = []
    for marker_id, ref in zip(ARUCO_IDS, REFERENCE_CENTRES):
        if marker_id not in found:
            continue
        diffs.append(float(np.linalg.norm(found[marker_id] - np.array(ref, dtype=np.float32))))
    if len(diffs) < 3:
        return None
    return float(mean(diffs))


def run_scenario(
    cropper: CropOnMarkers,
    sheets: list[np.ndarray],
    occluded_corners: list[int],
    *,
    severity: str = "full",
    label: str | None = None,
) -> dict:
    rng = np.random.default_rng(12345)
    successes = 0
    times_ms: list[float] = []
    errors_px: list[float] = []
    for idx, sheet in enumerate(sheets):
        rng_local = np.random.default_rng(rng.integers(0, 1 << 30))
        trial = (
            occlude(sheet, occluded_corners, rng_local, severity=severity)
            if occluded_corners
            else sheet
        )
        t0 = time.perf_counter()
        try:
            result = cropper._apply_aruco_filter(trial.copy(), f"bench-{idx}.png")
        except Exception:
            result = None
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        if result is None:
            continue
        successes += 1
        err = reprojection_error(result)
        if err is not None:
            errors_px.append(err)
    n = len(sheets)
    derived_label = (
        label
        or (
            f"{severity}_" + "+".join(["TL", "TR", "BL", "BR"][c] for c in occluded_corners)
            if occluded_corners
            else "clean"
        )
    )
    return {
        "scenario": derived_label,
        "n": n,
        "success_pct": round(100.0 * successes / n, 1) if n else 0.0,
        "successes": successes,
        "median_ms": round(median(times_ms), 1) if times_ms else None,
        "mean_ms": round(mean(times_ms), 1) if times_ms else None,
        "p95_ms": round(float(np.percentile(times_ms, 95)), 1) if times_ms else None,
        "warp_reproj_mean_px": round(mean(errors_px), 2) if errors_px else None,
        "warp_reproj_max_px": round(max(errors_px), 2) if errors_px else None,
        "n_with_warp_reproj_metric": len(errors_px),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=30, help="Number of synthetic sheets per scenario.")
    parser.add_argument("--label", type=str, default="run", help="Label for the output JSON file.")
    parser.add_argument("--out", type=Path, default=REPO / "bench_marker_robustness.json")
    args = parser.parse_args()

    print(f"Generating {args.rows} synthetic sheets...")
    sheets = [render_sheet(i) for i in range(args.rows)]

    cropper = make_cropper()

    # Scenario taxonomy (graded by damage tier):
    #   * partial (25%) — small dog-ear, only the page-edge tip of the
    #     marker is folded away. Inside the recoverable zone for a 24-px
    #     marker after the upgrade.
    #   * medium  (50%) — larger fold; on the edge of recoverable.
    #   * full          — entire marker zone painted white. Only the
    #     existing 3-of-4 affine extrapolation can save these (works on
    #     1 corner missing; 2+ corners is fundamentally unrecoverable
    #     without printing larger markers).
    #   * clip          — 12-px scanner edge clip (≈5 mm at 200 DPI A4).
    scenarios: list[tuple[list[int], str, str]] = [
        ([], "full", "clean"),
        ([0], "partial", "partial_TL"),
        ([3], "partial", "partial_BR"),
        ([0, 1], "partial", "partial_TL+TR"),
        ([0, 3], "partial", "partial_TL+BR"),
        ([0, 1, 2], "partial", "partial_TL+TR+BL"),
        ([0], "medium", "medium_TL"),
        ([0, 1], "medium", "medium_TL+TR"),
        ([0], "full", "full_TL"),
        ([0, 1], "full", "full_TL+TR"),
        ([0, 3], "full", "full_TL+BR"),
        ([0], "clip", "clip_TL_edge"),
        ([0, 1], "clip", "clip_top_edge"),
    ]
    results = []
    for occluded, severity, label in scenarios:
        r = run_scenario(cropper, sheets, occluded, severity=severity, label=label)
        print(
            f"{r['scenario']:>20} | success {r['success_pct']:>5.1f}% "
            f"| median {r['median_ms']:>6} ms | reproj {r['warp_reproj_mean_px']} px"
        )
        results.append(r)

    payload = {"label": args.label, "rows": args.rows, "scenarios": results}
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
