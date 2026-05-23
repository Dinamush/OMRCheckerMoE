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
import itertools
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
from src.constants.common import FIELD_TYPES
from src.constants.image_processing import QUADRANT_DIVISION  # noqa: F401  (used downstream)
from src.template import FieldBlock
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
CORNER_NAMES = ["TL", "TR", "BL", "BR"]
PAIR_TYPES = {
    frozenset((0, 1)): "same_edge_top",
    frozenset((0, 2)): "same_edge_left",
    frozenset((1, 3)): "same_edge_right",
    frozenset((2, 3)): "same_edge_bottom",
    frozenset((0, 3)): "diagonal",
    frozenset((1, 2)): "diagonal",
}


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


class _TemplateContext:
    def __init__(self, field_blocks: list[FieldBlock]) -> None:
        self.field_blocks = field_blocks


def make_prefilled_template_context() -> _TemplateContext:
    from webui.api import _PREFILLED_25Q_TEMPLATE

    field_blocks = []
    for block_name, field_block_object in _PREFILLED_25Q_TEMPLATE[
        "fieldBlocks"
    ].items():
        block_object = dict(field_block_object)
        if "fieldType" in block_object:
            block_object = {
                **FIELD_TYPES[block_object["fieldType"]],
                **block_object,
            }
        block_object = {
            "direction": "vertical",
            "emptyValue": "",
            "bubbleDimensions": _PREFILLED_25Q_TEMPLATE["bubbleDimensions"],
            **block_object,
        }
        field_blocks.append(FieldBlock(block_name, block_object))
    return _TemplateContext(field_blocks)


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
    proc.set_template_context(make_prefilled_template_context())
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
    # ``aruco_marker_boxes`` may legitimately return fewer than four
    # entries when a corner does not fit on the canvas, so look the box
    # up by its ``corner`` field instead of by list position.
    for box in prefill_module.aruco_marker_boxes(PAGE_W, PAGE_H):
        if box["corner"] == corner_index:
            return box["x0"], box["y0"], box["x1"], box["y1"]
    raise ValueError(
        f"No ArUco marker box on the {PAGE_W}x{PAGE_H} canvas for corner "
        f"index {corner_index}; the canvas may be too small to fit a marker."
    )


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


def _perspective_jitter(image: np.ndarray, amount: float) -> np.ndarray:
    h, w = image.shape[:2]
    dx = w * amount
    dy = h * amount
    src = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [dx * 0.2, dy],
            [w - 1 - dx, dy * 0.3],
            [w - 1 - dx * 0.3, h - 1 - dy],
            [dx, h - 1 - dy * 0.2],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _rotate_skew(image: np.ndarray, degrees: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else image


def _motion_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(image, -1, kernel)


def apply_transform(image: np.ndarray, transform: str) -> np.ndarray:
    out = image.copy()
    if transform == "none":
        return out
    if transform == "rotate_180":
        return cv2.rotate(out, cv2.ROTATE_180)
    if transform == "skew_2deg":
        return _rotate_skew(out, 2.0)
    if transform == "skew_5deg":
        return _rotate_skew(out, 5.0)
    if transform == "perspective_mild":
        return _perspective_jitter(out, 0.025)
    if transform == "perspective_strong":
        return _perspective_jitter(out, 0.055)
    if transform == "gaussian_blur":
        return cv2.GaussianBlur(out, (5, 5), 0)
    if transform == "motion_blur":
        return _motion_blur(out, 7)
    if transform == "jpeg_q55":
        return _jpeg_roundtrip(out, 55)
    if transform == "xerox_low_contrast":
        out = cv2.convertScaleAbs(out, alpha=0.72, beta=42)
        out = cv2.GaussianBlur(out, (3, 3), 0)
        return _jpeg_roundtrip(out, 65)
    if transform == "xerox_perspective":
        out = _perspective_jitter(out, 0.035)
        out = cv2.convertScaleAbs(out, alpha=0.76, beta=36)
        return _jpeg_roundtrip(out, 65)
    raise ValueError(f"Unknown transform: {transform}")


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


def _content_alignment_error(reference: np.ndarray, warped: np.ndarray) -> float | None:
    """Mean absolute difference against the clean sheet, ignoring marker zones."""
    if warped is None or warped.shape != reference.shape:
        return None
    mask = np.full(reference.shape[:2], 255, dtype=np.uint8)
    for corner_idx in range(4):
        x0, y0, x1, y1 = _marker_box_on_canvas(corner_idx)
        pad = 12
        mask[
            max(0, y0 - pad): min(mask.shape[0], y1 + pad),
            max(0, x0 - pad): min(mask.shape[1], x1 + pad),
        ] = 0
    diff = cv2.absdiff(reference, warped)
    masked = cv2.mean(diff, mask=mask)
    return float(sum(masked[:3]) / 3.0)


def _pair_type(corners: list[int]) -> str:
    if len(corners) == 0:
        return "none"
    if len(corners) == 1:
        return "single"
    if len(corners) == 2:
        return PAIR_TYPES.get(frozenset(corners), "unknown_pair")
    if len(corners) == 3:
        return "three_missing"
    return "all_missing"


def _scenario_label(corners: list[int], severity: str, transform: str) -> str:
    base = (
        "clean"
        if not corners
        else f"{severity}_" + "+".join(CORNER_NAMES[c] for c in corners)
    )
    return base if transform == "none" else f"{base}__{transform}"


def run_scenario(
    cropper: CropOnMarkers,
    sheets: list[np.ndarray],
    occluded_corners: list[int],
    *,
    severity: str = "full",
    transform: str = "none",
    label: str | None = None,
) -> dict:
    rng = np.random.default_rng(12345)
    successes = 0
    times_ms: list[float] = []
    errors_px: list[float] = []
    alignment_errors: list[float] = []
    confidence_scores: list[float] = []
    confidence_medians: list[float] = []
    confidence_coverages: list[float] = []
    confidence_rejections = 0
    pre_counts: list[int] = []
    failure_count = 0
    for idx, sheet in enumerate(sheets):
        rng_local = np.random.default_rng(rng.integers(0, 1 << 30))
        trial = (
            occlude(sheet, occluded_corners, rng_local, severity=severity)
            if occluded_corners
            else sheet
        )
        trial = apply_transform(trial, transform)
        pre_counts.append(len(detect_marker_centres(trial)))
        t0 = time.perf_counter()
        try:
            result = cropper._apply_aruco_filter(trial.copy(), f"bench-{idx}.png")
        except Exception:
            result = None
        confidence = getattr(cropper, "last_warp_bubble_confidence", None)
        if confidence is not None:
            confidence_scores.append(float(confidence.score))
            confidence_medians.append(float(confidence.median_contrast))
            confidence_coverages.append(float(confidence.coverage))
            if not confidence.ok:
                confidence_rejections += 1
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        if result is None:
            failure_count += 1
            continue
        successes += 1
        err = reprojection_error(result)
        if err is not None:
            errors_px.append(err)
        alignment_error = _content_alignment_error(sheet, result)
        if alignment_error is not None:
            alignment_errors.append(alignment_error)
    n = len(sheets)
    derived_label = label or _scenario_label(occluded_corners, severity, transform)
    homography_mode = (
        "similarity_2marker"
        if median(pre_counts) == 2 and successes
        else "ransac_3plus"
        if median(pre_counts) >= 3 and successes
        else "failed"
    )
    return {
        "scenario": derived_label,
        "n": n,
        "severity": severity,
        "transform": transform,
        "occluded_corners": [CORNER_NAMES[c] for c in occluded_corners],
        "pair_type": _pair_type(occluded_corners),
        "pre_detected_marker_count_median": median(pre_counts) if pre_counts else None,
        "homography_mode_inferred": homography_mode,
        "success_pct": round(100.0 * successes / n, 1) if n else 0.0,
        "successes": successes,
        "failures": failure_count,
        "median_ms": round(median(times_ms), 1) if times_ms else None,
        "mean_ms": round(mean(times_ms), 1) if times_ms else None,
        "p95_ms": round(float(np.percentile(times_ms, 95)), 1) if times_ms else None,
        "max_ms": round(max(times_ms), 1) if times_ms else None,
        "warp_reproj_mean_px": round(mean(errors_px), 2) if errors_px else None,
        "warp_reproj_p95_px": round(float(np.percentile(errors_px, 95)), 2) if errors_px else None,
        "warp_reproj_max_px": round(max(errors_px), 2) if errors_px else None,
        "n_with_warp_reproj_metric": len(errors_px),
        "content_alignment_mean_absdiff": round(mean(alignment_errors), 2) if alignment_errors else None,
        "content_alignment_p95_absdiff": round(float(np.percentile(alignment_errors, 95)), 2) if alignment_errors else None,
        "content_alignment_max_absdiff": round(max(alignment_errors), 2) if alignment_errors else None,
        "confidence_score_mean": round(mean(confidence_scores), 3) if confidence_scores else None,
        "confidence_score_min": round(min(confidence_scores), 3) if confidence_scores else None,
        "confidence_median_contrast_mean": round(mean(confidence_medians), 3) if confidence_medians else None,
        "confidence_coverage_mean": round(mean(confidence_coverages), 3) if confidence_coverages else None,
        "confidence_rejections": confidence_rejections,
    }


def build_scenarios(profile: str) -> list[tuple[list[int], str, str, str]]:
    scenarios: list[tuple[list[int], str, str, str]] = [([], "full", "none", "clean")]
    single_corners = [[idx] for idx in range(4)]
    two_corner_pairs = [list(pair) for pair in itertools.combinations(range(4), 2)]

    if profile == "quick":
        scenarios.extend(
            [
                ([0], "full", "none", "full_TL"),
                ([0, 1], "full", "none", "full_TL+TR"),
                ([2, 3], "full", "none", "full_BL+BR"),
                ([0, 3], "full", "none", "full_TL+BR"),
                ([1, 2], "full", "none", "full_TR+BL"),
                ([0, 1, 2], "full", "none", "full_TL+TR+BL"),
                ([], "full", "perspective_mild", "clean__perspective_mild"),
                ([], "full", "xerox_low_contrast", "clean__xerox_low_contrast"),
                ([0, 1], "full", "perspective_mild", "full_TL+TR__perspective_mild"),
                ([0, 3], "full", "perspective_mild", "full_TL+BR__perspective_mild"),
            ]
        )
        return scenarios

    for severity in ("partial", "medium", "full", "clip"):
        scenarios.extend(
            (corners, severity, "none", _scenario_label(corners, severity, "none"))
            for corners in single_corners
        )
        scenarios.extend(
            (corners, severity, "none", _scenario_label(corners, severity, "none"))
            for corners in two_corner_pairs
        )

    for transform in (
        "skew_2deg",
        "skew_5deg",
        "rotate_180",
        "perspective_mild",
        "perspective_strong",
        "gaussian_blur",
        "motion_blur",
        "jpeg_q55",
        "xerox_low_contrast",
        "xerox_perspective",
    ):
        scenarios.append(([], "full", transform, _scenario_label([], "full", transform)))

    # Combine the riskiest 2-marker topologies with controlled geometry/noise.
    for corners in ([0, 1], [2, 3], [0, 2], [1, 3], [0, 3], [1, 2]):
        for transform in ("perspective_mild", "perspective_strong", "xerox_perspective"):
            scenarios.append(
                (corners, "full", transform, _scenario_label(corners, "full", transform))
            )

    # Always include 3-marker-missing fail-closed probes.
    for corners in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]):
        scenarios.append((list(corners), "full", "none", _scenario_label(list(corners), "full", "none")))
    return scenarios


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=30, help="Number of synthetic sheets per scenario.")
    parser.add_argument("--label", type=str, default="run", help="Label for the output JSON file.")
    parser.add_argument("--out", type=Path, default=REPO / "bench_marker_robustness.json")
    parser.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="quick",
        help="Scenario matrix size. Use full for comprehensive occlusion + geometry coverage.",
    )
    args = parser.parse_args()

    print(f"Generating {args.rows} synthetic sheets...")
    sheets = [render_sheet(i) for i in range(args.rows)]

    cropper = make_cropper()

    # Scenario taxonomy:
    #   * partial / medium / full / clip cover marker occlusion severity.
    #   * transform variants isolate skew, perspective, blur, JPEG, and
    #     Xerox-like low-contrast degradation.
    #   * full profile enumerates all 1-marker and 2-marker combinations
    #     so same-edge and diagonal 2-marker recovery are measured separately.
    scenarios = build_scenarios(args.profile)
    results = []
    for occluded, severity, transform, label in scenarios:
        r = run_scenario(
            cropper,
            sheets,
            occluded,
            severity=severity,
            transform=transform,
            label=label,
        )
        print(
            f"{r['scenario']:>20} | success {r['success_pct']:>5.1f}% "
            f"| median {r['median_ms']:>6} ms "
            f"| align {r['content_alignment_mean_absdiff']} "
            f"| markers {r['pre_detected_marker_count_median']} "
            f"| mode {r['homography_mode_inferred']}"
        )
        results.append(r)

    payload = {
        "label": args.label,
        "rows": args.rows,
        "profile": args.profile,
        "scenario_count": len(scenarios),
        "scenarios": results,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
