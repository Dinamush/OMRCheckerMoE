import os
from dataclasses import dataclass

import cv2
import numpy as np

from src.constants.image_processing import (
    DEFAULT_BLACK_COLOR,
    DEFAULT_BORDER_REMOVE,
    DEFAULT_GAUSSIAN_BLUR_PARAMS_MARKER,
    DEFAULT_LINE_WIDTH,
    DEFAULT_NORMALIZE_PARAMS,
    DEFAULT_WHITE_COLOR,
    ERODE_RECT_COLOR,
    EROSION_PARAMS,
    MARKER_RECTANGLE_COLOR,
    NORMAL_RECT_COLOR,
    QUADRANT_DIVISION,
)
from src.logger import logger
from src.processors.interfaces.ImagePreprocessor import ImagePreprocessor
from src.utils.gpu import warp_perspective as gpu_warp_perspective
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils


# Corner index convention shared by both detection modes:
#   0 = top-left   1 = top-right
#   2 = bottom-left  3 = bottom-right
_CORNER_NAMES = ("top-left", "top-right", "bottom-left", "bottom-right")


# Tolerance band for the post-warp aspect-ratio sanity check. The warped
# page should match the template's expected aspect ratio (page_w / page_h)
# closely; allowing ±10% covers JPEG artefacts and 1-2 px detection noise
# without admitting badly-degenerate homographies.
_HOMOGRAPHY_ASPECT_TOLERANCE = 0.10
# Quad area must fall within these fractions of the full page area.
# Together with the convexity check this catches "page collapsed to a
# sliver" homographies produced by collinear / near-collinear marker sets.
_HOMOGRAPHY_AREA_MIN = 0.30
_HOMOGRAPHY_AREA_MAX = 1.50
_DEGRADED_BUBBLE_SAMPLE_LIMIT = 240
_DEGRADED_BUBBLE_MIN_SAMPLES = 16
_DEGRADED_BUBBLE_CONTRAST_FLOOR = 0.03
_DEGRADED_BUBBLE_MIN_MEDIAN_CONTRAST = 0.04
_DEGRADED_BUBBLE_MIN_COVERAGE = 0.70


@dataclass(frozen=True)
class WarpBubbleConfidence:
    sample_count: int
    median_contrast: float
    coverage: float
    score: float
    ok: bool
    reason: str


def _homography_is_sane(
    homography: np.ndarray,
    image_shape: tuple[int, int],
    expected_aspect: float,
) -> tuple[bool, str]:
    """Validate a marker-derived homography is geometrically reasonable.

    Returns ``(ok, reason)``. Used as a guardrail when ``refineDetectedMarkers``
    pulls a noisy candidate out of ``rejectedCorners`` and produces a
    technically valid but wildly skewed transform. Cheap to run (just a
    handful of numpy ops on 4 points), so it is safe to call on every
    successful detection.
    """
    if homography is None or homography.shape != (3, 3):
        return False, "homography is None or wrong shape"

    h, w = image_shape[:2]
    src = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    try:
        warped = cv2.perspectiveTransform(src, homography).reshape(-1, 2)
    except cv2.error as err:
        return False, f"perspectiveTransform failed: {err}"

    # Convexity: cross-product sign must agree across all 4 vertices.
    signs = []
    for i in range(4):
        a = warped[i]
        b = warped[(i + 1) % 4]
        c = warped[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(np.sign(cross))
    if len(set(signs)) > 1:
        return False, "warped page corners are not convex"

    # Aspect-ratio drift: average top/bottom side / left/right side.
    top = float(np.linalg.norm(warped[1] - warped[0]))
    bottom = float(np.linalg.norm(warped[2] - warped[3]))
    left = float(np.linalg.norm(warped[3] - warped[0]))
    right = float(np.linalg.norm(warped[2] - warped[1]))
    horiz = (top + bottom) * 0.5
    vert = (left + right) * 0.5
    if horiz <= 1.0 or vert <= 1.0:
        return False, "warped quad is degenerate (zero side)"
    aspect = horiz / vert
    if abs(aspect / expected_aspect - 1.0) > _HOMOGRAPHY_ASPECT_TOLERANCE:
        return False, (
            f"aspect drift: got {aspect:.3f} vs expected "
            f"{expected_aspect:.3f} (>±{int(_HOMOGRAPHY_ASPECT_TOLERANCE * 100)}%)"
        )

    # Warped area must be in the same ballpark as the source page.
    polygon = warped.reshape(-1, 1, 2).astype(np.float32)
    area = float(cv2.contourArea(polygon))
    page_area = float(w * h)
    if not (_HOMOGRAPHY_AREA_MIN * page_area <= area <= _HOMOGRAPHY_AREA_MAX * page_area):
        return False, (
            f"warped area {area:.0f}px² is outside "
            f"{int(_HOMOGRAPHY_AREA_MIN * 100)}-{int(_HOMOGRAPHY_AREA_MAX * 100)}% "
            "of the source page"
        )

    return True, "ok"


def _find_homography_robust(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray | None:
    """Compute a homography with RANSAC, falling back gracefully on older OpenCVs.

    With 4 correspondences the RANSAC step degenerates into the same
    least-squares solution as ``getPerspectiveTransform``, but it returns
    a 3x3 ``None`` instead of raising on degenerate input — matching the
    rest of the pipeline's error-handling style. With more correspondences
    (e.g. all 16 marker corners after a board refinement, see future work)
    RANSAC actively rejects mismatched outliers.
    """
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    homography, _ = cv2.findHomography(
        src_pts, dst_pts, method=method, ransacReprojThreshold=3.0
    )
    if homography is None:
        # MAGSAC sometimes refuses noisy 4-point sets; fall back to the
        # plain DLT solver (``method=0``) which always returns something
        # if the points are not all collinear.
        homography, _ = cv2.findHomography(src_pts, dst_pts, method=0)
    return homography


def _similarity_homography_from_pairs(
    src_pts: np.ndarray, dst_pts: np.ndarray
) -> np.ndarray | None:
    """Fit a similarity homography (rotation + uniform scale + translation).

    Used as a degraded-mode recovery when only two ArUco markers decode on
    a Xerox/feeder scan and the other two are present-but-corrupted (common
    when one half of the page has weak ink contrast). Restricting the
    recovery to a 4-DOF similarity transform means there is no extra
    perspective freedom to silently mis-warp content, and the existing
    geometric sanity check on the resulting homography (convexity, aspect,
    area) gates the output further. Returns ``None`` if the source points
    are coincident or the rigid solver could not converge.
    """
    src = np.asarray(src_pts, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_pts, dtype=np.float64).reshape(-1, 2)
    if src.shape != dst.shape or src.shape[0] < 2:
        return None
    matrix, _ = cv2.estimateAffinePartial2D(
        src.astype(np.float32),
        dst.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if matrix is None:
        return None
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = matrix
    return homography


def _iter_template_bubble_boxes(
    template, sample_limit: int = _DEGRADED_BUBBLE_SAMPLE_LIMIT
):
    boxes: list[tuple[int, int, int, int]] = []
    for field_block in getattr(template, "field_blocks", []) or []:
        try:
            box_w, box_h = (int(v) for v in field_block.bubble_dimensions)
        except (TypeError, ValueError):
            continue
        if box_w <= 0 or box_h <= 0:
            continue
        for field_block_bubbles in (
            getattr(field_block, "traverse_bubbles", []) or []
        ):
            for bubble in field_block_bubbles:
                boxes.append((int(bubble.x), int(bubble.y), box_w, box_h))

    if len(boxes) <= sample_limit:
        return boxes
    indices = np.linspace(0, len(boxes) - 1, num=sample_limit, dtype=np.int32)
    return [boxes[int(index)] for index in indices]


def _score_warp_bubble_confidence(image, template) -> WarpBubbleConfidence | None:
    """Score whether expected bubble outlines are still aligned after a warp.

    This is intentionally cheap and is only used on the degraded 2-marker
    path. A correct warp places each template bubble outline on dark printed
    ink, so the outline ring should be darker than nearby paper. Perspective
    drift in the similarity fallback pushes that ring off the printed bubble
    and collapses the contrast/coverage signal.
    """
    if template is None:
        return None

    boxes = _iter_template_bubble_boxes(template)
    if len(boxes) < _DEGRADED_BUBBLE_MIN_SAMPLES:
        return WarpBubbleConfidence(
            sample_count=len(boxes),
            median_contrast=0.0,
            coverage=0.0,
            score=0.0,
            ok=False,
            reason=f"only {len(boxes)} bubble samples available",
        )

    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3
        else image
    )
    height, width = gray.shape[:2]
    contrasts: list[float] = []
    for x, y, box_w, box_h in boxes:
        center_x = x + box_w / 2.0
        center_y = y + box_h / 2.0
        margin = max(4, int(max(box_w, box_h) * 0.6))
        x0 = max(0, int(x - margin))
        y0 = max(0, int(y - margin))
        x1 = min(width, int(x + box_w + margin))
        y1 = min(height, int(y + box_h + margin))
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            continue

        center = (int(round(center_x - x0)), int(round(center_y - y0)))
        outer_axes = (
            max(2, int(round(box_w * 0.55))),
            max(2, int(round(box_h * 0.55))),
        )
        inner_axes = (
            max(1, int(round(box_w * 0.30))),
            max(1, int(round(box_h * 0.30))),
        )
        background_axes = (
            max(3, int(round(box_w * 0.75))),
            max(3, int(round(box_h * 0.75))),
        )
        outer_mask = np.zeros(patch.shape, dtype=np.uint8)
        inner_mask = np.zeros(patch.shape, dtype=np.uint8)
        background_mask = np.full(patch.shape, 255, dtype=np.uint8)
        cv2.ellipse(outer_mask, center, outer_axes, 0, 0, 360, 255, -1)
        cv2.ellipse(inner_mask, center, inner_axes, 0, 0, 360, 255, -1)
        cv2.ellipse(background_mask, center, background_axes, 0, 0, 360, 0, -1)
        outline_mask = cv2.subtract(outer_mask, inner_mask)
        if (
            cv2.countNonZero(outline_mask) < 4
            or cv2.countNonZero(background_mask) < 4
        ):
            continue

        outline_mean = float(cv2.mean(patch, mask=outline_mask)[0])
        background_mean = float(cv2.mean(patch, mask=background_mask)[0])
        contrasts.append((background_mean - outline_mean) / 255.0)

    if len(contrasts) < _DEGRADED_BUBBLE_MIN_SAMPLES:
        return WarpBubbleConfidence(
            sample_count=len(contrasts),
            median_contrast=0.0,
            coverage=0.0,
            score=0.0,
            ok=False,
            reason=f"only {len(contrasts)} valid bubble samples",
        )

    contrast_array = np.asarray(contrasts, dtype=np.float32)
    median_contrast = float(np.median(contrast_array))
    coverage = float(np.mean(contrast_array > _DEGRADED_BUBBLE_CONTRAST_FLOOR))
    normalized_median = min(1.0, max(0.0, median_contrast / 0.16))
    score = 0.60 * normalized_median + 0.40 * coverage
    ok = (
        median_contrast >= _DEGRADED_BUBBLE_MIN_MEDIAN_CONTRAST
        and coverage >= _DEGRADED_BUBBLE_MIN_COVERAGE
    )
    reason = (
        "ok"
        if ok
        else (
            f"median_contrast={median_contrast:.3f} "
            f"coverage={coverage:.2f}"
        )
    )
    return WarpBubbleConfidence(
        sample_count=len(contrasts),
        median_contrast=median_contrast,
        coverage=coverage,
        score=score,
        ok=ok,
        reason=reason,
    )


class CropOnMarkers(ImagePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.tuning_config
        marker_ops = self.options
        self.threshold_circles = []
        self.template_context = None
        self.last_warp_bubble_confidence: WarpBubbleConfidence | None = None
        # img_utils = ImageUtils()

        # Marker detection type: "template_matching" (default) or "aruco"
        self.marker_type = marker_ops.get("type", "template_matching")

        self.preserve_full_image = bool(marker_ops.get("preserveFullImage", False))
        self.reference_marker_centers = self._parse_reference_centers(
            marker_ops.get("referenceMarkerCenters")
        )
        if self.preserve_full_image and self.reference_marker_centers is None:
            raise ValueError(
                "preserveFullImage=true requires referenceMarkerCenters to be "
                "set to the 4 expected marker centres in the (resized) "
                "processing canvas, in top-left, top-right, bottom-left, "
                "bottom-right order."
            )

        if self.marker_type == "aruco":
            dict_name = marker_ops.get("arucoDictionary", "DICT_4X4_50")
            aruco_dict_id = getattr(cv2.aruco, dict_name, None)
            if aruco_dict_id is None:
                raise ValueError(
                    f"Unknown arucoDictionary {dict_name!r}. "
                    "Use a name from cv2.aruco, e.g. 'DICT_4X4_50'."
                )
            raw_ids = marker_ops.get("arucoCornerIds", [0, 1, 2, 3])
            if len(raw_ids) != 4:
                raise ValueError("arucoCornerIds must contain exactly 4 integer IDs.")
            self.aruco_corner_ids: list[int] = [int(i) for i in raw_ids]
            aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
            params = cv2.aruco.DetectorParameters()
            # Detection parameters. ``adaptiveThreshWinSizeMax`` was previously
            # capped at 15 as a speed micro-optimisation, but at the engine's
            # 2x-oversampled processing canvas (e.g. 1332x1030 for the 666x515
            # landscape preset) a 15px adaptive-threshold window is too narrow
            # to separate the marker's black border from a smudge that bleeds
            # ink into the quiet zone. Restoring the OpenCV default of 23
            # rescues markers obscured by feeder-roller streaks or pen smears
            # while still being noticeably cheaper than the default 3..23
            # step-10 sweep because we keep ``step=4`` (more, but smaller,
            # passes).
            #
            # ``CORNER_REFINE_CONTOUR`` fits a polygon to each detected
            # marker's outline so corner positions snap to the true edge of
            # the printed marker rather than to the smudge ring around it.
            # This is what makes a single biased marker recoverable for the
            # candidate-number grid (10 px abutting bubbles). The downstream
            # leave-one-out fallback in ``_choose_best_warp`` then catches the
            # cases where contour refinement makes things worse (e.g. when a
            # marker is so degraded the polygon snaps to the wrong outline)
            # by trying each 3-marker subset and picking the warp with the
            # best bubble-alignment score.
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 23
            params.adaptiveThreshWinSizeStep = 4
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
            params.minMarkerPerimeterRate = 0.02
            params.maxMarkerPerimeterRate = 0.5
            self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            # ----------------------------------------------------------------
            # Board-based recovery for partially occluded sheets.
            #
            # Construct a planar ``cv2.aruco.Board`` from the four reference
            # marker centres so ``refineDetectedMarkers`` can reproject any
            # markers that the initial pass missed (commonly: dog-ears on
            # one or two corners, partial scanner clipping, or finger
            # occlusion during phone scans). With the board, 2 detected
            # markers can frequently be promoted back to 3-4 by re-examining
            # ``rejectedCorners`` near the homography-projected positions of
            # the missing IDs. This is a pure code change — no template
            # update or reprint required.
            # ----------------------------------------------------------------
            half_size = float(marker_ops.get("referenceMarkerHalfSize", 10.0))
            self.reference_marker_half_size = half_size
            if self.reference_marker_centers is not None:
                obj_points = np.zeros((4, 4, 3), dtype=np.float32)
                # ArUco corner ordering inside one marker: TL, TR, BR, BL.
                for i in range(4):
                    ref_corners = self._reference_marker_corners(i)
                    for j in range(4):
                        obj_points[i, j, 0] = ref_corners[j, 0]
                        obj_points[i, j, 1] = ref_corners[j, 1]
                        obj_points[i, j, 2] = 0.0
                board_ids = np.array(self.aruco_corner_ids, dtype=np.int32)
                self.aruco_board = cv2.aruco.Board(obj_points, aruco_dict, board_ids)
                # Permissive refine parameters: paper sheets are flat, the
                # marker layout is known to within sub-pixel accuracy, so we
                # accept candidates within ~30% of expected error and any
                # rotation order (the IDs disambiguate orientation).
                refine_params = cv2.aruco.RefineParameters()
                refine_params.minRepDistance = 10.0
                refine_params.errorCorrectionRate = 3.0
                refine_params.checkAllOrders = True
                self.aruco_detector.setRefineParameters(refine_params)
            else:
                self.aruco_board = None
            # template_matching fields not needed in ArUco mode
            self.marker = None
        else:
            # options with defaults (template_matching mode)
            self.marker_path = os.path.join(
                self.relative_dir, marker_ops.get("relativePath", "omr_marker.jpg")
            )
            self.min_matching_threshold = marker_ops.get("min_matching_threshold", 0.3)
            self.max_matching_variation = marker_ops.get("max_matching_variation", 0.41)
            self.marker_rescale_range = tuple(
                int(r) for r in marker_ops.get("marker_rescale_range", (35, 100))
            )
            self.marker_rescale_steps = int(marker_ops.get("marker_rescale_steps", 10))
            self.apply_erode_subtract = marker_ops.get("apply_erode_subtract", True)
            self.marker_corners = self._parse_marker_corners(
                marker_ops.get("markerCorners")
            )
            self.marker_search_padding = max(
                0, int(marker_ops.get("markerSearchPadding", 20))
            )
            self.fallback_to_expanded_marker_corners = bool(
                marker_ops.get("fallbackToExpandedMarkerCorners", True)
            )
            self.marker = self.load_marker(marker_ops, config)

    @staticmethod
    def _parse_reference_centers(raw):
        if raw is None:
            return None
        if len(raw) != 4:
            raise ValueError(
                "referenceMarkerCenters must contain 4 [x, y] pairs in order "
                "[top-left, top-right, bottom-left, bottom-right]"
            )
        parsed = []
        for idx, point in enumerate(raw):
            if len(point) != 2:
                raise ValueError(
                    f"referenceMarkerCenters[{idx}] must be [x, y]"
                )
            parsed.append((float(point[0]), float(point[1])))
        return parsed

    def _reference_marker_corners(self, corner_idx: int) -> np.ndarray:
        cx, cy = self.reference_marker_centers[corner_idx]
        half = float(self.reference_marker_half_size)
        marker_size = half * 2.0
        quiet_zone = max(6.0, marker_size / 8.0)
        width = float(self.tuning_config.dimensions.processing_width)
        height = float(self.tuning_config.dimensions.processing_height)
        x0 = max(quiet_zone, min(cx - half, width - marker_size - quiet_zone))
        y0 = max(quiet_zone, min(cy - half, height - marker_size - quiet_zone))
        x1 = x0 + marker_size
        y1 = y0 + marker_size
        return np.array(
            [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _parse_marker_corners(raw):
        """Parse per-corner search windows in ``[[y0, y1, x0, x1], ...]`` order.

        The four entries must correspond to top-left, top-right, bottom-left,
        bottom-right, matching the quadrant order used by the matcher. Returns
        ``None`` if unset (fall back to standard 50/50 quadrants).
        """
        if raw is None:
            return None
        if len(raw) != 4:
            raise ValueError(
                "markerCorners must contain exactly 4 entries in order "
                "[top-left, top-right, bottom-left, bottom-right]"
            )
        parsed = []
        for idx, window in enumerate(raw):
            if len(window) != 4:
                raise ValueError(
                    f"markerCorners[{idx}] must be [y0, y1, x0, x1]"
                )
            y0, y1, x0, x1 = (int(v) for v in window)
            if y1 <= y0 or x1 <= x0:
                raise ValueError(
                    f"markerCorners[{idx}] has non-positive size: {window}"
                )
            parsed.append((y0, y1, x0, x1))
        return parsed

    @staticmethod
    def _clip_window(window, height, width):
        y0, y1, x0, x1 = window
        y0c = max(0, min(int(y0), height))
        y1c = max(0, min(int(y1), height))
        x0c = max(0, min(int(x0), width))
        x1c = max(0, min(int(x1), width))
        return y0c, y1c, x0c, x1c

    @classmethod
    def _expand_window(cls, window, padding, height, width):
        y0, y1, x0, x1 = window
        return cls._clip_window(
            (y0 - padding, y1 + padding, x0 - padding, x1 + padding),
            height,
            width,
        )

    @staticmethod
    def _window_to_quad(image, window):
        y0, y1, x0, x1 = window
        return image[y0:y1, x0:x1], [x0, y0]

    @staticmethod
    def _quadrant_window(index, height, width):
        midh, midw = (
            height // QUADRANT_DIVISION["height_factor"],
            width // QUADRANT_DIVISION["width_factor"],
        )
        windows = {
            0: (0, midh, 0, midw),
            1: (0, midh, midw, width),
            2: (midh, height, 0, midw),
            3: (midh, height, midw, width),
        }
        return windows[index]

    @staticmethod
    def _match_marker_in_quad(quad, marker):
        if (
            quad.size == 0
            or quad.shape[0] < marker.shape[0]
            or quad.shape[1] < marker.shape[1]
        ):
            return None, None
        res = cv2.matchTemplate(quad, marker, cv2.TM_CCOEFF_NORMED)
        return res, float(res.max())

    def getBestMatchInWindows(self, search_windows):
        """Pick the marker scale that performs best across the corner search windows.

        ``all_max_t`` from a global ``matchTemplate`` can be inflated by
        non-marker features (filled bubbles, text edges) when the marker
        template has erode-subtract applied but the image does not. By
        scoring scales using only the configured corner windows, the chosen
        baseline is anchored to where the markers actually live.

        Returns ``(best_scale, anchor_max_t)`` where ``anchor_max_t`` is the
        max of per-corner scores at the chosen scale, or ``(None, 0.0)`` when
        no scale fits inside every window.
        """
        descent_per_step = (
            self.marker_rescale_range[1] - self.marker_rescale_range[0]
        ) // self.marker_rescale_steps
        _h, _w = self.marker.shape[:2]
        best_scale = None
        best_aggregate = -1.0
        best_corner_scores: list[float] = []
        for r0 in np.arange(
            self.marker_rescale_range[1],
            self.marker_rescale_range[0],
            -1 * descent_per_step,
        ):
            scale = float(r0 / 100)
            if scale == 0.0:
                continue
            rescaled_marker = ImageUtils.resize_util_h(
                self.marker, u_height=int(_h * scale)
            )
            corner_scores: list[float] = []
            for window_image in search_windows:
                _res, max_t = self._match_marker_in_quad(
                    window_image, rescaled_marker
                )
                if max_t is None:
                    corner_scores = []
                    break
                corner_scores.append(float(max_t))
            if len(corner_scores) != len(search_windows):
                continue
            aggregate = min(corner_scores)
            if aggregate > best_aggregate:
                best_aggregate = aggregate
                best_scale = scale
                best_corner_scores = corner_scores
        if best_scale is None or not best_corner_scores:
            return None, 0.0
        return best_scale, max(best_corner_scores)

    def __str__(self):
        if self.marker_type == "aruco":
            return f"CropOnMarkers[aruco ids={self.aruco_corner_ids}]"
        return self.marker_path

    def set_template_context(self, template) -> None:
        self.template_context = template

    def _attempt_warp_for_subset(
        self,
        *,
        image,
        detected_corners,
        id_to_corner,
        subset_ids,
        expected_aspect,
    ):
        """Compute homography → sanity-check → warp → score, for one ID subset.

        Returns ``(warped_image, confidence, reject_reason)``. ``warped_image``
        is ``None`` when the subset cannot produce a usable warp; in that case
        ``reject_reason`` carries a short description for logging.
        """
        if len(subset_ids) < 2:
            return None, None, f"subset {subset_ids} has <2 markers"

        src_pts, dst_pts = [], []
        for marker_id in subset_ids:
            corner_idx = id_to_corner[marker_id]
            src_pts.extend(detected_corners[marker_id])
            dst_pts.extend(self._reference_marker_corners(corner_idx))
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        if len(subset_ids) >= 3:
            homography = _find_homography_robust(src_pts, dst_pts)
        else:
            # 2-marker degraded recovery: restrict to a similarity transform
            # (rotation + uniform scale + translation) so we cannot silently
            # introduce perspective in the unobserved direction.
            homography = _similarity_homography_from_pairs(src_pts, dst_pts)

        if homography is None:
            return None, None, "homography solver returned None"

        ok, reason = _homography_is_sane(
            homography, image.shape, expected_aspect
        )
        if not ok:
            return None, None, f"sanity check failed ({reason})"

        warped = gpu_warp_perspective(
            image,
            homography,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REPLICATE,
        )
        confidence = _score_warp_bubble_confidence(warped, self.template_context)
        return warped, confidence, None

    def _choose_best_warp(
        self,
        *,
        image,
        detected_corners,
        id_to_corner,
        available_ids,
        expected_aspect,
        file_path,
    ):
        """Pick the homography fit with the strongest bubble-alignment score.

        Always tries the full fit on every detected marker first. If that fit
        fails geometric sanity OR its bubble-confidence score is poor AND we
        have 4 markers, also tries each 3-marker leave-one-out subset and
        picks the highest-scoring candidate. This auto-rejects a single
        biased marker (smudged, partially occluded by feeder ink, scuffed)
        without needing per-marker confidence estimation. With 2 detected
        markers, only the degraded similarity-transform path is tried.

        Returns ``(warped_image, confidence, chosen_subset_ids)`` or
        ``(None, last_confidence, None)`` if no candidate is acceptable.
        ``chosen_subset_ids`` is sorted ascending and equals ``available_ids``
        if the full fit was kept.
        """
        if len(available_ids) < 2:
            logger.error(
                file_path,
                "\nArUco: only "
                f"{len(available_ids)}/4 markers usable for warp — aborting.",
            )
            return None, None, None

        warped_full, conf_full, reason_full = self._attempt_warp_for_subset(
            image=image,
            detected_corners=detected_corners,
            id_to_corner=id_to_corner,
            subset_ids=available_ids,
            expected_aspect=expected_aspect,
        )

        # Score of a confidence record (None → -inf so any real score wins).
        def _score(conf):
            return -1.0 if conf is None else float(conf.score)

        best_warp = warped_full
        best_conf = conf_full
        best_subset = list(available_ids) if warped_full is not None else None

        full_acceptable = (
            warped_full is not None and conf_full is not None and conf_full.ok
        )

        # Only burn the cost of leave-one-out when the full fit is not
        # already passing the bubble-confidence gate. With 4 markers we have
        # 4 LOO subsets to try; with 3 we already are the LOO of the full 4,
        # so further LOO would leave only 2 (handled below as a fallback).
        if not full_acceptable and len(available_ids) >= 3:
            subsets_to_try: list[list[int]] = []
            if len(available_ids) == 4:
                for drop_id in available_ids:
                    subsets_to_try.append(
                        [mid for mid in available_ids if mid != drop_id]
                    )
            elif len(available_ids) == 3:
                # Trying 2-marker similarity fits as a last-ditch fallback
                # only makes sense when the 3-marker fit is geometrically
                # broken (sanity failed); otherwise we'd be downgrading from
                # 3 noisy markers to 2 noisier ones for no reason.
                if warped_full is None:
                    for drop_id in available_ids:
                        subsets_to_try.append(
                            [mid for mid in available_ids if mid != drop_id]
                        )

            for subset in subsets_to_try:
                warped_sub, conf_sub, _reason = self._attempt_warp_for_subset(
                    image=image,
                    detected_corners=detected_corners,
                    id_to_corner=id_to_corner,
                    subset_ids=subset,
                    expected_aspect=expected_aspect,
                )
                if warped_sub is None:
                    continue
                if _score(conf_sub) > _score(best_conf):
                    best_warp = warped_sub
                    best_conf = conf_sub
                    best_subset = sorted(subset)

        # Final accept/reject. With 4 markers detected and a passing full fit
        # we keep the previous loose policy (no bubble gate); the gate only
        # kicks in when we already had a reason to be suspicious (failed
        # sanity, low confidence on full fit, dropped to LOO subset, or
        # degraded 2-marker path). This preserves performance and bench
        # behaviour on the happy path while still catching warps that would
        # otherwise mis-score sheets.
        if best_warp is None:
            logger.error(
                file_path,
                "\nArUco: every marker subset failed to produce a usable "
                f"warp (last reason: {reason_full!r}).",
            )
            return None, best_conf, None

        applied_loo = (
            best_subset is not None and len(best_subset) < len(available_ids)
        )
        is_degraded_2 = len(available_ids) == 2
        suspicious_full = (
            len(available_ids) >= 3
            and not full_acceptable
            and best_subset == list(available_ids)
        )
        if applied_loo or is_degraded_2 or suspicious_full:
            if best_conf is None:
                logger.warning(
                    file_path,
                    "\nArUco: could not score bubble alignment "
                    "confidence on the chosen warp (template geometry "
                    "unavailable). Verify alignment.",
                )
            elif not best_conf.ok:
                logger.error(
                    file_path,
                    "\nArUco: rejected warp because bubble alignment "
                    f"confidence is too low (score={best_conf.score:.2f}, "
                    f"median_contrast={best_conf.median_contrast:.3f}, "
                    f"coverage={best_conf.coverage:.2f}, "
                    f"samples={best_conf.sample_count}; "
                    f"{best_conf.reason}).",
                )
                return None, best_conf, None
            else:
                logger.warning(
                    file_path,
                    "\nArUco: chosen warp passed bubble-alignment confidence "
                    f"gate (score={best_conf.score:.2f}, "
                    f"median_contrast={best_conf.median_contrast:.3f}, "
                    f"coverage={best_conf.coverage:.2f}, "
                    f"samples={best_conf.sample_count}). Verify alignment.",
                )

        return best_warp, best_conf, best_subset

    def exclude_files(self):
        if self.marker_type == "aruco":
            return []
        return [self.marker_path]

    def _apply_aruco_filter(self, image, file_path):
        """Detect 4 ArUco corner markers, orient them by ID, then warp.

        Each corner has a unique ArUco ID so the canonical orientation is
        determined from the IDs alone — no rotation ambiguity.

        Corner index → ID mapping comes from ``self.aruco_corner_ids``:
            index 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right
        """
        self.last_warp_bubble_confidence = None
        config = self.tuning_config

        # ArUco detector works best on grayscale
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image.copy()
        )
        # Pad with white before detection so markers near the image border have
        # a full quiet zone, which the adaptive threshold algorithm requires.
        _PAD = 60
        gray_padded = cv2.copyMakeBorder(
            gray, _PAD, _PAD, _PAD, _PAD, cv2.BORDER_CONSTANT, value=255
        )
        corners_raw, ids_raw, rejected = self.aruco_detector.detectMarkers(gray_padded)

        initial_count = 0 if ids_raw is None else len(ids_raw)

        # ----------------------------------------------------------------
        # Pass 2 — Board-based refinement.
        #
        # ``refineDetectedMarkers`` interpolates each missing marker's
        # projected position from the markers we *did* find (using a
        # global homography on the planar board), then searches
        # ``rejectedCorners`` near that projection for a candidate that
        # decodes correctly with the loose ``errorCorrectionRate`` we set
        # in ``__init__``. Only triggered when at least 1 marker was
        # detected and at least 1 rejected candidate exists, so the fast
        # path on clean sheets pays virtually nothing.
        # ----------------------------------------------------------------
        if (
            self.aruco_board is not None
            and ids_raw is not None
            and 0 < initial_count < 4
            and rejected is not None
            and len(rejected) > 0
        ):
            try:
                corners_raw, ids_raw, rejected, recovered = (
                    self.aruco_detector.refineDetectedMarkers(
                        gray_padded,
                        self.aruco_board,
                        list(corners_raw),
                        ids_raw,
                        list(rejected),
                        None,
                        None,
                    )
                )
                if recovered is not None and len(recovered) > 0:
                    logger.info(
                        file_path,
                        f"\nArUco: refineDetectedMarkers recovered "
                        f"{len(recovered)} marker(s).",
                    )
            except cv2.error as err:
                logger.warning(
                    file_path,
                    f"\nArUco: refineDetectedMarkers raised: {err}",
                )

        # Shift detected corner coordinates back to unpadded image space.
        if corners_raw is not None and len(corners_raw) > 0:
            corners_raw = [c - [[[_PAD, _PAD]]] for c in corners_raw]

        if ids_raw is None or len(ids_raw) == 0:
            logger.error(
                file_path,
                "\nArUco: no markers detected. "
                "Ensure the sheet has ArUco markers printed at the corners.",
            )
            return None

        # Build ID-keyed center and sub-corner maps. The sub-corners let the
        # 3-marker path solve a real projective transform from observed points
        # instead of synthesizing a fourth marker center.
        detected: dict[int, list[float]] = {}
        detected_corners: dict[int, np.ndarray] = {}
        for i, marker_id in enumerate(ids_raw.flatten()):
            # corners_raw[i] has shape (1, 4, 2); mean over the 4 sub-corners
            center = corners_raw[i][0].mean(axis=0).tolist()
            marker_id_int = int(marker_id)
            detected[marker_id_int] = center
            detected_corners[marker_id_int] = corners_raw[i][0].astype(np.float32)

        # Map expected IDs to canonical corner indices
        id_to_corner = {
            id_val: idx for idx, id_val in enumerate(self.aruco_corner_ids)
        }
        centres_by_index: list[list[float] | None] = [None, None, None, None]
        missing_indices: list[int] = []
        for id_val, corner_idx in id_to_corner.items():
            if id_val in detected:
                centres_by_index[corner_idx] = detected[id_val]
            else:
                missing_indices.append(corner_idx)
                logger.warning(
                    file_path,
                    f"\nArUco: marker ID {id_val} "
                    f"({_CORNER_NAMES[corner_idx]}) not detected.",
                )

        detected_count = 4 - len(missing_indices)
        if detected_count < 2:
            logger.error(
                file_path,
                f"\nArUco: only {detected_count}/4 markers detected — "
                "need at least 2 to attempt recovery.",
            )
            return None
        if detected_count == 2 and not self.preserve_full_image:
            # The non-preserveFullImage path warps via ``four_point_transform``
            # which needs all four page corners; the 2-marker similarity
            # recovery only feeds the homography path below.
            logger.error(
                file_path,
                "\nArUco: only 2/4 markers detected and preserveFullImage is "
                "off — cannot recover with the four_point_transform path.",
            )
            return None

        # Extrapolate the one missing corner using an affine fit over the 3
        # detected corners and their reference positions. This is retained for
        # the non-preserveFullImage path, which still needs four page corners
        # for four_point_transform. The preserveFullImage path below uses only
        # observed marker sub-corners for its homography.
        if missing_indices and not self.preserve_full_image:
            if self.reference_marker_centers is None:
                logger.error(
                    file_path,
                    "\nArUco: one marker missing and referenceMarkerCenters "
                    "is not set — cannot extrapolate. Set referenceMarkerCenters.",
                )
                return None
            missing_idx = missing_indices[0]
            good_indices = [k for k in range(4) if k != missing_idx]
            src_three = np.array(
                [centres_by_index[k] for k in good_indices], dtype=np.float32
            )
            dst_three = np.array(
                [self.reference_marker_centers[k] for k in good_indices],
                dtype=np.float32,
            )
            affine = cv2.getAffineTransform(dst_three, src_three)
            missing_ref = np.array(
                [[self.reference_marker_centers[missing_idx]]],
                dtype=np.float32,
            )
            estimated = cv2.transform(missing_ref, affine)[0, 0]
            centres_by_index[missing_idx] = [
                float(estimated[0]), float(estimated[1])
            ]
            logger.warning(
                file_path,
                f"\nArUco: extrapolated {_CORNER_NAMES[missing_idx]} corner "
                f"from 3 detected markers → "
                f"[{round(float(estimated[0]), 1)}, {round(float(estimated[1]), 1)}]",
            )

        if self.preserve_full_image:
            if self.reference_marker_centers is None:
                logger.error(
                    file_path,
                    "\nArUco: preserveFullImage=true requires referenceMarkerCenters.",
                )
                return None
            expected_aspect = (
                config.dimensions.processing_width
                / max(1, config.dimensions.processing_height)
            )
            available_ids = sorted(
                mid for mid in id_to_corner if mid in detected_corners
            )
            warped, chosen_confidence, chosen_subset = self._choose_best_warp(
                image=image,
                detected_corners=detected_corners,
                id_to_corner=id_to_corner,
                available_ids=available_ids,
                expected_aspect=expected_aspect,
                file_path=file_path,
            )
            self.last_warp_bubble_confidence = chosen_confidence
            if warped is None:
                return None
            image = warped
            if chosen_subset is not None and len(chosen_subset) < len(available_ids):
                dropped = sorted(set(available_ids) - set(chosen_subset))
                logger.warning(
                    file_path,
                    "\nArUco: leave-one-out dropped biased marker(s) "
                    f"{dropped} (likely smudged/scuffed) — fit on remaining "
                    f"{len(chosen_subset)} markers "
                    f"(IDs {chosen_subset}) had a better bubble-alignment score.",
                )
        else:
            centres = [c for c in centres_by_index if c is not None]
            image = ImageUtils.four_point_transform(image, np.array(centres))

        if config.outputs.show_image_level >= 2:
            InteractionUtils.show(
                f"ArUco Warped: {file_path}",
                ImageUtils.resize_util(image, config.dimensions.display_width),
                0,
                0,
                [0, 0],
                config=config,
            )
        return image

    def apply_filter(self, image, file_path):
        if self.marker_type == "aruco":
            return self._apply_aruco_filter(image, file_path)
        config = self.tuning_config
        image_instance_ops = self.image_instance_ops
        # Fix for audit finding CORE-4: the conditional was inverted —
        # ``apply_erode_subtract=True`` was returning the *raw* image, even
        # though ``load_marker`` always erode-subtracts the marker template
        # when this flag is set. The two halves of cv2.matchTemplate then
        # operated on mismatched representations, degrading scores.
        if self.apply_erode_subtract:
            image_eroded_sub = ImageUtils.normalize_util(
                image
                - cv2.erode(
                    image,
                    kernel=np.ones(EROSION_PARAMS["kernel_size"]),
                    iterations=EROSION_PARAMS["iterations"],
                )
            )
        else:
            image_eroded_sub = ImageUtils.normalize_util(image)
        # Build per-corner search windows. Prefer explicit ``markerCorners``
        # when provided; otherwise fall back to the classical 50/50 quadrant
        # split.
        quads = {}
        windows = {}
        fallback_quads = {}
        fallback_origins = {}
        fallback_windows = {}
        h1, w1 = image_eroded_sub.shape[:2]
        if self.marker_corners is not None:
            origins = []
            for idx, (y0, y1, x0, x1) in enumerate(self.marker_corners):
                window = self._expand_window(
                    (y0, y1, x0, x1), self.marker_search_padding, h1, w1
                )
                quad, origin = self._window_to_quad(image_eroded_sub, window)
                quads[idx] = quad
                windows[idx] = window
                origins.append(origin)
                fallback_padding = max(self.marker_search_padding * 2, 40)
                fallback_window = self._expand_window(
                    (y0, y1, x0, x1), fallback_padding, h1, w1
                )
                fallback_quad, fallback_origin = self._window_to_quad(
                    image_eroded_sub, fallback_window
                )
                fallback_quads[idx] = fallback_quad
                fallback_origins[idx] = fallback_origin
                fallback_windows[idx] = fallback_window
        else:
            midh, midw = (
                h1 // QUADRANT_DIVISION["height_factor"],
                w1 // QUADRANT_DIVISION["width_factor"],
            )
            origins = [[0, 0], [midw, 0], [0, midh], [midw, midh]]
            quads[0] = image_eroded_sub[0:midh, 0:midw]
            quads[1] = image_eroded_sub[0:midh, midw:w1]
            quads[2] = image_eroded_sub[midh:h1, 0:midw]
            quads[3] = image_eroded_sub[midh:h1, midw:w1]
            windows[0] = (0, midh, 0, midw)
            windows[1] = (0, midh, midw, w1)
            windows[2] = (midh, h1, 0, midw)
            windows[3] = (midh, h1, midw, w1)

            # Draw Quadlines only for the classical split so the debug image
            # keeps its familiar appearance.
            image_eroded_sub[:, midw : midw + 2] = DEFAULT_WHITE_COLOR
            image_eroded_sub[midh : midh + 2, :] = DEFAULT_WHITE_COLOR

        search_windows = [quads[0], quads[1], quads[2], quads[3]]
        best_scale, anchor_max_t = self.getBestMatchInWindows(search_windows)
        if best_scale is None:
            if config.outputs.show_image_level >= 1:
                InteractionUtils.show("Quads", image_eroded_sub, config=config)
            return None

        optimal_marker = ImageUtils.resize_util_h(
            self.marker, u_height=int(self.marker.shape[0] * best_scale)
        )
        _h, w = optimal_marker.shape[:2]
        centres_by_index: list[list[float] | None] = [None, None, None, None]
        failed_corners: list[dict] = []
        sum_t = 0.0
        successful_corners = 0
        corner_scores: list[float] = []
        quarter_match_log = "Matching Marker:  "
        for k in range(0, 4):
            res, max_t = self._match_marker_in_quad(quads[k], optimal_marker)
            used_fallback = False
            if res is None:
                logger.error(
                    file_path,
                    "\nError: marker search window is smaller than marker template in Quad",
                    k + 1,
                    "\n\t search_window",
                    quads[k].shape[:2],
                    "\t search_bounds",
                    windows.get(k),
                    "\t marker_template",
                    optimal_marker.shape[:2],
                    "\n\t Check that config dimensions match template pageDimensions.",
                )
                return None
            if (
                (
                    max_t < self.min_matching_threshold
                    or abs(anchor_max_t - max_t) >= self.max_matching_variation
                )
                and self.marker_corners is not None
                and self.fallback_to_expanded_marker_corners
            ):
                fallback_res, fallback_max_t = self._match_marker_in_quad(
                    fallback_quads[k], optimal_marker
                )
                if (
                    fallback_res is not None
                    and fallback_max_t >= self.min_matching_threshold
                    and abs(anchor_max_t - fallback_max_t)
                    < self.max_matching_variation
                    and fallback_max_t >= max_t
                ):
                    res = fallback_res
                    max_t = fallback_max_t
                    origins[k] = fallback_origins[k]
                    windows[k] = fallback_windows[k]
                    used_fallback = True
            corner_scores.append(round(max_t, 3))
            quarter_match_log += f"Quarter{str(k + 1)}: {str(round(max_t, 3))}\t"
            if (
                max_t < self.min_matching_threshold
                or abs(anchor_max_t - max_t) >= self.max_matching_variation
            ):
                failed_corners.append(
                    {
                        "index": k,
                        "max_t": max_t,
                        "search_window_shape": quads[k].shape[:2],
                        "search_bounds": windows.get(k),
                        "marker_template_shape": optimal_marker.shape[:2],
                        "res": res,
                    }
                )
                if used_fallback:
                    quarter_match_log += (
                        f"Quarter{str(k + 1)} fallback (failed)\t"
                    )
                continue

            pt = np.argwhere(res == max_t)[0]
            pt = [pt[1], pt[0]]
            pt[0] += origins[k][0]
            pt[1] += origins[k][1]
            image = cv2.rectangle(
                image,
                tuple(pt),
                (pt[0] + w, pt[1] + _h),
                MARKER_RECTANGLE_COLOR,
                DEFAULT_LINE_WIDTH,
            )
            image_eroded_sub = cv2.rectangle(
                image_eroded_sub,
                tuple(pt),
                (pt[0] + w, pt[1] + _h),
                ERODE_RECT_COLOR if self.apply_erode_subtract else NORMAL_RECT_COLOR,
                4,
            )
            centres_by_index[k] = [pt[0] + w / 2, pt[1] + _h / 2]
            sum_t += max_t
            successful_corners += 1
            if used_fallback:
                quarter_match_log += f"Quarter{str(k + 1)} fallback\t"

        if failed_corners:
            extrapolation_possible = (
                len(failed_corners) == 1
                and successful_corners == 3
                and self.preserve_full_image
                and self.reference_marker_centers is not None
                and len(self.reference_marker_centers) == 4
            )
            if extrapolation_possible:
                missing_idx = failed_corners[0]["index"]
                good_indices = [k for k in range(4) if k != missing_idx]
                src_three = np.array(
                    [centres_by_index[k] for k in good_indices], dtype=np.float32
                )
                dst_three = np.array(
                    [self.reference_marker_centers[k] for k in good_indices],
                    dtype=np.float32,
                )
                # 3 reference→image correspondences pin down an affine
                # transform; project the missing reference centre back into
                # image space to estimate where the failed marker sits.
                affine = cv2.getAffineTransform(dst_three, src_three)
                missing_ref = np.array(
                    [[self.reference_marker_centers[missing_idx]]],
                    dtype=np.float32,
                )
                estimated = cv2.transform(missing_ref, affine)[0, 0]
                centres_by_index[missing_idx] = [
                    float(estimated[0]),
                    float(estimated[1])
                ]
                logger.warning(
                    file_path,
                    "\nWarning: extrapolated marker for Quad",
                    missing_idx + 1,
                    "from 3 detected corners",
                    "\n\t corner_scores",
                    corner_scores,
                    "\t anchor_max_t",
                    round(anchor_max_t, 3),
                    "\t estimated_centre",
                    [round(estimated[0], 1), round(estimated[1], 1)],
                )
                quarter_match_log += (
                    f"Quarter{missing_idx + 1} extrapolated\t"
                )
            else:
                for failure in failed_corners:
                    logger.error(
                        file_path,
                        "\nError: No circle found in Quad",
                        failure["index"] + 1,
                        "\n\t min_matching_threshold",
                        self.min_matching_threshold,
                        "\t max_matching_variation",
                        self.max_matching_variation,
                        "\t max_t",
                        failure["max_t"],
                        "\t anchor_max_t",
                        anchor_max_t,
                        "\n\t search_window",
                        failure["search_window_shape"],
                        "\t search_bounds",
                        failure["search_bounds"],
                        "\t marker_template",
                        failure["marker_template_shape"],
                        "\n\t corner_scores",
                        corner_scores,
                    )
                if config.outputs.show_image_level >= 1:
                    InteractionUtils.show(
                        f"No markers: {file_path}",
                        image_eroded_sub,
                        0,
                        config=config,
                    )
                    for failure in failed_corners:
                        InteractionUtils.show(
                            f"res_Q{str(failure['index'] + 1)} ({str(failure['max_t'])})",
                            failure["res"],
                            1,
                            config=config,
                        )
                return None

        centres = [c for c in centres_by_index if c is not None]
        logger.info(quarter_match_log)
        logger.info(f"Optimal Scale: {best_scale}")
        # analysis data: average over corners that matched directly (excludes
        # extrapolated corner from the running threshold so a synthetic point
        # doesn't pull the per-batch threshold around).
        if successful_corners:
            self.threshold_circles.append(sum_t / successful_corners)

        if self.preserve_full_image:
            src_pts = np.array(centres, dtype=np.float32)
            dst_pts = np.array(self.reference_marker_centers, dtype=np.float32)
            homography, _ = cv2.findHomography(src_pts, dst_pts, method=0)
            if homography is None:
                logger.error(
                    file_path,
                    "\nError: could not compute homography from detected markers.",
                )
                return None
            # Audit fix CORE-10: the ArUco path already validates each
            # homography with _homography_is_sane (line ~822); the
            # template-matching preserve_full_image path skipped that check
            # and silently warped the image with degenerate (near-collinear
            # marker) homographies. Apply the same guard here.
            expected_aspect = (
                self.tuning_config.dimensions.processing_width
                / max(1.0, float(self.tuning_config.dimensions.processing_height))
            )
            ok, reason = _homography_is_sane(
                homography, image.shape[:2], expected_aspect
            )
            if not ok:
                logger.error(
                    file_path,
                    f"\nError: marker-derived homography failed sanity check: {reason}.",
                )
                return None
            image = gpu_warp_perspective(
                image,
                homography,
                (image.shape[1], image.shape[0]),
                flags=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REPLICATE,
            )
        else:
            image = ImageUtils.four_point_transform(image, np.array(centres))
        # appendSaveImg(1,image_eroded_sub)
        # appendSaveImg(1,image_norm)

        image_instance_ops.append_save_img(2, image_eroded_sub)
        # Debugging image -
        # res = cv2.matchTemplate(image_eroded_sub,optimal_marker,cv2.TM_CCOEFF_NORMED)
        # res[ : , midw:midw+2] = 255
        # res[ midh:midh+2, : ] = 255
        # show("Markers Matching",res)
        if config.outputs.show_image_level >= 2 and config.outputs.show_image_level < 4:
            image_eroded_sub = ImageUtils.resize_util_h(
                image_eroded_sub, image.shape[0]
            )
            image_eroded_sub[:, -DEFAULT_BORDER_REMOVE:] = DEFAULT_BLACK_COLOR
            h_stack = np.hstack((image_eroded_sub, image))
            InteractionUtils.show(
                f"Warped: {file_path}",
                ImageUtils.resize_util(
                    h_stack, int(config.dimensions.display_width * 1.6)
                ),
                0,
                0,
                [0, 0],
                config=config,
            )
        # iterations : Tuned to 2.
        # image_eroded_sub = image_norm - cv2.erode(image_norm, kernel=np.ones((5,5)),iterations=2)
        return image

    def load_marker(self, marker_ops, config):
        """Load and preprocess the template marker image (template_matching mode only)."""
        if self.marker_type == "aruco":
            return None
        if not os.path.exists(self.marker_path):
            # Audit fix CORE-5: previously called exit(31), which terminates
            # the entire worker process bypassing every finally/atexit
            # handler. Inside the web pipeline that orphaned the future and
            # could deadlock the orchestrator. Raise a typed error instead
            # so the worker can report the failure and the pool can recover.
            logger.error(
                "Marker not found at path provided in template: %s",
                self.marker_path,
            )
            raise FileNotFoundError(
                f"Marker image not found: {self.marker_path}"
            )

        marker = cv2.imread(self.marker_path, cv2.IMREAD_GRAYSCALE)

        if "sheetToMarkerWidthRatio" in marker_ops:
            marker = ImageUtils.resize_util(
                marker,
                config.dimensions.processing_width
                / int(marker_ops["sheetToMarkerWidthRatio"]),
            )
        marker = cv2.GaussianBlur(
            marker,
            DEFAULT_GAUSSIAN_BLUR_PARAMS_MARKER["kernel_size"],
            DEFAULT_GAUSSIAN_BLUR_PARAMS_MARKER["sigma_x"],
        )
        marker = cv2.normalize(
            marker,
            None,
            alpha=DEFAULT_NORMALIZE_PARAMS["alpha"],
            beta=DEFAULT_NORMALIZE_PARAMS["beta"],
            norm_type=cv2.NORM_MINMAX,
        )

        if self.apply_erode_subtract:
            marker -= cv2.erode(
                marker,
                kernel=np.ones(EROSION_PARAMS["kernel_size"]),
                iterations=EROSION_PARAMS["iterations"],
            )

        return marker

    # Resizing the marker within scaleRange at rate of descent_per_step to
    # find the best match.
    def getBestMatch(self, image_eroded_sub):
        config = self.tuning_config
        descent_per_step = (
            self.marker_rescale_range[1] - self.marker_rescale_range[0]
        ) // self.marker_rescale_steps
        _h, _w = self.marker.shape[:2]
        res, best_scale = None, None
        all_max_t = 0

        for r0 in np.arange(
            self.marker_rescale_range[1],
            self.marker_rescale_range[0],
            -1 * descent_per_step,
        ):  # reverse order
            s = float(r0 * 1 / 100)
            if s == 0.0:
                continue
            rescaled_marker = ImageUtils.resize_util_h(
                self.marker, u_height=int(_h * s)
            )
            if (
                image_eroded_sub.shape[0] < rescaled_marker.shape[0]
                or image_eroded_sub.shape[1] < rescaled_marker.shape[1]
            ):
                continue
            # res is the black image with white dots
            res = cv2.matchTemplate(
                image_eroded_sub, rescaled_marker, cv2.TM_CCOEFF_NORMED
            )

            max_t = res.max()
            if all_max_t < max_t:
                # print('Scale: '+str(s)+', Circle Match: '+str(round(max_t*100,2))+'%')
                best_scale, all_max_t = s, max_t

        if all_max_t < self.min_matching_threshold:
            logger.warning(
                "\tTemplate matching too low! Consider rechecking preProcessors applied before this."
            )
            # Audit fix CORE-11: if every rescaled marker is larger than
            # the input image the for-loop above never assigns res, so
            # cv2.imshow(None) inside InteractionUtils.show would raise
            # ``cv2.error: Image is empty``. Guard against it.
            if config.outputs.show_image_level >= 1 and res is not None:
                InteractionUtils.show("res", res, 1, 0, config=config)

        if best_scale is None:
            logger.warning(
                "No matchings for given scaleRange:", self.marker_rescale_range
            )
        return best_scale, all_max_t
