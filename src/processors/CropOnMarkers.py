import os

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
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils


class CropOnMarkers(ImagePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.tuning_config
        marker_ops = self.options
        self.threshold_circles = []
        # img_utils = ImageUtils()

        # options with defaults
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
        return self.marker_path

    def exclude_files(self):
        return [self.marker_path]

    def apply_filter(self, image, file_path):
        config = self.tuning_config
        image_instance_ops = self.image_instance_ops
        image_eroded_sub = ImageUtils.normalize_util(
            image
            if self.apply_erode_subtract
            else (
                image
                - cv2.erode(
                    image,
                    kernel=np.ones(EROSION_PARAMS["kernel_size"]),
                    iterations=EROSION_PARAMS["iterations"],
                )
            )
        )
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
            image = cv2.warpPerspective(
                image,
                homography,
                (image.shape[1], image.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
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
        if not os.path.exists(self.marker_path):
            logger.error(
                "Marker not found at path provided in template:",
                self.marker_path,
            )
            exit(31)

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
            if config.outputs.show_image_level >= 1:
                InteractionUtils.show("res", res, 1, 0, config=config)

        if best_scale is None:
            logger.warning(
                "No matchings for given scaleRange:", self.marker_rescale_range
            )
        return best_scale, all_max_t
