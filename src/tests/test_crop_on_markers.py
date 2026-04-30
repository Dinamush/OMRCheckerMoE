import cv2
import numpy as np

from src.processors.CropOnMarkers import CropOnMarkers


MARKER_SIZE = 24


def _build_synthetic_marker() -> np.ndarray:
    """A solid black square on white reads as a strong, scale-friendly marker."""
    marker = np.full((MARKER_SIZE, MARKER_SIZE), 255, dtype=np.uint8)
    cv2.rectangle(
        marker,
        (3, 3),
        (MARKER_SIZE - 4, MARKER_SIZE - 4),
        color=0,
        thickness=-1,
    )
    return marker


def _window_with_marker(
    marker: np.ndarray, window_shape=(80, 80), marker_origin=(20, 20)
) -> np.ndarray:
    window = np.full(window_shape, 255, dtype=np.uint8)
    y, x = marker_origin
    h, w = marker.shape[:2]
    window[y : y + h, x : x + w] = marker
    return window


def _make_processor_stub(marker: np.ndarray) -> CropOnMarkers:
    """Skip __init__ so we can unit-test geometry without disk fixtures."""
    processor = CropOnMarkers.__new__(CropOnMarkers)
    processor.marker = marker
    processor.marker_rescale_range = [90, 110]
    processor.marker_rescale_steps = 5
    return processor


def test_marker_window_padding_clips_at_image_boundaries():
    window = (0, 40, 626, 666)

    expanded = CropOnMarkers._expand_window(window, padding=20, height=515, width=666)

    assert expanded == (0, 60, 606, 666)


def test_marker_matching_returns_none_when_window_is_too_small():
    search_window = np.zeros((10, 10), dtype=np.uint8)
    marker = np.zeros((20, 20), dtype=np.uint8)

    result, score = CropOnMarkers._match_marker_in_quad(search_window, marker)

    assert result is None
    assert score is None


def test_get_best_match_in_windows_picks_scale_that_satisfies_all_corners():
    marker = _build_synthetic_marker()
    processor = _make_processor_stub(marker)

    windows = [_window_with_marker(marker) for _ in range(4)]

    best_scale, anchor_max_t = processor.getBestMatchInWindows(windows)

    assert best_scale is not None
    assert 0.85 <= best_scale <= 1.15
    # Per-window peak should be near-perfect when the marker fits cleanly.
    assert anchor_max_t > 0.9


def test_get_best_match_in_windows_returns_none_when_any_window_is_too_small():
    marker = _build_synthetic_marker()
    processor = _make_processor_stub(marker)

    windows = [
        _window_with_marker(marker),
        _window_with_marker(marker),
        _window_with_marker(marker),
        np.zeros((10, 10), dtype=np.uint8),
    ]

    best_scale, anchor_max_t = processor.getBestMatchInWindows(windows)

    assert best_scale is None
    assert anchor_max_t == 0.0


def test_get_best_match_in_windows_ignores_spurious_peak_outside_windows():
    """A heavy-bubble peak elsewhere on the page should never inflate the
    anchor score — only the configured corner windows feed into matching."""
    marker = _build_synthetic_marker()
    processor = _make_processor_stub(marker)

    # Each corner window contains a partially occluded marker so peak
    # correlation is moderate, not perfect.
    occluded_marker = marker.copy()
    occluded_marker[:6, :] = 200  # erase the top stripe of the marker
    rng = np.random.default_rng(42)
    corner_windows = []
    for _ in range(4):
        window = _window_with_marker(occluded_marker)
        # Add random gaussian-like noise so cross-correlation is imperfect.
        noise = rng.integers(0, 40, window.shape, dtype=np.int16)
        corner_windows.append(np.clip(window.astype(np.int16) - noise, 0, 255).astype(np.uint8))

    # Construct a separate "rest of the page" region that contains the *exact*
    # marker (high correlation) — by design, this region is NOT passed to
    # getBestMatchInWindows, so it must not influence the result.
    spurious_region = _window_with_marker(marker)
    full_page_max = float(
        cv2.matchTemplate(spurious_region, marker, cv2.TM_CCOEFF_NORMED).max()
    )

    _, anchor_max_t = processor.getBestMatchInWindows(corner_windows)

    assert full_page_max > 0.95
    # Anchor score reflects the corner windows only — must stay well below the
    # spurious global peak so the variation gate doesn't reject real markers.
    assert anchor_max_t < full_page_max - 0.1


def test_three_of_four_corner_extrapolation_recovers_missing_corner():
    """Affine projection from 3 reference→image correspondences should put the
    missing 4th corner within a couple of pixels of its true location."""
    reference = [
        [50.0, 50.0],
        [950.0, 50.0],
        [50.0, 950.0],
        [950.0, 950.0],
    ]
    # Apply a small affine (translation + tiny shear) to mimic a real scan.
    affine_truth = np.array([[1.01, 0.02, 8.0], [0.01, 0.99, 12.0]], dtype=np.float32)
    detected = []
    for pt in reference:
        ref_pt = np.array([[pt]], dtype=np.float32)
        warped = cv2.transform(ref_pt, affine_truth)[0, 0]
        detected.append([float(warped[0]), float(warped[1])])

    missing_idx = 1  # pretend top-right failed
    good_indices = [k for k in range(4) if k != missing_idx]
    src_three = np.array([detected[k] for k in good_indices], dtype=np.float32)
    dst_three = np.array([reference[k] for k in good_indices], dtype=np.float32)

    affine_est = cv2.getAffineTransform(dst_three, src_three)
    missing_ref = np.array([[reference[missing_idx]]], dtype=np.float32)
    estimated = cv2.transform(missing_ref, affine_est)[0, 0]

    expected = detected[missing_idx]
    assert abs(float(estimated[0]) - expected[0]) < 1.5
    assert abs(float(estimated[1]) - expected[1]) < 1.5
