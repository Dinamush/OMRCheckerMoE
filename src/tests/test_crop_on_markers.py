from dataclasses import dataclass, field

import cv2
import numpy as np

from src.processors.CropOnMarkers import (
    CropOnMarkers,
    WarpBubbleConfidence,
)


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


# ---------------------------------------------------------------------------
# Leave-one-out marker rejection
# ---------------------------------------------------------------------------


@dataclass
class _StubDims:
    processing_width: int = 666
    processing_height: int = 515


@dataclass
class _StubTuning:
    dimensions: _StubDims = field(default_factory=_StubDims)


def _make_aruco_processor_stub() -> CropOnMarkers:
    """Construct a ``CropOnMarkers`` without running ``__init__``.

    Avoids the production constructor's disk + OpenCV dependencies; the LOO
    helper only needs ``tuning_config``, ``reference_marker_centers``,
    ``reference_marker_half_size`` and ``template_context`` to be set.
    """
    processor = CropOnMarkers.__new__(CropOnMarkers)
    processor.tuning_config = _StubTuning()
    processor.reference_marker_centers = [
        (13.5, 13.2),
        (651.5, 13.2),
        (13.5, 499.0),
        (651.5, 499.0),
    ]
    processor.reference_marker_half_size = 10.0
    processor.template_context = None
    processor.last_warp_bubble_confidence = None
    return processor


def _reference_marker_corners_for_ids(processor, marker_ids):
    """Return a {id: (4, 2) ndarray} of perfect marker sub-corners.

    Builds a perfectly-aligned scan: the detected corners equal the reference
    corners, so a homography fit on every subset has zero residual.
    """
    detected = {}
    for marker_id in marker_ids:
        detected[marker_id] = processor._reference_marker_corners(marker_id)
    return detected


def _reference_marker_centers_for_ids(processor, marker_ids):
    centers = {}
    for marker_id in marker_ids:
        centers[marker_id] = np.mean(
            processor._reference_marker_corners(marker_id), axis=0
        ).tolist()
    return centers


def test_choose_best_warp_keeps_full_fit_when_all_markers_clean(monkeypatch):
    processor = _make_aruco_processor_stub()
    detected = _reference_marker_corners_for_ids(processor, [0, 1, 2, 3])
    image = np.full((515, 666, 3), 220, dtype=np.uint8)
    id_to_corner = {0: 0, 1: 1, 2: 2, 3: 3}

    accept = WarpBubbleConfidence(
        sample_count=100, median_contrast=0.20, coverage=0.95,
        score=0.85, ok=True, reason="ok",
    )

    monkeypatch.setattr(
        "src.processors.CropOnMarkers._score_warp_bubble_confidence",
        lambda *_args, **_kwargs: accept,
    )

    warped, conf, subset = processor._choose_best_warp(
        image=image,
        detected_corners=detected,
        detected_centers=_reference_marker_centers_for_ids(processor, [0, 1, 2, 3]),
        synthetic_marker_ids=set(),
        id_to_corner=id_to_corner,
        available_ids=[0, 1, 2, 3],
        expected_aspect=666 / 515,
        file_path="unit-test",
    )

    assert warped is not None
    assert conf is not None and conf.ok
    assert subset == [0, 1, 2, 3], (
        "full fit was acceptable; LOO must not run"
    )


def test_choose_best_warp_drops_biased_marker_via_loo(monkeypatch):
    """A biased detection on ID 1 must be dropped in favour of the
    3-marker subset {0, 2, 3} when its bubble-alignment score is better."""
    processor = _make_aruco_processor_stub()
    detected = _reference_marker_corners_for_ids(processor, [0, 2, 3])
    # Bias marker ID 1 by 25 px both x and y so any subset containing it
    # produces a measurably worse warp than the LOO subset that drops it.
    biased = processor._reference_marker_corners(1) + np.array(
        [[25.0, 25.0]], dtype=np.float32
    )
    detected[1] = biased
    image = np.full((515, 666, 3), 220, dtype=np.uint8)
    id_to_corner = {0: 0, 1: 1, 2: 2, 3: 3}

    accept = WarpBubbleConfidence(
        sample_count=100, median_contrast=0.20, coverage=0.95,
        score=0.85, ok=True, reason="ok",
    )
    reject = WarpBubbleConfidence(
        sample_count=100, median_contrast=0.01, coverage=0.10,
        score=0.10, ok=False, reason="median_contrast=0.01 coverage=0.10",
    )

    def score(_image, _template):
        # The full fit (includes biased marker) returns reject; the LOO
        # subset that drops ID 1 returns accept. Subsets that drop a
        # different marker still include the biased ID 1 so they also
        # return reject.
        return getattr(score, "_next", reject)

    call_subsets = []
    original = (
        "src.processors.CropOnMarkers"
        "._score_warp_bubble_confidence"
    )

    def fake_score(warped_image, template):
        # Find which subset produced this warp by inspecting which corner
        # of the warped image is closest to (0, 0) — easier: compare the
        # checksum to a recorded baseline. Cheaper proxy: round 1 of the
        # call sequence is full, then 4 LOO calls in order [drop-0,
        # drop-1, drop-2, drop-3].
        call_subsets.append(int(np.sum(warped_image[::40, ::40, 0])))
        nth = len(call_subsets)
        # nth=1 → full (includes biased ID 1)  → reject
        # nth=3 → drop-1 (LOO subset without biased marker) → accept
        if nth == 3:
            return accept
        return reject

    monkeypatch.setattr(original, fake_score)

    warped, conf, subset = processor._choose_best_warp(
        image=image,
        detected_corners=detected,
        detected_centers=_reference_marker_centers_for_ids(processor, [0, 1, 2, 3]),
        synthetic_marker_ids=set(),
        id_to_corner=id_to_corner,
        available_ids=[0, 1, 2, 3],
        expected_aspect=666 / 515,
        file_path="unit-test",
    )

    assert warped is not None
    assert conf is not None and conf.ok
    assert subset == [0, 2, 3], (
        f"expected LOO subset that drops biased ID 1, got {subset!r}"
    )


def test_choose_best_warp_warns_but_accepts_suspicious_full_with_bad_conf(
    monkeypatch,
):
    """When the full 4-marker fit is geometrically sane but bubble confidence
    is low (``suspicious_full`` path), the warp must still be *accepted* with
    a warning — not hard-rejected.  The confidence thresholds are calibrated
    for detecting smudge-biased corners, not for general scan quality; pale
    ink or photocopies trigger this path legitimately and must not route to
    ErrorFiles.
    """
    processor = _make_aruco_processor_stub()
    detected = _reference_marker_corners_for_ids(processor, [0, 1, 2, 3])
    image = np.full((515, 666, 3), 220, dtype=np.uint8)
    id_to_corner = {0: 0, 1: 1, 2: 2, 3: 3}

    reject = WarpBubbleConfidence(
        sample_count=100, median_contrast=0.01, coverage=0.10,
        score=0.10, ok=False, reason="too low",
    )

    monkeypatch.setattr(
        "src.processors.CropOnMarkers._score_warp_bubble_confidence",
        lambda *_args, **_kwargs: reject,
    )

    warped, conf, subset = processor._choose_best_warp(
        image=image,
        detected_corners=detected,
        detected_centers=_reference_marker_centers_for_ids(processor, [0, 1, 2, 3]),
        synthetic_marker_ids=set(),
        id_to_corner=id_to_corner,
        available_ids=[0, 1, 2, 3],
        expected_aspect=666 / 515,
        file_path="unit-test",
    )

    # Full fit was geometrically sane → must be accepted despite low confidence.
    assert warped is not None, (
        "suspicious_full path must warn, not reject a geometrically-valid warp"
    )
    assert subset == [0, 1, 2, 3], "full subset must be returned unchanged"
    assert conf is not None and not conf.ok, "low confidence must be propagated"


def test_choose_best_warp_rejects_loo_subset_when_all_confs_bad(monkeypatch):
    """When a LOO subset was chosen (full sanity failed, one subset produced a
    warp) but the best candidate still fails the bubble-alignment gate, the
    warp must be hard-rejected (``applied_loo=True`` triggers the strict check).
    """
    processor = _make_aruco_processor_stub()
    image = np.full((515, 666, 3), 220, dtype=np.uint8)
    id_to_corner = {0: 0, 1: 1, 2: 2, 3: 3}
    warp_placeholder = image.copy()

    reject = WarpBubbleConfidence(
        sample_count=100, median_contrast=0.01, coverage=0.10,
        score=0.10, ok=False, reason="too low",
    )

    call_n = {"n": 0}

    def fake_attempt(
        self_inner,
        *,
        image,
        detected_corners,
        detected_centers,
        synthetic_marker_ids,
        id_to_corner,
        subset_ids,
        expected_aspect,
    ):
        call_n["n"] += 1
        if len(subset_ids) == 4:
            # Full-set attempt fails sanity so LOO is triggered.
            return None, None, "sanity check failed (injected)"
        # First LOO subset returns a warp with bad confidence.
        return warp_placeholder, reject, None

    monkeypatch.setattr(
        "src.processors.CropOnMarkers.CropOnMarkers._attempt_warp_for_subset",
        fake_attempt,
    )

    warped, conf, subset = processor._choose_best_warp(
        image=image,
        detected_corners={},  # unused — _attempt_warp_for_subset is patched
        detected_centers={},
        synthetic_marker_ids=set(),
        id_to_corner=id_to_corner,
        available_ids=[0, 1, 2, 3],
        expected_aspect=666 / 515,
        file_path="unit-test",
    )

    # LOO subset was taken (applied_loo=True) but conf.ok=False → reject.
    assert warped is None, (
        "applied_loo path must hard-reject when best subset conf is not ok"
    )
    assert subset is None
    assert conf is not None and not conf.ok


def test_choose_best_warp_accepts_two_marker_when_relaxed_gate_passes(monkeypatch):
    """A two-marker degraded warp can be accepted when strict gate narrowly misses.

    Real scans can have valid geometry but slightly reduced bubble-outline
    coverage due to non-uniform shading. For the degraded two-marker path,
    the relaxed gate accepts only when contrast + coverage + score are all
    still strong enough.
    """
    processor = _make_aruco_processor_stub()
    image = np.full((515, 666, 3), 220, dtype=np.uint8)
    id_to_corner = {2: 2, 3: 3}
    warp_placeholder = image.copy()

    near_miss = WarpBubbleConfidence(
        sample_count=200,
        median_contrast=0.054,
        coverage=0.62,
        score=0.45,
        ok=False,
        reason="median_contrast=0.054 coverage=0.62",
    )

    def fake_attempt(
        self_inner,
        *,
        image,
        detected_corners,
        detected_centers,
        synthetic_marker_ids,
        id_to_corner,
        subset_ids,
        expected_aspect,
    ):
        assert subset_ids == [2, 3]
        return warp_placeholder, near_miss, None

    monkeypatch.setattr(
        "src.processors.CropOnMarkers.CropOnMarkers._attempt_warp_for_subset",
        fake_attempt,
    )

    warped, conf, subset = processor._choose_best_warp(
        image=image,
        detected_corners={},
        detected_centers={},
        synthetic_marker_ids=set(),
        id_to_corner=id_to_corner,
        available_ids=[2, 3],
        expected_aspect=666 / 515,
        file_path="unit-test",
    )

    assert warped is not None, "two-marker near-miss should be accepted"
    assert subset == [2, 3]
    assert conf is not None and not conf.ok


def test_choose_best_warp_rejects_two_marker_when_relaxed_gate_fails(monkeypatch):
    """Two-marker degraded warp must still reject when relaxed floor is not met."""
    processor = _make_aruco_processor_stub()
    image = np.full((515, 666, 3), 220, dtype=np.uint8)
    id_to_corner = {2: 2, 3: 3}
    warp_placeholder = image.copy()

    bad_conf = WarpBubbleConfidence(
        sample_count=200,
        median_contrast=0.049,
        coverage=0.62,
        score=0.43,
        ok=False,
        reason="median_contrast=0.049 coverage=0.62",
    )

    def fake_attempt(
        self_inner,
        *,
        image,
        detected_corners,
        detected_centers,
        synthetic_marker_ids,
        id_to_corner,
        subset_ids,
        expected_aspect,
    ):
        assert subset_ids == [2, 3]
        return warp_placeholder, bad_conf, None

    monkeypatch.setattr(
        "src.processors.CropOnMarkers.CropOnMarkers._attempt_warp_for_subset",
        fake_attempt,
    )

    warped, conf, subset = processor._choose_best_warp(
        image=image,
        detected_corners={},
        detected_centers={},
        synthetic_marker_ids=set(),
        id_to_corner=id_to_corner,
        available_ids=[2, 3],
        expected_aspect=666 / 515,
        file_path="unit-test",
    )

    assert warped is None, "two-marker path must still reject weak confidence"
    assert subset is None
    assert conf is not None and not conf.ok
