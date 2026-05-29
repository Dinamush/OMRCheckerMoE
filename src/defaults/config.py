from dotmap import DotMap

CONFIG_DEFAULTS = DotMap(
    {
        "dimensions": {
            "display_height": 2480,
            "display_width": 1640,
            "processing_height": 820,
            "processing_width": 666,
        },
        "threshold_params": {
            "GAMMA_LOW": 0.7,
            "MIN_GAP": 30,
            "MIN_JUMP": 25,
            "CONFIDENT_SURPLUS": 5,
            "JUMP_DELTA": 30,
            "PAGE_TYPE_FOR_THRESHOLD": "white",
            # Fraction of each bubble's width/height to trim off every side
            # before measuring its mean intensity. Because consecutive
            # bubbles in a strip can abut with zero gap (notably QTYPE_INT
            # candidate-number columns where bubblesGap == bubbleDimensions),
            # a slightly skewed scan lets a filled bubble bleed dark pixels
            # into its neighbour's sampling rectangle and trip a false
            # second mark (e.g. candidate "...74" read as "...MR(78)4").
            # Insetting the sampled rectangle creates a guard band between
            # neighbouring ROIs so residual skew no longer cross-contaminates
            # them, while still sampling the bubble centre where real marks
            # concentrate. 0.0 reproduces the legacy full-box behaviour.
            # NOTE: a uniform inset shifts the page-wide global threshold and
            # can perturb unrelated blocks, so it is left OFF by default; the
            # candidate-column bleed is handled directly by the bounded
            # vertical re-centering below.
            "BUBBLE_INSET_RATIO": 0.0,
            # Bounded per-strip vertical re-centering for vertically-stacked,
            # zero-gap strips (QTYPE_INT candidate/roll columns). A skewed or
            # partially-marker-detected scan leaves the warped digit grid
            # vertically misregistered, so a filled bubble straddles the shared
            # edge between its ROI and a neighbour's and trips a false MR(...)
            # (e.g. candidate "...74" read as "...MR(78)4"). For each such
            # strip we search a small vertical offset (up to this fraction of
            # the bubble pitch) and keep the offset that makes ONE bubble most
            # clearly the darkest, re-centring the mark inside its own ROI.
            # The search is deliberately capped below half the pitch so it can
            # only sharpen alignment toward the nearest printed bubble — never
            # far enough to lock onto a neighbour — so an unrecoverable skew
            # still resolves to a safe MR(...) rather than a silent wrong digit.
            # 0.0 disables re-centering.
            "VERTICAL_RECENTER_MAX_RATIO": 0.4,
            # Internal oversample multiplier for the measurement canvas. The
            # template's logical coordinate space is preserved (so the webui,
            # prefill pipeline, and student_fill calibration keep working at
            # the canonical resolution), but the engine resizes inputs and
            # scales bubble geometry by this factor before measuring. Bigger
            # = more pixels per bubble = a sharper darkest-vs-runner-up gap,
            # tighter ArUco corner refinement, and less residual skew after
            # the homography warp — directly improving candidate-number
            # robustness on partial-marker scans. Empirically benched on the
            # production sheet: 1.5x and 2.0x recover the doubly-degraded
            # cases that 1.0x quarantines, with ~5% time cost; 3.0x regresses
            # so the safe upper bound is 2.0x.
            "OVERSAMPLE_SCALE": 1.0,
        },
        "alignment_params": {
            # Note: 'auto_align' enables automatic template alignment, use if the scans show slight misalignments.
            "auto_align": False,
            "match_col": 5,
            "max_steps": 20,
            "stride": 1,
            "thickness": 3,
        },
        "outputs": {
            "show_image_level": 0,
            "save_image_level": 0,
            "save_detections": True,
            "filter_out_multimarked_files": False,
            "multi_mark_equal_delta": 0.06,
            "max_workers": 4,
            "candidate_regex": None,
        },
    },
    _dynamic=False,
)
