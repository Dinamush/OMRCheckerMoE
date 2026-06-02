"""Regression tests: the adjacent-gap local-threshold fallback must not
create ghost answers for truly-empty MCQ strips.

Background
----------
``get_local_threshold`` normally falls back to ``global_thr`` when it cannot
find a confident local jump.  On templates that contain a dark section (e.g.
candidate-number columns whose filled bubbles are at ~40 intensity), the
page-wide ``global_thr`` is pulled down to ~70–90 — far below every *answer*
bubble (~130–250).  The old code therefore classified all answer bubbles as
empty and returned NR for every lightly-filled answer question.

The fix uses the adjacent-gap midpoint ``(q_arr[0] + q_arr[1]) / 2`` when
all strip values are above ``global_thr`` AND the adjacent gap is ≥ 2 intensity
units (a signal above typical scan noise).  The 2-unit guard prevents the
fallback from firing on a truly-blank strip where the four near-uniform bubble
means differ only by scan noise (< 1 unit for clean scans).

These tests verify both sides of that boundary:

1. **Ghost protection** — a strip where all four bubbles are at the same
   "unfilled" intensity (or with ≤ 1 unit of variation) must still return NR,
   even when ``global_thr`` is far below every value in the strip.

2. **Light-fill detection** — a strip where the chosen bubble is measurably
   darker than the others (adj gap ≥ 2) must be detected, even when the fill
   is so faint that the global_thr would have missed it.

The tests use a synthetic image that replicates the MoE-Landscape geometry:
  • Candidate-number columns (filled at intensity 20) bias ``global_thr`` low.
  • MCQ answer strips are either blank (uniform high intensity) or lightly
    filled (one bubble a few units darker than the rest).
No ArUco markers or pre-processors are needed because the image is delivered
directly in template space.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from freezegun import freeze_time

from main import entry_point_for_args

# ---------------------------------------------------------------------------
# Template geometry — mirrors the MoE-April-2026-Landscape page dimensions
# and candidate-number block.
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = 666, 515

# Candidate-number block (identical to the production template, used to
# create a biased global_thr that mimics the real-scan problem).
CAND_ORIGIN_X = 430
CAND_ORIGIN_Y = 103
CAND_LABELS_GAP = 21.5    # X step between columns (cand1..10)
CAND_BUBBLES_GAP = 10.0   # Y step between digits 0..9 within a column
BUBBLE = 10

# Answer MCQ block — simplified 4-question layout with no pre-processors.
MCQ_ORIGIN_X = 50
MCQ_ORIGIN_Y = 200
MCQ_LABELS_GAP = 60       # X step between question groups
MCQ_BUBBLES_GAP = 20      # Y step between options A/B/C/D
MCQ_N_QUESTIONS = 4

# Pixel intensities
CAND_FILLED = 0       # pure black — anchors the image minimum so that
                      # normalize_util is an identity (min=0, max=255) and the
                      # carefully chosen MCQ intensity values survive unchanged
CAND_EMPTY = 255      # white (unfilled candidate bubble)

# Lightly-filled answer bubble: dark enough to stand apart with adj_gap ≥ 2,
# but so faint that global_thr alone would miss it (well above 86, the biased
# global_thr produced by this synthetic sheet).
LIGHT_FILLED = 145    # the "filled" option in a lightly-filled question
LIGHT_EMPTY = 155     # the three "empty" options in the same question

# Truly-blank question: all four options at a uniform unfilled intensity.
# adj_gap = 0 → the guard prevents the adjacent-midpoint fallback → NR.
BLANK_UNIFORM = 215

# Blank question with minimal scan noise (1-unit variation across 4 bubbles).
# adj_gap = 1 < 2 → guard still prevents the fallback → NR.
BLANK_NOISY = [213, 214, 214, 215]   # sorted intensities for the 4 options

# Template dict used for the test run (no pre-processors, template space).
TEMPLATE = {
    "pageDimensions": [PAGE_W, PAGE_H],
    "bubbleDimensions": [BUBBLE, BUBBLE],
    "preProcessors": [],
    "customLabels": {"CandidateNumber": ["cand1..10"]},
    "outputColumns": ["CandidateNumber", "q1", "q2", "q3", "q4"],
    "fieldBlocks": {
        "CandidateNumber": {
            "origin": [CAND_ORIGIN_X, CAND_ORIGIN_Y],
            "bubblesGap": CAND_BUBBLES_GAP,
            "labelsGap": CAND_LABELS_GAP,
            "fieldLabels": ["cand1..10"],
            "fieldType": "QTYPE_INT",
        },
        "MCQBlock": {
            "origin": [MCQ_ORIGIN_X, MCQ_ORIGIN_Y],
            "bubblesGap": MCQ_BUBBLES_GAP,
            "labelsGap": MCQ_LABELS_GAP,
            "fieldLabels": ["q1..4"],
            "emptyValue": "NR",
            "fieldType": "QTYPE_MCQ4",
        },
    },
}

CONFIG = {
    "dimensions": {
        "display_height": PAGE_H,
        "display_width": PAGE_W,
        "processing_height": PAGE_H,
        "processing_width": PAGE_W,
    },
    "outputs": {"show_image_level": 0, "save_image_level": 0},
    "threshold_params": {"MIN_JUMP": 15, "OVERSAMPLE_SCALE": 1},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_sheet() -> np.ndarray:
    return np.full((PAGE_H, PAGE_W), CAND_EMPTY, dtype=np.uint8)


def _fill_cand_bubble(img: np.ndarray, column: int, digit: int, intensity: int) -> None:
    x0 = round(CAND_ORIGIN_X + column * CAND_LABELS_GAP)
    y0 = round(CAND_ORIGIN_Y + digit * CAND_BUBBLES_GAP)
    img[y0 : y0 + BUBBLE, x0 : x0 + BUBBLE] = intensity


def _fill_mcq_bubble(
    img: np.ndarray, question_idx: int, option_idx: int, intensity: int
) -> None:
    """Paint one MCQ bubble.

    QTYPE_MCQ4 is ``direction: horizontal``, so:
      * ``bubblesGap`` steps in the **X** direction (between options A/B/C/D)
      * ``labelsGap``  steps in the **Y** direction (between questions)

    question_idx: 0-based (0 = q1, 1 = q2, …)
    option_idx:   0-based (0 = A, 1 = B, 2 = C, 3 = D)
    """
    x0 = round(MCQ_ORIGIN_X + option_idx * MCQ_BUBBLES_GAP)   # X for option
    y0 = round(MCQ_ORIGIN_Y + question_idx * MCQ_LABELS_GAP)  # Y for question
    img[y0 : y0 + BUBBLE, x0 : x0 + BUBBLE] = intensity


def _run_engine(tmp_path: Path, img: np.ndarray) -> dict[str, str]:
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "template.json").write_text(json.dumps(TEMPLATE), encoding="utf-8")
    (inputs / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    assert cv2.imwrite(str(inputs / "sheet.png"), img)

    output_dir = tmp_path / "out"
    with freeze_time("1970-01-01"):
        entry_point_for_args(
            {
                "input_paths": [str(inputs)],
                "output_dir": str(output_dir),
                "debug": False,
                "autoAlign": False,
                "setLayout": False,
                "silent": True,
            }
        )

    results = sorted(output_dir.rglob("Results_*.csv"))
    assert results, f"no Results CSV produced under {output_dir}"
    with results[0].open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "Results CSV had no data rows"
    return rows[0]


def _sheet_with_cand_bias(candidate_digit: int = 5) -> np.ndarray:
    """Return a blank sheet with candidate-number columns filled.

    Filling one digit per column with intensity CAND_FILLED=20 creates a
    distribution that forces ``global_thr`` down to ~70–90 — replicating the
    biased global_thr seen on the production MoE scans.
    """
    img = _blank_sheet()
    for col in range(10):
        _fill_cand_bubble(img, col, candidate_digit, CAND_FILLED)
    return img


# ---------------------------------------------------------------------------
# Ghost-protection tests
# ---------------------------------------------------------------------------

def test_truly_blank_question_returns_NR_uniform(tmp_path: Path) -> None:
    """A strip where all four bubbles are at the same intensity → NR.

    With a biased global_thr (~86) and all MCQ values at BLANK_UNIFORM=215,
    adj_gap = 0 < 2 → the guard prevents the adjacent-midpoint fallback →
    global_thr is used → all bubbles above global_thr → NR.
    """
    img = _sheet_with_cand_bias()
    # Leave all MCQ questions blank (same intensity for all 4 options).
    for q in range(MCQ_N_QUESTIONS):
        for opt in range(4):
            _fill_mcq_bubble(img, q, opt, BLANK_UNIFORM)

    row = _run_engine(tmp_path, img)

    for q in ("q1", "q2", "q3", "q4"):
        assert row[q] == "NR", (
            f"Ghost answer detected for truly-blank question {q}: "
            f"got {row[q]!r}, expected 'NR'"
        )


def test_truly_blank_question_returns_NR_noisy(tmp_path: Path) -> None:
    """A strip with ≤ 1-unit scan noise across four bubbles → NR.

    adj_gap = BLANK_NOISY[1] - BLANK_NOISY[0] = 1 < 2 → guard prevents
    fallback → NR despite biased global_thr.
    """
    img = _sheet_with_cand_bias()
    for q in range(MCQ_N_QUESTIONS):
        # Assign the four options with 1-unit noise — simulating realistic
        # scan variation on a blank paper.
        for opt, intensity in enumerate(BLANK_NOISY):
            _fill_mcq_bubble(img, q, opt, intensity)

    row = _run_engine(tmp_path, img)

    for q in ("q1", "q2", "q3", "q4"):
        assert row[q] == "NR", (
            f"Ghost answer from scan-noise strip for {q}: "
            f"got {row[q]!r}, expected 'NR'"
        )


# ---------------------------------------------------------------------------
# Light-fill detection tests (regression for the original NR miss)
# ---------------------------------------------------------------------------

def test_lightly_filled_question_detected(tmp_path: Path) -> None:
    """A lightly-filled bubble with adj_gap = LIGHT_EMPTY - LIGHT_FILLED = 10
    must be detected despite biased global_thr, and with all other bubbles
    at LIGHT_EMPTY (well above global_thr).

    q1 → A (option 0), q2 → C (option 2).  q3 and q4 are left blank.
    """
    img = _sheet_with_cand_bias()

    # q1: fill option A=0 (darkest); all others at LIGHT_EMPTY.
    for opt in range(4):
        _fill_mcq_bubble(img, 0, opt, LIGHT_EMPTY)
    _fill_mcq_bubble(img, 0, 0, LIGHT_FILLED)   # A is darkest

    # q2: fill option C=2.
    for opt in range(4):
        _fill_mcq_bubble(img, 1, opt, LIGHT_EMPTY)
    _fill_mcq_bubble(img, 1, 2, LIGHT_FILLED)   # C is darkest

    # q3, q4: blank (uniform).
    for q in (2, 3):
        for opt in range(4):
            _fill_mcq_bubble(img, q, opt, BLANK_UNIFORM)

    row = _run_engine(tmp_path, img)

    assert row["q1"] == "A", (
        f"Lightly-filled q1=A not detected: got {row['q1']!r}"
    )
    assert row["q2"] == "C", (
        f"Lightly-filled q2=C not detected: got {row['q2']!r}"
    )
    assert row["q3"] == "NR", (
        f"Ghost answer for blank q3: got {row['q3']!r}"
    )
    assert row["q4"] == "NR", (
        f"Ghost answer for blank q4: got {row['q4']!r}"
    )


def test_minimum_adj_gap_boundary(tmp_path: Path) -> None:
    """adj_gap = 2 exactly (the guard minimum) must trigger detection.

    adj_gap < 2 (e.g. 1) must NOT trigger detection (NR).
    This pins the guard boundary to prevent silent regressions if someone
    adjusts the 2.0 threshold in core.py.
    """
    img = _sheet_with_cand_bias()

    # q1: adj_gap = 2 (just at the boundary) → should detect the fill.
    JUST_FILLED = 153    # darkest bubble
    JUST_EMPTY = 155     # adj_gap = 155 - 153 = 2
    for opt in range(4):
        _fill_mcq_bubble(img, 0, opt, JUST_EMPTY)
    _fill_mcq_bubble(img, 0, 0, JUST_FILLED)   # option A

    # q2: adj_gap = 1 (below boundary) → should NOT detect → NR.
    SUB_FILLED = 154     # darkest bubble
    SUB_EMPTY = 155      # adj_gap = 155 - 154 = 1
    for opt in range(4):
        _fill_mcq_bubble(img, 1, opt, SUB_EMPTY)
    _fill_mcq_bubble(img, 1, 0, SUB_FILLED)

    # q3, q4: uniform blank.
    for q in (2, 3):
        for opt in range(4):
            _fill_mcq_bubble(img, q, opt, BLANK_UNIFORM)

    row = _run_engine(tmp_path, img)

    # adj_gap=2 should be detected.
    assert row["q1"] == "A", (
        f"adj_gap=2 (boundary) should detect fill but got {row['q1']!r}"
    )
    # adj_gap=1 should NOT be detected.
    assert row["q2"] == "NR", (
        f"adj_gap=1 (below boundary) should be NR but got {row['q2']!r}"
    )
    assert row["q3"] == "NR"
    assert row["q4"] == "NR"
