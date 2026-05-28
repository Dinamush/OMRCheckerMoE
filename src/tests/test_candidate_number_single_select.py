"""End-to-end regression for candidate-number single-select behaviour.

A candidate-number (``QTYPE_INT``) column is single-select: exactly one
digit per column. The engine previously routed INT columns through a
branch that *concatenated* every bubble above the per-strip threshold,
so a faint second mark (ghost / erasure / smudge) clearing the threshold
silently turned one column into two digits (e.g. ``"9" + "8" -> "98"``),
corrupting the whole ``CandidateNumber`` and, critically, without
flagging the sheet for review.

These tests run the *real* OMR engine (``entry_point_for_args``) on a
synthetic, perfectly-aligned sheet built directly from the canonical
``MoE-April-2026-Landscape-NNQ25-0`` candidate-block geometry. No ArUco
markers are needed because the template declares no pre-processors and
the image is already in template space.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from freezegun import freeze_time

from main import entry_point_for_args

# Geometry copied from MoE-April-2026-Landscape-NNQ25-0/template.json.
PAGE_W, PAGE_H = 666, 515
ORIGIN_X, ORIGIN_Y = 430, 103
LABELS_GAP = 21.5  # X step between digit columns (cand1..10)
BUBBLES_GAP = 10.0  # Y step between digits 0..9 within a column
BUBBLE = 10

DARK = 20  # a solid, deliberate pencil/pen mark
FAINT = 110  # a faint ghost/smudge that still clears the strip threshold
NEAR_TIE = 30  # a second mark almost as dark as DARK (genuine ambiguity)

TEMPLATE = {
    "pageDimensions": [PAGE_W, PAGE_H],
    "bubbleDimensions": [BUBBLE, BUBBLE],
    "customLabels": {"CandidateNumber": ["cand1..10"]},
    "outputColumns": ["CandidateNumber"],
    "fieldBlocks": {
        "CandidateNumber": {
            "origin": [ORIGIN_X, ORIGIN_Y],
            "bubblesGap": BUBBLES_GAP,
            "labelsGap": LABELS_GAP,
            "fieldLabels": ["cand1..10"],
            "fieldType": "QTYPE_INT",
        }
    },
    "preProcessors": [],
}

CONFIG = {
    "dimensions": {
        "display_height": PAGE_H,
        "display_width": PAGE_W,
        "processing_height": PAGE_H,
        "processing_width": PAGE_W,
    },
    "outputs": {"show_image_level": 0, "save_image_level": 0},
}


def _fill_bubble(img: np.ndarray, column: int, digit: int, intensity: int) -> None:
    """Paint a digit bubble at its template coordinates."""
    x0 = round(ORIGIN_X + column * LABELS_GAP)
    y0 = round(ORIGIN_Y + digit * BUBBLES_GAP)
    img[y0 : y0 + BUBBLE, x0 : x0 + BUBBLE] = intensity


def _blank_sheet() -> np.ndarray:
    return np.full((PAGE_H, PAGE_W), 255, dtype=np.uint8)


def _run_engine(tmp_path: Path, img: np.ndarray) -> dict[str, str]:
    """Process ``img`` and return the single Results row as a dict."""
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


def test_faint_second_mark_is_dropped_not_appended(tmp_path: Path) -> None:
    """A faint ghost mark in a column must not add a digit to the number."""
    candidate = "9012345678"
    img = _blank_sheet()
    for column, digit_char in enumerate(candidate):
        _fill_bubble(img, column, int(digit_char), DARK)
    # Inject a faint second mark in column 0 (real digit is 9): a ghost at 0.
    _fill_bubble(img, 0, 0, FAINT)

    row = _run_engine(tmp_path, img)

    assert row["CandidateNumber"] == candidate, (
        "faint second mark leaked into the candidate number: "
        f"got {row['CandidateNumber']!r}, expected {candidate!r}"
    )
    assert len(row["CandidateNumber"]) == 10


def test_clean_candidate_number_reads_exactly(tmp_path: Path) -> None:
    """A cleanly filled candidate number reads back digit-for-digit."""
    candidate = "1234509876"
    img = _blank_sheet()
    for column, digit_char in enumerate(candidate):
        _fill_bubble(img, column, int(digit_char), DARK)

    row = _run_engine(tmp_path, img)

    assert row["CandidateNumber"] == candidate
    assert len(row["CandidateNumber"]) == 10


def test_genuine_equal_double_mark_is_flagged(tmp_path: Path) -> None:
    """Two equally-dark marks in one column flag review instead of guessing."""
    candidate = "9012345678"
    img = _blank_sheet()
    for column, digit_char in enumerate(candidate):
        _fill_bubble(img, column, int(digit_char), DARK)
    # Column 0 real digit is 9; add a near-equal second mark at digit 0.
    _fill_bubble(img, 0, 0, NEAR_TIE)

    row = _run_engine(tmp_path, img)

    candidate_value = row["CandidateNumber"]
    assert "MR(" in candidate_value, (
        "a genuine double-marked column must surface an MR(...) marker, "
        f"got {candidate_value!r}"
    )
    # The remaining nine columns are still resolved as their single digit,
    # so the rest of the number is intact around the flagged column.
    assert candidate_value.endswith("012345678")
