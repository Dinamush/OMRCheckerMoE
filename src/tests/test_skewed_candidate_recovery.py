"""End-to-end regression for candidate-number recovery on a skewed scan.

Real-world failure (production batch ``356592fdd397``): a rotated Xerox scan
whose top-left ArUco marker was unreadable. With only 3 markers the warp is a
degraded homography that leaves the candidate-number grid uniformly shifted
~half a bubble vertically. Because the QTYPE_INT digit bubbles abut with zero
gap, each filled digit bled dark pixels into its neighbour's sampling
rectangle, so candidate ``9010292074`` was misread as ``90102920MR(78)4``
(column 9 flagged, column 10's bleed dropped).

The fix is a bounded, consistency-gated vertical re-centering of touching INT
columns: when every marked column agrees on the same systematic offset, the
grid is shifted uniformly to re-centre the marks (recovering the number);
when the per-column offsets disagree (an unrecoverable non-uniform skew), the
grid is left nominal so the ambiguous columns stay MR(...) and the sheet is
safely quarantined rather than scored wrong.

This test runs the *full* engine (ArUco crop + measurement) on the actual
failing scan committed under ``test_samples/skewed_candidate_3marker`` and
asserts the candidate number is now read exactly.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from freezegun import freeze_time

from main import entry_point_for_args

FIXTURE_DIR = Path(__file__).parent / "test_samples" / "skewed_candidate_3marker"
EXPECTED_CANDIDATE = "9010292074"


def test_skewed_3marker_scan_recovers_candidate_number(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for name in ("template.json", "config.json", "skewed_3marker_scan.jpg"):
        shutil.copy(FIXTURE_DIR / name, inputs / name)

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

    candidate = rows[0]["CandidateNumber"]
    assert candidate == EXPECTED_CANDIDATE, (
        "skewed 3-marker scan candidate number was not recovered: "
        f"got {candidate!r}, expected {EXPECTED_CANDIDATE!r}"
    )
    assert "MR(" not in candidate
