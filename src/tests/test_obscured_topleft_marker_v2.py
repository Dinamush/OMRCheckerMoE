"""Regression for a landscape Xerox scan whose top-left ArUco is smudged.

Companion to ``test_skewed_candidate_recovery.py``. The other fixture has
all four corner markers heavily obscured by feeder banding; this one only
loses the top-left to a single dark smudge bleeding outward from the
marker. Under the current production detector settings
(``adaptiveThreshWinSizeMax=15`` on the 2x-oversampled processing canvas)
the top-left fails to decode and the resulting 3-marker fit is biased
enough that the QTYPE_INT candidate-number column samples halfway between
rows — the scan was read as ``"78"`` instead of ``"0123456789"``.

The fixture is a synthetic diagonal-fill calibration pattern (column 1=0,
column 2=1, ..., column 10=9) so a correct read maps to the canonical
candidate number ``"0123456789"``. MCQ blocks are intentionally blank so
this test isolates the candidate-number grid (which is the location most
sensitive to a degraded warp because its bubbles abut with zero guard
band).
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from freezegun import freeze_time

from main import entry_point_for_args

FIXTURE_DIR = Path(__file__).parent / "test_samples" / "obscured_topleft_marker_v2"
EXPECTED_CANDIDATE = "0123456789"


def test_obscured_topleft_marker_recovers_candidate_number(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for name in ("template.json", "config.json", "obscured_topleft_scan.jpg"):
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
        "obscured-topleft scan candidate number was not recovered: "
        f"got {candidate!r}, expected {EXPECTED_CANDIDATE!r}"
    )
    assert "MR(" not in candidate
