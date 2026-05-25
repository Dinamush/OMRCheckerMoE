"""Run the OMR engine against the prefilled demo sheet and report results."""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

from portrait_25q.variants.sweep.generate_prefilled_demo import (
    ANSWER_KEY,
    CANDIDATE_NUMBER,
    OUTPUT_PATH,
    SPEC,
)
from portrait_25q.variants.sweep.generate_sheet import emit_config, emit_template


REPO_ROOT = Path(__file__).resolve().parents[3]
WORK_DIR = Path(__file__).resolve().parent / "reference_test"


def main() -> int:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    emit_template(SPEC, WORK_DIR)
    emit_config(WORK_DIR)
    shutil.copy(OUTPUT_PATH, WORK_DIR / "demo.png")

    out_dir = WORK_DIR / "out"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "main.py",
         "--inputDir", str(WORK_DIR),
         "--outputDir", str(out_dir)],
        cwd=REPO_ROOT, env=env, capture_output=True,
        encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        print("ENGINE FAILED")
        print(proc.stdout)
        print(proc.stderr)
        return 1

    results_dir = out_dir / "Results"
    csvs = sorted(results_dir.glob("Results_*.csv"))
    if not csvs:
        print("No Results CSV produced.")
        return 1
    with csvs[-1].open(encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    got_cand = row.get("CandidateNumber", "")
    print(f"Candidate Number  expected={CANDIDATE_NUMBER}  got={got_cand}  "
          f"{'OK' if got_cand == CANDIDATE_NUMBER else 'MISMATCH'}")

    correct = 0
    mismatches: list[str] = []
    for qnum, expected in ANSWER_KEY:
        got = row.get(f"q{qnum}", "")
        if got == expected:
            correct += 1
        else:
            mismatches.append(f"q{qnum}: expected {expected}, got {got!r}")
    print(f"Answers           {correct}/25 correct"
          f"{'  OK' if correct == 25 else ''}")
    for m in mismatches:
        print(f"  {m}")
    return 0 if (got_cand == CANDIDATE_NUMBER and correct == 25) else 1


if __name__ == "__main__":
    raise SystemExit(main())
