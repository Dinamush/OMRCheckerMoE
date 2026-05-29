"""Skew/rotation stress test for candidate-number reading.

Takes a known-ground-truth prefilled sheet and the real failing Xerox scan,
applies a grid of rotations + perspective keystones (simulating sheets where
not all 4 ArUco markers stay crisp), runs each through the full OMR engine,
and reports the candidate-number read vs the expected value.

A PASS is: exact match, OR an MR(...) quarantine (safe-fail, surfaced for
manual review). A FAIL is a silently WRONG candidate number.

Run:
    python scripts/bench/bench_skew_candidate.py
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.getLogger().setLevel(logging.ERROR)
for noisy in ("src", "src.entry", "src.core", "src.template", "src.processors",
              "src.processors.manager", "src.utils.file"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from main import entry_point_for_args

TEMPLATE_DIR = REPO_ROOT / "MoE-April-2026-Landscape-NNQ25-0"
BATCH_DIR = REPO_ROOT / "webui" / "storage" / "batches" / "356592fdd397"

# (image_path, expected_candidate)
CASES = [
    (BATCH_DIR / "sample_medium_pencil.png", "0123456789"),
    (BATCH_DIR / "sample_heavy_pencil.png", "0123456789"),
    (BATCH_DIR / "sample_pen.png", "0123456789"),
    (BATCH_DIR / "sample_careful_student.png", "0123456789"),
    (BATCH_DIR / "inputs" / "Xerox_Scan_05232026114627_page_0001.jpg", "9010292074"),
]


def rotate(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))


def keystone(img, k):
    """Apply a vertical perspective keystone of magnitude k (fraction)."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dx = k * w
    dst = np.float32([[dx, 0], [w - dx, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), borderValue=(255, 255, 255))


def run_one(img_path, work):
    """Run engine on one image dir, return the candidate number read."""
    out = work / "out"
    if out.exists():
        shutil.rmtree(out)
    entry_point_for_args({
        "input_paths": [str(work)],
        "output_dir": str(out),
        "debug": False, "autoAlign": False, "setLayout": False, "silent": True,
    })
    csvs = list((out / "Results").glob("*.csv"))
    results = {}
    for csv in csvs:
        for line in csv.read_text().splitlines()[1:]:
            cols = [c.strip('"') for c in line.split(",")]
            results[Path(cols[0]).name] = cols[4]
    # multimarked sheets land in Manual/MultiMarkedFiles with no Results row
    return results


def main() -> int:
    transforms = [
        ("original", lambda im: im),
        ("rot+2", lambda im: rotate(im, 2)),
        ("rot-2", lambda im: rotate(im, -2)),
        ("rot+4", lambda im: rotate(im, 4)),
        ("keystone+0.04", lambda im: keystone(im, 0.04)),
        ("rot+3,keystone0.03", lambda im: keystone(rotate(im, 3), 0.03)),
    ]

    total = ok = safe = fail = 0
    fails = []
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        shutil.copy(TEMPLATE_DIR / "template.json", work)
        shutil.copy(TEMPLATE_DIR / "config.json", work)
        for img_path, expected in CASES:
            base = cv2.imread(str(img_path))
            for tname, tfn in transforms:
                # clear stray images
                for old in work.glob("*.png"):
                    old.unlink()
                for old in work.glob("*.jpg"):
                    old.unlink()
                variant = tfn(base)
                vpath = work / f"case_{img_path.stem}_{tname}.png"
                cv2.imwrite(str(vpath), variant)
                results = run_one(vpath, work)
                read = results.get(vpath.name, "<MR/ERROR-quarantined>")
                total += 1
                if read == expected:
                    ok += 1
                    verdict = "OK"
                elif read.startswith("<") or "MR(" in read or "ERR" in read:
                    safe += 1
                    verdict = "SAFE (quarantined)"
                else:
                    fail += 1
                    verdict = "*** SILENT WRONG ***"
                    fails.append((img_path.stem, tname, expected, read))
                print(f"  {img_path.stem:<22s} {tname:<20s} expect={expected} "
                      f"read={read:<18s} {verdict}")
            print()

    print(f"=== TOTAL {total}: exact={ok}  safe-fail={safe}  SILENT-WRONG={fail} ===")
    if fails:
        print("SILENT WRONG cases (must be zero):")
        for s in fails:
            print("   ", s)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
