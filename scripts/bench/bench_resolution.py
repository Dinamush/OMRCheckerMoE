"""Resolution sweep for candidate-number accuracy + processing time.

For each scale factor we:
  1. Scale the template's pageDimensions, bubbleDimensions, bubblesGap,
     labelsGap, origin and referenceMarkerCenters by that factor.
  2. Set the config processing_width/height to the same scaled dimensions.
  3. Run the full OMR engine on:
       - the real failing skewed Xerox scan (ground truth 9010292074)
       - rotated variants of it (rot+2, rot-2)
       - clean prefilled samples (medium_pencil, heavy_pencil, pen,
         careful_student) with ground truth 0123456789
  4. Record candidate number and wall-clock time.

A PASS is: exact match. A SAFE-FAIL is: any MR(...) / quarantine (surfaced
for manual review). A SILENT-WRONG is a clean-looking but wrong digit (the
only outcome we must never ship).

Run:
    python scripts/bench/bench_resolution.py
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.getLogger().setLevel(logging.ERROR)
for n in ("src", "src.entry", "src.core", "src.template", "src.processors",
          "src.processors.manager", "src.utils.file"):
    logging.getLogger(n).setLevel(logging.ERROR)

from main import entry_point_for_args

TEMPLATE_DIR = REPO_ROOT / "MoE-April-2026-Landscape-NNQ25-0"
BATCH = REPO_ROOT / "webui" / "storage" / "batches" / "356592fdd397"

# (image_path, expected_candidate)
CASES = [
    (BATCH / "inputs" / "Xerox_Scan_05232026114627_page_0001.jpg", "9010292074", "xerox_skewed"),
    (BATCH / "sample_medium_pencil.png", "0123456789", "medium_pencil"),
    (BATCH / "sample_heavy_pencil.png", "0123456789", "heavy_pencil"),
    (BATCH / "sample_pen.png", "0123456789", "pen"),
    (BATCH / "sample_careful_student.png", "0123456789", "careful_student"),
]

SCALES = [1.0, 1.5, 2.0, 3.0]


def scale_template(tpl: dict, k: float) -> dict:
    """Scale every pixel-space quantity in the template by k."""
    out = json.loads(json.dumps(tpl))
    out["pageDimensions"] = [round(v * k) for v in tpl["pageDimensions"]]
    out["bubbleDimensions"] = [max(2, round(v * k)) for v in tpl["bubbleDimensions"]]
    for name, block in out["fieldBlocks"].items():
        block["origin"] = [v * k for v in block["origin"]]
        block["bubblesGap"] = block["bubblesGap"] * k
        block["labelsGap"] = block["labelsGap"] * k
    for pp in out.get("preProcessors", []):
        ops = pp.get("options", {})
        if "referenceMarkerCenters" in ops:
            ops["referenceMarkerCenters"] = [
                [c[0] * k, c[1] * k] for c in ops["referenceMarkerCenters"]
            ]
        # Always set the half-size explicitly: the default (10.0) is a 1x
        # constant, so at any other scale the homography sanity check rejects
        # the warp because the marker-to-page ratio no longer matches.
        ops["referenceMarkerHalfSize"] = ops.get("referenceMarkerHalfSize", 10.0) * k
    return out


def scale_config(cfg: dict, scaled_dims: list[int]) -> dict:
    out = json.loads(json.dumps(cfg))
    out.setdefault("dimensions", {})
    out["dimensions"]["processing_width"] = scaled_dims[0]
    out["dimensions"]["processing_height"] = scaled_dims[1]
    # The bench already scales the template + config externally; disable the
    # engine's internal oversample so we measure the pure scale effect rather
    # than double-applying it on top of the bench's scaling.
    out.setdefault("threshold_params", {})
    out["threshold_params"]["OVERSAMPLE_SCALE"] = 1.0
    return out


def rot(im, a):
    h, w = im.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
    return cv2.warpAffine(im, m, (w, h), borderValue=(255, 255, 255))


def run_one(img_path: Path, work: Path) -> tuple[str, float]:
    """Run the engine on the dir containing img_path; return (candidate, secs)."""
    out = work / "out"
    if out.exists():
        shutil.rmtree(out)
    t0 = time.perf_counter()
    entry_point_for_args({
        "input_paths": [str(work)],
        "output_dir": str(out),
        "debug": False, "autoAlign": False, "setLayout": False, "silent": True,
    })
    elapsed = time.perf_counter() - t0
    candidate = "<quarantined>"
    for csv in (out / "Results").glob("*.csv"):
        for line in csv.read_text().splitlines()[1:]:
            cols = [c.strip('"') for c in line.split(",")]
            if Path(cols[0]).name == img_path.name:
                candidate = cols[4]
    return candidate, elapsed


def classify(read: str, expected: str) -> str:
    if read == expected:
        return "EXACT"
    if read.startswith("<") or "MR(" in read or "ERR" in read.upper():
        return "SAFE"
    return "SILENT-WRONG"


def main() -> int:
    base_tpl = json.loads((TEMPLATE_DIR / "template.json").read_text())
    base_cfg = json.loads((TEMPLATE_DIR / "config.json").read_text())

    # Add rotation variants of the real failing scan so we test the bug
    # itself plus near-skew neighbours at every resolution.
    variants = []
    base_xerox = cv2.imread(str(BATCH / "inputs" / "Xerox_Scan_05232026114627_page_0001.jpg"))
    variants.append(("xerox_skewed", base_xerox, "9010292074"))
    variants.append(("xerox_rot+2", rot(base_xerox, 2), "9010292074"))
    variants.append(("xerox_rot-2", rot(base_xerox, -2), "9010292074"))
    for img_path, expected, label in CASES[1:]:
        variants.append((label, cv2.imread(str(img_path)), expected))

    print(f"Sweeping {len(SCALES)} scales x {len(variants)} cases = "
          f"{len(SCALES) * len(variants)} runs\n")

    rows = []
    for k in SCALES:
        scaled_dims = [round(v * k) for v in base_tpl["pageDimensions"]]
        scaled_tpl = scale_template(base_tpl, k)
        scaled_cfg = scale_config(base_cfg, scaled_dims)

        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            (work / "template.json").write_text(json.dumps(scaled_tpl))
            (work / "config.json").write_text(json.dumps(scaled_cfg))

            scale_total_time = 0.0
            scale_exact = scale_safe = scale_wrong = 0
            for label, img, expected in variants:
                for old in list(work.glob("*.png")) + list(work.glob("*.jpg")):
                    old.unlink()
                vpath = work / f"{label}.png"
                cv2.imwrite(str(vpath), img)
                read, elapsed = run_one(vpath, work)
                verdict = classify(read, expected)
                scale_total_time += elapsed
                if verdict == "EXACT":
                    scale_exact += 1
                elif verdict == "SAFE":
                    scale_safe += 1
                else:
                    scale_wrong += 1
                rows.append((k, scaled_dims, label, expected, read, verdict, elapsed))
                print(f"  scale={k:>4.1f}x ({scaled_dims[0]}x{scaled_dims[1]})  "
                      f"{label:<18s} expect={expected}  read={read:<22s} "
                      f"{verdict:<12s} t={elapsed:.2f}s")
            n = len(variants)
            print(f"  -> scale={k:.1f}x  exact={scale_exact}/{n}  safe={scale_safe}/{n}  "
                  f"silent-wrong={scale_wrong}/{n}  total_time={scale_total_time:.2f}s  "
                  f"avg/sheet={scale_total_time/n:.2f}s\n")

    # Summary table
    print("\n=== SUMMARY ===")
    print(f"{'scale':>6} {'dims':>13} {'exact':>6} {'safe':>5} {'wrong':>6} {'avg_s':>7}")
    for k in SCALES:
        runs = [r for r in rows if r[0] == k]
        ex = sum(1 for r in runs if r[5] == "EXACT")
        sa = sum(1 for r in runs if r[5] == "SAFE")
        wr = sum(1 for r in runs if r[5] == "SILENT-WRONG")
        avg = sum(r[6] for r in runs) / max(1, len(runs))
        dims = runs[0][1]
        print(f"{k:>5.1f}x {dims[0]:>4}x{dims[1]:<4} {ex:>6} {sa:>5} {wr:>6} {avg:>6.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
