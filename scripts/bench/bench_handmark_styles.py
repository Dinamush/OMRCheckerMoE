"""Hand-mark style benchmark for OMR engine safety margin.

Simulates eight realistic student-marking styles (solid, check, X, partial
top, off-center, hesitation, eraser ghost, ring-only) at four darkness
levels and runs each sheet through the **production canonical
``MoE-April-2026-Landscape-NNQ25-0`` template** (ArUco-cropped + current
detection logic, no extra preprocessing).

The verdict reports per-style answer/candidate accuracy + how often the
engine wisely refuses to guess (``MR(..)`` / NR) versus how often it
silently misreads. This shows the real safety margin against the
distribution of marks a student is likely to actually make.

Run:
    python scripts/bench/bench_handmark_styles.py
        [--styles solid_centered check_mark ...]
        [--levels 50 100 150 200]
        [--sheets-per-cell 25]
        [--out benchmarks/results/bench_handmark_styles.json]
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from freezegun import freeze_time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.getLogger().setLevel(logging.ERROR)
for noisy in (
    "src", "src.entry", "src.core", "src.utils.file", "src.template",
    "src.processors", "src.processors.manager",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from main import entry_point_for_args

CANONICAL_DIR = REPO_ROOT / "MoE-April-2026-Landscape-NNQ25-0"

PAGE_W, PAGE_H = 666, 515
BUBBLE = 10
CAND_ORIGIN = (430, 103)
CAND_LABELS_GAP = 21.5
CAND_BUBBLES_GAP = 10.0
N_CAND_COLS = 10
QBLOCKS = [
    {"origin": (52.7, 259.3), "labels": [f"q{i}" for i in range(1, 6)]},
    {"origin": (180.9, 259.3), "labels": [f"q{i}" for i in range(6, 11)]},
    {"origin": (309.8, 259.3), "labels": [f"q{i}" for i in range(11, 16)]},
    {"origin": (435.9, 259.3), "labels": [f"q{i}" for i in range(16, 21)]},
    {"origin": (566.3, 259.3), "labels": [f"q{i}" for i in range(21, 26)]},
]
QBUBBLES_GAP = 20.0
QLABELS_GAP = 41.9
ANSWER_VALUES = ["A", "B", "C", "D"]

# Render at 3x template resolution so ArUco markers are crisp and
# the engine's image rescaling does not destroy thin pencil strokes.
RENDER_SCALE = 3
RENDER_W, RENDER_H = PAGE_W * RENDER_SCALE, PAGE_H * RENDER_SCALE
SIDE = BUBBLE * RENDER_SCALE  # bubble side length in render space (30 px)


# ── Hand-mark style stamps ────────────────────────────────────────────────


def _circle_mask(side: int = SIDE, shrink: float = 1.0) -> np.ndarray:
    yy, xx = np.mgrid[0:side, 0:side]
    c = (side - 1) / 2.0
    r = ((side - 2) / 2.0) * shrink
    return (yy - c) ** 2 + (xx - c) ** 2 <= r * r


def _build_style_mask(style: str, rng: np.random.Generator) -> np.ndarray:
    """Return a boolean stamp mask of shape (SIDE, SIDE) for a marking style.

    Each style defines the SHAPE of the dark pixels; the caller multiplies
    by ``fill_intensity`` to produce the actual painted region.
    """
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    if style == "solid_centered":
        circle = _circle_mask()
        fill_pick = rng.random((SIDE, SIDE)) < 0.78
        mask = circle & fill_pick
    elif style == "check_mark":
        # A small ✓ — diagonal from (lower-left) up-right to peak, then short hook down-left.
        canvas = np.zeros((SIDE, SIDE), dtype=np.uint8)
        cv2.line(canvas, (int(SIDE * 0.25), int(SIDE * 0.55)),
                 (int(SIDE * 0.45), int(SIDE * 0.75)), 255, thickness=2)
        cv2.line(canvas, (int(SIDE * 0.45), int(SIDE * 0.75)),
                 (int(SIDE * 0.78), int(SIDE * 0.30)), 255, thickness=2)
        # Roughen the stroke a little (real ink isn't perfectly even).
        canvas = cv2.dilate(canvas, np.ones((2, 2), np.uint8), iterations=1)
        mask = canvas > 0
    elif style == "x_mark":
        canvas = np.zeros((SIDE, SIDE), dtype=np.uint8)
        cv2.line(canvas, (int(SIDE * 0.22), int(SIDE * 0.22)),
                 (int(SIDE * 0.78), int(SIDE * 0.78)), 255, thickness=2)
        cv2.line(canvas, (int(SIDE * 0.78), int(SIDE * 0.22)),
                 (int(SIDE * 0.22), int(SIDE * 0.78)), 255, thickness=2)
        canvas = cv2.dilate(canvas, np.ones((2, 2), np.uint8), iterations=1)
        mask = canvas > 0
    elif style == "partial_top":
        circle = _circle_mask()
        half = np.zeros_like(circle)
        half[: SIDE // 2, :] = True
        fill_pick = rng.random((SIDE, SIDE)) < 0.78
        mask = circle & half & fill_pick
    elif style == "off_center":
        # Shifted solid fill — bubble centre offset by 2–3 OMR-px (6–9 render-px).
        dx = int(rng.integers(-9, 10))
        dy = int(rng.integers(-9, 10))
        yy, xx = np.mgrid[0:SIDE, 0:SIDE]
        c = (SIDE - 1) / 2.0
        r = (SIDE - 2) / 2.0
        shifted = (yy - c - dy) ** 2 + (xx - c - dx) ** 2 <= r * r
        fill_pick = rng.random((SIDE, SIDE)) < 0.78
        mask = shifted & fill_pick
    elif style == "hesitation":
        # 2–3 overlapping partial strokes (each ~40% of bubble area).
        circle = _circle_mask()
        strokes = np.zeros_like(circle)
        for _ in range(int(rng.integers(2, 4))):
            cx_off = rng.normal(0, SIDE * 0.07)
            cy_off = rng.normal(0, SIDE * 0.07)
            yy, xx = np.mgrid[0:SIDE, 0:SIDE]
            c = (SIDE - 1) / 2.0
            r_s = (SIDE - 2) / 2.0 * 0.55
            stroke = (yy - c - cy_off) ** 2 + (xx - c - cx_off) ** 2 <= r_s * r_s
            strokes = strokes | (stroke & (rng.random((SIDE, SIDE)) < 0.7))
        mask = circle & strokes
    elif style == "eraser_ghost":
        # Sparse residual: 22% of pixels in the circle still carry graphite.
        circle = _circle_mask()
        residual = rng.random((SIDE, SIDE)) < 0.22
        mask = circle & residual
    elif style == "ring_only":
        # Outline traced by the student (no interior fill).
        outer = _circle_mask(shrink=1.0)
        inner = _circle_mask(shrink=0.72)
        ring = outer & ~inner
        fill_pick = rng.random((SIDE, SIDE)) < 0.75
        mask = ring & fill_pick
    else:
        raise ValueError(f"unknown style: {style}")
    return mask


STYLES = [
    "solid_centered",
    "check_mark",
    "x_mark",
    "partial_top",
    "off_center",
    "hesitation",
    "eraser_ghost",
    "ring_only",
]


# ── Sheet rendering ───────────────────────────────────────────────────────


def _stamp_aruco(img: np.ndarray) -> np.ndarray:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_px = 60
    centers_omr = [(13.5, 13.2), (651.5, 13.2), (13.5, 499.0), (651.5, 499.0)]
    sx, sy = RENDER_W / PAGE_W, RENDER_H / PAGE_H
    for marker_id, (cx, cy) in enumerate(centers_omr):
        x = int(round(cx * sx)) - marker_px // 2
        y = int(round(cy * sy)) - marker_px // 2
        x = max(6, min(RENDER_W - marker_px - 6, x))
        y = max(6, min(RENDER_H - marker_px - 6, y))
        m = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
        img[y : y + marker_px, x : x + marker_px] = m
    return img


def _paint_bubble(
    img: np.ndarray,
    cx_omr: float,
    cy_omr: float,
    intensity: int,
    style: str,
    rng: np.random.Generator,
) -> None:
    sx, sy = RENDER_W / PAGE_W, RENDER_H / PAGE_H
    x_render = int(round(cx_omr * sx))
    y_render = int(round(cy_omr * sy))
    mask = _build_style_mask(style, rng)
    jitter = rng.normal(0.0, 12.0, (SIDE, SIDE))
    painted = np.clip(intensity + jitter, 0, 255).astype(np.uint8)
    region = img[y_render : y_render + SIDE, x_render : x_render + SIDE]
    region[mask] = np.minimum(region[mask], painted[mask])


def _add_paper_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    arr = img.astype(np.float32)
    grad_y = np.linspace(-6, 6, RENDER_H, dtype=np.float32)[:, None]
    grad_x = np.linspace(-3, 3, RENDER_W, dtype=np.float32)[None, :]
    arr += grad_y + grad_x
    arr += rng.normal(0.0, 5.0, arr.shape)
    specks = rng.random(arr.shape) < 0.0005
    arr[specks] = rng.integers(40, 120, size=int(specks.sum()))
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if ok:
        arr = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return arr


def _make_sheet(
    intensity: int, style: str, rng: np.random.Generator
) -> tuple[np.ndarray, str, dict[str, str]]:
    img = np.full((RENDER_H, RENDER_W), 252, dtype=np.uint8)
    candidate = "".join(str(d) for d in rng.integers(0, 10, size=N_CAND_COLS))
    for col, d in enumerate(candidate):
        _paint_bubble(
            img,
            CAND_ORIGIN[0] + col * CAND_LABELS_GAP,
            CAND_ORIGIN[1] + int(d) * CAND_BUBBLES_GAP,
            intensity, style, rng,
        )
    answers: dict[str, str] = {}
    for block in QBLOCKS:
        ox, oy = block["origin"]
        for row, label in enumerate(block["labels"]):
            idx = int(rng.integers(0, len(ANSWER_VALUES)))
            _paint_bubble(
                img,
                ox + idx * QBUBBLES_GAP,
                oy + row * QLABELS_GAP,
                intensity, style, rng,
            )
            answers[label] = ANSWER_VALUES[idx]
    img = _add_paper_noise(img, rng)
    img = _stamp_aruco(img)
    return img, candidate, answers


# ── Scorer ────────────────────────────────────────────────────────────────


def _classify(engine_value: str, truth: str) -> str:
    """Bucket a single field result vs. its truth.

    Returns one of: ``"correct"``, ``"mr"``, ``"nr"``, ``"wrong"``.
    """
    val = (engine_value or "").strip()
    if val == truth:
        return "correct"
    if "MR(" in val:
        return "mr"
    if val == "" or val == "NR":
        return "nr"
    return "wrong"


def _score_csv(csv_path: Path, ground_truth: dict[str, tuple[str, dict[str, str]]]) -> dict:
    s = {
        "sheets": 0,
        "cand_exact": 0,
        "cand_mr": 0,
        "cand_wrong": 0,
        "cand_partial": 0,
        "ans_correct": 0,
        "ans_total": 0,
        "ans_mr": 0,
        "ans_nr": 0,
        "ans_wrong": 0,
    }
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            fname = row["file_id"]
            if fname not in ground_truth:
                continue
            cand_truth, ans_truth = ground_truth[fname]
            s["sheets"] += 1
            cand = row.get("CandidateNumber", "") or ""
            if cand == cand_truth:
                s["cand_exact"] += 1
            elif "MR(" in cand:
                s["cand_mr"] += 1
            elif cand.isdigit() and len(cand) == len(cand_truth):
                # Right length, wrong digit(s) — partial.
                s["cand_partial"] += 1
            else:
                s["cand_wrong"] += 1
            for qname, truth in ans_truth.items():
                s["ans_total"] += 1
                bucket = _classify(row.get(qname, ""), truth)
                s["ans_" + bucket] += 1
    return s


# ── Runner ────────────────────────────────────────────────────────────────


def _run_cell(
    style: str,
    intensity: int,
    sheets_per_cell: int,
    seed: int,
    work_root: Path,
) -> dict:
    rng = np.random.default_rng(seed + intensity * 113 + abs(hash(style)) % 10007)
    inputs = work_root / f"{style}_int{intensity:03d}"
    if inputs.exists():
        shutil.rmtree(inputs, ignore_errors=True)
    inputs.mkdir(parents=True, exist_ok=True)

    # Copy the production template + config so the engine runs the real pipeline
    shutil.copy2(CANONICAL_DIR / "template.json", inputs / "template.json")
    shutil.copy2(CANONICAL_DIR / "config.json", inputs / "config.json")

    ground_truth: dict[str, tuple[str, dict[str, str]]] = {}
    for i in range(sheets_per_cell):
        img, cand, ans = _make_sheet(intensity, style, rng)
        fname = f"sheet_{i:03d}.jpg"
        cv2.imwrite(str(inputs / fname), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        ground_truth[fname] = (cand, ans)

    out_dir = work_root / f"out_{style}_int{intensity:03d}"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    t0 = time.perf_counter()
    with freeze_time("1970-01-01"):
        entry_point_for_args(
            {
                "input_paths": [str(inputs)],
                "output_dir": str(out_dir),
                "debug": False,
                "autoAlign": False,
                "setLayout": False,
                "silent": True,
            }
        )
    elapsed = time.perf_counter() - t0

    csv_files = sorted(out_dir.rglob("Results_*.csv"))
    err_csvs = sorted(out_dir.rglob("ErrorFiles.csv"))
    preprocess_failures = 0
    if err_csvs:
        with err_csvs[0].open("r", encoding="utf-8", newline="") as fh:
            preprocess_failures = sum(1 for _ in _csv.DictReader(fh))

    s: dict = {}
    if csv_files:
        s = _score_csv(csv_files[0], ground_truth)
    else:
        s = {k: 0 for k in (
            "sheets","cand_exact","cand_mr","cand_wrong","cand_partial",
            "ans_correct","ans_total","ans_mr","ans_nr","ans_wrong",
        )}
    s["preprocess_failures"] = preprocess_failures
    s["elapsed_s"] = round(elapsed, 2)
    return s


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "benchmarks" / "results" / "bench_handmark_styles.json"),
    )
    parser.add_argument("--styles", nargs="*", default=STYLES)
    parser.add_argument("--levels", nargs="*", type=int, default=[50, 100, 150, 200])
    parser.add_argument("--sheets-per-cell", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260528)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="omr_handmark_") as work_root_str:
        work_root = Path(work_root_str)
        for style in args.styles:
            results[style] = {}
            for intensity in args.levels:
                print(
                    f"  style={style:<16s} int={intensity:>3d}  ...",
                    end="", flush=True,
                )
                s = _run_cell(
                    style, intensity, args.sheets_per_cell, args.seed, work_root,
                )
                results[style][intensity] = s
                ans_pct = (s["ans_correct"] / s["ans_total"] * 100) if s["ans_total"] else 0
                cand_pct = (s["cand_exact"] / s["sheets"] * 100) if s["sheets"] else 0
                fail = s["preprocess_failures"]
                print(
                    f"  cand_exact={cand_pct:5.1f}%  ans={ans_pct:5.1f}%  "
                    f"pp_fail={fail:>2d}  ({s['elapsed_s']:.1f}s)"
                )

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    print("\n=== VERDICT ===\n")
    print(_format_verdict(results, args.levels))
    return 0


def _format_verdict(results: dict[str, dict], levels: list[int]) -> str:
    lines: list[str] = []
    # Per-style answer-accuracy matrix
    lines.append("Answer accuracy (correct / total) per style × intensity:")
    head = f"{'style':<18s}" + "".join(f"   int{l:>3d}" for l in levels)
    lines.append(head)
    lines.append("-" * len(head))
    for style in results:
        cells = []
        for lvl in levels:
            r = results[style].get(lvl, {})
            if not r or not r.get("ans_total"):
                cells.append("    --   ")
                continue
            pct = r["ans_correct"] / r["ans_total"] * 100
            cells.append(f"  {pct:6.2f}%")
        lines.append(f"{style:<18s}" + "".join(cells))

    lines.append("")
    lines.append("Candidate-number exact match per style × intensity:")
    lines.append(head)
    lines.append("-" * len(head))
    for style in results:
        cells = []
        for lvl in levels:
            r = results[style].get(lvl, {})
            if not r or not r.get("sheets"):
                cells.append("    --   ")
                continue
            pct = r["cand_exact"] / r["sheets"] * 100
            cells.append(f"  {pct:6.2f}%")
        lines.append(f"{style:<18s}" + "".join(cells))

    lines.append("")
    lines.append("Engine 'safe-fail' rate (MR / NR) per style × intensity:")
    lines.append("(values = % of answer fields where engine flagged ambiguity")
    lines.append(" or returned blank, rather than guessing wrong)")
    lines.append(head)
    lines.append("-" * len(head))
    for style in results:
        cells = []
        for lvl in levels:
            r = results[style].get(lvl, {})
            if not r or not r.get("ans_total"):
                cells.append("    --   ")
                continue
            safe = (r["ans_mr"] + r["ans_nr"]) / r["ans_total"] * 100
            cells.append(f"  {safe:6.2f}%")
        lines.append(f"{style:<18s}" + "".join(cells))

    lines.append("")
    lines.append("Silent-wrong rate per style × intensity:")
    lines.append("(engine returned a plain wrong answer — worst case)")
    lines.append(head)
    lines.append("-" * len(head))
    for style in results:
        cells = []
        for lvl in levels:
            r = results[style].get(lvl, {})
            if not r or not r.get("ans_total"):
                cells.append("    --   ")
                continue
            wrong = r["ans_wrong"] / r["ans_total"] * 100
            cells.append(f"  {wrong:6.2f}%")
        lines.append(f"{style:<18s}" + "".join(cells))

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
