"""Rigorous preprocessing benchmark for faint marks.

Sweeps a grid of (fill darkness × preprocessing recipe) across deterministic
synthetic sheets built from the canonical
``MoE-April-2026-Landscape-NNQ25-0`` template geometry, runs the real
OMR engine on each combination, and scores accuracy.

The question: does an aggressive preprocessing recipe (CLAHE, gamma, unsharp,
levels stretch, bilateral, etc.) let the engine recover very-faint
student responses without sacrificing accuracy on normal/dark marks?

Run:
    python scripts/bench/bench_preprocess_darkness.py \
        --out benchmarks/results/bench_preprocess_darkness.json

Output: JSON report with per-(darkness, recipe) accuracy + a console
verdict picking the recipe with the largest faint-mark accuracy win
that does not regress on dark marks.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import entry_point_for_args

# Silence the engine's per-image INFO logs — they swamp the benchmark output.
logging.getLogger().setLevel(logging.ERROR)
for noisy in (
    "src", "src.entry", "src.core", "src.utils.file", "src.template",
    "src.processors", "src.processors.manager",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)

# ── Geometry copied from MoE-April-2026-Landscape-NNQ25-0/template.json ───
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

DARKNESS_LEVELS = [20, 50, 80, 110, 140, 170, 200, 220]
SHEETS_PER_LEVEL = 25
SEED = 20260528

TEMPLATE = {
    "pageDimensions": [PAGE_W, PAGE_H],
    "bubbleDimensions": [BUBBLE, BUBBLE],
    "customLabels": {"CandidateNumber": ["cand1..10"]},
    "outputColumns": ["CandidateNumber", "q1..25"],
    "fieldBlocks": {
        "CandidateNumber": {
            "origin": list(CAND_ORIGIN),
            "bubblesGap": CAND_BUBBLES_GAP,
            "labelsGap": CAND_LABELS_GAP,
            "fieldLabels": ["cand1..10"],
            "fieldType": "QTYPE_INT",
        },
        "q01block": {
            "origin": list(QBLOCKS[0]["origin"]),
            "bubblesGap": QBUBBLES_GAP,
            "labelsGap": QLABELS_GAP,
            "fieldLabels": ["q1..5"],
            "emptyValue": "NR",
            "fieldType": "QTYPE_MCQ4",
        },
        "q06block": {
            "origin": list(QBLOCKS[1]["origin"]),
            "bubblesGap": QBUBBLES_GAP,
            "labelsGap": QLABELS_GAP,
            "fieldLabels": ["q6..10"],
            "emptyValue": "NR",
            "fieldType": "QTYPE_MCQ4",
        },
        "q11block": {
            "origin": list(QBLOCKS[2]["origin"]),
            "bubblesGap": QBUBBLES_GAP,
            "labelsGap": QLABELS_GAP,
            "fieldLabels": ["q11..15"],
            "emptyValue": "NR",
            "fieldType": "QTYPE_MCQ4",
        },
        "q16block": {
            "origin": list(QBLOCKS[3]["origin"]),
            "bubblesGap": QBUBBLES_GAP,
            "labelsGap": QLABELS_GAP,
            "fieldLabels": ["q16..20"],
            "emptyValue": "NR",
            "fieldType": "QTYPE_MCQ4",
        },
        "q21block": {
            "origin": list(QBLOCKS[4]["origin"]),
            "bubblesGap": QBUBBLES_GAP,
            "labelsGap": QLABELS_GAP,
            "fieldLabels": ["q21..25"],
            "emptyValue": "NR",
            "fieldType": "QTYPE_MCQ4",
        },
    },
    "preProcessors": [],  # no preprocessors — we control darkness ourselves
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


# ── Sheet rendering ───────────────────────────────────────────────────────


def _fill_block(
    img: np.ndarray,
    cx: float,
    cy: float,
    intensity: int,
    rng: np.random.Generator,
    fill_ratio: float = 0.72,
) -> None:
    """Stamp a realistic pencil-like mark inside the bubble.

    Mimics a real student's mark: not a solid uniform block.
    - A circular mask covers most of the bubble area.
    - Only ``fill_ratio`` of the masked pixels actually darken (the
      student's pencil doesn't tile every pixel; the rest of the
      circle stays at the noisy paper background).
    - Each darkened pixel gets ``intensity`` plus jitter (graphite is
      uneven), then is clipped to [0, 255].

    Lighter ``intensity`` (closer to 255) -> fainter mark; the engine
    must still be able to discriminate it from paper.
    """
    x0, y0 = int(round(cx)), int(round(cy))
    h, w = img.shape
    y1, x1 = min(h, y0 + BUBBLE), min(w, x0 + BUBBLE)
    if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
        return
    region = img[y0:y1, x0:x1]
    rh, rw = region.shape
    yy, xx = np.mgrid[0:rh, 0:rw]
    cy_, cx_ = (rh - 1) / 2.0, (rw - 1) / 2.0
    r = (BUBBLE - 1) / 2.0
    inside_circle = (yy - cy_) ** 2 + (xx - cx_) ** 2 <= r * r
    fill_pick = rng.random(region.shape) < fill_ratio
    paint_mask = inside_circle & fill_pick
    jitter = rng.normal(0.0, 10.0, region.shape)
    painted = np.clip(intensity + jitter, 0, 255).astype(np.uint8)
    region[paint_mask] = np.minimum(region[paint_mask], painted[paint_mask])


def _apply_paper_realism(
    img: np.ndarray,
    rng: np.random.Generator,
    jpeg_quality: int = 78,
) -> np.ndarray:
    """Add scanner/printer noise and JPEG compression to the rendered sheet."""
    # 1. Mild brightness gradient (uneven scanner light)
    h, w = img.shape
    grad_y = np.linspace(-6, 6, h, dtype=np.float32)[:, None]
    grad_x = np.linspace(-3, 3, w, dtype=np.float32)[None, :]
    noisy = img.astype(np.float32) + grad_y + grad_x

    # 2. Gaussian sensor noise (sigma ~ 6 grey-levels mimics commodity scanners)
    noisy += rng.normal(0.0, 6.0, img.shape)

    # 3. Salt-pepper-ish speckle on paper at very low density (toner specks).
    speck = rng.random(img.shape) < 0.0008
    noisy[speck] = rng.integers(40, 120, size=int(speck.sum()))

    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    # 4. JPEG round-trip — adds ringing around mark edges.
    ok, buf = cv2.imencode(".jpg", noisy, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return noisy
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


@dataclass
class SheetTruth:
    candidate: str  # 10-char digit string
    answers: dict[str, str]  # q1..q25 -> A/B/C/D


def _generate_sheet(rng: np.random.Generator, intensity: int) -> tuple[np.ndarray, SheetTruth]:
    """Render one realistic sheet with random ground truth at the given fill intensity."""
    img = np.full((PAGE_H, PAGE_W), 252, dtype=np.uint8)  # off-white paper
    candidate = "".join(str(d) for d in rng.integers(0, 10, size=N_CAND_COLS))
    for col, digit_char in enumerate(candidate):
        cx = CAND_ORIGIN[0] + col * CAND_LABELS_GAP
        cy = CAND_ORIGIN[1] + int(digit_char) * CAND_BUBBLES_GAP
        _fill_block(img, cx, cy, intensity, rng)

    answers: dict[str, str] = {}
    for block in QBLOCKS:
        ox, oy = block["origin"]
        for row, label in enumerate(block["labels"]):
            choice_idx = int(rng.integers(0, len(ANSWER_VALUES)))
            choice = ANSWER_VALUES[choice_idx]
            cx = ox + choice_idx * QBUBBLES_GAP
            cy = oy + row * QLABELS_GAP
            _fill_block(img, cx, cy, intensity, rng)
            answers[label] = choice

    img = _apply_paper_realism(img, rng)
    return img, SheetTruth(candidate=candidate, answers=answers)


# ── Preprocessing recipes ─────────────────────────────────────────────────


def _baseline(img: np.ndarray) -> np.ndarray:
    return img


def _clahe(img: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8)).apply(img)


def _gamma(g: float):
    """Build a gamma LUT and return a recipe function applying it."""
    table = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in np.arange(256)], dtype=np.uint8)
    def _apply(img: np.ndarray) -> np.ndarray:
        return cv2.LUT(img, table)
    return _apply


def _levels(low: int, high: int, gamma: float):
    """Build a parameterised levels-stretch recipe."""
    inv = 1.0 / gamma
    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i <= low:
            v = 0.0
        elif i >= high:
            v = 255.0
        else:
            v = (((i - low) / (high - low)) ** inv) * 255
        table[i] = max(0, min(255, int(round(v))))
    def _apply(img: np.ndarray) -> np.ndarray:
        return cv2.LUT(img, table)
    return _apply


_levels_stretch = _levels(140, 255, 0.7)
_levels_aggr = _levels(120, 250, 0.6)
_levels_balanced = _levels(160, 255, 0.65)
_levels_strong = _levels(150, 248, 0.55)


def _unsharp(img: np.ndarray) -> np.ndarray:
    """Unsharp mask: amplify edges of faint marks without blowing dark ones."""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    return cv2.addWeighted(img, 1.0 + 1.2, blur, -1.2, 0)


def _clahe_gamma(img: np.ndarray) -> np.ndarray:
    return _gamma(0.5)(_clahe(img))


def _bilateral_gamma(img: np.ndarray) -> np.ndarray:
    smooth = cv2.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=40)
    return _gamma(0.5)(smooth)


def _morph_blackhat(img: np.ndarray) -> np.ndarray:
    """Morphological black-hat highlights dark blobs on light background.

    Result is bright-on-dark; invert so dark marks read as dark again
    (the engine assumes dark = mark).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BUBBLE + 2, BUBBLE + 2))
    bh = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    return 255 - bh


RECIPES: dict[str, callable] = {
    "baseline": _baseline,
    "clahe": _clahe,
    "gamma_0.5": _gamma(0.5),
    "gamma_0.3": _gamma(0.3),
    "levels_stretch": _levels_stretch,
    "levels_aggr": _levels_aggr,
    "levels_balanced": _levels_balanced,
    "levels_strong": _levels_strong,
    "unsharp": _unsharp,
    "clahe+gamma_0.5": _clahe_gamma,
    "bilateral+gamma_0.5": _bilateral_gamma,
    "morph_blackhat": _morph_blackhat,
}


# ── Benchmark runner ──────────────────────────────────────────────────────


@dataclass
class Score:
    sheets: int = 0
    candidate_exact: int = 0
    candidate_digits_correct: int = 0
    candidate_digits_total: int = 0
    candidate_mr_flags: int = 0
    answers_correct: int = 0
    answers_total: int = 0
    answers_nr: int = 0  # engine returned NR / empty for a real mark
    answers_mr_flags: int = 0  # engine flagged multi-mark on a single-mark truth
    elapsed_s: float = 0.0


def _score_results_csv(csv_path: Path, truths: list[SheetTruth]) -> Score:
    import csv as _csv
    truth_by_file = {f"sheet_{i:03d}.png": t for i, t in enumerate(truths)}
    s = Score()
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(_csv.DictReader(fh))
    for row in rows:
        file_id = row.get("file_id", "")
        truth = truth_by_file.get(file_id)
        if truth is None:
            continue
        s.sheets += 1
        cand = row.get("CandidateNumber", "") or ""
        if "MR(" in cand:
            s.candidate_mr_flags += 1
        if cand == truth.candidate:
            s.candidate_exact += 1
        # Per-digit accuracy: zip aligned to truth length, MR(..) digits count wrong.
        s.candidate_digits_total += len(truth.candidate)
        engine_digits = [ch for ch in cand if ch.isdigit() and "MR" not in cand[:cand.index(ch) + 3] if "MR(" not in cand]
        # Simpler: only count digits when engine result is a plain 10-digit string.
        if cand.isdigit() and len(cand) == len(truth.candidate):
            for d_truth, d_engine in zip(truth.candidate, cand):
                if d_truth == d_engine:
                    s.candidate_digits_correct += 1
        for qname, truth_ans in truth.answers.items():
            engine_ans = (row.get(qname, "") or "").strip()
            s.answers_total += 1
            if engine_ans == truth_ans:
                s.answers_correct += 1
            elif engine_ans == "NR" or engine_ans == "":
                s.answers_nr += 1
            elif engine_ans.startswith("MR("):
                s.answers_mr_flags += 1
    return s


def _run_recipe(
    recipe_name: str,
    recipe_fn,
    intensity: int,
    sheets_per_level: int,
    seed: int,
    work_root: Path,
) -> Score:
    rng = np.random.default_rng(seed + intensity)
    inputs = work_root / f"{recipe_name}_int{intensity:03d}"
    if inputs.exists():
        shutil.rmtree(inputs, ignore_errors=True)
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "template.json").write_text(json.dumps(TEMPLATE), encoding="utf-8")
    (inputs / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    truths: list[SheetTruth] = []
    for i in range(sheets_per_level):
        img, truth = _generate_sheet(rng, intensity)
        truths.append(truth)
        processed = recipe_fn(img)
        if processed.dtype != np.uint8:
            processed = np.clip(processed, 0, 255).astype(np.uint8)
        cv2.imwrite(str(inputs / f"sheet_{i:03d}.png"), processed)

    output_dir = work_root / f"out_{recipe_name}_int{intensity:03d}"
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    t0 = time.perf_counter()
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
    elapsed = time.perf_counter() - t0

    csv_files = sorted(output_dir.rglob("Results_*.csv"))
    assert csv_files, f"no Results CSV for {recipe_name} @ {intensity}"
    s = _score_results_csv(csv_files[0], truths)
    s.elapsed_s = elapsed
    return s


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "benchmarks" / "results" / "bench_preprocess_darkness.json"),
    )
    parser.add_argument("--sheets-per-level", type=int, default=SHEETS_PER_LEVEL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--recipes",
        nargs="*",
        default=list(RECIPES.keys()),
        help="Subset of recipes to run (default: all).",
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        type=int,
        default=DARKNESS_LEVELS,
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="omr_bench_") as work_root_str:
        work_root = Path(work_root_str)
        for recipe_name in args.recipes:
            recipe_fn = RECIPES[recipe_name]
            results[recipe_name] = {}
            for intensity in args.levels:
                print(
                    f"  recipe={recipe_name:<22s} intensity={intensity:>3d} ...",
                    end="",
                    flush=True,
                )
                s = _run_recipe(
                    recipe_name,
                    recipe_fn,
                    intensity,
                    args.sheets_per_level,
                    args.seed,
                    work_root,
                )
                results[recipe_name][intensity] = {
                    "sheets": s.sheets,
                    "candidate_exact": s.candidate_exact,
                    "candidate_digits_correct": s.candidate_digits_correct,
                    "candidate_digits_total": s.candidate_digits_total,
                    "candidate_mr_flags": s.candidate_mr_flags,
                    "answers_correct": s.answers_correct,
                    "answers_total": s.answers_total,
                    "answers_nr": s.answers_nr,
                    "answers_mr_flags": s.answers_mr_flags,
                    "elapsed_s": round(s.elapsed_s, 2),
                }
                cand_pct = (
                    s.candidate_exact / s.sheets * 100 if s.sheets else 0.0
                )
                ans_pct = (
                    s.answers_correct / s.answers_total * 100 if s.answers_total else 0.0
                )
                print(
                    f"  cand_exact={cand_pct:5.1f}%  "
                    f"ans={ans_pct:5.1f}%  ({s.elapsed_s:.1f}s)"
                )

    # ── False-positive check on empty sheets ───────────────────────────
    print("\n[Empty-sheet false-positive check]")
    empty_fp: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="omr_bench_empty_") as work_root_str:
        work_root = Path(work_root_str)
        for recipe_name in args.recipes:
            recipe_fn = RECIPES[recipe_name]
            s = _run_empty_sheets(
                recipe_name, recipe_fn, args.sheets_per_level, args.seed, work_root,
            )
            empty_fp[recipe_name] = s
            fp_cand = s["candidate_phantom_digits"]
            fp_ans = s["answers_marked_when_empty"]
            print(
                f"  recipe={recipe_name:<22s} "
                f"candidate phantom marks: {fp_cand:>3d}/{s['candidate_digits_total']:>3d}  "
                f"answer phantom marks: {fp_ans:>3d}/{s['answers_total']:>3d}"
            )

    out_path.write_text(
        json.dumps({"per_recipe": results, "empty_sheet_false_positives": empty_fp}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")

    # ── Verdict ─────────────────────────────────────────────────────────
    print("\n=== VERDICT ===")
    print(_format_verdict(results, args.levels, empty_fp))
    return 0


def _run_empty_sheets(
    recipe_name: str,
    recipe_fn,
    sheets_per_level: int,
    seed: int,
    work_root: Path,
) -> dict:
    """Render unmarked sheets, preprocess, and count any phantom marks."""
    import csv as _csv

    rng = np.random.default_rng(seed + 99999)
    inputs = work_root / f"empty_{recipe_name}"
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "template.json").write_text(json.dumps(TEMPLATE), encoding="utf-8")
    (inputs / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    for i in range(sheets_per_level):
        img = np.full((PAGE_H, PAGE_W), 252, dtype=np.uint8)
        img = _apply_paper_realism(img, rng)
        processed = recipe_fn(img)
        if processed.dtype != np.uint8:
            processed = np.clip(processed, 0, 255).astype(np.uint8)
        cv2.imwrite(str(inputs / f"sheet_{i:03d}.png"), processed)

    output_dir = work_root / f"out_empty_{recipe_name}"
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
    csv_files = sorted(output_dir.rglob("Results_*.csv"))
    assert csv_files, f"no Results CSV for empty {recipe_name}"
    sheets = 0
    cand_phantom = 0
    cand_total = 0
    ans_phantom = 0
    ans_total = 0
    with csv_files[0].open("r", encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            sheets += 1
            cand = (row.get("CandidateNumber", "") or "")
            # Strip any MR(...) groups; everything else that's a digit is a phantom.
            cand_stripped = cand
            while "MR(" in cand_stripped:
                start = cand_stripped.index("MR(")
                end = cand_stripped.index(")", start)
                cand_stripped = cand_stripped[:start] + cand_stripped[end + 1:]
            cand_phantom += sum(1 for ch in cand_stripped if ch.isdigit())
            if "MR(" in cand:
                cand_phantom += cand.count("MR(")  # ambiguity also phantoms
            cand_total += N_CAND_COLS
            for q in range(1, 26):
                v = (row.get(f"q{q}", "") or "").strip()
                ans_total += 1
                if v and v != "NR":
                    ans_phantom += 1
    return {
        "sheets": sheets,
        "candidate_phantom_digits": cand_phantom,
        "candidate_digits_total": cand_total,
        "answers_marked_when_empty": ans_phantom,
        "answers_total": ans_total,
    }


def _format_verdict(
    results: dict[str, dict],
    levels: list[int],
    empty_fp: dict[str, dict] | None = None,
) -> str:
    """Pick the recipe with the best faint-mark accuracy that does not
    regress on the darkest mark level."""
    lines: list[str] = []
    header = f"{'recipe':<22s}" + "".join(f"  int{l:>3d}" for l in levels)
    lines.append(header)
    lines.append("-" * len(header))
    for recipe_name, by_int in results.items():
        cells = []
        for level in levels:
            r = by_int.get(level, {})
            if not r:
                cells.append("    -- ")
                continue
            ans_pct = (r["answers_correct"] / r["answers_total"]) if r["answers_total"] else 0
            cells.append(f"  {ans_pct*100:5.1f}%")
        lines.append(f"{recipe_name:<22s}" + "".join(cells))
    lines.append("")
    lines.append("(values = q1..q25 answer accuracy, %)")
    lines.append("")

    baseline = results.get("baseline", {})
    dark_levels = [l for l in levels if l <= 80]
    faint_levels = [l for l in levels if l >= 170]

    def acc(recipe: str, lvl: int) -> float:
        r = results.get(recipe, {}).get(lvl, {})
        return (r.get("answers_correct", 0) / r["answers_total"]) if r.get("answers_total") else 0.0

    best_recipe = None
    best_score = -1.0
    for recipe in results:
        if not all(lvl in results[recipe] for lvl in levels):
            continue
        # require no dark-level regression vs baseline
        regress = any(acc(recipe, lvl) + 0.001 < acc("baseline", lvl) for lvl in dark_levels)
        if regress and recipe != "baseline":
            continue
        # require zero false positives on empty sheets if data available
        if empty_fp is not None:
            fp = empty_fp.get(recipe, {})
            if fp.get("answers_marked_when_empty", 0) > 0:
                continue
            if fp.get("candidate_phantom_digits", 0) > 0:
                continue
        # score = mean faint-level answer accuracy
        faint_acc = sum(acc(recipe, lvl) for lvl in faint_levels) / max(1, len(faint_levels))
        if faint_acc > best_score:
            best_score = faint_acc
            best_recipe = recipe
    if best_recipe is not None:
        lines.append(
            f"Recommended recipe: {best_recipe!r} "
            f"(mean faint-mark accuracy at intensity >= 170: {best_score*100:.1f}%)"
        )
        for lvl in faint_levels:
            base = acc("baseline", lvl) * 100
            best = acc(best_recipe, lvl) * 100
            lines.append(
                f"  intensity {lvl:>3d}: baseline {base:5.1f}%  ->  {best_recipe} {best:5.1f}%  "
                f"({best - base:+5.1f} pp)"
            )
    else:
        lines.append("No recipe beat baseline without regressing on dark marks.")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
