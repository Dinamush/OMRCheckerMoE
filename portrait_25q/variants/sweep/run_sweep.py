"""End-to-end runner for the candidate-bubble size sweep.

For each (bubble diameter × darkness preset) combination it:
  1. Renders a blank sheet at the right diameter.
  2. Fills a known 10-digit candidate number at the chosen darkness.
  3. Writes a matched ``template.json`` next to the filled image.
  4. Invokes ``python main.py -i <dir> -o <dir>/out`` (the OMR engine).
  5. Parses the per-image ``Results_*.csv`` and compares to ground truth.

Aggregated accuracy is printed as a (diameter × darkness) matrix and
also written to ``portrait_25q/variants/sweep/results/summary.json``.

Run::

    python -m portrait_25q.variants.sweep.run_sweep
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from portrait_25q.variants.sweep.fill_bubbles import (
    DARKNESS_PRESETS,
    FillStyle,
    fill_candidate_number,
)
from portrait_25q.variants.sweep.generate_sheet import (
    SheetSpec,
    default_sweep_specs,
    emit_config,
    emit_template,
    render_blank,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SWEEP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SWEEP_DIR / "results"
RUNS_DIR = SWEEP_DIR / "runs"

# A handful of randomly-chosen but deterministic candidate IDs. Using
# multiple distinct IDs per (size, darkness) cell averages out positional
# bias (e.g. column 0 always being "easier" than column 9).
GROUND_TRUTH_IDS = [
    "0123456789",
    "9876543210",
    "1357902468",
    "4204825196",
    "7531468290",
]


@dataclass
class TestCase:
    spec: SheetSpec
    darkness: str
    ground_truth: str
    image_path: Path
    template_path: Path
    output_dir: Path

    @property
    def case_id(self) -> str:
        return f"{self.spec.label}__{self.darkness}__{self.ground_truth}"


def prepare_case(spec: SheetSpec, darkness: str, ground_truth: str) -> TestCase:
    """Render + fill + write template into a per-case directory."""
    case_dir = RUNS_DIR / spec.label / f"{darkness}__{ground_truth}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    emit_config(case_dir)
    template_path = emit_template(spec, case_dir)
    img = render_blank(spec)
    style = FillStyle(color=DARKNESS_PRESETS[darkness])
    img = fill_candidate_number(img, spec, ground_truth, style)
    image_path = case_dir / f"filled_{ground_truth}.png"
    img.save(image_path, format="PNG", compress_level=1)
    output_dir = case_dir / "out"
    return TestCase(
        spec=spec,
        darkness=darkness,
        ground_truth=ground_truth,
        image_path=image_path,
        template_path=template_path,
        output_dir=output_dir,
    )


def run_engine(case: TestCase) -> tuple[int, str]:
    """Invoke ``python main.py -i case_dir -o case_dir/out``.

    Returns the (exit_code, combined stdout+stderr) so callers can log
    engine failures alongside accuracy results.
    """
    case_dir = case.image_path.parent
    cmd = [
        sys.executable,
        "main.py",
        "-i",
        str(case_dir),
        "-o",
        str(case.output_dir),
    ]
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        # Use UTF-8 + replace so the engine's Rich-formatted log output
        # (which emits Unicode glyphs like ✓ and box-drawing chars) does
        # not crash subprocess decoding on Windows cp1252 consoles.
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=120,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def parse_results_csv(case: TestCase) -> dict | None:
    """Find and parse the engine's ``Results_*.csv`` for this case.

    Returns the first row as a dict (column-name → value), or None when
    the engine failed to produce a results file.
    """
    results_root = case.output_dir
    if not results_root.exists():
        return None
    candidates = sorted(results_root.rglob("Results_*.csv"))
    if not candidates:
        return None
    with candidates[0].open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        return None
    return rows[0]


def score_case(case: TestCase, row: dict | None) -> dict:
    """Compute per-digit + whole-string accuracy for one case."""
    if row is None:
        return {
            "case_id": case.case_id,
            "engine_ok": False,
            "predicted": None,
            "digit_correct": 0,
            "digit_total": 10,
            "all_correct": False,
        }
    raw_predicted = row.get("CandidateNumber", "")
    predicted = (raw_predicted or "").strip()
    truth = case.ground_truth
    digit_correct = 0
    for i in range(10):
        truth_ch = truth[i]
        pred_ch = predicted[i] if i < len(predicted) else ""
        if pred_ch == truth_ch:
            digit_correct += 1
    return {
        "case_id": case.case_id,
        "engine_ok": True,
        "predicted": predicted,
        "digit_correct": digit_correct,
        "digit_total": 10,
        "all_correct": predicted == truth,
    }


def run_full_sweep() -> dict:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    specs = default_sweep_specs()

    cases: list[TestCase] = []
    for spec in specs:
        for darkness in DARKNESS_PRESETS:
            for ground_truth in GROUND_TRUTH_IDS:
                cases.append(prepare_case(spec, darkness, ground_truth))

    per_case_scores: list[dict] = []
    for idx, case in enumerate(cases, start=1):
        exit_code, output = run_engine(case)
        row = parse_results_csv(case) if exit_code == 0 else None
        score = score_case(case, row)
        score["exit_code"] = exit_code
        score["spec"] = {
            "label": case.spec.label,
            "cand_bubble_diam": case.spec.cand_bubble_diam,
            "cand_bubbles_gap_x": case.spec.cand_bubbles_gap_x,
            "cand_labels_gap_y": case.spec.cand_labels_gap_y,
        }
        score["darkness"] = case.darkness
        score["ground_truth"] = case.ground_truth
        if exit_code != 0:
            score["engine_tail"] = output[-600:]
        per_case_scores.append(score)
        print(
            f"[{idx:>3}/{len(cases)}] {case.case_id}"
            f"  predicted={score.get('predicted')!r:>14}"
            f"  digits={score['digit_correct']}/10"
            f"  whole={'OK' if score['all_correct'] else 'NO'}"
        )

    # Aggregate by (spec_label, darkness) → mean digit-accuracy.
    aggregate: dict[str, dict[str, dict]] = {}
    for spec in specs:
        aggregate[spec.label] = {}
        for darkness in DARKNESS_PRESETS:
            relevant = [
                s
                for s in per_case_scores
                if s["spec"]["label"] == spec.label and s["darkness"] == darkness
            ]
            total_digits = sum(s["digit_total"] for s in relevant)
            correct_digits = sum(s["digit_correct"] for s in relevant)
            whole_correct = sum(1 for s in relevant if s["all_correct"])
            aggregate[spec.label][darkness] = {
                "digit_accuracy": correct_digits / max(1, total_digits),
                "whole_id_accuracy": whole_correct / max(1, len(relevant)),
                "n_cases": len(relevant),
            }

    summary = {
        "specs": [asdict(spec) for spec in specs],
        "darknesses": list(DARKNESS_PRESETS.keys()),
        "ground_truth_ids": GROUND_TRUTH_IDS,
        "per_case": per_case_scores,
        "aggregate": aggregate,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print("Aggregate digit accuracy (rows = size, cols = darkness):")
    header = "size".ljust(10) + "".join(f"{d:>16s}" for d in DARKNESS_PRESETS)
    print(header)
    for spec in specs:
        row = spec.label.ljust(10)
        for darkness in DARKNESS_PRESETS:
            stats = aggregate[spec.label][darkness]
            row += f"{stats['digit_accuracy']*100:>14.1f} %"
        print(row)

    print()
    print("Aggregate whole-ID accuracy (rows = size, cols = darkness):")
    print(header)
    for spec in specs:
        row = spec.label.ljust(10)
        for darkness in DARKNESS_PRESETS:
            stats = aggregate[spec.label][darkness]
            row += f"{stats['whole_id_accuracy']*100:>14.1f} %"
        print(row)

    return summary


if __name__ == "__main__":
    run_full_sweep()
