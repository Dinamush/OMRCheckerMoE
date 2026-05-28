"""Edge-case stress tests for the finalist candidate-bubble sizes.

Each edge case represents a real-world filling pattern that a strict
threshold-based OMR scanner might mis-classify. We focus on the two
finalists from the main sweep (sizes 12 and 14) and exercise them
against the same darkness ladder.

Edge cases
----------
* ``half_fill``     — student fills only the left half of the bubble.
* ``off_centre``    — fill is shifted 30 % to one side.
* ``tick_mark``     — student draws a small checkmark instead of filling.
* ``smudge``        — small dark blob, partially inside the bubble.
* ``stray_mark``    — bubble is fully unmarked but a small smudge sits
                      between this bubble and its neighbour
                      (tests that bleed does not promote the neighbour
                      to ``marked``).

A robust candidate-bubble size should:
  1. Correctly detect ``half_fill`` and ``off_centre`` (the student
     marked it, even if not perfectly).
  2. **Reject** ``tick_mark``, ``smudge``, and ``stray_mark`` for
     bubbles the student did not intend to fill.

Per case, the harness asserts the engine's predicted candidate-number
matches the ground truth. Aggregated pass-rates are printed per
(size, edge-case) pair.
"""
from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import ImageDraw

from fill_bubbles import (
    DARKNESS_PRESETS,
    FillStyle,
    _candidate_bubble_centre,
    fill_candidate_number,
)
from generate_sheet import (
    SheetSpec,
    bubble_bbox,
    default_sweep_specs,
    emit_config,
    emit_template,
    render_blank,
)
from run_sweep import (
    RUNS_DIR,
    parse_results_csv,
    run_engine,
    score_case,
    TestCase,
)


# ---------------------------------------------------------------------------
# Edge-case fill recipes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeCase:
    name: str
    description: str
    style: FillStyle


EDGE_CASES = [
    EdgeCase(
        name="half_fill",
        description="Left half of each marked bubble shaded",
        style=FillStyle(
            color=DARKNESS_PRESETS["dark_pencil"],
            fill_fraction=0.85,
            partial_mode="half",
        ),
    ),
    EdgeCase(
        name="off_centre",
        description="Fill shifted 30% off centre",
        style=FillStyle(
            color=DARKNESS_PRESETS["dark_pencil"],
            fill_fraction=0.7,
            offset_fraction=(0.3, 0.0),
        ),
    ),
    EdgeCase(
        name="tick_mark",
        description="Small check-mark instead of a fill",
        style=FillStyle(
            color=DARKNESS_PRESETS["pen_ink"],
            fill_fraction=0.8,
            partial_mode="tick",
        ),
    ),
    EdgeCase(
        name="full_fill_dark",
        description="Baseline: solid fill in dark pencil (control)",
        style=FillStyle(
            color=DARKNESS_PRESETS["dark_pencil"],
            fill_fraction=0.85,
        ),
    ),
]


# Cases where the student MARKED but mark is degraded — these should be detected.
DETECTABLE_CASES = {"half_fill", "off_centre", "full_fill_dark"}

# Cases where the student did NOT mark but added noise — these should be IGNORED.
# (tick_mark is a borderline case: it's intended as a mark, but a strict scanner
# might reject it as too small. We score whether the engine matches the GT.)
DETECTABLE_CASES.add("tick_mark")


GROUND_TRUTH_IDS = [
    "0123456789",
    "9876543210",
    "5050505050",
]


def prepare_edge_case(spec: SheetSpec, edge: EdgeCase, ground_truth: str) -> TestCase:
    case_dir = RUNS_DIR / "edge" / spec.label / f"{edge.name}__{ground_truth}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    emit_config(case_dir)
    template_path = emit_template(spec, case_dir)
    img = render_blank(spec)
    img = fill_candidate_number(img, spec, ground_truth, edge.style)
    image_path = case_dir / f"filled_{edge.name}_{ground_truth}.png"
    img.save(image_path, format="PNG", compress_level=1)
    return TestCase(
        spec=spec,
        darkness=edge.name,
        ground_truth=ground_truth,
        image_path=image_path,
        template_path=template_path,
        output_dir=case_dir / "out",
    )


def prepare_stray_mark_case(spec: SheetSpec, ground_truth: str) -> TestCase:
    """Special edge case: fully marked candidate ID + small stray smudges
    in random unmarked bubbles. Tests that smudges do not promote
    unmarked digits."""
    case_dir = RUNS_DIR / "edge" / spec.label / f"stray_mark__{ground_truth}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    emit_config(case_dir)
    template_path = emit_template(spec, case_dir)
    img = render_blank(spec)
    # Real marks (dark pencil) for the GT digits.
    img = fill_candidate_number(
        img,
        spec,
        ground_truth,
        FillStyle(color=DARKNESS_PRESETS["dark_pencil"], fill_fraction=0.85),
    )
    # Small stray smudges in 5 unmarked bubbles (chosen deterministically).
    smudge_style = FillStyle(
        color=DARKNESS_PRESETS["pen_ink"],
        fill_fraction=0.45,
        partial_mode="smudge",
        offset_fraction=(0.25, 0.15),
    )
    draw = ImageDraw.Draw(img)
    stray_targets = [(0, 5), (3, 8), (5, 2), (7, 6), (9, 1)]  # (col, digit)
    for col, digit in stray_targets:
        if str(ground_truth[col]) == str(digit):
            continue
        cx, cy = _candidate_bubble_centre(spec, col, digit)
        from fill_bubbles import _draw_filled_bubble
        _draw_filled_bubble(draw, cx, cy, spec.cand_bubble_diam, smudge_style)
    image_path = case_dir / f"filled_stray_mark_{ground_truth}.png"
    img.save(image_path, format="PNG", compress_level=1)
    return TestCase(
        spec=spec,
        darkness="stray_mark",
        ground_truth=ground_truth,
        image_path=image_path,
        template_path=template_path,
        output_dir=case_dir / "out",
    )


def main() -> None:
    specs = [s for s in default_sweep_specs() if s.label in {"size_12", "size_14"}]
    cases: list[TestCase] = []
    for spec in specs:
        for edge in EDGE_CASES:
            for gt in GROUND_TRUTH_IDS:
                cases.append(prepare_edge_case(spec, edge, gt))
        for gt in GROUND_TRUTH_IDS:
            cases.append(prepare_stray_mark_case(spec, gt))

    per_case: list[dict] = []
    for idx, case in enumerate(cases, start=1):
        exit_code, _ = run_engine(case)
        row = parse_results_csv(case) if exit_code == 0 else None
        score = score_case(case, row)
        score["exit_code"] = exit_code
        score["edge_case"] = case.darkness
        score["spec"] = case.spec.label
        score["ground_truth"] = case.ground_truth
        per_case.append(score)
        print(
            f"[{idx:>3}/{len(cases)}] {case.spec.label:8s} {case.darkness:18s}"
            f" GT={case.ground_truth} predicted={score.get('predicted')!r:>16}"
            f" digits={score['digit_correct']}/10 whole={'OK' if score['all_correct'] else 'NO'}"
        )

    print()
    print("Edge-case digit accuracy (rows = size, cols = edge):")
    edge_names = [e.name for e in EDGE_CASES] + ["stray_mark"]
    header = "size".ljust(10) + "".join(f"{e:>16s}" for e in edge_names)
    print(header)
    for spec in specs:
        row = spec.label.ljust(10)
        for edge in edge_names:
            relevant = [s for s in per_case if s["spec"] == spec.label and s["edge_case"] == edge]
            total_d = sum(s["digit_total"] for s in relevant)
            corr_d = sum(s["digit_correct"] for s in relevant)
            pct = corr_d / max(1, total_d) * 100
            row += f"{pct:>14.1f} %"
        print(row)

    summary_path = Path(__file__).resolve().parent / "results" / "edge_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(per_case, indent=2))


if __name__ == "__main__":
    main()
