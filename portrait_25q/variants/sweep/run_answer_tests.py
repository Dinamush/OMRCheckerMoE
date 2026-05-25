"""Validate answer-grid bubble robustness at the fixed 5.9 mm diameter.

The candidate-bubble sweep settled on size 12 OMR-px (≈5.0 mm). The
answer bubbles are fixed at ANS_BUBBLE_DIAM = 14 OMR-px (≈5.9 mm) per
the user's brief that "the question bubbles are fine as is". This
harness verifies the answer-grid bubbles are at least as robust as
the chosen candidate-bubble size across the same darkness + edge-case
matrix.

For each test we:
  1. Render the blank sheet at the winning candidate size (12).
  2. Fill candidate digits with dark pencil (control, should always pass).
  3. Fill 25 known answer choices at one of 4 darkness presets, or
     under an edge-case fill pattern.
  4. Run the OMR engine and verify both the candidate and answer fields.

A robust answer grid should match the candidate grid's behaviour
across pen → light pencil and all edge cases.
"""
from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from portrait_25q.variants.sweep.fill_bubbles import (
    DARKNESS_PRESETS,
    FillStyle,
    fill_answers,
    fill_candidate_number,
)
from portrait_25q.variants.sweep.generate_sheet import (
    SheetSpec,
    default_sweep_specs,
    emit_config,
    emit_template,
    render_blank,
)
from portrait_25q.variants.sweep.run_sweep import (
    RUNS_DIR,
    parse_results_csv,
    run_engine,
    TestCase,
)


SIZE_12 = next(s for s in default_sweep_specs() if s.label == "size_12")
CANDIDATE_GT = "0123456789"

ANSWER_KEYS = [
    # 5 deterministic answer keys spanning many A/B/C/D distributions.
    list(zip(range(1, 26), "ABCDABCDABCDABCDABCDABCDA")),
    list(zip(range(1, 26), "DDDDAAAABBBBCCCCDDDDAAABB")),
    list(zip(range(1, 26), "ABABABABABABABABABABABABA")),
    list(zip(range(1, 26), "CDCDCDCDCDCDCDCDCDCDCDCDC")),
    list(zip(range(1, 26), "BACBADCBADCBADCBADCBADCBA")),
]


@dataclass(frozen=True)
class AnswerCase:
    label: str            # darkness preset name OR edge case name
    style: FillStyle


CASES = [
    AnswerCase("pen_ink",       FillStyle(color=DARKNESS_PRESETS["pen_ink"],       fill_fraction=0.85)),
    AnswerCase("dark_pencil",   FillStyle(color=DARKNESS_PRESETS["dark_pencil"],   fill_fraction=0.85)),
    AnswerCase("medium_pencil", FillStyle(color=DARKNESS_PRESETS["medium_pencil"], fill_fraction=0.85)),
    AnswerCase("light_pencil",  FillStyle(color=DARKNESS_PRESETS["light_pencil"],  fill_fraction=0.85)),
    AnswerCase("half_fill",     FillStyle(color=DARKNESS_PRESETS["dark_pencil"],   fill_fraction=0.85, partial_mode="half")),
    AnswerCase("off_centre",    FillStyle(color=DARKNESS_PRESETS["dark_pencil"],   fill_fraction=0.7, offset_fraction=(0.3, 0.0))),
    AnswerCase("tick_mark",     FillStyle(color=DARKNESS_PRESETS["pen_ink"],       fill_fraction=0.8, partial_mode="tick")),
]


def prepare(case: AnswerCase, key_index: int) -> tuple[TestCase, list[tuple[int, str]]]:
    answers = ANSWER_KEYS[key_index]
    case_dir = RUNS_DIR / "answer" / f"{case.label}__key{key_index}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    emit_config(case_dir)
    template_path = emit_template(SIZE_12, case_dir)
    img = render_blank(SIZE_12)
    img = fill_candidate_number(
        img, SIZE_12, CANDIDATE_GT,
        FillStyle(color=DARKNESS_PRESETS["dark_pencil"], fill_fraction=0.85),
    )
    img = fill_answers(img, answers, case.style)
    image_path = case_dir / f"filled_{case.label}_key{key_index}.png"
    img.save(image_path, format="PNG", compress_level=1)
    return TestCase(
        spec=SIZE_12,
        darkness=case.label,
        ground_truth=CANDIDATE_GT,
        image_path=image_path,
        template_path=template_path,
        output_dir=case_dir / "out",
    ), answers


def score_answers(row: dict | None, ground_truth_answers: list[tuple[int, str]]) -> dict:
    if row is None:
        return {
            "engine_ok": False,
            "answer_correct": 0,
            "answer_total": 25,
            "candidate_correct": 0,
            "wrong_predictions": [],
        }
    answer_correct = 0
    wrong = []
    for q, expected in ground_truth_answers:
        col_name = f"q{q}"
        predicted = (row.get(col_name) or "").strip()
        if predicted == expected:
            answer_correct += 1
        else:
            wrong.append((q, expected, predicted))
    cand_predicted = (row.get("CandidateNumber") or "").strip()
    cand_correct = sum(
        1 for i in range(10) if i < len(cand_predicted) and cand_predicted[i] == CANDIDATE_GT[i]
    )
    return {
        "engine_ok": True,
        "answer_correct": answer_correct,
        "answer_total": len(ground_truth_answers),
        "candidate_correct": cand_correct,
        "wrong_predictions": wrong[:5],  # cap for readability
    }


def main() -> None:
    results: list[dict] = []
    total_cases = len(CASES) * len(ANSWER_KEYS)
    case_idx = 0
    for case in CASES:
        for key_index in range(len(ANSWER_KEYS)):
            case_idx += 1
            tc, answers = prepare(case, key_index)
            exit_code, _ = run_engine(tc)
            row = parse_results_csv(tc) if exit_code == 0 else None
            score = score_answers(row, answers)
            score["case_label"] = case.label
            score["key_index"] = key_index
            results.append(score)
            ok = "OK" if score.get("engine_ok") and score["answer_correct"] == 25 else "NO"
            print(
                f"[{case_idx:>2}/{total_cases}] {case.label:14s} key{key_index}"
                f" answers={score['answer_correct']}/{score['answer_total']}"
                f" cand={score['candidate_correct']}/10  {ok}"
            )
            if score.get("wrong_predictions"):
                for q, expected, pred in score["wrong_predictions"]:
                    print(f"        q{q}: expected {expected}, got {pred!r}")

    print()
    print("Answer-grid accuracy summary (per case):")
    for case in CASES:
        relevant = [r for r in results if r["case_label"] == case.label]
        total_q = sum(r["answer_total"] for r in relevant)
        corr_q = sum(r["answer_correct"] for r in relevant)
        print(f"  {case.label:14s} {corr_q}/{total_q} ({corr_q/max(1,total_q)*100:5.1f} %)")

    out_path = Path(__file__).resolve().parent / "results" / "answer_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
