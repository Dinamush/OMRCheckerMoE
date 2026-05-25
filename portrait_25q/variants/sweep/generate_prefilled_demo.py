"""Render a single prefilled Variant-A portrait sheet for demo / preview.

Uses the same winning geometry as the production design:
  * Candidate bubble = 12 OMR-px (5.0 mm) — the sweep optimum.
  * Answer bubble    = 14 OMR-px (5.9 mm) — the fixed answer-grid size.
  * Fill style       = dark pencil at 85 % fill fraction, the realistic
                       "good student" mark profile that scored 100 % in
                       every robustness test.

The output is written to:
  portrait_25q/reference/prefilled_portrait_25q_demo.png
"""
from __future__ import annotations

from pathlib import Path

from portrait_25q.variants.sweep.fill_bubbles import (
    DARKNESS_PRESETS,
    FillStyle,
    fill_answers,
    fill_candidate_number,
)
from portrait_25q.variants.sweep.generate_sheet import SheetSpec, render_blank


CANDIDATE_NUMBER = "2026051234"
ANSWER_KEY: list[tuple[int, str]] = [
    (1,  "C"), (2,  "A"), (3,  "D"), (4,  "B"), (5,  "C"),
    (6,  "A"), (7,  "D"), (8,  "B"), (9,  "C"), (10, "A"),
    (11, "B"), (12, "D"), (13, "C"), (14, "A"), (15, "B"),
    (16, "D"), (17, "C"), (18, "A"), (19, "B"), (20, "D"),
    (21, "C"), (22, "A"), (23, "B"), (24, "D"), (25, "C"),
]

SPEC = SheetSpec(
    label="size_12",
    cand_bubble_diam=12,
    cand_bubbles_gap_x=28.0,
    cand_labels_gap_y=15.0,
    cand_origin=(118, 252),
)

OUTPUT_PATH = (
    Path(__file__).resolve().parents[2] / "reference" / "prefilled_portrait_25q_demo.png"
)


def main() -> None:
    img = render_blank(SPEC)
    style = FillStyle(color=DARKNESS_PRESETS["dark_pencil"], fill_fraction=0.85)
    fill_candidate_number(img, SPEC, CANDIDATE_NUMBER, style)
    fill_answers(img, ANSWER_KEY, style)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUTPUT_PATH, dpi=(200, 200))
    print(f"Wrote {OUTPUT_PATH}")
    print(f"  Candidate Number : {CANDIDATE_NUMBER}")
    print(f"  Answers          : {' '.join(f'{q}{a}' for q, a in ANSWER_KEY)}")


if __name__ == "__main__":
    main()
