# `MoE-May-2026-Variants-SMQ25-0/` — Portrait-orientation 25-question OMR sheet (research + design)

> **Status:** design + initial implementation. Calibration of the printed template
> against actual OMR readings is **not yet complete** — treat this as a research
> spike, not a drop-in replacement for `MoE-April-2026-Landscape-NNQ25-0/`.

## What's in here

| Path | Purpose |
| --- | --- |
| [`DESIGN.md`](DESIGN.md) | Full design rationale for the portrait layout: page geometry, ArUco placement, bubble grid, fonts, and trade-offs against the existing landscape sheet. |
| [`template.json`](template.json) | OMR engine template (`pageDimensions`, `bubbleDimensions`, `fieldBlocks`, `preProcessors`) for the portrait layout. Coordinates are in OMR-processing space (515×666 — aspect-matched to US Letter). Top-level `bubbleDimensions: [13, 13]` (5.5 mm Ø) drives the answer blocks; the candidate-number block overrides this to `[10, 10]` (4.2 mm Ø) so the 10×10 digit grid fits without spilling into the bottom ArUco quiet zone. |
| [`generate_blank.py`](generate_blank.py) | Standalone Python script that renders the blank printable PNG (`reference/blank_MoE-May-2026-Variants-SMQ25-0.png`) from the constants documented in `DESIGN.md §4`. Every constant carries an inline comment citing the industry / accessibility / motor-skill / camera-detection research line that justifies the value. Re-run after any layout change. |
| [`reference/blank_MoE-May-2026-Variants-SMQ25-0.png`](reference/blank_MoE-May-2026-Variants-SMQ25-0.png) | The generated blank reference PNG (**US Letter portrait** at 200 DPI, **1700×2200 px**). 5.5 mm answer bubbles, 14 pt header labels, 22 pt bold title, and 12.6 mm ArUco fiducials. Used by the prefill / student-fill pipeline as the canvas for student details + bubble fills. |
| [`inputs/`](inputs/) | Drop scanned (or pre-filled) sheets here when running OMR against this template via `python main.py -i MoE-May-2026-Variants-SMQ25-0/`. |

## Why portrait?

The existing 25Q sheet (`MoE-April-2026-Landscape-NNQ25-0/template.json`) is landscape:
666 × 515 px in OMR space, A4-landscape print. That layout was inherited from the
upstream OMRChecker examples and is awkward in practice because:

1. **Schools mostly print on portrait paper** — most classroom printers ship
   set to portrait US Letter (or portrait A4); landscape sheets force a print
   dialog change, which trips up exam invigilators. The portrait redesign
   targets US Letter (8.5" × 11") by default, since that's the dominant paper
   size in the North American / Caribbean markets the sheet serves.
2. **Filing and stapling assumes portrait** — answer sheets are usually filed
   alongside scripts that are written on portrait paper.
3. **The horizontal answer-block layout is visually noisy** — five blocks of five
   questions across the page means students must scan left-to-right repeatedly.
   A two-column 13/12 portrait layout matches the reading order students already
   use for written exam papers.
4. **Mobile-phone scanning is biased toward portrait** — the back camera and
   document-scanning UI on phones default to portrait, and skew correction is
   easier when the long axis is vertical.

This sub-package is a research spike towards replacing the landscape sheet with
an optimised portrait equivalent that captures the **same fields** (student
name, school, exam, region [optional], 10-digit candidate number, 25 × 4-option
questions) but with better-calibrated bubble geometry, clearer instructions,
and an optional **region** field that integrates with the new batch-grouping
feature on the prefill page.

## How to regenerate the blank PNG

```bash
python MoE-May-2026-Variants-SMQ25-0/generate_blank.py
```

This rewrites `reference/blank_MoE-May-2026-Variants-SMQ25-0.png` from scratch using the
constants defined inside the script (which match `DESIGN.md`). You should re-run
this after every layout adjustment, and then re-calibrate `template.json` using
`scripts/diagnostics/calibrate_bubbles.py` (adapted from the landscape calibration flow
described in [`docs/audits/audit_report_20260524.md`](../docs/audits/audit_report_20260524.md)).

## How to run OMR against this template (once calibrated)

```bash
python main.py -i MoE-May-2026-Variants-SMQ25-0/ -o MoE-May-2026-Variants-SMQ25-0/outputs
```

…with calibrated, pre-filled scans dropped into `MoE-May-2026-Variants-SMQ25-0/inputs/`.

## Next steps before this can replace the landscape sheet

1. Calibrate `template.json` against the generated reference using
   `scripts/diagnostics/calibrate_bubbles.py` (see `docs/audits/audit_report_20260524.md`).
2. Wire the prefill service so the user can pick the portrait template via
   a `template_id` query parameter on `/api/v1/prefill/single` and
   `/api/v1/prefill/batch`.
3. Add an OMR roundtrip test mirroring
   `webui/tests/test_student_fill_omr_roundtrip.py` for the portrait sheet.
4. Make `region` a first-class field on the printed sheet (already wired
   into the grouping feature on the prefill page).
