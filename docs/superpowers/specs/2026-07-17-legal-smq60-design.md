# Legal 60-Question MoE Answer Sheets (SMQ60)

**Status:** Approved for generation (not yet registered in WebUI).  
**Date:** 2026-07-17

## Goals

- Same MoE header language/layout family as `MoE-April-2026-Landscape-NNQ25-0` blank.
- **60** MCQ questions (A–D), arranged **6 columns × 10 rows**.
- **US Legal** paper (8.5″ × 14″), both **landscape** and **portrait**.
- ArUco centres inset **~0.5″** from page edges for clipping/smudge robustness.
- Ship complete `template.json` + `config.json` so scanning can be wired later without redesign.
- **Do not** register in `sheet_registry` / WebUI until MoE confirms.

## Folders

| Directory | Orientation |
|---|---|
| `MoE-July-2026-Landscape-SMQ60-0/` | Legal landscape |
| `MoE-July-2026-Portrait-SMQ60-0/` | Legal portrait |

Shared generator: `scripts/sheets/generate_legal_smq60.py`

## Geometry

| Property | Landscape | Portrait |
|---|---|---|
| Print @ 300 DPI | 4200 × 2550 | 2550 × 4200 |
| OMR canvas | 840 × 510 | 510 × 840 |
| Marker centres | 0.5″ inset | 0.5″ inset |
| Dictionary / IDs | `DICT_4X4_50` / `[0,1,2,3]` | same |
| Bubble size (OMR) | 10 × 10 | 10 × 10 |

## Fields

- Header: title, instructions, Student Name / School Name / Exam Name / Student Signature
- Candidate Number: 10×10 `QTYPE_INT`
- Answers: six `QTYPE_MCQ4` blocks (`q1..10` … `q51..60`)

## Out of scope (for now)

- WebUI preset registration
- Prefill CSV pipeline wiring
- Printed calibration against physical Xerox batches
