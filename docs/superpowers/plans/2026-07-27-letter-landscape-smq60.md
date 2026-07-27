# Letter Landscape SMQ60 Implementation Plan

> **For agentic workers:** Implement task-by-task. Steps use checkbox syntax.

**Goal:** Add a US Letter landscape 60Q MoE sheet folder with blank, templates, and a bubbled sample for candidate `0009027001`.

**Architecture:** New dedicated generator `scripts/sheets/generate_letter_smq60.py` (Legal July sheets untouched). Outputs under `MoE-July-2026-Letter-Landscape-SMQ60-0/`.

**Tech Stack:** Python, Pillow, OpenCV ArUco, existing OMRChecker template JSON shape.

## Global Constraints

- Letter landscape only: 11″ × 8.5″ @ 300 DPI (3300×2550 print, 660×510 OMR)
- Do not modify Legal July SMQ60 folders
- Identity labels: Student Name, Centre Name, Subject Name, Exam Name, Student Signature
- Write-in lines ~72% of available span
- Candidate table ~8–10% larger than Legal landscape cand wells
- ArUco: DICT_4X4_50, IDs [0,1,2,3], April-sized markers, ~0.5″ inset
- Blank must read 60/60 NR; sample must read CandidateNumber `0009027001`
- Not registered in WebUI

---

### Task 1: Generator + folder outputs

- [x] Write `scripts/sheets/generate_letter_smq60.py`
- [x] Emit `MoE-July-2026-Letter-Landscape-SMQ60-0/` (blank PNG/PDF, template, config, preview, README, helper)
- [x] Emit sample with candidate `0009027001`
- [x] Update `SHEETS.md`
- [x] Verify ArUco detect + blank NR + sample candidate read
