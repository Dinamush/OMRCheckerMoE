# Letter Landscape 60-Question MoE Answer Sheet (SMQ60)

**Status:** Approved 2026-07-27  
**Date:** 2026-07-27

## Goals

- New **US Letter landscape** (11″ × 8.5″) 60Q sheet, separate from Legal July SMQ60.
- Keep `MoE-July-2026-Landscape-SMQ60-0` (Legal) unchanged.
- Identity: Student Name, **Centre Name**, Exam Name, **Subject Name**, Student Signature.
- Shorter write-in lines; slightly larger candidate-number table.
- Ship blank + sample with candidate number **0009027001**.
- Not registered in WebUI until MoE confirms.

## Folder

| Directory | Paper |
|---|---|
| `MoE-July-2026-Letter-Landscape-SMQ60-0/` | Letter landscape |

Generator: `scripts/sheets/generate_letter_smq60.py`

## Geometry

| Property | Value |
|---|---|
| Print @ 300 DPI | 3300 × 2550 |
| OMR canvas | 660 × 510 |
| Marker centres | ~0.5″ inset |
| Dictionary / IDs | `DICT_4X4_50` / `[0,1,2,3]` |
| Marker size | Match April legacy (~108 px / ~0.36″) |
| Answers | 6×10 A–D; bubble diam tuned for Letter width |
| Candidate wells | ~8–10% larger than Legal landscape (Legal cand ≈ 12 OMR-px) |

## Identity

1. Student Name  
2. Centre Name  
3. Exam Name  
4. Subject Name  
5. Student Signature  

Write-in lines end **0.5″** before the candidate panel (not flush).

## Sample

- `reference/sample_0009027001.png` (+ PDF): blank with candidate **0009027001** bubbled and written in.

## Out of scope

- WebUI / `sheet_registry` registration  
- Portrait Letter variant  
- Changing Legal July sheets  
