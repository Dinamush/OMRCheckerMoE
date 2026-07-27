# MoE-July-2026-Letter-Landscape-SMQ60-0

US Letter **landscape** (11″ × 8.5″) MoE answer sheet — **60** MCQ questions (6×10).

## Contents

| File | Role |
|---|---|
| `template.json` | OMR field map (ArUco CropOnMarkers) |
| `config.json` | Processing dimensions |
| `reference/blank_landscape_smq60.png` | Printable blank |
| `reference/blank_landscape_smq60.pdf` | Printable PDF @ 300 DPI |
| `reference/sample_0009027001.png` | Sample with candidate bubbled |
| `reference/alignment_preview_landscape.png` | Template-bubble overlay QA |
| `generate_blank.py` | Regenerate from shared script |

## Notes

- Not registered in WebUI / `sheet_registry` yet (pending MoE approval).
- Identity: Student / Centre / Subject / Exam / Signature.
- Marker centres inset ~0.5″ from page edges.
- Shared generator: `scripts/sheets/generate_letter_smq60.py`
- Legal July landscape remains in `MoE-July-2026-Landscape-SMQ60-0/`.
