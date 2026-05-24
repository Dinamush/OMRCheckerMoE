# Portrait 25-Question OMR Sheet — Design Document

**Status:** v1 design + scaffolding. Not yet calibrated against printed output.
**Replaces:** `custom_25_definitive_final/` (landscape 25Q sheet) once
calibration completes.

---

## 1. Goals

Optimise the existing 25-question OMR sheet for:

1. **Real-world print/file conventions** — A4 portrait, the default page
   orientation for school papers in Caribbean and Commonwealth schools.
2. **Optical reliability** — wider bubble spacing, larger fiducial markers,
   high-contrast quiet zones, instructions that discourage stray marks.
3. **Phone-camera scanning** — portrait aspect ratio matches the default
   document-scanner UI on iOS and Android.
4. **Extra metadata** — first-class **Region** field (printed and bubblable),
   integrates with the new grouping feature on `/api/v1/prefill/batch`.

It captures the **same fields** as the existing landscape sheet:

| Field | Existing landscape | Portrait redesign |
| --- | --- | --- |
| Student name | text strip | text strip |
| School name | text strip | text strip |
| Exam name | text strip | text strip |
| **Region** | — | **new text strip + reserved space** |
| Candidate number | 10×10 bubble grid | 10×10 bubble grid (vertical stack) |
| Q1 – Q25 (A/B/C/D) | 5 blocks of 5 questions, horizontal | **2 columns of 13 / 12 questions, vertical** |

---

## 2. Page geometry

| Property | Value | Notes |
| --- | --- | --- |
| Paper | A4 portrait | 210 mm × 297 mm. Same engine works for US Letter portrait by rendering at a different print DPI. |
| Print DPI | 200 | Industry-standard for OMR. Yields 1654 × 2339 px print canvas. |
| OMR processing space | **515 × 728 px** | Aspect ≈ 1 : 1.414, matches A4 portrait. Picked so existing bubble dimensions (10×10) and gap conventions (≈20 px) keep their meaning. |
| Margins (OMR space) | 14 px each side | ≈ 7 mm at 200 DPI print. Big enough to leave a quiet zone around the ArUco markers without wasting page real estate. |

> **Why 515 × 728 and not the print resolution?** The OMR engine reads
> `pageDimensions` from the template and homographs the input image onto that
> internal canvas before sampling bubbles. Smaller canvas means faster sampling
> and less RAM. The 666 × 515 landscape canvas has worked well, so we pick a
> portrait equivalent at the **same pixel density** (≈ 0.4 px / mm).

---

## 3. ArUco fiducial markers

| Property | Value | Source / rationale |
| --- | --- | --- |
| Dictionary | `DICT_4X4_50` | Same as landscape sheet; the engine's `CropOnMarkers.py` already supports it. |
| Marker IDs | 0 (TL), 1 (TR), 2 (BL), 3 (BR) | Convention used by the landscape sheet — keeps `arucoCornerIds: [0, 1, 2, 3]` valid. |
| Marker size (OMR space) | **27 × 27 px** | ≈ 14 mm at 200 DPI. ~5 % of page width — within the 6–8 % range that OpenCV's ArUco detector handles reliably. |
| Marker centre coordinates | `[14, 14]`, `[501, 14]`, `[14, 714]`, `[501, 714]` | 14 px from each corner edge; gives a one-bit quiet-zone visible to the detector at any reasonable scan resolution. |
| Quiet zone | 1 marker-bit (≈ 6 px in OMR space) on all sides | Required by the ArUco spec; the layout reserves this implicitly via the 14 px corner offset. |

---

## 4. Bubble geometry

| Property | Value | Rationale |
| --- | --- | --- |
| Bubble shape | Circle outline | Matches landscape sheet, matches the `student_fill` drawing routines, OpenCV thresholding is well-tuned for circular targets. |
| Bubble dimensions (OMR space) | **10 × 10 px** | Same as the landscape sheet (`bubbleDimensions: [10, 10]`) so existing engine sampling code, `student_fill` profiles, and tests all carry over. |
| Bubble print diameter | ≈ 5 mm at 200 DPI | Industry-standard OMR bubble size; aligns with what students are taught to fill. |
| `bubblesGap` (horizontal between A/B/C/D) | 22 px | Wider than landscape's 20 px to make accidental adjacent-mark errors less likely on portrait. |
| `labelsGap` (vertical between questions) | 22 px | Tight vertical packing — portrait sheet has plenty of vertical room for 13 rows × 22 px = 286 px answer column. |
| Letter labels (A/B/C/D) | drawn **inside** each bubble outline | Matches the existing prefill template and `student_fill.draw_student_marks`. Industry convention is split: Scantron prints labels above bubbles; ZipGrade and most low-cost OMR providers print them inside. Inside is easier on a phone-camera scan because it keeps the label aligned with its bubble even under modest skew. |

---

## 5. Layout (OMR space, 515 wide × 728 tall)

```
+--------------------------------------------------------------+
|  [A] ArUco TL (14,14)                  [B] ArUco TR (501,14) |
|                                                              |
|  -- Title strip -------                                      |
|  y ≈ 40   "Grade 5 Reading - Answer Sheet"                   |
|  y ≈ 65   "Fill bubbles completely. Use #2 pencil only."     |
|                                                              |
|  -- Header text strips --                                    |
|  y ≈ 95   School:   _______________________________          |
|  y ≈ 120  Exam:     _______________________________          |
|  y ≈ 145  Region:   _______________________________  <NEW>   |
|  y ≈ 170  Student:  _______________________________          |
|                                                              |
|  -- Candidate Number (10 columns × 10 rows of digits 0-9) -- |
|  y ≈ 210-360, x ≈ 130-380   "Candidate Number" caption       |
|                                                              |
|  -- Answer Grid (2 columns, 13 + 12 questions) ------------  |
|  Col 1 (x ≈ 70-220):  q1-q13                                 |
|  Col 2 (x ≈ 290-440): q14-q25                                |
|  y range: 400-700                                            |
|                                                              |
|  [C] ArUco BL (14,714)                 [D] ArUco BR (501,714)|
+--------------------------------------------------------------+
```

### Field-by-field

#### 5.1 Title + instructions strip

- y ≈ 40, centered: `"OMR Answer Sheet — 25 Questions"`
- y ≈ 65, smaller font: `"Fill each bubble completely with a #2 pencil. Mark only one answer per question. Stray marks may be read as answers."`

These are static — printed at PNG generation time; not stored as template fields.

#### 5.2 Header text strips (4 lines)

Plain text fields drawn by the prefill pipeline (no bubbles), positioned to
match how `prefill_only_package/prefill_answer_sheet_final.py` already paints
the student name / school / exam labels.

| Field | Y (OMR space) | X range | Print font (200 DPI) |
| --- | --- | --- | --- |
| School Name | 95 | 70 – 460 | 16 pt sans-serif |
| Exam Name | 120 | 70 – 460 | 16 pt sans-serif |
| **Region (new)** | 145 | 70 – 460 | 16 pt sans-serif |
| Student Name | 170 | 70 – 460 | 16 pt sans-serif |

> The Region field is **text only** in v1 — the optional CSV `region` column
> is printed onto each sheet for human reference but is not yet bubblable.
> A future iteration could add a bubblable region field (e.g. 10 region codes
> 0-9) if a fixed region list is decided.

#### 5.3 Candidate number bubble grid

10 columns × 10 rows = 100 bubbles. Same semantics as the landscape sheet
(`fieldType: QTYPE_INT`).

| Property | Value |
| --- | --- |
| Block origin (top-left of the `0` row) | `[133, 210]` |
| `bubblesGap` (between columns) | 25 px |
| `labelsGap` (between digit rows) | 14.5 px |
| `fieldLabels` | `["cand1..10"]` |

Columns laid out **vertically** as in landscape, but the grid sits in the
upper-middle of the portrait page rather than the top-right.

#### 5.4 Answer grid (25 × 4)

5 blocks of 5 questions arranged **vertically** in two **columns**:

| Block | First question | Column | Origin (OMR space) |
| --- | --- | --- | --- |
| q01block | q1 | left | `[85, 400]` |
| q06block | q6 | left | `[85, 510]` (+110 from q1) |
| q11block | q11 | left | `[85, 620]` (+110) — overflows into right column for q14+ |
| q14block | q14 | right | `[305, 400]` |
| q19block | q19 | right | `[305, 510]` |
| q24block | q24 | right | `[305, 620]` (only 2 questions in this block) |

Wait — that's awkward. Let me restate as **two columns of equal-height blocks**:

- **Left column (q1 – q13):** 13 questions, vertical spacing 22 px → 286 px tall.
- **Right column (q14 – q25):** 12 questions, vertical spacing 22 px → 264 px tall.

So each *column* is one big "block" rather than five small ones. In OMR template
form that's two fieldBlocks:

| Block | Questions | Origin (top-left of q1-A / q14-A bubble) | `bubblesGap` | `labelsGap` | `fieldLabels` |
| --- | --- | --- | --- | --- | --- |
| `q01_q13_block` | q1 – q13 | `[85, 400]` | 22 | 22 | `["q1..13"]` |
| `q14_q25_block` | q14 – q25 | `[305, 400]` | 22 | 22 | `["q14..25"]` |

Each block has `fieldType: QTYPE_MCQ4` (same as landscape).

> If real-world calibration shows the engine prefers smaller blocks (e.g. it
> tolerates row-jitter per-block better than full-column), we can split each
> column back into two or three blocks. v1 starts with the simpler two-block
> shape because fewer blocks = fewer calibration parameters to tune.

---

## 6. Drawing the printed sheet

`generate_blank.py` (in this folder) renders the printable PNG using PIL:

1. Start with a white 1654 × 2339 RGBA canvas (A4 portrait at 200 DPI).
2. Project the OMR-space layout onto the print canvas with a uniform scale
   factor of `1654 / 515 ≈ 3.213`.
3. Draw the 4 ArUco markers from `cv2.aruco.generateImageMarker` at their
   computed corner positions.
4. Draw the title + instructions text using `ImageDraw`.
5. Draw the four header text strips with labels.
6. Draw the 100 candidate-number bubble outlines + digit labels.
7. Draw the 100 answer bubble outlines + letter labels (A/B/C/D) inside.
8. Draw a thin border between the two answer columns to delineate them.
9. Save to `reference/blank_portrait_25q.png` (PNG, compress_level=1).

The script is self-contained; it only depends on PIL/Pillow and `opencv-python`
(both already in `requirements.txt`).

---

## 7. Integration plan with existing pipeline

| Layer | Change |
| --- | --- |
| OMR engine | None required — `CropOnMarkers` + `core.py` already work with any `pageDimensions` + ArUco-corner template. |
| `prefill_only_package` | New module `portrait_prefill.py` analogous to `prefill_answer_sheet_final.py` but with the portrait constants. v1 keeps `prefill_only_package/` as the *landscape* implementation; the portrait version goes here under `portrait_25q/`. |
| `webui/services/prefill.py` | Add optional `template_id` parameter (`"landscape"` (default) or `"portrait"`) that switches between the two prefill modules and stamped-template caches. |
| `webui/services/student_fill.py` | Add a `portrait_25q` answer-block constant set parallel to the existing `_ANSWER_BLOCKS`. Drawing logic is unchanged. |
| `webui/api.py` | Accept `template_id` form param on `/api/v1/prefill/{single,batch}`. Validate it as an enum. |
| `webui/templates/prefill.html` + `.js` | Add a "Template" dropdown to pick landscape vs portrait. |
| Tests | New `webui/tests/test_portrait_25q_*.py` mirroring existing student-fill and OMR roundtrip tests. |

This wiring is **out of scope for v1** — this folder is the research +
foundational artifacts only. Wiring it into the live web UI is the next iteration.

---

## 8. Open calibration questions

These need to be answered empirically after `generate_blank.py` is run and the
first physical prints come back through a scanner:

1. Does `labelsGap = 22` produce enough vertical room for messy handwriting
   spillover, or do we need 25 – 28?
2. Is the title + instructions strip pulling student attention away from the
   header text fields, or does it help?
3. Does the two-column 13/12 layout cause confusion when students mistakenly
   fill the wrong column's bubble for q1 vs q14? (Solution: print column labels
   "Q1 – Q13" / "Q14 – Q25" at the head of each column.)
4. Does Region appear in time on the printed sheet for invigilators to verify
   it's correct? (Solution: bold typeface for Region.)
5. What is the bubble fill threshold (`minMarked`) for OMR detection at this
   geometry? Likely close to the landscape sheet's value, but should be
   re-measured.

---

## 9. References

- OMRChecker upstream wiki — bubble geometry conventions:
  <https://github.com/Udayraj123/OMRChecker/wiki>
- OpenCV ArUco detector parameter tuning notes:
  <https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html>
- ZipGrade default sheet layouts (real-world reference):
  <https://www.zipgrade.com/help/answer-sheets/>
- This repo's existing audit + calibration write-up:
  [`docs/audit_report_20260524.md`](../docs/audit_report_20260524.md)
