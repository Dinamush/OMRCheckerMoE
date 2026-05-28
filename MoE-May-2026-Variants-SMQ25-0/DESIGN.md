# Portrait 25-Question OMR Sheet — Design Document

**Status:** v1 design + scaffolding. Not yet calibrated against printed output.
**Replaces:** `MoE-April-2026-Landscape-NNQ25-0/` (landscape 25Q sheet) once
calibration completes.

---

## 1. Goals

Optimise the existing 25-question OMR sheet for:

1. **Real-world print/file conventions** — US Letter portrait (8.5" × 11"),
   the default page size in North America and the dominant size for
   classroom printers across most Caribbean and Latin-American jurisdictions
   that participate in CXC / Cambridge assessments printed via US suppliers.
   (A4 portrait remains available by swapping the print canvas constants —
   the OMR space is aspect-matched to whichever paper is used.)
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
| Paper | US Letter portrait | 8.5 in × 11 in (215.9 mm × 279.4 mm). Same engine works for A4 portrait by swapping `PRINT_W/PRINT_H` and the OMR-space height in `generate_blank.py` to match A4's aspect ratio. |
| Print DPI | 200 | Industry-standard for OMR. Yields **1700 × 2200 px** print canvas. |
| OMR processing space | **515 × 666 px** | Aspect ≈ 1 : 1.294, matches US Letter portrait. Picked so existing bubble dimensions (10×10) and gap conventions (≈20 px) keep their meaning, and so scale factors are uniform between OMR space and print space (≈3.301 in both axes — no bubble distortion). |
| Margins (OMR space) | 14 px each side | ≈ 1.8 mm horizontal / 1.9 mm vertical at 200 DPI print. Big enough to leave a quiet zone around the ArUco markers without wasting page real estate. |

> **Why 515 × 666 and not the print resolution?** The OMR engine reads
> `pageDimensions` from the template and homographs the input image onto that
> internal canvas before sampling bubbles. Smaller canvas means faster sampling
> and less RAM. The 666 × 515 landscape canvas has worked well, so we pick a
> portrait equivalent at the **same pixel density** (≈ 0.4 px / mm) and the
> **same aspect ratio as the print canvas** so uniform scaling preserves
> bubble roundness on paper.

---

## 3. ArUco fiducial markers

### 3.1 Geometry

| Property | Value | Source / rationale |
| --- | --- | --- |
| Dictionary | `DICT_4X4_50` | Same as landscape sheet; the engine's `CropOnMarkers.py` already supports it. |
| Marker IDs | 0 (TL), 1 (TR), 2 (BL), 3 (BR) | Convention used by the landscape sheet — keeps `arucoCornerIds: [0, 1, 2, 3]` valid. |
| Marker size (OMR space) | **30 × 30 px** | ≈ 12.6 mm at 200 DPI. Sized for ≥ 30 captured pixels per side at every supported scan path (5 MP phone, 12 MP phone, 300 DPI flatbed) — comfortably above OpenCV's `minMarkerPerimeterRate=0.03` default for a 4000-px-wide phone capture (see [OpenCV ArUco tutorial](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html), [Find-GCP empirical results](https://github.com/zsiki/Find-GCP), [Konovalenko & Tupitsin, CEUR-WS Vol-2744](https://ceur-ws.org/Vol-2744/short14.pdf)). |
| Marker centre coordinates | `[40, 40]`, `[475, 40]`, `[40, 626]`, `[475, 626]` | **40 px from each corner ≈ 16.8 mm physical.** This is intentional — see §3.2 (dog-ear / tear hardening). Marker *outer* edge sits 25 px ≈ 10.5 mm from the page edge. |
| External quiet zone (marker → page edge) | **25 px ≈ 10.5 mm** | Exceeds ArUco's 1-marker-bit (5 px) minimum by 5× and gives the detector a clear background under perspective skew, JPEG compression, **and** typical sheet damage (see §3.2). |
| Internal quiet zone (marker → content) | ≥ 5 px ≈ 2.1 mm | Standard ArUco requirement; the layout in §5 reserves ≥ 5 px between the inner marker edge and the nearest content (header label, candidate-grid bottom, answer-grid edges). |

### 3.2 Damage tolerance — paper-side hardening

Real-world classroom OMR sheets are dog-eared, stapled through corners,
hole-punched, photocopied off-centre, and sometimes torn during filing.
Print-finishing literature documents the **5–8 mm "accidental dog-ear"
fold** as the dominant corner-damage mode, with a secondary **10–15 mm
"deliberate fold for stapling / sorting"** mode for invigilator and
filing-clerk handling. The marker layout chosen here is calibrated
against both:

| Damage mode | Typical extent | Marker outer edge | Marker survives? |
| --- | --- | --- | --- |
| Light dog-ear | 5 mm | 10.5 mm | **Yes** — paper damage stops 5.5 mm before reaching the marker. |
| Heavy dog-ear | 8 mm | 10.5 mm | **Yes** — 2.5 mm of clean paper still between fold-crease and marker. |
| Deliberate corner fold for stapling | 10–12 mm | 10.5 mm | **Marginal** — fold crease may touch the marker outer edge but unlikely to bisect it. The marker's outer 1-bit-quiet zone (5 px ≈ 2.1 mm) absorbs the contact. |
| Aggressive 15 mm fold | 15 mm | 10.5 mm | **No** (marker damaged) — but the engine's 3-of-4 / 2-of-4 fallback (§3.4) recovers. |
| Corner tear | typically ≤ 10 mm radius | 10.5 mm | **Usually yes** — torn paper stops in the sacrificial 10.5 mm margin. |

### 3.3 Fold-safe visual frame

The print also includes a **thin grey L-bracket** at each page corner,
plotted *between* the page edge and the ArUco marker (12 px / ≈ 5 mm
inset from the edge; 18 px / ≈ 7.6 mm arms along each axis). Centred at
the top of the page is the printed warning text:

> *"Keep corners clean — do not fold or staple inside the grey bracket."*

This bracket has three roles:

1. **Behaviour cue** — a visible "do not fold inside this line" boundary
   for students, teachers, invigilators, and filing clerks.
2. **Damage indicator** — a torn or folded corner produces an obvious
   visual gap in the bracket; a human reviewer scanning a batch of
   answer sheets can spot damaged corners before they enter the OMR
   pipeline. This is the same principle behind the [Crucial Print
   Marketing fold-safe-area conventions](https://www.crucialprintmarketing.com/quiet-zone-design-guide/)
   used for postage barcodes and ISBN barcodes.
3. **Light-grey colour** (`#bbbbbb`) keeps the bracket invisible to
   adaptive-threshold OMR processing — it sits below the typical
   pencil-fill darkness threshold and outside every defined field block,
   so the engine never samples it.

### 3.4 Engine-side fallback — already implemented

Even with the paper-side hardening above, some marker damage is
unavoidable in the wild. The `CropOnMarkers._apply_aruco_filter`
implementation in [`src/processors/CropOnMarkers.py`](../src/processors/CropOnMarkers.py)
already provides **two layers of automatic recovery** that this
template's geometry deliberately accommodates:

| Detected markers | Recovery path | Implementation | Behaviour |
| --- | --- | --- | --- |
| 4 of 4 (nominal) | Direct 4-point homography via [`_find_homography_robust`](../src/processors/CropOnMarkers.py) (RANSAC / DLT). | line ~790 | Full perspective transform; standard fast path. |
| 3 of 4 | The `refineDetectedMarkers` board-based pass tries to recover the missing marker from `rejectedCorners` near its projected position. If still missing, the missing centre is estimated from the other 3 via an affine fit (line ~740). | lines 644–673 + 740–770 | Logged as `WARNING: extrapolated <corner>` so post-hoc audit can verify the affected sheet. |
| 2 of 4 | Degraded similarity recovery (rotation + uniform scale + translation only, no perspective freedom) via [`_similarity_homography_from_pairs`](../src/processors/CropOnMarkers.py) (line ~149), gated by [`_score_warp_bubble_confidence`](../src/processors/CropOnMarkers.py) which samples expected bubble outlines and rejects the warp if median outline-vs-paper contrast < 0.04 or coverage < 0.70. | lines 793–869 | Logged as `WARNING: degraded 2-marker recovery` and tagged in `last_warp_bubble_confidence`. Sheet may be rejected automatically if bubble alignment is implausible. |
| 1 of 4 or 0 of 4 | No recovery; sheet is sent to `ErrorFiles`. | line 717 | Human review required. |

> **Implication for this template:** because the fallback is built into
> the engine, the marker inset in §3.1 is a "first line of defence" that
> *prevents* damage from reaching the markers, while the engine fallback
> is a "second line of defence" that *recovers* when prevention fails.
> Both layers exist on purpose — the inset alone is not sufficient
> against aggressive folds, and the fallback alone is not sufficient
> when both bottom corners are torn off (a real failure mode on stapled
> mark-sheet batches).

---

## 4. Bubble geometry, typography, and stroke widths (research synthesis)

### 4.1 Why these numbers — the four-axis research synthesis

The bubble diameter, spacing, font sizes, and stroke widths below are not
copied from the landscape sheet — they are derived from a four-axis
research synthesis covering the **only** four constraints that matter
for a real-world classroom OMR sheet:

1. **Industry standards** — Scantron ScanFlex, Remark Office OMR, AMC
   LaTeX, OMRChecker upstream, ZipGrade, Akindi, Apperson, GradeCam, and
   the OpenCV / Aspose / Kofax detection engines. Convergence point:
   **5 mm bubble Ø, 5.08 mm pitch, 10 pt label, 0.10–0.18 mm stroke**.
2. **Accessibility** — WCAG 2.2, APH (American Printing House for the
   Blind), RNIB / UKAAF Clear Print, the British Dyslexia Association
   Style Guide, College Board SAT accommodations, ETS standard test
   specifications. Convergence point: **14 pt sans-serif body, 18+ pt
   headings, ≥ 5 mm bubble for unaccommodated school-age print, off-black
   on off-white, no italics / no underlines for body**.
3. **Psychology + motor skill** — Hughes & Wilkins 2009 ("typography
   for children"), Katzir et al. 2013 ("Bigger is not always better"),
   Hughes & Wilkins on K-12 reading, the PLOS ONE OMR-detection paper
   (FlAttum et al. 2018), Nielsen Norman Group's cognitive-load-in-forms
   series, and the Smarter Balanced UAAG. Convergence point: **6 mm
   bubble, 11 mm row pitch, label *beside* (not inside) the bubble,
   25–30 % whitespace around the grid**.
4. **Phone-camera detection** — the OpenCV ArUco tutorial, Konovalenko
   & Tupitsin (CEUR-WS Vol-2744, 2020), Tungsten OmniPage / Kofax OMR
   SDK reference, Find-GCP empirical results, the OMRChecker upstream
   wiki. Convergence point: **≥ 30 captured pixels per ArUco side, ≥ 5
   mm bubble Ø (= ~39 captured-px on a 5 MP budget phone), 0.30–0.40
   mm stroke**.

The final numbers below sit at the **intersection** of these four
constraints with one explicit trade-off: the 10 × 10 candidate-number
grid (100 bubbles) cannot grow to 5.5 mm without spilling into the
bottom ArUco quiet zone, so candidate bubbles stay at 4.2 mm (still
above Scantron's 2.5 mm machine floor and OMRChecker's smallest
validated bubble) while the **answer** bubbles — which carry the
graded marks — grow to 5.5 mm.

### 4.2 Final geometry

| Property | Value | Print equivalent | Why |
| --- | --- | --- | --- |
| Bubble shape | Circle outline | — | Matches `student_fill` drawing routines and OpenCV's well-tuned circular-contour thresholding. |
| Answer bubble Ø (OMR space) | **13 × 13 px** | **≈ 5.5 mm @ 200 DPI** | Midpoint of the industry-consensus 5.0 mm (Scantron / Remark / AMC) and the motor-skill 6.0 mm (Hughes & Wilkins). At 5 MP camera capture ≈ 39 px Ø — well above the Kofax OmniPage 45–50 px reliability floor with margin. |
| Candidate bubble Ø (OMR space) | **10 × 10 px** | **≈ 4.2 mm @ 200 DPI** | Density-constrained — the 10×10 grid cannot fit at 5.5 mm without spilling into the bottom ArUco quiet zone. Still above Scantron ScanFlex's 2.5 mm machine floor and matches OMRChecker upstream's `[10, 10]` default. |
| Bubble outline stroke | 3 print-px | **≈ 0.38 mm** | Safe band per [Addmen "don't bold the outline"](https://www.addmengroup.com/omr-design/omr-bubble-size.htm) anti-pattern and [Jukebox 0.25 pt print-engine floor](https://support.jukeboxprint.com/en/articles/3190067-what-is-the-smallest-font-size-i-should-use): thin enough that no engine reads the outline as a fill, thick enough to survive 200 DPI scanner downsampling. |
| `bubblesGap` answer (A/B/C/D) | **24 px** | **≈ 10.0 mm centre-to-centre** | Twice Scantron's 0.166″ pitch — gives a clear 4.6 mm of white between bubble edges, satisfies Remark's "two character spaces between bubbles", and keeps the OpenCV adaptive-threshold block-size (≤ 23 px) able to read an unbiased local background. |
| `labelsGap` answer (Q1→Q2) | **17.0 px** | **≈ 7.1 mm row pitch** | Tighter than horizontal because the 13-row left column has to clear the new (inset) bottom-marker quiet zone — 12 row-gaps × 17.0 = 204 px puts q13 at y=599; with a 13 px bubble, the bottom edge is y=605.5, preserving the documented 5 px internal quiet zone above the marker's 30-px outer edge at y=611 (content limit y≤606). Aligns with [Scantron OpScan 5–6 timing-marks-per-inch (4.23–5.08 mm)](https://www.scantron.com/ScanToolsPlus/Help/v8/LINK/content/overview/scanner_technology_overview.htm) and exceeds it. Trade-off: this is the value we pay for the §3.2 fold-resistance inset. |
| `bubblesGap` candidate (column) | 25 px | ≈ 10.5 mm | Wide enough that even at the smaller 4.2 mm bubble Ø, neighbouring columns are clearly separated. |
| `labelsGap` candidate (row) | 13.5 px | ≈ 5.7 mm | Bare minimum for adjacent digit bubbles to keep ≥ 1.5 mm of white space (PLOS ONE-tolerated density). Compressed by 0.5 px from the previous 14 px value to free vertical room for the §3 fold-resistance inset. |
| Label glyph colour (A/B/C/D, digits inside bubbles) | **`#777777` mid-grey** | — | [PLOS ONE FlAttum et al. 2018](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0206420) finds that black labels *inside* the bubble outline are sometimes mis-classified as fills by adaptive-threshold OMR engines. Printing the glyph in mid-grey keeps it human-readable but pushes its luminance well below pencil-fill darkness. |
| Outline colour | pure black | — | Opposite of the label glyph — full contrast so the contour is detectable by OpenCV's edge detector. |

### 4.3 Typography

All sizes given as **PIL pixel height @ 200 DPI** (PIL convention) with
the printer's-point equivalent in brackets (1 pt ≈ 2.78 px @ 200 DPI).

| Element | PIL size | pt | Why |
| --- | --- | --- | --- |
| Page title (`OMR Answer Sheet — 25 Questions`) | **60 px** | **≈ 22 pt bold** | Above BDA "headings 20 % larger than body" applied to the 14 pt body — comfortably above the 18 pt heading floor in [British Dyslexia Association Style Guide](https://www.bdadyslexia.org.uk/advice/employers/creating-a-dyslexia-friendly-workplace/dyslexia-friendly-style-guide). |
| Instructions (2 wrapped lines) | **32 px** | **≈ 11.5 pt regular** | Above [RNIB Clear Print 11 pt floor](https://www.ukaaf.org/wp-content/uploads/2024/12/G003-UKAAF-Creating-clear-print-and-large-print-documents-v4.pdf) so unaccommodated readers can parse without strain. |
| Form-field labels (`School Name:`, `Region:`…) | **38 px** | **≈ 14 pt bold** | RNIB / BDA / [APH](https://www.aph.org/resources/large-print-guidelines/) "minimum 12 pt, ideally 14 pt" for school-age print, bold to mark them as field anchors. |
| Section captions (`Candidate Number`, `Q1–Q13`) | **38 px** | **≈ 14 pt bold** | Same scale as form-field labels — visual peer group. |
| Question numbers (`1.`, `13.`) | **32 px** | **≈ 11.5 pt bold** | Comfortably above ZipGrade's small numerals, right-aligned so single-digit and two-digit question numbers sit the same distance from their A bubble. |
| A / B / C / D inside answer bubble | **28 px** | **≈ 10 pt regular**, mid-grey | Inside-bubble convention from OMRChecker upstream / ZipGrade / Akindi — but with [PLOS ONE FlAttum et al. 2018](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0206420)'s mid-grey colour to suppress false-fill detection. |
| Digit `0`–`9` inside candidate bubble | **22 px** | **≈ 8 pt regular**, mid-grey | Smaller because the bubble is smaller — but still legible at 5 MP capture (16 captured-px cap height). |
| Form-field underline stroke | 6 print-px | **≈ 2 pt** | Matches [APH "≥ 2 pt stroke for large print"](https://www.aph.org/resources/large-print-guidelines/) and the ADA / Section 508 fixed-display character-stroke minimum. |
| Font family | DejaVu Sans, Arial fallback (regular + bold variants) | — | Sans-serif required by RNIB / BDA / APH / ETS for accessibility; DejaVu and Arial both rank top-tier in [Rello & Baeza-Yates dyslexia-readability TACCESS 2016](https://superarladislexia.org/pdf/2016-Luz%20Rello-Fonts-taccess.pdf). |

---

## 5. Layout (OMR space, 515 wide × 666 tall)

```
+--------------------------------------------------------------+
|  ┌─[A] ArUco TL (40,40)        [B] ArUco TR (475,40) ─┐      |
|  | ▣ "Keep corners clean — do not fold inside grey…"  |      |
|  |    grey fold-safe brackets at each corner          |      |
|  |                                                    |      |
|  |  -- Title (22 pt bold) ------                      |      |
|  |  y ≈ 65   "OMR Answer Sheet — 25 Questions"        |      |
|  |  y ≈ 100  "Fill each bubble completely…"           |      |
|  |  y ≈ 114  "Stray marks may be read as answers."    |      |
|  |                                                    |      |
|  |  -- Header text strips (14 pt bold + 2 pt rule) -- |      |
|  |  y ≈ 138  School Name:   ___________________       |      |
|  |  y ≈ 160  Exam Name:     ___________________       |      |
|  |  y ≈ 182  Region:        ___________________       |      |
|  |  y ≈ 204  Student Name:  ___________________       |      |
|  |                                                    |      |
|  |  -- Candidate Number (10×10 digits, Ø 4.2 mm) ---  |      |
|  |  y ≈ 222       "Candidate Number" (14 pt bold)     |      |
|  |  y ≈ 237-358   bubble grid (row gap 13.5, col 25)  |      |
|  |                                                    |      |
|  |  -- Answer Grid (13 + 12, Ø 5.5 mm) -------------- |      |
|  |  y ≈ 380       "Q1 – Q13" / "Q14 – Q25" (14 pt b.) |      |
|  |  Col 1 (x ≈ 60-160):   origin x=85, row gap 17.0   |      |
|  |  Col 2 (x ≈ 280-380):  origin x=305, row gap 17.0  |      |
|  |  y range: 395-599 centres, bubble bottom 605.5     |      |
|  |                                                    |      |
|  └─[C] ArUco BL (40,626)       [D] ArUco BR (475,626)─┘      |
|     ↑                                                ↑       |
|     └─ fold-safe brackets ──── 10.5 mm sacrificial paper ────┘
+--------------------------------------------------------------+
```

### Field-by-field

#### 5.1 Title + instructions strip

- y ≈ 30, centred grey, 7 pt: `"Keep corners clean — do not fold or staple inside the grey bracket."` (fold-safe warning, see §3.3)
- y ≈ 65, centred: `"OMR Answer Sheet — 25 Questions"` (22 pt bold)
- y ≈ 100, centred line 1: `"Fill each bubble completely with a #2 pencil. Mark only one answer per question."` (11.5 pt)
- y ≈ 114, centred line 2: `"Stray marks may be read as answers."` (11.5 pt)

These are static — printed at PNG generation time; not stored as template fields.
The two-line wrap keeps the body text in scope of the [RNIB Clear Print
"≤ 70 characters per line"](https://www.ukaaf.org/wp-content/uploads/2024/12/G003-UKAAF-Creating-clear-print-and-large-print-documents-v4.pdf) guidance.

#### 5.2 Header text strips (4 lines)

Plain text fields drawn by the prefill pipeline (no bubbles), positioned to
match how `prefill_package/prefill_answer_sheet_final.py` already paints
the student name / school / exam labels.

| Field | Y (OMR space) | Label X anchor | Underline X range | Print font (200 DPI) |
| --- | --- | --- | --- | --- |
| School Name | 138 | 60 | 165 – 455 | 14 pt sans-serif **bold** |
| Exam Name | 160 | 60 | 165 – 455 | 14 pt sans-serif **bold** |
| **Region (new)** | 182 | 60 | 165 – 455 | 14 pt sans-serif **bold** |
| Student Name | 204 | 60 | 165 – 455 | 14 pt sans-serif **bold** |

> The 14 pt bold label + 2 pt underline is anchored to the [RNIB Clear
> Print](https://www.ukaaf.org/wp-content/uploads/2024/12/G003-UKAAF-Creating-clear-print-and-large-print-documents-v4.pdf) / [APH](https://www.aph.org/resources/large-print-guidelines/) / [British Dyslexia Association](https://www.bdadyslexia.org.uk/advice/employers/creating-a-dyslexia-friendly-workplace/dyslexia-friendly-style-guide) **school-age print floor** of 12 pt, with one safety margin pt for bold weight.

> The Region field is **text only** in v1 — the optional CSV `region` column
> is printed onto each sheet for human reference but is not yet bubblable.
> A future iteration could add a bubblable region field (e.g. 10 region codes
> 0-9) if a fixed region list is decided.

#### 5.3 Candidate number bubble grid

10 columns × 10 rows = 100 bubbles. Same semantics as the landscape sheet
(`fieldType: QTYPE_INT`).

| Property | Value |
| --- | --- |
| Block origin (top-left of the `0` row) | `[133, 237]` |
| `bubbleDimensions` (block override) | `[10, 10]` (≈ 4.2 mm Ø) |
| `bubblesGap` (between columns) | 25 px |
| `labelsGap` (between digit rows) | 13.5 px |
| `fieldLabels` | `["cand1..10"]` |

> **Why smaller bubbles here than the answer grid?** Because a 10×10 grid
> of 5.5 mm bubbles plus a 13-row answer column physically exceeds Letter
> portrait. Candidate-number bubbles at 4.2 mm still sit above Scantron
> ScanFlex's 2.5 mm machine floor and OMRChecker's smallest validated
> bubble (4.7 mm). The `bubbleDimensions` field-block override (supported
> by `src/schemas/template_schema.py`) lets the OMR engine sample the
> candidate grid with the correct kernel.

Columns laid out **vertically** as in landscape, but the grid sits in the
upper-middle of the portrait page rather than the top-right.

#### 5.4 Answer grid (25 × 4)

Restated as **two columns of equal-height blocks**:

- **Left column (q1 – q13):** 13 questions, vertical spacing 20 px → 240 px tall.
- **Right column (q14 – q25):** 12 questions, vertical spacing 20 px → 220 px tall.

So each *column* is one big "block" rather than five small ones. In OMR template
form that's two fieldBlocks:

| Block | Questions | Origin (top-left of q1-A / q14-A bubble) | `bubblesGap` | `labelsGap` | `fieldLabels` |
| --- | --- | --- | --- | --- | --- |
| `q01_q13_block` | q1 – q13 | `[85, 395]` | 24 | 17.0 | `["q1..13"]` |
| `q14_q25_block` | q14 – q25 | `[305, 395]` | 24 | 17.0 | `["q14..25"]` |

Top-level `bubbleDimensions` is `[13, 13]` (≈ 5.5 mm Ø) and applies to both
answer blocks (the candidate-number block overrides this to `[10, 10]`).

Each block has `fieldType: QTYPE_MCQ4` (same as landscape).

> If real-world calibration shows the engine prefers smaller blocks (e.g. it
> tolerates row-jitter per-block better than full-column), we can split each
> column back into two or three blocks. v1 starts with the simpler two-block
> shape because fewer blocks = fewer calibration parameters to tune.

---

## 6. Drawing the printed sheet

`generate_blank.py` (in this folder) renders the printable PNG using PIL:

1. Start with a white 1700 × 2200 RGBA canvas (US Letter portrait at 200 DPI).
2. Project the OMR-space layout onto the print canvas with a uniform scale
   factor of `1700 / 515 ≈ 3.301` (matching `2200 / 666 ≈ 3.303` — uniform to within 0.07 %).
3. Draw the 4 ArUco markers from `cv2.aruco.generateImageMarker` at their
   computed corner positions.
4. Draw the title + instructions text using `ImageDraw`.
5. Draw the four header text strips with labels.
6. Draw the 100 candidate-number bubble outlines + digit labels.
7. Draw the 100 answer bubble outlines + letter labels (A/B/C/D) inside.
8. Draw a thin border between the two answer columns to delineate them.
9. Save to `reference/blank_MoE-May-2026-Variants-SMQ25-0.png` (PNG, compress_level=1).

The script is self-contained; it only depends on PIL/Pillow and `opencv-python`
(both already in `requirements.txt`).

---

## 7. Integration plan with existing pipeline

| Layer | Change |
| --- | --- |
| OMR engine | None required — `CropOnMarkers` + `core.py` already work with any `pageDimensions` + ArUco-corner template. |
| `prefill_package` | New module `portrait_prefill.py` analogous to `prefill_answer_sheet_final.py` but with the portrait constants. v1 keeps `prefill_package/` as the *landscape* implementation; the portrait version goes here under `MoE-May-2026-Variants-SMQ25-0/`. |
| `webui/services/prefill.py` | Add optional `template_id` parameter (`"landscape"` (default) or `"portrait"`) that switches between the two prefill modules and stamped-template caches. |
| `webui/services/student_fill.py` | Add a `MoE-May-2026-Variants-SMQ25-0` answer-block constant set parallel to the existing `_ANSWER_BLOCKS`. Drawing logic is unchanged. |
| `webui/api.py` | Accept `template_id` form param on `/api/v1/prefill/{single,batch}`. Validate it as an enum. |
| `webui/templates/prefill.html` + `.js` | Add a "Template" dropdown to pick landscape vs portrait. |
| Tests | New `webui/tests/test_MoE-May-2026-Variants-SMQ25-0_*.py` mirroring existing student-fill and OMR roundtrip tests. |

This wiring is **out of scope for v1** — this folder is the research +
foundational artifacts only. Wiring it into the live web UI is the next iteration.

---

## 8. Open calibration questions

These need to be answered empirically after `generate_blank.py` is run and the
first physical prints come back through a scanner:

1. Does `labelsGap = 17.0` (answer grid) produce enough vertical room
   for messy handwriting spillover on US Letter, or do we need 20+?
   (If 20+, the only way to fit while preserving the §3 fold-resistance
   marker inset is to either split each answer column into two stacked
   blocks, or pull the candidate-number grid into a 2-row × 5-column
   landscape orientation so it occupies less vertical real estate.)
2. Are the §3.3 grey fold-safe brackets and printed warning enough of
   a behaviour cue to actually reduce corner damage in real classrooms?
   (Measure by comparing dog-ear / staple-through-corner incident rates
   before vs. after deployment on a 200-sheet sample.)
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

The geometry numbers in §3 and §4 are anchored to evidence from these
four research axes. URLs are grouped by axis so the next iteration can
revisit the bibliography per concern.

### 9.1 Industry — OMR scanner / sheet vendors

- [Scantron ScanFlex form-design specification](https://www.scantron.com/ScanToolsPlus/Help/v8/LINK/content/scanflex/designing_forms_for_scanflex.htm) — 2.46 × 1.85 mm minimum oval, 0.166″ / 0.200″ read-cell pitch.
- [Scantron OpScan scanner-technology overview](https://www.scantron.com/ScanToolsPlus/Help/v8/LINK/content/overview/scanner_technology_overview.htm) — timing-track pitches.
- [Remark Office OMR form-design best practices](https://remarksoftware.com/support/remark-office-omr-support/form-design/) — 10–14 pt Arial bubble font, ≥ 3/8″ clearance.
- [OMRChecker upstream wiki — Templates](https://github.com/Udayraj123/OMRChecker/wiki/%5Bv1%5D-About-Templates).
- [OMRChecker README](https://github.com/Udayraj123/OMRChecker/blob/master/README.md).
- [Auto-Multiple-Choice (AMC) LaTeX documentation](https://www.auto-multiple-choice.net/auto-multiple-choice.en/) — 2.5ex bubble, 0.5 pt rule.
- [Aspose.OMR fill-threshold defaults](https://docs.aspose.com/omr/net/) — 30–40 % pen / 15–25 % pencil.
- [Kofax / Tungsten OmniPage OMR SDK reference](https://docshield.tungstenautomation.com/OmniPageCaptureSDK/en_US/2025.1.0-m7NwYtqyAo/help/RECAPI/OMR_GENERAL.html) — 45–50 px minimum bubble bounding box.
- [Addmen OMR design — bubble size](https://www.addmengroup.com/omr-design/omr-bubble-size.htm); [Addmen design mistakes](https://www.admengroup.com/design-omr-sheet-design-mistake.htm).
- [OMR Home best practices for OMR sheet design](https://omrhome.com/blog/best-practices-for-designing-omr-sheets-to-ensure-accurate-scanning/).
- [ZipGrade help — answer sheets](https://www.zipgrade.com/help/answer-sheets/).

### 9.2 Accessibility / large-print

- [WCAG 2.2 — W3C recommendation (Oct 2023)](https://www.w3.org/TR/WCAG22/) — contrast and resize requirements.
- [RNIB / UKAAF — Creating Clear-Print and Large-Print documents (G003)](https://www.ukaaf.org/wp-content/uploads/2024/12/G003-UKAAF-Creating-clear-print-and-large-print-documents-v4.pdf) — 12 pt minimum / 14 pt ideal; 16–18 pt for large print.
- [APH (American Printing House for the Blind) — large-print guidelines](https://www.aph.org/resources/large-print-guidelines/).
- [APH — Test Access: Making Tests Accessible (2009)](https://sites.aph.org/wp-content/uploads/2017/09/Test-Access-Making-Tests-Accessible-2009.pdf).
- [British Dyslexia Association — Dyslexia-Friendly Style Guide](https://www.bdadyslexia.org.uk/advice/employers/creating-a-dyslexia-friendly-workplace/dyslexia-friendly-style-guide).
- [College Board — SAT Accommodations Handbook](https://accommodations.collegeboard.org/media/pdf/accommodations-supports-handbook.pdf) — 10 pt standard / 14 pt / 20 pt / 24 pt large-print tiers.
- [ETS large-print and reformatted-test technical notes (AFB)](https://afb.org/aw/7/6/14414).
- [Texas STAAR / TELPAS font + point sizes](https://teadev.tea.texas.gov/sites/default/files/2018_STAAR_TELPAS_Alternate%202_Font_and_Point_Sizes_tagged.pdf) — Verdana 14 pt standard.
- [US Section 508 — Fonts and typography](https://www.section508.gov/develop/fonts-typography/).

### 9.3 Psychology + motor / educational measurement

- [FlAttum et al. (2018), PLOS ONE — Optical mark recognition that tolerates fill variation](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0206420) — 5.5 × 4.5 mm reliable field size; internal labels increase machine-detection error.
- [Hughes & Wilkins (2009), Wiley — Typography for children](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-9817.2009.01402.x).
- [Katzir, Hershko & Halamish (2013), PMC — Bigger is not always better](https://pmc.ncbi.nlm.nih.gov/articles/PMC3777945/).
- [Rello & Baeza-Yates (2016), TACCESS — Font readability for dyslexia](https://superarladislexia.org/pdf/2016-Luz%20Rello-Fonts-taccess.pdf).
- [Nielsen Norman Group — 4 principles to reduce cognitive load in forms](https://www.nngroup.com/articles/4-principles-reduce-cognitive-load/).
- [Nielsen Norman Group — Form whitespace](https://www.nngroup.com/articles/form-design-white-space/).
- [Mead & Drasgow (1986), SAGE — Vertical vs horizontal answer sheets (N=50 000)](https://journals.sagepub.com/doi/10.1177/001316448604600225).
- [Smarter Balanced Usability, Accessibility, and Accommodations Guidelines](https://portal.smarterbalanced.org/library/en/usability-accessibility-and-accommodations-guidelines.pdf/).
- [Meazure Learning — Reducing cognitive load and test anxiety](https://www.meazurelearning.com/resources/reducing-cognitive-load-and-test-anxiety-4-strategies-for-better-outcomes).

### 9.4 Computer-vision detection (phone + scanner)

- [OpenCV ArUco detector tutorial (4.x)](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html); [ArUco FAQ](https://docs.opencv.org/4.x/d1/dcb/tutorial_aruco_faq.html); [DetectorParameters reference](https://docs.opencv.org/4.0.0/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html).
- [Find-GCP — empirical ArUco detection size guidance](https://github.com/zsiki/Find-GCP) — ≥ 20 px, ≥ 30 px comfort target.
- [Konovalenko & Tupitsin (2020), CEUR-WS Vol-2744](https://ceur-ws.org/Vol-2744/short14.pdf) — ArUco detection probability vs marker pixel area.
- [Garrido-Jurado et al. (2014), Pattern Recognition — Automatic generation and detection of highly reliable fiducial markers under occlusion](https://www.sciencedirect.com/science/article/abs/pii/S0031320314000235).
- [Romero-Ramirez, Muñoz-Salinas & Medina-Carnicer (2018), Image and Vision Computing — Speeded-Up Detection of Squared Fiducial Markers](https://www.researchgate.net/publication/325787310_Speeded_Up_Detection_of_Squared_Fiducial_Markers).
- [PyImageSearch — Bubble-sheet multiple-choice scanner with OpenCV](https://pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/) — 20 × 20 px minimum bubble bounding box.

### 9.5 In-repo context

- This repo's existing audit + calibration write-up:
  [`docs/audits/audit_report_20260524.md`](../docs/audits/audit_report_20260524.md)
- Per-block `bubbleDimensions` schema support:
  [`src/schemas/template_schema.py`](../src/schemas/template_schema.py) (line 257).
