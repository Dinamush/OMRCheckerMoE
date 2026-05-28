# Student Bubble Fill — Feature Design

**Date:** 2026-05-24
**Author:** Claude Opus 4.7 (Cursor agent)
**Status:** approved-by-implementation

## Why

Today, the prefill pipeline stamps only candidate metadata (name, school, exam,
candidate number) and the 10 candidate-number bubbles. The 100 answer bubbles
(q1–q25 × A–D) are left blank because real students fill those by hand. That
makes it impossible to:

1. Generate synthetic OMR test data that includes filled answers
2. Run end-to-end "prefill → OMR → verify" with known ground truth
3. Stress-test the OMR engine against realistic *student-style* marks (light
   pencil, partial fills, check marks, multi-marks) without a stack of real
   sheets

This feature adds an opt-in "student fill" stage that simulates how a child
would mark the answer bubbles, with several **marking profiles** that vary
darkness, pressure, mark style and consistency.

## Scope

- **Single-sheet** prefill: optional `answers` + `marking_profile` form fields.
- **Batch** prefill: optional CSV `answers` column + global `marking_profile`.
- **Programmatic API** for use in tests and Python scripts.
- The feature is **orthogonal** to the existing `realism_preset` scan
  simulation. Marking profile controls the *student's* hand. Realism preset
  controls the *scanner's* artifacts. Both can be combined.

## Marking profiles

| name              | gray range  | scale range | shift_px | style       | multi% | stray% | use case                         |
|-------------------|-------------|-------------|----------|-------------|--------|--------|----------------------------------|
| `none`            | —           | —           | —        | skip        | 0      | 0      | metadata-only (current behaviour)|
| `light_pencil`    | 140–190     | 0.70–0.90   | 2        | ellipse     | 1      | 1      | tentative, faint marks           |
| `medium_pencil`   | 60–110      | 0.85–1.05   | 2        | ellipse     | 2      | 2      | typical student                  |
| `heavy_pencil`    | 10–45       | 0.95–1.10   | 1        | ellipse     | 1      | 1      | confident, dark marks            |
| `pen_ballpoint`   | 5–30        | 0.92–1.05   | 1        | ellipse     | 0      | 0      | pen marker, crisp                |
| `check_mark`      | 15–60       | —           | 2        | check (✓)   | 0      | 0      | check instead of fill            |
| `cross_mark`      | 15–60       | —           | 2        | cross (✗)   | 0      | 0      | X instead of fill                |
| `partial_fill`    | 60–120      | 0.60–0.85   | 3        | partial arc | 3      | 4      | half-shaded bubbles              |
| `messy_student`   | 30–180      | 0.50–1.15   | 4        | ellipse     | 5      | 8      | erratic, variable                |
| `careful_student` | 15–60       | 0.97–1.05   | 1        | ellipse     | 0      | 1      | neat, full fills                 |

All profiles share the same alpha-blended ellipse drawing primitive
(borrowed from `scan_simulation._draw_imperfect_bubbles`), seeded
deterministically per `(candidate_number, marking_profile, answers)`.

## Answer format

`parse_answers(value, *, num_questions=25)` is permissive:

- `None` / `""` / `"none"` / `"blank"` → all questions blank
- `"random"` → `random.choice("ABCD")` for each question
- `"random_with_skips"` → random with ~10% blanks
- `"all_a"`, `"all_b"`, `"all_c"`, `"all_d"` → uniform
- `"alternating"` → `ABCDABCDA…`
- A dict `{"q1": "A", "q3": "BC", "q5": None}`
- A list `["A", "B", "", "C", …]`
- A 25-character string `"ABCD-ABCD-ABCD-ABCD-ABCDA"` (`-` / `X` / space = skip)
- A JSON encoding of any of the above

Each per-question value can be:
- A single option letter `A`/`B`/`C`/`D` (case-insensitive)
- Multiple letters `AB`, `BCD` (student marked 2+ bubbles)
- Empty/skip marker `""`, `"-"`, `"X"`, `"NR"`, `null`

Returns `dict[int, list[int]]` keyed by 1-indexed question, valued by
sorted 0-indexed option lists. Missing questions are blank.

## Geometry

`answer_bubble_geometry(w, h)` projects the OMR template's `fieldBlocks`
coordinates (in 666×515 space) into the prefill render canvas (typically
1426×1103) by uniform scaling:

```python
scale_x = w / 666.0
scale_y = h / 515.0
for block in (q01block, q06block, q11block, q16block, q21block):
    for q_idx in 0..4:              # 5 questions per block
        for opt_idx in 0..3:        # 4 options per question
            cx = (origin_x + opt_idx * bubblesGap) * scale_x
            cy = (origin_y + q_idx  * labelsGap)  * scale_y
            radius = round(5 * min(scale_x, scale_y))
            yield {q, opt, cx, cy, radius}
```

Because the OMR template and the prefill canvas share the same ArUco
marker reference fractions, this proportional scaling is identical to
the perspective transform that `CropOnMarkers` will later apply when
scanning. Bubbles drawn here will land exactly on the OMR engine's
field-block centres after the round-trip.

## Integration

### Backend

```
webui/services/student_fill.py         (new)
webui/services/prefill.py              (modify _thread_render + generate_*)
webui/api.py                           (extend /prefill/single, /prefill/batch)
```

### Frontend

```
webui/templates/prefill.html           (add "Fill answers" section)
webui/static/prefill.js                (send answers/marking_profile fields)
```

### Tests

```
webui/tests/test_student_fill_unit.py           (geometry, parser, drawing)
webui/tests/test_student_fill_api.py            (API endpoints)
webui/tests/test_student_fill_omr_roundtrip.py  (end-to-end, OMR-readable)
```

## Determinism

The drawing RNG is seeded by `blake2b(answers, marking_profile,
candidate_number, "student-fill")`, so identical inputs produce
byte-identical output even though it looks hand-drawn.

## Safety / robustness rails

- Input validation rejects malformed answer strings with HTTP 422
- The empty-answers fast path is byte-identical to today's output
- The drawing function clamps all coordinates to the image bounds
- Multi-marks per question are capped at all 4 options
- Stray marks always stay within ±2 bubble radii of a real bubble
- Random seed is hashed from inputs so tests are reproducible
