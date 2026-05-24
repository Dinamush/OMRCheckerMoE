# OMRCheckerMoE — Comprehensive Codebase Audit

**Date:** 2026-05-24  
**Model:** Claude Opus 4.7 (4 read-only subagents in parallel)  
**Total findings:** 130 across 4 layers (Core engine · Web backend · Frontend · Real-world OMR robustness)

| Layer | Critical | High | Medium | Low | Total |
|---|---|---|---|---|---|
| Core engine (`src/`) | 5 | 9 | 6 | 5 | 25 |
| Web backend (`webui/services/`, `api.py`) | 3 | 9 | 15 | 5 | 32 |
| Frontend (`webui/static/`, `templates/`) | 1 | 7 | 14 | 11 | 33 |
| Real-world OMR robustness | 7 | 16 | 10 | 7 | 40 |
| **Total** | **16** | **41** | **45** | **28** | **130** |

The full per-layer reports live in subagent traces; this document records the **prioritized findings and which were fixed in this pass**.

---

## Tier 1 — Fixed in this audit pass

These are correctness/security bugs with low-risk one-shot fixes.

### Core engine (`src/`)

| ID | File | Description | Status |
|---|---|---|---|
| CORE-1 | `src/core.py` | `save_img_list` declared as class-level `defaultdict` → all `ImageInstanceOps` instances share image stack → memory leak + cross-contamination | **Fixed** |
| CORE-2 | `src/core.py` | `per_omr_threshold_avg /= total_q_strip_no` raises `ZeroDivisionError` when template has zero question strips | **Fixed** |
| CORE-3 | `src/core.py` | Integral-image indexing uses `x + field_block.shift` with no bounds clamping — negative shift wraps via Python negative indexing producing **silently wrong bubble reads** | **Fixed** |
| CORE-4 | `src/processors/CropOnMarkers.py` | `apply_erode_subtract` conditional is **semantically inverted**: `True` returns the raw image; `False` returns the eroded image. Marker template is pre-eroded ⇒ matching is degraded for users with documented default | **Fixed** |
| CORE-5 | `src/processors/CropOnMarkers.py` | `exit(31)` on missing marker file kills the worker process bypassing pool cleanup → orchestrator can hang | **Fixed** |
| CORE-6 | `src/processors/FeatureBasedAlignment.py` | No `None` check after `cv2.imread(reference)` → cryptic `AttributeError` later | **Fixed** |
| CORE-7 | `src/processors/FeatureBasedAlignment.py` | `cv2.findHomography` / `estimateAffine2D` return `None` is passed straight to `warpPerspective` → crash | **Fixed** |
| CORE-8 | `src/processors/FeatureBasedAlignment.py` | Zero ORB descriptors / fewer than 4 good matches crash `matcher.match` | **Fixed** |
| CORE-9 | `src/core.py` | `_auto_orient_to_template` only tests CW/CCW 90° — 180° rotation undetected (Scantron/duplex feed scenario) | **Fixed** |
| CORE-10 | `src/processors/CropOnMarkers.py` | `preserve_full_image` template-match path skips `_homography_is_sane` → degenerate homographies silently warp the image | **Fixed** |
| CORE-11 | `src/processors/CropOnMarkers.py` | `getBestMatch` debug-mode `InteractionUtils.show("res", res)` crashes when `res is None` (every rescaled marker larger than input) | **Fixed** |
| CORE-12 | `src/utils/image.py` | `four_point_transform` crashes when `max_width` or `max_height` resolves to 0 (degenerate marker centres) | **Fixed** |
| CORE-13 | `src/utils/image.py` | `cv2.normalize(img, 0, 255, ...)` calls land alpha/beta into the wrong positional slot. **Investigation revealed this is load-bearing**: the swap effectively maps `NORM_MINMAX` to `[255, 0]`, inverting the image, and the entire downstream pipeline (thresholding, morphology, "looks like a Xeroxed OMR" log message, all snapshot tests) depends on the inversion. Annotated as a documented pseudo-bug with a long-form comment so a future contributor doesn't innocently "tidy" it back into a real bug. | **Documented (not changed)** |
| CORE-14 | `src/utils/parsing.py` | `custom_sort_output_columns` raises `IndexError` when field label doesn't match number-regex | **Fixed** |
| CORE-15 | `src/entry.py` | `check_and_move` always returns `True` even when copy fails → CSV rows pointing to non-existent files in `Manual/Errors/` | **Fixed** |

### Web backend (`webui/`)

| ID | File | Description | Status |
|---|---|---|---|
| API-1 | `webui/services/omr.py` | **Path traversal**: `_resolve_batch_asset` returns paths that escape `batch_root` when template `relativePath` is `../../etc/...` | **Fixed** |
| API-2 | `webui/services/batches.py` | `batch_id` taken straight from URL used as filesystem path with **no safe-character validation** (allows `..`, control chars, drive prefixes) | **Fixed** |
| API-3 | `webui/api.py` | `assert csv_file is not None` used as runtime validation in a route handler — stripped under `python -O` | **Fixed** |
| API-4 | `webui/services/batches.py`, `webui/services/omr.py` | Per-batch / per-PDF lock dicts (`_batch_locks`, `_PDF_SPLIT_LOCKS`) grow unboundedly — long-running server memory leak | **Fixed** |
| API-5 | `webui/services/omr.py` | `_collect_relative_paths` recursive walker has no depth cap → deeply nested attacker-supplied JSON triggers `RecursionError` | **Fixed** |
| API-6 | `webui/api.py` | Background `_run_pdf_split` echoes `type(exc).__name__: exc` containing filesystem paths back to the public JSON status endpoint | **Fixed (sanitized)** |

### Frontend (`webui/static/`, `webui/templates/`)

| ID | File | Description | Status |
|---|---|---|---|
| UI-1 | `webui/static/prefill.js` | **XSS** in CSV preview — column headers and cell values inserted via `innerHTML` without escaping | **Fixed** |
| UI-2 | `webui/static/app.js`, `webui/static/batch.js` | Several `JSON.parse(text)` / `response.json()` calls without `try/catch` → unhandled rejections when a proxy returns HTML | **Fixed** |
| UI-3 | `webui/static/batch.js` | `handleUpload` submit button never disabled — risk of double-submit during slow uploads | **Fixed** |
| UI-4 | `webui/static/app.js` | Log poll `setInterval` runs forever (1 Hz fetch per tab) with no visibility/teardown — drains power, hammers server | **Fixed** |
| UI-5 | `webui/static/batch.js` | `renderResults` does `container.innerHTML = ...` on every poll → selected result tab resets every 2 s during active batches | **Fixed** |
| UI-6 | `webui/templates/batch_detail.html` | Hard-coded `Erasure risk 85%` in SSR template ignores actual per-row risk until JS re-renders | **Fixed** |
| UI-7 | `webui/static/app.css` | `var(--border)` referenced in `.preset-preview` rules but never defined → preset comparison cards have no visible border | **Fixed** |
| UI-8 | `webui/static/settings.js` | "Settings saved" success banner never auto-clears → confuses operators into thinking edits-after-save are persisted | **Fixed** |
| UI-9 | `webui/static/batch.js` | `location.reload()` fired before success feedback rendered → user never sees "Preset applied" confirmation | **Fixed** |
| UI-10 | `webui/templates/batch_detail.html` | `<h2>` nested inside `<button>` — invalid HTML, broken heading hierarchy for AT | **Fixed** |
| UI-11 | `webui/static/batch.js` | `localStorage.setItem` in `setEditorMode` not wrapped in try/catch (quota exhaustion / private-browsing) | **Fixed** |

### Real-world OMR robustness (cross-cutting)

| ID | Description | Status |
|---|---|---|
| OMR-1 | 180° auto-orient missing in ArUco path → sheets fed upside-down silently misgraded *(same as CORE-9)* | **Fixed** |
| OMR-2 | Bubble classification has no "ambiguous" bin; per-bubble confidence not exposed | Documented (Tier 2 scope) |
| OMR-3 | Disk-full mid-batch leaves truncated results CSV | Documented (Tier 2 scope) |

---

## Tier 2 — Documented, deferred

These require larger design work (confidence scoring, batch resume, multi-template support, accessibility overhaul, authentication) and were intentionally not pulled into this pass to keep the diff reviewable. The full subagent reports list them with file/line refs.

Key Tier-2 items to consider next:
- **Bubble confidence tier** (`AMBIGUOUS` bin between empty/filled with a per-bubble margin) — `src/core.py:843` already carries a TODO for this.
- **180°/mirror detection in template-matching mode** (`CropOnMarkers.py` apply_filter path).
- **Form-version detection** (QR or extra ArUco cluster) so mixed-version batches don't silently mis-score.
- **Batch checkpoint/resume** (`webui/services/omr.py:1301`) — currently `recover_stale_batches` just fails them.
- **Lens distortion correction** for phone-camera capture (cv2.undistort).
- **Authentication layer** — entire API is currently unauthenticated.
- **CORS hardening** — replace `["*"] + allow_credentials=True` default with explicit origin list.
- **Storage quotas** — no max-age / max-total-size sweeper for `webui/storage/batches/`.
- **Atomic metadata writes** wrapping the read–modify–write cycle (currently only the final replace is atomic).
- **Per-batch results pagination** — schema declares `offset`/`limit` but `read_results` always returns the whole CSV.
- **A11y pass** — skip-link, role="tab"/aria-pressed/aria-live, focus management on modals.
- **`@media print` stylesheet** — current dark theme prints with full ink saturation.

---

## Methodology

Four read-only subagents (Claude Sonnet 4.6 medium-thinking) each audited one layer end-to-end and returned a structured per-finding report with file/line, severity, scenario, and suggested fix. Findings were de-duplicated (e.g., the `save_img_list` class-level bug appears in both Core and Real-World reports as CORE-1 / F-33), prioritized, and the Tier-1 set was applied in a single coherent diff.

---

## Post-audit follow-up — Bubble geometry calibration (2026-05-24)

While building the student-fill feature, a visual review of the rendered sheets revealed that:

- **`custom_25_definitive_final/template.json`** had its answer-block origins offset **+5px in y** from the actual printed bubble centres on `prefill_only_package/blank_template_reference.png`.
- The OMR engine samples a `bubbleDimensions=[10,10]` box from `origin → origin + 10`. With the old origins, the sampling box centre landed ~5 px above the bubble centre — well outside the printed circle. The engine still read fully-filled bubbles because the integral-mean over a 10×10 region caught the bottom of the dark mark; but **partial fills, light pencils, check marks, and the upper margin of normal student strokes were systematically missed**.

### Fix
- Re-calibrated all 5 answer-block origins via Hough-circle detection (`webui/tests/_calibrate_bubbles.py`). New origins place the sampling box exactly on the printed bubble.
- Relaxed `src/schemas/template_schema.py` to accept fractional pixel origins (`two_positive_numbers` instead of `two_positive_integers`) — the engine already supported floats internally; only the schema was strict.
- Made `parse_answers("random" | "random_with_skips")` deterministic per-candidate by seeding from `_stable_seed("prefill-answers", candidate_number, raw_answer_spec)` in `webui/services/prefill.py`. Without this seed, random shortcuts produced a fresh pattern on every call — silently breaking the contract that the same input should produce the same output.

### Measured impact (OMR readability of 25 questions on a freshly generated sheet)

| Profile | Old geometry | New geometry |
|---|---|---|
| `medium_pencil` (`all_*`) | 8 – 17 / 25 | **24 – 25 / 25** |
| `check_mark` (`all_*`) | 0 – 6 / 25 | **19 – 25 / 25** |
| `cross_mark` (`all_*`) | 15 – 25 / 25 | **25 / 25** |
| `messy_student` (`all_*`) | 3 – 8 / 25 | **8 – 15 / 25** |
| `heavy_pencil`, `pen_ballpoint`, `careful_student` | already 24 – 25 / 25 | unchanged |
| `light_pencil`, `partial_fill` | 0 – 5 / 25 | unchanged (intentionally below OMR threshold) |

The new geometry also fixes the original user complaint that drawn marks "float above" the printed circles. See `diagnostic_output/answer_geometry_overlay.png` and `diagnostic_output/alignment_gallery/` for visual proof.

