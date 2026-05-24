"""Detect actual answer-bubble centres on the blank reference template.

Uses Hough circle detection on the answer-grid region to find the real
pixel positions of the 100 answer bubbles, then back-projects them into
the 666x515 OMR processing canvas so we can update both the student-fill
geometry constants AND the OMR template.json.

Run from repo root: ``python scripts/calibrate_bubbles.py``
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PNG = REPO_ROOT / "prefill_only_package" / "blank_template_reference.png"
OUT_DIR = REPO_ROOT / "diagnostic_output"
OUT_DIR.mkdir(exist_ok=True)

# OMR space: 666x515. Prefill canvas: 1426x1103.
OMR_W, OMR_H = 666, 515


def detect_bubbles() -> list[tuple[int, int, int]]:
    """Return (cx, cy, r) for each detected bubble on the prefill canvas."""
    img_rgb = np.array(Image.open(TEMPLATE_PNG).convert("RGB"))
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # The answer grid lives in roughly the bottom half: y > 500 in 1103-tall canvas.
    # Crop to that region to keep Hough from being confused by the candidate grid.
    y_top = int(h * 0.45)
    y_bot = int(h * 0.99)
    crop = gray[y_top:y_bot, :]
    blurred = cv2.medianBlur(crop, 3)
    # Bubble outlines are circles ~22-26 px in diameter on the 1426 canvas.
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=18,
        param1=80,
        param2=18,
        minRadius=9,
        maxRadius=15,
    )
    if circles is None:
        raise RuntimeError("No bubbles detected — tune Hough params.")
    detected = []
    for c in circles[0]:
        cx, cy, r = float(c[0]), float(c[1]) + y_top, float(c[2])
        detected.append((int(round(cx)), int(round(cy)), int(round(r))))
    return detected


# Approximate x windows for the 5 horizontal answer blocks on the 1426-wide
# canvas. These come from the existing (off) OMR template ranges plus a
# generous pad, used only to bucket detections per block.
_BLOCK_X_WINDOWS_PREFILL = [
    (90, 290),    # q1-q5
    (370, 570),   # q6-q10
    (640, 850),   # q11-q15
    (910, 1120),  # q16-q20
    (1170, 1380), # q21-q25
]


def cluster_into_grid(detected: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    """Cluster detections into per-question groups by horizontal block.

    Returns a list ordered by question 1..25, each containing 4 (cx, cy, r)
    tuples for options A, B, C, D.

    Approach: for each of the 5 horizontal blocks, take the detections that
    fall in its x window. Sort by y, then merge near-duplicates (same row),
    cluster into 5 rows of 4 bubbles. Robust against per-row jitter.
    """
    questions: list[list[tuple[int, int, int]]] = []
    block_diagnostics: list[str] = []
    for block_idx, (xmin, xmax) in enumerate(_BLOCK_X_WINDOWS_PREFILL):
        block = [b for b in detected if xmin <= b[0] <= xmax]
        # Sort by y so row clustering is straightforward.
        block.sort(key=lambda b: (b[1], b[0]))
        # Greedy row clustering with a y-tolerance of 20 px (rows are ~88 px
        # apart so this comfortably separates them while tolerating jitter).
        rows: list[list[tuple[int, int, int]]] = []
        for b in block:
            placed = False
            for row in rows:
                if abs(row[0][1] - b[1]) <= 20:
                    row.append(b)
                    placed = True
                    break
            if not placed:
                rows.append([b])
        # Sort each row by x and prune duplicates (same x within 15 px).
        cleaned: list[list[tuple[int, int, int]]] = []
        for row in rows:
            row.sort(key=lambda b: b[0])
            deduped: list[tuple[int, int, int]] = []
            for b in row:
                if deduped and (b[0] - deduped[-1][0]) < 15:
                    if b[2] > deduped[-1][2]:
                        deduped[-1] = b
                    continue
                deduped.append(b)
            cleaned.append(deduped)
        # Keep only the top 5 rows that have exactly 4 bubbles (a complete
        # question). Reject rows with extra bubbles by trimming furthest
        # outliers from the median x positions.
        complete_rows: list[list[tuple[int, int, int]]] = []
        for row in cleaned:
            if len(row) == 4:
                complete_rows.append(row)
                continue
            if len(row) > 4:
                while len(row) > 4:
                    xs = [b[0] for b in row]
                    mean_x = sum(xs) / len(xs)
                    worst = max(range(len(row)), key=lambda i: abs(row[i][0] - mean_x))
                    row.pop(worst)
                complete_rows.append(row)
        complete_rows.sort(key=lambda r: r[0][1])
        block_diagnostics.append(
            f"block {block_idx+1}: detected_in_window={len(block)} -> "
            f"rows={len(rows)} complete={len(complete_rows)}"
        )
        if len(complete_rows) < 5:
            raise RuntimeError(
                f"Block {block_idx+1}: got only {len(complete_rows)} complete rows "
                f"out of 5. Diagnostics:\n  " + "\n  ".join(block_diagnostics)
            )
        for row in complete_rows[:5]:
            questions.append(row)
    print("Cluster diagnostics:")
    for d in block_diagnostics:
        print(f"  {d}")
    return questions


def overlay_detections(detected: list[tuple[int, int, int]]) -> None:
    img = Image.open(TEMPLATE_PNG).convert("RGB")
    draw = ImageDraw.Draw(img)
    for cx, cy, r in detected:
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 200, 0), width=2)
        draw.line([(cx - 2, cy), (cx + 2, cy)], fill=(255, 0, 0), width=1)
        draw.line([(cx, cy - 2), (cx, cy + 2)], fill=(255, 0, 0), width=1)
    out = OUT_DIR / "detected_bubbles.png"
    img.save(out)
    print(f"Saved detections overlay -> {out}")


def project_to_omr(cx_prefill: int, cy_prefill: int, prefill_w: int, prefill_h: int) -> tuple[float, float]:
    """Convert prefill-canvas pixel to OMR (666x515) pixel via proportional scale.

    This assumes ArUco markers at the same relative corner positions on both
    canvases, which is how the prefill template is designed.
    """
    return cx_prefill * OMR_W / prefill_w, cy_prefill * OMR_H / prefill_h


def emit_constants(
    questions: list[list[tuple[int, int, int]]],
    prefill_w: int,
    prefill_h: int,
) -> list[tuple[float, float, float, float, int]]:
    """Print + return the proposed _ANSWER_BLOCKS for both code & template.

    For each block, derives:
      - origin = (mean A-bubble x across 5 rows, top-row mean y)
      - bubblesGap = median horizontal step between A→B→C→D
      - labelsGap = median vertical step between successive rows
    All in OMR (666x515) space.
    """
    print("\n--- proposed _ANSWER_BLOCKS (OMR space, regularised) ---")
    blocks: list[tuple[float, float, float, float, int]] = []
    for block_idx in range(5):
        first_q_idx = block_idx * 5
        block_qs = questions[first_q_idx:first_q_idx + 5]
        row_ys = [sum(b[1] for b in row) / 4.0 for row in block_qs]
        col_xs = [
            sum(row[col][0] for row in block_qs) / 5.0
            for col in range(4)
        ]
        ox_p = col_xs[0]
        oy_p = row_ys[0]
        bubble_gaps_p = [col_xs[i + 1] - col_xs[i] for i in range(3)]
        bubbles_gap_p = sum(bubble_gaps_p) / len(bubble_gaps_p)
        label_gaps_p = [row_ys[i + 1] - row_ys[i] for i in range(4)]
        labels_gap_p = sum(label_gaps_p) / len(label_gaps_p)
        ox_omr, oy_omr = project_to_omr(ox_p, oy_p, prefill_w, prefill_h)
        bubbles_gap_omr = bubbles_gap_p * OMR_W / prefill_w
        labels_gap_omr = labels_gap_p * OMR_H / prefill_h
        first_q = first_q_idx + 1
        blocks.append((ox_omr, oy_omr, bubbles_gap_omr, labels_gap_omr, first_q))
        print(
            f"    ({ox_omr:6.2f}, {oy_omr:6.2f}, {bubbles_gap_omr:5.2f}, "
            f"{labels_gap_omr:5.2f}, {first_q:2d}),  # q{first_q}-q{first_q+4} "
            f"[row_ys={[f'{y:.0f}' for y in row_ys]}]"
        )
    print("\n--- OMR template fieldBlocks (CENTER coords) ---")
    print("Note: OMR template.json origin = these center coords - 5 (top-left).")
    for ox, oy, bg, lg, fq in blocks:
        print(
            f'  "q{fq:02d}block": '
            f'{{"origin": [{ox-5:.1f}, {oy-5:.1f}], '
            f'"bubblesGap": {bg:.2f}, "labelsGap": {lg:.2f}, ...}},'
        )
    return blocks


if __name__ == "__main__":
    detected = detect_bubbles()
    overlay_detections(detected)
    questions = cluster_into_grid(detected)
    with Image.open(TEMPLATE_PNG) as im:
        prefill_w, prefill_h = im.size
    print(f"Prefill canvas: {prefill_w}x{prefill_h}")
    print("\n--- q1-q5 detected bubble centres (prefill canvas) ---")
    for q_idx in range(5):
        cells = questions[q_idx]
        print(
            f"q{q_idx+1}: A={cells[0][:2]} B={cells[1][:2]} "
            f"C={cells[2][:2]} D={cells[3][:2]}"
        )
    blocks = emit_constants(questions, prefill_w, prefill_h)
    # Re-draw overlay with the REGULAR-GRID positions (what OMR will sample
    # and what student_fill will draw at) so we can visually confirm fit.
    img = Image.open(TEMPLATE_PNG).convert("RGB")
    draw = ImageDraw.Draw(img)
    for ox_omr, oy_omr, bg, lg, fq in blocks:
        for row in range(5):
            for col in range(4):
                cx_omr = ox_omr + col * bg
                cy_omr = oy_omr + row * lg
                cx_p = cx_omr * prefill_w / OMR_W
                cy_p = cy_omr * prefill_h / OMR_H
                r = 11
                draw.line([(cx_p - r, cy_p), (cx_p + r, cy_p)], fill=(0, 150, 255), width=2)
                draw.line([(cx_p, cy_p - r), (cx_p, cy_p + r)], fill=(0, 150, 255), width=2)
                draw.ellipse((cx_p - r, cy_p - r, cx_p + r, cy_p + r), outline=(0, 150, 255), width=2)
    out = OUT_DIR / "regular_grid_overlay.png"
    img.save(out)
    print(f"Saved regular-grid overlay -> {out}")
