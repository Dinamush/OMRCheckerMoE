"""Generate the blank printable PNG for the portrait 25Q OMR sheet.

The script paints, in OMR-processing space first then scaled to the print
canvas, every static element described in ``DESIGN.md`` — ArUco fiducials,
title + instructions, header text labels, the 10x10 candidate-number bubble
grid, the 2 x 13/12 answer grid, and the two-column divider.

Run from the repo root::

    python portrait_25q/generate_blank.py

Output: ``portrait_25q/reference/blank_portrait_25q.png``

Dependencies: PIL (Pillow) + opencv-python. Both are already in
``requirements.txt``.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# OMR-space constants — MUST match portrait_25q/template.json.
# ---------------------------------------------------------------------------

OMR_W, OMR_H = 515, 728           # internal OMR processing canvas (px)
PRINT_W, PRINT_H = 1654, 2339     # A4 portrait at 200 DPI

# Marker centres in OMR space (top-left, top-right, bottom-left, bottom-right).
ARUCO_CENTRES = [(14, 14), (501, 14), (14, 714), (501, 714)]
ARUCO_CORNER_IDS = [0, 1, 2, 3]
ARUCO_MARKER_SIZE_OMR = 27        # OMR-space marker width/height

# Header text strip y-coordinates (OMR space) and x ranges.
HEADER_FIELDS = [
    ("School Name", 95),
    ("Exam Name", 120),
    ("Region", 145),
    ("Student Name", 170),
]
HEADER_X_START, HEADER_X_END = 70, 460

# Candidate-number bubble grid (10 columns x 10 rows of digits 0-9).
# Origin = top-left of the digit-0 row of the first column.
CAND_ORIGIN_X, CAND_ORIGIN_Y = 133, 210
CAND_BUBBLES_GAP_X = 25.0   # column spacing
CAND_LABELS_GAP_Y = 14.5    # row spacing (between digit values 0..9)
CAND_BUBBLE_DIAM = 10       # bubble diameter (OMR space, matches bubbleDimensions)

# Answer grid (25 questions, 2 columns of 13/12).
ANS_BUBBLE_DIAM = 10
ANS_BUBBLES_GAP_X = 22.0
ANS_LABELS_GAP_Y = 22.0
ANS_BLOCK_LEFT_ORIGIN = (85, 400)
ANS_BLOCK_RIGHT_ORIGIN = (305, 400)

# Scaling factor from OMR space to print space.
SCALE_X = PRINT_W / OMR_W
SCALE_Y = PRINT_H / OMR_H

# Anchor for the printed page title / instructions.
TITLE_Y_OMR = 40
INSTR_Y_OMR = 65
TITLE_TEXT = "OMR Answer Sheet — 25 Questions"
INSTR_TEXT = (
    "Fill each bubble completely with a #2 pencil.  "
    "Mark only one answer per question.  Stray marks may be read as answers."
)


# ---------------------------------------------------------------------------
# Font loading — falls back gracefully across platforms.
# ---------------------------------------------------------------------------

def load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "DejaVuSans.ttf",
        "Arial.ttf",
        "arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Coordinate helpers.
# ---------------------------------------------------------------------------

def omr_to_print(x_omr: float, y_omr: float) -> tuple[float, float]:
    return x_omr * SCALE_X, y_omr * SCALE_Y


def omr_bubble_to_print(cx_omr: float, cy_omr: float, diam_omr: float) -> tuple[float, float, float, float]:
    """Return (x0, y0, x1, y1) print-space bbox for a centred bubble."""
    cx, cy = omr_to_print(cx_omr, cy_omr)
    r = (diam_omr / 2.0) * SCALE_X
    return cx - r, cy - r, cx + r, cy + r


# ---------------------------------------------------------------------------
# Drawing routines.
# ---------------------------------------------------------------------------

def draw_aruco(img_pil: Image.Image) -> Image.Image:
    """Stamp the 4 corner ArUco markers onto *img_pil* (in place semantics)."""
    img_cv = np.array(img_pil.convert("RGB"))[:, :, ::-1].copy()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_px = round(ARUCO_MARKER_SIZE_OMR * SCALE_X)
    for centre_omr, marker_id in zip(ARUCO_CENTRES, ARUCO_CORNER_IDS):
        cx_px, cy_px = omr_to_print(*centre_omr)
        x0 = int(round(cx_px - marker_px / 2))
        y0 = int(round(cy_px - marker_px / 2))
        x1 = x0 + marker_px
        y1 = y0 + marker_px
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        img_cv[y0:y1, x0:x1] = marker_bgr
    return Image.fromarray(img_cv[:, :, ::-1])


def draw_title_and_instructions(draw: ImageDraw.ImageDraw) -> None:
    title_font = load_font(36)
    instr_font = load_font(18)
    # Centred title.
    title_w = draw.textlength(TITLE_TEXT, font=title_font)
    title_x = (PRINT_W - title_w) / 2
    _, title_y = omr_to_print(0, TITLE_Y_OMR)
    draw.text((title_x, title_y), TITLE_TEXT, fill="black", font=title_font)
    instr_w = draw.textlength(INSTR_TEXT, font=instr_font)
    instr_x = (PRINT_W - instr_w) / 2
    _, instr_y = omr_to_print(0, INSTR_Y_OMR)
    draw.text((instr_x, instr_y), INSTR_TEXT, fill="dimgray", font=instr_font)


def draw_header_strips(draw: ImageDraw.ImageDraw) -> None:
    label_font = load_font(18)
    for label, y_omr in HEADER_FIELDS:
        x0_px, y_px = omr_to_print(HEADER_X_START, y_omr)
        x1_px, _ = omr_to_print(HEADER_X_END, y_omr)
        # Label (e.g. "Region:")
        draw.text(
            (x0_px - 10 * SCALE_X, y_px - 6),
            f"{label}:",
            fill="black",
            font=label_font,
            anchor="lt",
        )
        # Underline that the user / prefill engine writes onto.
        draw.line(
            [(x0_px + 90 * SCALE_X / OMR_W * OMR_W, y_px + 22), (x1_px, y_px + 22)],
            fill="black",
            width=2,
        )


def draw_candidate_grid(draw: ImageDraw.ImageDraw) -> None:
    digit_font = load_font(14)
    caption_font = load_font(16)
    # Caption above the grid.
    cap_x_omr = CAND_ORIGIN_X + (CAND_BUBBLES_GAP_X * 9) / 2
    cap_y_omr = CAND_ORIGIN_Y - 20
    cap_x, cap_y = omr_to_print(cap_x_omr, cap_y_omr)
    draw.text((cap_x, cap_y), "Candidate Number", fill="black", font=caption_font, anchor="mm")

    # Column digit headers (above row 0): write the column index 1..10 at top? No —
    # the candidate-number grid is "digit rows" not "column digits". The 10 rows
    # of each column carry digits 0..9. So we don't need column headers.
    for col in range(10):
        cx_omr = CAND_ORIGIN_X + col * CAND_BUBBLES_GAP_X
        for digit in range(10):
            cy_omr = CAND_ORIGIN_Y + digit * CAND_LABELS_GAP_Y
            bbox = omr_bubble_to_print(cx_omr, cy_omr, CAND_BUBBLE_DIAM)
            draw.ellipse(bbox, outline="black", width=2)
            # Digit label inside the bubble.
            draw.text(
                ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                str(digit),
                fill="black",
                font=digit_font,
                anchor="mm",
            )


def draw_answer_block(
    draw: ImageDraw.ImageDraw,
    *,
    origin_omr: tuple[int, int],
    first_q: int,
    last_q: int,
) -> None:
    """Draw one column of question bubbles.

    Each row: "Qn." label + 4 bubbles A/B/C/D arranged horizontally.
    """
    letter_font = load_font(14)
    qnum_font = load_font(15)
    ox, oy = origin_omr
    for row_idx, qnum in enumerate(range(first_q, last_q + 1)):
        cy_omr = oy + row_idx * ANS_LABELS_GAP_Y
        # Question number label to the left of the first bubble.
        q_label_x_omr = ox - 25
        q_label_x_px, q_label_y_px = omr_to_print(q_label_x_omr, cy_omr)
        draw.text(
            (q_label_x_px, q_label_y_px),
            f"{qnum}.",
            fill="black",
            font=qnum_font,
            anchor="lm",
        )
        for option_idx, letter in enumerate("ABCD"):
            cx_omr = ox + option_idx * ANS_BUBBLES_GAP_X
            bbox = omr_bubble_to_print(cx_omr, cy_omr, ANS_BUBBLE_DIAM)
            draw.ellipse(bbox, outline="black", width=2)
            draw.text(
                ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                letter,
                fill="black",
                font=letter_font,
                anchor="mm",
            )


def draw_answer_grid(draw: ImageDraw.ImageDraw) -> None:
    draw_answer_block(draw, origin_omr=ANS_BLOCK_LEFT_ORIGIN, first_q=1, last_q=13)
    draw_answer_block(draw, origin_omr=ANS_BLOCK_RIGHT_ORIGIN, first_q=14, last_q=25)
    # Column header text above each block.
    header_font = load_font(16)
    left_h_x, left_h_y = omr_to_print(
        ANS_BLOCK_LEFT_ORIGIN[0] + ANS_BUBBLES_GAP_X * 1.5,
        ANS_BLOCK_LEFT_ORIGIN[1] - 24,
    )
    right_h_x, right_h_y = omr_to_print(
        ANS_BLOCK_RIGHT_ORIGIN[0] + ANS_BUBBLES_GAP_X * 1.5,
        ANS_BLOCK_RIGHT_ORIGIN[1] - 24,
    )
    draw.text((left_h_x, left_h_y), "Q1 – Q13", fill="black", font=header_font, anchor="mm")
    draw.text((right_h_x, right_h_y), "Q14 – Q25", fill="black", font=header_font, anchor="mm")


def draw_column_divider(draw: ImageDraw.ImageDraw) -> None:
    # Vertical divider between the two answer columns.
    div_x_omr = (ANS_BLOCK_LEFT_ORIGIN[0] + ANS_BLOCK_RIGHT_ORIGIN[0]) / 2
    div_y_top_omr = ANS_BLOCK_LEFT_ORIGIN[1] - 32
    div_y_bot_omr = ANS_BLOCK_LEFT_ORIGIN[1] + ANS_LABELS_GAP_Y * 13 + 8
    x0, y0 = omr_to_print(div_x_omr, div_y_top_omr)
    x1, y1 = omr_to_print(div_x_omr, div_y_bot_omr)
    draw.line([(x0, y0), (x1, y1)], fill="lightgray", width=1)


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------

def main() -> Path:
    out_dir = Path(__file__).resolve().parent / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "blank_portrait_25q.png"

    img = Image.new("RGB", (PRINT_W, PRINT_H), "white")
    draw = ImageDraw.Draw(img)

    draw_title_and_instructions(draw)
    draw_header_strips(draw)
    draw_candidate_grid(draw)
    draw_column_divider(draw)
    draw_answer_grid(draw)

    img = draw_aruco(img)
    img.save(out_path, format="PNG", compress_level=1)
    print(f"Wrote {out_path} ({img.size[0]}x{img.size[1]})")
    return out_path


if __name__ == "__main__":
    main()
