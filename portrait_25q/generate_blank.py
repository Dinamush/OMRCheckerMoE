"""Generate the blank printable PNG for the portrait 25Q OMR sheet.

The script paints, in OMR-processing space first then scaled to the print
canvas, every static element described in ``DESIGN.md`` — ArUco fiducials,
title + instructions, header text labels, the 10x10 candidate-number bubble
grid, the 2 x 13/12 answer grid, and the two-column divider.

Sizing is anchored to the multi-source research synthesis recorded in
``DESIGN.md §4`` — Scantron / Remark / OMRChecker (industry), APH / RNIB /
British Dyslexia Association / SAT (accessibility), PLOS ONE FlAttum et al.
+ MDPI elementary-reader studies (motor + psychology), and OpenCV ArUco /
Kofax OmniPage / CEUR-WS marker-detection thresholds (camera reliability).

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
# Canvas — OMR processing space + US Letter print space @ 200 DPI.
# OMR space aspect matches Letter (515 : 666 ≈ 8.5 : 11) so the scale
# factor is uniform (≈ 3.301 print-px / OMR-px) and bubbles stay round.
# 1 OMR-px ≈ 0.419 mm ≈ 1.19 pt.
# ---------------------------------------------------------------------------

OMR_W, OMR_H = 515, 666
PRINT_W, PRINT_H = 1700, 2200

SCALE_X = PRINT_W / OMR_W       # ≈ 3.301
SCALE_Y = PRINT_H / OMR_H       # ≈ 3.303

# ---------------------------------------------------------------------------
# ArUco corner fiducials — damage-resistant inset.
#
# Marker side: 30 OMR-px ≈ 12.6 mm physical — exceeds OpenCV's 30-capture-px
# comfort threshold for DICT_4X4_50 at every supported scan path (5 MP
# phone, 12 MP phone, 300 DPI flatbed).
#
# Centre inset: 40 OMR-px from each page edge ≈ 16.8 mm physical. The
# marker *outer* edge therefore sits 25 OMR-px / ~10.5 mm from the page
# edge — safely outside the 5–8 mm accidental dog-ear zone documented in
# print-finishing literature, and giving a meaningful (if not absolute)
# defence against the 10–15 mm "deliberate fold for stapling" damage
# pattern. The remaining 10.5 mm of margin paper is a sacrificial zone
# that absorbs corner wear without touching the fiducial.
#
# Defence-in-depth: even if one or two markers are destroyed, the
# OMRChecker engine's ``CropOnMarkers._apply_aruco_filter`` already
# implements affine 3-of-4 recovery and similarity 2-of-4 recovery
# (gated by a bubble-confidence score). See DESIGN.md §3.4.
# ---------------------------------------------------------------------------

ARUCO_CENTRES = [(40, 40), (475, 40), (40, 626), (475, 626)]
ARUCO_CORNER_IDS = [0, 1, 2, 3]
ARUCO_MARKER_SIZE_OMR = 30

# Fold-safe frame — thin grey corner brackets just OUTSIDE the markers
# (between marker outer edge and page edge). Three roles:
#   1. visual "do not fold inside this line" cue to students/invigilators,
#   2. obvious damage indicator (a torn corner produces a visible gap),
#   3. secondary visual anchor for human re-alignment if the printed
#      output is photocopied off-centre.
# Drawn in light grey so it does not interfere with adaptive thresholding.
FOLD_SAFE_INSET_OMR = 12       # bracket runs at this inset from page edge
FOLD_SAFE_ARM_OMR = 18         # bracket arm length per axis
FOLD_SAFE_STROKE_PX = 2
FOLD_SAFE_COLOR = "#bbbbbb"
FOLD_WARNING_TEXT = "Keep corners clean — do not fold or staple inside the grey bracket."
FOLD_WARNING_Y_OMR = 30        # centred between top markers

# ---------------------------------------------------------------------------
# Title + instructions (anchored at top of usable content area).
# All y-anchors sit below the top-marker quiet zone (marker centre y=40,
# half-size 15, quiet zone +5 → content starts at y≥60).
# Title uses 22 pt bold sans-serif (BDA "headings 20 % larger than body"
# applied to a 14 pt body floor → 18 pt min, rounded up).
# Instructions use 11.5 pt — comfortable plain-text body on Letter paper
# and above the RNIB Clear-Print 11 pt floor.
# ---------------------------------------------------------------------------

TITLE_Y_OMR = 65
INSTR_LINE_1_Y_OMR = 100
INSTR_LINE_2_Y_OMR = 114
TITLE_TEXT = "OMR Answer Sheet — 25 Questions"
INSTR_LINE_1 = "Fill each bubble completely with a #2 pencil. Mark only one answer per question."
INSTR_LINE_2 = "Stray marks may be read as answers."

# ---------------------------------------------------------------------------
# Header text strips ("School Name:", "Exam Name:", "Region:", "Student
# Name:"). 14 pt sans-serif labels — RNIB / BDA accessibility floor for
# unaccommodated school-age print.
# Horizontal extent restricted to the marker quiet zone:
#   left limit  = 40 (marker centre) + 15 (half) + 5 (quiet zone)  = 60
#   right limit = 475 (marker centre) - 15 (half) - 5 (quiet zone) = 455
# ---------------------------------------------------------------------------

HEADER_FIELDS = [
    ("School Name", 138),
    ("Exam Name", 160),
    ("Region", 182),
    ("Student Name", 204),
]
HEADER_LABEL_X_OMR = 60        # left anchor — clear of top-left marker quiet zone
HEADER_LINE_START_X_OMR = 165  # underline begins after the widest label
HEADER_LINE_END_X_OMR = 455    # underline ends before top-right marker quiet zone
HEADER_UNDERLINE_PX = 6        # ≈ 0.75 mm — APH "≥ 2 pt stroke" for print

# ---------------------------------------------------------------------------
# Candidate-number bubble grid (10 columns × 10 rows of digits 0-9).
# Kept at 4.2 mm Ø because growing the bubbles to 5.5 mm cannot fit the
# 100-bubble grid + 13-row answer column inside Letter portrait without
# spilling into the ArUco quiet zone. 4.2 mm is still well above
# Scantron's 2.5 mm machine floor and OMRChecker's smallest validated
# bubble (≈ 4.7 mm at 40 px on a 1846-wide A4 canvas).
# ---------------------------------------------------------------------------

CAND_CAPTION_Y_OMR = 222
CAND_ORIGIN_X, CAND_ORIGIN_Y = 133, 237
CAND_BUBBLES_GAP_X = 25.0
CAND_LABELS_GAP_Y = 13.5
CAND_BUBBLE_DIAM = 10          # 4.2 mm

# ---------------------------------------------------------------------------
# Answer grid (25 questions, 2 columns of 13 + 12).
# 13 OMR-px Ø ≈ 5.5 mm — at the centre of the industry / accessibility /
# motor-skill / phone-camera consensus (Scantron 5.08 mm, RNIB 5–6 mm,
# Hughes & Wilkins 6.0 mm, OpenCV/Kofax ≥ 5 mm for reliable detection).
# Centre-to-centre 24 OMR-px (≈ 10 mm) horizontal — matches Scantron 0.166″
# pitch ×2 and gives ≈ 4.6 mm between bubble edges. Row gap is intentionally
# tighter (20 OMR-px ≈ 8.4 mm) so 13 rows fit above the bottom ArUco
# quiet zone.
# ---------------------------------------------------------------------------

ANS_HEADER_Y_OMR = 380
ANS_BLOCK_LEFT_ORIGIN = (85, 395)
ANS_BLOCK_RIGHT_ORIGIN = (305, 395)
ANS_BUBBLE_DIAM = 13           # 5.5 mm
ANS_BUBBLES_GAP_X = 24.0       # 10.0 mm centre-to-centre
ANS_LABELS_GAP_Y = 17.0        # 7.1 mm row pitch — compressed to fit between
                               # the new bottom-marker quiet zone (y ≤ 606)
                               # and the answer-header at y=380.
                               # 12 gaps × 17.0 = 204; q13 centre y = 599
                               # and bubble bottom y = 605.5, preserving the
                               # 5 px internal quiet zone above marker top y=611.

# Outline stroke — 3 print-px ≈ 0.38 mm. Inside the safe band: thin
# enough that no OMR engine reads the outline as a fill (Addmen warning),
# thick enough to survive 200 DPI scanner downsampling and JPEG.
BUBBLE_STROKE_PX = 3

# ---------------------------------------------------------------------------
# Font sizes (PIL .truetype size = pixel height @ 200 DPI canvas).
# 1 pt ≈ 2.78 px.
# ---------------------------------------------------------------------------

FONT_TITLE_PX = 60             # ≈ 22 pt — page title
FONT_INSTR_PX = 32             # ≈ 11.5 pt — instructions
FONT_HEADER_PX = 38            # ≈ 14 pt — "School Name:", section captions
FONT_QNUM_PX = 32              # ≈ 11.5 pt — "1.", "13."
FONT_LETTER_PX = 28            # ≈ 10 pt — A/B/C/D inside answer bubble
FONT_DIGIT_PX = 22             # ≈ 8 pt — 0-9 inside candidate bubble

# Bubble-internal glyphs (A/B/C/D, 0-9) are drawn in mid-grey rather than
# black so the printed letter cannot be mistaken for a pencil fill by the
# OMR detector (PLOS ONE FlAttum et al., 2018) — the bubble outline
# remains pure black.
LABEL_GLYPH_COLOR = "#777777"
LABEL_FILL_COLOR = "black"
SUBTLE_COLOR = "#333333"


# ---------------------------------------------------------------------------
# Font loading — falls back gracefully across platforms.
# ---------------------------------------------------------------------------

def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a sans-serif TTF at the requested *pixel* size."""
    bold_candidates = [
        "DejaVuSans-Bold.ttf",
        "Arial Bold.ttf",
        "arialbd.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    regular_candidates = [
        "DejaVuSans.ttf",
        "Arial.ttf",
        "arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in (bold_candidates if bold else regular_candidates):
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

def draw_fold_safe_frame(draw: ImageDraw.ImageDraw) -> None:
    """Draw the four grey L-shaped corner brackets and the printed warning.

    Each bracket sits *outside* the ArUco marker on the sacrificial paper
    margin. It runs FOLD_SAFE_INSET_OMR from the page edge and extends
    FOLD_SAFE_ARM_OMR along each axis from the corner.

    A torn or folded corner will visibly interrupt the bracket, giving
    the OMR engine (and any human invigilator) an unambiguous "this page
    has been damaged in the marker region" signal.
    """
    arm = FOLD_SAFE_ARM_OMR
    inset = FOLD_SAFE_INSET_OMR
    corners = [
        # (corner_x, corner_y, axis_dx, axis_dy)
        (inset, inset, +1, +1),                          # TL
        (OMR_W - inset, inset, -1, +1),                  # TR
        (inset, OMR_H - inset, +1, -1),                  # BL
        (OMR_W - inset, OMR_H - inset, -1, -1),          # BR
    ]
    for cx, cy, dx, dy in corners:
        cx_px, cy_px = omr_to_print(cx, cy)
        end_h_x_px, _ = omr_to_print(cx + dx * arm, cy)
        _, end_v_y_px = omr_to_print(cx, cy + dy * arm)
        draw.line(
            [(cx_px, cy_px), (end_h_x_px, cy_px)],
            fill=FOLD_SAFE_COLOR,
            width=FOLD_SAFE_STROKE_PX,
        )
        draw.line(
            [(cx_px, cy_px), (cx_px, end_v_y_px)],
            fill=FOLD_SAFE_COLOR,
            width=FOLD_SAFE_STROKE_PX,
        )

    warn_font = load_font(20)
    warn_w = draw.textlength(FOLD_WARNING_TEXT, font=warn_font)
    warn_x = (PRINT_W - warn_w) / 2
    _, warn_y = omr_to_print(0, FOLD_WARNING_Y_OMR)
    draw.text(
        (warn_x, warn_y),
        FOLD_WARNING_TEXT,
        fill=FOLD_SAFE_COLOR,
        font=warn_font,
        anchor="lt",
    )


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
    title_font = load_font(FONT_TITLE_PX, bold=True)
    instr_font = load_font(FONT_INSTR_PX)
    title_w = draw.textlength(TITLE_TEXT, font=title_font)
    title_x = (PRINT_W - title_w) / 2
    _, title_y = omr_to_print(0, TITLE_Y_OMR)
    draw.text((title_x, title_y), TITLE_TEXT, fill=LABEL_FILL_COLOR, font=title_font, anchor="lt")
    for text, y_omr in (
        (INSTR_LINE_1, INSTR_LINE_1_Y_OMR),
        (INSTR_LINE_2, INSTR_LINE_2_Y_OMR),
    ):
        line_w = draw.textlength(text, font=instr_font)
        line_x = (PRINT_W - line_w) / 2
        _, line_y = omr_to_print(0, y_omr)
        draw.text((line_x, line_y), text, fill=SUBTLE_COLOR, font=instr_font, anchor="lt")


# Print-space margin between the label's bbox bottom and the underline.
# Small enough to read as part of the same field, large enough not to
# clip descenders or kiss the visible baseline.
HEADER_UNDERLINE_MARGIN_PX = 4


def draw_header_strips(draw: ImageDraw.ImageDraw) -> None:
    label_font = load_font(FONT_HEADER_PX, bold=True)
    for label, y_omr in HEADER_FIELDS:
        # With ``anchor="lm"`` PIL centres the text bbox vertically on the
        # given y, so ``text_mid_y_px`` is the MIDDLE of the rendered text
        # (not the baseline). The bbox bottom therefore sits at
        # ``text_mid_y_px + FONT_HEADER_PX / 2`` — that's where the
        # underline must clear.
        label_x_px, text_mid_y_px = omr_to_print(HEADER_LABEL_X_OMR, y_omr)
        draw.text(
            (label_x_px, text_mid_y_px),
            f"{label}:",
            fill=LABEL_FILL_COLOR,
            font=label_font,
            anchor="lm",
        )
        line_x0, _ = omr_to_print(HEADER_LINE_START_X_OMR, y_omr)
        line_x1, _ = omr_to_print(HEADER_LINE_END_X_OMR, y_omr)
        underline_y_px = (
            text_mid_y_px + FONT_HEADER_PX / 2 + HEADER_UNDERLINE_MARGIN_PX
        )
        draw.line(
            [(line_x0, underline_y_px), (line_x1, underline_y_px)],
            fill=LABEL_FILL_COLOR,
            width=HEADER_UNDERLINE_PX,
        )


def draw_candidate_grid(draw: ImageDraw.ImageDraw) -> None:
    digit_font = load_font(FONT_DIGIT_PX)
    caption_font = load_font(FONT_HEADER_PX, bold=True)
    cap_x_omr = CAND_ORIGIN_X + (CAND_BUBBLES_GAP_X * 9) / 2
    cap_x, cap_y = omr_to_print(cap_x_omr, CAND_CAPTION_Y_OMR)
    draw.text(
        (cap_x, cap_y),
        "Candidate Number",
        fill=LABEL_FILL_COLOR,
        font=caption_font,
        anchor="mm",
    )

    for col in range(10):
        cx_omr = CAND_ORIGIN_X + col * CAND_BUBBLES_GAP_X
        for digit in range(10):
            cy_omr = CAND_ORIGIN_Y + digit * CAND_LABELS_GAP_Y
            bbox = omr_bubble_to_print(cx_omr, cy_omr, CAND_BUBBLE_DIAM)
            draw.ellipse(bbox, outline=LABEL_FILL_COLOR, width=BUBBLE_STROKE_PX)
            draw.text(
                ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                str(digit),
                fill=LABEL_GLYPH_COLOR,
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
    letter_font = load_font(FONT_LETTER_PX)
    qnum_font = load_font(FONT_QNUM_PX, bold=True)
    ox, oy = origin_omr
    for row_idx, qnum in enumerate(range(first_q, last_q + 1)):
        cy_omr = oy + row_idx * ANS_LABELS_GAP_Y
        # Right-align question numbers so "1." and "13." sit the same
        # distance from the first bubble.
        q_label_x_omr = ox - 12
        q_label_x_px, q_label_y_px = omr_to_print(q_label_x_omr, cy_omr)
        draw.text(
            (q_label_x_px, q_label_y_px),
            f"{qnum}.",
            fill=LABEL_FILL_COLOR,
            font=qnum_font,
            anchor="rm",
        )
        for option_idx, letter in enumerate("ABCD"):
            cx_omr = ox + option_idx * ANS_BUBBLES_GAP_X
            bbox = omr_bubble_to_print(cx_omr, cy_omr, ANS_BUBBLE_DIAM)
            draw.ellipse(bbox, outline=LABEL_FILL_COLOR, width=BUBBLE_STROKE_PX)
            draw.text(
                ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                letter,
                fill=LABEL_GLYPH_COLOR,
                font=letter_font,
                anchor="mm",
            )


def draw_answer_grid(draw: ImageDraw.ImageDraw) -> None:
    draw_answer_block(draw, origin_omr=ANS_BLOCK_LEFT_ORIGIN, first_q=1, last_q=13)
    draw_answer_block(draw, origin_omr=ANS_BLOCK_RIGHT_ORIGIN, first_q=14, last_q=25)
    header_font = load_font(FONT_HEADER_PX, bold=True)
    left_h_x, left_h_y = omr_to_print(
        ANS_BLOCK_LEFT_ORIGIN[0] + ANS_BUBBLES_GAP_X * 1.5,
        ANS_HEADER_Y_OMR,
    )
    right_h_x, right_h_y = omr_to_print(
        ANS_BLOCK_RIGHT_ORIGIN[0] + ANS_BUBBLES_GAP_X * 1.5,
        ANS_HEADER_Y_OMR,
    )
    draw.text((left_h_x, left_h_y), "Q1 – Q13", fill=LABEL_FILL_COLOR, font=header_font, anchor="mm")
    draw.text((right_h_x, right_h_y), "Q14 – Q25", fill=LABEL_FILL_COLOR, font=header_font, anchor="mm")


def draw_column_divider(draw: ImageDraw.ImageDraw) -> None:
    div_x_omr = (ANS_BLOCK_LEFT_ORIGIN[0] + ANS_BLOCK_RIGHT_ORIGIN[0]) / 2
    div_y_top_omr = ANS_HEADER_Y_OMR - 8
    # Extend to the exact bottom edge of the q13 bubble (q1 -> q13 is
    # 12 row-gaps, plus one bubble radius) so the divider visually
    # separates both columns for their full height. With the current
    # geometry (origin 395, gap 17.0, diameter 13) this lands at
    # y=605.5 OMR-px -- still safely inside the y<=606 marker-quiet-zone
    # content limit documented in DESIGN.md §3.1.
    div_y_bot_omr = (
        ANS_BLOCK_LEFT_ORIGIN[1] + ANS_LABELS_GAP_Y * 12 + ANS_BUBBLE_DIAM / 2
    )
    x0, y0 = omr_to_print(div_x_omr, div_y_top_omr)
    x1, y1 = omr_to_print(div_x_omr, div_y_bot_omr)
    draw.line([(x0, y0), (x1, y1)], fill="#cccccc", width=2)


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------

def main() -> Path:
    out_dir = Path(__file__).resolve().parent / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "blank_portrait_25q.png"

    img = Image.new("RGB", (PRINT_W, PRINT_H), "white")
    draw = ImageDraw.Draw(img)

    draw_fold_safe_frame(draw)
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
