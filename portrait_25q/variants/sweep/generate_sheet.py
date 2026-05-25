"""Parameterised generator for the Variant-A portrait sheet under sweep.

Emits a matched (template.json + blank PNG) pair for a given candidate-
number bubble diameter. The sweep harness uses this to create N
candidate-size variants and compare detection accuracy across them.

Critical correctness note
-------------------------
For OMRChecker, ``QTYPE_INT`` defaults to ``direction="vertical"``, which
means:

  * ``bubblesGap`` = step BETWEEN bubbles within a single field
    (i.e. vertical spacing between digits 0..9 inside one column).
  * ``labelsGap`` = step BETWEEN fields
    (i.e. horizontal spacing between digit-position columns).

The previous in-tree ``template.json`` had these swapped for
``CandidateNumber`` (bubblesGap=25, labelsGap=13.5), causing the engine
to look for bubbles at transposed positions. This generator emits the
**corrected** values so the engine reads the bubbles where we draw them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Canvas (US Letter portrait, 200 DPI).
# ---------------------------------------------------------------------------

OMR_W, OMR_H = 515, 666
PRINT_W, PRINT_H = 1700, 2200
SCALE_X = PRINT_W / OMR_W
SCALE_Y = PRINT_H / OMR_H

ARUCO_CENTRES = [(40, 40), (475, 40), (40, 626), (475, 626)]
ARUCO_CORNER_IDS = [0, 1, 2, 3]
ARUCO_MARKER_SIZE_OMR = 30

FOLD_SAFE_INSET_OMR = 12
FOLD_SAFE_ARM_OMR = 18
FOLD_SAFE_STROKE_PX = 2
FOLD_SAFE_COLOR = "#bbbbbb"
FOLD_WARNING_TEXT = "Keep corners clean — do not fold or staple inside the grey bracket."
FOLD_WARNING_Y_OMR = 30

TITLE_LINE_1 = "Ministry of Education"
TITLE_LINE_2 = "Multiple Choice Answer Sheet"
INSTR_TEXT = "Fill bubbles completely with a #2 pencil. One answer per question."
METADATA_FIELDS = ("Name", "School", "Exam")

LABEL_GLYPH_COLOR = "#777777"
LABEL_FILL_COLOR = "black"
SUBTLE_COLOR = "#333333"
BUBBLE_STROKE_PX = 3

ANS_BUBBLE_DIAM = 14
ANS_BUBBLES_GAP_X = 24.0
ANS_LABELS_GAP_Y = 17.5
ANS_BLOCK_LEFT_ORIGIN = (85, 420)
ANS_BLOCK_RIGHT_ORIGIN = (305, 420)
ANS_HEADER_Y = 405


# ---------------------------------------------------------------------------
# Per-sweep parameters.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SheetSpec:
    """Layout knobs for one row of the candidate-bubble sweep."""

    label: str                    # e.g. "size_10"
    cand_bubble_diam: int         # OMR-px
    cand_bubbles_gap_x: float     # horizontal column spacing (OMR-px)
    cand_labels_gap_y: float      # vertical row spacing (OMR-px)
    cand_origin: tuple[int, int]  # OMR-px

    @property
    def cand_bottom_y(self) -> float:
        return self.cand_origin[1] + 9 * self.cand_labels_gap_y + self.cand_bubble_diam


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def omr_to_print(x: float, y: float) -> tuple[float, float]:
    return x * SCALE_X, y * SCALE_Y


def bubble_bbox(cx_omr: float, cy_omr: float, diam_omr: float) -> tuple[float, float, float, float]:
    cx, cy = omr_to_print(cx_omr, cy_omr)
    r = (diam_omr / 2.0) * SCALE_X
    return cx - r, cy - r, cx + r, cy + r


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    bold_candidates = [
        "DejaVuSans-Bold.ttf",
        "arialbd.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    regular_candidates = [
        "DejaVuSans.ttf",
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in bold_candidates if bold else regular_candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Drawing.
# ---------------------------------------------------------------------------


def draw_fold_safe_frame(draw: ImageDraw.ImageDraw) -> None:
    arm = FOLD_SAFE_ARM_OMR
    inset = FOLD_SAFE_INSET_OMR
    corners = [
        (inset, inset, +1, +1),
        (OMR_W - inset, inset, -1, +1),
        (inset, OMR_H - inset, +1, -1),
        (OMR_W - inset, OMR_H - inset, -1, -1),
    ]
    for cx, cy, dx, dy in corners:
        cx_px, cy_px = omr_to_print(cx, cy)
        end_h_x_px, _ = omr_to_print(cx + dx * arm, cy)
        _, end_v_y_px = omr_to_print(cx, cy + dy * arm)
        draw.line([(cx_px, cy_px), (end_h_x_px, cy_px)], fill=FOLD_SAFE_COLOR, width=FOLD_SAFE_STROKE_PX)
        draw.line([(cx_px, cy_px), (cx_px, end_v_y_px)], fill=FOLD_SAFE_COLOR, width=FOLD_SAFE_STROKE_PX)
    warn_font = load_font(20)
    warn_w = draw.textlength(FOLD_WARNING_TEXT, font=warn_font)
    _, warn_y = omr_to_print(0, FOLD_WARNING_Y_OMR)
    draw.text(((PRINT_W - warn_w) / 2, warn_y), FOLD_WARNING_TEXT, fill=FOLD_SAFE_COLOR, font=warn_font, anchor="lt")


def draw_aruco(img_pil: Image.Image) -> Image.Image:
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


def draw_title_strip(draw: ImageDraw.ImageDraw) -> None:
    title_font = load_font(60, bold=True)
    subtitle_font = load_font(36, bold=False)
    instr_font = load_font(32)
    _, y1 = omr_to_print(0, 65)
    line1_w = draw.textlength(TITLE_LINE_1, font=title_font)
    draw.text(((PRINT_W - line1_w) / 2, y1), TITLE_LINE_1, fill=LABEL_FILL_COLOR, font=title_font, anchor="lt")
    _, y2 = omr_to_print(0, 87)
    line2_w = draw.textlength(TITLE_LINE_2, font=subtitle_font)
    draw.text(((PRINT_W - line2_w) / 2, y2), TITLE_LINE_2, fill=SUBTLE_COLOR, font=subtitle_font, anchor="lt")
    _, y3 = omr_to_print(0, 105)
    instr_w = draw.textlength(INSTR_TEXT, font=instr_font)
    draw.text(((PRINT_W - instr_w) / 2, y3), INSTR_TEXT, fill=SUBTLE_COLOR, font=instr_font, anchor="lt")


def draw_header(draw: ImageDraw.ImageDraw) -> None:
    label_font = load_font(36, bold=True)
    rows = [("Name", 150), ("School", 180), ("Exam", 210)]
    for label, y_omr in rows:
        label_x, baseline_y = omr_to_print(60, y_omr)
        draw.text((label_x, baseline_y), f"{label}:", fill=LABEL_FILL_COLOR, font=label_font, anchor="lm")
        line_x0, _ = omr_to_print(160, y_omr)
        line_x1, _ = omr_to_print(475, y_omr)
        underline_y = baseline_y + label_font.size * 0.35
        draw.line([(line_x0, underline_y), (line_x1, underline_y)], fill=LABEL_FILL_COLOR, width=5)


def draw_candidate_grid(draw: ImageDraw.ImageDraw, spec: SheetSpec) -> None:
    digit_font = load_font(max(14, int(spec.cand_bubble_diam * 2.0)))
    caption_font = load_font(32, bold=True)
    ox, oy = spec.cand_origin
    cap_x_omr = ox + (spec.cand_bubbles_gap_x * 9) / 2
    cap_y_omr = oy - 15
    cap_x, cap_y = omr_to_print(cap_x_omr, cap_y_omr)
    draw.text((cap_x, cap_y), "Candidate Number", fill=LABEL_FILL_COLOR, font=caption_font, anchor="mm")
    for col in range(10):
        cx_omr = ox + col * spec.cand_bubbles_gap_x
        for digit in range(10):
            cy_omr = oy + digit * spec.cand_labels_gap_y
            bbox = bubble_bbox(cx_omr, cy_omr, spec.cand_bubble_diam)
            draw.ellipse(bbox, outline=LABEL_FILL_COLOR, width=BUBBLE_STROKE_PX)
            draw.text(
                ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                str(digit),
                fill=LABEL_GLYPH_COLOR,
                font=digit_font,
                anchor="mm",
            )


def draw_answer_grid(draw: ImageDraw.ImageDraw) -> None:
    letter_font = load_font(30)
    qnum_font = load_font(34, bold=True)
    header_font = load_font(34, bold=True)
    for origin, q_first, q_last, label in [
        (ANS_BLOCK_LEFT_ORIGIN, 1, 13, "Q1–Q13"),
        (ANS_BLOCK_RIGHT_ORIGIN, 14, 25, "Q14–Q25"),
    ]:
        ox, oy = origin
        for row_idx, qnum in enumerate(range(q_first, q_last + 1)):
            cy_omr = oy + row_idx * ANS_LABELS_GAP_Y
            q_label_x, q_label_y = omr_to_print(ox - 12, cy_omr)
            draw.text((q_label_x, q_label_y), f"{qnum}.", fill=LABEL_FILL_COLOR, font=qnum_font, anchor="rm")
            for option_idx, letter in enumerate("ABCD"):
                cx_omr = ox + option_idx * ANS_BUBBLES_GAP_X
                bbox = bubble_bbox(cx_omr, cy_omr, ANS_BUBBLE_DIAM)
                draw.ellipse(bbox, outline=LABEL_FILL_COLOR, width=BUBBLE_STROKE_PX)
                draw.text(
                    ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                    letter,
                    fill=LABEL_GLYPH_COLOR,
                    font=letter_font,
                    anchor="mm",
                )
        h_x, h_y = omr_to_print(ox + ANS_BUBBLES_GAP_X * 1.5, ANS_HEADER_Y)
        draw.text((h_x, h_y), label, fill=LABEL_FILL_COLOR, font=header_font, anchor="mm")
    div_x = (ANS_BLOCK_LEFT_ORIGIN[0] + ANS_BLOCK_RIGHT_ORIGIN[0]) / 2 + ANS_BUBBLES_GAP_X * 1.25
    x0, y0 = omr_to_print(div_x, ANS_HEADER_Y - 8)
    x1, y1 = omr_to_print(div_x, ANS_BLOCK_LEFT_ORIGIN[1] + ANS_LABELS_GAP_Y * 12 + 3)
    draw.line([(x0, y0), (x1, y1)], fill="#cccccc", width=2)


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------


def render_blank(spec: SheetSpec) -> Image.Image:
    img = Image.new("RGB", (PRINT_W, PRINT_H), "white")
    draw = ImageDraw.Draw(img)
    draw_fold_safe_frame(draw)
    draw_title_strip(draw)
    draw_header(draw)
    draw_candidate_grid(draw, spec)
    draw_answer_grid(draw)
    return draw_aruco(img)


def emit_config(out_dir: Path) -> Path:
    """Write a ``config.json`` matched to this template's coordinate space.

    Choosing ``processing_width = OMR_W`` (and analogously for height)
    means every coordinate in the template (marker centres, field-block
    origins) is interpreted by the engine in **the same units we used
    when drawing the sheet**. This avoids a double-scaling between
    OMR-space and processing-canvas-space that previously trip-warped
    the homography sanity check.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "dimensions": {
            "processing_width": OMR_W,
            "processing_height": OMR_H,
        },
        "outputs": {
            "save_image_level": 0,
            "show_image_level": 0,
        },
    }
    path = out_dir / "config.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


def emit_template(spec: SheetSpec, out_dir: Path) -> Path:
    """Write a ``template.json`` matched to ``spec`` into ``out_dir``.

    Origin / sampling-box convention
    --------------------------------
    The engine's per-bubble integral-image sample uses ``(pt.x, pt.y)``
    as the **top-left corner** of a ``bubbleDimensions``-sized box. Our
    generator draws each bubble **centred** on its grid position, so the
    template origin we emit is ``centre − bubbleDimensions / 2`` for the
    first bubble of each block (and subsequent bubbles inherit the
    offset because ``bubblesGap`` / ``labelsGap`` are measured between
    centres in both the generator and the engine).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_origin = [
        spec.cand_origin[0] - spec.cand_bubble_diam / 2,
        spec.cand_origin[1] - spec.cand_bubble_diam / 2,
    ]
    ans_left = [
        ANS_BLOCK_LEFT_ORIGIN[0] - ANS_BUBBLE_DIAM / 2,
        ANS_BLOCK_LEFT_ORIGIN[1] - ANS_BUBBLE_DIAM / 2,
    ]
    ans_right = [
        ANS_BLOCK_RIGHT_ORIGIN[0] - ANS_BUBBLE_DIAM / 2,
        ANS_BLOCK_RIGHT_ORIGIN[1] - ANS_BUBBLE_DIAM / 2,
    ]
    template = {
        "pageDimensions": [OMR_W, OMR_H],
        "bubbleDimensions": [ANS_BUBBLE_DIAM, ANS_BUBBLE_DIAM],
        "customLabels": {
            "CandidateNumber": ["cand1..10"],
        },
        "outputColumns": [
            "CandidateNumber",
            "q1..25",
        ],
        "fieldBlocks": {
            "CandidateNumber": {
                "origin": cand_origin,
                "bubbleDimensions": [spec.cand_bubble_diam, spec.cand_bubble_diam],
                # CRITICAL: bubblesGap is the VERTICAL step (digits 0..9 within a column)
                # and labelsGap is the HORIZONTAL step (between digit-position columns).
                # The pre-existing in-tree template had these swapped.
                "bubblesGap": float(spec.cand_labels_gap_y),
                "labelsGap": float(spec.cand_bubbles_gap_x),
                "fieldLabels": ["cand1..10"],
                "fieldType": "QTYPE_INT",
            },
            "q01_q13_block": {
                "origin": ans_left,
                "bubblesGap": float(ANS_BUBBLES_GAP_X),
                "labelsGap": float(ANS_LABELS_GAP_Y),
                "fieldLabels": ["q1..13"],
                "emptyValue": "NR",
                "fieldType": "QTYPE_MCQ4",
            },
            "q14_q25_block": {
                "origin": ans_right,
                "bubblesGap": float(ANS_BUBBLES_GAP_X),
                "labelsGap": float(ANS_LABELS_GAP_Y),
                "fieldLabels": ["q14..25"],
                "emptyValue": "NR",
                "fieldType": "QTYPE_MCQ4",
            },
        },
        "preProcessors": [
            {
                "name": "CropOnMarkers",
                "options": {
                    "type": "aruco",
                    "arucoDictionary": "DICT_4X4_50",
                    "arucoCornerIds": [0, 1, 2, 3],
                    "preserveFullImage": True,
                    # Reference marker side is `ARUCO_MARKER_SIZE_OMR` in OMR
                    # space (matches what the generator draws). Default of 10
                    # is for 20×20 markers and would create an inconsistent
                    # per-marker vs between-marker scale that warps the page.
                    "referenceMarkerHalfSize": ARUCO_MARKER_SIZE_OMR / 2,
                    "referenceMarkerCenters": [list(c) for c in ARUCO_CENTRES],
                },
            }
        ],
    }
    path = out_dir / "template.json"
    path.write_text(json.dumps(template, indent=2))
    return path


# ---------------------------------------------------------------------------
# Default sweep specs.
# ---------------------------------------------------------------------------


def default_sweep_specs() -> list[SheetSpec]:
    """5 candidate diameters spanning 3.4 mm → 6.7 mm at 200 DPI.

    Origins/spacings are chosen so the grid fits cleanly between the
    header (which ends near y_omr=225) and the answer grid (which starts
    at y_omr=405) — all without re-flowing the rest of the page.
    """
    return [
        # (label,        diam, gap_x, gap_y, origin)
        SheetSpec("size_08",  8, 22.0, 12.0, (143, 258)),
        SheetSpec("size_10", 10, 25.0, 13.5, (133, 255)),
        SheetSpec("size_12", 12, 28.0, 15.0, (118, 252)),
        SheetSpec("size_14", 14, 31.0, 16.5, (102, 248)),
        SheetSpec("size_16", 16, 34.0, 18.0,  (86, 245)),
    ]
