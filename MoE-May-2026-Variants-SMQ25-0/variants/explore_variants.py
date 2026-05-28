"""Exploratory variant generator for the 25-question OMR answer sheet.

Generates **6 candidate designs** = 3 header styles × 2 orientations on
US Letter (8.5" × 11"). The metadata strip is intentionally minimised
(only Name / School / Exam — Region has been removed per user feedback)
so the reclaimed vertical space can grow the answer-grid bubbles.

Variants:

* **A — Two-row header**     | 3 fields stacked compactly on 2 rows
* **B — Single-line header** | 3 fields on a single horizontal line
* **C — Corner-block header**| 3 fields in a compact left-side block,
                              candidate-number grid moved up beside it

Orientations:

* Portrait  (8.5 × 11)   — 2 columns of 13 + 12 questions
* Landscape (11 × 8.5)   — 5 columns of 5 questions each (Scantron-style)

Run::

    python MoE-May-2026-Variants-SMQ25-0/variants/explore_variants.py

Outputs 6 PNGs into `MoE-May-2026-Variants-SMQ25-0/variants/`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Canvas — wraps OMR + print dimensions and shared geometry for one
# orientation. All draw helpers take a Canvas so the same painters work
# for both portrait and landscape.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Canvas:
    name: str
    omr_w: int
    omr_h: int
    print_w: int
    print_h: int
    aruco_centres: tuple[tuple[int, int], ...]

    @property
    def scale_x(self) -> float:
        return self.print_w / self.omr_w

    @property
    def scale_y(self) -> float:
        return self.print_h / self.omr_h

    def omr_to_print(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale_x, y * self.scale_y

    def bubble_bbox(
        self, cx_omr: float, cy_omr: float, diam_omr: float
    ) -> tuple[float, float, float, float]:
        cx, cy = self.omr_to_print(cx_omr, cy_omr)
        r = (diam_omr / 2.0) * self.scale_x
        return cx - r, cy - r, cx + r, cy + r


PORTRAIT = Canvas(
    name="portrait",
    omr_w=515,
    omr_h=666,
    print_w=1700,
    print_h=2200,
    aruco_centres=((40, 40), (475, 40), (40, 626), (475, 626)),
)
LANDSCAPE = Canvas(
    name="landscape",
    omr_w=666,
    omr_h=515,
    print_w=2200,
    print_h=1700,
    aruco_centres=((40, 40), (626, 40), (40, 475), (626, 475)),
)


# ---------------------------------------------------------------------------
# Shared rendering constants.
# ---------------------------------------------------------------------------

ARUCO_CORNER_IDS = (0, 1, 2, 3)
ARUCO_MARKER_SIZE_OMR = 30

FOLD_SAFE_INSET_OMR = 12
FOLD_SAFE_ARM_OMR = 18
FOLD_SAFE_STROKE_PX = 2
FOLD_SAFE_COLOR = "#bbbbbb"
FOLD_WARNING_TEXT = "Keep corners clean — do not fold or staple inside the grey bracket."
FOLD_WARNING_Y_OMR = 30

LABEL_GLYPH_COLOR = "#777777"
LABEL_FILL_COLOR = "black"
SUBTLE_COLOR = "#333333"
BUBBLE_STROKE_PX = 3

METADATA_FIELDS = ("Name", "School", "Exam")
INSTR_TEXT = "Fill bubbles completely with a #2 pencil. One answer per question."

TITLE_LINE_1 = "Ministry of Education"
TITLE_LINE_2 = "Multiple Choice Answer Sheet"


# ---------------------------------------------------------------------------
# Font loader.
# ---------------------------------------------------------------------------


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
# Variant + answer-grid layout.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnswerColumn:
    origin_x: int
    origin_y: int
    q_first: int
    q_last: int


@dataclass(frozen=True)
class Variant:
    label: str                      # "a", "b", "c"
    canvas: Canvas
    title: str
    header_renderer: Callable[[ImageDraw.ImageDraw, "Variant"], int]
    answer_columns: tuple[AnswerColumn, ...]
    answer_header_y: int            # y-OMR where "Q1–Q5" / "Q1–Q13" labels sit
    candidate_origin: tuple[int, int]
    ans_bubble_diam: int
    ans_bubbles_gap_x: float
    ans_labels_gap_y: float
    cand_bubble_diam: int = 10
    cand_bubbles_gap_x: float = 25.0
    cand_labels_gap_y: float = 13.5
    font_title_px: int = 60
    font_instr_px: int = 32
    font_header_px: int = 36
    font_qnum_px: int = 32
    font_letter_px: int = 28
    font_digit_px: int = 22


# ---------------------------------------------------------------------------
# Shared painters.
# ---------------------------------------------------------------------------


def draw_fold_safe_frame(draw: ImageDraw.ImageDraw, canvas: Canvas) -> None:
    arm = FOLD_SAFE_ARM_OMR
    inset = FOLD_SAFE_INSET_OMR
    corners = [
        (inset, inset, +1, +1),
        (canvas.omr_w - inset, inset, -1, +1),
        (inset, canvas.omr_h - inset, +1, -1),
        (canvas.omr_w - inset, canvas.omr_h - inset, -1, -1),
    ]
    for cx, cy, dx, dy in corners:
        cx_px, cy_px = canvas.omr_to_print(cx, cy)
        end_h_x_px, _ = canvas.omr_to_print(cx + dx * arm, cy)
        _, end_v_y_px = canvas.omr_to_print(cx, cy + dy * arm)
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
    warn_x = (canvas.print_w - warn_w) / 2
    _, warn_y = canvas.omr_to_print(0, FOLD_WARNING_Y_OMR)
    draw.text(
        (warn_x, warn_y),
        FOLD_WARNING_TEXT,
        fill=FOLD_SAFE_COLOR,
        font=warn_font,
        anchor="lt",
    )


def draw_aruco(img_pil: Image.Image, canvas: Canvas) -> Image.Image:
    img_cv = np.array(img_pil.convert("RGB"))[:, :, ::-1].copy()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_px = round(ARUCO_MARKER_SIZE_OMR * canvas.scale_x)
    for centre_omr, marker_id in zip(canvas.aruco_centres, ARUCO_CORNER_IDS):
        cx_px, cy_px = canvas.omr_to_print(*centre_omr)
        x0 = int(round(cx_px - marker_px / 2))
        y0 = int(round(cy_px - marker_px / 2))
        x1 = x0 + marker_px
        y1 = y0 + marker_px
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        img_cv[y0:y1, x0:x1] = marker_bgr
    return Image.fromarray(img_cv[:, :, ::-1])


def draw_title_strip(
    draw: ImageDraw.ImageDraw,
    variant: Variant,
    *,
    y_omr: int,
) -> int:
    canvas = variant.canvas
    title_font = load_font(variant.font_title_px, bold=True)
    subtitle_font = load_font(int(variant.font_title_px * 0.6), bold=False)
    instr_font = load_font(variant.font_instr_px)
    # Two-line MoE title.
    line1_w = draw.textlength(TITLE_LINE_1, font=title_font)
    _, title_y_px = canvas.omr_to_print(0, y_omr)
    draw.text(((canvas.print_w - line1_w) / 2, title_y_px), TITLE_LINE_1, fill=LABEL_FILL_COLOR, font=title_font, anchor="lt")
    subtitle_y_omr = y_omr + 22
    line2_w = draw.textlength(TITLE_LINE_2, font=subtitle_font)
    _, subtitle_y_px = canvas.omr_to_print(0, subtitle_y_omr)
    draw.text(((canvas.print_w - line2_w) / 2, subtitle_y_px), TITLE_LINE_2, fill=SUBTLE_COLOR, font=subtitle_font, anchor="lt")
    instr_y_omr = subtitle_y_omr + 18
    line_w = draw.textlength(INSTR_TEXT, font=instr_font)
    _, line_y_px = canvas.omr_to_print(0, instr_y_omr)
    draw.text(((canvas.print_w - line_w) / 2, line_y_px), INSTR_TEXT, fill=SUBTLE_COLOR, font=instr_font, anchor="lt")
    return instr_y_omr + 12


def _draw_field_line(
    draw: ImageDraw.ImageDraw,
    canvas: Canvas,
    *,
    label: str,
    label_x_omr: int,
    line_start_x_omr: int,
    line_end_x_omr: int,
    y_omr: int,
    label_font: ImageFont.FreeTypeFont,
    underline_width_px: int = 5,
) -> None:
    label_x_px, baseline_y_px = canvas.omr_to_print(label_x_omr, y_omr)
    draw.text(
        (label_x_px, baseline_y_px),
        f"{label}:",
        fill=LABEL_FILL_COLOR,
        font=label_font,
        anchor="lm",
    )
    line_x0, _ = canvas.omr_to_print(line_start_x_omr, y_omr)
    line_x1, _ = canvas.omr_to_print(line_end_x_omr, y_omr)
    underline_y_px = baseline_y_px + label_font.size * 0.35
    draw.line(
        [(line_x0, underline_y_px), (line_x1, underline_y_px)],
        fill=LABEL_FILL_COLOR,
        width=underline_width_px,
    )


def draw_candidate_grid(
    draw: ImageDraw.ImageDraw,
    variant: Variant,
    *,
    origin_x_omr: int,
    origin_y_omr: int,
    caption_offset_y: int = 15,
) -> int:
    canvas = variant.canvas
    digit_font = load_font(variant.font_digit_px)
    caption_font = load_font(variant.font_header_px - 4, bold=True)

    cap_x_omr = origin_x_omr + (variant.cand_bubbles_gap_x * 9) / 2
    cap_y_omr = origin_y_omr - caption_offset_y
    cap_x, cap_y = canvas.omr_to_print(cap_x_omr, cap_y_omr)
    draw.text(
        (cap_x, cap_y),
        "Candidate Number",
        fill=LABEL_FILL_COLOR,
        font=caption_font,
        anchor="mm",
    )
    for col in range(10):
        cx_omr = origin_x_omr + col * variant.cand_bubbles_gap_x
        for digit in range(10):
            cy_omr = origin_y_omr + digit * variant.cand_labels_gap_y
            bbox = canvas.bubble_bbox(cx_omr, cy_omr, variant.cand_bubble_diam)
            draw.ellipse(bbox, outline=LABEL_FILL_COLOR, width=BUBBLE_STROKE_PX)
            draw.text(
                ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                str(digit),
                fill=LABEL_GLYPH_COLOR,
                font=digit_font,
                anchor="mm",
            )
    return int(origin_y_omr + 9 * variant.cand_labels_gap_y + variant.cand_bubble_diam)


def draw_answer_block(
    draw: ImageDraw.ImageDraw,
    variant: Variant,
    column: AnswerColumn,
) -> None:
    canvas = variant.canvas
    letter_font = load_font(variant.font_letter_px)
    qnum_font = load_font(variant.font_qnum_px, bold=True)
    for row_idx, qnum in enumerate(range(column.q_first, column.q_last + 1)):
        cy_omr = column.origin_y + row_idx * variant.ans_labels_gap_y
        q_label_x_omr = column.origin_x - 12
        q_label_x_px, q_label_y_px = canvas.omr_to_print(q_label_x_omr, cy_omr)
        draw.text(
            (q_label_x_px, q_label_y_px),
            f"{qnum}.",
            fill=LABEL_FILL_COLOR,
            font=qnum_font,
            anchor="rm",
        )
        for option_idx, letter in enumerate("ABCD"):
            cx_omr = column.origin_x + option_idx * variant.ans_bubbles_gap_x
            bbox = canvas.bubble_bbox(cx_omr, cy_omr, variant.ans_bubble_diam)
            draw.ellipse(bbox, outline=LABEL_FILL_COLOR, width=BUBBLE_STROKE_PX)
            draw.text(
                ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                letter,
                fill=LABEL_GLYPH_COLOR,
                font=letter_font,
                anchor="mm",
            )


def draw_answer_grid(draw: ImageDraw.ImageDraw, variant: Variant) -> None:
    canvas = variant.canvas
    header_font = load_font(variant.font_header_px - 2, bold=True)
    for col in variant.answer_columns:
        draw_answer_block(draw, variant, col)
        header_x_omr = col.origin_x + variant.ans_bubbles_gap_x * 1.5
        header_x_px, header_y_px = canvas.omr_to_print(header_x_omr, variant.answer_header_y)
        label = f"Q{col.q_first}–Q{col.q_last}"
        draw.text(
            (header_x_px, header_y_px),
            label,
            fill=LABEL_FILL_COLOR,
            font=header_font,
            anchor="mm",
        )
    # Column dividers between adjacent columns.
    for i in range(len(variant.answer_columns) - 1):
        a = variant.answer_columns[i]
        b = variant.answer_columns[i + 1]
        div_x_omr = (
            a.origin_x + variant.ans_bubbles_gap_x * 3 + b.origin_x - variant.ans_bubbles_gap_x * 0.5
        ) / 2
        # Top of divider just above column headers
        div_y_top_omr = variant.answer_header_y - 8
        rows_in_col = a.q_last - a.q_first
        div_y_bot_omr = a.origin_y + variant.ans_labels_gap_y * rows_in_col + 3
        x0, y0 = canvas.omr_to_print(div_x_omr, div_y_top_omr)
        x1, y1 = canvas.omr_to_print(div_x_omr, div_y_bot_omr)
        draw.line([(x0, y0), (x1, y1)], fill="#cccccc", width=2)


# ---------------------------------------------------------------------------
# Header renderers — each variant has 1 portrait + 1 landscape function.
# ---------------------------------------------------------------------------


def header_a_portrait(draw: ImageDraw.ImageDraw, variant: Variant) -> int:
    """Variant A portrait: 3 stacked full-width rows (one field per row).

    Row order: Name → School → Exam. Underline spans almost the full
    content width so students have plenty of room for long inputs.
    """
    label_font = load_font(variant.font_header_px, bold=True)
    rows = [
        ("Name",   150, 60, 150, 475),
        ("School", 180, 60, 160, 475),
        ("Exam",   210, 60, 150, 475),
    ]
    for label, y_omr, lx, sx, ex in rows:
        _draw_field_line(
            draw,
            variant.canvas,
            label=label,
            label_x_omr=lx,
            line_start_x_omr=sx,
            line_end_x_omr=ex,
            y_omr=y_omr,
            label_font=label_font,
        )
    return 245


def header_a_landscape(draw: ImageDraw.ImageDraw, variant: Variant) -> int:
    """Variant A landscape: 3 fields on the LEFT half (x ≤ 340).

    Candidate-number grid lives on the right half via ``candidate_origin``,
    so the two sit side-by-side rather than stacked.
    """
    label_font = load_font(variant.font_header_px - 2, bold=True)
    rows = [
        (165, [("Name", 60, 145, 340)]),
        (195, [("School", 60, 160, 340)]),
        (225, [("Exam", 60, 145, 340)]),
    ]
    for y_omr, fields in rows:
        for label, lx, sx, ex in fields:
            _draw_field_line(
                draw,
                variant.canvas,
                label=label,
                label_x_omr=lx,
                line_start_x_omr=sx,
                line_end_x_omr=ex,
                y_omr=y_omr,
                label_font=label_font,
            )
    return 245


def header_b_portrait(draw: ImageDraw.ImageDraw, variant: Variant) -> int:
    """Variant B portrait: 3 fields on a single line, evenly spaced."""
    label_font = load_font(variant.font_header_px - 2, bold=True)
    y_omr = 165
    slot_x_start = 60
    slot_x_end = 455
    slots = len(METADATA_FIELDS)
    slot_w = (slot_x_end - slot_x_start) / slots
    for i, name in enumerate(METADATA_FIELDS):
        slot_x0 = slot_x_start + i * slot_w
        slot_x1 = slot_x0 + slot_w - 8
        text_x_px, baseline_y_px = variant.canvas.omr_to_print(slot_x0, y_omr)
        label_text = f"{name}:"
        draw.text(
            (text_x_px, baseline_y_px),
            label_text,
            fill=LABEL_FILL_COLOR,
            font=label_font,
            anchor="lm",
        )
        label_w_px = draw.textlength(label_text, font=label_font)
        line_x0_px = text_x_px + label_w_px + 5
        line_x1_px, _ = variant.canvas.omr_to_print(slot_x1, y_omr)
        underline_y_px = baseline_y_px + label_font.size * 0.35
        draw.line(
            [(line_x0_px, underline_y_px), (line_x1_px, underline_y_px)],
            fill=LABEL_FILL_COLOR,
            width=4,
        )
    return 195


def header_b_landscape(draw: ImageDraw.ImageDraw, variant: Variant) -> int:
    """Variant B landscape: 3 fields on a single line on the LEFT half.

    Candidate-number grid sits to the right of these fields. Slots are
    ~93 OMR-px each (39 mm) — comfortable for a name + a 6-character
    code + an exam ID.
    """
    label_font = load_font(variant.font_header_px - 4, bold=True)
    y_omr = 175
    slot_x_start = 60
    slot_x_end = 340
    slots = len(METADATA_FIELDS)
    slot_w = (slot_x_end - slot_x_start) / slots
    for i, name in enumerate(METADATA_FIELDS):
        slot_x0 = slot_x_start + i * slot_w
        slot_x1 = slot_x0 + slot_w - 6
        text_x_px, baseline_y_px = variant.canvas.omr_to_print(slot_x0, y_omr)
        label_text = f"{name}:"
        draw.text(
            (text_x_px, baseline_y_px),
            label_text,
            fill=LABEL_FILL_COLOR,
            font=label_font,
            anchor="lm",
        )
        label_w_px = draw.textlength(label_text, font=label_font)
        line_x0_px = text_x_px + label_w_px + 5
        line_x1_px, _ = variant.canvas.omr_to_print(slot_x1, y_omr)
        underline_y_px = baseline_y_px + label_font.size * 0.35
        draw.line(
            [(line_x0_px, underline_y_px), (line_x1_px, underline_y_px)],
            fill=LABEL_FILL_COLOR,
            width=4,
        )
    return 200


def header_c_portrait(draw: ImageDraw.ImageDraw, variant: Variant) -> int:
    """Variant C portrait: 3 fields stacked compactly in top-left block.

    Candidate-number grid sits to the right (placed by render()).
    """
    label_font = load_font(variant.font_header_px - 6, bold=True)
    fields = [
        ("Name", 60, 150, 235),
        ("School", 60, 180, 235),
        ("Exam", 60, 210, 235),
    ]
    for label, lx, ly, ex in fields:
        _draw_field_line(
            draw,
            variant.canvas,
            label=label,
            label_x_omr=lx,
            line_start_x_omr=lx + 60,
            line_end_x_omr=ex,
            y_omr=ly,
            label_font=label_font,
            underline_width_px=4,
        )
    return 230  # the candidate grid extends further down; render() uses max()


def header_c_landscape(draw: ImageDraw.ImageDraw, variant: Variant) -> int:
    """Variant C landscape: 3 fields stacked compactly in top-left block.

    Candidate-number grid sits to the right (placed by render()).
    """
    label_font = load_font(variant.font_header_px - 4, bold=True)
    fields = [
        ("Name", 60, 165, 285),
        ("School", 60, 195, 285),
        ("Exam", 60, 225, 285),
    ]
    for label, lx, ly, ex in fields:
        _draw_field_line(
            draw,
            variant.canvas,
            label=label,
            label_x_omr=lx,
            line_start_x_omr=lx + 70,
            line_end_x_omr=ex,
            y_omr=ly,
            label_font=label_font,
            underline_width_px=4,
        )
    return 245


# ---------------------------------------------------------------------------
# Variant catalogue (6 entries = 3 header styles × 2 orientations).
# ---------------------------------------------------------------------------


VARIANTS: list[Variant] = [
    # ----- Variant A: 2-row header -----
    Variant(
        label="a_portrait",
        canvas=PORTRAIT,
        title="Ministry of Education",
        header_renderer=header_a_portrait,
        answer_columns=(
            AnswerColumn(origin_x=85, origin_y=420, q_first=1, q_last=13),
            AnswerColumn(origin_x=305, origin_y=420, q_first=14, q_last=25),
        ),
        answer_header_y=405,
        candidate_origin=(118, 252),
        ans_bubble_diam=14,         # 5.9 mm
        ans_bubbles_gap_x=24.0,     # 10.0 mm
        ans_labels_gap_y=17.5,      # 7.3 mm
        cand_bubble_diam=12,        # 5.0 mm — sweep winner (100% all darknesses)
        cand_bubbles_gap_x=28.0,
        cand_labels_gap_y=15.0,
        font_letter_px=30,
        font_qnum_px=34,
    ),
    Variant(
        label="a_landscape",
        canvas=LANDSCAPE,
        title="OMR Answer Sheet — A · Landscape",
        header_renderer=header_a_landscape,
        answer_columns=(
            AnswerColumn(origin_x=85, origin_y=325, q_first=1, q_last=5),
            AnswerColumn(origin_x=200, origin_y=325, q_first=6, q_last=10),
            AnswerColumn(origin_x=315, origin_y=325, q_first=11, q_last=15),
            AnswerColumn(origin_x=430, origin_y=325, q_first=16, q_last=20),
            AnswerColumn(origin_x=545, origin_y=325, q_first=21, q_last=25),
        ),
        answer_header_y=305,
        candidate_origin=(380, 165),
        ans_bubble_diam=14,
        ans_bubbles_gap_x=22.0,
        ans_labels_gap_y=28.0,
        cand_bubble_diam=10,
        cand_bubbles_gap_x=22.0,
        cand_labels_gap_y=11.0,
        font_letter_px=30,
        font_qnum_px=34,
    ),
    # ----- Variant B: single-line header -----
    Variant(
        label="b_portrait",
        canvas=PORTRAIT,
        title="OMR Answer Sheet — B · Portrait",
        header_renderer=header_b_portrait,
        answer_columns=(
            AnswerColumn(origin_x=85, origin_y=395, q_first=1, q_last=13),
            AnswerColumn(origin_x=305, origin_y=395, q_first=14, q_last=25),
        ),
        answer_header_y=380,
        candidate_origin=(133, 230),
        ans_bubble_diam=15,         # 6.3 mm
        ans_bubbles_gap_x=26.0,     # 10.9 mm
        ans_labels_gap_y=18.5,      # 7.8 mm
        font_letter_px=32,
        font_qnum_px=36,
    ),
    Variant(
        label="b_landscape",
        canvas=LANDSCAPE,
        title="OMR Answer Sheet — B · Landscape",
        header_renderer=header_b_landscape,
        answer_columns=(
            AnswerColumn(origin_x=85, origin_y=300, q_first=1, q_last=5),
            AnswerColumn(origin_x=200, origin_y=300, q_first=6, q_last=10),
            AnswerColumn(origin_x=315, origin_y=300, q_first=11, q_last=15),
            AnswerColumn(origin_x=430, origin_y=300, q_first=16, q_last=20),
            AnswerColumn(origin_x=545, origin_y=300, q_first=21, q_last=25),
        ),
        answer_header_y=280,
        candidate_origin=(380, 165),
        ans_bubble_diam=15,
        ans_bubbles_gap_x=22.0,
        ans_labels_gap_y=27.0,
        cand_bubble_diam=10,
        cand_bubbles_gap_x=22.0,
        cand_labels_gap_y=11.0,
        font_letter_px=32,
        font_qnum_px=36,
    ),
    # ----- Variant C: corner-block header -----
    Variant(
        label="c_portrait",
        canvas=PORTRAIT,
        title="OMR Answer Sheet — C · Portrait",
        header_renderer=header_c_portrait,
        answer_columns=(
            AnswerColumn(origin_x=85, origin_y=395, q_first=1, q_last=13),
            AnswerColumn(origin_x=305, origin_y=395, q_first=14, q_last=25),
        ),
        answer_header_y=380,
        candidate_origin=(280, 165),
        ans_bubble_diam=16,         # 6.7 mm
        ans_bubbles_gap_x=28.0,     # 11.7 mm
        ans_labels_gap_y=19.5,      # 8.2 mm
        cand_bubbles_gap_x=20.0,
        cand_labels_gap_y=14.0,
        font_letter_px=34,
        font_qnum_px=38,
    ),
    Variant(
        label="c_landscape",
        canvas=LANDSCAPE,
        title="OMR Answer Sheet — C · Landscape",
        header_renderer=header_c_landscape,
        answer_columns=(
            AnswerColumn(origin_x=85, origin_y=295, q_first=1, q_last=5),
            AnswerColumn(origin_x=200, origin_y=295, q_first=6, q_last=10),
            AnswerColumn(origin_x=315, origin_y=295, q_first=11, q_last=15),
            AnswerColumn(origin_x=430, origin_y=295, q_first=16, q_last=20),
            AnswerColumn(origin_x=545, origin_y=295, q_first=21, q_last=25),
        ),
        answer_header_y=275,
        candidate_origin=(340, 165),
        ans_bubble_diam=16,
        ans_bubbles_gap_x=22.0,
        ans_labels_gap_y=28.0,
        cand_bubble_diam=10,
        cand_bubbles_gap_x=22.0,
        cand_labels_gap_y=11.0,
        font_letter_px=34,
        font_qnum_px=38,
    ),
]


# ---------------------------------------------------------------------------
# Renderer.
# ---------------------------------------------------------------------------


def render(variant: Variant) -> Image.Image:
    canvas = variant.canvas
    img = Image.new("RGB", (canvas.print_w, canvas.print_h), "white")
    draw = ImageDraw.Draw(img)

    draw_fold_safe_frame(draw, canvas)
    draw_title_strip(draw, variant, y_omr=65)

    variant.header_renderer(draw, variant)
    draw_candidate_grid(
        draw,
        variant,
        origin_x_omr=variant.candidate_origin[0],
        origin_y_omr=variant.candidate_origin[1],
        caption_offset_y=15,
    )
    draw_answer_grid(draw, variant)
    img = draw_aruco(img, canvas)
    return img


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        img = render(variant)
        out_path = out_dir / f"variant_{variant.label}.png"
        img.save(out_path, format="PNG", compress_level=1)
        print(
            f"Wrote {out_path.name:30s} | "
            f"orientation={variant.canvas.name:9s} | "
            f"bubble Ø={variant.ans_bubble_diam:>2d} OMR-px "
            f"({variant.ans_bubble_diam * 0.419:.1f} mm), "
            f"row gap={variant.ans_labels_gap_y} OMR-px "
            f"({variant.ans_labels_gap_y * 0.419:.1f} mm)"
        )


if __name__ == "__main__":
    main()
