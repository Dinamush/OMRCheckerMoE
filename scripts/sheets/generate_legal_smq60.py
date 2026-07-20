"""Generate MoE Legal 60Q answer sheets (landscape + portrait).

Produces blank PNG/PDF, template.json, and config.json under:

* ``MoE-July-2026-Landscape-SMQ60-0/``
* ``MoE-July-2026-Portrait-SMQ60-0/``

OMR coordinates match what is drawn so the templates are scan-ready once
registered. Not wired into the WebUI yet.

Run from repo root::

    python scripts/sheets/generate_legal_smq60.py
    python scripts/sheets/generate_legal_smq60.py --orientation landscape
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]

PRINT_DPI = 300
# Legal inches → print pixels @ 300 DPI
LEGAL_SHORT_IN, LEGAL_LONG_IN = 8.5, 14.0
PRINT_SHORT = int(LEGAL_SHORT_IN * PRINT_DPI)  # 2550
PRINT_LONG = int(LEGAL_LONG_IN * PRINT_DPI)  # 4200

# OMR canvas: ~60 px/inch so 0.5" = 30 OMR-px (matches design inset).
OMR_PX_PER_IN = 60.0
OMR_SHORT = int(round(LEGAL_SHORT_IN * OMR_PX_PER_IN))  # 510
OMR_LONG = int(round(LEGAL_LONG_IN * OMR_PX_PER_IN))  # 840

ARUCO_DICT = cv2.aruco.DICT_4X4_50
ARUCO_IDS = [0, 1, 2, 3]
ARUCO_INSET_OMR = 30.0  # 0.5" marker centres (robustness)
ARUCO_SIZE_OMR = 28

# Bubble diameters (OMR-px). Landscape gets the larger wells; portrait is
# constrained by the short Legal edge width. Keep enough row pitch
# (labelsGap ≳ diam + 2) so blank scans do not false-trigger.
BUBBLE_DIAM_LANDSCAPE = 18
# Portrait short-edge width: leave enough inter-column room for "60." labels
# so they do not paint into the previous column's D bubble (false D on blank).
BUBBLE_DIAM_PORTRAIT = 13
# Candidate wells are 1.5× smaller than answer wells (answer / 1.5).
CAND_TO_ANSWER_SCALE = 1.5
BUBBLE_STROKE_PRINT = 3
# Extra content inset beyond the marker outer edge (was 12 — tightened).
CONTENT_PAST_MARKER_OMR = 2.0

TITLE = "Ministry of Education - Multiple Choice Answer Sheet"
INSTRUCTIONS = (
    "Write your name clearly. Fill one bubble for each answer. Use a dark pencil."
)

LABEL_FILL = "black"
# Mid-grey in-bubble glyphs: readable on print, still below OMR mark threshold
# when spacing is sound (verified on blank Legal SMQ60 scans).
LABEL_GLYPH = "#787878"
SUBTLE = "#333333"
BOX_STROKE = "#111111"
FOLD_SAFE = "#bbbbbb"


@dataclass(frozen=True)
class OrientationLayout:
    name: str
    folder: str
    omr_w: int
    omr_h: int
    print_w: int
    print_h: int


LANDSCAPE = OrientationLayout(
    name="landscape",
    folder="MoE-July-2026-Landscape-SMQ60-0",
    omr_w=OMR_LONG,
    omr_h=OMR_SHORT,
    print_w=PRINT_LONG,
    print_h=PRINT_SHORT,
)
PORTRAIT = OrientationLayout(
    name="portrait",
    folder="MoE-July-2026-Portrait-SMQ60-0",
    omr_w=OMR_SHORT,
    omr_h=OMR_LONG,
    print_w=PRINT_SHORT,
    print_h=PRINT_LONG,
)


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    bold_candidates = [
        "C:/Windows/Fonts/arialbd.ttf",
        "arialbd.ttf",
        "DejaVuSans-Bold.ttf",
    ]
    regular_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "arial.ttf",
        "DejaVuSans.ttf",
    ]
    for path in bold_candidates if bold else regular_candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def aruco_centres(layout: OrientationLayout) -> list[tuple[float, float]]:
    i = ARUCO_INSET_OMR
    return [
        (i, i),
        (layout.omr_w - i, i),
        (i, layout.omr_h - i),
        (layout.omr_w - i, layout.omr_h - i),
    ]


def scale_xy(layout: OrientationLayout) -> tuple[float, float]:
    return layout.print_w / layout.omr_w, layout.print_h / layout.omr_h


def to_print(
    layout: OrientationLayout, x_omr: float, y_omr: float
) -> tuple[float, float]:
    sx, sy = scale_xy(layout)
    return x_omr * sx, y_omr * sy


def bubble_diam_for(layout: OrientationLayout) -> float:
    return float(
        BUBBLE_DIAM_LANDSCAPE
        if layout.name == "landscape"
        else BUBBLE_DIAM_PORTRAIT
    )


def cand_bubble_diam_for(layout: OrientationLayout) -> float:
    """Candidate bubbles are ``CAND_TO_ANSWER_SCALE``× smaller than answers."""
    return bubble_diam_for(layout) / CAND_TO_ANSWER_SCALE


def bubble_bbox_from_origin(
    layout: OrientationLayout, x0: float, y0: float, diam: float | None = None
) -> tuple[float, float, float, float]:
    """Print-space ellipse bbox for a bubble whose OMR top-left is (x0, y0)."""
    if diam is None:
        diam = bubble_diam_for(layout)
    sx, sy = scale_xy(layout)
    x0p, y0p = x0 * sx, y0 * sy
    return x0p, y0p, x0p + diam * sx, y0p + diam * sy


def pt_to_px(pt: float) -> int:
    """Convert typographic points to print pixels at PRINT_DPI."""
    return max(8, int(round(pt * PRINT_DPI / 72.0)))


def fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    max_width_px: float,
    start_pt: float,
    min_pt: float = 8.0,
    bold: bool = False,
) -> ImageFont.ImageFont:
    """Shrink a font until ``text`` fits within ``max_width_px``."""
    pt = start_pt
    while pt >= min_pt:
        font = load_font(pt_to_px(pt), bold=bold)
        if draw.textlength(text, font=font) <= max_width_px:
            return font
        pt -= 0.5
    return load_font(pt_to_px(min_pt), bold=bold)


def content_margin(layout: OrientationLayout) -> float:
    """Left/right/bottom content inset past marker outer edge."""
    return ARUCO_INSET_OMR + ARUCO_SIZE_OMR / 2 + CONTENT_PAST_MARKER_OMR


def compute_geometry(layout: OrientationLayout) -> dict:
    """Derive all OMR coordinates for drawing + template export."""
    diam = bubble_diam_for(layout)
    cand_diam = cand_bubble_diam_for(layout)
    m = content_margin(layout)
    right = layout.omr_w - m
    bottom = layout.omr_h - m
    # Keep bubbles clear of the answer-box stroke (avoids blank false-D).
    box_inner_pad = 4.0
    usable_w = (right - m) - 2 * box_inner_pad
    n_cols, n_rows = 6, 10

    # Title centred on the top marker mid-line (between left/right ArUco).
    # Instructions sit just under the title still inside the marker band.
    title_y = ARUCO_INSET_OMR
    instr_y = ARUCO_INSET_OMR + 10
    marker_bottom = ARUCO_INSET_OMR + ARUCO_SIZE_OMR / 2
    header_top = marker_bottom + CONTENT_PAST_MARKER_OMR

    if layout.name == "landscape":
        # Compact header; larger bubbles; spread answer columns.
        cand_labels_gap = cand_diam + 4.0
        cand_bubbles_gap = cand_diam + 1.5
        cand_w = cand_labels_gap * 9 + cand_diam
        cand_h = cand_bubbles_gap * 9 + cand_diam
        write_in_h = max(12.0, cand_diam + 2)
        side_pad = max(2.0, (cand_labels_gap - cand_diam) / 2)
        caption_h = 12.0
        cand_origin = (
            right - cand_w - side_pad - 2,
            header_top + caption_h + write_in_h + 6,
        )
        cand_table = {
            "x0": cand_origin[0] - side_pad,
            "y0": cand_origin[1] - write_in_h,
            "x1": cand_origin[0] + cand_w + side_pad,
            "y1": cand_origin[1] + cand_h + side_pad,
            "write_in_h": write_in_h,
            "side_pad": side_pad,
        }
        identity_line_end = cand_table["x0"] - 8
        # Identity fields need room for four spaced write-in lines.
        identity_bottom = header_top + 150
        header_bottom = max(
            identity_bottom,
            header_top + max(110.0, (cand_table["y1"] - header_top) + 8),
        )
        ans_top = header_bottom + 5
        ans_bottom = bottom - 2
        # Room for right-aligned "60." between prev D and this column's A.
        q_label_pad = 16.0
        # Keep A–D compact; spend leftover width on inter-column gutters
        # so the six columns fill the answer rectangle evenly.
        ans_bubbles_gap = diam + 2.0
        option_span = 3 * ans_bubbles_gap + diam
        col_body = q_label_pad + option_span
        min_gutter = 4.0  # must clear two-digit question labels
        total_col_bodies = n_cols * col_body
        if total_col_bodies + (n_cols - 1) * min_gutter > usable_w:
            overflow = total_col_bodies + (n_cols - 1) * min_gutter - usable_w
            ans_bubbles_gap = max(
                diam + 1.0, ans_bubbles_gap - overflow / (3 * n_cols)
            )
            option_span = 3 * ans_bubbles_gap + diam
            col_body = q_label_pad + option_span
            total_col_bodies = n_cols * col_body
        leftover = usable_w - total_col_bodies
        gutter = leftover / (n_cols - 1) if n_cols > 1 else min_gutter
        if gutter < min_gutter:
            raise ValueError(
                f"landscape: inter-column gutter too tight "
                f"(gutter={gutter:.2f}, min={min_gutter})"
            )
        total_needed = total_col_bodies + (n_cols - 1) * gutter
        group_start = m + box_inner_pad
        block_origins_x = [
            group_start + q_label_pad + i * (col_body + gutter) for i in range(n_cols)
        ]
        row_inset = 5.0
        row_span = ans_bottom - ans_top - 2 * row_inset - diam
        ans_labels_gap = row_span / (n_rows - 1)
        if ans_labels_gap < diam + 2.0:
            raise ValueError(
                f"landscape: row pitch too tight "
                f"(labelsGap={ans_labels_gap:.2f}, diam={diam})"
            )
        ans_origin_y = ans_top + row_inset
    else:
        cand_labels_gap = cand_diam + 2.5
        cand_bubbles_gap = cand_diam + 1.0
        cand_w = cand_labels_gap * 9 + cand_diam
        cand_h = cand_bubbles_gap * 9 + cand_diam
        write_in_h = max(10.0, cand_diam + 1.5)
        side_pad = max(1.5, (cand_labels_gap - cand_diam) / 2)
        caption_h = 11.0
        cand_origin = (
            right - cand_w - side_pad - 2,
            header_top + caption_h + write_in_h + 5,
        )
        cand_table = {
            "x0": cand_origin[0] - side_pad,
            "y0": cand_origin[1] - write_in_h,
            "x1": cand_origin[0] + cand_w + side_pad,
            "y1": cand_origin[1] + cand_h + side_pad,
            "write_in_h": write_in_h,
            "side_pad": side_pad,
        }
        identity_line_end = cand_table["x0"] - 8
        identity_bottom = header_top + 150
        header_bottom = max(
            identity_bottom,
            header_top + max(100.0, (cand_table["y1"] - header_top) + 8),
        )
        ans_top = header_bottom + 6
        ans_bottom = bottom - 2
        # q_label_pad + gutter must fit two-digit labels ("60.") without
        # overlapping the previous column's D bubble.
        q_label_pad = 11.5
        gutter = 2.8
        # Solve A–D pitch so 6 columns + label clearance fit exactly.
        # col_body = q_label_pad + 3*gap + diam
        ans_bubbles_gap = (
            usable_w - (n_cols - 1) * gutter - n_cols * (q_label_pad + diam)
        ) / (3 * n_cols)
        if ans_bubbles_gap < diam + 0.75:
            raise ValueError(
                f"portrait: A–D pitch too tight "
                f"(gap={ans_bubbles_gap:.2f}, diam={diam}, usable={usable_w:.1f})"
            )
        option_span = 3 * ans_bubbles_gap + diam
        col_body = q_label_pad + option_span
        total_needed = n_cols * col_body + (n_cols - 1) * gutter
        group_start = m + box_inner_pad + max(0.0, (usable_w - total_needed) / 2)
        block_origins_x = [
            group_start + q_label_pad + i * (col_body + gutter) for i in range(n_cols)
        ]
        # Clear the rounded answer-box stroke on the first/last rows
        # (otherwise corner questions blank-scan as false D).
        row_inset = 8.0
        row_span = ans_bottom - ans_top - 2 * row_inset - diam
        ans_labels_gap = row_span / (n_rows - 1)
        ans_origin_y = ans_top + row_inset

    last_opt_right = block_origins_x[-1] + 3 * ans_bubbles_gap + diam
    if last_opt_right > right - box_inner_pad + 0.01:
        raise ValueError(
            f"{layout.name}: answer columns overflow page "
            f"(last_opt_right={last_opt_right:.1f} > right={right:.1f}; "
            f"diam={diam}, gap={ans_bubbles_gap:.2f})"
        )

    q_blocks = []
    for i in range(n_cols):
        q0 = i * n_rows + 1
        q1 = q0 + n_rows - 1
        q_blocks.append(
            {
                "name": f"q{q0:02d}block",
                "origin": [round(block_origins_x[i], 2), round(ans_origin_y, 2)],
                "bubblesGap": round(float(ans_bubbles_gap), 2),
                "labelsGap": round(float(ans_labels_gap), 2),
                "fieldLabels": [f"q{q0}..{q1}"],
                "first_q": q0,
                "last_q": q1,
            }
        )

    return {
        "bubble_diam": diam,
        "cand_bubble_diam": cand_diam,
        "title_y": title_y,
        "instr_y": instr_y,
        "header_top": header_top,
        "header_bottom": header_bottom,
        "margin": m,
        "cand_origin": cand_origin,
        "cand_table": cand_table,
        "cand_labels_gap": cand_labels_gap,
        "cand_bubbles_gap": cand_bubbles_gap,
        "cand_w": cand_w,
        "cand_h": cand_h,
        "identity_line_end": identity_line_end,
        "ans_top": ans_top,
        "ans_bubbles_gap": ans_bubbles_gap,
        "ans_labels_gap": ans_labels_gap,
        "q_label_pad": q_label_pad,
        "q_blocks": q_blocks,
        "aruco_centres": aruco_centres(layout),
        "title_max_x0": ARUCO_INSET_OMR + ARUCO_SIZE_OMR / 2 + 6,
        "title_max_x1": layout.omr_w - (ARUCO_INSET_OMR + ARUCO_SIZE_OMR / 2 + 6),
    }


def draw_rounded_rect(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[float, float, float, float],
    *,
    radius: float,
    width: int,
) -> None:
    draw.rounded_rectangle(bbox, radius=radius, outline=BOX_STROKE, width=width)


def draw_fold_brackets(draw: ImageDraw.ImageDraw, layout: OrientationLayout) -> None:
    inset = 12
    arm = 18
    sx, sy = scale_xy(layout)
    stroke = max(2, int(round(2 * sx / 3)))
    corners = [
        (inset, inset, 1, 1),
        (layout.omr_w - inset, inset, -1, 1),
        (inset, layout.omr_h - inset, 1, -1),
        (layout.omr_w - inset, layout.omr_h - inset, -1, -1),
    ]
    for cx, cy, dx, dy in corners:
        x0, y0 = to_print(layout, cx, cy)
        x1, _ = to_print(layout, cx + dx * arm, cy)
        _, y1 = to_print(layout, cx, cy + dy * arm)
        draw.line([(x0, y0), (x1, y0)], fill=FOLD_SAFE, width=stroke)
        draw.line([(x0, y0), (x0, y1)], fill=FOLD_SAFE, width=stroke)


def stamp_aruco(img: Image.Image, layout: OrientationLayout, centres) -> Image.Image:
    sx, _ = scale_xy(layout)
    marker_px = max(24, int(round(ARUCO_SIZE_OMR * sx)))
    # Prefer odd/even consistency for generateImageMarker
    if marker_px % 2:
        marker_px += 1
    arr = np.array(img.convert("RGB"))[:, :, ::-1].copy()
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    for (cx, cy), mid in zip(centres, ARUCO_IDS):
        px, py = to_print(layout, cx, cy)
        x0 = int(round(px - marker_px / 2))
        y0 = int(round(py - marker_px / 2))
        marker = cv2.aruco.generateImageMarker(dictionary, mid, marker_px)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        arr[y0 : y0 + marker_px, x0 : x0 + marker_px] = marker_bgr
    return Image.fromarray(arr[:, :, ::-1])


def draw_header(
    draw: ImageDraw.ImageDraw, layout: OrientationLayout, geo: dict
) -> None:
    m = geo["margin"]
    # Title/instructions sit between the top markers — use that band width.
    title_left_px, _ = to_print(layout, geo["title_max_x0"], 0)
    title_right_px, _ = to_print(layout, geo["title_max_x1"], 0)
    max_title_w = title_right_px - title_left_px

    title_font = fit_font(
        draw,
        TITLE,
        max_width_px=max_title_w,
        start_pt=14 if layout.name == "portrait" else 15,
        min_pt=9,
        bold=True,
    )
    instr_font = fit_font(
        draw,
        INSTRUCTIONS,
        max_width_px=max_title_w,
        start_pt=8 if layout.name == "portrait" else 8.5,
        min_pt=7,
        bold=False,
    )
    label_font = load_font(pt_to_px(10 if layout.name == "portrait" else 11), bold=True)

    _, ty = to_print(layout, 0, geo["title_y"])
    draw.text(
        (layout.print_w / 2, ty),
        TITLE,
        fill=LABEL_FILL,
        font=title_font,
        anchor="mm",
    )
    _, iy = to_print(layout, 0, geo["instr_y"])
    draw.text(
        (layout.print_w / 2, iy),
        INSTRUCTIONS,
        fill=SUBTLE,
        font=instr_font,
        anchor="mm",
    )

    x0, y0 = to_print(layout, m, geo["header_top"])
    x1, y1 = to_print(layout, layout.omr_w - m, geo["header_bottom"])
    draw_rounded_rect(draw, (x0, y0, x1, y1), radius=14 * scale_xy(layout)[0], width=3)

    cand_x0 = geo["cand_table"]["x0"]
    sx0, _ = to_print(layout, cand_x0, geo["header_top"])
    draw.line([(sx0, y0 + 8), (sx0, y1 - 8)], fill=BOX_STROKE, width=2)

    # Spaced write-in lines (~38 OMR-px apart) for comfortable handwriting.
    field_y = [
        geo["header_top"] + 24,
        geo["header_top"] + 62,
        geo["header_top"] + 100,
        geo["header_top"] + 138,
    ]
    labels = ["Student Name:", "School Name:", "Exam Name:", "Student Signature:"]
    line_x0 = m + (78 if layout.name == "portrait" else 95)
    line_x1 = geo["identity_line_end"]
    for label, y in zip(labels, field_y):
        lx, ly = to_print(layout, m + 8, y)
        draw.text((lx, ly), label, fill=LABEL_FILL, font=label_font, anchor="lm")
        x_a, y_a = to_print(layout, line_x0, y + (4 if "Signature" in label else 7))
        x_b, _ = to_print(layout, line_x1, y)
        draw.line([(x_a, y_a), (x_b, y_a)], fill=LABEL_FILL, width=2)

    draw_candidate_table(draw, layout, geo)


def draw_candidate_table(
    draw: ImageDraw.ImageDraw, layout: OrientationLayout, geo: dict
) -> None:
    """Legacy-style candidate grid: write-in row + column lines + digit bubbles."""
    cand_diam = geo["cand_bubble_diam"]
    table = geo["cand_table"]
    cx0, cy0 = geo["cand_origin"]
    pitch = geo["cand_labels_gap"]
    v_gap = geo["cand_bubbles_gap"]
    write_in_h = table["write_in_h"]
    stroke = max(2, int(round(2 * scale_xy(layout)[0] / 2.5)))

    cap_font = load_font(pt_to_px(10 if layout.name == "portrait" else 11), bold=True)
    digit_font = load_font(pt_to_px(6 if layout.name == "portrait" else 6.5))

    # Caption sits just above the table, left-aligned like the April sheet.
    cap_x, cap_y = to_print(layout, table["x0"], table["y0"] - 4)
    draw.text(
        (cap_x, cap_y),
        "Candidate Number",
        fill=LABEL_FILL,
        font=cap_font,
        anchor="lb",
    )

    tx0, ty0 = to_print(layout, table["x0"], table["y0"])
    tx1, ty1 = to_print(layout, table["x1"], table["y1"])
    write_y = table["y0"] + write_in_h
    _, wy = to_print(layout, 0, write_y)

    draw.rectangle((tx0, ty0, tx1, ty1), outline=BOX_STROKE, width=stroke)
    draw.line([(tx0, wy), (tx1, wy)], fill=BOX_STROKE, width=stroke)

    # Vertical column dividers (9 interior lines).
    for col in range(1, 10):
        vx = table["x0"] + col * pitch
        px, _ = to_print(layout, vx, 0)
        draw.line([(px, ty0), (px, ty1)], fill=BOX_STROKE, width=stroke)

    for col in range(10):
        for digit in range(10):
            x0b = cx0 + col * pitch
            y0b = cy0 + digit * v_gap
            bbox = bubble_bbox_from_origin(layout, x0b, y0b, cand_diam)
            draw.ellipse(bbox, outline=LABEL_FILL, width=BUBBLE_STROKE_PRINT)
            draw.text(
                ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                str(digit),
                fill=LABEL_GLYPH,
                font=digit_font,
                anchor="mm",
            )


def draw_answers(
    draw: ImageDraw.ImageDraw, layout: OrientationLayout, geo: dict
) -> None:
    diam = geo["bubble_diam"]
    m = geo["margin"]
    x0, y0 = to_print(layout, m, geo["ans_top"])
    x1, y1 = to_print(layout, layout.omr_w - m, layout.omr_h - m)
    draw_rounded_rect(draw, (x0, y0, x1, y1), radius=18 * scale_xy(layout)[0], width=3)

    qnum_font = load_font(pt_to_px(9 if layout.name == "portrait" else 10), bold=True)
    letter_font = load_font(pt_to_px(6.5 if layout.name == "portrait" else 7.5))

    for block in geo["q_blocks"]:
        ox, oy = block["origin"]
        for row, qnum in enumerate(range(block["first_q"], block["last_q"] + 1)):
            y = oy + row * block["labelsGap"]
            # Question number left of first bubble
            qx, qy = to_print(layout, ox - 4, y + diam / 2)
            draw.text(
                (qx, qy),
                f"{qnum}.",
                fill=LABEL_FILL,
                font=qnum_font,
                anchor="rm",
            )
            for oi, letter in enumerate("ABCD"):
                bx = ox + oi * block["bubblesGap"]
                bbox = bubble_bbox_from_origin(layout, bx, y, diam)
                draw.ellipse(bbox, outline=LABEL_FILL, width=BUBBLE_STROKE_PRINT)
                draw.text(
                    ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                    letter,
                    fill=LABEL_GLYPH,
                    font=letter_font,
                    anchor="mm",
                )


def build_template(layout: OrientationLayout, geo: dict) -> dict:
    diam = int(round(geo["bubble_diam"]))
    # Keep one decimal so portrait 13/1.5 stays exact (8.7), not rounded to 9.
    cand_diam = round(float(geo["cand_bubble_diam"]), 1)
    centres = geo["aruco_centres"]
    field_blocks = {
        "CandidateNumber": {
            "origin": [
                round(geo["cand_origin"][0], 2),
                round(geo["cand_origin"][1], 2),
            ],
            "bubbleDimensions": [cand_diam, cand_diam],
            "bubblesGap": float(geo["cand_bubbles_gap"]),
            "labelsGap": float(geo["cand_labels_gap"]),
            "fieldLabels": ["cand1..10"],
            "fieldType": "QTYPE_INT",
        }
    }
    for block in geo["q_blocks"]:
        field_blocks[block["name"]] = {
            "origin": block["origin"],
            "bubblesGap": float(block["bubblesGap"]),
            "labelsGap": float(block["labelsGap"]),
            "fieldLabels": block["fieldLabels"],
            "emptyValue": "NR",
            "fieldType": "QTYPE_MCQ4",
        }

    return {
        "pageDimensions": [layout.omr_w, layout.omr_h],
        "bubbleDimensions": [diam, diam],
        "customLabels": {"CandidateNumber": ["cand1..10"]},
        "outputColumns": ["CandidateNumber", "q1..60"],
        "fieldBlocks": field_blocks,
        "preProcessors": [
            {
                "name": "CropOnMarkers",
                "options": {
                    "type": "aruco",
                    "arucoDictionary": "DICT_4X4_50",
                    "arucoCornerIds": ARUCO_IDS,
                    "preserveFullImage": True,
                    "referenceMarkerCenters": [
                        [round(c[0], 2), round(c[1], 2)] for c in centres
                    ],
                    "referenceMarkerHalfSize": ARUCO_SIZE_OMR / 2,
                },
            }
        ],
    }


def build_config(layout: OrientationLayout) -> dict:
    return {
        "dimensions": {
            "display_height": layout.omr_h,
            "display_width": layout.omr_w,
            "processing_height": layout.omr_h,
            "processing_width": layout.omr_w,
        },
        "threshold_params": {"OVERSAMPLE_SCALE": 2.0},
        "outputs": {"show_image_level": 0},
    }


def render_blank(layout: OrientationLayout, geo: dict) -> Image.Image:
    img = Image.new("RGB", (layout.print_w, layout.print_h), "white")
    draw = ImageDraw.Draw(img)
    draw_fold_brackets(draw, layout)
    draw_header(draw, layout, geo)
    draw_answers(draw, layout, geo)
    img = stamp_aruco(img, layout, geo["aruco_centres"])
    return img


def write_alignment_preview(
    layout: OrientationLayout, geo: dict, blank: Image.Image, out_path: Path
) -> None:
    """Overlay template bubble boxes on a downscaled blank for visual QA."""
    diam = geo["bubble_diam"]
    cand_diam = geo["cand_bubble_diam"]
    preview = blank.copy()
    # Scale preview to OMR size for 1:1 box overlay, then upscale for viewing
    omr_img = preview.resize((layout.omr_w, layout.omr_h), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(omr_img)
    # Candidate
    ox, oy = geo["cand_origin"]
    for col in range(10):
        for row in range(10):
            x = ox + col * geo["cand_labels_gap"]
            y = oy + row * geo["cand_bubbles_gap"]
            draw.rectangle(
                [x, y, x + cand_diam, y + cand_diam], outline="#00aa00", width=1
            )
    for block in geo["q_blocks"]:
        bx, by = block["origin"]
        for row in range(10):
            for opt in range(4):
                x = bx + opt * block["bubblesGap"]
                y = by + row * block["labelsGap"]
                draw.rectangle(
                    [x, y, x + diam, y + diam],
                    outline="#0066ff",
                    width=1,
                )
    # Upscale for readability
    view = omr_img.resize(
        (layout.omr_w * 2, layout.omr_h * 2), Image.Resampling.NEAREST
    )
    view.save(out_path, format="PNG", optimize=True)


def generate_one(layout: OrientationLayout) -> Path:
    out_dir = REPO / layout.folder
    ref_dir = out_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    geo = compute_geometry(layout)
    blank = render_blank(layout, geo)

    png_path = ref_dir / f"blank_{layout.name}_smq60.png"
    pdf_path = ref_dir / f"blank_{layout.name}_smq60.pdf"
    preview_path = ref_dir / f"alignment_preview_{layout.name}.png"

    blank.save(png_path, format="PNG", optimize=True)
    blank.save(pdf_path, format="PDF", resolution=float(PRINT_DPI))
    write_alignment_preview(layout, geo, blank, preview_path)

    template = build_template(layout, geo)
    config = build_config(layout)
    (out_dir / "template.json").write_text(
        json.dumps(template, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    # Thin per-folder regenerate helper
    helper = f'''"""Regenerate the {layout.name} Legal 60Q blank + template.

Run from repo root::

    python {layout.folder}/generate_blank.py
"""
from __future__ import annotations

import runpy
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sheets" / "generate_legal_smq60.py"

if __name__ == "__main__":
    # Re-exec shared generator for this orientation only.
    import sys
    sys.argv = [str(SCRIPT), "--orientation", "{layout.name}"]
    runpy.run_path(str(SCRIPT), run_name="__main__")
'''
    (out_dir / "generate_blank.py").write_text(helper, encoding="utf-8")

    readme = f"""# {layout.folder}

US Legal **{layout.name}** MoE answer sheet — **60** MCQ questions (6×10).

## Contents

| File | Role |
|---|---|
| `template.json` | OMR field map (ArUco CropOnMarkers) |
| `config.json` | Processing dimensions |
| `reference/blank_{layout.name}_smq60.png` | Printable blank |
| `reference/blank_{layout.name}_smq60.pdf` | Printable PDF @ 300 DPI |
| `reference/alignment_preview_{layout.name}.png` | Template-bubble overlay QA |
| `generate_blank.py` | Regenerate from shared script |

## Notes

- Not registered in WebUI / `sheet_registry` yet (pending MoE approval).
- Marker centres inset ~0.5″ from page edges.
- Shared generator: `scripts/sheets/generate_legal_smq60.py`
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"[{layout.name}] wrote {out_dir}")
    print(f"  PNG  {png_path}  ({blank.size[0]}x{blank.size[1]})")
    print(f"  PDF  {pdf_path}")
    print(f"  QA   {preview_path}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orientation",
        choices=("landscape", "portrait", "both"),
        default="both",
    )
    args = parser.parse_args()
    targets = []
    if args.orientation in ("landscape", "both"):
        targets.append(LANDSCAPE)
    if args.orientation in ("portrait", "both"):
        targets.append(PORTRAIT)
    for layout in targets:
        generate_one(layout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
