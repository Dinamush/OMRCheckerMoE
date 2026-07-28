"""Generate MoE Letter-landscape 60Q answer sheet (SMQ60).

Produces blank PNG/PDF, bubbled sample, template.json, and config.json under:

* ``MoE-July-2026-Letter-Landscape-SMQ60-0/``

Legal July SMQ60 sheets are unchanged (see ``generate_legal_smq60.py``).

Run from repo root::

    python scripts/sheets/generate_letter_smq60.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]

PRINT_DPI = 300
LETTER_SHORT_IN, LETTER_LONG_IN = 8.5, 11.0
PRINT_SHORT = int(LETTER_SHORT_IN * PRINT_DPI)  # 2550
PRINT_LONG = int(LETTER_LONG_IN * PRINT_DPI)  # 3300

OMR_PX_PER_IN = 60.0
OMR_SHORT = int(round(LETTER_SHORT_IN * OMR_PX_PER_IN))  # 510
OMR_LONG = int(round(LETTER_LONG_IN * OMR_PX_PER_IN))  # 660

ARUCO_DICT = cv2.aruco.DICT_4X4_50
ARUCO_IDS = [0, 1, 2, 3]
ARUCO_INSET_OMR = 30.0
# Match April 2026 legacy print size: 108 px @ 300 DPI ≈ 0.36".
ARUCO_SIZE_OMR = 108.0 / (PRINT_DPI / OMR_PX_PER_IN)

# Letter is 3" narrower than Legal — slightly smaller answer wells.
BUBBLE_DIAM = 16.0
# Legal landscape cand ≈ 12; bump ~8% → 13.
CAND_BUBBLE_DIAM = 13.0
BUBBLE_STROKE_PRINT = 3
CONTENT_PAST_MARKER_OMR = 2.0
# Write-in lines end 0.5" short of the candidate-number box.
IDENTITY_LINE_GAP_OMR = 0.5 * OMR_PX_PER_IN
SAMPLE_CANDIDATE = "0009027001"

TITLE = "Ministry of Education - Multiple Choice Answer Sheet"
INSTRUCTIONS = (
    "Write your name clearly. Fill one bubble for each answer. Use a dark pencil."
)
IDENTITY_LABELS = [
    "Student Name:",
    "Centre Name:",
    "Exam Name:",
    "Subject Name:",
    "Student Signature:",
]

LABEL_FILL = "black"
LABEL_GLYPH = "#787878"
SUBTLE = "#333333"
BOX_STROKE = "#111111"
MARK_FILL = "black"


@dataclass(frozen=True)
class OrientationLayout:
    name: str
    folder: str
    omr_w: int
    omr_h: int
    print_w: int
    print_h: int


LAYOUT = OrientationLayout(
    name="landscape",
    folder="MoE-July-2026-Letter-Landscape-SMQ60-0",
    omr_w=OMR_LONG,
    omr_h=OMR_SHORT,
    print_w=PRINT_LONG,
    print_h=PRINT_SHORT,
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


def bubble_bbox_from_origin(
    layout: OrientationLayout, x0: float, y0: float, diam: float
) -> tuple[float, float, float, float]:
    sx, sy = scale_xy(layout)
    x0p, y0p = x0 * sx, y0 * sy
    return x0p, y0p, x0p + diam * sx, y0p + diam * sy


def pt_to_px(pt: float) -> int:
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
    pt = start_pt
    while pt >= min_pt:
        font = load_font(pt_to_px(pt), bold=bold)
        if draw.textlength(text, font=font) <= max_width_px:
            return font
        pt -= 0.5
    return load_font(pt_to_px(min_pt), bold=bold)


def content_margin() -> float:
    return ARUCO_INSET_OMR + ARUCO_SIZE_OMR / 2 + CONTENT_PAST_MARKER_OMR


def compute_geometry(layout: OrientationLayout) -> dict:
    diam = BUBBLE_DIAM
    cand_diam = CAND_BUBBLE_DIAM
    m = content_margin()
    right = layout.omr_w - m
    bottom = layout.omr_h - m
    box_inner_pad = 4.0
    usable_w = (right - m) - 2 * box_inner_pad
    n_cols, n_rows = 6, 10

    title_y = ARUCO_INSET_OMR
    instr_y = ARUCO_INSET_OMR + 10
    marker_bottom = ARUCO_INSET_OMR + ARUCO_SIZE_OMR / 2
    header_top = marker_bottom + CONTENT_PAST_MARKER_OMR

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

    line_x0 = m + 95
    identity_line_end = cand_table["x0"] - IDENTITY_LINE_GAP_OMR
    if identity_line_end <= line_x0 + 20:
        raise ValueError(
            f"identity write-in lines too short "
            f"(line_x0={line_x0:.1f}, end={identity_line_end:.1f})"
        )

    # Five identity lines; keep handwriting room while fitting the cand table.
    field_ys = [
        header_top + 20,
        header_top + 50,
        header_top + 80,
        header_top + 110,
        header_top + 140,
    ]
    identity_bottom = field_ys[-1] + 14
    header_bottom = max(
        identity_bottom,
        header_top + max(120.0, (cand_table["y1"] - header_top) + 8),
    )

    ans_top = header_bottom + 5
    ans_bottom = bottom - 2
    q_label_pad = 14.0
    ans_bubbles_gap = diam + 2.0
    option_span = 3 * ans_bubbles_gap + diam
    col_body = q_label_pad + option_span
    min_gutter = 4.0
    total_col_bodies = n_cols * col_body
    if total_col_bodies + (n_cols - 1) * min_gutter > usable_w:
        overflow = total_col_bodies + (n_cols - 1) * min_gutter - usable_w
        ans_bubbles_gap = max(diam + 1.0, ans_bubbles_gap - overflow / (3 * n_cols))
        option_span = 3 * ans_bubbles_gap + diam
        col_body = q_label_pad + option_span
        total_col_bodies = n_cols * col_body
    leftover = usable_w - total_col_bodies
    gutter = leftover / (n_cols - 1) if n_cols > 1 else min_gutter
    if gutter < min_gutter:
        raise ValueError(
            f"letter landscape: inter-column gutter too tight "
            f"(gutter={gutter:.2f}, min={min_gutter}, usable={usable_w:.1f})"
        )
    group_start = m + box_inner_pad
    block_origins_x = [
        group_start + q_label_pad + i * (col_body + gutter) for i in range(n_cols)
    ]
    row_inset = 5.0
    row_span = ans_bottom - ans_top - 2 * row_inset - diam
    ans_labels_gap = row_span / (n_rows - 1)
    if ans_labels_gap < diam + 2.0:
        raise ValueError(
            f"letter landscape: row pitch too tight "
            f"(labelsGap={ans_labels_gap:.2f}, diam={diam})"
        )
    ans_origin_y = ans_top + row_inset

    last_opt_right = block_origins_x[-1] + 3 * ans_bubbles_gap + diam
    if last_opt_right > right - box_inner_pad + 0.01:
        raise ValueError(
            f"letter landscape: answer columns overflow page "
            f"(last_opt_right={last_opt_right:.1f} > right={right:.1f})"
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
        "line_x0": line_x0,
        "identity_line_end": identity_line_end,
        "identity_field_ys": field_ys,
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


def stamp_aruco(img: Image.Image, layout: OrientationLayout, centres) -> Image.Image:
    sx, _ = scale_xy(layout)
    marker_px = max(24, int(round(ARUCO_SIZE_OMR * sx)))
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
    title_left_px, _ = to_print(layout, geo["title_max_x0"], 0)
    title_right_px, _ = to_print(layout, geo["title_max_x1"], 0)
    max_title_w = title_right_px - title_left_px

    title_font = fit_font(
        draw, TITLE, max_width_px=max_title_w, start_pt=14, min_pt=9, bold=True
    )
    instr_font = fit_font(
        draw, INSTRUCTIONS, max_width_px=max_title_w, start_pt=8.5, min_pt=7
    )
    label_font = load_font(pt_to_px(10.5), bold=True)

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

    line_x0 = geo["line_x0"]
    line_x1 = geo["identity_line_end"]
    for label, y in zip(IDENTITY_LABELS, geo["identity_field_ys"]):
        lx, ly = to_print(layout, m + 8, y)
        draw.text((lx, ly), label, fill=LABEL_FILL, font=label_font, anchor="lm")
        baseline = y + (4 if "Signature" in label else 7)
        x_a, y_a = to_print(layout, line_x0, baseline)
        x_b, _ = to_print(layout, line_x1, baseline)
        draw.line([(x_a, y_a), (x_b, y_a)], fill=LABEL_FILL, width=2)

    draw_candidate_table(draw, layout, geo)


def draw_candidate_table(
    draw: ImageDraw.ImageDraw,
    layout: OrientationLayout,
    geo: dict,
    *,
    filled_digits: str | None = None,
) -> None:
    cand_diam = geo["cand_bubble_diam"]
    table = geo["cand_table"]
    cx0, cy0 = geo["cand_origin"]
    pitch = geo["cand_labels_gap"]
    v_gap = geo["cand_bubbles_gap"]
    write_in_h = table["write_in_h"]
    stroke = max(2, int(round(2 * scale_xy(layout)[0] / 2.5)))

    cap_font = load_font(pt_to_px(11), bold=True)
    digit_font = load_font(pt_to_px(6.5))
    write_font = load_font(pt_to_px(12), bold=True)

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

    for col in range(1, 10):
        vx = table["x0"] + col * pitch
        px, _ = to_print(layout, vx, 0)
        draw.line([(px, ty0), (px, ty1)], fill=BOX_STROKE, width=stroke)

    digits = None
    if filled_digits is not None:
        if len(filled_digits) != 10 or not filled_digits.isdigit():
            raise ValueError(f"expected 10-digit candidate, got {filled_digits!r}")
        digits = [int(ch) for ch in filled_digits]
        for col, digit in enumerate(digits):
            cell_cx = cx0 + col * pitch + cand_diam / 2
            cell_cy = table["y0"] + write_in_h / 2
            px, py = to_print(layout, cell_cx, cell_cy)
            draw.text(
                (px, py), str(digit), fill=MARK_FILL, font=write_font, anchor="mm"
            )

    for col in range(10):
        for digit in range(10):
            x0b = cx0 + col * pitch
            y0b = cy0 + digit * v_gap
            bbox = bubble_bbox_from_origin(layout, x0b, y0b, cand_diam)
            marked = digits is not None and digits[col] == digit
            if marked:
                draw.ellipse(bbox, fill=MARK_FILL, outline=MARK_FILL)
            else:
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

    qnum_font = load_font(pt_to_px(10), bold=True)
    letter_font = load_font(pt_to_px(7))

    for block in geo["q_blocks"]:
        ox, oy = block["origin"]
        for row, qnum in enumerate(range(block["first_q"], block["last_q"] + 1)):
            y = oy + row * block["labelsGap"]
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


def render_sheet(
    layout: OrientationLayout,
    geo: dict,
    *,
    filled_candidate: str | None = None,
) -> Image.Image:
    img = Image.new("RGB", (layout.print_w, layout.print_h), "white")
    draw = ImageDraw.Draw(img)
    # Header draws empty candidate table; redraw with fills after answers if needed.
    draw_header(draw, layout, geo)
    draw_answers(draw, layout, geo)
    if filled_candidate is not None:
        draw_candidate_table(draw, layout, geo, filled_digits=filled_candidate)
    img = stamp_aruco(img, layout, geo["aruco_centres"])
    return img


def write_alignment_preview(
    layout: OrientationLayout, geo: dict, blank: Image.Image, out_path: Path
) -> None:
    diam = geo["bubble_diam"]
    cand_diam = geo["cand_bubble_diam"]
    preview = blank.copy()
    omr_img = preview.resize((layout.omr_w, layout.omr_h), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(omr_img)
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
    view = omr_img.resize(
        (layout.omr_w * 2, layout.omr_h * 2), Image.Resampling.NEAREST
    )
    view.save(out_path, format="PNG", optimize=True)


def generate() -> Path:
    layout = LAYOUT
    out_dir = REPO / layout.folder
    ref_dir = out_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    geo = compute_geometry(layout)
    blank = render_sheet(layout, geo)
    sample = render_sheet(layout, geo, filled_candidate=SAMPLE_CANDIDATE)

    png_path = ref_dir / f"blank_{layout.name}_smq60.png"
    pdf_path = ref_dir / f"blank_{layout.name}_smq60.pdf"
    sample_png = ref_dir / f"sample_{SAMPLE_CANDIDATE}.png"
    sample_pdf = ref_dir / f"sample_{SAMPLE_CANDIDATE}.pdf"
    preview_path = ref_dir / f"alignment_preview_{layout.name}.png"

    blank.save(png_path, format="PNG", optimize=True)
    blank.save(pdf_path, format="PDF", resolution=float(PRINT_DPI))
    sample.save(sample_png, format="PNG", optimize=True)
    sample.save(sample_pdf, format="PDF", resolution=float(PRINT_DPI))
    write_alignment_preview(layout, geo, blank, preview_path)

    template = build_template(layout, geo)
    config = build_config(layout)
    (out_dir / "template.json").write_text(
        json.dumps(template, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    helper = f'''"""Regenerate the Letter-landscape 60Q blank + sample + template.

Run from repo root::

    python {layout.folder}/generate_blank.py
"""
from __future__ import annotations

import runpy
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sheets" / "generate_letter_smq60.py"

if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")
'''
    (out_dir / "generate_blank.py").write_text(helper, encoding="utf-8")

    readme = f"""# {layout.folder}

US Letter **landscape** (11″ × 8.5″) MoE answer sheet — **60** MCQ questions (6×10).

## Contents

| File | Role |
|---|---|
| `template.json` | OMR field map (ArUco CropOnMarkers) |
| `config.json` | Processing dimensions |
| `reference/blank_landscape_smq60.png` | Printable blank |
| `reference/blank_landscape_smq60.pdf` | Printable PDF @ 300 DPI |
| `reference/sample_{SAMPLE_CANDIDATE}.png` | Sample with candidate bubbled |
| `reference/alignment_preview_landscape.png` | Template-bubble overlay QA |
| `generate_blank.py` | Regenerate from shared script |

## Notes

- Registered as the WebUI **default** blank-sheet variant and default preset.
- Identity: Student / Centre / Exam / Subject / Signature.
- Marker centres inset ~0.5″ from page edges.
- Shared generator: `scripts/sheets/generate_letter_smq60.py`
- Legal July landscape remains in `MoE-July-2026-Landscape-SMQ60-0/`.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"[{layout.name}] wrote {out_dir}")
    print(f"  PNG  {png_path}  ({blank.size[0]}x{blank.size[1]})")
    print(f"  PDF  {pdf_path}")
    print(f"  SMP  {sample_png}")
    print(f"  QA   {preview_path}")
    return out_dir


def main() -> int:
    generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
