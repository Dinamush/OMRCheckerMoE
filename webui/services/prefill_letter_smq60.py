"""Letter landscape SMQ60 geometry for prefill rendering.

Coordinates are derived from ``scripts/sheets/generate_letter_smq60.py`` /
``MoE-July-2026-Letter-Landscape-SMQ60-0/template.json`` and scale from the
OMR canvas (660×510) onto the print blank (3300×2550 @ 300 DPI).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from webui.sheet_registry import LETTER_LANDSCAPE_SMQ60_DIR

LETTER_BLANK_PNG = (
    LETTER_LANDSCAPE_SMQ60_DIR / "reference" / "blank_landscape_smq60.png"
)

OMR_W = 660.0
OMR_H = 510.0

# Identity write-in baselines (OMR-y of label mid-line + small offset for ink).
# Order: Student, Centre, Exam, Subject (signature left blank).
_IDENTITY_Y_OMR = (62.8, 92.8, 122.8, 152.8)
_LINE_X0_OMR = 137.8
_LINE_X1_OMR = 415.2

# CandidateNumber fieldBlock from template.json (top-left of digit-0 / col-0).
_CAND_ORIGIN_OMR = (447.2, 75.8)
_CAND_DIAM_OMR = 13.0
_CAND_LABELS_GAP_OMR = 17.0
_CAND_BUBBLES_GAP_OMR = 14.5
_CAND_WRITE_IN_Y0_OMR = 60.8
_CAND_WRITE_IN_H_OMR = 15.0

# Answer blocks: (origin_x, origin_y, bubbles_gap, labels_gap, first_q)
# origin = top-left of A for the block's first question (template.json).
_ANSWER_BLOCKS_OMR: tuple[tuple[float, float, float, float, int], ...] = (
    (60.8, 239.3, 18.0, 22.77, 1),
    (157.28, 239.3, 18.0, 22.77, 11),
    (253.76, 239.3, 18.0, 22.77, 21),
    (350.24, 239.3, 18.0, 22.77, 31),
    (446.72, 239.3, 18.0, 22.77, 41),
    (543.2, 239.3, 18.0, 22.77, 51),
)
_QUESTIONS_PER_BLOCK = 10
_ANSWER_DIAM_OMR = 16.0
NUM_QUESTIONS = 60

_font_cache: dict[tuple[int, bool], ImageFont.ImageFont] = {}


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    paths = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for path in paths:
        try:
            font = ImageFont.truetype(path, size)
            _font_cache[key] = font
            return font
        except OSError:
            continue
    font = ImageFont.load_default()
    _font_cache[key] = font
    return font


def load_letter_blank() -> Image.Image:
    """Load the Letter SMQ60 blank (ArUco already stamped)."""
    if not LETTER_BLANK_PNG.exists():
        raise FileNotFoundError(
            f"Letter SMQ60 blank missing at {LETTER_BLANK_PNG}. "
            "Run: python scripts/sheets/generate_letter_smq60.py"
        )
    return Image.open(LETTER_BLANK_PNG).convert("RGB")


def _scale(w: int, h: int) -> tuple[float, float]:
    return w / OMR_W, h / OMR_H


def _draw_text_on_line(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    x0: float,
    x1: float,
    y_mid: float,
    start_size: int,
) -> None:
    text = (text or "").strip()
    if not text:
        return
    max_w = max(8.0, x1 - x0)
    size = start_size
    font = _load_font(size)
    while size >= 14:
        font = _load_font(size)
        tw = draw.textlength(text, font=font)
        if tw <= max_w:
            break
        size -= 1
    draw.text((x0, y_mid), text, fill="black", font=font, anchor="lm")


def draw_letter_sheet_content(
    img: Image.Image,
    *,
    student_name: str,
    centre_name: str,
    exam_name: str,
    subject_name: str,
    candidate_number: str,
) -> Image.Image:
    """Draw identity text + candidate write-in / bubbles onto a Letter blank."""
    if len(candidate_number) != 10 or not candidate_number.isdigit():
        raise ValueError("Candidate number must be exactly 10 digits.")

    draw = ImageDraw.Draw(img)
    w, h = img.size
    sx, sy = _scale(w, h)

    values = (student_name, centre_name, exam_name, subject_name)
    x0 = _LINE_X0_OMR * sx
    x1 = _LINE_X1_OMR * sx
    for value, y_omr in zip(values, _IDENTITY_Y_OMR):
        _draw_text_on_line(
            draw,
            value,
            x0=x0,
            x1=x1,
            y_mid=(y_omr + 2.0) * sy,
            start_size=max(18, int(round(11 * sx))),
        )

    ox, oy = _CAND_ORIGIN_OMR
    diam = _CAND_DIAM_OMR
    pitch = _CAND_LABELS_GAP_OMR
    v_gap = _CAND_BUBBLES_GAP_OMR
    write_cy = (_CAND_WRITE_IN_Y0_OMR + _CAND_WRITE_IN_H_OMR / 2.0) * sy
    digit_font = _load_font(max(16, int(round(12 * sx))), bold=True)
    radius = max(4.0, (diam * min(sx, sy)) / 2.0)

    for col, ch in enumerate(candidate_number):
        digit = int(ch)
        cx = (ox + col * pitch + diam / 2.0) * sx
        draw.text((cx, write_cy), ch, fill="black", font=digit_font, anchor="mm")
        cy = (oy + digit * v_gap + diam / 2.0) * sy
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill="black",
        )
    return img


def answer_bubble_geometry(w: int, h: int) -> list[dict[str, Any]]:
    """Pixel centres for q1–q60 × A–D on a Letter SMQ60 canvas of size (w, h)."""
    if w <= 0 or h <= 0:
        raise ValueError(f"Canvas dimensions must be positive (got {w}x{h}).")
    sx, sy = _scale(w, h)
    # Use ~42% of bubble diam as radius (not 50%) so heavy/messy marks keep
    # clearance to the next A–D well. Full-radius fills were bleeding across
    # the 18 OMR-px option pitch and collapsing CropOnMarkers bubble-alignment
    # confidence on multi-option keys like ``alternating``.
    radius = max(2, int(round((_ANSWER_DIAM_OMR * 0.42) * min(sx, sy))))
    bubbles: list[dict[str, Any]] = []
    for origin_x, origin_y, bubbles_gap, labels_gap, first_q in _ANSWER_BLOCKS_OMR:
        for q_idx in range(_QUESTIONS_PER_BLOCK):
            for opt_idx, letter in enumerate("ABCD"):
                # template origin is top-left; convert to centre
                cx_omr = origin_x + _ANSWER_DIAM_OMR / 2.0 + opt_idx * bubbles_gap
                cy_omr = origin_y + _ANSWER_DIAM_OMR / 2.0 + q_idx * labels_gap
                cx = int(round(cx_omr * sx))
                cy = int(round(cy_omr * sy))
                bubbles.append(
                    {
                        "q": first_q + q_idx,
                        "option": opt_idx,
                        "option_letter": letter,
                        "cx": min(max(cx, 0), w - 1),
                        "cy": min(max(cy, 0), h - 1),
                        "radius": radius,
                    }
                )
    return bubbles


def candidate_bubble_geometry(
    w: int, h: int, candidate_number: str | None = None
) -> list[dict[str, Any]]:
    """Candidate-number bubble centres for scan-simulation protection."""
    sx, sy = _scale(w, h)
    ox, oy = _CAND_ORIGIN_OMR
    diam = _CAND_DIAM_OMR
    pitch = _CAND_LABELS_GAP_OMR
    v_gap = _CAND_BUBBLES_GAP_OMR
    radius = max(2, int(round((diam / 2.0) * min(sx, sy))))
    filled: dict[int, int] = {}
    if candidate_number:
        for i, ch in enumerate(candidate_number[:10]):
            if ch.isdigit():
                filled[i] = int(ch)
    out: list[dict[str, Any]] = []
    for col in range(10):
        for digit in range(10):
            cx = int(round((ox + col * pitch + diam / 2.0) * sx))
            cy = int(round((oy + digit * v_gap + diam / 2.0) * sy))
            out.append(
                {
                    "column": col,
                    "digit": digit,
                    "cx": cx,
                    "cy": cy,
                    "radius": radius,
                    "filled": filled.get(col) == digit,
                }
            )
    return out


def aruco_marker_boxes(w: int, h: int) -> list[dict[str, Any]]:
    """Approximate ArUco boxes (0.5\" inset, April-sized markers) for scan sim."""
    sx, sy = _scale(w, h)
    inset = 30.0
    half = 10.8
    centres = (
        (inset, inset),
        (OMR_W - inset, inset),
        (inset, OMR_H - inset),
        (OMR_W - inset, OMR_H - inset),
    )
    boxes = []
    for corner, (cx, cy) in enumerate(centres):
        x0 = int(round((cx - half) * sx))
        y0 = int(round((cy - half) * sy))
        x1 = int(round((cx + half) * sx))
        y1 = int(round((cy + half) * sy))
        boxes.append(
            {
                "corner": corner,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
            }
        )
    return boxes


def candidate_region_box(w: int, h: int) -> tuple[int, int, int, int]:
    sx, sy = _scale(w, h)
    # cand_table bounds from generator
    return (
        int(round(445.2 * sx)),
        int(round(60.8 * sy)),
        int(round(615.2 * sx)),
        int(round(221.3 * sy)),
    )
