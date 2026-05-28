"""Synthetic bubble fillers for the candidate-bubble sweep harness.

The OMRChecker engine evaluates each bubble by computing the **mean grey
intensity** of the box defined by ``bubbleDimensions``. Lower intensity
(closer to 0) means a darker fill, which the threshold algorithm in
``src.core.ImageInstanceOps.get_global_threshold`` interprets as a mark.

We model 4 realistic darkness presets keyed to the "pencil sharpness"
spectrum a teacher might encounter in a real exam:

================  ===============  =========================================
Preset            RGB fill value   Real-world analogue
================  ===============  =========================================
pen_ink           (20, 20, 20)     Ballpoint pen / dark marker
dark_pencil       (60, 60, 60)     Sharp #2 pencil, firm pressure
medium_pencil     (110, 110, 110)  Average #2 pencil, normal pressure
light_pencil      (170, 170, 170)  Worn pencil or feather-touch student
================  ===============  =========================================

These cover the engine's MIN_JUMP=25 threshold: pen_ink and dark_pencil
should be trivially detected against a 255-white background (jump of
~195/180), while light_pencil at intensity 170 leaves only an 85-unit
jump — close to the engine's confidence floor for ambiguous strips.

Each filler returns the modified PIL image so callers can chain
(e.g. fill candidate digits and then question answers).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PIL import Image, ImageDraw

from generate_sheet import (
    ANS_BLOCK_LEFT_ORIGIN,
    ANS_BLOCK_RIGHT_ORIGIN,
    ANS_BUBBLE_DIAM,
    ANS_BUBBLES_GAP_X,
    ANS_LABELS_GAP_Y,
    SheetSpec,
    bubble_bbox,
)


DARKNESS_PRESETS = {
    "pen_ink":       (20, 20, 20),
    "dark_pencil":   (60, 60, 60),
    "medium_pencil": (110, 110, 110),
    "light_pencil":  (170, 170, 170),
}


@dataclass(frozen=True)
class FillStyle:
    """How a single bubble should be filled."""

    color: tuple[int, int, int]
    # Shrink factor for the drawn fill ellipse, expressed as a fraction of
    # the OMR diameter. 1.0 = exact fit to the printed outline, 0.85 =
    # student stays slightly inside the outline (typical real-world).
    fill_fraction: float = 0.85
    # Optional override for filling only PART of the bubble (e.g. "half" or
    # "tick" for edge-case tests). When None the full ellipse is drawn.
    partial_mode: str | None = None
    # Offset applied to the centre, in fractions of the bubble diameter
    # (e.g. (0.3, 0.0) shifts 30 % to the right for an off-centre test).
    offset_fraction: tuple[float, float] = (0.0, 0.0)


def _candidate_bubble_centre(spec: SheetSpec, col: int, digit: int) -> tuple[float, float]:
    ox, oy = spec.cand_origin
    cx_omr = ox + col * spec.cand_bubbles_gap_x
    cy_omr = oy + digit * spec.cand_labels_gap_y
    return cx_omr, cy_omr


def _answer_bubble_centre(qnum: int, option: str) -> tuple[float, float]:
    if 1 <= qnum <= 13:
        origin = ANS_BLOCK_LEFT_ORIGIN
        row = qnum - 1
    else:
        origin = ANS_BLOCK_RIGHT_ORIGIN
        row = qnum - 14
    ox, oy = origin
    option_idx = "ABCD".index(option)
    cx = ox + option_idx * ANS_BUBBLES_GAP_X
    cy = oy + row * ANS_LABELS_GAP_Y
    return cx, cy


def _draw_filled_bubble(
    draw: ImageDraw.ImageDraw,
    cx_omr: float,
    cy_omr: float,
    diam_omr: float,
    style: FillStyle,
) -> None:
    """Draw a filled mark at the given OMR-space centre.

    Handles the partial-fill modes used by edge-case tests:
      * ``None`` (default) — solid ellipse at fill_fraction × diam.
      * ``"half"`` — fill the left half of the bubble only.
      * ``"tick"`` — short diagonal line, simulating a checkmark.
      * ``"smudge"`` — small dark blob offset from centre.
    """
    drawn_diam = diam_omr * style.fill_fraction
    dx, dy = style.offset_fraction
    cx = cx_omr + dx * diam_omr
    cy = cy_omr + dy * diam_omr

    if style.partial_mode is None:
        bbox = bubble_bbox(cx, cy, drawn_diam)
        draw.ellipse(bbox, fill=style.color)
    elif style.partial_mode == "half":
        bbox = bubble_bbox(cx, cy, drawn_diam)
        draw.pieslice(bbox, 90, 270, fill=style.color)
    elif style.partial_mode == "tick":
        bbox = bubble_bbox(cx, cy, drawn_diam * 0.7)
        # Diagonal line from lower-left to upper-right
        draw.line([(bbox[0], bbox[3]), (bbox[2], bbox[1])], fill=style.color, width=4)
    elif style.partial_mode == "smudge":
        bbox = bubble_bbox(cx, cy, drawn_diam * 0.55)
        draw.ellipse(bbox, fill=style.color)
    else:
        raise ValueError(f"Unknown partial_mode: {style.partial_mode}")


def fill_candidate_number(
    img: Image.Image,
    spec: SheetSpec,
    digits: str,
    style: FillStyle,
) -> Image.Image:
    """Fill the 10-digit candidate-number grid with ``digits``.

    ``digits`` must be a 10-character string of '0'..'9' (or '.' to skip a
    column for edge-case tests of missing marks).
    """
    if len(digits) != 10:
        raise ValueError(f"digits must be 10 characters, got {len(digits)}: {digits!r}")
    draw = ImageDraw.Draw(img)
    for col, ch in enumerate(digits):
        if ch == ".":
            continue
        if ch not in "0123456789":
            raise ValueError(f"digits must be 0-9 or '.', got {ch!r} at col {col}")
        digit = int(ch)
        cx, cy = _candidate_bubble_centre(spec, col, digit)
        _draw_filled_bubble(draw, cx, cy, spec.cand_bubble_diam, style)
    return img


def fill_answers(
    img: Image.Image,
    answers: Iterable[tuple[int, str]],
    style: FillStyle,
) -> Image.Image:
    """Fill a sequence of ``(question_number, option_letter)`` answers."""
    draw = ImageDraw.Draw(img)
    for qnum, option in answers:
        if not (1 <= qnum <= 25):
            raise ValueError(f"question number out of range: {qnum}")
        if option not in "ABCD":
            raise ValueError(f"option must be A-D, got {option!r}")
        cx, cy = _answer_bubble_centre(qnum, option)
        _draw_filled_bubble(draw, cx, cy, ANS_BUBBLE_DIAM, style)
    return img
