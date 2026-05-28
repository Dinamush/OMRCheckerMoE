"""Student-style bubble fill for prefilled answer sheets.

This module simulates how a real student would mark the answer bubbles
(q1-q25 × A-D) on a prefilled sheet. It complements ``scan_simulation``:
the marking profile controls the *student's hand* (light pencil, heavy
pencil, check marks, partial fills, multi-marks, etc.), and the realism
preset controls the *scanner's artifacts*. Both can be combined.

The geometry uses uniform proportional scaling from the OMR template's
``fieldBlocks`` coordinates (in the 666×515 processing canvas) into the
prefill render canvas (typically 1426×1103). Because both share the same
ArUco marker reference fractions, this scaling matches what CropOnMarkers
will apply on the scanned input — so bubbles drawn here land exactly on
the field-block centres after the round-trip.

Drawing is fully deterministic for a given (answers, marking_profile,
candidate_number, seed) tuple, so tests can assert byte-identical output.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OMR template constants (mirror webui.api._PREFILLED_25Q_TEMPLATE).
# Kept here as a tiny standalone copy so this module never has to import the
# 1500-line ``webui.api`` module (which would create a circular dependency:
# api.py imports student_fill, student_fill imports api). Any change to the
# template constant in api.py must be mirrored here.
# ---------------------------------------------------------------------------
_OMR_PAGE_WIDTH = 666
_OMR_PAGE_HEIGHT = 515

# (block_first_bubble_center_x, ..._center_y, bubbles_gap, labels_gap, first_q)
#
# Stores CENTER coordinates of the first bubble in each block (q1-A, q6-A,
# …) in the 666x515 OMR processing canvas. These were empirically
# calibrated via Hough-circle detection against the actual printed bubble
# outlines on ``prefill_package/blank_template_reference.png`` (see
# ``webui/tests/_calibrate_bubbles.py``).
#
# Note: the OMR template's ``origin`` field stores the TOP-LEFT of the
# bubble box (the engine samples a ``bubbleDimensions``-wide rectangle
# starting at origin). The values below are SHIFTED +box/2 from the OMR
# template origin so they represent bubble *centres* — which is what the
# drawing code needs to place ellipses dead-on the printed circles. Keep
# the OMR template ``MoE-April-2026-Landscape-NNQ25-0/template.json`` in sync
# (template origin = these center coords - 5 in each axis).
_ANSWER_BLOCKS: tuple[tuple[float, float, float, float, int], ...] = (
    (57.7,  264.3, 20.0, 41.9,  1),   # q1-q5
    (185.9, 264.3, 20.0, 41.9,  6),   # q6-q10
    (314.8, 264.3, 19.8, 41.9, 11),   # q11-q15
    (440.9, 264.3, 20.0, 41.9, 16),   # q16-q20
    (571.3, 264.3, 20.0, 41.9, 21),   # q21-q25
)
_QUESTIONS_PER_BLOCK = 5
_OPTIONS_PER_QUESTION = 4
_OPTION_LETTERS = ("A", "B", "C", "D")
NUM_QUESTIONS = 25
NUM_OPTIONS = 4


# ---------------------------------------------------------------------------
# Marking profiles.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MarkingProfile:
    """Parameters describing a student's marking behaviour.

    All ranges are inclusive ``(low, high)`` tuples.
    """

    name: str
    description: str
    style: str  # "ellipse" | "check" | "cross" | "partial" | "none"
    gray_range: tuple[int, int] = (40, 90)
    scale_range: tuple[float, float] = (0.85, 1.05)
    shift_px: int = 2
    noise_sigma: float = 6.0
    angle_deg: float = 8.0
    multi_mark_prob: float = 0.0    # chance a single-answer becomes 2 marks
    stray_mark_prob: float = 0.0    # chance of a tiny stray near a real bubble
    erasure_prob: float = 0.0       # chance of a faint ghost smear on a bubble
    # Per-mark partial-fill coverage (only used when style == "partial").
    partial_arc_range: tuple[int, int] = (180, 270)


MARKING_PROFILES: dict[str, MarkingProfile] = {
    "none": MarkingProfile(
        name="none",
        description="Don't draw any answer bubbles (metadata-only behaviour).",
        style="none",
    ),
    "light_pencil": MarkingProfile(
        name="light_pencil",
        description="Faint, tentative pencil strokes.",
        style="ellipse",
        gray_range=(140, 190),
        scale_range=(0.70, 0.90),
        shift_px=2,
        noise_sigma=8.0,
        angle_deg=14.0,
        multi_mark_prob=0.01,
        stray_mark_prob=0.01,
    ),
    "medium_pencil": MarkingProfile(
        name="medium_pencil",
        description="Typical student — full but slightly imperfect fills.",
        style="ellipse",
        gray_range=(60, 110),
        scale_range=(0.85, 1.05),
        shift_px=2,
        noise_sigma=6.0,
        angle_deg=10.0,
        multi_mark_prob=0.02,
        stray_mark_prob=0.02,
    ),
    "heavy_pencil": MarkingProfile(
        name="heavy_pencil",
        description="Confident, dark, slightly-larger marks.",
        style="ellipse",
        gray_range=(10, 45),
        scale_range=(0.95, 1.10),
        shift_px=1,
        noise_sigma=4.0,
        angle_deg=6.0,
        multi_mark_prob=0.01,
        stray_mark_prob=0.01,
    ),
    "pen_ballpoint": MarkingProfile(
        name="pen_ballpoint",
        description="Ballpoint pen — uniformly dark, crisp.",
        style="ellipse",
        gray_range=(5, 30),
        scale_range=(0.92, 1.05),
        shift_px=1,
        noise_sigma=2.5,
        angle_deg=4.0,
        multi_mark_prob=0.0,
        stray_mark_prob=0.0,
    ),
    "check_mark": MarkingProfile(
        name="check_mark",
        description="Check mark (✓) drawn inside the bubble.",
        style="check",
        gray_range=(15, 60),
        shift_px=2,
        noise_sigma=3.0,
    ),
    "cross_mark": MarkingProfile(
        name="cross_mark",
        description="X drawn through the bubble.",
        style="cross",
        gray_range=(15, 60),
        shift_px=2,
        noise_sigma=3.0,
    ),
    "partial_fill": MarkingProfile(
        name="partial_fill",
        description="Bubble only partially shaded (half-arc).",
        style="partial",
        gray_range=(60, 120),
        scale_range=(0.60, 0.85),
        shift_px=3,
        noise_sigma=6.0,
        angle_deg=10.0,
        multi_mark_prob=0.03,
        stray_mark_prob=0.04,
        partial_arc_range=(140, 260),
    ),
    "messy_student": MarkingProfile(
        name="messy_student",
        description="Highly inconsistent: variable darkness, jittery placement, occasional multi-marks.",
        style="ellipse",
        gray_range=(30, 180),
        scale_range=(0.50, 1.15),
        shift_px=4,
        noise_sigma=10.0,
        angle_deg=20.0,
        multi_mark_prob=0.05,
        stray_mark_prob=0.08,
        erasure_prob=0.04,
    ),
    "careful_student": MarkingProfile(
        name="careful_student",
        description="Neat, fully shaded bubbles, low jitter.",
        style="ellipse",
        gray_range=(15, 60),
        scale_range=(0.97, 1.05),
        shift_px=1,
        noise_sigma=3.0,
        angle_deg=4.0,
        multi_mark_prob=0.0,
        stray_mark_prob=0.01,
    ),
}

DEFAULT_MARKING_PROFILE = "medium_pencil"


_PROFILE_LABELS: dict[str, str] = {
    "none": "None — leave blank",
    "light_pencil": "Light pencil",
    "medium_pencil": "Medium pencil",
    "heavy_pencil": "Heavy pencil",
    "pen_ballpoint": "Pen (ballpoint)",
    "check_mark": "Check mark (✓)",
    "cross_mark": "Cross mark (✗)",
    "partial_fill": "Partial fill",
    "messy_student": "Messy student",
    "careful_student": "Careful student",
}


def list_marking_profiles() -> list[dict[str, str]]:
    """Return a UI-friendly list of available profiles."""
    return [
        {
            "id": p.name,
            "label": _PROFILE_LABELS.get(p.name, p.name.replace("_", " ").title()),
            "description": p.description,
        }
        for p in MARKING_PROFILES.values()
    ]


def normalize_marking_profile(value: str | None) -> str:
    """Return a canonical marking profile name or raise ``ValueError``."""
    key = (value or "none").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in MARKING_PROFILES:
        allowed = ", ".join(sorted(MARKING_PROFILES))
        raise ValueError(f"marking_profile must be one of: {allowed}.")
    return key


# ---------------------------------------------------------------------------
# Geometry.
# ---------------------------------------------------------------------------
def answer_bubble_geometry(w: int, h: int) -> list[dict[str, Any]]:
    """Return pixel-space geometry for the 100 answer bubbles on a sheet
    of size ``(w, h)``.

    The output is a flat list of dicts ordered (q ascending, opt ascending):
    ``{q: 1..25, option: 0..3, cx, cy, radius}``.

    ``cx``/``cy`` are integer pixel centres in the (w, h) canvas; ``radius``
    is half the smaller scaled bubble dimension. All values are clamped to
    the image bounds so caller can pass them straight to ``cv2.ellipse``.
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"Canvas dimensions must be positive (got {w}x{h}).")

    sx = w / _OMR_PAGE_WIDTH
    sy = h / _OMR_PAGE_HEIGHT
    # Bubble half-size of 5 mirrors bubbleDimensions=[10,10] / 2.
    radius_omr = 5.0
    radius = max(2, int(round(radius_omr * min(sx, sy))))

    bubbles: list[dict[str, Any]] = []
    for origin_x, origin_y, bubbles_gap, labels_gap, first_q in _ANSWER_BLOCKS:
        for q_idx in range(_QUESTIONS_PER_BLOCK):
            for opt_idx in range(_OPTIONS_PER_QUESTION):
                cx_omr = origin_x + opt_idx * bubbles_gap
                cy_omr = origin_y + q_idx * labels_gap
                cx = int(round(cx_omr * sx))
                cy = int(round(cy_omr * sy))
                cx = min(max(cx, 0), w - 1)
                cy = min(max(cy, 0), h - 1)
                bubbles.append({
                    "q": first_q + q_idx,
                    "option": opt_idx,
                    "option_letter": _OPTION_LETTERS[opt_idx],
                    "cx": cx,
                    "cy": cy,
                    "radius": radius,
                })
    return bubbles


def _build_bubble_index(geometry: Sequence[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {(b["q"], b["option"]): b for b in geometry}


# ---------------------------------------------------------------------------
# Answer parsing.
# ---------------------------------------------------------------------------
_BLANK_TOKENS = {"", "-", "_", "nr", "x", "skip", "none", "null", "blank"}


def _option_letter_to_idx(letter: str) -> int | None:
    """Return option index 0..3 or ``None`` if the letter is not A-D."""
    c = letter.strip().upper()
    if c in {"A", "B", "C", "D"}:
        return ord(c) - ord("A")
    return None


def _normalise_one_answer(value: Any) -> list[int]:
    """Normalise a per-question answer value into a sorted list of option
    indices. Empty / blank values return an empty list.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        idxs: set[int] = set()
        for item in value:
            sub = _normalise_one_answer(item)
            idxs.update(sub)
        return sorted(idxs)
    if isinstance(value, int):
        if 0 <= value < NUM_OPTIONS:
            return [value]
        return []
    text = str(value).strip()
    if not text or text.lower() in _BLANK_TOKENS:
        return []
    idxs = set()
    for ch in text:
        if ch in {",", " ", "|", "+"}:
            continue
        opt = _option_letter_to_idx(ch)
        if opt is not None:
            idxs.add(opt)
    return sorted(idxs)


def _make_random(seed: int | None) -> random.Random:
    if seed is None:
        return random.Random()
    return random.Random(int(seed) & 0xFFFFFFFF)


def parse_answers(
    value: Any,
    *,
    num_questions: int = NUM_QUESTIONS,
    seed: int | None = None,
) -> dict[int, list[int]]:
    """Parse a permissive answer specification into a canonical dict.

    The output maps 1-indexed question number → sorted list of 0-indexed
    option indices the student marked. Missing keys are blank by convention.

    Accepted inputs (case-insensitive):
        - ``None`` / ``""`` / ``"none"`` / ``"blank"``: all blank
        - ``"random"``: random A-D per question
        - ``"random_with_skips"``: ~10% blanks
        - ``"all_a"`` / ``"all_b"`` / ``"all_c"`` / ``"all_d"``: uniform
        - ``"alternating"``: ABCDA…
        - dict with ``q1`` / ``q01`` / ``"1"`` keys
        - list / tuple aligned with question numbers
        - 25-character string ``"ABCD-ABCD..."`` (``-`` / ``X`` / space = skip)
        - JSON encoding of any of the above
    """
    rng = _make_random(seed)

    if value is None:
        return {}
    if isinstance(value, (list, tuple)):
        out: dict[int, list[int]] = {}
        for idx, item in enumerate(value, start=1):
            if idx > num_questions:
                break
            opts = _normalise_one_answer(item)
            if opts:
                out[idx] = opts
        return out
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            try:
                q = int(re.sub(r"\D", "", str(k)) or "-1")
            except ValueError:
                continue
            if 1 <= q <= num_questions:
                opts = _normalise_one_answer(v)
                if opts:
                    out[q] = opts
        return out

    text = str(value).strip()
    if not text:
        return {}

    # JSON?
    if text[0] in "{[":
        try:
            return parse_answers(json.loads(text), num_questions=num_questions, seed=seed)
        except (json.JSONDecodeError, ValueError):
            pass

    lower = text.lower()
    if lower in _BLANK_TOKENS or lower == "blank":
        return {}
    if lower == "random":
        return {q: [rng.randrange(NUM_OPTIONS)] for q in range(1, num_questions + 1)}
    if lower == "random_with_skips":
        out = {}
        for q in range(1, num_questions + 1):
            if rng.random() < 0.10:
                continue
            out[q] = [rng.randrange(NUM_OPTIONS)]
        return out
    if lower.startswith("all_"):
        letter = lower[4:].strip()
        idx = _option_letter_to_idx(letter)
        if idx is None:
            raise ValueError(f"Unknown 'all_X' shortcut: {value!r}")
        return {q: [idx] for q in range(1, num_questions + 1)}
    if lower == "alternating":
        return {q: [(q - 1) % NUM_OPTIONS] for q in range(1, num_questions + 1)}

    # Treat as positional letter string (whitespace / separator-stripped).
    compact = re.sub(r"[,;|]", "", text).replace(" ", "")
    out = {}
    # Special markers in positional strings stand for "skip this question"
    # and must each consume exactly one slot rather than being collapsed
    # into the surrounding answer.
    SKIP_CHARS = {"-", "_", "."}
    q = 1
    for ch in compact:
        if q > num_questions:
            break
        if ch in SKIP_CHARS:
            q += 1
            continue
        if ch.upper() == "X":
            q += 1
            continue
        opt = _option_letter_to_idx(ch)
        if opt is None:
            # Ignore stray characters but log once.
            logger.debug("parse_answers: ignored char %r in %r", ch, value)
            continue
        out[q] = [opt]
        q += 1
    return out


# ---------------------------------------------------------------------------
# Seeding.
# ---------------------------------------------------------------------------
def _stable_seed(*parts: object) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "little") & 0xFFFFFFFF


def _rng_for(
    answers: dict[int, list[int]],
    profile: MarkingProfile,
    candidate_number: str | None,
    seed: int | None,
) -> np.random.Generator:
    if seed is not None:
        return np.random.default_rng(int(seed) & 0xFFFFFFFF)
    key = _stable_seed(
        "student-fill",
        profile.name,
        candidate_number or "",
        tuple(sorted((q, tuple(opts)) for q, opts in answers.items())),
    )
    return np.random.default_rng(key)


# ---------------------------------------------------------------------------
# Drawing primitives.
# ---------------------------------------------------------------------------
def _clip_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def _draw_one_ellipse(
    arr: np.ndarray,
    bubble: dict[str, Any],
    profile: MarkingProfile,
    rng: np.random.Generator,
    *,
    partial: bool = False,
) -> None:
    """Draw a single hand-drawn ellipse into ``arr`` (modified in place).

    Uses the alpha-blended approach from ``scan_simulation._draw_imperfect_bubbles``:
    build an anti-aliased mask, draw a noisy grayscale fill plane, and
    blend. Mask is feathered with a 3×3 Gaussian so the edges look pencil-like.
    """
    cx, cy, radius = bubble["cx"], bubble["cy"], bubble["radius"]
    h_img, w_img = arr.shape[:2]

    shift = profile.shift_px
    dx = int(rng.integers(-shift, shift + 1)) if shift > 0 else 0
    dy = int(rng.integers(-shift, shift + 1)) if shift > 0 else 0
    scale_x = float(rng.uniform(*profile.scale_range))
    scale_y = float(rng.uniform(*profile.scale_range))
    axes = (
        max(2, int(radius * scale_x)),
        max(2, int(radius * scale_y)),
    )
    angle = float(rng.uniform(-profile.angle_deg, profile.angle_deg)) if profile.angle_deg > 0 else 0.0
    gray = int(rng.integers(profile.gray_range[0], profile.gray_range[1] + 1))

    if partial:
        # Random arc covering 140-260 degrees, starting at a random angle.
        arc_low, arc_high = profile.partial_arc_range
        arc = int(rng.integers(arc_low, arc_high + 1))
        start = float(rng.uniform(0, 360))
        end = start + arc
    else:
        start, end = 0.0, 360.0

    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (cx + dx, cy + dy),
        axes,
        angle,
        start,
        end,
        255,
        -1,
        lineType=cv2.LINE_AA,
    )
    noise = rng.normal(0, profile.noise_sigma, (h_img, w_img)).astype(np.int16)
    fill_plane = _clip_uint8(np.full((h_img, w_img), gray, dtype=np.int16) + noise)
    alpha = (cv2.GaussianBlur(mask, (3, 3), 0.5).astype(np.float32) / 255.0)[..., None]
    fill_rgb = np.dstack([fill_plane] * arr.shape[2])
    blended = arr.astype(np.float32) * (1.0 - alpha) + fill_rgb.astype(np.float32) * alpha
    arr[:] = _clip_uint8(blended)


def _draw_one_check(
    arr: np.ndarray,
    bubble: dict[str, Any],
    profile: MarkingProfile,
    rng: np.random.Generator,
) -> None:
    """Draw a check mark (✓) inside the bubble."""
    cx, cy, radius = bubble["cx"], bubble["cy"], bubble["radius"]
    gray = int(rng.integers(profile.gray_range[0], profile.gray_range[1] + 1))
    color = (gray, gray, gray) if arr.ndim == 3 else gray
    dx = int(rng.integers(-profile.shift_px, profile.shift_px + 1))
    dy = int(rng.integers(-profile.shift_px, profile.shift_px + 1))
    # Check mark: short stroke down-right, longer stroke up-right.
    p1 = (cx - radius + dx, cy + dy)
    p2 = (cx - radius // 3 + dx, cy + int(radius * 0.7) + dy)
    p3 = (cx + radius + dx, cy - radius + dy)
    thickness = max(1, int(round(radius / 3)))
    cv2.line(arr, p1, p2, color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.line(arr, p2, p3, color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_one_cross(
    arr: np.ndarray,
    bubble: dict[str, Any],
    profile: MarkingProfile,
    rng: np.random.Generator,
) -> None:
    """Draw an X through the bubble."""
    cx, cy, radius = bubble["cx"], bubble["cy"], bubble["radius"]
    gray = int(rng.integers(profile.gray_range[0], profile.gray_range[1] + 1))
    color = (gray, gray, gray) if arr.ndim == 3 else gray
    dx = int(rng.integers(-profile.shift_px, profile.shift_px + 1))
    dy = int(rng.integers(-profile.shift_px, profile.shift_px + 1))
    r = radius + 1
    thickness = max(1, int(round(radius / 3)))
    cv2.line(
        arr,
        (cx - r + dx, cy - r + dy),
        (cx + r + dx, cy + r + dy),
        color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.line(
        arr,
        (cx - r + dx, cy + r + dy),
        (cx + r + dx, cy - r + dy),
        color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_one_mark(
    arr: np.ndarray,
    bubble: dict[str, Any],
    profile: MarkingProfile,
    rng: np.random.Generator,
) -> None:
    style = profile.style
    if style == "ellipse":
        _draw_one_ellipse(arr, bubble, profile, rng, partial=False)
    elif style == "partial":
        _draw_one_ellipse(arr, bubble, profile, rng, partial=True)
    elif style == "check":
        _draw_one_check(arr, bubble, profile, rng)
    elif style == "cross":
        _draw_one_cross(arr, bubble, profile, rng)
    elif style == "none":
        return
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown marking style: {style!r}")


def _draw_stray_mark(
    arr: np.ndarray,
    bubble: dict[str, Any],
    profile: MarkingProfile,
    rng: np.random.Generator,
) -> None:
    """A tiny mark/scribble near (but outside) a real bubble.

    Stays within ±2 bubble radii of the bubble centre so it never lands far
    from the legitimate region. Stray marks are *visual* noise — they should
    be small enough that the OMR engine ignores them when integrating over
    the canonical bubble cell.
    """
    cx, cy, radius = bubble["cx"], bubble["cy"], bubble["radius"]
    h_img, w_img = arr.shape[:2]
    # Pick a random offset just outside the bubble (radius * 1.4..2.0).
    angle = float(rng.uniform(0, 2 * np.pi))
    dist = float(rng.uniform(radius * 1.4, radius * 2.0))
    ox = int(round(np.cos(angle) * dist))
    oy = int(round(np.sin(angle) * dist))
    sx = max(0, min(w_img - 1, cx + ox))
    sy = max(0, min(h_img - 1, cy + oy))
    stray_len = max(2, int(round(radius * rng.uniform(0.3, 0.8))))
    gray = int(rng.integers(80, 180))
    color = (gray, gray, gray) if arr.ndim == 3 else gray
    cv2.line(
        arr,
        (sx, sy),
        (sx + int(rng.integers(-stray_len, stray_len + 1)),
         sy + int(rng.integers(-stray_len, stray_len + 1))),
        color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )


def _draw_erasure(
    arr: np.ndarray,
    bubble: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    """A faint smudge across a bubble that was rubbed out."""
    cx, cy, radius = bubble["cx"], bubble["cy"], bubble["radius"]
    h_img, w_img = arr.shape[:2]
    patch_w = max(6, int(radius * rng.uniform(1.4, 2.2)))
    patch_h = max(3, int(radius * rng.uniform(0.5, 0.9)))
    x0 = max(0, cx - patch_w // 2)
    y0 = max(0, cy - patch_h // 2)
    x1 = min(w_img, x0 + patch_w)
    y1 = min(h_img, y0 + patch_h)
    smudge = int(rng.integers(180, 220))
    arr[y0:y1, x0:x1] = np.maximum(arr[y0:y1, x0:x1], smudge)


# ---------------------------------------------------------------------------
# Public drawing entry point.
# ---------------------------------------------------------------------------
def draw_student_marks(
    image: Image.Image,
    *,
    answers: dict[int, list[int]] | Any,
    marking_profile: str | MarkingProfile = DEFAULT_MARKING_PROFILE,
    candidate_number: str | None = None,
    seed: int | None = None,
) -> Image.Image:
    """Return a new ``PIL.Image`` with answer bubbles marked.

    ``answers`` may be a canonical ``dict[int, list[int]]`` (as produced by
    :func:`parse_answers`) or anything that :func:`parse_answers` accepts.

    The original image is never modified. The output is a new PIL Image of
    the same size and mode. The result is deterministic for a given input
    tuple (answers, marking_profile, candidate_number, seed).
    """
    if image is None:
        raise ValueError("image must not be None.")

    if isinstance(marking_profile, MarkingProfile):
        profile = marking_profile
    else:
        profile = MARKING_PROFILES[normalize_marking_profile(marking_profile)]

    if profile.style == "none":
        return image  # No-op; original returned unchanged.

    if not isinstance(answers, dict) or not all(
        isinstance(k, int) and isinstance(v, list) for k, v in answers.items()
    ):
        answers = parse_answers(answers, seed=seed)

    if not answers:
        return image

    rng = _rng_for(answers, profile, candidate_number, seed)

    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    geometry = answer_bubble_geometry(image.size[0], image.size[1])
    bubble_idx = _build_bubble_index(geometry)

    # Build the actual draw list with multi-marks expanded.
    draw_list: list[dict[str, Any]] = []
    for q, opts in sorted(answers.items()):
        if not (1 <= q <= NUM_QUESTIONS):
            continue
        # Multi-mark expansion: with probability multi_mark_prob, add one
        # extra random option (different from the chosen one) to simulate
        # a student who hedged. Skip when ``opts`` already has multiple
        # marks — the operator's explicit choice wins.
        effective_opts = list(opts)
        if (
            len(effective_opts) == 1
            and profile.multi_mark_prob > 0
            and float(rng.uniform(0, 1)) < profile.multi_mark_prob
        ):
            alt_candidates = [o for o in range(NUM_OPTIONS) if o not in effective_opts]
            if alt_candidates:
                effective_opts.append(int(rng.choice(alt_candidates)))
                effective_opts.sort()
        for opt in effective_opts:
            if not (0 <= opt < NUM_OPTIONS):
                continue
            bubble = bubble_idx.get((q, opt))
            if bubble is None:
                continue
            draw_list.append(bubble)

    for bubble in draw_list:
        _draw_one_mark(arr, bubble, profile, rng)
        if profile.stray_mark_prob > 0 and float(rng.uniform(0, 1)) < profile.stray_mark_prob:
            _draw_stray_mark(arr, bubble, profile, rng)
        if profile.erasure_prob > 0 and float(rng.uniform(0, 1)) < profile.erasure_prob:
            _draw_erasure(arr, bubble, rng)

    return Image.fromarray(arr, mode="RGB")


def answers_summary(answers: dict[int, list[int]]) -> str:
    """Return a compact human-readable string of the canonical answers map."""
    parts = []
    for q in range(1, NUM_QUESTIONS + 1):
        opts = answers.get(q, [])
        if not opts:
            parts.append("-")
        else:
            parts.append("".join(_OPTION_LETTERS[o] for o in opts))
    return "".join(parts)
