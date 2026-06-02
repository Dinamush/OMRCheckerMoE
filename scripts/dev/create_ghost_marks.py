"""
create_ghost_marks.py

Reads the real MoE landscape scan, detects its ArUco corner markers to compute
the perspective warp, then adds ghost marks (light pencil touches) at *wrong*
answer bubble positions by mapping template coordinates → raw image coordinates
via the inverse perspective transform.

Ghost mark levels (relative to the real scan's empty-bubble background):
  VERY_FAINT  adj_gap  1-3  → should be NR   (1-unit scan noise)
  FAINT       adj_gap  5-10 → likely NR       (below adj_gap=2 guard by feel)
  MEDIUM      adj_gap 15-25 → WILL trigger    (realistic lightly-filled bubble)

MEDIUM marks are the adversarial case: a student lightly touches the wrong
bubble — will the OMR produce a ghost answer?

Output: prefilled_sheet_ghost_marks.png  (same folder as the original input)
"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

INPUT   = ROOT / "webui/storage/batches/4303aca72e01/inputs/prefilled_sheet_moderate_medium_pencil.png"
OUT_DIR = ROOT / "webui/storage/batches/4303aca72e01/inputs"

OVERSAMPLE = 2   # must match batch config OVERSAMPLE_SCALE

# Template dimensions (OMR-space)
PAGE_W, PAGE_H = 666, 515
BUBBLE = 10       # bubble side in OMR-space pixels

# Reference marker centres in OMR-space (from template.json)
MARKER_CENTRES_OMR = np.float32([
    [13.5, 13.2],    # id 0  top-left
    [651.5, 13.2],   # id 1  top-right
    [13.5, 499.0],   # id 2  bottom-left
    [651.5, 499.0],  # id 3  bottom-right
])

# ── Template block definitions ────────────────────────────────────────────────
# QTYPE_MCQ4 (horizontal): bubblesGap in X (A/B/C/D), labelsGap in Y (q1…qN)
BLOCKS = {
    "q01": {"origin": (52.7,  259.3), "bubblesGap": 20.0, "labelsGap": 41.9, "qs": list(range(1, 6))},
    "q06": {"origin": (180.9, 259.3), "bubblesGap": 20.0, "labelsGap": 41.9, "qs": list(range(6, 11))},
    "q11": {"origin": (309.8, 259.3), "bubblesGap": 19.8, "labelsGap": 41.9, "qs": list(range(11, 16))},
    "q16": {"origin": (435.9, 259.3), "bubblesGap": 20.0, "labelsGap": 41.9, "qs": list(range(16, 21))},
    "q21": {"origin": (566.3, 259.3), "bubblesGap": 20.0, "labelsGap": 41.9, "qs": list(range(21, 26))},
}
OPTIONS = ["A", "B", "C", "D"]

CORRECT_ANSWERS = {
    1: "A", 2: "C", 3: "C", 4: "B", 5: "C",
    6: "C", 7: "C", 8: "B", 9: "A", 10: "B",
    11: "B", 12: "B", 13: "A", 14: "D", 15: "B",
    16: "D", 17: "D", 18: "D", 19: "D", 20: "D",
    21: "A", 22: "D", 23: "B", 24: "A", 25: "A",
}

# ── Ghost mark scenarios ───────────────────────────────────────────────────────
# (question, wrong_option, intensity_offset_below_background, shape, label)
# intensity_offset is subtracted from the measured background mean to get the
# absolute intensity painted into the bubble ROI.
GHOST_SCENARIOS = [
    # VERY FAINT — intensity offset 2-3 (adj_gap ≈ 2-3) — on the boundary
    (1,  "B",  2, "corner",  "boundary_corner"),   # q1=A, ghost B
    (8,  "C",  2, "dot",     "boundary_dot"),       # q8=B, ghost C
    (18, "A",  3, "dot",     "boundary_dot"),       # q18=D, ghost A
    # FAINT — offset 8 (adj_gap ≈ 8) — clearly above guard
    (3,  "D",  8, "full",    "faint_full"),          # q3=C, ghost D
    (10, "A",  8, "corner",  "faint_corner"),        # q10=B, ghost A
    (15, "C",  8, "full",    "faint_full"),          # q15=B, ghost C
    # MEDIUM — offset 20 (adj_gap ≈ 20) — realistic "light pencil touch on wrong bubble"
    (5,  "A", 20, "full",    "medium_full"),         # q5=C, ghost A
    (13, "B", 20, "full",    "medium_full"),         # q13=A, ghost B
    (21, "D", 20, "corner",  "medium_corner"),       # q21=A, ghost D (far option)
    (24, "B", 20, "full",    "medium_full"),         # q24=A, ghost B
]


def block_for_q(q):
    for name, blk in BLOCKS.items():
        if q in blk["qs"]:
            return name
    raise ValueError(f"Question {q} not in any block")


def bubble_centre_omr(q, option):
    """Centre of bubble in OMR-space (float)."""
    blk = BLOCKS[block_for_q(q)]
    ox, oy = blk["origin"]
    q_idx  = blk["qs"].index(q)
    opt_idx = OPTIONS.index(option)
    cx = ox + opt_idx * blk["bubblesGap"] + BUBBLE / 2
    cy = oy + q_idx   * blk["labelsGap"]  + BUBBLE / 2
    return cx, cy


def detect_aruco_corners(gray):
    """Return detected marker centres keyed by marker id."""
    aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        raise RuntimeError(f"Need 4 ArUco markers, detected {0 if ids is None else len(ids)}")
    centres = {}
    for i, mid in enumerate(ids.flatten()):
        c = corners[i][0]
        centres[int(mid)] = c.mean(axis=0)  # (x, y) centre of marker
    return centres


def build_inverse_warp(raw_h, raw_w):
    """
    Compute the 3×3 homography that maps OMR-space coords → raw image coords.
    The forward warp (raw → OMR) is what CropOnMarkers applies; we invert it.
    """
    img_bgr = cv2.imread(str(INPUT))
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    centres = detect_aruco_corners(gray)

    # Raw-image marker centres (detected), ordered by marker id 0-3
    src_pts = np.float32([centres[i] for i in range(4)])   # in raw-image space
    # OMR-space destination (template reference centres * oversample)
    dst_pts = MARKER_CENTRES_OMR * OVERSAMPLE               # in 2×OMR space

    # Forward: raw → 2×OMR space
    H_fwd, _ = cv2.findHomography(src_pts, dst_pts)
    # Inverse: 2×OMR space → raw image
    H_inv = np.linalg.inv(H_fwd)
    return H_inv, gray


def omr_to_raw(cx_omr, cy_omr, H_inv):
    """Map a single point from OMR-space to raw-image space."""
    pt = np.array([[[cx_omr * OVERSAMPLE, cy_omr * OVERSAMPLE]]], dtype=np.float32)
    raw_pt = cv2.perspectiveTransform(pt, H_inv)
    return float(raw_pt[0, 0, 0]), float(raw_pt[0, 0, 1])


def fill_bubble_raw(img_gray, raw_cx, raw_cy, intensity, shape, bubble_r):
    """
    Paint a ghost mark around (raw_cx, raw_cy) in raw-image coords.
    bubble_r: half-side of the bubble in raw-image pixels.
    """
    h, w = img_gray.shape
    x0 = max(0, round(raw_cx - bubble_r))
    y0 = max(0, round(raw_cy - bubble_r))
    x1 = min(w, round(raw_cx + bubble_r))
    y1 = min(h, round(raw_cy + bubble_r))
    if x1 <= x0 or y1 <= y0:
        return

    if shape == "full":
        img_gray[y0:y1, x0:x1] = intensity
    elif shape == "corner":
        mid_x = (x0 + x1) // 2
        mid_y = (y0 + y1) // 2
        img_gray[y0:mid_y, x0:mid_x] = intensity
    elif shape == "dot":
        cx, cy = round(raw_cx), round(raw_cy)
        r = max(1, bubble_r // 3)
        img_gray[max(0,cy-r):cy+r, max(0,cx-r):cx+r] = intensity


def main():
    img_bgr = cv2.imread(str(INPUT))
    if img_bgr is None:
        raise FileNotFoundError(f"Input not found: {INPUT}")

    raw_h, raw_w = img_bgr.shape[:2]
    H_inv, gray = build_inverse_warp(raw_h, raw_w)

    # Bubble radius in raw-image pixels: BUBBLE/2 scaled to raw space.
    # Approximate scale: raw_w / (PAGE_W * OVERSAMPLE)
    raw_scale = raw_w / (PAGE_W * OVERSAMPLE)
    bubble_r  = max(3, round((BUBBLE / 2) * raw_scale))

    # ── Measure empty-bubble background in raw image (via inverse warp) ───────
    bg_vals = []
    for q, correct in CORRECT_ANSWERS.items():
        for opt in OPTIONS:
            if opt == correct:
                continue
            cx_omr, cy_omr = bubble_centre_omr(q, opt)
            rx, ry = omr_to_raw(cx_omr, cy_omr, H_inv)
            x0 = max(0, round(rx - bubble_r)); x1 = min(raw_w, round(rx + bubble_r))
            y0 = max(0, round(ry - bubble_r)); y1 = min(raw_h, round(ry + bubble_r))
            roi = gray[y0:y1, x0:x1]
            if roi.size:
                bg_vals.append(float(np.mean(roi)))

    fill_vals = []
    for q, correct in CORRECT_ANSWERS.items():
        cx_omr, cy_omr = bubble_centre_omr(q, correct)
        rx, ry = omr_to_raw(cx_omr, cy_omr, H_inv)
        x0 = max(0, round(rx - bubble_r)); x1 = min(raw_w, round(rx + bubble_r))
        y0 = max(0, round(ry - bubble_r)); y1 = min(raw_h, round(ry + bubble_r))
        roi = gray[y0:y1, x0:x1]
        if roi.size:
            fill_vals.append(float(np.mean(roi)))

    bg_mean = float(np.mean(bg_vals))
    fill_mean = float(np.mean(fill_vals))
    print(f"Background (empty bubble):  mean={bg_mean:.1f}  "
          f"min={min(bg_vals):.1f}  max={max(bg_vals):.1f}")
    print(f"Filled (correct answer):    mean={fill_mean:.1f}  "
          f"min={min(fill_vals):.1f}  max={max(fill_vals):.1f}")
    print(f"Real fill-to-bg gap:        {bg_mean - fill_mean:.1f} intensity units")
    print(f"Raw-image bubble radius:    {bubble_r} px  (raw_scale={raw_scale:.3f})\n")

    # ── Paint ghost marks ─────────────────────────────────────────────────────
    ghost_gray = gray.copy()
    for q, opt, offset, shape, label in GHOST_SCENARIOS:
        intensity = round(bg_mean - offset)
        cx_omr, cy_omr = bubble_centre_omr(q, opt)
        rx, ry = omr_to_raw(cx_omr, cy_omr, H_inv)
        fill_bubble_raw(ghost_gray, rx, ry, intensity, shape, bubble_r)
        correct = CORRECT_ANSWERS[q]
        print(f"  q{q:2d} {correct}→{opt}  intensity={intensity} "
              f"(bg-{offset})  adj_gap≈{offset}  shape={shape}  [{label}]")

    ghost_bgr = cv2.cvtColor(ghost_gray, cv2.COLOR_GRAY2BGR)
    out_path = OUT_DIR / "prefilled_sheet_ghost_marks.png"
    cv2.imwrite(str(out_path), ghost_bgr)
    print(f"\nSaved: {out_path}")
    print("Upload this file to the batch and run OMR to test for ghost detections.")
    print("\nExpected results:")
    print("  VERY FAINT (adj_gap 2-3): NR  — should NOT trigger adj_gap guard")
    print("  FAINT      (adj_gap 8):   NR  — above guard but correct answer dominates")
    print("  MEDIUM     (adj_gap 20):  ??? — adversarial; correct answer should still win")


if __name__ == "__main__":
    main()
