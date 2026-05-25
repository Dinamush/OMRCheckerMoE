"""Sample bubble intensities directly to debug the threshold logic.

For a chosen test case, this loads the filled PNG, computes the mean
intensity at each candidate-number bubble using the same integral-image
sampling the engine uses, and prints them sorted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from portrait_25q.variants.sweep.generate_sheet import (
    OMR_H,
    OMR_W,
    SheetSpec,
    default_sweep_specs,
)


def sample_intensities(image_path: Path, spec: SheetSpec, processing_w: int = OMR_W, processing_h: int = OMR_H) -> list[tuple[int, int, float]]:
    """Return (col, digit, mean_intensity) for every candidate bubble."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    # Mimic engine: resize input to processing dims (same as our OMR space).
    img = cv2.resize(img, (processing_w, processing_h), interpolation=cv2.INTER_AREA)
    integral = cv2.integral(img)
    box_w = box_h = spec.cand_bubble_diam
    results = []
    for col in range(10):
        for digit in range(10):
            # Match the engine: pt is the TOP-LEFT of the sampling box.
            cx = spec.cand_origin[0] + col * spec.cand_bubbles_gap_x
            cy = spec.cand_origin[1] + digit * spec.cand_labels_gap_y
            x1 = int(cx - box_w / 2)
            y1 = int(cy - box_h / 2)
            x2 = x1 + box_w
            y2 = y1 + box_h
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(integral.shape[1] - 1, x2)
            y2 = min(integral.shape[0] - 1, y2)
            mean = (
                integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
            ) / max(1, (x2 - x1) * (y2 - y1))
            results.append((col, digit, float(mean)))
    return results


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    runs = repo_root / "portrait_25q" / "variants" / "sweep" / "runs"
    spec_by_label = {s.label: s for s in default_sweep_specs()}

    # Pick a handful of representative cases that span both passing and failing.
    cases = [
        ("size_10", "pen_ink",       "1234567890"),
        ("size_10", "dark_pencil",   "1234567890"),
        ("size_10", "medium_pencil", "0123456789"),
        ("size_10", "light_pencil",  "0123456789"),
        ("size_14", "pen_ink",       "0123456789"),
        ("size_14", "medium_pencil", "0123456789"),
    ]
    for size_label, darkness, gt in cases:
        spec = spec_by_label[size_label]
        case_dir = runs / size_label / f"{darkness}__{gt}"
        image_path = case_dir / f"filled_{gt}.png"
        if not image_path.exists():
            print(f"SKIP: {image_path} not found")
            continue
        intensities = sample_intensities(image_path, spec)
        intensities.sort(key=lambda t: t[2])
        print(f"\n=== {size_label} / {darkness} / GT={gt} ===")
        print("  Darkest 12 bubbles (col, digit, mean_intensity) — should be the 10 GT marks:")
        for col, digit, mean in intensities[:12]:
            tag = " <- GT" if str(gt[col]) == str(digit) else ""
            print(f"    col={col} digit={digit} mean={mean:7.2f}{tag}")
        print("  Brightest 5 bubbles:")
        for col, digit, mean in intensities[-5:]:
            print(f"    col={col} digit={digit} mean={mean:7.2f}")
        # Gap analysis
        sorted_means = [m for *_, m in intensities]
        gaps = [(i, sorted_means[i+1] - sorted_means[i]) for i in range(len(sorted_means) - 1)]
        largest_gap = max(gaps, key=lambda t: t[1])
        print(f"  Largest gap in sorted intensities: position {largest_gap[0]} "
              f"({sorted_means[largest_gap[0]]:.1f} -> {sorted_means[largest_gap[0]+1]:.1f}, "
              f"jump={largest_gap[1]:.1f})")


if __name__ == "__main__":
    main()
