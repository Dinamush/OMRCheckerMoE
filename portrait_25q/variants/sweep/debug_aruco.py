"""One-off ArUco detection debug for the generated sheet.

Loads the most recent smoke-test image, runs the engine's own detector,
and prints detected marker centres + the resulting marker-quad aspect.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def detect_aruco(path: Path, *, resize: tuple[int, int] | None = None) -> dict:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": f"could not read {path}"}
    h0, w0 = img.shape[:2]
    if resize is not None:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    pad = 60
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners_raw, ids_raw, rejected = detector.detectMarkers(padded)
    if ids_raw is None or len(ids_raw) == 0:
        return {
            "original_shape": (h0, w0),
            "resized_shape": (h, w),
            "detected": {},
            "rejected_count": 0 if rejected is None else len(rejected),
        }
    detected = {}
    for idx, marker_id in enumerate(ids_raw.flatten()):
        centre = (corners_raw[idx] - [[pad, pad]])[0].mean(axis=0)
        detected[int(marker_id)] = (float(centre[0]), float(centre[1]))
    return {
        "original_shape": (h0, w0),
        "resized_shape": (h, w),
        "detected": detected,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    paths = [
        repo_root / "portrait_25q" / "reference" / "blank_portrait_25q.png",
        repo_root / "portrait_25q" / "variants" / "variant_a_portrait.png",
    ]
    sweep_runs = repo_root / "portrait_25q" / "variants" / "sweep" / "runs"
    if sweep_runs.exists():
        paths.extend(sorted(sweep_runs.rglob("filled_*.png"))[:1])

    for p in paths:
        print(f"\n--- {p.relative_to(repo_root)} ---")
        for label, resize in [
            ("native", None),
            ("processing 666x862", (666, 862)),
            ("processing 666x820", (666, 820)),
        ]:
            result = detect_aruco(p, resize=resize)
            print(f"  [{label}] {result['original_shape']} -> {result.get('resized_shape')}")
            det = result.get("detected", {})
            if not det:
                print("    no markers detected")
                continue
            for marker_id in sorted(det):
                x, y = det[marker_id]
                print(f"    marker {marker_id}: ({x:7.2f}, {y:7.2f})")
            if {0, 1, 2, 3}.issubset(det):
                tl = np.array(det[0])
                tr = np.array(det[1])
                bl = np.array(det[2])
                br = np.array(det[3])
                top_w = np.linalg.norm(tr - tl)
                bot_w = np.linalg.norm(br - bl)
                left_h = np.linalg.norm(bl - tl)
                right_h = np.linalg.norm(br - tr)
                horiz = (top_w + bot_w) / 2
                vert = (left_h + right_h) / 2
                print(
                    f"    marker quad: horiz={horiz:.2f}, vert={vert:.2f},"
                    f" aspect={horiz/vert:.3f}"
                )


if __name__ == "__main__":
    main()
