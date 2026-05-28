"""End-to-end realism bench: how many moderate/adversarial scan-simulated
sheets does the cropper successfully process?

This is the metric that matters in production. The synthetic dog-ear
benchmark exercises the absolute worst cases (aggressive folds at 24-px
markers); this script exercises the realistic distribution of damage that
``apply_scan_simulation`` produces, including all the small geometric
warps, lighting drifts, JPEG roundtrips, and noise that the user's actual
sheets see.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from statistics import median, mean

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotmap import DotMap

from webui.services import prefill as prefill_service
from src.processors.CropOnMarkers import CropOnMarkers


PAGE_W, PAGE_H = 666, 515
REFERENCE_CENTRES = [
    [13.5, 13.2], [651.5, 13.2], [13.5, 499.0], [651.5, 499.0],
]


class _Stub:
    def __init__(self) -> None:
        self.tuning_config = DotMap(
            {
                "outputs": {"show_image_level": 0},
                "dimensions": {
                    "display_width": PAGE_W,
                    "display_height": PAGE_H,
                    "processing_width": PAGE_W,
                    "processing_height": PAGE_H,
                },
            },
            _dynamic=False,
        )

    def append_save_img(self, *a, **k) -> None:
        pass


def make_cropper() -> CropOnMarkers:
    return CropOnMarkers(
        options={
            "type": "aruco",
            "arucoDictionary": "DICT_4X4_50",
            "arucoCornerIds": [0, 1, 2, 3],
            "preserveFullImage": True,
            "referenceMarkerCenters": REFERENCE_CENTRES,
        },
        relative_dir=str(REPO),
        image_instance_ops=_Stub(),
    )


def render_sheet(seed: int, preset: str) -> np.ndarray:
    candidate = f"{9000000000 + seed:010d}"
    png = prefill_service.generate_single_png(
        f"Student {seed}", "Test School", "National Test", candidate, realism_preset=preset
    )
    pil = Image.open(io.BytesIO(png)).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (PAGE_W, PAGE_H), interpolation=cv2.INTER_AREA)
    return bgr


def run(rows: int, preset: str) -> dict:
    cropper = make_cropper()
    successes = 0
    times_ms: list[float] = []
    for i in range(rows):
        sheet = render_sheet(i, preset)
        t0 = time.perf_counter()
        try:
            res = cropper._apply_aruco_filter(sheet, f"sheet-{i}.png")
        except Exception:
            res = None
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        if res is not None:
            successes += 1
    return {
        "preset": preset,
        "rows": rows,
        "success_pct": round(100.0 * successes / rows, 1) if rows else 0.0,
        "successes": successes,
        "median_ms": round(median(times_ms), 1),
        "mean_ms": round(mean(times_ms), 1),
        "p95_ms": round(float(np.percentile(times_ms, 95)), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--label", type=str, default="run")
    parser.add_argument("--out", type=Path, default=REPO / "bench_realism.json")
    args = parser.parse_args()

    presets = ["none", "subtle", "moderate", "adversarial"]
    results = []
    for preset in presets:
        r = run(args.rows, preset)
        print(
            f"{preset:>12} | success {r['success_pct']:>5.1f}% "
            f"| median {r['median_ms']:>6} ms"
        )
        results.append(r)

    payload = {"label": args.label, "rows": args.rows, "presets": results}
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
