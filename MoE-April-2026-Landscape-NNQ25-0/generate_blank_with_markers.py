"""Render the legacy MoE-April-2026-Landscape-NNQ25-0 blank answer sheet with the 4
corner ArUco fiducials stamped on, matching the template's
``referenceMarkerCenters``. Produces both a PNG preview and a US-Letter
landscape PDF.

Run from the repo root::

    python MoE-April-2026-Landscape-NNQ25-0/generate_blank_with_markers.py

Outputs (next to the source JPG):
    blank_legacy_landscape_answer_sheet_with_markers.png
    blank_legacy_landscape_answer_sheet_with_markers.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SRC_JPG = HERE / "MCFormat25questions_page-0001.jpg"
TEMPLATE_JSON = HERE / "template.json"
OUT_PNG = HERE / "blank_legacy_landscape_answer_sheet_with_markers.png"
OUT_PDF = HERE / "blank_legacy_landscape_answer_sheet_with_markers.pdf"

PRINT_DPI = 300

# Marker side length in OMR-space units. The template's reference centres
# sit (13.5, 13.2) OMR-px from the corner; a 22 OMR-px square keeps an
# 11 OMR-px half side and leaves ~2.2 OMR-px (~1 mm physical) breathing
# room to the page edge, matching the existing filled_sample_markers.jpg.
ARUCO_MARKER_SIZE_OMR = 22
ARUCO_DICT = cv2.aruco.DICT_4X4_50


def main() -> None:
    template = json.loads(TEMPLATE_JSON.read_text(encoding="utf-8"))
    omr_w, omr_h = template["pageDimensions"]
    crop_opts = next(
        p["options"] for p in template["preProcessors"] if p["name"] == "CropOnMarkers"
    )
    centres_omr = crop_opts["referenceMarkerCenters"]
    marker_ids = crop_opts["arucoCornerIds"]

    img_pil = Image.open(SRC_JPG).convert("RGB")
    print_w, print_h = img_pil.size
    scale_x = print_w / omr_w
    scale_y = print_h / omr_h

    img_cv = np.array(img_pil)[:, :, ::-1].copy()
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker_px = round(ARUCO_MARKER_SIZE_OMR * scale_x)

    for (cx_omr, cy_omr), marker_id in zip(centres_omr, marker_ids):
        cx = int(round(cx_omr * scale_x))
        cy = int(round(cy_omr * scale_y))
        x0 = cx - marker_px // 2
        y0 = cy - marker_px // 2
        x1 = x0 + marker_px
        y1 = y0 + marker_px
        marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        img_cv[y0:y1, x0:x1] = marker_bgr

    stamped = Image.fromarray(img_cv[:, :, ::-1])
    stamped.save(OUT_PNG, format="PNG", optimize=True)
    stamped.save(OUT_PDF, format="PDF", resolution=float(PRINT_DPI))
    print(f"Wrote {OUT_PNG}  ({stamped.size[0]}x{stamped.size[1]})")
    print(f"Wrote {OUT_PDF}  ({OUT_PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
