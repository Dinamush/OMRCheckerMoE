"""Diagnose where student-fill geometry lands vs actual bubble outlines.

Draws red crosshairs at every computed bubble centre on top of the blank
reference template, saves the result, and also runs the OMR engine on a
clean prefilled sheet to verify the OMR coordinates against the visible
bubbles.

Run from repo root: ``python scripts/debug_alignment.py``
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from webui.services import prefill as prefill_service  # noqa: E402
from webui.services.student_fill import answer_bubble_geometry  # noqa: E402

TEMPLATE_PNG = REPO_ROOT / "prefill_only_package" / "blank_template_reference.png"
OUT_DIR = REPO_ROOT / "diagnostic_output"
OUT_DIR.mkdir(exist_ok=True)


def overlay_geometry() -> None:
    """Draw red crosshairs where each answer-bubble centre lands."""
    img = Image.open(TEMPLATE_PNG).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    bubbles = answer_bubble_geometry(w, h)
    print(f"Canvas: {w}x{h} | bubbles: {len(bubbles)}")
    print(f"Q1 option A: cx={bubbles[0]['cx']} cy={bubbles[0]['cy']} r={bubbles[0]['radius']}")
    for b in bubbles:
        cx, cy, r = b["cx"], b["cy"], b["radius"]
        draw.line([(cx - r * 2, cy), (cx + r * 2, cy)], fill=(255, 0, 0), width=2)
        draw.line([(cx, cy - r * 2), (cx, cy + r * 2)], fill=(255, 0, 0), width=2)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 0, 0), width=2)
    out = OUT_DIR / "answer_geometry_overlay.png"
    img.save(out)
    print(f"Saved overlay -> {out}")


def render_with_fill(profile: str = "heavy_pencil", answers: str = "all_a") -> Path:
    """Generate a sheet with student-fill applied and save it."""
    png = prefill_service.generate_single_png(
        "Alignment Test", "Test School", "Test Exam", "9010690012",
        marking_profile=profile, answers=answers,
    )
    out = OUT_DIR / f"filled_{profile}_{answers}.png"
    out.write_bytes(png)
    print(f"Saved filled -> {out}")
    return out


def run_omr(png_path: Path) -> dict:
    """Run the OMR engine on a single sheet and return the read row."""
    template_json = REPO_ROOT / "custom_25_definitive_final" / "template.json"
    config_payload = {
        "dimensions": {
            "processing_height": 515,
            "processing_width": 666,
            "display_height": 515,
            "display_width": 666,
        },
        "outputs": {"show_image_level": 0},
    }
    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        (td / "template_base").mkdir()
        shutil.copy2(template_json, td / "template_base" / "template.json")
        (td / "omr_out").mkdir()
        from src.entry import entry_point_for_image
        entry_point_for_image(
            image_path=str(png_path),
            output_dir=str(td / "omr_out"),
            template_payload=json.loads(template_json.read_text(encoding="utf-8")),
            config_payload=config_payload,
            template_dir=str(td / "template_base"),
            rotation_degrees=0,
        )
        csvs = list((td / "omr_out" / "Results").glob("Results_*.csv"))
        if not csvs:
            return {}
        checked = list((td / "omr_out" / "CheckedOMRs").glob("*"))
        if checked:
            shutil.copy2(checked[0], OUT_DIR / f"checked_{png_path.stem}.png")
        return list(csv.DictReader(csvs[0].open(encoding="utf-8")))[0]


if __name__ == "__main__":
    overlay_geometry()
    png = render_with_fill("heavy_pencil", "all_a")
    row = run_omr(png)
    print(f"OMR read q1={row.get('q1')!r} q2={row.get('q2')!r} q5={row.get('q5')!r}")
    correct = sum(
        1 for q in range(1, 26)
        if (row.get(f"q{q}") or "NR").strip() == "A"
    )
    print(f"OMR correct: {correct}/25 as A")
