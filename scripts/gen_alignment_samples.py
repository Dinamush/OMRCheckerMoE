"""Generate a gallery of prefilled sheets across all profiles + key answer
patterns, then OMR each one and report read accuracy.

Used to visually + numerically verify the bubble geometry calibration.

Run from repo root: ``python scripts/gen_alignment_samples.py``
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from webui.services import prefill as prefill_service  # noqa: E402

OUT_DIR = REPO_ROOT / "diagnostic_output" / "alignment_gallery"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_JSON = REPO_ROOT / "custom_25_definitive_final" / "template.json"

PROFILES = [
    "light_pencil",
    "medium_pencil",
    "heavy_pencil",
    "pen_ballpoint",
    "check_mark",
    "cross_mark",
    "partial_fill",
    "messy_student",
    "careful_student",
]
ANSWER_KEYS = [
    ("all_a", "all_a"),
    ("all_b", "all_b"),
    ("all_c", "all_c"),
    ("all_d", "all_d"),
    ("alternating", "alternating"),
    ("random_seed", "random"),
]


def render(profile: str, answers: str) -> Path:
    png = prefill_service.generate_single_png(
        f"{profile}/{answers}", "Test School", "Alignment Test", "9010690012",
        marking_profile=profile, answers=answers,
    )
    out = OUT_DIR / f"{profile}__{answers}.png"
    out.write_bytes(png)
    return out


def run_omr(png_path: Path) -> dict:
    cfg = {
        "dimensions": {
            "processing_height": 515, "processing_width": 666,
            "display_height": 515, "display_width": 666,
        },
        "outputs": {"show_image_level": 0},
    }
    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        (td / "template_base").mkdir()
        shutil.copy2(TEMPLATE_JSON, td / "template_base" / "template.json")
        (td / "omr_out").mkdir()
        from src.entry import entry_point_for_image
        try:
            entry_point_for_image(
                image_path=str(png_path),
                output_dir=str(td / "omr_out"),
                template_payload=json.loads(TEMPLATE_JSON.read_text(encoding="utf-8")),
                config_payload=cfg,
                template_dir=str(td / "template_base"),
                rotation_degrees=0,
            )
        except Exception as e:
            print(f"  OMR error: {e!s}")
            return {}
        csvs = list((td / "omr_out" / "Results").glob("Results_*.csv"))
        if not csvs:
            return {}
        return list(csv.DictReader(csvs[0].open(encoding="utf-8")))[0]


def expected_answers(key: str) -> dict[int, str]:
    if key in ("all_a", "all_b", "all_c", "all_d"):
        letter = key[-1].upper()
        return {q: letter for q in range(1, 26)}
    if key == "alternating":
        return {q: ["A", "B", "C", "D"][(q - 1) % 4] for q in range(1, 26)}
    return {}


if __name__ == "__main__":
    print(f"Output dir: {OUT_DIR}")
    print(f"{'profile':<16} {'answers':<14} | OMR matches")
    print("-" * 50)
    for profile in PROFILES:
        for label, ans_key in ANSWER_KEYS:
            png = render(profile, ans_key)
            row = run_omr(png)
            expected = expected_answers(ans_key)
            if not expected:
                print(f"{profile:<16} {label:<14} | (no expected — rendered only)")
                continue
            correct = sum(
                1 for q, letter in expected.items()
                if (row.get(f"q{q}") or "NR").strip() == letter
            )
            total = len(expected)
            print(f"{profile:<16} {label:<14} | {correct}/{total}")
