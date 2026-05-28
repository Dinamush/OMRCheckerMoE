"""One-shot materialiser for the ``MoE-May-2026-Portrait-SMQ25-1`` preset.

Writes static ``template.json``, ``config.json`` and a reference blank PNG
into ``MoE-May-2026-Portrait-SMQ25-1/`` using the sweep-winning Variant-A geometry
(12 OMR-px candidate bubbles, 14 OMR-px answer bubbles, optimised spacings
and origins). The output is intentionally tracked in git so the preset
behaves like any other static preset and does not depend on the sweep
harness at runtime.
"""
from __future__ import annotations

from pathlib import Path

from generate_sheet import (
    SheetSpec,
    emit_config,
    emit_template,
    render_blank,
)


PRESET_DIR = Path(__file__).resolve().parents[3] / "MoE-May-2026-Portrait-SMQ25-1"
REFERENCE_PNG = PRESET_DIR / "reference" / "blank_MoE-May-2026-Portrait-SMQ25-1.png"

WINNING_SPEC = SheetSpec(
    label="size_12",
    cand_bubble_diam=12,
    cand_bubbles_gap_x=28.0,
    cand_labels_gap_y=15.0,
    cand_origin=(118, 252),
)


def main() -> None:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    (PRESET_DIR / "reference").mkdir(parents=True, exist_ok=True)
    tpl_path = emit_template(WINNING_SPEC, PRESET_DIR)
    cfg_path = emit_config(PRESET_DIR)
    img = render_blank(WINNING_SPEC)
    img.save(REFERENCE_PNG, dpi=(200, 200))
    print(f"Wrote {tpl_path.relative_to(PRESET_DIR.parent)}")
    print(f"Wrote {cfg_path.relative_to(PRESET_DIR.parent)}")
    print(f"Wrote {REFERENCE_PNG.relative_to(PRESET_DIR.parent)}")


if __name__ == "__main__":
    main()
