"""Regenerate the Letter-landscape 60Q blank + sample + template.

Run from repo root::

    python MoE-July-2026-Letter-Landscape-SMQ60-0/generate_blank.py
"""
from __future__ import annotations

import runpy
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sheets" / "generate_letter_smq60.py"

if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")
