"""Regenerate the portrait Legal 60Q blank + template.

Run from repo root::

    python MoE-July-2026-Portrait-SMQ60-0/generate_blank.py
"""
from __future__ import annotations

import runpy
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sheets" / "generate_legal_smq60.py"

if __name__ == "__main__":
    # Re-exec shared generator for this orientation only.
    import sys
    sys.argv = [str(SCRIPT), "--orientation", "portrait"]
    runpy.run_path(str(SCRIPT), run_name="__main__")
