"""Canonical answer-sheet directory names and legacy aliases.

This module is the single source of truth for MoE sheet folder names after
the April/May 2026 naming cleanup. Import paths from here instead of
hard-coding old directory strings (``custom_25_definitive_final``, etc.).

Physical directories live at the repo root. Names use hyphens because they
are data/preset bundles, not Python packages. The portrait variants
tooling tree (sweep scripts) lives inside
:data:`VARIANTS_V1_DIR`; those scripts use sibling imports when run from
the ``variants/sweep/`` directory.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Current canonical directory names (repo root) ─────────────────────

LANDSCAPE_NNQ25_0 = "MoE-April-2026-Landscape-NNQ25-0"
PORTRAIT_NNQ25_0 = "MoE-April-2026-Portrait-NNQ25-0"
VARIANTS_SMQ25_0 = "MoE-May-2026-Variants-SMQ25-0"
PORTRAIT_SMQ25_1 = "MoE-May-2026-Portrait-SMQ25-1"

PREFILL_PACKAGE_DIR = "prefill_package"

# Logical preset for portrait v1/v2 variant routing (not a physical folder).
PORTRAIT_SMQ25_LOGICAL = "MoE-May-2026-Portrait-SMQ25"

# ── Legacy names (pre-2026-05 rename) ───────────────────────────────────

LEGACY_LANDSCAPE = "custom_25_definitive_final"
LEGACY_PORTRAIT_NN = "old_custom25_answer_sheet_v1"
LEGACY_VARIANTS_V1 = "portrait_25q"
LEGACY_PORTRAIT_V2 = "portrait_25q_v2"
LEGACY_PREFILL_PACKAGE = "prefill_only_package"

# preset name / settings string -> canonical name (logical or physical)
PRESET_LEGACY_ALIASES: dict[str, str] = {
    LEGACY_LANDSCAPE: LANDSCAPE_NNQ25_0,
    LEGACY_PORTRAIT_NN: PORTRAIT_NNQ25_0,
    LEGACY_VARIANTS_V1: PORTRAIT_SMQ25_LOGICAL,
    LEGACY_PORTRAIT_V2: PORTRAIT_SMQ25_1,
}

# ── Resolved paths ────────────────────────────────────────────────────

LANDSCAPE_DIR = REPO_ROOT / LANDSCAPE_NNQ25_0
PORTRAIT_NN_DIR = REPO_ROOT / PORTRAIT_NNQ25_0
VARIANTS_V1_DIR = REPO_ROOT / VARIANTS_SMQ25_0
PORTRAIT_V2_DIR = REPO_ROOT / PORTRAIT_SMQ25_1
PREFILL_PACKAGE_PATH = REPO_ROOT / PREFILL_PACKAGE_DIR


def normalize_preset_name(name: str) -> str:
    """Map a preset identifier to its canonical form (aliases + strip)."""
    text = (name or "").strip()
    if not text:
        return text
    return PRESET_LEGACY_ALIASES.get(text, text)
