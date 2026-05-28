"""Preset management — named template+config bundles.

A preset is any direct subdirectory of ``settings.presets_dir`` that
contains a ``template.json`` file.  The directory may also contain
``config.json``, ``evaluation.json``, and non-JSON template asset files
(marker images etc.) which are all copied together when a preset is
applied to a batch.

Variant routing
---------------

The user-facing preset :data:`PORTRAIT_SMQ25_LOGICAL` is *variant-aware*:
which on-disk directory backs it depends on the runtime setting
``sheet_variant``. By default (``v1_legacy``) the variants tree
:data:`VARIANTS_SMQ25_0` is used; flipping the setting to ``v2_optimized``
swaps in :data:`PORTRAIT_SMQ25_1` without renaming anything in batches,
exports, or templates.

Canonical directory names and legacy aliases live in
:mod:`webui.sheet_registry`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from webui.settings import Settings, get_settings
from webui.sheet_registry import (
    PORTRAIT_SMQ25_1,
    PORTRAIT_SMQ25_LOGICAL,
    PRESET_LEGACY_ALIASES,
    VARIANTS_SMQ25_0,
    normalize_preset_name,
)

# Logical preset name -> { variant_value: physical_directory_name }
_VARIANT_ROUTES: dict[str, dict[str, str]] = {
    PORTRAIT_SMQ25_LOGICAL: {
        "v1_legacy": VARIANTS_SMQ25_0,
        "v2_optimized": PORTRAIT_SMQ25_1,
    },
}

# Physical dirs only reachable via variant routing — hidden from preset lists.
_HIDDEN_VARIANT_DIRS: frozenset[str] = frozenset({
    VARIANTS_SMQ25_0,
    PORTRAIT_SMQ25_1,
})


def _active_variant(settings: Settings, preset_name: str) -> str | None:
    """Return the active variant for ``preset_name`` or ``None`` if untouched."""
    if preset_name in _VARIANT_ROUTES:
        return settings.sheet_variant
    return None


def _resolve_preset_dir(settings: Settings, preset_name: str) -> Path:
    """Map a user-facing preset name to its on-disk directory."""
    canonical = normalize_preset_name(preset_name)
    variant = _active_variant(settings, canonical)
    if variant is not None:
        physical_name = _VARIANT_ROUTES[canonical].get(variant)
        if physical_name is None:
            raise ValueError(
                f"Unknown variant {variant!r} for preset {canonical!r}; "
                f"valid: {sorted(_VARIANT_ROUTES[canonical])}."
            )
        return _safe_preset_path(settings, physical_name)
    return _safe_preset_path(settings, canonical)


def list_presets(settings: Settings | None = None) -> list[str]:
    """Return sorted names of all available presets."""
    s = settings or get_settings()
    results: list[str] = []
    try:
        for subdir in sorted(s.presets_dir.iterdir()):
            if subdir.name in _HIDDEN_VARIANT_DIRS:
                continue
            if subdir.is_dir() and (subdir / "template.json").exists():
                results.append(subdir.name)
    except OSError:
        pass
    for logical in sorted(_VARIANT_ROUTES):
        if logical not in results:
            results.append(logical)
    return sorted(results)


def get_preset_documents(
    preset_name: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Return a dict of {doc_name: content} for all JSON docs in the preset."""
    s = settings or get_settings()
    preset_dir = _resolve_preset_dir(s, preset_name)
    if not preset_dir.is_dir():
        return {}
    docs: dict[str, Any] = {}
    for doc in ("template", "config", "evaluation"):
        path = preset_dir / f"{doc}.json"
        if path.exists():
            docs[doc] = json.loads(path.read_text(encoding="utf-8"))
    return docs


def apply_preset_to_batch(
    batch_root: Path,
    preset_name: str,
    settings: Settings | None = None,
) -> None:
    """Copy all files from a preset directory into a batch root."""
    s = settings or get_settings()
    preset_dir = _resolve_preset_dir(s, preset_name)
    if not preset_dir.is_dir():
        raise ValueError(f"Preset {preset_name!r} not found.")
    for item in preset_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, batch_root / item.name)


def _safe_preset_path(settings: Settings, preset_name: str) -> Path:
    """Resolve a preset name to its directory, rejecting path-traversal attempts."""
    candidate = (settings.presets_dir / preset_name).resolve()
    if settings.presets_dir.resolve() not in candidate.parents:
        raise ValueError(f"Invalid preset name: {preset_name!r}")
    return candidate


# Re-export for tests that introspect legacy alias table.
LEGACY_PRESET_ALIASES = PRESET_LEGACY_ALIASES
