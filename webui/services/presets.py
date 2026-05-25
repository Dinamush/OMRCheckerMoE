"""Preset management — named template+config bundles.

A preset is any direct subdirectory of ``settings.presets_dir`` that
contains a ``template.json`` file.  The directory may also contain
``config.json``, ``evaluation.json``, and non-JSON template asset files
(marker images etc.) which are all copied together when a preset is
applied to a batch.

Variant routing
---------------

The user-visible preset name ``portrait_25q`` is *variant-aware*: which
on-disk directory backs it depends on the runtime setting
``sheet_variant``. By default (``v1_legacy``) the original ``portrait_25q/``
directory is used; flipping the setting to ``v2_optimized`` swaps in the
sweep-validated ``portrait_25q_v2/`` preset without renaming anything in
batches, exports, or templates.
The list / get / apply functions all funnel through
:func:`_resolve_preset_dir` so the routing happens in exactly one place.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from webui.settings import Settings, get_settings


# Public preset name (what users / batches reference) -> the per-variant
# directory it should resolve to. Keys are the LOGICAL preset names; the
# inner mapping is { variant_value: physical_directory_name }.
#
# This indirection means a batch whose preset metadata says "portrait_25q"
# automatically picks up whichever variant is active at run time, without
# anyone having to migrate batch records on a settings change.
_VARIANT_ROUTES: dict[str, dict[str, str]] = {
    "portrait_25q": {
        "v1_legacy": "portrait_25q",
        "v2_optimized": "portrait_25q_v2",
    },
}

# Reverse: directory names that should not appear under their own name in
# the public preset listing because they are only reachable via variant
# routing. ``portrait_25q_v2`` is an implementation detail of the
# ``portrait_25q`` preset; surfacing it as a separate preset would
# duplicate it in the UI and let users pick the wrong one.
_HIDDEN_VARIANT_DIRS: frozenset[str] = frozenset({"portrait_25q_v2"})


def _active_variant(settings: Settings, preset_name: str) -> str | None:
    """Return the active variant for ``preset_name`` or ``None`` if untouched.

    Centralises the "which setting picks which physical dir" lookup so
    adding another variant-aware preset later is one new entry in
    ``_VARIANT_ROUTES`` plus one attribute on :class:`Settings`.
    """
    if preset_name == "portrait_25q":
        return settings.sheet_variant
    return None


def _resolve_preset_dir(settings: Settings, preset_name: str) -> Path:
    """Map a user-facing preset name to its on-disk directory.

    Variant-aware presets (currently only ``portrait_25q``) honour the
    active variant setting. Every other name resolves to a same-named
    directory under :attr:`Settings.presets_dir`. Path-traversal attempts
    (``..``, absolute paths) are rejected just like a flat lookup.
    """
    variant = _active_variant(settings, preset_name)
    if variant is not None:
        physical_name = _VARIANT_ROUTES[preset_name].get(variant)
        if physical_name is None:
            raise ValueError(
                f"Unknown variant {variant!r} for preset {preset_name!r}; "
                f"valid: {sorted(_VARIANT_ROUTES[preset_name])}."
            )
        return _safe_preset_path(settings, physical_name)
    return _safe_preset_path(settings, preset_name)


def list_presets(settings: Settings | None = None) -> list[str]:
    """Return sorted names of all available presets.

    Physical directories that exist only as variant backings (e.g.
    ``portrait_25q_v2``) are hidden so the UI shows the user-facing
    preset name once, not once per variant.
    """
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
    return results


def get_preset_documents(
    preset_name: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Return a dict of {doc_name: content} for all JSON docs in the preset.

    Returns an empty dict if the preset does not exist. Variant-aware
    presets resolve to the directory selected by their variant setting.
    """
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
    """Copy all files from a preset directory into a batch root.

    JSON documents (template, config, evaluation) are copied directly.
    Non-JSON files (marker images etc.) are also copied as template
    assets. Variant-aware presets resolve to the directory selected by
    their variant setting at call time. Raises ``ValueError`` if the
    preset does not exist.
    """
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
