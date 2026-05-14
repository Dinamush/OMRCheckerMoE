"""Preset management — named template+config bundles.

A preset is any direct subdirectory of ``settings.presets_dir`` that
contains a ``template.json`` file.  The directory may also contain
``config.json``, ``evaluation.json``, and non-JSON template asset files
(marker images etc.) which are all copied together when a preset is
applied to a batch.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from webui.settings import Settings, get_settings


def list_presets(settings: Settings | None = None) -> list[str]:
    """Return sorted names of all available presets."""
    s = settings or get_settings()
    results: list[str] = []
    try:
        for subdir in sorted(s.presets_dir.iterdir()):
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

    Returns an empty dict if the preset does not exist.
    """
    s = settings or get_settings()
    preset_dir = _safe_preset_path(s, preset_name)
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
    Non-JSON files (marker images etc.) are also copied as template assets.
    Raises ``ValueError`` if the preset does not exist.
    """
    s = settings or get_settings()
    preset_dir = _safe_preset_path(s, preset_name)
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
