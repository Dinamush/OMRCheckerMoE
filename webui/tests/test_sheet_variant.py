"""Tests for the ``sheet_variant`` runtime setting.

The setting toggles which physical directory backs the user-facing
``portrait_25q`` preset:

* ``v1_legacy`` (default)  -> ``portrait_25q/`` (the original layout).
* ``v2_optimized``         -> ``portrait_25q_v2/`` (sweep-validated layout).

These tests verify the routing in three places it matters:

1. :func:`webui.services.presets.list_presets` hides the variant
   backing directory (``portrait_25q_v2``) so the public listing is
   unambiguous regardless of which variant is active.
2. :func:`webui.services.presets.get_preset_documents` returns the
   template/config of whichever variant is currently active.
3. :func:`webui.services.presets.apply_preset_to_batch` copies the
   active variant's files into the batch directory.

All tests run against the real ``Settings`` model with a tmp_path
``presets_dir`` so we can prove the routing without touching the live
repo presets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest

from webui.services import presets as presets_service
from webui.settings import Settings, get_settings


@pytest.fixture
def presets_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Build a tmp presets_dir with both portrait variants and one decoy."""
    monkeypatch.setenv("OMR_WEBUI_PRESETS_DIR", str(tmp_path))
    get_settings.cache_clear()

    (tmp_path / "portrait_25q").mkdir()
    (tmp_path / "portrait_25q" / "template.json").write_text(
        json.dumps({"variant_marker": "v1_legacy"}), encoding="utf-8"
    )

    (tmp_path / "portrait_25q_v2").mkdir()
    (tmp_path / "portrait_25q_v2" / "template.json").write_text(
        json.dumps({"variant_marker": "v2_optimized"}), encoding="utf-8"
    )
    (tmp_path / "portrait_25q_v2" / "config.json").write_text(
        json.dumps({"variant_marker": "v2_optimized_cfg"}), encoding="utf-8"
    )

    (tmp_path / "other_preset").mkdir()
    (tmp_path / "other_preset" / "template.json").write_text(
        json.dumps({"name": "other"}), encoding="utf-8"
    )

    yield tmp_path

    get_settings.cache_clear()


def _settings_with_variant(presets_dir: Path, variant: str) -> Settings:
    """Build a Settings instance with the chosen variant and tmp presets_dir."""
    return get_settings().model_copy(
        update={"presets_dir": presets_dir, "sheet_variant": variant}
    )


@pytest.mark.parametrize("variant", ["v1_legacy", "v2_optimized"])
def test_list_presets_hides_variant_backing_dir(
    presets_dir: Path, variant: str
) -> None:
    """``portrait_25q_v2`` must not appear as its own selectable preset."""
    s = _settings_with_variant(presets_dir, variant)
    listing = presets_service.list_presets(s)
    assert "portrait_25q" in listing, listing
    assert "portrait_25q_v2" not in listing, listing
    assert "other_preset" in listing, listing


def test_get_preset_documents_serves_v1_legacy_by_default(
    presets_dir: Path,
) -> None:
    s = _settings_with_variant(presets_dir, "v1_legacy")
    docs = presets_service.get_preset_documents("portrait_25q", s)
    assert docs["template"]["variant_marker"] == "v1_legacy"
    assert "config" not in docs


def test_get_preset_documents_serves_v2_when_setting_flipped(
    presets_dir: Path,
) -> None:
    s = _settings_with_variant(presets_dir, "v2_optimized")
    docs = presets_service.get_preset_documents("portrait_25q", s)
    assert docs["template"]["variant_marker"] == "v2_optimized"
    assert docs["config"]["variant_marker"] == "v2_optimized_cfg"


def test_non_routed_preset_is_unaffected_by_variant_setting(
    presets_dir: Path,
) -> None:
    """Only ``portrait_25q`` honours the variant setting."""
    for variant in ("v1_legacy", "v2_optimized"):
        s = _settings_with_variant(presets_dir, variant)
        docs = presets_service.get_preset_documents("other_preset", s)
        assert docs["template"] == {"name": "other"}


@pytest.mark.parametrize(
    "variant,expected_marker,expect_config",
    [
        ("v1_legacy", "v1_legacy", False),
        ("v2_optimized", "v2_optimized", True),
    ],
)
def test_apply_preset_copies_variant_files_into_batch(
    presets_dir: Path,
    tmp_path: Path,
    variant: str,
    expected_marker: str,
    expect_config: bool,
) -> None:
    """``apply_preset_to_batch("portrait_25q", ...)`` uses the active variant."""
    batch_root = tmp_path / f"batch_{variant}"
    batch_root.mkdir()
    s = _settings_with_variant(presets_dir, variant)
    presets_service.apply_preset_to_batch(batch_root, "portrait_25q", s)

    template_path = batch_root / "template.json"
    assert template_path.exists()
    body = json.loads(template_path.read_text(encoding="utf-8"))
    assert body["variant_marker"] == expected_marker

    config_path = batch_root / "config.json"
    assert config_path.exists() is expect_config


def test_apply_preset_raises_for_unknown_preset(
    presets_dir: Path, tmp_path: Path
) -> None:
    """Routing must not mask the "preset not found" error path."""
    batch_root = tmp_path / "batch_unknown"
    batch_root.mkdir()
    s = _settings_with_variant(presets_dir, "v1_legacy")
    with pytest.raises(ValueError, match="not found"):
        presets_service.apply_preset_to_batch(batch_root, "no_such_preset", s)
