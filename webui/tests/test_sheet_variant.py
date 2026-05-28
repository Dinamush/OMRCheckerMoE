"""Tests for the ``sheet_variant`` runtime setting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest

from webui.services import presets as presets_service
from webui.settings import Settings, get_settings
from webui.sheet_registry import (
    PORTRAIT_SMQ25_1,
    PORTRAIT_SMQ25_LOGICAL,
    VARIANTS_SMQ25_0,
)


@pytest.fixture
def presets_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Build a tmp presets_dir with both portrait variants and one decoy."""
    monkeypatch.setenv("OMR_WEBUI_PRESETS_DIR", str(tmp_path))
    get_settings.cache_clear()

    (tmp_path / VARIANTS_SMQ25_0).mkdir()
    (tmp_path / VARIANTS_SMQ25_0 / "template.json").write_text(
        json.dumps({"variant_marker": "v1_legacy"}), encoding="utf-8"
    )
    (tmp_path / VARIANTS_SMQ25_0 / "generate_blank.py").write_text(
        "print('maintainer tooling')\n", encoding="utf-8"
    )

    (tmp_path / PORTRAIT_SMQ25_1).mkdir()
    (tmp_path / PORTRAIT_SMQ25_1 / "template.json").write_text(
        json.dumps({"variant_marker": "v2_optimized"}), encoding="utf-8"
    )
    (tmp_path / PORTRAIT_SMQ25_1 / "config.json").write_text(
        json.dumps({"variant_marker": "v2_optimized_cfg"}), encoding="utf-8"
    )

    (tmp_path / "other_preset").mkdir()
    (tmp_path / "other_preset" / "template.json").write_text(
        json.dumps({"name": "other"}), encoding="utf-8"
    )

    yield tmp_path

    get_settings.cache_clear()


def _settings_with_variant(presets_dir: Path, variant: str) -> Settings:
    return get_settings().model_copy(
        update={"presets_dir": presets_dir, "sheet_variant": variant}
    )


@pytest.mark.parametrize("variant", ["v1_legacy", "v2_optimized"])
def test_list_presets_hides_variant_backing_dirs(
    presets_dir: Path, variant: str
) -> None:
    s = _settings_with_variant(presets_dir, variant)
    listing = presets_service.list_presets(s)
    assert PORTRAIT_SMQ25_LOGICAL in listing, listing
    assert VARIANTS_SMQ25_0 not in listing, listing
    assert PORTRAIT_SMQ25_1 not in listing, listing
    assert "other_preset" in listing, listing


def test_get_preset_documents_serves_v1_legacy_by_default(
    presets_dir: Path,
) -> None:
    s = _settings_with_variant(presets_dir, "v1_legacy")
    docs = presets_service.get_preset_documents(PORTRAIT_SMQ25_LOGICAL, s)
    assert docs["template"]["variant_marker"] == "v1_legacy"
    assert "config" not in docs


def test_get_preset_documents_serves_v2_when_setting_flipped(
    presets_dir: Path,
) -> None:
    s = _settings_with_variant(presets_dir, "v2_optimized")
    docs = presets_service.get_preset_documents(PORTRAIT_SMQ25_LOGICAL, s)
    assert docs["template"]["variant_marker"] == "v2_optimized"
    assert docs["config"]["variant_marker"] == "v2_optimized_cfg"


def test_legacy_portrait_preset_alias_still_resolves(
    presets_dir: Path,
) -> None:
    s = _settings_with_variant(presets_dir, "v2_optimized")
    docs = presets_service.get_preset_documents("portrait_25q", s)
    assert docs["template"]["variant_marker"] == "v2_optimized"


def test_non_routed_preset_is_unaffected_by_variant_setting(
    presets_dir: Path,
) -> None:
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
    batch_root = tmp_path / f"batch_{variant}"
    batch_root.mkdir()
    s = _settings_with_variant(presets_dir, variant)
    presets_service.apply_preset_to_batch(batch_root, PORTRAIT_SMQ25_LOGICAL, s)

    template_path = batch_root / "template.json"
    assert template_path.exists()
    body = json.loads(template_path.read_text(encoding="utf-8"))
    assert body["variant_marker"] == expected_marker
    assert not (batch_root / "generate_blank.py").exists()

    config_path = batch_root / "config.json"
    assert config_path.exists() is expect_config


def test_apply_preset_raises_for_unknown_preset(
    presets_dir: Path, tmp_path: Path
) -> None:
    batch_root = tmp_path / "batch_unknown"
    batch_root.mkdir()
    s = _settings_with_variant(presets_dir, "v1_legacy")
    with pytest.raises(ValueError, match="not found"):
        presets_service.apply_preset_to_batch(batch_root, "no_such_preset", s)
