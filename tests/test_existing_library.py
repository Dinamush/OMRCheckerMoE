"""Tests for library-folder name matching."""

from pathlib import Path

import pytest

from existing_library import (
    build_library_normalized_names,
    matches_existing_library,
    resolve_optional_library_directory,
)


def test_matches_exact_normalized(tmp_path: Path) -> None:
    (tmp_path / "My_Cool_Video.mp4").write_bytes(b"x")
    norms = build_library_normalized_names(tmp_path, recursive=False)
    assert matches_existing_library("My Cool Video", norms)


def test_resolve_empty_returns_none(tmp_path: Path) -> None:
    assert resolve_optional_library_directory("", tmp_path) is None
    assert resolve_optional_library_directory("   ", tmp_path) is None


def test_resolve_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        resolve_optional_library_directory("/nonexistent/path/zzz", tmp_path)
