"""Tests for library-folder name matching."""

import unittest
from pathlib import Path

from existing_library import (
    build_library_index,
    build_library_normalized_names,
    matches_existing_library,
    matches_pornhub_in_library,
    resolve_optional_library_directory,
    viewkey_from_pornhub_url,
)


class TestExistingLibrary(unittest.TestCase):
    def test_matches_exact_normalized(self) -> None:
        tmp = self._tmpdir()
        (tmp / "My_Cool_Video.mp4").write_bytes(b"x")
        norms = build_library_normalized_names(tmp, recursive=False)
        self.assertTrue(matches_existing_library("My Cool Video", norms))

    def test_resolve_empty_returns_none(self) -> None:
        tmp = self._tmpdir()
        self.assertIsNone(resolve_optional_library_directory("", tmp))
        self.assertIsNone(resolve_optional_library_directory("   ", tmp))

    def test_resolve_missing_raises(self) -> None:
        tmp = self._tmpdir()
        with self.assertRaises(ValueError):
            resolve_optional_library_directory("/nonexistent/path/zzz", tmp)

    def test_pornhub_viewkey_in_ph_subfolder(self) -> None:
        tmp = self._tmpdir()
        ph = tmp / "ph"
        ph.mkdir()
        (ph / "Cool Clip_ph5fe50e391efeb.mp4").write_bytes(b"x")
        idx = build_library_index(tmp, recursive=False)
        url = "https://www.pornhub.com/view_video.php?viewkey=ph5fe50e391efeb"
        self.assertEqual(viewkey_from_pornhub_url(url), "ph5fe50e391efeb")
        self.assertTrue(matches_pornhub_in_library(url, "Cool Clip", idx))

    def test_pornhub_title_key_from_filename(self) -> None:
        tmp = self._tmpdir()
        (tmp / "My Cool Video_ph5fe50e391efeb.mp4").write_bytes(b"x")
        idx = build_library_index(tmp, recursive=True)
        self.assertTrue(
            matches_pornhub_in_library(
                "https://www.pornhub.com/view_video.php?viewkey=ph5fe50e391efeb",
                "My Cool Video",
                idx,
            )
        )

    def _tmpdir(self) -> Path:
        import tempfile

        return Path(tempfile.mkdtemp())


if __name__ == "__main__":
    unittest.main()
