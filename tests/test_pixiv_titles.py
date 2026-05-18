"""Tests for Pixiv filename title translation helpers."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pixiv_titles import (
    CACHE_FILENAME,
    _needs_translation,
    resolve_pixiv_filename_title,
    sanitize_filename_slug,
)
from settings import AppSettings


class TestSanitizeFilenameSlug(unittest.TestCase):
    def test_forbidden_chars(self) -> None:
        self.assertEqual(sanitize_filename_slug('a<b>c:d"e'), "a_b_c_d_e")

    def test_whitespace(self) -> None:
        self.assertEqual(sanitize_filename_slug("hello world"), "hello_world")

    def test_max_len(self) -> None:
        long = "a" * 100
        self.assertEqual(len(sanitize_filename_slug(long, max_len=80)), 80)


class TestNeedsTranslation(unittest.TestCase):
    def test_ascii_skip(self) -> None:
        self.assertFalse(_needs_translation("R-18 sketch"))

    def test_japanese(self) -> None:
        self.assertTrue(_needs_translation("オリジナル"))


class TestResolvePixivFilenameTitle(unittest.TestCase):
    def test_disabled_returns_raw(self) -> None:
        cfg = AppSettings(pixiv_translate_titles=False)
        out = resolve_pixiv_filename_title("オリジナル", "123", settings=cfg)
        self.assertEqual(out, "オリジナル")

    def test_ascii_unchanged_without_api(self) -> None:
        cfg = AppSettings(pixiv_translate_titles=True)
        out = resolve_pixiv_filename_title("Original Art", "456", settings=cfg)
        self.assertEqual(out, "Original Art")

    def test_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / CACHE_FILENAME
            cache_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "entries": {
                            "999": {
                                "source": "テスト",
                                "en": "Test title",
                                "updated": 1,
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            cfg = AppSettings(pixiv_translate_titles=True)
            with patch("pixiv_titles._translate_remote") as mock_tr:
                out = resolve_pixiv_filename_title(
                    "テスト", "999", settings=cfg, user_data_dir=root
                )
                self.assertEqual(out, "Test title")
                mock_tr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
