"""Tests for Pixiv manga ZIP bundling and filename fallbacks."""

import unittest

from pixiv_ph import (
    PIXIV_ILLUST_TYPE_MANGA,
    _cdn_page_stem,
    _manga_zip_filename,
    _safe_filename,
    is_manga_body,
)


class TestMangaDetection(unittest.TestCase):
    def test_manga_type(self) -> None:
        self.assertTrue(is_manga_body({"illustType": PIXIV_ILLUST_TYPE_MANGA}))


class TestMangaZipName(unittest.TestCase):
    def test_includes_title(self) -> None:
        name = _manga_zip_filename("My Manga Title", "12345")
        self.assertTrue(name.startswith("12345_"))
        self.assertTrue(name.endswith(".zip"))

    def test_long_title_falls_back_to_id(self) -> None:
        long_id = "9" * 190
        self.assertEqual(_manga_zip_filename("My Title", long_id), f"{long_id}.zip")


class TestSafeFilenameFallback(unittest.TestCase):
    def test_uses_post_title(self) -> None:
        name = _safe_filename("Hello World", "1", 0, ".jpg")
        self.assertIn("Hello_World", name)

    def test_cdn_fallback_when_too_long(self) -> None:
        long_title = "あ" * 80
        url = "https://i.pximg.net/img-original/img/2020/01/01/81214321_p0.png"
        name = _safe_filename(long_title, "81214321", 0, ".png", image_url=url, max_total_len=50)
        self.assertIn("81214321_p0", name)

    def test_cdn_stem(self) -> None:
        url = "https://i.pximg.net/img-original/img/2020/01/01/999_p2.jpg"
        self.assertEqual(_cdn_page_stem(url, 2), "999_p2")


class TestWindowsFilenameEdgeCases(unittest.TestCase):
    """Windows-oriented edge cases for Pixiv on-disk names."""

    def test_empty_title_uses_cdn_stem(self) -> None:
        url = "https://i.pximg.net/img-original/img/2020/01/01/81214321_p0.png"
        self.assertEqual(
            _safe_filename("", "81214321", 0, ".jpg", image_url=url),
            "81214321_p0_81214321_p0.jpg",
        )

    def test_whitespace_only_title_uses_page_fallback(self) -> None:
        self.assertEqual(_safe_filename("   ", "1", 0, ".jpg"), "1_p0_p000.jpg")

    def test_emoji_only_title_preserved(self) -> None:
        name = _safe_filename("\U0001f3a8\U00002728", "12345", 0, ".jpg")
        self.assertEqual(name, "12345_p0_\U0001f3a8\U00002728.jpg")

    def test_reserved_word_with_id_prefix(self) -> None:
        self.assertEqual(_safe_filename("CON", "12345", 0, ".jpg"), "12345_p0_CON.jpg")
        self.assertEqual(_manga_zip_filename("CON", "12345"), "12345_CON.zip")

    def test_trailing_dots_stripped_from_slug(self) -> None:
        self.assertEqual(_safe_filename("Hello...", "1", 0, ".jpg"), "1_p0_Hello.jpg")

    def test_component_length_capped_at_default_max(self) -> None:
        name = _safe_filename("A" * 500, "12345", 0, ".jpg")
        self.assertLessEqual(len(name), 200)

    def test_cdn_stem_without_url(self) -> None:
        self.assertEqual(_cdn_page_stem(None, 5), "p005")

    def test_mixed_jp_en_translated_style_title(self) -> None:
        title = (
            "Original Character Design - School Uniform Version (Commission)"
        )
        name = _safe_filename(title, "98765432", 0, ".jpg")
        self.assertIn("Original_Character_Design", name)
        self.assertLessEqual(len(name), 200)

    def test_manga_long_raw_uses_slug_when_fits(self) -> None:
        raw = "word " * 50
        name = _manga_zip_filename(raw, "12345")
        self.assertTrue(name.startswith("12345_"))
        self.assertTrue(name.endswith(".zip"))
        self.assertNotEqual(name, "12345.zip")

    def test_manga_empty_title_is_id_only_zip(self) -> None:
        self.assertEqual(_manga_zip_filename("", "12345"), "12345.zip")


if __name__ == "__main__":
    unittest.main()
