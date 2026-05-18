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
        long_title = "x" * 300
        self.assertEqual(_manga_zip_filename(long_title, "99"), "99.zip")


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


if __name__ == "__main__":
    unittest.main()
