"""Tests for Pixiv image quality presets."""

import unittest

from pixiv_ph import pick_pixiv_image_url, pick_ugoira_zip_url

_URLS = {
    "original": "https://i.pximg.net/img-original/a.png",
    "regular": "https://i.pximg.net/img-master/a.png",
    "small": "https://i.pximg.net/c/48x48/a.png",
}


class TestPickPixivImageUrl(unittest.TestCase):
    def test_original_prefers_original(self) -> None:
        self.assertEqual(pick_pixiv_image_url(_URLS, "original"), _URLS["original"])

    def test_regular_skips_original_when_present(self) -> None:
        self.assertEqual(pick_pixiv_image_url(_URLS, "regular"), _URLS["regular"])

    def test_small_prefers_small(self) -> None:
        self.assertEqual(pick_pixiv_image_url(_URLS, "small"), _URLS["small"])

    def test_best_matches_original(self) -> None:
        self.assertEqual(pick_pixiv_image_url(_URLS, "best"), _URLS["original"])

    def test_regular_fallback_to_small_before_original(self) -> None:
        urls = {"original": _URLS["original"], "small": _URLS["small"]}
        self.assertEqual(pick_pixiv_image_url(urls, "regular"), _URLS["small"])

    def test_thumb_alias_for_small(self) -> None:
        urls = {"thumb": _URLS["small"]}
        self.assertEqual(pick_pixiv_image_url(urls, "small"), _URLS["small"])


class TestPickUgoiraZipUrl(unittest.TestCase):
    def test_original_quality(self) -> None:
        meta = {
            "src": "https://example.com/preview.zip",
            "originalSrc": "https://example.com/full.zip",
        }
        self.assertEqual(
            pick_ugoira_zip_url(meta, quality="original"), meta["originalSrc"]
        )

    def test_small_uses_preview(self) -> None:
        meta = {
            "src": "https://example.com/preview.zip",
            "originalSrc": "https://example.com/full.zip",
        }
        self.assertEqual(pick_ugoira_zip_url(meta, quality="small"), meta["src"])


if __name__ == "__main__":
    unittest.main()
