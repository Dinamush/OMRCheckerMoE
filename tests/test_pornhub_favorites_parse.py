"""Tests for PornHub favourites link parsing."""

import unittest
from pathlib import Path

from pornhub_ph import parse_video_links_pornhub

_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "website-code"
    / "ph-favourites-page.html"
)


class TestPornhubFavoritesParse(unittest.TestCase):
    @unittest.skipUnless(_FIXTURE.is_file(), "website-code/ph-favourites-page.html missing")
    def test_parse_favourites_fixture(self) -> None:
        html = _FIXTURE.read_text(encoding="utf-8", errors="replace")
        links = parse_video_links_pornhub(html)
        self.assertGreater(len(links), 0)
        self.assertTrue(all("viewkey=" in u for u in links))
        self.assertIn(
            "https://www.pornhub.com/view_video.php?viewkey=66d854c760254",
            links,
        )


if __name__ == "__main__":
    unittest.main()
