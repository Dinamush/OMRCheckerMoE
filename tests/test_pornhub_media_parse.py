"""Tests for PornHub flashvars / mediaDefinitions parsing (ph.py algorithm)."""

import unittest
from pathlib import Path

from pornhub_ph import extract_media_from_page

_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "website-code"
    / "ph-video.html"
)


class TestPornhubMediaParse(unittest.TestCase):
    @unittest.skipUnless(_FIXTURE.is_file(), "website-code/ph-video.html missing")
    def test_extract_media_from_fixture(self) -> None:
        html = _FIXTURE.read_text(encoding="utf-8", errors="replace")
        url = "https://www.pornhub.com/view_video.php?viewkey=ph5fe50e391efeb"
        result = extract_media_from_page(html, url)
        self.assertIsNotNone(result, "expected flashvars mediaDefinitions in fixture")
        media_url, fmt, title, viewkey = result  # type: ignore[misc]
        self.assertEqual(viewkey, "ph5fe50e391efeb")
        self.assertTrue(title)
        self.assertTrue(media_url.startswith("http"))
        self.assertIn(fmt, ("mp4", "hls"))


if __name__ == "__main__":
    unittest.main()
