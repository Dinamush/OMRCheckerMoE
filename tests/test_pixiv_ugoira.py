"""Tests for Pixiv ugoira detection and metadata parsing."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pixiv_ph import (
    _write_ugoira_ffconcat,
    is_ugoira_body,
    parse_illust_id,
    pick_ugoira_zip_url,
    resolve_pixiv_target,
)


class TestUgoiraDetection(unittest.TestCase):
    def test_illust_type_2(self) -> None:
        self.assertTrue(is_ugoira_body({"illustType": 2}))

    def test_static_illust(self) -> None:
        self.assertFalse(is_ugoira_body({"illustType": 0}))
        self.assertFalse(is_ugoira_body({}))


class TestParseIllustId(unittest.TestCase):
    def test_artworks_url(self) -> None:
        self.assertEqual(
            parse_illust_id("https://www.pixiv.net/en/artworks/127692715"),
            "127692715",
        )

    def test_user_url_not_illust(self) -> None:
        self.assertIsNone(
            parse_illust_id("https://www.pixiv.net/en/users/33191689/bookmarks/artworks")
        )


class TestResolvePixivTarget(unittest.TestCase):
    def test_artwork_url_auto(self) -> None:
        mode, uid, iid = resolve_pixiv_target(
            "https://www.pixiv.net/en/artworks/127692715", single_artwork=False
        )
        self.assertEqual(mode, "artwork")
        self.assertIsNone(uid)
        self.assertEqual(iid, "127692715")

    def test_numeric_with_checkbox(self) -> None:
        mode, uid, iid = resolve_pixiv_target("127692715", single_artwork=True)
        self.assertEqual(mode, "artwork")
        self.assertEqual(iid, "127692715")

    def test_numeric_without_checkbox_is_user(self) -> None:
        mode, uid, iid = resolve_pixiv_target("33191689", single_artwork=False)
        self.assertEqual(mode, "bookmarks")
        self.assertEqual(uid, "33191689")
        self.assertIsNone(iid)


class TestUgoiraMetaPick(unittest.TestCase):
    def test_prefers_original_src(self) -> None:
        meta = {
            "src": "https://example.com/sample.zip",
            "originalSrc": "https://example.com/original.zip",
        }
        self.assertEqual(pick_ugoira_zip_url(meta), meta["originalSrc"])

    def test_falls_back_to_src(self) -> None:
        meta = {"src": "https://example.com/sample.zip"}
        self.assertEqual(pick_ugoira_zip_url(meta), meta["src"])

    def test_missing_urls(self) -> None:
        self.assertIsNone(pick_ugoira_zip_url({}))


class TestUgoiraFfconcat(unittest.TestCase):
    def test_writes_frame_durations(self) -> None:
        frames = [
            {"file": "000.jpg", "delay": 80},
            {"file": "001.jpg", "delay": 120},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ffconcat.txt"
            _write_ugoira_ffconcat(frames, out)
            text = out.read_text(encoding="utf-8")
        self.assertIn("ffconcat version 1.0", text)
        self.assertIn("file 000.jpg", text)
        self.assertIn("duration 0.08", text)
        self.assertIn("file 001.jpg", text)


class TestFetchUgoiraMeta(unittest.TestCase):
    @patch("pixiv_ph.challenge_detect.detect_challenge", return_value="none")
    def test_parses_body(self, _detect: MagicMock) -> None:
        from pixiv_ph import fetch_ugoira_meta

        session = MagicMock()
        resp = MagicMock()
        resp.text = ""
        resp.url = "https://www.pixiv.net/ajax/illust/1/ugoira_meta"
        resp.json.return_value = {
            "error": False,
            "body": {
                "originalSrc": "https://i.pximg.net/ugoira.zip",
                "frames": [{"file": "0.jpg", "delay": 100}],
            },
        }
        session.headers = {}
        session.get.return_value = resp

        body = fetch_ugoira_meta(session, "1")
        self.assertEqual(body["originalSrc"], "https://i.pximg.net/ugoira.zip")
        self.assertIn("ugoira_meta", session.get.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
