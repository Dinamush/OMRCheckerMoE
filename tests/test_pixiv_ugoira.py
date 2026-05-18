"""Tests for Pixiv ugoira detection and metadata parsing."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pixiv_ph import (
    _write_ugoira_ffconcat,
    is_ugoira_body,
    pick_ugoira_zip_url,
)


class TestUgoiraDetection(unittest.TestCase):
    def test_illust_type_2(self) -> None:
        self.assertTrue(is_ugoira_body({"illustType": 2}))

    def test_static_illust(self) -> None:
        self.assertFalse(is_ugoira_body({"illustType": 0}))
        self.assertFalse(is_ugoira_body({}))


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
        session.get.assert_called_once()
        self.assertIn("ugoira_meta", session.get.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
