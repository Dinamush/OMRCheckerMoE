"""Tests for PornHub favorites collection algorithm."""

import unittest
from unittest.mock import MagicMock, patch

from pornhub_ph import (
    collect_favorites_urls_requests,
    collect_favorites_urls_with_driver,
    parse_video_links_pornhub,
)

_MOD = "pornhub_ph"
_FAV_URL = "https://www.pornhub.com/users/testuser/videos/favorites"


def _html(*viewkeys):
    links = "".join(
        f'<a href="/view_video.php?viewkey={vk}">v</a>' for vk in viewkeys
    )
    return f"<html><body>{links}</body></html>"


def _url(vk):
    return f"https://www.pornhub.com/view_video.php?viewkey={vk}"


# ---------------------------------------------------------------------------
# parse_video_links_pornhub
# ---------------------------------------------------------------------------

class TestParseVideoLinks(unittest.TestCase):

    def test_basic_extraction(self):
        links = parse_video_links_pornhub(_html("abc123", "def456"))
        self.assertEqual(len(links), 2)
        self.assertTrue(all("viewkey=" in u for u in links))

    def test_deduplication(self):
        links = parse_video_links_pornhub(_html("dup", "dup", "other"))
        self.assertEqual(len(links), 2)

    def test_empty_page(self):
        self.assertEqual(parse_video_links_pornhub("<html><body>no videos</body></html>"), [])


# ---------------------------------------------------------------------------
# collect_favorites_urls_with_driver
# ---------------------------------------------------------------------------

class TestCollectWithDriver(unittest.TestCase):

    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.scroll_to_bottom_pornhub")
    @patch(f"{_MOD}.try_click_load_more")
    def test_single_page_no_load_more(self, mock_click, mock_scroll, mock_sleep):
        driver = MagicMock()
        driver.page_source = _html("vid1")
        mock_click.return_value = False

        result = collect_favorites_urls_with_driver(
            driver, _FAV_URL, already_on_favorites=True
        )

        self.assertEqual(result, [_url("vid1")])
        mock_scroll.assert_called_once()

    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.scroll_to_bottom_pornhub")
    @patch(f"{_MOD}.try_click_load_more")
    @patch(f"{_MOD}.parse_video_links_pornhub")
    def test_collects_all_pages_when_button_appears_late(
        self, mock_parse, mock_click, mock_scroll, mock_sleep
    ):
        """
        Regression test for BUG 1 fix: the #moreDataBtn may not be rendered on
        the first scroll because the 'moreData' section is lazy-loaded by JS.
        The fix retries try_click_load_more() once after a 2.5 s wait before
        concluding there are no more pages.

        Sequence with the fix applied:
          Iteration 1 → parse p1v1; click check #1 = False (not rendered yet);
                        retry click check #2 = True (rendered after wait)
          Iteration 2 → parse p1v1+p2v1; click check #3 = True
          Iteration 3 → parse p1v1+p2v1+p3v1; click check #4 = False → done
        Expected result: 3 unique videos.
        """
        mock_parse.side_effect = [
            [_url("p1v1")],
            [_url("p1v1"), _url("p2v1")],
            [_url("p1v1"), _url("p2v1"), _url("p3v1")],
        ]
        # 4 calls: initial check (False), retry (True), after pg2 (True), after pg3 (False)
        mock_click.side_effect = [False, True, True, False]

        driver = MagicMock()
        result = collect_favorites_urls_with_driver(
            driver, _FAV_URL, already_on_favorites=True
        )

        self.assertEqual(
            len(result),
            3,
            "All 3 pages must be collected even when button is absent on first check.",
        )


# ---------------------------------------------------------------------------
# collect_favorites_urls_requests
# ---------------------------------------------------------------------------

class TestCollectWithRequests(unittest.TestCase):

    def _resp(self, html, url=""):
        r = MagicMock()
        r.text = html
        r.url = url
        r.raise_for_status = MagicMock()
        return r

    @patch(f"{_MOD}._session_with_cookies")
    def test_pagination_terminates_when_no_new_links(self, mock_factory):
        session = MagicMock()
        mock_factory.return_value = session
        session.get.side_effect = [
            self._resp(_html("vid001", "vid002")),  # page 1 — 2 new URLs
            self._resp(_html("vid001", "vid002")),  # page 2 — 0 new → stop
        ]

        result = collect_favorites_urls_requests("cookies.txt", _FAV_URL)

        self.assertEqual(len(result), 2)
        self.assertEqual(session.get.call_count, 2)

    @patch(f"{_MOD}._session_with_cookies")
    def test_partial_failure_returns_partial_list(self, mock_factory):
        session = MagicMock()
        mock_factory.return_value = session
        session.get.side_effect = [
            self._resp(_html("vid101", "vid102")),  # page 1 — ok
            self._resp(_html("vid103")),            # page 2 — ok
            Exception("connection reset"),          # page 3 — network error → stop
        ]

        result = collect_favorites_urls_requests("cookies.txt", _FAV_URL)

        self.assertEqual(len(result), 3)
        self.assertTrue(all("viewkey=" in u for u in result))

    @patch(f"{_MOD}._session_with_cookies")
    def test_login_redirect_stops_collection(self, mock_factory):
        """BUG 3 fix: PornHub returns HTTP 200 on login redirect — detect it."""
        session = MagicMock()
        mock_factory.return_value = session
        login_html = '<form><input name="username" /></form>'
        session.get.side_effect = [
            self._resp(_html("vid201", "vid202")),   # page 1 — ok
            self._resp(login_html, url="https://www.pornhub.com/login"),  # page 2 — redirect
        ]

        result = collect_favorites_urls_requests("cookies.txt", _FAV_URL)

        # Should return only the videos collected before the redirect was detected.
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
