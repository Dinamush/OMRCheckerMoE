"""PornHub favourites URL resolution (ph.py /users/<name>/videos/favorites)."""
import unittest
from pathlib import Path

from pornhub_ph import (
    _username_from_profile_html,
    favorites_url_for_username,
)

FIXTURE = Path(__file__).resolve().parents[1] / "website-code" / "ph-favourites-page.html"


class TestPornhubFavoritesUrl(unittest.TestCase):
    def test_favorites_url_for_username(self):
        url = favorites_url_for_username("burninglove999")
        self.assertEqual(
            url,
            "https://www.pornhub.com/users/burninglove999/videos/favorites",
        )

    def test_username_from_favourites_fixture(self):
        html = FIXTURE.read_text(encoding="utf-8", errors="replace")
        self.assertEqual(_username_from_profile_html(html), "burninglove999")


if __name__ == "__main__":
    unittest.main()
