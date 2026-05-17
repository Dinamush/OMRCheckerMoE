"""Tests for Pixiv URL/id parsing."""

import unittest

from pixiv_ph import parse_user_id


class TestPixivParse(unittest.TestCase):
    def test_digits(self) -> None:
        self.assertEqual(parse_user_id("33191689"), "33191689")

    def test_profile_url(self) -> None:
        self.assertEqual(
            parse_user_id(
                "https://www.pixiv.net/en/users/33191689/bookmarks/artworks?rest=hide&mode=all"
            ),
            "33191689",
        )

    def test_empty(self) -> None:
        self.assertIsNone(parse_user_id(""))
        self.assertIsNone(parse_user_id("   "))


if __name__ == "__main__":
    unittest.main()
