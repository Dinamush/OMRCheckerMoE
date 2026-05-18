"""Tests for challenge_detect heuristics."""

import unittest

from challenge_detect import detect_challenge


class TestChallengeDetect(unittest.TestCase):
    def test_cloudflare_iuam(self) -> None:
        html = "<html><title>Just a moment...</title><body>Checking your browser</body></html>"
        self.assertEqual(detect_challenge(html, "https://example.com/", "Just a moment..."), "cloudflare_iuam")

    def test_captcha(self) -> None:
        html = '<div class="g-recaptcha" data-sitekey="x"></div>'
        self.assertEqual(detect_challenge(html, "https://example.com/", "Verify"), "captcha")

    def test_login_path(self) -> None:
        self.assertEqual(
            detect_challenge("<html></html>", "https://xhamster.com/login", "Login"),
            "login",
        )

    def test_none_on_favorites(self) -> None:
        html = "<html><body><a href='/videos/foo'>vid</a> favorites</body></html>"
        self.assertEqual(
            detect_challenge(html, "https://xhamster.com/my/favorites/videos", "Favorites"),
            "none",
        )


if __name__ == "__main__":
    unittest.main()
