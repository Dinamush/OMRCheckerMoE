"""Heuristic detection of login walls, Cloudflare IUAM, and CAPTCHA interstitials."""

from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlparse

ChallengeKind = Literal["none", "login", "cloudflare_iuam", "captcha"]

_CF_MARKERS = (
    "just a moment",
    "checking your browser",
    "checking if the site connection is secure",
    "cf-browser-verification",
    "cf-challenge",
    "challenge-platform",
    "turnstile",
)
_CAPTCHA_MARKERS = (
    "g-recaptcha",
    "hcaptcha",
    "cf-turnstile",
    "captcha",
    "are you human",
)
_LOGIN_PATH = ("/login", "/signin", "/sign-in", "/auth")


def detect_challenge(html: str, url: str, title: str = "") -> ChallengeKind:
    """Classify the current page; ``none`` when content looks like a normal site page."""
    low_html = (html or "")[:120_000].lower()
    low_title = (title or "").lower()
    path = (urlparse(url or "").path or "").lower()

    if any(m in low_html or m in low_title for m in _CAPTCHA_MARKERS):
        if "captcha" in low_title or "g-recaptcha" in low_html or "hcaptcha" in low_html:
            return "captcha"

    if any(m in low_html or m in low_title for m in _CF_MARKERS):
        return "cloudflare_iuam"

    if "accounts.pixiv.net" in (urlparse(url or "").netloc or "").lower():
        return "login"

    if any(seg in path for seg in _LOGIN_PATH):
        if "favorites" not in path and "my/" not in path and "/bookmarks/" not in path:
            return "login"

    if re.search(r"\b(log\s*in|sign\s*in)\b", low_title) and len(low_html) < 80_000:
        if "favorites" not in url.lower():
            return "login"

    return "none"
