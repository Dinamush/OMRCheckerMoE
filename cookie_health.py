"""Netscape cookie jar health checks per site."""

from __future__ import annotations

import os
from http.cookiejar import MozillaCookieJar
from typing import Tuple

import requests

import pornhub_ph


def check_pornhub_cookies(cookie_file: str, proxy_url: str = "") -> Tuple[bool, str]:
    return pornhub_ph.check_cookie_health(cookie_file, proxy_url)


def check_xhamster_cookies(cookie_file: str, proxy_url: str = "") -> Tuple[bool, str]:
    if not os.path.isfile(cookie_file):
        return False, "Cookie file not found."
    session = requests.Session()
    pv = (proxy_url or "").strip()
    if pv:
        session.proxies.update({"http": pv, "https": pv})
    jar = MozillaCookieJar(cookie_file)
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
    except Exception as e:
        return False, f"Could not read cookie file: {e}"
    session.cookies = jar
    try:
        r = session.get(
            "https://xhamster.com/my/favorites/videos",
            timeout=20,
            allow_redirects=True,
        )
        final = (r.url or "").lower()
        if "login" in final or "signin" in final:
            return False, "Session expired — redirected to login."
        if r.status_code >= 400:
            return False, f"Favorites page returned HTTP {r.status_code}."
        html = (r.text or "")[:80_000].lower()
        if "xh-video" in html or "/videos/" in html or "favorites" in html:
            return True, "Cookies valid — xHamster favorites reachable."
        return False, "Favorites page did not look authenticated."
    except Exception as e:
        return False, f"Cookie health check failed: {e}"


def check_site_cookies(site: str, cookie_file: str, proxy_url: str = "") -> Tuple[bool, str]:
    if site == "pornhub":
        return check_pornhub_cookies(cookie_file, proxy_url)
    if site == "xhamster":
        return check_xhamster_cookies(cookie_file, proxy_url)
    return False, f"Unknown site: {site}"
