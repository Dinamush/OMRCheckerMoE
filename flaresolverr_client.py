"""Optional FlareSolverr client for Cloudflare IUAM (HDoujin-style plugin)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


def solve_url(
    target_url: str,
    *,
    base_url: str = "http://127.0.0.1:8191/v1",
    timeout_sec: float = 120.0,
) -> Tuple[Dict[str, str], str]:
    """
    POST FlareSolverr ``request.get``; return ``(cookie_name -> value, user_agent)``.

    Raises on HTTP/API errors or non-success status.
    """
    api = base_url.rstrip("/")
    payload = {
        "cmd": "request.get",
        "url": target_url,
        "maxTimeout": int(timeout_sec * 1000),
    }
    r = requests.post(api, json=payload, timeout=timeout_sec + 15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "ok":
        raise RuntimeError(data.get("message") or "FlareSolverr returned non-ok status")
    solution = data.get("solution") or {}
    cookies_list: List[Dict[str, Any]] = solution.get("cookies") or []
    cookies: Dict[str, str] = {}
    for c in cookies_list:
        name = c.get("name")
        value = c.get("value")
        if name and value is not None:
            cookies[str(name)] = str(value)
    ua = str(solution.get("userAgent") or "")
    logger.info("FlareSolverr solved %s (%s cookies)", target_url, len(cookies))
    return cookies, ua


def inject_cookies_into_driver(driver, cookies: Dict[str, str], landing_url: str) -> None:
    """Navigate to origin then add cookies (best-effort per domain)."""
    from urllib.parse import urlparse

    driver.get(landing_url)
    origin = urlparse(landing_url)
    domain = origin.hostname or ""
    for name, value in cookies.items():
        spec: dict = {"name": name, "value": value, "path": "/"}
        if domain:
            spec["domain"] = domain if domain.startswith(".") else f".{domain}"
        try:
            driver.add_cookie(spec)
        except Exception as e:
            logger.debug("FlareSolverr cookie skip %s: %s", name, e)
