"""
Shared browser lifecycle for download workflows: cookie reuse, auth vs scrape drivers, challenges.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

from selenium import webdriver

import browser_antibot
import browser_challenge_wait
import challenge_detect
import cookie_health
import flaresolverr_client
from settings import AppSettings

logger = logging.getLogger(__name__)


def scrape_headless_enabled(form_headless: bool, cfg: AppSettings) -> bool:
    return bool(form_headless or cfg.headless_scraping)


def site_profile_dir(site: str, cfg: AppSettings):
    return browser_antibot.site_profile_dir(site, cfg.browser_profile_per_site)


def cookies_valid_for_site(site: str, cookie_file: str, cfg: AppSettings) -> Tuple[bool, str]:
    if not cfg.skip_login_if_cookies_valid:
        return False, "Cookie reuse disabled in settings."
    if not os.path.isfile(cookie_file):
        return False, "No saved cookie file yet."
    return cookie_health.check_site_cookies(site, cookie_file, cfg.proxy_url)


def create_driver(
    site: str,
    cfg: AppSettings,
    *,
    headless: bool,
    purpose: browser_antibot.DriverPurpose,
) -> webdriver.Chrome:
    profile = site_profile_dir(site, cfg)
    return browser_antibot.create_chrome_driver(
        headless=headless,
        proxy_url=cfg.proxy_url,
        profile_dir=profile,
        purpose=purpose,
        use_undetected=cfg.use_undetected_chrome,
    )


def apply_flaresolverr_preflight(
    driver: webdriver.Chrome,
    start_url: str,
    cfg: AppSettings,
) -> None:
    if cfg.challenge_solver != "flaresolverr":
        return
    try:
        cookies, ua = flaresolverr_client.solve_url(
            start_url,
            base_url=cfg.flaresolverr_base_url,
        )
        flaresolverr_client.inject_cookies_into_driver(driver, cookies, start_url)
        if ua:
            driver.execute_cdp_cmd("Network.setUserAgentOverride", {"userAgent": ua})
    except Exception as e:
        logger.warning("FlareSolverr preflight failed (%s); continuing manually", e)


def quit_driver(driver: Optional[webdriver.Chrome]) -> None:
    if driver is None:
        return
    try:
        driver.quit()
        logging.info("Chrome WebDriver closed successfully.")
    except Exception as e:
        logging.error("Error closing Chrome WebDriver: %s", e)


def escalate_visible_for_challenge(
    session_id: str,
    driver: webdriver.Chrome,
    site: str,
    cfg: AppSettings,
    url: str,
) -> webdriver.Chrome:
    """Replace headless (or blocked) driver with visible Chrome; wait for progress-page confirm."""
    quit_driver(driver)
    import workflow_heuristics

    workflow_heuristics.advance(session_id, "challenge_wait")
    workflow_heuristics.set_detail(session_id, browser_challenge_wait.challenge_progress_hint())
    visible = create_driver(site, cfg, headless=False, purpose="auth")
    visible.get(url)
    browser_challenge_wait.wait_for_challenge_cleared(session_id)
    resume_step = "extract_list" if site == "pornhub" else "collect_urls"
    workflow_heuristics.advance(session_id, resume_step)
    workflow_heuristics.set_detail(session_id, None)
    return visible


def check_page_challenge(
    driver: webdriver.Chrome,
    url: str,
    *,
    session_id: Optional[str] = None,
    site: str = "xhamster",
    cfg: Optional[AppSettings] = None,
) -> Optional[webdriver.Chrome]:
    """
    If the current page looks like a challenge/login wall, escalate to visible browser when configured.

    Returns replacement driver when escalated, else ``None``.
    """
    if session_id is None or cfg is None:
        return None
    try:
        html = driver.page_source or ""
        title = driver.title or ""
    except Exception:
        return None
    kind = challenge_detect.detect_challenge(html, driver.current_url or url, title)
    if kind == "none":
        return None
    logging.warning("Challenge detected (%s) at %s — escalating to visible browser", kind, url)
    return escalate_visible_for_challenge(session_id, driver, site, cfg, url)
