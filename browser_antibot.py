"""
Chrome WebDriver factory: profiles, headless=new, optional undetected-chromedriver (scrape only).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

from paths import get_user_data_dir
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)

BROWSER_ANTIBOT_VERSION = 1

DriverPurpose = Literal["auth", "scrape"]

_UNDETECTED_AVAILABLE = False
try:
    import undetected_chromedriver as uc  # type: ignore

    _UNDETECTED_AVAILABLE = True
except ImportError:
    uc = None  # type: ignore


def site_profile_dir(site: str, enabled: bool = True) -> Optional[Path]:
    if not enabled:
        return None
    d = get_user_data_dir() / "browser_profiles" / site
    d.mkdir(parents=True, exist_ok=True)
    return d


def undetected_chromedriver_available() -> bool:
    return _UNDETECTED_AVAILABLE


def _apply_common_options(
    options: Options,
    *,
    headless: bool,
    proxy_url: str,
    profile_dir: Optional[Path],
) -> None:
    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
    pv = (proxy_url or "").strip()
    if pv:
        options.add_argument("--proxy-server=" + pv)
    if profile_dir is not None:
        options.add_argument(f"--user-data-dir={profile_dir}")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--log-level=3")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)


def _finalize_driver(driver: webdriver.Chrome) -> webdriver.Chrome:
    driver.set_page_load_timeout(30)
    driver.implicitly_wait(0)
    try:
        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
    except Exception:
        pass
    return driver


def create_chrome_driver(
    *,
    headless: bool,
    proxy_url: str = "",
    profile_dir: Optional[Path] = None,
    purpose: DriverPurpose = "scrape",
    use_undetected: bool = True,
) -> webdriver.Chrome:
    """
    Create Chrome for SHUCK3R automation.

    - ``purpose='auth'`` always runs headed (never headless).
    - ``purpose='scrape'`` may use headless=new and optional undetected-chromedriver.
    """
    if purpose == "auth":
        headless = False

    want_uc = (
        purpose == "scrape"
        and use_undetected
        and _UNDETECTED_AVAILABLE
        and uc is not None
    )

    if want_uc:
        try:
            options = uc.ChromeOptions()
            _apply_common_options(
                options,
                headless=headless,
                proxy_url=proxy_url,
                profile_dir=profile_dir,
            )
            driver = uc.Chrome(options=options, use_subprocess=True)
            logger.info(
                "Chrome (undetected) initialized purpose=%s headless=%s profile=%s",
                purpose,
                headless,
                profile_dir,
            )
            return _finalize_driver(driver)
        except Exception as e:
            logger.warning("undetected-chromedriver failed (%s); falling back to stock Selenium", e)

    options = Options()
    _apply_common_options(
        options,
        headless=headless,
        proxy_url=proxy_url,
        profile_dir=profile_dir,
    )
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        logger.info(
            "Chrome WebDriver initialized purpose=%s headless=%s profile=%s",
            purpose,
            headless,
            profile_dir,
        )
        return _finalize_driver(driver)
    except WebDriverException as e:
        logger.error("Chrome WebDriver initialization failed: %s", e)
        raise
