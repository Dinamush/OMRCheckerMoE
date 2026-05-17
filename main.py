#!/usr/bin/env python3
"""
SHUCK3R — FastAPI application
Download favorites from PornHub, xHamster, and Pixiv bookmarks via a web UI.
"""

from __future__ import annotations

import os
import logging
import mimetypes
import time
import concurrent.futures
import json
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from fastapi import FastAPI, Request, BackgroundTasks, Form, HTTPException, Query
from fastapi.responses import RedirectResponse, FileResponse, Response, PlainTextResponse
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from urllib.parse import parse_qs, urljoin, urlparse
from webdriver_manager.chrome import ChromeDriverManager
import yt_dlp
import requests
from contextlib import asynccontextmanager
from dataclasses import asdict

from paths import get_resource_root, get_user_data_dir
from duplicate_finder import (
    resolve_scan_directory,
    resolve_deletable_file,
    iter_files,
    cluster_duplicates,
    build_groups_payload,
)
from existing_library import (
    build_library_normalized_names,
    matches_existing_library,
    resolve_optional_library_directory,
)
import webview_login_bridge
import workflow_heuristics
import pornhub_ph
import pixiv_ph
import browser_antibot
import browser_challenge_wait
import chrome_login_confirm
import workflow_browser
from settings import (
    AppSettings,
    apply_delay,
    effective_download_directory,
    site_download_directory,
    load_queue_snapshot,
    load_settings,
    maybe_notify_complete,
    save_settings,
    save_queue_snapshot,
    yt_dlp_format_string,
)

# ----------------------------- Configuration ----------------------------- #

RESOURCE_ROOT = get_resource_root()
USER_DATA_DIR = get_user_data_dir()
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = str((USER_DATA_DIR / "logs").resolve())
DOWNLOAD_DIR = str((USER_DATA_DIR / "downloads").resolve())
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
THUMBS_DIR = USER_DATA_DIR / "thumbs"
THUMBS_DIR.mkdir(parents=True, exist_ok=True)
SESSION_HISTORY_FILE = USER_DATA_DIR / "session_history.jsonl"
SESSION_REGISTRY_FILE = USER_DATA_DIR / "session_registry.json"

# --- Cancellation registry ---
import threading as _threading
_cancel_events: Dict[str, _threading.Event] = {}
_cancel_lock = _threading.Lock()

def _get_cancel_event(session_id: str) -> _threading.Event:
    """Create and register a fresh cancel event for a session."""
    ev = _threading.Event()
    with _cancel_lock:
        _cancel_events[session_id] = ev
    return ev

def _is_cancelled(session_id: str) -> bool:
    """Return True if the session has been asked to stop."""
    with _cancel_lock:
        ev = _cancel_events.get(session_id)
    return ev is not None and ev.is_set()

def _clear_cancel_event(session_id: str) -> None:
    """Remove the cancel event for a session (called during cleanup)."""
    with _cancel_lock:
        _cancel_events.pop(session_id, None)


_watcher_shutdown = _threading.Event()


def _watcher_main_loop() -> None:
    """Heartbeat when favourites watcher is enabled (MVP: log-only reminders)."""
    while True:
        if _watcher_shutdown.is_set():
            return
        s = load_settings()
        interval = max(60.0, float(s.watcher_interval_minutes) * 60.0)
        if s.watcher_enabled:
            logging.info(
                "[favorites watcher] enabled — heartbeat (~%s min); run a download session when ready",
                s.watcher_interval_minutes,
            )
        if _watcher_shutdown.wait(timeout=interval):
            return


@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    logging.info(
        "SHUCK3R browser stack login_v%s antibot_v%s (%s)",
        chrome_login_confirm.CHROME_LOGIN_WAIT_VERSION,
        browser_antibot.BROWSER_ANTIBOT_VERSION,
        getattr(browser_antibot, "__file__", "?"),
    )
    t = _threading.Thread(target=_watcher_main_loop, name="hamster-watcher", daemon=True)
    t.start()
    yield
    _watcher_shutdown.set()
    t.join(timeout=3.0)

from progress_tracker import get_tracker, cleanup_tracker, set_default_log_dir

set_default_log_dir(LOG_DIR)

app = FastAPI(
    title="SHUCK3R",
    description="Download videos from PornHub and xHamster favorites",
    lifespan=_app_lifespan,
)


def _desktop_session_progress_url(session_id: str, site: str) -> str:
    """Stable GET URL for the post-submit progress page (WebView-friendly after POST redirect)."""
    host = os.environ.get("HAMSTER_HOST", "127.0.0.1").strip()
    if host in ("0.0.0.0", "::", "[::]"):
        host = "127.0.0.1"
    port = int(os.environ.get("HAMSTER_PORT", "8001"))
    timestamp = int(time.time())
    return f"http://{host}:{port}/download/session/{session_id}?timestamp={timestamp}&site={site}"

templates = Jinja2Templates(directory=str(RESOURCE_ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(RESOURCE_ROOT / "static")), name="static")


def _workflow_cookie_file(site: str, session_id: str, cfg: AppSettings) -> str:
    if cfg.persistent_cookies:
        names = {
            "pornhub": "pornhub_cookies.txt",
            "xhamster": "xhamster_cookies.txt",
            "pixiv": "pixiv_cookies.txt",
        }
        name = names.get(site, f"{site}_cookies.txt")
        return str((USER_DATA_DIR / name).resolve())
    return os.path.join(LOG_DIR, f"{site}_cookies_{session_id}.txt")


def _pixiv_bookmark_rests(mode: str) -> List[str]:
    m = (mode or "public").strip().lower()
    if m in ("all", "both"):
        return ["show", "hide"]
    if m in ("private", "hide"):
        return ["hide"]
    return ["show"]


def _active_download_root() -> Path:
    return effective_download_directory(load_settings())

_LIBRARY_VIDEO_EXTS = frozenset({
    ".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v", ".wmv", ".flv",
})


def _resolve_under_downloads(relative: str) -> Path:
    """Resolve a URL-relative path under the downloads root (blocks .. traversal)."""
    rel = relative.replace("\\", "/").strip().lstrip("/")
    if not rel or ".." in rel.split("/"):
        raise HTTPException(status_code=400, detail="Invalid path.")
    root = _active_download_root().resolve()
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path.")
    return candidate


def _thumb_cache_name(relative: str) -> str:
    """Stable thumb filename for a video at *relative* path under downloads."""
    return relative.replace("\\", "__").replace("/", "__")

# Allowed suffixes for duplicate-checker inline video preview (same validation as path checks).
DUPLICATE_PREVIEW_VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v", ".wmv", ".flv", ".mpeg", ".mpg", ".3gp", ".ogv",
})


class DuplicateScanBody(BaseModel):
    directory: Optional[str] = None
    recursive: bool = False
    threshold: float = Field(0.72, ge=0.5, le=1.0)


class DuplicateDeleteBody(BaseModel):
    paths: List[str] = Field(..., min_length=1)


class SeleniumLoginConfirmBody(BaseModel):
    """Progress-page confirmation after logging in inside Selenium Chrome (non-embedded mode)."""

    session_id: str = Field(..., min_length=1)


class BrowserChallengeConfirmBody(BaseModel):
    """Progress-page confirmation after solving Cloudflare/CAPTCHA in visible Chrome."""

    session_id: str = Field(..., min_length=1)


# ----------------------------- Helper Functions ----------------------------- #

def setup_logging(log_file: str) -> None:
    """Configure logging."""
    # Clear existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create new handlers
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

def ensure_download_dir(directory: str) -> None:
    """Ensure that the download directory exists."""
    try:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Download directory is set to: {directory}")
    except Exception as e:
        logging.error(f"Failed to create download directory '{directory}': {e}")
        raise

def _selenium_headless_after_embedded_login() -> bool:
    """
    After site login inside pywebview, a visible Chrome window would cover SHUCK3R and block
    the menu (and feels like it replaced the app). Run automation headless unless opted in.
    """
    show = os.environ.get("HAMSTER_SHOW_SELENIUM", "").strip().lower()
    if show in ("1", "true", "yes"):
        return False
    return True


def setup_chrome_driver(headless: bool = False, proxy_url: str = "") -> webdriver.Chrome:
    """Backward-compatible wrapper — prefer ``workflow_browser.create_driver``."""
    cfg = load_settings()
    purpose = "scrape" if headless else "auth"
    return workflow_browser.create_driver(
        "xhamster",
        cfg,
        headless=headless,
        purpose=purpose,  # type: ignore[arg-type]
    )


def wait_for_manual_login(driver, login_url: str, session_id: str) -> None:
    """Prompt the user to log in manually and wait for confirmation (terminal or progress page)."""
    try:
        driver.get(login_url)
        logging.info("Navigated to login page. Please log in manually in the browser window.")
        workflow_heuristics.set_detail(
            session_id,
            chrome_login_confirm.chrome_login_progress_hint(),
        )
        chrome_login_confirm.wait_for_chrome_login(session_id)
        logging.info("User confirmed login.")
    except Exception as e:
        logging.error(f"Error during manual login: {e}")
        raise

def save_cookies_netscape(driver, cookie_file: str) -> None:
    """Save cookies in Netscape format for yt-dlp."""
    cookies = driver.get_cookies()
    with open(cookie_file, "w") as f:
        f.write("# Netscape HTTP Cookie File\n")
        f.write("# This file was generated by Selenium\n")
        for cookie in cookies:
            domain = cookie.get("domain", "")
            flag = "TRUE" if domain.startswith(".") else "FALSE"
            path = cookie.get("path", "/")
            secure = "TRUE" if cookie.get("secure", False) else "FALSE"
            expiry = cookie.get("expiry", 0)
            name = cookie.get("name", "")
            value = cookie.get("value", "")
            f.write(f"{domain}\t{flag}\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n")
    logging.info(f"Cookies saved to {cookie_file}")


def load_netscape_cookies_into_driver(driver: webdriver.Chrome, cookie_file: str, landing_url: str) -> None:
    """Apply a Netscape cookie jar to Selenium after opening landing_url (same registrable domain)."""
    driver.get(landing_url)
    time.sleep(1.5)
    try:
        with open(cookie_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    except OSError as e:
        logging.error("Could not read cookie file for Selenium: %s", e)
        return

    for line in lines:
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        domain, _flag, path, secure, expiry_s, name, value = parts[:7]
        spec: dict = {
            "name": name,
            "value": value,
            "domain": domain,
            "path": path or "/",
            "secure": secure.upper() == "TRUE",
        }
        try:
            exp_int = int(expiry_s)
            if exp_int > 0:
                spec["expiry"] = exp_int
        except ValueError:
            pass
        try:
            driver.add_cookie(spec)
        except Exception as e:
            logging.debug("Skipping cookie %s: %s", name, e)

    driver.get(landing_url)
    time.sleep(1.0)

def _pornhub_tracker_titles(video_urls: List[str]) -> List[str]:
    """Short labels for progress UI (viewkey when present)."""
    out: List[str] = []
    for u in video_urls:
        q = parse_qs(urlparse(u).query)
        vk = (q.get("viewkey") or [""])[0]
        out.append(vk or u[-20:])
    return out


def fetch_html_selenium(
    driver,
    url: str,
    *,
    session_id: Optional[str] = None,
    site: str = "xhamster",
    cfg: Optional[AppSettings] = None,
) -> Tuple[Optional[str], Any]:
    """Fetch HTML via Selenium with a full scroll to trigger lazy-loaded content.

    Returns ``(html_or_none, driver)`` — *driver* may be replaced after challenge escalation.
    """
    try:
        driver.get(url)
        logging.info(f"Navigated to {url}")
        
        # Wait for page to load
        time.sleep(2)

        if session_id and cfg is not None:
            replacement = workflow_browser.check_page_challenge(
                driver, url, session_id=session_id, site=site, cfg=cfg
            )
            if replacement is not None:
                driver = replacement
                driver.get(url)
                time.sleep(2)
        
        # Check if we're on a valid page (not 404 or error page)
        if "404" in driver.title or "error" in driver.title.lower() or "not found" in driver.title.lower():
            logging.warning(f"Page appears to be an error page: {driver.title}")
            return None, driver
        
        # Check if we're still on the favorites page (not redirected to login)
        current_url = driver.current_url
        if "login" in current_url or "signin" in current_url:
            logging.warning(f"Redirected to login page: {current_url}")
            return None, driver
        
        scroll_to_bottom(driver)
        
        # Additional wait after scrolling
        time.sleep(1)
        
        return driver.page_source, driver
    except Exception as e:
        logging.error(f"Error fetching HTML from {url}: {e}")
        return None, driver

def scroll_to_bottom(driver, pause_time: float = 2.0) -> None:
    """Scroll to the bottom of the page to load all content."""
    try:
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause_time)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                logging.info("Reached the bottom of the page.")
                break
            last_height = new_height
    except Exception as e:
        logging.error(f"Error during scrolling: {e}")

def try_click_next_button(driver) -> bool:
    """Click the next-page button if one is visible and enabled.

    Tries a prioritised list of xHamster-specific and generic CSS selectors.
    Returns True if a button was found and clicked, False if pagination is exhausted.
    """
    try:
        # Look for xHamster-specific next page button selectors
        next_selectors = [
            "a[data-page='next']",  # xHamster specific
            "li.next a",  # xHamster specific
            "a[aria-label='Next page']",
            "a[title='Next page']",
            "a.next",
            "a[class*='next']",
            "button[aria-label='Next page']",
            "button[title='Next page']",
            "button.next",
            "button[class*='next']",
            "a[href*='/favorites/videos/']",  # xHamster pagination links
        ]
        
        for selector in next_selectors:
            try:
                next_button = driver.find_element(By.CSS_SELECTOR, selector)
                if next_button.is_enabled() and next_button.is_displayed():
                    logging.info(f"Found next button with selector: {selector}")
                    driver.execute_script("arguments[0].click();", next_button)
                    time.sleep(3)  # Wait for page to load
                    return True
            except NoSuchElementException:
                continue
        
        logging.info("No next page button found")
        return False
    except Exception as e:
        logging.error(f"Error trying to click next button: {e}")
        return False

def parse_video_links_xhamster(html: str, base_url: str) -> List[str]:
    """Parse video links for xHamster."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        video_links = []
        
        # Try multiple selectors for video links
        selectors = [
            'a.thumb-image-container',
            'a[class*="thumb"]',
            'a[href*="/videos/"]'
        ]
        
        for selector in selectors:
            video_anchors = soup.select(selector)
            logging.debug(f"Found {len(video_anchors)} elements with selector '{selector}'")
            
            for anchor in video_anchors:
                href = anchor.get('href')
                if href and '/videos/' in href:
                    video_url = urljoin(base_url, href)
                    if video_url not in video_links:  # Avoid duplicates
                        video_links.append(video_url)
                        logging.debug(f"Found xHamster video URL: {video_url}")
        
        logging.info(f"Parsed {len(video_links)} xHamster video links.")
        return video_links
    except Exception as e:
        logging.error(f"Error parsing xHamster video links: {e}")
        return []

def extract_direct_video_url(driver, video_page_url: str) -> Tuple[Optional[str], str]:
    """Extract direct video URL and title from a video page using browser session."""
    try:
        logging.info(f"Extracting direct video URL from: {video_page_url}")
        
        # Set shorter timeout for individual page loads
        driver.set_page_load_timeout(15)
        driver.get(video_page_url)
        time.sleep(1)  # Minimal wait time
        
        # Extract video title first
        video_title = "Unknown Video"
        try:
            # Try to get title from page title
            page_title = driver.title
            if page_title and "|" in page_title:
                # xHamster format: "Video Title: Category | xHamster"
                video_title = page_title.split("|")[0].strip()
                if ":" in video_title:
                    video_title = video_title.split(":")[0].strip()
            elif page_title:
                video_title = page_title.replace(" | xHamster", "").strip()
            
            # Sanitise title for use as a filename
            video_title = re.sub(r'[<>:"/\\|?*]', '_', video_title)
            video_title = re.sub(r'\s+', '_', video_title)
            video_title = video_title[:100]  # Limit length
            
            logging.info(f"Extracted video title: {video_title}")
        except Exception as e:
            logging.warning(f"Could not extract video title: {e}")
        
        # Method 1: Look for video element
        try:
            video_elements = driver.find_elements("tag name", "video")
            for video in video_elements:
                src = video.get_attribute("src")
                if src and ('.mp4' in src or '.m3u8' in src):
                    logging.info(f"Found video src: {src}")
                    return src, video_title
        except Exception as e:
            logging.warning(f"Error checking video elements: {e}")
        
        # Method 2: Look in page source for video URLs
        try:
            page_source = driver.page_source
            
            # Look for HLS URLs (preferred)
            m3u8_patterns = [
                r'https://video-cf\.xhcdn\.com/[^"\']*\.m3u8[^"\']*',
                r'https://video[0-9]*\.xhcdn\.com/[^"\']*\.m3u8[^"\']*',
                r'https://video-nss\.xhcdn\.com/[^"\']*\.m3u8[^"\']*',
            ]
            
            for pattern in m3u8_patterns:
                matches = re.findall(pattern, page_source)
                for match in matches:
                    if '.m3u8' in match:
                        logging.info(f"Found HLS URL: {match}")
                        return match, video_title
            
            # Look for direct MP4 URLs (fallback)
            mp4_patterns = [
                r'https://galleryn[0-9]+\.vcmdiawe\.com/[^"\']*\.mp4[^"\']*',
                r'https://video[0-9]*\.xhcdn\.com/[^"\']*\.mp4[^"\']*',
            ]
            
            for pattern in mp4_patterns:
                matches = re.findall(pattern, page_source)
                for match in matches:
                    if '.mp4' in match:
                        logging.info(f"Found MP4 URL: {match}")
                        return match, video_title
        except Exception as e:
            logging.warning(f"Error searching page source: {e}")
        
        logging.warning(f"No direct video URL found for: {video_page_url}")
        return None, video_title
        
    except Exception as e:
        logging.error(f"Error extracting direct video URL from {video_page_url}: {e}")
        return None, "Unknown_Video"
    finally:
        # Reset timeout to original value
        try:
            driver.set_page_load_timeout(30)
        except:
            pass

def create_authenticated_session(driver) -> requests.Session:
    """Create authenticated requests session from Selenium cookies."""
    try:
        session = requests.Session()
        cookies = driver.get_cookies()
        for cookie in cookies:
            domain = cookie['domain'].lstrip('.')
            session.cookies.set(cookie['name'], cookie['value'], domain=domain)
        logging.info("Authenticated requests session created.")
        return session
    except Exception as e:
        logging.error(f"Error creating authenticated session: {e}")
        raise

def get_video_title(session: requests.Session, video_url: str) -> str:
    """Get video title from video page."""
    try:
        response = session.get(video_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('h1')
        if title_tag:
            title = title_tag.get_text(strip=True)
            sanitized_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else "_" for c in title)
            return sanitized_title
        else:
            logging.warning(f"Title not found for {video_url}")
            return f"video_{int(time.time())}"
    except Exception as e:
        logging.error(f"Error fetching title for {video_url}: {e}")
        return f"video_{int(time.time())}"
def _ytdlp_file_exists_for_title(download_dir: str, title: str) -> bool:
    base = Path(download_dir)
    for ext in (".mp4", ".mkv", ".webm", ".avi", ".m4v"):
        if (base / f"{title}{ext}").is_file():
            return True
    return False


def download_video_ytdlp(
    video_url: str,
    title: str,
    cookie_file: str,
    download_dir: str,
    session_id: str,
    *,
    settings_snapshot: Optional[AppSettings] = None,
) -> None:
    """Download video using yt-dlp with progress tracking and authentication."""
    cfg = settings_snapshot or load_settings()
    retries = cfg.yt_dlp_retries if cfg.yt_dlp_retries >= 0 else 3
    tracker = get_tracker(session_id)
    if cfg.skip_existing_in_download_dir and _ytdlp_file_exists_for_title(download_dir, title):
        tracker.skip_download(video_url, "File already exists in download folder")
        return

    tracker.start_download(video_url)

    filepath = os.path.join(download_dir, f"{title}.%(ext)s")
    fmt = yt_dlp_format_string(cfg)
    ydl_opts: Dict[str, Any] = {
        "cookiesfrombrowser": None,  # We'll use cookies file instead
        "cookiefile": cookie_file,
        "outtmpl": filepath,
        "format": fmt,
        "merge_output_format": "mp4",
        "hls_prefer_native": True,
        "quiet": False,
        "no_warnings": False,
        "retries": retries,
        "ignoreerrors": False,
        "extractor_retries": retries,
        "fragment_retries": retries,
        "writeinfojson": False,
        "writesubtitles": False,
        "writeautomaticsub": False,
    }
    pv = (cfg.proxy_url or "").strip()
    if pv:
        ydl_opts["proxy"] = pv

    def progress_hook(d):
        if d["status"] == "downloading":
            if "total_bytes" in d and d["total_bytes"]:
                percent = (d["downloaded_bytes"] / d["total_bytes"]) * 100
                logging.info(f"Downloading {title}: {percent:.1f}%")
        elif d["status"] == "finished":
            logging.info(f"Finished downloading {title}")

    ydl_opts["progress_hooks"] = [progress_hook]

    try:
        logging.info(f"Downloading video with authentication: {video_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Find the actual downloaded file
        actual_file = None
        for ext in [".mp4", ".mkv", ".webm", ".avi"]:
            potential_file = os.path.join(download_dir, f"{title}{ext}")
            if os.path.exists(potential_file):
                actual_file = potential_file
                break

        if actual_file:
            file_size = os.path.getsize(actual_file)
            tracker.complete_download(video_url, actual_file, file_size)
            logging.info(f"Successfully downloaded: {actual_file} ({file_size} bytes)")
        else:
            tracker.fail_download(video_url, "Downloaded file not found")
            logging.error(f"No file found for {title}")

    except Exception as e:
        tracker.fail_download(video_url, str(e))
        logging.error(f"Exception downloading {video_url}: {e}")


def download_video_direct(
    video_url: str,
    title: str,
    download_dir: str,
    session_id: str,
    *,
    settings_snapshot: Optional[AppSettings] = None,
) -> None:
    """Download video using yt-dlp without cookies with progress tracking."""
    cfg = settings_snapshot or load_settings()
    retries = cfg.yt_dlp_retries if cfg.yt_dlp_retries >= 0 else 3
    tracker = get_tracker(session_id)
    if cfg.skip_existing_in_download_dir and _ytdlp_file_exists_for_title(download_dir, title):
        tracker.skip_download(video_url, "File already exists in download folder")
        return

    tracker.start_download(video_url)

    filepath = os.path.join(download_dir, f"{title}.%(ext)s")
    fmt = yt_dlp_format_string(cfg)
    ydl_opts: Dict[str, Any] = {
        "outtmpl": filepath,
        "format": fmt,
        "merge_output_format": "mp4",
        "hls_prefer_native": True,
        "quiet": False,
        "no_warnings": False,
        "retries": retries,
        "ignoreerrors": False,
        "extractor_retries": retries,
        "fragment_retries": retries,
        "writeinfojson": False,
        "writesubtitles": False,
        "writeautomaticsub": False,
    }
    pv = (cfg.proxy_url or "").strip()
    if pv:
        ydl_opts["proxy"] = pv

    def progress_hook(d):
        if d["status"] == "downloading":
            if "total_bytes" in d and d["total_bytes"]:
                percent = (d["downloaded_bytes"] / d["total_bytes"]) * 100
                logging.info(f"Downloading {title}: {percent:.1f}%")
        elif d["status"] == "finished":
            logging.info(f"Finished downloading {title}")

    ydl_opts["progress_hooks"] = [progress_hook]

    try:
        logging.info(f"Downloading xHamster video: {video_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Find the actual downloaded file
        actual_file = None
        for ext in [".mp4", ".mkv", ".webm", ".avi"]:
            potential_file = os.path.join(download_dir, f"{title}{ext}")
            if os.path.exists(potential_file):
                actual_file = potential_file
                break

        if actual_file:
            file_size = os.path.getsize(actual_file)
            tracker.complete_download(video_url, actual_file, file_size)
            logging.info(f"Successfully downloaded: {actual_file} ({file_size} bytes)")
        else:
            tracker.fail_download(video_url, "Downloaded file not found")
            logging.error(f"No file found for {title}")

    except Exception as e:
        tracker.fail_download(video_url, str(e))
        logging.error(f"Exception downloading {video_url}: {e}")


def download_videos_parallel(
    video_info_list: List[Tuple[str, str]],
    download_dir: str,
    max_workers: int = 4,
    cookie_file: str = None,
    session_id: str = None,
    *,
    settings_snapshot: Optional[AppSettings] = None,
) -> None:
    """Download a batch of videos in parallel and track progress.

    Args:
        video_info_list: list of (url, title) pairs to download.
        download_dir: destination directory for downloaded files.
        max_workers: thread-pool size (default 4).
        cookie_file: path to a Netscape cookie file; if provided each worker calls
            download_video_ytdlp, otherwise download_video_direct.
        session_id: progress-tracker session key; a timestamp-based fallback is used
            when None.

    Each worker is given a 1-hour timeout; stragglers are cancelled on exit.
    """
    if not session_id:
        session_id = str(int(time.time()))
        
    try:
        tracker = get_tracker(session_id)
        
        # Only start session if not already started
        if not tracker.stats.start_time:
            tracker.start_session(len(video_info_list))
            # Add URLs to tracking if not already added
            urls = [item[0] for item in video_info_list]
            titles = [item[1] for item in video_info_list]
            tracker.add_urls(urls, titles)
        
        cfg = settings_snapshot or load_settings()
        logging.info(f"Starting download of {len(video_info_list)} videos with {max_workers} workers")

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = []
            for video_url, title in video_info_list:
                apply_delay(cfg.download_delay_seconds, cfg.delay_variance_seconds)
                if cookie_file:
                    futures.append(
                        executor.submit(
                            download_video_ytdlp,
                            video_url,
                            title,
                            cookie_file,
                            download_dir,
                            session_id,
                            settings_snapshot=cfg,
                        )
                    )
                else:
                    futures.append(
                        executor.submit(
                            download_video_direct,
                            video_url,
                            title,
                            download_dir,
                            session_id,
                            settings_snapshot=cfg,
                        )
                    )

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result(timeout=3600)
                except concurrent.futures.TimeoutError:
                    logging.error("A download worker timed out after 1 hour; skipping.")
                except Exception as e:
                    logging.error(f"Error in video download: {e}")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        logging.info("Parallel download workers finished for this batch.")

    except Exception as e:
        logging.critical(f"Critical error in download_videos_parallel: {e}")

# ----------------------------- Site-Specific Workflows ----------------------------- #

def _session_meta_path(session_id: str) -> Path:
    return Path(LOG_DIR) / f"session_meta_{session_id}.json"


def _write_session_meta(session_id: str, site: str) -> None:
    try:
        _session_meta_path(session_id).write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "site": site,
                    "started_ts": datetime.now().isoformat(timespec="seconds"),
                }
            ),
            encoding="utf-8",
        )
    except Exception as e:
        logging.warning("Could not write session meta for %s: %s", session_id, e)


def _load_session_registry() -> Dict[str, dict]:
    if not SESSION_REGISTRY_FILE.is_file():
        return {}
    try:
        raw = json.loads(SESSION_REGISTRY_FILE.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _save_session_registry(registry: Dict[str, dict]) -> None:
    SESSION_REGISTRY_FILE.write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )


def _stats_from_tracker(tracker) -> dict:
    stats = tracker.stats
    duration_s = 0
    if stats.start_time:
        end = stats.end_time or datetime.now()
        duration_s = int((end - stats.start_time).total_seconds())
    return {
        "total": stats.total_urls,
        "downloaded": stats.downloads_completed,
        "skipped": stats.files_skipped,
        "failed": stats.downloads_failed,
        "duration_s": duration_s,
    }


def session_history_register(
    session_id: str, site: str, *, timestamp: Optional[int] = None
) -> None:
    """Record a new run as soon as the user starts a download (before background work finishes)."""
    try:
        reg = _load_session_registry()
        now = datetime.now().isoformat(timespec="seconds")
        reg[session_id] = {
            "ts": now,
            "updated_ts": now,
            "timestamp": timestamp or int(time.time()),
            "site": site,
            "session_id": session_id,
            "status": "running",
            "total": 0,
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "duration_s": 0,
            "error_message": None,
        }
        _save_session_registry(reg)
        _write_session_meta(session_id, site)
    except Exception as e:
        logging.warning("Could not register session history: %s", e)


def session_history_update(
    session_id: str,
    site: str,
    *,
    status: str,
    tracker=None,
    error_message: Optional[str] = None,
) -> None:
    """Update registry entry (running, completed, failed, cancelled)."""
    try:
        reg = _load_session_registry()
        now = datetime.now().isoformat(timespec="seconds")
        rec = dict(reg.get(session_id) or {})
        rec.setdefault("ts", now)
        rec["updated_ts"] = now
        rec["site"] = site
        rec["session_id"] = session_id
        rec["status"] = status
        if tracker is not None:
            rec.update(_stats_from_tracker(tracker))
        if error_message:
            rec["error_message"] = error_message
        reg[session_id] = rec
        _save_session_registry(reg)
    except Exception as e:
        logging.warning("Could not update session history: %s", e)


def _append_session_history(session_id: str, site: str, tracker) -> None:
    """Finalize: registry + append JSONL line for completed sessions."""
    session_history_update(session_id, site, status="completed", tracker=tracker)
    try:
        rec = _load_session_registry().get(session_id) or {}
        line = {
            "ts": rec.get("updated_ts") or datetime.now().isoformat(timespec="seconds"),
            "site": site,
            "session_id": session_id,
            "status": "completed",
            "total": rec.get("total", 0),
            "downloaded": rec.get("downloaded", 0),
            "skipped": rec.get("skipped", 0),
            "failed": rec.get("failed", 0),
            "duration_s": rec.get("duration_s", 0),
        }
        with open(SESSION_HISTORY_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
    except Exception as e:
        logging.warning("Could not append session history JSONL: %s", e)


def _history_from_progress_files(registry: Dict[str, dict]) -> Dict[str, dict]:
    """Merge in-progress sessions from progress_*.json on disk."""
    out = dict(registry)
    log_root = Path(LOG_DIR)
    for prog_path in log_root.glob("progress_*.json"):
        sid = prog_path.stem.replace("progress_", "", 1)
        if not sid:
            continue
        existing = out.get(sid) or {}
        if existing.get("status") == "completed":
            continue
        try:
            data = json.loads(prog_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stats = data.get("stats") or {}
        site = existing.get("site")
        meta_path = _session_meta_path(sid)
        if meta_path.is_file():
            try:
                site = json.loads(meta_path.read_text(encoding="utf-8")).get("site") or site
            except Exception:
                pass
        started = stats.get("start_time") or existing.get("ts")
        rec = {
            "ts": existing.get("ts") or started or datetime.now().isoformat(timespec="seconds"),
            "updated_ts": data.get("timestamp") or existing.get("updated_ts"),
            "site": site or "unknown",
            "session_id": sid,
            "status": existing.get("status") or "running",
            "total": int(stats.get("total_urls") or 0),
            "downloaded": int(stats.get("downloads_completed") or 0),
            "skipped": int(stats.get("files_skipped") or 0),
            "failed": int(stats.get("downloads_failed") or 0),
            "duration_s": int(existing.get("duration_s") or 0),
            "error_message": existing.get("error_message"),
        }
        tl = workflow_heuristics.get_timeline(sid)
        if tl and tl.get("error_message"):
            rec["status"] = "failed"
            rec["error_message"] = tl["error_message"]
        out[sid] = rec
    return out


def _read_progress_json(session_id: str) -> Optional[dict]:
    """Read the persisted progress JSON for a session (survives cleanup_tracker)."""
    p = Path(LOG_DIR) / f"progress_{session_id}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_ffprobe(path: str) -> Tuple[bool, str]:
    """Use ffprobe to verify a video file's integrity.

    Returns ``(True, '')`` when OK, ``(False, reason)`` when corrupt or unreadable,
    and ``(True, 'skipped')`` when ffprobe is not on PATH.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return True, "skipped — ffprobe not found"
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_type",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0 or result.stderr.strip():
            return False, (result.stderr.strip()[:300] or "ffprobe reported an error")
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "ffprobe timed out"
    except Exception as e:
        return False, str(e)


def _generate_thumbnail(video_path: str, thumb_path: str) -> bool:
    """Extract a single frame at t=5 s from *video_path* and save as JPEG to *thumb_path*.

    Returns ``True`` on success, ``False`` when ffmpeg is unavailable or fails.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-ss", "5", "-i", video_path, "-frames:v", "1", "-q:v", "4", "-y", thumb_path],
            capture_output=True, text=True, timeout=60,
        )
        return result.returncode == 0 and os.path.isfile(thumb_path)
    except Exception:
        return False


def pornhub_workflow(
    playlist_url: str,
    download_dir: str,
    headless: bool,
    log_file: str,
    session_id: str = None,
    *,
    library_dir: Optional[str] = None,
    library_recursive: bool = False,
    url_override: Optional[List[str]] = None,
    cancel_event: Optional[_threading.Event] = None,
) -> None:
    """PornHub workflow: ph.py-style favourites collection + mediaDefinitions downloads."""
    if not session_id:
        session_id = str(int(time.time()))

    setup_logging(log_file)
    cfg = load_settings()
    download_root = str(site_download_directory("pornhub", cfg))
    ensure_download_dir(download_root)

    cookie_file = _workflow_cookie_file("pornhub", session_id, cfg)
    driver = None
    ok = False

    try:
        use_embedded = webview_login_bridge.is_active() and webview_login_bridge.embedded_login_env_enabled()
        if use_embedded:
            workflow_heuristics.advance(session_id, "site_login")
            webview_login_bridge.begin_embedded_login(
                "https://www.pornhub.com/",
                cookie_file,
                progress_url=_desktop_session_progress_url(session_id, "pornhub"),
            )
            time.sleep(1.0)
        else:
            scrape_headless = workflow_browser.scrape_headless_enabled(headless, cfg)
            workflow_heuristics.advance(session_id, "cookie_reuse")
            cookies_ok, cookie_msg = workflow_browser.cookies_valid_for_site(
                "pornhub", cookie_file, cfg
            )
            workflow_heuristics.set_detail(session_id, cookie_msg)
            if cookies_ok:
                logging.info("PornHub cookie reuse: %s", cookie_msg)
                workflow_heuristics.set_detail(
                    session_id, "Using saved session — skipping login…",
                )
                if scrape_headless:
                    workflow_heuristics.advance(session_id, "headless_scrape")
                workflow_heuristics.advance(session_id, "extract_list")
                driver = workflow_browser.create_driver(
                    "pornhub", cfg, headless=scrape_headless, purpose="scrape"
                )
                workflow_browser.apply_flaresolverr_preflight(
                    driver, "https://www.pornhub.com/", cfg
                )
                load_netscape_cookies_into_driver(
                    driver, cookie_file, "https://www.pornhub.com/"
                )
            else:
                workflow_heuristics.advance(session_id, "chrome_login")
                auth_driver = workflow_browser.create_driver(
                    "pornhub", cfg, headless=False, purpose="auth"
                )
                auth_driver.get("https://www.pornhub.com")
                logging.info("Please log in to PornHub in the opened browser.")
                workflow_heuristics.set_detail(
                    session_id,
                    chrome_login_confirm.chrome_login_progress_hint(),
                )
                chrome_login_confirm.wait_for_chrome_login(session_id)
                time.sleep(5)
                save_cookies_netscape(auth_driver, cookie_file)
                healthy, health_msg = pornhub_ph.check_cookie_health(
                    cookie_file, cfg.proxy_url
                )
                if not healthy:
                    logging.warning("PornHub cookie health check failed: %s", health_msg)
                    workflow_heuristics.mark_error(
                        session_id, f"Login check failed — {health_msg}"
                    )
                    workflow_browser.quit_driver(auth_driver)
                    return
                logging.info("PornHub cookie health check passed: %s", health_msg)
                if scrape_headless:
                    workflow_heuristics.advance(session_id, "headless_scrape")
                    workflow_browser.quit_driver(auth_driver)
                    driver = workflow_browser.create_driver(
                        "pornhub", cfg, headless=True, purpose="scrape"
                    )
                    load_netscape_cookies_into_driver(
                        driver, cookie_file, "https://www.pornhub.com/"
                    )
                else:
                    driver = auth_driver
                workflow_heuristics.advance(session_id, "extract_list")

        if use_embedded:
            workflow_heuristics.advance(session_id, "extract_list")
        if url_override is not None:
            video_urls = url_override
            workflow_heuristics.set_detail(session_id, f"Retrying {len(url_override)} failed URLs…")
        elif use_embedded:
            workflow_heuristics.set_detail(
                session_id,
                "Collecting favourites (HTTP, paginated) — same approach as ph.py…",
            )
            video_urls = pornhub_ph.collect_favorites_urls_requests(
                cookie_file,
                playlist_url,
                proxy_url=cfg.proxy_url,
            )
        else:
            workflow_heuristics.set_detail(
                session_id,
                "Collecting favourites in Chrome (scroll + Load more) — ph.py style…",
            )
            video_urls = pornhub_ph.collect_favorites_urls_with_driver(driver, playlist_url)
            # Keep Chrome open — video pages must be opened in the authenticated
            # browser so flashvars_* (mediaDefinitions) is present in the HTML.
            # ph.py does the same: driver stays alive for the full download batch.

        if not video_urls:
            logging.error("No video URLs extracted.")
            workflow_heuristics.mark_error(
                session_id,
                "No videos found in favourites (empty list, login required, or parsing found no links).",
            )
            return

        workflow_heuristics.set_detail(session_id, f"Found {len(video_urls)} videos in favourites")

        if cfg.persist_queue_snapshots:
            try:
                save_queue_snapshot(session_id, "pornhub", video_urls, USER_DATA_DIR)
            except Exception as e:
                logging.warning("Could not save queue snapshot: %s", e)

        workflow_heuristics.advance(session_id, "download")

        if library_dir:
            tracker = get_tracker(session_id)
            tracker.start_session(len(video_urls))
            norms = build_library_normalized_names(
                Path(library_dir), recursive=library_recursive
            )
            workflow_heuristics.set_detail(
                session_id,
                f"Resolving titles ({len(video_urls)} videos) · comparing to your library folder…",
            )
            to_download: List[Tuple[str, str]] = []
            skipped_n = 0
            for i, url in enumerate(video_urls):
                title = pornhub_ph.fetch_video_title_from_page(cookie_file, url, proxy_url=cfg.proxy_url)
                if matches_existing_library(title, norms):
                    tracker.add_urls([url], [title])
                    tracker.skip_download(url, "Already in library folder")
                    skipped_n += 1
                else:
                    to_download.append((url, title))
                if (i + 1) % 40 == 0 or (i + 1) == len(video_urls):
                    logging.info(
                        "Library compare: %s/%s URLs (skipped %s matches so far)",
                        i + 1,
                        len(video_urls),
                        skipped_n,
                    )
            logging.info(
                "Library filter complete: %s skipped (already on disk), %s to download",
                skipped_n,
                len(to_download),
            )
            workflow_heuristics.set_detail(
                session_id,
                f"Downloading {len(to_download)} new videos ({skipped_n} skipped · already in library)",
            )
            if to_download:
                if cancel_event and cancel_event.is_set():
                    logging.info("Cancellation requested before PH download batch — stopping.")
                    workflow_heuristics.mark_error(session_id, "Download cancelled by user.")
                    return
                tracker.add_urls(
                    [pair[0] for pair in to_download],
                    [pair[1] for pair in to_download],
                )
                pornhub_ph.download_pornhub_videos_parallel(
                    [pair[0] for pair in to_download],
                    download_root,
                    cookie_file,
                    session_id,
                    max_workers=max(1, cfg.max_parallel_downloads),
                    driver=driver,
                    proxy_url=cfg.proxy_url,
                    skip_if_exists=cfg.skip_existing_in_download_dir,
                    yt_dlp_retries=cfg.yt_dlp_retries,
                )
            else:
                logging.info("All favorites matched files in the library folder; nothing to download.")
        else:
            tracker = get_tracker(session_id)
            tracker.start_session(len(video_urls))
            tracker.add_urls(video_urls, _pornhub_tracker_titles(video_urls))
            if cancel_event and cancel_event.is_set():
                logging.info("Cancellation requested before PH download — stopping.")
                workflow_heuristics.mark_error(session_id, "Download cancelled by user.")
                return
            pornhub_ph.download_pornhub_videos_parallel(
                video_urls,
                download_root,
                cookie_file,
                session_id,
                max_workers=max(1, cfg.max_parallel_downloads),
                driver=driver,
                proxy_url=cfg.proxy_url,
                skip_if_exists=cfg.skip_existing_in_download_dir,
                yt_dlp_retries=cfg.yt_dlp_retries,
            )

        ok = True

    except Exception as e:
        logging.critical(f"Error in PornHub workflow: {e}")
        workflow_heuristics.mark_error(session_id, str(e))
    finally:
        workflow_browser.quit_driver(driver)
        try:
            if not cfg.persistent_cookies and os.path.exists(cookie_file):
                os.unlink(cookie_file)
        except Exception as e:
            logging.warning(f"Could not delete cookie file {cookie_file}: {e}")
        if ok:
            try:
                _append_session_history(session_id, "pornhub", get_tracker(session_id))
                cleanup_tracker(session_id)
            except Exception as e:
                logging.error(f"Error cleaning up tracker: {e}")
            workflow_heuristics.complete_success(session_id)
            maybe_notify_complete("SHUCK3R", "PornHub download session finished.", cfg)
        else:
            _record_failed_session_history(session_id, "pornhub")
        _clear_cancel_event(session_id)


def _record_failed_session_history(session_id: str, site: str) -> None:
    tl = workflow_heuristics.get_timeline(session_id) or {}
    err = tl.get("error_message")
    status = "cancelled"
    if not err or "cancel" not in err.lower():
        status = "failed"
    tracker = None
    try:
        tracker = get_tracker(session_id)
    except Exception:
        pass
    session_history_update(
        session_id, site, status=status, tracker=tracker, error_message=err
    )


def pixiv_workflow(
    target_user_id: str,
    bookmark_mode: str,
    download_dir: str,
    headless: bool,
    log_file: str,
    session_id: str,
    *,
    cancel_event: Optional[_threading.Event] = None,
) -> None:
    """Pixiv bookmark workflow: Chrome login → Ajax bookmark list → image download."""
    if not session_id:
        session_id = str(int(time.time()))

    setup_logging(log_file)
    cfg = load_settings()
    download_root = str(site_download_directory("pixiv", cfg))
    ensure_download_dir(download_root)

    cookie_file = _workflow_cookie_file("pixiv", session_id, cfg)
    driver = None
    ok = False
    rests = _pixiv_bookmark_rests(bookmark_mode)

    def _cancelled() -> bool:
        return bool(cancel_event and cancel_event.is_set())

    try:
        workflow_heuristics.advance(session_id, "cookie_reuse")
        cookies_ok, cookie_msg = workflow_browser.cookies_valid_for_site(
            "pixiv", cookie_file, cfg
        )
        workflow_heuristics.set_detail(session_id, cookie_msg)

        if cookies_ok:
            logging.info("Pixiv cookie reuse: %s", cookie_msg)
            workflow_heuristics.set_detail(
                session_id, "Using saved Pixiv session — skipping login…",
            )
            session = pixiv_ph.load_session(cookie_file, cfg.proxy_url)
        else:
            workflow_heuristics.advance(session_id, "chrome_login")
            driver = workflow_browser.create_driver(
                "pixiv", cfg, headless=False, purpose="auth"
            )
            workflow_browser.apply_flaresolverr_preflight(
                driver, pixiv_ph.PIXIV_HOME, cfg
            )
            try:
                driver.get(pixiv_ph.PIXIV_HOME)
                time.sleep(1)
                driver.delete_all_cookies()
                logging.info(
                    "Cleared Pixiv Chrome profile cookies (avoids stale PHPSESSID)."
                )
            except Exception as e:
                logging.warning("Could not clear Pixiv profile cookies: %s", e)
            driver.get(pixiv_ph.PIXIV_LOGIN)
            logging.info("Please log in to Pixiv in the opened Chrome window.")
            workflow_heuristics.set_detail(
                session_id,
                pixiv_ph.PIXIV_LOGIN_PROGRESS_HINT,
            )
            try:
                chrome_login_confirm.wait_for_chrome_login(
                    session_id, should_stop=_cancelled
                )
            except InterruptedError:
                workflow_heuristics.mark_error(
                    session_id, "Download cancelled by user."
                )
                return
            try:
                pixiv_ph.establish_www_session(
                    driver,
                    timeout=120.0,
                    target_user_id=target_user_id,
                    bookmark_rest=rests[0] if rests else "show",
                    cancel_check=_cancelled,
                )
            except pixiv_ph.DownloadCancelled:
                workflow_heuristics.mark_error(
                    session_id, "Download cancelled by user."
                )
                return
            except RuntimeError as e:
                workflow_heuristics.mark_error(
                    session_id, f"Pixiv login check failed — {e}"
                )
                return
            workflow_heuristics.advance(session_id, "save_cookies")
            live_session = pixiv_ph.session_from_driver(driver, cfg.proxy_url)
            ok_sess, sess_msg = pixiv_ph.check_cookie_health_with_session(live_session)
            if not ok_sess:
                logging.warning(
                    "Requests session check failed (%s); using browser-verified cookies anyway",
                    sess_msg,
                )
            pixiv_ph.save_session_from_driver(driver, cookie_file)
            workflow_browser.quit_driver(driver)
            driver = None
            session = live_session

        if cfg.challenge_solver == "flaresolverr" and cookies_ok:
            try:
                pixiv_ph.apply_flaresolverr_to_session(
                    session, base_url=cfg.flaresolverr_base_url
                )
            except Exception as e:
                logging.warning("FlareSolverr preflight for Pixiv failed: %s", e)

        uid = pixiv_ph.parse_user_id(target_user_id) or pixiv_ph.get_logged_in_user_id(
            session
        )
        logging.info("Pixiv target user id: %s (rests=%s)", uid, rests)

        workflow_heuristics.advance(session_id, "collect_urls")
        workflow_heuristics.set_detail(
            session_id,
            f"Collecting bookmarks for user {uid} ({', '.join(rests)})…",
        )
        works = pixiv_ph.collect_bookmark_works(
            session, uid, rests, cancel_check=_cancelled
        )
        if _cancelled():
            workflow_heuristics.mark_error(
                session_id, "Download cancelled by user."
            )
            return
        if not works:
            workflow_heuristics.mark_error(
                session_id,
                "No bookmarks found (empty list, wrong user id, or login required).",
            )
            return

        logging.info("Collected %s Pixiv bookmark works", len(works))
        workflow_heuristics.set_detail(
            session_id, f"Found {len(works)} bookmarked works — downloading…",
        )

        if cfg.persist_queue_snapshots:
            try:
                urls = [f"{pixiv_ph.PIXIV_HOME}artworks/{w.get('id')}" for w in works]
                save_queue_snapshot(session_id, "pixiv", urls, USER_DATA_DIR)
            except Exception as e:
                logging.warning("Could not save queue snapshot: %s", e)

        workflow_heuristics.advance(session_id, "download")
        tracker = get_tracker(session_id)
        tracker.start_session(len(works))
        stopped = pixiv_ph.download_bookmark_works(
            session,
            works,
            download_root,
            session_id,
            max_workers=max(1, cfg.max_parallel_downloads),
            skip_if_exists=cfg.skip_existing_in_download_dir,
            cancel_check=_cancelled,
        )
        if stopped or _cancelled():
            workflow_heuristics.mark_error(
                session_id, "Download cancelled by user."
            )
            return
        ok = True

    except pixiv_ph.DownloadCancelled:
        workflow_heuristics.mark_error(session_id, "Download cancelled by user.")
    except Exception as e:
        logging.critical("Error in Pixiv workflow: %s", e)
        workflow_heuristics.mark_error(session_id, str(e))
    finally:
        workflow_browser.quit_driver(driver)
        try:
            if not cfg.persistent_cookies and os.path.exists(cookie_file):
                os.unlink(cookie_file)
        except Exception as e:
            logging.warning("Could not delete cookie file %s: %s", cookie_file, e)
        if ok:
            try:
                _append_session_history(session_id, "pixiv", get_tracker(session_id))
                cleanup_tracker(session_id)
            except Exception as e:
                logging.error("Error cleaning up tracker: %s", e)
            workflow_heuristics.complete_success(session_id)
            maybe_notify_complete("SHUCK3R", "Pixiv download session finished.", cfg)
        else:
            _record_failed_session_history(session_id, "pixiv")
        _clear_cancel_event(session_id)


def _xh_watch_page_candidate(url: str) -> bool:
    lu = url.lower()
    return "xhamster." in lu and ("/videos/" in lu or "/movies/" in lu)


def extract_video_info_sequential(
    driver,
    video_urls: List[str],
    *,
    delay_settings: Optional[AppSettings] = None,
) -> List[Tuple[str, Optional[str], str]]:
    """Extract video info sequentially to avoid driver conflicts."""
    results = []
    for i, video_url in enumerate(video_urls):
        try:
            logging.info(f"Extracting video {i + 1}/{len(video_urls)}: {video_url}")
            direct_url, video_title = extract_direct_video_url(driver, video_url)
            results.append((video_url, direct_url, video_title))

            time.sleep(0.5)
            if delay_settings:
                apply_delay(delay_settings.page_delay_seconds, delay_settings.delay_variance_seconds)

        except Exception as e:
            logging.error(f"Error extracting info for {video_url}: {e}")
            results.append((video_url, None, f"video_{int(time.time())}"))

    return results

def xhamster_workflow(
    favorites_url: str,
    download_dir: str,
    headless: bool,
    log_file: str,
    session_id: str = None,
    *,
    library_dir: Optional[str] = None,
    library_recursive: bool = False,
    cancel_event: Optional[_threading.Event] = None,
    favorite_urls_override: Optional[List[str]] = None,
    retry_pairs: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """xHamster workflow using Chrome and yt-dlp with two-phase approach."""
    if not session_id:
        session_id = str(int(time.time()))

    cfg = load_settings()
    download_root = str(site_download_directory("xhamster", cfg))

    setup_logging(log_file)
    ensure_download_dir(download_root)

    cookie_file = _workflow_cookie_file("xhamster", session_id, cfg)
    driver = None
    ok = False
    try:
        login_url = "https://xhamster.com/login"
        use_embedded = webview_login_bridge.is_active() and webview_login_bridge.embedded_login_env_enabled()
        scrape_headless = workflow_browser.scrape_headless_enabled(headless, cfg)
        if not use_embedded:
            workflow_heuristics.advance(session_id, "cookie_reuse")
        cookies_ok, cookie_msg = workflow_browser.cookies_valid_for_site(
            "xhamster", cookie_file, cfg
        )
        if not use_embedded:
            workflow_heuristics.set_detail(session_id, cookie_msg)
        if use_embedded:
            workflow_heuristics.advance(session_id, "site_login")
            webview_login_bridge.begin_embedded_login(
                login_url,
                cookie_file,
                progress_url=_desktop_session_progress_url(session_id, "xhamster"),
            )
            if not scrape_headless:
                scrape_headless = _selenium_headless_after_embedded_login()
            if scrape_headless:
                logging.info(
                    "Selenium runs headless after in-app login so Chrome does not cover SHUCK3R "
                    "(set HAMSTER_SHOW_SELENIUM=1 for a visible automation window)."
                )
            workflow_heuristics.advance(session_id, "save_cookies")
            workflow_heuristics.set_detail(session_id, None)
            if scrape_headless:
                workflow_heuristics.advance(session_id, "headless_scrape")
            workflow_heuristics.advance(session_id, "browser_start")
            driver = workflow_browser.create_driver(
                "xhamster", cfg, headless=scrape_headless, purpose="scrape"
            )
            workflow_browser.apply_flaresolverr_preflight(
                driver, "https://xhamster.com/", cfg
            )
            load_netscape_cookies_into_driver(driver, cookie_file, "https://xhamster.com/")
        elif cookies_ok:
            logging.info("xHamster cookie reuse: %s", cookie_msg)
            workflow_heuristics.set_detail(
                session_id, "Using saved session — skipping login…",
            )
            if scrape_headless:
                workflow_heuristics.advance(session_id, "headless_scrape")
            workflow_heuristics.advance(session_id, "browser_start")
            driver = workflow_browser.create_driver(
                "xhamster", cfg, headless=scrape_headless, purpose="scrape"
            )
            workflow_browser.apply_flaresolverr_preflight(
                driver, "https://xhamster.com/", cfg
            )
            load_netscape_cookies_into_driver(driver, cookie_file, "https://xhamster.com/")
        else:
            workflow_heuristics.advance(session_id, "chrome_login")
            auth_driver = workflow_browser.create_driver(
                "xhamster", cfg, headless=False, purpose="auth"
            )
            wait_for_manual_login(auth_driver, login_url, session_id)
            time.sleep(5)
            save_cookies_netscape(auth_driver, cookie_file)
            workflow_heuristics.advance(session_id, "save_cookies")
            if scrape_headless:
                workflow_heuristics.advance(session_id, "headless_scrape")
                workflow_browser.quit_driver(auth_driver)
                driver = workflow_browser.create_driver(
                    "xhamster", cfg, headless=True, purpose="scrape"
                )
                load_netscape_cookies_into_driver(
                    driver, cookie_file, "https://xhamster.com/"
                )
            else:
                driver = auth_driver
            workflow_heuristics.advance(session_id, "browser_start")

        tracker = get_tracker(session_id)
        if retry_pairs is not None:
            workflow_heuristics.advance(session_id, "collect_urls")
            workflow_heuristics.set_detail(session_id, f"Retrying {len(retry_pairs)} failed URLs…")
            workflow_heuristics.advance(session_id, "download")
            tracker.start_session(len(retry_pairs))
            workflow_heuristics.set_detail(
                session_id,
                f"{len(retry_pairs)} retries · fetching streams again",
            )
            library_norms_retry = None
            if library_dir:
                library_norms_retry = build_library_normalized_names(
                    Path(library_dir), recursive=library_recursive
                )
            batch_sz_retry = 10
            total_dn_retry = 0
            for i_retry in range(0, len(retry_pairs), batch_sz_retry):
                batch_pairs = retry_pairs[i_retry : i_retry + batch_sz_retry]
                batch_num_r = i_retry // batch_sz_retry + 1
                total_batches_r = (len(retry_pairs) + batch_sz_retry - 1) // batch_sz_retry
                logging.info("Retry batch %s/%s (%s items)", batch_num_r, total_batches_r, len(batch_pairs))
                video_rows: List[Tuple[str, Optional[str], str]] = []
                for raw_u, title_hint in batch_pairs:
                    if cancel_event and cancel_event.is_set():
                        break
                    if _xh_watch_page_candidate(raw_u):
                        du, vt = extract_direct_video_url(driver, raw_u)
                        video_rows.append((raw_u, du, vt or title_hint))
                    else:
                        video_rows.append((raw_u, raw_u, title_hint))
                    apply_delay(cfg.page_delay_seconds, cfg.delay_variance_seconds)
                vids_td: List[Tuple[str, str]] = []
                for orig_u, du2, ttl2 in video_rows:
                    if cancel_event and cancel_event.is_set():
                        break
                    if du2:
                        if library_norms_retry and matches_existing_library(ttl2, library_norms_retry):
                            tracker.add_urls([du2], [ttl2])
                            tracker.skip_download(du2, "Already in library folder")
                            logging.info("Skipping retry (library match): %s", ttl2)
                            continue
                        vids_td.append((du2, ttl2))
                        tracker.add_urls([du2], [ttl2])
                    else:
                        fb_ttl = ttl2 or f"video_{total_dn_retry + len(vids_td)}"
                        tracker.add_urls([orig_u], [fb_ttl])
                        tracker.fail_download(orig_u, "Could not extract direct video URL (retry)")
                if vids_td:
                    download_videos_parallel(
                        vids_td,
                        download_root,
                        max_workers=max(1, cfg.max_parallel_downloads),
                        cookie_file=cookie_file,
                        session_id=session_id,
                        settings_snapshot=cfg,
                    )
                    total_dn_retry += len(vids_td)
                time.sleep(2)
                apply_delay(cfg.download_delay_seconds, cfg.delay_variance_seconds)
                if cancel_event and cancel_event.is_set():
                    logging.info(
                        "Cancellation requested between xHamster retry batches — stopping."
                    )
                    workflow_heuristics.mark_error(session_id, "Download cancelled by user.")
                    break
            logging.info(
                "=== RETRY PHASE COMPLETE: processed %s items (successful downloads counted per batch)",
                len(retry_pairs),
            )

        else:
            workflow_heuristics.advance(session_id, "collect_urls")
            workflow_heuristics.set_detail(session_id, "Loading your favorites pages…")
    
            # PHASE 1: Collect all video URLs from all pages
            logging.info("=== PHASE 1: Collecting all video URLs from all pages ===")
            if favorite_urls_override is not None:
                all_video_links: Set[str] = set(dict.fromkeys(favorite_urls_override))
                logging.info(
                    "=== PHASE 1 SKIPPED: using saved/injected favourites list (%s items) ===",
                    len(all_video_links),
                )
                logging.info(
                    "=== PHASE 1 COMPLETE (injected URL list): %s unique watch-page links ===",
                    len(all_video_links),
                )
            else:
                all_video_links = set()
                page_number = 1
                consecutive_empty_pages = 0
                max_consecutive_empty = 3

                while True:
                    # Try different pagination approaches for xHamster
                    if page_number == 1:
                        current_page_url = favorites_url
                        html_content, driver = fetch_html_selenium(
                            driver,
                            current_page_url,
                            session_id=session_id,
                            site="xhamster",
                            cfg=cfg,
                        )
                    else:
                        # xHamster uses /page_number format for pagination
                        if page_number == 2:
                            current_page_url = f"{favorites_url}/2"
                        else:
                            current_page_url = f"{favorites_url}/{page_number}"

                        html_content, driver = fetch_html_selenium(
                            driver,
                            current_page_url,
                            session_id=session_id,
                            site="xhamster",
                            cfg=cfg,
                        )

                        # If URL pagination didn't work, try clicking next button
                        if not html_content:
                            logging.info(f"URL pagination didn't work for page {page_number}, trying next button...")
                            if try_click_next_button(driver):
                                html_content = driver.page_source
                                current_page_url = driver.current_url
                            else:
                                logging.info(f"No next button found, stopping pagination")
                                break

                    logging.info(f"Collecting URLs from page {page_number}: {current_page_url}")

                    if not html_content:
                        logging.info(f"No content found on page {page_number}, stopping pagination")
                        break

                    # Debug: Check if we're getting the same content
                    current_url = driver.current_url
                    logging.info(f"Current browser URL after navigation: {current_url}")

                    # Check if we've been redirected to a different page (indicating end of pagination)
                    if "favorites/videos" not in current_url or "login" in current_url or "signin" in current_url:
                        logging.warning(f"Redirected away from favorites page: {current_url} - stopping pagination")
                        break

                    # Check if we're on an error page (404, etc.) - stop pagination immediately
                    if "404" in driver.title or "error" in driver.title.lower() or "not found" in driver.title.lower():
                        logging.warning(f"Page {page_number} appears to be an error page: {driver.title} - stopping pagination")
                        break

                    video_links = parse_video_links_xhamster(html_content, "https://xhamster.com")
                    if not video_links:
                        logging.info(f"No video links found on page {page_number}, stopping pagination")
                        break

                    # Debug: Log some sample URLs to see if they're different
                    sample_urls = list(video_links)[:3] if video_links else []
                    logging.info(f"Sample video URLs from page {page_number}: {sample_urls}")

                    # Check if we're getting the same sample URLs as previous pages (indicating end of pagination)
                    if page_number > 1 and sample_urls:
                        # Store sample URLs from previous page for comparison
                        if not hasattr(try_click_next_button, 'prev_sample_urls'):
                            try_click_next_button.prev_sample_urls = []

                        if sample_urls == try_click_next_button.prev_sample_urls:
                            logging.warning(f"Page {page_number} returned identical sample URLs as previous page - likely end of pagination")
                            consecutive_empty_pages += 1

                        try_click_next_button.prev_sample_urls = sample_urls

                    new_links = set(video_links) - all_video_links
                    if not new_links:
                        consecutive_empty_pages += 1
                        logging.info(f"No new videos found on page {page_number} (consecutive empty: {consecutive_empty_pages})")

                        # Check if we're getting the exact same URLs (indicating pagination isn't working)
                        if video_links and len(video_links) == len(all_video_links) and set(video_links).issubset(all_video_links):
                            logging.warning(f"Page {page_number} returned identical content to previous pages - pagination may not be working")
                            consecutive_empty_pages += 1  # Count this as an empty page

                        if consecutive_empty_pages >= max_consecutive_empty:
                            logging.info(f"Stopping pagination after {consecutive_empty_pages} consecutive pages with no new videos")
                            break
                    else:
                        consecutive_empty_pages = 0  # Reset counter when we find new videos
                        logging.info(f"Found {len(new_links)} new videos on page {page_number}")

                    all_video_links.update(video_links)  # Update with all links from this page
                    workflow_heuristics.set_detail(
                        session_id,
                        f"Favorites page {page_number} · {len(all_video_links)} links collected so far",
                    )
                    page_number += 1

                    if cancel_event and cancel_event.is_set():
                        logging.info("Cancellation requested during page collection - stopping.")
                        workflow_heuristics.mark_error(session_id, "Download cancelled by user.")
                        return

                    # Safety limit to prevent infinite loops
                    if page_number > 100:
                        logging.warning("Reached maximum page limit (100), stopping pagination")
                        break

                    apply_delay(cfg.page_delay_seconds, cfg.delay_variance_seconds)

                logging.info(
                    "=== PHASE 1 COMPLETE: Collected %s unique video URLs from %s favourites pages ===",
                    len(all_video_links),
                    max(0, page_number - 1),
                )

            if cfg.persist_queue_snapshots:
                try:
                    save_queue_snapshot(session_id, "xhamster", list(all_video_links), USER_DATA_DIR)
                except Exception as e:
                    logging.warning("Could not save queue snapshot: %s", e)
            # PHASE 2: Download all videos
            logging.info("=== PHASE 2: Downloading all videos ===")
            all_video_links_list = list(all_video_links)
            tracker.start_session(len(all_video_links_list))
            workflow_heuristics.advance(session_id, "download")
    
            library_norms = None
            if library_dir:
                library_norms = build_library_normalized_names(
                    Path(library_dir), recursive=library_recursive
                )
                logging.info(
                    "Existing-library filter active: %s normalized video names under %s",
                    len(library_norms),
                    library_dir,
                )
                workflow_heuristics.set_detail(
                    session_id,
                    f"{len(all_video_links_list)} favorites · skipping names already in your library folder",
                )
            else:
                workflow_heuristics.set_detail(
                    session_id,
                    f"{len(all_video_links_list)} videos queued · processing in batches",
                )
    
            batch_size = 10
            total_downloaded = 0
    
            for i in range(0, len(all_video_links_list), batch_size):
                batch = all_video_links_list[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(all_video_links_list) + batch_size - 1) // batch_size
    
                logging.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} videos)")
                workflow_heuristics.set_detail(
                    session_id,
                    f"Batch {batch_num} of {total_batches} · resolving streams and downloading",
                )
                
                # Extract video info sequentially (to avoid driver conflicts)
                video_info_results = extract_video_info_sequential(
                    driver, batch, delay_settings=cfg
                )
                
                videos_to_download = []
                for original_url, direct_url, video_title in video_info_results:
                    if direct_url:
                        if library_norms and matches_existing_library(
                            video_title, library_norms
                        ):
                            tracker.add_urls([direct_url], [video_title])
                            tracker.skip_download(
                                direct_url, "Already in library folder"
                            )
                            logging.info(
                                "Skipping download (library match): %s", video_title
                            )
                            continue
                        videos_to_download.append((direct_url, video_title))
                        tracker.add_urls([direct_url], [video_title])
                    else:
                        fallback_title = f"video_{total_downloaded + len(videos_to_download)}"
                        tracker.add_urls([original_url], [fallback_title])
                        tracker.fail_download(original_url, "Could not extract direct video URL")
                
                # Download videos in parallel
                if videos_to_download:
                    logging.info(f"Downloading {len(videos_to_download)} videos from batch {batch_num}...")
                    download_videos_parallel(
                        videos_to_download,
                        download_root,
                        max_workers=max(1, cfg.max_parallel_downloads),
                        cookie_file=cookie_file,
                        session_id=session_id,
                        settings_snapshot=cfg,
                    )
                    total_downloaded += len(videos_to_download)
                
                # Small delay between batches
                time.sleep(2)
                apply_delay(cfg.download_delay_seconds, cfg.delay_variance_seconds)

                if cancel_event and cancel_event.is_set():
                    logging.info("Cancellation requested between xHamster batches \u2014 stopping.")
                    workflow_heuristics.mark_error(session_id, "Download cancelled by user.")
                    break
            
            logging.info(f"=== PHASE 2 COMPLETE: Downloaded {total_downloaded} videos ===")
            logging.info(f"xHamster workflow completed. Total videos processed: {total_downloaded}")
    
            workflow_heuristics.set_detail(session_id, None)
        ok = True

    except Exception as e:
        logging.critical(f"Error in xHamster workflow: {e}")
        workflow_heuristics.mark_error(session_id, str(e))
    finally:
        workflow_browser.quit_driver(driver)
        try:
            if not cfg.persistent_cookies and os.path.exists(cookie_file):
                os.unlink(cookie_file)
        except Exception as e:
            logging.warning(f"Could not delete cookie file {cookie_file}: {e}")
        if ok:
            try:
                _append_session_history(session_id, "xhamster", get_tracker(session_id))
                cleanup_tracker(session_id)
            except Exception as e:
                logging.error(f"Error cleaning up tracker: {e}")
            workflow_heuristics.complete_success(session_id)
            maybe_notify_complete("SHUCK3R", "xHamster download session finished.", cfg)
        else:
            _record_failed_session_history(session_id, "xhamster")
        _clear_cancel_event(session_id)

# ----------------------------- FastAPI Routes ----------------------------- #


@app.get("/api/build-info", include_in_schema=False)
def api_build_info():
    """Quick check that the running server loaded the current browser automation code."""
    return {
        "chrome_login_wait_version": chrome_login_confirm.CHROME_LOGIN_WAIT_VERSION,
        "browser_antibot_version": browser_antibot.BROWSER_ANTIBOT_VERSION,
        "challenge_wait_version": browser_challenge_wait.CHALLENGE_WAIT_VERSION,
        "undetected_chromedriver_available": browser_antibot.undetected_chromedriver_available(),
        "chrome_login_confirm_module": getattr(chrome_login_confirm, "__file__", None),
        "browser_antibot_module": getattr(browser_antibot, "__file__", None),
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico():
    """Browsers request /favicon.ico by default; serve SVG so logs stay clean."""
    icon = RESOURCE_ROOT / "static" / "favicon.svg"
    if icon.is_file():
        return FileResponse(str(icon), media_type="image/svg+xml")
    return Response(status_code=204)


@app.get("/")
def read_form(request: Request):
    """Display the main form."""
    return templates.TemplateResponse(request, "index.html", {"form_error": None})


@app.get("/settings")
def settings_page(request: Request, saved: int = Query(0)):
    cfg = load_settings()
    ctx = {
        "settings": asdict(cfg),
        "effective_download_dir": str(effective_download_directory(cfg)),
        "site_download_dirs": {
            "PornHub": str(site_download_directory("pornhub", cfg)),
            "xHamster": str(site_download_directory("xhamster", cfg)),
            "Pixiv": str(site_download_directory("pixiv", cfg)),
        },
        "saved": bool(saved),
        "queue_snapshots_hint": USER_DATA_DIR / "queue_snapshots",
    }
    return templates.TemplateResponse(request, "settings.html", ctx)


@app.post("/settings")
async def settings_save(request: Request):
    fd = await request.form()

    def _combo_on(key: str) -> bool:
        return "on" in {str(v).strip().lower() for v in fd.getlist(key)}

    def _pf(key: str, default: float = 0.0) -> float:
        raw = fd.get(key)
        if raw is None or str(raw).strip() == "":
            return default
        try:
            return float(str(raw).strip())
        except Exception:
            return default

    def _pi(key: str, default: int = 0) -> int:
        raw = fd.get(key)
        if raw is None or str(raw).strip() == "":
            return default
        try:
            return int(str(raw).strip())
        except Exception:
            return default

    cfg = load_settings()

    dq = fd.get("video_quality")
    if dq is None:
        vq = cfg.video_quality
    else:
        vq = str(dq).strip().lower()
    if vq not in ("best", "720", "1080"):
        vq = "720"

    cfg.download_directory = str(fd.get("download_directory") or "").strip()
    cfg.max_parallel_downloads = max(1, min(32, _pi("max_parallel_downloads", cfg.max_parallel_downloads)))
    cfg.video_quality = vq  # type: ignore[assignment]
    cfg.skip_existing_in_download_dir = _combo_on("skip_existing_in_download_dir")
    cfg.persistent_cookies = _combo_on("persistent_cookies")
    cfg.page_delay_seconds = max(0.0, _pf("page_delay_seconds", cfg.page_delay_seconds))
    cfg.download_delay_seconds = max(0.0, _pf("download_delay_seconds", cfg.download_delay_seconds))
    cfg.delay_variance_seconds = max(0.0, _pf("delay_variance_seconds", cfg.delay_variance_seconds))
    cfg.proxy_url = str(fd.get("proxy_url") or "").strip()
    cfg.yt_dlp_retries = max(0, min(50, _pi("yt_dlp_retries", cfg.yt_dlp_retries)))
    cfg.persist_queue_snapshots = _combo_on("persist_queue_snapshots")
    cfg.watcher_enabled = _combo_on("watcher_enabled")
    cfg.watcher_interval_minutes = max(1, min(10_080, _pi("watcher_interval_minutes", cfg.watcher_interval_minutes)))
    cfg.notify_on_complete = _combo_on("notify_on_complete")
    cfg.headless_scraping = _combo_on("headless_scraping")
    cfg.browser_profile_per_site = _combo_on("browser_profile_per_site")
    cfg.skip_login_if_cookies_valid = _combo_on("skip_login_if_cookies_valid")
    cfg.use_undetected_chrome = _combo_on("use_undetected_chrome")
    solver = str(fd.get("challenge_solver") or cfg.challenge_solver).strip().lower()
    cfg.challenge_solver = solver if solver in ("manual", "flaresolverr") else "manual"
    cfg.flaresolverr_base_url = str(
        fd.get("flaresolverr_base_url") or cfg.flaresolverr_base_url
    ).strip() or "http://127.0.0.1:8191/v1"

    save_settings(cfg)
    return RedirectResponse("/settings?saved=1", status_code=303)


@app.post("/download")
def handle_form(
    request: Request,
    background_tasks: BackgroundTasks,
    site: str = Form(...),
    headless: str = Form("false"),
    existing_library_dir: str = Form(""),
    library_recursive: str = Form("false"),
    pixiv_user_id: str = Form(""),
    pixiv_bookmarks: str = Form("public"),
):
    """Handle form submission and start the appropriate workflow."""
    site = (site or "").strip().lower()
    timestamp = int(time.time())
    session_id = str(uuid.uuid4())
    log_file = os.path.join(LOG_DIR, f"video_downloader_{session_id}.log")

    headless_bool = headless.lower() in ("true", "on", "1", "yes")
    library_recursive_bool = library_recursive.lower() == "true"

    try:
        library_path = resolve_optional_library_directory(
            existing_library_dir, USER_DATA_DIR
        )
    except ValueError as e:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"form_error": str(e)},
            status_code=400,
        )

    library_dir_arg = str(library_path) if library_path else None

    if site == "pornhub":
        playlist_url = "https://www.pornhub.com/my/favorites/videos"
        cancel_ev = _get_cancel_event(session_id)
        background_tasks.add_task(
            pornhub_workflow,
            playlist_url,
            DOWNLOAD_DIR,
            headless_bool,
            log_file,
            session_id,
            library_dir=library_dir_arg,
            library_recursive=library_recursive_bool,
            cancel_event=cancel_ev,
        )
    elif site == "xhamster":
        favorites_url = "https://xhamster.com/my/favorites/videos"
        cancel_ev = _get_cancel_event(session_id)
        background_tasks.add_task(
            xhamster_workflow,
            favorites_url,
            DOWNLOAD_DIR,
            headless_bool,
            log_file,
            session_id,
            library_dir=library_dir_arg,
            library_recursive=library_recursive_bool,
            cancel_event=cancel_ev,
        )
    elif site == "pixiv":
        cancel_ev = _get_cancel_event(session_id)
        background_tasks.add_task(
            pixiv_workflow,
            pixiv_user_id,
            pixiv_bookmarks,
            DOWNLOAD_DIR,
            headless_bool,
            log_file,
            session_id,
            cancel_event=cancel_ev,
        )
    else:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "form_error": (
                    f'Unknown site "{site}". Choose PornHub, xHamster, or Pixiv. '
                    "If Pixiv is missing from the form, restart the app from the latest code."
                ),
            },
            status_code=400,
        )

    embedded_login = (
        site != "pixiv"
        and webview_login_bridge.is_active()
        and webview_login_bridge.embedded_login_env_enabled()
    )
    workflow_heuristics.register_session(session_id, site, embedded=embedded_login)
    session_history_register(session_id, site, timestamp=timestamp)
    dest = f"/download/session/{session_id}?timestamp={timestamp}&site={site}"
    return RedirectResponse(url=dest, status_code=303)


@app.get("/download/session/{session_id}")
def download_session_progress(
    request: Request,
    session_id: str,
    timestamp: int = Query(...),
    site: str = Query(...),
):
    """Bookmarkable progress page (GET) so the desktop WebView keeps the download UI after login."""
    if site not in ("pornhub", "xhamster", "pixiv"):
        raise HTTPException(status_code=400, detail="Invalid site")

    embedded_login = webview_login_bridge.is_active() and webview_login_bridge.embedded_login_env_enabled()
    return templates.TemplateResponse(
        request,
        "submitted.html",
        context={
            "timestamp": timestamp,
            "session_id": session_id,
            "site": site,
            "embedded_login": embedded_login,
        },
    )


@app.post("/api/embedded-login/confirm")
def api_embedded_login_confirm():
    """Same as menu 'Done logging in' — on-page button for users who do not see the native menu bar."""
    if not webview_login_bridge.is_active() or not webview_login_bridge.embedded_login_env_enabled():
        raise HTTPException(
            status_code=400,
            detail="Embedded login is not active here. Use the SHUCK3R desktop window.",
        )
    if not webview_login_bridge.has_pending_login():
        raise HTTPException(
            status_code=400,
            detail="Nothing is waiting for login confirmation. If you already continued, watch progress below.",
        )
    webview_login_bridge.finish_embedded_login(skip_native_confirm=True)
    return {"ok": True}


@app.post("/api/selenium-login/confirm")
def api_selenium_login_confirm(body: SeleniumLoginConfirmBody):
    """Unblocks legacy Chrome login when there is no interactive terminal (e.g. `python main.py`)."""
    sid = body.session_id.strip()
    if chrome_login_confirm.confirm_chrome_login(sid):
        return {"ok": True, "chrome_login_wait_version": chrome_login_confirm.CHROME_LOGIN_WAIT_VERSION}
    raise HTTPException(
        status_code=400,
        detail=(
            "No Chrome login step is waiting for that session. Open the matching download tab, "
            "wait until Chrome has appeared, sign in, then try again — or the job may have already continued."
        ),
    )


@app.post("/api/browser-challenge/confirm")
def api_browser_challenge_confirm(body: BrowserChallengeConfirmBody):
    """Unblocks scrape after user solves Cloudflare/CAPTCHA in visible Chrome."""
    sid = body.session_id.strip()
    if browser_challenge_wait.confirm_challenge_cleared(sid):
        return {"ok": True, "challenge_wait_version": browser_challenge_wait.CHALLENGE_WAIT_VERSION}
    raise HTTPException(
        status_code=400,
        detail=(
            "No browser challenge step is waiting for that session. "
            "Complete the check in Chrome, then try again."
        ),
    )


@app.get("/download/log/{session_id}")
def get_log(session_id: str):
    """Download log file."""
    log_file = os.path.join(LOG_DIR, f"video_downloader_{session_id}.log")
    if os.path.exists(log_file):
        return FileResponse(path=log_file, filename=os.path.basename(log_file), media_type='text/plain')
    else:
        return {"error": "Log file not found."}


def _read_log_tail_text(log_path: Path, *, lines: int, max_bytes: int = 56_000) -> str:
    """Return the last `lines` lines without loading huge files whole."""
    try:
        raw = log_path.read_bytes()
    except OSError:
        return ""
    if len(raw) <= max_bytes:
        text = raw.decode("utf-8", errors="replace")
    else:
        chunk = raw[-max_bytes:]
        nl = chunk.find(b"\n")
        if nl != -1:
            chunk = chunk[nl + 1 :]
        text = chunk.decode("utf-8", errors="replace")
    all_lines = text.splitlines()
    if len(all_lines) <= lines:
        return "\n".join(all_lines)
    return "\n".join(all_lines[-lines:])


@app.get("/download/log/{session_id}/tail", response_class=PlainTextResponse)
def get_log_tail(session_id: str, lines: int = Query(48, ge=8, le=200)):
    """Last lines of the session log for the live dashboard (plain text)."""
    log_file = Path(LOG_DIR) / f"video_downloader_{session_id}.log"
    if not log_file.is_file():
        raise HTTPException(status_code=404, detail="Log file not found.")
    body = _read_log_tail_text(log_file, lines=lines)
    return body or "(log is empty so far)"

@app.get("/downloaded/{file_path:path}")
def get_downloaded_file(file_path: str):
    """Serve a downloaded video file (supports nested paths under the downloads folder)."""
    resolved = _resolve_under_downloads(file_path)
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    media_type, _ = mimetypes.guess_type(str(resolved))
    return FileResponse(
        path=str(resolved),
        filename=resolved.name,
        media_type=media_type or "video/mp4",
    )

@app.get("/progress/{session_id}")
def get_progress(session_id: str):
    """JSON snapshot of progress (machine-readable). Prefer /api/progress for new integrations."""
    try:
        tracker = get_tracker(session_id)
        return tracker.get_progress()
    except Exception as e:
        return {"error": f"Session not found: {e}"}


@app.get("/api/progress/{session_id}")
def api_get_progress(session_id: str):
    """Same payload as /progress/{session_id}; use this path in docs and UI so users are not sent here by mistake."""
    try:
        tracker = get_tracker(session_id)
        return tracker.get_progress()
    except Exception as e:
        return {"error": f"Session not found: {e}"}

@app.get("/progress/{session_id}/summary")
def get_progress_summary(session_id: str):
    """Get progress summary for a download session."""
    summary: Dict[str, Any]
    try:
        tracker = get_tracker(session_id)
        summary = tracker.get_summary()
    except Exception as e:
        summary = {"error": f"Session not found: {e}"}
    wf = workflow_heuristics.get_timeline(session_id)
    if wf is not None:
        summary["workflow"] = wf
    return summary

@app.get("/progress/{session_id}/json")
def get_progress_json(session_id: str):
    """Get progress as JSON file."""
    progress_file = os.path.join(LOG_DIR, f"progress_{session_id}.json")
    if os.path.exists(progress_file):
        return FileResponse(path=progress_file, filename=f"progress_{session_id}.json", media_type='application/json')
    else:
        return {"error": "Progress file not found."}


@app.get("/duplicates")
def duplicates_page(request: Request):
    """Duplicate file checker UI — scan any directory; empty field uses project downloads."""
    return templates.TemplateResponse(
        request,
        "duplicates.html",
        context={
            "default_scan_directory": str(_active_download_root()),
            # Alias for older cached templates / mixed deploys
            "allowed_downloads_root": str(_active_download_root()),
        },
    )


@app.post("/api/duplicates/scan")
def api_duplicates_scan(body: DuplicateScanBody):
    try:
        root = resolve_scan_directory(body.directory or "", USER_DATA_DIR, "downloads")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    files = iter_files(root, body.recursive)
    groups = cluster_duplicates(files, threshold=body.threshold)
    payload = build_groups_payload(files, groups)
    return {
        "groups": payload,
        "scanned_root": str(root),
        "file_count": len(files),
        "group_count": len(payload),
    }


@app.get("/api/duplicates/preview-file")
def api_duplicates_preview_file(path: str = Query(..., description="Absolute path to a video file")):
    """Stream a video file for HTML5 preview thumbnails (validated path, video extensions only)."""
    try:
        fp = resolve_deletable_file(path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    ext = fp.suffix.lower()
    if ext not in DUPLICATE_PREVIEW_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Preview is only available for common video file types.",
        )
    media_type, _ = mimetypes.guess_type(str(fp))
    return FileResponse(path=str(fp), media_type=media_type or "video/mp4")


@app.post("/api/duplicates/delete")
def api_duplicates_delete(body: DuplicateDeleteBody):
    deleted: List[str] = []
    errors: List[dict] = []
    for path_str in body.paths:
        try:
            fp = resolve_deletable_file(path_str)
            fp.unlink()
            deleted.append(str(fp))
        except ValueError as e:
            errors.append({"path": path_str, "error": str(e)})
        except OSError as e:
            errors.append({"path": path_str, "error": str(e)})
    return {"deleted": deleted, "errors": errors}


# ─────────────────── Feature 2: Session URL export & retry ───────────────────

@app.post("/api/session/{session_id}/cancel")
def api_session_cancel(session_id: str):
    """Signal the running workflow for this session to stop after the current item."""
    with _cancel_lock:
        ev = _cancel_events.get(session_id)
    if ev is None:
        # Register so a workflow that has not started yet still sees cancel.
        ev = _get_cancel_event(session_id)
    if chrome_login_confirm.is_waiting(session_id):
        logging.info(
            "Cancel during Pixiv Chrome login wait (session %s)", session_id
        )
    if ev.is_set():
        return {"ok": True, "message": "Cancel already requested."}
    ev.set()
    logging.info("Cancel requested for session %s", session_id)
    workflow_heuristics.mark_error(session_id, "Download cancelled by user.")
    site = (_load_session_registry().get(session_id) or {}).get("site")
    if not site and _session_meta_path(session_id).is_file():
        try:
            site = json.loads(
                _session_meta_path(session_id).read_text(encoding="utf-8")
            ).get("site")
        except Exception:
            site = None
    if site:
        session_history_update(
            session_id,
            site,
            status="cancelled",
            error_message="Download cancelled by user.",
        )
    return {"ok": True, "message": "Cancel signal sent. The workflow will stop after the current item."}


@app.get("/api/session/{session_id}/failed")
def api_session_failed(session_id: str):
    """Return all failed download items for a session (reads the persisted progress JSON)."""
    data = _read_progress_json(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    failed = [
        {"url": url, "title": item.get("title", ""), "error": item.get("error_message", "")}
        for url, item in data.get("download_items", {}).items()
        if item.get("status") == "failed"
    ]
    return {"session_id": session_id, "failed": failed, "count": len(failed)}


@app.post("/api/session/{session_id}/retry-failed")
def api_session_retry_failed(
    session_id: str,
    background_tasks: BackgroundTasks,
    site: str = Query(..., description='"pornhub" or "xhamster"'),
    headless: bool = Query(False),
):
    """Start a new download session that retries only the failed URLs from a previous session."""
    site_n = site.lower().strip()
    if site_n not in ("pornhub", "xhamster", "pixiv"):
        raise HTTPException(
            status_code=400,
            detail='site must be "pornhub", "xhamster", or "pixiv".',
        )

    data = _read_progress_json(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Source session not found.")
    failed_items = [
        (url, str(item.get("title", "") or ""))
        for url, item in data.get("download_items", {}).items()
        if item.get("status") == "failed"
    ]
    if not failed_items:
        return {"ok": True, "message": "No failed downloads to retry.", "retry_session_id": None}
    new_session_id = str(uuid.uuid4())
    log_file = os.path.join(LOG_DIR, f"video_downloader_{new_session_id}.log")
    timestamp = int(time.time())
    cancel_ev = _get_cancel_event(new_session_id)
    dl_arg = DOWNLOAD_DIR

    if site_n == "pornhub":
        background_tasks.add_task(
            pornhub_workflow,
            "https://www.pornhub.com/my/favorites/videos",
            dl_arg,
            headless,
            log_file,
            new_session_id,
            url_override=[pair[0] for pair in failed_items],
            cancel_event=cancel_ev,
        )
    else:
        background_tasks.add_task(
            xhamster_workflow,
            "https://xhamster.com/my/favorites/videos",
            dl_arg,
            headless,
            log_file,
            new_session_id,
            retry_pairs=failed_items,
            cancel_event=cancel_ev,
        )

    workflow_heuristics.register_session(new_session_id, site_n, embedded=False)
    session_history_register(new_session_id, site_n, timestamp=timestamp)
    return {
        "ok": True,
        "retry_session_id": new_session_id,
        "url_count": len(failed_items),
        "progress_url": f"/download/session/{new_session_id}?timestamp={timestamp}&site={site_n}",
    }


@app.post("/api/queue/resume/{snapshot_session_id}")
def api_queue_resume(snapshot_session_id: str, background_tasks: BackgroundTasks, headless: bool = Query(False)):
    """Start a fresh session using a saved favourites URL list from a prior run (crash recovery)."""
    snap = load_queue_snapshot(snapshot_session_id, USER_DATA_DIR)
    if not snap:
        raise HTTPException(status_code=404, detail="No queue snapshot for that session.")

    site = str(snap.get("site") or "").lower().strip()
    urls = snap.get("urls") or []
    if site not in ("pornhub", "xhamster", "pixiv") or not isinstance(urls, list) or not urls:
        raise HTTPException(status_code=400, detail="Snapshot is incomplete or malformed.")

    new_session_id = str(uuid.uuid4())
    log_file = os.path.join(LOG_DIR, f"video_downloader_{new_session_id}.log")
    timestamp = int(time.time())
    cancel_ev = _get_cancel_event(new_session_id)
    dl_arg = DOWNLOAD_DIR

    if site == "pornhub":
        background_tasks.add_task(
            pornhub_workflow,
            "https://www.pornhub.com/my/favorites/videos",
            dl_arg,
            headless,
            log_file,
            new_session_id,
            url_override=[str(u) for u in urls],
            cancel_event=cancel_ev,
        )
    elif site == "xhamster":
        background_tasks.add_task(
            xhamster_workflow,
            "https://xhamster.com/my/favorites/videos",
            dl_arg,
            headless,
            log_file,
            new_session_id,
            favorite_urls_override=[str(u) for u in urls],
            cancel_event=cancel_ev,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail="Queue resume for Pixiv is not supported yet; start a new download.",
        )

    workflow_heuristics.register_session(new_session_id, site, embedded=False)
    session_history_register(new_session_id, site, timestamp=timestamp)
    return {
        "ok": True,
        "resume_session_id": new_session_id,
        "url_count": len(urls),
        "progress_url": f"/download/session/{new_session_id}?timestamp={timestamp}&site={site}",
    }


# ─────────────────── Feature 3: URL export ───────────────────

@app.get("/api/session/{session_id}/urls.txt", response_class=PlainTextResponse)
def api_session_urls_txt(session_id: str):
    """All tracked URLs for a session, one per line."""
    data = _read_progress_json(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return "\n".join(data.get("download_items", {}).keys())


@app.get("/api/session/{session_id}/urls.json")
def api_session_urls_json(session_id: str):
    """All tracked items (url, title, status) for a session."""
    data = _read_progress_json(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    items = [
        {"url": url, "title": item.get("title", ""), "status": item.get("status", "")}
        for url, item in data.get("download_items", {}).items()
    ]
    return {"session_id": session_id, "items": items, "count": len(items)}


# ─────────────────── Feature 4: Integrity scanner ───────────────────

class IntegrityScanBody(BaseModel):
    directory: Optional[str] = None
    recursive: bool = False


@app.get("/integrity")
def integrity_page(request: Request):
    """File integrity scanner — flags truncated or corrupt video files."""
    return templates.TemplateResponse(
        request, "integrity.html",
        {"default_scan_directory": str(_active_download_root())},
    )


@app.post("/api/integrity/scan")
def api_integrity_scan(body: IntegrityScanBody):
    try:
        root = resolve_scan_directory(body.directory or "", USER_DATA_DIR, "downloads")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    video_exts = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v", ".wmv", ".flv", ".mpeg", ".mpg"}
    file_records = iter_files(root, body.recursive)
    results = []
    for rec in file_records:
        path, name, size = rec["path"], rec["name"], rec["size"]
        if Path(name).suffix.lower() not in video_exts:
            continue
        if size == 0:
            results.append({"path": path, "name": name, "size": 0, "ok": False, "detail": "Empty file (0 bytes)"})
            continue
        ok, detail = _run_ffprobe(path)
        results.append({"path": path, "name": name, "size": size, "ok": ok, "detail": detail})
    corrupt = [r for r in results if not r["ok"]]
    return {
        "scanned_root": str(root),
        "file_count": len(results),
        "corrupt_count": len(corrupt),
        "results": results,
    }


# ─────────────────── Feature 5: Thumbnail library browser ───────────────────

@app.get("/library")
def library_page(request: Request):
    """Thumbnail grid browser for all downloaded video files (including subfolders)."""
    dl_root = _active_download_root()
    videos: List[Dict[str, Any]] = []
    if dl_root.is_dir():
        for f in sorted(dl_root.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in _LIBRARY_VIDEO_EXTS:
                continue
            rel = f.relative_to(dl_root).as_posix()
            videos.append({
                "rel_path": rel,
                "filename": f.name,
                "size": f.stat().st_size,
            })
    return templates.TemplateResponse(
        request,
        "library.html",
        {"videos": videos, "downloads_dir": str(_active_download_root())},
    )


@app.get("/thumbs/{file_path:path}")
def get_thumb(file_path: str):
    """Serve a thumbnail for a video file, generating it on demand via ffmpeg."""
    video_path = _resolve_under_downloads(file_path)
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found.")
    thumb_path = THUMBS_DIR / f"{_thumb_cache_name(file_path)}.jpg"
    if not thumb_path.is_file():
        _generate_thumbnail(str(video_path), str(thumb_path))
    if thumb_path.is_file():
        return FileResponse(str(thumb_path), media_type="image/jpeg")
    raise HTTPException(
        status_code=404,
        detail="Thumbnail unavailable — ffmpeg not installed or extraction failed.",
    )


# ─────────────────── Feature 6: Session history ───────────────────

@app.get("/history")
def history_page(request: Request):
    """Persistent log of all completed download runs."""
    return templates.TemplateResponse(request, "history.html", {})


@app.get("/api/history")
def api_history(limit: int = Query(100, ge=1, le=1000)):
    """Return recent sessions: running, failed, cancelled, and completed (most recent first)."""
    try:
        merged = _history_from_progress_files(_load_session_registry())
        if SESSION_HISTORY_FILE.is_file():
            for ln in SESSION_HISTORY_FILE.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                sid = rec.get("session_id")
                if sid and sid not in merged:
                    rec.setdefault("status", "completed")
                    rec.setdefault("updated_ts", rec.get("ts"))
                    merged[sid] = rec
        records = list(merged.values())
        records.sort(
            key=lambda r: r.get("updated_ts") or r.get("ts") or "",
            reverse=True,
        )
        return records[:limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/paths")
def api_paths():
    """Effective download and data directories for the running app."""
    cfg = load_settings()
    base = effective_download_directory(cfg)
    return {
        "user_data_dir": str(USER_DATA_DIR),
        "download_directory": str(base),
        "site_download_directories": {
            site: str(site_download_directory(site, cfg))
            for site in ("pornhub", "xhamster", "pixiv")
        },
        "logs_directory": str(LOG_DIR),
        "session_history_file": str(SESSION_HISTORY_FILE),
    }


if __name__ == "__main__":
    try:
        import sys

        if getattr(sys, "frozen", False):
            from paths import ensure_runtime_cwd

            ensure_runtime_cwd()

        if not getattr(sys, "frozen", False):
            from check_requirements import check_and_update_requirements

            req_path = os.path.join(str(RESOURCE_ROOT), "requirements.txt")
            if os.path.isfile(req_path) and not check_and_update_requirements(req_path):
                print("Warning: some dependencies could not be updated. Continuing anyway.")

        import uvicorn

        host = os.environ.get("HAMSTER_HOST", "0.0.0.0")
        port = int(os.environ.get("HAMSTER_PORT", "8001"))
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        raise
