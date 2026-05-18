"""
PornHub favourites + download logic aligned with archive/ph.py.

- Favourites: Selenium scroll + #moreDataBtn (when a driver exists), or HTTP ?page=N
  pagination with a Netscape cookie jar (embedded login).
- Per video: fetch view_video page HTML, read flashvars_*.mediaDefinitions, prefer get_media
  MP4 then HLS; download with requests or yt-dlp (same strategy as ph.py).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
import concurrent.futures
from http.cookiejar import MozillaCookieJar
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

logger = logging.getLogger(__name__)

PH_BASE = "https://www.pornhub.com"
# ph.py uses /users/<username>/videos/favorites (not /my/favorites/videos).
PH_PLAYLIST_URL_PLACEHOLDER = f"{PH_BASE}/my/favorites/videos"


def favorites_url_for_username(username: str) -> str:
    """Canonical favourites URL (matches archive/ph.py and website-code fixture)."""
    uname = re.sub(r"^/|/$", "", (username or "").strip())
    if not uname:
        return PH_PLAYLIST_URL_PLACEHOLDER
    return f"{PH_BASE}/users/{uname}/videos/favorites"


def _username_from_profile_html(html: str) -> str:
    """Logged-in profile slug from favourites or header markup."""
    if not html:
        return ""
    for pattern in (
        r'href="/users/([^"/]+)/videos/favorites"',
        r'class="username"\s+href="/users/([^"/]+)"',
        r'href="/users/([^"/]+)"[^>]*class="username"',
        r'topUserMenu[^>]+href="/users/([^"/]+)/videos/favorites"',
    ):
        m = re.search(pattern, html, re.I)
        if m:
            return m.group(1).strip()
    return ""


def _username_from_user_info_payload(data) -> str:
    if not isinstance(data, dict):
        return ""
    for key in ("username", "user_name", "profile_username", "name", "display_name"):
        val = data.get(key)
        if isinstance(val, str) and re.fullmatch(r"[\w.-]+", val.strip()):
            return val.strip()
    user = data.get("user")
    if isinstance(user, dict):
        found = _username_from_user_info_payload(user)
        if found:
            return found
    for val in data.values():
        if isinstance(val, dict):
            found = _username_from_user_info_payload(val)
            if found:
                return found
    return ""


def parse_video_links_pornhub(html: str, base_url: str = PH_BASE) -> List[str]:
    """Collect view_video.php?viewkey=... URLs from favourites or listing HTML."""
    soup = BeautifulSoup(html, "html.parser")
    video_links: List[str] = []
    seen: set[str] = set()
    for anchor in soup.select('a[href*="view_video"], a[href*="viewkey="]'):
        href = anchor.get("href")
        if not href or "viewkey=" not in href:
            continue
        video_url = urljoin(base_url, href)
        if video_url in seen:
            continue
        seen.add(video_url)
        video_links.append(video_url)
    return video_links


DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _cookie_ua_path(cookie_file: str) -> str:
    return f"{cookie_file}.ua"


def _session_user_agent(cookie_file: str) -> str:
    ua_path = _cookie_ua_path(cookie_file)
    if os.path.isfile(ua_path):
        try:
            ua = open(ua_path, encoding="utf-8").read().strip()
            if ua:
                return ua
        except OSError:
            pass
    return DEFAULT_UA


def save_user_agent_sidecar(cookie_file: str, user_agent: str) -> None:
    ua = (user_agent or "").strip()
    if ua:
        try:
            with open(_cookie_ua_path(cookie_file), "w", encoding="utf-8") as fh:
                fh.write(ua)
        except OSError as e:
            logger.warning("Could not save PornHub UA sidecar: %s", e)


def _session_with_cookies(cookie_file: str, proxy_url: str = "") -> requests.Session:
    session = requests.Session()
    jar = MozillaCookieJar(cookie_file)
    jar.load(ignore_discard=True, ignore_expires=True)
    session.cookies = jar
    pv = (proxy_url or "").strip()
    if pv:
        session.proxies.update({"http": pv, "https": pv})
    session.headers.update(
        {
            "User-Agent": _session_user_agent(cookie_file),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"{PH_BASE}/",
        }
    )
    return session


def favorites_url_with_page(playlist_url: str, page: int) -> str:
    """Merge or set ?page=N on the favourites URL."""
    parsed = urlparse(playlist_url)
    q = parse_qs(parsed.query, keep_blank_values=True)
    for k in list(q.keys()):
        q[k] = q[k][0] if q[k] else ""
    q["page"] = str(page)
    new_query = urlencode({k: v for k, v in q.items() if v is not None})
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment)
    )


def collect_favorites_urls_requests(
    cookie_file: str,
    playlist_url: str,
    *,
    max_pages: int = 500,
    request_timeout: float = 30.0,
    proxy_url: str = "",
) -> List[str]:
    """
    Collect favourite video page URLs using authenticated GETs and ?page=1,2,...
    Used when there is no Selenium driver (e.g. embedded WebView login).
    """
    session = _session_with_cookies(cookie_file, proxy_url)
    all_urls: set[str] = set()
    q0 = urlparse(playlist_url).query.lower()
    for page in range(1, max_pages + 1):
        if page == 1 and "page=" not in q0:
            url = playlist_url
        else:
            url = favorites_url_with_page(playlist_url, page)
        try:
            r = session.get(url, timeout=request_timeout)
            r.raise_for_status()
        except Exception as e:
            logger.warning("Favourites page %s failed: %s", page, e)
            break
        # PornHub returns HTTP 200 even for login redirects — detect and bail out.
        final_url = (getattr(r, 'url', '') or '').lower()
        if 'login' in final_url or 'name="username"' in r.text:
            logger.error(
                "PornHub session expired — got login page on page %s (URL: %s)", page, r.url
            )
            break
        new_links = parse_video_links_pornhub(r.text, PH_BASE)
        before = len(all_urls)
        all_urls.update(new_links)
        added = len(all_urls) - before
        logger.info("PornHub favourites page %s: +%s URLs (total %s)", page, added, len(all_urls))
        if added == 0:
            break
    result = list(all_urls)
    logger.info("Collected %s PornHub video URLs via HTTP pagination.", len(result))
    return result


def scroll_to_bottom_pornhub(driver, pause_time: float = 2.0, max_scrolls: int = 25) -> None:
    try:
        last_height = driver.execute_script("return document.body.scrollHeight")
        scrolls = 0
        while scrolls < max_scrolls:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause_time)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            scrolls += 1
    except Exception as e:
        logger.error("PornHub scroll_to_bottom: %s", e)


def resolve_favorites_playlist_url(driver, fallback: str = "") -> str:
    """
    Resolve /users/<username>/videos/favorites (ph.py). Never leaves /my/favorites/videos
    when the logged-in username can be determined.
    """
    fb = (fallback or PH_PLAYLIST_URL_PLACEHOLDER).strip()
    if "/users/" in fb and "/videos/favorites" in fb:
        return fb

    try:
        html = driver.page_source or ""
        uname = _username_from_profile_html(html)
        if uname:
            resolved = favorites_url_for_username(uname)
            logger.info("PornHub favourites URL from page HTML: %s", resolved)
            return resolved
    except Exception as e:
        logger.debug("PornHub username from HTML failed: %s", e)

    try:
        payload = driver.execute_async_script(
            """
            const done = arguments[arguments.length - 1];
            fetch('https://www.pornhub.com/user/get_user_info', {
              credentials: 'include',
              headers: { 'Accept': 'application/json' }
            })
              .then(r => r.json())
              .then(d => done(d || {}))
              .catch(e => done({error: String(e)}));
            """
        )
        if isinstance(payload, dict):
            uname = _username_from_user_info_payload(payload)
            if uname:
                resolved = favorites_url_for_username(uname)
                logger.info("PornHub favourites URL from get_user_info: %s", resolved)
                return resolved
    except Exception as e:
        logger.debug("Could not resolve PornHub username via API: %s", e)

    cur = driver.current_url or ""
    m = re.search(r"/users/([^/]+)/videos/favorites", cur, re.I)
    if m:
        return favorites_url_for_username(m.group(1))

    # /my/favorites/videos redirects to /users/<you>/videos/favorites when logged in.
    if "my/favorites" in fb.lower() or not fb:
        try:
            driver.get(PH_PLAYLIST_URL_PLACEHOLDER)
            time.sleep(2.5)
            m = re.search(r"/users/([^/]+)/videos/favorites", driver.current_url or "", re.I)
            if m:
                resolved = favorites_url_for_username(m.group(1))
                logger.info("PornHub favourites URL from redirect: %s", resolved)
                return resolved
            uname = _username_from_profile_html(driver.page_source or "")
            if uname:
                resolved = favorites_url_for_username(uname)
                logger.info("PornHub favourites URL after /my/favorites redirect: %s", resolved)
                return resolved
        except Exception as e:
            logger.debug("PornHub /my/favorites redirect probe failed: %s", e)

    return fb


def navigate_to_favorites_page(
    driver,
    playlist_url: str,
    *,
    wait_seconds: float = 20.0,
) -> Tuple[bool, str, str]:
    """
    ph.py extract_video_urls_selenium step 1: open favourites and wait for the list.
    Returns (ok, message, effective_url).
    """
    url = (playlist_url or PH_PLAYLIST_URL_PLACEHOLDER).strip()
    if not url.startswith("http"):
        url = urljoin(PH_BASE, url)
    logger.info("Navigating to PornHub favourites page: %s", url)
    try:
        driver.get(url)
    except Exception as e:
        return False, f"Could not open favourites page: {e}", url

    deadline = time.time() + max(5.0, wait_seconds)
    while time.time() < deadline:
        time.sleep(1.0)
        cur = (driver.current_url or "").lower()
        if "login" in cur and "view_video" not in cur:
            return (
                False,
                "Redirected to login — sign in, open your favourites, then continue.",
                driver.current_url or url,
            )
        html = driver.page_source or ""
        if 'id="moreData"' in html or "view_video.php" in html:
            effective = driver.current_url or url
            if "/videos/favorites" in effective.lower() or "my/favorites" in effective.lower():
                logger.info("Favourites page loaded: %s", effective)
                return True, "Favourites page ready.", effective
        # Redirect may land on canonical /users/.../videos/favorites after /my/favorites/videos
        if "/videos/favorites" in cur and "login" not in cur:
            time.sleep(1.5)
            effective = driver.current_url or url
            return True, "Favourites page ready.", effective

    return (
        False,
        "Favourites page did not show a video list — open your favourites in Chrome, then retry.",
        driver.current_url or url,
    )


def try_click_load_more(driver) -> bool:
    try:
        btn = driver.find_element(By.ID, "moreDataBtn")
        if btn.is_displayed() and btn.is_enabled():
            driver.execute_script("arguments[0].click();", btn)
            return True
    except NoSuchElementException:
        pass
    except Exception as e:
        logger.debug("Load More click: %s", e)
    return False


def collect_favorites_urls_with_driver(
    driver,
    playlist_url: str,
    *,
    base_url: str = PH_BASE,
    max_load_more: int = 200,
    already_on_favorites: bool = False,
) -> List[str]:
    """
    Scroll favourites, parse links, click #moreDataBtn until exhausted.
    Caller must navigate first (navigate_to_favorites_page) unless already_on_favorites.
    Mirrors archive/ph.py extract_video_urls_selenium (after driver.get(playlist_url)).
    """
    if not already_on_favorites:
        ok, msg, playlist_url = navigate_to_favorites_page(driver, playlist_url)
        if not ok:
            logger.warning("Favourites navigation before scrape: %s", msg)
    else:
        logger.info("Scraping PornHub favourites (already on page): %s", driver.current_url or playlist_url)
    time.sleep(1.0)

    all_urls: set[str] = set()
    load_more_attempts = 0
    clicked_load_more = False
    blank_after_click = 0  # consecutive zero-new-URL iterations after a Load More click

    while load_more_attempts < max_load_more:
        scroll_to_bottom_pornhub(driver, pause_time=2.0)
        time.sleep(1)

        html = driver.page_source
        new_urls = parse_video_links_pornhub(html, base_url)
        before = len(all_urls)
        all_urls.update(new_urls)
        added = len(all_urls) - before
        logger.info("PornHub favourites scrape: total %s URLs (+%s this pass)", len(all_urls), added)

        if clicked_load_more and added == 0:
            # BUG 4 fix: allow one extra iteration in case the AJAX response was still
            # loading when we read page_source (slow network / CDN lag after click).
            blank_after_click += 1
            if blank_after_click >= 2:
                logger.info("No new videos after Load More (2 checks); finished.")
                break
            logger.info("No new URLs yet after Load More click — waiting one more cycle…")
            time.sleep(3)
            continue
        else:
            blank_after_click = 0

        clicked_load_more = try_click_load_more(driver)
        if not clicked_load_more:
            # BUG 1 fix: on the very first iteration the #moreDataBtn may not have
            # rendered yet (lazy-loaded JS section). Give it a brief extra wait and
            # retry once before concluding there are no more pages.
            if load_more_attempts == 0:
                logger.info("No Load More button on first check — retrying after short wait…")
                time.sleep(2.5)
                clicked_load_more = try_click_load_more(driver)
            if not clicked_load_more:
                logger.info("No Load More button; finished collecting favourites.")
                break
        load_more_attempts += 1
        logger.info("Clicked Load More (%s/%s), waiting…", load_more_attempts, max_load_more)
        time.sleep(3)

    result = list(all_urls)
    logger.info("Extracted %s PornHub video URLs via Selenium.", len(result))
    return result


def _find_matching_bracket(html: str, start_pos: int, open_c: str, close_c: str) -> int:
    depth = 1
    i = start_pos + 1
    while i < len(html) and depth:
        if html[i] == '"':
            i += 1
            while i < len(html):
                if html[i] == "\\" and i + 1 < len(html):
                    i += 2
                    continue
                if html[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if html[i] == open_c:
            depth += 1
        elif html[i] == close_c:
            depth -= 1
        i += 1
    return i - 1 if depth == 0 else -1


def _extract_flashvars_block(html: str) -> Optional[str]:
    m = re.search(r"var\s+flashvars_\d+\s*=\s*\{", html)
    if not m:
        return None
    start_brace = m.end() - 1
    end_brace = _find_matching_bracket(html, start_brace, "{", "}")
    if end_brace == -1:
        return None
    return html[start_brace : end_brace + 1]


def extract_media_from_page(html: str, video_url: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Returns (media_url, fmt, title, viewkey) with fmt 'mp4' or 'hls', or None.
    """
    parsed = urlparse(video_url)
    qs = parse_qs(parsed.query)
    viewkey = (qs.get("viewkey") or [None])[0] or "unknown"

    flashvars = _extract_flashvars_block(html)
    if not flashvars:
        return None

    md_start = flashvars.find('"mediaDefinitions":[')
    if md_start == -1:
        return None
    start = md_start + len('"mediaDefinitions":[')
    end_bracket = _find_matching_bracket(flashvars, start - 1, "[", "]")
    if end_bracket == -1:
        return None
    array_str = flashvars[start : end_bracket]

    definitions: List[Tuple[str, str]] = []
    pos = 0
    while pos < len(array_str):
        obj_start = array_str.find("{", pos)
        if obj_start == -1:
            break
        end_brace = _find_matching_bracket(array_str, obj_start, "{", "}")
        if end_brace == -1:
            break
        block = array_str[obj_start : end_brace + 1]
        pos = end_brace + 1
        fm = re.search(r'"format"\s*:\s*"([^"]+)"', block)
        vm = re.search(r'"videoUrl"\s*:\s*"((?:[^"\\]|\\.)*)"', block)
        if not fm or not vm:
            continue
        fmt = fm.group(1).strip()
        url = vm.group(1).replace("\\/", "/")
        definitions.append((fmt, url))

    if not definitions:
        return None

    media_url = None
    chosen_fmt = None
    for fmt, url in definitions:
        if "get_media" in url:
            media_url = url
            chosen_fmt = "mp4"
            break
    if not media_url:
        for fmt, url in definitions:
            if "m3u8" in url or fmt == "hls":
                media_url = url
                chosen_fmt = "hls"
                break
    if not media_url:
        media_url = definitions[0][1]
        chosen_fmt = definitions[0][0]

    title = None
    tt = re.search(r'"video_title"\s*:\s*"((?:[^"\\]|\\.)*)"', flashvars)
    if tt:
        title = tt.group(1).replace("\\/", "/").replace('\\"', '"')
    if not title:
        title = viewkey

    return (media_url, chosen_fmt or "hls", title, viewkey)


def _sanitize_filename(s: str, max_len: int = 100) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = re.sub(r"\s+", "_", s.strip()).strip(".") or "video"
    return s[:max_len] if len(s) > max_len else s


def fetch_video_title_from_page(
    cookie_file: str,
    video_page_url: str,
    timeout: float = 30.0,
    proxy_url: str = "",
    *,
    driver=None,
) -> str:
    """Resolve title from flashvars (for library dedupe). Uses browser when driver given (ph.py)."""
    html = _get_video_page_html(
        video_page_url, cookie_file, driver=driver, timeout=timeout, proxy_url=proxy_url
    )
    if html:
        extracted = extract_media_from_page(html, video_page_url)
        if extracted:
            _, _, title, _vk = extracted
            return _sanitize_filename(title)
    return f"video_{abs(hash(video_page_url)) % 10_000_000}"


def fetch_video_page_data(
    cookie_file: str,
    video_page_url: str,
    *,
    driver=None,
    timeout: float = 30.0,
    proxy_url: str = "",
) -> Tuple[str, Optional[Tuple]]:
    """Fetch a video page and return ``(title, extracted_tuple_or_None)``.

    Returns the full ``(media_url, fmt, title, viewkey)`` tuple alongside the
    title so callers can pass it to ``download_pornhub_video_page`` via
    ``pre_extracted``, avoiding a second browser navigation for the same page.
    """
    html = _get_video_page_html(
        video_page_url, cookie_file, driver=driver, timeout=timeout, proxy_url=proxy_url
    )
    if html:
        extracted = extract_media_from_page(html, video_page_url)
        if extracted:
            _, _, title, _vk = extracted
            return _sanitize_filename(title), extracted
    return f"video_{abs(hash(video_page_url)) % 10_000_000}", None


def _get_video_page_html(
    video_url: str,
    cookie_file: str,
    *,
    driver=None,
    timeout: float = 30.0,
    proxy_url: str = "",
) -> Optional[str]:
    """Fetch a PornHub video page HTML.

    Prefer the authenticated browser (driver) when available — ph.py style.
    Falls back to requests + cookie file when driver is None.
    """
    if driver is not None:
        try:
            driver.get(video_url)
            time.sleep(2.5)  # ph.py: wait for flashvars_* in page_source
            return driver.page_source
        except Exception as e:
            logger.error("Error loading PornHub video page in browser %s: %s", video_url, e)
            return None
    session = _session_with_cookies(cookie_file, proxy_url)
    try:
        r = session.get(video_url, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.error("Error fetching PornHub video page %s: %s", video_url, e)
        return None


def _download_from_media_url(
    media_url: str,
    fmt: str,
    output_path: str,
    cookie_file: str,
    *,
    proxy_url: str = "",
    yt_dlp_retries: int = 3,
    referer: str = "",
    yt_dlp_format: str = "",
) -> bool:
    pv = (proxy_url or "").strip()
    if fmt == "mp4" or "get_media" in media_url:
        session = _session_with_cookies(cookie_file, proxy_url=pv)
        stream_headers = {"Referer": referer or f"{PH_BASE}/"}
        try:
            r = session.get(
                media_url, stream=True, timeout=120, headers=stream_headers
            )
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            logger.error("PornHub MP4 download failed: %s", e)
            return False

    retries_s = str(max(0, yt_dlp_retries))
    cmd = [sys.executable, "-m", "yt_dlp", "--cookies", cookie_file]
    if pv:
        cmd.extend(["--proxy", pv])
    format_sel = (yt_dlp_format or "").strip() or "best"
    cmd.extend(
        [
            "-f",
            format_sel,
            "-o",
            output_path,
            "--merge-output-format",
            "mp4",
            "--no-warnings",
            "--retries",
            retries_s,
            "--fragment-retries",
            retries_s,
            media_url,
        ]
    )
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            logger.error("yt-dlp HLS failed: %s", r.stderr[-2000:] if r.stderr else r.returncode)
        return r.returncode == 0
    except Exception as e:
        logger.error("yt-dlp subprocess error: %s", e)
        return False


def download_pornhub_video_page(
    video_page_url: str,
    cookie_file: str,
    download_dir: str,
    session_id: str,
    *,
    driver=None,
    proxy_url: str = "",
    skip_if_exists: bool = True,
    yt_dlp_retries: int = 3,
    yt_dlp_format: str = "",
    pre_extracted: Optional[Tuple] = None,
) -> None:
    """
    Download one PornHub video: page URL → mediaDefinitions → file on disk.
    Integrates with progress_tracker using video_page_url as the item key.
    When driver is provided, the video page is opened in the authenticated browser
    (same approach as ph.py) so flashvars_* is always present.
    Pass ``pre_extracted`` (from ``fetch_video_page_data``) to skip re-fetching the
    page when it was already loaded during a prior library-compare phase.
    """
    from progress_tracker import get_tracker

    tracker = get_tracker(session_id)

    if pre_extracted is not None:
        extracted = pre_extracted
        logger.info("Using pre-fetched page data for download: %s", video_page_url)
    else:
        logger.info("Opening PornHub video page for download: %s", video_page_url)
        html = _get_video_page_html(video_page_url, cookie_file, driver=driver, proxy_url=proxy_url)
        if not html:
            tracker.fail_download(video_page_url, "Could not load video page")
            return

        extracted = extract_media_from_page(html, video_page_url)
        if not extracted:
            tracker.fail_download(video_page_url, "No flashvars / mediaDefinitions on page")
            return

    media_url, fmt, title, viewkey = extracted
    safe_title = _sanitize_filename(title)
    output_path = os.path.join(download_dir, f"{safe_title}_{viewkey}.mp4")

    if skip_if_exists and os.path.exists(output_path):
        tracker.skip_download(video_page_url, "File already exists")
        return

    tracker.start_download(video_page_url)

    ok = _download_from_media_url(
        media_url,
        fmt,
        output_path,
        cookie_file,
        proxy_url=proxy_url,
        yt_dlp_retries=yt_dlp_retries,
        referer=video_page_url,
        yt_dlp_format=yt_dlp_format,
    )
    if ok and os.path.exists(output_path):
        sz = os.path.getsize(output_path)
        tracker.complete_download(video_page_url, output_path, sz)
        logger.info("Downloaded PornHub video: %s", output_path)
    else:
        tracker.fail_download(video_page_url, "Download failed or output missing")


def download_pornhub_videos_parallel(
    video_page_urls: List[str],
    download_dir: str,
    cookie_file: str,
    session_id: str,
    *,
    max_workers: int = 4,
    driver=None,
    proxy_url: str = "",
    skip_if_exists: bool = True,
    yt_dlp_retries: int = 3,
    yt_dlp_format: str = "",
    cancel_check: Optional[Callable[[], bool]] = None,
    pre_extracted_map: Optional[Dict[str, Tuple]] = None,
) -> None:
    """Download PornHub videos from their page URLs.

    When driver is provided (Chrome browser mode), pages are opened in the
    authenticated browser sequentially — exactly as ph.py does — so bot
    detection is bypassed and flashvars_* is always present in the HTML.

    When driver is None (embedded-login mode), falls back to parallel
    requests with the Netscape cookie file.

    Pass ``pre_extracted_map`` (a dict mapping URL → extracted tuple from
    ``fetch_video_page_data``) to reuse already-fetched page data and avoid
    navigating the browser to each page a second time.
    """
    if not video_page_urls:
        return
    from progress_tracker import get_tracker

    tracker = get_tracker(session_id)
    if driver is not None:
        logger.info(
            "Starting %s PornHub downloads (browser mode, sequential — ph.py style)",
            len(video_page_urls),
        )
        for url in video_page_urls:
            if cancel_check and cancel_check():
                tracker.fail_nonterminal("Download cancelled by user.")
                logger.info("PornHub download batch cancelled by user.")
                return
            download_pornhub_video_page(
                url,
                cookie_file,
                download_dir,
                session_id,
                driver=driver,
                proxy_url=proxy_url,
                skip_if_exists=skip_if_exists,
                yt_dlp_retries=yt_dlp_retries,
                yt_dlp_format=yt_dlp_format,
                pre_extracted=(pre_extracted_map or {}).get(url),
            )
        logger.info("PornHub browser-mode download batch finished.")
        return
    logger.info("Starting %s PornHub downloads (%s workers, requests mode)", len(video_page_urls), max_workers)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    cancelled = False
    try:
        futures = []
        for url in video_page_urls:
            if cancel_check and cancel_check():
                cancelled = True
                break
            futures.append(
                executor.submit(
                    download_pornhub_video_page,
                    url,
                    cookie_file,
                    download_dir,
                    session_id,
                    proxy_url=proxy_url,
                    skip_if_exists=skip_if_exists,
                    yt_dlp_retries=yt_dlp_retries,
                    yt_dlp_format=yt_dlp_format,
                )
            )
        for fut in concurrent.futures.as_completed(futures):
            if cancel_check and cancel_check():
                cancelled = True
                break
            try:
                fut.result(timeout=7200)
            except Exception as e:
                logger.error("PornHub parallel worker error: %s", e)
    finally:
        if cancelled:
            tracker.fail_nonterminal("Download cancelled by user.")
        executor.shutdown(wait=cancelled, cancel_futures=cancelled)
    logger.info("PornHub parallel download batch finished.")


def _page_looks_logged_in(html: str, current_url: str = "") -> bool:
    cur = (current_url or "").lower()
    if "login" in cur and "view_video" not in cur:
        return False
    markers = (
        '"isLoggedIn":true',
        '"loggedin":1',
        'class="usernameBadge"',
        'data-loggedin="1"',
        'id="profileMenu"',
        'class="profileMenu"',
        "/users/",
        "logout",
        "Log out",
        "Sign Out",
        "my-favorites",
        "view_video.php",
    )
    return any(m in html for m in markers)


def check_cookie_health_with_driver(
    driver,
    *,
    probe_url: str = "",
) -> Tuple[bool, str]:
    """Verify login inside Chrome (reliable right after manual sign-in)."""
    landing = (probe_url or f"{PH_BASE}/users/likes/videos").strip()
    try:
        driver.get(landing)
        time.sleep(2.5)
    except Exception as e:
        return False, f"Could not open PornHub in Chrome: {e}"

    current_url = driver.current_url or ""
    if "login" in current_url.lower() and "view_video" not in current_url.lower():
        return (
            False,
            "Chrome was sent to the login page — sign in on pornhub.com, then click Continue.",
        )

    try:
        payload = driver.execute_async_script(
            """
            const done = arguments[arguments.length - 1];
            fetch('https://www.pornhub.com/user/get_user_info', {
              credentials: 'include',
              headers: { 'Accept': 'application/json' }
            })
              .then(r => r.json())
              .then(d => done({
                ok: d.loggedin == 1 || d.loggedin === true || d.loggedin === '1',
                d: d
              }))
              .catch(e => done({ok: false, error: String(e)}));
            """
        )
        if isinstance(payload, dict) and payload.get("ok"):
            return True, "Browser session authenticated on PornHub."
        if isinstance(payload, dict) and payload.get("error"):
            logger.debug("PornHub get_user_info in browser: %s", payload.get("error"))
    except Exception as e:
        logger.debug("PornHub in-browser get_user_info failed: %s", e)

    try:
        html = driver.page_source or ""
        if _page_looks_logged_in(html, current_url):
            return True, "Browser session authenticated on PornHub."
    except Exception as e:
        return False, f"Cookie health check failed: {e}"

    return (
        False,
        "Chrome does not look logged in — open https://www.pornhub.com/, sign in, "
        "visit your favourites, then click Continue after Chrome login.",
    )


def check_cookie_health(cookie_file: str, proxy_url: str = "") -> Tuple[bool, str]:
    """Quick sanity-check that the saved Netscape cookie file still authenticates on PornHub.

    Returns ``(True, message)`` when the session appears valid, or ``(False, reason)``
    when cookies are missing, expired, or the request fails.
    """
    if not os.path.isfile(cookie_file):
        return False, "Cookie file not found."
    session = _session_with_cookies(cookie_file, proxy_url)
    try:
        r = session.get(f"{PH_BASE}/user/get_user_info", timeout=15)
        if r.status_code == 200:
            try:
                data = r.json()
                if data.get("loggedin") in (1, True, "1"):
                    return True, "Cookies valid — PornHub user is authenticated."
            except Exception:
                pass
        # Fallback: favourites page or homepage login markers
        for probe in (
            f"{PH_BASE}/users/likes/videos",
            f"{PH_BASE}/",
        ):
            r2 = session.get(probe, timeout=20)
            html = r2.text or ""
            final = (r2.url or "").lower()
            if "login" in final and "view_video" not in final:
                continue
            if _page_looks_logged_in(html, final):
                return True, "Cookies valid — PornHub user is authenticated."
        return False, "Cookies appear invalid or session expired — please sign in again."
    except Exception as e:
        return False, f"Cookie health check request failed: {e}"
