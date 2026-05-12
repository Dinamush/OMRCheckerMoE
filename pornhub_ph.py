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
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

logger = logging.getLogger(__name__)

PH_BASE = "https://www.pornhub.com"


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


def _session_with_cookies(cookie_file: str) -> requests.Session:
    session = requests.Session()
    jar = MozillaCookieJar(cookie_file)
    jar.load(ignore_discard=True, ignore_expires=True)
    session.cookies = jar
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
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
) -> List[str]:
    """
    Collect favourite video page URLs using authenticated GETs and ?page=1,2,...
    Used when there is no Selenium driver (e.g. embedded WebView login).
    """
    session = _session_with_cookies(cookie_file)
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
) -> List[str]:
    """
    Load favourites in the logged-in browser: scroll, parse, click #moreDataBtn until exhausted.
    Mirrors archive/ph.py extract_video_urls_selenium.
    """
    logger.info("Loading PornHub favourites (Selenium): %s", playlist_url)
    driver.get(playlist_url)
    time.sleep(2)

    all_urls: set[str] = set()
    load_more_attempts = 0
    clicked_load_more = False

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
            logger.info("No new videos after last Load More; finished.")
            break

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


def fetch_video_title_from_page(cookie_file: str, video_page_url: str, timeout: float = 30.0) -> str:
    """Resolve title from flashvars (for library dedupe)."""
    session = _session_with_cookies(cookie_file)
    try:
        r = session.get(video_page_url, timeout=timeout)
        r.raise_for_status()
        extracted = extract_media_from_page(r.text, video_page_url)
        if extracted:
            _, _, title, _vk = extracted
            return _sanitize_filename(title)
    except Exception as e:
        logger.warning("Could not read PornHub title for %s: %s", video_page_url, e)
    return f"video_{abs(hash(video_page_url)) % 10_000_000}"


def _get_video_page_html(video_url: str, cookie_file: str, timeout: float = 30.0) -> Optional[str]:
    session = _session_with_cookies(cookie_file)
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
) -> bool:
    if fmt == "mp4" or "get_media" in media_url:
        session = _session_with_cookies(cookie_file)
        try:
            r = session.get(media_url, stream=True, timeout=120)
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            logger.error("PornHub MP4 download failed: %s", e)
            return False

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--cookies",
        cookie_file,
        "-o",
        output_path,
        "--merge-output-format",
        "mp4",
        "--no-warnings",
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        media_url,
    ]
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
) -> None:
    """
    Download one PornHub video: page URL → mediaDefinitions → file on disk.
    Integrates with progress_tracker using video_page_url as the item key.
    """
    from progress_tracker import get_tracker

    tracker = get_tracker(session_id)

    html = _get_video_page_html(video_page_url, cookie_file)
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

    if os.path.exists(output_path):
        tracker.skip_download(video_page_url, "File already exists")
        return

    tracker.start_download(video_page_url)

    ok = _download_from_media_url(media_url, fmt, output_path, cookie_file)
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
) -> None:
    """Parallel per-video page fetch + extract + download (requests-based, thread-safe)."""
    if not video_page_urls:
        return
    logger.info("Starting %s PornHub downloads (%s workers)", len(video_page_urls), max_workers)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = [
            executor.submit(
                download_pornhub_video_page,
                url,
                cookie_file,
                download_dir,
                session_id,
            )
            for url in video_page_urls
        ]
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result(timeout=7200)
            except Exception as e:
                logger.error("PornHub parallel worker error: %s", e)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    logger.info("PornHub parallel download batch finished.")
