"""
Pixiv bookmark collection and image download via Ajax API (session cookies).

Public bookmarks: rest=show — https://www.pixiv.net/users/{id}/bookmarks/artworks
Owner full list: rest=hide — requires login as that user.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse

import requests

import challenge_detect
import flaresolverr_client
from paths import bundled_ffmpeg_path
from pixiv_titles import resolve_pixiv_filename_title, sanitize_filename_slug
from progress_tracker import get_tracker

logger = logging.getLogger(__name__)


class DownloadCancelled(Exception):
    """User stopped the workflow (Stop download)."""


PIXIV_HOME = "https://www.pixiv.net/"
PIXIV_LOGIN = "https://accounts.pixiv.net/login"
PIXIV_REFERER = "https://www.pixiv.net/"
DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
BOOKMARK_PAGE_LIMIT = 48
PIXIV_ILLUST_TYPE_MANGA = 1
PIXIV_ILLUST_TYPE_UGOIRA = 2
_MAX_FILENAME_TOTAL = 200
PixivUgoiraFormat = Literal["zip", "gif", "both"]
_PIXIV_PARENT_DOMAIN = ".pixiv.net"
_PIXIV_AUTH_NAMES = frozenset({"PHPSESSID", "__cf_bm", "cf_clearance"})

PIXIV_LOGIN_PROGRESS_HINT = (
    "Log in on accounts.pixiv.net, then in the **same Chrome window** open "
    "https://www.pixiv.net/ and confirm you see your account. "
    "Only then click **Continue after Chrome login** on this page."
)


def parse_user_id(value: str) -> Optional[str]:
    """Extract numeric Pixiv user id from raw digits or a profile/bookmarks URL."""
    raw = (value or "").strip()
    if not raw:
        return None
    if parse_illust_id(raw):
        return None
    if raw.isdigit():
        return raw
    m = re.search(r"/users/(\d+)", raw)
    return m.group(1) if m else None


def parse_illust_id(value: str) -> Optional[str]:
    """Extract illustration id from an artwork URL (not bare user ids)."""
    raw = (value or "").strip()
    if not raw:
        return None
    for pattern in (
        r"/artworks/(\d+)",
        r"[?&]illust_id=(\d+)",
    ):
        m = re.search(pattern, raw)
        if m:
            return m.group(1)
    return None


def resolve_pixiv_target(
    value: str,
    *,
    single_artwork: bool = False,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Classify Pixiv form input.

    Returns ``(mode, user_id, illust_id)`` where mode is ``artwork`` or ``bookmarks``.
    """
    raw = (value or "").strip()
    illust = parse_illust_id(raw)
    if illust and (single_artwork or "/artworks/" in raw or "illust_id=" in raw):
        return "artwork", None, illust
    if single_artwork and raw.isdigit():
        return "artwork", None, raw
    uid = parse_user_id(raw)
    if uid:
        return "bookmarks", uid, None
    raise ValueError(
        "Unrecognized Pixiv input. Use a bookmarks URL (/users/…/bookmarks), "
        "an artwork URL (/artworks/{id}), or enable single artwork with a numeric illust ID."
    )


def work_from_illust_body(body: Dict[str, Any], illust_id: str) -> Dict[str, Any]:
    title = str(
        body.get("title")
        or body.get("illustTitle")
        or body.get("illustId")
        or illust_id
    )
    return {"id": illust_id, "illustId": illust_id, "title": title}


def work_from_illust(session: requests.Session, illust_id: str) -> Dict[str, Any]:
    body = fetch_illust_body(session, illust_id)
    return work_from_illust_body(body, illust_id)


def _ua_sidecar_path(cookie_file: str) -> str:
    return f"{cookie_file}.ua"


def _should_scope_to_parent_domain(name: str, domain: str) -> bool:
    d = (domain or "").lower()
    return name in _PIXIV_AUTH_NAMES or "pixiv.net" in d


def _normalize_cookie_domain(name: str, domain: str) -> str:
    if _should_scope_to_parent_domain(name, domain):
        return _PIXIV_PARENT_DOMAIN
    return domain or _PIXIV_PARENT_DOMAIN


def _parse_user_self(data: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """Return (user_id, error_message). error_message empty on success."""
    if data.get("error"):
        return None, str(data.get("message") or "Pixiv Ajax error")

    user_data = data.get("userData")
    if isinstance(user_data, dict):
        uid = user_data.get("userId") or user_data.get("id")
        if uid:
            return str(uid), ""

    body = data.get("body")
    if isinstance(body, dict):
        uid = body.get("userId") or body.get("id") or body.get("user_id")
        if uid:
            return str(uid), ""

    # Guest bootstrap: site config only (userData null, no body)
    if "token" in data and data.get("userData") is None and "body" not in data:
        return (
            None,
            "Ajax session not authenticated (stale PHPSESSID?). Log out in Chrome, "
            "log in again, visit www.pixiv.net, then click Continue.",
        )

    if body is None or body == []:
        return None, "Session expired or invalid PHPSESSID — log in again."
    return None, "Session response missing user id."


def _bookmarks_probe_ok(data: Dict[str, Any]) -> bool:
    if data.get("error"):
        return False
    body = data.get("body")
    return isinstance(body, dict) and ("works" in body or "total" in body)


def _apply_session_headers(session: requests.Session, user_agent: str) -> None:
    session.headers.update(
        {
            "User-Agent": user_agent or DEFAULT_UA,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": PIXIV_REFERER,
            "X-Requested-With": "XMLHttpRequest",
        }
    )


def _driver_phpsessid(driver) -> Optional[str]:
    for cookie in driver.get_cookies():
        if cookie.get("name") == "PHPSESSID" and cookie.get("value"):
            return str(cookie["value"])
    return None


def ajax_get_json_via_driver(driver, ajax_path: str) -> Dict[str, Any]:
    """Call Pixiv Ajax from inside Chrome so cookies match the visible login."""
    path = ajax_path.lstrip("/")
    url = f"{PIXIV_HOME}ajax/{path}" if not path.startswith("http") else path
    raw = driver.execute_async_script(
        """
        const url = arguments[0];
        const done = arguments[arguments.length - 1];
        fetch(url, {
            credentials: 'include',
            headers: { 'Accept': 'application/json', 'X-Requested-With': 'XMLHttpRequest' }
        })
        .then(r => r.text())
        .then(t => done(t))
        .catch(e => done(JSON.stringify({error: true, message: String(e)})));
        """,
        url,
    )
    if not raw:
        raise RuntimeError("Empty response from in-browser Pixiv Ajax.")
    data = json.loads(raw)
    if data.get("error") and not isinstance(data.get("body"), dict):
        raise RuntimeError(data.get("message") or "Pixiv Ajax error (browser)")
    return data


def establish_www_session(
    driver,
    timeout: float = 60.0,
    *,
    target_user_id: Optional[str] = None,
    bookmark_rest: str = "show",
    cancel_check: Optional[Callable[[], bool]] = None,
) -> str:
    """
    After accounts login, open www (or target bookmarks page) and verify Ajax works.

    When *target_user_id* is set, bookmark Ajax for that user is enough (no /user/self).
    Returns the user id to use for collection (target or logged-in self).
    """
    target_uid = parse_user_id(target_user_id or "")
    landing = PIXIV_HOME
    if target_uid:
        landing = f"{PIXIV_HOME}en/users/{target_uid}/bookmarks/artworks"
    driver.get(landing)
    time.sleep(3)
    deadline = time.time() + timeout
    last_err = ""

    while time.time() < deadline:
        if cancel_check and cancel_check():
            raise DownloadCancelled("Download cancelled by user.")
        if not _driver_phpsessid(driver):
            names = sorted({c.get("name", "") for c in driver.get_cookies()})
            logging.debug("Waiting for PHPSESSID; have cookies: %s", names)
            time.sleep(2.0)
            driver.get(landing)
            continue

        if target_uid:
            probe = (
                f"user/{target_uid}/illusts/bookmarks"
                f"?tag=&offset=0&limit=1&rest={bookmark_rest}"
            )
            try:
                data = ajax_get_json_via_driver(driver, probe)
                if _bookmarks_probe_ok(data):
                    total = (data.get("body") or {}).get("total", "?")
                    logging.info(
                        "Pixiv bookmarks Ajax OK for user %s (total=%s, rest=%s)",
                        target_uid,
                        total,
                        bookmark_rest,
                    )
                    return target_uid
                last_err = data.get("message") or "Bookmarks Ajax returned no works"
            except Exception as e:
                last_err = str(e)
                logging.debug("Pixiv bookmarks probe failed: %s", e)
        else:
            try:
                data = ajax_get_json_via_driver(driver, "user/self")
                uid, err = _parse_user_self(data)
                if uid:
                    logging.info("Pixiv www session ready (logged-in user %s)", uid)
                    return uid
                last_err = err
            except Exception as e:
                last_err = str(e)
                logging.debug("Pixiv /ajax/user/self probe failed: %s", e)

        time.sleep(2.0)
        driver.get(landing)

    cookie_names = sorted({c.get("name", "") for c in driver.get_cookies()})
    profile_hint = (
        " If you keep seeing this, delete the folder "
        "%LOCALAPPDATA%\\HamsterScraper\\browser_profiles\\pixiv and try again."
    )
    raise RuntimeError(
        "Timed out waiting for Pixiv Ajax (PHPSESSID present but API still guest or blocked). "
        f"Cookies in Chrome: {', '.join(cookie_names[:12])}. "
        f"Last check: {last_err or 'unknown'}. "
        "Log out in Chrome, log in fresh at accounts.pixiv.net, open www.pixiv.net, "
        f"then click Continue.{profile_hint}"
    )


def session_from_driver(driver, proxy_url: str = "") -> requests.Session:
    """Build requests session from live Chrome cookies (avoids Netscape domain loss)."""
    session = requests.Session()
    try:
        ua = driver.execute_script("return navigator.userAgent") or DEFAULT_UA
    except Exception:
        ua = DEFAULT_UA
    _apply_session_headers(session, ua)
    pv = (proxy_url or "").strip()
    if pv:
        session.proxies.update({"http": pv, "https": pv})
    for cookie in driver.get_cookies():
        name = cookie.get("name")
        value = cookie.get("value")
        if not name or value is None:
            continue
        domain = _normalize_cookie_domain(name, cookie.get("domain") or "")
        d = domain.lstrip(".")
        session.cookies.set(name, value, domain=d, path=cookie.get("path") or "/")
        if name in _PIXIV_AUTH_NAMES:
            session.cookies.set(name, value, domain="www.pixiv.net", path="/")
    return session


def save_session_from_driver(driver, cookie_file: str) -> None:
    """Persist Netscape cookies + matching User-Agent for later httpx/requests runs."""
    cookies = driver.get_cookies()
    with open(cookie_file, "w", encoding="utf-8") as f:
        f.write("# Netscape HTTP Cookie File\n")
        f.write("# This file was generated by Selenium\n")
        for cookie in cookies:
            domain = _normalize_cookie_domain(
                cookie.get("name", ""), cookie.get("domain", "")
            )
            flag = "TRUE" if domain.startswith(".") else "FALSE"
            path = cookie.get("path", "/")
            secure = "TRUE" if cookie.get("secure", False) else "FALSE"
            expiry = int(cookie.get("expiry") or 0)
            name = cookie.get("name", "")
            value = cookie.get("value", "")
            f.write(
                f"{domain}\t{flag}\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n"
            )
    try:
        ua = driver.execute_script("return navigator.userAgent") or DEFAULT_UA
    except Exception:
        ua = DEFAULT_UA
    with open(_ua_sidecar_path(cookie_file), "w", encoding="utf-8") as uf:
        uf.write(ua.strip())
    logging.info("Pixiv cookies and User-Agent saved to %s", cookie_file)


def load_session(cookie_file: str, proxy_url: str = "") -> requests.Session:
    session = requests.Session()
    ua = DEFAULT_UA
    ua_path = _ua_sidecar_path(cookie_file)
    if os.path.isfile(ua_path):
        try:
            ua = Path(ua_path).read_text(encoding="utf-8").strip() or DEFAULT_UA
        except OSError:
            pass
    _apply_session_headers(session, ua)
    pv = (proxy_url or "").strip()
    if pv:
        session.proxies.update({"http": pv, "https": pv})
    jar = MozillaCookieJar(cookie_file)
    jar.load(ignore_discard=True, ignore_expires=True)
    for c in jar:
        domain = _normalize_cookie_domain(c.name, c.domain or "")
        session.cookies.set(
            c.name,
            c.value,
            domain=domain.lstrip("."),
            path=c.path or "/",
        )
    return session


def inject_cookies_into_session(
    session: requests.Session,
    cookies: Dict[str, str],
    *,
    domain: str = ".pixiv.net",
) -> None:
    for name, value in cookies.items():
        session.cookies.set(name, value, domain=domain, path="/")


def apply_flaresolverr_to_session(
    session: requests.Session,
    *,
    base_url: str,
    target_url: str = PIXIV_HOME,
) -> None:
    cookies, ua = flaresolverr_client.solve_url(target_url, base_url=base_url)
    inject_cookies_into_session(session, cookies)
    if ua:
        session.headers["User-Agent"] = ua


def _ajax_json(session: requests.Session, url: str, *, timeout: float = 30.0) -> Dict[str, Any]:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    text = r.text or ""
    if challenge_detect.detect_challenge(text, r.url, "") != "none":
        raise RuntimeError(
            f"Pixiv returned a challenge or login wall (HTTP {r.status_code}). "
            "Complete login in Chrome or enable FlareSolverr in Settings."
        )
    data = r.json()
    if data.get("error"):
        raise RuntimeError(data.get("message") or "Pixiv Ajax error")
    return data


def check_cookie_health_with_driver(driver) -> Tuple[bool, str]:
    """Verify login using the browser's own cookie jar (most reliable)."""
    try:
        if not _driver_phpsessid(driver):
            return (
                False,
                "Chrome has no PHPSESSID yet — open https://www.pixiv.net/ while logged in, "
                "then click Continue after Chrome login.",
            )
        data = ajax_get_json_via_driver(driver, "user/self")
        uid, err = _parse_user_self(data)
        if uid:
            return True, f"Cookies valid — logged in as user {uid}."
        return False, err or "Session response missing user id."
    except Exception as e:
        return False, f"Cookie health check failed: {e}"


def check_cookie_health_with_session(session: requests.Session) -> Tuple[bool, str]:
    try:
        session.get(PIXIV_HOME, timeout=25)
        data = _ajax_json(session, f"{PIXIV_HOME}ajax/user/self")
        uid, err = _parse_user_self(data)
        if uid:
            return True, f"Cookies valid — logged in as user {uid}."
        return False, err or "Session response missing user id."
    except Exception as e:
        return False, f"Cookie health check failed: {e}"


def check_cookie_health(cookie_file: str, proxy_url: str = "") -> Tuple[bool, str]:
    if not os.path.isfile(cookie_file):
        return False, "Cookie file not found."
    try:
        session = load_session(cookie_file, proxy_url)
        has_sess = any(
            c.name == "PHPSESSID" and "pixiv" in (c.domain or "")
            for c in session.cookies
        )
        if not has_sess:
            return (
                False,
                "Cookie file has no PHPSESSID for pixiv.net — log in, visit www.pixiv.net, "
                "then save cookies again.",
            )
        return check_cookie_health_with_session(session)
    except Exception as e:
        return False, f"Cookie health check failed: {e}"


def get_logged_in_user_id(session: requests.Session) -> str:
    session.get(PIXIV_HOME, timeout=25)
    data = _ajax_json(session, f"{PIXIV_HOME}ajax/user/self")
    uid, err = _parse_user_self(data)
    if not uid:
        raise RuntimeError(err or "Could not read Pixiv user id from /ajax/user/self.")
    return uid


def fetch_bookmarks_page(
    session: requests.Session,
    user_id: str,
    *,
    offset: int = 0,
    limit: int = BOOKMARK_PAGE_LIMIT,
    rest: str = "show",
) -> Tuple[List[Dict[str, Any]], int]:
    url = (
        f"{PIXIV_HOME}ajax/user/{user_id}/illusts/bookmarks"
        f"?tag=&offset={offset}&limit={limit}&rest={rest}"
    )
    data = _ajax_json(session, url)
    body = data.get("body") or {}
    works = body.get("works") or []
    total = int(body.get("total") or 0)
    return works, total


def collect_bookmark_works(
    session: requests.Session,
    user_id: str,
    rests: List[str],
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> List[Dict[str, Any]]:
    """Collect bookmark metadata across pages for each rest filter (show/hide)."""
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for rest in rests:
        offset = 0
        total = None
        while True:
            if cancel_check and cancel_check():
                break
            works, total = fetch_bookmarks_page(
                session, user_id, offset=offset, limit=BOOKMARK_PAGE_LIMIT, rest=rest
            )
            if not works:
                break
            for work in works:
                iid = str(work.get("id") or work.get("illustId") or "")
                if iid and iid not in seen:
                    seen.add(iid)
                    out.append(work)
            offset += BOOKMARK_PAGE_LIMIT
            if total is not None and offset >= total:
                break
        logging.info(
            "Pixiv bookmarks rest=%s: %s unique works so far for user %s",
            rest,
            len(out),
            user_id,
        )
    return out


def illust_artwork_referer(illust_id: str) -> str:
    return f"{PIXIV_HOME}artworks/{illust_id}"


def fetch_illust_body(session: requests.Session, illust_id: str) -> Dict[str, Any]:
    """Fetch ``/ajax/illust/{id}`` body (type, urls, page count, …)."""
    detail = _ajax_json(session, f"{PIXIV_HOME}ajax/illust/{illust_id}")
    body = detail.get("body")
    if not isinstance(body, dict):
        raise RuntimeError(f"Illust {illust_id}: missing detail body")
    return body


def is_manga_body(body: Dict[str, Any]) -> bool:
    return int(body.get("illustType") or 0) == PIXIV_ILLUST_TYPE_MANGA


def is_ugoira_body(body: Dict[str, Any]) -> bool:
    return int(body.get("illustType") or 0) == PIXIV_ILLUST_TYPE_UGOIRA


def fetch_ugoira_meta(session: requests.Session, illust_id: str) -> Dict[str, Any]:
    """Ugoira ZIP URL and per-frame delays from ``/ajax/illust/{id}/ugoira_meta``."""
    url = f"{PIXIV_HOME}ajax/illust/{illust_id}/ugoira_meta"
    headers = {
        **session.headers,
        "Referer": illust_artwork_referer(illust_id),
    }
    r = session.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    text = r.text or ""
    if challenge_detect.detect_challenge(text, r.url, "") != "none":
        raise RuntimeError(
            f"Pixiv ugoira meta challenge for {illust_id}. "
            "Log in in Chrome or enable FlareSolverr in Settings."
        )
    data = r.json()
    if data.get("error"):
        raise RuntimeError(data.get("message") or f"Pixiv ugoira meta error for {illust_id}")
    body = data.get("body")
    if not isinstance(body, dict):
        raise RuntimeError(f"Illust {illust_id}: missing ugoira_meta body")
    return body


def pick_ugoira_zip_url(meta_body: Dict[str, Any]) -> Optional[str]:
    return meta_body.get("originalSrc") or meta_body.get("src")


def fetch_illust_image_urls(
    session: requests.Session,
    illust_id: str,
    *,
    body: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Resolve original (or regular) image URLs for all pages of a static illustration."""
    if body is None:
        body = fetch_illust_body(session, illust_id)
    page_count = int(body.get("pageCount") or 1)
    urls: List[str] = []

    def _pick_url(urls_obj: Any) -> Optional[str]:
        if not isinstance(urls_obj, dict):
            return None
        return (
            urls_obj.get("original")
            or urls_obj.get("regular")
            or urls_obj.get("small")
            or urls_obj.get("thumb")
        )

    if page_count <= 1:
        u = _pick_url(body.get("urls"))
        if u:
            urls.append(u)
        return urls

    pages_data = _ajax_json(session, f"{PIXIV_HOME}ajax/illust/{illust_id}/pages")
    for page in pages_data.get("body") or []:
        u = _pick_url((page or {}).get("urls"))
        if u:
            urls.append(u)
    return urls


def _cdn_page_stem(url: Optional[str], page: int) -> str:
    """Pixiv CDN basename (e.g. ``12345_p0``) when post title is unusable or too long."""
    if url:
        name = Path(urlparse(url).path).name
        if name:
            slug = sanitize_filename_slug(Path(name).stem, max_len=80)
            if slug:
                return slug
    return f"p{page:03d}"


def _safe_filename(
    title: str,
    illust_id: str,
    page: int,
    ext: str,
    *,
    image_url: Optional[str] = None,
    max_total_len: int = _MAX_FILENAME_TOTAL,
) -> str:
    """``{id}_p{n}_{post_title}{ext}``; fall back to CDN page stem if path too long."""
    raw_title = (title or "").strip()
    if not raw_title:
        base = _cdn_page_stem(image_url, page)
    else:
        base = sanitize_filename_slug(raw_title, max_len=80)
    prefix = f"{illust_id}_p{page}_"
    bare = f"{illust_id}_p{page}{ext}"
    name = f"{prefix}{base}{ext}" if base else bare
    if len(name) > max_total_len:
        fallback = _cdn_page_stem(image_url, page)
        name = f"{prefix}{fallback}{ext}"
    if len(name) > max_total_len:
        return bare
    return name


def _manga_zip_filename(file_title: str, illust_id: str) -> str:
    """Outer archive name for multi-page manga: ``{id}_{title}.zip``."""
    bare = f"{illust_id}.zip"
    raw = (file_title or "").strip()
    if not raw:
        return bare
    budget = _MAX_FILENAME_TOTAL - len(illust_id) - len(".zip") - 1
    if budget < 8:
        return bare
    slug = sanitize_filename_slug(raw, max_len=min(80, budget))
    if not slug:
        return bare
    name = f"{illust_id}_{slug}.zip"
    if len(name) > _MAX_FILENAME_TOTAL:
        return bare
    return name


def _dest_is_complete(dest: Path) -> bool:
    if not dest.is_file() or dest.stat().st_size == 0:
        return False
    if dest.suffix.lower() == ".zip":
        return zipfile.is_zipfile(dest)
    return True


def _clone_session(base: requests.Session) -> requests.Session:
    s = requests.Session()
    s.headers.update(base.headers)
    s.cookies.update(base.cookies)
    if base.proxies:
        s.proxies.update(base.proxies)
    return s


_thread_local = threading.local()


def _worker_session(base: requests.Session) -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = _clone_session(base)
        _thread_local.session = s
    return s


def _ext_from_url(url: str) -> str:
    path = urlparse(url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".zip"):
        if path.endswith(ext):
            return ext
    return ".jpg"


def download_binary(
    session: requests.Session,
    url: str,
    dest: Path,
    *,
    referer: str = PIXIV_REFERER,
    skip_if_exists: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> bool:
    if skip_if_exists and _dest_is_complete(dest):
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    headers = {**session.headers, "Referer": referer}
    try:
        r = session.get(url, headers=headers, timeout=120, stream=True)
        r.raise_for_status()
        with open(partial, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if cancel_check and cancel_check():
                    try:
                        partial.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise DownloadCancelled("Download cancelled by user.")
                if chunk:
                    f.write(chunk)
        if not partial.is_file() or partial.stat().st_size == 0:
            partial.unlink(missing_ok=True)
            return False
        partial.replace(dest)
        return _dest_is_complete(dest)
    except Exception:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def download_image(
    session: requests.Session,
    url: str,
    dest: Path,
    *,
    skip_if_exists: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> bool:
    return download_binary(
        session,
        url,
        dest,
        skip_if_exists=skip_if_exists,
        cancel_check=cancel_check,
    )


def _write_ugoira_ffconcat(frames: List[Dict[str, Any]], path: Path) -> None:
    """Write an FFmpeg concat demuxer file from Pixiv ``frames`` metadata."""
    lines = ["ffconcat version 1.0", ""]
    seq = list(frames)
    if seq:
        last = dict(seq[-1])
        last["delay"] = 1
        seq = seq + [last]
    for frame in seq:
        name = frame.get("file")
        if not name:
            continue
        delay_ms = int(frame.get("delay") or 1)
        duration = max(0.001, round(delay_ms / 1000.0, 4))
        lines.append(f"file {name}")
        lines.append(f"duration {duration}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def find_ffmpeg() -> Optional[str]:
    """Locate ffmpeg: bundled ``bin/``, PATH, or common install locations."""
    bundled = bundled_ffmpeg_path()
    if bundled:
        return str(bundled)
    found = shutil.which("ffmpeg")
    if found:
        return found
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pfx86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    local = os.environ.get("LOCALAPPDATA", "")
    for candidate in (
        Path(pf) / "ffmpeg" / "bin" / "ffmpeg.exe",
        Path(pfx86) / "ffmpeg" / "bin" / "ffmpeg.exe",
        Path(local) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
        Path(local) / "ffmpeg" / "bin" / "ffmpeg.exe",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _extract_ugoira_frames(zip_path: Path, meta_body: Dict[str, Any]) -> Tuple[Path, List[Dict[str, Any]]]:
    """Extract ZIP to a temp dir; return (work_dir, frames list). Caller must clean up work_dir parent."""
    frames = meta_body.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("ugoira_meta has no frames")
    work = Path(tempfile.mkdtemp(prefix="ugoira_"))
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(work)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return work, frames


def ugoira_zip_to_gif_pillow(
    zip_path: Path,
    meta_body: Dict[str, Any],
    dest: Path,
) -> bool:
    """Build animated GIF with Pillow (no external ffmpeg required)."""
    try:
        from PIL import Image
    except ImportError:
        logger.debug("Pillow not installed — cannot build GIF in-process")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    work: Optional[Path] = None
    try:
        work, frames_meta = _extract_ugoira_frames(zip_path, meta_body)
        images: List[Any] = []
        durations: List[int] = []
        for fr in frames_meta:
            fname = fr.get("file")
            if not fname:
                continue
            fp = work / str(fname)
            if not fp.is_file():
                logger.warning("Ugoira frame missing in zip: %s", fname)
                continue
            im = Image.open(fp).convert("RGBA")
            images.append(im)
            durations.append(max(1, int(fr.get("delay") or 100)))
        if not images:
            return False
        paletted = [im.convert("P", palette=Image.ADAPTIVE, colors=256) for im in images]
        paletted[0].save(
            dest,
            save_all=True,
            append_images=paletted[1:],
            duration=durations,
            loop=0,
            disposal=2,
            optimize=False,
        )
    except Exception as e:
        logger.warning("Pillow ugoira GIF failed for %s: %s", zip_path, e)
        if dest.is_file():
            try:
                dest.unlink()
            except OSError:
                pass
        return False
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
    return dest.is_file() and dest.stat().st_size > 0


def ugoira_zip_to_gif_ffmpeg(
    zip_path: Path,
    meta_body: Dict[str, Any],
    dest: Path,
) -> bool:
    """Mux extracted ugoira frames to GIF via ffmpeg (optional; often smaller files)."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        logger.warning("ffmpeg not found — skipping ugoira GIF conversion for %s", zip_path)
        return False
    frames = meta_body.get("frames")
    if not isinstance(frames, list) or not frames:
        logger.warning("ugoira_meta has no frames — cannot convert %s", zip_path)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="ugoira_") as tmp:
            work = Path(tmp)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(work)
            ffconcat = work / "ffconcat.txt"
            _write_ugoira_ffconcat(frames, ffconcat)
            cmd = [
                ffmpeg,
                "-hide_banner",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(ffconcat),
                "-lavfi",
                "split[s0][s1];[s0]palettegen=stats_mode=single[p];[s1][p]paletteuse=dither=bayer",
                "-loop",
                "0",
                str(dest),
            ]
            proc = subprocess.run(
                cmd,
                cwd=work,
                check=True,
                capture_output=True,
                timeout=600,
            )
            if proc.stderr:
                logger.debug("ffmpeg stderr: %s", proc.stderr.decode(errors="replace")[:500])
    except subprocess.CalledProcessError as e:
        err_tail = (e.stderr or b"").decode(errors="replace")[-400:]
        logger.warning(
            "Ugoira GIF conversion failed for %s (exit %s): %s",
            zip_path,
            e.returncode,
            err_tail or e,
        )
        if dest.is_file():
            try:
                dest.unlink()
            except OSError:
                pass
        return False
    except subprocess.TimeoutExpired as e:
        logger.warning("Ugoira GIF conversion timed out for %s: %s", zip_path, e)
        if dest.is_file():
            try:
                dest.unlink()
            except OSError:
                pass
        return False
    except (OSError, zipfile.BadZipFile) as e:
        logger.warning("Ugoira GIF conversion failed for %s: %s", zip_path, e)
        if dest.is_file():
            try:
                dest.unlink()
            except OSError:
                pass
        return False
    return dest.is_file() and dest.stat().st_size > 0


def ugoira_zip_to_gif(
    zip_path: Path,
    meta_body: Dict[str, Any],
    dest: Path,
) -> bool:
    """GIF from ugoira ZIP: Pillow first (bundled dep), then ffmpeg if available."""
    if ugoira_zip_to_gif_pillow(zip_path, meta_body, dest):
        return True
    return ugoira_zip_to_gif_ffmpeg(zip_path, meta_body, dest)


def download_ugoira_for_illust(
    session: requests.Session,
    illust_id: str,
    title: str,
    root: Path,
    *,
    ugoira_format: PixivUgoiraFormat = "gif",
    skip_if_exists: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> List[Path]:
    """Download ugoira ZIP and/or GIF for one illustration."""
    meta = fetch_ugoira_meta(session, illust_id)
    zip_url = pick_ugoira_zip_url(meta)
    if not zip_url:
        raise RuntimeError("No ugoira ZIP URL in ugoira_meta (originalSrc/src missing)")

    referer = illust_artwork_referer(illust_id)
    want_zip = ugoira_format in ("zip", "both")
    want_gif = ugoira_format in ("gif", "both")
    zip_dest = root / _safe_filename(title, illust_id, 0, ".zip")
    gif_dest = root / _safe_filename(title, illust_id, 0, ".gif")
    saved: List[Path] = []

    zip_ready = skip_if_exists and want_zip and _dest_is_complete(zip_dest)
    if want_zip and not zip_ready:
        if download_binary(
            session,
            zip_url,
            zip_dest,
            referer=referer,
            skip_if_exists=skip_if_exists,
            cancel_check=cancel_check,
        ):
            saved.append(zip_dest)
    elif want_zip and zip_dest.is_file():
        saved.append(zip_dest)

    if want_gif:
        gif_exists = skip_if_exists and _dest_is_complete(gif_dest)
        if gif_exists:
            saved.append(gif_dest)
        else:
            try:
                if not zip_dest.is_file() or zip_dest.stat().st_size == 0:
                    download_binary(
                        session,
                        zip_url,
                        zip_dest,
                        referer=referer,
                        skip_if_exists=False,
                        cancel_check=cancel_check,
                    )
                if ugoira_zip_to_gif(zip_dest, meta, gif_dest):
                    saved.append(gif_dest)
                elif not want_zip:
                    raise RuntimeError(
                        "Downloaded ugoira frames but GIF conversion failed. "
                        "Run: pip install Pillow — or install ffmpeg / drop ffmpeg.exe "
                        "in %LOCALAPPDATA%\\HamsterScraper\\bin\\"
                    )
            finally:
                if not want_zip and zip_dest.is_file():
                    try:
                        zip_dest.unlink()
                    except OSError:
                        pass

    return saved


def download_manga_for_illust(
    session: requests.Session,
    illust_id: str,
    file_title: str,
    root: Path,
    body: Dict[str, Any],
    *,
    skip_if_exists: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> List[Path]:
    """Download all manga pages into one ZIP (post title on the archive name)."""
    urls = fetch_illust_image_urls(session, illust_id, body=body)
    if not urls:
        raise RuntimeError("No image URLs for manga illust")
    zip_dest = root / _manga_zip_filename(file_title, illust_id)
    if skip_if_exists and _dest_is_complete(zip_dest):
        return [zip_dest]

    referer = illust_artwork_referer(illust_id)
    work = Path(tempfile.mkdtemp(prefix="pixiv_manga_"))
    zip_part = zip_dest.with_suffix(".zip.part")
    try:
        pages_ok = 0
        for i, img_url in enumerate(urls):
            if cancel_check and cancel_check():
                raise DownloadCancelled("Download cancelled by user.")
            ext = _ext_from_url(img_url)
            member = f"{i:03d}{ext}"
            page_path = work / member
            if download_binary(
                session,
                img_url,
                page_path,
                referer=referer,
                skip_if_exists=False,
                cancel_check=cancel_check,
            ):
                pages_ok += 1
        if pages_ok != len(urls):
            raise RuntimeError(
                f"Manga download incomplete for {illust_id}: {pages_ok}/{len(urls)} pages"
            )
        zip_dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            zip_part, "w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            for page_path in sorted(work.iterdir()):
                if page_path.is_file():
                    zf.write(page_path, page_path.name)
        if zip_part.is_file() and zip_part.stat().st_size > 0:
            zip_part.replace(zip_dest)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        try:
            zip_part.unlink(missing_ok=True)
        except OSError:
            pass

    if _dest_is_complete(zip_dest):
        return [zip_dest]
    return []


def download_bookmark_works(
    session: requests.Session,
    works: List[Dict[str, Any]],
    download_dir: str,
    session_id: str,
    *,
    max_workers: int = 4,
    skip_if_exists: bool = True,
    ugoira_format: PixivUgoiraFormat = "gif",
    cancel_check: Optional[Callable[[], bool]] = None,
    settings: Optional[Any] = None,
) -> bool:
    """Download bookmarked works. Returns True if stopped early by user cancel."""
    tracker = get_tracker(session_id)
    root = Path(download_dir)
    cancelled = False

    def _one(work: Dict[str, Any]) -> None:
        ws = _worker_session(session)
        illust_id = str(work.get("id") or work.get("illustId") or "")
        title = str(work.get("title") or work.get("illustTitle") or illust_id)
        page_url = f"{PIXIV_HOME}artworks/{illust_id}"
        if not illust_id:
            return
        if cancel_check and cancel_check():
            return
        tracker.add_urls([page_url], [title])
        tracker.start_download(page_url)
        try:
            body = fetch_illust_body(ws, illust_id)
            display_title = work_from_illust_body(body, illust_id)["title"]
            file_title = resolve_pixiv_filename_title(
                display_title, illust_id, settings=settings
            )
            saved: List[Path] = []
            if is_ugoira_body(body):
                saved = download_ugoira_for_illust(
                    ws,
                    illust_id,
                    file_title,
                    root,
                    ugoira_format=ugoira_format,
                    skip_if_exists=skip_if_exists,
                    cancel_check=cancel_check,
                )
            elif is_manga_body(body):
                saved = download_manga_for_illust(
                    ws,
                    illust_id,
                    file_title,
                    root,
                    body,
                    skip_if_exists=skip_if_exists,
                    cancel_check=cancel_check,
                )
            else:
                urls = fetch_illust_image_urls(ws, illust_id, body=body)
                if not urls:
                    tracker.fail_download(page_url, "No image URLs in illust detail")
                    return
                for i, img_url in enumerate(urls):
                    if cancel_check and cancel_check():
                        raise DownloadCancelled("Download cancelled by user.")
                    ext = _ext_from_url(img_url)
                    dest = root / _safe_filename(
                        file_title, illust_id, i, ext, image_url=img_url
                    )
                    if download_image(
                        ws,
                        img_url,
                        dest,
                        skip_if_exists=skip_if_exists,
                        cancel_check=cancel_check,
                    ):
                        saved.append(dest)
            if saved:
                total_sz = sum(p.stat().st_size for p in saved)
                tracker.complete_download(page_url, str(saved[0]), total_sz)
            else:
                tracker.fail_download(page_url, "All pages failed or empty files")
        except DownloadCancelled:
            raise
        except Exception as e:
            logger.warning("Pixiv download failed for %s: %s", illust_id, e)
            tracker.fail_download(page_url, str(e))

    executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
    futures = []
    try:
        for work in works:
            if cancel_check and cancel_check():
                cancelled = True
                break
            futures.append(executor.submit(_one, work))

        for fut in as_completed(futures):
            if cancel_check and cancel_check():
                cancelled = True
                break
            try:
                fut.result(timeout=3600)
            except DownloadCancelled:
                cancelled = True
                break
            except Exception as e:
                logger.error("Pixiv worker error: %s", e)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return cancelled or bool(cancel_check and cancel_check())
