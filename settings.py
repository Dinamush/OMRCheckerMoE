"""
Persistent SHUCK3R tuning (Tier 1–3 parity with operational patterns).

Stored as JSON next to logs/downloads under ``get_user_data_dir()``.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from paths import get_user_data_dir


CONFIG_FILENAME = "config.json"
QUEUE_SNAPSHOT_DIRNAME = "queue_snapshots"


@dataclass
class AppSettings:
    """All tunables saved to disk; omitted keys load as defaults."""

    # Tier 1
    download_directory: str = ""  # empty → default %(USER_DATA)/downloads
    max_parallel_downloads: int = 4
    video_quality: Literal["best", "720", "1080"] = "720"
    pixiv_ugoira_format: Literal["zip", "gif", "both"] = "gif"
    skip_existing_in_download_dir: bool = True
    persistent_cookies: bool = False

    # Tier 2
    page_delay_seconds: float = 0.0  # sleeps after Selenium page loads where applicable
    download_delay_seconds: float = 0.0  # between parallel worker submissions / batches
    delay_variance_seconds: float = 0.0  # random uniform additive to delays
    proxy_url: str = ""  # http(s):// or socks5:// — passed to yt-dlp / requests
    yt_dlp_retries: int = 3
    persist_queue_snapshots: bool = True

    # Tier 3
    watcher_enabled: bool = False
    watcher_interval_minutes: int = 60  # honoured by background watcher loop
    notify_on_complete: bool = False  # Windows informational MessageBox thread (frozen-friendly)

    # Browser / anti-bot (HDoujin-inspired)
    headless_scraping: bool = False  # scrape phase only; login is always interactive
    browser_profile_per_site: bool = True
    skip_login_if_cookies_valid: bool = True
    use_undetected_chrome: bool = True
    challenge_solver: Literal["manual", "flaresolverr"] = "manual"
    flaresolverr_base_url: str = "http://127.0.0.1:8191/v1"

    # PornHub: empty → resolve /users/<you>/videos/favorites after login (ph.py)
    pornhub_username: str = ""


DEFAULT_SETTINGS = AppSettings()

_VALID_QUALITY = frozenset({"best", "720", "1080"})
_VALID_UGOIRA_FORMAT = frozenset({"zip", "gif", "both"})
_VALID_CHALLENGE_SOLVER = frozenset({"manual", "flaresolverr"})


def _config_path(user_data_dir: Optional[Path] = None) -> Path:
    root = user_data_dir or get_user_data_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root / CONFIG_FILENAME


def _queue_snap_dir(user_data_dir: Optional[Path] = None) -> Path:
    root = user_data_dir or get_user_data_dir()
    d = root / QUEUE_SNAPSHOT_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_settings(user_data_dir: Optional[Path] = None) -> AppSettings:
    path = _config_path(user_data_dir)
    if not path.is_file():
        return AppSettings(**asdict(DEFAULT_SETTINGS))
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return AppSettings(**asdict(DEFAULT_SETTINGS))
    data: Dict[str, Any] = {f.name: getattr(DEFAULT_SETTINGS, f.name) for f in fields(AppSettings)}
    for k in data.keys():
        if k in raw:
            data[k] = raw[k]
    if data.get("video_quality") not in _VALID_QUALITY:
        data["video_quality"] = "720"
    ugoira_fmt = data.get("pixiv_ugoira_format")
    if ugoira_fmt == "webm":
        ugoira_fmt = "gif"
    if ugoira_fmt not in _VALID_UGOIRA_FORMAT:
        ugoira_fmt = "gif"
    data["pixiv_ugoira_format"] = ugoira_fmt
    if data.get("challenge_solver") not in _VALID_CHALLENGE_SOLVER:
        data["challenge_solver"] = "manual"
    try:
        out = AppSettings(**data)  # type: ignore[arg-type]
    except TypeError:
        out = AppSettings(**asdict(DEFAULT_SETTINGS))
    out.max_parallel_downloads = max(1, min(32, int(out.max_parallel_downloads)))
    out.yt_dlp_retries = max(0, min(50, int(out.yt_dlp_retries)))
    out.watcher_interval_minutes = max(1, min(10_080, int(out.watcher_interval_minutes)))
    out.page_delay_seconds = max(0.0, float(out.page_delay_seconds))
    out.download_delay_seconds = max(0.0, float(out.download_delay_seconds))
    out.delay_variance_seconds = max(0.0, float(out.delay_variance_seconds))
    return out


def save_settings(settings: AppSettings, user_data_dir: Optional[Path] = None) -> None:
    path = _config_path(user_data_dir)
    payload = asdict(settings)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def effective_download_directory(settings: Optional[AppSettings] = None) -> Path:
    s = settings or load_settings()
    user = get_user_data_dir()
    raw = (s.download_directory or "").strip()
    if not raw:
        return (user / "downloads").resolve()
    p = Path(raw).expanduser().resolve()
    return p


# Per-site subfolder under the download root (e.g. downloads/ph/).
SITE_DOWNLOAD_SUBDIRS: dict[str, str] = {
    "pornhub": "ph",
    "xhamster": "xh",
    "pixiv": "pixiv",
}


def site_download_directory(
    site: str, settings: Optional[AppSettings] = None
) -> Path:
    """Resolved path for one site's downloads: {download_root}/{ph|xh|pixiv|…}."""
    base = effective_download_directory(settings)
    key = (site or "").strip().lower()
    sub = SITE_DOWNLOAD_SUBDIRS.get(key, key or "other")
    return (base / sub).resolve()


def yt_dlp_format_string(settings: Optional[AppSettings] = None) -> str:
    s = settings or load_settings()
    if s.video_quality == "best":
        return "bestvideo+bestaudio/best"
    if s.video_quality == "1080":
        return "best[height<=1080]/bestvideo[height<=1080]+bestaudio/best[height<=1080]"
    return "best[height<=720]/bestvideo[height<=720]+bestaudio/best[height<=720]"


def apply_delay(base: float, variance: float) -> None:
    v = variance if variance > 0 else 0.0
    extra = random.uniform(-v, v) if v > 0 else 0.0
    t = max(0.0, base + extra)
    if t > 0:
        time.sleep(t)


def maybe_notify_complete(title: str, body: str, settings: Optional[AppSettings] = None) -> None:
    s = settings or load_settings()
    if not s.notify_on_complete:
        return
    import sys
    import threading

    if sys.platform != "win32":
        return

    def _box() -> None:
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                0,
                body[:1024],
                title[:128],
                0x40,  # MB_ICONINFORMATION
            )
        except Exception:
            pass

    threading.Thread(target=_box, daemon=True).start()


def save_queue_snapshot(session_id: str, site: str, urls: list[str], user_data_dir: Optional[Path] = None) -> Path:
    p = _queue_snap_dir(user_data_dir) / f"{session_id}.json"
    p.write_text(
        json.dumps({"site": site, "urls": urls, "saved_at_epoch": time.time()}, indent=2),
        encoding="utf-8",
    )
    return p


def load_queue_snapshot(session_id: str, user_data_dir: Optional[Path] = None) -> Optional[dict]:
    p = _queue_snap_dir(user_data_dir) / f"{session_id}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
