"""
User-facing workflow timeline (linked-list of steps) for download sessions.

Each session follows an ordered chain of StepDefinitions (id → next_id). The UI
polls merged summary JSON and renders every step with status: pending, active,
done, or error — so users always know where they are in the pipeline.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

StepStatus = Literal["pending", "active", "done", "error"]

SiteId = Literal["xhamster", "pornhub"]
SITE_XHAMSTER: SiteId = "xhamster"
SITE_PORNHUB: SiteId = "pornhub"


@dataclass(frozen=True)
class StepDefinition:
    """One node in the workflow linked list."""

    id: str
    title: str
    user_message: str
    next_id: Optional[str]


def _chain(*steps: Tuple[str, str, str]) -> List[StepDefinition]:
    """Build StepDefinition list with next_id wired sequentially."""
    out: List[StepDefinition] = []
    ids = [s[0] for s in steps]
    for i, (sid, title, msg) in enumerate(steps):
        nxt = ids[i + 1] if i + 1 < len(ids) else None
        out.append(StepDefinition(id=sid, title=title, user_message=msg, next_id=nxt))
    return out


# --- xHamster: embedded WebView login ---
_XH_EMB = _chain(
    (
        "boot",
        "Session queued",
        "Your download job is running. This screen updates automatically — no need to refresh.",
    ),
    (
        "site_login",
        "Sign in to the site",
        "Use the site login window (or this window if login opened here). When you are signed in, tap Continue download — I'm logged in so cookies can be saved.",
    ),
    (
        "save_cookies",
        "Saving your session",
        "Cookies are being written for yt-dlp and the automation browser. The login window may close — that is normal.",
    ),
    (
        "browser_start",
        "Starting automation browser",
        "Chrome is launching to read your favorites (hidden by default on xHamster so it does not cover SHUCK3R).",
    ),
    (
        "collect_urls",
        "Scanning your favorites",
        "Walking each favorites page and collecting video links. Counters may stay at zero until this phase finishes.",
    ),
    (
        "download",
        "Downloading videos",
        "Resolving streams and saving files with yt-dlp in batches. Watch Completed / Failed above.",
    ),
    (
        "wrap_up",
        "Closing browser",
        "Shutting down automation and finalizing logs.",
    ),
    (
        "complete",
        "Session finished",
        "All queued work for this run is done. Check the full log if any items failed.",
    ),
)

# --- xHamster: legacy Selenium login window ---
_XH_LEGACY = _chain(
    (
        "boot",
        "Session queued",
        "Your download job is running. This screen updates automatically.",
    ),
    (
        "chrome_login",
        "Log in in Chrome",
        "A Chrome window should open — log in to xHamster there. When finished, return to the terminal or prompt as instructed.",
    ),
    (
        "save_cookies",
        "Saving cookies",
        "Session cookies are saved from Chrome for downloads.",
    ),
    (
        "browser_start",
        "Browser ready",
        "Automation is ready to read your favorites.",
    ),
    (
        "collect_urls",
        "Scanning your favorites",
        "Collecting video links from every favorites page.",
    ),
    (
        "download",
        "Downloading videos",
        "Saving videos with yt-dlp in batches.",
    ),
    (
        "wrap_up",
        "Closing browser",
        "Shutting down automation.",
    ),
    (
        "complete",
        "Session finished",
        "This run is complete.",
    ),
)

# --- PornHub: embedded ---
_PH_EMB = _chain(
    (
        "boot",
        "Session queued",
        "Your download job is running. Keep this page open.",
    ),
    (
        "site_login",
        "Sign in to Pornhub",
        "Use the site login window. When signed in, tap Continue download — I'm logged in (or use the SHUCK3R menu).",
    ),
    (
        "extract_list",
        "Reading your favorites list",
        "yt-dlp is pulling video URLs from your favorites playlist.",
    ),
    (
        "download",
        "Downloading videos",
        "Saving files to your downloads folder.",
    ),
    (
        "wrap_up",
        "Finishing",
        "Cleaning up session state.",
    ),
    (
        "complete",
        "Session finished",
        "This run is complete.",
    ),
)

# --- PornHub: legacy (terminal Enter) ---
_PH_LEGACY = _chain(
    (
        "boot",
        "Session queued",
        "Your download job is running.",
    ),
    (
        "chrome_login",
        "Log in in Chrome",
        "Log in to Pornhub in the opened browser, then press Enter in the terminal when prompted.",
    ),
    (
        "extract_list",
        "Reading your favorites list",
        "Extracting video URLs from your favorites.",
    ),
    (
        "download",
        "Downloading videos",
        "Saving files with yt-dlp.",
    ),
    (
        "wrap_up",
        "Finishing",
        "Cleaning up.",
    ),
    (
        "complete",
        "Session finished",
        "This run is complete.",
    ),
)


def _template(site: SiteId, embedded: bool) -> List[StepDefinition]:
    if site == SITE_XHAMSTER:
        return list(_XH_EMB if embedded else _XH_LEGACY)
    if site == SITE_PORNHUB:
        return list(_PH_EMB if embedded else _PH_LEGACY)
    raise ValueError(f"unknown site {site!r}")


class _Runtime:
    __slots__ = ("site", "embedded", "steps", "order", "status", "current_id", "detail", "error_message")

    def __init__(self, site: SiteId, embedded: bool) -> None:
        self.site = site
        self.embedded = embedded
        self.steps = _template(site, embedded)
        self.order = [s.id for s in self.steps]
        self.status: Dict[str, StepStatus] = {s.id: "pending" for s in self.steps}
        self.current_id = self.order[0]
        self.detail: Optional[str] = None
        self.error_message: Optional[str] = None
        self.status[self.current_id] = "active"

    def _apply_index(self, idx: int) -> None:
        for i, sid in enumerate(self.order):
            if i < idx:
                self.status[sid] = "done"
            elif i == idx:
                self.status[sid] = "active"
                self.current_id = sid
            else:
                self.status[sid] = "pending"

    def advance_to(self, step_id: str) -> None:
        if step_id not in self.order:
            return
        idx = self.order.index(step_id)
        self._apply_index(idx)

    def mark_error(self, message: str) -> None:
        self.error_message = message
        if self.current_id in self.status:
            self.status[self.current_id] = "error"

    def set_detail(self, detail: Optional[str]) -> None:
        self.detail = detail

    def to_payload(self) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        for s in self.steps:
            st = self.status.get(s.id, "pending")
            nodes.append(
                {
                    "id": s.id,
                    "title": s.title,
                    "user_message": s.user_message,
                    "next_id": s.next_id,
                    "status": st,
                }
            )
        return {
            "site": self.site,
            "embedded": self.embedded,
            "current_step_id": self.current_id,
            "detail": self.detail,
            "error_message": self.error_message,
            "steps": nodes,
        }


_lock = threading.Lock()
_sessions: Dict[str, _Runtime] = {}


def register_session(session_id: str, site: SiteId, *, embedded: bool) -> None:
    """Call from POST /download before returning redirect (so first poll has steps)."""
    with _lock:
        _sessions[session_id] = _Runtime(site, embedded)


def advance(session_id: str, step_id: str) -> None:
    with _lock:
        rt = _sessions.get(session_id)
        if rt is None:
            return
        rt.advance_to(step_id)


def set_detail(session_id: str, detail: Optional[str]) -> None:
    with _lock:
        rt = _sessions.get(session_id)
        if rt is None:
            return
        rt.set_detail(detail)


def mark_error(session_id: str, message: str) -> None:
    with _lock:
        rt = _sessions.get(session_id)
        if rt is None:
            return
        rt.mark_error(message)


def get_timeline(session_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        rt = _sessions.get(session_id)
        if rt is None:
            return None
        return rt.to_payload()


def complete_success(session_id: str) -> None:
    """Mark every step done (terminal green checklist)."""
    with _lock:
        rt = _sessions.get(session_id)
        if rt is None:
            return
        last = rt.order[-1]
        for sid in rt.order:
            rt.status[sid] = "done"
        rt.current_id = last
        rt.detail = None
        rt.error_message = None


def dispose_session(session_id: str) -> None:
    """Remove timeline after UI no longer needs it (optional)."""
    with _lock:
        _sessions.pop(session_id, None)
