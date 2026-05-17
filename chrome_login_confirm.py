"""
Progress-page confirmation for legacy Selenium Chrome login (Web UI / uvicorn).

Never uses stdin ``input()`` — hosted servers have no interactive terminal.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Bump when login-wait behavior changes (logged at startup and on each wait).
CHROME_LOGIN_WAIT_VERSION = 3

CHROME_LOGIN_PROGRESS_HINT = (
    "When Chrome shows you are signed in, click **Continue after Chrome login** on this "
    "progress page. Keep the Chrome window open until you do."
)

_lock = threading.Lock()
_events: Dict[str, threading.Event] = {}


def chrome_login_progress_hint() -> str:
    return CHROME_LOGIN_PROGRESS_HINT


def wait_for_chrome_login(session_id: str) -> None:
    """Block until POST /api/selenium-login/confirm signals this session (up to 2 hours)."""
    ev = threading.Event()
    with _lock:
        _events[session_id] = ev
    logger.info(
        "[chrome_login v%s] Waiting for progress-page confirm (session %s). "
        "Click “Continue after Chrome login” on your download tab.",
        CHROME_LOGIN_WAIT_VERSION,
        session_id,
    )
    try:
        if not ev.wait(timeout=7200.0):
            raise TimeoutError(
                "Timed out after 2 hours waiting for “Continue after Chrome login” on the progress page."
            )
        logger.info(
            "[chrome_login v%s] Progress-page confirm received (session %s).",
            CHROME_LOGIN_WAIT_VERSION,
            session_id,
        )
    finally:
        with _lock:
            _events.pop(session_id, None)


def confirm_chrome_login(session_id: str) -> bool:
    with _lock:
        ev = _events.get(session_id)
        if ev is None:
            logger.warning(
                "[chrome_login v%s] Confirm for %s but no waiter (active: %s)",
                CHROME_LOGIN_WAIT_VERSION,
                session_id,
                list(_events.keys())[:8],
            )
            return False
        ev.set()
        return True


def is_waiting(session_id: str) -> bool:
    with _lock:
        return session_id in _events
