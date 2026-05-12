"""
Wait for the user to finish logging in via Selenium Chrome when stdin is not interactive
(e.g. SHUCK3R started with `python main.py` / uvicorn — no terminal attached to `input()`).

Browser users confirm on the session progress page (POST /api/selenium-login/confirm).
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Dict

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_events: Dict[str, threading.Event] = {}


def wait_for_user_after_chrome_login(session_id: str, tty_message: str) -> None:
    """
    Block until the user confirms login.

    - Interactive terminal: same as ``input()`` after printing ``tty_message``.
    - No TTY (typical web server): wait until ``confirm_chrome_login_done(session_id)``
      is called (progress page button), up to 2 hours.
    """
    stdin = getattr(sys, "stdin", None)
    if stdin is not None and stdin.isatty():
        print(tty_message, flush=True)
        try:
            input()
        except EOFError as e:
            logger.error("EOF while waiting for Enter after Chrome login.")
            raise RuntimeError(
                "No terminal input available (EOF). If you started SHUCK3R from the web UI, "
                "switch to the download progress tab and click **Continue after Chrome login** "
                "when you have signed in in Chrome — do not rely on pressing Enter in a server "
                "console."
            ) from e
        return

    ev = threading.Event()
    with _lock:
        _events[session_id] = ev
    logger.info(
        "Chrome login: waiting for web confirmation (session %s). "
        "Use the progress page button “Continue after Chrome login”.",
        session_id,
    )
    try:
        if not ev.wait(timeout=7200.0):
            raise TimeoutError(
                "Timed out after 2 hours waiting for “Continue after Chrome login” on the progress page."
            )
    finally:
        with _lock:
            _events.pop(session_id, None)


def confirm_chrome_login_done(session_id: str) -> bool:
    """Signal the waiting workflow for ``session_id``. Returns False if nothing was waiting."""
    with _lock:
        ev = _events.get(session_id)
        if ev is None:
            return False
        ev.set()
        return True


def is_waiting(session_id: str) -> bool:
    with _lock:
        return session_id in _events
