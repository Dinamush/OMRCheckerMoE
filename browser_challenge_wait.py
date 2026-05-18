"""Progress-page confirmation when a visible browser must solve a challenge."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict

logger = logging.getLogger(__name__)

CHALLENGE_WAIT_VERSION = 1

CHALLENGE_PROGRESS_HINT = (
    "A security check or login wall appeared. Complete it in the Chrome window, then click "
    "**Continue after browser challenge** on this page."
)

_lock = threading.Lock()
_events: Dict[str, threading.Event] = {}


def challenge_progress_hint() -> str:
    return CHALLENGE_PROGRESS_HINT


def wait_for_challenge_cleared(session_id: str) -> None:
    ev = threading.Event()
    with _lock:
        _events[session_id] = ev
    logger.info(
        "[challenge_wait v%s] Waiting for user (session %s). Use **Continue after browser challenge**.",
        CHALLENGE_WAIT_VERSION,
        session_id,
    )
    try:
        if not ev.wait(timeout=7200.0):
            raise TimeoutError(
                "Timed out after 2 hours waiting for **Continue after browser challenge**."
            )
    finally:
        with _lock:
            _events.pop(session_id, None)


def confirm_challenge_cleared(session_id: str) -> bool:
    with _lock:
        ev = _events.get(session_id)
        if ev is None:
            return False
        ev.set()
        return True


def is_waiting(session_id: str) -> bool:
    with _lock:
        return session_id in _events
