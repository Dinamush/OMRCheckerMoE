"""Backward-compatible re-exports — implementation lives in chrome_login_confirm."""

from __future__ import annotations

from chrome_login_confirm import (
    CHROME_LOGIN_PROGRESS_HINT,
    CHROME_LOGIN_WAIT_VERSION,
    chrome_login_progress_hint,
    confirm_chrome_login,
    is_waiting,
    wait_for_chrome_login,
)


def is_web_hosted() -> bool:
    import sys

    return bool(
        {"fastapi", "starlette", "uvicorn", "hypercorn", "daphne", "granian", "waitress"}
        & set(sys.modules.keys())
    )


def wait_for_user_after_chrome_login(session_id: str, tty_message: str) -> None:
    wait_for_chrome_login(session_id)


def confirm_chrome_login_done(session_id: str) -> bool:
    return confirm_chrome_login(session_id)
