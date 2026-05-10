#!/usr/bin/env python3
"""
Desktop launcher: local server + embedded window (pywebview on Windows).

Use HAMSTER_EXTERNAL_BROWSER=1 to open the system browser instead (dev / fallback).
Use `python main.py` for a normal dev server (0.0.0.0).
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
import webbrowser
from urllib.error import URLError
from urllib.request import urlopen

from paths import ensure_runtime_cwd, get_user_data_dir

# Help PyInstaller trace pywebview when building HamsterScraper.exe
try:
    import webview as _webview_pkg  # noqa: F401
except ImportError:
    pass

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _fatal_exit(message: str, exc: BaseException | None = None) -> None:
    """No console in frozen GUI builds — log + optional Windows message box."""
    lines = [message]
    if exc is not None:
        lines.append("")
        lines.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    text = "\n".join(lines)

    if _is_frozen():
        try:
            ud = get_user_data_dir()
            ud.mkdir(parents=True, exist_ok=True)
            log_path = ud / "hamster_scraper_error.log"
            log_path.write_text(text, encoding="utf-8")
        except OSError:
            pass
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, text[:2000], "Hamster Scraper", 0x10)
        except Exception:
            pass
    else:
        print(text, file=sys.stderr)
    sys.exit(1)


def _http_ok(resp) -> bool:
    code = getattr(resp, "status", None)
    if code is None and hasattr(resp, "getcode"):
        code = resp.getcode()
    return code == 200


def wait_for_server_ready(host: str, port: int, timeout: float = 30.0) -> bool:
    url = f"http://{host}:{port}/"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as resp:
                if _http_ok(resp):
                    return True
        except (URLError, OSError):
            time.sleep(0.12)
    return False


def start_uvicorn_background(app, host: str, port: int) -> tuple[threading.Thread, list]:
    errors: list[BaseException] = []

    def _serve() -> None:
        try:
            import uvicorn

            # Frozen / embedded UI: no access-log spam; dev keeps richer logs via python main.py
            quiet = _is_frozen() or os.environ.get("HAMSTER_QUIET_UVICORN", "").lower() in (
                "1",
                "true",
                "yes",
            )
            uvicorn.run(
                app,
                host=host,
                port=port,
                log_level="warning" if quiet else "info",
                access_log=not quiet,
            )
        except BaseException as e:
            traceback.print_exc(file=sys.stderr)
            errors.append(e)

    thread = threading.Thread(target=_serve, daemon=True, name="uvicorn")
    thread.start()
    return thread, errors


def run_desktop(*, external_browser: bool | None = None) -> None:
    ensure_runtime_cwd()

    if external_browser is None:
        external_browser = os.environ.get("HAMSTER_EXTERNAL_BROWSER", "").lower() in (
            "1",
            "true",
            "yes",
        )

    host = os.environ.get("HAMSTER_HOST", DEFAULT_HOST)
    port = int(os.environ.get("HAMSTER_PORT", str(DEFAULT_PORT)))

    from main import app

    _thread, uvicorn_errors = start_uvicorn_background(app, host, port)
    time.sleep(0.4)
    if uvicorn_errors:
        _fatal_exit(
            f"The server failed to start. If port {port} is busy, set HAMSTER_PORT.\n",
            uvicorn_errors[0],
        )

    if not wait_for_server_ready(host, port):
        _fatal_exit("Server did not become ready in time.")

    url = f"http://{host}:{port}/"

    if not external_browser:
        try:
            import webview

            webview.create_window("Hamster Scraper", url, width=1100, height=800)
            webview.start(debug=False)
            return
        except ImportError:
            if _is_frozen():
                _fatal_exit(
                    "Embedded window (pywebview) is missing from this build.\n"
                    "Rebuild with pywebview included or set HAMSTER_EXTERNAL_BROWSER=1."
                )
            print(
                "pywebview not installed; opening your default browser instead.\n"
                "Install with: pip install pywebview",
                file=sys.stderr,
            )
        except Exception as e:
            if _is_frozen():
                _fatal_exit("Could not open the app window (WebView2 / pywebview).", e)
            print(f"Embedded window failed ({e}); opening browser.", file=sys.stderr)

    webbrowser.open(url)
    print(f"Hamster Scraper is running at {url}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    run_desktop()
