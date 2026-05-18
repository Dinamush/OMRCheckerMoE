"""
Resolve read-only bundle assets vs writable per-user data (SHUCK3R).

- RESOURCE_ROOT: templates/, static/ (MEIPASS when frozen).
- USER_DATA_DIR (via get_user_data_dir): logs/, downloads/ under the current user's profile
  (Windows %LOCALAPPDATA%\\HamsterScraper, macOS Application Support, Linux XDG).
  Folder names are legacy; override with env HAMSTER_USER_DATA if needed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_resource_root() -> Path:
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_user_data_dir() -> Path:
    """
    Per-user writable root so installs work for every Windows/macOS/Linux account
    without relying on the repo folder or exe location.
    """
    override = os.environ.get("HAMSTER_USER_DATA", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "").strip()
        if local:
            return (Path(local).expanduser().resolve() / "HamsterScraper")
        return (Path.home() / "AppData" / "Local" / "HamsterScraper").resolve()

    if sys.platform == "darwin":
        return (Path.home() / "Library" / "Application Support" / "HamsterScraper").resolve()

    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    if xdg:
        return (Path(xdg).expanduser().resolve() / "hamster_scraper")
    return (Path.home() / ".local" / "share" / "hamster_scraper").resolve()


def bundled_ffmpeg_path() -> Optional[Path]:
    """Optional ffmpeg shipped next to the app: ``{resource}/bin/ffmpeg.exe`` or user ``bin/``."""
    names = ("ffmpeg.exe", "ffmpeg") if sys.platform == "win32" else ("ffmpeg",)
    for base in (get_resource_root(), get_user_data_dir()):
        for name in names:
            candidate = base / "bin" / name
            if candidate.is_file():
                return candidate
    return None


def ensure_runtime_cwd() -> None:
    """Normalize cwd when launching the desktop app (Explorer often starts with System32)."""
    if is_frozen():
        data = get_user_data_dir()
        data.mkdir(parents=True, exist_ok=True)
        os.chdir(data)
