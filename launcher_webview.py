#!/usr/bin/env python3
"""Same as `launcher.py` (embedded window is already the default). Kept for older docs / alternate PyInstaller entry."""

from __future__ import annotations

import multiprocessing

from launcher import run_desktop

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_desktop(external_browser=False)
