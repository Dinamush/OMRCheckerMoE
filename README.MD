<div align="center">

```
███████╗██╗  ██╗██╗   ██╗ ██████╗██╗  ██╗███████╗██████╗
██╔════╝██║  ██║██║   ██║██╔════╝██║ ██╔╝╚════██║██╔══██╗
███████╗███████║██║   ██║██║     █████╔╝     ██╔╝██████╔╝
╚════██║██╔══██║██║   ██║██║     ██╔═██╗    ██╔╝ ██╔══██╗
███████║██║  ██║╚██████╔╝╚██████╗██║  ██╗   ██║  ██║  ██║
╚══════╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝   ╚═╝  ╚═╝  ╚═╝
```

**Download your favorites. Skip what you have. Delete the rest.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-00ff88?style=flat-square)](#)
[![Tests](https://img.shields.io/badge/Tests-passing-00ff88?style=flat-square)](#testing)

</div>

---

SHUCK3R is a **local web application** that downloads your saved favorites from PornHub and xHamster. Log in once in the browser the app opens; SHUCK3R scrapes every URL from your favorites list, downloads each video, and tracks progress in real time. Point it at a folder of videos you already own — it skips anything whose title matches a file you already have, so you never re-download.

A built-in **duplicate file checker** lets you scan any directory, review fuzzy-matched duplicate groups side-by-side, keep one copy per group, and permanently delete the rest — all from the same UI.

---

## Table of contents

- [Features](#features)
- [Requirements](#requirements)
- [Install](#install)
- [Running the app](#running-the-app)
  - [Browser mode](#browser-mode-python-mainpy)
  - [Desktop window mode](#desktop-window-mode-python-launcherpy)
  - [Environment variables](#environment-variables)
  - [Where files are saved](#where-files-are-saved)
- [How to use](#how-to-use)
  - [Downloading favorites](#downloading-favorites)
  - [Duplicate file checker](#duplicate-file-checker)
- [Building a Windows executable](#building-a-windows-executable)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Features

| Capability | Detail |
|---|---|
| **Favorites scraper** | Paginates through your full PornHub or xHamster favorites list |
| **Smart skip** | Point at an existing folder — files whose titles match are skipped |
| **Progress dashboard** | Live session view: workflow timeline, metrics, and recent log lines |
| **Headless mode** | Login and scraping run without a visible browser window |
| **Duplicate checker** | Fuzzy-match by name + size across any directory; keep one, delete the rest |
| **Desktop launcher** | Embedded pywebview window — no separate browser tab needed |
| **Portable exe** | Single-file `SHUCK3R.exe` via PyInstaller; data stays in your user profile |

---

## Requirements

| Dependency | Notes |
|---|---|
| **Python 3.10+** | 3.13 recommended |
| **Google Chrome** | Must be installed — used by Selenium for login and scraping |
| **ChromeDriver** | Auto-downloaded on first run via `webdriver-manager` |
| **WebView2** | Desktop window only (Windows) — pre-installed on Windows 11 |

---

## Install

```bash
# 1. Clone the repo
git clone <repo-url>
cd SHuck3r

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the app

### Browser mode (`python main.py`)

Starts the API server and serves the UI at **http://localhost:8001**. Keep the terminal open while you use the app.

```bash
python main.py
```

Open **http://localhost:8001** in any browser. Press **Ctrl+C** in the terminal to stop.

---

### Desktop window mode (`python launcher.py`)

Starts the API on loopback and opens the UI in an **embedded window** (pywebview / Edge WebView2) — no separate browser tab.

```bash
python launcher.py
```

**Login flow in the desktop window:**

1. Submit the download form — a **site login popup** opens.
2. Log in on the site. Use the menu bar (**SHUCK3R → Show main window / Show site login window**) to switch between windows if needed.
3. Click **Continue download — I'm logged in** (or use **SHUCK3R → Done logging in**) when finished. Cookies are saved and the download begins.
4. The login window closes; the main window returns to the session dashboard.

> **xHamster note:** Selenium runs headless by default after login. Set `HAMSTER_SHOW_SELENIUM=1` to see the Chrome window.

---

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `HAMSTER_HOST` | `0.0.0.0` | Bind address. Use `127.0.0.1` for local-only access. |
| `HAMSTER_PORT` | `8001` | Port the server listens on. |
| `HAMSTER_USER_DATA` | *(platform default)* | Override the root folder for downloads and logs. |
| `HAMSTER_EMBEDDED_LOGIN` | `1` | Set to `0` to use a separate Chrome window for login instead of the popup WebView. |
| `HAMSTER_SHOW_SELENIUM` | `0` | Set to `1` to show the Chrome window during xHamster scraping. |
| `HAMSTER_EXTERNAL_BROWSER` | `0` | Set to `1` to open the UI in your default browser instead of the embedded window. |

**Examples (Windows PowerShell):**

```powershell
# Local-only access on a custom port
$env:HAMSTER_HOST = "127.0.0.1"; $env:HAMSTER_PORT = "8002"; python main.py

# Custom data directory
$env:HAMSTER_USER_DATA = "D:\MyVideos"; python main.py
```

---

### Where files are saved

Downloads and logs are stored in a **per-user folder**, not inside the repo:

| Platform | Path |
|---|---|
| Windows | `%LOCALAPPDATA%\HamsterScraper\downloads\` and `...\logs\` |
| macOS | `~/Library/Application Support/HamsterScraper/` |
| Linux | `~/.local/share/hamster_scraper/` |

Override the root with the `HAMSTER_USER_DATA` environment variable (absolute path).

---

## How to use

### Downloading favorites

1. **Open the app** — go to `http://localhost:8001`.
2. **Choose a site** — select **PornHub** or **xHamster**.
3. **Optional: set an existing library folder** — paste the path to videos you already have. SHUCK3R skips favorites whose titles match files in that folder.
4. **Optional: enable headless mode** — runs login and scraping without a visible browser window.
5. **Click Start Download.**
6. **Log in** in the browser (or popup) window that opens. SHUCK3R then:
   - Paginates through all your favorites to collect every video URL.
   - Downloads each video with progress tracking.
7. **Monitor progress** on the live session dashboard — workflow timeline, counts, and recent log lines.
8. From the dashboard you can **open the full log**, **export progress as JSON**, or **start a new download**.

---

### Duplicate file checker

1. Navigate to **`http://localhost:8001/duplicates`** (or use the nav).
2. **Enter a directory path** to scan (leave empty to use the default downloads folder).
3. **Adjust the similarity threshold** — higher = stricter (fewer results).
4. **Click Scan for Duplicates.**
5. For each group, check the file you want to **keep** (one per group is pre-selected).
6. Click **Delete unchecked files** — a confirmation modal lists every file before deletion. This action cannot be undone.

---

## Building a Windows executable

Produces a single portable **`release\SHUCK3R.exe`**. Chrome must still be installed on the target machine.

```powershell
# Install runtime + build dependencies
pip install -r requirements.txt
pip install -r requirements-build.txt

# Build
.\build_windows.ps1
```

The exe is written to `release\SHUCK3R.exe`. Run it directly — data goes to `%LOCALAPPDATA%\HamsterScraper\`, not next to the exe. Startup errors are written to `hamster_scraper_error.log` in that folder.

**Manual build command:**

```powershell
pyinstaller hamster_scraper.spec --distpath ./release --workpath build/pyinstaller --noconfirm
```

**Optional second exe** (`SHUCK3RWebView.exe`):

```powershell
pyinstaller hamster_scraper_webview.spec --distpath ./release --workpath build/pyinstaller_webview --noconfirm
```

> `build/` and `release/*.exe` are gitignored. The `release/` directory is tracked with `.gitkeep` so the path exists in fresh clones. First builds may hit missing modules — add names to `hiddenimports` in the `.spec` file and rebuild.

---

## Project structure

```
SHuck3r/
├── main.py                      # FastAPI app — `python main.py` starts the server
├── launcher.py                  # Desktop launcher: loopback API + embedded window
├── launcher_webview.py          # Thin wrapper for the webview build
├── webview_login_bridge.py      # Popup WebView site login + cookie export
├── workflow_heuristics.py       # Download workflow state machine
├── existing_library.py          # Fuzzy skip-list against an existing video folder
├── paths.py                     # RESOURCE_ROOT + per-user data dir resolution
├── progress_tracker.py          # Per-session download progress state
├── duplicate_finder.py          # Fuzzy duplicate grouping and cluster logic
├── check_requirements.py        # Runtime package version validation
│
├── templates/                   # Jinja2 HTML templates
│   ├── base.html                # Shared layout shell (Ghost Protocol dark theme)
│   ├── index.html               # Download form
│   ├── submitted.html           # Live session dashboard
│   └── duplicates.html          # Duplicate checker UI
│
├── static/
│   ├── style.css                # Design tokens + Ghost Protocol dark theme
│   └── script.js                # Shared client-side logic
│
├── tests/
│   └── test_duplicate_finder.py # pytest suite
│
├── requirements.txt             # Runtime dependencies
├── requirements-build.txt       # Build-only dependencies (PyInstaller, pywebview)
├── build_windows.ps1            # One-command Windows exe build
├── hamster_scraper.spec         # PyInstaller spec — browser launcher
├── hamster_scraper_webview.spec # PyInstaller spec — webview launcher
│
├── docs/                        # Implementation notes and site structure docs
├── archive/                     # Legacy standalone scripts (not used by main.py)
└── website-code/                # Reference HTML snapshots of scraped pages
```

---

## Testing

```bash
python -m pytest tests/ -v
```

Tests cover the duplicate-finder logic. No network or browser access required.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| **Port 8001 already in use** | Set `HAMSTER_PORT` to a free port: `$env:HAMSTER_PORT=8002; python main.py` |
| **Chrome / ChromeDriver error** | Ensure Chrome is installed. ChromeDriver is auto-downloaded — check for version mismatches if Chrome was recently updated. |
| **Login required error** | You must be signed in to the site in the window SHUCK3R opens. The favorites list requires authentication. |
| **"Press Enter to exit"** | A startup error occurred. Read the traceback printed above that line. |
| **exe startup failure (no console)** | Check `hamster_scraper_error.log` inside `%LOCALAPPDATA%\HamsterScraper\` (Windows). |
| **Embedded window is blank** | WebView2 may be missing or outdated. Install the [WebView2 Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/) or use `python main.py` and open a browser instead. |
