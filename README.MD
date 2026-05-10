# SHUCK3R

Download videos from your **PornHub** and **xHamster** favorites. You log in once in a browser window; the app collects video URLs and downloads them with progress tracking. Optionally point at a folder of videos you already have: SHUCK3R **skips** favorites whose titles match those filenames and only downloads what’s missing.

---

## Requirements

- **Python 3.8+**
- **Chrome** (installed for Selenium/automation)
- **ChromeDriver** is installed automatically via `webdriver-manager` when you run the app
- **WebView2** (for the desktop app window on Windows — usually already installed on Windows 11)

---

## Install

1. Clone or download this repo and open a terminal in the project folder.

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Run the system

From the project root:

```bash
python main.py
```

The server starts at **http://0.0.0.0:8001** (all interfaces). Open in your browser:

**http://localhost:8001**

Leave the terminal open while you use the app.

### Bind address and port (`python main.py`)

- **`HAMSTER_HOST`** — default `0.0.0.0` (all interfaces). Use `127.0.0.1` if you only want local access.
- **`HAMSTER_PORT`** — default `8001`.

Example (local only): `set HAMSTER_HOST=127.0.0.1` then `python main.py`.

Logs and downloads use a **per-user data folder** (same for source and the `.exe`): Windows **`%LOCALAPPDATA%\HamsterScraper`** (`downloads\`, `logs\`), macOS **`~/Library/Application Support/HamsterScraper`**, Linux **`~/.local/share/hamster_scraper`**. Set **`HAMSTER_USER_DATA`** to force a different root. (Those folder names are legacy; the app is **SHUCK3R**.)

### Desktop launcher (`python launcher.py`)

Starts the API on **loopback** and opens the UI in an **embedded window** (pywebview / Edge WebView2), not a separate browser tab.

When you start a download, a **popup WebView window** opens for the site login while the **main window stays on SHUCK3R**. Use the menu to switch (**SHUCK3R → Show main window** / **Show site login window**), then **SHUCK3R → Done logging in — continue download** when finished (cookies are saved for yt-dlp / Selenium). If the popup cannot be created, login falls back to the main window. Set **`HAMSTER_EMBEDDED_LOGIN=0`** to use the old separate Chrome login instead.

For **xHamster**, after that in-app login step, Selenium Chrome runs **headless** by default so a full Chrome window does not cover the SHUCK3R window. Set **`HAMSTER_SHOW_SELENIUM=1`** if you want a visible Chrome window while it scrapes.

The packaged app asks for **Yes/No confirmation** after **Done logging in** so you can verify you finished signing in before cookies are saved.

Submitting the form **redirects** to a stable **`GET /download/session/{id}`** progress URL so the embedded browser reliably keeps the progress UI (then returns there again after login). That page is the **live session dashboard**: an ordered **workflow timeline** (linked steps with plain-language copy), live metrics, and recent log lines. **`/progress/{id}`** is JSON for tools only — use **Open full log** or **Load progress JSON** on the dashboard instead of opening `/progress/...` in a browser (you would only see raw JSON).

After signing in on the site popup, use the **Continue download — I'm logged in** button on that page (not only the SHUCK3R menu), especially if the menu bar is hidden. The **site login window closes** after cookies are saved; the main window stays on the dashboard.

- **`HAMSTER_EXTERNAL_BROWSER=1`** — use your **default browser** instead and keep a terminal open (useful for debugging).
- **`HAMSTER_HOST`** / **`HAMSTER_PORT`** — same as above.

The packaged **`SHUCK3R.exe`** uses this launcher with **no console window**. If startup fails, check **`hamster_scraper_error.log`** inside your **`HAMSTER_USER_DATA`** folder (default: `%LOCALAPPDATA%\HamsterScraper` on Windows); a message box may also appear.

---

## Building a Windows executable

The default build is a **single portable `release/SHUCK3R.exe`** (one-file PyInstaller), so the repo root stays source-only.

1. Install runtime and build deps:

   ```powershell
   pip install -r requirements.txt
   pip install -r requirements-build.txt
   ```

2. From the **project root**:

   ```powershell
   .\build_windows.ps1
   ```

   When it finishes, run **`.\release\SHUCK3R.exe`**. The app opens in **its own window**. Videos and logs go under **your user profile** (see above), not next to the exe. Startup errors are written to **`hamster_scraper_error.log`** in that same user-data folder.

   Equivalent manual command:

   ```powershell
   pyinstaller hamster_scraper.spec --distpath ./release --workpath build/pyinstaller --noconfirm
   ```

3. Optional second exe (**`SHUCK3RWebView.exe`**) — same embedded UI as the main build; only use **`hamster_scraper_webview.spec`** if you want a separate artifact name.

   ```powershell
   Remove-Item -Force ./release/SHUCK3RWebView.exe -ErrorAction SilentlyContinue
   pyinstaller hamster_scraper_webview.spec --distpath ./release --workpath build/pyinstaller_webview --noconfirm
   ```

**Notes:**

- When frozen, the launcher switches the working directory to your **user-data** folder so paths behave consistently.
- The exe itself can live anywhere (Desktop, `release\`, etc.); data still goes under the **per-user** paths above.
- **Chrome** must still be installed for downloads; PyInstaller does not bundle it.
- First builds may hit missing modules; add names to `hiddenimports` in the `.spec` file and rebuild.
- The bundled app skips `pip` dependency checks at startup.
- **`build/`** is gitignored; **`release/*.exe`** is gitignored. The **`release/`** folder is kept with **`.gitkeep`** so the path exists in fresh clones.

---

## How to use

1. **Open the app**  
   Go to http://localhost:8001 in your browser.

2. **Choose site**  
   Select **PornHub** or **xHamster** (your favorites from that site will be used).

3. **Optional: Run headlessly**  
   Check “Run Headlessly” if you don’t want to see the browser window (login and scraping happen in the background).

4. **Start download**  
   Click **Start Download**.

5. **Log in when the browser opens**  
   A Chrome window will open. Log in to the selected site if you aren’t already. The app will then:
   - Collect all video URLs from your favorites (and follow “Next” for multiple pages).
   - Download videos into the `downloads` folder with progress tracking.

6. **After submission**  
   You’re taken to a “Download Started” page where you can:
   - **View Logs** – open the session log file
   - **View Progress** – see live progress in the browser
   - **Download Progress JSON** – export progress as JSON
   - **Start New Download** – return to the form

7. **Where files go**  
   Default **`downloads`** and **`logs`** are under your **user profile**, not the repo:

   - **Windows:** `%LOCALAPPDATA%\HamsterScraper\downloads` and `...\logs`
   - **macOS:** `~/Library/Application Support/HamsterScraper/`
   - **Linux:** `~/.local/share/hamster_scraper/`  

   Override the root with **`HAMSTER_USER_DATA`** (absolute path).

---

## Project layout (what the app uses)

```
hamster_scraper/
├── main.py              # Entry point: `python main.py` starts the server
├── launcher.py          # Desktop: loopback + default browser
├── launcher_webview.py  # Thin wrapper around launcher (embedded window)
├── webview_login_bridge.py  # Embedded WebView site login + cookie export (exe / launcher.py)
├── paths.py             # RESOURCE_ROOT + per-user data dir (LOCALAPPDATA etc.)
├── progress_tracker.py  # Used by main.py for download progress
├── duplicate_finder.py  # Duplicate grouping logic (duplicate checker UI)
├── requirements.txt     # Python dependencies
├── requirements-build.txt  # pyinstaller, pywebview (optional builds)
├── release/             # Build output: SHUCK3R.exe (gitignored); .gitkeep tracked
├── build_windows.ps1    # Builds release/SHUCK3R.exe
├── hamster_scraper.spec    # PyInstaller one-file spec (browser launcher)
├── hamster_scraper_webview.spec  # PyInstaller one-file spec (webview)
├── templates/           # HTML (Jinja); extends base.html where noted
│   ├── base.html
│   ├── index.html
│   ├── submitted.html
│   └── duplicates.html
├── static/              # CSS (style.css), assets
├── tests/               # e.g. test_duplicate_finder.py
├── .shared/             # Optional local tools only (see .shared/README.md)
├── logs/                # Gitignored; often unused — defaults go to user profile (see below)
├── downloads/           # Gitignored; same — app uses %LOCALAPPDATA%\HamsterScraper\downloads by default
└── archive/             # Old/standalone scripts – not used by main.py
```

Only `main.py`, `paths.py`, `progress_tracker.py`, `duplicate_finder.py`, `templates/`, and `static/` are required for the web app. The `archive/` folder contains legacy scripts and is not used by the running system.

---

## UI design tooling (optional)

This repo can use the Cursor **ui-ux-pro-max** workflow (design-system search scripts). The bundle is **not** committed: clone it into `.shared/ui-ux-pro-max/` when you want it. See **[`.shared/README.md`](.shared/README.md)** for install steps and **[upstream ui-ux-pro-max-skill](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill)** for the source.

---

## Stopping the server

In the terminal where you ran `python main.py`, press **Ctrl+C**.

---

## If something goes wrong

- **“Press Enter to exit”** – the app prints the error and waits; read the traceback in the terminal.
- **Port 8001 in use** – close the other app using that port or set **`HAMSTER_PORT`** (e.g. `set HAMSTER_PORT=8002` before `python main.py` or `python launcher.py`).
- **Chrome/ChromeDriver issues** – ensure Chrome is installed and that no other automation is blocking it. ChromeDriver is auto-downloaded on first run.
- **Login required** – both sites need you to be logged in (in the window the app opens) so it can see your favorites.
