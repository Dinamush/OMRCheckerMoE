# Unified Video Downloader

Web app to download videos from your **PornHub** and **xHamster** favorites. You log in once in a browser window; the app collects video URLs and downloads them with progress tracking.

---

## Requirements

- **Python 3.8+**
- **Chrome** (installed for Selenium/automation)
- **ChromeDriver** is installed automatically via `webdriver-manager` when you run the app

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
   Videos are saved under the project folder in:

   ```
   downloads/
   ```

   Logs and progress data are under:

   ```
   logs/
   ```

---

## Project layout (what the app uses)

```
hamster_scraper/
├── main.py              # Entry point: run this to start the server
├── progress_tracker.py  # Used by main.py for download progress
├── requirements.txt    # Python dependencies
├── templates/           # HTML (index, submitted)
│   ├── index.html
│   └── submitted.html
├── static/              # CSS, JS
├── logs/                # Created at runtime (logs, progress JSON)
├── downloads/           # Created at runtime (downloaded videos)
└── archive/             # Old/standalone scripts – not used by main.py
```

Only `main.py`, `progress_tracker.py`, `templates/`, and `static/` are required for the web app. The `archive/` folder contains legacy scripts and is not used by the running system.

---

## Stopping the server

In the terminal where you ran `python main.py`, press **Ctrl+C**.

---

## If something goes wrong

- **“Press Enter to exit”** – the app prints the error and waits; read the traceback in the terminal.
- **Port 8001 in use** – close the other app using 8001 or change the port in `main.py` (last line: `uvicorn.run(app, host="0.0.0.0", port=8001)`).
- **Chrome/ChromeDriver issues** – ensure Chrome is installed and that no other automation is blocking it. ChromeDriver is auto-downloaded on first run.
- **Login required** – both sites need you to be logged in (in the window the app opens) so it can see your favorites.
