# main.py

import os
from fastapi import FastAPI, Request, BackgroundTasks, Form
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from UI_version.downloader import main_workflow
import shutil

app = FastAPI()

# Absolute path for template and static directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Directory to store logs
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Directory to store downloaded videos
DOWNLOAD_DIR = "downloaded_videos"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

@app.get("/")
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/download")
def handle_form(request: Request,
                background_tasks: BackgroundTasks,
                base_url: str = Form(...),
                favorites_url: str = Form(...),
                headless: bool = Form(False)):
    """
    Handle the form submission and start the downloading process.
    """
    # Generate a unique log file name based on timestamp
    import time
    timestamp = int(time.time())
    log_file = os.path.join(LOG_DIR, f"video_downloader_{timestamp}.log")

    # Start the background task
    background_tasks.add_task(main_workflow,
                              base_url=base_url,
                              favorites_url=favorites_url,
                              download_dir=DOWNLOAD_DIR,
                              headless=headless,
                              log_file=log_file)

    # Redirect to a status page or inform the user
    return templates.TemplateResponse("submitted.html", {"request": request, "timestamp": timestamp})

@app.get("/download/log/{timestamp}")
def get_log(timestamp: int):
    """
    Allow users to download the log file.
    """
    log_file = os.path.join(LOG_DIR, f"video_downloader_{timestamp}.log")
    if os.path.exists(log_file):
        return FileResponse(path=log_file, filename=os.path.basename(log_file), media_type='text/plain')
    else:
        return {"error": "Log file not found."}

@app.get("/downloaded/{filename}")
def get_downloaded_file(filename: str):
    """
    Serve the downloaded video files.
    """
    file_path = os.path.join(DOWNLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename, media_type='video/mp4')
    else:
        return {"error": "File not found."}
