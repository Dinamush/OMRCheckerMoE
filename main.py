#!/usr/bin/env python3
"""
Unified Video Downloader - FastAPI Application
Supports both PornHub (ph.py) and xHamster (xh.py) video downloading
"""

import os
import logging
import time
import subprocess
import concurrent.futures
from typing import List, Optional, Tuple, Set
from fastapi import FastAPI, Request, BackgroundTasks, Form
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from urllib.parse import urljoin
from webdriver_manager.chrome import ChromeDriverManager
import yt_dlp
import requests
from progress_tracker import get_tracker, cleanup_tracker

# ----------------------------- Configuration ----------------------------- #

app = FastAPI(title="Unified Video Downloader", description="Download videos from PornHub and xHamster")

# Absolute path for template and static directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Directory to store logs
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Directory to store downloaded videos
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ----------------------------- Helper Functions ----------------------------- #

def setup_logging(log_file: str) -> None:
    """Configure logging."""
    # Clear existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create new handlers
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

def ensure_download_dir(directory: str) -> None:
    """Ensure that the download directory exists."""
    try:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Download directory is set to: {directory}")
    except Exception as e:
        logging.error(f"Failed to create download directory '{directory}': {e}")
        raise

def setup_chrome_driver(headless: bool = False) -> webdriver.Chrome:
    """Set up Chrome WebDriver with webdriver-manager."""
    options = Options()
    if headless:
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--log-level=3")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-web-security")
    options.add_argument("--allow-running-insecure-content")
    options.add_argument("--disable-features=VizDisplayCompositor")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    try:
        # Use ChromeDriverManager with caching
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        # Set timeouts to prevent hanging
        driver.set_page_load_timeout(30)
        driver.implicitly_wait(10)
        
        # Execute script to remove webdriver property
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        logging.info("Chrome WebDriver initialized successfully.")
        return driver
    except WebDriverException as e:
        logging.error(f"Chrome WebDriver initialization failed: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error initializing Chrome WebDriver: {e}")
        raise


def wait_for_manual_login(driver, login_url: str) -> None:
    """Prompt the user to log in manually and wait for confirmation."""
    try:
        driver.get(login_url)
        logging.info("Navigated to login page. Please log in manually in the browser window.")
        input("After logging in, press Enter here to continue...")
        logging.info("User confirmed login.")
    except Exception as e:
        logging.error(f"Error during manual login: {e}")
        raise

def save_cookies_netscape(driver, cookie_file: str) -> None:
    """Save cookies in Netscape format for yt-dlp."""
    cookies = driver.get_cookies()
    with open(cookie_file, "w") as f:
        f.write("# Netscape HTTP Cookie File\n")
        f.write("# This file was generated by Selenium\n")
        for cookie in cookies:
            domain = cookie.get("domain", "")
            flag = "TRUE" if domain.startswith(".") else "FALSE"
            path = cookie.get("path", "/")
            secure = "TRUE" if cookie.get("secure", False) else "FALSE"
            expiry = cookie.get("expiry", 0)
            name = cookie.get("name", "")
            value = cookie.get("value", "")
            f.write(f"{domain}\t{flag}\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n")
    logging.info(f"Cookies saved to {cookie_file}")

def extract_video_urls_ytdlp(cookie_file: str, playlist_url: str) -> List[str]:
    """Extract video URLs using yt-dlp (for PornHub)."""
    command = [
        "python", "-m", "yt_dlp",
        "--cookies", cookie_file,
        "--flat-playlist",
        "--print", "url",
        playlist_url
    ]
    logging.info("Extracting video URLs with yt-dlp...")
    
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error("Error extracting video URLs with yt-dlp:")
        logging.error(result.stderr)
        return []
    
    urls = result.stdout.strip().splitlines()
    logging.info(f"Extracted {len(urls)} video URLs with yt-dlp.")
    return urls

def fetch_html_selenium(driver, url: str) -> Optional[str]:
    """Fetch HTML content using Selenium with scrolling."""
    try:
        driver.get(url)
        logging.info(f"Navigated to {url}")
        
        # Wait for page to load
        time.sleep(2)
        
        # Check if we're on a valid page (not 404 or error page)
        if "404" in driver.title or "error" in driver.title.lower():
            logging.warning(f"Page appears to be an error page: {driver.title}")
            return None
        
        # Check if we're still on the favorites page (not redirected to login)
        current_url = driver.current_url
        if "login" in current_url or "signin" in current_url:
            logging.warning(f"Redirected to login page: {current_url}")
            return None
        
        scroll_to_bottom(driver)
        
        # Additional wait after scrolling
        time.sleep(1)
        
        return driver.page_source
    except Exception as e:
        logging.error(f"Error fetching HTML from {url}: {e}")
        return None

def scroll_to_bottom(driver, pause_time: float = 2.0) -> None:
    """Scroll to the bottom of the page to load all content."""
    try:
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause_time)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                logging.info("Reached the bottom of the page.")
                break
            last_height = new_height
    except Exception as e:
        logging.error(f"Error during scrolling: {e}")

def parse_video_links_xhamster(html: str, base_url: str) -> List[str]:
    """Parse video links for xHamster."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        video_links = []
        
        # Try multiple selectors for video links
        selectors = [
            'a.thumb-image-container',
            'a[class*="thumb"]',
            'a[href*="/videos/"]'
        ]
        
        for selector in selectors:
            video_anchors = soup.select(selector)
            logging.debug(f"Found {len(video_anchors)} elements with selector '{selector}'")
            
            for anchor in video_anchors:
                href = anchor.get('href')
                if href and '/videos/' in href:
                    video_url = urljoin(base_url, href)
                    if video_url not in video_links:  # Avoid duplicates
                        video_links.append(video_url)
                        logging.debug(f"Found xHamster video URL: {video_url}")
        
        logging.info(f"Parsed {len(video_links)} xHamster video links.")
        return video_links
    except Exception as e:
        logging.error(f"Error parsing xHamster video links: {e}")
        return []

def create_authenticated_session(driver) -> requests.Session:
    """Create authenticated requests session from Selenium cookies."""
    try:
        session = requests.Session()
        cookies = driver.get_cookies()
        for cookie in cookies:
            domain = cookie['domain'].lstrip('.')
            session.cookies.set(cookie['name'], cookie['value'], domain=domain)
        logging.info("Authenticated requests session created.")
        return session
    except Exception as e:
        logging.error(f"Error creating authenticated session: {e}")
        raise

def get_video_title(session: requests.Session, video_url: str) -> str:
    """Get video title from video page."""
    try:
        response = session.get(video_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('h1')
        if title_tag:
            title = title_tag.get_text(strip=True)
            sanitized_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else "_" for c in title)
            return sanitized_title
        else:
            logging.warning(f"Title not found for {video_url}")
            return f"video_{int(time.time())}"
    except Exception as e:
        logging.error(f"Error fetching title for {video_url}: {e}")
        return f"video_{int(time.time())}"

def download_video_ytdlp(video_url: str, title: str, cookie_file: str, download_dir: str, session_id: str) -> None:
    """Download video using yt-dlp with progress tracking."""
    tracker = get_tracker(session_id)
    tracker.start_download(video_url)
    
    filepath = os.path.join(download_dir, f"{title}.%(ext)s")
    command = [
        "python", "-m", "yt_dlp",
        "--cookies", cookie_file,
        "-o", filepath,
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        video_url
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Unknown error"
            tracker.fail_download(video_url, error_msg)
        else:
            # Find the actual downloaded file
            actual_file = None
            for ext in ['.mp4', '.mkv', '.webm', '.avi']:
                potential_file = os.path.join(download_dir, f"{title}{ext}")
                if os.path.exists(potential_file):
                    actual_file = potential_file
                    break
            
            if actual_file:
                file_size = os.path.getsize(actual_file)
                tracker.complete_download(video_url, actual_file, file_size)
            else:
                tracker.fail_download(video_url, "Downloaded file not found")
                
    except Exception as e:
        tracker.fail_download(video_url, str(e))

def download_video_direct(video_url: str, title: str, download_dir: str, session_id: str) -> None:
    """Download video using yt-dlp without cookies with progress tracking."""
    tracker = get_tracker(session_id)
    tracker.start_download(video_url)
    
    filepath = os.path.join(download_dir, f"{title}.%(ext)s")
    ydl_opts = {
        'outtmpl': filepath,
        'format': 'best',
        'quiet': False,
        'no_warnings': True,
        'retries': 3,
        'ignoreerrors': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            
        # Find the actual downloaded file
        actual_file = None
        for ext in ['.mp4', '.mkv', '.webm', '.avi']:
            potential_file = os.path.join(download_dir, f"{title}{ext}")
            if os.path.exists(potential_file):
                actual_file = potential_file
                break
        
        if actual_file:
            file_size = os.path.getsize(actual_file)
            tracker.complete_download(video_url, actual_file, file_size)
        else:
            tracker.fail_download(video_url, "Downloaded file not found")
            
    except Exception as e:
        tracker.fail_download(video_url, str(e))

def download_videos_parallel(video_info_list: List[Tuple[str, str]], download_dir: str, 
                           max_workers: int = 4, cookie_file: str = None, session_id: str = None) -> None:
    """Download videos in parallel with progress tracking."""
    if not session_id:
        session_id = str(int(time.time()))
        
    try:
        tracker = get_tracker(session_id)
        
        # Only start session if not already started
        if not tracker.stats.start_time:
            tracker.start_session(len(video_info_list))
            # Add URLs to tracking if not already added
            urls = [item[0] for item in video_info_list]
            titles = [item[1] for item in video_info_list]
            tracker.add_urls(urls, titles)
        
        logging.info(f"Starting download of {len(video_info_list)} videos with {max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for video_url, title in video_info_list:
                if cookie_file:
                    futures.append(executor.submit(download_video_ytdlp, video_url, title, cookie_file, download_dir, session_id))
                else:
                    futures.append(executor.submit(download_video_direct, video_url, title, download_dir, session_id))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error in video download: {e}")
        
        # End the session
        tracker.end_session()
        logging.info("Download session completed successfully")
        
    except Exception as e:
        logging.critical(f"Critical error in download_videos_parallel: {e}")
    finally:
        try:
            cleanup_tracker(session_id)
        except Exception as e:
            logging.error(f"Error cleaning up tracker: {e}")

# ----------------------------- Site-Specific Workflows ----------------------------- #

def pornhub_workflow(playlist_url: str, download_dir: str, headless: bool, log_file: str, session_id: str = None) -> None:
    """PornHub workflow using Chrome and yt-dlp."""
    if not session_id:
        session_id = str(int(time.time()))
        
    setup_logging(log_file)
    ensure_download_dir(download_dir)
    
    cookie_file = os.path.join(LOG_DIR, f"pornhub_cookies_{session_id}.txt")
    driver = None
    
    try:
        driver = setup_chrome_driver(headless)
        driver.get("https://www.pornhub.com")
        logging.info("Please log in to PornHub in the opened browser.")
        input("After logging in, press Enter to continue...")
        
        time.sleep(5)  # Wait for cookies to be set
        save_cookies_netscape(driver, cookie_file)
        
        # Extract video URLs
        video_urls = extract_video_urls_ytdlp(cookie_file, playlist_url)
        if not video_urls:
            logging.error("No video URLs extracted.")
            return
        
        # Download videos
        video_info_list = [(url, f"video_{i}") for i, url in enumerate(video_urls)]
        download_videos_parallel(video_info_list, download_dir, cookie_file=cookie_file, session_id=session_id)
        
    except Exception as e:
        logging.critical(f"Error in PornHub workflow: {e}")
    finally:
        if driver:
            try:
                driver.quit()
                logging.info("Chrome WebDriver closed successfully.")
            except Exception as e:
                logging.error(f"Error closing Chrome WebDriver: {e}")

def xhamster_workflow(favorites_url: str, download_dir: str, headless: bool, log_file: str, session_id: str = None) -> None:
    """xHamster workflow using Chrome and direct scraping."""
    if not session_id:
        session_id = str(int(time.time()))
        
    setup_logging(log_file)
    ensure_download_dir(download_dir)
    
    driver = None
    try:
        driver = setup_chrome_driver(headless)
        login_url = "https://xhamster.com/login"
        wait_for_manual_login(driver, login_url)
        
        all_video_links: Set[str] = set()
        page_number = 1
        
        # Pagination loop
        consecutive_empty_pages = 0
        max_consecutive_empty = 3  # Stop after 3 consecutive pages with no new videos
        
        while True:
            current_page_url = f"{favorites_url}?page={page_number}"
            logging.info(f"Processing page {page_number}: {current_page_url}")
            
            html_content = fetch_html_selenium(driver, current_page_url)
            if not html_content:
                logging.info(f"No content found on page {page_number}, stopping pagination")
                break
            
            video_links = parse_video_links_xhamster(html_content, "https://xhamster.com")
            if not video_links:
                logging.info(f"No video links found on page {page_number}, stopping pagination")
                break
            
            new_links = set(video_links) - all_video_links
            if not new_links:
                consecutive_empty_pages += 1
                logging.info(f"No new videos found on page {page_number} (consecutive empty: {consecutive_empty_pages})")
                
                if consecutive_empty_pages >= max_consecutive_empty:
                    logging.info(f"Stopping pagination after {consecutive_empty_pages} consecutive pages with no new videos")
                    break
            else:
                consecutive_empty_pages = 0  # Reset counter when we find new videos
                logging.info(f"Found {len(new_links)} new videos on page {page_number}")
            
            all_video_links.update(video_links)  # Update with all links from this page
            page_number += 1
            
            # Safety limit to prevent infinite loops
            if page_number > 100:
                logging.warning("Reached maximum page limit (100), stopping pagination")
                break
        
        if all_video_links:
            logging.info(f"Collected {len(all_video_links)} video links.")
            session = create_authenticated_session(driver)
            
            # Start progress tracking immediately
            tracker = get_tracker(session_id)
            tracker.start_session(len(all_video_links))
            
            # Fetch titles in parallel with progress updates
            video_info_list = []
            completed_count = 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all title fetching tasks
                future_to_url = {
                    executor.submit(get_video_title, session, url): url 
                    for url in all_video_links
                }
                
                # Process completed tasks
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    completed_count += 1
                    
                    try:
                        title = future.result()
                        video_info_list.append((url, title))
                        tracker.add_urls([url], [title])
                        tracker.update_title_fetch_progress(completed_count, len(all_video_links))
                        logging.info(f"Fetched title for video {completed_count}/{len(all_video_links)}: {url}")
                    except Exception as e:
                        logging.error(f"Error fetching title for {url}: {e}")
                        video_info_list.append((url, f"video_{int(time.time())}"))
                        tracker.add_urls([url], [f"video_{int(time.time())}"])
                        tracker.update_title_fetch_progress(completed_count, len(all_video_links))
            
            # Now start the actual downloads
            download_videos_parallel(video_info_list, download_dir, session_id=session_id)
        
    except Exception as e:
        logging.critical(f"Error in xHamster workflow: {e}")
    finally:
        if driver:
            try:
                driver.quit()
                logging.info("Chrome WebDriver closed successfully.")
            except Exception as e:
                logging.error(f"Error closing Chrome WebDriver: {e}")

# ----------------------------- FastAPI Routes ----------------------------- #

@app.get("/")
def read_form(request: Request):
    """Display the main form."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/download")
def handle_form(request: Request,
                background_tasks: BackgroundTasks,
                site: str = Form(...),
                headless: str = Form("false")):
    """Handle form submission and start the appropriate workflow."""
    timestamp = int(time.time())
    session_id = str(timestamp)
    log_file = os.path.join(LOG_DIR, f"video_downloader_{timestamp}.log")
    
    # Convert headless string to boolean
    headless_bool = headless.lower() == "true"
    
    if site == "pornhub":
        # PornHub favorites URL - user will need to be logged in to their account
        playlist_url = "https://www.pornhub.com/my/favorites/videos"
        background_tasks.add_task(pornhub_workflow, playlist_url, DOWNLOAD_DIR, headless_bool, log_file, session_id)
    elif site == "xhamster":
        # Default xHamster favorites URL - user will need to be logged in
        favorites_url = "https://xhamster.com/my/favorites/videos"
        background_tasks.add_task(xhamster_workflow, favorites_url, DOWNLOAD_DIR, headless_bool, log_file, session_id)
    else:
        return {"error": "Invalid site selected"}
    
    return templates.TemplateResponse("submitted.html", {
        "request": request, 
        "timestamp": timestamp,
        "session_id": session_id,
        "site": site
    })

@app.get("/download/log/{timestamp}")
def get_log(timestamp: int):
    """Download log file."""
    log_file = os.path.join(LOG_DIR, f"video_downloader_{timestamp}.log")
    if os.path.exists(log_file):
        return FileResponse(path=log_file, filename=os.path.basename(log_file), media_type='text/plain')
    else:
        return {"error": "Log file not found."}

@app.get("/downloaded/{filename}")
def get_downloaded_file(filename: str):
    """Serve downloaded video files."""
    file_path = os.path.join(DOWNLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename, media_type='video/mp4')
    else:
        return {"error": "File not found."}

@app.get("/progress/{session_id}")
def get_progress(session_id: str):
    """Get progress information for a download session."""
    try:
        tracker = get_tracker(session_id)
        return tracker.get_progress()
    except Exception as e:
        return {"error": f"Session not found: {e}"}

@app.get("/progress/{session_id}/summary")
def get_progress_summary(session_id: str):
    """Get progress summary for a download session."""
    try:
        tracker = get_tracker(session_id)
        return tracker.get_summary()
    except Exception as e:
        return {"error": f"Session not found: {e}"}

@app.get("/progress/{session_id}/json")
def get_progress_json(session_id: str):
    """Get progress as JSON file."""
    progress_file = os.path.join(LOG_DIR, f"progress_{session_id}.json")
    if os.path.exists(progress_file):
        return FileResponse(path=progress_file, filename=f"progress_{session_id}.json", media_type='application/json')
    else:
        return {"error": "Progress file not found."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
