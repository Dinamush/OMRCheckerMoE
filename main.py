#!/usr/bin/env python3
"""
Unified Video Downloader - FastAPI Application
Download favorites from PornHub and xHamster via a web UI.
"""

import os
import logging
import time
import subprocess
import concurrent.futures
import re
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
        if "404" in driver.title or "error" in driver.title.lower() or "not found" in driver.title.lower():
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

def try_click_next_button(driver) -> bool:
    """Try to click the next page button if it exists."""
    try:
        # Look for xHamster-specific next page button selectors
        next_selectors = [
            "a[data-page='next']",  # xHamster specific
            "li.next a",  # xHamster specific
            "a[aria-label='Next page']",
            "a[title='Next page']",
            "a.next",
            "a[class*='next']",
            "button[aria-label='Next page']",
            "button[title='Next page']",
            "button.next",
            "button[class*='next']",
            "a[href*='/favorites/videos/']",  # xHamster pagination links
        ]
        
        for selector in next_selectors:
            try:
                next_button = driver.find_element(By.CSS_SELECTOR, selector)
                if next_button.is_enabled() and next_button.is_displayed():
                    logging.info(f"Found next button with selector: {selector}")
                    driver.execute_script("arguments[0].click();", next_button)
                    time.sleep(3)  # Wait for page to load
                    return True
            except NoSuchElementException:
                continue
        
        logging.info("No next page button found")
        return False
    except Exception as e:
        logging.error(f"Error trying to click next button: {e}")
        return False

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

def extract_direct_video_url(driver, video_page_url: str) -> Tuple[Optional[str], str]:
    """Extract direct video URL and title from a video page using browser session."""
    try:
        logging.info(f"Extracting direct video URL from: {video_page_url}")
        
        # Set shorter timeout for individual page loads
        driver.set_page_load_timeout(15)
        driver.get(video_page_url)
        time.sleep(1)  # Minimal wait time
        
        # Extract video title first
        video_title = "Unknown Video"
        try:
            # Try to get title from page title
            page_title = driver.title
            if page_title and "|" in page_title:
                # xHamster format: "Video Title: Category | xHamster"
                video_title = page_title.split("|")[0].strip()
                if ":" in video_title:
                    video_title = video_title.split(":")[0].strip()
            elif page_title:
                video_title = page_title.replace(" | xHamster", "").strip()
            
            # Clean up the title for filename
            video_title = re.sub(r'[<>:"/\\|?*]', '_', video_title)  # Replace invalid filename chars
            video_title = re.sub(r'\s+', '_', video_title)  # Replace spaces with underscores
            video_title = video_title[:100]  # Limit length
            
            logging.info(f"Extracted video title: {video_title}")
        except Exception as e:
            logging.warning(f"Could not extract video title: {e}")
        
        # Method 1: Look for video element
        try:
            video_elements = driver.find_elements("tag name", "video")
            for video in video_elements:
                src = video.get_attribute("src")
                if src and ('.mp4' in src or '.m3u8' in src):
                    logging.info(f"Found video src: {src}")
                    return src, video_title
        except Exception as e:
            logging.warning(f"Error checking video elements: {e}")
        
        # Method 2: Look in page source for video URLs
        try:
            page_source = driver.page_source
            
            # Look for HLS URLs (preferred)
            m3u8_patterns = [
                r'https://video-cf\.xhcdn\.com/[^"\']*\.m3u8[^"\']*',
                r'https://video[0-9]*\.xhcdn\.com/[^"\']*\.m3u8[^"\']*',
                r'https://video-nss\.xhcdn\.com/[^"\']*\.m3u8[^"\']*',
            ]
            
            for pattern in m3u8_patterns:
                matches = re.findall(pattern, page_source)
                for match in matches:
                    if '.m3u8' in match:
                        logging.info(f"Found HLS URL: {match}")
                        return match, video_title
            
            # Look for direct MP4 URLs (fallback)
            mp4_patterns = [
                r'https://galleryn[0-9]+\.vcmdiawe\.com/[^"\']*\.mp4[^"\']*',
                r'https://video[0-9]*\.xhcdn\.com/[^"\']*\.mp4[^"\']*',
            ]
            
            for pattern in mp4_patterns:
                matches = re.findall(pattern, page_source)
                for match in matches:
                    if '.mp4' in match:
                        logging.info(f"Found MP4 URL: {match}")
                        return match, video_title
        except Exception as e:
            logging.warning(f"Error searching page source: {e}")
        
        logging.warning(f"No direct video URL found for: {video_page_url}")
        return None, video_title
        
    except Exception as e:
        logging.error(f"Error extracting direct video URL from {video_page_url}: {e}")
        return None, "Unknown_Video"
    finally:
        # Reset timeout to original value
        try:
            driver.set_page_load_timeout(30)
        except:
            pass

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
    """Download video using yt-dlp with progress tracking and authentication."""
    tracker = get_tracker(session_id)
    tracker.start_download(video_url)
    
    filepath = os.path.join(download_dir, f"{title}.%(ext)s")
    ydl_opts = {
        'cookiesfrombrowser': None,  # We'll use cookies file instead
        'cookiefile': cookie_file,
        'outtmpl': filepath,
        'format': 'best[height<=720]',  # Limit to 720p for reasonable file sizes
        'merge_output_format': 'mp4',   # Convert to MP4
        'hls_prefer_native': True,      # Use native HLS support for better compatibility
        'quiet': False,
        'no_warnings': False,           # Show warnings for debugging
        'retries': 3,
        'ignoreerrors': False,          # Don't ignore errors for debugging
        'extractor_retries': 3,
        'fragment_retries': 3,
        'writeinfojson': False,         # Don't write info JSON files
        'writesubtitles': False,        # Don't download subtitles
        'writeautomaticsub': False,     # Don't download auto-generated subtitles
    }
    
    def progress_hook(d):
        if d['status'] == 'downloading':
            if 'total_bytes' in d and d['total_bytes']:
                percent = (d['downloaded_bytes'] / d['total_bytes']) * 100
                logging.info(f"Downloading {title}: {percent:.1f}%")
        elif d['status'] == 'finished':
            logging.info(f"Finished downloading {title}")
    
    ydl_opts['progress_hooks'] = [progress_hook]
    
    try:
        logging.info(f"Downloading video with authentication: {video_url}")
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
            logging.info(f"Successfully downloaded: {actual_file} ({file_size} bytes)")
        else:
            tracker.fail_download(video_url, "Downloaded file not found")
            logging.error(f"No file found for {title}")
            
    except Exception as e:
        tracker.fail_download(video_url, str(e))
        logging.error(f"Exception downloading {video_url}: {e}")

def download_video_direct(video_url: str, title: str, download_dir: str, session_id: str) -> None:
    """Download video using yt-dlp without cookies with progress tracking."""
    tracker = get_tracker(session_id)
    tracker.start_download(video_url)
    
    filepath = os.path.join(download_dir, f"{title}.%(ext)s")
    ydl_opts = {
        'outtmpl': filepath,
        'format': 'best[height<=720]',  # Limit to 720p for reasonable file sizes
        'merge_output_format': 'mp4',   # Convert to MP4
        'hls_prefer_native': True,      # Use native HLS support for better compatibility
        'quiet': False,
        'no_warnings': False,           # Show warnings for debugging
        'retries': 3,
        'ignoreerrors': False,          # Don't ignore errors for debugging
        'extractor_retries': 3,
        'fragment_retries': 3,
        'writeinfojson': False,         # Don't write info JSON files
        'writesubtitles': False,        # Don't download subtitles
        'writeautomaticsub': False,     # Don't download auto-generated subtitles
    }
    
    def progress_hook(d):
        if d['status'] == 'downloading':
            if 'total_bytes' in d and d['total_bytes']:
                percent = (d['downloaded_bytes'] / d['total_bytes']) * 100
                logging.info(f"Downloading {title}: {percent:.1f}%")
        elif d['status'] == 'finished':
            logging.info(f"Finished downloading {title}")
    
    ydl_opts['progress_hooks'] = [progress_hook]
    
    try:
        logging.info(f"Downloading xHamster video: {video_url}")
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
            logging.info(f"Successfully downloaded: {actual_file} ({file_size} bytes)")
        else:
            tracker.fail_download(video_url, "Downloaded file not found")
            logging.error(f"No file found for {title}")
            
    except Exception as e:
        tracker.fail_download(video_url, str(e))
        logging.error(f"Exception downloading {video_url}: {e}")

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

def extract_video_info_sequential(driver, video_urls: List[str]) -> List[Tuple[str, str, str]]:
    """Extract video info sequentially to avoid driver conflicts."""
    results = []
    for i, video_url in enumerate(video_urls):
        try:
            logging.info(f"Extracting video {i+1}/{len(video_urls)}: {video_url}")
            direct_url, video_title = extract_direct_video_url(driver, video_url)
            results.append((video_url, direct_url, video_title))
            
            # Small delay between extractions to avoid overwhelming the site
            time.sleep(0.5)
            
        except Exception as e:
            logging.error(f"Error extracting info for {video_url}: {e}")
            results.append((video_url, None, f"video_{int(time.time())}"))
    
    return results

def xhamster_workflow(favorites_url: str, download_dir: str, headless: bool, log_file: str, session_id: str = None) -> None:
    """xHamster workflow using Chrome and yt-dlp with two-phase approach."""
    if not session_id:
        session_id = str(int(time.time()))
        
    setup_logging(log_file)
    ensure_download_dir(download_dir)
    
    cookie_file = os.path.join(LOG_DIR, f"xhamster_cookies_{session_id}.txt")
    driver = None
    try:
        driver = setup_chrome_driver(headless)
        login_url = "https://xhamster.com/login"
        wait_for_manual_login(driver, login_url)
        
        # Save cookies for yt-dlp authentication
        time.sleep(5)  # Wait for cookies to be set
        save_cookies_netscape(driver, cookie_file)
        
        # Initialize progress tracking
        tracker = get_tracker(session_id)
        tracker.start_session(0)  # We'll update this as we find videos
        
        # PHASE 1: Collect all video URLs from all pages
        logging.info("=== PHASE 1: Collecting all video URLs from all pages ===")
        all_video_links: Set[str] = set()
        page_number = 1
        consecutive_empty_pages = 0
        max_consecutive_empty = 3
        
        while True:
            # Try different pagination approaches for xHamster
            if page_number == 1:
                current_page_url = favorites_url
                html_content = fetch_html_selenium(driver, current_page_url)
            else:
                # xHamster uses /page_number format for pagination
                if page_number == 2:
                    current_page_url = f"{favorites_url}/2"
                else:
                    current_page_url = f"{favorites_url}/{page_number}"
                
                html_content = fetch_html_selenium(driver, current_page_url)
                
                # If URL pagination didn't work, try clicking next button
                if not html_content:
                    logging.info(f"URL pagination didn't work for page {page_number}, trying next button...")
                    if try_click_next_button(driver):
                        html_content = driver.page_source
                        current_page_url = driver.current_url
                    else:
                        logging.info(f"No next button found, stopping pagination")
                        break
            
            logging.info(f"Collecting URLs from page {page_number}: {current_page_url}")
            
            if not html_content:
                logging.info(f"No content found on page {page_number}, stopping pagination")
                break
            
            # Debug: Check if we're getting the same content
            current_url = driver.current_url
            logging.info(f"Current browser URL after navigation: {current_url}")
            
            # Check if we've been redirected to a different page (indicating end of pagination)
            if "favorites/videos" not in current_url or "login" in current_url or "signin" in current_url:
                logging.warning(f"Redirected away from favorites page: {current_url} - stopping pagination")
                break
            
            # Check if we're on an error page (404, etc.) - stop pagination immediately
            if "404" in driver.title or "error" in driver.title.lower() or "not found" in driver.title.lower():
                logging.warning(f"Page {page_number} appears to be an error page: {driver.title} - stopping pagination")
                break
            
            video_links = parse_video_links_xhamster(html_content, "https://xhamster.com")
            if not video_links:
                logging.info(f"No video links found on page {page_number}, stopping pagination")
                break
            
            # Debug: Log some sample URLs to see if they're different
            sample_urls = list(video_links)[:3] if video_links else []
            logging.info(f"Sample video URLs from page {page_number}: {sample_urls}")
            
            # Check if we're getting the same sample URLs as previous pages (indicating end of pagination)
            if page_number > 1 and sample_urls:
                # Store sample URLs from previous page for comparison
                if not hasattr(try_click_next_button, 'prev_sample_urls'):
                    try_click_next_button.prev_sample_urls = []
                
                if sample_urls == try_click_next_button.prev_sample_urls:
                    logging.warning(f"Page {page_number} returned identical sample URLs as previous page - likely end of pagination")
                    consecutive_empty_pages += 1
                
                try_click_next_button.prev_sample_urls = sample_urls
            
            new_links = set(video_links) - all_video_links
            if not new_links:
                consecutive_empty_pages += 1
                logging.info(f"No new videos found on page {page_number} (consecutive empty: {consecutive_empty_pages})")
                
                # Check if we're getting the exact same URLs (indicating pagination isn't working)
                if video_links and len(video_links) == len(all_video_links) and set(video_links).issubset(all_video_links):
                    logging.warning(f"Page {page_number} returned identical content to previous pages - pagination may not be working")
                    consecutive_empty_pages += 1  # Count this as an empty page
                
                
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
        
        logging.info(f"=== PHASE 1 COMPLETE: Collected {len(all_video_links)} unique video URLs from {page_number-1} pages ===")
        
        # PHASE 2: Download all videos
        logging.info("=== PHASE 2: Downloading all videos ===")
        all_video_links_list = list(all_video_links)
        tracker.start_session(len(all_video_links_list))
        
        # Process videos in batches for better performance
        batch_size = 10  # Process 10 videos at a time
        total_downloaded = 0
        
        for i in range(0, len(all_video_links_list), batch_size):
            batch = all_video_links_list[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(all_video_links_list) + batch_size - 1) // batch_size
            
            logging.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} videos)")
            
            # Extract video info sequentially (to avoid driver conflicts)
            video_info_results = extract_video_info_sequential(driver, batch)
            
            # Prepare videos for download
            videos_to_download = []
            for original_url, direct_url, video_title in video_info_results:
                if direct_url:
                    videos_to_download.append((direct_url, video_title))
                    tracker.add_urls([direct_url], [video_title])
                else:
                    fallback_title = f"video_{total_downloaded + len(videos_to_download)}"
                    tracker.add_urls([original_url], [fallback_title])
                    tracker.fail_download(original_url, "Could not extract direct video URL")
            
            # Download videos in parallel
            if videos_to_download:
                logging.info(f"Downloading {len(videos_to_download)} videos from batch {batch_num}...")
                download_videos_parallel(videos_to_download, download_dir, cookie_file=cookie_file, session_id=session_id)
                total_downloaded += len(videos_to_download)
            
            # Small delay between batches
            time.sleep(2)
        
        logging.info(f"=== PHASE 2 COMPLETE: Downloaded {total_downloaded} videos ===")
        logging.info(f"xHamster workflow completed. Total videos processed: {total_downloaded}")
        
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
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        raise
