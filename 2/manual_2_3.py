import os
import logging
import requests
import json
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import concurrent.futures
import time
from selenium import webdriver
from urllib.parse import urljoin
import threading
import subprocess  # For combining segments

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SCRAPE_URL = 'https://xhamster.com/my/favorites/videos'
DOWNLOAD_DIR = "downloaded_videos"

# Ensure download directory exists
def ensure_download_dir():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        logging.info(f"Created directory {DOWNLOAD_DIR}")

# Selenium WebDriver settings
CHROME_DRIVER_PATH = r"C:\chromedriver.exe"  # Path to ChromeDriver executable
HEADLESS = False  # Set to True to run headlessly

def setup_selenium():
    options = Options()
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)" \
                 " Chrome/85.0.4183.83 Safari/537.36"
    options.add_argument(f'user-agent={user_agent}')
    if HEADLESS:
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
    service = Service(CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def wait_for_manual_login(driver):
    """Allow user to log in manually and wait for confirmation."""
    driver.get("https://xhamster.com/login")
    logging.info("Please log in manually in the browser window.")
    input("After logging in, press Enter here to continue...")

def transfer_cookies(driver, session):
    logging.info("Transferring cookies from Selenium to requests.Session")
    for cookie in driver.get_cookies():
        c = requests.cookies.create_cookie(
            domain=cookie.get('domain'),
            name=cookie.get('name'),
            value=cookie.get('value'),
            path=cookie.get('path'),
            secure=cookie.get('secure'),
            expires=cookie.get('expiry'),
            rest={'httpOnly': cookie.get('httpOnly')}
        )
        session.cookies.set_cookie(c)

def download_segment(session, base_url, segment_number, output_path):
    segment_url = f"{base_url}{segment_number}-v1-a1.m4s"
    segment_file = os.path.join(output_path, f"segment_{segment_number}.m4s")
    try:
        response = session.get(segment_url, stream=True)
        response.raise_for_status()
        with open(segment_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Downloaded segment {segment_number}")
    except requests.RequestException as e:
        logging.error(f"Failed to download segment {segment_number}: {e}")
    return segment_file

def download_all_segments(session, base_url, num_segments, output_path):
    segment_files = []
    for i in range(1, num_segments + 1):
        segment_file = download_segment(session, base_url, i, output_path)
        segment_files.append(segment_file)
    return segment_files

def combine_segments(segment_files, output_file="output_video.mp4"):
    with open("file_list.txt", "w") as f:
        for segment_file in segment_files:
            f.write(f"file '{segment_file}'\n")
    
    # Use FFmpeg to combine segments
    subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", "file_list.txt", "-c", "copy", output_file])
    logging.info(f"Combined video saved as {output_file}")

def process_video_segments(driver, session, video_page_url, idx):
    try:
        # Step 1: Get the base URL of the segments (this may vary)
        base_url = f"https://video-nss.xhcdn.com/seg-"  # Example base URL for segments
        output_path = os.path.join(DOWNLOAD_DIR, f"video_{idx}")
        os.makedirs(output_path, exist_ok=True)

        # Step 2: Download all segments
        num_segments = 10  # Change based on observed segments
        segment_files = download_all_segments(session, base_url, num_segments, output_path)
        
        # Step 3: Combine all segments into a single video
        output_file = os.path.join(DOWNLOAD_DIR, f"video_{idx}.mp4")
        combine_segments(segment_files, output_file)
        
    except Exception as e:
        logging.error(f"Error processing video segments for {video_page_url}: {e}")

def download_videos(driver, session, video_page_links):
    ensure_download_dir()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for idx, video_page_url in enumerate(video_page_links):
            futures.append(executor.submit(process_video_segments, driver, session, video_page_url, idx))
            time.sleep(1)  # Delay to avoid rate limits
        for future in concurrent.futures.as_completed(futures):
            pass

def main():
    driver = None
    try:
        driver = setup_selenium()

        # Step 1: Let the user log in manually
        wait_for_manual_login(driver)

        # Step 2: Fetch HTML content
        driver.get(SCRAPE_URL)
        time.sleep(2)  # Allow some time for page to load

        # Example static video links
        video_page_links = [SCRAPE_URL]  # List of URLs for example

        # Step 3: Transfer cookies to requests.Session
        session = requests.Session()
        transfer_cookies(driver, session)

        # Step 4: Download videos
        download_videos(driver, session, video_page_links)
        
    except Exception as e:
        logging.error(f"An error occurred in main: {e}")
    finally:
        # Close the Selenium driver
        if driver:
            driver.quit()

if __name__ == "__main__":
    main()
