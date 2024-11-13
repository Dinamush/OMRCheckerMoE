import os
import logging
import time
import concurrent.futures
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from urllib.parse import urljoin
import yt_dlp

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

def fetch_html(driver, url):
    try:
        driver.get(url)
        logging.info(f"Navigated to {url}")
        scroll_to_bottom(driver)
        return driver.page_source
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None

def scroll_to_bottom(driver):
    SCROLL_PAUSE_TIME = 2
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def parse_video_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    video_links = []
    video_anchors = soup.find_all('a', class_='thumb-image-container')  # Adjust selector if necessary
    for anchor in video_anchors:
        href = anchor.get('href')
        if href:
            video_url = urljoin('https://xhamster.com', href)
            video_links.append(video_url)
            logging.debug(f"Found video URL: {video_url}")
    return video_links

def download_video_with_ytdlp(video_url, idx):
    """Download video using yt-dlp."""
    filepath = os.path.join(DOWNLOAD_DIR, f"video_{idx + 1}.mp4")
    ydl_opts = {
        'outtmpl': filepath,  # Output template to save with unique names
        'format': 'best',  # Choose the best quality format
        'quiet': False,     # Show download progress
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logging.info(f"Starting download for {video_url}")
            ydl.download([video_url])
            logging.info(f"Downloaded: {filepath}")
    except Exception as e:
        logging.error(f"Failed to download {video_url}: {e}")

def download_videos(video_page_links):
    ensure_download_dir()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for idx, video_page_url in enumerate(video_page_links):
            futures.append(executor.submit(download_video_with_ytdlp, video_page_url, idx))
            time.sleep(1)  # Delay to avoid rate limits
        for future in concurrent.futures.as_completed(futures):
            pass

def main():
    driver = setup_selenium()

    # Step 1: Let the user log in manually
    wait_for_manual_login(driver)

    # Step 2: Fetch HTML content
    html_content = fetch_html(driver, SCRAPE_URL)
    if not html_content:
        logging.error("Failed to retrieve HTML content.")
        driver.quit()
        return

    # Step 3: Parse video links
    video_page_links = parse_video_links(html_content)
    if video_page_links:
        logging.info(f"Found {len(video_page_links)} video page links.")
        download_videos(video_page_links)
    else:
        logging.warning("No video links found.")
    
    # Close the Selenium driver
    driver.quit()

if __name__ == "__main__":
    main()
