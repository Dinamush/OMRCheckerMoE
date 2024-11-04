import os
import logging
import requests
import json
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urljoin
import concurrent.futures
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

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

def transfer_cookies(driver, session):
    for cookie in driver.get_cookies():
        session.cookies.set(cookie['name'], cookie['value'], domain=cookie['domain'])

def get_video_download_url_selenium(driver, video_page_url):
    """Retrieve the video download URL using Selenium to avoid 520 errors."""
    try:
        driver.get(video_page_url)
        time.sleep(2)  # Wait for page to load fully
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        scripts = soup.find_all('script')
        for script in scripts:
            if 'window.initials=' in script.text:
                json_text = script.text.split('window.initials=')[1].rstrip(';')
                data = json.loads(json_text)
                video_url = data['videoModel']['sources']['mp4'][0]['link']
                return video_url
    except Exception as e:
        logging.error(f"Failed to get video URL from {video_page_url}: {e}")
    return None

def download_video(session, video_url, filepath):
    try:
        with session.get(video_url, stream=True) as response:
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        logging.info(f"Downloaded: {filepath}")
    except Exception as e:
        logging.error(f"Failed to download {video_url}: {e}")

def download_videos(driver, video_page_links):
    ensure_download_dir()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for idx, video_page_url in enumerate(video_page_links):
            video_url = get_video_download_url_selenium(driver, video_page_url)
            if video_url:
                filepath = os.path.join(DOWNLOAD_DIR, f"video_{idx + 1}.mp4")
                futures.append(executor.submit(download_video, requests.Session(), video_url, filepath))
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
        download_videos(driver, video_page_links)
    else:
        logging.warning("No video links found.")
    
    # Close the Selenium driver
    driver.quit()

if __name__ == "__main__":
    main()
