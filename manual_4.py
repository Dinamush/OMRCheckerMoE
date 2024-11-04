import os
import logging
import requests
import time
import random
from seleniumwire import webdriver  # Import selenium-wire for request interception
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from urllib.parse import urljoin
from bs4 import BeautifulSoup  # Import BeautifulSoup
import concurrent.futures
import subprocess

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SCRAPE_URL = 'https://xhamster.com/my/favorites/videos'
DOWNLOAD_DIR = "downloaded_videos"

# Ensure download directory exists
def ensure_download_dir():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        logging.info(f"Created directory {DOWNLOAD_DIR}")
    else:
        logging.info(f"Download directory already exists: {DOWNLOAD_DIR}")

# Path to ChromeDriver executable
CHROME_DRIVER_PATH = r"C:\chromedriver.exe"  # Update this path as necessary
HEADLESS = False  # Set to True to run headlessly

# Selenium WebDriver setup with selenium-wire
def setup_selenium():
    options = Options()
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " \
                 "AppleWebKit/537.36 (KHTML, like Gecko) " \
                 "Chrome/85.0.4183.83 Safari/537.36"
    options.add_argument(f'user-agent={user_agent}')
    if HEADLESS:
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
    
    # Selenium Wire options (optional: you can add more if needed)
    seleniumwire_options = {
        'disable_encoding': True,  # Disable encoding to simplify response processing
    }
    
    service = Service(CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options, seleniumwire_options=seleniumwire_options)
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

def scroll_to_bottom(driver, max_scrolls=50):
    SCROLL_PAUSE_TIME = 2
    scrolls = 0
    last_height = driver.execute_script("return document.body.scrollHeight")
    while scrolls < max_scrolls:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        scrolls += 1
        if new_height == last_height:
            logging.info("Reached the bottom of the page.")
            break
        last_height = new_height
    if scrolls == max_scrolls:
        logging.warning("Reached maximum scroll limit.")

def parse_video_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    video_links = []
    
    # Updated selectors based on the current website structure
    video_anchors = soup.find_all('a', class_='thumb-image-container')  # Example selector
    if not video_anchors:
        logging.warning("No video anchors found with 'thumb-image-container', trying alternative selector.")
        video_anchors = soup.select('a.video-thumb__image-link')  # Alternative selector

    for anchor in video_anchors:
        href = anchor.get('href')
        if href:
            video_url = urljoin('https://xhamster.com', href)
            video_links.append(video_url)
            logging.debug(f"Found video URL: {video_url}")

    if not video_links:
        logging.warning("No video links found in the page.")
    else:
        logging.info(f"Total video links found: {len(video_links)}")

    return video_links

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
            rest={'HttpOnly': cookie.get('httpOnly')}
        )
        session.cookies.set_cookie(c)
        logging.debug(f"Transferred cookie: {c}")

def capture_media_requests(driver):
    """Capture media requests by monitoring network traffic for video MIME types."""
    media_urls = []
    for request in driver.requests:
        if request.response:
            content_type = request.response.headers.get('Content-Type', '')
            if "video" in content_type:
                media_urls.append(request.url)
                logging.info(f"Captured video URL: {request.url} with Content-Type: {content_type}")
    return media_urls

def download_video(session, video_url, filepath, retries=3):
    attempt = 0
    headers = {}
    while attempt < retries:
        try:
            logging.info(f"Starting download: {video_url} (Attempt {attempt + 1})")
            with session.get(video_url, stream=True, timeout=60, headers=headers) as response:
                if response.status_code == 206:
                    logging.warning(f"Received 206 Partial Content for {video_url}. Attempting to resume.")
                    # Extract the 'Content-Range' header
                    content_range = response.headers.get('Content-Range')
                    if content_range:
                        # Parse the range
                        range_start = int(content_range.split(' ')[1].split('-')[0])
                        headers['Range'] = f"bytes={range_start}-"
                    else:
                        logging.error("No 'Content-Range' header found. Cannot resume download.")
                        break
                response.raise_for_status()
                mode = 'ab' if response.status_code == 206 else 'wb'
                with open(filepath, mode) as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            logging.info(f"Downloaded: {filepath}")
            break
        except requests.exceptions.RequestException as e:
            attempt += 1
            logging.error(f"Failed to download {video_url} (Attempt {attempt}): {e}")
            if attempt < retries:
                logging.info(f"Retrying download ({attempt}/{retries}) after delay...")
                time.sleep(5)  # Wait before retrying
            else:
                logging.error(f"Exhausted retries for {video_url}")
        except Exception as e:
            logging.error(f"Error saving video {filepath}: {e}")
            break  # For other exceptions, don't retry

def convert_video(input_path, output_path):
    try:
        subprocess.run([
            'ffmpeg',
            '-i', input_path,
            '-pix_fmt', 'yuv420p',  # Specify a supported pixel format
            output_path
        ], check=True)
        logging.info(f"Converted video saved to {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg conversion failed: {e}")

def validate_video(filepath):
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        logging.info(f"Video {filepath} downloaded successfully.")
    else:
        logging.error(f"Video {filepath} is missing or empty.")

def process_video(driver, session, idx, video_page_url):
    try:
        logging.info(f"Processing video page {idx + 1}: {video_page_url}")
        driver.get(video_page_url)
        time.sleep(random.uniform(2, 4))  # Randomized delay

        # Optional: Interact with the video player to ensure media requests are made
        try:
            play_button = driver.find_element(By.CSS_SELECTOR, 'button.play-button')  # Update selector as necessary
            if play_button:
                ActionChains(driver).click(play_button).perform()
                logging.info("Clicked play button to initiate video playback.")
                time.sleep(random.uniform(3, 6))  # Wait for media requests to be captured
        except Exception as e:
            logging.warning(f"Could not interact with the play button: {e}")

        # Capture media requests on this page
        media_urls = capture_media_requests(driver)
        logging.info(f"Found {len(media_urls)} media URLs on page {idx + 1}")
        if media_urls:
            for media_idx, media_url in enumerate(media_urls):
                # Create a unique filename for each media URL
                filename = os.path.basename(media_url).split('?')[0]
                filepath = os.path.join(DOWNLOAD_DIR, f"video_{idx + 1}_{media_idx + 1}_{filename}")
                download_video(session, media_url, filepath)
                validate_video(filepath)
        else:
            logging.warning(f"No media URLs captured for {video_page_url}")
        
        # Introduce delay after processing
        time.sleep(random.uniform(1, 3))
    except Exception as e:
        logging.error(f"Error processing video {video_page_url}: {e}")

def main():
    driver = None
    try:
        driver = setup_selenium()
        ensure_download_dir()

        # Step 1: Let the user log in manually
        wait_for_manual_login(driver)

        # Step 2: Fetch HTML content
        html_content = fetch_html(driver, SCRAPE_URL)
        if not html_content:
            logging.error("Failed to retrieve HTML content.")
            return

        # Step 3: Parse video links
        video_page_links = parse_video_links(html_content)
        if video_page_links:
            logging.info(f"Found {len(video_page_links)} video page links.")
            # Transfer cookies to requests.Session
            session = requests.Session()
            transfer_cookies(driver, session)

            # Optional: Set headers for the session
            session.headers.update({
                'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " \
                              "AppleWebKit/537.36 (KHTML, like Gecko) " \
                              "Chrome/85.0.4183.83 Safari/537.36"
            })

            # Process videos concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(process_video, driver, session, idx, url)
                    for idx, url in enumerate(video_page_links)
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Error in concurrent processing: {e}")
        else:
            logging.warning("No video links found.")
    except Exception as e:
        logging.error(f"An error occurred in main: {e}")
    finally:
        # Close the Selenium driver
        if driver:
            driver.quit()
        logging.info("Scraping process completed.")

if __name__ == "__main__":
    main()
