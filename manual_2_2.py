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
from urllib.parse import urljoin  # Add this import
import threading  # Added for running the monitor in a separate thread

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
    video_anchors = soup.find_all('a', class_='thumb-image-container')
    if not video_anchors:
        logging.warning("No video anchors found, trying alternative selector.")
        video_anchors = soup.select('a.video-thumb__image-link')
    for anchor in video_anchors:
        href = anchor.get('href')
        if href:
            video_url = urljoin('https://xhamster.com', href)
            video_links.append(video_url)
            logging.debug(f"Found video URL: {video_url}")
    if not video_links:
        logging.warning("No video links found in the page.")
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
            rest={'httpOnly': cookie.get('httpOnly')}
        )
        session.cookies.set_cookie(c)

def get_video_download_url_selenium(driver, video_page_url):
    """Retrieve the video download URL by parsing the <noscript> tag."""
    try:
        driver.get(video_page_url)
        logging.info(f"Fetching video URL from {video_page_url}")
        time.sleep(2)  # Wait for page to load fully

        # Parse the page to locate the <noscript> tag
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        noscript_tag = soup.find('noscript')
        if noscript_tag:
            # Decode the <noscript> tag to extract video URL
            noscript_html = noscript_tag.decode_contents()
            nosoup = BeautifulSoup(noscript_html, 'html.parser')
            video_tag = nosoup.find('video')
            if video_tag and video_tag.get('src'):
                video_url = video_tag['src']
                logging.info(f"Found video URL in noscript tag: {video_url}")
                return video_url
            else:
                logging.warning("Video tag not found inside noscript.")
        else:
            logging.warning("Noscript tag not found in page.")

        # Alternative: Extract from JSON in the script tags if noscript fails
        scripts = soup.find_all('script')
        for script in scripts:
            if 'window.initials=' in script.text:
                json_text = script.text.split('window.initials=')[1].rstrip(';')
                data = json.loads(json_text)
                sources = data.get('videoModel', {}).get('sources', {}).get('mp4', [])
                if sources:
                    video_url = sources[-1]['link']
                    logging.info(f"Found video URL in JSON data: {video_url}")
                    return video_url
                else:
                    logging.warning("No video sources found in JSON data.")
                break
    except Exception as e:
        logging.error(f"Failed to get video URL from {video_page_url}: {e}")
    return None


def download_video(session, video_url, filepath, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            logging.info(f"Starting download: {video_url} (Attempt {attempt+1})")
            with session.get(video_url, stream=True, timeout=30) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            logging.debug(f"Downloaded {downloaded} of {total_size} bytes")
            logging.info(f"Downloaded: {filepath}")
            break  # Exit loop if download succeeded
        except requests.exceptions.RequestException as e:
            attempt += 1
            logging.error(f"Failed to download {video_url} (Attempt {attempt}): {e}")
            if attempt == retries:
                logging.error(f"Exhausted retries for {video_url}")
        except Exception as e:
            logging.error(f"Error saving video {filepath}: {e}")
            break  # For other exceptions, don't retry

def process_video(driver, session, idx, video_page_url):
    try:
        video_url = get_video_download_url_selenium(driver, video_page_url)
        if video_url:
            filepath = os.path.join(DOWNLOAD_DIR, f"video_{idx + 1}.mp4")
            download_video(session, video_url, filepath)
        else:
            logging.warning(f"Could not get video URL for {video_page_url}")
    except Exception as e:
        logging.error(f"Error processing video {video_page_url}: {e}")

def download_videos(driver, session, video_page_links):
    ensure_download_dir()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for idx, video_page_url in enumerate(video_page_links):
            futures.append(executor.submit(process_video, driver, session, idx, video_page_url))
            time.sleep(1)  # Delay to avoid rate limits
        for future in concurrent.futures.as_completed(futures):
            pass

def monitor_video_playback(driver, session, stop_event):
    """Monitor video playback on the page and download videos when they start playing."""
    logging.info("Started monitoring for video playback.")
    already_downloaded = set()
    while not stop_event.is_set():
        try:
            # Execute JavaScript to get all video elements and their src and paused state
            script = """
            var videos = document.getElementsByTagName('video');
            var video_info = [];
            for (var i = 0; i < videos.length; i++) {
                video_info.push({
                    src: videos[i].currentSrc || videos[i].src,
                    paused: videos[i].paused
                });
            }
            return video_info;
            """
            video_info = driver.execute_script(script)
            for idx, video in enumerate(video_info):
                src = video['src']
                paused = video['paused']
                if not paused and src and src not in already_downloaded:
                    logging.info(f"Detected playing video: {src}")
                    already_downloaded.add(src)
                    # Download the video
                    filename = src.split('/')[-1].split('?')[0]  # Extract filename from URL
                    filepath = os.path.join(DOWNLOAD_DIR, f"playing_{filename}")
                    download_video(session, src, filepath)
            time.sleep(5)  # Wait before next check
        except Exception as e:
            logging.error(f"Error in monitoring playback: {e}")
            time.sleep(5)
    logging.info("Stopped monitoring for video playback.")

def main():
    driver = None
    stop_event = threading.Event()
    monitor_thread = None
    try:
        driver = setup_selenium()

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
            # Start downloading existing videos
            download_videos(driver, session, video_page_links)
        else:
            logging.warning("No video links found.")

        # Start monitoring for video playback in a separate thread
        if driver.current_url == SCRAPE_URL:
            session = requests.Session()
            transfer_cookies(driver, session)
            monitor_thread = threading.Thread(target=monitor_video_playback, args=(driver, session, stop_event))
            monitor_thread.start()
            logging.info("Monitoring for video playback. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logging.info("Keyboard interrupt received. Stopping monitoring...")
                stop_event.set()
                monitor_thread.join()
    except Exception as e:
        logging.error(f"An error occurred in main: {e}")
    finally:
        # Close the Selenium driver
        if driver:
            driver.quit()

if __name__ == "__main__":
    main()
