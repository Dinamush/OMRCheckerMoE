import os
import logging
import time
import concurrent.futures
from typing import List, Optional
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from urllib.parse import urljoin
import yt_dlp
import requests

# ----------------------------- Configuration ----------------------------- #

# Logging configuration
LOG_FILE = 'video_downloader.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Constants
SCRAPE_URL = 'https://xhamster.com/my/favorites/videos'
DOWNLOAD_DIR = "downloaded_videos"

# Selenium WebDriver settings
CHROME_DRIVER_PATH = r"C:\chromedriver.exe"  # Path to ChromeDriver executable
HEADLESS = False  # Set to True to run headlessly

# Concurrency settings
MAX_WORKERS = 5
DOWNLOAD_DELAY = 1  # Seconds between starting downloads to avoid rate limits

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # Seconds between retries

# ----------------------------- Helper Functions ----------------------------- #

def ensure_download_dir(directory: str = DOWNLOAD_DIR) -> None:
    """Ensure that the download directory exists."""
    os.makedirs(directory, exist_ok=True)
    logging.info(f"Download directory is set to: {directory}")

def setup_selenium() -> webdriver.Chrome:
    """Set up the Selenium WebDriver with specified options."""
    options = Options()
    if HEADLESS:
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--log-level=3")  # Suppress Selenium logs
    service = Service(CHROME_DRIVER_PATH)
    try:
        driver = webdriver.Chrome(service=service, options=options)
        logging.info("Selenium WebDriver initialized successfully.")
        return driver
    except Exception as e:
        logging.error(f"Error initializing Selenium WebDriver: {e}")
        raise

def wait_for_manual_login(driver: webdriver.Chrome) -> None:
    """
    Prompt the user to log in manually and wait for confirmation.
    
    Args:
        driver (webdriver.Chrome): The Selenium WebDriver instance.
    """
    login_url = "https://xhamster.com/login"
    driver.get(login_url)
    logging.info("Navigated to login page. Please log in manually in the browser window.")
    input("After logging in, press Enter here to continue...")

def fetch_html(driver: webdriver.Chrome, url: str) -> Optional[str]:
    """
    Fetch the HTML content of the specified URL after scrolling to the bottom.

    Args:
        driver (webdriver.Chrome): The Selenium WebDriver instance.
        url (str): The URL to fetch.

    Returns:
        Optional[str]: The page source if successful, else None.
    """
    try:
        driver.get(url)
        logging.info(f"Navigated to {url}")
        scroll_to_bottom(driver)
        return driver.page_source
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None

def scroll_to_bottom(driver: webdriver.Chrome, pause_time: float = 2.0) -> None:
    """
    Scroll to the bottom of the page to ensure all dynamic content is loaded.

    Args:
        driver (webdriver.Chrome): The Selenium WebDriver instance.
        pause_time (float): Time to wait after each scroll.
    """
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_time)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            logging.info("Reached the bottom of the page.")
            break
        last_height = new_height

def parse_video_links(html: str) -> List[str]:
    """
    Parse the HTML content to extract video page links.

    Args:
        html (str): The HTML content of the favorites page.

    Returns:
        List[str]: A list of video page URLs.
    """
    soup = BeautifulSoup(html, 'html.parser')
    video_links = []
    video_anchors = soup.find_all('a', class_='thumb-image-container')  # Adjust selector if necessary
    for anchor in video_anchors:
        href = anchor.get('href')
        if href:
            video_url = urljoin('https://xhamster.com', href)
            video_links.append(video_url)
            logging.debug(f"Found video URL: {video_url}")
    logging.info(f"Parsed {len(video_links)} video links from the page.")
    return video_links

def get_video_title(session: requests.Session, video_url: str) -> str:
    """
    Fetch the video title from the video page.

    Args:
        session (requests.Session): The requests session with authenticated cookies.
        video_url (str): The URL of the video page.

    Returns:
        str: The sanitized video title.
    """
    try:
        response = session.get(video_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('h1')  # Adjust selector based on xHamster's HTML structure
        if title_tag:
            title = title_tag.get_text(strip=True)
            sanitized_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else "_" for c in title)
            logging.debug(f"Fetched title for {video_url}: {sanitized_title}")
            return sanitized_title.replace(" ", "_")
        else:
            logging.warning(f"Title not found for {video_url}. Using fallback title.")
            return f"video_{int(time.time())}"
    except Exception as e:
        logging.error(f"Error fetching title for {video_url}: {e}")
        return f"video_{int(time.time())}"

def create_authenticated_session(driver: webdriver.Chrome) -> requests.Session:
    """
    Create a requests.Session object with authenticated cookies from Selenium.

    Args:
        driver (webdriver.Chrome): The Selenium WebDriver instance.

    Returns:
        requests.Session: The authenticated session.
    """
    session = requests.Session()
    cookies = driver.get_cookies()
    for cookie in cookies:
        session.cookies.set(cookie['name'], cookie['value'], domain=cookie['domain'])
    logging.info("Authenticated requests session created from Selenium cookies.")
    return session

def download_video(video_url: str, title: str) -> None:
    """
    Download a video using yt-dlp with the given title as the filename.

    Args:
        video_url (str): The URL of the video to download.
        title (str): The title to use for the downloaded file.
    """
    filepath = os.path.join(DOWNLOAD_DIR, f"{title}.mp4")
    ydl_opts = {
        'outtmpl': filepath,  # Output template to save with title-based name
        'format': 'best',     # Choose the best quality format
        'quiet': False,       # Show download progress
        'no_warnings': True,
        'retries': MAX_RETRIES,
        'ignoreerrors': True,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logging.info(f"Starting download for '{title}' from {video_url} (Attempt {attempt})")
                ydl.download([video_url])
            logging.info(f"Successfully downloaded: {filepath}")
            break  # Exit the retry loop on success
        except Exception as e:
            logging.error(f"Failed to download {video_url} on attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                logging.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logging.error(f"Exceeded maximum retries for {video_url}. Skipping.")

def download_videos_concurrently(video_info_list: List[tuple]) -> None:
    """
    Download multiple videos concurrently.

    Args:
        video_info_list (List[tuple]): A list of tuples containing video URLs and their titles.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for video_url, title in video_info_list:
            futures.append(executor.submit(download_video, video_url, title))
            time.sleep(DOWNLOAD_DELAY)  # Delay to avoid rate limits
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"An error occurred during video download: {e}")

# ----------------------------- Main Workflow ----------------------------- #

def main() -> None:
    """Main function to orchestrate the scraping and downloading process."""
    ensure_download_dir()

    # Initialize Selenium WebDriver within a context manager to ensure proper cleanup
    with setup_selenium() as driver:
        try:
            # Step 1: Let the user log in manually
            wait_for_manual_login(driver)

            # Step 2: Fetch HTML content of favorites
            html_content = fetch_html(driver, SCRAPE_URL)
            if not html_content:
                logging.error("Failed to retrieve HTML content. Exiting.")
                return

            # Step 3: Parse video links
            video_page_links = parse_video_links(html_content)
            if not video_page_links:
                logging.warning("No video links found. Exiting.")
                return

            # Step 4: Create an authenticated requests session
            session = create_authenticated_session(driver)

            # Step 5: Fetch all video titles
            video_info_list = []
            for video_url in video_page_links:
                title = get_video_title(session, video_url)
                video_info_list.append((video_url, title))

            logging.info(f"Prepared {len(video_info_list)} videos for download.")

            # Step 6: Download videos concurrently
            download_videos_concurrently(video_info_list)

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
        finally:
            # Ensure the driver is closed
            driver.quit()
            logging.info("Selenium WebDriver has been closed.")

if __name__ == "__main__":
    main()
