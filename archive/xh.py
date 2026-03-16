import os
import logging
import time
import concurrent.futures
from typing import List, Optional, Tuple, Set
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

# ----------------------------- Configuration ----------------------------- #

# Logging configuration
LOG_FILE = 'logs/video_downloader.log'
# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Constants
BASE_URL = 'https://xhamster.com'
FAVORITES_URL = f'{BASE_URL}/my/favorites/videos'
DOWNLOAD_DIR = "downloads"

# Selenium WebDriver settings
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
    try:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Download directory is set to: {directory}")
    except Exception as e:
        logging.error(f"Failed to create download directory '{directory}': {e}")
        raise

def setup_selenium() -> webdriver.Chrome:
    """Set up the Selenium WebDriver with specified options."""
    options = Options()
    if HEADLESS:
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--log-level=3")  # Suppress Selenium logs
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    try:
        # Use webdriver-manager to automatically download and manage ChromeDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        # Execute script to remove webdriver property
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        logging.info("Selenium WebDriver initialized successfully.")
        return driver
    except WebDriverException as e:
        logging.error(f"Selenium WebDriver initialization failed: {e}")
        raise

def wait_for_manual_login(driver: webdriver.Chrome) -> None:
    """
    Prompt the user to log in manually and wait for confirmation.

    Args:
        driver (webdriver.Chrome): The Selenium WebDriver instance.
    """
    login_url = "https://xhamster.com/login"
    try:
        driver.get(login_url)
        logging.info("Navigated to login page. Please log in manually in the browser window.")
        input("After logging in, press Enter here to continue...")
        logging.info("User confirmed login.")
    except Exception as e:
        logging.error(f"Error during manual login: {e}")
        raise

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
    except TimeoutException:
        logging.error(f"Timeout while fetching {url}")
    except WebDriverException as e:
        logging.error(f"WebDriver exception while fetching {url}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while fetching {url}: {e}")
    return None

def scroll_to_bottom(driver: webdriver.Chrome, pause_time: float = 2.0) -> None:
    """
    Scroll to the bottom of the page to ensure all dynamic content is loaded.

    Args:
        driver (webdriver.Chrome): The Selenium WebDriver instance.
        pause_time (float): Time to wait after each scroll.
    """
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
    except WebDriverException as e:
        logging.error(f"Error during scrolling: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during scrolling: {e}")

def parse_video_links(html: str) -> List[str]:
    """
    Parse the HTML content to extract video page links.

    Args:
        html (str): The HTML content of the favorites page.

    Returns:
        List[str]: A list of video page URLs.
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
        video_links = []
        video_anchors = soup.find_all('a', class_='thumb-image-container')  # Adjust selector if necessary
        if not video_anchors:
            logging.warning("No video anchors found with class 'thumb-image-container'. Check the selector.")
        for anchor in video_anchors:
            href = anchor.get('href')
            if href:
                video_url = urljoin(BASE_URL, href)
                video_links.append(video_url)
                logging.debug(f"Found video URL: {video_url}")
        logging.info(f"Parsed {len(video_links)} video links from the page.")
        return video_links
    except Exception as e:
        logging.error(f"Error parsing video links: {e}")
        return []

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
        response = session.get(video_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('h1')  # Adjust selector based on xHamster's HTML structure
        if title_tag:
            title = title_tag.get_text(strip=True)
            sanitized_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else "_" for c in title)
            sanitized_title = sanitized_title.replace(" ", "_")
            logging.debug(f"Fetched title for {video_url}: {sanitized_title}")
            return sanitized_title
        else:
            logging.warning(f"Title not found for {video_url}. Using fallback title.")
            return f"video_{int(time.time())}"
    except requests.RequestException as e:
        logging.error(f"Request error while fetching title for {video_url}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while fetching title for {video_url}: {e}")
    return f"video_{int(time.time())}"  # Fallback title

def create_authenticated_session(driver: webdriver.Chrome) -> requests.Session:
    """
    Create a requests.Session object with authenticated cookies from Selenium.

    Args:
        driver (webdriver.Chrome): The Selenium WebDriver instance.

    Returns:
        requests.Session: The authenticated session.
    """
    try:
        session = requests.Session()
        cookies = driver.get_cookies()
        for cookie in cookies:
            domain = cookie['domain'].lstrip('.')  # Remove leading dot
            session.cookies.set(cookie['name'], cookie['value'], domain=domain)
        logging.info("Authenticated requests session created from Selenium cookies.")
        return session
    except Exception as e:
        logging.error(f"Error creating authenticated session: {e}")
        raise

def download_video(video_url: str, title: str, failed_downloads: List[Tuple[str, str]]) -> None:
    """
    Download a video using yt-dlp with the given title as the filename.

    Args:
        video_url (str): The URL of the video to download.
        title (str): The title to use for the downloaded file.
        failed_downloads (List[Tuple[str, str]]): List to record failed downloads.
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
        except yt_dlp.utils.DownloadError as e:
            logging.error(f"DownloadError for {video_url} on attempt {attempt}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during download of {video_url} on attempt {attempt}: {e}")
        if attempt < MAX_RETRIES:
            sleep_time = RETRY_DELAY * attempt  # Exponential backoff
            logging.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
        else:
            logging.error(f"Exceeded maximum retries for {video_url}. Skipping.")
            failed_downloads.append((video_url, title))

def download_videos_concurrently(video_info_list: List[Tuple[str, str]]) -> None:
    """
    Download multiple videos concurrently.

    Args:
        video_info_list (List[Tuple[str, str]]): A list of tuples containing video URLs and their titles.
    """
    failed_downloads: List[Tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for video_url, title in video_info_list:
            futures.append(executor.submit(download_video, video_url, title, failed_downloads))
            time.sleep(DOWNLOAD_DELAY)  # Delay to avoid rate limits
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"An unexpected error occurred during video download: {e}")
    if failed_downloads:
        logging.warning(f"{len(failed_downloads)} videos failed to download:")
        for url, title in failed_downloads:
            logging.warning(f" - {title}: {url}")
    else:
        logging.info("All videos downloaded successfully.")

def find_next_page(driver: webdriver.Chrome) -> Optional[str]:
    """
    Find the URL of the next page in favorites.

    Args:
        driver (webdriver.Chrome): The Selenium WebDriver instance.

    Returns:
        Optional[str]: The URL of the next page if found, else None.
    """
    try:
        # Attempt to find the "Next" button by link text
        next_button = driver.find_element(By.LINK_TEXT, 'Next')
        if next_button:
            next_page_url = next_button.get_attribute('href')
            logging.info(f"Found next page URL: {next_page_url}")
            return next_page_url
    except NoSuchElementException:
        logging.info("No 'Next' button found. Reached the last page.")
    except Exception as e:
        logging.error(f"Error finding next page: {e}")
    return None

# ----------------------------- Main Workflow ----------------------------- #

def main() -> None:
    """Main function to orchestrate the scraping and downloading process."""
    ensure_download_dir()
    driver = None

    # Initialize Selenium WebDriver within a context manager to ensure proper cleanup
    try:
        driver = setup_selenium()
    except Exception as e:
        logging.critical(f"Failed to initialize Selenium WebDriver: {e}")
        raise

    try:
        # Step 1: Let the user log in manually
        wait_for_manual_login(driver)

        # Step 2: Initialize variables for pagination
        current_page_url = FAVORITES_URL
        all_video_links: Set[str] = set()  # Use a set to avoid duplicates
        page_number = 1

        # Step 3: Loop through all pages
        while current_page_url:
            logging.info(f"Processing page {page_number}: {current_page_url}")
            html_content = fetch_html(driver, current_page_url)
            if not html_content:
                logging.error(f"Failed to retrieve HTML content for page {page_number}. Exiting pagination loop.")
                break

            # Parse video links from the current page
            video_page_links = parse_video_links(html_content)
            if not video_page_links:
                logging.warning(f"No video links found on page {page_number}.")
                # Decide whether to continue or break based on application logic
                break

            new_links = set(video_page_links) - all_video_links
            if not new_links:
                logging.info(f"No new video links found on page {page_number}. Exiting pagination loop.")
                break
            all_video_links.update(new_links)

            logging.info(f"Total unique videos collected so far: {len(all_video_links)}")

            # Find the next page URL
            next_page_url = find_next_page(driver)
            if next_page_url and next_page_url != current_page_url:
                current_page_url = next_page_url
                page_number += 1
            else:
                logging.info("No further pages to process.")
                break

        if all_video_links:
            logging.info(f"Collected a total of {len(all_video_links)} unique video links.")

            # Step 4: Create an authenticated requests session
            try:
                session = create_authenticated_session(driver)
            except Exception as e:
                logging.critical(f"Failed to create authenticated session: {e}")
                raise

            # Step 5: Fetch all video titles
            video_info_list: List[Tuple[str, str]] = []
            for idx, video_url in enumerate(all_video_links, start=1):
                try:
                    logging.info(f"Fetching title for video {idx}/{len(all_video_links)}: {video_url}")
                    title = get_video_title(session, video_url)
                    video_info_list.append((video_url, title))
                except Exception as e:
                    logging.error(f"Failed to fetch title for {video_url}: {e}")
                    # Optionally, skip adding this video or add with a fallback title
                    video_info_list.append((video_url, f"video_{int(time.time())}"))

            logging.info(f"Prepared {len(video_info_list)} videos for download.")

            # Step 6: Download videos concurrently
            download_videos_concurrently(video_info_list)

        else:
            logging.warning("No video links found across all pages.")

    except Exception as e:
        logging.critical(f"An unexpected error occurred in the main workflow: {e}")
    finally:
        # Ensure the driver is closed
        if driver is not None:
            try:
                driver.quit()
                logging.info("Selenium WebDriver has been closed.")
            except Exception as e:
                logging.error(f"Error closing Selenium WebDriver: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Fatal error")
        input("\nPress Enter to exit...")
        raise
