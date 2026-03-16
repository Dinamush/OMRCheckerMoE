# downloader.py

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
import yt_dlp
import requests

# ----------------------------- Helper Functions ----------------------------- #

def setup_logging(log_file: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def ensure_download_dir(directory: str) -> None:
    """Ensure that the download directory exists."""
    try:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Download directory is set to: {directory}")
    except Exception as e:
        logging.error(f"Failed to create download directory '{directory}': {e}")
        raise

def setup_selenium(headless: bool = False) -> webdriver.Chrome:
    """Set up the Selenium WebDriver with specified options."""
    options = Options()
    if headless:
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--log-level=3")  # Suppress Selenium logs
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--no-sandbox")
    try:
        driver = webdriver.Chrome(options=options)  # 'chromedriver' is assumed to be on PATH
        logging.info("Selenium WebDriver initialized successfully.")
        return driver
    except WebDriverException as e:
        logging.error(f"Selenium WebDriver initialization failed: {e}")
        raise

def wait_for_manual_login(driver: webdriver.Chrome, login_url: str) -> None:
    """Prompt the user to log in manually and wait for confirmation."""
    try:
        driver.get(login_url)
        logging.info("Navigated to login page. Please log in manually in the browser window.")
        input("After logging in, press Enter here to continue...")
        logging.info("User confirmed login.")
    except Exception as e:
        logging.error(f"Error during manual login: {e}")
        raise

def fetch_html(driver: webdriver.Chrome, url: str) -> Optional[str]:
    """Fetch the HTML content of the specified URL after scrolling to the bottom."""
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
    """Scroll to the bottom of the page to ensure all dynamic content is loaded."""
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

def parse_video_links(html: str, base_url: str) -> List[str]:
    """Parse the HTML content to extract video page links based on the site structure."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        video_links = []

        # Site-specific condition for Pornhub
        if "pornhub" in base_url.lower():
            container = soup.find('ul', id='moreData')
            if container:
                video_items = container.find_all('li', class_='pcVideoListItem')
                for item in video_items:
                    data_id = item.get('data-id')
                    if data_id:
                        video_url = urljoin(base_url, f"/view_video.php?viewkey={data_id}")
                        video_links.append(video_url)
                        logging.debug(f"Found Pornhub video URL: {video_url}")
            else:
                logging.warning("No `moreData` container found.")
        else:
            video_anchors = soup.find_all('a', class_='thumb-image-container')
            for anchor in video_anchors:
                href = anchor.get('href')
                if href:
                    video_url = urljoin(base_url, href)
                    video_links.append(video_url)
                    logging.debug(f"Found video URL: {video_url}")

        logging.info(f"Parsed {len(video_links)} video links from the page.")
        return video_links
    except Exception as e:
        logging.error(f"Error parsing video links: {e}")
        return []

def create_authenticated_session(driver: webdriver.Chrome) -> requests.Session:
    """Create a requests.Session object with authenticated cookies from Selenium."""
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

def get_video_title(session: requests.Session, video_url: str) -> str:
    """Fetch the video title from the video page."""
    try:
        response = session.get(video_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('h1')  # Adjust selector based on site structure
        if title_tag:
            title = title_tag.get_text(strip=True)
            sanitized_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else "_" for c in title)
            return sanitized_title
        else:
            logging.warning(f"Title not found for {video_url}. Using fallback title.")
            return f"video_{int(time.time())}"
    except requests.RequestException as e:
        logging.error(f"Request error while fetching title for {video_url}: {e}")
        return f"video_{int(time.time())}"

def main_workflow(base_url: str,
                  favorites_url: str,
                  download_dir: str = "downloaded_videos",
                  headless: bool = False,
                  max_workers: int =5,
                  download_delay: float =1.0,
                  max_retries: int =3,
                  retry_delay: int =5,
                  log_file: str = 'video_downloader.log') -> None:
    """Main function to orchestrate the scraping and downloading process."""
    setup_logging(log_file)
    ensure_download_dir(download_dir)

    try:
        driver = setup_selenium(headless)
    except Exception as e:
        logging.critical(f"Failed to initialize Selenium WebDriver: {e}")
        return

    try:
        login_url = f"{base_url}/login"
        wait_for_manual_login(driver, login_url)

        all_video_links: Set[str] = set()
        page_number = 1

        # Loop through pages until no more content is available
        while True:
            current_page_url = f"{favorites_url}?page={page_number}"
            logging.info(f"Processing page {page_number}: {current_page_url}")
            html_content = fetch_html(driver, current_page_url)
            if not html_content:
                logging.error(f"Failed to retrieve HTML content for page {page_number}. Exiting pagination loop.")
                break

            video_page_links = parse_video_links(html_content, base_url)
            if not video_page_links:
                logging.warning(f"No video links found on page {page_number}.")
                break

            new_links = set(video_page_links) - all_video_links
            if not new_links:
                logging.info(f"No new video links found on page {page_number}. Exiting pagination loop.")
                break
            all_video_links.update(new_links)

            logging.info(f"Total unique videos collected so far: {len(all_video_links)}")
            page_number += 1

        if all_video_links:
            logging.info(f"Collected a total of {len(all_video_links)} unique video links.")
            session = create_authenticated_session(driver)
            video_info_list = [(url, get_video_title(session, url)) for url in all_video_links]
            download_videos_concurrently(video_info_list, download_dir, max_workers, download_delay)

    except Exception as e:
        logging.critical(f"An unexpected error occurred in the main workflow: {e}")
    finally:
        driver.quit()
        logging.info("Selenium WebDriver has been closed.")
