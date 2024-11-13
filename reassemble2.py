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
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SCRAPE_URL = 'https://xhamster.com/my/favorites/videos'
DOWNLOAD_DIR = "downloaded_videos"
CHROME_DRIVER_PATH = r"C:\chromedriver.exe"
FFMPEG_PATH = "ffmpeg"  # Adjust this if ffmpeg is not in PATH
HEADLESS = False

def ensure_download_dir():
    """Ensure that the download directory exists."""
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        logging.info(f"Created directory {DOWNLOAD_DIR}")

def setup_selenium():
    """Setup Selenium WebDriver with options."""
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
    """Fetch HTML from URL and scroll to load all content."""
    try:
        driver.get(url)
        logging.info(f"Navigated to {url}")
        scroll_to_bottom(driver)
        return driver.page_source
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None

def scroll_to_bottom(driver):
    """Scroll to the bottom of the page to load dynamic content."""
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
    """Extract video page URLs from the HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    video_links = []
    video_anchors = soup.find_all('a', class_='thumb-image-container')
    for anchor in video_anchors:
        href = anchor.get('href')
        if href:
            video_url = urljoin('https://xhamster.com', href)
            video_links.append(video_url)
    return video_links

def transfer_cookies(driver, session):
    """Transfer cookies from Selenium to requests session for authenticated requests."""
    for cookie in driver.get_cookies():
        session.cookies.set(cookie['name'], cookie['value'], domain=cookie['domain'])

def get_video_download_url_selenium(driver, video_page_url):
    """Retrieve video download URL, handle HLS (.m3u8) if present."""
    try:
        driver.get(video_page_url)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        scripts = soup.find_all('script')
        for script in scripts:
            if 'window.initials=' in script.text:
                json_text = script.text.split('window.initials=')[1].rstrip(';')
                data = json.loads(json_text)
                video_sources = data['videoModel']['sources']['mp4']
                if video_sources:
                    return video_sources[0]['link']  # Direct MP4 link
                # Check for HLS stream
                hls_sources = data['videoModel']['sources'].get('hls', [])
                if hls_sources:
                    return hls_sources[0]['link']  # HLS playlist link
    except Exception as e:
        logging.error(f"Failed to get video URL from {video_page_url}: {e}")
    return None

def download_segment(session, segment_url, output_path):
    """Download a single HLS segment."""
    try:
        with session.get(segment_url, stream=True) as response:
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        logging.info(f"Downloaded segment: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to download segment {segment_url}: {e}")
        return False

def download_hls_segments(session, playlist_url, output_dir):
    """Download segments from an HLS playlist."""
    response = session.get(playlist_url)
    playlist_content = response.text
    segment_urls = [urljoin(playlist_url, line.strip()) for line in playlist_content.splitlines()
                    if line and not line.startswith("#") and line.endswith('.m4s')]
    segment_files = []
    for idx, seg_url in enumerate(segment_urls):
        segment_filename = f"segment_{idx + 1}.m4s"
        segment_filepath = os.path.join(output_dir, segment_filename)
        if download_segment(session, seg_url, segment_filepath):
            segment_files.append(segment_filepath)
    return segment_files

def assemble_video_ffmpeg(segment_files, output_file):
    """Use FFmpeg to concatenate video segments."""
    file_list_path = os.path.join(os.path.dirname(output_file), "file_list.txt")
    with open(file_list_path, "w") as f:
        for segment_file in segment_files:
            f.write(f"file '{os.path.abspath(segment_file)}'\n")
    try:
        subprocess.run([FFMPEG_PATH, "-f", "concat", "-safe", "0", "-i", file_list_path, "-c", "copy", output_file],
                       check=True)
        logging.info(f"Combined video saved as {output_file}")
    finally:
        os.remove(file_list_path)
        for seg_file in segment_files:
            os.remove(seg_file)

def download_video(session, video_url, filepath):
    """Direct download for single MP4 videos."""
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
    """Download all videos, handle HLS recombination if needed."""
    ensure_download_dir()
    session = requests.Session()
    transfer_cookies(driver, session)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for idx, video_page_url in enumerate(video_page_links):
            video_url = get_video_download_url_selenium(driver, video_page_url)
            if video_url:
                filepath = os.path.join(DOWNLOAD_DIR, f"video_{idx + 1}.mp4")
                if video_url.endswith('.m3u8'):
                    output_subdir = os.path.join(DOWNLOAD_DIR, f"video_{idx + 1}_segments")
                    os.makedirs(output_subdir, exist_ok=True)
                    segment_files = download_hls_segments(session, video_url, output_subdir)
                    if segment_files:
                        assemble_video_ffmpeg(segment_files, filepath)
                else:
                    executor.submit(download_video, session, video_url, filepath)
                time.sleep(1)

def main():
    driver = setup_selenium()
    wait_for_manual_login(driver)
    html_content = fetch_html(driver, SCRAPE_URL)
    if not html_content:
        driver.quit()
        return
    video_page_links = parse_video_links(html_content)
    if video_page_links:
        logging.info(f"Found {len(video_page_links)} video page links.")
        download_videos(driver, video_page_links)
    driver.quit()

if __name__ == "__main__":
    main()
