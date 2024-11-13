import os
import logging
import requests
import json
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urljoin
import concurrent.futures
import time
from selenium import webdriver
import subprocess

# --------------------------- Configuration ---------------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
SCRAPE_URL = 'https://xhamster.com/my/favorites/videos'
DOWNLOAD_DIR = "downloaded_videos"
CHROME_DRIVER_PATH = r"C:\chromedriver.exe"  # Update this path
HEADLESS = False  # Set to True to run headlessly
MAX_WORKERS = 4  # Number of concurrent downloads
FFMPEG_PATH = "ffmpeg"  # Set to "ffmpeg" if in PATH, else full path like r"C:\ffmpeg\bin\ffmpeg.exe"

# --------------------------- Helper Functions ---------------------------

def ensure_download_dir():
    """Ensure that the download directory exists."""
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        logging.info(f"Created directory {DOWNLOAD_DIR}")

def setup_selenium():
    """Configure and initialize the Selenium WebDriver."""
    options = Options()
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    user_agent = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36")
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

def scroll_to_bottom(driver, max_scrolls=50):
    """Scroll the page to load all dynamic content."""
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

def fetch_html(driver, url):
    """Navigate to the URL and return the page source after scrolling."""
    try:
        driver.get(url)
        logging.info(f"Navigated to {url}")
        scroll_to_bottom(driver)
        return driver.page_source
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None

def parse_video_links(html):
    """Parse the HTML to extract individual video page URLs."""
    soup = BeautifulSoup(html, 'html.parser')
    video_links = []
    # Attempt multiple selectors to find video links
    selectors = [
        'a.thumb-image-container',
        'a.video-thumb__image-link',
        'a[href*="/video/"]'
    ]
    for selector in selectors:
        video_anchors = soup.select(selector)
        if video_anchors:
            for anchor in video_anchors:
                href = anchor.get('href')
                if href and '/video/' in href:
                    video_url = urljoin('https://xhamster.com', href)
                    video_links.append(video_url)
                    logging.debug(f"Found video URL: {video_url}")
            if video_links:
                break  # Stop if links are found with a selector
    if not video_links:
        logging.warning("No video links found in the page.")
    else:
        logging.info(f"Found {len(video_links)} unique video links.")
    return list(set(video_links))  # Remove duplicates

def transfer_cookies(driver, session):
    """Transfer cookies from Selenium to requests.Session and set necessary headers."""
    logging.info("Transferring cookies from Selenium to requests.Session")
    for cookie in driver.get_cookies():
        c = requests.cookies.create_cookie(
            domain=cookie.get('domain'),
            name=cookie.get('name'),
            value=cookie.get('value'),
            path=cookie.get('path', '/'),
            secure=cookie.get('secure', False),
            expires=cookie.get('expiry'),
            rest={'httpOnly': cookie.get('httpOnly', False)}
        )
        session.cookies.set_cookie(c)
    # Set headers to mimic browser
    session.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36"),
        "Referer": SCRAPE_URL,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    })
    logging.info("Transferred cookies and set headers to requests session.")

def get_video_download_url_selenium(driver, video_page_url):
    """Retrieve the direct video download URL by parsing the <noscript> tag or embedded JSON."""
    try:
        driver.get(video_page_url)
        logging.info(f"Fetching video download URL from {video_page_url}")
        # Wait for the video player to load
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'video')))
        time.sleep(2)  # Additional wait for dynamic content

        # Parse the page source
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        # Attempt to find the <noscript> tag
        noscript_tag = soup.find('noscript')
        if noscript_tag:
            noscript_html = noscript_tag.decode_contents()
            nosoup = BeautifulSoup(noscript_html, 'html.parser')
            video_tag = nosoup.find('video')
            if video_tag and video_tag.get('src'):
                video_url = video_tag['src']
                logging.info(f"Found video URL in <noscript> tag: {video_url}")
                return video_url
            else:
                logging.warning("Video tag not found inside <noscript>.")
        else:
            logging.warning("<noscript> tag not found in page.")

        # Alternative: Extract from JSON in script tags
        scripts = soup.find_all('script')
        for script in scripts:
            if 'window.initials=' in script.text:
                try:
                    json_text = script.text.split('window.initials=')[1].rstrip(';')
                    data = json.loads(json_text)
                    sources = data.get('videoModel', {}).get('sources', {}).get('mp4', [])
                    if sources:
                        # Assuming the last source is the highest quality
                        video_url = sources[-1].get('link')
                        if video_url:
                            logging.info(f"Found video URL in JSON data: {video_url}")
                            return video_url
                    logging.warning("No video sources found in JSON data.")
                except (json.JSONDecodeError, IndexError, KeyError) as e:
                    logging.error(f"Error parsing JSON data: {e}")
        logging.warning(f"Could not find direct video URL for {video_page_url}")
        return None
    except Exception as e:
        logging.error(f"Failed to get video download URL from {video_page_url}: {e}")
        return None

def download_video(session, video_url, filepath, retries=3):
    """Download a single video file with retry mechanism."""
    attempt = 0
    while attempt < retries:
        try:
            logging.info(f"Starting download: {video_url} (Attempt {attempt + 1})")
            with session.get(video_url, stream=True, timeout=60) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size:
                                percent = downloaded / total_size * 100
                                logging.debug(f"Downloaded {downloaded} of {total_size} bytes ({percent:.2f}%)")
            logging.info(f"Downloaded: {filepath}")
            return True  # Success
        except requests.exceptions.RequestException as e:
            attempt += 1
            logging.error(f"Failed to download {video_url} (Attempt {attempt}): {e}")
            if attempt == retries:
                logging.error(f"Exhausted retries for {video_url}")
        except Exception as e:
            logging.error(f"Error saving video {filepath}: {e}")
            break  # Don't retry on other exceptions
    return False  # Failed

def download_segment(session, segment_url, output_path, retries=3):
    """Download a single video segment with retry mechanism."""
    attempt = 0
    while attempt < retries:
        try:
            logging.info(f"Downloading segment: {segment_url} (Attempt {attempt + 1})")
            with session.get(segment_url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            logging.info(f"Downloaded segment: {output_path}")
            return True
        except requests.RequestException as e:
            attempt += 1
            logging.error(f"Failed to download segment {segment_url} (Attempt {attempt}): {e}")
            if attempt == retries:
                logging.error(f"Exhausted retries for segment {segment_url}")
        except Exception as e:
            logging.error(f"Error saving segment {output_path}: {e}")
            break  # Don't retry on other exceptions
    return False  # Failed

def download_hls_segments(session, playlist_url, output_dir):
    """Download all segments from an HLS playlist."""
    try:
        logging.info(f"Fetching HLS playlist: {playlist_url}")
        response = session.get(playlist_url, timeout=30)
        response.raise_for_status()
        playlist_content = response.text

        # Parse the playlist to extract segment URLs
        segment_urls = []
        for line in playlist_content.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and line.endswith('.m4s'):
                segment_url = urljoin(playlist_url, line)
                segment_urls.append(segment_url)

        if not segment_urls:
            logging.warning(f"No segments found in playlist: {playlist_url}")
            return []

        logging.info(f"Found {len(segment_urls)} segments in playlist.")

        # Download all segments
        segment_files = []
        for idx, seg_url in enumerate(segment_urls):
            segment_filename = f"segment_{idx + 1}.m4s"
            segment_filepath = os.path.join(output_dir, segment_filename)
            success = download_segment(session, seg_url, segment_filepath)
            if success:
                segment_files.append(segment_filepath)
            else:
                logging.error(f"Failed to download segment: {seg_url}")
                # Optionally, implement retry logic or abort
        return segment_files
    except requests.RequestException as e:
        logging.error(f"Failed to fetch HLS playlist {playlist_url}: {e}")
        return []
    except Exception as e:
        logging.error(f"Error processing HLS playlist {playlist_url}: {e}")
        return []

def assemble_video_ffmpeg(segment_files, output_file):
    """Assemble video segments into a single video file using FFmpeg."""
    try:
        # Create a file list for FFmpeg
        file_list_path = os.path.join(os.path.dirname(output_file), "file_list.txt")
        with open(file_list_path, "w") as f:
            for segment_file in segment_files:
                f.write(f"file '{os.path.abspath(segment_file)}'\n")

        # Run FFmpeg to concatenate segments
        logging.info(f"Assembling video using FFmpeg: {output_file}")
        result = subprocess.run(
            [FFMPEG_PATH, "-f", "concat", "-safe", "0", "-i", file_list_path, "-c", "copy", output_file],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info(f"Combined video saved as {output_file}")

        # Clean up segment files and file list
        for segment_file in segment_files:
            try:
                os.remove(segment_file)
                logging.debug(f"Removed segment file: {segment_file}")
            except OSError as e:
                logging.warning(f"Could not remove segment file {segment_file}: {e}")
        try:
            os.remove(file_list_path)
            logging.debug(f"Removed file list: {file_list_path}")
        except OSError as e:
            logging.warning(f"Could not remove file list {file_list_path}: {e}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed to assemble video {output_file}: {e.stderr.decode().strip()}")
    except Exception as e:
        logging.error(f"Error assembling video {output_file}: {e}")

def process_video(driver, session, idx, video_page_url):
    """Process a single video: extract download URL and download the video."""
    try:
        logging.info(f"Processing video {idx + 1}: {video_page_url}")
        video_url = get_video_download_url_selenium(driver, video_page_url)
        if video_url:
            # Determine if the video URL is a direct MP4 or an HLS playlist (.m3u8)
            if video_url.endswith('.m3u8'):
                logging.info(f"Detected HLS stream for video {idx + 1}: {video_url}")
                output_subdir = os.path.join(DOWNLOAD_DIR, f"video_{idx + 1}_segments")
                os.makedirs(output_subdir, exist_ok=True)
                segment_files = download_hls_segments(session, video_url, output_subdir)
                if segment_files:
                    output_file = os.path.join(DOWNLOAD_DIR, f"video_{idx + 1}.mp4")
                    assemble_video_ffmpeg(segment_files, output_file)
                else:
                    logging.error(f"No segments downloaded for video {idx + 1}")
            else:
                # Assume it's a direct download
                filename = os.path.basename(video_url.split('?')[0])  # Remove query params
                # Ensure unique filenames by prefixing with index
                filename = f"video_{idx + 1}_{filename}"
                filepath = os.path.join(DOWNLOAD_DIR, filename)
                success = download_video(session, video_url, filepath)
                if not success:
                    logging.error(f"Failed to download video {video_page_url}")
        else:
            logging.warning(f"No download URL found for video {video_page_url}")
    except Exception as e:
        logging.error(f"Error processing video {video_page_url}: {e}")

def download_videos(driver, session, video_page_links):
    """Download all videos from the list of video page URLs using multithreading."""
    ensure_download_dir()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for idx, video_page_url in enumerate(video_page_links):
            # Submit the process_video function to the executor
            futures.append(executor.submit(process_video, driver, session, idx, video_page_url))
            time.sleep(1)  # Delay to avoid rate limits
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in thread: {e}")

# --------------------------- Main Function ---------------------------

def main():
    driver = None
    try:
        driver = setup_selenium()

        # Step 1: Let the user log in manually
        wait_for_manual_login(driver)

        # Step 2: Fetch HTML content of the favorites page
        html_content = fetch_html(driver, SCRAPE_URL)
        if not html_content:
            logging.error("Failed to retrieve HTML content.")
            return

        # Step 3: Parse video links from the favorites page
        video_page_links = parse_video_links(html_content)
        if video_page_links:
            logging.info(f"Found {len(video_page_links)} video page links.")
            # Step 4: Transfer cookies to requests.Session for authenticated downloads
            session = requests.Session()
            transfer_cookies(driver, session)
            # Step 5: Download all videos
            download_videos(driver, session, video_page_links)
        else:
            logging.warning("No video links found.")
    except Exception as e:
        logging.error(f"An error occurred in main: {e}")
    finally:
        # Close the Selenium driver
        if driver:
            driver.quit()
            logging.info("Selenium driver closed.")

# --------------------------- Entry Point ---------------------------

if __name__ == "__main__":
    main()
