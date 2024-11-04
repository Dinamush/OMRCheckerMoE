import os
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants (Replace these with actual values)
LOGIN_URL = 'https://xhamster.com/login'  # Replace with the actual login URL
SCRAPE_URL = 'https://xhamster.com/my/favorites/videos'  # Replace with the actual URL to scrape
USERNAME = 'Burninglove999'  # Replace with your username
PASSWORD = 'aceofblades'  # Replace with your password
USERNAME_FIELD = 'Burninglove999'  # Replace with the actual form field name for username
PASSWORD_FIELD = 'aceofblades'  # Replace with the actual form field name for password
VIDEO_CONTAINER_CLASS = "thumb-list_item video-thumb--type-video"
VIDEO_LINK_CLASS = "video-thumb__image-container"
DOWNLOAD_DIR = "downloaded_videos"

def ensure_download_dir():
    """Ensure the download directory exists."""
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        logging.info(f"Created directory {DOWNLOAD_DIR}")

def login(session):
    """Perform login and return True if successful."""
    try:
        # Fetch the login page to get CSRF token (if required)
        response = session.get(LOGIN_URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract CSRF token if it's part of the login form
        csrf_token = soup.find('input', {'name': 'csrf_token'})
        login_data = {
            USERNAME_FIELD: USERNAME,
            PASSWORD_FIELD: PASSWORD
        }
        if csrf_token:
            login_data['csrf_token'] = csrf_token['value']

        # Send POST request to log in
        response = session.post(LOGIN_URL, data=login_data)
        response.raise_for_status()

        # Verify login success (modify this check according to the website's response)
        if "incorrect username or password" in response.text.lower():
            logging.error("Incorrect username or password.")
            return False

        logging.info("Logged in successfully.")
        return True

    except requests.RequestException as e:
        logging.error(f"Login request failed: {e}")
        return False
    except Exception as e:
        logging.error(f"An error occurred during login: {e}")
        return False

def fetch_html(session, url):
    """Fetch the HTML content of the webpage using the given session."""
    try:
        response = session.get(url)
        response.raise_for_status()
        logging.info(f"Fetched HTML content from {url}")
        return response.text
    except requests.RequestException as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None

def parse_video_links(html, base_url):
    """Parse video links from HTML content."""
    soup = BeautifulSoup(html, 'html.parser')
    video_divs = soup.find_all("div", class_=VIDEO_CONTAINER_CLASS)
    video_links = []

    for video_div in video_divs:
        link_tag = video_div.find("a", class_=VIDEO_LINK_CLASS)
        if link_tag and 'href' in link_tag.attrs:
            video_url = urljoin(base_url, link_tag['href'])
            video_links.append(video_url)
            logging.debug(f"Found video URL: {video_url}")

    return video_links

def download_video(session, url, filepath):
    """Download a video from the given URL."""
    try:
        with session.get(url, stream=True) as response:
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        logging.info(f"Downloaded: {filepath}")
    except requests.RequestException as e:
        logging.error(f"Failed to download {url}: {e}")
    except Exception as e:
        logging.error(f"An error occurred while downloading {url}: {e}")

def download_videos(session, video_links):
    """Download all videos from the list of video links."""
    ensure_download_dir()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for idx, video_url in enumerate(video_links):
            filename = f"video_{idx + 1}.mp4"  # Adjust extension as necessary
            filepath = os.path.join(DOWNLOAD_DIR, filename)
            futures.append(executor.submit(download_video, session, video_url, filepath))

        # Wait for all downloads to complete
        for future in concurrent.futures.as_completed(futures):
            pass  # Errors are logged in download_video

def main():
    # Initialize session
    session = requests.Session()

    # Step 1: Log in
    if not login(session):
        logging.error("Exiting due to login failure.")
        return

    # Step 2: Fetch HTML content
    html_content = fetch_html(session, SCRAPE_URL)
    if html_content is None:
        logging.error("Failed to retrieve HTML content.")
        return

    # Step 3: Parse video links
    video_links = parse_video_links(html_content, SCRAPE_URL)
    if not video_links:
        logging.warning("No video links found.")
        return
    logging.info(f"Found {len(video_links)} video links.")

    # Step 4: Download videos
    download_videos(session, video_links)

if __name__ == "__main__":
    main()
