#!/usr/bin/env python3
"""
Usage:
    python video_scraper.py "https://www.tnaflix.com/cum-videos/Cum-Swapping-Babes-PMV/video5930362?from_pop=1" --output downloads [--use_ffmpeg]

This script:
  1. Downloads the HTML content of the provided URL.
  2. Extracts a direct video URL by looking for a <video> element or an <a> tag with '.mp4' in the href.
  3. Downloads the video into the specified output folder using either yt-dlp (default) or ffmpeg (with --use_ffmpeg).

When using ffmpeg, the script captures the video stream as it plays and saves it as "downloaded_video.mp4".
"""

import os
import argparse
import logging
import subprocess
import requests
from bs4 import BeautifulSoup

def extract_video_url(html):
    """
    Parses HTML to extract a direct video URL.
    
    - First, searches for a <video> element. If found, returns its 'src' or that of a nested <source>.
    - If not, scans for any <a> tag whose href contains '.mp4'.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Look for a <video> tag.
    video = soup.find('video')
    if video:
        src = video.get('src')
        if src:
            logging.info(f"Found <video> element with src: {src}")
            return src
        # If <video> tag lacks a src, try a nested <source>.
        source = video.find('source')
        if source:
            src = source.get('src')
            if src:
                logging.info(f"Found <source> element with src: {src}")
                return src

    # Fallback: look for an <a> tag containing '.mp4'.
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '.mp4' in href:
            logging.info(f"Found <a> tag linking to an MP4: {href}")
            return href

    logging.error("No direct video URL found in the HTML.")
    return None

def download_video_ytdlp(video_url, output_folder):
    """
    Uses yt-dlp to download the video from the provided direct URL.
    Saves the video using a title-based output template in the specified folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_template = os.path.join(output_folder, "%(title)s.%(ext)s")
    cmd = ["yt-dlp", "-o", output_template, video_url]
    logging.info("Running yt-dlp command: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        logging.info("Download via yt-dlp successful!")
    except subprocess.CalledProcessError as e:
        logging.error(f"yt-dlp failed: {e}")
    except FileNotFoundError:
        logging.error("yt-dlp not found. Please ensure it is installed and in your PATH.")

def download_video_ffmpeg(video_url, output_folder):
    """
    Uses ffmpeg to capture and download the video stream from the provided direct URL.
    Saves the stream as 'downloaded_video.mp4' in the specified folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, "downloaded_video.mp4")
    # The command uses ffmpeg to capture the stream (-i video_url) and copy the codecs without re-encoding.
    cmd = ["ffmpeg", "-y", "-i", video_url, "-c", "copy", output_file]
    logging.info("Running ffmpeg command: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        logging.info("Download via ffmpeg successful!")
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg failed: {e}")
    except FileNotFoundError:
        logging.error("ffmpeg not found. Please ensure it is installed and in your PATH.")

def main():
    parser = argparse.ArgumentParser(description="Scrape and download a video using direct URL access (no browser).")
    parser.add_argument("url", help="The URL of the webpage containing the video.")
    parser.add_argument("--output", default="downloads", help="Output folder for the downloaded video (default: downloads).")
    parser.add_argument("--use_ffmpeg", action="store_true", help="Use ffmpeg to download the video instead of yt-dlp.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting video scraper...")

    # Fetch the webpage.
    try:
        response = requests.get(args.url)
        if response.status_code != 200:
            logging.error(f"Failed to fetch the page. Status code: {response.status_code}")
            return
    except Exception as e:
        logging.error(f"Error fetching URL {args.url}: {e}")
        return

    # Extract video URL from the HTML.
    video_url = extract_video_url(response.text)
    if not video_url:
        logging.error("Could not extract a direct video URL from the page.")
        return
    logging.info(f"Direct video URL extracted: {video_url}")

    # Download the video using the chosen method.
    if args.use_ffmpeg:
        download_video_ffmpeg(video_url, args.output)
    else:
        download_video_ytdlp(video_url, args.output)

if __name__ == "__main__":
    main()
