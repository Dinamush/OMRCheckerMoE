#!/usr/bin/env python3
"""
Progress Tracker for Video Downloader
Handles progress tracking, logging, and statistics for video downloads
"""

import os
import time
import json
import logging
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from pathlib import Path

@dataclass
class DownloadStats:
    """Statistics for download progress"""
    total_urls: int = 0
    urls_collected: int = 0
    downloads_started: int = 0
    downloads_completed: int = 0
    downloads_failed: int = 0
    files_not_found: int = 0
    files_skipped: int = 0
    total_size_downloaded: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total_urls == 0:
            return 0.0
        return (self.downloads_completed + self.downloads_failed + self.files_skipped) / self.total_urls * 100
    
    @property
    def remaining_downloads(self) -> int:
        """Calculate remaining downloads"""
        return self.total_urls - (self.downloads_completed + self.downloads_failed + self.files_skipped)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.downloads_started == 0:
            return 0.0
        return self.downloads_completed / self.downloads_started * 100

@dataclass
class DownloadItem:
    """Individual download item tracking"""
    url: str
    title: str
    status: str = "pending"  # pending, downloading, completed, failed, skipped, not_found
    file_path: Optional[str] = None
    file_size: int = 0
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0

class ProgressTracker:
    """Main progress tracking class"""
    
    def __init__(self, session_id: str, log_dir: str = "logs"):
        self.session_id = session_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize tracking data
        self.stats = DownloadStats()
        self.download_items: Dict[str, DownloadItem] = {}
        self.failed_downloads: List[DownloadItem] = []
        self.successful_downloads: List[DownloadItem] = []
        self.skipped_downloads: List[DownloadItem] = []
        self.not_found_downloads: List[DownloadItem] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Setup logging
        self._setup_logging()
        
        # Progress file for web interface
        self.progress_file = self.log_dir / f"progress_{session_id}.json"
        
    def _setup_logging(self):
        """Setup detailed logging for this session"""
        log_file = self.log_dir / f"download_progress_{self.session_id}.log"
        
        # Create a separate logger for this session
        self.logger = logging.getLogger(f"progress_{self.session_id}")
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Prevent propagation to root logger to avoid conflicts
        self.logger.propagate = False
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
    def start_session(self, total_urls: int):
        """Start a new download session"""
        with self._lock:
            self.stats.total_urls = total_urls
            self.stats.start_time = datetime.now()
            self.logger.info(f"Starting download session with {total_urls} URLs")
            self._save_progress()
            
    def add_urls(self, urls: List[str], titles: List[str]):
        """Add URLs to be tracked"""
        with self._lock:
            for url, title in zip(urls, titles):
                if url not in self.download_items:
                    self.download_items[url] = DownloadItem(url=url, title=title)
                    self.stats.urls_collected += 1
                    
            self.logger.info(f"Added {len(urls)} URLs to tracking. Total: {self.stats.urls_collected}")
            self._save_progress()
    
    def update_title_fetch_progress(self, current: int, total: int):
        """Update progress during title fetching"""
        with self._lock:
            self.logger.info(f"Title fetching progress: {current}/{total}")
            self._save_progress()
            
    def start_download(self, url: str):
        """Mark a download as started"""
        with self._lock:
            if url in self.download_items:
                self.download_items[url].status = "downloading"
                self.download_items[url].start_time = datetime.now()
                self.stats.downloads_started += 1
                self.logger.info(f"Started downloading: {self.download_items[url].title}")
                self._save_progress()
                
    def complete_download(self, url: str, file_path: str, file_size: int = 0):
        """Mark a download as completed"""
        with self._lock:
            if url in self.download_items:
                item = self.download_items[url]
                item.status = "completed"
                item.file_path = file_path
                item.file_size = file_size
                item.end_time = datetime.now()
                
                self.stats.downloads_completed += 1
                self.stats.total_size_downloaded += file_size
                
                self.successful_downloads.append(item)
                
                duration = (item.end_time - item.start_time).total_seconds() if item.start_time else 0
                self.logger.info(f"Completed: {item.title} ({file_size} bytes, {duration:.1f}s)")
                self._save_progress()
                
    def fail_download(self, url: str, error_message: str, retry: bool = False):
        """Mark a download as failed"""
        with self._lock:
            if url in self.download_items:
                item = self.download_items[url]
                item.status = "failed" if not retry else "pending"
                item.error_message = error_message
                item.end_time = datetime.now()
                
                if retry:
                    item.retry_count += 1
                    self.logger.warning(f"Retrying download: {item.title} (attempt {item.retry_count})")
                else:
                    self.stats.downloads_failed += 1
                    self.failed_downloads.append(item)
                    self.logger.error(f"Failed: {item.title} - {error_message}")
                    
                self._save_progress()
                
    def skip_download(self, url: str, reason: str = "File already exists"):
        """Mark a download as skipped"""
        with self._lock:
            if url in self.download_items:
                item = self.download_items[url]
                item.status = "skipped"
                item.error_message = reason
                item.end_time = datetime.now()
                
                self.stats.files_skipped += 1
                self.skipped_downloads.append(item)
                self.logger.info(f"Skipped: {item.title} - {reason}")
                self._save_progress()
                
    def not_found(self, url: str, reason: str = "File not found"):
        """Mark a download as not found"""
        with self._lock:
            if url in self.download_items:
                item = self.download_items[url]
                item.status = "not_found"
                item.error_message = reason
                item.end_time = datetime.now()
                
                self.stats.files_not_found += 1
                self.not_found_downloads.append(item)
                self.logger.warning(f"Not found: {item.title} - {reason}")
                self._save_progress()
                
    def end_session(self):
        """End the download session"""
        with self._lock:
            self.stats.end_time = datetime.now()
            duration = (self.stats.end_time - self.stats.start_time).total_seconds() if self.stats.start_time else 0
            
            self.logger.info("=" * 60)
            self.logger.info("DOWNLOAD SESSION COMPLETED")
            self.logger.info("=" * 60)
            self.logger.info(f"Total URLs: {self.stats.total_urls}")
            self.logger.info(f"URLs Collected: {self.stats.urls_collected}")
            self.logger.info(f"Downloads Started: {self.stats.downloads_started}")
            self.logger.info(f"Downloads Completed: {self.stats.downloads_completed}")
            self.logger.info(f"Downloads Failed: {self.stats.downloads_failed}")
            self.logger.info(f"Files Skipped: {self.stats.files_skipped}")
            self.logger.info(f"Files Not Found: {self.stats.files_not_found}")
            self.logger.info(f"Success Rate: {self.stats.success_rate:.1f}%")
            self.logger.info(f"Total Size Downloaded: {self.stats.total_size_downloaded:,} bytes")
            self.logger.info(f"Total Duration: {duration:.1f} seconds")
            self.logger.info("=" * 60)
            
            # Log detailed results
            self._log_detailed_results()
            self._save_progress()
            
    def _log_detailed_results(self):
        """Log detailed results by category"""
        
        if self.successful_downloads:
            self.logger.info("\nSUCCESSFUL DOWNLOADS:")
            for item in self.successful_downloads:
                self.logger.info(f"  ✓ {item.title} -> {item.file_path}")
                
        if self.failed_downloads:
            self.logger.info("\nFAILED DOWNLOADS:")
            for item in self.failed_downloads:
                self.logger.info(f"  ✗ {item.title} - {item.error_message}")
                
        if self.skipped_downloads:
            self.logger.info("\nSKIPPED DOWNLOADS:")
            for item in self.skipped_downloads:
                self.logger.info(f"  ⏭ {item.title} - {item.error_message}")
                
        if self.not_found_downloads:
            self.logger.info("\nNOT FOUND DOWNLOADS:")
            for item in self.not_found_downloads:
                self.logger.info(f"  ❓ {item.title} - {item.error_message}")
                
    def _save_progress(self):
        """Save progress to JSON file for web interface"""
        progress_data = {
            "session_id": self.session_id,
            "stats": asdict(self.stats),
            "download_items": {url: asdict(item) for url, item in self.download_items.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")
            
    def get_progress(self) -> Dict:
        """Get current progress data"""
        with self._lock:
            return {
                "session_id": self.session_id,
                "stats": asdict(self.stats),
                "download_items": {url: asdict(item) for url, item in self.download_items.items()},
                "timestamp": datetime.now().isoformat()
            }
            
    def get_summary(self) -> Dict:
        """Get a summary of the download session"""
        with self._lock:
            # Calculate title fetching progress
            title_fetch_progress = 0
            if self.stats.total_urls > 0:
                title_fetch_progress = (self.stats.urls_collected / self.stats.total_urls) * 100
            
            return {
                "total_urls": self.stats.total_urls,
                "urls_collected": self.stats.urls_collected,
                "downloads_started": self.stats.downloads_started,
                "downloads_completed": self.stats.downloads_completed,
                "downloads_failed": self.stats.downloads_failed,
                "files_skipped": self.stats.files_skipped,
                "files_not_found": self.stats.files_not_found,
                "success_rate": self.stats.success_rate,
                "progress_percentage": self.stats.progress_percentage,
                "title_fetch_progress": round(title_fetch_progress, 2),
                "remaining_downloads": self.stats.remaining_downloads,
                "total_size_downloaded": self.stats.total_size_downloaded,
                "is_complete": self.stats.remaining_downloads == 0
            }

# Global tracker instances (one per session)
_trackers: Dict[str, ProgressTracker] = {}

def get_tracker(session_id: str) -> ProgressTracker:
    """Get or create a progress tracker for a session"""
    if session_id not in _trackers:
        _trackers[session_id] = ProgressTracker(session_id)
    return _trackers[session_id]

def cleanup_tracker(session_id: str):
    """Clean up a tracker after session completion"""
    if session_id in _trackers:
        _trackers[session_id].end_session()
        del _trackers[session_id]
