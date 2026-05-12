# Graph Report - .  (2026-05-12)

## Corpus Check
- 41 files · ~95,677 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 669 nodes · 1023 edges · 40 communities (35 shown, 5 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 86 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]

## God Nodes (most connected - your core abstractions)
1. `archive/ph.py` - 30 edges
2. `xhamster_workflow()` - 20 edges
3. `ProgressTracker` - 19 edges
4. `normalize_name()` - 17 edges
5. `cluster_duplicates()` - 14 edges
6. `pornhub_workflow()` - 13 edges
7. `archive/xh.py` - 13 edges
8. `get_tracker()` - 12 edges
9. `main()` - 11 edges
10. `main_workflow()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `check_and_update_requirements()`  [INFERRED]
  archive/ph.py → check_requirements.py
- `api_duplicates_scan()` --calls--> `resolve_scan_directory()`  [INFERRED]
  main.py → duplicate_finder.py
- `api_duplicates_preview_file()` --calls--> `resolve_deletable_file()`  [INFERRED]
  main.py → duplicate_finder.py
- `normalize_compare_key_from_filename()` --calls--> `normalize_name()`  [INFERRED]
  existing_library.py → duplicate_finder.py
- `normalize_compare_key_from_title()` --calls--> `normalize_name()`  [INFERRED]
  existing_library.py → duplicate_finder.py

## Communities (40 total, 5 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.0
Nodes (48): build_groups_payload(), cluster_duplicates(), _collapse_ws(), DuplicateGroup, _file_best_peer_score(), FileMatchRow, FileRecord, _group_max_pairwise_score() (+40 more)

### Community 1 - "Community 1"
Cohesion: 0.0
Nodes (25): Enum, DownloadItem, DownloadStats, DownloadStatus, ProgressTracker, Setup detailed logging for this session, Start a new download session, Add URLs to be tracked (+17 more)

### Community 2 - "Community 2"
Cohesion: 0.0
Nodes (47): Local Downloaded Media Asset, pornhub_favourites_page_and_pagination.md, pornhub_video_handling.md, pornhub_website_video_storage.md, Favorites Scraper Feature, ph-favourites-page.html, ph-video.html, download_videos_parallel() in main.py (+39 more)

### Community 3 - "Community 3"
Cohesion: 0.0
Nodes (46): API POST /api/duplicates/delete, API GET /api/duplicates/preview-file, API POST /api/duplicates/scan, API POST /api/embedded-login/confirm, API GET /download/log/{session_id}, API GET /download/log/{session_id}/tail, API GET /progress/{session_id}/json, API GET /progress/{session_id}/summary (+38 more)

### Community 4 - "Community 4"
Cohesion: 0.0
Nodes (34): Local Non-System Files Asset, Session Logs and Progress JSON Asset, Attacker: Local Process, Attacker: Malicious Local Webpage, ProgressTracker class, Duplicate Finder Component, Local FastAPI Server, Progress/Log Subsystem (+26 more)

### Community 5 - "Community 5"
Cohesion: 0.0
Nodes (33): create_authenticated_session(), download_bookmarks(), download_image(), get_bookmarks(), get_user_id(), login(), main(), archive/pixiv.py (+25 more)

### Community 6 - "Community 6"
Cohesion: 0.0
Nodes (25): begin_embedded_login(), _cleanup_login_ui(), configure(), embedded_login_env_enabled(), finish_embedded_login(), _finish_embedded_login_impl(), has_pending_login(), is_active() (+17 more)

### Community 7 - "Community 7"
Cohesion: 0.0
Nodes (25): collect_favorites_urls_requests(), collect_favorites_urls_with_driver(), _download_from_media_url(), download_pornhub_video_page(), download_pornhub_videos_parallel(), _extract_flashvars_block(), extract_media_from_page(), favorites_url_with_page() (+17 more)

### Community 8 - "Community 8"
Cohesion: 0.0
Nodes (25): download_videos_parallel(), extract_video_urls_selenium(), login(), main(), parse_video_links_pornhub(), archive/ph.py, Click the 'Load More' button (#moreDataBtn) if present.     Returns True if cli, Load the favorites page, scroll to trigger lazy load, click 'Load More'     unt (+17 more)

### Community 9 - "Community 9"
Cohesion: 0.0
Nodes (25): create_authenticated_session(), download_video(), download_videos_concurrently(), ensure_download_dir(), fetch_html(), find_next_page(), get_video_title(), main() (+17 more)

### Community 10 - "Community 10"
Cohesion: 0.0
Nodes (15): advance(), _chain(), complete_success(), dispose_session(), get_timeline(), User-facing workflow timeline (linked-list of steps) for download sessions.  E, One node in the workflow linked list., Call from POST /download before returning redirect (so first poll has steps). (+7 more)

### Community 11 - "Community 11"
Cohesion: 0.0
Nodes (21): BaseModel, api_embedded_login_confirm(), api_selenium_login_confirm(), create_authenticated_session(), download_session_progress(), DuplicateDeleteBody, duplicates_page(), DuplicateScanBody (+13 more)

### Community 12 - "Community 12"
Cohesion: 0.0
Nodes (18): build_library_normalized_names(), _collapse_for_compare(), matches_existing_library(), normalize_compare_key_from_filename(), normalize_compare_key_from_title(), Compare candidate video titles against filenames already on disk (library folder, True if this download title likely corresponds to a file already in the library., Underscores in filenames and spaces in titles converge for comparison. (+10 more)

### Community 13 - "Community 13"
Cohesion: 0.0
Nodes (20): Attacker: Network-Adjacent (if misbound), HAMSTER_EXTERNAL_BROWSER env var, HAMSTER_HOST env var, HAMSTER_PORT env var, HAMSTER_USER_DATA env var, main.py, requirements.txt, download_video() in archive/ph.py (+12 more)

### Community 14 - "Community 14"
Cohesion: 0.0
Nodes (19): create_authenticated_session(), ensure_download_dir(), fetch_html(), get_video_title(), main_workflow(), parse_video_links(), Parse the HTML content to extract video page links based on the site structure., Create a requests.Session object with authenticated cookies from Selenium. (+11 more)

### Community 15 - "Community 15"
Cohesion: 0.0
Nodes (19): SHUCK3R Application, Google Chrome Browser, Edge WebView2, Desktop Launcher (Uvicorn + pywebview), Download Workflows (PornHub + xHamster), ChromeDriver, Python 3.10+, HAMSTER_SHOW_SELENIUM env var (+11 more)

### Community 16 - "Community 16"
Cohesion: 0.0
Nodes (15): NamedTuple, check_and_update_requirements(), _get_installed_version(), _parse_requirements_line(), _parse_version(), ParsedRequirement, _pip_install_upgrade(), Run pip install -U for each package. Return True if all succeeded. (+7 more)

### Community 17 - "Community 17"
Cohesion: 0.0
Nodes (14): _fatal_exit(), _http_ok(), _is_frozen(), No console in frozen GUI builds — log + optional Windows message box., run_desktop(), start_uvicorn_background(), wait_for_server_ready(), ensure_runtime_cwd() (+6 more)

### Community 18 - "Community 18"
Cohesion: 0.0
Nodes (15): _desktop_session_progress_url(), ensure_download_dir(), _pornhub_tracker_titles(), pornhub_workflow(), Ensure that the download directory exists., Set up Chrome WebDriver with webdriver-manager., Save cookies in Netscape format for yt-dlp., Short labels for progress UI (viewkey when present). (+7 more)

### Community 19 - "Community 19"
Cohesion: 0.0
Nodes (14): api_get_progress(), download_video_direct(), download_video_ytdlp(), download_videos_parallel(), get_progress(), get_progress_summary(), JSON snapshot of progress (machine-readable). Prefer /api/progress for new integ, Same payload as /progress/{session_id}; use this path in docs and UI so users ar (+6 more)

### Community 20 - "Community 20"
Cohesion: 0.0
Nodes (12): load_netscape_cookies_into_driver(), parse_video_links_xhamster(), After site login inside pywebview, a visible Chrome window would cover SHUCK3R a, Prompt the user to log in manually and wait for confirmation (terminal or progre, Apply a Netscape cookie jar to Selenium after opening landing_url (same registra, Click the next-page button if one is visible and enabled.      Tries a priorit, Parse video links for xHamster., xHamster workflow using Chrome and yt-dlp with two-phase approach. (+4 more)

### Community 21 - "Community 21"
Cohesion: 0.0
Nodes (11): Session Cookies Asset, Embedded Login Bridge, HAMSTER_EMBEDDED_LOGIN env var, webview_login_bridge.py, finish_embedded_login(), save_cookies_netscape(), simplecookie_list_to_netscape_file(), Mitigation: Restrictive Cookie File Permissions (+3 more)

### Community 22 - "Community 22"
Cohesion: 0.0
Nodes (10): iter_files(), List regular files under root., api_duplicates_preview_file(), api_duplicates_scan(), favicon_ico(), get_downloaded_file(), Browsers request /favicon.ico by default; serve SVG so logs stay clean., Serve downloaded video files. (+2 more)

### Community 23 - "Community 23"
Cohesion: 0.0
Nodes (10): _download_from_media_url(), download_video(), _get_video_page_html(), Load Netscape cookie file into a requests session., Replace invalid path chars and limit length., Download from a direct stream URL (get_media or m3u8) using cookies., Get HTML of a PornHub video page (the same structure as website-code/ph-video.ht, Opens the video page (with Selenium if driver given, else requests), finds the (+2 more)

### Community 24 - "Community 24"
Cohesion: 0.0
Nodes (6): get_downloaded_file(), get_log(), handle_form(), Handle the form submission and start the downloading process., Allow users to download the log file., Serve the downloaded video files.

### Community 25 - "Community 25"
Cohesion: 0.0
Nodes (8): PH DOM: #globalCookieBanner / #cookieBanner, PH DOM: body.logged-in class, PH DOM: #js-networkBar, PH JS Var: WIDGET_PLAYLIST_HEADER, PH URL Pattern: /users/{username}/videos/favorites, PH URL Pattern: /view_video.php?viewkey={key}, PH Scraped: Favourites Page (ph-favourites-page.html), PH Scraped: Video Page (ph-video.html)

### Community 26 - "Community 26"
Cohesion: 0.0
Nodes (5): confirm_chrome_login_done(), Wait for the user to finish logging in via Selenium Chrome when stdin is not int, Block until the user confirms login.      - Interactive terminal: same as ``in, Signal the waiting workflow for ``session_id``. Returns False if nothing was wai, wait_for_user_after_chrome_login()

### Community 27 - "Community 27"
Cohesion: 0.0
Nodes (5): main(), render(), _scale(), _X(), _Y()

### Community 28 - "Community 28"
Cohesion: 0.0
Nodes (6): _extract_flashvars_block(), _extract_media_from_page(), _find_matching_bracket(), Find the position of the matching close_c, skipping strings. start_pos is the in, Find the flashvars_<id> = {...}; block as in ph-video.html (the specific actual, Extract the best stream URL and video title from a PornHub video page HTML.

### Community 29 - "Community 29"
Cohesion: 0.0
Nodes (4): extract_direct_video_url(), extract_video_info_sequential(), Extract direct video URL and title from a video page using browser session., Extract video info sequentially to avoid driver conflicts.

### Community 30 - "Community 30"
Cohesion: 0.0
Nodes (4): fetch_html_selenium(), Fetch HTML via Selenium with a full scroll to trigger lazy-loaded content., Scroll to the bottom of the page to load all content., scroll_to_bottom()

### Community 31 - "Community 31"
Cohesion: 0.0
Nodes (4): get_log_tail(), Return the last `lines` lines without loading huge files whole., Last lines of the session log for the live dashboard (plain text)., _read_log_tail_text()

### Community 32 - "Community 32"
Cohesion: 0.0
Nodes (4): build_windows.ps1, Portable EXE Feature, requirements-build.txt, pyinstaller

## Knowledge Gaps
- **178 isolated node(s):** `A parsed requirements.txt line: package name plus optional version specifier.`, `Parse a line like 'yt-dlp>=2025.1.1' or 'fastapi' -> ParsedRequirement, or None`, `Turn '2025.1.1' or '1.2' into comparable tuple. Non-numeric tails ignored.`, `Check if installed version satisfies the spec (>=, ==, etc.). Simple implementat`, `Return installed version string for package, or None if not installed.` (+173 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.