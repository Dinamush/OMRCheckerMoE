# Changelog

All notable changes to **SHUCK3R** (formerly HamsterScraper) are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  
Versioning follows [Semantic Versioning](https://semver.org/) — see [VERSIONING.md](VERSIONING.md).

**Repository:** 57 commits from 2024-11-03 through 2026-05-17 (inception → current).

---

## [Unreleased]

### Planned / known gaps

- Pixiv queue resume API
- Library skip on home form for Pixiv downloads
- Optional `pornhub_username` in Settings UI

---

## [1.2.0] - 2026-05-17

Pixiv mature feature set: animated ugoira, manga archives, translated filenames, and workflow hardening.

### Added

- **Ugoira** (animated illusts): Ajax `ugoira_meta`, ZIP download, GIF via Pillow (ffmpeg optional); settings `pixiv_ugoira_format` (`gif` / `zip` / `both`)
- **Manga** (`illustType` 1): all pages bundled into `{id}_{title}.zip`
- **Title translation** for filenames: `pixiv_titles.py`, Google Translate (default) or DeepL; cache at `pixiv_title_cache.json`
- Single-artwork mode: artwork URLs and numeric illust ID with checkbox
- Pixiv retry failed downloads via `POST /api/session/{id}/retry-failed?site=pixiv`
- Tests: `test_pixiv_ugoira`, `test_pixiv_manga`, `test_pixiv_titles`, `test_pixiv_parse` (74 tests total)

### Changed

- Atomic downloads (`.part` → rename), ZIP integrity checks, thread-local HTTP sessions for parallel Pixiv workers
- Settings: DeepL key preserved when password field left blank; ugoira copy reflects Pillow-first GIF
- Home form: headless checkbox defaults from `headless_scraping` setting

### Fixed

- Translation cache: no longer caches API failures; file lock on cache R/W; `CACHE_VERSION` invalidation
- Manga incomplete ZIPs when a page failed; corrupt ZIP skip behavior
- Pixiv retry routing (was incorrectly starting xHamster workflow)
- `establish_www_session`: target-user bookmark probe no longer falls through to `user/self`
- Unparseable Pixiv input raises clear error instead of silently using own bookmarks

**Commits:** `6ad5340` … `0a081ee`

---

## [1.1.1] - 2026-05-17

### Added

- Site-specific download subfolders: `downloads/ph`, `downloads/xh`, `downloads/pixiv`
- Session history UI improvements for download runs

### Changed

- PornHub workflow and cookie management refinements
- Library indexing enhancements for existing-library matching

**Commits:** `1f4d048`, `e783495`, `5163066`, `6ad5340` (partial)

---

## [1.1.0] - 2026-05-17

### Added

- **Pixiv** bookmark downloader: Chrome login, Ajax API, public/hidden bookmark modes (`rest=show` / `hide`)
- `pixiv_ph.py` integration in FastAPI `main.py` and PyInstaller specs

**Commits:** `855cdbe`

---

## [1.0.0-beta.3] - 2026-05-17

Browser automation, settings page, and operational features.

### Added

- **Settings** page: video quality, delays, proxy, persistent cookies, queue snapshots, watcher interval
- `chrome_login_confirm` (replaces `selenium_login_wait`) with versioned wait protocol
- Headless scraping default, per-site Chrome profiles, FlareSolverr option, challenge detection
- Favorites watcher with configurable reminders
- Proxy support for downloads and cookie checks

**Commits:** `aa274e9`, `7107964`, `0e64a5a`

---

## [1.0.0-beta.2] - 2026-05-12

Session lifecycle and library tooling.

### Added

- **Session history** (`session_history.jsonl`) and History UI
- **Integrity scanner** page
- **Library browser** for downloaded media
- Download **cancellation** API and Stop button
- Nested path serving for library videos; path traversal guards

**Commits:** `0a83cf0`, `adc8192`, `a15f09a`

---

## [1.0.0-beta.1] - 2026-05-11

PornHub reliability and “Corn Protocol” UI polish.

### Added

- Selenium login confirmation flow for PornHub
- Corn Protocol design system (typography, tokens, layout)
- PR #1 audit fixes: typed duplicate payloads, `ProgressTracker` status enum, PEP 440 requirement parsing

### Changed

- PornHub media extraction with optional browser driver
- Duplicate finder: system path protection, custom delete modal

**Commits:** `0d6a302` … `44ab919` (merge PR #2)

---

## [0.9.0] - 2026-05-10

SHUCK3R rebrand and desktop-grade UX.

### Added

- Rebrand **HamsterScraper → SHUCK3R** (executables, templates, docs)
- **Headless** Selenium after embedded login (xHamster)
- Embedded **pywebview** login bridge and menu handoff
- **Library comparison**: skip downloads already in an existing folder (PH/xH)
- User-data directory under `%LOCALAPPDATA%\HamsterScraper` (`paths.py`)
- Regenerated app icon from favicon

**Commits:** `2959b50`, `464240d`, `cb2e569`, `af678a3`, `646eba6`, `0590449`

---

## [0.8.0] - 2026-05-10

Duplicate file checker and UI foundation.

### Added

- **Duplicate file checker** UI and API (`duplicate_finder.py`)
- Fuzzy grouping by name/size; episode/page index detection; release-metadata stripping
- Jinja **base layout**; Plus Jakarta Sans / JetBrains Mono; semantic `<main>` structure
- Video preview seek helper for duplicate thumbnails

**Commits:** `580b2ff`, `226db82`, `cc91ee6`, `1250886`, `c8d73a1`, `39a7269`

---

## [0.7.0] - 2026-03-15

PornHub scraper modernization.

### Changed

- Moved legacy scripts to `archive/`; **FastAPI `main.py`** as primary entry
- PornHub favorites: HTML parsing + Selenium scroll / “Load More”
- Video download via **requests** + page HTML media definitions (not only yt-dlp page fetch)
- yt-dlp version pin; non-HLS format preference

**Commits:** `e0eff95`, `cb3e57c`, `36c50e4`, `8b7b928`

---

## [0.6.0] - 2025-09-21

Multi-site downloader core.

### Added

- **xHamster** favorites: direct URL extraction, pagination, batch downloads
- **Progress tracker** with live session JSON and log directory
- `python -m yt_dlp` invocation; **webdriver-manager** for ChromeDriver
- Enhanced download UI and session metrics

**Commits:** `210559a`, `160fee0`, `d4be103`

---

## [0.5.0] - 2025-02-13

### Added

- Project restructure (“initial add”); renamed project identity

**Commits:** `456ccf8`, `efc47b2`, `9dc6740`

---

## [0.4.0] - 2024-11-12

### Added

- Working end-to-end prototype (“it works now”)

**Commits:** `2159dae`, `309444e`

---

## [0.1.0] - 2024-11-03

### Added

- Initial repository and requirements scaffold

**Commits:** `341d90d`, `5225846`

---

[1.2.0]: https://github.com/Dinamush/hamster_scraper/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/Dinamush/hamster_scraper/compare/v1.0.0-beta.3...v1.1.0
[1.0.0-beta.3]: https://github.com/Dinamush/hamster_scraper/compare/v1.0.0-beta.2...v1.0.0-beta.3
