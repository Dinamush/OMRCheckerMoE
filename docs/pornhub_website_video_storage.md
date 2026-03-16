# PornHub Website: How Videos Are Stored and Handled

This document is derived from the **live video page** structure captured in `website-code/ph-video.html`. It describes how PornHub’s video watch page stores and delivers video data **on the website**, and how earlier versions of the page handled playback.

---

## 1. Page and Container Structure

### 1.1 Main wrapper

- **Container:** `<div id="main-container" class="clearfix vpContainer noAds videoVoted">`
- **Origin URL (script):** `var originUrl = '/view_video.php?viewkey=ph5fe50e391efeb';`
- The page is the canonical **view_video** page; the **viewkey** (e.g. `ph5fe50e391efeb`) identifies the video.

### 1.2 Global JavaScript objects (per video)

| Variable | Purpose |
|----------|---------|
| **`VIDEO_SHOW`** | Page-level metadata: `vkey`, `video_id`, `videoId` (player div id), `videoTitle`, playlists, user/tracking IDs, etc. **Does not** contain direct video stream URLs. |
| **`flashvars_<video_id>`** | e.g. `flashvars_379015542`. Main payload for the player: stream URLs, quality list, thumbs, ads, next-video, etc. |
| **`playerObjList.playerDiv_<video_id>`** | Player registration: `flashvars` (e.g. `embedId`), and legacy **`embedSWF`** (Flash fallback config). |

Video **stream URLs and formats** are provided via **`flashvars_<id>.mediaDefinitions`** (see below).

---

## 2. How Videos Are Stored and Exposed (Current Page)

### 2.1 `mediaDefinitions` (primary source of stream URLs)

Inside **`flashvars_<video_id>`**, the **`mediaDefinitions`** array lists all available delivery options. Each entry has:

| Field | Type | Description |
|-------|------|-------------|
| **`format`** | string | `"hls"` or `"mp4"` |
| **`videoUrl`** | string | Full URL to the stream (see below) |
| **`quality`** | string or array | e.g. `"1080"`, `"720"`, `"480"`, `"240"`, or `[]` for generic MP4 |
| **`defaultQuality`** | boolean | Which quality is pre-selected (e.g. 720p) |
| **`group`** | number | Grouping for quality switching (e.g. `1`) |
| **`segmentFormats`** | object | For HLS: `{"audio":"ts_aac","video":"mpeg2_ts"}` |
| **`remote`** | boolean | If true, URL is a **redirect/proxy** (e.g. `get_media`) rather than direct CDN |

**Example HLS entries (from ph-video.html):**

- **1080p:** `videoUrl`: `https://ev-h.phncdn.com/hls/videos/202012/24/379015542/1080P_8000K_379015542.mp4/master.m3u8?validfrom=...&validto=...&ipa=1&hdl=-1&hash=...`
- **720p (default):** same path pattern, `720P_4000K_...`, `defaultQuality: true`
- **480p, 240p:** same CDN path, different resolution/bitrate in path.

**Example MP4 (remote) entry:**

- **`format`:** `"mp4"`
- **`videoUrl`:** `https://www.pornhub.com/video/get_media?s=<encrypted>&v=ph5fe50e391efeb&e=0&t=p`
- **`quality`:** `[]`
- **`remote`:** `true`  
So the **actual file** is not stored in the page; the page stores a **signed/encrypted `get_media` URL** that the server resolves to a CDN URL (often time-limited).

### 2.2 CDN and URL patterns

- **HLS (current primary):** Host **`ev-h.phncdn.com`** (and similar). Path pattern:  
  `hls/videos/<YYYYMM>/<DD>/<video_id>/<QUALITY>_<BITRATE>K_<video_id>.mp4/master.m3u8`  
  Query params include **`validfrom`**, **`validto`**, **`ipa`**, **`hdl`**, **`hash`** (time-limited and integrity-checked).
- **Thumbnails/previews:** **`ei.phncdn.com`** (images), **`kw.phncdn.com`** (e.g. preview clips). Paths like `videos/<YYYYMM>/<DD>/<video_id>/original/...` or `timeline/160x90/...`.
- **MP4 download/proxy:** **`www.pornhub.com/video/get_media`** with encrypted `s`, viewkey `v`, and flags `e`, `t`. No direct CDN URL in HTML.

### 2.3 Other flashvars used for playback

| Key | Role |
|-----|------|
| **`mediaPriority`** | `"hls"` – player prefers HLS over MP4 when both exist. |
| **`link_url`** | Canonical watch URL (e.g. `https://www.pornhub.com/view_video.php?viewkey=ph5fe50e391efeb`). |
| **`video_duration`** | Duration in seconds. |
| **`video_title`** | Title string. |
| **`image_url`** | Poster/thumbnail URL. |
| **`mp4_seek`** | Seek parameter for MP4 (e.g. `"ms"`). |
| **`thumbs`** | Timeline thumbnails: `urlPattern`, `spritePatterns`, `samplingFrequency`, `thumbWidth`, `thumbHeight`, `cdnType`, `isVault`. |
| **`hotspots`** | Array of timestamps for “hotspot” segments on the seek bar. |
| **`nextVideo`** | Next-video info: `thumb`, `title`, `nextUrl`, `vkey`, etc. |

So **on the website**, videos are “stored” as:

1. **HLS:** Time-limited **m3u8 URLs** pointing to CDN segments (TS + AAC).
2. **MP4:** Time-limited **get_media** URLs that the server maps to CDN or stream.

---

## 3. How the Player Uses This (Current Page)

### 3.1 Player config (getPlayerConfig)

The player is configured from **flashvars** (see inline script and player JS in ph-video.html). Relevant mapping:

- **`mainRoll.mediaDefinition`** ← **`flashvars.mediaDefinitions`**
- **`mainRoll.mediaPriority`** ← **`flashvars.mediaPriority`** (e.g. `"hls"`)
- **`mainRoll.poster`** ← **`flashvars.image_url`**
- **`mainRoll.videoUrl`** ← **`flashvars.link_url`** (page URL, not stream URL)
- **`mainRoll.duration`** / **`mainRoll.title`** ← **`flashvars.video_duration`** / **`flashvars.video_title`**
- **`mainRoll.thumbs`** ← **`flashvars.thumbs`** (timeline sprites/patterns)
- **Seek params:** **`seekParams: { mp4: flashvars.mp4_seek, flv: 'fv' }`** (MP4 and legacy FLV).

So the **single source of truth** for stream URLs on the page is **`flashvars.mediaDefinitions`**; the rest is metadata and UI.

### 3.2 Playback method (from debug UI in ph-video.html)

- **Method:** “Adaptive (TS)”
- **Quality:** “720p (Forced)” (or selected quality)
- **Adaptive library:** “Hlsjs (^1.6.5)”
- **Audio format:** TS_AAC
- **Video format:** MPEG2_TS  
So the **current** watch page uses **HLS (Hls.js)** with TS segments; the **video element** ends up with a blob or HLS-driven stream, not a raw MP4 URL in the DOM.

### 3.3 HTML5 video element

- The visible player is an HTML5 **`<video>`** inside a wrapper (e.g. `id="playerDiv_379015542"`).
- **`<source src="">`** is initially empty; the real source is set by the player JS from **mediaDefinitions** (HLS or, when used, MP4/get_media).

---

## 4. Previous Versions of the Page and How They Handled Videos

The following is inferred from **the same ph-video.html** file (legacy paths and naming), not from separate old snapshots.

### 4.1 Flash era (legacy)

- **Naming:** The object holding stream and UI config is still called **“flashvars”** and is keyed by video id (e.g. `flashvars_379015542`). This naming comes from the old **Flash** player, which received these variables when embedded.
- **Embed config still present:**  
  **`playerObjList.playerDiv_379015542.embedSWF`** =  
  `{ "url": "https://ei.phncdn.com/www-static/flash/", "element": "playerDiv_379015542", "width": "100%", "height": "100%", "version": "9.0.0" }`  
  So the page still **references** a Flash SWF base URL and element; the actual playback is now HTML5/HLS.
- **Seek params:** **`seekParams: { mp4: flashvars.mp4_seek, flv: 'fv' }`** – the **`flv: 'fv'`** indicates that **FLV** was previously supported (seek parameter `fv` for FLV).
- **Conclusion (previous handling):** Earlier, the same **flashvars** (or a subset) were passed to a **Flash** player. Streams were likely **FLV** and/or **MP4**; the **mediaDefinitions** array and **HLS** are a later addition. The **flashvars** name and **embedSWF** block are leftovers from that era.

### 4.2 Legacy player version and events

- In the player code: **`parseFloat(playerVersion) < 6.1 ? Self.getLegacyEventsConfig(events) : events`**  
  So **player version &lt; 6.1** used a **different event config** (`getLegacyEventsConfig`). Older page/player versions therefore had a different event model (and possibly different stream handling) than the current one.
- **Conclusion:** “Previous versions” of the **player** (pre–6.1) are still reflected in code paths; behavior would have been Flash/FLV or early HTML5, before the current HLS-first setup.

### 4.3 Evolution summary (inferred)

| Phase | How videos were handled (inferred) |
|-------|-------------------------------------|
| **Flash** | SWF from `ei.phncdn.com/www-static/flash/`, config via **flashvars**. Streams: **FLV** (and possibly MP4), seek param **`flv: 'fv'`**. |
| **Transition** | Same **flashvars** reused; **embedSWF** kept for fallback or legacy; **mediaDefinitions** and **mediaPriority** added. |
| **Current** | **HTML5** + **Hls.js**; primary streams in **mediaDefinitions** as **HLS** (m3u8 on **ev-h.phncdn.com**); optional **MP4** via **get_media** proxy. **FLV** no longer used in config; **mp4_seek** still present. |

---

## 5. Implications for Downloaders (yt-dlp, ph.py, main.py)

- **Page URL** is always **view_video.php?viewkey=...**. Favorites and playlists only give **lists of these viewkeys/links**; they do **not** expose **mediaDefinitions** or raw CDN URLs.
- **Stream URLs** are **inside the watch page** in **`flashvars_<id>.mediaDefinitions`**: **HLS** (m3u8) and/or **MP4** (get_media). They are **time-limited** (`validfrom`/`validto`/hash or encrypted `s`).
- **yt-dlp** (and similar tools) must either:
  - **Parse the watch page** (or its JSON/API) to get **mediaDefinitions**, then use **m3u8** or **get_media** with the same cookies/session, or
  - Rely on **extractors** that do this and handle **HLS** (and optional MP4) and **token expiry** (e.g. 412 Gone, 404 on segments).
- **Fragment 404s** when downloading are usually **expired or invalid HLS/segment URLs** (or geo/access), not “wrong page”; the **website** itself serves those URLs with short-lived validity.

---

## 6. Reference: mediaDefinitions shape (from ph-video.html)

```text
mediaDefinitions: [
  { group: 1, height: 0, width: 0, defaultQuality: false, format: "hls",
    videoUrl: "https://ev-h.phncdn.com/hls/videos/.../1080P_8000K_.../master.m3u8?validfrom=...&validto=...",
    quality: "1080", segmentFormats: { audio: "ts_aac", video: "mpeg2_ts" } },
  { ..., format: "hls", quality: "240", ... },
  { ..., format: "hls", quality: "480", ... },
  { ..., format: "hls", quality: "720", defaultQuality: true, ... },
  { ..., format: "mp4", videoUrl: "https://www.pornhub.com/video/get_media?s=...&v=ph5fe50e391efeb&e=0&t=p",
    quality: [], remote: true }
]
```

**Source file:** `website-code/ph-video.html` (variable `flashvars_379015542`, script block around the main player container).  
**See also:** [pornhub_video_handling.md](pornhub_video_handling.md) for how this repo’s code (main.py, ph.py) uses PornHub URLs and yt-dlp.
