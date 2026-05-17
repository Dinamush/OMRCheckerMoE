# PornHub Favourites Page and Pagination

This document describes the **favourites page** structure and **pagination** behaviour (on the website and in our code), so we can collect **every** favourite video across all pages. It is based on the saved HTML in `website-code/ph-favourites-page.html` and the user-described behaviour (URL `?page=N`, “Load more”, scroll).

---

## 1. Favourites Page URLs

| URL | Meaning |
|-----|--------|
| **`https://www.pornhub.com/users/<username>/videos/favorites`** | First page (default). Same as `.../favorites?` with no query. |
| **`https://www.pornhub.com/users/<username>/videos/favorites?page=2`** | Second page. |
| **`https://www.pornhub.com/users/<username>/videos/favorites?page=N`** | Page N. |

**Sort options** (can be combined with `page`):

- **`?o=newest`** – Newest first (default when “Newest” is selected in the UI).
- **`?o=oldest`** – Oldest first.
- **`?o=mr`** – Most Recent.
- **`?o=mv`** – Most Viewed.
- **`?o=tr`** – Top Rated.
- **`?o=lg`** – Longest.

Example:  
`https://www.pornhub.com/users/burninglove999/videos/favorites?o=newest&page=2`  
= second page of favourites, newest first.

**Script variables (from ph-favourites-page.html):**

- `originPart = 'users/burninglove999/videos/favorites'`
- `originUrl = '/users/burninglove999/videos/favorites'`

---

## 2. Page Structure (from ph-favourites-page.html)

### 2.1 Container and video list

- **Profile section:** `<div class="profileClass">`, with profile header and tabs.
- **Videos container:**  
  - **`<ul id="moreData" class="videos row-3-thumbs gap-row-15">`** – list of video items. Each item is a **`<li class="pcVideoListItem js-pop videoblock videoBox ...">`** with `id="vfavouriteVideo<video_id>"`, `data-video-id`, `data-video-vkey`.
- **Video links:** Each video has one or more **`<a href="/view_video.php?viewkey=...">`** (thumb and title). Same parsing as watch page: any `a` with `href` containing `view_video` or `viewkey=` gives a video page URL.

### 2.2 “Load More” button

- **Button:**  
  `<button id="moreDataBtn" class="greyButton big light" type="button" onclick="loadMoreData('/users/burninglove999/videos/favorites/ajax?o=newest', '6', '1');">Load More</button>`
- **Behaviour (site):** When the user clicks “Load More”, the page calls **`loadMoreData(ajaxUrl, ...)`** with:
  - **AJAX URL:** `/users/<username>/videos/favorites/ajax?o=newest` (sort matches the current filter).
  - Extra arguments (e.g. `'6'`, `'1'`) are likely page/offset or batch identifiers for the next chunk.
- The server returns more HTML (or JSON that is rendered); the new items are **appended** into the **`#moreData`** list, and the page **scrolls down** so the new content is visible. The button may stay visible for the next batch or disappear when there are no more results.

So on the **website**, “Load more” = **in-page AJAX** that appends to the same list and scrolls; **URL pagination** = navigating to `.../favorites?page=2`, etc., to load a full new page.

### 2.3 Lazy loading (scroll to reveal)

- **`page_params.lazyLoad.sections.push('moreData');`** – the **`moreData`** section is registered for lazy loading.
- **Scrolling down** can trigger loading of more content (e.g. more thumbnails or more items). So in practice the site uses:
  1. **Initial load** – first batch of videos.
  2. **Scroll** – may load more (lazy load).
  3. **“Load more”** – explicit next batch via AJAX, then scroll.

To get **every** favourite in an automated way, we can either:
- Use **URL pagination** (`?page=1`, `?page=2`, …) and parse each full page HTML, or
- Simulate **scroll + “Load more”** in a browser (Selenium) until no more button or no new links.

---

## 3. How Our Code Handles Favourites (Current and Past)

### 3.1 archive/ph.py (current)

- **Scroll + “Load more” to get all videos.**  
  - **`extract_video_urls_selenium(driver, playlist_url)`** loads the favourites URL, then: **scrolls to bottom** (to trigger lazy load on `#moreData`), parses video links, then repeatedly **clicks the “Load More” button** (`#moreDataBtn`) until it is not found or a click yields no new URLs. After each click it waits, scrolls again, and re-parses. All URLs are deduplicated into a set.
  - **`scroll_to_bottom(driver)`** scrolls until `document.body.scrollHeight` stops increasing (with a short pause between scrolls).  
  - **`try_click_load_more(driver)`** finds `#moreDataBtn` and clicks it via JS; returns `True` if clicked, `False` otherwise.
- **Result:** All favourites visible after scrolling and loading more are collected (not just the first page).

### 3.2 main.py – PornHub workflow (current)

- Uses **`pornhub_ph.collect_favorites_urls_with_driver`** (Selenium scroll + `#moreDataBtn`), same algorithm as ph.py’s `extract_video_urls_selenium`. Default playlist URL: `https://www.pornhub.com/my/favorites/videos`. Embedded WebView login still uses HTTP `?page=N` pagination when no Chrome driver is available.

### 3.3 main.py – xHamster workflow (for comparison)

- **Full pagination** is implemented for xHamster:
  - **URL pagination:** Page 1 = `favorites_url`, page 2 = `favorites_url/2` or `favorites_url/2`, page N = `favorites_url/{page_number}`.
  - **`fetch_html_selenium`** is used per page; it calls **`scroll_to_bottom(driver)`** so that lazy-loaded content on **that** page is included in **`driver.page_source`**.
  - **Next page:** Either by navigating to the next URL or by **`try_click_next_button(driver)`** (e.g. “Next” link/button).
  - Stops when: redirect away from favorites, error page, no new video links, or same content as previous page (consecutive empty detection), or after a max page limit (e.g. 100).

So for **xHamster** we combine **scroll on each page** + **multi-page** to get every favourite; for **PornHub** we do **neither** in the current code.

### 3.4 How it “worked in the past” (inference)

- **ph.py:** Has only ever loaded a **single** favorites URL; there is no past version in this repo that did multi-page or “Load more” for PornHub.
- **main.py:** When PornHub used Selenium + HTML parsing (reverted), it used **`fetch_html_selenium`**, which **does** call **`scroll_to_bottom`**. So we would have gotten **everything that appears on the first page after scrolling** (lazy-loaded items included), but still **only the first page** – no `?page=2` or “Load more” clicks.
- **Conclusion:** In this codebase, we have **never** implemented full PornHub favourites pagination (multiple pages or repeated “Load more”). We only ever collected the first page (with or without scroll).

---

## 4. Recommended Way to Get Every Favourite (PornHub)

To get **every** favourite video for a user:

1. **Option A – URL pagination (simplest to implement)**  
   - Start at `.../users/<username>/videos/favorites` (or `?o=newest` if you want a fixed sort).  
   - For each **page** `N = 1, 2, 3, ...`:  
     - Open **`.../videos/favorites?page=N`** (and keep `?o=newest` if used).  
     - Optionally **scroll to bottom** once so lazy-loaded items on that page are in the DOM.  
     - Parse **`driver.page_source`** for all `a[href*="view_video"]` / `viewkey=` and add to a set.  
   - Stop when a page returns **no new** video links (or empty list).

2. **Option B – “Load more” in a single page**  
   - Open `.../videos/favorites` once.  
   - Loop: **scroll to bottom**, then **click `#moreDataBtn`** (“Load More”) if present; wait for new content; parse `#moreData` (or full `page_source`) for video links.  
   - Stop when the button is gone or no new links appear after a click.

Option A is usually easier and more robust (same URL pattern as the user’s description: `?page=2`, etc.).

---

## 5. Summary Table

| Aspect | Website behaviour | ph.py (current) | main.py PornHub (current) |
|--------|-------------------|------------------|---------------------------|
| **First page URL** | `.../favorites` or `.../favorites?` | Loads that URL only | yt-dlp on playlist URL (404-prone) |
| **Next pages URL** | `.../favorites?page=2`, etc. | Not used | Not used |
| **“Load more”** | AJAX append + scroll | Not used | Not used |
| **Scroll** | Lazy load + after “Load more” | No scroll | No Selenium for PH |
| **Videos per page** | Many (grid in `#moreData`) | First page only | — |
| **Get every favourite** | Yes (user can page / load more) | No (first page only) | No |

---

## 6. Reference: Favourites Page Snippets (from ph-favourites-page.html)

- **Video list:** `<ul id="moreData" class="videos row-3-thumbs gap-row-15">` → `<li class="pcVideoListItem ...">` → `<a href="/view_video.php?viewkey=...">`.
- **Load More button:** `<button id="moreDataBtn" ... onclick="loadMoreData('/users/burninglove999/videos/favorites/ajax?o=newest', '6', '1');">Load More</button>`.
- **Sort links:** `.../favorites` (Newest), `.../favorites?o=oldest`, `?o=mr`, `?o=mv`, `?o=tr`, `?o=lg`.

**Source file:** `website-code/ph-favourites-page.html`.  
**See also:** [pornhub_video_handling.md](pornhub_video_handling.md) (how we use PornHub URLs and yt-dlp), [pornhub_website_video_storage.md](pornhub_website_video_storage.md) (watch page and stream URLs).
