# Website code (saved pages)

This folder holds **saved HTML** (and related assets if any) from partner sites, used as reference for how those sites structure pages and deliver video.

## Contents

| File | Description |
|------|-------------|
| **ph-video.html** | PornHub video watch page (`view_video.php?viewkey=...`). Contains the main container, **VIDEO_SHOW**, **flashvars_&lt;video_id&gt;** (including **mediaDefinitions** with HLS/MP4 URLs), and the HTML5 player markup. |
| **ph-favourites-page.html** | PornHub user favourites page (`/users/<username>/videos/favorites`). Contains **#moreData** (video list), **#moreDataBtn** (“Load More”), sort links (`?o=newest`, etc.), and URL/pagination behaviour. |

## Documentation

- **Watch page – how videos are stored and handled (current and legacy):**  
  [../docs/pornhub_website_video_storage.md](../docs/pornhub_website_video_storage.md)
- **Favourites page and pagination (URL ?page=N, Load more, scroll):**  
  [../docs/pornhub_favourites_page_and_pagination.md](../docs/pornhub_favourites_page_and_pagination.md)
- **How this repo’s code uses PornHub (main.py, ph.py):**  
  [../docs/pornhub_video_handling.md](../docs/pornhub_video_handling.md)

## Usage

- Use **ph-video.html** as the source of truth for PornHub watch-page structure (flashvars, mediaDefinitions, CDN URLs, player config).
- Use **ph-favourites-page.html** for favourites layout, video list (`#moreData`), “Load More” (`#moreDataBtn`), and pagination (URL `?page=N`, sort `?o=...`).
- When the live site changes, re-save the relevant page here and update the corresponding doc if structure or URLs change.
