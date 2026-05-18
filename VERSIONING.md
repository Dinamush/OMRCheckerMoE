# Versioning policy — SHUCK3R

## Single source of truth

- **Version string:** `version.py` → `__version__`
- **Release notes:** `CHANGELOG.md` (Keep a Changelog format)
- **Git tags:** `v{MAJOR}.{MINOR}.{PATCH}` (e.g. `v1.2.0`)

## Semver rules

| Bump | When |
|------|------|
| **MAJOR** | Breaking UX, config migration, or removal of a supported site/workflow |
| **MINOR** | New site, major feature (Pixiv, duplicate checker, library browser), or notable capability |
| **PATCH** | Bug fixes, copy/UI tweaks, dependency pins, test-only changes |

While the app is stabilizing multi-site support, **1.x** is appropriate. Pre-**1.0.0** history is documented in `CHANGELOG.md` as **0.x** retroactive releases.

## Pre-1.0 vs 1.0

Ship **1.0.0** when:

- PornHub, xHamster, and Pixiv bookmark flows are stable for a full release cycle
- Settings and user-data paths are documented and unlikely to break on upgrade
- PyInstaller builds (`SHUCK3R.exe` / `SHUCK3RWebView.exe`) are tested on a clean Windows profile

Until then, use **1.x** for feature releases (current line: **1.2.0** = Pixiv ugoira, manga ZIP, title translation).

## Git tags

### Backfill (recommended once)

Tag historical milestones so `git log` and GitHub releases align with `CHANGELOG.md`:

```bash
git tag -a v0.1.0 341d90d -m "Initial prototype"
git tag -a v0.3.0 210559a -m "Progress tracking and xHamster"
git tag -a v0.8.0 2959b50 -m "SHUCK3R rebrand"
git tag -a v1.0.0-beta.2 0a83cf0 -m "Session history, library, cancel"
git tag -a v1.1.0 855cdbe -m "Pixiv bookmarks"
git tag -a v1.2.0 HEAD -m "Pixiv ugoira, manga, translation"
```

Adjust hashes if your branch diverges; run `git log --oneline` to confirm.

### Going forward

1. Update `version.py` and `CHANGELOG.md` `[Unreleased]` section
2. Commit: `Release v1.2.1` (or similar)
3. `git tag -a v1.2.1 -m "v1.2.1"`
4. Push tags when ready: `git push origin v1.2.1`

## Files that should reference the version

| File | Purpose |
|------|---------|
| `version.py` | Definition |
| `CHANGELOG.md` | Human release notes |
| `main.py` | `/api/version`, Jinja global `app_version` |
| `templates/base.html` | Footer display |
| `hamster_scraper.spec` / `hamster_scraper_webview.spec` | `hiddenimports` includes `version` |
| `README.MD` | Optional version badge (manual or CI) |

Internal schema versions (e.g. `CACHE_VERSION` in `pixiv_titles.py`, `CHROME_LOGIN_WAIT_VERSION`) are **not** app semver — bump only when migrating stored data.

## Commit message hints

- `feat:` → MINOR
- `fix:` → PATCH
- `feat!:` or `BREAKING:` → MAJOR
