"""
Pixiv illust title translation for English (or other) filesystem filenames.

Uses ``deep-translator`` (Google by default; optional DeepL free API key).
Translations are cached by illust id under ``get_user_data_dir()``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from paths import get_user_data_dir

if TYPE_CHECKING:
    from settings import AppSettings

logger = logging.getLogger(__name__)

CACHE_VERSION = 1
CACHE_FILENAME = "pixiv_title_cache.json"

_translate_lock = threading.Lock()
_cache_io_lock = threading.Lock()
_last_translate_at = 0.0


def _cache_path(user_data_dir: Optional[Path] = None) -> Path:
    root = user_data_dir or get_user_data_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root / CACHE_FILENAME


def _load_cache(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"version": CACHE_VERSION, "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": CACHE_VERSION, "entries": {}}
    if not isinstance(data, dict):
        return {"version": CACHE_VERSION, "entries": {}}
    if data.get("version") != CACHE_VERSION:
        return {"version": CACHE_VERSION, "entries": {}}
    entries = data.get("entries")
    if not isinstance(entries, dict):
        data["entries"] = {}
    data.setdefault("version", CACHE_VERSION)
    return data


def _save_cache(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _deepl_api_key(settings: AppSettings) -> str:
    key = (getattr(settings, "pixiv_deepl_api_key", None) or "").strip()
    if key:
        return key
    return os.environ.get("DEEPL_AUTH_KEY", "").strip()


def _needs_translation(title: str) -> bool:
    text = (title or "").strip()
    if not text:
        return False
    try:
        text.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def _translate_remote(title: str, target: str, api_key: str, delay: float) -> tuple[str, bool]:
    """Return ``(text, ok)``; ``ok`` is False when the remote API failed."""
    global _last_translate_at
    snippet = title[:5000]
    with _translate_lock:
        if delay > 0:
            wait = delay - (time.monotonic() - _last_translate_at)
            if wait > 0:
                time.sleep(wait)
        try:
            from deep_translator import DeeplTranslator, GoogleTranslator

            if api_key:
                translator = DeeplTranslator(
                    api_key=api_key,
                    target=target,
                    use_free_api=api_key.endswith(":fx"),
                )
            else:
                translator = GoogleTranslator(source="auto", target=target)
            result = (translator.translate(snippet) or "").strip()
            _last_translate_at = time.monotonic()
            return (result or title), True
        except Exception as e:
            logger.warning("Pixiv title translation failed: %s", e)
            _last_translate_at = time.monotonic()
            return title, False


def resolve_pixiv_filename_title(
    title: str,
    illust_id: str,
    *,
    settings: Optional[AppSettings] = None,
    user_data_dir: Optional[Path] = None,
) -> str:
    """
    Return a title string suitable for ``_safe_filename`` (translated when enabled).

    Original ``title`` is unchanged for UI/tracker; call this only for on-disk names.
    """
    from settings import load_settings

    s = settings or load_settings()
    raw = (title or "").strip() or str(illust_id)

    if not getattr(s, "pixiv_translate_titles", True):
        return raw

    target = (getattr(s, "pixiv_title_target_lang", None) or "en").strip().lower() or "en"
    if not _needs_translation(raw):
        return raw

    path = _cache_path(user_data_dir)
    with _cache_io_lock:
        cache = _load_cache(path)
        entries: Dict[str, Any] = cache.setdefault("entries", {})
        ent = entries.get(illust_id)
        if isinstance(ent, dict) and ent.get("source") == raw:
            cached = ent.get(target)
            if isinstance(cached, str) and cached.strip():
                return cached.strip()

    api_key = _deepl_api_key(s)
    delay = float(getattr(s, "pixiv_translate_delay_seconds", 0.35) or 0.0)
    if not api_key:
        delay = max(delay, 0.35)

    translated, ok = _translate_remote(raw, target, api_key, delay)
    if ok:
        with _cache_io_lock:
            cache = _load_cache(path)
            entries = cache.setdefault("entries", {})
            entries[illust_id] = {
                "source": raw,
                target: translated,
                "updated": int(time.time()),
            }
            cache["version"] = CACHE_VERSION
            _save_cache(path, cache)
        return translated
    return raw


_WIN_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def sanitize_filename_slug(text: str, max_len: int = 80) -> str:
    """Windows-safe slug: forbidden chars, whitespace collapsed, length cap."""
    s = re.sub(r"[\x00-\x1f\x7f]", "_", (text or "").strip())
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = re.sub(r"\s+", "_", s).strip("._") or "untitled"
    stem = s.split(".")[0].upper()
    if stem in _WIN_RESERVED or re.match(r"^COM[1-9]\d?$", stem) or re.match(r"^LPT[1-9]\d?$", stem):
        s = f"pixiv_{s}"
    if len(s) > max_len:
        s = s[:max_len].rstrip("._") or "untitled"
    return s
