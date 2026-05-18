"""
Compare candidate video titles against filenames already on disk (library folder).

Uses the same name normalization as duplicate_finder so spacing/punctuation variants
still match sensibly.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from duplicate_finder import _is_system_path
from typing import FrozenSet, Optional, Set
from urllib.parse import parse_qs, urlparse

from duplicate_finder import normalize_name

VIDEO_EXTENSIONS = frozenset(
    {
        ".mp4",
        ".mkv",
        ".webm",
        ".avi",
        ".mov",
        ".m4v",
        ".flv",
        ".wmv",
        ".mpeg",
        ".mpg",
    }
)

MIN_SUBSTRING_LEN = 15

# PornHub saves as {title}_{viewkey}.mp4 (archive/ph.py)
_PH_VIEWKEY_SUFFIX = re.compile(r"_(ph[0-9a-f]{8,}|[0-9a-f]{10,})$", re.I)


@dataclass(frozen=True)
class LibraryIndex:
    """Normalized title keys plus PornHub viewkeys found in on-disk filenames."""

    normalized_titles: FrozenSet[str]
    viewkeys: FrozenSet[str]


def _collapse_for_compare(s: str) -> str:
    """Underscores in filenames and spaces in titles converge for comparison."""
    s = re.sub(r"_+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_compare_key_from_filename(filename: str) -> str:
    return _collapse_for_compare(normalize_name(filename))


def normalize_compare_key_from_title(title: str) -> str:
    t = (title or "").strip()
    return _collapse_for_compare(normalize_name(f"{t}.mp4"))


def _strip_outer_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1].strip()
    return s


def resolve_optional_library_directory(
    user_path: Optional[str],
    base_dir: Path,
) -> Optional[Path]:
    """
    Empty / whitespace-only → None (feature disabled).
    Otherwise resolve like duplicate_finder scan paths (cwd + base_dir fallback for relative).
    """
    stripped = _strip_outer_quotes(user_path or "")
    if not stripped:
        return None

    p = Path(stripped).expanduser()
    if p.is_absolute():
        candidate = p.resolve()
    else:
        cwd_try = (Path.cwd() / p).resolve()
        base_try = (base_dir / p).resolve()
        if cwd_try.is_dir():
            candidate = cwd_try
        elif base_try.is_dir():
            candidate = base_try
        else:
            raise ValueError(
                "Existing-library path is not a directory or does not exist. "
                f"Tried (cwd): {cwd_try}; (profile): {base_try}."
            )

    if not candidate.is_dir():
        raise ValueError("Existing-library path is not a directory or does not exist.")

    if _is_system_path(candidate):
        raise ValueError("System directories cannot be used as an existing-library folder.")

    return candidate


def _consider_library_file(
    fp: Path, titles: Set[str], viewkeys: Set[str]
) -> None:
    try:
        if not fp.is_file() or fp.suffix.lower() not in VIDEO_EXTENSIONS:
            return
        key = normalize_compare_key_from_filename(fp.name)
        if key:
            titles.add(key)
        stem = fp.stem
        m = _PH_VIEWKEY_SUFFIX.search(stem)
        if m:
            viewkeys.add(m.group(1).lower())
            title_part = stem[: m.start()].strip()
            if title_part:
                tk = _collapse_for_compare(normalize_name(f"{title_part}.mp4"))
                if tk:
                    titles.add(tk)
    except OSError:
        return


def build_library_index(root: Path, *, recursive: bool) -> LibraryIndex:
    """Scan a library folder for title keys and PornHub viewkeys (incl. ph/ subfolders)."""
    titles: Set[str] = set()
    viewkeys: Set[str] = set()

    if recursive:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                _consider_library_file(Path(dirpath) / fn, titles, viewkeys)
    else:
        try:
            for fp in root.iterdir():
                if fp.is_file():
                    _consider_library_file(fp, titles, viewkeys)
                elif fp.is_dir():
                    # Site subfolders (ph/, xh/, pixiv/) when user points at downloads root
                    for child in fp.iterdir():
                        _consider_library_file(child, titles, viewkeys)
        except OSError:
            pass

    return LibraryIndex(
        normalized_titles=frozenset(titles),
        viewkeys=frozenset(viewkeys),
    )


def build_library_normalized_names(root: Path, *, recursive: bool) -> FrozenSet[str]:
    """Normalized stems for video-like files under root."""
    return build_library_index(root, recursive=recursive).normalized_titles


def viewkey_from_pornhub_url(video_page_url: str) -> str:
    qs = parse_qs(urlparse(video_page_url).query)
    return ((qs.get("viewkey") or [""])[0] or "").strip()


def matches_pornhub_in_library(
    video_page_url: str, title: str, index: LibraryIndex
) -> bool:
    """Match favourites against on-disk PornHub files ({title}_{viewkey}.mp4)."""
    vk = viewkey_from_pornhub_url(video_page_url).lower()
    if vk and vk in index.viewkeys:
        return True
    if matches_existing_library(title, index.normalized_titles):
        return True
    if vk and (title or "").strip():
        fn_key = normalize_compare_key_from_filename(f"{title}_{vk}.mp4")
        if fn_key in index.normalized_titles:
            return True
    return False


def matches_existing_library(title: str, library_norms: FrozenSet[str]) -> bool:
    """
    True if this download title likely corresponds to a file already in the library.

    Compares normalize_name(title) to normalized library stems, plus a guarded
    substring rule for truncation / minor yt-dlp vs disk naming differences.
    """
    if not library_norms or not (title or "").strip():
        return False

    nt = normalize_compare_key_from_title(title)
    if not nt:
        return False

    if nt in library_norms:
        return True

    for ln in library_norms:
        if len(nt) >= MIN_SUBSTRING_LEN and len(ln) >= MIN_SUBSTRING_LEN:
            shorter, longer = (nt, ln) if len(nt) <= len(ln) else (ln, nt)
            if shorter in longer and len(shorter) / len(longer) >= 0.92:
                return True

    return False
