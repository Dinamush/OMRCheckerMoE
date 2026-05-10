"""
Compare candidate video titles against filenames already on disk (library folder).

Uses the same name normalization as duplicate_finder so spacing/punctuation variants
still match sensibly.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import FrozenSet, Optional, Set

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

    return candidate


def build_library_normalized_names(root: Path, *, recursive: bool) -> FrozenSet[str]:
    """Normalized stems for video-like files under root."""
    found: Set[str] = set()

    def consider_file(fp: Path) -> None:
        try:
            if not fp.is_file():
                return
            if fp.suffix.lower() not in VIDEO_EXTENSIONS:
                return
            key = normalize_compare_key_from_filename(fp.name)
            if key:
                found.add(key)
        except OSError:
            return

    if recursive:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                consider_file(Path(dirpath) / fn)
    else:
        try:
            for fp in root.iterdir():
                consider_file(fp)
        except OSError:
            pass

    return frozenset(found)


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
            if nt in ln or ln in nt:
                return True

    return False
