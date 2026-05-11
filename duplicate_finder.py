"""
Duplicate file detection: fuzzy filename matching plus size similarity.

Scanning targets any user-specified directory, except OS system directories which are
blocked (see resolve_scan_directory).
Deletes validate that paths resolve to existing regular files outside system directories
(resolve_deletable_file).
"""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

# ---------------------------------------------------------------------------
# Typed schemas for file records and API payload
# ---------------------------------------------------------------------------

class FileRecord(TypedDict):
    """A single file entry produced by iter_files()."""
    path: str
    name: str
    size: int


class FileMatchRow(TypedDict):
    """One file's row inside a DuplicateGroup payload."""
    path: str
    name: str
    size: int
    size_display: str
    match_score: float


class DuplicateGroup(TypedDict):
    """A group of probable duplicate files as returned by build_groups_payload()."""
    group_id: int
    confidence: float
    normalized_hint: str
    files: List[FileMatchRow]

# Prefixes of system directories that must never be scanned, previewed, or deleted from.
_SYSTEM_PATH_PREFIXES: tuple[str, ...] = (
    # Windows
    os.environ.get("SystemRoot", r"C:\Windows").lower(),
    r"c:\program files",
    r"c:\program files (x86)",
    r"c:\programdata",
    # Unix / macOS
    "/bin",
    "/sbin",
    "/usr/bin",
    "/usr/sbin",
    "/etc",
    "/boot",
    "/sys",
    "/proc",
    "/dev",
    "/Library/System",
    "/System/Library",
)


def _is_system_path(p: Path) -> bool:
    """Return True if *p* starts with a known OS/system directory prefix."""
    s = str(p).lower().replace("\\", "/")
    return any(s.startswith(prefix.lower().replace("\\", "/")) for prefix in _SYSTEM_PATH_PREFIXES)


def _strip_outer_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1].strip()
    return s


def resolve_scan_directory(
    user_path: str,
    base_dir: Path,
    download_dir_name: str = "downloads",
) -> Path:
    """
    Resolve the folder to scan (any existing, non-system directory on the machine).

    - Empty input: project ``base_dir / download_dir_name`` (created if missing).
    - Non-empty: ``expanduser``, strip outer quotes. Absolute paths are accepted but
      must not fall inside OS system directories (Windows, /bin, /etc, etc.).
      Relative paths: try ``cwd / path``, then ``base_dir / path`` so project-relative
      folders still work when the server was started from another working directory.
    """
    stripped = _strip_outer_quotes(user_path or "")
    if not stripped:
        root = (base_dir / download_dir_name).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

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
                "Path is not a directory or does not exist. "
                f"Tried (cwd): {cwd_try}; (project): {base_try}."
            )

    if not candidate.is_dir():
        raise ValueError("Path is not a directory or does not exist.")

    if _is_system_path(candidate):
        raise ValueError("Scanning system directories is not permitted.")

    return candidate


def resolve_deletable_file(path_str: str) -> Path:
    """Resolve a path that must exist, be a regular file, and not reside in a system directory."""
    p = Path(_strip_outer_quotes(path_str)).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    if not p.exists():
        raise ValueError("Path does not exist.")
    if not p.is_file():
        raise ValueError("Not a regular file.")
    if _is_system_path(p):
        raise ValueError("This path is in a protected system directory.")
    return p


def normalize_name(filename: str) -> str:
    """Lowercase, strip extension, collapse non-alphanumeric to spaces."""
    base = Path(filename).stem.lower()
    base = re.sub(r"[\[\](){}]", " ", base)
    base = re.sub(r"[^\w\s]+", " ", base, flags=re.UNICODE)
    base = re.sub(r"\s+", " ", base).strip()
    return base


def iter_files(root: Path, recursive: bool) -> List[FileRecord]:
    """List regular files under root."""
    out: List[FileRecord] = []
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                fp = Path(dirpath) / fn
                try:
                    if fp.is_file():
                        out.append(
                            {
                                "path": str(fp.resolve()),
                                "name": fn,
                                "size": fp.stat().st_size,
                            }
                        )
                except OSError:
                    continue
    else:
        try:
            for fp in root.iterdir():
                try:
                    if fp.is_file():
                        out.append(
                            {
                                "path": str(fp.resolve()),
                                "name": fp.name,
                                "size": fp.stat().st_size,
                            }
                        )
                except OSError:
                    continue
        except OSError:
            pass
    return out


def _size_similarity(sa: int, sb: int) -> float:
    if sa <= 0 and sb <= 0:
        return 1.0
    m = max(sa, sb, 1)
    return max(0.0, 1.0 - abs(sa - sb) / m)


def _pair_score(
    norm_a: str,
    norm_b: str,
    sa: int,
    sb: int,
    name_weight: float,
    size_weight: float,
) -> float:
    name_score = SequenceMatcher(None, norm_a, norm_b).ratio()
    sz_score = _size_similarity(sa, sb)
    return name_weight * name_score + size_weight * sz_score


def _size_gate(sa: int, sb: int, max_ratio_diff: float = 0.28) -> bool:
    """Skip pair if sizes differ too much (relative)."""
    if sa == 0 and sb == 0:
        return True
    m = max(sa, sb, 1)
    return abs(sa - sb) / m <= max_ratio_diff


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


# Trailing tokens removed before episode / page-index detection (whole-token, case-insensitive).
_RELEASE_TAIL_TOKENS = frozenset({
    "720p", "1080p", "480p", "360p", "2160p", "4320p", "4k", "8k", "uhd", "fhd", "hd",
    "eng", "english", "sub", "subbed", "dub", "dubbed", "raw",
    "uncen", "uncensored", "cen", "censored",
    "x264", "x265", "hevc", "h264", "h265", "aac", "opus",
    "bluray", "bdrip", "webrip", "webdl", "dvdrip",
    "underhentai", "hanime", "xvideos", "pornhub",
})


def _strip_release_metadata(norm: str) -> str:
    """Drop resolution / encoder / site noise from the end so episode digits can be parsed."""
    parts = norm.strip().split()
    if not parts:
        return ""
    changed = True
    while changed and parts:
        changed = False
        last = parts[-1].lower()
        if last in _RELEASE_TAIL_TOKENS:
            parts.pop()
            changed = True
            continue
        if re.fullmatch(r"v\d+x", last):
            parts.pop()
            changed = True
            continue
        if last in ("net", "com", "org", "tv", "io") and len(parts) >= 2:
            parts.pop()
            changed = True
            continue
        if last == "www" and len(parts) >= 2:
            parts.pop()
            changed = True
            continue
    return " ".join(parts).strip()


def _hex_like_token(tok: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{16,}", tok.lower()))


def _series_episode_split(norm: str) -> Optional[Tuple[str, Tuple[int, ...]]]:
    """
    If the normalized name looks like the same title with an episode/sequel index,
    return (title_root, index_tuple). Used to avoid merging sequels (video_1 vs video_2).

    Order matters: try SxxEyy before a plain trailing number so names like ..._s01e02
    are not split on the final digits only.
    """
    n = _strip_release_metadata(norm.strip())
    if not n:
        return None

    # ... S01E02 / s1e10 (spacing or separators before S)
    m = re.match(r"^(.*?)[.\s_-]*s(\d+)e(\d+)$", n, re.I)
    if m:
        base = _collapse_ws(m.group(1))
        if base:
            return (base, (int(m.group(2)), int(m.group(3))))

    # ... episode 3 / ep_12
    m = re.match(r"^(.*?)[.\s_-]+(?:ep|episode)[.\s_-]*(\d+)$", n, re.I)
    if m:
        base = _collapse_ws(m.group(1))
        if base:
            return (base, (int(m.group(2)),))

    # ... part 2 / pt_3
    m = re.match(r"^(.*?)[.\s_-]+(?:part|pt)[.\s_-]*(\d+)$", n, re.I)
    if m:
        base = _collapse_ws(m.group(1))
        if base:
            return (base, (int(m.group(2)),))

    # Pixiv / gallery page index: ..._p7 / ... p12 (underscore may survive normalize_name)
    m = re.match(r"^(.*)[_\s]p(\d+)$", n, re.I)
    if m:
        base = _collapse_ws(m.group(1))
        if base and not base.endswith("p"):  # avoid splitting "... ep12" wrongly if ep normalized odd
            return (base, (int(m.group(2)),))

    # ...Series_05 / ...series 05 (episode before ENG etc.)
    m = re.match(r"^(.*?)[_\s]series[_\s](\d+)$", n, re.I)
    if m:
        base = _collapse_ws(m.group(1))
        if base:
            return (base, (int(m.group(2)),))

    # ..._12 / ...-03 at end (video_1, sequel_2, "... kankei 1" after strip)
    m = re.match(r"^(.*)[_\s-](\d+)$", n)
    if m:
        base = _collapse_ws(m.group(1))
        if base:
            return (base, (int(m.group(2)),))

    return None


def _is_likely_duplicate_copy(norm_a: str, norm_b: str, sa: int, sb: int) -> bool:
    """
    Windows-style copy: same stem except trailing ' (n)' → normalized as extra trailing digit group.
    Require nearly identical file sizes so we do not treat episode 2 vs episode 1 as 'copy'.
    """
    a, b = norm_a.strip(), norm_b.strip()
    if a == b:
        return True
    mlen = max(sa, sb, 1)
    if abs(sa - sb) / mlen > 0.03:
        return False
    for x, y in ((a, b), (b, a)):
        mt = re.match(r"^(.+?)\s+(\d+)$", x)
        if mt:
            base = mt.group(1).strip()
            num = int(mt.group(2))
            if base == y and num >= 2:
                return True
    return False


def _numeric_token_installment_mismatch(norm_a: str, norm_b: str, min_ratio: float = 0.92) -> bool:
    """
    High-similarity names whose token streams differ only in one numeric slot (same length).
    Also one-extra trailing numeric token (episode suffix on one side only), if not hex-heavy.
    """
    a, b = _strip_release_metadata(norm_a.strip()), _strip_release_metadata(norm_b.strip())
    if not a or not b:
        return False

    def _too_hex(s: str) -> bool:
        toks = [t for t in re.split(r"[\s_-]+", s) if t]
        if len(toks) == 1 and _hex_like_token(toks[0]):
            return True
        if toks and _hex_like_token(toks[0]) and len(toks[0]) >= 24:
            return True
        return False

    if _too_hex(a) or _too_hex(b):
        return False

    if SequenceMatcher(None, a, b).ratio() < min_ratio:
        return False

    ta = [t for t in re.split(r"[\s_-]+", a) if t]
    tb = [t for t in re.split(r"[\s_-]+", b) if t]

    if len(ta) == len(tb):
        for i in range(len(ta)):
            if ta[i] != tb[i]:
                if (
                    ta[i].isdigit()
                    and tb[i].isdigit()
                    and int(ta[i]) != int(tb[i])
                    and ta[i + 1 :] == tb[i + 1 :]
                ):
                    return True
                return False
        return False

    if len(ta) == len(tb) + 1 and ta[-1].isdigit() and ta[:-1] == tb:
        return True
    if len(tb) == len(ta) + 1 and tb[-1].isdigit() and tb[:-1] == ta:
        return True

    return False


def _same_series_different_installment(norm_a: str, norm_b: str) -> bool:
    """
    True when both names share the same parsed title root but differ in episode/sequel
    number — these should not be clustered as file duplicates.
    """
    pa = _series_episode_split(norm_a)
    pb = _series_episode_split(norm_b)
    if pa is None or pb is None:
        return False
    base_a, idx_a = pa
    base_b, idx_b = pb
    if base_a != base_b:
        return False
    return idx_a != idx_b


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._n = n
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        if not (0 <= x < self._n):
            raise IndexError(f"Index {x} out of range [0, {self._n})")
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def cluster_duplicates(
    files: List[FileRecord],
    name_weight: float = 0.85,
    size_weight: float = 0.15,
    threshold: float = 0.72,
    max_size_ratio_diff: float = 0.28,
) -> List[List[int]]:
    """
    Return lists of file indices that form duplicate groups (group size >= 2).

    Pairs are scored as: ``name_weight * name_similarity + size_weight * size_similarity``.
    A pair is grouped when score >= ``threshold`` and the size ratio difference is within
    ``max_size_ratio_diff``.  Series-episode and copy-number heuristics can fast-accept or
    fast-reject pairs before the score is computed.

    Args:
        files: list of dicts with at least ``name`` (str) and ``size`` (int) keys.
        name_weight: contribution of filename similarity (default 0.85).
        size_weight: contribution of file-size similarity (default 0.15).
        threshold: minimum combined score to treat a pair as duplicates (default 0.72).
        max_size_ratio_diff: reject pairs whose size ratio differs by more than this
            fraction (default 0.28, i.e. files must be within ~22% of each other in size).

    Returns:
        List of groups; each group is a list of indices into *files*.
    """
    n = len(files)
    if n < 2:
        return []

    norms = [normalize_name(f["name"]) for f in files]
    sizes = [int(f["size"]) for f in files]
    uf = _UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if not _size_gate(sizes[i], sizes[j], max_size_ratio_diff):
                continue
            if not _is_likely_duplicate_copy(norms[i], norms[j], sizes[i], sizes[j]):
                if _same_series_different_installment(norms[i], norms[j]):
                    continue
                if _numeric_token_installment_mismatch(norms[i], norms[j]):
                    continue
            score = _pair_score(
                norms[i], norms[j], sizes[i], sizes[j], name_weight, size_weight
            )
            if score >= threshold:
                uf.union(i, j)

    buckets: Dict[int, List[int]] = {}
    for i in range(n):
        r = uf.find(i)
        buckets.setdefault(r, []).append(i)

    groups = [sorted(ids) for ids in buckets.values() if len(ids) >= 2]
    groups.sort(key=lambda g: g[0])
    return groups


def _group_max_pairwise_score(
    indices: List[int],
    files: List[FileRecord],
    norms: List[str],
    sizes: List[int],
    name_weight: float,
    size_weight: float,
) -> float:
    best = 0.0
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            i, j = indices[a], indices[b]
            best = max(
                best,
                _pair_score(norms[i], norms[j], sizes[i], sizes[j], name_weight, size_weight),
            )
    return best


def _file_best_peer_score(
    idx: int,
    indices: List[int],
    files: List[FileRecord],
    norms: List[str],
    sizes: List[int],
    name_weight: float,
    size_weight: float,
) -> float:
    best = 0.0
    for j in indices:
        if j == idx:
            continue
        best = max(
            best,
            _pair_score(
                norms[idx], norms[j], sizes[idx], sizes[j], name_weight, size_weight
            ),
        )
    return best


def human_size(n: int) -> str:
    x = float(max(0, n))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(x)} B"
            return f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{int(max(0, n))} B"


def build_groups_payload(
    files: List[FileRecord],
    groups: List[List[int]],
    name_weight: float = 0.85,
    size_weight: float = 0.15,
) -> List[DuplicateGroup]:
    """
    Build API/UI payload. Confidence per group is the maximum pairwise combined score
    within that group (documented in the UI).
    """
    norms = [normalize_name(f["name"]) for f in files]
    sizes = [int(f["size"]) for f in files]

    payload: List[DuplicateGroup] = []
    for gid, indices in enumerate(groups):
        conf = _group_max_pairwise_score(
            indices, files, norms, sizes, name_weight, size_weight
        )
        rep = norms[indices[0]]
        rows = []
        for i in indices:
            rows.append(
                {
                    "path": files[i]["path"],
                    "name": files[i]["name"],
                    "size": files[i]["size"],
                    "size_display": human_size(files[i]["size"]),
                    "match_score": round(
                        _file_best_peer_score(
                            i, indices, files, norms, sizes, name_weight, size_weight
                        ),
                        3,
                    ),
                }
            )
        payload.append(
            {
                "group_id": gid,
                "confidence": round(conf, 3),
                "normalized_hint": rep,
                "files": rows,
            }
        )
    return payload
