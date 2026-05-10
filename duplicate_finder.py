"""
Duplicate file detection: fuzzy filename matching plus size similarity.

Scanning may target any directory the user specifies (see resolve_scan_directory).
Deletes validate that paths resolve to existing regular files (resolve_deletable_file).
"""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
    Resolve the folder to scan (any existing directory on the machine).

    - Empty input: project ``base_dir / download_dir_name`` (created if missing).
    - Non-empty: ``expanduser``, strip outer quotes. Absolute paths may point anywhere.
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

    return candidate


def resolve_deletable_file(path_str: str) -> Path:
    """Resolve a path that must exist and refer to a regular file."""
    p = Path(_strip_outer_quotes(path_str)).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    if not p.exists():
        raise ValueError("Path does not exist.")
    if not p.is_file():
        raise ValueError("Not a regular file.")
    return p


def normalize_name(filename: str) -> str:
    """Lowercase, strip extension, collapse non-alphanumeric to spaces."""
    base = Path(filename).stem.lower()
    base = re.sub(r"[\[\](){}]", " ", base)
    base = re.sub(r"[^\w\s]+", " ", base, flags=re.UNICODE)
    base = re.sub(r"\s+", " ", base).strip()
    return base


def iter_files(root: Path, recursive: bool) -> List[Dict[str, Any]]:
    """List regular files under root."""
    out: List[Dict[str, Any]] = []
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


def _series_episode_split(norm: str) -> Optional[Tuple[str, Tuple[int, ...]]]:
    """
    If the normalized name looks like the same title with an episode/sequel index,
    return (title_root, index_tuple). Used to avoid merging sequels (video_1 vs video_2).

    Order matters: try SxxEyy before a plain trailing number so names like ..._s01e02
    are not split on the final digits only.
    """
    n = norm.strip()
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

    # ..._12 / ...-03 at end (video_1, sequel_2)
    m = re.match(r"^(.*)[_\s-](\d+)$", n)
    if m:
        base = _collapse_ws(m.group(1))
        if base:
            return (base, (int(m.group(2)),))

    return None


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
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_duplicates(
    files: List[Dict[str, Any]],
    name_weight: float = 0.85,
    size_weight: float = 0.15,
    threshold: float = 0.72,
    max_size_ratio_diff: float = 0.28,
) -> List[List[int]]:
    """
    Return lists of indices into files that form duplicate groups (size >= 2).
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
            if _same_series_different_installment(norms[i], norms[j]):
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
    files: List[Dict[str, Any]],
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
    files: List[Dict[str, Any]],
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
    files: List[Dict[str, Any]],
    groups: List[List[int]],
    name_weight: float = 0.85,
    size_weight: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Build API/UI payload. Confidence per group is the maximum pairwise combined score
    within that group (documented in the UI).
    """
    norms = [normalize_name(f["name"]) for f in files]
    sizes = [int(f["size"]) for f in files]

    payload: List[Dict[str, Any]] = []
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
