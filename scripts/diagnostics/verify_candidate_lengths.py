"""Verify candidate-number accuracy in a batch Results CSV.

Reports any row whose CandidateNumber is not exactly 10 characters, or
whose value contains MR(...) (ambiguity flag), or whose last digits do
not match the page index (when files are named ..._page_NNNN.jpg).

Usage:
    python scripts/diagnostics/verify_candidate_lengths.py <results_csv>
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path


def main(csv_path: str) -> int:
    path = Path(csv_path)
    total = 0
    by_len: Counter[int] = Counter()
    mr_rows: list[tuple[str, str]] = []
    page_mismatch: list[tuple[str, str, str]] = []
    contains_letter: list[tuple[str, str]] = []
    page_re = re.compile(r"_page_(\d+)\.")

    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            total += 1
            file_id = row.get("file_id", "")
            cand = row.get("CandidateNumber", "")
            by_len[len(cand)] += 1
            if "MR(" in cand:
                mr_rows.append((file_id, cand))
            if not cand.isdigit() and "MR(" not in cand:
                contains_letter.append((file_id, cand))
            m = page_re.search(file_id)
            if m and cand.isdigit() and len(cand) == 10:
                page_suffix = m.group(1).lstrip("0") or "0"
                expected_tail = page_suffix.zfill(min(len(cand), 7))
                if not cand.endswith(expected_tail):
                    page_mismatch.append((file_id, cand, expected_tail))

    print(f"Total rows: {total}")
    print(f"CandidateNumber length distribution: {dict(sorted(by_len.items()))}")
    print(f"Rows with MR(...) ambiguity flag: {len(mr_rows)}")
    print(f"Rows with non-digit / non-MR content: {len(contains_letter)}")
    print(f"Rows where page-suffix mismatch tail: {len(page_mismatch)}")
    for label, sample in (("MR samples", mr_rows[:5]), ("page mismatch samples", page_mismatch[:5])):
        if sample:
            print(f"\n{label}:")
            for item in sample:
                print(f"  {item}")
    healthy = by_len.get(10, 0) - len(mr_rows)
    pct = (healthy / total * 100) if total else 0.0
    print(f"\nHealthy single-digit-per-column rows: {healthy}/{total} ({pct:.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
