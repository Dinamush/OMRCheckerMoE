"""Generate a 5000-row prefill benchmark CSV."""
from __future__ import annotations

import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent / "benchmark_5000.csv"
COLS = ["student_name", "school_name", "exam_name", "candidate_number"]


def main() -> None:
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLS)
        writer.writeheader()
        for i in range(5000):
            writer.writerow(
                {
                    "student_name": f"Benchmark Student {i:04d}",
                    "school_name": "Benchmark High School",
                    "exam_name": "GPU CPU Timing Test",
                    "candidate_number": f"{1000000000 + i}",
                }
            )
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
