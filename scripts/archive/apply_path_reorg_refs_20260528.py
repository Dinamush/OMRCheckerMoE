"""One-shot updater for the 2026-05-28 repo organisation.

Rewrites in-text references from the pre-reorg paths to the new layout:

* ``scripts/<name>.py``                     ->  ``scripts/<bucket>/<name>.py``
* ``docs/<dated_report>.md``                ->  ``docs/audits/<dated_report>.md``
* ``docs/research_brief_*.md``              ->  ``docs/research/...``
* ``docs/student_fill_feature_design.md``   ->  ``docs/features/student-fill/...``
* ``docs/ARCHITECTURE.md``                  ->  ``docs/architecture/ARCHITECTURE.md``

Run from repo root once:

    python scripts/archive/apply_path_reorg_refs_20260528.py

Idempotent (no-op if every reference has already been migrated).
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SCRIPT_BUCKETS: dict[str, str] = {
    "bench_marker_robustness.py": "bench",
    "bench_moderate_realism.py": "bench",
    "bench_pdf_split.py": "bench",
    "compare_marker_bench.py": "bench",
    "calibrate_bubbles.py": "diagnostics",
    "debug_alignment.py": "diagnostics",
    "debug_realism_samples.py": "diagnostics",
    "gen_alignment_samples.py": "diagnostics",
    "omr_stress_test.py": "diagnostics",
    "make_test_pdf.py": "dev",
    "probe_edge_cases.py": "dev",
    "smoke_test_pdf_upload.py": "dev",
    "stress_test.py": "dev",
    "stress_test_parallel_split.py": "dev",
    "verify_inmemory_fewer_writes.py": "dev",
    "verify_moderate_e2e.py": "dev",
    "add_defender_exclusion.ps1": "archive",
}

DOC_REWRITES: dict[str, str] = {
    "docs/audit_report_20260524.md": "docs/audits/audit_report_20260524.md",
    "docs/marker_robustness_benchmark_20260523.md": "docs/audits/marker_robustness_benchmark_20260523.md",
    "docs/student_fill_e2e_report_20260524.md": "docs/audits/student_fill_e2e_report_20260524.md",
    "docs/student_fill_feature_design.md": "docs/features/student-fill/student_fill_feature_design.md",
    "docs/research_brief_omr_throughput_2026.md": "docs/research/research_brief_omr_throughput_2026.md",
    "docs/research_brief_scan_simulation_2026.md": "docs/research/research_brief_scan_simulation_2026.md",
    "docs/ARCHITECTURE.md": "docs/architecture/ARCHITECTURE.md",
}

TARGETS: list[Path] = [
    REPO / "README.md",
    REPO / "SHEETS.md",
    REPO / "MoE-May-2026-Variants-SMQ25-0" / "README.md",
    REPO / "MoE-May-2026-Variants-SMQ25-0" / "DESIGN.md",
    REPO / "docs" / "audits" / "marker_robustness_benchmark_20260523.md",
    REPO / "docs" / "audits" / "student_fill_e2e_report_20260524.md",
    REPO / "docs" / "architecture" / "ARCHITECTURE.md",
    REPO / "scripts" / "bench" / "bench_marker_robustness.py",
    REPO / "scripts" / "bench" / "bench_pdf_split.py",
    REPO / "scripts" / "dev" / "smoke_test_pdf_upload.py",
    REPO / "scripts" / "dev" / "stress_test_parallel_split.py",
    REPO / "scripts" / "dev" / "verify_inmemory_fewer_writes.py",
    REPO / "scripts" / "diagnostics" / "calibrate_bubbles.py",
    REPO / "scripts" / "diagnostics" / "debug_alignment.py",
    REPO / "scripts" / "diagnostics" / "debug_realism_samples.py",
    REPO / "scripts" / "diagnostics" / "gen_alignment_samples.py",
    REPO / "webui" / "services" / "scan_simulation.py",
]


def rewrite(text: str) -> tuple[str, int]:
    n = 0
    for name, bucket in SCRIPT_BUCKETS.items():
        old = f"scripts/{name}"
        new = f"scripts/{bucket}/{name}"
        if old in text and new not in text.replace(old, ""):
            cnt = text.count(old)
            text = text.replace(old, new)
            n += cnt
    for old, new in DOC_REWRITES.items():
        if old in text:
            cnt = text.count(old)
            text = text.replace(old, new)
            n += cnt
    return text, n


def main() -> None:
    changed = 0
    for path in TARGETS:
        if not path.exists():
            print(f"  skip (missing): {path.relative_to(REPO)}")
            continue
        original = path.read_text(encoding="utf-8")
        updated, n = rewrite(original)
        if n and updated != original:
            path.write_text(updated, encoding="utf-8")
            print(f"  rewrote {n:>3} refs in {path.relative_to(REPO)}")
            changed += 1
        else:
            print(f"  ok     :        {path.relative_to(REPO)}")
    print(f"\nDone. Files modified: {changed}")


if __name__ == "__main__":
    main()
