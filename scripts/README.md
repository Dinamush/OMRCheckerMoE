# `scripts/` — tooling

Operational, dev, bench, and diagnostic scripts. Organised into four buckets
on **2026-05-28** to keep the repo root clean.

| Bucket | When to put a script here |
|---|---|
| `bench/` | Repeatable performance measurement that emits a JSON / report. Outputs go to `benchmarks/results/`. |
| `dev/` | Live smoke tests / end-to-end verifications that hit a running server or generate fixtures. |
| `diagnostics/` | Calibration, alignment overlay, or per-template debug tooling that operators run during template work. |
| `archive/` | Completed one-shot migrations. Never deleted; kept for history. |
| (root) | Permanent operational tools used by IT (e.g. `add_av_exclusion.ps1`). |

## Current inventory

```
scripts/
├── add_av_exclusion.ps1
├── archive/
│   ├── add_defender_exclusion.ps1                 (deprecated shim → add_av_exclusion.ps1)
│   ├── apply_path_reorg_refs_20260528.py          (this reorg)
│   └── apply_sheet_rename_refs_20260525.py        (prior MoE-* rename)
├── bench/
│   ├── bench_marker_robustness.py
│   ├── bench_moderate_realism.py
│   ├── bench_pdf_split.py
│   └── compare_marker_bench.py
├── dev/
│   ├── make_test_pdf.py
│   ├── probe_edge_cases.py
│   ├── smoke_test_pdf_upload.py
│   ├── stress_test.py
│   ├── stress_test_parallel_split.py
│   ├── verify_inmemory_fewer_writes.py
│   └── verify_moderate_e2e.py
└── diagnostics/
    ├── calibrate_bubbles.py
    ├── debug_alignment.py
    ├── debug_realism_samples.py
    ├── gen_alignment_samples.py
    └── omr_stress_test.py
```

All scripts are designed to be run from the repo root:

```powershell
python scripts/bench/bench_marker_robustness.py --rows 30 --label baseline
python scripts/diagnostics/calibrate_bubbles.py
python scripts/dev/smoke_test_pdf_upload.py --base-url http://127.0.0.1:5051
```

See [`REPO_LAYOUT.md`](../REPO_LAYOUT.md) for the wider repository map.
