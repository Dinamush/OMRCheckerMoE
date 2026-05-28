# `benchmarks/` — committed baselines + run outputs

| Path | Status | Purpose |
|---|---|---|
| `benchmarks/results/` | **Committed** | Regression baselines from the 2026-05 marker-robustness + throughput work |
| `benchmarks/results/bench_baseline.json` | Committed | Throughput baseline before the moderate-realism refine |
| `benchmarks/results/bench_refine.json` | Committed | Throughput after refinement |
| `benchmarks/results/bench_realism_baseline.json` | Committed | Realism preset baseline |
| `benchmarks/results/bench_realism_refine.json` | Committed | Realism preset after refinement |
| `benchmarks/results/robustness_full_20260523.json` | Committed | Full ArUco marker robustness sweep |
| `benchmarks/results/robustness_confidence_full_20260523.json` | Committed | Confidence-bound version of the above |

New benchmark *runs* should write into per-run subfolders here, which the
gitignore (`benchmark_results*/`, `bench_*.json` at root) keeps out of git.
Only deliberately promoted baselines get committed.

## How to compare runs

```powershell
python scripts/bench/compare_marker_bench.py `
  --before benchmarks/results/robustness_full_20260523.json `
  --after  benchmarks/results/<new-run>.json
```

See [`docs/audits/marker_robustness_benchmark_20260523.md`](../docs/audits/marker_robustness_benchmark_20260523.md)
for the report that produced the committed baselines.
