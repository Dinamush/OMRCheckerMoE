# OMRCheckerMoE — Repository Layout

> The one-page map of every top-level directory. Last reorganisation:
> **2026-05-28**. For sheet-specific naming, see [`SHEETS.md`](SHEETS.md). For
> end-to-end flows, see [`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md).

```text
OMRCheckerMoE/
├── main.py                      ← CLI entry point (OMR runner)
├── desktop.py                   ← Desktop (pywebview) launcher
├── OMRChecker.spec              ← PyInstaller spec
├── Dockerfile, pyproject.toml, pytest.ini, requirements*.txt
├── README.md, SHEETS.md, REPO_LAYOUT.md, LICENSE, CONTRIBUTING.md, CODE_OF_CONDUCT.md
│
├── src/                         ← Core OMR engine (entry, template, processors, tests)
├── webui/                       ← FastAPI service, templates, static, services, tests
├── prefill_package/             ← Python package: renders prefilled answer sheets
│
├── MoE-April-2026-Landscape-NNQ25-0/   ← Production landscape preset (default)
├── MoE-April-2026-Portrait-NNQ25-0/    ← Retired portrait preset (archive)
├── MoE-May-2026-Variants-SMQ25-0/      ← Portrait variants + sweep tooling (v1)
├── MoE-May-2026-Portrait-SMQ25-1/      ← Optimised portrait preset (v2)
│
├── samples/                     ← Example sheets for engine smoke tests
│
├── scripts/                     ← Operational, dev, bench, and diagnostic tooling
│   ├── add_av_exclusion.ps1
│   ├── archive/                 ←   one-shot migrations (kept for history)
│   ├── bench/                   ←   benchmark + comparison harnesses
│   ├── dev/                     ←   live smoke / stress / verification scripts
│   └── diagnostics/             ←   calibration & alignment debugging
│
├── benchmarks/                  ← Benchmark inputs / outputs
│   └── results/                 ←   committed JSON baselines (bench_*, robustness_*)
│
├── docs/                        ← Markdown documentation (see index below)
│   ├── architecture/            ←   evergreen reference (ARCHITECTURE, NERS diagram)
│   ├── audits/                  ←   dated audit + benchmark reports
│   ├── features/                ←   per-feature design docs (e.g. student-fill)
│   ├── research/                ←   long-form research briefs
│   └── assets/                  ←   embedded images
│
├── inputs/, outputs/, logs/     ← Runtime scratch (gitignored except .gitkeep)
├── build/, dist/                ← PyInstaller artefacts (gitignored)
├── .github/                     ← CI workflows
└── .venv/, .pytest_cache/, __pycache__/   ← Local-only (gitignored)
```

## Top-level files

| File | Purpose | Owner / Notes |
|---|---|---|
| `main.py` | CLI entry point used by tests, `webui/services/omr.py`, and PyInstaller | Keep at root |
| `desktop.py` | Spawns Uvicorn + pywebview; the packaged Windows app target | Keep at root |
| `OMRChecker.spec` | PyInstaller build spec | Edit when adding bundled data |
| `Dockerfile` | Container build for the FastAPI service | |
| `pyproject.toml` / `pytest.ini` / `.pylintrc` / `.pre-commit-config.yaml` | Tooling | |
| `requirements.txt` / `requirements.dev.txt` | Runtime + dev deps | |
| `README.md` | Project overview | Outward-facing |
| `SHEETS.md` | Canonical sheet-directory index + legacy aliases | Source of truth for preset names |
| `REPO_LAYOUT.md` | **This file** | Source of truth for directory layout |
| `LICENSE`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `Contributors.md` | Project meta | |

## `src/` — OMR engine

The pure-Python OMR engine. Imported by `main.py` and `webui/services/omr.py`.
Tests live in `src/tests/`. See `src/entry.py` for the processing pipeline and
`src/processors/CropOnMarkers.py` for ArUco alignment.

## `webui/` — FastAPI service

| Subfolder | Role |
|---|---|
| `webui/api.py`, `app.py` | Route definitions + app factory |
| `webui/services/` | Business logic (`batches`, `omr`, `prefill`, `presets`, `scan_simulation`) |
| `webui/templates/`, `static/` | Jinja2 + vanilla JS frontend |
| `webui/sheet_registry.py` | **Single source of truth for sheet directory names + legacy aliases** |
| `webui/settings.py`, `schemas*.py` | Pydantic settings + API schemas |
| `webui/tests/` | Pytest suite (fixtures in `conftest.py`) |

## `prefill_package/` — Prefill renderer

Python package (no hyphens — Python import constraint) that renders filled
answer sheets from a template image plus a CSV of candidate data. Used by
`webui/services/prefill.py`.

## Sheet preset directories

See [`SHEETS.md`](SHEETS.md) for the full preset/variant table. Quick map:

| Directory | Status |
|---|---|
| `MoE-April-2026-Landscape-NNQ25-0` | **Production** (default preset) |
| `MoE-April-2026-Portrait-NNQ25-0` | Archive (retired NN portrait) |
| `MoE-May-2026-Variants-SMQ25-0` | Variants harness + `v1_legacy` portrait body |
| `MoE-May-2026-Portrait-SMQ25-1` | `v2_optimized` portrait body (hidden from picker) |

## `scripts/`

Reorganised on 2026-05-28 into four buckets:

| Bucket | Contents | Run as |
|---|---|---|
| `scripts/` (root) | `add_av_exclusion.ps1` — antivirus exclusion helper | `.\scripts\add_av_exclusion.ps1` |
| `scripts/bench/` | `bench_marker_robustness.py`, `bench_moderate_realism.py`, `bench_pdf_split.py`, `compare_marker_bench.py` | `python scripts/bench/<name>.py` |
| `scripts/dev/` | Live smoke tests + verifications: `make_test_pdf.py`, `probe_edge_cases.py`, `smoke_test_pdf_upload.py`, `stress_test.py`, `stress_test_parallel_split.py`, `verify_inmemory_fewer_writes.py`, `verify_moderate_e2e.py` | `python scripts/dev/<name>.py` |
| `scripts/diagnostics/` | Calibration / alignment debugging: `calibrate_bubbles.py`, `debug_alignment.py`, `debug_realism_samples.py`, `gen_alignment_samples.py`, `omr_stress_test.py` | `python scripts/diagnostics/<name>.py` |
| `scripts/archive/` | Completed one-shot migrations: `apply_sheet_rename_refs_20260525.py`, `apply_path_reorg_refs_20260528.py`, `add_defender_exclusion.ps1` (shim) | Kept for traceability; do not re-run without reason |

## `benchmarks/`

Output directory for benchmark JSONs. The committed baselines under
`benchmarks/results/` (`bench_baseline.json`, `bench_refine.json`,
`bench_realism_*.json`, `robustness_*.json`) are the regression line for the
marker-robustness + throughput work. Future benchmark runs append here; the
gitignore intentionally excludes new run subfolders so only deliberately
committed baselines persist.

## `docs/`

Reorganised on 2026-05-28 into four buckets:

| Bucket | Contents |
|---|---|
| `docs/architecture/` | `ARCHITECTURE.md`, `NERS_OMR_Workflow.excalidraw`, `NERS_OMR_Workflow_preview.svg` |
| `docs/audits/` | Dated reports: `audit_report_20260524.md`, `marker_robustness_benchmark_20260523.md`, `student_fill_e2e_report_20260524.md` |
| `docs/features/` | Per-feature design docs (e.g. `features/student-fill/student_fill_feature_design.md`) |
| `docs/research/` | Long-form research briefs (`research_brief_omr_throughput_2026.md`, `research_brief_scan_simulation_2026.md`) |
| `docs/assets/` | Embedded images |

## Runtime / build dirs (gitignored)

| Directory | Purpose | Lifetime |
|---|---|---|
| `inputs/` | CLI default input drop zone | Per-run |
| `outputs/` | CLI default output drop zone | Per-run |
| `logs/` | Runtime log files | Per-run |
| `webui/storage/` | Web-UI batches (user data) | Persistent on disk only |
| `build/`, `dist/` | PyInstaller artefacts | Build-time |
| `.venv/`, `.pytest_cache/`, `__pycache__/` | Local tooling caches | Local-only |

## Where to add new things

| Adding... | Put it in |
|---|---|
| A new answer sheet variant | New `MoE-<Month>-<Year>-<Orientation>-<Initials>Q25-<Rev>/` + register in `webui/sheet_registry.py` |
| A new one-off measurement script | `scripts/bench/` |
| A new live smoke / verification | `scripts/dev/` |
| A new alignment / template debug tool | `scripts/diagnostics/` |
| A new web route | `webui/api.py` + service in `webui/services/` |
| A new pydantic setting | `webui/settings.py` + `webui/schemas_settings.py` |
| A dated audit/benchmark report | `docs/audits/<topic>_<YYYYMMDD>.md` |
| A feature design doc | `docs/features/<feature>/...` |
| A research brief | `docs/research/research_brief_<topic>_<YYYY>.md` |
| A baseline benchmark JSON to commit | `benchmarks/results/` |

## Cleanup that has already been done

* `__pycache__/` and `.pyc` files outside `.venv` purged (2026-05-28)
* Tracked scratch removed: `6dd617c31d59_Results_02PM.csv`, `build_log.txt`,
  `build_output.txt`, root `omr_marker.jpg` (duplicate), `NERS_OMR_Workflow.png`
  (regenerable), 92-byte `package-lock.json` placeholder
* Untracked scratch removed: `benchmark_1000.py`, `run_5000_benchmark.py`,
  `extract_5000.py`, `extract_pdf.py`, `test_output.log`,
  `prefilled_sheets_moderate (1).pdf`
* `.gitignore` rewritten with the new directory names + retained legacy
  patterns so stale checkouts stay clean
