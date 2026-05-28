# `docs/` — documentation index

Reorganised on **2026-05-28** into four buckets. New docs MUST live in one of
these; do not drop new markdown at `docs/` root.

| Bucket | Purpose | Examples |
|---|---|---|
| `architecture/` | Evergreen architectural reference + diagrams | `ARCHITECTURE.md`, `NERS_OMR_Workflow.excalidraw` |
| `audits/` | Dated audit + benchmark reports (snapshot in time) | `audit_report_20260524.md`, `marker_robustness_benchmark_20260523.md` |
| `features/<feature>/` | Per-feature design + e2e docs | `features/student-fill/student_fill_feature_design.md` |
| `research/` | Long-form research briefs that inform design decisions | `research_brief_omr_throughput_2026.md` |
| `assets/` | Images embedded by other docs | `colored_output.jpg` |

## Naming conventions

* **Dated reports**: `<topic>_<YYYYMMDD>.md` — e.g. `audit_report_20260524.md`
* **Research briefs**: `research_brief_<topic>_<YYYY>.md`
* **Feature designs**: `features/<feature>/<feature>_feature_design.md`
* **Architecture**: a single `ARCHITECTURE.md` + supporting diagrams

## Top-level entry points

* [`../REPO_LAYOUT.md`](../REPO_LAYOUT.md) — repository map
* [`../SHEETS.md`](../SHEETS.md) — sheet directory + preset index
* [`architecture/ARCHITECTURE.md`](architecture/ARCHITECTURE.md) — preset / variant / template model and end-to-end flows
