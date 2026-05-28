# MoE Answer Sheet Index

Canonical names for every answer-sheet bundle in this repository. Use these
directory names in settings, presets, and documentation — **not** the legacy
folder names from before May 2026.

## Quick reference

| Canonical directory | Role | Replaces (legacy) |
|---|---|---|
| `MoE-April-2026-Landscape-NNQ25-0` | **Production** landscape 25Q sheet (default preset) | `custom_25_definitive_final` |
| `MoE-April-2026-Portrait-NNQ25-0` | Retired portrait NN layout (archive) | `old_custom25_answer_sheet_v1` |
| `MoE-May-2026-Variants-SMQ25-0` | Portrait variants / sweep tooling (`v1_legacy`) | `portrait_25q` |
| `MoE-May-2026-Portrait-SMQ25-1` | Optimised portrait layout (`v2_optimized`) | `portrait_25q_v2` |
| `prefill_package/` | Prefill renderer (Python package) | `prefill_only_package` |

**Naming pattern:** `MoE-<Month>-<Year>-<Orientation>-<Initials>Q25-<Revision>`

- **NN** — Nkasi Nedd (manager / original landscape author)
- **SM** — Samir Mohammed sheet maintainer initials for the May 2026 portrait programme
- **Q25** — 25-question multiple-choice layout
- **0 / 1** — revision index within that programme

## Logical preset: portrait variant router

The setting **Sheet variant** (`sheet_variant` on `/settings`) toggles which
physical folder backs the logical preset:

| Logical preset (UI) | `v1_legacy` → | `v2_optimized` → |
|---|---|---|
| `MoE-May-2026-Portrait-SMQ25` | `MoE-May-2026-Variants-SMQ25-0` | `MoE-May-2026-Portrait-SMQ25-1` |

The two physical portrait directories are **hidden** from the preset picker;
only the logical name appears. Legacy alias `portrait_25q` still resolves.

## What each directory contains

### `MoE-April-2026-Landscape-NNQ25-0` (default)

- `template.json` — OMR field map (CropOnMarkers / ArUco)
- `config.json` — processing dimensions
- `blank_legacy_landscape_answer_sheet_with_markers.pdf` — print blank (ArUco)
- `generate_blank_with_markers.py` — regenerate blank PDF
- `inputs/` — sample scans for tests

### `MoE-April-2026-Portrait-NNQ25-0`

- Legacy portrait NN sheet (superseded). Kept for regression comparisons.

### `MoE-May-2026-Variants-SMQ25-0`

- `template.json`, `DESIGN.md`, `generate_blank.py`
- `variants/sweep/` — bubble-size sweeps and materialisation scripts
- `reference/` — rendered PNG/PDF references

### `MoE-May-2026-Portrait-SMQ25-1`

- Sweep-winning optimised portrait preset (`template.json`, `config.json`)

### `prefill_package/`

- `prefill_answer_sheet_final.py` — landscape prefill renderer
- `blank_template_reference.png` — template image for legacy landscape prefill

## Code entry points

| Concern | Module |
|---|---|
| Canonical path constants | `webui/sheet_registry.py` |
| Preset list / apply / variant routing | `webui/services/presets.py` |
| Default preset | `webui/settings.py` → `default_preset` |
| Prefill + blank-sheet print | `webui/services/prefill.py` |
| Built-in prefill template for uploads | `webui/api.py` → `_PREFILLED_25Q_TEMPLATE` |

## Legacy aliases (automatic)

Old names in batch metadata, CSV docs, or operator muscle-memory still work
via `PRESET_LEGACY_ALIASES` in `webui/sheet_registry.py`:

| Legacy | Resolves to |
|---|---|
| `custom_25_definitive_final` | `MoE-April-2026-Landscape-NNQ25-0` |
| `old_custom25_answer_sheet_v1` | `MoE-April-2026-Portrait-NNQ25-0` |
| `portrait_25q` | `MoE-May-2026-Portrait-SMQ25` (logical) |
| `portrait_25q_v2` | `MoE-May-2026-Portrait-SMQ25-1` |

See [`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md) for end-to-end prefill and scan flows.
