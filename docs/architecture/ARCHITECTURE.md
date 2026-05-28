# OMRCheckerMoE — Architecture Index

Executive summary: the app has **three sheet-related concepts** — (1)
**preset directories** on disk (`template.json` + friends), (2) a **logical
portrait preset** whose physical folder is chosen by `sheet_variant`, and
(3) the **prefill package** which renders filled sheets from a separate
template image. Scanning always needs a `template.json`; prefilling uses
`prefill_package` and may auto-attach the built-in landscape template on
upload. Canonical directory names are documented in [`../SHEETS.md`](../SHEETS.md).

## 1. Preset vs sheet variant vs template

| Term | Meaning | Defined in |
|---|---|---|
| **Preset** | A repo-root folder with `template.json` (optional `config.json`) | Discovered by `webui/services/presets.py` → `list_presets()` |
| **Sheet variant** | Runtime switch (`v1_legacy` / `v2_optimized`) for the logical portrait preset | `webui/settings.py` → `sheet_variant` |
| **Template** | The JSON field map consumed by the OMR engine (`src/template.py`) | Each preset's `template.json` |

They are distinct: a batch stores a **preset name** (string). Only
`MoE-May-2026-Portrait-SMQ25` honours `sheet_variant`; all other presets
map 1:1 to their directory name.

## 2. How the system picks a preset

1. **New batch** — `default_preset` in settings (currently
   `MoE-April-2026-Landscape-NNQ25-0`) is applied via
   `presets.apply_preset_to_batch` when configured.
2. **Operator override** — UI or API applies any name from `list_presets()`.
3. **Prefill uploads** — filenames containing `prefilled_sheet` trigger
   `_attach_prefilled_25q_defaults` in `webui/api.py` (embedded template,
   not disk preset).
4. **Blank-sheet print** — `/api/v1/prefill/blank` uses
   `BLANK_SHEET_VARIANTS` in `webui/services/prefill.py` (landscape PDF).

Legacy preset strings are normalised in `webui/sheet_registry.py` →
`normalize_preset_name()`.

## 3. Files the engine consumes

| File | Loader |
|---|---|
| `template.json` | `src/template.py` — `Template` class |
| `config.json` | `src/defaults/config.py` merged in entry path |
| Marker / sheet images | Referenced by template preProcessors; copied with preset |

Preset files are copied into the batch directory by
`presets.apply_preset_to_batch()` before processing.

## 4. Directory classification

| Directory | Type | Role |
|---|---|---|
| `MoE-April-2026-Landscape-NNQ25-0` | **Preset (production)** | Default scan + blank print |
| `MoE-April-2026-Portrait-NNQ25-0` | Preset (archive) | Retired NN portrait |
| `MoE-May-2026-Variants-SMQ25-0` | Preset + tooling | v1 portrait + sweep harness |
| `MoE-May-2026-Portrait-SMQ25-1` | Preset (hidden) | v2 optimised portrait |
| `prefill_package` | **Python package** | Landscape prefill renderer |

## 5. Canonical flows

### (a) Prefill via `/prefill`

`prefill.html` → `POST /api/v1/prefill/single|batch|blank`
→ `webui/services/prefill.py` → `prefill_package.prefill_answer_sheet_final`
→ download token → operator prints.

### (b) Scan uploaded sheets

Upload → batch storage → (optional) auto template attach →
`POST /api/v1/batches/{id}/process` → `webui/services/omr.py` →
`src/entry.py` → `Template` + `CropOnMarkers` → results CSV.

## 6. Preset name → directory resolution

```
normalize_preset_name(name)     # webui/sheet_registry.py
        ↓
_resolve_preset_dir(settings, name)   # webui/services/presets.py
        ↓
  variant-aware? → _VARIANT_ROUTES[sheet_variant]
  else           → presets_dir / name
```

## 7. User-visible strings (frontend)

| Location | Content |
|---|---|
| `webui/templates/prefill.html` | Blank sheet variant labels |
| `webui/templates/settings.html` | `sheet_variant`, `default_preset` |
| `webui/templates/generate_csv.html` | Hints referencing preset names |

Update labels when adding presets; paths live only in `sheet_registry.py`.
