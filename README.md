# OMR Checker

Read OMR sheets fast and accurately using a scanner 🖨 or your phone 🤳.

## What is OMR?

OMR stands for Optical Mark Recognition, used to detect and interpret human-marked data on documents. OMR refers to the process of reading and evaluating OMR sheets, commonly used in exams, surveys, and other forms.

## What this fork adds

This repository is a fork of [Udayraj123/OMRChecker](https://github.com/Udayraj123/OMRChecker). The upstream engine handles the hard work of image alignment, bubble detection, and CSV output. On top of that foundation this fork adds:

- **FastAPI web UI + JSON API** — a full-featured browser interface and REST API for creating batches, uploading scans, editing templates, running OMR, and downloading results (see [Web UI and JSON API](#web-ui-and-json-api)).
- **Prefill Sheets workflow** — generate personalised pre-filled answer sheets from a blank landscape template using a calibrated 25-question OMR template (`MoE-April-2026-Landscape-NNQ25-0/`) and a standalone prefill library (`prefill_package/`).
- **Student-style bubble fill** — simulate how real students mark bubbles, with ten distinct marking profiles and flexible answer-key shortcuts, enabling end-to-end pipeline testing without manual scanning (see [Student-style bubble fill](#student-style-bubble-fill)).
- **High-throughput pipelined processing** — per-image dimension inference and background-task processing for large batch runs without blocking (see [`docs/research/research_brief_omr_throughput_2026.md`](docs/research/research_brief_omr_throughput_2026.md)).
- **Scan simulation** — realistic degradation and scan-artefact injection for synthetic test datasets (see [`docs/research/research_brief_scan_simulation_2026.md`](docs/research/research_brief_scan_simulation_2026.md)).
- **Codebase audit** — a documented review of the codebase with tracked fixes (see [`docs/audits/audit_report_20260524.md`](docs/audits/audit_report_20260524.md)).

#### **Quick Links**

- [Installation](#getting-started)
- [Repository map](REPO_LAYOUT.md) — every top-level directory explained
- [Sheet directory index](SHEETS.md) — canonical preset names + legacy aliases
- [Architecture overview](docs/architecture/ARCHITECTURE.md) — preset / variant / template model
- [User Guide](https://github.com/Udayraj123/OMRChecker/wiki)
- [Contributor Guide](https://github.com/Udayraj123/OMRChecker/blob/master/CONTRIBUTING.md)
- [Project Ideas List](https://github.com/users/Udayraj123/projects/2/views/1)

<hr />

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/Udayraj123/OMRChecker/pull/new/master) <!-- [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/Udayraj123/OMRChecker/wiki/TODOs) -->
[![GitHub pull-requests closed](https://img.shields.io/github/issues-pr-closed/Udayraj123/OMRChecker.svg)](https://github.com/Udayraj123/OMRChecker/pulls?q=is%3Aclosed)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/Udayraj123/OMRChecker.svg)](https://GitHub.com/Udayraj123/OMRChecker/issues?q=is%3Aissue+is%3Aclosed)
[![Ask me](https://img.shields.io/badge/Discuss-on_Github-purple.svg?style=flat-square)](https://github.com/Udayraj123/OMRChecker/issues/5)

<!-- [![GitHub contributors](https://img.shields.io/github/contributors/Udayraj123/OMRChecker.svg)](https://GitHub.com/Udayraj123/OMRChecker/graphs/contributors/) -->

[![GitHub stars](https://img.shields.io/github/stars/Udayraj123/OMRChecker.svg?style=social&label=Stars✯)](https://GitHub.com/Udayraj123/OMRChecker/stargazers/)
[![Join](https://img.shields.io/badge/Join-Discord_group-purple.svg?style=flat-square)](https://discord.gg/qFv2Vqf)

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/Udayraj123/a125b1531c61cceed5f06994329cba66/omrchecker-on-cloud.ipynb) -->

## 🎯 Features

A full-fledged OMR checking software that can read and evaluate OMR sheets scanned at any angle and having any color.

| Specs <img width=200/> | ![Current_Speed](https://img.shields.io/badge/Speed-200+_OMRs/min-blue.svg?style=flat-square) ![Min Resolution](https://img.shields.io/badge/Min_Resolution-640x480-blue.svg?style=flat-square) <img width=200/> |
| :--------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 💯 **Accurate**        | Currently nearly 100% accurate on good quality document scans; and about 90% accurate on mobile images.                                                                                                          |
| 💪🏿 **Robust**          | Supports low resolution, xeroxed sheets. See [**Robustness**](https://github.com/Udayraj123/OMRChecker/wiki/Robustness) for more.                                                                                |
| ⏩ **Fast**            | Current processing speed without any optimization is 200 OMRs/minute.                                                                                                                                            |
| ✅ **Customizable**    | [Easily apply](https://github.com/Udayraj123/OMRChecker/wiki/User-Guide) to custom OMR layouts, surveys, etc.                                                                                                    |
| 📊 **Visually Rich**   | [Get insights](https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals) to configure and debug easily.                                                                                                        |
| 🎈 **Lightweight**     | Very minimal core code size.                                                                                                                                                                                     |
| 🏫 **Large Scale**     | Tested on a large scale at [Technothlon](https://en.wikipedia.org/wiki/Technothlon).                                                                                                                             |
| 👩🏿‍💻 **Dev Friendly**    | [Pylinted](http://pylint.pycqa.org/) and [Black formatted](https://github.com/psf/black) code. Also has a [developer community](https://discord.gg/qFv2Vqf) on discord.                                          |

Note: For solving interesting challenges, developers can check out [**TODOs**](https://github.com/Udayraj123/OMRChecker/wiki/TODOs).

See the complete guide and details at [Project Wiki](https://github.com/Udayraj123/OMRChecker/wiki/).

<!-- 💁🏿‍♂️ **User Friendly** - WIP, Help by contributing! -->

## 💡 What can OMRChecker do for me?

Once you configure the OMR layout, just throw images of the sheets at the software; and you'll get back the marked responses in an excel sheet!

Images can be taken from various angles as shown below-

<p align="center">
	<img alt="sample_input" width="400" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/sample_input.PNG">
</p>

### Code in action on images taken by scanner:

<p align="center">
	<img alt="document_scanner" height="300" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/mini_scripts/outputs/gif/document_scanner.gif">

</p>

### Code in action on images taken by a mobile phone:

<p align="center">
	<img alt="checking_xeroxed_mobile" height="300" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/mini_scripts/outputs/gif/checking_xeroxed_mobile.gif">
</p>

## Visuals

### Processing steps

See step-by-step processing of any OMR sheet:

<p align="center">
	<a href="https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals">
		<img alt="rotation_stack" width="650" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/rotation.PNG">
	</a>
	<br>
	*Note: This image is generated by the code itself!*
</p>

### Output

Get a CSV sheet containing the detected responses and evaluated scores:

<p align="center">
	<a href="https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals">
		<img alt="csv_output" width="550" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/csv_output.PNG">
	</a>
</p>

We now support [colored outputs](https://github.com/Udayraj123/OMRChecker/wiki/%5Bv2%5D-About-Evaluation) as well. Here's a sample output on another image -
<p align="center">
	<a href="https://github.com/Udayraj123/OMRChecker/wiki/%5Bv2%5D-About-Evaluation">
		<img alt="colored_output" width="550" src="./docs/assets/colored_output.jpg">
	</a>
</p>

#### There are many more visuals in the wiki. Check them out [here!](https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals)

## Getting started

![Setup Time](https://img.shields.io/badge/Setup_Time-20_min-blue.svg)

**Operating system:** OSX or Linux is recommended although Windows is also supported.

### 1. Install global dependencies

![opencv-python ≥4.8.0](https://img.shields.io/badge/opencv--python-%E2%89%A54.8.0-blue.svg) ![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

To check if python3 and pip is already installed:

```bash
python3 --version
python3 -m pip --version
```

<details>
	<summary><b>Install Python3</b></summary>

To install python3 follow instructions [here](https://www.python.org/downloads/)

To install pip - follow instructions [here](https://pip.pypa.io/en/stable/installation/)

</details>
<details>
<summary><b>Install OpenCV</b></summary>

**Any installation method is fine.**

Recommended:

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user opencv-python
python3 -m pip install --user opencv-contrib-python
```

More details on pip install openCV [here](https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/).

</details>

<details>

<summary><b>Extra steps(for Linux users only)</b></summary>

<b>Installing missing libraries(if any):</b>

On a fresh computer, some of the libraries may get missing in event after a successful pip install. Install them using following commands[(ref)](https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/):

```bash
sudo apt-get install -y build-essential cmake unzip pkg-config
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libatlas-base-dev gfortran
```

</details>

### 2. Install project dependencies

Clone the repo

```bash
git clone https://github.com/Udayraj123/OMRChecker
cd OMRChecker/
```

Install pip requirements

```bash
python3 -m pip install --user -r requirements.txt
```

_**Note:** If you face a distutils error in pip, use `--ignore-installed` flag in above command._

<!-- Wiki should not get cloned -->

### 3. Run the code

1. First copy and examine the sample data to know how to structure your inputs:
   ```bash
   cp -r ./samples/sample1 inputs/
   # Note: you may remove previous inputs (if any) with `mv inputs/* ~/.trash`
   # Change the number N in sampleN to see more examples
   ```
2. Run OMRChecker:
   ```bash
   python3 main.py
   ```

Alternatively you can also use `python3 main.py -i ./samples/sample1`.

Each example in the samples folder demonstrates different ways in which OMRChecker can be used.

## Web UI and JSON API

A thin FastAPI wrapper around the same engine is available for batch-style workflows. It serves both a minimal HTML UI and a JSON API from a single process.

Start it with:

```bash
uvicorn webui.app:create_app --factory --reload
# or choose a port
uvicorn webui.app:create_app --factory --reload --port 5050
```

Then open:

- `http://127.0.0.1:8000/` – HTML UI for creating batches, uploading scans, editing `template.json` / `config.json` / `evaluation.json`, running OMR, and reviewing results.
- `http://127.0.0.1:8000/prefill` – Pre-fill answer sheets with student details (see [Prefill Sheets](#prefill-sheets) below).
- `http://127.0.0.1:8000/docs` – auto-generated Swagger UI for the `/api/v1/*` endpoints.
- `http://127.0.0.1:8000/openapi.json` – OpenAPI schema for machine consumption.

Key UI features:

- **Batch workflow**: create a batch, upload/import images or PDFs, attach JSON docs, run OMR, download results.
- **PDF preprocessing**: uploaded/imported PDFs are split into one PNG image per page before OMR processing, so every page appears as an input sheet in the batch.
- **Template/config/evaluation editor**: toggle between:
  - **Code mode** (raw JSON)
  - **UI mode** (form controls like sliders, tag/chip editors, repeatable cards, and ordered lists)
  - Switching between the two preserves edits; saving writes the same JSON back to disk.
- **Image previews**:
  - input sheets are viewable directly on the batch page (large preview)
  - template assets (e.g. `omr_marker.jpg`) are previewable alongside the asset list
- **Results review**:
  - per-`file_id` view for assessing one sheet at a time
  - a raw CSV table view contained in a horizontal scroller (no page overflow)
  - flags likely multi-mark/erasure cases as a review heuristic when a question output contains multiple options (e.g. `AB`)

Each batch is stored as a self-contained directory under `webui/storage/batches/<id>/`, mirroring the layout the CLI already expects (`inputs/`, `outputs/`, `template.json`, `config.json`, `evaluation.json`). That means a batch created via the UI can still be processed with `python main.py -i <batch>/inputs -o <batch>/outputs` as a fallback.

Configuration lives in `webui/settings.py` and can be overridden with environment variables (prefix `OMR_WEBUI_`) or a `.env` file. Key flags:

- `OMR_WEBUI_STORAGE_ROOT` – where batches are stored.
- `OMR_WEBUI_ALLOW_DIRECTORY_IMPORT` – enables the "import from a server-side directory" endpoint and UI control. Keep this on for local use; turn it off before hosting to avoid arbitrary filesystem reads.
- `OMR_WEBUI_CORS_ORIGINS` – comma-free JSON list of CORS allow-origins for external frontends.

Useful API endpoints (in addition to `/api/v1/batches/*`):

- Preview an input image: `/api/v1/batches/{batch_id}/files/{filename}/preview`
- Preview a template asset image: `/api/v1/batches/{batch_id}/assets/{filename}/preview`

For local-directory batch marking, the intended flow is:

1. Create a batch.
2. Use "Import from directory" to pull all supported images/PDFs from a local folder into the batch `inputs/` directory. PDFs are rendered into page images automatically.
3. Save or upload `template.json` and, optionally, `config.json` / `evaluation.json`.
4. Run OMR once for the batch.

During web/API processing, OMRChecker now computes dimensions per image instead of assuming one static size for the entire batch:

- `processing_width` / `processing_height` are derived from the actual image size and clamped to the engine's default processing bounds while preserving aspect ratio.
- `display_width` / `display_height` are derived the same way for consistency.
- The runtime config always forces `outputs.show_image_level = 0` so background processing never blocks on OpenCV windows.

Because `config.json` is schema-validated, only the latest computed `dimensions` block is written back to the batch `config.json`. Per-file autosizing details are stored in `metadata.json` and are surfaced by `GET /api/v1/batches/{id}/status` as `latest_dynamic_dimensions`, `latest_processed_file`, `processed_files`, and `total_files`.

This means:

- multiple sheets from a local directory can be processed in one batch without manually tuning dimensions for each file
- the batch `config.json` remains valid for future runs
- API clients can poll status and see what dimensions were used most recently

Processing is handled in-process via FastAPI `BackgroundTasks`, which is fine for local use. For hosted deployments, keep the service layer as-is and replace the task handoff with a real queue (RQ, Arq, or Celery with Redis); no engine changes required.

Tests for the web UI live in `webui/tests/` and run as part of the standard `pytest` invocation.

## Prefill Sheets

The `/prefill` page generates pre-filled answer sheets from the blank landscape template, bubbling in a student's candidate number and printing their name, school, and exam name into the text boxes. It is useful for issuing personalised sheets before an exam.

The feature lives in `prefill_package/` and is exposed through the web UI and the JSON API.

**Web UI** — navigate to `http://127.0.0.1:8000/prefill`:

| Mode | Input | Output |
| ---- | ----- | ------ |
| **Single Sheet** | Form fields for name, school, exam, candidate number | Download PNG or PDF |
| **Batch** | Upload a CSV file *or* enter rows in the inline table | Download combined PDF or ZIP of PNGs |

**JSON API** endpoints (under `/api/v1/prefill/`):

- `POST /api/v1/prefill/single` — form fields: `student_name`, `school_name`, `exam_name`, `candidate_number`, `output_format` (`png` or `pdf`). Returns a streaming download.
- `POST /api/v1/prefill/batch` — form fields: `csv_text` or `csv_file`, `output_mode` (`pdf` or `zip`). Returns a streaming download.

**CSV format** (required columns: `student_name`, `school_name`, `exam_name`, `candidate_number`; optional: `output_file`):

```csv
student_name,school_name,exam_name,candidate_number,output_file
Johnathan Ragnauth Brigmohan,The New Sapodilla Primary,Grade 4 Reading,9010690012,johnathan.png
```

Candidate numbers must be exactly 10 digits. Bubble placement is calibrated to `prefill_package/blank_template_reference.png` — do not swap in a different template without re-calibrating the `CFG` values in `prefill_answer_sheet_final.py`.

## Student-style bubble fill

The student-fill feature lets you simulate how real students mark bubbles on answer sheets, producing realistic synthetic scans for end-to-end pipeline testing — no physical scanning required.

### Marking profiles

Ten profiles control how each bubble is drawn:

| Profile | Description |
| --- | --- |
| `none` | Leave bubbles blank (useful for blank-sheet generation) |
| `light_pencil` | Faint grey fill, as if marked with a soft pencil stroke |
| `medium_pencil` | Standard HB-pencil darkness |
| `heavy_pencil` | Dark, heavy pencil fill |
| `pen_ballpoint` | Solid, dark ballpoint-pen fill |
| `check_mark` | A ✓ drawn inside the bubble |
| `cross_mark` | An ✗ drawn inside the bubble |
| `partial_fill` | Only part of the bubble is filled |
| `messy_student` | Irregular, smudged fill with slight overruns |
| `careful_student` | Neat, well-centred fill |

### Answer-key formats

**Shortcut strings**

| Shortcut | Meaning |
| --- | --- |
| `all_a` / `all_b` / `all_c` / `all_d` | All 25 questions answered with that option |
| `alternating` | A, B, A, B, … |
| `random` | Deterministically random per candidate number |
| `random_with_skips` | Like `random` but some questions left blank |
| `blank` | All questions skipped |
| 25-letter string e.g. `ABCD-ABCD-ABCDA-BCDAB-CDABC` | Explicit per-question answers; `-` means skip |

**JSON object**

```json
{"q1": "A", "q3": "BC"}
```

Keys are question names; values are one or more option letters (multi-mark is supported). Questions not listed are left blank.

### Per-row CSV override

When running a batch, each CSV row may include an `answers` or `answers_json` column to override the answer spec for that individual candidate:

```csv
student_name,school_name,exam_name,candidate_number,answers
Jane Smith,Riverview Primary,Grade 5 Maths,1234567890,ABCDABCDABCDABCDABCDABCDABC
```

### Determinism

Fills are **deterministic** per `(candidate_number, answer_spec)` pair — the same inputs always produce identical output. This makes the feature well-suited for regression testing.

### API surface

```
GET  /api/v1/prefill/marking-profiles   — list available profile names
POST /api/v1/prefill/single             — single sheet (add marking_profile + answers form fields)
POST /api/v1/prefill/batch              — batch (add marking_profile + answers column in CSV)
```

Pass `marking_profile=heavy_pencil` (or any profile name from the list endpoint) and `answers=all_a` (or any format described above) as additional form fields alongside the existing prefill parameters.

### Further reading

- [`docs/features/student-fill/student_fill_feature_design.md`](docs/features/student-fill/student_fill_feature_design.md) — full feature specification
- [`docs/audits/student_fill_e2e_report_20260524.md`](docs/audits/student_fill_e2e_report_20260524.md) — end-to-end test report

### Common Issues

<details>
<summary>
	1. [Windows] ERROR: Could not open requirements file<br>
	</summary>
Command: <code>python3 -m pip install --user -r requirements.txt</code>
<br>
	Link to Solution:  <a href="https://github.com/Udayraj123/OMRChecker/issues/54#issuecomment-1264569006">#54</a>
</details>
<details>
<summary>
2. [Linux] ERROR: No module named pip<br>
</summary>
Command: <code>python3 -m pip install --user --upgrade pip</code>
<br>
	Link to Solution: <a href="https://github.com/Udayraj123/OMRChecker/issues/70#issuecomment-1268094136">#70</a>
</details>

## OMRChecker for custom OMR Sheets

1. First, [create your own template.json](https://github.com/Udayraj123/OMRChecker/wiki/User-Guide).
2. Configure the tuning parameters.
3. Run OMRChecker with appropriate arguments (See full usage).
<!-- 4. Add answer key( TODO: add answer key/marking scheme guide)  -->

## Full Usage

```
python3 main.py [--setLayout] [--inputDir dir1] [--outputDir dir1]
```

Explanation for the arguments:

`--setLayout`: Set up OMR template layout - modify your json file and run again until the template is set.

`--inputDir`: Specify an input directory.

`--outputDir`: Specify an output directory.

<details>
<summary>
 <b>Deprecation logs</b>
</summary>

- The old `--noCropping` flag has been replaced with the 'CropPage' plugin in "preProcessors" of the template.json(see [samples](https://github.com/Udayraj123/OMRChecker/tree/master/samples)).
- The `--autoAlign` flag is deprecated due to low performance on a generic OMR sheet
- The `--template` flag is deprecated and instead it's recommended to keep the template file at the parent folder containing folders of different images
</details>

<!-- #### Testing the code
Datasets to test on :
Low Quality Dataset(For CV Based methods)) (1.5 GB)
Standard Quality Dataset(For ML Based methods) (3 GB)
High Quality Dataset(For custom processing) (6 GB)
-->

## Repository layout

```
OMRCheckerMoE/
├── src/                          # Core OMR engine (image processing, template parsing, evaluation)
│   └── tests/                    # Unit tests for the engine
├── webui/                        # FastAPI service + Jinja templates + JS for the web UI
│   └── tests/                    # Web UI test suite (runs as part of pytest)
├── prefill_package/         # Standalone answer-sheet prefill module (used as a library by webui too)
├── MoE-April-2026-Landscape-NNQ25-0/   # Current calibrated 25-question OMR template + sample inputs
├── MoE-April-2026-Portrait-NNQ25-0/ # Legacy 25Q template variant (kept for reference)
├── samples/                      # Example sheets and templates for the core engine
├── scripts/                      # One-off tools, benchmarks, smoke tests, and bubble-geometry calibration helpers
└── docs/                         # Markdown reports, design docs, and research briefs
```

## Development

**Runtime dependencies**

```bash
python -m pip install -r requirements.txt
```

**Dev / test dependencies**

```bash
python -m pip install -r requirements.dev.txt
```

**Running the test suite**

```bash
pytest
```

`pytest.ini` is configured to discover tests in both `src/tests/` and `webui/tests/`. The test `webui/tests/test_student_fill_omr_roundtrip.py` exercises the full prefill → OMR → CSV roundtrip.

**Pre-commit hooks**

```bash
pre-commit install
```

Hooks are defined in `.pre-commit-config.yaml` and run Black, isort, and other linters automatically before each commit.

## Additional documentation

| Document | Description |
| --- | --- |
| [`docs/audits/audit_report_20260524.md`](docs/audits/audit_report_20260524.md) | Codebase audit and documented fixes |
| [`docs/features/student-fill/student_fill_feature_design.md`](docs/features/student-fill/student_fill_feature_design.md) | Student-fill feature specification |
| [`docs/audits/student_fill_e2e_report_20260524.md`](docs/audits/student_fill_e2e_report_20260524.md) | End-to-end validation report for student-fill |
| [`docs/research/research_brief_omr_throughput_2026.md`](docs/research/research_brief_omr_throughput_2026.md) | High-throughput OMR pipeline research |
| [`docs/research/research_brief_scan_simulation_2026.md`](docs/research/research_brief_scan_simulation_2026.md) | Scan simulation research |
| [`docs/audits/marker_robustness_benchmark_20260523.md`](docs/audits/marker_robustness_benchmark_20260523.md) | ArUco marker robustness benchmark |

## FAQ

<details>
<summary>
<b>Why is this software free?</b>
</summary>

This project was born out of a student-led organization called as [Technothlon](https://technothlon.techniche.org.in). It is a logic-based international school championship organized by students of IIT Guwahati. Being a non-profit organization, and after seeing it work fabulously at such a large scale we decided to share this tool with the world. The OMR checking processes still involves so much tediousness which we aim to reduce dramatically.

We believe in the power of open source! Currently, OMRChecker is in an intermediate stage where only developers can use it. We hope to see it become more user-friendly as well as robust from exposure to different inputs from you all!

[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

</details>

<details>
<summary>
<b>Can I use this code in my (public) work?</b>
</summary>

OMRChecker can be forked and modified. You are encouraged to play with it and we would love to see your own projects in action!

It is published under the [MIT license](https://github.com/Udayraj123/OMRChecker/blob/master/LICENSE).

</details>

<details>
<summary>
<b>What are the ways to contribute?</b>
</summary>

<!-- - Help OMRChecker reach more people by giving a star! The Goal is to reach top position for the [OMR Topic](https://github.com/topics/omr) -->

- Join the developer community on [Discord](https://discord.gg/qFv2Vqf) to fix [issues](https://github.com/Udayraj123/OMRChecker/issues) with OMRChecker.

- If this project saved you large costs on OMR Software licenses, or saved efforts to make one. Consider donating an amount of your choice(donate section).

<!-- ![☕](https://miro.medium.com/fit/c/256/256/1*br7aoq_JVfxeg73x5tF_Sw.png) -->
<!-- [![paypal.me](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_SM.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=Z5BNNK7AVFVH8&source=url) -->
<!-- https://www.amazon.in/hz/wishlist/ls/3V0TDQBI3T8IL -->

</details>

## Credits

_A Huge thanks to:_
_**Adrian Rosebrock** for his exemplary blog:_ https://pyimagesearch.com

_**Harrison Kinsley** aka sentdex for his [video tutorials](https://www.youtube.com/watch?v=Z78zbnLlPUA&list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq) and many other resources._

_**Satya Mallic** for his resourceful blog:_ https://www.learnopencv.com

_And to other amazing people from all over the globe who've made significant improvements in this project._

_Thank you!_

<!--
OpencV
matplotlib
some SO answers from roughworks
prof
-->

## Related Projects

Here's a snapshot of the [Android OMR Helper App (archived)](https://github.com/Udayraj123/AndroidOMRHelper):

<p align="center">
	<a href="https://github.com/Udayraj123/AndroidOMRHelper">
		<img height="300" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/app_flow.PNG">
	</a>
</p>

## Stargazers over time

[![Stargazers over time](https://starchart.cc/Udayraj123/OMRChecker.svg)](https://starchart.cc/Udayraj123/OMRChecker)

---

<h2 align="center">Made with ❤️ by Awesome Contributors</h2>

<a href="https://github.com/Udayraj123/OMRChecker/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Udayraj123/OMRChecker" />
</a>

---

### License

[![GitHub license](https://img.shields.io/github/license/Udayraj123/OMRChecker.svg)](https://github.com/Udayraj123/OMRChecker/blob/master/LICENSE)

For more details see [LICENSE](https://github.com/Udayraj123/OMRChecker/blob/master/LICENSE).

### Donate

<a href="https://www.buymeacoffee.com/Udayraj123" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a> [![paypal](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_LG.gif)](https://www.paypal.me/Udayraj123/500)

_Find OMRChecker on_ [**_Product Hunt_**](https://www.producthunt.com/posts/omr-checker/) **|** [**_Reddit_**](https://www.reddit.com/r/computervision/comments/ccbj6f/omrchecker_grade_exams_using_python_and_opencv/) **|** [**Discord**](https://discord.gg/qFv2Vqf) **|** [**Linkedin**](https://www.linkedin.com/pulse/open-source-talks-udayraj-udayraj-deshmukh/) **|** [**goodfirstissue.dev**](https://goodfirstissue.dev/language/python) **|** [**codepeak.tech**](https://www.codepeak.tech/) **|** [**fossoverflow.dev**](https://fossoverflow.dev/projects) **|** [**Interview on Console by CodeSee**](https://console.substack.com/p/console-140) **|** [**Open Source Hub**](https://opensourcehub.io/udayraj123/omrchecker)

 <!-- [***Hacker News***](https://news.ycombinator.com/item?id=20420602) **|** -->
 <!-- **|** [***Swyya***](https://www.swyya.com/projects/omrchecker) -->
