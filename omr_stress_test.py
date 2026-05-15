"""OMR robustness stress test - direct subprocess approach.

Generates adversarial scans from real prefilled sheets, then runs the OMR
engine directly as a subprocess with a hard per-test timeout.  No HTTP
server required, no polling loops that silently time-out.

Every step prints a timestamped log line so you can see exactly where things
hang or break.

Usage:
    python omr_stress_test.py              # default: 45 s per-test timeout
    OMR_STRESS_TIMEOUT=60 python omr_stress_test.py

Pre-reqs: custom_25_definitive_final/ present, .venv active.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Force UTF-8 output on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = REPO_ROOT / "custom_25_definitive_final"
PREFILL_SCRIPT = REPO_ROOT / "prefill_only_package" / "prefill_answer_sheet_final.py"
TEMPLATE_JSON = TEMPLATE_DIR / "template.json"
CONFIG_JSON = TEMPLATE_DIR / "config.json"
MARKER_ASSET = TEMPLATE_DIR / "omr_marker.jpg"
ARTIFACTS = REPO_ROOT / "omr_stress_artifacts"
ARTIFACTS.mkdir(exist_ok=True)

PYTHON = sys.executable
# Per-test hard timeout (seconds). OMR on one image should finish in <10s.
PER_TEST_TIMEOUT = int(os.environ.get("OMR_STRESS_TIMEOUT", "45"))

# Non-interactive config injected into every test dir
_NON_INTERACTIVE_CONFIG = {
    "dimensions": {"display_height": 515, "display_width": 666,
                   "processing_height": 515, "processing_width": 666},
    "outputs": {"show_image_level": 0},
}


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

_t0 = time.perf_counter()


def _ts() -> str:
    return f"[{time.perf_counter() - _t0:7.2f}]"


def log(msg: str) -> None:
    print(f"{_ts()} {msg}", flush=True)


# ---------------------------------------------------------------------------
# Step 1 - Generate a real prefilled sheet as the clean baseline
# ---------------------------------------------------------------------------

def generate_prefilled_baseline() -> np.ndarray:
    """Run the prefill CLI to produce one filled sheet PNG, load it with cv2."""
    out_path = ARTIFACTS / "prefilled_baseline.png"
    if out_path.exists():
        log(f"PREFILL  baseline already exists at {out_path.name}, reusing")
        img = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
        if img is not None:
            return img
        log("PREFILL  cached file unreadable, regenerating")
        out_path.unlink(missing_ok=True)

    log(f"PREFILL  generating baseline via {PREFILL_SCRIPT.name} ...")

    template_png = PREFILL_SCRIPT.parent / "blank_template_reference.png"
    if not template_png.exists():
        # Fall back to the existing filled sample scan
        fallback = TEMPLATE_DIR / "MCFormat25questions_page-0001_filled_sample_markers.jpg"
        if fallback.exists():
            log(f"PREFILL  blank_template_reference.png missing; using fallback: {fallback.name}")
            img = cv2.imread(str(fallback), cv2.IMREAD_COLOR)
            if img is not None:
                # Save a copy so we have a baseline PNG for later transforms
                cv2.imwrite(str(out_path), img)
                return img
        raise SystemExit("FATAL: no baseline image available")

    cmd = [
        PYTHON, str(PREFILL_SCRIPT),
        "--template", str(template_png),
        "--student-name", "Ahmed Al-Rashid",
        "--school-name", "NERS Test School",
        "--exam-name", "Maths Grade 10",
        "--candidate-number", "1234567890",
        "--output", str(out_path),
    ]
    log(f"PREFILL  cmd: {' '.join(str(c) for c in cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT))
    elapsed = time.perf_counter() - t0
    log(f"PREFILL  exit={proc.returncode} elapsed={elapsed:.1f}s")
    for line in (proc.stdout or "").strip().splitlines():
        log(f"PREFILL  stdout: {line}")
    for line in (proc.stderr or "").strip().splitlines()[-10:]:
        log(f"PREFILL  stderr: {line}")
    if proc.returncode != 0 or not out_path.exists():
        raise SystemExit(f"FATAL: prefill failed (exit={proc.returncode})")

    img = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"FATAL: cv2 could not read prefilled output: {out_path}")
    log(f"PREFILL  baseline loaded: shape={img.shape}")
    return img


# ---------------------------------------------------------------------------
# Step 2 - Adversarial image generators
# ---------------------------------------------------------------------------

def _rot_expand(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), borderValue=(255, 255, 255))


def _perspective(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * 0.05, h * 0.02], [w * 0.95, h * 0.08],
                      [w * 0.92, h * 0.96], [w * 0.08, h * 0.94]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))


def _occlude_corner(img: np.ndarray, corner: int) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    s = int(min(h, w) * 0.12)
    positions = [(0, 0), (w - s, 0), (0, h - s), (w - s, h - s)]
    x, y = positions[corner]
    cv2.rectangle(out, (x, y), (x + s, y + s), (255, 255, 255), -1)
    return out


def _crease(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    cv2.line(out, (0, h // 3), (w, h // 2), (70, 70, 70), thickness=8)
    return out


def _shadow(img: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32)
    h, w = img.shape[:2]
    fade = np.linspace(0.35, 1.0, w // 3)
    out[:, : w // 3] *= fade[np.newaxis, :, np.newaxis]
    return np.clip(out, 0, 255).astype(np.uint8)


def _encode_jpg(arr: np.ndarray, q: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, q])
    assert ok
    return buf.tobytes()


def _encode_png(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return buf.tobytes()


def build_variants(base: np.ndarray) -> list[tuple[str, str, bytes, str]]:
    """Return list of (label, expectation, image_bytes, ext)."""
    cases: list[tuple[str, str, bytes, str]] = []
    h, w = base.shape[:2]

    def add(label: str, exp: str, img: np.ndarray, ext: str = "jpg") -> None:
        data = _encode_jpg(img) if ext == "jpg" else _encode_png(img)
        cases.append((label, exp, data, ext))

    # --- Clean baseline ---
    add("01_baseline_clean", "pass", base)

    # --- Geometric ---
    add("02_rotate_2deg", "pass", _rot_expand(base, 2))
    add("03_rotate_5deg", "either", _rot_expand(base, 5))
    add("04_rotate_15deg", "either", _rot_expand(base, 15))
    add("05_rotate_45deg", "either", _rot_expand(base, 45))
    add("06_rotate_90deg", "either", _rot_expand(base, 90))
    add("07_rotate_180deg", "either", _rot_expand(base, 180))
    add("08_perspective_warp", "either", _perspective(base))

    # --- Blur ---
    add("09_blur_mild", "pass", cv2.GaussianBlur(base, (5, 5), 0))
    add("10_blur_heavy", "either", cv2.GaussianBlur(base, (21, 21), 0))
    add("11_blur_extreme", "graceful_fail", cv2.GaussianBlur(base, (51, 51), 0))

    # --- Low resolution ---
    lo25 = cv2.resize(cv2.resize(base, (w // 4, h // 4), cv2.INTER_AREA), (w, h), cv2.INTER_LINEAR)
    lo10 = cv2.resize(cv2.resize(base, (w // 10, h // 10), cv2.INTER_AREA), (w, h), cv2.INTER_LINEAR)
    add("12_low_res_25pct", "either", lo25)
    add("13_low_res_10pct", "graceful_fail", lo10)

    # --- JPEG crush ---
    ok30, buf30 = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 30])
    cases.append(("14_jpeg_q30", "either", buf30.tobytes(), "jpg"))
    ok5, buf5 = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 5])
    cases.append(("15_jpeg_q5", "either", buf5.tobytes(), "jpg"))

    # --- Exposure ---
    add("16_very_dark", "either",
        np.clip(base.astype(np.float32) * 0.2, 0, 255).astype(np.uint8))
    add("17_very_bright", "either",
        np.clip(base.astype(np.float32) * 1.8 + 50, 0, 255).astype(np.uint8))

    # --- Noise ---
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 40, base.shape).astype(np.int16)
    add("18_heavy_noise", "either",
        np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8))

    # --- Physical defects ---
    add("19_fold_crease", "either", _crease(base))
    add("20_scan_shadow", "either", _shadow(base))

    # --- Marker occlusion ---
    add("21_occlude_TL", "either", _occlude_corner(base, 0))
    add("22_occlude_TR", "either", _occlude_corner(base, 1))
    all_occ = base.copy()
    for c in range(4):
        all_occ = _occlude_corner(all_occ, c)
    add("23_occlude_all", "graceful_fail", all_occ)

    # --- Partial crops ---
    add("24_crop_bottom_25pct", "graceful_fail", base[: int(h * 0.75), :])
    add("25_crop_right_25pct", "graceful_fail", base[:, : int(w * 0.75)])

    # --- Wrong scale ---
    add("26_upscale_4x", "either", cv2.resize(base, (w * 4, h * 4), cv2.INTER_LINEAR))
    add("27_downscale_half", "either", cv2.resize(base, (w // 2, h // 2), cv2.INTER_AREA))

    # --- Blank pages ---
    add("28_blank_white", "graceful_fail", np.full_like(base, 255))
    add("29_blank_black", "graceful_fail", np.zeros_like(base))

    # --- PNG format ---
    add("30_png_format", "pass", base, "png")

    # --- Corrupt / garbage files (raw bytes, not ndarray) ---
    cases.append(("31_zero_bytes", "graceful_fail", b"", "jpg"))
    cases.append(("32_garbage_bytes", "graceful_fail", b"\x00\x01\x02NOT_IMAGE" * 200, "jpg"))
    cases.append(("33_truncated_jpeg", "graceful_fail", _encode_jpg(base)[:512], "jpg"))
    cases.append(("34_html_as_jpg", "graceful_fail",
                  b"<html><body>not an image</body></html>", "jpg"))

    return cases


# ---------------------------------------------------------------------------
# Step 3 - Run one test via direct subprocess
# ---------------------------------------------------------------------------

def _drain_output(proc: subprocess.Popen, buf: list[str]) -> None:
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        buf.append(line)
        kw = line.lower()
        if any(w in kw for w in ("error", "exception", "traceback", "warning",
                                  "marker", "process", "found", "no valid")):
            log(f"  ENGINE > {line}")


@dataclass
class RunResult:
    label: str
    expectation: str
    outcome: str       # 'pass' | 'graceful_fail' | 'unexpected_pass' | 'timeout' | 'crash'
    detail: str
    elapsed_s: float


def run_omr_direct(label: str, expectation: str, image_bytes: bytes, ext: str) -> RunResult:
    """Run OMR engine directly for one image variant.

    Layout under omr_stress_artifacts/<label>/:
        scan.<ext>      — the adversarial image
        template.json   — copied from custom_25_definitive_final
        config.json     — non-interactive override (show_image_level=0)
        omr_marker.jpg  — marker reference (if present)
    """
    t0 = time.perf_counter()
    test_dir = ARTIFACTS / label
    out_dir = test_dir / "outputs"

    # --- Setup ---
    log(f"  SETUP   creating {test_dir.name}/")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    img_path = test_dir / f"scan.{ext}"
    img_path.write_bytes(image_bytes)
    log(f"  SETUP   scan.{ext} written ({len(image_bytes):,} bytes)")

    shutil.copy2(TEMPLATE_JSON, test_dir / "template.json")
    with (test_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(_NON_INTERACTIVE_CONFIG, fh, indent=2)
    if MARKER_ASSET.exists():
        shutil.copy2(MARKER_ASSET, test_dir / MARKER_ASSET.name)
    log(f"  SETUP   template.json + config.json + marker ready")

    # --- Launch ---
    cmd = [PYTHON, str(REPO_ROOT / "main.py"), "-i", str(test_dir), "-o", str(out_dir)]
    log(f"  LAUNCH  python main.py -i {test_dir.name}/ -o outputs/  [timeout={PER_TEST_TIMEOUT}s]")

    stdout_buf: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            cwd=str(REPO_ROOT),
        )
        drain_thread = threading.Thread(
            target=_drain_output, args=(proc, stdout_buf), daemon=True
        )
        drain_thread.start()

        try:
            proc.wait(timeout=PER_TEST_TIMEOUT)
        except subprocess.TimeoutExpired:
            log(f"  TIMEOUT killing process after {PER_TEST_TIMEOUT}s")
            proc.kill()
            proc.wait()
            drain_thread.join(timeout=2)
            tail = " | ".join(stdout_buf[-5:])
            return RunResult(label, expectation, "timeout",
                             f"killed after {PER_TEST_TIMEOUT}s. last output: {tail[:300]}",
                             time.perf_counter() - t0)

        drain_thread.join(timeout=2)
        exit_code = proc.returncode
        elapsed = time.perf_counter() - t0
        log(f"  ENGINE  exit={exit_code}  elapsed={elapsed:.2f}s")

    except Exception as exc:
        return RunResult(label, expectation, "crash",
                         f"Popen failed: {exc}", time.perf_counter() - t0)

    # --- Analyse outputs ---
    outcome, detail = _classify_outputs(out_dir, exit_code, stdout_buf)
    final = _apply_expectation(expectation, outcome)
    log(f"  RESULT  raw={outcome}  expect={expectation}  =>  {final}  | {detail}")
    return RunResult(label, expectation, final, detail, time.perf_counter() - t0)


def _classify_outputs(out_dir: Path, exit_code: int,
                      stdout_lines: list[str]) -> tuple[str, str]:
    results_csvs = list(out_dir.rglob("Results_*.csv"))
    errors_dir = out_dir / "Manual" / "ErrorFiles"
    mm_dir = out_dir / "Manual" / "MultiMarkedFiles"

    has_results = bool(results_csvs)
    has_errors = errors_dir.exists() and any(errors_dir.iterdir())
    has_mm = mm_dir.exists() and any(mm_dir.iterdir())

    stdout_text = "\n".join(stdout_lines).lower()
    has_traceback = ("traceback" in stdout_text or "attributeerror" in stdout_text
                     or "typeerror" in stdout_text)

    log(f"  ANALYSE results_csv={len(results_csvs)}  errors={has_errors}  "
        f"multimarked={has_mm}  exit={exit_code}  traceback={has_traceback}")

    if exit_code != 0 and has_traceback:
        snippet = next((l for l in stdout_lines if "error" in l.lower()), "")
        return "crash", f"exit={exit_code} + traceback: {snippet[:120]}"

    if has_results:
        try:
            rows = [l for l in results_csvs[0]
                    .read_text(encoding="utf-8", errors="replace")
                    .splitlines()[1:] if l.strip()]
            if rows:
                return "pass", f"Results CSV with {len(rows)} row(s)"
            # Engine ran but produced no scored rows — graceful degradation
            return "graceful_fail", "Results CSV present but empty (0 data rows)"
        except Exception as e:
            return "pass", f"Results CSV present (read error: {e})"

    if has_errors:
        return "graceful_fail", "image in Manual/ErrorFiles (marker/preprocessing failed)"

    if has_mm:
        return "graceful_fail", "image in Manual/MultiMarkedFiles"

    if exit_code == 0:
        if "no valid images" in stdout_text or "empty" in stdout_text:
            return "graceful_fail", "engine reported no valid images"
        return "graceful_fail", "engine exited 0 but no result rows produced"

    # non-zero exit, no traceback — handled error
    return "graceful_fail", f"engine exited {exit_code} without traceback"


def _apply_expectation(expectation: str, raw: str) -> str:
    if raw in ("timeout", "crash"):
        return raw
    if expectation == "pass":
        return raw
    if expectation == "graceful_fail":
        return "pass" if raw == "graceful_fail" else "unexpected_pass"
    # either
    return "pass"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    log("=" * 72)
    log("OMR Robustness Stress Test  (direct subprocess, no HTTP)")
    log(f"  repo:    {REPO_ROOT}")
    log(f"  timeout: {PER_TEST_TIMEOUT}s per test")
    log(f"  python:  {PYTHON}")
    log("=" * 72)

    for p in (TEMPLATE_JSON,):
        if not p.exists():
            log(f"FATAL missing: {p}")
            return 2

    log("")
    log("--- STEP 1: Generate prefilled baseline ---")
    try:
        baseline = generate_prefilled_baseline()
    except SystemExit as e:
        log(str(e))
        return 2

    log("")
    log("--- STEP 2: Build adversarial variants ---")
    variants = build_variants(baseline)
    log(f"  {len(variants)} variants built")

    log("")
    log("--- STEP 3: Run OMR engine on each variant ---")
    results: list[RunResult] = []

    for i, (label, exp, data, ext) in enumerate(variants, 1):
        log("")
        log(f"{'='*60}")
        log(f"TEST {i:02d}/{len(variants)}: {label}  [expect={exp}]")
        r = run_omr_direct(label, exp, data, ext)
        results.append(r)

    # Summary table
    log("")
    log("=" * 72)
    log("FINAL SUMMARY")
    log("=" * 72)
    counts: dict[str, int] = {}
    for r in results:
        counts[r.outcome] = counts.get(r.outcome, 0) + 1

    label_w = max(len(r.label) for r in results)
    header = f"  {'TEST':<{label_w}}  {'EXPECT':<14}  {'RESULT':<16}  TIME    DETAIL"
    log(header)
    log("  " + "-" * (len(header) - 2))
    for r in results:
        marker = ("OK  " if r.outcome == "pass"
                  else "WARN" if r.outcome in ("graceful_fail", "unexpected_pass")
                  else "FAIL")
        log(f"  {r.label:<{label_w}}  {r.expectation:<14}  [{marker}] {r.outcome:<10}  "
            f"{r.elapsed_s:5.1f}s  {r.detail[:60]}")

    log("")
    for k in ("pass", "graceful_fail", "unexpected_pass", "timeout", "crash"):
        n = counts.get(k, 0)
        if n:
            sym = "OK  " if k == "pass" else "WARN" if k in ("graceful_fail",) else "FAIL"
            log(f"  [{sym}] {k:20s}: {n}")

    log("")
    failures = [r for r in results if r.outcome in ("timeout", "crash")]
    unexpected_fails = [r for r in results
                        if r.outcome == "graceful_fail" and r.expectation == "pass"]
    if failures:
        log("FAILURES requiring hardening:")
        for r in failures:
            log(f"  [{r.outcome.upper():7}] {r.label}")
            log(f"           {r.detail}")
    if unexpected_fails:
        log("UNEXPECTED failures on 'pass' tests:")
        for r in unexpected_fails:
            log(f"  [GRACE  ] {r.label}: {r.detail}")

    total = len(results)
    ok = counts.get("pass", 0)
    log("")
    log(f"RESULT: {ok}/{total} passed | {counts.get('timeout', 0)} timeouts"
        f" | {counts.get('crash', 0)} crashes")

    # Prune large artifacts
    for td in ARTIFACTS.iterdir():
        if not td.is_dir():
            continue
        sz = sum(f.stat().st_size for f in td.rglob("*") if f.is_file())
        if sz > 50 * 1024 * 1024:
            shutil.rmtree(td, ignore_errors=True)

    return 0 if not (failures or unexpected_fails) else 1


if __name__ == "__main__":
    raise SystemExit(main())
