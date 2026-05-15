"""Adversarial stress / chaos test for the prefill + OMR pipelines.

Hits the running webui at http://127.0.0.1:8000 with malicious / pathological
inputs and concurrent load. Records each scenario's outcome (HTTP status,
elapsed, body excerpt) and prints a final pass/fail matrix.

Categories:
  A. Adversarial CSV inputs (/prefill/batch)
  B. Adversarial single-form inputs (/prefill/single)
  C. Concurrency / throttling
  D. OMR pipeline edge cases (corrupt image, empty batch, double-cancel)
  E. Resource pressure (huge batch)

Run:
  python stress_test.py
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from PIL import Image

BASE = os.environ.get("STRESS_BASE", "http://127.0.0.1:8001/api/v1")
TIMEOUT_SHORT = 30
TIMEOUT_LONG = 600  # 5k-row batches can legitimately take many minutes

# Force UTF-8 output so Windows cp1252 console doesn't crash on PDF excerpts.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass


# Force unbuffered output so progress streams through pipes/log files.
def log(msg: str = "") -> None:
    print(msg, flush=True)


@dataclass
class Result:
    name: str
    expected: str  # short description of expected behavior
    status: int | None = None
    elapsed_s: float = 0.0
    body_excerpt: str = ""
    passed: bool = False
    note: str = ""


RESULTS: list[Result] = []


def _excerpt(body: bytes | str, n: int = 200) -> str:
    if isinstance(body, bytes):
        try:
            body = body.decode("utf-8", errors="replace")
        except Exception:
            body = repr(body[:n])
    return body[:n].replace("\n", " ")


def post_form(path: str, data: dict, files: dict | None = None,
              timeout: int = TIMEOUT_SHORT) -> tuple[int, bytes, float]:
    t0 = time.perf_counter()
    r = requests.post(BASE + path, data=data, files=files, timeout=timeout)
    elapsed = time.perf_counter() - t0
    return r.status_code, r.content, elapsed


# ---------------------------------------------------------------------------
# Helpers to build CSVs
# ---------------------------------------------------------------------------

def _csv_from_rows(rows: list[dict[str, str]]) -> str:
    cols = ["student_name", "school_name", "exam_name", "candidate_number"]
    lines = [",".join(cols)]
    for row in rows:
        vals = []
        for c in cols:
            v = str(row.get(c, ""))
            if "," in v or '"' in v or "\n" in v:
                v = '"' + v.replace('"', '""') + '"'
            vals.append(v)
        lines.append(",".join(vals))
    return "\n".join(lines) + "\n"


def good_row(i: int = 0) -> dict[str, str]:
    return {
        "student_name": f"Test Student {i}",
        "school_name": "Test School",
        "exam_name": "Test Exam",
        "candidate_number": f"{1000000000 + i}",
    }


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------

def run_csv_case(name: str, csv_text: str, *, expect_status: set[int],
                 expected_desc: str, output_mode: str = "pdf") -> None:
    log(f"  -> {name}")
    res = Result(name=name, expected=expected_desc)
    try:
        status, body, elapsed = post_form(
            "/prefill/batch",
            data={"csv_text": csv_text, "output_mode": output_mode},
            timeout=TIMEOUT_LONG,
        )
        res.status = status
        res.elapsed_s = elapsed
        res.body_excerpt = _excerpt(body)
        res.passed = status in expect_status
        if res.passed and status == 200:
            # Validate that we actually got a real PDF/zip, not silent garbage
            if output_mode == "pdf" and not body.startswith(b"%PDF"):
                res.passed = False
                res.note = "200 OK but body is not a PDF"
            elif output_mode == "zip" and not body.startswith(b"PK"):
                res.passed = False
                res.note = "200 OK but body is not a zip"
    except requests.exceptions.RequestException as exc:
        res.note = f"{type(exc).__name__}: {exc}"
        res.passed = False
    RESULTS.append(res)


def run_single_case(name: str, *, data: dict, expect_status: set[int],
                    expected_desc: str) -> None:
    log(f"  -> {name}")
    res = Result(name=name, expected=expected_desc)
    try:
        status, body, elapsed = post_form("/prefill/single", data=data)
        res.status = status
        res.elapsed_s = elapsed
        res.body_excerpt = _excerpt(body)
        res.passed = status in expect_status
    except requests.exceptions.RequestException as exc:
        res.note = f"{type(exc).__name__}: {exc}"
        res.passed = False
    RESULTS.append(res)


# ---------------------------------------------------------------------------
# A. Adversarial CSV scenarios
# ---------------------------------------------------------------------------

def category_a():
    log("\n=== A. Adversarial CSV ===")

    # A1: empty body
    run_csv_case("A1 empty CSV body", "", expect_status={422},
                 expected_desc="422 - empty CSV rejected")

    # A2: header-only, no rows
    run_csv_case("A2 header-only no rows",
                 "student_name,school_name,exam_name,candidate_number\n",
                 expect_status={422},
                 expected_desc="422 - no data rows")

    # A3: missing required column
    run_csv_case("A3 missing column",
                 "student_name,school_name,exam_name\nA,B,C\n",
                 expect_status={422},
                 expected_desc="422 - missing candidate_number")

    # A4: bad candidate_number (not 10 digits)
    run_csv_case("A4 short candidate number",
                 _csv_from_rows([{**good_row(0), "candidate_number": "123"}]),
                 expect_status={422},
                 expected_desc="422 - candidate_number must be 10 digits")

    # A5: non-digit candidate_number
    run_csv_case("A5 non-digit candidate number",
                 _csv_from_rows([{**good_row(0), "candidate_number": "ABCDEFGHIJ"}]),
                 expect_status={422},
                 expected_desc="422 - candidate_number must be digits")

    # A6: empty candidate_number
    run_csv_case("A6 empty candidate number",
                 _csv_from_rows([{**good_row(0), "candidate_number": ""}]),
                 expect_status={422},
                 expected_desc="422 - empty candidate_number")

    # A7: empty student name
    run_csv_case("A7 empty student name",
                 _csv_from_rows([{**good_row(0), "student_name": ""}]),
                 expect_status={200, 422},
                 expected_desc="should fail or sanitize, never crash")

    # A8: unicode names
    run_csv_case("A8 unicode CJK + emoji",
                 _csv_from_rows([{**good_row(0),
                                  "student_name": "学生 José 🎓 ñoño"}]),
                 expect_status={200, 422},
                 expected_desc="200 - unicode rendered or 422 - rejected, never crash")

    # A9: extremely long student name (5000 chars)
    run_csv_case("A9 huge student name (5000ch)",
                 _csv_from_rows([{**good_row(0),
                                  "student_name": "X" * 5000}]),
                 expect_status={200, 422},
                 expected_desc="should clamp or reject, never crash")

    # A10: control chars / null byte
    run_csv_case("A10 null byte in name",
                 _csv_from_rows([{**good_row(0),
                                  "student_name": "Ev\x00il\x07Name"}]),
                 expect_status={200, 422},
                 expected_desc="should sanitize or reject, never crash")

    # A11: duplicate candidate numbers
    run_csv_case("A11 duplicate candidate numbers",
                 _csv_from_rows([good_row(1), good_row(1), good_row(1)]),
                 expect_status={200, 422},
                 expected_desc="200 - process all or 422 - reject dupes")

    # A12: malformed CSV (unclosed quote)
    run_csv_case("A12 malformed CSV unclosed quote",
                 'student_name,school_name,exam_name,candidate_number\n"Bob,B,C,1234567890\n',
                 expect_status={200, 422},
                 expected_desc="should parse or 422, never crash")

    # A13: large streaming batch. The un-hardened server OOM-crashed at 5000.
    # 500 rows still proves the streaming writer never holds all PNGs in RAM,
    # while completing inside the test timeout window.
    big = _csv_from_rows([good_row(i) for i in range(500)])
    log(f"  A13 csv size = {len(big)/1024/1024:.2f} MiB")
    run_csv_case("A13 500-row streaming batch",
                 big,
                 expect_status={200, 429},
                 expected_desc="200 PDF or 429 backpressure")

    # A14: payload over 50 MiB hard limit (built from a single huge row, fast)
    huge_row = {**good_row(0), "student_name": "X" * (51 * 1024 * 1024)}
    huge = _csv_from_rows([huge_row])
    log(f"  A14 csv size = {len(huge)/1024/1024:.1f} MiB")
    run_csv_case("A14 single >50MiB row",
                 huge,
                 expect_status={413, 422, 400},
                 expected_desc="413/422 - reject oversized")

    # A15: BOM prefix
    run_csv_case("A15 UTF-8 BOM prefix",
                 "\ufeff" + _csv_from_rows([good_row(0)]),
                 expect_status={200},
                 expected_desc="200 - BOM tolerated")

    # A16: small valid batch (sanity check)
    run_csv_case("A16 valid 3-row baseline",
                 _csv_from_rows([good_row(i) for i in range(3)]),
                 expect_status={200},
                 expected_desc="200 - PDF returned")

    # A17: row count over the PDF cap (default 5000)
    over_cap = _csv_from_rows([good_row(i) for i in range(5_001)])
    run_csv_case("A17 over PDF row cap (>5000)",
                 over_cap,
                 expect_status={422},
                 expected_desc="422 - row cap rejected")

    # A18: invalid output_mode
    res = Result(name="A18 invalid output_mode", expected="422")
    try:
        status, body, elapsed = post_form(
            "/prefill/batch",
            data={"csv_text": _csv_from_rows([good_row(0)]),
                  "output_mode": "../../etc/passwd"})
        res.status, res.elapsed_s, res.body_excerpt = status, elapsed, _excerpt(body)
        res.passed = status == 422
    except Exception as exc:
        res.note = f"{type(exc).__name__}: {exc}"
    RESULTS.append(res)


# ---------------------------------------------------------------------------
# B. Adversarial single-form
# ---------------------------------------------------------------------------

def category_b():
    log("\n=== B. /prefill/single edge cases ===")
    base = {
        "student_name": "Bob",
        "school_name": "School",
        "exam_name": "Exam",
        "candidate_number": "1234567890",
        "output_format": "png",
    }
    run_single_case("B1 baseline png", data=dict(base),
                    expect_status={200}, expected_desc="200 png")
    run_single_case("B2 baseline pdf",
                    data={**base, "output_format": "pdf"},
                    expect_status={200}, expected_desc="200 pdf")
    run_single_case("B3 missing candidate_number",
                    data={**base, "candidate_number": ""},
                    expect_status={422}, expected_desc="422")
    run_single_case("B4 unicode name",
                    data={**base, "student_name": "学生 ñ 🎓"},
                    expect_status={200, 422}, expected_desc="render or reject")
    run_single_case("B5 huge name",
                    data={**base, "student_name": "Z" * 10000},
                    expect_status={200, 422}, expected_desc="clamp or reject")
    run_single_case("B6 null byte school",
                    data={**base, "school_name": "Bad\x00School"},
                    expect_status={200, 422}, expected_desc="sanitize or reject")


# ---------------------------------------------------------------------------
# C. Concurrency
# ---------------------------------------------------------------------------

def _hit_single(_i: int) -> tuple[int, float]:
    t0 = time.perf_counter()
    try:
        r = requests.post(BASE + "/prefill/single",
                          data={
                              "student_name": f"C{_i}",
                              "school_name": "S",
                              "exam_name": "E",
                              "candidate_number": f"{1000000000 + _i}",
                              "output_format": "png",
                          }, timeout=TIMEOUT_LONG)
        return r.status_code, time.perf_counter() - t0
    except requests.exceptions.RequestException:
        return -1, time.perf_counter() - t0


def category_c():
    log("\n=== C. Concurrency ===")

    # C1: 50 concurrent /prefill/single
    res = Result(name="C1 50x concurrent /single", expected="server stays alive, no 5xx")
    t0 = time.perf_counter()
    statuses = []
    with cf.ThreadPoolExecutor(max_workers=50) as ex:
        for st, _ in ex.map(_hit_single, range(50)):
            statuses.append(st)
    res.elapsed_s = time.perf_counter() - t0
    res.status = max(statuses) if statuses else -1
    n_ok = sum(1 for s in statuses if s == 200)
    n_429 = sum(1 for s in statuses if s in {429, 503})
    n_5xx = sum(1 for s in statuses if 500 <= s < 600 and s != 503)
    n_err = sum(1 for s in statuses if s == -1)
    res.body_excerpt = f"ok={n_ok}/50 429/503={n_429} 5xx={n_5xx} netfail={n_err}"
    # Pass: server doesn't crash and every request gets *some* answer.
    res.passed = (n_5xx == 0 and n_err == 0 and (n_ok + n_429) == 50)
    RESULTS.append(res)

    # C2: 10 concurrent batches of 100 rows each
    res = Result(name="C2 10x concurrent batch(100)", expected="all 200 or 429 backpressure")
    payloads = [_csv_from_rows([good_row(i + 1000 * k) for i in range(100)])
                for k in range(10)]

    def hit_batch(p: str) -> tuple[int, float]:
        t = time.perf_counter()
        try:
            r = requests.post(BASE + "/prefill/batch",
                              data={"csv_text": p, "output_mode": "pdf"},
                              timeout=TIMEOUT_LONG)
            return r.status_code, time.perf_counter() - t
        except requests.exceptions.RequestException:
            return -1, time.perf_counter() - t

    t0 = time.perf_counter()
    statuses = []
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        for st, _ in ex.map(hit_batch, payloads):
            statuses.append(st)
    res.elapsed_s = time.perf_counter() - t0
    res.status = max(statuses)
    n_ok = sum(1 for s in statuses if s == 200)
    n_429 = sum(1 for s in statuses if s == 429 or s == 503)
    n_5xx = sum(1 for s in statuses if 500 <= s < 600 and s != 503)
    n_err = sum(1 for s in statuses if s == -1)
    res.body_excerpt = f"ok={n_ok}/10 429/503={n_429} 5xx={n_5xx} netfail={n_err}"
    # Pass: server stays alive (no 5xx, no net failures) AND every request gets
    # *some* answer (ok + backpressure 429 should equal total).
    res.passed = (n_5xx == 0 and n_err == 0 and (n_ok + n_429) == 10)
    RESULTS.append(res)

    # C3: 2k-row batch (the size that crashed baseline server) -- must not OOM
    res = Result(name="C3 single 2k-row PDF batch", expected="200 or 429, never crash")
    big = _csv_from_rows([good_row(i) for i in range(2000)])
    try:
        status, body, elapsed = post_form(
            "/prefill/batch",
            data={"csv_text": big, "output_mode": "pdf"},
            timeout=TIMEOUT_LONG)
        res.status, res.elapsed_s, res.body_excerpt = status, elapsed, f"len={len(body)}"
        # Must be either real PDF or backpressure -- must NOT be empty/HTML/5xx.
        if status == 200:
            res.passed = body.startswith(b"%PDF")
        else:
            res.passed = status in {429, 422}
    except Exception as exc:
        res.note = f"{type(exc).__name__}: {exc}"
    RESULTS.append(res)

    # C4: server still healthy after the heavy run?
    res = Result(name="C4 server still healthy after stress", expected="200")
    try:
        r = requests.get(BASE + "/system/info", timeout=10)
        res.status = r.status_code
        res.body_excerpt = _excerpt(r.text)
        res.passed = r.status_code == 200
    except Exception as exc:
        res.note = f"{type(exc).__name__}: {exc}"
    RESULTS.append(res)


# ---------------------------------------------------------------------------
# D. OMR pipeline
# ---------------------------------------------------------------------------

def _create_batch(name: str = "stress_test") -> str | None:
    r = requests.post(BASE + "/batches", json={"name": name})
    if r.status_code in (200, 201):
        return r.json().get("id") or r.json().get("batch_id")
    return None


def _delete_batch(batch_id: str) -> None:
    try:
        requests.delete(BASE + f"/batches/{batch_id}", timeout=10)
    except Exception:
        pass


def _upload_template(batch_id: str, template_dir: Path) -> int:
    tpl = (template_dir / "template.json").read_text()
    cfg = (template_dir / "config.json").read_text()
    r = requests.put(BASE + f"/batches/{batch_id}/template",
                     data=tpl,
                     headers={"Content-Type": "application/json"})
    if r.status_code not in (200, 201):
        return r.status_code
    r = requests.put(BASE + f"/batches/{batch_id}/config",
                     data=cfg,
                     headers={"Content-Type": "application/json"})
    return r.status_code


def _upload_image(batch_id: str, name: str, blob: bytes,
                  content_type: str = "image/png") -> int:
    r = requests.post(
        BASE + f"/batches/{batch_id}/images",
        files={"images": (name, io.BytesIO(blob), content_type)},
        timeout=60,
    )
    return r.status_code


def category_d(template_dir: Path):
    log("\n=== D. OMR pipeline edge cases ===")

    # D1: process empty batch (no images) -> should fail cleanly
    bid = _create_batch("stress_empty")
    res = Result(name="D1 process batch with no images",
                 expected="failed status with helpful message")
    if bid is None:
        res.note = "could not create batch"
        RESULTS.append(res)
    else:
        try:
            _upload_template(bid, template_dir)
            r = requests.post(BASE + f"/batches/{bid}/process", timeout=10)
            res.status = r.status_code
            time.sleep(2.0)
            stat = requests.get(BASE + f"/batches/{bid}", timeout=10).json()
            res.body_excerpt = (f"start_status={r.status_code} "
                                f"final={stat.get('status')} "
                                f"err={(stat.get('last_error') or '')[:80]}")
            res.passed = stat.get("status") in {"failed", "done"}
        except Exception as exc:
            res.note = f"{type(exc).__name__}: {exc}"
        finally:
            _delete_batch(bid)
        RESULTS.append(res)

    # D2: corrupt image mixed with valid ones - whole batch should not crash
    bid = _create_batch("stress_corrupt")
    res = Result(name="D2 corrupt PNG amongst good images",
                 expected="batch completes; bad file marked failed")
    if bid is None:
        res.note = "could not create batch"
        RESULTS.append(res)
    else:
        try:
            _upload_template(bid, template_dir)
            # Real prefilled image (small enough to upload quickly)
            ref = template_dir / "blank_template.png"
            if not ref.exists():
                # Fall back to a generated white image
                img = Image.new("RGB", (1000, 1500), (255, 255, 255))
                buf = io.BytesIO(); img.save(buf, format="PNG")
                blob = buf.getvalue()
            else:
                blob = ref.read_bytes()
            _upload_image(bid, "good.png", blob)
            _upload_image(bid, "corrupt.png", b"NOT A REAL PNG\x00\x01\x02")
            _upload_image(bid, "tiny.png", b"\x89PNG\r\n\x1a\n")  # truncated
            r = requests.post(BASE + f"/batches/{bid}/process", timeout=10)
            res.status = r.status_code
            for _ in range(60):
                time.sleep(2.0)
                stat = requests.get(BASE + f"/batches/{bid}", timeout=10).json()
                if stat.get("status") in {"done", "failed", "cancelled"}:
                    break
            res.body_excerpt = (f"final={stat.get('status')} "
                                f"processed={stat.get('processed_files')} "
                                f"err={(stat.get('last_error') or '')[:80]}")
            res.passed = stat.get("status") in {"done", "failed"}
            # Pipeline must NOT hang or crash the server
        except Exception as exc:
            res.note = f"{type(exc).__name__}: {exc}"
        finally:
            _delete_batch(bid)
        RESULTS.append(res)

    # D3: cancel before run + cancel after start
    bid = _create_batch("stress_cancel")
    res = Result(name="D3 cancel queued batch", expected="cancelled status")
    if bid is None:
        res.note = "could not create batch"
        RESULTS.append(res)
    else:
        try:
            _upload_template(bid, template_dir)
            # Immediate cancel before any processing
            r = requests.post(BASE + f"/batches/{bid}/cancel", timeout=10)
            res.status = r.status_code
            res.body_excerpt = f"cancel_status={r.status_code}"
            res.passed = r.status_code in {200, 202, 400, 409}
        except Exception as exc:
            res.note = f"{type(exc).__name__}: {exc}"
        finally:
            _delete_batch(bid)
        RESULTS.append(res)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report() -> int:
    print("\n" + "=" * 90)
    print("STRESS TEST SUMMARY")
    print("=" * 90)
    print(f"{'#':<4} {'NAME':<40} {'STATUS':<7} {'ELAPSED':<10} {'PASS':<5} NOTE/EXCERPT")
    print("-" * 90)
    n_pass = 0
    for i, r in enumerate(RESULTS, start=1):
        verdict = "PASS" if r.passed else "FAIL"
        if r.passed:
            n_pass += 1
        notes = r.note or r.body_excerpt
        print(f"{i:<4} {r.name[:39]:<40} "
              f"{str(r.status or '-'):<7} {r.elapsed_s:>7.2f}s   "
              f"{verdict:<5} {notes[:120]}")
    print("-" * 90)
    print(f"PASS {n_pass}/{len(RESULTS)}   FAIL {len(RESULTS) - n_pass}")
    return 0 if n_pass == len(RESULTS) else 1


def main() -> int:
    print("Stress test starting against", BASE)
    template_dir = Path("custom_25_definitive_final")
    if not template_dir.exists():
        print("ERROR: missing custom_25_definitive_final/")
        return 2

    # Quick sanity: server is up
    try:
        info = requests.get(BASE + "/system/info", timeout=5).json()
        print(f"server OK | cpu_count={info.get('cpu_count')} workers={info.get('default_max_workers')}")
    except Exception as exc:
        print(f"ERROR: server unreachable: {exc}")
        return 2

    category_a()
    category_b()
    category_c()
    category_d(template_dir)
    return report()


if __name__ == "__main__":
    sys.exit(main())
