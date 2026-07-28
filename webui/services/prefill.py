"""Service layer for the prefill answer-sheet feature.

Wraps ``prefill_package.prefill_answer_sheet_final``. Batch outputs are
streamed directly to a temp file on disk so peak memory is bounded regardless
of row count (a 5k-row PDF must not OOM the server).
"""

from __future__ import annotations

import io
import logging
import math
import os
import re
import sys
import tempfile
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

from webui.services.scan_simulation import (
    BubbleGeometry,
    MarkerBox,
    apply_scan_simulation,
    normalize_realism_preset,
)
from webui.services.student_fill import (
    DEFAULT_MARKING_PROFILE,
    MARKING_PROFILES,
    _stable_seed,
    draw_student_marks,
    normalize_marking_profile,
    parse_answers,
)
from webui.services import prefill_letter_smq60 as letter_layout
from webui.sheet_registry import (
    LANDSCAPE_DIR,
    LANDSCAPE_NNQ25_0,
    LEGACY_LANDSCAPE,
    LETTER_LANDSCAPE_SMQ60_0,
    LETTER_LANDSCAPE_SMQ60_DIR,
    normalize_preset_name,
)

# Active single/batch prefill layout (Letter SMQ60 is the product default).
PREFILL_SHEET_LAYOUT = "letter_smq60"
PREFILL_NUM_QUESTIONS = letter_layout.NUM_QUESTIONS

# When running as a PyInstaller frozen bundle sys._MEIPASS is the _internal/
# directory where data files are extracted; fall back to the source-tree path.
if getattr(sys, "frozen", False):
    _PKG_ROOT = Path(sys._MEIPASS)  # type: ignore[attr-defined]
else:
    _PKG_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = _PKG_ROOT / "prefill_package" / "blank_template_reference.png"
LETTER_TEMPLATE = letter_layout.LETTER_BLANK_PNG

# Registry of "print blank sheet" variants. Each entry maps a stable key
# (used by the ``/api/v1/prefill/blank`` form and persisted in tests) to a
# 1-page source PDF on disk plus a human-readable label for the UI. The
# blank-sheet generator clones the source page N times into the output
# PDF, optionally stamping continuous page numbers and post-splitting into
# printer-safe segments — the same workflow the existing batch endpoint
# uses, minus per-row rendering.
#
# When adding a new variant make sure its source PDF is bundled by
# ``OMRChecker.spec`` (otherwise the desktop build will 404 at runtime).

BLANK_SHEET_VARIANTS: dict[str, dict[str, Any]] = {
    LETTER_LANDSCAPE_SMQ60_0: {
        "label": (
            "July 2026 Letter landscape (SMQ60) — 60Q US Letter with ArUco markers"
        ),
        "source_pdf": (
            LETTER_LANDSCAPE_SMQ60_DIR
            / "reference"
            / "blank_landscape_smq60.pdf"
        ),
        "default_stem": f"blank_{LETTER_LANDSCAPE_SMQ60_0}",
    },
    LANDSCAPE_NNQ25_0: {
        "label": (
            "April 2026 landscape (NNQ25) — production sheet with ArUco markers"
        ),
        "source_pdf": (
            LANDSCAPE_DIR / "blank_legacy_landscape_answer_sheet_with_markers.pdf"
        ),
        "default_stem": "blank_MoE-April-2026-Landscape-NNQ25-0",
    },
}

DEFAULT_BLANK_SHEET_VARIANT = LETTER_LANDSCAPE_SMQ60_0

logger = logging.getLogger(__name__)

# Per-field clamps. Names that overflow drawable area produced runtime
# overflow / pillow ValueError in earlier stress runs; clamp at the source.
_MAX_FIELD_LEN = 200
# Strip control characters except common whitespace.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean_field(value: Any, *, max_len: int = _MAX_FIELD_LEN) -> str:
    """Strip control chars & clamp length so renderer never sees pathological input."""
    text = "" if value is None else str(value)
    text = _CONTROL_RE.sub("", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "…"
    return text


# Canonical CSV column names the payload builders and grouping logic expect,
# plus the permissive header aliases operators export from their student
# registries. The Centre field in particular is exported under several
# spellings ("center_name", "centre_name", "centre", …); map them all onto
# the canonical ``school_name`` so downstream code stays simple. Matching is
# case-insensitive and whitespace-trimmed.
_CSV_COLUMN_ALIASES: dict[str, str] = {
    "student_name": "student_name",
    "student": "student_name",
    "name": "student_name",
    "candidate_number": "candidate_number",
    "candidate_no": "candidate_number",
    "candidate": "candidate_number",
    "school_name": "school_name",
    "school": "school_name",
    "centre_name": "school_name",
    "center_name": "school_name",
    "centre": "school_name",
    "center": "school_name",
    "exam_name": "exam_name",
    "exam": "exam_name",
    "subject_name": "subject_name",
    "subject": "subject_name",
    "region": "region",
    "region_name": "region",
    "answers": "answers",
    "answers_json": "answers_json",
    "marking_profile": "marking_profile",
    "output_file": "output_file",
}


def normalize_row_keys(row: dict[str, Any]) -> dict[str, Any]:
    """Map permissive CSV header aliases onto canonical payload keys.

    Header matching is case-insensitive and whitespace-trimmed, so a CSV with
    a ``center_name`` (or ``Centre Name``) column is treated the same as one
    using the canonical ``school_name``. Unknown columns are preserved
    verbatim so internal fields (e.g. ``_source_row_number``) survive. When
    two aliases map to the same canonical key, the first non-empty value wins.
    """
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key is None:
            continue
        lookup = re.sub(r"[\s\-]+", "_", str(key).strip().lower())
        canonical = _CSV_COLUMN_ALIASES.get(lookup)
        if canonical is None:
            out.setdefault(str(key), value)
            continue
        existing = out.get(canonical)
        if canonical not in out or (
            not _clean_field(existing) and _clean_field(value)
        ):
            out[canonical] = value
    return out


# Columns every prefill CSV row must resolve to (after alias normalisation).
# ``exam_name``/``subject_name``/``region`` are optional: sheets simply leave
# the corresponding write-in line blank when they are absent.
REQUIRED_CSV_COLUMNS: frozenset[str] = frozenset(
    {"student_name", "school_name", "candidate_number"}
)



def _import_prefill():
    """Lazy import to avoid loading PIL at module-level if not needed."""
    from prefill_package.prefill_answer_sheet_final import prefill_sheet

    return prefill_sheet


def _import_prefill_module():
    """Lazy import of the full prefill module (needed for batch helpers)."""
    import prefill_package.prefill_answer_sheet_final as m

    return m


def _images_to_pdf_bytes(images) -> bytes:
    """Convert a list of PIL Images to a PDF byte string in memory."""
    imgs = [im.convert("RGB") for im in images]
    if not imgs:
        raise ValueError("No images to convert.")
    buf = io.BytesIO()
    imgs[0].save(
        buf,
        format="PDF",
        save_all=True,
        append_images=imgs[1:],
        resolution=300.0,
    )
    return buf.getvalue()


def _images_to_pdf_bytes_fast(png_bytes_list: list[bytes]) -> bytes:
    """Assemble PNG bytes into PDF using PyMuPDF (faster, lower peak RAM)."""
    import fitz

    doc = fitz.open()
    for png_bytes in png_bytes_list:
        img_doc = fitz.open("png", png_bytes)
        pdf_bytes = img_doc.convert_to_pdf()
        img_doc.close()
        src = fitz.open("pdf", pdf_bytes)
        doc.insert_pdf(src)
        src.close()
    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True)
    doc.close()
    return buf.getvalue()


def _validate_candidate_number(candidate_number: str) -> None:
    if len(candidate_number) != 10 or not candidate_number.isdigit():
        display_value = candidate_number if candidate_number else "<blank>"
        raise ValueError(
            "Candidate number must be exactly 10 digits; "
            f"got {display_value!r}."
        )


def _build_payload(stamped_bytes: bytes, row: dict[str, Any]) -> dict[str, Any]:
    """Validate + sanitise a row into a legacy worker payload (includes stamped_bytes)."""
    candidate_number = _clean_field(row.get("candidate_number"), max_len=64)
    _validate_candidate_number(candidate_number)
    return {
        "stamped_bytes": stamped_bytes,
        "student_name": _clean_field(row.get("student_name")),
        "school_name": _clean_field(row.get("school_name")),
        "exam_name": _clean_field(row.get("exam_name")),
        "subject_name": _clean_field(row.get("subject_name")),
        "candidate_number": candidate_number,
    }


def _build_fast_payload(
    row: dict[str, Any],
    output_format: str = "png",
    realism_preset: str = "none",
    marking_profile: str = "none",
    answers: Any = None,
) -> dict[str, Any]:
    """Validate + sanitise a row into a fast worker payload (no stamped_bytes).

    Per-row ``answers`` (CSV column) override the batch-level default. The
    row's ``marking_profile`` column overrides the batch-level profile when
    present, otherwise the batch default is used.
    """
    candidate_number = _clean_field(row.get("candidate_number"), max_len=64)
    _validate_candidate_number(candidate_number)

    row_profile = _clean_field(row.get("marking_profile"), max_len=64) or marking_profile
    row_profile = normalize_marking_profile(row_profile or "none")

    # Per-row answers may live in either ``answers`` or ``answers_json``.
    row_answers_raw = row.get("answers")
    if row_answers_raw in (None, ""):
        row_answers_raw = row.get("answers_json")
    if row_answers_raw in (None, ""):
        row_answers_raw = answers

    # Seed random-answer generation deterministically per candidate_number +
    # answer spec so each row is reproducible but each sheet in a batch
    # gets a distinct pattern. Without a seed, ``random`` / ``random_with_skips``
    # would silently produce a fresh pattern every call.
    answer_seed = _stable_seed("prefill-answers", candidate_number, str(row_answers_raw))
    parsed_answers = (
        parse_answers(
            row_answers_raw, num_questions=PREFILL_NUM_QUESTIONS, seed=answer_seed
        )
        if row_answers_raw is not None
        else {}
    )

    return {
        "student_name": _clean_field(row.get("student_name")),
        "school_name": _clean_field(row.get("school_name")),
        "exam_name": _clean_field(row.get("exam_name")),
        "subject_name": _clean_field(row.get("subject_name")),
        "candidate_number": candidate_number,
        "output_format": output_format,
        "realism_preset": normalize_realism_preset(realism_preset),
        "marking_profile": row_profile,
        "answers": parsed_answers,
    }


def _valid_fast_payloads(
    rows: list[dict[str, Any]],
    *,
    output_format: str = "png",
    realism_preset: str = "none",
    marking_profile: str = "none",
    answers: Any = None,
    include_page_numbers: bool = False,
) -> tuple[list[dict[str, Any]], list[int], list[str]]:
    """Build render payloads, preserving row-level errors for invalid rows.

    Validation happens before worker submission, so a single bad CSV row
    should be skipped rather than aborting the whole batch or region group.
    ``source_rows`` maps payload indexes back to the 1-based CSV row number
    within the current batch/group for accurate error messages.
    """
    payloads: list[dict[str, Any]] = []
    source_rows: list[int] = []
    errors: list[str] = []

    for row_number, row in enumerate(rows, start=1):
        source_row_number = int(row.get("_source_row_number") or row_number)
        try:
            payload = _build_fast_payload(
                row,
                output_format=output_format,
                realism_preset=realism_preset,
                marking_profile=marking_profile,
                answers=answers,
            )
        except Exception as exc:  # noqa: BLE001 - report per-row validation
            errors.append(f"row {source_row_number}: {type(exc).__name__}: {exc}")
            continue

        if include_page_numbers:
            # Number only the sheets that will actually be rendered so the
            # produced PDF stays continuous even when invalid rows are skipped.
            payload["page_number"] = len(payloads) + 1
        payloads.append(payload)
        source_rows.append(source_row_number)

    return payloads, source_rows, errors


# ---------------------------------------------------------------------------
# Module-level stamped template cache.
# The stamped PIL image (ArUco corners drawn) is constant for the lifetime of
# the process.  Caching it here avoids ~50 ms of disk-read + ArUco work on
# every single-sheet HTTP request.
# ---------------------------------------------------------------------------
_STAMPED_IMG_CACHE: 'Image.Image | None' = None  # type: ignore[name-defined]  # noqa: F821
_STAMPED_ARR_CACHE: 'np.ndarray | None' = None  # type: ignore[name-defined]  # noqa: F821


def _get_stamped_img():
    """Return the stamped PIL Image, building and caching it on first call."""
    global _STAMPED_IMG_CACHE, _STAMPED_ARR_CACHE
    if _STAMPED_IMG_CACHE is None:
        t = time.perf_counter()
        if PREFILL_SHEET_LAYOUT == "letter_smq60":
            # Letter blank already includes ArUco markers.
            _STAMPED_IMG_CACHE = letter_layout.load_letter_blank()
        else:
            m = _import_prefill_module()
            _STAMPED_IMG_CACHE = m.load_stamped_template(DEFAULT_TEMPLATE)
        _STAMPED_ARR_CACHE = np.array(_STAMPED_IMG_CACHE)
        logger.info("Stamped template cached | %.1fms", (time.perf_counter() - t) * 1000)
    return _STAMPED_IMG_CACHE, _STAMPED_ARR_CACHE


def _draw_identity_and_candidate(
    img,
    *,
    student_name: str,
    school_name: str,
    exam_name: str,
    candidate_number: str,
    subject_name: str = "",
):
    """Draw student identity + candidate bubbles for the active sheet layout."""
    if PREFILL_SHEET_LAYOUT == "letter_smq60":
        return letter_layout.draw_letter_sheet_content(
            img,
            student_name=student_name,
            centre_name=school_name,
            exam_name=exam_name,
            subject_name=subject_name,
            candidate_number=candidate_number,
        )
    m = _import_prefill_module()
    return m._draw_sheet_content(
        img, student_name, school_name, exam_name, candidate_number
    )


def _stamp_template_once() -> bytes:
    """Render the ArUco-stamped template once, returning its PNG bytes."""
    stamped_img, _ = _get_stamped_img()
    assert stamped_img is not None
    stamped_buf = io.BytesIO()
    stamped_img.save(stamped_buf, format="PNG", compress_level=1)
    return stamped_buf.getvalue()


def _max_workers() -> int:
    return max(1, min((os.cpu_count() or 2) - 1, 8))


def _simulate_scan_if_needed(
    image,
    prefill_module,
    *,
    candidate_number: str,
    realism_preset: str = "none",
):
    """Apply optional deterministic scan simulation after content drawing."""
    preset = normalize_realism_preset(realism_preset)
    if preset == "none":
        return image

    w, h = image.size
    if PREFILL_SHEET_LAYOUT == "letter_smq60":
        cand_geom = letter_layout.candidate_bubble_geometry(w, h, candidate_number)
        marker_geom = letter_layout.aruco_marker_boxes(w, h)
        candidate_region = letter_layout.candidate_region_box(w, h)
    else:
        cand_geom = prefill_module.candidate_bubble_geometry(w, h, candidate_number)
        marker_geom = prefill_module.aruco_marker_boxes(w, h)
        candidate_region = prefill_module.candidate_region_box(w, h)
    bubbles = [
        BubbleGeometry(
            column=int(item["column"]),
            digit=int(item["digit"]),
            cx=int(item["cx"]),
            cy=int(item["cy"]),
            radius=int(item["radius"]),
            filled=bool(item.get("filled", False)),
        )
        for item in cand_geom
    ]
    markers = [
        MarkerBox(
            corner=int(item["corner"]),
            x0=int(item["x0"]),
            y0=int(item["y0"]),
            x1=int(item["x1"]),
            y1=int(item["y1"]),
        )
        for item in marker_geom
    ]
    # Treat the candidate-number block as "printed and never written
    # over" — keep it pristine even when the rest of the page is heavily
    # degraded by moderate/adversarial scan effects.
    return apply_scan_simulation(
        image,
        preset=preset,
        candidate_number=candidate_number,
        bubbles=bubbles,
        markers=markers,
        candidate_region=candidate_region,
    )


def _maybe_fill_student_marks(
    image,
    *,
    candidate_number: str,
    marking_profile: str | None,
    answers: Any,
):
    """Optionally apply student-style bubble fills to ``image``.

    No-op when ``marking_profile`` is ``"none"`` or ``answers`` is empty.
    Returns the resulting image (a new instance when marks were drawn).
    """
    profile = (marking_profile or "none").lower()
    if profile == "none":
        return image
    if not answers:
        return image
    return draw_student_marks(
        image,
        answers=answers,
        marking_profile=profile,
        candidate_number=candidate_number,
        sheet_layout=PREFILL_SHEET_LAYOUT,
        num_questions=PREFILL_NUM_QUESTIONS,
    )


def _thread_render(payload: dict) -> bytes:
    """Thread worker: renders one prefill sheet using the shared in-process template cache.

    Threads share the stamped PIL Image already held in ``_STAMPED_IMG_CACHE``;
    no IPC or pickling is required.  PIL image operations release the GIL so
    multiple threads make real progress in parallel.
    """
    m = _import_prefill_module()
    stamped_img, _ = _get_stamped_img()
    assert stamped_img is not None
    img = stamped_img.copy()
    img = _draw_identity_and_candidate(
        img,
        student_name=payload['student_name'],
        school_name=payload['school_name'],
        exam_name=payload['exam_name'],
        candidate_number=payload['candidate_number'],
        subject_name=payload.get('subject_name', ''),
    )
    page_number = payload.get('page_number')
    if page_number is not None:
        # Stamp BEFORE simulation so the number degrades like any other
        # printed mark. Placement guarantees no overlap with bubbles or
        # ArUco markers (see prefill_answer_sheet_final.page_number_anchor).
        img = m.draw_page_number(img, page_number)
    img = _maybe_fill_student_marks(
        img,
        candidate_number=payload['candidate_number'],
        marking_profile=payload.get('marking_profile', 'none'),
        answers=payload.get('answers') or {},
    )
    img = _simulate_scan_if_needed(
        img,
        m,
        candidate_number=payload['candidate_number'],
        realism_preset=payload.get('realism_preset', 'none'),
    )
    buf = io.BytesIO()
    fmt = payload.get('output_format', 'png').lower()
    if fmt == 'jpeg':
        img.save(buf, format='JPEG', quality=payload.get('jpeg_quality', 75))
    else:
        img.save(buf, format='PNG', compress_level=1)
    return buf.getvalue()


def _iter_pngs_fast(payloads: list[dict], *, preserve_order: bool = True):
    """Yield ``(index, png_bytes_or_None, error_or_None)`` using a thread pool.

    Uses ``ThreadPoolExecutor`` (not ``ProcessPoolExecutor``) so spawned threads
    never inherit the server's listening socket — which was the root cause of
    orphaned workers stealing connections and hanging the server.

    PIL image operations release the GIL, so threads achieve real parallelism
    for the CPU-bound rendering work.  Falls back to serial on any executor
    error.
    """
    n = len(payloads)
    max_workers = _max_workers()
    window_size = max_workers * 4
    # If workers produce nothing for this many seconds, abort rather than
    # looping forever.  300 s (5 min) is generous even for very large batches.
    _MAX_IDLE_S = 300

    yielded_indexes: set[int] = set()
    ex = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures: dict = {}
        completed: dict[int, tuple[bytes | None, str | None]] = {}
        next_submit = [0]  # list so the inner closure can mutate it
        next_yield = 0
        last_progress = time.perf_counter()

        def submit_until_window() -> None:
            while next_submit[0] < n and len(futures) < window_size:
                idx = next_submit[0]
                future = ex.submit(_thread_render, payloads[idx])
                futures[future] = idx
                next_submit[0] += 1

        submit_until_window()
        while futures:
            done, _ = wait(
                list(futures.keys()), timeout=30, return_when=FIRST_COMPLETED
            )
            if not done:
                idle_secs = time.perf_counter() - last_progress
                logger.warning(
                    "Prefill worker pool idle for %.0fs | pending=%d/%d",
                    idle_secs,
                    len(futures),
                    n,
                )
                if idle_secs > _MAX_IDLE_S:
                    logger.error(
                        "Aborting prefill pool — no progress for %.0fs (limit=%ds)",
                        idle_secs,
                        _MAX_IDLE_S,
                    )
                    break
                continue
            for future in done:
                idx = futures.pop(future)
                last_progress = time.perf_counter()
                try:
                    result: tuple[bytes | None, str | None] = (future.result(), None)
                except Exception as exc:  # noqa: BLE001
                    result = (None, f"{type(exc).__name__}: {exc}")

                if preserve_order:
                    completed[idx] = result
                else:
                    yielded_indexes.add(idx)
                    yield idx, result[0], result[1]

            if preserve_order:
                while next_yield in completed:
                    result = completed.pop(next_yield)
                    yielded_indexes.add(next_yield)
                    yield next_yield, result[0], result[1]
                    next_yield += 1

            submit_until_window()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Thread worker pool error after %d/%d rows; falling back to serial: %s",
            len(yielded_indexes), n, exc,
        )
    finally:
        # Non-blocking shutdown: don't wait for stuck threads.
        # cancel_futures=True cancels any not-yet-started submissions.
        ex.shutdown(wait=False, cancel_futures=True)

    # Serial fallback for remaining.
    for idx in range(n):
        if idx in yielded_indexes:
            continue
        try:
            # Re-use stamped img directly in-process to avoid another pool.
            stamped_img, _ = _get_stamped_img()
            assert stamped_img is not None
            img = stamped_img.copy()
            m2 = _import_prefill_module()
            img = _draw_identity_and_candidate(
                img,
                student_name=payloads[idx]['student_name'],
                school_name=payloads[idx]['school_name'],
                exam_name=payloads[idx]['exam_name'],
                candidate_number=payloads[idx]['candidate_number'],
                subject_name=payloads[idx].get('subject_name', ''),
            )
            fallback_page = payloads[idx].get('page_number')
            if fallback_page is not None:
                img = m2.draw_page_number(img, fallback_page)
            img = _maybe_fill_student_marks(
                img,
                candidate_number=payloads[idx]['candidate_number'],
                marking_profile=payloads[idx].get('marking_profile', 'none'),
                answers=payloads[idx].get('answers') or {},
            )
            img = _simulate_scan_if_needed(
                img,
                m2,
                candidate_number=payloads[idx]['candidate_number'],
                realism_preset=payloads[idx].get('realism_preset', 'none'),
            )
            buf = io.BytesIO()
            img.save(buf, format='PNG', compress_level=1)
            yield idx, buf.getvalue(), None
        except Exception as inner:  # noqa: BLE001
            yield idx, None, f"{type(inner).__name__}: {inner}"


def _iter_pngs_with_fallback(payloads: list[dict]):
    """Legacy path kept for backward compatibility. Delegates to fast path."""
    fast_payloads = [
        {k: v for k, v in p.items() if k != "stamped_bytes"} for p in payloads
    ]
    yield from _iter_pngs_fast(fast_payloads)


def generate_single_png(
    student_name: str,
    school_name: str,
    exam_name: str,
    candidate_number: str,
    realism_preset: str = "none",
    marking_profile: str = "none",
    answers: Any = None,
    subject_name: str = "",
) -> bytes:
    candidate_number = _clean_field(candidate_number, max_len=64)
    _validate_candidate_number(candidate_number)
    realism_preset = normalize_realism_preset(realism_preset)
    marking_profile = normalize_marking_profile(marking_profile or "none")
    answer_seed = _stable_seed("prefill-answers", candidate_number, str(answers))
    parsed_answers = (
        parse_answers(
            answers, num_questions=PREFILL_NUM_QUESTIONS, seed=answer_seed
        )
        if answers is not None
        else {}
    )
    m = _import_prefill_module()
    stamped_img, _ = _get_stamped_img()
    assert stamped_img is not None
    image = _draw_identity_and_candidate(
        stamped_img.copy(),
        student_name=_clean_field(student_name),
        school_name=_clean_field(school_name),
        exam_name=_clean_field(exam_name),
        candidate_number=candidate_number,
        subject_name=_clean_field(subject_name),
    )
    image = _maybe_fill_student_marks(
        image,
        candidate_number=candidate_number,
        marking_profile=marking_profile,
        answers=parsed_answers,
    )
    image = _simulate_scan_if_needed(
        image,
        m,
        candidate_number=candidate_number,
        realism_preset=realism_preset,
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def generate_single_pdf(
    student_name: str,
    school_name: str,
    exam_name: str,
    candidate_number: str,
    realism_preset: str = "none",
    marking_profile: str = "none",
    answers: Any = None,
    subject_name: str = "",
) -> bytes:
    import fitz
    import struct
    candidate_number = _clean_field(candidate_number, max_len=64)
    _validate_candidate_number(candidate_number)
    png_bytes = generate_single_png(
        student_name,
        school_name,
        exam_name,
        candidate_number,
        realism_preset=realism_preset,
        marking_profile=marking_profile,
        answers=answers,
        subject_name=subject_name,
    )
    w, h = struct.unpack('>II', png_bytes[16:24])
    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    page.insert_image(page.rect, stream=png_bytes)
    buf = io.BytesIO()
    doc.save(buf, garbage=0)
    doc.close()
    return buf.getvalue()


def generate_batch_pdf_to_file(
    rows: list[dict[str, Any]],
    dst_path: Path,
    realism_preset: str = "none",
    include_page_numbers: bool = False,
    marking_profile: str = "none",
    answers: Any = None,
) -> dict:
    """Stream PDF generation directly to ``dst_path``.

    Workers output JPEG bytes; JPEG is stored natively in PDF as DCT so no
    re-encoding or deflate pass is needed.  Progress is logged every 500 sheets.
    Returns a metadata dict: ``{count, successes, errors, elapsed_s, size_bytes}``.

    When ``include_page_numbers`` is ``True``, each rendered sheet receives
    a sequential page-number stamp (1, 2, 3, …) in the bottom-right corner,
    matching the order of ``rows``. Output ordering is guaranteed by
    :func:`_iter_pngs_fast` so the stamp on each PDF page matches its
    physical position in the document.
    """
    import fitz  # PyMuPDF

    count = len(rows)
    logger.info(
        "Prefill batch PDF started | count=%d | page_numbers=%s",
        count, include_page_numbers,
    )
    t_batch = time.perf_counter()

    realism_preset = normalize_realism_preset(realism_preset)
    marking_profile = normalize_marking_profile(marking_profile or "none")
    payloads, source_rows, errors = _valid_fast_payloads(
        rows,
        output_format="jpeg",
        realism_preset=realism_preset,
        marking_profile=marking_profile,
        answers=answers,
        include_page_numbers=include_page_numbers,
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    successes = 0
    last_log = time.perf_counter()

    # Log progress at ~10% intervals (min every 100 rows, max every 500).
    _progress_step = max(100, min(500, count // 10 or 1))
    try:
        for idx, img_bytes, err in _iter_pngs_fast(payloads):
            row_number = source_rows[idx] if idx < len(source_rows) else idx + 1
            if err or img_bytes is None:
                errors.append(f"row {row_number}: {err or 'empty result'}")
                continue
            try:
                # JPEG bytes: read dimensions via fitz (avoids struct parsing JPEG SOF)
                tmp = fitz.open("jpeg", img_bytes)
                w, h = tmp[0].rect.width, tmp[0].rect.height
                tmp.close()
                page = doc.new_page(width=int(w), height=int(h))
                page.insert_image(page.rect, stream=img_bytes)
                successes += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"row {row_number}: {type(exc).__name__}: {exc}")
            # Progress log at ~10% intervals or every 30s
            now = time.perf_counter()
            if successes % _progress_step == 0 and successes > 0 or now - last_log > 30:
                rate = successes / (now - t_batch) * 60
                logger.info(
                    "Prefill PDF progress | %d/%d (%.0f/min) | err=%d",
                    successes, count, rate, len(errors),
                )
                last_log = now
        save_start = time.perf_counter()
        logger.info(
            "Prefill PDF final save started | pages=%d | target=%s",
            successes,
            dst_path,
        )
        if successes > 0:
            # Full garbage collection (garbage=4) is very expensive on thousands
            # of image-only pages and looks like a hang after rendering finishes.
            # These files are newly built, so a plain save is enough and much faster.
            doc.save(str(dst_path), garbage=0, deflate=False)
        else:
            dst_path.write_bytes(b"")
        logger.info(
            "Prefill PDF final save complete | pages=%d | elapsed=%.1fs",
            successes,
            time.perf_counter() - save_start,
        )
    finally:
        doc.close()

    elapsed = time.perf_counter() - t_batch
    size_bytes = dst_path.stat().st_size if dst_path.exists() else 0
    rate = (successes / elapsed) * 60 if elapsed > 0 else 0
    logger.info(
        "Prefill batch PDF complete | count=%d | ok=%d | err=%d | elapsed=%.1fs | "
        "rate=%.0f/min | size_mb=%.1f",
        count, successes, len(errors), elapsed, rate, size_bytes / (1024 * 1024),
    )
    if errors:
        preview = " | ".join(errors[:10])
        if len(errors) > 10:
            preview += f" | ...and {len(errors) - 10} more"
        logger.warning(
            "Prefill batch PDF row errors | showing=%d/%d | %s",
            min(len(errors), 10),
            len(errors),
            preview,
        )
    return {
        "count": count,
        "successes": successes,
        "errors": errors[:50],
        "elapsed_s": round(elapsed, 2),
        "size_bytes": size_bytes,
    }


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

# Accepted ``group_by`` values for the batch endpoint. ``none`` reproduces the
# original flat output; the others bundle rows into a ZIP-of-PDFs with one
# file per group.
GROUP_BY_VALUES = ("none", "school", "region", "region_school")
DEFAULT_GROUP_BY = "none"

# A control-character / path-traversal-safe pattern for group filenames.
_GROUP_NAME_BAD_CHARS = re.compile(r"[\\/:*?\"<>|\x00-\x1f\x7f]+")


def normalize_group_by(value: Any) -> str:
    """Return a validated ``group_by`` value or raise :class:`ValueError`."""
    text = (str(value) if value is not None else "").strip().lower()
    if not text:
        return DEFAULT_GROUP_BY
    if text not in GROUP_BY_VALUES:
        raise ValueError(
            f"Unknown group_by={value!r}. Must be one of: {', '.join(GROUP_BY_VALUES)}."
        )
    return text


def _safe_group_filename(name: str, *, fallback: str = "_ungrouped") -> str:
    """Coerce a row's school/region into a safe ZIP-entry filename component."""
    cleaned = _clean_field(name, max_len=80)
    cleaned = _GROUP_NAME_BAD_CHARS.sub("_", cleaned).strip(" ._-")
    if not cleaned:
        return fallback
    return cleaned


def _row_group_keys(row: dict[str, Any], group_by: str) -> tuple[str, ...]:
    """Return one or two normalised group keys for a row.

    ``school`` -> (school,)
    ``region`` -> (region,)
    ``region_school`` -> (region, school)  # nested ZIP path
    """
    school = _safe_group_filename(row.get("school_name", ""), fallback="_unknown_school")
    region = _safe_group_filename(row.get("region", ""), fallback="_unknown_region")
    if group_by == "school":
        return (school,)
    if group_by == "region":
        return (region,)
    if group_by == "region_school":
        return (region, school)
    raise ValueError(f"Cannot derive group keys for group_by={group_by!r}.")


def _group_rows(
    rows: list[dict[str, Any]], group_by: str
) -> dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]]:
    """Bucket ``(original_index, row)`` tuples by their group key.

    Insertion order is preserved within each bucket so that page numbering
    inside a group still follows CSV order.
    """
    groups: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = {}
    for idx, row in enumerate(rows):
        key = _row_group_keys(row, group_by)
        groups.setdefault(key, []).append((idx, row))
    return groups


def _group_zip_entry_name(key: tuple[str, ...]) -> str:
    """Translate a group key tuple into the ZIP entry path (always .pdf)."""
    parts = list(key)
    parts[-1] = f"{parts[-1]}.pdf"
    return "/".join(parts)


def generate_batch_grouped_zip_to_file(
    rows: list[dict[str, Any]],
    dst_path: Path,
    *,
    group_by: str,
    realism_preset: str = "none",
    include_page_numbers: bool = False,
    marking_profile: str = "none",
    answers: Any = None,
) -> dict:
    """Write a ZIP of one PDF per group to ``dst_path``.

    ``group_by`` must be one of ``"school"``, ``"region"``, ``"region_school"``.
    Page numbering, when requested, resets to 1 within each group so each
    group's PDF reads as a self-contained document.

    Returns ``{count, successes, errors, elapsed_s, size_bytes, groups}``
    where ``groups`` is a list of ``{name, count, successes}`` summaries.
    """
    group_by = normalize_group_by(group_by)
    if group_by == "none":
        raise ValueError(
            "generate_batch_grouped_zip_to_file requires a non-'none' group_by."
        )

    count = len(rows)
    logger.info(
        "Prefill grouped ZIP started | count=%d | group_by=%s | page_numbers=%s",
        count,
        group_by,
        include_page_numbers,
    )
    t_batch = time.perf_counter()

    grouped = _group_rows(rows, group_by)
    logger.info(
        "Prefill grouped ZIP groups | count=%d | groups=%d",
        count,
        len(grouped),
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    total_successes = 0
    errors: list[str] = []
    group_summaries: list[dict[str, Any]] = []

    with zipfile.ZipFile(
        dst_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as zf:
        for key, indexed_rows in grouped.items():
            entry_name = _group_zip_entry_name(key)
            group_rows = [
                {**row, "_source_row_number": original_index + 1}
                for original_index, row in indexed_rows
            ]
            with tempfile.NamedTemporaryFile(
                prefix="prefill_group_", suffix=".pdf", delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
            try:
                meta = generate_batch_pdf_to_file(
                    group_rows,
                    tmp_path,
                    realism_preset=realism_preset,
                    include_page_numbers=include_page_numbers,
                    marking_profile=marking_profile,
                    answers=answers,
                )
                if meta["successes"] > 0:
                    zf.writestr(entry_name, tmp_path.read_bytes())
                    total_successes += meta["successes"]
                else:
                    errors.append(f"group {entry_name!r}: all rows failed")
                group_summaries.append(
                    {
                        "name": entry_name,
                        "count": meta["count"],
                        "successes": meta["successes"],
                        "errors": meta["errors"],
                    }
                )
                if meta.get("errors"):
                    errors.extend(
                        f"group {entry_name!r}: {e}" for e in meta["errors"]
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"group {entry_name!r}: {type(exc).__name__}: {exc}")
                group_summaries.append(
                    {
                        "name": entry_name,
                        "count": len(group_rows),
                        "successes": 0,
                        "errors": [f"{type(exc).__name__}: {exc}"],
                    }
                )
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    elapsed = time.perf_counter() - t_batch
    size_bytes = dst_path.stat().st_size if dst_path.exists() else 0
    rate = (total_successes / elapsed) * 60 if elapsed > 0 else 0
    logger.info(
        "Prefill grouped ZIP complete | count=%d | groups=%d | ok=%d | err=%d | "
        "elapsed=%.1fs | rate=%.0f/min | size_mb=%.1f",
        count,
        len(grouped),
        total_successes,
        len(errors),
        elapsed,
        rate,
        size_bytes / (1024 * 1024),
    )
    return {
        "count": count,
        "successes": total_successes,
        "errors": errors[:50],
        "elapsed_s": round(elapsed, 2),
        "size_bytes": size_bytes,
        "groups": group_summaries,
    }


def generate_batch_zip_to_file(
    rows: list[dict[str, Any]],
    dst_path: Path,
    realism_preset: str = "none",
    marking_profile: str = "none",
    answers: Any = None,
) -> dict:
    """Stream ZIP generation directly to ``dst_path``. Bounded memory."""
    count = len(rows)
    logger.info("Prefill batch ZIP started | count=%d", count)
    t_batch = time.perf_counter()

    marking_profile = normalize_marking_profile(marking_profile or "none")
    payloads: list[dict[str, Any]] = []
    source_rows: list[int] = []
    filenames: list[str] = []
    errors: list[str] = []
    for i, row in enumerate(rows, start=1):
        source_row_number = int(row.get("_source_row_number") or i)
        filename = Path(_clean_field(row.get("output_file", "")) or "").name \
            or f"sheet_{i:03d}.png"
        if not filename.lower().endswith(".png"):
            filename += ".png"
        try:
            payload = _build_fast_payload(
                row,
                realism_preset=realism_preset,
                marking_profile=marking_profile,
                answers=answers,
            )
        except Exception as exc:  # noqa: BLE001 - skip bad rows, keep batch alive
            errors.append(
                f"row {source_row_number} ({filename}): {type(exc).__name__}: {exc}"
            )
            continue
        payloads.append(payload)
        source_rows.append(source_row_number)
        filenames.append(filename)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    successes = 0
    with zipfile.ZipFile(
        dst_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as zf:
        for idx, png_bytes, err in _iter_pngs_fast(payloads, preserve_order=False):
            row_number = source_rows[idx] if idx < len(source_rows) else idx + 1
            if err or png_bytes is None:
                errors.append(f"row {row_number} ({filenames[idx]}): {err or 'empty result'}")
                continue
            try:
                zf.writestr(filenames[idx], png_bytes)
                successes += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"row {row_number}: {type(exc).__name__}: {exc}")

    elapsed = time.perf_counter() - t_batch
    size_bytes = dst_path.stat().st_size if dst_path.exists() else 0
    rate = (successes / elapsed) * 60 if elapsed > 0 else 0
    logger.info(
        "Prefill batch ZIP complete | count=%d | ok=%d | err=%d | elapsed=%.1fs | "
        "rate=%.0f/min | size_mb=%.1f",
        count, successes, len(errors), elapsed, rate, size_bytes / (1024 * 1024),
    )
    if errors:
        preview = " | ".join(errors[:10])
        if len(errors) > 10:
            preview += f" | ...and {len(errors) - 10} more"
        logger.warning(
            "Prefill batch ZIP row errors | showing=%d/%d | %s",
            min(len(errors), 10),
            len(errors),
            preview,
        )
    return {
        "count": count,
        "successes": successes,
        "errors": errors[:50],
        "elapsed_s": round(elapsed, 2),
        "size_bytes": size_bytes,
    }


# Backwards-compatible in-memory wrappers (still used by older callers / tests).
# These now stream to a temp file first then read it back, so peak memory matches
# the streaming path even when the caller wants raw bytes.
def generate_batch_pdf(
    rows: list[dict[str, Any]],
    realism_preset: str = "none",
    include_page_numbers: bool = False,
    marking_profile: str = "none",
    answers: Any = None,
) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        generate_batch_pdf_to_file(
            rows,
            tmp_path,
            realism_preset=realism_preset,
            include_page_numbers=include_page_numbers,
            marking_profile=marking_profile,
            answers=answers,
        )
        return tmp_path.read_bytes()
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def generate_batch_zip(
    rows: list[dict[str, Any]],
    realism_preset: str = "none",
    marking_profile: str = "none",
    answers: Any = None,
) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        generate_batch_zip_to_file(
            rows,
            tmp_path,
            realism_preset=realism_preset,
            marking_profile=marking_profile,
            answers=answers,
        )
        return tmp_path.read_bytes()
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Size-aware PDF segmentation
# ---------------------------------------------------------------------------
#
# Real-world office MFPs (Xerox VersaLink C-series, Ricoh IM, Canon iR-ADV,
# Konica bizhub, HP LaserJet Enterprise) reliably accept print jobs up to
# ~50 MiB and ~500-999 pages but fail in opaque ways above those limits
# (Fault 016-751 on Xerox, PostScript ``limitcheck`` on HP, silent
# half-prints on Ricoh). The two helpers below let the prefill pipeline
# split an assembled PDF into bounded segments AFTER rendering. Splitting
# post-render means page numbers stay continuous across segments for free,
# because the underlying ``generate_batch_pdf_to_file`` already numbers
# pages by CSV order (1, 2, 3, ...).

# Output filename conventions for split segments. Embedding the part index
# AND total upfront ("part_02_of_07.pdf") rather than padding to a fixed
# width avoids the "01.pdf 02.pdf ... 10.pdf" lexicographic-vs-numeric
# sort confusion that breaks alphabetised file managers.
_PART_NAME_TEMPLATE = "{stem}_part_{index:02d}_of_{total:02d}.pdf"


def _format_part_name(stem: str, index: int, total: int) -> str:
    """Return the canonical filename for segment ``index`` of ``total``.

    ``stem`` is the base name without the ``.pdf`` extension. ``index`` is
    1-based. Single-segment outputs collapse to ``{stem}.pdf`` so the user
    sees a plain file rather than a confusing ``..._part_01_of_01.pdf``.
    """
    if total <= 1:
        return f"{stem}.pdf"
    return _PART_NAME_TEMPLATE.format(stem=stem, index=index, total=total)


def _chunk_ranges_for_split(
    page_byte_sizes: list[int],
    *,
    max_bytes: int,
    max_pages: int,
) -> list[tuple[int, int]]:
    """Return inclusive ``(start_index, end_index)`` page ranges for each segment.

    Walks the per-page byte sizes greedily and starts a new segment when
    adding the next page would exceed ``max_bytes`` OR the segment has
    already reached ``max_pages``. A single page larger than ``max_bytes``
    still occupies its own segment (we never produce a zero-page segment).
    """
    if not page_byte_sizes:
        return []
    ranges: list[tuple[int, int]] = []
    seg_start = 0
    seg_bytes = 0
    for i, page_bytes in enumerate(page_byte_sizes):
        seg_pages = i - seg_start + 1
        projected_bytes = seg_bytes + page_bytes
        too_big = projected_bytes > max_bytes and seg_pages > 1
        too_long = seg_pages > max_pages
        if too_big or too_long:
            ranges.append((seg_start, i - 1))
            seg_start = i
            seg_bytes = page_bytes
        else:
            seg_bytes = projected_bytes
    ranges.append((seg_start, len(page_byte_sizes) - 1))
    return ranges


def _split_pdf_into_segments(
    src_pdf_path: Path,
    *,
    max_pdf_mb: int,
    max_pdf_pages: int,
) -> list[tuple[Path, int, int, int]]:
    """Split ``src_pdf_path`` into temp PDFs each within both caps.

    Returns a list of ``(segment_path, first_page_number, last_page_number,
    size_bytes)`` tuples. ``first_page_number`` / ``last_page_number`` are
    1-based and refer to the original combined PDF, which preserves the
    continuous numbering already baked into each page by
    :func:`generate_batch_pdf_to_file`. The caller owns the returned temp
    files and is responsible for unlinking them.

    The single-segment case (source already within caps) returns the
    source path itself wrapped in a one-element list — the caller can
    treat that uniformly without special-casing the "no split needed"
    branch. The source file is never modified.
    """
    import fitz

    max_bytes = max(1, max_pdf_mb) * 1024 * 1024
    max_pages = max(1, max_pdf_pages)

    src = fitz.open(str(src_pdf_path))
    try:
        page_count = src.page_count
        if page_count == 0:
            return []
        total_size = src_pdf_path.stat().st_size
        if total_size <= max_bytes and page_count <= max_pages:
            return [(src_pdf_path, 1, page_count, total_size)]

        # We do not have per-page on-disk byte sizes inexpensively. The
        # PDF object stream is shared (xref table, fonts, images). For
        # segmentation it is enough to budget by AVERAGE page size; the
        # per-segment ``doc.save`` reports the real size which we
        # surface in the metadata. Using a constant per-page average
        # also makes the segment count deterministic and predictable
        # for tests, which is more valuable than chasing exact bytes.
        avg_per_page = max(1, math.ceil(total_size / page_count))
        page_byte_sizes = [avg_per_page] * page_count

        ranges = _chunk_ranges_for_split(
            page_byte_sizes,
            max_bytes=max_bytes,
            max_pages=max_pages,
        )

        results: list[tuple[Path, int, int, int]] = []
        for start, end in ranges:
            with tempfile.NamedTemporaryFile(
                prefix="prefill_segment_", suffix=".pdf", delete=False
            ) as tmp:
                seg_path = Path(tmp.name)
            seg_doc = fitz.open()
            try:
                seg_doc.insert_pdf(src, from_page=start, to_page=end)
                # Keep ``garbage=0 deflate=False`` to match the parent
                # save flags - we already paid the cost of building
                # bubble-free greyscale JPEG pages once; re-deflating
                # here for marginal savings would just add latency.
                seg_doc.save(str(seg_path), garbage=0, deflate=False)
            finally:
                seg_doc.close()
            size_bytes = seg_path.stat().st_size if seg_path.exists() else 0
            results.append((seg_path, start + 1, end + 1, size_bytes))
        return results
    finally:
        src.close()


def generate_batch_split_pdf_to_zip(
    rows: list[dict[str, Any]],
    dst_path: Path,
    *,
    max_pdf_mb: int,
    max_pdf_pages: int,
    realism_preset: str = "none",
    include_page_numbers: bool = False,
    marking_profile: str = "none",
    answers: Any = None,
    stem: str = "prefilled_sheets",
) -> dict:
    """Render a combined PDF then split it into ``<= cap`` segments inside a ZIP.

    The combined PDF is rendered exactly once via
    :func:`generate_batch_pdf_to_file` so page numbers (when
    ``include_page_numbers`` is true) are assigned in CSV order across the
    full batch. The post-render split preserves that ordering, which is
    why ``segment_02_of_05`` continues the numbering of
    ``segment_01_of_05`` rather than resetting to 1.

    Returns a metadata dict shaped like the other batch entry points,
    plus a ``segments`` list of ``{name, first_page, last_page, pages,
    size_bytes}`` rows describing each segment in the ZIP.
    """
    if max_pdf_mb < 1:
        raise ValueError("max_pdf_mb must be >= 1.")
    if max_pdf_pages < 1:
        raise ValueError("max_pdf_pages must be >= 1.")

    logger.info(
        "Prefill split PDF -> ZIP started | rows=%d | max_mb=%d | max_pages=%d",
        len(rows),
        max_pdf_mb,
        max_pdf_pages,
    )
    t_batch = time.perf_counter()

    with tempfile.NamedTemporaryFile(
        prefix="prefill_combined_", suffix=".pdf", delete=False
    ) as combined_tmp:
        combined_path = Path(combined_tmp.name)

    segment_paths: list[Path] = []
    try:
        combined_meta = generate_batch_pdf_to_file(
            rows,
            combined_path,
            realism_preset=realism_preset,
            include_page_numbers=include_page_numbers,
            marking_profile=marking_profile,
            answers=answers,
        )
        segments = _split_pdf_into_segments(
            combined_path,
            max_pdf_mb=max_pdf_mb,
            max_pdf_pages=max_pdf_pages,
        )

        # If a single segment came back AND it is the source path itself,
        # we still wrap it in the ZIP under the canonical single-part name
        # so the response format is predictable for the caller.
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        segment_summaries: list[dict[str, Any]] = []
        total = len(segments)
        with zipfile.ZipFile(
            dst_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as zf:
            for idx, (seg_path, first_page, last_page, size_bytes) in enumerate(
                segments, start=1
            ):
                entry_name = _format_part_name(stem, idx, total)
                zf.writestr(entry_name, seg_path.read_bytes())
                if seg_path != combined_path:
                    segment_paths.append(seg_path)
                segment_summaries.append(
                    {
                        "name": entry_name,
                        "first_page": first_page,
                        "last_page": last_page,
                        "pages": last_page - first_page + 1,
                        "size_bytes": size_bytes,
                    }
                )
    finally:
        # Always remove the combined temp file (we have either zipped
        # its bytes already or copied them into segment temps).
        try:
            combined_path.unlink(missing_ok=True)
        except OSError:
            pass
        for sp in segment_paths:
            try:
                sp.unlink(missing_ok=True)
            except OSError:
                pass

    elapsed = time.perf_counter() - t_batch
    size_bytes = dst_path.stat().st_size if dst_path.exists() else 0
    logger.info(
        "Prefill split PDF -> ZIP complete | rows=%d | segments=%d | "
        "elapsed=%.1fs | zip_size_mb=%.1f",
        combined_meta["count"],
        len(segment_summaries),
        elapsed,
        size_bytes / (1024 * 1024),
    )
    return {
        "count": combined_meta["count"],
        "successes": combined_meta["successes"],
        "errors": combined_meta["errors"],
        "elapsed_s": round(elapsed, 2),
        "size_bytes": size_bytes,
        "segments": segment_summaries,
    }


def generate_batch_grouped_split_zip_to_file(
    rows: list[dict[str, Any]],
    dst_path: Path,
    *,
    group_by: str,
    max_pdf_mb: int,
    max_pdf_pages: int,
    realism_preset: str = "none",
    include_page_numbers: bool = False,
    marking_profile: str = "none",
    answers: Any = None,
) -> dict:
    """Grouped output where each group's PDF is itself split into bounded segments.

    Behaves like :func:`generate_batch_grouped_zip_to_file` but every
    per-group PDF that exceeds the caps is post-split into ZIP entries of
    the form ``<group>_part_NN_of_TT.pdf``. Page numbering inside each
    group remains continuous across that group's segments because the
    per-group PDF is rendered once before splitting.
    """
    group_by = normalize_group_by(group_by)
    if group_by == "none":
        raise ValueError(
            "generate_batch_grouped_split_zip_to_file requires a non-'none' "
            "group_by."
        )
    if max_pdf_mb < 1:
        raise ValueError("max_pdf_mb must be >= 1.")
    if max_pdf_pages < 1:
        raise ValueError("max_pdf_pages must be >= 1.")

    count = len(rows)
    logger.info(
        "Prefill grouped+split ZIP started | count=%d | group_by=%s | "
        "max_mb=%d | max_pages=%d | page_numbers=%s",
        count,
        group_by,
        max_pdf_mb,
        max_pdf_pages,
        include_page_numbers,
    )
    t_batch = time.perf_counter()

    grouped = _group_rows(rows, group_by)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    total_successes = 0
    errors: list[str] = []
    group_summaries: list[dict[str, Any]] = []

    with zipfile.ZipFile(
        dst_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as zf:
        for key, indexed_rows in grouped.items():
            entry_base = _group_zip_entry_name(key)
            # Strip the trailing ``.pdf`` so we can interpolate split
            # suffixes; ``entry_base`` may also contain ``region/school``
            # which we want to preserve as a folder prefix in the ZIP.
            stem = entry_base[:-4] if entry_base.lower().endswith(".pdf") else entry_base
            group_rows = [
                {**row, "_source_row_number": original_index + 1}
                for original_index, row in indexed_rows
            ]

            with tempfile.NamedTemporaryFile(
                prefix="prefill_group_", suffix=".pdf", delete=False
            ) as tmp:
                group_pdf_path = Path(tmp.name)

            segment_paths: list[Path] = []
            try:
                group_meta = generate_batch_pdf_to_file(
                    group_rows,
                    group_pdf_path,
                    realism_preset=realism_preset,
                    include_page_numbers=include_page_numbers,
                    marking_profile=marking_profile,
                    answers=answers,
                )
                if group_meta["successes"] == 0:
                    errors.append(f"group {entry_base!r}: all rows failed")
                    if group_meta.get("errors"):
                        errors.extend(
                            f"group {entry_base!r}: {e}" for e in group_meta["errors"]
                        )
                    group_summaries.append(
                        {
                            "name": entry_base,
                            "count": group_meta["count"],
                            "successes": 0,
                            "errors": group_meta["errors"],
                            "segments": [],
                        }
                    )
                    continue

                segments = _split_pdf_into_segments(
                    group_pdf_path,
                    max_pdf_mb=max_pdf_mb,
                    max_pdf_pages=max_pdf_pages,
                )
                total = len(segments)
                segment_entries: list[dict[str, Any]] = []
                for idx, (seg_path, first_page, last_page, size_bytes) in enumerate(
                    segments, start=1
                ):
                    entry_name = _format_part_name(stem, idx, total)
                    zf.writestr(entry_name, seg_path.read_bytes())
                    if seg_path != group_pdf_path:
                        segment_paths.append(seg_path)
                    segment_entries.append(
                        {
                            "name": entry_name,
                            "first_page": first_page,
                            "last_page": last_page,
                            "pages": last_page - first_page + 1,
                            "size_bytes": size_bytes,
                        }
                    )
                total_successes += group_meta["successes"]
                group_summaries.append(
                    {
                        "name": entry_base,
                        "count": group_meta["count"],
                        "successes": group_meta["successes"],
                        "errors": group_meta["errors"],
                        "segments": segment_entries,
                    }
                )
                if group_meta.get("errors"):
                    errors.extend(
                        f"group {entry_base!r}: {e}" for e in group_meta["errors"]
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"group {entry_base!r}: {type(exc).__name__}: {exc}")
                group_summaries.append(
                    {
                        "name": entry_base,
                        "count": len(group_rows),
                        "successes": 0,
                        "errors": [f"{type(exc).__name__}: {exc}"],
                        "segments": [],
                    }
                )
            finally:
                try:
                    group_pdf_path.unlink(missing_ok=True)
                except OSError:
                    pass
                for sp in segment_paths:
                    try:
                        sp.unlink(missing_ok=True)
                    except OSError:
                        pass

    elapsed = time.perf_counter() - t_batch
    size_bytes = dst_path.stat().st_size if dst_path.exists() else 0
    logger.info(
        "Prefill grouped+split ZIP complete | count=%d | groups=%d | ok=%d | "
        "err=%d | elapsed=%.1fs | zip_size_mb=%.1f",
        count,
        len(grouped),
        total_successes,
        len(errors),
        elapsed,
        size_bytes / (1024 * 1024),
    )
    return {
        "count": count,
        "successes": total_successes,
        "errors": errors[:50],
        "elapsed_s": round(elapsed, 2),
        "size_bytes": size_bytes,
        "groups": group_summaries,
    }


# ---------------------------------------------------------------------------
# Blank-sheet bulk printing.
#
# Operators frequently need N physical copies of the *unfilled* answer
# sheet to hand out at exam venues — completely separate from the
# candidate-data-bubbled batch flow. This section turns that into a
# first-class feature: choose a variant, choose N, optionally number the
# pages and/or split the output into printer-safe segments. The source
# PDF is a 1-page asset bundled with the application (see
# :data:`BLANK_SHEET_VARIANTS`); we clone its page N times via
# :func:`fitz.Document.insert_pdf` so the embedded page raster is shared
# across all N pages and the output stays compact (~one image worth of
# bytes plus per-page xref overhead) regardless of N.
# ---------------------------------------------------------------------------


def normalize_blank_sheet_variant(value: Any) -> str:
    """Return a registered blank-sheet variant key or raise ``ValueError``.

    Accepts ``None``/empty strings as a request for the default variant
    so the API layer can treat the form field as optional. Anything else
    must match a key in :data:`BLANK_SHEET_VARIANTS` exactly.
    """
    if value is None:
        return DEFAULT_BLANK_SHEET_VARIANT
    text = str(value).strip()
    if not text:
        return DEFAULT_BLANK_SHEET_VARIANT
    text = normalize_preset_name(text)
    if text == LEGACY_LANDSCAPE:
        text = LANDSCAPE_NNQ25_0
    if text not in BLANK_SHEET_VARIANTS:
        allowed = ", ".join(sorted(BLANK_SHEET_VARIANTS))
        raise ValueError(
            f"Unknown blank sheet variant {text!r}. Allowed values: {allowed}."
        )
    return text


def list_blank_sheet_variants() -> list[dict[str, str]]:
    """Return the registered variants in a UI-friendly shape."""
    return [
        {"key": key, "label": meta["label"]}
        for key, meta in BLANK_SHEET_VARIANTS.items()
    ]


def _resolve_blank_sheet_source(variant: str) -> Path:
    """Return the on-disk source PDF for ``variant`` (validated and existing)."""
    meta = BLANK_SHEET_VARIANTS.get(variant)
    if meta is None:
        raise ValueError(f"Unknown blank sheet variant {variant!r}.")
    src: Path = meta["source_pdf"]
    if not src.exists():
        raise FileNotFoundError(
            f"Blank sheet source PDF for variant {variant!r} is missing at "
            f"{src}. Re-run the generator or check the PyInstaller spec "
            "bundles the source directory."
        )
    return src


def _stamp_page_number_on_pdf_page(page, page_number: int, total: int) -> None:
    """Stamp ``page_number / total`` in the bottom-right of a PyMuPDF page.

    Positioned to clear the bottom-right ArUco fiducial of the custom-25
    landscape sheet (marker centre ≈ x=782.5, y=593.9 pt; marker side
    ≈ 26 pt) by drawing the text at y_baseline = page_height - 12 pt
    (i.e. comfortably inside the bottom margin) and right-aligned with a
    32 pt right-edge margin so the text sits to the LEFT of the corner
    marker rather than overlapping it. Any correctly-laid-out US-letter
    answer sheet leaves the bottom-edge strip empty of bubbles, so this
    stays safe for future variants too.
    """
    import fitz  # PyMuPDF: already a hard dependency of this module

    text = f"{page_number} / {total}"
    fontname = "helv"
    fontsize = 10
    text_width = fitz.get_text_length(text, fontsize=fontsize, fontname=fontname)
    page_rect = page.rect
    x_right_margin = 32  # pt — inside the right edge, outside the corner marker
    y_baseline = page_rect.height - 12  # 12 pt up from the bottom edge
    x = max(0.0, page_rect.width - x_right_margin - text_width)
    page.insert_text(
        (x, y_baseline),
        text,
        fontsize=fontsize,
        fontname=fontname,
        color=(0, 0, 0),
    )


def generate_blank_sheets_pdf_to_file(
    dst_path: Path,
    *,
    variant: str = DEFAULT_BLANK_SHEET_VARIANT,
    count: int,
    include_page_numbers: bool = False,
) -> dict:
    """Render N copies of a blank answer-sheet variant into a single PDF.

    The source PDF is opened once and its single page is inserted into a
    fresh PyMuPDF document N times. ``garbage=4, deflate=True`` on save
    dedupes the shared embedded image so the output stays compact for
    large N (e.g. 5000 sheets stays in the low-MB range rather than
    growing linearly with N). Returns a metadata dict in the same shape
    as the batch helpers so the API layer can reuse its download-token
    plumbing unchanged.
    """
    import fitz

    if count < 1:
        raise ValueError("count must be >= 1.")
    variant = normalize_blank_sheet_variant(variant)
    src_path = _resolve_blank_sheet_source(variant)

    logger.info(
        "Blank sheet PDF started | variant=%s | count=%d | page_numbers=%s",
        variant, count, include_page_numbers,
    )
    t_batch = time.perf_counter()

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    src = fitz.open(str(src_path))
    try:
        if src.page_count < 1:
            raise ValueError(
                f"Blank sheet source PDF for variant {variant!r} has no pages."
            )
        out = fitz.open()
        try:
            for _ in range(count):
                out.insert_pdf(src, from_page=0, to_page=0)
            if include_page_numbers:
                for idx in range(count):
                    _stamp_page_number_on_pdf_page(
                        out.load_page(idx),
                        page_number=idx + 1,
                        total=count,
                    )
            out.save(str(dst_path), garbage=4, deflate=True)
        finally:
            out.close()
    finally:
        src.close()

    elapsed = time.perf_counter() - t_batch
    size_bytes = dst_path.stat().st_size if dst_path.exists() else 0
    logger.info(
        "Blank sheet PDF complete | variant=%s | count=%d | elapsed=%.1fs | size_mb=%.1f",
        variant, count, elapsed, size_bytes / (1024 * 1024),
    )
    return {
        "count": count,
        "successes": count,
        "errors": [],
        "elapsed_s": round(elapsed, 2),
        "size_bytes": size_bytes,
        "variant": variant,
    }


def generate_blank_sheets_split_zip_to_file(
    dst_path: Path,
    *,
    variant: str = DEFAULT_BLANK_SHEET_VARIANT,
    count: int,
    max_pdf_mb: int,
    max_pdf_pages: int,
    include_page_numbers: bool = False,
) -> dict:
    """Render N blank sheets and split into printer-safe ZIP segments.

    Builds the combined PDF first (so any page-numbering stays continuous
    across segments) and then reuses :func:`_split_pdf_into_segments`
    plus the existing ZIP writer used by the batch split path. Returns
    the same shape as :func:`generate_batch_split_pdf_to_zip`.
    """
    if max_pdf_mb < 1:
        raise ValueError("max_pdf_mb must be >= 1.")
    if max_pdf_pages < 1:
        raise ValueError("max_pdf_pages must be >= 1.")

    variant = normalize_blank_sheet_variant(variant)
    meta_variant = BLANK_SHEET_VARIANTS[variant]
    stem = meta_variant.get("default_stem", f"blank_{variant}")

    logger.info(
        "Blank sheet split ZIP started | variant=%s | count=%d | "
        "max_mb=%d | max_pages=%d | page_numbers=%s",
        variant, count, max_pdf_mb, max_pdf_pages, include_page_numbers,
    )
    t_batch = time.perf_counter()

    with tempfile.NamedTemporaryFile(
        prefix="blank_combined_", suffix=".pdf", delete=False
    ) as combined_tmp:
        combined_path = Path(combined_tmp.name)

    segment_paths: list[Path] = []
    try:
        combined_meta = generate_blank_sheets_pdf_to_file(
            combined_path,
            variant=variant,
            count=count,
            include_page_numbers=include_page_numbers,
        )
        segments = _split_pdf_into_segments(
            combined_path,
            max_pdf_mb=max_pdf_mb,
            max_pdf_pages=max_pdf_pages,
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        segment_summaries: list[dict[str, Any]] = []
        total = len(segments)
        with zipfile.ZipFile(
            dst_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as zf:
            for idx, (seg_path, first_page, last_page, size_bytes) in enumerate(
                segments, start=1
            ):
                entry_name = _format_part_name(stem, idx, total)
                zf.writestr(entry_name, seg_path.read_bytes())
                if seg_path != combined_path:
                    segment_paths.append(seg_path)
                segment_summaries.append(
                    {
                        "name": entry_name,
                        "first_page": first_page,
                        "last_page": last_page,
                        "pages": last_page - first_page + 1,
                        "size_bytes": size_bytes,
                    }
                )
    finally:
        try:
            combined_path.unlink(missing_ok=True)
        except OSError:
            pass
        for sp in segment_paths:
            try:
                sp.unlink(missing_ok=True)
            except OSError:
                pass

    elapsed = time.perf_counter() - t_batch
    size_bytes = dst_path.stat().st_size if dst_path.exists() else 0
    logger.info(
        "Blank sheet split ZIP complete | variant=%s | count=%d | "
        "segments=%d | elapsed=%.1fs | zip_size_mb=%.1f",
        variant, combined_meta["count"], len(segment_summaries),
        elapsed, size_bytes / (1024 * 1024),
    )
    return {
        "count": combined_meta["count"],
        "successes": combined_meta["successes"],
        "errors": combined_meta["errors"],
        "elapsed_s": round(elapsed, 2),
        "size_bytes": size_bytes,
        "variant": variant,
        "segments": segment_summaries,
    }
