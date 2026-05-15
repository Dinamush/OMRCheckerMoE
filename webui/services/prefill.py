"""Service layer for the prefill answer-sheet feature.

Wraps ``prefill_only_package.prefill_answer_sheet_final``. Batch outputs are
streamed directly to a temp file on disk so peak memory is bounded regardless
of row count (a 5k-row PDF must not OOM the server).
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import time
import zipfile
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import Any

# Built-in blank template shipped with the package
DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "prefill_only_package"
    / "blank_template_reference.png"
)

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


def _import_prefill():
    """Lazy import to avoid loading PIL at module-level if not needed."""
    from prefill_only_package.prefill_answer_sheet_final import prefill_sheet

    return prefill_sheet


def _import_prefill_module():
    """Lazy import of the full prefill module (needed for batch helpers)."""
    import prefill_only_package.prefill_answer_sheet_final as m

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
        raise ValueError("Candidate number must be exactly 10 digits.")


def _build_payload(stamped_bytes: bytes, row: dict[str, Any]) -> dict[str, Any]:
    """Validate + sanitise a row into a worker payload."""
    candidate_number = _clean_field(row.get("candidate_number"), max_len=10)
    _validate_candidate_number(candidate_number)
    return {
        "stamped_bytes": stamped_bytes,
        "student_name": _clean_field(row.get("student_name")),
        "school_name": _clean_field(row.get("school_name")),
        "exam_name": _clean_field(row.get("exam_name")),
        "candidate_number": candidate_number,
    }


def _stamp_template_once() -> bytes:
    """Render the ArUco-stamped template once, returning its PNG bytes."""
    m = _import_prefill_module()
    t_stamp = time.perf_counter()
    stamped_img = m.load_stamped_template(DEFAULT_TEMPLATE)
    stamped_buf = io.BytesIO()
    stamped_img.save(stamped_buf, format="PNG", compress_level=1)
    stamped_bytes = stamped_buf.getvalue()
    logger.info("Template stamped | %.1fms", (time.perf_counter() - t_stamp) * 1000)
    return stamped_bytes


def _max_workers() -> int:
    return max(1, min((os.cpu_count() or 2) - 1, 8))


def _iter_pngs_with_fallback(payloads: list[dict]):
    """Yield ``(index, png_bytes_or_None, error_or_None)`` for each payload.

    Uses a process pool with chunked map for throughput. If the pool dies
    (``BrokenExecutor``), falls back to in-process serial processing for the
    remaining payloads so a single bad input cannot kill the whole batch.
    """
    m = _import_prefill_module()
    n = len(payloads)
    yielded = 0
    try:
        with ProcessPoolExecutor(max_workers=_max_workers()) as ex:
            for png_bytes in ex.map(m._prefill_worker, payloads, chunksize=20):
                yield yielded, png_bytes, None
                yielded += 1
    except BrokenExecutor as exc:
        logger.warning(
            "Prefill worker pool broken after %d/%d rows; falling back to serial: %s",
            yielded, n, exc,
        )
    except Exception as exc:  # noqa: BLE001 - graceful: any worker fault → serial
        logger.warning(
            "Prefill worker pool error after %d/%d rows; falling back to serial: %s",
            yielded, n, exc,
        )

    # Serial fallback for any remaining payloads.
    for idx in range(yielded, n):
        try:
            png_bytes = m._prefill_worker(payloads[idx])
            yield idx, png_bytes, None
        except Exception as inner:  # noqa: BLE001 - never abort whole batch
            yield idx, None, f"{type(inner).__name__}: {inner}"


def generate_single_png(
    student_name: str,
    school_name: str,
    exam_name: str,
    candidate_number: str,
) -> bytes:
    candidate_number = _clean_field(candidate_number, max_len=10)
    _validate_candidate_number(candidate_number)
    prefill_sheet = _import_prefill()
    image = prefill_sheet(
        DEFAULT_TEMPLATE,
        _clean_field(student_name),
        _clean_field(school_name),
        _clean_field(exam_name),
        candidate_number,
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def generate_single_pdf(
    student_name: str,
    school_name: str,
    exam_name: str,
    candidate_number: str,
) -> bytes:
    candidate_number = _clean_field(candidate_number, max_len=10)
    _validate_candidate_number(candidate_number)
    prefill_sheet = _import_prefill()
    image = prefill_sheet(
        DEFAULT_TEMPLATE,
        _clean_field(student_name),
        _clean_field(school_name),
        _clean_field(exam_name),
        candidate_number,
    )
    return _images_to_pdf_bytes([image])


def generate_batch_pdf_to_file(rows: list[dict[str, Any]], dst_path: Path) -> dict:
    """Stream PDF generation directly to ``dst_path``.

    Memory peak is one PNG (~5MB) + the open PyMuPDF doc (incremental).
    Returns a metadata dict: ``{count, successes, errors, elapsed_s, size_bytes}``.
    Caller must validate row count beforehand.
    """
    import fitz  # PyMuPDF

    count = len(rows)
    logger.info("Prefill batch started (streaming) | count=%d output=pdf", count)
    t_batch = time.perf_counter()

    stamped_bytes = _stamp_template_once()
    payloads = [_build_payload(stamped_bytes, row) for row in rows]

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    successes = 0
    errors: list[str] = []

    try:
        for idx, png_bytes, err in _iter_pngs_with_fallback(payloads):
            if err or png_bytes is None:
                errors.append(f"row {idx}: {err or 'empty result'}")
                continue
            try:
                img_doc = fitz.open("png", png_bytes)
                pdf_bytes = img_doc.convert_to_pdf()
                img_doc.close()
                src = fitz.open("pdf", pdf_bytes)
                doc.insert_pdf(src)
                src.close()
                successes += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"row {idx}: {type(exc).__name__}: {exc}")
        doc.save(str(dst_path), garbage=4, deflate=True)
    finally:
        doc.close()

    elapsed = time.perf_counter() - t_batch
    size_bytes = dst_path.stat().st_size if dst_path.exists() else 0
    rate = (successes / elapsed) * 60 if elapsed > 0 else 0
    logger.info(
        "Prefill batch complete | count=%d | ok=%d | err=%d | elapsed=%.1fs | "
        "rate=%.0f/min | size_mb=%.1f",
        count, successes, len(errors), elapsed, rate, size_bytes / (1024 * 1024),
    )
    return {
        "count": count,
        "successes": successes,
        "errors": errors[:50],
        "elapsed_s": round(elapsed, 2),
        "size_bytes": size_bytes,
    }


def generate_batch_zip_to_file(rows: list[dict[str, Any]], dst_path: Path) -> dict:
    """Stream ZIP generation directly to ``dst_path``. Bounded memory."""
    count = len(rows)
    logger.info("Prefill batch started (streaming) | count=%d output=zip", count)
    t_batch = time.perf_counter()

    stamped_bytes = _stamp_template_once()
    payloads: list[dict] = []
    filenames: list[str] = []
    for i, row in enumerate(rows, start=1):
        payloads.append(_build_payload(stamped_bytes, row))
        filename = Path(_clean_field(row.get("output_file", "")) or "").name \
            or f"sheet_{i:03d}.png"
        if not filename.lower().endswith(".png"):
            filename += ".png"
        filenames.append(filename)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    successes = 0
    errors: list[str] = []
    with zipfile.ZipFile(
        dst_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as zf:
        for idx, png_bytes, err in _iter_pngs_with_fallback(payloads):
            if err or png_bytes is None:
                errors.append(f"row {idx} ({filenames[idx]}): {err or 'empty result'}")
                continue
            try:
                zf.writestr(filenames[idx], png_bytes)
                successes += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"row {idx}: {type(exc).__name__}: {exc}")

    elapsed = time.perf_counter() - t_batch
    size_bytes = dst_path.stat().st_size if dst_path.exists() else 0
    rate = (successes / elapsed) * 60 if elapsed > 0 else 0
    logger.info(
        "Prefill batch complete | count=%d | ok=%d | err=%d | elapsed=%.1fs | "
        "rate=%.0f/min | size_mb=%.1f",
        count, successes, len(errors), elapsed, rate, size_bytes / (1024 * 1024),
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
def generate_batch_pdf(rows: list[dict[str, Any]]) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        generate_batch_pdf_to_file(rows, tmp_path)
        return tmp_path.read_bytes()
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def generate_batch_zip(rows: list[dict[str, Any]]) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        generate_batch_zip_to_file(rows, tmp_path)
        return tmp_path.read_bytes()
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
