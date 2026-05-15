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

import numpy as np

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
    """Validate + sanitise a row into a legacy worker payload (includes stamped_bytes)."""
    candidate_number = _clean_field(row.get("candidate_number"), max_len=10)
    _validate_candidate_number(candidate_number)
    return {
        "stamped_bytes": stamped_bytes,
        "student_name": _clean_field(row.get("student_name")),
        "school_name": _clean_field(row.get("school_name")),
        "exam_name": _clean_field(row.get("exam_name")),
        "candidate_number": candidate_number,
    }


def _build_fast_payload(row: dict[str, Any], output_format: str = "png") -> dict[str, Any]:
    """Validate + sanitise a row into a fast worker payload (no stamped_bytes)."""
    candidate_number = _clean_field(row.get("candidate_number"), max_len=10)
    _validate_candidate_number(candidate_number)
    return {
        "student_name": _clean_field(row.get("student_name")),
        "school_name": _clean_field(row.get("school_name")),
        "exam_name": _clean_field(row.get("exam_name")),
        "candidate_number": candidate_number,
        "output_format": output_format,
    }


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
        m = _import_prefill_module()
        t = time.perf_counter()
        _STAMPED_IMG_CACHE = m.load_stamped_template(DEFAULT_TEMPLATE)
        _STAMPED_ARR_CACHE = np.array(_STAMPED_IMG_CACHE)
        logger.info("Stamped template cached | %.1fms", (time.perf_counter() - t) * 1000)
    return _STAMPED_IMG_CACHE, _STAMPED_ARR_CACHE


def _stamp_template_once() -> bytes:
    """Render the ArUco-stamped template once, returning its PNG bytes."""
    stamped_img, _ = _get_stamped_img()
    stamped_buf = io.BytesIO()
    stamped_img.save(stamped_buf, format="PNG", compress_level=1)
    return stamped_buf.getvalue()


def _max_workers() -> int:
    return max(1, min((os.cpu_count() or 2) - 1, 8))


def _iter_pngs_fast(payloads: list[dict]):
    """Yield ``(index, png_bytes_or_None, error_or_None)`` using the fast worker.

    Each worker process receives the stamped template as a numpy array via the
    pool initializer (once per worker process), so task payloads carry only the
    small per-student text fields.  Falls back to serial on BrokenExecutor.
    """
    import numpy as np
    m = _import_prefill_module()
    _, stamped_arr = _get_stamped_img()
    arr_bytes = stamped_arr.tobytes()
    shape = stamped_arr.shape
    n = len(payloads)
    # Larger chunksize reduces IPC round-trips; cap so short batches still parallelise.
    chunksize = max(1, min(200, n // (_max_workers() * 4) + 1))
    yielded = 0
    try:
        with ProcessPoolExecutor(
            max_workers=_max_workers(),
            initializer=m._init_worker_stamped,
            initargs=(arr_bytes, shape),
        ) as ex:
            for png_bytes in ex.map(m._prefill_worker_fast, payloads, chunksize=chunksize):
                yield yielded, png_bytes, None
                yielded += 1
    except BrokenExecutor as exc:
        logger.warning(
            "Fast worker pool broken after %d/%d rows; falling back to serial: %s",
            yielded, n, exc,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Fast worker pool error after %d/%d rows; falling back to serial: %s",
            yielded, n, exc,
        )

    # Serial fallback for remaining.
    for idx in range(yielded, n):
        try:
            # Re-use stamped img directly in-process to avoid another pool.
            stamped_img, _ = _get_stamped_img()
            img = stamped_img.copy()
            m2 = _import_prefill_module()
            img = m2._draw_sheet_content(
                img,
                payloads[idx]['student_name'],
                payloads[idx]['school_name'],
                payloads[idx]['exam_name'],
                payloads[idx]['candidate_number'],
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
) -> bytes:
    candidate_number = _clean_field(candidate_number, max_len=10)
    _validate_candidate_number(candidate_number)
    m = _import_prefill_module()
    stamped_img, _ = _get_stamped_img()
    image = m._draw_sheet_content(
        stamped_img.copy(),
        _clean_field(student_name),
        _clean_field(school_name),
        _clean_field(exam_name),
        candidate_number,
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def generate_single_pdf(
    student_name: str,
    school_name: str,
    exam_name: str,
    candidate_number: str,
) -> bytes:
    import fitz
    import struct
    candidate_number = _clean_field(candidate_number, max_len=10)
    _validate_candidate_number(candidate_number)
    png_bytes = generate_single_png(student_name, school_name, exam_name, candidate_number)
    w, h = struct.unpack('>II', png_bytes[16:24])
    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    page.insert_image(page.rect, stream=png_bytes)
    buf = io.BytesIO()
    doc.save(buf, garbage=0)
    doc.close()
    return buf.getvalue()


def generate_batch_pdf_to_file(rows: list[dict[str, Any]], dst_path: Path) -> dict:
    """Stream PDF generation directly to ``dst_path``.

    Workers output JPEG bytes; JPEG is stored natively in PDF as DCT so no
    re-encoding or deflate pass is needed.  Progress is logged every 500 sheets.
    Returns a metadata dict: ``{count, successes, errors, elapsed_s, size_bytes}``.
    """
    import fitz  # PyMuPDF

    count = len(rows)
    logger.info("Prefill batch PDF started | count=%d", count)
    t_batch = time.perf_counter()

    payloads = [_build_fast_payload(row, output_format="jpeg") for row in rows]

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    successes = 0
    errors: list[str] = []
    last_log = time.perf_counter()

    try:
        for idx, img_bytes, err in _iter_pngs_fast(payloads):
            if err or img_bytes is None:
                errors.append(f"row {idx}: {err or 'empty result'}")
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
                errors.append(f"row {idx}: {type(exc).__name__}: {exc}")
            # Progress log every ~500 sheets or every 30s
            now = time.perf_counter()
            if successes % 500 == 0 and successes > 0 or now - last_log > 30:
                rate = successes / (now - t_batch) * 60
                logger.info(
                    "Prefill PDF progress | %d/%d (%.0f/min) | err=%d",
                    successes, count, rate, len(errors),
                )
                last_log = now
        doc.save(str(dst_path), garbage=4)
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
    logger.info("Prefill batch ZIP started | count=%d", count)
    t_batch = time.perf_counter()

    payloads: list[dict] = []
    filenames: list[str] = []
    for i, row in enumerate(rows, start=1):
        payloads.append(_build_fast_payload(row))
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
        for idx, png_bytes, err in _iter_pngs_fast(payloads):
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
        "Prefill batch ZIP complete | count=%d | ok=%d | err=%d | elapsed=%.1fs | "
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
