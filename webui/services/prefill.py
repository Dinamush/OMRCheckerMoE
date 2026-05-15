"""Service layer for the prefill answer-sheet feature.

Wraps ``prefill_only_package.prefill_answer_sheet_final`` and returns
in-memory bytes so HTTP handlers can stream them directly.
"""

from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

# Built-in blank template shipped with the package
DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "prefill_only_package"
    / "blank_template_reference.png"
)

logger = logging.getLogger(__name__)


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


def generate_single_png(
    student_name: str,
    school_name: str,
    exam_name: str,
    candidate_number: str,
) -> bytes:
    _validate_candidate_number(candidate_number)
    prefill_sheet = _import_prefill()
    image = prefill_sheet(DEFAULT_TEMPLATE, student_name, school_name, exam_name, candidate_number)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def generate_single_pdf(
    student_name: str,
    school_name: str,
    exam_name: str,
    candidate_number: str,
) -> bytes:
    _validate_candidate_number(candidate_number)
    prefill_sheet = _import_prefill()
    image = prefill_sheet(DEFAULT_TEMPLATE, student_name, school_name, exam_name, candidate_number)
    return _images_to_pdf_bytes([image])


def generate_batch_pdf(rows: list[dict[str, Any]]) -> bytes:
    m = _import_prefill_module()
    count = len(rows)
    logger.info("Prefill batch started | count=%d output=pdf", count)
    t_batch = time.perf_counter()

    # Pre-stamp template once — biggest per-sheet speedup
    t_stamp = time.perf_counter()
    stamped_img = m.load_stamped_template(DEFAULT_TEMPLATE)
    stamped_buf = io.BytesIO()
    stamped_img.save(stamped_buf, format="PNG", compress_level=1)
    stamped_bytes = stamped_buf.getvalue()
    logger.info("Template stamped | %.1fms", (time.perf_counter() - t_stamp) * 1000)

    payloads: list[dict] = []
    for row in rows:
        _validate_candidate_number(str(row["candidate_number"]).strip())
        payloads.append({
            "stamped_bytes": stamped_bytes,
            "student_name": str(row["student_name"]).strip(),
            "school_name": str(row["school_name"]).strip(),
            "exam_name": str(row["exam_name"]).strip(),
            "candidate_number": str(row["candidate_number"]).strip(),
        })

    max_workers = max(1, min((os.cpu_count() or 2) - 1, 8))
    logger.info("Parallel workers started | workers=%d", max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        png_bytes_list = list(executor.map(m._prefill_worker, payloads, chunksize=20))

    pdf_bytes = _images_to_pdf_bytes_fast(png_bytes_list)
    elapsed = time.perf_counter() - t_batch
    rate = (count / elapsed) * 60 if elapsed > 0 else 0
    size_mb = len(pdf_bytes) / (1024 * 1024)
    logger.info("Prefill batch complete | count=%d | elapsed=%.1fs | rate=%.0f/min | size_mb=%.1f", count, elapsed, rate, size_mb)
    return pdf_bytes


def generate_batch_zip(rows: list[dict[str, Any]]) -> bytes:
    m = _import_prefill_module()
    count = len(rows)
    logger.info("Prefill batch started | count=%d output=zip", count)
    t_batch = time.perf_counter()

    # Pre-stamp template once
    t_stamp = time.perf_counter()
    stamped_img = m.load_stamped_template(DEFAULT_TEMPLATE)
    stamped_buf = io.BytesIO()
    stamped_img.save(stamped_buf, format="PNG", compress_level=1)
    stamped_bytes = stamped_buf.getvalue()
    logger.info("Template stamped | %.1fms", (time.perf_counter() - t_stamp) * 1000)

    payloads: list[dict] = []
    filenames: list[str] = []
    for i, row in enumerate(rows, start=1):
        _validate_candidate_number(str(row["candidate_number"]).strip())
        payloads.append({
            "stamped_bytes": stamped_bytes,
            "student_name": str(row["student_name"]).strip(),
            "school_name": str(row["school_name"]).strip(),
            "exam_name": str(row["exam_name"]).strip(),
            "candidate_number": str(row["candidate_number"]).strip(),
        })
        filename = Path((str(row.get("output_file") or "").strip())).name or f"sheet_{i:03d}.png"
        if not filename.lower().endswith(".png"):
            filename += ".png"
        filenames.append(filename)

    max_workers = max(1, min((os.cpu_count() or 2) - 1, 8))
    logger.info("Parallel workers started | workers=%d", max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        png_bytes_list = list(executor.map(m._prefill_worker, payloads, chunksize=20))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, png_bytes in zip(filenames, png_bytes_list):
            zf.writestr(filename, png_bytes)
    zip_bytes = buf.getvalue()
    elapsed = time.perf_counter() - t_batch
    rate = (count / elapsed) * 60 if elapsed > 0 else 0
    size_mb = len(zip_bytes) / (1024 * 1024)
    logger.info("Prefill batch complete | count=%d | elapsed=%.1fs | rate=%.0f/min | size_mb=%.1f", count, elapsed, rate, size_mb)
    return zip_bytes
