"""Service layer for the prefill answer-sheet feature.

Wraps ``prefill_only_package.prefill_answer_sheet_final`` and returns
in-memory bytes so HTTP handlers can stream them directly.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

# Built-in blank template shipped with the package
DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "prefill_only_package"
    / "blank_template_reference.png"
)


def _import_prefill():
    """Lazy import to avoid loading PIL at module-level if not needed."""
    from prefill_only_package.prefill_answer_sheet_final import prefill_sheet

    return prefill_sheet


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
    prefill_sheet = _import_prefill()
    images = []
    for row in rows:
        _validate_candidate_number(str(row["candidate_number"]).strip())
        image = prefill_sheet(
            DEFAULT_TEMPLATE,
            str(row["student_name"]).strip(),
            str(row["school_name"]).strip(),
            str(row["exam_name"]).strip(),
            str(row["candidate_number"]).strip(),
        )
        images.append(image)
    return _images_to_pdf_bytes(images)


def generate_batch_zip(rows: list[dict[str, Any]]) -> bytes:
    prefill_sheet = _import_prefill()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, row in enumerate(rows, start=1):
            _validate_candidate_number(str(row["candidate_number"]).strip())
            image = prefill_sheet(
                DEFAULT_TEMPLATE,
                str(row["student_name"]).strip(),
                str(row["school_name"]).strip(),
                str(row["exam_name"]).strip(),
                str(row["candidate_number"]).strip(),
            )
            filename = Path((str(row.get("output_file") or "").strip())).name or f"sheet_{i:03d}.png"
            if not filename.lower().endswith(".png"):
                filename += ".png"
            img_buf = io.BytesIO()
            image.save(img_buf, format="PNG")
            zf.writestr(filename, img_buf.getvalue())
    return buf.getvalue()
