"""Prove the path-based split survives a >2 GiB PDF (no MuPDF stream crash).

Renders only the first few pages so it finishes quickly while exercising the
exact code path the upload handler now uses.

Usage: python scripts/diagnostics/probe_large_pdf.py "<path-to-pdf>" [num_pages]
"""
from __future__ import annotations

import functools
import sys
import tempfile
import time
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001


def main() -> None:
    pdf = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    size = pdf.stat().st_size
    print(f"file={pdf.name} size={size/1024/1024:.0f} MiB ({size} bytes)")

    from webui.services.batches import (
        _save_pdf_pages_parallel,
        _save_pdf_pages_serial,
    )

    with tempfile.TemporaryDirectory() as td:
        inputs = Path(td) / "inputs"
        inputs.mkdir()

        # Open by path (memory-mapped) — this is what staging gives us.
        t0 = time.perf_counter()
        serial_stored, serial_failed = _save_pdf_pages_serial(
            inputs=inputs,
            safe_filename="big.pdf",
            pdf_path=str(pdf),
            stem="big",
            page_count=n,
            dpi=150,
            grayscale=True,
            ext=".jpg",
            jpeg_quality=90,
            batch_id=None,
            settings=None,
        )
        print(
            f"[{time.perf_counter()-t0:5.1f}s] SERIAL ok="
            f"{len(serial_stored)} failed={serial_failed}"
        )

        inputs2 = Path(td) / "inputs2"
        inputs2.mkdir()
        t0 = time.perf_counter()
        par_stored, par_failed = _save_pdf_pages_parallel(
            inputs=inputs2,
            safe_filename="big.pdf",
            pdf_path=str(pdf),
            stem="big",
            page_count=n,
            dpi=150,
            grayscale=True,
            ext=".jpg",
            jpeg_quality=90,
            workers=2,
            batch_id=None,
            settings=None,
        )
        print(
            f"[{time.perf_counter()-t0:5.1f}s] PARALLEL ok="
            f"{len(par_stored)} failed={par_failed}"
        )

    print("PASS: >2 GiB PDF rendered via path-open with no crash.")


if __name__ == "__main__":
    main()
