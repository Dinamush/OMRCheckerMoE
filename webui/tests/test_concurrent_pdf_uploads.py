"""Tests for concurrent same-stem PDF uploads (Fix #3).

Verifies that two PDFs uploaded simultaneously with the same filename to the
same batch do not clobber each other's rendered page images.

Strategy: Option A (per-(batch_id, stem) threading.Lock) is in place inside
_save_pdf_pages_as_images.  These tests exercise both the concurrent path and
a simple serial sanity-check.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_batch(client: TestClient, name: str = "Concurrent PDF test") -> str:
    resp = client.post("/api/v1/batches", json={"name": name})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _make_pdf_bytes(page_count: int, width: int = 210, height: int = 297) -> bytes:
    """Build a minimal fitz PDF with *page_count* blank pages."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page(width=width, height=height)
        page.insert_text((72, 140), f"Page {i + 1} ({width}x{height})")
    data = doc.tobytes()
    doc.close()
    return data


def _upload_pdf_sync(
    client: TestClient,
    batch_id: str,
    filename: str,
    content: bytes,
    results: list,
    index: int,
) -> None:
    """Thread target: upload *content* as *filename* and store the response."""
    resp = client.post(
        f"/api/v1/batches/{batch_id}/files",
        files=[("files", (filename, content, "application/pdf"))],
    )
    results[index] = resp


# ---------------------------------------------------------------------------
# Serial sanity-check: two sequential same-stem uploads yield pages from the
# second upload only (the first is replaced) — existing behaviour is preserved.
# ---------------------------------------------------------------------------

def test_serial_same_stem_second_upload_replaces_first(client: TestClient) -> None:
    """Serial re-upload of the same PDF stem replaces the first set of pages."""
    pytest.importorskip("fitz")
    batch_id = _create_batch(client, "Serial same-stem test")

    pdf_a = _make_pdf_bytes(3, width=210, height=297)
    pdf_b = _make_pdf_bytes(5, width=210, height=297)

    for pdf_bytes in (pdf_a, pdf_b):
        resp = client.post(
            f"/api/v1/batches/{batch_id}/files",
            files=[("files", ("scans.pdf", pdf_bytes, "application/pdf"))],
        )
        assert resp.status_code == 202, resp.text

    files_resp = client.get(f"/api/v1/batches/{batch_id}/files")
    assert files_resp.status_code == 200
    files = files_resp.json()
    # The second upload (5 pages) replaced the first (3 pages).
    assert len(files) == 5, (
        f"Expected 5 pages after second upload replaced first; got {len(files)}"
    )
    names = [f["name"] for f in files]
    assert "scans_page_0005.jpg" in names
    # No duplicate names.
    assert len(names) == len(set(names)), f"Duplicate filenames detected: {names}"


# ---------------------------------------------------------------------------
# Concurrent upload: two *distinct* PDFs with the same filename to the same
# batch.  The lock serialises them so the output is one complete set of pages
# (whichever ran second wins), not an interleaved corrupt mix.
# ---------------------------------------------------------------------------

def test_concurrent_same_stem_no_collision(client: TestClient) -> None:
    """Concurrent same-stem PDFs must not produce duplicate or missing pages."""
    pytest.importorskip("fitz")
    batch_id = _create_batch(client, "Concurrent same-stem test")

    # Use different page counts so we can tell which upload won.
    pdf_5 = _make_pdf_bytes(5, width=200, height=280)
    pdf_7 = _make_pdf_bytes(7, width=200, height=280)

    results: list = [None, None]

    t1 = threading.Thread(
        target=_upload_pdf_sync,
        args=(client, batch_id, "scans.pdf", pdf_5, results, 0),
    )
    t2 = threading.Thread(
        target=_upload_pdf_sync,
        args=(client, batch_id, "scans.pdf", pdf_7, results, 1),
    )

    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert results[0] is not None and results[1] is not None, (
        "One or both upload threads did not complete in time"
    )
    assert results[0].status_code == 202, results[0].text
    assert results[1].status_code == 202, results[1].text

    # Allow background tasks a moment to settle (TestClient runs them
    # synchronously within each request, but both requests have already
    # returned their 202 responses by the time we reach here).
    time.sleep(0.5)

    files_resp = client.get(f"/api/v1/batches/{batch_id}/files")
    assert files_resp.status_code == 200
    files = files_resp.json()
    names = [f["name"] for f in files]

    # No duplicate filenames (page clobbering would produce identical names).
    assert len(names) == len(set(names)), (
        f"Duplicate page filenames detected — concurrent clobbering occurred: {names}"
    )

    # The lock ensures exactly one complete set of pages landed (either 5 or 7).
    assert len(files) in {5, 7}, (
        f"Expected 5 or 7 pages (one complete set); got {len(files)}: {names}"
    )


# ---------------------------------------------------------------------------
# Concurrent upload of *different* stems must run fully in parallel (each
# should produce its own page set without interfering with the other).
# ---------------------------------------------------------------------------

def test_concurrent_different_stems_both_complete(client: TestClient) -> None:
    """Two PDFs with different stems uploaded concurrently must both produce pages."""
    pytest.importorskip("fitz")
    batch_id = _create_batch(client, "Concurrent different-stem test")

    pdf_a = _make_pdf_bytes(3, width=210, height=297)
    pdf_b = _make_pdf_bytes(4, width=148, height=210)

    results: list = [None, None]

    t1 = threading.Thread(
        target=_upload_pdf_sync,
        args=(client, batch_id, "alpha.pdf", pdf_a, results, 0),
    )
    t2 = threading.Thread(
        target=_upload_pdf_sync,
        args=(client, batch_id, "beta.pdf", pdf_b, results, 1),
    )

    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert results[0].status_code == 202
    assert results[1].status_code == 202

    files_resp = client.get(f"/api/v1/batches/{batch_id}/files")
    assert files_resp.status_code == 200
    files = files_resp.json()
    names = {f["name"] for f in files}

    alpha_pages = {f"alpha_page_{i:04d}.jpg" for i in range(1, 4)}
    beta_pages = {f"beta_page_{i:04d}.jpg" for i in range(1, 5)}

    assert alpha_pages <= names, (
        f"Missing alpha pages; got: {sorted(names)}"
    )
    assert beta_pages <= names, (
        f"Missing beta pages; got: {sorted(names)}"
    )
    assert len(names) == len(set(names)), "Duplicate filenames detected"


# ---------------------------------------------------------------------------
# Concurrent metadata writes must not crash on the temp-file rename.
#
# Regression: with a shared "metadata.tmp" name, many threads writing batch
# metadata at once (the PDF-split progress writer for several uploads plus a
# /process status transition) raced on os.replace and raised
# FileNotFoundError [WinError 2] '...metadata.tmp' -> '...metadata.json',
# 500-ing the /process request on large multi-PDF batches.
# ---------------------------------------------------------------------------

def test_concurrent_metadata_writes_do_not_crash(client: TestClient) -> None:
    """Hammering metadata writes from many threads must never raise."""
    from webui.services import batches as bm
    from webui.schemas import BatchStatus, SourceMode
    from webui.settings import get_settings

    batch_id = _create_batch(client, "Concurrent metadata test")
    settings = get_settings()

    errors: list[BaseException] = []
    barrier = threading.Barrier(12)

    def writer(kind: int) -> None:
        try:
            barrier.wait(timeout=5)
            for i in range(40):
                if kind % 3 == 0:
                    bm._write_pdf_split_progress(batch_id, settings, i, 40)
                elif kind % 3 == 1:
                    bm.update_status(batch_id, BatchStatus.queued, settings=settings)
                else:
                    bm.set_source(batch_id, SourceMode.upload, None, settings)
        except BaseException as exc:  # noqa: BLE001 — capture for assertion
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(k,)) for k in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert not errors, f"Concurrent metadata writes raised: {errors!r}"

    # Metadata is still valid, readable JSON after the storm.
    meta = bm.get_batch_metadata(batch_id, settings)
    assert meta.get("id") == batch_id
    assert meta.get("pdf_split_total") in {0, 40}


def test_concurrent_metadata_reads_during_replace_do_not_raise(
    client: TestClient,
) -> None:
    """Readers must ride out the Windows-replace permission/missing race.

    Regression: while one thread runs ``os.replace(metadata.tmp,
    metadata.json)`` a concurrent reader could get ``PermissionError
    [Errno 13]`` (Windows briefly denies the open) or ``FileNotFoundError``.
    The OMR run polled ``cancel_requested`` via this path on every page,
    so a heavy upload regularly killed the OMR run.
    """
    from webui.services import batches as bm
    from webui.schemas import BatchStatus
    from webui.settings import get_settings

    batch_id = _create_batch(client, "Read race test")
    settings = get_settings()

    errors: list[BaseException] = []
    stop = threading.Event()
    barrier = threading.Barrier(8)

    def writer() -> None:
        try:
            barrier.wait(timeout=5)
            i = 0
            while not stop.is_set():
                bm._write_pdf_split_progress(batch_id, settings, i, 1000)
                i += 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader() -> None:
        try:
            barrier.wait(timeout=5)
            for _ in range(400):
                bm.get_batch_metadata(batch_id, settings)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(4)] + \
              [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    # Let readers finish, then stop writers.
    for t in threads[4:]:
        t.join(timeout=15)
    stop.set()
    for t in threads[:4]:
        t.join(timeout=5)

    assert not errors, f"Concurrent metadata reads raised: {errors!r}"

    # Final state is still a valid Batch and round-trips through update_status.
    bm.update_status(batch_id, BatchStatus.queued, settings=settings)
    meta = bm.get_batch_metadata(batch_id, settings)
    assert meta.get("status") == "queued"
