"""Tests for the embedded-image fast path in batches._try_extract_embedded_page_image.

Fix #1 — Direct embedded-image fast path
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gray_jpeg(width: int = 100, height: int = 100, quality: int = 90) -> bytes:
    """Return grayscale JPEG bytes for use as an embedded page image."""
    fitz = pytest.importorskip("fitz")
    import fitz as _fitz
    pix = _fitz.Pixmap(_fitz.csGRAY, (0, 0, width, height), False)
    # set_rect requires a sequence for color; pass a 1-tuple for gray.
    pix.set_rect(pix.irect, (128,))
    try:
        return pix.tobytes("jpeg", jpg_quality=quality)
    except TypeError:
        return pix.tobytes("jpeg")


def _make_rgb_jpeg(width: int = 100, height: int = 100, quality: int = 90) -> bytes:
    """Return RGB JPEG bytes (red fill) for use as an embedded page image."""
    import fitz as _fitz
    pix = _fitz.Pixmap(_fitz.csRGB, (0, 0, width, height), False)
    pix.set_rect(pix.irect, (220, 50, 50))
    try:
        return pix.tobytes("jpeg", jpg_quality=quality)
    except TypeError:
        return pix.tobytes("jpeg")


def _make_gray_png(width: int = 100, height: int = 100) -> bytes:
    """Return grayscale PNG bytes for use as an embedded page image."""
    import fitz as _fitz
    pix = _fitz.Pixmap(_fitz.csGRAY, (0, 0, width, height), False)
    pix.set_rect(pix.irect, (200,))
    return pix.tobytes("png")


def _build_pdf_with_full_page_image(image_bytes: bytes, width: int = 100, height: int = 100) -> "tuple[Any, Any]":
    """Return (doc, page) for a PDF whose first page is entirely *image_bytes*."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.insert_image(page.rect, stream=image_bytes)
    return doc, page


def _build_pdf_with_text_only(width: int = 100, height: int = 100) -> "tuple[Any, Any]":
    """Return (doc, page) for a PDF with text only — no embedded images."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.insert_text((10, 50), "No images here — vector only")
    return doc, page


def _build_pdf_with_small_corner_image(
    page_width: int = 200,
    page_height: int = 300,
    image_size: int = 20,
) -> "tuple[Any, Any]":
    """Return (doc, page) where a small image occupies only the top-left corner.

    Coverage = 20×20 / (200×300) ≈ 0.67 % — well below the 90 % threshold.
    """
    import fitz
    gray_jpeg = _make_gray_jpeg(image_size, image_size)
    doc = fitz.open()
    page = doc.new_page(width=page_width, height=page_height)
    corner_rect = fitz.Rect(0, 0, image_size, image_size)
    page.insert_image(corner_rect, stream=gray_jpeg)
    return doc, page


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fast_path_extracts_embedded_jpeg_directly(tmp_path: Path) -> None:
    """A full-page embedded JPEG is returned without any re-encoding.

    The saved bytes must equal what PyMuPDF stores as the embedded stream
    (i.e. no second JPEG quantisation round-trip at our layer).
    """
    pytest.importorskip("fitz")
    src_jpeg = _make_gray_jpeg()
    doc, page = _build_pdf_with_full_page_image(src_jpeg)

    # What PyMuPDF actually stored (our reference — may differ from src_jpeg
    # if PyMuPDF normalised the header, but won't differ from what we extract).
    images = page.get_images(full=True)
    xref = images[0][0]
    ref_bytes = doc.extract_image(xref)["image"]

    from webui.services.batches import _try_extract_embedded_page_image
    result = _try_extract_embedded_page_image(
        doc, page, 1,
        dpi=150, grayscale=True, ext=".jpg", jpeg_quality=90,
    )

    assert result is not None, "Fast path should return bytes for a full-page gray JPEG"
    assert result == ref_bytes, (
        "Fast path must return the raw embedded stream — no additional re-encoding"
    )
    doc.close()


def test_fast_path_falls_back_for_vector_pages(tmp_path: Path) -> None:
    """A page with text only (no embedded images) must trigger the fallback path."""
    pytest.importorskip("fitz")
    doc, page = _build_pdf_with_text_only()

    from webui.services.batches import _try_extract_embedded_page_image
    result = _try_extract_embedded_page_image(
        doc, page, 1,
        dpi=150, grayscale=True, ext=".jpg", jpeg_quality=90,
    )

    assert result is None, (
        "Vector-only page must return None so the caller falls back to rasterisation"
    )
    doc.close()


def test_fast_path_falls_back_for_partial_image_pages(tmp_path: Path) -> None:
    """A small corner image (< 90 % coverage) must not trigger the fast path."""
    pytest.importorskip("fitz")
    doc, page = _build_pdf_with_small_corner_image(
        page_width=200, page_height=300, image_size=20
    )

    from webui.services.batches import _try_extract_embedded_page_image
    result = _try_extract_embedded_page_image(
        doc, page, 1,
        dpi=150, grayscale=True, ext=".jpg", jpeg_quality=90,
    )

    assert result is None, (
        "Partial-coverage image must return None (coverage ≈ 0.67 %% < 90 %%)"
    )
    doc.close()


def test_fast_path_handles_format_mismatch(tmp_path: Path) -> None:
    """Embedded PNG with JPEG output requested → re-encoded to JPEG, not raw PNG."""
    pytest.importorskip("fitz")
    png_bytes = _make_gray_png()
    doc, page = _build_pdf_with_full_page_image(png_bytes)

    from webui.services.batches import _try_extract_embedded_page_image
    result = _try_extract_embedded_page_image(
        doc, page, 1,
        dpi=150, grayscale=True, ext=".jpg", jpeg_quality=90,
    )

    assert result is not None, "PNG→JPEG conversion must succeed"
    # JPEG magic bytes: FF D8
    assert result[:2] == b"\xff\xd8", (
        f"Output must be a JPEG (expected FF D8 header), got {result[:4].hex()}"
    )
    # Must NOT be raw PNG bytes (PNG magic: 89 50 4E 47)
    assert result[:4] != b"\x89PNG", "Output must not be raw PNG"
    doc.close()


def test_fast_path_handles_color_to_grayscale_conversion(tmp_path: Path) -> None:
    """Embedded RGB JPEG with grayscale=True → output must be a grayscale JPEG."""
    pytest.importorskip("fitz")
    rgb_jpeg = _make_rgb_jpeg()
    doc, page = _build_pdf_with_full_page_image(rgb_jpeg)

    from webui.services.batches import _try_extract_embedded_page_image
    result = _try_extract_embedded_page_image(
        doc, page, 1,
        dpi=150, grayscale=True, ext=".jpg", jpeg_quality=90,
    )

    assert result is not None, "RGB→grayscale JPEG conversion must succeed"
    # Decode the output with PyMuPDF and check colorspace components.
    import fitz as _fitz
    pix = _fitz.Pixmap(result)
    n_components = pix.n - (1 if pix.alpha else 0)
    assert n_components == 1, (
        f"Expected 1 grayscale component in output, got {n_components} "
        f"(colorspace={pix.colorspace})"
    )
    doc.close()


# ---------------------------------------------------------------------------
# Integration: _save_pdf_pages_serial correctly writes fast-path pages
# ---------------------------------------------------------------------------

def test_serial_fast_path_produces_valid_image_files(tmp_path: Path) -> None:
    """_save_pdf_pages_serial must write a valid image when fast path triggers."""
    pytest.importorskip("fitz")
    import fitz as _fitz

    # Build a 2-page PDF where each page is a full-page grayscale JPEG.
    src_jpeg = _make_gray_jpeg(width=200, height=280)
    doc = _fitz.open()
    for _ in range(2):
        pg = doc.new_page(width=200, height=280)
        pg.insert_image(pg.rect, stream=src_jpeg)
    pdf_bytes = doc.tobytes()
    doc.close()

    inputs = tmp_path / "inputs"
    inputs.mkdir()

    # The renderer now opens the PDF by path (memory-mapped) rather than from
    # an in-memory stream, so stage the bytes to a temp file first.
    staged = tmp_path / "scan_staged.pdf"
    staged.write_bytes(pdf_bytes)

    from webui.services.batches import _save_pdf_pages_serial
    stored, failed = _save_pdf_pages_serial(
        inputs=inputs,
        safe_filename="scan.pdf",
        pdf_path=str(staged),
        stem="scan",
        page_count=2,
        dpi=150,
        grayscale=True,
        ext=".jpg",
        jpeg_quality=90,
        batch_id=None,
        settings=None,
    )

    assert failed == [], f"No pages should fail; failed={failed}"
    assert len(stored) == 2
    for ref in stored:
        img_path = inputs / ref.name
        assert img_path.exists(), f"Expected {ref.name} on disk"
        # Verify it's a valid JPEG.
        assert img_path.read_bytes()[:2] == b"\xff\xd8", f"{ref.name} is not a JPEG"


# ---------------------------------------------------------------------------
# Helpers for flipped-transform tests
# ---------------------------------------------------------------------------

def _make_topbright_botdark_png(width: int = 100, height: int = 100) -> bytes:
    """Return a grayscale PNG where the top half is bright (200) and bottom is dark (50).

    The asymmetric brightness makes it easy to detect vertical flips in tests.
    """
    import fitz as _fitz
    pix = _fitz.Pixmap(_fitz.csGRAY, (0, 0, width, height), False)
    pix.set_rect(_fitz.IRect(0, 0, width, height // 2), (200,))
    pix.set_rect(_fitz.IRect(0, height // 2, width, height), (50,))
    return pix.tobytes("png")


def _build_pdf_with_flipped_image_transform(
    img_bytes: bytes,
    width: int = 100,
    height: int = 100,
) -> "tuple[Any, Any]":
    """Return *(doc, page)* for a PDF whose image has a vertical-flip content-stream matrix.

    This replicates the pattern produced by mobile scan apps: the raw JPEG/PNG
    bytes are stored in one orientation and the content stream uses
    ``W 0 0 -H 0 H cm`` (d < 0 in fitz coords) to flip them back so the page
    renders correctly in PDF viewers.
    """
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    # Register the image in the xref table via the normal insert path.
    page.insert_image(page.rect, stream=img_bytes)
    # Get the resource name assigned to the image by PyMuPDF.
    images = page.get_images(full=True)
    img_name = images[0][7]
    # Rewrite the content stream with a vertical-flip placement matrix.
    # Normal: ``W 0 0 H 0 0 cm``  →  d = +H (positive)
    # Flipped: ``W 0 0 -H 0 H cm`` →  d = -H (negative)
    flipped = f"q\n{width} 0 0 -{height} 0 {height} cm\n/{img_name} Do\nQ\n"
    content_xrefs = page.get_contents()
    doc.update_stream(content_xrefs[0], flipped.encode())
    page = doc.reload_page(page)
    return doc, page


# ---------------------------------------------------------------------------
# Regression tests: scanner PDFs with flipped content-stream transforms
# ---------------------------------------------------------------------------

def test_fast_path_falls_back_for_flipped_content_stream_transform() -> None:
    """Fast path must return None when the image has a vertical-flip placement transform.

    Regression test for the scanner-PDF bug: some scan apps store the page
    image with a ``d < 0`` content-stream matrix so the page renders correctly
    in PDF viewers.  Before the fix, the fast path returned the raw (flipped)
    bytes, causing the OMR engine to receive an upside-down image and fail to
    locate ArUco markers.
    """
    pytest.importorskip("fitz")
    W, H = 100, 100
    png_bytes = _make_topbright_botdark_png(W, H)
    doc, page = _build_pdf_with_flipped_image_transform(png_bytes, width=W, height=H)

    # Sanity-check that the transform really does have d < 0.
    info = page.get_image_info(xrefs=True)
    assert info, "Expected image info on the test page"
    t = info[0]["transform"]
    assert t[3] < 0, f"Test setup failed: expected d < 0, got d={t[3]}"

    from webui.services.batches import _try_extract_embedded_page_image
    result = _try_extract_embedded_page_image(
        doc, page, 1,
        dpi=150, grayscale=True, ext=".jpg", jpeg_quality=90,
    )
    assert result is None, (
        "Fast path must return None for a flipped-transform image so the caller "
        "falls back to rasterisation (which correctly applies the flip)."
    )
    doc.close()


def test_serial_render_matches_rasterised_output_for_flipped_transform_pdf(
    tmp_path: Path,
) -> None:
    """_save_pdf_pages_serial must produce an image matching page.get_pixmap() for flipped PDFs.

    Regression test: before the fix, the serial renderer used the fast path and
    returned the raw (vertically flipped) bytes.  After the fix it falls back to
    rasterisation, whose output must match the ground-truth get_pixmap() render.
    """
    pytest.importorskip("fitz")
    import fitz as _fitz
    import numpy as _np

    try:
        import cv2 as _cv2
    except ImportError:
        pytest.skip("cv2 not available")

    W, H = 100, 100
    png_bytes = _make_topbright_botdark_png(W, H)
    doc, page = _build_pdf_with_flipped_image_transform(png_bytes, width=W, height=H)

    # Ground truth: rasterised render of the page (correct orientation).
    # The flip transform means: raw top-bright→rendered top-dark, raw bot-dark→rendered bot-bright.
    ref_pix = page.get_pixmap(dpi=150, alpha=False, colorspace=_fitz.csGRAY)
    ref_arr = _np.frombuffer(ref_pix.samples, dtype=_np.uint8).reshape(
        ref_pix.height, ref_pix.width
    )
    ref_top_mean = float(ref_arr[: H // 4].mean())
    ref_bot_mean = float(ref_arr[-H // 4 :].mean())

    pdf_bytes = doc.tobytes()
    doc.close()

    staged = tmp_path / "scan_staged.pdf"
    staged.write_bytes(pdf_bytes)
    inputs = tmp_path / "inputs"
    inputs.mkdir()

    from webui.services.batches import _save_pdf_pages_serial

    stored, failed = _save_pdf_pages_serial(
        inputs=inputs,
        safe_filename="scan.pdf",
        pdf_path=str(staged),
        stem="scan",
        page_count=1,
        dpi=150,
        grayscale=True,
        ext=".jpg",
        jpeg_quality=90,
        batch_id=None,
        settings=None,
    )

    assert failed == [], f"Page should not fail; failed={failed}"
    assert len(stored) == 1

    out_path = inputs / stored[0].name
    out_arr = _cv2.imread(str(out_path), _cv2.IMREAD_GRAYSCALE)
    assert out_arr is not None, "Output image must be readable"

    out_top_mean = float(out_arr[: H // 4].mean())
    out_bot_mean = float(out_arr[-H // 4 :].mean())

    # The output brightness profile must match the rasterised ground truth,
    # not the raw (flipped) bytes.  Allow a 10-point tolerance for JPEG artifacts.
    _TOL = 10.0
    assert abs(out_top_mean - ref_top_mean) < _TOL, (
        f"Output top brightness ({out_top_mean:.1f}) does not match "
        f"rasterised top brightness ({ref_top_mean:.1f}) — image may be flipped."
    )
    assert abs(out_bot_mean - ref_bot_mean) < _TOL, (
        f"Output bottom brightness ({out_bot_mean:.1f}) does not match "
        f"rasterised bottom brightness ({ref_bot_mean:.1f}) — image may be flipped."
    )
