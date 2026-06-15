"""Comprehensive guardrail tests for PDF orientation / transform edge-cases.

The fast-path in ``_try_extract_embedded_page_image`` extracts a page's
embedded image bytes directly without rasterisation.  It must fall back to
``page.get_pixmap()`` whenever the content-stream transform or the page's
own /Rotate entry would produce a different pixel orientation than the raw
bytes.

Scenarios tested
----------------
Content-stream transform variants (plausible from real scanner apps):
  1. Normal (d > 0)            – fast path MUST be used (regression guard)
  2. Vertical flip (d < 0)     – original bug; scanner stores upside-down image
  3. Horizontal flip (a < 0)   – scanner stores mirror image
  4. 180° rotation             – paper placed upside-down in scanner
  5. 90° clockwise rotation    – landscape paper, content-stream rotation
  6. 90° counter-clockwise     – landscape paper, other direction

Page /Rotate variants (PDF page dictionary key):
  7. /Rotate = 90
  8. /Rotate = 180
  9. /Rotate = 270

Combined scenarios:
  10. V-flip + /Rotate = 90   – some advanced scan apps use both

Integration tests (for each non-normal scenario):
  • ``_save_pdf_pages_serial`` must produce an image whose pixel orientation
    matches ``page.get_pixmap()`` (ground-truth), not the raw embedded bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Image / PDF helpers
# ---------------------------------------------------------------------------

_W, _H = 100, 150  # default page / image dimensions for most tests


def _make_quad_gray_png(width: int = _W, height: int = _H) -> bytes:
    """Return a grayscale PNG with four distinct-brightness quadrants.

    Top-left=220, top-right=160, bottom-left=100, bottom-right=50.
    The asymmetry in both axes lets us detect horizontal *and* vertical flips.
    """
    fitz = pytest.importorskip("fitz")
    pix = fitz.Pixmap(fitz.csGRAY, (0, 0, width, height), False)
    hw, hh = width // 2, height // 2
    pix.set_rect(fitz.IRect(0, 0, hw, hh), (220,))          # top-left
    pix.set_rect(fitz.IRect(hw, 0, width, hh), (160,))       # top-right
    pix.set_rect(fitz.IRect(0, hh, hw, height), (100,))      # bottom-left
    pix.set_rect(fitz.IRect(hw, hh, width, height), (50,))   # bottom-right
    return pix.tobytes("png")


def _build_pdf_with_content_stream_transform(
    img_bytes: bytes,
    width: int,
    height: int,
    cm_matrix: str,
) -> "tuple[Any, Any]":
    """Return *(doc, page)* with the image placed via *cm_matrix*.

    ``cm_matrix`` is a PDF content-stream ``cm`` argument string, e.g.
    ``"100 0 0 -150 0 150"`` for a vertical flip.
    The image resource name is substituted as ``{img}``.
    """
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.insert_image(page.rect, stream=img_bytes)
    images = page.get_images(full=True)
    img_name = images[0][7]
    content = f"q\n{cm_matrix} cm\n/{img_name} Do\nQ\n"
    cxrefs = page.get_contents()
    doc.update_stream(cxrefs[0], content.encode())
    page = doc.reload_page(page)
    return doc, page


def _build_pdf_with_page_rotation(
    img_bytes: bytes,
    width: int,
    height: int,
    rotation: int,
) -> "tuple[Any, Any]":
    """Return *(doc, page)* with a /Rotate *rotation* entry in the page dict."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.insert_image(page.rect, stream=img_bytes)
    page.set_rotation(rotation)
    return doc, page


def _build_pdf_combined(
    img_bytes: bytes,
    width: int,
    height: int,
    cm_matrix: str,
    rotation: int,
) -> "tuple[Any, Any]":
    """Return *(doc, page)* combining a content-stream transform *and* /Rotate."""
    doc, page = _build_pdf_with_content_stream_transform(
        img_bytes, width, height, cm_matrix
    )
    page.set_rotation(rotation)
    return doc, page


def _render_and_save(
    doc: "Any",
    page: "Any",
    tmp_path: Path,
    dpi: int = 150,
) -> "tuple[Any, Any]":
    """Render *page* via ``_save_pdf_pages_serial`` and return *(ref_arr, out_arr)*.

    *ref_arr* is the ground-truth numpy array from ``page.get_pixmap()``.
    *out_arr* is the saved-then-decoded image from the serial render pipeline.
    """
    import fitz as _fitz
    import numpy as _np

    cv2 = pytest.importorskip("cv2")

    pdf_bytes = doc.tobytes()

    # Ground-truth: correct rasterisation.
    ref_pix = page.get_pixmap(dpi=dpi, alpha=False, colorspace=_fitz.csGRAY)
    ref_arr = _np.frombuffer(ref_pix.samples, dtype=_np.uint8).reshape(
        ref_pix.height, ref_pix.width
    )

    staged = tmp_path / "staged.pdf"
    staged.write_bytes(pdf_bytes)
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)

    from webui.services.batches import _save_pdf_pages_serial

    stored, failed = _save_pdf_pages_serial(
        inputs=inputs,
        safe_filename="test.pdf",
        pdf_path=str(staged),
        stem="test",
        page_count=1,
        dpi=dpi,
        grayscale=True,
        ext=".jpg",
        jpeg_quality=90,
        batch_id=None,
        settings=None,
    )
    assert failed == [], f"Page render failed: {failed}"
    assert len(stored) == 1

    out_raw = cv2.imread(str(inputs / stored[0].name), cv2.IMREAD_GRAYSCALE)
    assert out_raw is not None, "Could not read saved image"

    # Resize to match reference (DPI is the same, but small rounding diffs possible).
    out_arr = cv2.resize(out_raw, (ref_arr.shape[1], ref_arr.shape[0]))
    return ref_arr, out_arr


def _assert_orientation_matches(
    ref_arr: "Any",
    out_arr: "Any",
    label: str,
    tol: float = 8.0,
) -> None:
    """Assert *out_arr* has the same orientation as *ref_arr* within *tol* MAE."""
    import numpy as _np

    mae = float(_np.abs(ref_arr.astype(_np.int32) - out_arr.astype(_np.int32)).mean())
    assert mae < tol, (
        f"[{label}] Output orientation does not match rasterised ground truth "
        f"(MAE={mae:.2f} >= tol={tol}).  The image may be flipped or rotated."
    )


# ---------------------------------------------------------------------------
# Parametrised data
# ---------------------------------------------------------------------------

# Each entry: (id_label, cm_matrix_string)
_CONTENT_STREAM_TRANSFORMS = [
    ("v_flip",   f"{_W} 0 0 -{_H} 0 {_H}"),
    ("h_flip",   f"-{_W} 0 0 {_H} {_W} 0"),
    ("rot_180",  f"-{_W} 0 0 -{_H} {_W} {_H}"),
    ("rot_90cw", f"0 -{_W} {_H} 0 0 {_W}"),
    ("rot_90ccw",f"0 {_W} -{_H} 0 {_H} 0"),
]

_PAGE_ROTATIONS = [90, 180, 270]


# ---------------------------------------------------------------------------
# 1. Fast path returns None for every non-normal transform
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,cm_matrix", _CONTENT_STREAM_TRANSFORMS, ids=[t[0] for t in _CONTENT_STREAM_TRANSFORMS])
def test_fast_path_falls_back_for_content_stream_transform(label: str, cm_matrix: str) -> None:
    """Fast path must return None for any non-identity content-stream transform."""
    pytest.importorskip("fitz")
    png = _make_quad_gray_png()
    doc, page = _build_pdf_with_content_stream_transform(png, _W, _H, cm_matrix)

    from webui.services.batches import _try_extract_embedded_page_image

    result = _try_extract_embedded_page_image(
        doc, page, 1,
        dpi=150, grayscale=True, ext=".jpg", jpeg_quality=90,
    )
    assert result is None, (
        f"[{label}] Fast path must return None so the rasteriser applies the "
        f"content-stream transform; returning raw bytes would give a "
        f"mis-oriented image."
    )
    doc.close()


@pytest.mark.parametrize("rotation", _PAGE_ROTATIONS)
def test_fast_path_falls_back_for_page_rotation(rotation: int) -> None:
    """Fast path must return None for any non-zero page /Rotate entry."""
    pytest.importorskip("fitz")
    png = _make_quad_gray_png()
    doc, page = _build_pdf_with_page_rotation(png, _W, _H, rotation)

    from webui.services.batches import _try_extract_embedded_page_image

    result = _try_extract_embedded_page_image(
        doc, page, 1,
        dpi=150, grayscale=True, ext=".jpg", jpeg_quality=90,
    )
    assert result is None, (
        f"[/Rotate={rotation}] Fast path must return None; the page /Rotate "
        f"is applied by the rasteriser but lost when returning raw bytes."
    )
    doc.close()


def test_fast_path_falls_back_for_combined_vflip_and_page_rotation() -> None:
    """Fast path must return None when both a content-stream flip and /Rotate are present."""
    pytest.importorskip("fitz")
    png = _make_quad_gray_png()
    cm_matrix = f"{_W} 0 0 -{_H} 0 {_H}"  # v-flip
    doc, page = _build_pdf_combined(png, _W, _H, cm_matrix, rotation=90)

    from webui.services.batches import _try_extract_embedded_page_image

    result = _try_extract_embedded_page_image(
        doc, page, 1,
        dpi=150, grayscale=True, ext=".jpg", jpeg_quality=90,
    )
    assert result is None, (
        "Fast path must return None for combined v-flip + /Rotate=90."
    )
    doc.close()


# ---------------------------------------------------------------------------
# 2. Normal transform: fast path MUST be used (regression guard)
# ---------------------------------------------------------------------------

def test_fast_path_active_for_normal_transform() -> None:
    """Fast path must NOT fall back for a normal positive-scale placement (regression)."""
    pytest.importorskip("fitz")
    import fitz as _fitz

    pix_src = _fitz.Pixmap(_fitz.csGRAY, (0, 0, _W, _H), False)
    pix_src.set_rect(pix_src.irect, (180,))
    img_bytes = pix_src.tobytes("png")

    doc = _fitz.open()
    page = doc.new_page(width=_W, height=_H)
    page.insert_image(page.rect, stream=img_bytes)

    from webui.services.batches import _try_extract_embedded_page_image

    result = _try_extract_embedded_page_image(
        doc, page, 1,
        dpi=150, grayscale=True, ext=".jpg", jpeg_quality=90,
    )
    assert result is not None, (
        "Fast path must return bytes for a normally-placed image; "
        "this regression would hurt performance for every valid PDF."
    )
    doc.close()


# ---------------------------------------------------------------------------
# 3. Integration: _save_pdf_pages_serial produces correctly-oriented output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,cm_matrix", _CONTENT_STREAM_TRANSFORMS, ids=[t[0] for t in _CONTENT_STREAM_TRANSFORMS])
def test_serial_output_orientation_for_content_stream_transform(
    label: str, cm_matrix: str, tmp_path: Path
) -> None:
    """_save_pdf_pages_serial output must match get_pixmap() for every transform."""
    pytest.importorskip("fitz")
    pytest.importorskip("cv2")

    png = _make_quad_gray_png()
    doc, page = _build_pdf_with_content_stream_transform(png, _W, _H, cm_matrix)

    ref_arr, out_arr = _render_and_save(doc, page, tmp_path)
    doc.close()

    _assert_orientation_matches(ref_arr, out_arr, label)


@pytest.mark.parametrize("rotation", _PAGE_ROTATIONS)
def test_serial_output_orientation_for_page_rotation(
    rotation: int, tmp_path: Path
) -> None:
    """_save_pdf_pages_serial output must match get_pixmap() for every /Rotate."""
    pytest.importorskip("fitz")
    pytest.importorskip("cv2")

    png = _make_quad_gray_png()
    doc, page = _build_pdf_with_page_rotation(png, _W, _H, rotation)

    ref_arr, out_arr = _render_and_save(doc, page, tmp_path)
    doc.close()

    _assert_orientation_matches(ref_arr, out_arr, f"rotate_{rotation}")


def test_serial_output_orientation_for_combined_vflip_and_page_rotation(
    tmp_path: Path,
) -> None:
    """_save_pdf_pages_serial output must match get_pixmap() for v-flip + /Rotate=90."""
    pytest.importorskip("fitz")
    pytest.importorskip("cv2")

    png = _make_quad_gray_png()
    cm_matrix = f"{_W} 0 0 -{_H} 0 {_H}"
    doc, page = _build_pdf_combined(png, _W, _H, cm_matrix, rotation=90)

    ref_arr, out_arr = _render_and_save(doc, page, tmp_path)
    doc.close()

    _assert_orientation_matches(ref_arr, out_arr, "vflip+rotate90")


# ---------------------------------------------------------------------------
# 4. Edge-cases: multi-image page, partial coverage, zero-area page
# ---------------------------------------------------------------------------

def test_fast_path_falls_back_for_two_images_on_page() -> None:
    """Page with two embedded images must not use fast path (coverage/uniqueness check)."""
    pytest.importorskip("fitz")
    import fitz as _fitz

    pix = _fitz.Pixmap(_fitz.csGRAY, (0, 0, _W, _H), False)
    pix.set_rect(pix.irect, (128,))
    img_bytes = pix.tobytes("png")

    doc = _fitz.open()
    page = doc.new_page(width=_W, height=_H)
    half = _fitz.Rect(0, 0, _W, _H // 2)
    page.insert_image(half, stream=img_bytes)
    page.insert_image(_fitz.Rect(0, _H // 2, _W, _H), stream=img_bytes)

    from webui.services.batches import _try_extract_embedded_page_image

    result = _try_extract_embedded_page_image(
        doc, page, 1,
        dpi=150, grayscale=True, ext=".jpg", jpeg_quality=90,
    )
    assert result is None, "Two-image page must not use the fast path."
    doc.close()


def test_fast_path_falls_back_for_small_corner_image() -> None:
    """Small image with < 90 % page coverage must not use fast path."""
    pytest.importorskip("fitz")
    import fitz as _fitz

    pix = _fitz.Pixmap(_fitz.csGRAY, (0, 0, 20, 20), False)
    pix.set_rect(pix.irect, (128,))
    img_bytes = pix.tobytes("png")

    doc = _fitz.open()
    page = doc.new_page(width=200, height=300)
    page.insert_image(_fitz.Rect(0, 0, 20, 20), stream=img_bytes)

    from webui.services.batches import _try_extract_embedded_page_image

    result = _try_extract_embedded_page_image(
        doc, page, 1,
        dpi=150, grayscale=True, ext=".jpg", jpeg_quality=90,
    )
    assert result is None, "Small-coverage image must not use the fast path."
    doc.close()


# ---------------------------------------------------------------------------
# 5. Smoke test: scanner-style PDF (simulates the real-world bug)
# ---------------------------------------------------------------------------

def test_scanner_style_vflip_pdf_produces_correct_output(tmp_path: Path) -> None:
    """End-to-end smoke test simulating the original scanner-PDF bug.

    The raw embedded PNG bytes have top=bright, bottom=dark.
    The content stream applies a vertical flip, so the rendered page has
    top=dark, bottom=bright.  The pipeline must produce the rendered
    orientation (top-dark), not the raw orientation (top-bright).
    """
    pytest.importorskip("fitz")
    pytest.importorskip("cv2")
    import numpy as _np

    cv2 = pytest.importorskip("cv2")

    import fitz as _fitz

    W, H = 100, 100
    pix = _fitz.Pixmap(_fitz.csGRAY, (0, 0, W, H), False)
    pix.set_rect(_fitz.IRect(0, 0, W, H // 2), (200,))   # top half bright
    pix.set_rect(_fitz.IRect(0, H // 2, W, H), (50,))    # bottom half dark
    png_bytes = pix.tobytes("png")

    cm_matrix = f"{W} 0 0 -{H} 0 {H}"  # vertical flip: d = -H
    doc, page = _build_pdf_with_content_stream_transform(png_bytes, W, H, cm_matrix)

    # Ground truth: rendered page has top=dark (50), bottom=bright (200).
    ref_pix = page.get_pixmap(dpi=150, alpha=False, colorspace=_fitz.csGRAY)
    ref_arr = _np.frombuffer(ref_pix.samples, dtype=_np.uint8).reshape(
        ref_pix.height, ref_pix.width
    )

    ref_top = float(ref_arr[: H // 4].mean())
    ref_bot = float(ref_arr[-H // 4 :].mean())
    assert ref_top < ref_bot, (
        "Test setup error: rendered page should have darker top than bottom "
        f"(ref_top={ref_top:.1f}, ref_bot={ref_bot:.1f})"
    )

    pdf_bytes = doc.tobytes()
    doc.close()

    staged = tmp_path / "scanner.pdf"
    staged.write_bytes(pdf_bytes)
    inputs = tmp_path / "inputs"
    inputs.mkdir()

    from webui.services.batches import _save_pdf_pages_serial

    stored, failed = _save_pdf_pages_serial(
        inputs=inputs,
        safe_filename="scanner.pdf",
        pdf_path=str(staged),
        stem="scanner",
        page_count=1,
        dpi=150,
        grayscale=True,
        ext=".jpg",
        jpeg_quality=90,
        batch_id=None,
        settings=None,
    )

    assert failed == [], f"No pages should fail; failed={failed}"
    out_arr = cv2.imread(str(inputs / stored[0].name), cv2.IMREAD_GRAYSCALE)
    assert out_arr is not None

    out_top = float(out_arr[: H // 4].mean())
    out_bot = float(out_arr[-H // 4 :].mean())

    _TOL = 10.0
    assert abs(out_top - ref_top) < _TOL, (
        f"Top brightness mismatch: output={out_top:.1f}, expected≈{ref_top:.1f}. "
        "Image may be vertically flipped (original scanner-PDF bug)."
    )
    assert abs(out_bot - ref_bot) < _TOL, (
        f"Bottom brightness mismatch: output={out_bot:.1f}, expected≈{ref_bot:.1f}."
    )
