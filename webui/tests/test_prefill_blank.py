"""Tests for the print-N-blank-sheets feature.

Coverage:

*   ``normalize_blank_sheet_variant`` accepts the default, the registered
    key, and rejects unknown values.
*   ``generate_blank_sheets_pdf_to_file`` produces exactly N pages of
    US-letter landscape geometry, optionally stamps continuous
    ``n / N`` page numbers, and dedupes the embedded raster on save
    so output size scales sublinearly with N.
*   ``generate_blank_sheets_split_zip_to_file`` post-splits the combined
    PDF into ZIP segments while keeping page numbers continuous.
*   ``POST /api/v1/prefill/blank`` returns a download token, downloads
    a valid PDF/ZIP, and enforces the per-request cap, count >= 1, and
    the variant allowlist.
*   The bundled source PDF carries all four ArUco markers so each
    generated blank page is alignable by the OMR engine.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import cv2
import fitz
import pytest
from fastapi.testclient import TestClient

from webui.services import prefill as prefill_service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _page_count(pdf_path: Path) -> int:
    doc = fitz.open(str(pdf_path))
    try:
        return doc.page_count
    finally:
        doc.close()


def _page_texts(pdf_path: Path) -> list[str]:
    doc = fitz.open(str(pdf_path))
    try:
        return [doc.load_page(i).get_text().strip() for i in range(doc.page_count)]
    finally:
        doc.close()


def _page_dimensions(pdf_path: Path) -> list[tuple[float, float]]:
    doc = fitz.open(str(pdf_path))
    try:
        return [(doc.load_page(i).rect.width, doc.load_page(i).rect.height)
                for i in range(doc.page_count)]
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Variant normalisation
# ---------------------------------------------------------------------------


def test_normalize_blank_sheet_variant_returns_default_for_empty() -> None:
    assert (
        prefill_service.normalize_blank_sheet_variant(None)
        == prefill_service.DEFAULT_BLANK_SHEET_VARIANT
    )
    assert (
        prefill_service.normalize_blank_sheet_variant("")
        == prefill_service.DEFAULT_BLANK_SHEET_VARIANT
    )
    assert (
        prefill_service.normalize_blank_sheet_variant("   ")
        == prefill_service.DEFAULT_BLANK_SHEET_VARIANT
    )


def test_normalize_blank_sheet_variant_accepts_registered_keys() -> None:
    for key in prefill_service.BLANK_SHEET_VARIANTS:
        assert prefill_service.normalize_blank_sheet_variant(key) == key


def test_normalize_blank_sheet_variant_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown blank sheet variant"):
        prefill_service.normalize_blank_sheet_variant("not_a_real_variant")


def test_list_blank_sheet_variants_shape_matches_ui_contract() -> None:
    variants = prefill_service.list_blank_sheet_variants()
    assert variants, "Expected at least one registered blank sheet variant"
    for entry in variants:
        assert set(entry.keys()) == {"key", "label"}
        assert isinstance(entry["key"], str) and entry["key"]
        assert isinstance(entry["label"], str) and entry["label"]


# ---------------------------------------------------------------------------
# Service-layer generation
# ---------------------------------------------------------------------------


def test_generate_blank_sheets_pdf_to_file_writes_n_pages(tmp_path: Path) -> None:
    dst = tmp_path / "blank_3.pdf"
    meta = prefill_service.generate_blank_sheets_pdf_to_file(
        dst, variant="MoE-April-2026-Landscape-NNQ25-0", count=3
    )
    assert dst.exists()
    assert meta["count"] == 3
    assert meta["successes"] == 3
    assert meta["errors"] == []
    assert meta["variant"] == "MoE-April-2026-Landscape-NNQ25-0"
    assert _page_count(dst) == 3


def test_generate_blank_sheets_pdf_to_file_pages_are_us_letter_landscape(
    tmp_path: Path,
) -> None:
    dst = tmp_path / "blank_2.pdf"
    prefill_service.generate_blank_sheets_pdf_to_file(dst, count=2)
    for width_pt, height_pt in _page_dimensions(dst):
        assert width_pt == pytest.approx(792.0, abs=1.0)
        assert height_pt == pytest.approx(612.0, abs=1.0)


def test_generate_blank_sheets_pdf_to_file_rejects_zero_or_negative_count(
    tmp_path: Path,
) -> None:
    dst = tmp_path / "blank_bad.pdf"
    with pytest.raises(ValueError, match="count must be >= 1"):
        prefill_service.generate_blank_sheets_pdf_to_file(dst, count=0)
    with pytest.raises(ValueError, match="count must be >= 1"):
        prefill_service.generate_blank_sheets_pdf_to_file(dst, count=-7)


def test_generate_blank_sheets_pdf_to_file_unknown_variant_raises(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Unknown blank sheet variant"):
        prefill_service.generate_blank_sheets_pdf_to_file(
            tmp_path / "x.pdf", variant="bogus", count=1
        )


def test_generate_blank_sheets_pdf_to_file_stamps_continuous_page_numbers(
    tmp_path: Path,
) -> None:
    dst = tmp_path / "blank_numbered.pdf"
    prefill_service.generate_blank_sheets_pdf_to_file(
        dst, count=4, include_page_numbers=True
    )
    texts = _page_texts(dst)
    assert len(texts) == 4
    for i, body in enumerate(texts, start=1):
        assert f"{i} / 4" in body, (
            f"Page {i} text {body!r} did not contain its 'n / N' stamp"
        )


def test_generate_blank_sheets_pdf_to_file_without_numbers_has_no_page_text(
    tmp_path: Path,
) -> None:
    dst = tmp_path / "blank_unnumbered.pdf"
    prefill_service.generate_blank_sheets_pdf_to_file(dst, count=2)
    texts = _page_texts(dst)
    assert all("/" not in t for t in texts), (
        "Unnumbered output should not contain 'n / N' fragments"
    )


def test_generate_blank_sheets_pdf_size_scales_sublinearly_with_n(
    tmp_path: Path,
) -> None:
    """The shared raster must be deduped so 50 pages << 50× single-page size."""
    one = tmp_path / "one.pdf"
    fifty = tmp_path / "fifty.pdf"
    prefill_service.generate_blank_sheets_pdf_to_file(one, count=1)
    prefill_service.generate_blank_sheets_pdf_to_file(fifty, count=50)
    one_bytes = one.stat().st_size
    fifty_bytes = fifty.stat().st_size
    assert fifty_bytes < one_bytes * 4, (
        f"Expected dedupe of shared raster — got {fifty_bytes} vs "
        f"50x1={one_bytes * 50}; 4x ceiling is generous and only sanity-checks "
        f"that the raster is not embedded 50 times."
    )


# ---------------------------------------------------------------------------
# ArUco markers must survive the clone+save round-trip
# ---------------------------------------------------------------------------


def _render_page_to_array(pdf_path: Path, page_index: int):
    doc = fitz.open(str(pdf_path))
    try:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=300, alpha=False)
        return cv2.imdecode(
            __import__("numpy").frombuffer(pix.tobytes("png"), dtype="uint8"),
            cv2.IMREAD_COLOR,
        )
    finally:
        doc.close()


def test_generated_blank_pages_keep_all_four_aruco_markers(tmp_path: Path) -> None:
    dst = tmp_path / "blank_aruco.pdf"
    prefill_service.generate_blank_sheets_pdf_to_file(dst, count=2)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    for i in range(2):
        img = _render_page_to_array(dst, i)
        _corners, ids, _rejected = detector.detectMarkers(img)
        detected = sorted(int(x) for x in (ids.flatten() if ids is not None else []))
        assert detected == [0, 1, 2, 3], (
            f"Page {i + 1} lost ArUco markers — detected {detected}"
        )


# ---------------------------------------------------------------------------
# Split / ZIP output
# ---------------------------------------------------------------------------


def test_generate_blank_sheets_split_zip_segments_match_page_cap(
    tmp_path: Path,
) -> None:
    dst = tmp_path / "blank_split.zip"
    meta = prefill_service.generate_blank_sheets_split_zip_to_file(
        dst,
        variant="MoE-April-2026-Landscape-NNQ25-0",
        count=7,
        max_pdf_mb=200,
        max_pdf_pages=3,
        include_page_numbers=True,
    )
    assert dst.exists()
    assert meta["count"] == 7
    assert meta["successes"] == 7
    assert len(meta["segments"]) == 3  # 3, 3, 1
    names = [s["name"] for s in meta["segments"]]
    assert names == [
        "blank_MoE-April-2026-Landscape-NNQ25-0_part_01_of_03.pdf",
        "blank_MoE-April-2026-Landscape-NNQ25-0_part_02_of_03.pdf",
        "blank_MoE-April-2026-Landscape-NNQ25-0_part_03_of_03.pdf",
    ]
    # First-page numbers in each segment should be 1, 4, 7 because the
    # page numbering is baked into the combined PDF before splitting.
    assert [s["first_page"] for s in meta["segments"]] == [1, 4, 7]
    assert [s["last_page"] for s in meta["segments"]] == [3, 6, 7]


def test_blank_split_zip_segments_are_valid_pdfs_with_continuous_numbering(
    tmp_path: Path,
) -> None:
    dst = tmp_path / "blank_split_continuous.zip"
    prefill_service.generate_blank_sheets_split_zip_to_file(
        dst,
        count=5,
        max_pdf_mb=200,
        max_pdf_pages=2,
        include_page_numbers=True,
    )
    expected_label_per_page: list[str] = []
    with zipfile.ZipFile(dst) as zf:
        entries = [n for n in zf.namelist() if n.lower().endswith(".pdf")]
        assert entries, "Split ZIP must contain at least one PDF segment"
        for name in entries:
            data = zf.read(name)
            seg_path = tmp_path / name.replace("/", "_")
            seg_path.write_bytes(data)
            for text in _page_texts(seg_path):
                expected_label_per_page.append(text)
    # Every page must include its (1-based) n / 5 stamp, continuous across segments.
    assert len(expected_label_per_page) == 5
    for i, body in enumerate(expected_label_per_page, start=1):
        assert f"{i} / 5" in body


def test_generate_blank_sheets_split_zip_rejects_invalid_caps(
    tmp_path: Path,
) -> None:
    dst = tmp_path / "split_bad.zip"
    with pytest.raises(ValueError, match="max_pdf_mb"):
        prefill_service.generate_blank_sheets_split_zip_to_file(
            dst, count=1, max_pdf_mb=0, max_pdf_pages=10
        )
    with pytest.raises(ValueError, match="max_pdf_pages"):
        prefill_service.generate_blank_sheets_split_zip_to_file(
            dst, count=1, max_pdf_mb=10, max_pdf_pages=0
        )


# ---------------------------------------------------------------------------
# HTTP API surface
# ---------------------------------------------------------------------------


def test_blank_variants_endpoint_lists_registered_options(
    client: TestClient,
) -> None:
    res = client.get("/api/v1/prefill/blank/variants")
    assert res.status_code == 200
    body = res.json()
    keys = {v["key"] for v in body["variants"]}
    assert "MoE-April-2026-Landscape-NNQ25-0" in keys
    assert "MoE-July-2026-Letter-Landscape-SMQ60-0" in keys
    assert body["default"] == "MoE-July-2026-Letter-Landscape-SMQ60-0"


def test_blank_post_returns_token_and_downloads_pdf(client: TestClient) -> None:
    res = client.post(
        "/api/v1/prefill/blank",
        data={
            "variant": "MoE-April-2026-Landscape-NNQ25-0",
            "count": "2",
            "include_page_numbers": "true",
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["count"] == 2
    assert body["successes"] == 2
    assert body["filename"].endswith(".pdf")
    assert body["filename"].startswith("blank_MoE-April-2026-Landscape-NNQ25-0_x2")
    assert body["split_pdfs"] is False
    assert body["page_numbers"] is True

    download = client.get(body["download_url"])
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/pdf"
    doc = fitz.open(stream=download.content, filetype="pdf")
    try:
        assert doc.page_count == 2
        texts = [doc.load_page(i).get_text() for i in range(doc.page_count)]
    finally:
        doc.close()
    assert any("1 / 2" in t for t in texts)
    assert any("2 / 2" in t for t in texts)


def test_blank_post_split_returns_zip_with_part_named_segments(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The runtime cap ``prefill_split_max_pdf_pages`` is schema-bounded to
    # ``ge=50`` so we can't drive it below that via the env var path. The
    # API endpoint reads its caps from ``webui.api.get_settings``, which is
    # imported as a name into the api module — monkeypatching that name
    # gives us a small-cap settings instance for this request without
    # rejecting at validation time. ``model_copy`` skips validators on
    # update so the test stays within Pydantic v2 contract.
    from webui import api as api_module
    from webui.settings import get_settings as real_get_settings

    real_settings = real_get_settings()
    patched_settings = real_settings.model_copy(
        update={"prefill_split_max_pdf_pages": 2}
    )
    monkeypatch.setattr(api_module, "get_settings", lambda: patched_settings)

    res = client.post(
        "/api/v1/prefill/blank",
        data={
            "variant": "MoE-April-2026-Landscape-NNQ25-0",
            "count": "5",
            "split_pdfs": "true",
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["split_pdfs"] is True
    assert body["filename"].endswith("_split.zip")
    assert len(body["segments"]) == 3
    segment_names = [s["name"] for s in body["segments"]]
    assert segment_names == [
        "blank_MoE-April-2026-Landscape-NNQ25-0_part_01_of_03.pdf",
        "blank_MoE-April-2026-Landscape-NNQ25-0_part_02_of_03.pdf",
        "blank_MoE-April-2026-Landscape-NNQ25-0_part_03_of_03.pdf",
    ]

    download = client.get(body["download_url"])
    assert download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(download.content)) as zf:
        assert sorted(zf.namelist()) == sorted(segment_names)


def test_blank_post_rejects_zero_count(client: TestClient) -> None:
    res = client.post("/api/v1/prefill/blank", data={"count": "0"})
    assert res.status_code == 422
    assert "count must be >= 1" in res.json()["detail"]


def test_blank_post_rejects_non_numeric_count(client: TestClient) -> None:
    res = client.post("/api/v1/prefill/blank", data={"count": "lots"})
    assert res.status_code == 422
    assert "count" in res.json()["detail"]


def test_blank_post_rejects_unknown_variant(client: TestClient) -> None:
    res = client.post(
        "/api/v1/prefill/blank",
        data={"variant": "no_such_variant", "count": "1"},
    )
    assert res.status_code == 422
    assert "Unknown blank sheet variant" in res.json()["detail"]


def test_blank_post_rejects_count_above_cap(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same monkeypatch strategy as the split test: override the
    # ``get_settings`` name in the api module so the endpoint sees a
    # bespoke cap without touching env-var validation.
    from webui import api as api_module
    from webui.settings import get_settings as real_get_settings

    patched_settings = real_get_settings().model_copy(
        update={"prefill_blank_max_sheets": 5}
    )
    monkeypatch.setattr(api_module, "get_settings", lambda: patched_settings)

    res = client.post(
        "/api/v1/prefill/blank",
        data={"variant": "MoE-April-2026-Landscape-NNQ25-0", "count": "6"},
    )
    assert res.status_code == 422
    detail = res.json()["detail"]
    assert "exceeds the per-request cap of 5" in detail
