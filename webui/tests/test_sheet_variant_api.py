"""End-to-end /api/v1/settings tests for the ``sheet_variant`` field."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.tests.utils import setup_mocker_patches
from webui.app import create_app
from webui.settings import get_settings


@pytest.fixture
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    cache_dir = tmp_path / "settings_cache"
    cache_dir.mkdir()
    monkeypatch.setenv("OMR_WEBUI_CACHE_ROOT", str(cache_dir))
    get_settings.cache_clear()
    yield cache_dir
    get_settings.cache_clear()


@pytest.fixture
def client(storage_root: Path, cache_root: Path, mocker) -> Iterator[TestClient]:
    setup_mocker_patches(mocker)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_meta_includes_variant_with_v1_legacy_default(client: TestClient) -> None:
    body = client.get("/api/v1/settings/meta").json()
    assert "sheet_variant" in body["mutable_keys"]
    assert body["defaults"]["sheet_variant"] == "v1_legacy"
    desc = body["descriptions"]["sheet_variant"]
    assert "v1_legacy" in desc and "v2_optimized" in desc


def test_put_then_get_round_trip_persists_v2(client: TestClient) -> None:
    put_resp = client.put(
        "/api/v1/settings", json={"sheet_variant": "v2_optimized"}
    )
    assert put_resp.status_code == 200, put_resp.text
    assert put_resp.json()["sheet_variant"] == "v2_optimized"

    follow_up = client.get("/api/v1/settings").json()
    assert follow_up["sheet_variant"] == "v2_optimized"


def test_put_rejects_invalid_variant_through_api(client: TestClient) -> None:
    resp = client.put(
        "/api/v1/settings", json={"sheet_variant": "v3_experimental"}
    )
    assert resp.status_code == 422, resp.text
