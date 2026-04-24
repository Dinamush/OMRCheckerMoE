"""End-to-end tests for the /api/v1 surface.

The flow under test mirrors how a real client uses the API::

    create batch -> upload file -> set template/config -> process -> results

We run the engine against the existing ``samples/sample2/AdrianSample``
images so no new fixtures are needed. OpenCV UI calls are mocked in the
shared ``conftest.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webui.schemas import BatchStatus
from webui.services import batches as batches_service
from webui.services.omr import (
    _compute_dynamic_dimensions,
    _write_non_interactive_config,
)
from webui.settings import get_settings

def _create_batch(client: TestClient, name: str = "Integration Test") -> str:
    response = client.post("/api/v1/batches", json={"name": name})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _upload_image(client: TestClient, batch_id: str, path: Path) -> list[dict]:
    with path.open("rb") as fh:
        response = client.post(
            f"/api/v1/batches/{batch_id}/files",
            files=[("files", (path.name, fh, "image/png"))],
        )
    assert response.status_code == 201, response.text
    return response.json()


def _wait_for_status(
    client: TestClient, batch_id: str, terminal={"done", "failed"}, timeout: float = 30.0
) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/v1/batches/{batch_id}/status")
        assert response.status_code == 200
        data = response.json()
        if data["status"] in terminal:
            return data
        time.sleep(0.1)
    pytest.fail(f"Timed out waiting for terminal status; last payload: {data}")


def test_batch_lifecycle_crud(client: TestClient) -> None:
    response = client.get("/api/v1/batches")
    assert response.status_code == 200
    assert response.json() == []

    batch_id = _create_batch(client, "CRUD batch")

    response = client.get(f"/api/v1/batches/{batch_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "CRUD batch"
    assert body["status"] == "created"
    assert body["file_count"] == 0

    response = client.get("/api/v1/batches")
    assert len(response.json()) == 1

    response = client.delete(f"/api/v1/batches/{batch_id}")
    assert response.status_code == 204

    response = client.get(f"/api/v1/batches/{batch_id}")
    assert response.status_code == 404


def test_upload_and_list_files(
    client: TestClient, adrian_images: list[Path]
) -> None:
    batch_id = _create_batch(client, "Upload test")
    _upload_image(client, batch_id, adrian_images[0])

    response = client.get(f"/api/v1/batches/{batch_id}/files")
    assert response.status_code == 200
    files = response.json()
    assert len(files) == 1
    assert files[0]["name"].endswith(".png")
    assert files[0]["size_bytes"] > 0


def test_rejects_unsupported_filetype(client: TestClient) -> None:
    batch_id = _create_batch(client, "Bad upload")
    response = client.post(
        f"/api/v1/batches/{batch_id}/files",
        files=[("files", ("notes.txt", b"hello", "text/plain"))],
    )
    assert response.status_code == 400


def test_directory_import(
    client: TestClient, adrian_images: list[Path]
) -> None:
    batch_id = _create_batch(client, "Directory import test")
    source_dir = adrian_images[0].parent
    response = client.post(
        f"/api/v1/batches/{batch_id}/files/import",
        json={"source_dir": str(source_dir), "copy": True},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert len(payload["imported"]) == len(adrian_images)
    assert payload["skipped"] == []


def test_template_config_round_trip(
    client: TestClient, sample_template_body: dict, sample_config_body: dict
) -> None:
    batch_id = _create_batch(client, "Template round trip")

    response = client.put(
        f"/api/v1/batches/{batch_id}/template", json=sample_template_body
    )
    assert response.status_code == 200
    assert response.json()["status"] == "saved"

    response = client.put(
        f"/api/v1/batches/{batch_id}/config", json=sample_config_body
    )
    assert response.status_code == 200

    response = client.get(f"/api/v1/batches/{batch_id}/template")
    assert response.status_code == 200
    assert response.json() == sample_template_body

    response = client.get(f"/api/v1/batches/{batch_id}")
    batch = response.json()
    assert batch["has_template"] is True
    assert batch["has_config"] is True


def test_process_requires_template(
    client: TestClient, adrian_images: list[Path]
) -> None:
    batch_id = _create_batch(client, "Missing template")
    _upload_image(client, batch_id, adrian_images[0])

    response = client.post(f"/api/v1/batches/{batch_id}/process")
    assert response.status_code == 400
    assert "template" in response.json()["detail"].lower()


def test_full_process_flow_produces_results(
    client: TestClient,
    adrian_images: list[Path],
    sample_template_body: dict,
    sample_config_body: dict,
) -> None:
    batch_id = _create_batch(client, "Happy path")

    for image in adrian_images:
        _upload_image(client, batch_id, image)

    client.put(
        f"/api/v1/batches/{batch_id}/template", json=sample_template_body
    )
    client.put(
        f"/api/v1/batches/{batch_id}/config", json=sample_config_body
    )

    response = client.post(f"/api/v1/batches/{batch_id}/process")
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "queued"

    final = _wait_for_status(client, batch_id)
    assert final["status"] == "done", final
    assert final["processed_files"] == len(adrian_images)
    assert final["total_files"] == len(adrian_images)
    assert final["latest_processed_file"] == adrian_images[-1].name
    assert final["latest_dynamic_dimensions"] == _compute_dynamic_dimensions(
        adrian_images[-1]
    )

    response = client.get(f"/api/v1/batches/{batch_id}/results")
    assert response.status_code == 200
    results = response.json()
    assert results["batch_id"] == batch_id
    assert results["generated_csv"] is not None
    assert len(results["rows"]) >= 1
    assert "file_id" in results["columns"]
    assert "score" in results["columns"]

    response = client.get(f"/api/v1/batches/{batch_id}/results/download")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert len(response.content) > 0

    response = client.get(f"/api/v1/batches/{batch_id}/config")
    assert response.status_code == 200
    persisted_config = response.json()
    expected_dimensions = _compute_dynamic_dimensions(adrian_images[-1])
    assert persisted_config["dimensions"]["display_height"] == expected_dimensions["display_height"]
    assert persisted_config["dimensions"]["display_width"] == expected_dimensions["display_width"]
    assert (
        persisted_config["dimensions"]["processing_height"]
        == expected_dimensions["processing_height"]
    )
    assert (
        persisted_config["dimensions"]["processing_width"]
        == expected_dimensions["processing_width"]
    )


def test_ui_pages_render(
    client: TestClient, sample_template_body: dict, adrian_images: list[Path]
) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "OMRChecker" in response.text

    batch_id = _create_batch(client, "HTML test")
    _upload_image(client, batch_id, adrian_images[0])
    client.put(f"/api/v1/batches/{batch_id}/template", json=sample_template_body)

    response = client.get(f"/batches/{batch_id}")
    assert response.status_code == 200
    assert "HTML test" in response.text
    assert "template.json" in response.text
    assert "Template parameter help" in response.text
    assert "origin" in response.text
    assert "bubblesGap" in response.text
    assert "Template assets" in response.text


def test_staged_config_forces_non_interactive(tmp_path: Path) -> None:
    src = tmp_path / "config.json"
    dst = tmp_path / "staged_config.json"
    src.write_text(
        '{"dimensions":{"display_height":2480},"outputs":{"show_image_level":5}}',
        encoding="utf-8",
    )

    _write_non_interactive_config(src, dst)

    data = json.loads(dst.read_text(encoding="utf-8"))
    assert data["outputs"]["show_image_level"] == 0


_MARKER_TEMPLATE = {
    "pageDimensions": [300, 400],
    "bubbleDimensions": [25, 25],
    "fieldBlocks": {
        "MCQ_Block_1": {
            "fieldType": "QTYPE_MCQ4",
            "origin": [65, 60],
            "fieldLabels": ["q1..2"],
            "labelsGap": 52,
            "bubblesGap": 41,
        }
    },
    "preProcessors": [
        {
            "name": "CropOnMarkers",
            "options": {
                "relativePath": "omr_marker.jpg",
                "sheetToMarkerWidthRatio": 17,
            },
        }
    ],
}


def test_missing_template_asset_blocks_processing_with_clear_error(
    client: TestClient, adrian_images: list[Path]
) -> None:
    batch_id = _create_batch(client, "Missing marker asset")
    _upload_image(client, batch_id, adrian_images[0])
    client.put(f"/api/v1/batches/{batch_id}/template", json=_MARKER_TEMPLATE)

    response = client.post(f"/api/v1/batches/{batch_id}/process")
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "omr_marker.jpg" in detail
    assert "Template assets" in detail

    response = client.get(f"/api/v1/batches/{batch_id}/status")
    assert response.json()["status"] == "created"


def test_list_template_assets_reports_missing_required_asset(
    client: TestClient, adrian_images: list[Path]
) -> None:
    batch_id = _create_batch(client, "Assets listing")
    _upload_image(client, batch_id, adrian_images[0])
    client.put(f"/api/v1/batches/{batch_id}/template", json=_MARKER_TEMPLATE)

    response = client.get(f"/api/v1/batches/{batch_id}/assets")
    assert response.status_code == 200
    assets = response.json()
    assert len(assets) == 1
    assert assets[0]["name"] == "omr_marker.jpg"
    assert assets[0]["required"] is True
    assert assets[0]["present"] is False


def test_upload_template_asset_satisfies_preflight(
    client: TestClient, adrian_images: list[Path]
) -> None:
    batch_id = _create_batch(client, "Upload asset")
    _upload_image(client, batch_id, adrian_images[0])
    client.put(f"/api/v1/batches/{batch_id}/template", json=_MARKER_TEMPLATE)

    response = client.post(f"/api/v1/batches/{batch_id}/process")
    assert response.status_code == 400

    fake_marker_bytes = b"\xff\xd8\xff\xe0" + b"0" * 64
    response = client.post(
        f"/api/v1/batches/{batch_id}/assets",
        files=[("files", ("omr_marker.jpg", fake_marker_bytes, "image/jpeg"))],
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body[0]["name"] == "omr_marker.jpg"
    assert body[0]["present"] is True

    response = client.get(f"/api/v1/batches/{batch_id}/assets")
    assets = response.json()
    assert assets[0]["present"] is True
    assert assets[0]["size_bytes"] == len(fake_marker_bytes)

    settings = get_settings()
    missing = batches_service.missing_template_assets(batch_id, settings)
    assert missing == []

    response = client.delete(
        f"/api/v1/batches/{batch_id}/assets/omr_marker.jpg"
    )
    assert response.status_code == 204

    response = client.get(f"/api/v1/batches/{batch_id}/assets")
    assert response.json()[0]["present"] is False

    response = client.post(f"/api/v1/batches/{batch_id}/process")
    assert response.status_code == 400
    assert "omr_marker.jpg" in response.json()["detail"]


def test_asset_upload_rejects_bad_filenames(
    client: TestClient, adrian_images: list[Path]
) -> None:
    batch_id = _create_batch(client, "Bad asset upload")
    _upload_image(client, batch_id, adrian_images[0])

    response = client.post(
        f"/api/v1/batches/{batch_id}/assets",
        files=[("files", ("notes.txt", b"hello", "text/plain"))],
    )
    assert response.status_code == 400

    response = client.post(
        f"/api/v1/batches/{batch_id}/assets",
        files=[("files", ("template.json", b"{}", "application/json"))],
    )
    assert response.status_code == 400


def test_cancel_endpoint_cancels_queued_batch(
    client: TestClient, adrian_images: list[Path], sample_template_body: dict
) -> None:
    batch_id = _create_batch(client, "Queued cancel")
    _upload_image(client, batch_id, adrian_images[0])
    client.put(f"/api/v1/batches/{batch_id}/template", json=sample_template_body)

    settings = get_settings()
    batches_service.update_status(batch_id, BatchStatus.queued, settings=settings)

    response = client.post(f"/api/v1/batches/{batch_id}/cancel")
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "cancelled"

    response = client.get(f"/api/v1/batches/{batch_id}/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "cancelled"
    assert payload["cancel_requested"] is True


def test_restart_endpoint_requeues_non_running_batch(
    client: TestClient, adrian_images: list[Path], sample_template_body: dict, monkeypatch
) -> None:
    batch_id = _create_batch(client, "Restart me")
    _upload_image(client, batch_id, adrian_images[0])
    client.put(f"/api/v1/batches/{batch_id}/template", json=sample_template_body)

    settings = get_settings()
    batches_service.update_status(
        batch_id,
        BatchStatus.failed,
        last_error="Synthetic failure",
        settings=settings,
    )

    from webui.services import omr as omr_service

    def fake_run(batch_id_arg, settings_arg=None):
        batches_service.update_status(
            batch_id_arg,
            BatchStatus.done,
            settings=settings_arg or settings,
        )

    monkeypatch.setattr(omr_service, "run_batch_sync", fake_run)

    response = client.post(f"/api/v1/batches/{batch_id}/restart")
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "queued"

    response = client.get(f"/api/v1/batches/{batch_id}/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "done"


def test_directory_import_processes_with_dynamic_dimensions(
    client: TestClient,
    adrian_images: list[Path],
    sample_template_body: dict,
    sample_config_body: dict,
) -> None:
    batch_id = _create_batch(client, "Dynamic directory batch")
    source_dir = adrian_images[0].parent

    response = client.post(
        f"/api/v1/batches/{batch_id}/files/import",
        json={"source_dir": str(source_dir), "copy": True},
    )
    assert response.status_code == 201, response.text

    client.put(f"/api/v1/batches/{batch_id}/template", json=sample_template_body)
    client.put(f"/api/v1/batches/{batch_id}/config", json=sample_config_body)

    response = client.post(f"/api/v1/batches/{batch_id}/process")
    assert response.status_code == 202, response.text

    final = _wait_for_status(client, batch_id)
    assert final["status"] == "done", final
    assert final["processed_files"] == len(adrian_images)
    assert final["total_files"] == len(adrian_images)

    latest_expected = _compute_dynamic_dimensions(adrian_images[-1])
    assert final["latest_dynamic_dimensions"] == latest_expected

    response = client.get(f"/api/v1/batches/{batch_id}/config")
    assert response.status_code == 200
    persisted_config = response.json()
    assert persisted_config["dimensions"]["processing_width"] == latest_expected["processing_width"]
    assert persisted_config["dimensions"]["processing_height"] == latest_expected["processing_height"]
    assert persisted_config["outputs"]["show_image_level"] == sample_config_body["outputs"]["show_image_level"]
