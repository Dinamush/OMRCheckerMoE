"""JSON API router for the OMRChecker Web UI.

Exposed under ``/api/v1``. All mutations live here so that the HTML UI
and any third-party API consumer go through identical codepaths.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import secrets
import threading
import time as _time_mod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import csv
import io
import os
import tempfile
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, StreamingResponse

from webui.schemas import (
    Batch,
    BatchCreate,
    BatchRotationUpdate,
    BatchStatus,
    BatchStatusResponse,
    DirectoryImportRequest,
    FileRef,
    ImportResult,
    ProcessAccepted,
    ResultsPayload,
    TemplateAssetRef,
)
from webui.services import batches as batches_service
from webui.services import omr as omr_service
from webui.services import prefill as prefill_service
from webui.services import presets as presets_service
from webui.services import test_csv as test_csv_service
from webui.services.scan_simulation import normalize_realism_preset
from webui.services.student_fill import (
    MARKING_PROFILES,
    list_marking_profiles,
    normalize_marking_profile,
)
from webui import log_stream
from webui.schemas_settings import (
    RuntimeSettingsResponse,
    RuntimeSettingsUpdate,
    SettingsMetaResponse,
    build_meta_response,
)
from webui.services.batches import BatchNotFound, InvalidBatchRequest
from webui.settings import (
    RUNTIME_MUTABLE_SETTINGS,
    Settings,
    _load_overrides,
    get_settings,
    reload_settings,
    write_overrides,
)

router = APIRouter(prefix="/api/v1", tags=["omr"])


# Built-in template/config for sheets produced by the Prefill page.
#
# The generated sheets include four ArUco corner markers, but the OMR engine
# still needs a template to map the cropped page into fields and bubbles.
# Auto-attaching these documents keeps the generated-sheet workflow one-click:
# generate/download prefilled sheets -> upload/process without manually adding
# template.json/config.json.
_PREFILLED_25Q_TEMPLATE: dict[str, Any] = {
    "pageDimensions": [666, 515],
    "bubbleDimensions": [10, 10],
    "customLabels": {"CandidateNumber": ["cand1..10"]},
    "outputColumns": ["CandidateNumber", "q1..25"],
    "fieldBlocks": {
        "CandidateNumber": {
            "origin": [430, 103],
            "bubblesGap": 10.0,
            "labelsGap": 21.5,
            "fieldLabels": ["cand1..10"],
            "fieldType": "QTYPE_INT",
        },
        "q01block": {
            "origin": [52.7, 259.3],
            "bubblesGap": 20.0,
            "labelsGap": 41.9,
            "fieldLabels": ["q1..5"],
            "emptyValue": "NR",
            "fieldType": "QTYPE_MCQ4",
        },
        "q06block": {
            "origin": [180.9, 259.3],
            "bubblesGap": 20.0,
            "labelsGap": 41.9,
            "fieldLabels": ["q6..10"],
            "emptyValue": "NR",
            "fieldType": "QTYPE_MCQ4",
        },
        "q11block": {
            "origin": [309.8, 259.3],
            "bubblesGap": 19.8,
            "labelsGap": 41.9,
            "fieldLabels": ["q11..15"],
            "emptyValue": "NR",
            "fieldType": "QTYPE_MCQ4",
        },
        "q16block": {
            "origin": [435.9, 259.3],
            "bubblesGap": 20.0,
            "labelsGap": 41.9,
            "fieldLabels": ["q16..20"],
            "emptyValue": "NR",
            "fieldType": "QTYPE_MCQ4",
        },
        "q21block": {
            "origin": [566.3, 259.3],
            "bubblesGap": 20.0,
            "labelsGap": 41.9,
            "fieldLabels": ["q21..25"],
            "emptyValue": "NR",
            "fieldType": "QTYPE_MCQ4",
        },
    },
    "preProcessors": [
        {
            "name": "CropOnMarkers",
            "options": {
                "type": "aruco",
                "arucoDictionary": "DICT_4X4_50",
                "arucoCornerIds": [0, 1, 2, 3],
                "preserveFullImage": True,
                "referenceMarkerCenters": [
                    [13.5, 13.2],
                    [651.5, 13.2],
                    [13.5, 499.0],
                    [651.5, 499.0],
                ],
            },
        }
    ],
}

_PREFILLED_25Q_CONFIG: dict[str, Any] = {
    "dimensions": {
        "display_height": 515,
        "display_width": 666,
        "processing_height": 515,
        "processing_width": 666,
    },
    "outputs": {"show_image_level": 0},
}

# ---------------------------------------------------------------------------
# Prefill backpressure: cap concurrent batch jobs server-side. A 5k-row PDF
# crashed the server in stress testing because each concurrent batch spawns
# its own pool of (cpu_count - 1) workers — N concurrent batches × W workers
# easily exceeds RAM. The semaphore enforces "at most this many heavy
# prefill batches at once" and excess requests get a fast HTTP 429.
_PREFILL_BATCH_LIMIT = max(1, int(os.environ.get("OMR_WEBUI_PREFILL_CONCURRENCY", "2")))
_PREFILL_BATCH_SEM = threading.BoundedSemaphore(_PREFILL_BATCH_LIMIT)

# Single-sheet endpoint also forks a process pool for PNG/PDF rendering, so
# uncapped concurrency (e.g. 50 simultaneous requests) can wedge the host.
# Allow a higher limit than batches but still bounded.
_PREFILL_SINGLE_LIMIT = max(2, int(os.environ.get("OMR_WEBUI_PREFILL_SINGLE_CONCURRENCY", "8")))
_PREFILL_SINGLE_SEM = threading.BoundedSemaphore(_PREFILL_SINGLE_LIMIT)

# /prefill/sample is fired in parallel by the in-page comparison gallery
# (4 concurrent requests on page load). Cap server-side concurrency so a
# misbehaving client (or a tight retry loop) can't pile up dozens of PNG
# renders simultaneously.
_PREFILL_SAMPLE_LIMIT = max(2, int(os.environ.get("OMR_WEBUI_PREFILL_SAMPLE_CONCURRENCY", "6")))
_PREFILL_SAMPLE_SEM = threading.BoundedSemaphore(_PREFILL_SAMPLE_LIMIT)

# Hard caps on prefill batch sizes are now sourced from the runtime
# Settings model: ``settings.prefill_pdf_max_rows``, ``settings.prefill_zip_max_rows``,
# and ``settings.prefill_csv_max_bytes``. Reading them per-request means
# operators can change limits via the /settings page (or
# OMR_WEBUI_PREFILL_*_MAX_ROWS env vars) without restarting the server.

async def _ensure_large_multipart(request: Request) -> None:
    """Pre-parse multipart bodies with a generous ``max_part_size``.

    Starlette's ``MultiPartParser`` caps each individual form part at
    1 MiB by default, which causes routes like ``/prefill/batch`` to
    reject 30 000-row CSVs (~3.5 MiB submitted as a single ``csv_text``
    field) with *"Part exceeded maximum size of 1024KB."* even though
    the application-layer caps allow far larger payloads.

    Calling ``request.form(max_part_size=...)`` here parses and caches
    the form on ``request._form`` once with the operator-controlled
    ``settings.max_upload_bytes`` cap. FastAPI's later ``Form``/``File``
    machinery picks up the cached ``FormData`` instead of re-parsing,
    so the larger limit takes effect transparently for the route.
    Application-layer caps (``prefill_csv_max_bytes`` and per-row
    counts) still enforce upper bounds downstream.
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type.lower():
        return
    settings = get_settings()
    await request.form(
        max_part_size=settings.max_upload_bytes,
        max_files=10_000,
        max_fields=10_000,
    )


def _run_batch_pdf(
    rows: list[dict[str, Any]],
    dst_path: Path,
    realism_preset: str,
    include_page_numbers: bool,
    marking_profile: str = "none",
    answers: Any = None,
) -> dict:
    """Adapter that calls ``generate_batch_pdf_to_file`` with the right kwargs.

    Tests monkeypatch ``generate_batch_pdf_to_file`` with a stub that may
    not yet know newer parameters (``include_page_numbers``, ``marking_profile``,
    ``answers``); introspecting the signature here keeps those fixtures green
    while still threading every flag through for production callers.
    """
    import inspect

    target = prefill_service.generate_batch_pdf_to_file
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs: dict[str, Any] = {"realism_preset": realism_preset}
    if "include_page_numbers" in params:
        kwargs["include_page_numbers"] = include_page_numbers
    if "marking_profile" in params:
        kwargs["marking_profile"] = marking_profile
    if "answers" in params:
        kwargs["answers"] = answers
    return target(rows, dst_path, **kwargs)


def _run_batch_zip(
    rows: list[dict[str, Any]],
    dst_path: Path,
    realism_preset: str,
    marking_profile: str = "none",
    answers: Any = None,
) -> dict:
    """Adapter for ``generate_batch_zip_to_file`` that mirrors :func:`_run_batch_pdf`."""
    import inspect

    target = prefill_service.generate_batch_zip_to_file
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs: dict[str, Any] = {"realism_preset": realism_preset}
    if "marking_profile" in params:
        kwargs["marking_profile"] = marking_profile
    if "answers" in params:
        kwargs["answers"] = answers
    return target(rows, dst_path, **kwargs)


def _run_batch_grouped_zip(
    rows: list[dict[str, Any]],
    dst_path: Path,
    realism_preset: str,
    group_by: str,
    include_page_numbers: bool,
    marking_profile: str = "none",
    answers: Any = None,
) -> dict:
    """Adapter for ``generate_batch_grouped_zip_to_file`` matching the
    same shape as :func:`_run_batch_pdf` / :func:`_run_batch_zip`. Tests may
    monkeypatch the underlying service function with a stub that lacks
    newer kwargs, so we feature-detect the signature like the siblings do.
    """
    import inspect

    target = prefill_service.generate_batch_grouped_zip_to_file
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs: dict[str, Any] = {
        "group_by": group_by,
        "realism_preset": realism_preset,
    }
    if "include_page_numbers" in params:
        kwargs["include_page_numbers"] = include_page_numbers
    if "marking_profile" in params:
        kwargs["marking_profile"] = marking_profile
    if "answers" in params:
        kwargs["answers"] = answers
    return target(rows, dst_path, **kwargs)


def _run_batch_split_pdf(
    rows: list[dict[str, Any]],
    dst_path: Path,
    realism_preset: str,
    max_pdf_mb: int,
    max_pdf_pages: int,
    include_page_numbers: bool,
    marking_profile: str = "none",
    answers: Any = None,
    stem: str = "prefilled_sheets",
) -> dict:
    """Adapter for ``generate_batch_split_pdf_to_zip`` matching the
    signature pattern of the other ``_run_batch_*`` adapters.

    Falls back gracefully when older test monkeypatches replace the
    service function with a stub that lacks newer kwargs.
    """
    import inspect

    target = prefill_service.generate_batch_split_pdf_to_zip
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs: dict[str, Any] = {
        "max_pdf_mb": max_pdf_mb,
        "max_pdf_pages": max_pdf_pages,
        "realism_preset": realism_preset,
    }
    if "include_page_numbers" in params:
        kwargs["include_page_numbers"] = include_page_numbers
    if "marking_profile" in params:
        kwargs["marking_profile"] = marking_profile
    if "answers" in params:
        kwargs["answers"] = answers
    if "stem" in params:
        kwargs["stem"] = stem
    return target(rows, dst_path, **kwargs)


def _run_batch_grouped_split_zip(
    rows: list[dict[str, Any]],
    dst_path: Path,
    realism_preset: str,
    group_by: str,
    max_pdf_mb: int,
    max_pdf_pages: int,
    include_page_numbers: bool,
    marking_profile: str = "none",
    answers: Any = None,
) -> dict:
    """Adapter for ``generate_batch_grouped_split_zip_to_file``."""
    import inspect

    target = prefill_service.generate_batch_grouped_split_zip_to_file
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs: dict[str, Any] = {
        "group_by": group_by,
        "max_pdf_mb": max_pdf_mb,
        "max_pdf_pages": max_pdf_pages,
        "realism_preset": realism_preset,
    }
    if "include_page_numbers" in params:
        kwargs["include_page_numbers"] = include_page_numbers
    if "marking_profile" in params:
        kwargs["marking_profile"] = marking_profile
    if "answers" in params:
        kwargs["answers"] = answers
    return target(rows, dst_path, **kwargs)


# Download token store: maps token -> (tmp_path, media_type, filename, expires_at)
# Tokens are single-use and expire after 10 minutes so orphaned files are cleaned up.
_DOWNLOAD_STORE: dict[str, tuple[Path, str, str, float]] = {}
_DOWNLOAD_STORE_LOCK = threading.Lock()

def _register_download(tmp_path: Path, media_type: str, filename: str) -> str:
    """Store a completed batch file and return a one-time download token."""
    token = secrets.token_urlsafe(24)
    expires_at = _time_mod.monotonic() + 600  # 10 minutes
    with _DOWNLOAD_STORE_LOCK:
        # Evict any expired tokens first
        expired = [k for k, (_, _, _, exp) in _DOWNLOAD_STORE.items() if _time_mod.monotonic() > exp]
        for k in expired:
            try:
                _DOWNLOAD_STORE[k][0].unlink(missing_ok=True)
            except OSError:
                pass
            del _DOWNLOAD_STORE[k]
        _DOWNLOAD_STORE[token] = (tmp_path, media_type, filename, expires_at)
    return token


def _pop_download_entry(token: str) -> tuple[Path, str, str, float]:
    """Return and consume a one-time download entry."""
    with _DOWNLOAD_STORE_LOCK:
        entry = _DOWNLOAD_STORE.pop(token, None)
    if entry is None:
        raise HTTPException(status_code=404, detail="Download link not found or already used.")
    return entry


def _download_entry_response(
    token: str,
    background_tasks: BackgroundTasks,
) -> FileResponse:
    """Serve a registered one-time download token and clean up its temp file."""
    tmp_path, media_type, filename, expires_at = _pop_download_entry(token)
    if not tmp_path.exists():
        raise HTTPException(status_code=410, detail="File no longer available.")
    if _time_mod.monotonic() > expires_at:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=410, detail="Download link has expired.")

    def _cleanup():
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    background_tasks.add_task(_cleanup)
    return FileResponse(
        path=str(tmp_path),
        media_type=media_type,
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        background=background_tasks,
    )


def _is_prefilled_sheet_upload(filename: str | None) -> bool:
    """Return true for files produced by this app's Prefill page."""
    name = (filename or "").lower()
    return "prefilled_sheet" in name or "prefilled_sheets" in name


def _attach_prefilled_25q_defaults(
    batch_id: str,
    settings: Settings,
    *,
    reason: str,
) -> None:
    """Attach the built-in 25Q prefill template/config if absent.

    We never overwrite user-supplied documents. This only fills the common
    gap where the upload was generated by the app's Prefill page and therefore
    has ArUco markers but no explicit batch template yet.
    """
    attached: list[str] = []
    if batches_service.get_json_document(batch_id, "template", settings) is None:
        batches_service.save_json_document(
            batch_id,
            "template",
            copy.deepcopy(_PREFILLED_25Q_TEMPLATE),
            settings,
        )
        attached.append("template.json")
    if batches_service.get_json_document(batch_id, "config", settings) is None:
        batches_service.save_json_document(
            batch_id,
            "config",
            copy.deepcopy(_PREFILLED_25Q_CONFIG),
            settings,
        )
        attached.append("config.json")

    batches_service.update_batch_metadata(
        batch_id,
        {
            "input_profile": "prefilled_25q",
            "auto_attached_template": True,
            "auto_attached_template_reason": reason,
        },
        settings,
    )
    if attached:
        logger.info(
            "Auto-attached prefilled 25Q defaults | batch=%s | files=%s | reason=%s",
            batch_id,
            ", ".join(attached),
            reason,
        )


def _maybe_attach_prefilled_25q_defaults(
    batch_id: str,
    settings: Settings,
    *,
    reason: str,
) -> bool:
    """Attach defaults for known prefilled-sheet batches.

    Returns ``True`` when the batch is or has been marked as a prefilled 25Q
    batch. This is used by the process guard to repair existing batches that
    were uploaded before the template was auto-attached.
    """
    metadata = batches_service.get_batch_metadata(batch_id, settings)
    if metadata.get("input_profile") != "prefilled_25q":
        return False
    _attach_prefilled_25q_defaults(batch_id, settings, reason=reason)
    return True


# ---------------------------------------------------------------------------
# Log streaming
# ---------------------------------------------------------------------------


@router.get("/logs/poll")
async def logs_poll(since: int = -1) -> dict:
    """Return log entries with sequence number > ``since``.

    Used by the front-end log panel which polls once per second over plain
    HTTP. WebView2 has known SSE buffering issues so this is the preferred
    transport in the desktop wrapper.
    """
    return log_stream.poll(since=since)


@router.get("/logs/stream")
async def logs_stream() -> StreamingResponse:
    """Server-Sent Events stream of log lines (non-WebView2 clients)."""
    return StreamingResponse(
        log_stream.stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------


@router.get("/system/info")
async def system_info() -> dict:
    """Return static system capabilities: GPU status and default worker count."""
    import os
    from src.utils.gpu import gpu_status, is_gpu_available
    from webui.services.omr import _default_max_workers
    return {
        "gpu_available": is_gpu_available(),
        "gpu_status": gpu_status(),
        "cpu_count": os.cpu_count(),
        "default_max_workers": _default_max_workers(),
    }


# ---------------------------------------------------------------------------
# Runtime-mutable settings (used by the /settings UI)
# ---------------------------------------------------------------------------


def _settings_response(settings: Settings) -> RuntimeSettingsResponse:
    """Project the live :class:`Settings` instance into the API response."""
    return RuntimeSettingsResponse(
        **{key: getattr(settings, key) for key in RUNTIME_MUTABLE_SETTINGS}
    )


@router.get("/settings", response_model=RuntimeSettingsResponse)
async def get_runtime_settings(
    settings: Settings = Depends(get_settings),
) -> RuntimeSettingsResponse:
    """Return the current value of every runtime-mutable setting."""
    return _settings_response(settings)


@router.put("/settings", response_model=RuntimeSettingsResponse)
async def update_runtime_settings(
    payload: RuntimeSettingsUpdate,
    settings: Settings = Depends(get_settings),
) -> RuntimeSettingsResponse:
    """Persist runtime overrides, reload settings, and return the new state.

    Only fields explicitly present in the request body are written. This
    lets the UI send PATCH-style partial updates (toggle a single switch)
    without round-tripping every setting on every save. Unknown keys are
    rejected by ``RuntimeSettingsUpdate`` (``extra=\"forbid\"``) so the
    allowlist is enforced at the schema layer, not by post-hoc filtering.
    """
    new_values = payload.model_dump(exclude_unset=True)
    if not new_values:
        # No-op PUT: don't touch the overrides file, just echo current state.
        return _settings_response(settings)

    # Storage_root is the SEP-friendly default location for the overrides
    # file (see ``Settings.overrides_path``). ``_load_overrides`` and
    # ``write_overrides`` both accept either storage_root or cache_root
    # and resolve the actual path through the live Settings instance.
    overrides_root = settings.storage_root
    merged = _load_overrides(overrides_root)
    # Log every actual change before writing so a failed disk write still
    # produces an audit trail of what the operator attempted.
    for key, new_val in new_values.items():
        old_val = getattr(settings, key)
        if old_val != new_val:
            logger.info(
                "Settings updated | key=%s | old=%s | new=%s",
                key, old_val, new_val,
            )
        merged[key] = new_val

    write_overrides(merged, overrides_root)
    fresh = reload_settings()
    return _settings_response(fresh)


@router.get("/settings/meta", response_model=SettingsMetaResponse)
async def get_runtime_settings_meta() -> SettingsMetaResponse:
    """Return descriptions + defaults for every mutable setting.

    The ``/settings`` UI uses this to render field labels, helper text
    under each input, and a per-field \"Reset to default\" button.
    """
    return build_meta_response()


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


@router.get("/presets")
async def list_presets(
    settings: Settings = Depends(get_settings),
) -> list[str]:
    """Return the names of all available presets."""
    return presets_service.list_presets(settings)


@router.get("/presets/{preset_name}")
async def get_preset(
    preset_name: str,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Return all JSON documents (template/config/evaluation) for a preset."""
    try:
        docs = presets_service.get_preset_documents(preset_name, settings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not docs:
        raise HTTPException(status_code=404, detail=f"Preset {preset_name!r} not found.")
    return docs


@router.post("/batches/{batch_id}/preset")
async def apply_preset(
    batch_id: str,
    preset_name: str = Body(..., embed=True),
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    """Copy all files from a preset (template, config, assets) into a batch."""
    batch_root = settings.ensure_storage() / batch_id
    if not batch_root.is_dir():
        raise HTTPException(status_code=404, detail=f"Batch {batch_id!r} not found.")
    try:
        presets_service.apply_preset_to_batch(batch_root, preset_name, settings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"status": "ok", "preset": preset_name}


def _handle_errors(func):
    """Wrap service calls so our custom exceptions map to HTTP status codes."""
    from functools import wraps

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except BatchNotFound as exc:
            raise HTTPException(status_code=404, detail=f"Batch not found: {exc}")
        except InvalidBatchRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return wrapper


@router.post("/batches", response_model=Batch, status_code=status.HTTP_201_CREATED)
@_handle_errors
async def create_batch(
    payload: BatchCreate,
    settings: Settings = Depends(get_settings),
) -> Batch:
    return batches_service.create_batch(payload.name, settings)


@router.get("/batches", response_model=list[Batch])
@_handle_errors
async def list_batches(settings: Settings = Depends(get_settings)) -> list[Batch]:
    return batches_service.list_batches(settings)


@router.get("/batches/{batch_id}", response_model=Batch)
@_handle_errors
async def get_batch(
    batch_id: str, settings: Settings = Depends(get_settings)
) -> Batch:
    return batches_service.get_batch(batch_id, settings)


@router.delete("/batches/{batch_id}", status_code=status.HTTP_204_NO_CONTENT)
@_handle_errors
async def delete_batch(
    batch_id: str, settings: Settings = Depends(get_settings)
) -> None:
    batches_service.delete_batch(batch_id, settings)


@router.put("/batches/{batch_id}/rotation", response_model=Batch)
@_handle_errors
async def update_batch_rotation(
    batch_id: str,
    payload: BatchRotationUpdate,
    settings: Settings = Depends(get_settings),
) -> Batch:
    return batches_service.set_rotation(batch_id, payload.rotation_degrees, settings)


@router.get("/batches/{batch_id}/files", response_model=list[FileRef])
@_handle_errors
async def list_files(
    batch_id: str, settings: Settings = Depends(get_settings)
) -> list[FileRef]:
    return batches_service.list_files(batch_id, settings)


def _is_pdf_upload(upload: UploadFile) -> bool:
    """Return True when the uploaded file is a PDF by name or content-type."""
    name = (upload.filename or "").lower()
    if name.endswith(".pdf"):
        return True
    content_type = (upload.content_type or "").lower()
    return content_type == "application/pdf"


@router.post(
    "/batches/{batch_id}/files",
    dependencies=[Depends(_ensure_large_multipart)],
)
@_handle_errors
async def upload_files(
    batch_id: str,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
):
    """Accept image + PDF uploads.

    Image uploads (PNG / JPG / JPEG) are written synchronously and the
    endpoint returns ``201`` with the resulting :class:`FileRef` list.

    PDF uploads are scheduled as a background task because a 20 000-page PDF
    can take minutes to render; the endpoint returns ``202`` with
    ``{"processing": True, "files": [...image refs already saved...]}``.
    The frontend polls ``/batches/{batch_id}/status`` for the split
    progress and refreshes its file list when ``pdf_split_total`` returns
    to zero (i.e. the background task finished).
    """
    from fastapi.responses import JSONResponse

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    logger.info(
        "Upload received | batch=%s | files=%d | names=%s",
        batch_id,
        len(files),
        [u.filename for u in files],
    )
    image_refs: list[FileRef] = []
    # Pre-staged PDF uploads: each entry is (display_filename, on-disk path,
    # size_bytes).  The bytes have already been streamed to that path so we
    # never hold the whole PDF in RAM — critical for multi-GiB uploads.
    pdf_jobs: list[tuple[str, Path, int]] = []
    inferred_preset: str | None = None
    has_prefilled_sheet_upload = False
    scratch_root = batches_service.pdf_split_scratch_root(settings)
    for upload in files:
        name_lower = (upload.filename or "").lower()
        if _is_prefilled_sheet_upload(upload.filename):
            has_prefilled_sheet_upload = True
        for candidate in ("adversarial", "moderate", "subtle"):
            if candidate in name_lower:
                if inferred_preset is None:
                    inferred_preset = candidate
                break
        if _is_pdf_upload(upload):
            # Stream the multipart body chunks straight to disk.  For a 3.6 GiB
            # PDF this avoids both a ~30s in-RAM slurp via ``upload.read()``
            # and a subsequent identical-size second write inside the splitter
            # — total time-to-first-page is dominated by a single sequential
            # write of ~15s instead of two passes of ~30s.
            staged_name = (
                f"{uuid.uuid4().hex}_{Path(upload.filename or 'upload.pdf').name}"
            )
            staged_path = scratch_root / staged_name
            total = 0
            try:
                with staged_path.open("wb") as fh:
                    while True:
                        chunk = await upload.read(1 << 20)  # 1 MiB chunks
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > settings.max_upload_bytes:
                            fh.close()
                            staged_path.unlink(missing_ok=True)
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    f"File {upload.filename!r} exceeds "
                                    f"max_upload_bytes "
                                    f"({settings.max_upload_bytes} bytes)"
                                ),
                            )
                        fh.write(chunk)
            except HTTPException:
                raise
            except Exception:
                staged_path.unlink(missing_ok=True)
                raise
            pdf_jobs.append(
                (upload.filename or "upload.pdf", staged_path, total)
            )
            continue
        # Non-PDF (image) uploads stay on the in-memory path: they're tiny
        # relative to PDFs and the synchronous save_uploaded_file expects
        # bytes.  Enforce max_upload_bytes here.
        data = await upload.read()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File {upload.filename!r} exceeds max_upload_bytes "
                    f"({settings.max_upload_bytes} bytes)"
                ),
            )
        refs = await asyncio.to_thread(
            batches_service.save_uploaded_file,
            batch_id,
            upload.filename or "upload",
            data,
            settings,
        )
        image_refs.extend(refs)

    if inferred_preset is not None:
        batches_service.update_batch_metadata(
            batch_id,
            {"inferred_realism_preset": inferred_preset},
            settings,
        )
        logger.info(
            "Inferred realism preset from upload filename | batch=%s | preset=%s",
            batch_id,
            inferred_preset,
        )

    if has_prefilled_sheet_upload:
        _attach_prefilled_25q_defaults(
            batch_id,
            settings,
            reason="upload filename matched prefilled_sheet(s)",
        )

    if pdf_jobs:
        # Schedule PDF rendering as background work. BackgroundTasks runs
        # after the response is sent in production; in TestClient it runs
        # synchronously, which is exactly what the test harness expects.
        #
        # Each task is wrapped in a closure so that any exception (corrupt
        # PDF, disk full, etc.) is caught and persisted into batch metadata
        # rather than silently swallowed by the BackgroundTasks runner.
        for filename, staged_path, size_bytes in pdf_jobs:
            stem = Path(filename).stem

            def _run_pdf_split(fn=filename, sp=staged_path, sz=size_bytes, s=stem):
                logger.info(
                    "PDF split start | batch=%s | file=%s | bytes=%d",
                    batch_id, fn, sz,
                )
                try:
                    refs = batches_service.save_uploaded_pdf_from_path(
                        batch_id, fn, sp, settings
                    )
                    logger.info(
                        "PDF split done | batch=%s | file=%s | pages=%d",
                        batch_id, fn, len(refs),
                    )
                except Exception as exc:  # noqa: BLE001
                    # Audit fix API-6: previously the full exception string
                    # (often containing filesystem paths) was persisted into
                    # metadata and echoed to the public status endpoint.
                    # Log the full traceback server-side; record a generic
                    # user-facing message keyed by stem only.
                    logger.exception(
                        "Background PDF split failed | batch=%s | file=%s | exc_type=%s",
                        batch_id, fn, type(exc).__name__,
                    )
                    error_msg = (
                        f"{s}: PDF split failed ({type(exc).__name__}). "
                        f"See server logs for details."
                    )
                    batches_service._record_pdf_split_error(batch_id, settings, error_msg)
                finally:
                    # save_uploaded_pdf_from_path already deletes the staged
                    # file on success; defensively clean up if it leaked
                    # (e.g. before this background task even started running).
                    try:
                        Path(sp).unlink(missing_ok=True)
                    except OSError:
                        pass

            background_tasks.add_task(_run_pdf_split)
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "processing": True,
                "files": [ref.model_dump() for ref in image_refs],
                "pdf_count": len(pdf_jobs),
            },
        )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content=[ref.model_dump() for ref in image_refs],
    )


@router.post(
    "/batches/{batch_id}/files/import",
    response_model=ImportResult,
    status_code=status.HTTP_201_CREATED,
)
@_handle_errors
async def import_from_directory(
    batch_id: str,
    payload: DirectoryImportRequest,
    settings: Settings = Depends(get_settings),
) -> ImportResult:
    imported, skipped = batches_service.import_directory(
        batch_id, payload.source_dir, payload.copy_files, settings
    )
    return ImportResult(imported=imported, skipped=skipped)


@router.delete(
    "/batches/{batch_id}/files/{filename}",
    status_code=status.HTTP_204_NO_CONTENT,
)
@_handle_errors
async def delete_file(
    batch_id: str,
    filename: str,
    settings: Settings = Depends(get_settings),
) -> None:
    batches_service.delete_file(batch_id, filename, settings)


@router.get("/batches/{batch_id}/files/{filename}/preview")
@_handle_errors
async def preview_file(
    batch_id: str,
    filename: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    resolved = batches_service.resolve_input_file(batch_id, filename, settings)
    return FileResponse(resolved, filename=resolved.name)


@router.get(
    "/batches/{batch_id}/assets",
    response_model=list[TemplateAssetRef],
)
@_handle_errors
async def list_template_assets(
    batch_id: str, settings: Settings = Depends(get_settings)
) -> list[TemplateAssetRef]:
    return batches_service.list_template_assets(batch_id, settings)


@router.post(
    "/batches/{batch_id}/assets",
    response_model=list[TemplateAssetRef],
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(_ensure_large_multipart)],
)
@_handle_errors
async def upload_template_assets(
    batch_id: str,
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
) -> list[TemplateAssetRef]:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    stored: list[TemplateAssetRef] = []
    for upload in files:
        data = await upload.read()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File {upload.filename!r} exceeds max_upload_bytes "
                    f"({settings.max_upload_bytes} bytes)"
                ),
            )
        stored.append(
            batches_service.save_template_asset(
                batch_id,
                upload.filename or "asset",
                data,
                settings,
            )
        )
    return stored


@router.delete(
    "/batches/{batch_id}/assets/{filename}",
    status_code=status.HTTP_204_NO_CONTENT,
)
@_handle_errors
async def delete_template_asset(
    batch_id: str,
    filename: str,
    settings: Settings = Depends(get_settings),
) -> None:
    batches_service.delete_template_asset(batch_id, filename, settings)


@router.get("/batches/{batch_id}/assets/{filename}/preview")
@_handle_errors
async def preview_template_asset(
    batch_id: str,
    filename: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    resolved = batches_service.resolve_template_asset(batch_id, filename, settings)
    return FileResponse(resolved, filename=resolved.name)


def _make_json_endpoints(doc_name: str) -> None:
    """Attach GET/PUT routes for each optional JSON document."""

    @router.get(f"/batches/{{batch_id}}/{doc_name}", name=f"get_{doc_name}")
    @_handle_errors
    async def get_doc(
        batch_id: str, settings: Settings = Depends(get_settings)
    ) -> dict[str, Any] | None:
        return batches_service.get_json_document(batch_id, doc_name, settings)

    @router.put(f"/batches/{{batch_id}}/{doc_name}", name=f"put_{doc_name}")
    @_handle_errors
    async def put_doc(
        batch_id: str,
        content: dict[str, Any] | None = Body(
            default=None,
            description=f"Full JSON body for {doc_name}.json (null to delete)",
        ),
        settings: Settings = Depends(get_settings),
    ) -> dict[str, str]:
        batches_service.save_json_document(batch_id, doc_name, content, settings)
        return {"status": "saved" if content is not None else "deleted"}


for _doc in ("template", "config", "evaluation"):
    _make_json_endpoints(_doc)


def _assert_batch_ready_to_run(batch: Batch, settings: Settings) -> None:
    """Fail fast with a clear message if anything would block a run."""
    if batch.file_count == 0:
        raise HTTPException(status_code=400, detail="Batch has no input images.")
    if not batch.has_template:
        if _maybe_attach_prefilled_25q_defaults(
            batch.id,
            settings,
            reason="process requested for prefilled_25q batch without template",
        ):
            # The process guard is called with a Batch snapshot that was
            # loaded before this repair, so do not inspect batch.has_template
            # again here. The following missing-asset check reads from disk.
            pass
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Batch is missing template.json. If this batch came from "
                    "the Prefill page, upload the original file whose name "
                    "starts with 'prefilled_sheet' so the built-in 25Q "
                    "template can be attached automatically."
                ),
            )
    missing = batches_service.missing_template_assets(batch.id, settings)
    if missing:
        names = ", ".join(missing)
        raise HTTPException(
            status_code=400,
            detail=(
                f"template.json references missing asset(s): {names}. "
                "Upload them under Template assets before running."
            ),
        )


@router.post(
    "/batches/{batch_id}/process",
    response_model=ProcessAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
@_handle_errors
async def process_batch(
    batch_id: str,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> ProcessAccepted:
    batch = batches_service.get_batch(batch_id, settings)
    _assert_batch_ready_to_run(batch, settings)
    omr_service.queue_run(batch_id, settings)
    logger.info(
        "Process queued | batch=%s | files=%d",
        batch_id,
        batch.file_count,
    )
    background_tasks.add_task(omr_service.run_batch_sync, batch_id, settings)
    return ProcessAccepted(batch_id=batch_id, status=BatchStatus.queued)


@router.post(
    "/batches/{batch_id}/cancel",
    response_model=ProcessAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
@_handle_errors
async def cancel_batch(
    batch_id: str,
    settings: Settings = Depends(get_settings),
) -> ProcessAccepted:
    next_status = omr_service.request_cancel(batch_id, settings)
    return ProcessAccepted(batch_id=batch_id, status=next_status)


@router.post(
    "/batches/{batch_id}/restart",
    response_model=ProcessAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
@_handle_errors
async def restart_batch(
    batch_id: str,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> ProcessAccepted:
    batch = batches_service.get_batch(batch_id, settings)
    if batch.status in {BatchStatus.queued, BatchStatus.running}:
        raise HTTPException(
            status_code=409,
            detail="Stop the current run before restarting this batch.",
        )
    _assert_batch_ready_to_run(batch, settings)

    batches_service.reset_batch_runtime_state(batch_id, settings)
    omr_service.queue_run(batch_id, settings)
    background_tasks.add_task(omr_service.run_batch_sync, batch_id, settings)
    return ProcessAccepted(batch_id=batch_id, status=BatchStatus.queued)


@router.get(
    "/batches/{batch_id}/status",
    response_model=BatchStatusResponse,
)
@_handle_errors
async def batch_status(
    batch_id: str, settings: Settings = Depends(get_settings)
) -> BatchStatusResponse:
    import time as _time
    batch = batches_service.get_batch(batch_id, settings)
    metadata = batches_service.get_batch_metadata(batch_id, settings)
    processed = int(metadata.get("processed_files", 0))
    total = int(metadata.get("total_files", batch.file_count))
    elapsed_s: float | None = None
    rate_per_min: float | None = None
    eta_s: float | None = None
    run_started_at = metadata.get("run_started_at")
    run_elapsed = metadata.get("run_elapsed_s")
    if run_elapsed is not None:
        # Prefer the stored final value — avoids the timer ticking on after the
        # batch finishes while the status write is still in-flight.
        elapsed_s = float(run_elapsed)
    elif batch.status.value == "running" and run_started_at is not None:
        # No checkpoint written yet; derive from wall-clock start time.
        elapsed_s = round(_time.time() - float(run_started_at), 1)
    if elapsed_s and elapsed_s > 0 and processed > 0:
        rate_per_min = round(processed / elapsed_s * 60, 1)
        if batch.status.value == "running":
            remaining = total - processed
            if rate_per_min > 0:
                eta_s = round(remaining / (processed / elapsed_s))
    return BatchStatusResponse(
        id=batch.id,
        status=batch.status,
        last_error=batch.last_error,
        file_count=batch.file_count,
        updated_at=batch.updated_at,
        processed_files=processed,
        total_files=total,
        latest_processed_file=metadata.get("latest_processed_file"),
        latest_dynamic_dimensions=metadata.get("latest_dynamic_dimensions"),
        cancel_requested=bool(metadata.get("cancel_requested", False)),
        preprocess_failures=list(metadata.get("preprocess_failures", [])),
        elapsed_s=elapsed_s,
        rate_per_min=rate_per_min,
        eta_s=eta_s,
        pdf_split_pages=int(metadata.get("pdf_split_pages", 0)),
        pdf_split_total=int(metadata.get("pdf_split_total", 0)),
        pdf_split_error=metadata.get("pdf_split_error") or None,
        pipelined_run=bool(metadata.get("pipelined_run", False)),
        # Mirror runtime-mutable auto-start settings so the frontend
        # poller can decide whether to auto-fire /process without
        # making a separate /api/v1/settings request per tick.
        auto_start_omr_with_split=settings.auto_start_omr_with_split,
        auto_start_omr_min_pages=settings.auto_start_omr_min_pages,
        auto_start_omr_require_config=settings.auto_start_omr_require_config,
        has_template=batch.has_template,
        has_config=batch.has_config,
    )


@router.get("/batches/{batch_id}/results", response_model=ResultsPayload)
@_handle_errors
async def batch_results(
    batch_id: str, settings: Settings = Depends(get_settings)
) -> ResultsPayload:
    batches_service.get_batch(batch_id, settings)
    return omr_service.read_results(batch_id, settings)


@router.get("/batches/{batch_id}/results/download")
@_handle_errors
async def download_results(
    batch_id: str, settings: Settings = Depends(get_settings)
) -> FileResponse:
    batches_service.get_batch(batch_id, settings)
    path = omr_service.results_csv_path(batch_id, settings)
    if path is None:
        raise HTTPException(
            status_code=404, detail="No results CSV yet. Run the batch first."
        )
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"{batch_id}_{path.name}",
    )


@router.get("/batches/{batch_id}/outputs/{file_path:path}")
@_handle_errors
async def download_output_file(
    batch_id: str,
    file_path: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    resolved = batches_service.resolve_output_file(batch_id, file_path, settings)
    return FileResponse(resolved, filename=resolved.name)


@router.get("/batches/{batch_id}/results/{filename}/checked")
@_handle_errors
async def get_checked_output_image(
    batch_id: str,
    filename: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Serve the OMR-annotated output image for a given input filename.

    Searches CheckedOMRs first, then MultiMarkedFiles, then ErrorFiles so a
    single URL works regardless of which output subdirectory the engine used.
    """
    safe_name = Path(filename).name  # strip any path components
    if not safe_name or safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    batch_root = batches_service.get_batch_root(batch_id, settings)
    outputs_dir = batch_root / "outputs"
    for subdir in ("CheckedOMRs", "Manual/MultiMarkedFiles", "Manual/ErrorFiles"):
        candidate = (outputs_dir / subdir / safe_name).resolve()
        try:
            candidate.relative_to(outputs_dir.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return FileResponse(candidate, filename=safe_name)
    raise HTTPException(status_code=404, detail=f"No checked output image found for {filename!r}.")


# ---------------------------------------------------------------------------
# Prefill endpoints
# ---------------------------------------------------------------------------

@router.get("/prefill/marking-profiles")
async def prefill_marking_profiles() -> dict[str, Any]:
    """Return the available student marking profiles (id, label, description).

    Lets the UI dynamically populate the marking-profile dropdown so any
    server-side additions show up without a frontend redeploy.
    """
    return {"profiles": list_marking_profiles()}


@router.get("/prefill/sample")
async def prefill_sample(
    preset: str = "none",
    candidate_number: str = "9010690012",
    student_name: str = "Jane Doe",
    school_name: str = "Sample School",
    exam_name: str = "Sample Exam",
    marking_profile: str = "none",
    answers: str | None = None,
) -> StreamingResponse:
    """Return an inline PNG preview of a single preset.

    Used by the in-page "Compare presets" gallery so users can see exactly
    what each realism preset produces without downloading anything. Response
    is marked ``Cache-Control: no-store`` so WebView2 / browser caches cannot
    serve a stale version after the simulator code changes.

    ``marking_profile`` and ``answers`` enable previewing the new student-fill
    feature: the sample gallery can show the same answer key drawn with each
    marking profile to make profile selection visual.
    """
    try:
        preset_norm = normalize_realism_preset(preset)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        marking_profile_norm = normalize_marking_profile(marking_profile)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not _PREFILL_SAMPLE_SEM.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Sample renderer busy; try again shortly "
                f"(max {_PREFILL_SAMPLE_LIMIT} concurrent previews)."
            ),
        )
    try:
        try:
            data = await asyncio.to_thread(
                prefill_service.generate_single_png,
                student_name, school_name, exam_name, candidate_number, preset_norm,
                marking_profile_norm, answers,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"Sample render failed: {type(exc).__name__}: {exc}",
            )
    finally:
        _PREFILL_SAMPLE_SEM.release()
    profile_suffix = "" if marking_profile_norm == "none" else f"_{marking_profile_norm}"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="preview_{preset_norm}{profile_suffix}.png"',
            "Cache-Control": "no-store, max-age=0",
        },
    )


@router.post("/generate-csv")
async def generate_test_csv(
    count: int = Form(...),
    school_name: str = Form(...),
    exam_name: str = Form(...),
    candidate_start: str = Form(...),
    name_style: str = Form("numbered"),
    include_output_file: bool = Form(False),
) -> dict:
    """Generate a student-record test CSV as a server-backed download token."""
    settings = get_settings()
    row_cap = settings.prefill_zip_max_rows
    school_name = school_name.strip()
    exam_name = exam_name.strip()
    candidate_start = candidate_start.strip()
    name_style = (name_style or "numbered").strip().lower()

    if count < 1 or count > row_cap:
        raise HTTPException(
            status_code=422,
            detail=f"Number of students must be between 1 and {row_cap:,}.",
        )
    if not school_name:
        raise HTTPException(status_code=422, detail="School name is required.")
    if not exam_name:
        raise HTTPException(status_code=422, detail="Exam name is required.")
    if name_style not in {"numbered", "random"}:
        raise HTTPException(status_code=422, detail="name_style must be 'numbered' or 'random'.")
    if not (candidate_start.isdigit() and len(candidate_start) == 10):
        raise HTTPException(status_code=422, detail="Candidate number start must be exactly 10 digits.")

    last_candidate = int(candidate_start) + count - 1
    if last_candidate > 9_999_999_999:
        raise HTTPException(
            status_code=422,
            detail=(
                "Candidate numbers would exceed 10 digits. Lower the row count "
                "or use a smaller Candidate Number Start."
            ),
        )

    suffix = "random" if name_style == "random" else "numbered"
    filename = f"test_students_{count}_{suffix}_{int(_time_mod.time())}.csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp_path = Path(tmp.name)
    tmp.close()

    try:
        meta = test_csv_service.write_test_csv(
            dst_path=tmp_path,
            count=count,
            school_name=school_name,
            exam_name=exam_name,
            candidate_start=candidate_start,
            name_style=name_style,
            include_output_file=include_output_file,
        )
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    token = _register_download(tmp_path, "text/csv; charset=utf-8", filename)
    return {
        "download_url": f"/api/v1/generate-csv/download/{token}",
        "filename": filename,
        "count": meta["count"],
        "size_bytes": meta["size_bytes"],
        "include_output_file": include_output_file,
        "pdf_max_rows": settings.prefill_pdf_max_rows,
        "zip_max_rows": settings.prefill_zip_max_rows,
    }


@router.get("/generate-csv/download/{token}")
async def generate_test_csv_download(token: str, background_tasks: BackgroundTasks):
    """One-time download endpoint for generated test CSV files."""
    return _download_entry_response(token, background_tasks)


@router.post("/prefill/single")
async def prefill_single(
    student_name: str = Form(...),
    school_name: str = Form(...),
    exam_name: str = Form(...),
    candidate_number: str = Form(...),
    output_format: str = Form("png"),
    realism_preset: str = Form("none"),
    marking_profile: str = Form("none"),
    answers: str | None = Form(None),
    subject_name: str = Form(""),
) -> StreamingResponse:
    """Generate a single pre-filled answer sheet and stream it as a download.

    The optional ``marking_profile`` and ``answers`` parameters drive the
    new student-fill feature (see :mod:`webui.services.student_fill`). When
    ``marking_profile != "none"`` and ``answers`` is non-empty, the chosen
    answer bubbles are darkened on the sheet using a student-style hand.
    Both fields default to disabled so existing callers see no behaviour
    change.
    """
    output_format = (output_format or "").strip().lower()
    if output_format not in {"png", "pdf"}:
        raise HTTPException(
            status_code=422,
            detail="output_format must be 'png' or 'pdf'.",
        )
    try:
        realism_preset = normalize_realism_preset(realism_preset)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        marking_profile_norm = normalize_marking_profile(marking_profile)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    # Backpressure: bounded concurrency so a flood of requests cannot exhaust
    # the threadpool / RAM. Excess requests get a fast 429 with Retry-After.
    if not _PREFILL_SINGLE_SEM.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Server busy: max {_PREFILL_SINGLE_LIMIT} concurrent single "
                "prefill requests in flight. Please retry shortly."
            ),
            headers={"Retry-After": "2"},
        )
    try:
        # Offload CPU-heavy rendering off the event loop so a flood of
        # /prefill/single requests cannot block other endpoints (e.g. health).
        # Suffix the filename with the preset (when not "none") so users can
        # immediately tell which realism preset produced a given download.
        preset_suffix = "" if realism_preset == "none" else f"_{realism_preset}"
        profile_suffix = "" if marking_profile_norm == "none" else f"_{marking_profile_norm}"
        if output_format == "pdf":
            data = await asyncio.to_thread(
                prefill_service.generate_single_pdf,
                student_name, school_name, exam_name, candidate_number, realism_preset,
                marking_profile_norm, answers, subject_name,
            )
            media_type = "application/pdf"
            filename = f"prefilled_sheet{preset_suffix}{profile_suffix}.pdf"
        else:
            data = await asyncio.to_thread(
                prefill_service.generate_single_png,
                student_name, school_name, exam_name, candidate_number, realism_preset,
                marking_profile_norm, answers, subject_name,
            )
            media_type = "image/png"
            filename = f"prefilled_sheet{preset_suffix}{profile_suffix}.png"
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - never leak stack traces
        raise HTTPException(
            status_code=500,
            detail=f"Single prefill failed: {type(exc).__name__}: {exc}",
        )
    finally:
        _PREFILL_SINGLE_SEM.release()

    return StreamingResponse(
        io.BytesIO(data),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/prefill/batch")
async def prefill_batch(
    background_tasks: BackgroundTasks,
    request: Request,
) -> dict:
    """Generate pre-filled answer sheets for multiple students.

    Streams generation to a server-side temp file then returns a one-time
    download token as JSON. The client uses window.location.href on the
    download URL so large files stream directly to disk (no browser buffering).
    """
    # Parse the form manually instead of using FastAPI's automatic
    # Form/File parameters. Automatic parsing uses Starlette's 1 MiB default
    # max_part_size and rejects large uploaded CSVs before endpoint code can
    # run. Manual parsing lets us apply the operator-controlled upload cap.
    settings = get_settings()
    try:
        form = await request.form(
            max_part_size=settings.max_upload_bytes,
            max_files=10_000,
            max_fields=10_000,
        )
    except Exception as exc:  # noqa: BLE001 - normalize parser failures
        raise HTTPException(status_code=400, detail=str(exc))

    csv_text_value = form.get("csv_text")
    csv_text = csv_text_value if isinstance(csv_text_value, str) else None
    csv_file_value = form.get("csv_file")
    csv_file = csv_file_value if hasattr(csv_file_value, "read") else None
    output_mode = str(form.get("output_mode") or "pdf")
    realism_preset = str(form.get("realism_preset") or "none")
    marking_profile = str(form.get("marking_profile") or "none")
    group_by_raw = form.get("group_by")
    group_by_value = str(group_by_raw) if group_by_raw is not None else "none"
    answers_default_value = form.get("answers")
    answers_default = (
        answers_default_value if isinstance(answers_default_value, str) and answers_default_value.strip() else None
    )
    include_page_numbers_value = form.get("include_page_numbers")
    include_page_numbers = str(include_page_numbers_value).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Optional "split combined PDF into smaller printer-safe segments"
    # toggle (off by default). The size and page caps come from
    # operator-tunable runtime settings rather than form fields so the
    # operator can validate them against their printer fleet once on
    # the /settings page instead of asking every user to re-discover
    # safe values for every batch.
    split_pdfs_value = form.get("split_pdfs")
    split_pdfs = str(split_pdfs_value).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # 1) Validate output_mode early — reject unknown values explicitly.
    output_mode = (output_mode or "").strip().lower()
    if output_mode not in {"pdf", "zip"}:
        raise HTTPException(
            status_code=422,
            detail="output_mode must be 'pdf' or 'zip'.",
        )
    try:
        realism_preset = normalize_realism_preset(realism_preset)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        marking_profile = normalize_marking_profile(marking_profile)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        group_by = prefill_service.normalize_group_by(group_by_value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if not csv_text and (csv_file is None or not getattr(csv_file, "filename", "")):
        raise HTTPException(
            status_code=422,
            detail="Provide either csv_text or a csv_file.",
        )

    # Read caps from the live Settings instance so operators can mutate them
    # via /settings without restarting the server.
    pdf_max_rows = settings.prefill_pdf_max_rows
    zip_max_rows = settings.prefill_zip_max_rows
    max_bytes = settings.prefill_csv_max_bytes

    # 2) Bound the CSV body size BEFORE materialising it. For uploads we read
    # in chunks so a hostile client can't blow up RAM by sending a multi-GB file.
    if csv_text and csv_text.strip():
        encoded = csv_text.strip().encode("utf-8")
        if len(encoded) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"CSV text exceeds the {max_bytes // (1024*1024)} MiB limit.",
            )
        raw = encoded.decode("utf-8-sig")
    else:
        # csv_text was empty/blank, so csv_file must be present.
        # Audit fix API-3: replace assert (stripped under python -O) with
        # an explicit HTTPException so this branch is robust in production.
        if csv_file is None:
            raise HTTPException(
                status_code=400,
                detail="No CSV provided: pass either csv_text or csv_file.",
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await csv_file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File {csv_file.filename!r} exceeds the "
                           f"{max_bytes // (1024*1024)} MiB limit.",
                )
            chunks.append(chunk)
        try:
            raw = b"".join(chunks).decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"CSV must be UTF-8 encoded: {exc}",
            )

    # 3) Parse CSV defensively. csv.DictReader raises for some malformed inputs.
    try:
        reader = csv.DictReader(io.StringIO(raw))
        rows = [row for row in reader if row]
    except csv.Error as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse CSV: {exc}")
    except Exception as exc:  # noqa: BLE001 - normalise to 422
        raise HTTPException(status_code=422, detail=f"Failed to parse CSV: {exc}")

    if not rows:
        raise HTTPException(status_code=422, detail="CSV contains no data rows.")

    # 3b) Normalise header aliases (e.g. center_name/centre_name -> school_name)
    # so operator exports with varied column spellings work unchanged.
    rows = [prefill_service.normalize_row_keys(row) for row in rows]

    # 4) Required column check happens once on the first row. ``exam_name`` and
    # ``subject_name`` are optional (their write-in lines are left blank when
    # absent); the Centre field may arrive as center_name/centre_name and is
    # normalised to ``school_name`` above.
    required_cols = set(prefill_service.REQUIRED_CSV_COLUMNS)
    missing_cols = required_cols - set(rows[0].keys())
    if missing_cols:
        friendly = sorted(missing_cols)
        hint = ""
        if "school_name" in missing_cols:
            hint = (
                " (the Centre column may be named "
                "'school_name', 'center_name', or 'centre_name')"
            )
        raise HTTPException(
            status_code=422,
            detail=(
                "CSV is missing required columns: "
                f"{', '.join(friendly)}{hint}"
            ),
        )


    # 4b) When grouping by region the CSV must carry a ``region`` column.
    # We do not enforce non-empty values (rows with empty region fall into
    # an explicit ``_unknown_region`` bucket) but the column itself must
    # exist so the user knows their CSV missed it.
    if group_by in {"region", "region_school"} and "region" not in rows[0]:
        raise HTTPException(
            status_code=422,
            detail=(
                "group_by="
                + group_by
                + " requires a 'region' column in the CSV "
                "(empty cells are allowed and bucket into '_unknown_region')."
            ),
        )

    # 5) Row-count cap so a runaway batch can't dominate the server.
    # Grouped output is always packaged as a ZIP regardless of ``output_mode``
    # (the ZIP contains one PDF per group), so use the ZIP cap there.
    effective_mode_for_cap = "zip" if group_by != "none" else output_mode
    row_cap = pdf_max_rows if effective_mode_for_cap == "pdf" else zip_max_rows
    if len(rows) > row_cap:
        raise HTTPException(
            status_code=422,
            detail=(
                f"CSV has {len(rows)} rows but the per-batch limit for "
                f"{effective_mode_for_cap.upper()} output is {row_cap}. Split the file into "
                "smaller batches or raise the cap on the /settings page."
            ),
        )

    # 6) Backpressure heavy jobs server-wide.
    if not _PREFILL_BATCH_SEM.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Server is already running {_PREFILL_BATCH_LIMIT} prefill batch "
                "job(s). Try again shortly."
            ),
            headers={"Retry-After": "10"},
        )

    # Grouped output is always a ZIP of per-group PDFs.
    # The split-PDFs toggle is meaningful only when the underlying
    # output is at least one PDF; for a flat ZIP of single-page PNGs
    # (output_mode=zip, no grouping) splitting has nothing to operate
    # on and is silently dropped, mirroring the existing behaviour
    # for ``include_page_numbers`` on flat PNG ZIPs.
    split_enabled = split_pdfs and (group_by != "none" or output_mode == "pdf")
    if split_enabled:
        # Splitting always packages output as a ZIP (one or more PDF
        # segments inside) so the one-file-per-token download contract
        # still holds and the user sees a single consistent download.
        effective_mode = "zip"
    else:
        effective_mode = "zip" if group_by != "none" else output_mode
    suffix = ".pdf" if effective_mode == "pdf" else ".zip"
    media_type = "application/pdf" if effective_mode == "pdf" else "application/zip"
    preset_suffix = "" if realism_preset == "none" else f"_{realism_preset}"
    group_suffix = "" if group_by == "none" else f"_by_{group_by}"
    split_suffix = "_split" if split_enabled else ""
    filename = f"prefilled_sheets{preset_suffix}{group_suffix}{split_suffix}{suffix}"

    # 7) Write to a temp file the response will stream from. The file is
    # deleted after the response finishes via background_tasks.
    fd, tmp_path_str = tempfile.mkstemp(prefix="prefill_", suffix=suffix)
    os.close(fd)
    tmp_path = Path(tmp_path_str)

    def _cleanup() -> None:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        _PREFILL_BATCH_SEM.release()

    def _release_sem() -> None:
        _PREFILL_BATCH_SEM.release()

    # Snapshot split caps from settings so the values used during
    # rendering are pinned to the moment the request was accepted, not
    # whatever the operator might change mid-job on /settings.
    split_max_mb = settings.prefill_split_max_pdf_mb
    split_max_pages = settings.prefill_split_max_pdf_pages

    try:
        # Offload to a thread so a long-running batch cannot block the event
        # loop and stall every other request (incl. health checks).
        if split_enabled and group_by != "none":
            # Grouped output with each group's PDF post-split into bounded
            # segments. Each group's pages remain numbered continuously
            # across that group's segments because the per-group PDF is
            # rendered once before being split.
            meta = await asyncio.to_thread(
                _run_batch_grouped_split_zip,
                rows,
                tmp_path,
                realism_preset,
                group_by,
                split_max_mb,
                split_max_pages,
                include_page_numbers,
                marking_profile,
                answers_default,
            )
        elif split_enabled:
            # Flat combined PDF post-split into bounded segments, packaged
            # as a ZIP containing one segment per ``..._part_NN_of_TT.pdf``
            # entry. Numbering is continuous across the segments.
            meta = await asyncio.to_thread(
                _run_batch_split_pdf,
                rows,
                tmp_path,
                realism_preset,
                split_max_mb,
                split_max_pages,
                include_page_numbers,
                marking_profile,
                answers_default,
            )
        elif group_by != "none":
            # Grouped output: one PDF per school/region inside a ZIP. The
            # ``output_mode`` field is intentionally ignored here — when the
            # user asks for grouping they always get a ZIP-of-PDFs, since
            # a single PDF can't represent multiple group documents.
            meta = await asyncio.to_thread(
                _run_batch_grouped_zip,
                rows,
                tmp_path,
                realism_preset,
                group_by,
                include_page_numbers,
                marking_profile,
                answers_default,
            )
        elif output_mode == "zip":
            # Page numbers are a multi-page PDF feature only — silently
            # ignore the flag when the user picked a ZIP of single PNGs
            # so the JS UI doesn't have to enforce it client-side.
            meta = await asyncio.to_thread(
                _run_batch_zip,
                rows,
                tmp_path,
                realism_preset,
                marking_profile,
                answers_default,
            )
        else:
            meta = await asyncio.to_thread(
                _run_batch_pdf,
                rows,
                tmp_path,
                realism_preset,
                include_page_numbers,
                marking_profile,
                answers_default,
            )
    except ValueError as exc:
        _cleanup()
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - never leak stack traces over HTTP
        _cleanup()
        raise HTTPException(
            status_code=500,
            detail=f"Prefill batch generation failed: {type(exc).__name__}: {exc}",
        )

    # If literally every row failed, surface that as a 422 instead of returning
    # an empty PDF/zip the user has to inspect to discover.
    if meta["successes"] == 0:
        _cleanup()
        error_preview = "; ".join(str(err) for err in meta.get("errors", [])[:10])
        if len(meta.get("errors", [])) > 10:
            error_preview += f"; ...and {len(meta['errors']) - 10} more error(s)"
        raise HTTPException(
            status_code=422,
            detail="All rows failed to generate."
            + (f" Errors: {error_preview}" if error_preview else ""),
        )

    # Release the semaphore now — generation is done, file is held for download.
    _release_sem()
    # Register the file for a one-time token-based GET download instead of
    # streaming directly. This allows the frontend to use window.location.href
    # which bypasses browser PDF viewer buffering for large files.
    token = _register_download(tmp_path, media_type, filename)
    return {
        "download_url": f"/api/v1/prefill/batch/download/{token}",
        "filename": filename,
        "count": meta["count"],
        "successes": meta["successes"],
        "errors": meta["errors"],
        "elapsed_s": meta["elapsed_s"],
        "size_bytes": meta["size_bytes"],
        # Echo whether page numbers were applied. Page numbers DO apply
        # inside each per-group PDF when grouping is on; they're only
        # dropped for a flat ZIP of single PNGs.
        "page_numbers": include_page_numbers and (
            group_by != "none" or output_mode == "pdf"
        ),
        "group_by": group_by,
        "groups": meta.get("groups", []) if group_by != "none" else [],
        # Surface split metadata so the JS UI can show "Split into N
        # segments of <= M MiB each" instead of leaving the user to
        # discover that from a stranger-looking ZIP. When splitting
        # was not active, ``segments`` is an empty list and the cap
        # echoes the operator setting for transparency.
        "split_pdfs": split_enabled,
        "split_max_pdf_mb": split_max_mb if split_enabled else None,
        "split_max_pdf_pages": split_max_pages if split_enabled else None,
        "segments": meta.get("segments", []) if split_enabled and group_by == "none" else [],
    }


@router.get("/prefill/batch/download/{token}")
async def prefill_batch_download(token: str, background_tasks: BackgroundTasks):
    """One-time token download endpoint. Returns the generated file and deletes it."""
    return _download_entry_response(token, background_tasks)


# ---------------------------------------------------------------------------
# Print-N-blank-sheets endpoint.
# ---------------------------------------------------------------------------


def _run_blank_sheets_pdf(
    dst_path: Path,
    variant: str,
    count: int,
    include_page_numbers: bool,
) -> dict:
    """Adapter that calls ``generate_blank_sheets_pdf_to_file`` with introspection."""
    import inspect

    target = prefill_service.generate_blank_sheets_pdf_to_file
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs: dict[str, Any] = {"variant": variant, "count": count}
    if "include_page_numbers" in params:
        kwargs["include_page_numbers"] = include_page_numbers
    return target(dst_path, **kwargs)


def _run_blank_sheets_split_zip(
    dst_path: Path,
    variant: str,
    count: int,
    max_pdf_mb: int,
    max_pdf_pages: int,
    include_page_numbers: bool,
) -> dict:
    """Adapter that calls ``generate_blank_sheets_split_zip_to_file``."""
    import inspect

    target = prefill_service.generate_blank_sheets_split_zip_to_file
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs: dict[str, Any] = {
        "variant": variant,
        "count": count,
        "max_pdf_mb": max_pdf_mb,
        "max_pdf_pages": max_pdf_pages,
    }
    if "include_page_numbers" in params:
        kwargs["include_page_numbers"] = include_page_numbers
    return target(dst_path, **kwargs)


@router.get("/prefill/blank/variants")
async def prefill_blank_variants() -> dict:
    """Return the registered blank-sheet variants for the UI dropdown."""
    return {
        "variants": prefill_service.list_blank_sheet_variants(),
        "default": prefill_service.DEFAULT_BLANK_SHEET_VARIANT,
    }


@router.post("/prefill/blank")
async def prefill_blank(
    background_tasks: BackgroundTasks,
    request: Request,
) -> dict:
    """Generate N copies of a blank answer-sheet variant as a printable PDF.

    Designed for the common "I need 500 unfilled sheets to hand out at
    the venue" workflow that has nothing to do with candidate prefill.
    Sources the page from a 1-page asset bundled with the application
    (default: July 2026 Letter landscape SMQ60; April 2026 NNQ25 also
    available) and clones it N times into a single PDF. When
    ``split_pdfs=true`` the combined PDF is post-split into
    printer-safe ZIP segments using the same size/page caps as the
    batch endpoint, with page numbering continuous across segments.
    """
    settings = get_settings()
    try:
        form = await request.form(
            max_part_size=settings.max_upload_bytes,
            max_files=10,
            max_fields=100,
        )
    except Exception as exc:  # noqa: BLE001 - normalise parser failures
        raise HTTPException(status_code=400, detail=str(exc))

    variant_raw = form.get("variant")
    count_raw = form.get("count")
    include_page_numbers = str(form.get("include_page_numbers") or "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    split_pdfs = str(form.get("split_pdfs") or "").strip().lower() in {
        "1", "true", "yes", "on",
    }

    try:
        variant = prefill_service.normalize_blank_sheet_variant(variant_raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        count = int(str(count_raw).strip()) if count_raw is not None else 0
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="count must be a positive integer.",
        )
    if count < 1:
        raise HTTPException(
            status_code=422,
            detail="count must be >= 1.",
        )

    blank_cap = settings.prefill_blank_max_sheets
    if count > blank_cap:
        raise HTTPException(
            status_code=422,
            detail=(
                f"count={count} exceeds the per-request cap of {blank_cap}. "
                "Lower the count or raise prefill_blank_max_sheets on the "
                "/settings page."
            ),
        )

    if not _PREFILL_BATCH_SEM.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Server is already running {_PREFILL_BATCH_LIMIT} prefill batch "
                "job(s). Try again shortly."
            ),
            headers={"Retry-After": "10"},
        )

    suffix = ".zip" if split_pdfs else ".pdf"
    media_type = "application/zip" if split_pdfs else "application/pdf"
    stem = prefill_service.BLANK_SHEET_VARIANTS[variant].get(
        "default_stem", f"blank_{variant}"
    )
    filename = f"{stem}_x{count}{'_split' if split_pdfs else ''}{suffix}"

    fd, tmp_path_str = tempfile.mkstemp(prefix="blank_", suffix=suffix)
    os.close(fd)
    tmp_path = Path(tmp_path_str)

    def _cleanup() -> None:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        _PREFILL_BATCH_SEM.release()

    def _release_sem() -> None:
        _PREFILL_BATCH_SEM.release()

    split_max_mb = settings.prefill_split_max_pdf_mb
    split_max_pages = settings.prefill_split_max_pdf_pages

    try:
        if split_pdfs:
            meta = await asyncio.to_thread(
                _run_blank_sheets_split_zip,
                tmp_path,
                variant,
                count,
                split_max_mb,
                split_max_pages,
                include_page_numbers,
            )
        else:
            meta = await asyncio.to_thread(
                _run_blank_sheets_pdf,
                tmp_path,
                variant,
                count,
                include_page_numbers,
            )
    except (ValueError, FileNotFoundError) as exc:
        _cleanup()
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        _cleanup()
        raise HTTPException(
            status_code=500,
            detail=f"Blank sheet generation failed: {type(exc).__name__}: {exc}",
        )

    if meta["successes"] == 0:
        _cleanup()
        raise HTTPException(
            status_code=500,
            detail="Blank sheet generation produced no pages.",
        )

    _release_sem()
    token = _register_download(tmp_path, media_type, filename)
    return {
        "download_url": f"/api/v1/prefill/batch/download/{token}",
        "filename": filename,
        "variant": variant,
        "count": meta["count"],
        "successes": meta["successes"],
        "errors": meta["errors"],
        "elapsed_s": meta["elapsed_s"],
        "size_bytes": meta["size_bytes"],
        "page_numbers": include_page_numbers,
        "split_pdfs": split_pdfs,
        "split_max_pdf_mb": split_max_mb if split_pdfs else None,
        "split_max_pdf_pages": split_max_pages if split_pdfs else None,
        "segments": meta.get("segments", []) if split_pdfs else [],
    }
