"""JSON API router for the OMRChecker Web UI.

Exposed under ``/api/v1``. All mutations live here so that the HTML UI
and any third-party API consumer go through identical codepaths.
"""

from __future__ import annotations

from typing import Any

import csv
import io

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
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
from webui.services.batches import BatchNotFound, InvalidBatchRequest
from webui.settings import Settings, get_settings

router = APIRouter(prefix="/api/v1", tags=["omr"])


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


@router.post(
    "/batches/{batch_id}/files",
    response_model=list[FileRef],
    status_code=status.HTTP_201_CREATED,
)
@_handle_errors
async def upload_files(
    batch_id: str,
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
) -> list[FileRef]:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    stored: list[FileRef] = []
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
        stored.extend(
            batches_service.save_uploaded_file(
                batch_id,
                upload.filename or "upload",
                data,
                settings,
            )
        )
    return stored


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
        raise HTTPException(
            status_code=400,
            detail="Batch is missing template.json; upload one before processing.",
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
    batch = batches_service.get_batch(batch_id, settings)
    metadata = batches_service.get_batch_metadata(batch_id, settings)
    return BatchStatusResponse(
        id=batch.id,
        status=batch.status,
        last_error=batch.last_error,
        file_count=batch.file_count,
        updated_at=batch.updated_at,
        processed_files=metadata.get("processed_files", 0),
        total_files=metadata.get("total_files", batch.file_count),
        latest_processed_file=metadata.get("latest_processed_file"),
        latest_dynamic_dimensions=metadata.get("latest_dynamic_dimensions"),
        cancel_requested=bool(metadata.get("cancel_requested", False)),
        preprocess_failures=list(metadata.get("preprocess_failures", [])),
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


# ---------------------------------------------------------------------------
# Prefill endpoints
# ---------------------------------------------------------------------------

@router.post("/prefill/single")
async def prefill_single(
    student_name: str = Form(...),
    school_name: str = Form(...),
    exam_name: str = Form(...),
    candidate_number: str = Form(...),
    output_format: str = Form("png"),
) -> StreamingResponse:
    """Generate a single pre-filled answer sheet and stream it as a download."""
    try:
        if output_format == "pdf":
            data = prefill_service.generate_single_pdf(
                student_name, school_name, exam_name, candidate_number
            )
            media_type = "application/pdf"
            filename = "prefilled_sheet.pdf"
        else:
            data = prefill_service.generate_single_png(
                student_name, school_name, exam_name, candidate_number
            )
            media_type = "image/png"
            filename = "prefilled_sheet.png"
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return StreamingResponse(
        io.BytesIO(data),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/prefill/batch")
async def prefill_batch(
    csv_text: str | None = Form(default=None),
    csv_file: UploadFile | None = File(default=None),
    output_mode: str = Form("pdf"),
) -> StreamingResponse:
    """Generate pre-filled answer sheets for multiple students and stream as PDF or ZIP."""
    if not csv_text and (csv_file is None or not csv_file.filename):
        raise HTTPException(
            status_code=422,
            detail="Provide either csv_text or a csv_file.",
        )

    _CSV_MAX =50 * 1024 * 1024  # 10 MiB — covers ~100 000 student rows

    if csv_text and csv_text.strip():
        raw = csv_text.strip()
        if len(raw.encode()) > _CSV_MAX:
            raise HTTPException(status_code=413, detail="CSV text exceeds 1 MiB limit.")
    else:
        file_bytes = await csv_file.read()
        if len(file_bytes) > _CSV_MAX:
            raise HTTPException(status_code=413, detail=f"File {csv_file.filename!r} exceeds 1 MiB limit.")
        raw = file_bytes.decode("utf-8-sig")

    try:
        reader = csv.DictReader(io.StringIO(raw))
        rows = [row for row in reader]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse CSV: {exc}")

    if not rows:
        raise HTTPException(status_code=422, detail="CSV contains no data rows.")

    required_cols = {"student_name", "school_name", "exam_name", "candidate_number"}
    missing_cols = required_cols - set(rows[0].keys())
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"CSV is missing required columns: {', '.join(sorted(missing_cols))}",
        )

    try:
        if output_mode == "zip":
            data = prefill_service.generate_batch_zip(rows)
            media_type = "application/zip"
            filename = "prefilled_sheets.zip"
        else:
            data = prefill_service.generate_batch_pdf(rows)
            media_type = "application/pdf"
            filename = "prefilled_sheets.pdf"
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return StreamingResponse(
        io.BytesIO(data),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
