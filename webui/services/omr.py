"""Wrapper that drives the OMRChecker engine for a given batch.

The engine reads ``template.json``, ``config.json`` and ``evaluation.json``
from the *same* directory as the images. To keep the engine untouched, we
copy any batch-level JSON files into ``inputs/`` just before processing
and tidy them up afterwards.
"""

from __future__ import annotations

import csv
import copy
import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Any

import cv2

from src.defaults.config import CONFIG_DEFAULTS
from webui.schemas import (
    BatchStatus,
    ResultsPayload,
    ResultsRow,
)
from webui.services import batches as batches_service
from webui.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_COPIED_JSON_FILES = ("template.json", "config.json", "evaluation.json")
_RUNTIME_DIR_NAME = "_runtime"
_DISPLAY_KEYS = ("display_height", "display_width", "processing_height", "processing_width")

_batch_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(batch_id: str) -> threading.Lock:
    with _locks_guard:
        lock = _batch_locks.get(batch_id)
        if lock is None:
            lock = threading.Lock()
            _batch_locks[batch_id] = lock
        return lock


def _is_cancel_requested(batch_id: str, settings: Settings) -> bool:
    metadata = batches_service.get_batch_metadata(batch_id, settings)
    return bool(metadata.get("cancel_requested", False))


def _image_ended_up_in_errors(outputs_dir: Path, file_name: str) -> bool:
    """Return True when the OMR engine moved ``file_name`` into the
    ``Manual/ErrorFiles`` directory (which happens when a pre-processor
    like ``CropOnMarkers`` could not locate its markers)."""
    error_dir = outputs_dir / "Manual" / "ErrorFiles"
    return (error_dir / file_name).exists()


def _mark_cancelled(
    batch_id: str,
    reason: str,
    settings: Settings,
) -> None:
    batches_service.update_status(
        batch_id,
        BatchStatus.cancelled,
        last_error=reason,
        settings=settings,
    )


def _discover_input_images(batch_id: str, settings: Settings) -> list[Path]:
    """Return all processable images for a batch."""
    batch_root = batches_service.get_batch_root(batch_id, settings)
    asset_names = {Path(asset).name for asset in _get_template_relative_assets(batch_root)}
    return [
        path
        for path in batches_service.list_input_image_paths(batch_id, settings)
        if path.name not in asset_names
    ]


def _scale_dimensions(
    source_width: int,
    source_height: int,
    max_width: int,
    max_height: int,
) -> tuple[int, int]:
    """Scale dimensions down to fit within a bounding box without upscaling."""
    scale = min(max_width / source_width, max_height / source_height, 1.0)
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    return width, height


def _template_page_dimensions(template_payload: dict[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(template_payload, dict):
        return None
    page_dimensions = template_payload.get("pageDimensions")
    if (
        not isinstance(page_dimensions, list)
        or len(page_dimensions) != 2
        or not all(isinstance(value, int | float) for value in page_dimensions)
    ):
        return None
    width, height = (int(page_dimensions[0]), int(page_dimensions[1]))
    if width <= 0 or height <= 0:
        return None
    return width, height


def _uses_marker_template_space(template_payload: dict[str, Any] | None) -> bool:
    if not isinstance(template_payload, dict):
        return False
    preprocessors = template_payload.get("preProcessors")
    if not isinstance(preprocessors, list):
        return False
    return any(
        isinstance(processor, dict) and processor.get("name") == "CropOnMarkers"
        for processor in preprocessors
    )


def _compute_dynamic_dimensions(
    image_path: Path,
    template_payload: dict[str, Any] | None = None,
    rotation_degrees: int = 0,
) -> dict[str, int]:
    """Compute per-image display and processing dimensions."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path.as_posix()}")

    source_height, source_width = image.shape[:2]
    if rotation_degrees in {90, 270}:
        source_width, source_height = source_height, source_width
    display_width, display_height = _scale_dimensions(
        source_width,
        source_height,
        int(CONFIG_DEFAULTS.dimensions.display_width),
        int(CONFIG_DEFAULTS.dimensions.display_height),
    )
    template_dimensions = (
        _template_page_dimensions(template_payload)
        if _uses_marker_template_space(template_payload)
        else None
    )
    if template_dimensions is None:
        processing_width, processing_height = _scale_dimensions(
            source_width,
            source_height,
            int(CONFIG_DEFAULTS.dimensions.processing_width),
            int(CONFIG_DEFAULTS.dimensions.processing_height),
        )
    else:
        processing_width, processing_height = template_dimensions

    return {
        "source_height": int(source_height),
        "source_width": int(source_width),
        "display_height": int(display_height),
        "display_width": int(display_width),
        "processing_height": int(processing_height),
        "processing_width": int(processing_width),
    }


def _merge_dimensions_into_config(
    config: dict | None, dynamic_dimensions: dict[str, int]
) -> dict:
    """Merge computed dimensions into an existing config payload."""
    merged = copy.deepcopy(config) if isinstance(config, dict) else {}
    dimensions = merged.get("dimensions")
    if not isinstance(dimensions, dict):
        dimensions = {}
    for key in _DISPLAY_KEYS:
        dimensions[key] = int(dynamic_dimensions[key])
    merged["dimensions"] = dimensions
    return merged


def _write_runtime_config(config: dict, dst: Path) -> None:
    """Write staged config while forcing non-interactive web execution."""
    staged = copy.deepcopy(config)
    outputs = staged.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
    outputs["show_image_level"] = 0
    staged["outputs"] = outputs
    with dst.open("w", encoding="utf-8") as fh:
        json.dump(staged, fh, indent=2, sort_keys=True)


def _load_template_payload(batch_root: Path) -> dict[str, Any]:
    template_path = batch_root / "template.json"
    if not template_path.exists():
        return {}
    with template_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _collect_relative_paths(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "relativePath" and isinstance(item, str):
                found.append(item)
            else:
                found.extend(_collect_relative_paths(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_relative_paths(item))
    return found


def _get_template_relative_assets(batch_root: Path) -> list[str]:
    """Return relative asset paths referenced by the template."""
    template_payload = _load_template_payload(batch_root)
    return _collect_relative_paths(template_payload.get("preProcessors", []))


def _resolve_batch_asset(batch_root: Path, relative_path: str) -> Path:
    """Resolve a template-referenced relative asset from batch root or inputs."""
    direct = (batch_root / relative_path).resolve()
    if direct.exists():
        return direct

    inputs_candidate = (batch_root / "inputs" / Path(relative_path).name).resolve()
    if inputs_candidate.exists():
        return inputs_candidate

    raise FileNotFoundError(
        f"Missing template asset '{relative_path}'. Place it in the batch root or inputs folder."
    )


def _copy_runtime_assets(batch_root: Path, runtime_root: Path) -> None:
    """Copy any template-referenced relative assets into the runtime folder."""
    for relative_path in _get_template_relative_assets(batch_root):
        src = _resolve_batch_asset(batch_root, relative_path)
        dst = runtime_root / relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _rotate_image_for_runtime(src: Path, dst: Path, rotation_degrees: int) -> None:
    if rotation_degrees == 0:
        shutil.copy2(src, dst)
        return

    rotate_codes = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    rotate_code = rotate_codes.get(rotation_degrees)
    if rotate_code is None:
        raise ValueError(f"Unsupported rotation: {rotation_degrees}")

    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image for rotation: {src.as_posix()}")
    rotated = cv2.rotate(image, rotate_code)
    if not cv2.imwrite(str(dst), rotated):
        raise ValueError(f"Could not write rotated runtime image: {dst.as_posix()}")


def _prepare_runtime_dir(
    batch_root: Path,
    image_path: Path,
    runtime_config: dict,
    index: int,
    rotation_degrees: int = 0,
) -> Path:
    """Build an isolated single-image runtime directory for engine execution."""
    runtime_root = batch_root / _RUNTIME_DIR_NAME / f"{index:04d}_{image_path.stem}"
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    _rotate_image_for_runtime(
        image_path, runtime_root / image_path.name, rotation_degrees
    )
    for name in ("template.json", "evaluation.json"):
        src = batch_root / name
        if src.exists():
            shutil.copy2(src, runtime_root / name)
    _write_runtime_config(runtime_config, runtime_root / "config.json")
    _copy_runtime_assets(batch_root, runtime_root)
    return runtime_root


def _cleanup_runtime_dir(runtime_dir: Path | None) -> None:
    if runtime_dir and runtime_dir.exists():
        shutil.rmtree(runtime_dir, ignore_errors=True)


def _write_non_interactive_config(src: Path, dst: Path) -> None:
    """Copy config while forcing non-interactive processing in web runs.

    The core engine uses ``outputs.show_image_level`` to decide whether to open
    OpenCV windows and block on ``waitKey``. In API/background usage this can
    hang processing, so we always override it to 0 for staged run config.
    """
    with src.open("r", encoding="utf-8") as fh:
        content = json.load(fh)
    outputs = content.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
    outputs["show_image_level"] = 0
    content["outputs"] = outputs
    with dst.open("w", encoding="utf-8") as fh:
        json.dump(content, fh, indent=2, sort_keys=True)

def run_batch_sync(batch_id: str, settings: Settings | None = None) -> None:
    """Run OMR processing for a single batch synchronously.

    Updates the batch status throughout (``running`` → ``done``/``failed``).
    Safe to call from a ``BackgroundTasks`` task or directly from tests.
    """
    from main import entry_point_for_args

    settings = settings or get_settings()
    lock = _lock_for(batch_id)
    if not lock.acquire(blocking=False):
        logger.info("Batch %s is already processing; skipping duplicate run", batch_id)
        return

    runtime_dir: Path | None = None
    try:
        batch_root = batches_service.get_batch_root(batch_id, settings)
        outputs_dir = batch_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        input_images = _discover_input_images(batch_id, settings)

        if _is_cancel_requested(batch_id, settings):
            _mark_cancelled(batch_id, "Cancelled before processing started.", settings)
            return

        if not input_images:
            batches_service.update_status(
                batch_id,
                BatchStatus.failed,
                last_error="No input images present in batch.",
                settings=settings,
            )
            return

        if not (batch_root / "template.json").exists():
            batches_service.update_status(
                batch_id,
                BatchStatus.failed,
                last_error=(
                    "template.json is required before running OMR. "
                    "Upload or PUT one for this batch."
                ),
                settings=settings,
            )
            return

        batches_service.update_status(batch_id, BatchStatus.running, settings=settings)
        batches_service.update_batch_metadata(
            batch_id,
            {
                "processed_files": 0,
                "total_files": len(input_images),
                "latest_processed_file": None,
                "latest_dynamic_dimensions": None,
                "dynamic_dimensions_by_file": {},
                "preprocess_failures": [],
            },
            settings,
        )
        preprocess_failures: list[str] = []

        base_config = batches_service.get_json_document(batch_id, "config", settings) or {}
        template_payload = _load_template_payload(batch_root)
        metadata = batches_service.get_batch_metadata(batch_id, settings)
        rotation_degrees = int(metadata.get("rotation_degrees", 0))
        dynamic_dimensions_by_file: dict[str, dict[str, int]] = {}
        latest_persisted_config = copy.deepcopy(base_config)

        for index, image_path in enumerate(input_images, start=1):
            if _is_cancel_requested(batch_id, settings):
                _mark_cancelled(
                    batch_id,
                    "Stopped by user before the next image started.",
                    settings,
                )
                return
            dynamic_dimensions = _compute_dynamic_dimensions(
                image_path, template_payload, rotation_degrees
            )
            dynamic_dimensions_by_file[image_path.name] = dynamic_dimensions
            latest_persisted_config = _merge_dimensions_into_config(
                base_config, dynamic_dimensions
            )
            runtime_dir = _prepare_runtime_dir(
                batch_root,
                image_path,
                latest_persisted_config,
                index,
                rotation_degrees,
            )

            args = {
                "input_paths": [str(runtime_dir)],
                "output_dir": str(outputs_dir),
                "debug": False,
                "autoAlign": False,
                "setLayout": False,
            }
            entry_point_for_args(args)
            _cleanup_runtime_dir(runtime_dir)
            runtime_dir = None

            if _image_ended_up_in_errors(outputs_dir, image_path.name):
                preprocess_failures.append(image_path.name)

            batches_service.update_batch_metadata(
                batch_id,
                {
                    "processed_files": index,
                    "total_files": len(input_images),
                    "latest_processed_file": image_path.name,
                    "latest_dynamic_dimensions": dynamic_dimensions,
                    "dynamic_dimensions_by_file": dynamic_dimensions_by_file,
                    "preprocess_failures": list(preprocess_failures),
                },
                settings,
            )

            if _is_cancel_requested(batch_id, settings):
                _mark_cancelled(
                    batch_id,
                    "Stopped by user after the current image finished.",
                    settings,
                )
                return

        batches_service.save_json_document(
            batch_id, "config", latest_persisted_config, settings
        )

        if preprocess_failures and len(preprocess_failures) == len(input_images):
            batches_service.update_status(
                batch_id,
                BatchStatus.failed,
                last_error=(
                    "Marker / page preprocessing failed for every input image. "
                    "Check that template preprocessors can locate markers on "
                    "your sheets. Failed files: "
                    + ", ".join(preprocess_failures)
                ),
                settings=settings,
            )
        elif preprocess_failures:
            batches_service.update_status(
                batch_id,
                BatchStatus.done,
                last_error=(
                    f"{len(preprocess_failures)} of {len(input_images)} file(s) "
                    "failed marker / page preprocessing and were moved to "
                    "outputs/Manual/ErrorFiles: "
                    + ", ".join(preprocess_failures)
                ),
                settings=settings,
            )
        else:
            batches_service.update_status(
                batch_id, BatchStatus.done, settings=settings
            )
    except BaseException as exc:
        logger.exception("OMR run failed for batch %s", batch_id)
        batches_service.update_status(
            batch_id,
            BatchStatus.failed,
            last_error=f"{type(exc).__name__}: {exc}",
            settings=settings,
        )
    finally:
        _cleanup_runtime_dir(runtime_dir)
        lock.release()


def queue_run(batch_id: str, settings: Settings | None = None) -> None:
    """Mark a batch as queued so the caller can hand off to a background task."""
    settings = settings or get_settings()
    batches_service.update_batch_metadata(
        batch_id,
        {
            "cancel_requested": False,
            "processed_files": 0,
            "total_files": 0,
            "latest_processed_file": None,
            "latest_dynamic_dimensions": None,
            "dynamic_dimensions_by_file": {},
        },
        settings,
    )
    batches_service.update_status(batch_id, BatchStatus.queued, settings=settings)


def request_cancel(batch_id: str, settings: Settings | None = None) -> BatchStatus:
    """Request cooperative cancellation for a queued or running batch."""
    settings = settings or get_settings()
    batch = batches_service.get_batch(batch_id, settings)
    if batch.status not in {BatchStatus.queued, BatchStatus.running}:
        raise batches_service.InvalidBatchRequest(
            "Only queued or running batches can be stopped."
        )

    batches_service.update_batch_metadata(
        batch_id,
        {"cancel_requested": True},
        settings,
    )

    if batch.status == BatchStatus.queued:
        _mark_cancelled(batch_id, "Cancelled before processing started.", settings)
        return BatchStatus.cancelled

    batches_service.update_status(
        batch_id,
        BatchStatus.running,
        last_error="Stop requested. The current image will finish first.",
        settings=settings,
    )
    return BatchStatus.running


def _read_csv_records(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        raw_rows = list(reader)

    if not raw_rows:
        return [], []

    header, *data_rows = raw_rows
    records = [
        dict(zip(header, row))
        for row in data_rows
        if row and any(str(value).strip() for value in row)
    ]
    return header, records


def _find_error_files_csvs(batch_id: str, settings: Settings) -> list[Path]:
    batch_root = batches_service.get_batch_root(batch_id, settings)
    outputs = batch_root / "outputs"
    if not outputs.exists():
        return []
    candidates = [
        path
        for manual_dir in outputs.rglob("Manual")
        if manual_dir.is_dir()
        for path in manual_dir.glob("ErrorFiles*.csv")
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime)
    return candidates


def _result_row_from_record(record: dict[str, str], status: str = "ok") -> ResultsRow:
    responses = {
        key: value
        for key, value in record.items()
        if key not in {"file_id", "input_path", "output_path", "score"}
    }
    error_reason = None
    if status == "failed":
        error_reason = "Marker preprocessing failed (image moved to Manual/ErrorFiles)."
    return ResultsRow(
        file_id=record.get("file_id") or "",
        input_path=record.get("input_path"),
        output_path=record.get("output_path"),
        score=record.get("score"),
        status=status,
        error_reason=error_reason,
        responses=responses,
    )


def read_results(batch_id: str, settings: Settings | None = None) -> ResultsPayload:
    """Parse the latest ``Results_*.csv`` for a batch, if any."""
    settings = settings or get_settings()
    csv_path = batches_service.find_results_csv(batch_id, settings)
    error_csvs = _find_error_files_csvs(batch_id, settings)
    if csv_path is None and not error_csvs:
        return ResultsPayload(
            batch_id=batch_id, columns=[], rows=[], generated_csv=None
        )

    header: list[str] = []
    rows: list[ResultsRow] = []
    seen_error_file_ids: set[str] = set()

    if csv_path is not None:
        header, records = _read_csv_records(csv_path)
        rows.extend(_result_row_from_record(record) for record in records)

    for error_csv in error_csvs:
        error_header, error_records = _read_csv_records(error_csv)
        if not header:
            header = error_header
        for record in error_records:
            file_id = record.get("file_id") or ""
            if file_id in seen_error_file_ids:
                continue
            seen_error_file_ids.add(file_id)
            rows.append(_result_row_from_record(record, status="failed"))

    return ResultsPayload(
        batch_id=batch_id,
        columns=header,
        rows=rows,
        generated_csv=csv_path.as_posix() if csv_path is not None else None,
    )


def results_csv_path(batch_id: str, settings: Settings | None = None) -> Path | None:
    """Return the raw path to the latest Results CSV (if any)."""
    settings = settings or get_settings()
    return batches_service.find_results_csv(batch_id, settings)
