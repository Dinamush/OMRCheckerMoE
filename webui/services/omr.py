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


def _compute_dynamic_dimensions(image_path: Path) -> dict[str, int]:
    """Compute per-image display and processing dimensions."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path.as_posix()}")

    source_height, source_width = image.shape[:2]
    display_width, display_height = _scale_dimensions(
        source_width,
        source_height,
        int(CONFIG_DEFAULTS.dimensions.display_width),
        int(CONFIG_DEFAULTS.dimensions.display_height),
    )
    processing_width, processing_height = _scale_dimensions(
        source_width,
        source_height,
        int(CONFIG_DEFAULTS.dimensions.processing_width),
        int(CONFIG_DEFAULTS.dimensions.processing_height),
    )

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


def _prepare_runtime_dir(
    batch_root: Path,
    image_path: Path,
    runtime_config: dict,
    index: int,
) -> Path:
    """Build an isolated single-image runtime directory for engine execution."""
    runtime_root = batch_root / _RUNTIME_DIR_NAME / f"{index:04d}_{image_path.stem}"
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, runtime_root / image_path.name)
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
            dynamic_dimensions = _compute_dynamic_dimensions(image_path)
            dynamic_dimensions_by_file[image_path.name] = dynamic_dimensions
            latest_persisted_config = _merge_dimensions_into_config(
                base_config, dynamic_dimensions
            )
            runtime_dir = _prepare_runtime_dir(
                batch_root,
                image_path,
                latest_persisted_config,
                index,
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


def read_results(batch_id: str, settings: Settings | None = None) -> ResultsPayload:
    """Parse the latest ``Results_*.csv`` for a batch, if any."""
    settings = settings or get_settings()
    csv_path = batches_service.find_results_csv(batch_id, settings)
    if csv_path is None:
        return ResultsPayload(
            batch_id=batch_id, columns=[], rows=[], generated_csv=None
        )

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        raw_rows = list(reader)

    if not raw_rows:
        return ResultsPayload(
            batch_id=batch_id,
            columns=[],
            rows=[],
            generated_csv=csv_path.as_posix(),
        )

    header, *data_rows = raw_rows
    rows: list[ResultsRow] = []
    for row in data_rows:
        if not row:
            continue
        record = dict(zip(header, row))
        file_id = record.get("file_id") or (row[0] if row else "")
        responses = {
            key: value
            for key, value in record.items()
            if key not in {"file_id", "input_path", "output_path", "score"}
        }
        rows.append(
            ResultsRow(
                file_id=file_id,
                input_path=record.get("input_path"),
                output_path=record.get("output_path"),
                score=record.get("score"),
                responses=responses,
            )
        )

    return ResultsPayload(
        batch_id=batch_id,
        columns=header,
        rows=rows,
        generated_csv=csv_path.as_posix(),
    )


def results_csv_path(batch_id: str, settings: Settings | None = None) -> Path | None:
    """Return the raw path to the latest Results CSV (if any)."""
    settings = settings or get_settings()
    return batches_service.find_results_csv(batch_id, settings)
