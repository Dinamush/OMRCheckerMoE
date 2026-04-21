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


def _discover_input_images(batch_id: str, settings: Settings) -> list[Path]:
    """Return all processable images for a batch."""
    return batches_service.list_input_image_paths(batch_id, settings)


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
            },
            settings,
        )

        base_config = batches_service.get_json_document(batch_id, "config", settings) or {}
        dynamic_dimensions_by_file: dict[str, dict[str, int]] = {}
        latest_persisted_config = copy.deepcopy(base_config)

        for index, image_path in enumerate(input_images, start=1):
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

            batches_service.update_batch_metadata(
                batch_id,
                {
                    "processed_files": index,
                    "total_files": len(input_images),
                    "latest_processed_file": image_path.name,
                    "latest_dynamic_dimensions": dynamic_dimensions,
                    "dynamic_dimensions_by_file": dynamic_dimensions_by_file,
                },
                settings,
            )

        batches_service.save_json_document(
            batch_id, "config", latest_persisted_config, settings
        )

        batches_service.update_status(batch_id, BatchStatus.done, settings=settings)
    except Exception as exc:
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
    batches_service.update_status(batch_id, BatchStatus.queued, settings=settings)


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
