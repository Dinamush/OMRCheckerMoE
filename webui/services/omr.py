"""Wrapper that drives the OMRChecker engine for a given batch.

The engine reads ``template.json``, ``config.json`` and ``evaluation.json``
from the *same* directory as the images. To keep the engine untouched, we
copy any batch-level JSON files into ``inputs/`` just before processing
and tidy them up afterwards.
"""

from __future__ import annotations

import csv
import logging
import shutil
import threading
from pathlib import Path

from webui.schemas import (
    BatchStatus,
    ResultsPayload,
    ResultsRow,
)
from webui.services import batches as batches_service
from webui.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_COPIED_JSON_FILES = ("template.json", "config.json", "evaluation.json")

_batch_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(batch_id: str) -> threading.Lock:
    with _locks_guard:
        lock = _batch_locks.get(batch_id)
        if lock is None:
            lock = threading.Lock()
            _batch_locks[batch_id] = lock
        return lock


def _stage_json_files(batch_root: Path) -> list[Path]:
    """Copy template/config/evaluation into ``inputs/`` for the engine.

    Returns the list of files that were actually copied so the caller can
    remove them after the run finishes.
    """
    staged: list[Path] = []
    inputs_dir = batch_root / "inputs"
    for name in _COPIED_JSON_FILES:
        src = batch_root / name
        if not src.exists():
            continue
        dst = inputs_dir / name
        shutil.copy2(src, dst)
        staged.append(dst)
    return staged


def _unstage_json_files(staged: list[Path]) -> None:
    for path in staged:
        try:
            path.unlink()
        except OSError:
            logger.exception("Failed to remove staged file %s", path)


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

    staged: list[Path] = []
    try:
        batch_root = batches_service.get_batch_root(batch_id, settings)
        inputs_dir = batch_root / "inputs"
        outputs_dir = batch_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        if not any(
            p.is_file() and p.suffix.lower() in batches_service.IMAGE_EXTENSIONS
            for p in inputs_dir.iterdir()
        ):
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
        staged = _stage_json_files(batch_root)

        args = {
            "input_paths": [str(inputs_dir)],
            "output_dir": str(outputs_dir),
            "debug": False,
            "autoAlign": False,
            "setLayout": False,
        }
        entry_point_for_args(args)

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
        _unstage_json_files(staged)
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
