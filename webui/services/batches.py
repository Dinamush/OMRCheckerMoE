"""Filesystem-backed batch management service.

A batch is a self-contained directory that mirrors what the OMRChecker
engine already expects::

    <storage_root>/<batch_id>/
        metadata.json
        inputs/
        outputs/
        template.json      (optional)
        config.json        (optional)
        evaluation.json    (optional)

Using the engine's native layout means we can process a batch via the
existing ``entry_point_for_args`` without modifying any engine code.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from webui.schemas import (
    Batch,
    BatchStatus,
    FileRef,
    SourceMode,
    TemplateAssetRef,
)
from webui.settings import Settings, get_settings

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
TEMPLATE_FILENAME = "template.json"
CONFIG_FILENAME = "config.json"
EVALUATION_FILENAME = "evaluation.json"
METADATA_FILENAME = "metadata.json"
ASSET_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
RESERVED_BATCH_FILES = {
    TEMPLATE_FILENAME,
    CONFIG_FILENAME,
    EVALUATION_FILENAME,
    METADATA_FILENAME,
}

JSON_DOC_NAMES = {
    "template": TEMPLATE_FILENAME,
    "config": CONFIG_FILENAME,
    "evaluation": EVALUATION_FILENAME,
}

_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


class BatchNotFound(Exception):
    """Raised when a batch id does not map to an existing directory."""


class InvalidBatchRequest(Exception):
    """Raised on bad input (e.g. unsafe filenames, missing directory)."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sanitise_filename(name: str) -> str:
    """Return a filesystem-safe version of a user-supplied filename."""
    stem = Path(name).name
    cleaned = _SAFE_FILENAME.sub("_", stem).strip("._")
    if not cleaned:
        raise InvalidBatchRequest(f"Invalid filename: {name!r}")
    return cleaned


def _serialise(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _batch_root(settings: Settings, batch_id: str) -> Path:
    return settings.ensure_storage() / batch_id


def _metadata_path(settings: Settings, batch_id: str) -> Path:
    return _batch_root(settings, batch_id) / METADATA_FILENAME


def _inputs_dir(settings: Settings, batch_id: str) -> Path:
    return _batch_root(settings, batch_id) / "inputs"


def _outputs_dir(settings: Settings, batch_id: str) -> Path:
    return _batch_root(settings, batch_id) / "outputs"


def _load_metadata(settings: Settings, batch_id: str) -> dict[str, Any]:
    meta_path = _metadata_path(settings, batch_id)
    if not meta_path.exists():
        raise BatchNotFound(batch_id)
    with meta_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_metadata(settings: Settings, batch_id: str, data: dict[str, Any]) -> None:
    meta_path = _metadata_path(settings, batch_id)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: _serialise(v) for k, v in data.items()}
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def _file_count(batch_dir: Path) -> int:
    inputs = batch_dir / "inputs"
    if not inputs.exists():
        return 0
    return sum(
        1
        for child in inputs.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    )


def _to_batch(settings: Settings, batch_id: str, meta: dict[str, Any]) -> Batch:
    batch_dir = _batch_root(settings, batch_id)
    return Batch(
        id=batch_id,
        name=meta.get("name", batch_id),
        status=BatchStatus(meta.get("status", BatchStatus.created.value)),
        created_at=meta.get("created_at") or _now().isoformat(),
        updated_at=meta.get("updated_at") or meta.get("created_at") or _now().isoformat(),
        source_mode=(
            SourceMode(meta["source_mode"]) if meta.get("source_mode") else None
        ),
        source_dir=meta.get("source_dir"),
        last_error=meta.get("last_error"),
        file_count=_file_count(batch_dir),
        has_template=(batch_dir / TEMPLATE_FILENAME).exists(),
        has_config=(batch_dir / CONFIG_FILENAME).exists(),
        has_evaluation=(batch_dir / EVALUATION_FILENAME).exists(),
    )


def create_batch(name: str, settings: Settings | None = None) -> Batch:
    """Create a new empty batch and return its metadata."""
    settings = settings or get_settings()
    batch_id = uuid.uuid4().hex[:12]
    batch_dir = _batch_root(settings, batch_id)
    (batch_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (batch_dir / "outputs").mkdir(parents=True, exist_ok=True)
    now = _now()
    meta = {
        "id": batch_id,
        "name": name.strip() or batch_id,
        "status": BatchStatus.created.value,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "source_mode": None,
        "source_dir": None,
        "last_error": None,
    }
    _save_metadata(settings, batch_id, meta)
    return _to_batch(settings, batch_id, meta)


def list_batches(settings: Settings | None = None) -> list[Batch]:
    """Return all known batches, newest first."""
    settings = settings or get_settings()
    root = settings.ensure_storage()
    batches: list[Batch] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        meta_path = child / METADATA_FILENAME
        if not meta_path.exists():
            continue
        try:
            meta = _load_metadata(settings, child.name)
        except (OSError, json.JSONDecodeError):
            continue
        batches.append(_to_batch(settings, child.name, meta))
    batches.sort(key=lambda b: b.created_at, reverse=True)
    return batches


def get_batch(batch_id: str, settings: Settings | None = None) -> Batch:
    """Return a single batch or raise ``BatchNotFound``."""
    settings = settings or get_settings()
    meta = _load_metadata(settings, batch_id)
    return _to_batch(settings, batch_id, meta)


def delete_batch(batch_id: str, settings: Settings | None = None) -> None:
    """Delete the batch directory from disk."""
    settings = settings or get_settings()
    batch_dir = _batch_root(settings, batch_id)
    if not batch_dir.exists():
        raise BatchNotFound(batch_id)
    shutil.rmtree(batch_dir)


def update_status(
    batch_id: str,
    status: BatchStatus,
    last_error: str | None = None,
    settings: Settings | None = None,
) -> Batch:
    """Persist a status transition for a batch."""
    settings = settings or get_settings()
    meta = _load_metadata(settings, batch_id)
    meta["status"] = status.value
    meta["updated_at"] = _now().isoformat()
    if last_error is not None or status == BatchStatus.failed:
        meta["last_error"] = last_error
    elif status in {BatchStatus.queued, BatchStatus.running, BatchStatus.done}:
        meta["last_error"] = None
    _save_metadata(settings, batch_id, meta)
    return _to_batch(settings, batch_id, meta)


def set_source(
    batch_id: str,
    source_mode: SourceMode,
    source_dir: str | None,
    settings: Settings | None = None,
) -> None:
    """Record the last source used to add files to a batch."""
    settings = settings or get_settings()
    meta = _load_metadata(settings, batch_id)
    existing = meta.get("source_mode")
    if existing and existing != source_mode.value:
        meta["source_mode"] = SourceMode.mixed.value
    else:
        meta["source_mode"] = source_mode.value
    if source_dir is not None:
        meta["source_dir"] = source_dir
    meta["updated_at"] = _now().isoformat()
    _save_metadata(settings, batch_id, meta)


def list_files(batch_id: str, settings: Settings | None = None) -> list[FileRef]:
    """Return the image files currently attached to a batch."""
    settings = settings or get_settings()
    inputs = _inputs_dir(settings, batch_id)
    if not inputs.exists():
        raise BatchNotFound(batch_id)
    return [
        FileRef(name=child.name, size_bytes=child.stat().st_size)
        for child in sorted(inputs.iterdir())
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    ]


def list_input_image_paths(
    batch_id: str, settings: Settings | None = None
) -> list[Path]:
    """Return on-disk paths for input images in a batch."""
    settings = settings or get_settings()
    inputs = _inputs_dir(settings, batch_id)
    if not inputs.exists():
        raise BatchNotFound(batch_id)
    return [
        child
        for child in sorted(inputs.iterdir())
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    ]


def save_uploaded_file(
    batch_id: str,
    filename: str,
    data: bytes,
    settings: Settings | None = None,
) -> FileRef:
    """Write an uploaded image into the batch's ``inputs/`` folder."""
    settings = settings or get_settings()
    inputs = _inputs_dir(settings, batch_id)
    if not inputs.exists():
        raise BatchNotFound(batch_id)
    safe = _sanitise_filename(filename)
    suffix = Path(safe).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise InvalidBatchRequest(
            f"Unsupported file type {suffix!r}; allowed: {sorted(IMAGE_EXTENSIONS)}"
        )
    target = inputs / safe
    if target.exists():
        stem, ext = target.stem, target.suffix
        counter = 1
        while (inputs / f"{stem}_{counter}{ext}").exists():
            counter += 1
        target = inputs / f"{stem}_{counter}{ext}"
    target.write_bytes(data)
    set_source(batch_id, SourceMode.upload, None, settings)
    return FileRef(name=target.name, size_bytes=target.stat().st_size)


def delete_file(
    batch_id: str,
    filename: str,
    settings: Settings | None = None,
) -> None:
    """Remove a single file from a batch's inputs directory."""
    settings = settings or get_settings()
    inputs = _inputs_dir(settings, batch_id)
    if not inputs.exists():
        raise BatchNotFound(batch_id)
    safe = _sanitise_filename(filename)
    target = inputs / safe
    if not target.exists() or not target.is_file():
        raise InvalidBatchRequest(f"File not found: {safe}")
    target.unlink()


def resolve_input_file(
    batch_id: str,
    filename: str,
    settings: Settings | None = None,
) -> Path:
    """Return a safely resolved input image path for preview/download."""
    settings = settings or get_settings()
    inputs = _inputs_dir(settings, batch_id)
    if not inputs.exists():
        raise BatchNotFound(batch_id)
    safe = _sanitise_filename(filename)
    target = (inputs / safe).resolve()
    try:
        target.relative_to(inputs.resolve())
    except ValueError as exc:
        raise InvalidBatchRequest("Path traversal not allowed") from exc
    if (
        not target.exists()
        or not target.is_file()
        or target.suffix.lower() not in IMAGE_EXTENSIONS
    ):
        raise InvalidBatchRequest(f"Input image not found: {safe}")
    return target


def import_directory(
    batch_id: str,
    source_dir: str,
    copy: bool = True,
    settings: Settings | None = None,
) -> tuple[list[FileRef], list[str]]:
    """Copy (or link) all supported images from ``source_dir`` into the batch."""
    settings = settings or get_settings()
    if not settings.allow_directory_import:
        raise InvalidBatchRequest(
            "Directory import is disabled on this server (ALLOW_DIRECTORY_IMPORT=false)."
        )
    src = Path(source_dir).expanduser()
    if not src.exists() or not src.is_dir():
        raise InvalidBatchRequest(f"Source directory does not exist: {source_dir}")

    inputs = _inputs_dir(settings, batch_id)
    if not inputs.exists():
        raise BatchNotFound(batch_id)

    imported: list[FileRef] = []
    skipped: list[str] = []
    for child in sorted(src.iterdir()):
        if not child.is_file():
            continue
        if child.suffix.lower() not in IMAGE_EXTENSIONS:
            skipped.append(child.name)
            continue
        safe = _sanitise_filename(child.name)
        target = inputs / safe
        if target.exists():
            skipped.append(child.name)
            continue
        if copy:
            shutil.copy2(child, target)
        else:
            try:
                target.symlink_to(child.resolve())
            except (OSError, NotImplementedError):
                shutil.copy2(child, target)
        imported.append(FileRef(name=target.name, size_bytes=target.stat().st_size))

    if imported:
        set_source(batch_id, SourceMode.directory, str(src), settings)
    return imported, skipped


def get_json_document(
    batch_id: str,
    doc_name: str,
    settings: Settings | None = None,
) -> dict[str, Any] | None:
    """Return a parsed template/config/evaluation JSON or ``None`` if absent."""
    settings = settings or get_settings()
    if doc_name not in JSON_DOC_NAMES:
        raise InvalidBatchRequest(f"Unknown document {doc_name!r}")
    path = _batch_root(settings, batch_id) / JSON_DOC_NAMES[doc_name]
    if not _batch_root(settings, batch_id).exists():
        raise BatchNotFound(batch_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json_document(
    batch_id: str,
    doc_name: str,
    content: dict[str, Any] | None,
    settings: Settings | None = None,
) -> None:
    """Write or delete one of the optional template/config/evaluation files."""
    settings = settings or get_settings()
    if doc_name not in JSON_DOC_NAMES:
        raise InvalidBatchRequest(f"Unknown document {doc_name!r}")
    root = _batch_root(settings, batch_id)
    if not root.exists():
        raise BatchNotFound(batch_id)
    path = root / JSON_DOC_NAMES[doc_name]
    if content is None:
        if path.exists():
            path.unlink()
        return
    if not isinstance(content, dict):
        raise InvalidBatchRequest(f"{doc_name}.json must be a JSON object")
    with path.open("w", encoding="utf-8") as fh:
        json.dump(content, fh, indent=2, sort_keys=True)
    meta = _load_metadata(settings, batch_id)
    meta["updated_at"] = _now().isoformat()
    _save_metadata(settings, batch_id, meta)


def get_batch_metadata(
    batch_id: str, settings: Settings | None = None
) -> dict[str, Any]:
    """Return raw metadata.json content for a batch."""
    settings = settings or get_settings()
    return _load_metadata(settings, batch_id)


def update_batch_metadata(
    batch_id: str,
    updates: dict[str, Any],
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Merge arbitrary keys into metadata.json and return the new payload."""
    settings = settings or get_settings()
    meta = _load_metadata(settings, batch_id)
    meta.update(updates)
    meta["updated_at"] = _now().isoformat()
    _save_metadata(settings, batch_id, meta)
    return meta


def find_results_csv(batch_id: str, settings: Settings | None = None) -> Path | None:
    """Locate the most recent ``Results_*.csv`` produced by the engine."""
    settings = settings or get_settings()
    outputs = _outputs_dir(settings, batch_id)
    if not outputs.exists():
        return None
    candidates: list[Path] = []
    for results_dir in outputs.rglob("Results"):
        if not results_dir.is_dir():
            continue
        candidates.extend(results_dir.glob("Results_*.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def iter_output_files(batch_id: str, settings: Settings | None = None) -> Iterable[Path]:
    """Yield every file produced in the batch's ``outputs/`` tree."""
    settings = settings or get_settings()
    outputs = _outputs_dir(settings, batch_id)
    if not outputs.exists():
        return []
    return (p for p in outputs.rglob("*") if p.is_file())


def resolve_output_file(
    batch_id: str,
    relative: str,
    settings: Settings | None = None,
) -> Path:
    """Resolve ``relative`` safely against the batch's ``outputs/`` folder."""
    settings = settings or get_settings()
    outputs = _outputs_dir(settings, batch_id).resolve()
    if not outputs.exists():
        raise BatchNotFound(batch_id)
    candidate = (outputs / relative).resolve()
    try:
        candidate.relative_to(outputs)
    except ValueError as exc:
        raise InvalidBatchRequest("Path traversal not allowed") from exc
    if not candidate.exists() or not candidate.is_file():
        raise InvalidBatchRequest(f"Output file not found: {relative}")
    return candidate


def get_batch_root(batch_id: str, settings: Settings | None = None) -> Path:
    """Return the on-disk directory for a batch (for engine invocation)."""
    settings = settings or get_settings()
    root = _batch_root(settings, batch_id)
    if not root.exists():
        raise BatchNotFound(batch_id)
    return root


def _collect_template_relative_paths(value: Any) -> list[str]:
    """Walk a parsed template.json payload and collect ``relativePath`` values."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "relativePath" and isinstance(item, str):
                found.append(item)
            else:
                found.extend(_collect_template_relative_paths(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_template_relative_paths(item))
    return found


def _template_required_asset_names(
    batch_id: str, settings: Settings
) -> list[str]:
    """Return deduplicated asset filenames referenced by template.json."""
    template_doc = get_json_document(batch_id, "template", settings)
    if not template_doc:
        return []
    raw_paths = _collect_template_relative_paths(
        template_doc.get("preProcessors", [])
    )
    seen: list[str] = []
    for rel in raw_paths:
        name = Path(rel).name
        if name and name not in seen:
            seen.append(name)
    return seen


def _asset_is_present(batch_dir: Path, name: str) -> Path | None:
    """Return the path to an asset if it exists in root or inputs/, else None."""
    candidates = (batch_dir / name, batch_dir / "inputs" / name)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def list_template_assets(
    batch_id: str, settings: Settings | None = None
) -> list[TemplateAssetRef]:
    """Return template-referenced assets and whether each is present on disk."""
    settings = settings or get_settings()
    batch_dir = _batch_root(settings, batch_id)
    if not batch_dir.exists():
        raise BatchNotFound(batch_id)

    required = _template_required_asset_names(batch_id, settings)
    refs: list[TemplateAssetRef] = []
    for name in required:
        resolved = _asset_is_present(batch_dir, name)
        refs.append(
            TemplateAssetRef(
                name=name,
                required=True,
                present=resolved is not None,
                size_bytes=resolved.stat().st_size if resolved else None,
            )
        )
    return refs


def missing_template_assets(
    batch_id: str, settings: Settings | None = None
) -> list[str]:
    """Return names of template-referenced assets that are not yet on disk."""
    return [asset.name for asset in list_template_assets(batch_id, settings) if not asset.present]


def _validate_asset_filename(name: str) -> str:
    """Sanitize and validate a user-supplied asset filename."""
    safe = _sanitise_filename(name)
    if safe in RESERVED_BATCH_FILES:
        raise InvalidBatchRequest(
            f"{safe!r} is reserved for the batch itself; use a different filename."
        )
    suffix = Path(safe).suffix.lower()
    if suffix not in ASSET_EXTENSIONS:
        raise InvalidBatchRequest(
            f"Unsupported asset type {suffix!r}; allowed: {sorted(ASSET_EXTENSIONS)}"
        )
    return safe


def save_template_asset(
    batch_id: str,
    filename: str,
    data: bytes,
    settings: Settings | None = None,
) -> TemplateAssetRef:
    """Write a template asset (e.g. ``omr_marker.jpg``) into the batch root."""
    settings = settings or get_settings()
    batch_dir = _batch_root(settings, batch_id)
    if not batch_dir.exists():
        raise BatchNotFound(batch_id)
    safe = _validate_asset_filename(filename)
    target = batch_dir / safe
    target.write_bytes(data)

    required = _template_required_asset_names(batch_id, settings)
    return TemplateAssetRef(
        name=safe,
        required=safe in required,
        present=True,
        size_bytes=target.stat().st_size,
    )


def delete_template_asset(
    batch_id: str,
    filename: str,
    settings: Settings | None = None,
) -> None:
    """Remove a previously uploaded template asset from the batch root."""
    settings = settings or get_settings()
    batch_dir = _batch_root(settings, batch_id)
    if not batch_dir.exists():
        raise BatchNotFound(batch_id)
    safe = _validate_asset_filename(filename)
    target = batch_dir / safe
    if not target.exists() or not target.is_file():
        raise InvalidBatchRequest(f"Asset not found: {safe}")
    target.unlink()


def resolve_template_asset(
    batch_id: str,
    filename: str,
    settings: Settings | None = None,
) -> Path:
    """Return a safely resolved template asset image path for preview/download."""
    settings = settings or get_settings()
    batch_dir = _batch_root(settings, batch_id)
    if not batch_dir.exists():
        raise BatchNotFound(batch_id)
    safe = _validate_asset_filename(filename)
    resolved = _asset_is_present(batch_dir, safe)
    if resolved is None:
        raise InvalidBatchRequest(f"Asset not found: {safe}")
    return resolved.resolve()


def reset_batch_runtime_state(
    batch_id: str, settings: Settings | None = None
) -> None:
    """Remove generated runtime/output artifacts and reset run metadata."""
    settings = settings or get_settings()
    root = get_batch_root(batch_id, settings)

    for name in ("outputs", "_runtime"):
        target = root / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

    (root / "outputs").mkdir(parents=True, exist_ok=True)

    meta = _load_metadata(settings, batch_id)
    meta["status"] = BatchStatus.created.value
    meta["last_error"] = None
    meta["updated_at"] = _now().isoformat()
    for key in (
        "processed_files",
        "total_files",
        "latest_processed_file",
        "latest_dynamic_dimensions",
        "dynamic_dimensions_by_file",
        "cancel_requested",
    ):
        meta.pop(key, None)
    _save_metadata(settings, batch_id, meta)
