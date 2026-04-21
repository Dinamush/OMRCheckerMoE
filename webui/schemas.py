"""Pydantic schemas for the OMRChecker Web UI."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BatchStatus(str, Enum):
    """Lifecycle states for a batch."""

    created = "created"
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


class SourceMode(str, Enum):
    """How input files got into the batch."""

    upload = "upload"
    directory = "directory"
    mixed = "mixed"


class BatchCreate(BaseModel):
    """Payload for creating a new batch."""

    name: str = Field(min_length=1, max_length=120)


class Batch(BaseModel):
    """Persisted metadata about a batch."""

    id: str
    name: str
    status: BatchStatus = BatchStatus.created
    created_at: datetime
    updated_at: datetime
    source_mode: SourceMode | None = None
    source_dir: str | None = None
    last_error: str | None = None
    file_count: int = 0
    has_template: bool = False
    has_config: bool = False
    has_evaluation: bool = False


class FileRef(BaseModel):
    """A single image file inside a batch."""

    name: str
    size_bytes: int


class DirectoryImportRequest(BaseModel):
    """Payload for importing scanned images from a server-side directory."""

    model_config = {"protected_namespaces": ()}

    source_dir: str = Field(min_length=1)
    copy_files: bool = Field(
        default=True,
        alias="copy",
        description="Copy files into the batch (true) or symlink where supported (false).",
    )


class ImportResult(BaseModel):
    """Result of a directory import."""

    imported: list[FileRef]
    skipped: list[str] = Field(default_factory=list)


class JsonDocument(BaseModel):
    """Wrapper for optional template/config/evaluation JSON blobs."""

    name: str
    content: dict[str, Any] | None = None


class ProcessAccepted(BaseModel):
    """Response after queueing a processing run."""

    batch_id: str
    status: BatchStatus


class DynamicDimensions(BaseModel):
    """Derived dimensions used for a single processed image."""

    source_height: int
    source_width: int
    display_height: int
    display_width: int
    processing_height: int
    processing_width: int


class BatchStatusResponse(BaseModel):
    """Polled status payload."""

    id: str
    status: BatchStatus
    last_error: str | None = None
    file_count: int
    updated_at: datetime
    processed_files: int = 0
    total_files: int = 0
    latest_processed_file: str | None = None
    latest_dynamic_dimensions: DynamicDimensions | None = None


class ResultsRow(BaseModel):
    """A single parsed row from the generated Results CSV.

    The OMR engine writes arbitrary per-template columns; those are kept
    in ``responses`` so this model stays stable across templates.
    """

    file_id: str
    input_path: str | None = None
    output_path: str | None = None
    score: str | None = None
    responses: dict[str, str] = Field(default_factory=dict)


class ResultsPayload(BaseModel):
    """Full results payload exposed by the API."""

    batch_id: str
    columns: list[str]
    rows: list[ResultsRow]
    generated_csv: str | None = None
