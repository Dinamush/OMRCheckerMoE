"""Runtime configuration for the OMRChecker Web UI.

Values can be overridden via environment variables (prefix ``OMR_WEBUI_``)
or a ``.env`` file at the repo root. The defaults are chosen to match the
plan's local-first posture.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings for the web UI."""

    model_config = SettingsConfigDict(
        env_prefix="OMR_WEBUI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    storage_root: Path = Field(
        default=REPO_ROOT / "webui" / "storage" / "batches",
        description="Directory where each batch's files and outputs live.",
    )

    allow_directory_import: bool = Field(
        default=True,
        description=(
            "When true, /api/v1/batches/{id}/files/import can read from an "
            "arbitrary server-side directory. Turn off in hosted deployments."
        ),
    )

    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed origins. Use an explicit list in production.",
    )

    max_upload_bytes: int = Field(
        default=50 * 1024 * 1024,
        description="Per-file upload limit in bytes (default 50 MiB).",
    )

    def ensure_storage(self) -> Path:
        """Create and return the batches root directory."""
        self.storage_root.mkdir(parents=True, exist_ok=True)
        return self.storage_root


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a memoised ``Settings`` instance."""
    return Settings()
