"""Optional GPU acceleration helpers.

Uses ``cv2.cuda`` when an OpenCV CUDA build **and** a compatible GPU are
present.  Every public function falls back gracefully to its CPU equivalent
and emits a one-time diagnostic so operators know exactly why the GPU path
was skipped and what to do about it.

Typical failure reasons and remedies
--------------------------------------
* "OpenCV CUDA module not found"
  → Install an OpenCV wheel compiled with CUDA support:
      pip install opencv-contrib-python-headless
    or build OpenCV from source with -DWITH_CUDA=ON.

* "getCudaEnabledDeviceCount() raised …"
  → Your CUDA runtime / driver is outdated or mismatched.
    Update your GPU driver and ensure the CUDA toolkit version
    matches the OpenCV build (run: nvcc --version).

* "No CUDA-capable GPU was found"
  → This machine has no NVIDIA GPU, or the GPU is not CUDA-capable.
    CPU fallback is the only option.

* "GPU call failed … falling back to CPU"
  → A one-off runtime error (out-of-memory, driver hiccup).
    Check GPU VRAM usage; reduce batch size if memory is the cause.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU availability detection (runs exactly once at import time)
# ---------------------------------------------------------------------------

def _detect_gpu() -> tuple[bool, str]:
    """Return ``(is_available, human_readable_reason)``."""
    if not hasattr(cv2, "cuda"):
        return False, (
            "OpenCV CUDA module not found. "
            "To enable GPU acceleration install an OpenCV wheel built with CUDA "
            "(e.g. pip install opencv-contrib-python-headless with a CUDA build), "
            "or build OpenCV from source with -DWITH_CUDA=ON. "
            "Running on CPU."
        )
    try:
        n_devices = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception as exc:
        return False, (
            f"cv2.cuda is present but getCudaEnabledDeviceCount() raised "
            f"{type(exc).__name__}: {exc}. "
            "Ensure your NVIDIA driver and CUDA toolkit are up to date "
            "(run: nvidia-smi and nvcc --version). "
            "Running on CPU."
        )
    if n_devices == 0:
        return False, (
            "cv2.cuda is available but no CUDA-capable GPU was detected on this machine. "
            "Running on CPU."
        )
    try:
        dev_name: str = cv2.cuda.DeviceInfo(0).name()
    except Exception:
        dev_name = "unknown"
    return True, f"GPU acceleration enabled — using CUDA device 0: {dev_name}"


_GPU_AVAILABLE, _GPU_REASON = _detect_gpu()

# Log once at module load so the startup log makes the GPU status clear.
if _GPU_AVAILABLE:
    logger.info("[gpu] %s", _GPU_REASON)
else:
    # Debug level so it doesn't pollute normal log output on CPU-only machines.
    logger.debug("[gpu] %s", _GPU_REASON)


def is_gpu_available() -> bool:
    """Return ``True`` if a CUDA-capable GPU is ready for use."""
    return _GPU_AVAILABLE


def gpu_status() -> str:
    """Return a human-readable GPU status string (for health endpoints etc.)."""
    return _GPU_REASON


# ---------------------------------------------------------------------------
# Public GPU-accelerated helpers with CPU fallback
# ---------------------------------------------------------------------------

def warp_perspective(
    image: np.ndarray,
    homography: np.ndarray,
    dsize: tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    """Perspective warp with optional CUDA acceleration.

    Equivalent to ``cv2.warpPerspective``.  When a CUDA GPU is available the
    warp runs on the GPU (typically 3–10× faster for full-resolution images);
    if anything goes wrong it falls back to the CPU implementation and logs a
    warning with actionable guidance.
    """
    if _GPU_AVAILABLE:
        try:
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(image)
            gpu_dst = cv2.cuda.warpPerspective(
                gpu_src, homography, dsize,
                flags=flags,
                borderMode=border_mode,
            )
            return gpu_dst.download()
        except cv2.error as exc:
            logger.warning(
                "[gpu] warp_perspective: OpenCV CUDA error (%s). "
                "Falling back to CPU. "
                "If this is an out-of-memory error, close other GPU applications "
                "or reduce the image resolution.",
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[gpu] warp_perspective: unexpected error %s: %s. "
                "Falling back to CPU.",
                type(exc).__name__, exc,
            )
    return cv2.warpPerspective(
        image, homography, dsize, flags=flags, borderMode=border_mode
    )
