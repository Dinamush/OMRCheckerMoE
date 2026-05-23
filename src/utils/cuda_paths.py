"""Register CUDA/cuDNN DLL search paths on Windows (must run before ``import cv2``)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_registered = False


def register_cuda_dll_directories() -> None:
    """Add CUDA Toolkit and pip-shipped NVIDIA runtime folders to the DLL search path."""
    global _registered
    if _registered or sys.platform != "win32":
        return

    candidates: list[Path] = []
    cuda_root = os.environ.get(
        "CUDA_PATH",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
    )
    candidates.append(Path(cuda_root) / "bin")

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    if not site_packages.is_dir():
        try:
            import site

            for candidate in reversed(site.getsitepackages()):
                path = Path(candidate)
                if (path / "cv2").is_dir() or (path / "nvidia").is_dir():
                    site_packages = path
                    break
        except (ImportError, IndexError):
            site_packages = None

    if site_packages is not None and site_packages.is_dir():
        for relative in (
            "nvidia/cudnn/bin",
            "nvidia/cublas/bin",
            "nvidia/cuda_nvrtc/bin",
            "cv2",
        ):
            candidates.append(site_packages / relative)

    for directory in candidates:
        if not directory.is_dir():
            continue
        path_str = str(directory)
        os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(path_str)
            except OSError:
                pass

    _registered = True


register_cuda_dll_directories()
