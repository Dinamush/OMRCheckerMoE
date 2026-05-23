# Install CUDA-enabled OpenCV into the project venv (Windows).
# Prerequisites:
#   1. NVIDIA driver (nvidia-smi works)
#   2. CUDA Toolkit 12.8: https://developer.nvidia.com/cuda-12-8-0-download-archive
#   3. cuDNN 9.7 for CUDA 12.x — extract into the CUDA directory OR add its bin path to cv2/config.py
#      (see https://github.com/cudawarped/opencv-python-cuda-wheels/releases/tag/4.11.0.20250124)

$ErrorActionPreference = "Stop"
# pip writes benign warnings to stderr; do not treat them as terminating errors.
$prevEap = $ErrorActionPreference
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$WheelUrl = "https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.11.0.20250124/opencv_contrib_python_rolling-4.12.0.86-cp37-abi3-win_amd64.whl"

if (-not (Test-Path $VenvPython)) {
    Write-Error "No .venv found. Run: python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt"
}

Write-Host "Removing CPU-only opencv-python..."
$ErrorActionPreference = "Continue"
& (Join-Path $Root ".venv\Scripts\pip.exe") uninstall -y opencv-python opencv-contrib-python *> $null
$ErrorActionPreference = $prevEap

Write-Host "Installing cuDNN runtime wheels (pip)..."
& (Join-Path $Root ".venv\Scripts\pip.exe") install nvidia-cudnn-cu12
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Installing CUDA OpenCV wheel (~295 MB)..."
& (Join-Path $Root ".venv\Scripts\pip.exe") install $WheelUrl
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Verifying GPU detection..."
& $VenvPython -c "from src.utils.gpu import is_gpu_available, gpu_status; print('available:', is_gpu_available()); print(gpu_status())"
