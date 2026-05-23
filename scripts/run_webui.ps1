# Start the OMRChecker Web UI (activate venv if present).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

# Ensure CUDA/cuDNN DLLs are visible to OpenCV (see src/utils/gpu.py).
$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
$sitePackages = Join-Path $Root ".venv\Lib\site-packages"
$dllPaths = @(
    $cudaBin,
    (Join-Path $sitePackages "nvidia\cudnn\bin"),
    (Join-Path $sitePackages "nvidia\cublas\bin"),
    (Join-Path $sitePackages "nvidia\cuda_nvrtc\bin")
) | Where-Object { Test-Path $_ }
if ($dllPaths.Count -gt 0) {
    $env:Path = ($dllPaths -join ";") + ";" + $env:Path
}

$VenvUvicorn = Join-Path $Root ".venv\Scripts\uvicorn.exe"
if (Test-Path $VenvUvicorn) {
    & $VenvUvicorn webui.app:create_app --factory --host 127.0.0.1 --port 8000 @args
} else {
    uvicorn webui.app:create_app --factory --host 127.0.0.1 --port 8000 @args
}
