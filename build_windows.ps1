# Build a single portable HamsterScraper.exe under ./release/ (keeps repo root tidy).
# Prerequisites: pip install -r requirements.txt -r requirements-build.txt

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

New-Item -ItemType Directory -Force -Path "./release" | Out-Null

Write-Host "Building release/HamsterScraper.exe ..."
Get-Process -Name "HamsterScraper" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1
Remove-Item -Force -ErrorAction SilentlyContinue "./release/HamsterScraper.exe"
pyinstaller hamster_scraper.spec --distpath ./release --workpath build/pyinstaller --noconfirm
if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller failed (exit $LASTEXITCODE)."
    exit $LASTEXITCODE
}

if (-not (Test-Path "./release/HamsterScraper.exe")) {
    Write-Error "Build failed: release/HamsterScraper.exe not found."
    exit 1
}

Write-Host "Done: .\release\HamsterScraper.exe"
Write-Host 'Default data folder (Windows): %LOCALAPPDATA%\HamsterScraper  (per user)'
