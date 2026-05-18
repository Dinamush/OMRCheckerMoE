# Backfill annotated git tags for SHUCK3R releases documented in CHANGELOG.md.
# Run from repo root: .\scripts\tag_historical_releases.ps1
# Review tags with: git tag -l "v*" && git show v1.2.0
# Push when ready: git push origin --tags

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$tags = @(
    @{ Name = "v0.1.0"; Hash = "341d90d"; Message = "Initial prototype" },
    @{ Name = "v0.6.0"; Hash = "210559a"; Message = "Progress tracking and xHamster" },
    @{ Name = "v0.8.0"; Hash = "2959b50"; Message = "SHUCK3R rebrand" },
    @{ Name = "v1.0.0-beta.2"; Hash = "0a83cf0"; Message = "Session history, library, cancel" },
    @{ Name = "v1.1.0"; Hash = "855cdbe"; Message = "Pixiv bookmarks" },
    @{ Name = "v1.2.0"; Hash = "HEAD"; Message = "Pixiv ugoira, manga, translation" }
)

foreach ($t in $tags) {
    $exists = git tag -l $t.Name
    if ($exists) {
        Write-Host "Skip $($t.Name) (already exists)"
        continue
    }
    git tag -a $t.Name $t.Hash -m $t.Message
    Write-Host "Created $($t.Name) -> $($t.Hash)"
}

Write-Host "Done. List: git tag -l `"v*`""
