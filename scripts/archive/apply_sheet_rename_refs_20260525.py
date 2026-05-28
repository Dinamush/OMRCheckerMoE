"""One-shot reference updater for the MoE sheet directory rename.

Run from repo root after ``git mv``::

    python scripts/apply_sheet_rename_refs.py

Replaces legacy directory/package strings in text files. Skips binary
dirs and VCS/venv artefacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Order matters: longer / more specific names first.
REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("portrait_25q_v2", "MoE-May-2026-Portrait-SMQ25-1"),
    ("old_custom25_answer_sheet_v1", "MoE-April-2026-Portrait-NNQ25-0"),
    ("custom_25_definitive_final", "MoE-April-2026-Landscape-NNQ25-0"),
    ("portrait_25q", "MoE-May-2026-Variants-SMQ25-0"),
    ("prefill_only_package", "prefill_package"),
)

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    ".pytest_cache",
}

# Never rewrite the alias table — it must keep legacy *keys*.
SKIP_FILES = {
    "webui/sheet_registry.py",
}

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".html",
    ".js",
    ".txt",
    ".spec",
    ".yaml",
    ".yml",
    ".svg",
    ".excalidraw",
    ".ini",
    ".toml",
}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def main() -> int:
    changed: list[Path] = []
    for path in REPO.rglob("*"):
        if not path.is_file() or should_skip(path):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            "OMRChecker.spec",
            "Dockerfile",
        }:
            continue
        if path.name == Path(__file__).name:
            continue
        rel = path.relative_to(REPO).as_posix()
        if rel in SKIP_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        original = text
        for old, new in REPLACEMENTS:
            text = text.replace(old, new)
        if text != original:
            path.write_text(text, encoding="utf-8", newline="\n")
            changed.append(path)
    print(f"Updated {len(changed)} files")
    for p in sorted(changed)[:40]:
        print(f"  {p.relative_to(REPO)}")
    if len(changed) > 40:
        print(f"  ... and {len(changed) - 40} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
