#!/usr/bin/env python3
"""
Check that installed packages meet the versions in requirements.txt
and run pip install -U for any that are missing or outdated.
"""

from __future__ import annotations

import re
import subprocess
import sys


def _parse_requirements_line(line: str) -> tuple[str, str | None] | None:
    """Parse a line like 'yt-dlp>=2025.1.1' or 'fastapi' -> (package_name, spec_or_none)."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # Match package name (letters, digits, hyphens, underscores) and optional specifier
    m = re.match(r"^([a-zA-Z0-9_.-]+)\s*([<=>!~]+.*)?$", line)
    if not m:
        return None
    name, spec = m.group(1), m.group(2)
    if spec and spec.strip():
        return (name, spec.strip())
    return (name, None)


def _parse_version(v: str) -> tuple[int, ...]:
    """Turn '2025.1.1' or '1.2' into comparable tuple. Non-numeric tails ignored."""
    parts = []
    for part in re.split(r"\.", v):
        part = re.sub(r"[^0-9].*", "", part)
        parts.append(int(part) if part.isdigit() else 0)
    return tuple(parts) if parts else (0,)


def _spec_satisfied(installed: str, spec: str) -> bool:
    """Check if installed version satisfies the spec (>=, ==, etc.). Simple implementation."""
    try:
        inst = _parse_version(installed)
    except (ValueError, TypeError):
        return False
    spec = spec.strip()
    if spec.startswith(">="):
        req = _parse_version(spec[2:].strip())
        return inst >= req
    if spec.startswith("=="):
        req = _parse_version(spec[2:].strip())
        return inst == req
    if spec.startswith("<="):
        req = _parse_version(spec[2:].strip())
        return inst <= req
    if spec.startswith(">"):
        req = _parse_version(spec[1:].strip())
        return inst > req
    if spec.startswith("<"):
        req = _parse_version(spec[1:].strip())
        return inst < req
    if spec.startswith("~="):
        req_str = spec[2:].strip()
        req = _parse_version(req_str)
        # Compatible release: >= req AND < (major+1,)
        upper = (req[0] + 1,) if req else (1,)
        return inst >= req and inst < upper
    return True


def _get_installed_version(package: str) -> str | None:
    """Return installed version string for package, or None if not installed."""
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "show", package],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode != 0:
            return None
        for line in (r.stdout or "").splitlines():
            if line.strip().lower().startswith("version:"):
                return line.split(":", 1)[1].strip()
        return None
    except Exception:
        return None


def _pip_install_upgrade(packages: list[str]) -> bool:
    """Run pip install -U for each package. Return True if all succeeded."""
    for pkg in packages:
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-U", pkg],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if r.returncode != 0:
                print(f"Warning: pip install -U {pkg} failed: {r.stderr or r.stdout}", file=sys.stderr)
                return False
        except Exception as e:
            print(f"Warning: pip install -U {pkg} failed: {e}", file=sys.stderr)
            return False
    return True


def check_and_update_requirements(requirements_path: str) -> bool:
    """
    Read requirements.txt, check installed versions, and upgrade any package
    that is missing or does not satisfy its version spec. Return True if
    all requirements are satisfied (after any upgrades).
    """
    try:
        with open(requirements_path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        print(f"Could not read requirements: {requirements_path} -> {e}", file=sys.stderr)
        return True

    to_upgrade: list[str] = []
    for line in lines:
        parsed = _parse_requirements_line(line)
        if not parsed:
            continue
        name, spec = parsed
        installed = _get_installed_version(name)
        if installed is None:
            to_upgrade.append(name)
            continue
        if spec and not _spec_satisfied(installed, spec):
            to_upgrade.append(name)
            continue

    if not to_upgrade:
        return True

    print(f"Updating packages to match {requirements_path}: {', '.join(to_upgrade)}")
    if _pip_install_upgrade(to_upgrade):
        print("Done.")
        return True
    return False


if __name__ == "__main__":
    import os
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    ok = check_and_update_requirements(req_path)
    sys.exit(0 if ok else 1)
