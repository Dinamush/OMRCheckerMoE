#!/usr/bin/env python3
"""
Build static/app.ico for Windows (PyInstaller) from the same 32x32 layout as static/favicon.svg.

Uses Pillow only (no Cairo). Re-run after changing the favicon design:
  python scripts/build_app_icon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError as e:
    print("Install Pillow: pip install Pillow", file=sys.stderr)
    raise SystemExit(1) from e

ROOT = Path(__file__).resolve().parents[1]
OUT_ICO = ROOT / "static" / "app.ico"

# Align with static/favicon.svg (approximate ellipses / polygons for raster)
COL_TILE = (30, 64, 175, 255)
COL_HUSK_L = (34, 197, 94, 255)
COL_HUSK_R = (21, 128, 61, 255)
COL_COB = (234, 179, 8, 255)
COL_COB_EDGE = (161, 98, 7, 255)
COL_KERNEL = (133, 77, 14, 200)
COL_SILK = (254, 249, 195, 255)
COL_FLO_SHADOW = (15, 23, 42, 100)
COL_FLO_BODY = (226, 232, 240, 255)
COL_FLO_BAND = (148, 163, 184, 255)
COL_FLO_LABEL = (100, 116, 139, 255)
COL_FLO_SHUTTER = (71, 85, 105, 255)
COL_FLO_WIN = (37, 99, 235, 255)
COL_FLO_HUB = (219, 234, 254, 255)


def _scale(dim: int) -> float:
    return dim / 32.0


def _X(x: float, dim: int) -> int:
    return int(round(x * _scale(dim)))


def _Y(y: float, dim: int) -> int:
    return int(round(y * _scale(dim)))


def render(dim: int) -> Image.Image:
    s = _scale(dim)
    img = Image.new("RGBA", (dim, dim), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rc = max(_X(8, dim), dim // 16)
    d.rounded_rectangle([0, 0, dim - 1, dim - 1], radius=rc, fill=COL_TILE)

    # Husks
    d.polygon(
        [
            (_X(3.5, dim), _Y(26.5, dim)),
            (_X(3.5, dim), _Y(15, dim)),
            (_X(11.5, dim), _Y(9.5, dim)),
            (_X(10.5, dim), _Y(26.5, dim)),
        ],
        fill=COL_HUSK_L,
    )
    d.polygon(
        [
            (_X(28.5, dim), _Y(26.5, dim)),
            (_X(28.5, dim), _Y(15, dim)),
            (_X(20.5, dim), _Y(9.5, dim)),
            (_X(21.5, dim), _Y(26.5, dim)),
        ],
        fill=COL_HUSK_R,
    )

    # Cob (ellipse matches tapered silhouette closely at small sizes)
    cx, cy = _X(15.5, dim), _Y(15, dim)
    rw, rh = max(2, _X(4.8, dim)), max(3, _Y(11, dim))
    d.ellipse(
        [cx - rw, cy - rh, cx + rw, cy + rh],
        fill=COL_COB,
        outline=COL_COB_EDGE,
        width=max(1, int(round(0.4 * s))),
    )

    # Silk
    d.arc(
        [_X(12, dim), _Y(2.5, dim), _X(19, dim), _Y(8, dim)],
        start=200,
        end=340,
        fill=COL_SILK,
        width=max(1, int(round(1 * s))),
    )

    # Kernels (positions from favicon.svg)
    kr = max(1, int(round(1.1 * s)))
    pts = [
        (14.2, 9.8),
        (16.2, 9.8),
        (13.4, 13.2),
        (15.2, 13.2),
        (17.1, 13.2),
        (14.2, 16.6),
        (16.2, 16.6),
        (13.4, 20),
        (15.2, 20),
        (17.1, 20),
        (14.2, 23.2),
        (16.2, 23.2),
    ]
    for x, y in pts:
        px, py = _X(x, dim), _Y(y, dim)
        d.ellipse([px - kr, py - kr, px + kr, py + kr], fill=COL_KERNEL)

    # Floppy (bottom-right)
    ox, oy = _X(18.75, dim), _Y(17, dim)
    d.rounded_rectangle(
        [ox - _X(0.4, dim), oy - _Y(0.4, dim), ox + _X(11.9, dim), oy + _Y(10.3, dim)],
        radius=max(1, int(round(1.15 * s))),
        fill=COL_FLO_SHADOW,
    )
    d.rounded_rectangle(
        [ox, oy, ox + _X(11.5, dim), oy + _Y(9.9, dim)],
        radius=max(1, int(round(1.05 * s))),
        fill=COL_FLO_BODY,
    )
    d.rounded_rectangle(
        [ox, oy, ox + _X(11.5, dim), oy + _Y(3.55, dim)],
        radius=max(1, int(round(1.05 * s))),
        fill=COL_FLO_BAND,
    )
    d.rounded_rectangle(
        [
            ox + _X(0.65, dim),
            oy + _Y(0.65, dim),
            ox + _X(5.25, dim),
            oy + _Y(2.9, dim),
        ],
        radius=max(1, int(round(0.4 * s))),
        fill=COL_FLO_LABEL,
    )
    d.rounded_rectangle(
        [
            ox + _X(1.85, dim),
            oy + _Y(3.85, dim),
            ox + _X(9.65, dim),
            oy + _Y(4.85, dim),
        ],
        radius=max(1, int(round(0.2 * s))),
        fill=COL_FLO_SHUTTER,
    )
    d.rounded_rectangle(
        [
            ox + _X(2.15, dim),
            oy + _Y(5.45, dim),
            ox + _X(9.35, dim),
            oy + _Y(9.55, dim),
        ],
        radius=max(1, int(round(0.55 * s))),
        fill=COL_FLO_WIN,
    )
    hx = ox + _X(5.75, dim)
    hy = oy + _Y(7.45, dim)
    hr = max(1, int(round(1.1 * s)))
    d.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=COL_FLO_HUB)

    return img


def main() -> None:
    sizes = (256, 48, 32, 16)
    images = [render(d) for d in sizes]
    OUT_ICO.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        OUT_ICO,
        format="ICO",
        sizes=[(im.width, im.height) for im in images],
        append_images=images[1:],
    )
    print(f"Wrote {OUT_ICO} ({', '.join(str(s) for s in sizes)} px)")


if __name__ == "__main__":
    main()
