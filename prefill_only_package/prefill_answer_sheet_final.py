#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_TEMPLATE = Path('blank_template_reference.png')

# Relative coordinates for the current landscape blank template.
CFG = {
    'student_name_box': (0.040, 0.178, 0.245, 0.288),
    'school_name_box': (0.262, 0.178, 0.430, 0.288),
    'exam_name_box': (0.438, 0.178, 0.600, 0.330),

    'candidate_grid': (0.636, 0.121, 0.962, 0.400),  # x1, y1, x2, y2
    'candidate_header_band': (0.121, 0.200),         # y1, y2 relative to image
    'candidate_bubble_top': 0.2094,
    'candidate_bubble_step': 0.01985,

    # Text sizes (approximate; auto-reduced when needed)
    'student_start_size': 28,
    'school_start_size': 26,
    'exam_start_size': 25,
    'header_digit_size': 18,

    'header_digit_y_offset': 22,
    'bubble_row_y_offsets': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}


def load_font(size: int, bold: bool = False):
    paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/dejavu/DejaVuSans.ttf',
        'C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def wrap_and_draw(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], start_size: int) -> None:
    x1, y1, x2, y2 = box
    words = text.split()
    max_width = x2 - x1
    max_height = y2 - y1
    for size in range(start_size, 11, -1):
        font = load_font(size)
        line_h = draw.textbbox((0, 0), 'Ag', font=font)[3]
        lines: list[str] = []
        current = ''
        for word in words:
            trial = word if not current else current + ' ' + word
            width = draw.textbbox((0, 0), trial, font=font)[2]
            if width <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        total_h = len(lines) * line_h + max(0, len(lines) - 1) * 4
        if total_h <= max_height:
            y = y1
            for line in lines:
                draw.text((x1, y), line, fill='black', font=font)
                y += line_h + 4
            return
    draw.text((x1, y1), text, fill='black', font=load_font(12))


def relative_box(rel_box: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = rel_box
    return round(x1 * w), round(y1 * h), round(x2 * w), round(y2 * h)


# ArUco dictionary and ID convention — must match template.json.
_ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
# Corner order: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
_ARUCO_CORNER_IDS = [0, 1, 2, 3]
# Reference marker centres as fractions of the processing canvas (666 x 515).
# These mirror referenceMarkerCenters in template.json.
_REF_CENTERS_RELATIVE = [
    (13.5 / 666, 13.2 / 515),   # TL
    (651.5 / 666, 13.2 / 515),  # TR
    (13.5 / 666, 499.0 / 515),  # BL
    (651.5 / 666, 499.0 / 515), # BR
]
# Marker size relative to the longer image dimension (landscape).
# On the 1426×1103 reference template, this gives ~43 px; on the 666×515
# processing canvas it gives ~24 px — both large enough for reliable ArUco
# detection with the 60 px white-border padding used by the scanner.
# Minimum is clamped to 24 px.
_ARUCO_MARKER_SIZE_RATIO = 0.03


def draw_aruco_corners(img: Image.Image) -> Image.Image:
    """Stamp 4 ArUco fiducial markers at the corners of *img* and return the
    modified image.  The markers use ``DICT_4X4_50`` with IDs 0-3 corresponding
    to top-left, top-right, bottom-left, bottom-right.

    If the underlying template already has printed square markers at the
    corners, a white rectangle is drawn first to erase them before the ArUco
    pattern is stamped, so old ink does not interfere with detection.
    """
    img_cv = np.array(img.convert("RGB"))[:, :, ::-1].copy()  # PIL→BGR
    h, w = img_cv.shape[:2]

    aruco_dict = cv2.aruco.getPredefinedDictionary(_ARUCO_DICT_ID)
    marker_px = max(24, int(max(w, h) * _ARUCO_MARKER_SIZE_RATIO))
    # ArUco requires a white quiet zone on all sides.
    quiet_zone = max(6, marker_px // 8)

    for corner_idx, marker_id in enumerate(_ARUCO_CORNER_IDS):
        rel_cx, rel_cy = _REF_CENTERS_RELATIVE[corner_idx]
        cx, cy = round(rel_cx * w), round(rel_cy * h)
        half = marker_px // 2
        # Clamp so the full marker fits and has quiet zone on all sides.
        x0 = max(quiet_zone, min(cx - half, w - marker_px - quiet_zone))
        y0 = max(quiet_zone, min(cy - half, h - marker_px - quiet_zone))
        x1 = x0 + marker_px
        y1 = y0 + marker_px
        if x1 > w or y1 > h or marker_px < 8:
            continue
        # Erase any existing printed marker in this region with white.
        img_cv[y0:y1, x0:x1] = 255
        # Generate ArUco marker at the required size.
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
        # Convert to BGR for placement in the colour image.
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        img_cv[y0:y1, x0:x1] = marker_bgr

    # BGR→RGB, then back to PIL
    return Image.fromarray(img_cv[:, :, ::-1])


def prefill_sheet(template_path: Path, student_name: str, school_name: str, exam_name: str, candidate_number: str) -> Image.Image:
    if len(candidate_number) != 10 or not candidate_number.isdigit():
        raise ValueError('Candidate number must be exactly 10 digits.')

    img = Image.open(template_path).convert('RGB')
    # Stamp ArUco corner markers (erases any pre-printed square markers first).
    img = draw_aruco_corners(img)
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Fill top fields
    wrap_and_draw(draw, student_name, relative_box(CFG['student_name_box'], w, h), CFG['student_start_size'])
    wrap_and_draw(draw, school_name, relative_box(CFG['school_name_box'], w, h), CFG['school_start_size'])
    wrap_and_draw(draw, exam_name, relative_box(CFG['exam_name_box'], w, h), CFG['exam_start_size'])

    # Candidate number grid
    gx1, gy1, gx2, gy2 = relative_box(CFG['candidate_grid'], w, h)
    grid_w = gx2 - gx1
    col_centers = [round(gx1 + (i + 0.5) * grid_w / 10) for i in range(10)]
    header_y = round(((CFG['candidate_header_band'][0] + CFG['candidate_header_band'][1]) / 2) * h)
    row_centers = [round((CFG['candidate_bubble_top'] + i * CFG['candidate_bubble_step']) * h) for i in range(10)]

    header_font = load_font(CFG['header_digit_size'], bold=True)
    radius = max(8, round(h * 0.009))
    for idx, digit in enumerate(candidate_number):
        cx = col_centers[idx]
        bbox = draw.textbbox((0, 0), digit, font=header_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (cx - tw / 2, header_y - th / 2 - 1 + CFG['header_digit_y_offset']),
            digit,
            fill='black',
            font=header_font
        )
        cy = row_centers[int(digit)]
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill='black')

    return img


def save_pdf(images: Iterable[Image.Image], output_path: Path) -> None:
    imgs = [im.convert('RGB') for im in images]
    if not imgs:
        raise ValueError('No images to save.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(output_path, format='PDF', save_all=True, append_images=imgs[1:], resolution=300.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prefill the current landscape multiple choice answer sheet template.')
    parser.add_argument('--template', type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument('--student-name', type=str)
    parser.add_argument('--school-name', type=str)
    parser.add_argument('--exam-name', type=str)
    parser.add_argument('--candidate-number', type=str)
    parser.add_argument('--output', type=Path, help='Single-sheet output PNG or PDF')
    parser.add_argument('--csv', type=Path, help='Batch CSV with columns: student_name,school_name,exam_name,candidate_number[,output_file]')
    parser.add_argument('--output-dir', type=Path, help='Directory for individual PNGs in batch mode')
    parser.add_argument('--combined-pdf', type=Path, help='Combined PDF in batch mode')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.csv:
        rows = list(csv.DictReader(args.csv.open('r', newline='', encoding='utf-8-sig')))
        images = []
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(rows, start=1):
            image = prefill_sheet(
                args.template,
                row['student_name'].strip(),
                row['school_name'].strip(),
                row['exam_name'].strip(),
                str(row['candidate_number']).strip(),
            )
            images.append(image)
            if args.output_dir:
                filename = (row.get('output_file') or f'sheet_{i:03d}.png').strip()
                image.save(args.output_dir / filename)
        if args.combined_pdf:
            save_pdf(images, args.combined_pdf)
            print(f'Created {args.combined_pdf}')
        elif args.output:
            save_pdf(images, args.output)
            print(f'Created {args.output}')
        else:
            raise SystemExit('For batch mode, provide --combined-pdf or --output.')
        return

    required = [args.student_name, args.school_name, args.exam_name, args.candidate_number, args.output]
    if any(v is None for v in required):
        raise SystemExit('For a single sheet, provide --student-name, --school-name, --exam-name, --candidate-number, and --output.')

    image = prefill_sheet(args.template, args.student_name, args.school_name, args.exam_name, args.candidate_number)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == '.pdf':
        save_pdf([image], args.output)
    else:
        image.save(args.output)
    print(f'Created {args.output}')


if __name__ == '__main__':
    main()
