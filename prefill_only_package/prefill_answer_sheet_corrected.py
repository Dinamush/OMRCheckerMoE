#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

DEFAULT_TEMPLATE = Path('blank_template_reference.png')

# Relative coordinates for the current landscape blank template.
CFG = {
    'student_name_box': (0.040, 0.178, 0.245, 0.288),
    'school_name_box': (0.262, 0.178, 0.430, 0.288),
    'exam_name_box': (0.438, 0.178, 0.600, 0.330),

    'candidate_grid': (0.636, 0.121, 0.962, 0.400),  # x1, y1, x2, y2
    'candidate_header_band': (0.121, 0.200),         # y1, y2 relative to image
    'candidate_bubble_top': 0.215,
    'candidate_bubble_step': 0.0192,

    # Text sizes (approximate; auto-reduced when needed)
    'student_start_size': 28,
    'school_start_size': 26,
    'exam_start_size': 25,
    'header_digit_size': 18,

    'header_digit_y_offset': 22,
    'bubble_row_y_offsets': [-6, -5, -5, 0, 0, 0, -2, 0, 0, 0],
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


def prefill_sheet(template_path: Path, student_name: str, school_name: str, exam_name: str, candidate_number: str) -> Image.Image:
    if len(candidate_number) != 10 or not candidate_number.isdigit():
        raise ValueError('Candidate number must be exactly 10 digits.')

    img = Image.open(template_path).convert('RGB')
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
        digit_index = int(digit)
        cy = row_centers[digit_index] + CFG['bubble_row_y_offsets'][digit_index]
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
