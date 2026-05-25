"""Geometry regression tests for the legacy portrait_25q sheet.

Each test pins down one constraint from ``portrait_25q/DESIGN.md`` that
has previously regressed (or is at risk of regressing) when constants
are tuned. Keeping them as arithmetic invariants -- rather than pixel
diffs of the rendered PNG -- means refactoring the renderer is safe as
long as the geometry stays correct.
"""

from __future__ import annotations

import json
from pathlib import Path

from portrait_25q import generate_blank


def test_answer_grid_respects_bottom_marker_quiet_zone() -> None:
    """The lowest answer bubble must stay above the marker quiet-zone limit."""
    bottom_marker_top = (
        generate_blank.ARUCO_CENTRES[2][1] - generate_blank.ARUCO_MARKER_SIZE_OMR / 2
    )
    content_limit_y = bottom_marker_top - 5
    lowest_left_center_y = (
        generate_blank.ANS_BLOCK_LEFT_ORIGIN[1] + 12 * generate_blank.ANS_LABELS_GAP_Y
    )
    lowest_right_center_y = (
        generate_blank.ANS_BLOCK_RIGHT_ORIGIN[1] + 11 * generate_blank.ANS_LABELS_GAP_Y
    )

    assert lowest_left_center_y + generate_blank.ANS_BUBBLE_DIAM / 2 <= content_limit_y
    assert lowest_right_center_y + generate_blank.ANS_BUBBLE_DIAM / 2 <= content_limit_y


def test_answer_grid_template_gap_matches_generator() -> None:
    """The static template must stay in sync with the rendered row pitch."""
    template_path = Path(generate_blank.__file__).with_name("template.json")
    template = json.loads(template_path.read_text(encoding="utf-8"))

    assert (
        template["fieldBlocks"]["q01_q13_block"]["labelsGap"]
        == generate_blank.ANS_LABELS_GAP_Y
    )
    assert (
        template["fieldBlocks"]["q14_q25_block"]["labelsGap"]
        == generate_blank.ANS_LABELS_GAP_Y
    )


def test_column_divider_reaches_bottom_of_last_answer_bubble() -> None:
    """The grey column divider must span the full height of the q13 bubble."""
    expected_div_bot_omr = (
        generate_blank.ANS_BLOCK_LEFT_ORIGIN[1]
        + 12 * generate_blank.ANS_LABELS_GAP_Y
        + generate_blank.ANS_BUBBLE_DIAM / 2
    )
    lowest_left_bubble_bottom_omr = (
        generate_blank.ANS_BLOCK_LEFT_ORIGIN[1]
        + 12 * generate_blank.ANS_LABELS_GAP_Y
        + generate_blank.ANS_BUBBLE_DIAM / 2
    )

    assert expected_div_bot_omr == lowest_left_bubble_bottom_omr

    bottom_marker_top_omr = (
        generate_blank.ARUCO_CENTRES[2][1] - generate_blank.ARUCO_MARKER_SIZE_OMR / 2
    )
    content_limit_y = bottom_marker_top_omr - 5
    assert expected_div_bot_omr <= content_limit_y


def test_header_underline_clears_label_bbox() -> None:
    """The header underline must sit fully below the rendered text bbox.

    Using ``anchor="lm"`` (left, middle) places the text bbox centred on
    ``y_input``, so the bbox spans ``[y - FONT/2, y + FONT/2]``. The
    underline must therefore be at least ``FONT/2`` print-px below
    ``y_input`` -- anything less lands inside the text and clips
    descenders or kisses the visible baseline.
    """
    underline_offset = (
        generate_blank.FONT_HEADER_PX / 2
        + generate_blank.HEADER_UNDERLINE_MARGIN_PX
    )
    assert underline_offset >= generate_blank.FONT_HEADER_PX / 2 + 1

    next_row_gap_print_px = 22 * generate_blank.SCALE_Y
    next_row_text_top_offset = (
        next_row_gap_print_px - generate_blank.FONT_HEADER_PX / 2
    )
    assert underline_offset + generate_blank.HEADER_UNDERLINE_PX < next_row_text_top_offset
