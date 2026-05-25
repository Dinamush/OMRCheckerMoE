"""Geometry regression tests for the legacy portrait_25q sheet."""

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
