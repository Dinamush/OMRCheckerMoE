import pytest

from src.core import (
    choose_uniform_vertical_shift,
    resolve_marked_options,
    select_question_response,
)


@pytest.mark.parametrize(
    "marked,empty,delta,expected,is_multi",
    [
        ([], "NR", 0.06, "NR", False),
        ([("A", 10.0)], "NR", 0.06, "A", False),
        ([("B", 10.0), ("A", 12.0)], "NR", 0.06, "MR(AB)", True),
        ([("A", 10.0), ("B", 40.0)], "NR", 0.06, "A", False),
        # Integer/candidate-number digit columns are single-select too: a
        # faint second digit (well outside the equal-mark delta) must be
        # dropped in favour of the single darkest digit, never concatenated.
        ([("9", 12.0), ("8", 70.0)], "", 0.06, "9", False),
        ([("0", 15.0)], "", 0.06, "0", False),
        # A genuine, equally-dark double-mark in one digit column is flagged
        # for manual review instead of being silently appended.
        ([("8", 11.0), ("9", 13.0)], "", 0.06, "MR(89)", True),
        ([], "", 0.06, "", False),
    ],
)
def test_select_question_response(marked, empty, delta, expected, is_multi):
    response, multi = select_question_response(
        marked_options=marked,
        empty_value=empty,
        multi_mark_equal_delta=delta,
    )
    assert response == expected
    assert multi is is_multi


def test_resolve_marked_options_excludes_non_winning_ghost_mark() -> None:
    """Checked overlays should annotate only the final winning option."""
    response, is_multi, selected = resolve_marked_options(
        marked_options=[("5", 10.0), ("4", 35.0)],
        empty_value="",
        multi_mark_equal_delta=0.06,
    )
    assert response == "5"
    assert is_multi is False
    assert selected == {"5"}


def test_resolve_marked_options_marks_only_true_mr_tie() -> None:
    """For genuine ties, overlay selection must contain all tied options."""
    response, is_multi, selected = resolve_marked_options(
        marked_options=[("7", 10.0), ("8", 12.0), ("5", 40.0)],
        empty_value="",
        multi_mark_equal_delta=0.06,
    )
    assert response == "MR(78)"
    assert is_multi is True
    assert selected == {"7", "8"}


def test_choose_uniform_vertical_shift_strict_consensus() -> None:
    assert choose_uniform_vertical_shift([2, 2, 3, 2, 1]) == 2


def test_choose_uniform_vertical_shift_majority_with_outliers() -> None:
    # Reproduces degraded-corner behavior where most strips agree on a
    # positive shift but a few outlier strips vote negative.
    assert choose_uniform_vertical_shift([8, 5, 8, 5, 8, 8, -8, 6, -7, 8]) == 8


def test_choose_uniform_vertical_shift_same_sign_cluster_without_mode_majority() -> None:
    # Folded-edge / partial-marker scans can widen the per-strip vote spread
    # while still keeping almost every strip aligned to the same upward shift.
    # No single dy wins an absolute majority here, but the dominant-sign median
    # still captures the systematic correction.
    assert choose_uniform_vertical_shift([8, 3, 7, 4, 4, 8, 7, 4, 5, 5]) == 5


def test_choose_uniform_vertical_shift_no_clear_majority() -> None:
    assert choose_uniform_vertical_shift([3, -3, 2, -2, 1, -1]) == 0

