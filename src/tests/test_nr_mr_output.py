import pytest

from src.core import select_question_response


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

