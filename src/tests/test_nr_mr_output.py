import pytest

from src.core import select_question_response


@pytest.mark.parametrize(
    "marked,empty,delta,expected,is_multi",
    [
        ([], "NR", 0.06, "NR", False),
        ([("A", 10.0)], "NR", 0.06, "A", False),
        ([("B", 10.0), ("A", 12.0)], "NR", 0.06, "MR(AB)", True),
        ([("A", 10.0), ("B", 40.0)], "NR", 0.06, "A", False),
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

