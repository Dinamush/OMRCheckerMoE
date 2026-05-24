"""Unit tests for the student-fill feature (no FastAPI / no OMR engine).

Covers:
  * geometry math (answer_bubble_geometry)
  * answer parser (parse_answers) — every accepted format + edge cases
  * drawing primitives (draw_student_marks) — each profile is visible & deterministic
  * helper utilities (normalize_marking_profile, answers_summary, list_marking_profiles)
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from webui.services import student_fill


# ---------------------------------------------------------------------------
# answer_bubble_geometry
# ---------------------------------------------------------------------------
def test_answer_bubble_geometry_returns_100_bubbles_on_reference_canvas():
    bubbles = student_fill.answer_bubble_geometry(1426, 1103)
    assert len(bubbles) == student_fill.NUM_QUESTIONS * student_fill.NUM_OPTIONS == 100


def test_answer_bubble_geometry_covers_q1_through_q25():
    bubbles = student_fill.answer_bubble_geometry(1426, 1103)
    qs = sorted({b["q"] for b in bubbles})
    assert qs == list(range(1, 26))


def test_answer_bubble_geometry_has_four_options_per_question():
    bubbles = student_fill.answer_bubble_geometry(1426, 1103)
    by_q: dict[int, list[int]] = {}
    for b in bubbles:
        by_q.setdefault(b["q"], []).append(b["option"])
    for q, opts in by_q.items():
        assert sorted(opts) == [0, 1, 2, 3], f"q{q} missing options: {opts}"


def test_answer_bubble_geometry_centers_are_within_canvas():
    bubbles = student_fill.answer_bubble_geometry(666, 515)
    for b in bubbles:
        assert 0 <= b["cx"] < 666
        assert 0 <= b["cy"] < 515
        assert b["radius"] >= 2


def test_answer_bubble_geometry_scales_with_canvas_size():
    small = student_fill.answer_bubble_geometry(666, 515)
    large = student_fill.answer_bubble_geometry(1426, 1103)
    # The same bubble (q1, option A) should appear at roughly 2.14x further
    # right on the larger canvas than on the smaller one.
    small_q1a = next(b for b in small if b["q"] == 1 and b["option"] == 0)
    large_q1a = next(b for b in large if b["q"] == 1 and b["option"] == 0)
    scale = 1426 / 666
    assert abs(large_q1a["cx"] - round(small_q1a["cx"] * scale)) <= 2
    assert large_q1a["radius"] > small_q1a["radius"]


def test_answer_bubble_geometry_rejects_zero_size():
    with pytest.raises(ValueError):
        student_fill.answer_bubble_geometry(0, 100)
    with pytest.raises(ValueError):
        student_fill.answer_bubble_geometry(100, 0)


# ---------------------------------------------------------------------------
# parse_answers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, {}),
        ("", {}),
        ("blank", {}),
        ("none", {}),
    ],
)
def test_parse_answers_blank_inputs(value, expected):
    assert student_fill.parse_answers(value) == expected


def test_parse_answers_all_a():
    out = student_fill.parse_answers("all_a")
    assert len(out) == 25
    assert all(v == [0] for v in out.values())


def test_parse_answers_all_b_c_d():
    assert all(v == [1] for v in student_fill.parse_answers("all_b").values())
    assert all(v == [2] for v in student_fill.parse_answers("all_c").values())
    assert all(v == [3] for v in student_fill.parse_answers("all_d").values())


def test_parse_answers_alternating():
    out = student_fill.parse_answers("alternating")
    assert out[1] == [0]   # A
    assert out[2] == [1]   # B
    assert out[3] == [2]   # C
    assert out[4] == [3]   # D
    assert out[5] == [0]   # back to A


def test_parse_answers_25_letter_string():
    spec = "ABCDABCDABCDABCDABCDABCDA"
    out = student_fill.parse_answers(spec)
    assert len(out) == 25
    assert out[1] == [0]
    assert out[25] == [0]
    assert out[24] == [3]


def test_parse_answers_with_dashes_and_xes_for_skips():
    spec = "ABCD-ABCD-ABCD-ABCD-ABCDA"
    out = student_fill.parse_answers(spec)
    # Dashes are skips; questions 5, 10, 15, 20 should be unmarked.
    assert 5 not in out
    assert 10 not in out
    assert 15 not in out
    assert 20 not in out
    assert out[1] == [0]
    assert out[25] == [0]


def test_parse_answers_dict_with_q_keys():
    out = student_fill.parse_answers({"q1": "A", "q5": "B", "q25": "D"})
    assert out == {1: [0], 5: [1], 25: [3]}


def test_parse_answers_dict_with_numeric_keys():
    out = student_fill.parse_answers({"1": "A", "12": "BC"})
    assert out[1] == [0]
    assert out[12] == [1, 2]


def test_parse_answers_list_form():
    out = student_fill.parse_answers(["A", "B", "", "D", None, "C"])
    assert out == {1: [0], 2: [1], 4: [3], 6: [2]}


def test_parse_answers_json_dict():
    out = student_fill.parse_answers('{"q1": "A", "q2": "B", "q3": null}')
    assert out == {1: [0], 2: [1]}


def test_parse_answers_random_is_deterministic_with_seed():
    a = student_fill.parse_answers("random", seed=42)
    b = student_fill.parse_answers("random", seed=42)
    assert a == b
    c = student_fill.parse_answers("random", seed=43)
    assert a != c  # different seed → different answers


def test_parse_answers_random_with_skips_has_some_skipped():
    out = student_fill.parse_answers("random_with_skips", seed=1234)
    # Probabilistically expect ~10% blanks. Just assert it's not 25 and not 0.
    assert 0 < len(out) <= 25


def test_parse_answers_multi_marks():
    out = student_fill.parse_answers({"q1": "AB", "q2": "BCD"})
    assert out[1] == [0, 1]
    assert out[2] == [1, 2, 3]


def test_parse_answers_ignores_unknown_letters():
    out = student_fill.parse_answers({"q1": "E"})
    assert out == {}


def test_parse_answers_invalid_all_shortcut_raises():
    with pytest.raises(ValueError):
        student_fill.parse_answers("all_e")


def test_parse_answers_case_insensitive():
    out = student_fill.parse_answers("abCDa-bcdAB")
    assert out[1] == [0]
    assert out[3] == [2]
    assert out[5] == [0]  # 'a' at position 5
    # Position 6 is the dash → skip → next letter 'b' falls to position 7.
    assert 6 not in out
    assert out[7] == [1]


# ---------------------------------------------------------------------------
# normalize_marking_profile
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        ("none", "none"),
        ("Light Pencil", "light_pencil"),
        ("MEDIUM_PENCIL", "medium_pencil"),
        ("check-mark", "check_mark"),
        ("  pen ballpoint  ", "pen_ballpoint"),
    ],
)
def test_normalize_marking_profile_canonicalises(value, expected):
    assert student_fill.normalize_marking_profile(value) == expected


def test_normalize_marking_profile_rejects_unknown():
    with pytest.raises(ValueError):
        student_fill.normalize_marking_profile("disco_pencil")


def test_normalize_marking_profile_none_value_returns_none():
    assert student_fill.normalize_marking_profile(None) == "none"


# ---------------------------------------------------------------------------
# list_marking_profiles
# ---------------------------------------------------------------------------
def test_list_marking_profiles_has_expected_keys():
    profiles = student_fill.list_marking_profiles()
    ids = {p["id"] for p in profiles}
    for required in {
        "none",
        "light_pencil",
        "medium_pencil",
        "heavy_pencil",
        "pen_ballpoint",
        "check_mark",
        "cross_mark",
        "partial_fill",
        "messy_student",
        "careful_student",
    }:
        assert required in ids


def test_list_marking_profiles_entries_have_label_and_description():
    for p in student_fill.list_marking_profiles():
        assert p["label"]
        assert p["description"]


# ---------------------------------------------------------------------------
# draw_student_marks
# ---------------------------------------------------------------------------
def _white_canvas(w: int = 1426, h: int = 1103) -> Image.Image:
    return Image.new("RGB", (w, h), color=(255, 255, 255))


def _mean_pixel_diff(a: Image.Image, b: Image.Image) -> float:
    aa = np.array(a, dtype=np.int16)
    bb = np.array(b, dtype=np.int16)
    return float(np.abs(aa - bb).mean())


def test_draw_student_marks_noop_when_profile_is_none():
    img = _white_canvas()
    out = student_fill.draw_student_marks(img, answers={1: [0]}, marking_profile="none")
    assert out is img  # noop returns same instance


def test_draw_student_marks_noop_when_answers_empty():
    img = _white_canvas()
    out = student_fill.draw_student_marks(img, answers={}, marking_profile="medium_pencil")
    assert _mean_pixel_diff(img, out) == 0.0


@pytest.mark.parametrize(
    "profile",
    ["light_pencil", "medium_pencil", "heavy_pencil", "pen_ballpoint",
     "check_mark", "cross_mark", "partial_fill", "careful_student"],
)
def test_draw_student_marks_is_visible(profile):
    img = _white_canvas(666, 515)
    answers = {q: [q % 4] for q in range(1, 26)}
    out = student_fill.draw_student_marks(
        img, answers=answers, marking_profile=profile, candidate_number="9010690012"
    )
    assert out is not img
    diff = _mean_pixel_diff(img, out)
    assert diff > 0.05, f"profile={profile} produced too little change ({diff})"


def test_draw_student_marks_heavy_is_darker_than_light():
    img = _white_canvas(666, 515)
    answers = {q: [0] for q in range(1, 26)}
    light = student_fill.draw_student_marks(
        img, answers=answers, marking_profile="light_pencil", candidate_number="9010690012"
    )
    heavy = student_fill.draw_student_marks(
        img, answers=answers, marking_profile="heavy_pencil", candidate_number="9010690012"
    )
    light_mean = float(np.array(light).mean())
    heavy_mean = float(np.array(heavy).mean())
    # Lower mean = darker overall, since paper is white (255) and marks are dark.
    assert heavy_mean < light_mean


def test_draw_student_marks_is_deterministic_per_input():
    img = _white_canvas(666, 515)
    answers = {1: [0], 2: [1], 3: [2], 4: [3], 5: [0]}
    a = student_fill.draw_student_marks(
        img, answers=answers, marking_profile="medium_pencil", candidate_number="9010690012"
    )
    b = student_fill.draw_student_marks(
        img, answers=answers, marking_profile="medium_pencil", candidate_number="9010690012"
    )
    assert _mean_pixel_diff(a, b) == 0.0


def test_draw_student_marks_changes_with_candidate_number():
    img = _white_canvas(666, 515)
    answers = {q: [0] for q in range(1, 26)}
    a = student_fill.draw_student_marks(
        img, answers=answers, marking_profile="medium_pencil", candidate_number="9010690012"
    )
    b = student_fill.draw_student_marks(
        img, answers=answers, marking_profile="medium_pencil", candidate_number="9999999999"
    )
    # Different candidates → different RNG → different jitter pattern.
    assert _mean_pixel_diff(a, b) > 0.0


def test_draw_student_marks_explicit_seed_overrides_candidate_number():
    img = _white_canvas(666, 515)
    answers = {q: [0] for q in range(1, 26)}
    a = student_fill.draw_student_marks(
        img, answers=answers, marking_profile="medium_pencil",
        candidate_number="9010690012", seed=42,
    )
    b = student_fill.draw_student_marks(
        img, answers=answers, marking_profile="medium_pencil",
        candidate_number="9999999999", seed=42,
    )
    assert _mean_pixel_diff(a, b) == 0.0


def test_draw_student_marks_accepts_unparsed_answer_string():
    img = _white_canvas(666, 515)
    # Pass a raw 25-letter string instead of a dict — draw_student_marks
    # should fall back to parse_answers internally.
    out = student_fill.draw_student_marks(
        img,
        answers="ABCDABCDABCDABCDABCDABCDA",
        marking_profile="medium_pencil",
        candidate_number="9010690012",
    )
    assert _mean_pixel_diff(img, out) > 0.05


def test_draw_student_marks_marks_only_chosen_bubbles():
    """When q1=A is the only answer, only the q1A bubble area should change."""
    img = _white_canvas(666, 515)
    out = student_fill.draw_student_marks(
        img, answers={1: [0]}, marking_profile="heavy_pencil",
        candidate_number="9010690012",
    )
    arr_in = np.array(img)
    arr_out = np.array(out)
    diff = np.abs(arr_in.astype(np.int16) - arr_out.astype(np.int16)).sum(axis=2)
    # Find the bounding box of pixels that changed.
    ys, xs = np.where(diff > 5)
    assert len(xs) > 0, "no pixels changed at all"
    bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    # The q1-A bubble lives at OMR (~57.7, ~264.3) on the 666x515 canvas
    # (calibrated against the actual printed reference template). Allow
    # generous padding for jitter and partial overlap.
    target_cx, target_cy = 58, 264
    assert abs((bbox[0] + bbox[2]) / 2 - target_cx) < 20
    assert abs((bbox[1] + bbox[3]) / 2 - target_cy) < 20


def test_draw_student_marks_handles_multi_marks_per_question():
    img = _white_canvas(666, 515)
    single = student_fill.draw_student_marks(
        img, answers={1: [0]}, marking_profile="heavy_pencil",
        candidate_number="9010690012",
    )
    multi = student_fill.draw_student_marks(
        img, answers={1: [0, 1, 2, 3]}, marking_profile="heavy_pencil",
        candidate_number="9010690012",
    )
    # Multi-mark must produce noticeably more darkened pixels than single.
    assert _mean_pixel_diff(img, multi) > _mean_pixel_diff(img, single) * 2


def test_draw_student_marks_invalid_profile_raises():
    img = _white_canvas(666, 515)
    with pytest.raises(ValueError):
        student_fill.draw_student_marks(
            img, answers={1: [0]}, marking_profile="invalid_disco_pen",
        )


def test_draw_student_marks_rejects_none_image():
    with pytest.raises(ValueError):
        student_fill.draw_student_marks(None, answers={1: [0]}, marking_profile="medium_pencil")


# ---------------------------------------------------------------------------
# answers_summary
# ---------------------------------------------------------------------------
def test_answers_summary_blank():
    assert student_fill.answers_summary({}) == "-" * 25


def test_answers_summary_single_answers():
    out = student_fill.answers_summary({1: [0], 2: [1], 25: [3]})
    assert out[0] == "A"
    assert out[1] == "B"
    assert out[24] == "D"
    assert out[2] == "-"


def test_answers_summary_multi_marks():
    out = student_fill.answers_summary({1: [0, 1], 2: [2, 3]})
    assert out.startswith("ABCD")
