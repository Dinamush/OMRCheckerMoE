"""End-to-end OMR roundtrip tests for the student-fill feature.

These tests generate a prefilled sheet with student-style answer marks,
run it through the real OMR engine (``entry_point_for_image``), and assert
that the output CSV matches the known ground-truth answer key.

This is the first test in the repo that exercises the full pipeline:
  prefill → student fill → CropOnMarkers → bubble read → CSV

Test cases cover:
  * all_a / all_b / all_c / all_d (uniform answers)
  * alternating (ABCDA…)
  * random_with_skips (some NR)
  * each marking profile (heavy_pencil, check_mark, partial_fill, etc.)
  * combined with realism preset (subtle scan simulation)
  * multi-mark per question (AB on q1)
  * blank sheet (no answers) → all NR
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path

import pytest

from webui.services import prefill as prefill_service
from webui.services.prefill import PREFILL_NUM_QUESTIONS as NUM_QUESTIONS
from webui.services.student_fill import parse_answers

# ---------------------------------------------------------------------------
# Repo / fixture paths
# ---------------------------------------------------------------------------
# Prefill defaults to the July 2026 Letter landscape SMQ60 (60Q) sheet, so the
# roundtrip harness must scan against that template — not the legacy 25Q one.
REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_DIR = REPO_ROOT / "MoE-July-2026-Letter-Landscape-SMQ60-0"
SAMPLE_TEMPLATE = CUSTOM_DIR / "template.json"

_CONFIG_PAYLOAD = {
    "dimensions": {
        "processing_height": 510,
        "processing_width": 660,
        "display_height": 510,
        "display_width": 660,
    },
    # The Letter SMQ60 sheet ships with OVERSAMPLE_SCALE=2.0 in its config.json;
    # the ArUco warp-confidence gate needs it to clear the alignment threshold.
    "threshold_params": {"OVERSAMPLE_SCALE": 2.0},
    "outputs": {"show_image_level": 0},
}


def _template_payload() -> dict:
    return json.loads(SAMPLE_TEMPLATE.read_text(encoding="utf-8"))


def _setup_template_dir(tmp_path: Path) -> Path:
    import shutil
    tdir = tmp_path / "template_base"
    tdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SAMPLE_TEMPLATE, tdir / "template.json")
    return tdir


def _generate_filled_png(
    *,
    answers: str,
    marking_profile: str = "heavy_pencil",
    realism_preset: str = "none",
    candidate_number: str = "9010690012",
) -> bytes:
    return prefill_service.generate_single_png(
        "Roundtrip Student",
        "Roundtrip School",
        "Roundtrip Exam",
        candidate_number,
        realism_preset=realism_preset,
        marking_profile=marking_profile,
        answers=answers,
    )


def _run_omr_on_png(png_bytes: bytes, tmp_path: Path) -> dict[str, str]:
    """Save PNG to disk, run OMR, return the first results row as a dict."""
    from src.entry import entry_point_for_image  # noqa: PLC0415

    tmp_path.mkdir(parents=True, exist_ok=True)
    image_path = tmp_path / "sheet.png"
    image_path.write_bytes(png_bytes)
    template_dir = _setup_template_dir(tmp_path)
    output_dir = tmp_path / "omr_out"
    output_dir.mkdir(parents=True, exist_ok=True)

    entry_point_for_image(
        image_path=str(image_path),
        output_dir=str(output_dir),
        template_payload=_template_payload(),
        config_payload=_CONFIG_PAYLOAD,
        template_dir=str(template_dir),
        rotation_degrees=0,
    )

    results_dir = output_dir / "Results"
    csv_files = list(results_dir.glob("Results_*.csv")) if results_dir.exists() else []
    if not csv_files:
        error_dir = output_dir / "Manual" / "ErrorFiles"
        err_files = list(error_dir.iterdir()) if error_dir.exists() else []
        raise AssertionError(
            f"OMR produced no Results CSV. ErrorFiles: {err_files}. "
            f"output_dir: {list(output_dir.rglob('*'))}"
        )

    with csv_files[0].open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "Results CSV is empty"
    return rows[0]


def _expected_answer_letter(parsed: dict[int, list[int]], q: int) -> str:
    opts = parsed.get(q, [])
    if not opts:
        return "NR"
    return "".join(chr(ord("A") + o) for o in opts)


def _normalise_omr_response(raw: str) -> str:
    """Strip MR(...) wrapper the engine uses for ambiguous multi-marks."""
    text = (raw or "NR").strip()
    if text.startswith("MR(") and text.endswith(")"):
        return text[3:-1]
    return text


def _response_matches_expected(actual_raw: str, expected: str, *, allow_multi: bool) -> bool:
    actual = _normalise_omr_response(actual_raw)
    if actual == expected:
        return True
    if expected == "NR":
        return actual in ("", "NR")
    if allow_multi and expected and expected[0] in actual:
        return True
    return False


def _assert_answers_match(
    omr_row: dict[str, str],
    answers_spec: str,
    *,
    allow_multi: bool = False,
    skip_questions: set[int] | None = None,
    min_correct: int | None = None,
) -> None:
    """Assert that the OMR row matches the expected answer key.

    When ``allow_multi`` is True, responses like ``MR(BD)`` count as a match
    for expected ``B`` as long as the primary letter is present.

    ``min_correct`` overrides the default all-or-nothing check and instead
    requires at least that many questions to match (useful for faint profiles).
    """
    parsed = parse_answers(answers_spec, num_questions=NUM_QUESTIONS)
    skip = skip_questions or set()
    mismatches: list[str] = []
    correct = 0
    checked = 0
    for q in range(1, NUM_QUESTIONS + 1):
        if q in skip:
            continue
        if q not in parsed:
            continue
        checked += 1
        expected = _expected_answer_letter(parsed, q)
        actual_raw = omr_row.get(f"q{q}", "NR")
        if _response_matches_expected(actual_raw, expected, allow_multi=allow_multi):
            correct += 1
        else:
            mismatches.append(
                f"q{q}: expected {expected!r}, got {_normalise_omr_response(actual_raw)!r}"
            )
    if min_correct is not None:
        assert correct >= min_correct, (
            f"Only {correct}/{checked} matched (need ≥{min_correct}). "
            f"First mismatches: {mismatches[:5]}"
        )
        return
    assert not mismatches, (
        f"Answer mismatches ({len(mismatches)}/{checked}):\n"
        + "\n".join(mismatches[:10])
        + (f"\n… and {len(mismatches) - 10} more" if len(mismatches) > 10 else "")
    )


# ---------------------------------------------------------------------------
# Core roundtrip tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("answers_spec", ["all_a", "all_b", "all_c", "all_d"])
def test_omr_roundtrip_uniform_answers(answers_spec: str, tmp_path: Path) -> None:
    # pen_ballpoint: dark, crisp, zero multi_mark_prob → reliable 25/25 reads.
    png = _generate_filled_png(answers=answers_spec, marking_profile="pen_ballpoint")
    row = _run_omr_on_png(png, tmp_path)
    _assert_answers_match(row, answers_spec, allow_multi=True)


def test_omr_roundtrip_alternating(tmp_path: Path) -> None:
    png = _generate_filled_png(answers="alternating", marking_profile="pen_ballpoint")
    row = _run_omr_on_png(png, tmp_path)
    _assert_answers_match(row, "alternating", allow_multi=True)


def test_omr_roundtrip_blank_sheet_all_nr(tmp_path: Path) -> None:
    """A sheet with no student marks should produce NR for every question."""
    png = prefill_service.generate_single_png(
        "Blank Student", "School", "Exam", "9010690012",
        realism_preset="none", marking_profile="none",
    )
    row = _run_omr_on_png(png, tmp_path)
    for q in range(1, NUM_QUESTIONS + 1):
        val = (row.get(f"q{q}") or "NR").strip()
        assert val == "NR", f"q{q} expected NR, got {val!r}"


def test_omr_roundtrip_with_skips(tmp_path: Path) -> None:
    """Non-skipped questions must match; skipped rows may bleed from neighbours."""
    answers_spec = "ABCD-ABCD-ABCD-ABCD-ABCDA"
    png = _generate_filled_png(answers=answers_spec, marking_profile="pen_ballpoint")
    row = _run_omr_on_png(png, tmp_path)
    parsed = parse_answers(answers_spec, num_questions=NUM_QUESTIONS)
    skip_qs = {q for q in range(1, NUM_QUESTIONS + 1) if q not in parsed}
    for q in range(1, NUM_QUESTIONS + 1):
        if q in skip_qs:
            continue
        expected = _expected_answer_letter(parsed, q)
        actual_raw = row.get(f"q{q}", "NR")
        assert _response_matches_expected(actual_raw, expected, allow_multi=True), (
            f"q{q}: expected {expected!r}, got {_normalise_omr_response(actual_raw)!r}"
        )


def test_omr_roundtrip_candidate_number_preserved(tmp_path: Path) -> None:
    candidate = "9010690012"
    png = _generate_filled_png(
        answers="all_a", marking_profile="heavy_pencil", candidate_number=candidate,
    )
    row = _run_omr_on_png(png, tmp_path)
    cand = (row.get("CandidateNumber") or "").strip()
    assert cand == candidate, f"CandidateNumber mismatch: {cand!r} != {candidate!r}"


@pytest.mark.parametrize(
    "profile",
    ["heavy_pencil", "pen_ballpoint", "careful_student"],
)
def test_omr_roundtrip_dark_profiles(profile: str, tmp_path: Path) -> None:
    """Dark, full-fill profiles should be reliably OMR-readable."""
    png = _generate_filled_png(answers="all_a", marking_profile=profile)
    row = _run_omr_on_png(png, tmp_path)
    _assert_answers_match(row, "all_a")


def test_omr_roundtrip_with_subtle_realism(tmp_path: Path) -> None:
    """Student fill + subtle scan simulation should still be OMR-readable."""
    png = _generate_filled_png(
        answers="all_a",
        marking_profile="pen_ballpoint",
        realism_preset="subtle",
    )
    row = _run_omr_on_png(png, tmp_path)
    _assert_answers_match(row, "all_a", allow_multi=True, min_correct=20)


def test_omr_roundtrip_light_pencil_below_omr_threshold(tmp_path: Path) -> None:
    """Light pencil marks are intentionally below the OMR detection threshold.

    The marks are visible to the human eye but too faint for the integral
    bubble reader. This test documents that boundary — pipeline must complete
    without error even when no answers are read.
    """
    png = _generate_filled_png(answers="all_a", marking_profile="light_pencil")
    row = _run_omr_on_png(png, tmp_path)
    assert row.get("CandidateNumber", "").strip() == "9010690012"


def test_omr_roundtrip_multi_mark_detected(tmp_path: Path) -> None:
    """When q1 is marked AB, OMR should detect at least one of the marks."""
    answers_json = json.dumps({"q1": "AB", "q2": "C"})
    png = _generate_filled_png(
        answers=answers_json, marking_profile="pen_ballpoint",
    )
    row = _run_omr_on_png(png, tmp_path)
    q1 = _normalise_omr_response(row.get("q1", "NR"))
    q2 = _normalise_omr_response(row.get("q2", "NR"))
    # Both A and B were drawn; OMR may report one or both depending on fill overlap.
    assert q1 not in ("", "NR") and ("A" in q1 or "B" in q1), f"q1 not detected: {q1!r}"
    assert "C" in q2, f"q2 expected C, got {q2!r}"


def test_omr_roundtrip_random_with_skips_has_nrs(tmp_path: Path) -> None:
    """random_with_skips should produce at least one NR in the output."""
    png = _generate_filled_png(
        answers="random_with_skips",
        marking_profile="heavy_pencil",
        candidate_number="5555555555",
    )
    row = _run_omr_on_png(png, tmp_path)
    nr_count = sum(
        1 for q in range(1, NUM_QUESTIONS + 1)
        if (row.get(f"q{q}") or "NR").strip() == "NR"
    )
    assert nr_count >= 1, "Expected at least one NR from random_with_skips"


def test_omr_roundtrip_check_mark_profile_processes(tmp_path: Path) -> None:
    """Check marks draw visible strokes but OMR bubble-read may not detect them.

    This documents a known boundary: check/cross styles are useful for visual
    realism and adversarial testing but are not guaranteed OMR-readable without
    engine tuning. The test only asserts the pipeline completes.
    """
    png = _generate_filled_png(answers="all_a", marking_profile="check_mark")
    row = _run_omr_on_png(png, tmp_path)
    assert row.get("CandidateNumber", "").strip() == "9010690012"


def test_omr_roundtrip_partial_fill_profile_processes(tmp_path: Path) -> None:
    """Partial arc fills may fall below OMR detection threshold — pipeline must complete."""
    png = _generate_filled_png(answers="all_a", marking_profile="partial_fill")
    row = _run_omr_on_png(png, tmp_path)
    assert row.get("CandidateNumber", "").strip() == "9010690012"


def test_omr_roundtrip_messy_student_still_processes(tmp_path: Path) -> None:
    """Messy student profile should not crash the OMR engine."""
    png = _generate_filled_png(answers="random", marking_profile="messy_student")
    row = _run_omr_on_png(png, tmp_path)
    # Just verify we got a row back with some non-NR answers.
    non_nr = sum(
        1 for q in range(1, NUM_QUESTIONS + 1)
        if (row.get(f"q{q}") or "NR").strip() != "NR"
    )
    assert non_nr >= 1, "Expected at least one non-NR answer from messy_student sheet"


def test_omr_roundtrip_deterministic_across_two_runs(tmp_path: Path) -> None:
    """Two identical generate+OMR runs should produce identical CSV rows."""
    png1 = _generate_filled_png(answers="alternating", marking_profile="heavy_pencil")
    png2 = _generate_filled_png(answers="alternating", marking_profile="heavy_pencil")
    assert png1 == png2, "Prefill output must be deterministic"

    row1 = _run_omr_on_png(png1, tmp_path / "run1")
    row2 = _run_omr_on_png(png2, tmp_path / "run2")
    for q in range(1, NUM_QUESTIONS + 1):
        assert row1.get(f"q{q}") == row2.get(f"q{q}"), (
            f"q{q} differs between runs: {row1.get(f'q{q}')!r} vs {row2.get(f'q{q}')!r}"
        )
