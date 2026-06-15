"""Manifest-driven regression matrix for historically fragile OMR scenarios.

This test suite keeps a curated list of real artifacts that previously broke
and now must stay green. Cases are defined in
``src/tests/test_samples/regression_matrix_cases.json``.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
from freezegun import freeze_time

from main import entry_point_for_args

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    Path(__file__).parent / "test_samples" / "regression_matrix_cases.json"
)


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _enabled_cases() -> list[dict]:
    return [case for case in _load_manifest().get("cases", []) if case.get("enabled")]


def _required_categories() -> set[str]:
    return {
        "marker_occlusion_one_corner",
        "degraded_three_marker_skew",
        "historical_candidate_misread_fixed",
        "folded_edge_partial_marker",
        "hard_batch_known_good",
    }


def _case_context(case: dict) -> str:
    story = case.get("regression_story", {})
    return (
        f"[what_went_wrong={story.get('what_went_wrong', 'n/a')}; "
        f"guardrail={story.get('guardrail', 'n/a')}]"
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _run_case(case: dict, tmp_path: Path) -> tuple[str, dict[str, str] | None]:
    template_path = REPO_ROOT / case["template"]
    config_path = REPO_ROOT / case["config"]
    image_path = REPO_ROOT / case["image"]

    missing = [
        str(path)
        for path in (template_path, config_path, image_path)
        if not path.exists()
    ]
    if missing:
        if case.get("optional", False):
            pytest.skip(
                f"optional regression artifact missing for {case['id']}: "
                + ", ".join(missing)
            )
        raise AssertionError(
            f"required regression artifact missing for {case['id']}: "
            + ", ".join(missing)
            + " "
            + _case_context(case)
        )

    inputs = tmp_path / case["id"] / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    shutil.copy(template_path, inputs / "template.json")
    shutil.copy(config_path, inputs / "config.json")
    shutil.copy(image_path, inputs / image_path.name)

    output_dir = tmp_path / case["id"] / "out"
    with freeze_time("1970-01-01"):
        entry_point_for_args(
            {
                "input_paths": [str(inputs)],
                "output_dir": str(output_dir),
                "debug": False,
                "autoAlign": False,
                "setLayout": False,
                "silent": True,
            }
        )

    results_files = sorted(output_dir.rglob("Results_*.csv"))
    assert results_files, f"no Results CSV produced under {output_dir}"
    result_rows = _read_csv_rows(results_files[0])

    error_rows = _read_csv_rows(output_dir / "Manual" / "ErrorFiles.csv")
    expected_error = bool(case.get("expect_error", False))
    file_id = case.get("error_file_id", image_path.name)

    if expected_error:
        assert not result_rows, (
            f"regression case {case['id']} expected no Results row, "
            f"but got {len(result_rows)}"
            f" {_case_context(case)}"
        )
        assert any(row.get("file_id") == file_id for row in error_rows), (
            f"regression case {case['id']} expected ErrorFiles.csv entry "
            f"for {file_id!r}"
            f" {_case_context(case)}"
        )
        return "error", None

    assert result_rows, "Results CSV had no data rows"
    if case.get("forbid_error_file", True):
        assert not any(row.get("file_id") == file_id for row in error_rows), (
            f"regression case {case['id']} unexpectedly produced ErrorFiles.csv "
            f"entry for {file_id!r}"
            f" {_case_context(case)}"
        )
    return "ok", result_rows[0]


@pytest.mark.parametrize(
    "case",
    _enabled_cases(),
    ids=lambda c: c["id"],
)
def test_regression_matrix_case(case: dict, tmp_path: Path) -> None:
    status, row = _run_case(case, tmp_path)
    if case.get("expect_error", False):
        assert status == "error"
        return

    assert row is not None
    expected = case.get("expected", {})

    for field, expected_value in expected.items():
        if field == "must_not_contain":
            continue
        assert row.get(field) == expected_value, (
            f"regression case {case['id']} failed for field {field}: "
            f"got {row.get(field)!r}, expected {expected_value!r}"
            f" {_case_context(case)}"
        )

    forbidden = expected.get("must_not_contain", [])
    candidate = row.get("CandidateNumber", "")
    for marker in forbidden:
        assert marker not in candidate, (
            f"regression case {case['id']} candidate number unexpectedly "
            f"contains {marker!r}: {candidate!r}"
            f" {_case_context(case)}"
        )


def test_regression_manifest_sanity() -> None:
    manifest = _load_manifest()
    case_type_docs = manifest.get("case_type_docs", {})
    assert case_type_docs, "manifest must define case_type_docs"

    for category, doc in case_type_docs.items():
        assert category, "case_type_docs keys must be non-empty"
        assert doc.get("what_went_wrong"), (
            f"case_type_docs[{category!r}] missing what_went_wrong"
        )
        assert doc.get("regression_risk"), (
            f"case_type_docs[{category!r}] missing regression_risk"
        )
        assert doc.get("test_oracle"), (
            f"case_type_docs[{category!r}] missing test_oracle"
        )

    cases = manifest.get("cases", [])
    assert cases, "regression manifest must define at least one case"

    ids = [case.get("id") for case in cases]
    assert all(ids), "every regression case must have a non-empty id"
    assert len(set(ids)) == len(ids), "regression case ids must be unique"

    enabled_categories = {
        case.get("category") for case in cases if case.get("enabled", False)
    }
    missing = _required_categories() - enabled_categories
    assert not missing, (
        "regression matrix is missing required scenario categories: "
        f"{sorted(missing)}"
    )

    for case in cases:
        category = case.get("category")
        assert category, f"regression case {case.get('id')} missing category"
        assert category in case_type_docs, (
            f"regression case {case.get('id')} references undocumented "
            f"category {category!r}"
        )
        if case.get("enabled", False):
            story = case.get("regression_story", {})
            assert story.get("what_went_wrong"), (
                f"enabled case {case.get('id')} missing regression_story.what_went_wrong"
            )
            assert story.get("why_this_case_exists"), (
                f"enabled case {case.get('id')} missing regression_story.why_this_case_exists"
            )
            assert story.get("guardrail"), (
                f"enabled case {case.get('id')} missing regression_story.guardrail"
            )

    backlog = manifest.get("backlog", [])
    assert backlog, "maintain backlog entries so missing scenarios stay visible"
    for entry in backlog:
        assert entry.get("id"), "backlog entry must have id"
        assert entry.get("category"), "backlog entry must have category"
        assert entry.get("status"), "backlog entry must have status"
        assert entry.get("why"), "backlog entry must document why"