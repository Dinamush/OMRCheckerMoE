"""One-case smoke test for the sweep harness."""
from __future__ import annotations

from run_sweep import (
    parse_results_csv,
    prepare_case,
    run_engine,
    score_case,
)
from generate_sheet import default_sweep_specs


def main() -> None:
    spec = default_sweep_specs()[1]  # size_10
    case = prepare_case(spec, darkness="dark_pencil", ground_truth="1234567890")
    print(f"Image:    {case.image_path}")
    print(f"Template: {case.template_path}")
    exit_code, output = run_engine(case)
    print(f"Engine exit code: {exit_code}")
    print("Engine output tail:")
    print(output[-1000:])
    row = parse_results_csv(case)
    print(f"\nParsed row: {row}")
    score = score_case(case, row)
    print(f"Score: {score}")


if __name__ == "__main__":
    main()
