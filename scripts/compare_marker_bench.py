"""Print a before/after comparison table from benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_scenarios(before: dict, after: dict) -> None:
    before_map = {s["scenario"]: s for s in before["scenarios"]}
    after_map = {s["scenario"]: s for s in after["scenarios"]}
    scenarios = sorted(set(before_map) | set(after_map))
    print(f"\n{'Scenario':<22} {'Before':>8} {'After':>8} {'Delta':>10} {'Before ms':>10} {'After ms':>10}")
    print("-" * 72)
    for name in scenarios:
        b = before_map.get(name, {})
        a = after_map.get(name, {})
        b_pct = b.get("success_pct", 0.0)
        a_pct = a.get("success_pct", 0.0)
        delta = a_pct - b_pct
        b_ms = b.get("median_ms", "-")
        a_ms = a.get("median_ms", "-")
        sign = "+" if delta > 0 else ""
        print(
            f"{name:<22} {b_pct:>7.1f}% {a_pct:>7.1f}% {sign}{delta:>9.1f}% "
            f"{str(b_ms):>10} {str(a_ms):>10}"
        )


def compare_realism(before: dict, after: dict) -> None:
    before_map = {p["preset"]: p for p in before["presets"]}
    after_map = {p["preset"]: p for p in after["presets"]}
    print(f"\n{'Preset':<14} {'Before':>8} {'After':>8} {'Delta':>10} {'Before ms':>10} {'After ms':>10}")
    print("-" * 64)
    for preset in ["none", "subtle", "moderate", "adversarial"]:
        b = before_map.get(preset, {})
        a = after_map.get(preset, {})
        b_pct = b.get("success_pct", 0.0)
        a_pct = a.get("success_pct", 0.0)
        delta = a_pct - b_pct
        sign = "+" if delta > 0 else ""
        print(
            f"{preset:<14} {b_pct:>7.1f}% {a_pct:>7.1f}% {sign}{delta:>9.1f}% "
            f"{str(b.get('median_ms','-')):>10} {str(a.get('median_ms','-')):>10}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-occlusion", type=Path, default=Path("bench_baseline.json"))
    parser.add_argument("--after-occlusion", type=Path, default=Path("bench_refine.json"))
    parser.add_argument("--before-realism", type=Path, default=Path("bench_realism_baseline.json"))
    parser.add_argument("--after-realism", type=Path, default=Path("bench_realism_refine.json"))
    args = parser.parse_args()

    print("=== Synthetic occlusion benchmark (30 sheets/scenario) ===")
    compare_scenarios(load(args.before_occlusion), load(args.after_occlusion))

    print("\n=== Realism preset benchmark (50 sheets/preset) ===")
    compare_realism(load(args.before_realism), load(args.after_realism))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
