# Marker Robustness Benchmark - 2026-05-23

## Scope

This benchmark stress-tests the ArUco marker preprocessing path for generated
25-question OMR sheets. It covers:

- 1-marker and 2-marker occlusions across all corner combinations.
- Same-edge and diagonal 2-marker recovery.
- Skew, 180-degree rotation, perspective, blur, JPEG compression, and
  Xerox-style low contrast.
- Fail-closed behavior for 0-1 decoded marker cases.

The structured run log is stored at:

`robustness_full_20260523.json`

After adding degraded 2-marker bubble confidence scoring, the updated
structured run log is stored at:

`robustness_confidence_full_20260523.json`

Command:

```powershell
python scripts/bench/bench_marker_robustness.py --rows 10 --profile full --label robustness-full-20260523 --out robustness_full_20260523.json
```

Updated confidence-gated command:

```powershell
python scripts/bench/bench_marker_robustness.py --rows 10 --profile full --label robustness-confidence-full-20260523 --out robustness_confidence_full_20260523.json
```

## Result Summary

- Scenarios: 73
- Sheets per scenario: 10
- Clean sheets: 100% success
- 180-degree rotation: 100% success
- 1 missing marker: 100% success for full occlusion, except synthetic edge-clip cases that intentionally remove too much border context
- 2 missing markers: 100% success for full occlusion across same-edge and diagonal pairs when two markers remain decodable
- 2 missing markers with same-edge + perspective drift: risky right-edge and bottom-edge combinations are now rejected by the bubble confidence gate
- 3 missing markers: 0% success, as intended
- 0-1 decoded markers: fail closed, as intended

Worst successful cases remain Xerox/perspective combinations where the bubble
outline confidence stays above threshold. They should still be treated as
auditable recovery runs because the 2-marker path uses a similarity transform
rather than a full projective transform.

## Confidence Gate Results

The confidence gate only runs on `similarity_2marker` recovery. It samples
template-derived bubble outlines after the warp and checks whether the expected
bubble rings still sit on printed dark ink. Healthy 2-marker recoveries scored
around `0.96` with zero rejections.

Important confidence-gated rejections in the full benchmark:

- `full_TR+BR__perspective_mild`: 0% success, 10/10 rejected, mean score `0.307`
- `full_TR+BR__perspective_strong`: 0% success, 10/10 rejected, mean score `0.236`
- `full_TR+BR__xerox_perspective`: 0% success, 10/10 rejected, mean score `0.262`
- `full_BL+BR__perspective_strong`: 0% success, 10/10 rejected, mean score `0.363`
- `full_BL+BR__xerox_perspective`: 0% success, 10/10 rejected, mean score `0.425`

## Important Failure Modes

The following failures are expected and desirable:

- `full_*+*+*`: only 1 marker remains, so there is not enough geometry to align safely.
- `clip_*`: the current synthetic clip removes full image edges, often leaving 0-1 markers. This models severe scanner clipping and should fail closed unless a future page-border fallback is added.
- `clean__motion_blur`: marker bit patterns are too degraded to decode. This is safer as an error than a guessed alignment.

The following failures are stress-test artifacts to interpret carefully:

- `clean__skew_2deg` and `clean__skew_5deg` rotate the already-tight 666x515 processing canvas, clipping edge markers. Real scanner skew should ideally be tested on the higher-resolution source scan before resize.

## Research Notes

OpenCV guidance for ArUco boards supports the current approach: use
`refineDetectedMarkers` with a known board layout so missing markers can be
searched from rejected candidates. Without camera calibration, OpenCV uses a
global homography for this refinement, so geometric sanity checks are still
needed.

General OMR guidance also supports conservative behavior: use anchors/fiducials
to transform the known template zones, but reject sheets when the alignment
evidence is insufficient. This is especially important because OMR scoring can
look plausible even when a page is subtly misaligned.

## Current Recovery Policy

- 4 decoded markers: marker-corner homography.
- 3 decoded markers: marker-corner homography using observed corners only.
- 2 decoded markers: degraded similarity transform, gated by the existing
  convexity, aspect-ratio, area sanity checks, and post-warp bubble confidence.
- 0-1 decoded markers: fail closed.

The 2-marker path is intentionally limited to rotation, uniform scale, and
translation. It does not estimate perspective, which reduces silent
misalignment risk.

## Recommended Next Hardening

- Add high-resolution source-scan skew tests before the image is resized to the
  processing canvas.
- Keep all 2-marker recoveries visible in logs with the existing
  `degraded similarity-transform recovery` warning and the new bubble confidence
  score.
- Consider a page-border fallback only as a separate opt-in path with strict
  debug overlays and a confidence threshold.
