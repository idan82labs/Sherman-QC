# V11 simple-family failure analysis — 2026-09-16

## Answer about the score

The full V11 score remains the audited 71-line development result: control F1
`0.4285714286` and V2 challenger F1 `0.3818181818`.  Neither is the final
engine score or a historical-benchmark win.  The focused simple-family run is
a separate diagnostic over 12 lines; it must not replace the full denominator.

## Focused result

The fixed subset contains one case from each of `37361005`, `49001000`, and
`49024000`.  The V2 candidate produced 10 lines: 9 TP, 1 FP, and 3 FN, giving
precision `0.9000`, recall `0.7500`, and F1 `0.8181818182`.

| family | TP / FP / FN | precision | recall | F1 | pose |
| --- | ---: | ---: | ---: | ---: | --- |
| `37361005` | 4 / 0 / 0 | 1.00 | 1.00 | 1.00 | retained accepted context |
| `49001000` | 2 / 1 / 3 | 0.67 | 0.40 | 0.50 | retained accepted context |
| `49024000` | 3 / 0 / 0 | 1.00 | 1.00 | 1.00 | gated rejected context |
| subset | 9 / 1 / 3 | 0.90 | 0.75 | 0.8182 | — |

The result is retained in
`gold11_v11_simple_family_diagnostic_v2/RESULT.json` (SHA-256
`0540e53b3e1652d1bd4b1b3bcac69c73bda29efbd104b5a00255cedbbbba6e91`).  The
three post-inference overlays are in its `overlays/` directory.

## What the overlays and raw-support audit show

`37361005` is visually coherent: all four solid predicted lines lie on the
four dashed reference lines.  Two accepted lines have transverse offsets of
`3.395` and `3.779 mm`, so the apparent success already sits close to the
unchanged scorer's `4.0 mm` offset limit.

`49024000` also recovers all three bends.  Its predicted finite spans are only
`14.2–17.9 mm` against reference spans `19.5–28.3 mm`, and 3 mm support retains
only `0.64–0.77` of each axis.  This is a near-boundary partial-support success,
not evidence that the line geometry is calibrated.

`49001000` is the decisive failure.  Its raw scan supports all five reference
lines at 3 mm (support fraction `1.0`; reference nearest-point medians
`0.128–0.719 mm`, p90 `0.174–1.202 mm`).  Therefore the three missed references
are not absent from the scan and are not caused by a scan-coverage void.

The transformed native axes tell a different story:

| native axis | 3 mm support | median / p90 distance | result |
| --- | ---: | ---: | --- |
| `STEP_BEND_001` | 0.527 | 2.938 / 4.828 mm | truncated; unmatched long line |
| `STEP_BEND_002` | 0.129 | 3.047 / 3.091 mm | rejected; near `M04` |
| `STEP_BEND_003` | 0.000 | 4.603 / 4.666 mm | rejected; near `M02` |
| `STEP_BEND_004` | 1.000 | 2.258 / 2.293 mm | matches `M05` |
| `STEP_BEND_005` | 1.000 | 2.846 / 2.869 mm | matches `M03` |

At 3 mm, the support gate therefore discards `STEP_BEND_002` and
`STEP_BEND_003`, while the long `STEP_BEND_001` is trimmed to roughly half its
400 mm span.  These produce the three false negatives.  The accepted native
axes are consistently displaced from their matched references by about
`3.7 mm`, indicating a line-placement or CAD-axis-to-scan convention offset.

## Radius sensitivity result

The frozen support-radius sweep varied only the recovery gate and retained the
same poses, topology, scan bytes, and matcher.  It is in
`gold11_v11_simple_family_support_radius_sweep_v1/RESULT.json` (SHA-256
`b98829c5466a384562d3397b3c711fb65a2a14f782550af5cb4cbc8cbfbaf392`).

| support radius | TP / FP / FN | precision | recall | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 3.0 mm | 9 / 1 / 3 | 0.900 | 0.750 | 0.8182 |
| 3.5 mm | 10 / 1 / 2 | 0.909 | 0.833 | 0.8696 |
| 4.0 mm | 10 / 1 / 2 | 0.909 | 0.833 | 0.8696 |
| 4.5 mm | 10 / 2 / 2 | 0.833 | 0.833 | 0.8333 |
| 5.0 mm | 11 / 1 / 1 | 0.917 | 0.917 | 0.9167 |

The sweep confirms that the 3 mm gate is too tight for this raw scan/axis
convention: `STEP_BEND_002` becomes usable at 3.5 mm.  It does not solve the
whole problem.  At 5 mm all five native axes are emitted, but `STEP_BEND_003`
still fails the unchanged matcher because its reference offset is `4.728 mm`,
above the scorer's `4.0 mm` limit.  At 4.5 mm that axis becomes an additional
false positive instead of a correct match.  Loosening support alone therefore
trades abstentions for unsupported or mislocated lines.

## Root cause and next engineering target

The simple-family evidence separates two coupled mechanisms:

1. The support gate is calibrated to a nominal axis-to-scan distance that is too
   small for at least this material/line convention.  It rejects valid axes
   even when the reference line has dense raw support.
2. Native STEP axis placement is systematically offset from the adjudicated
   line convention by roughly 3–5 mm.  A larger support radius can expose the
   scan but cannot move the published line onto the observed bend line.  The
   remaining `49001000` miss is a correspondence/line-placement error, not a
   missing point-cloud feature.

The justified next candidate is a label-independent, scan-derived centerline
fit around each CAD axis: retain the CAD axis as a topology prior, estimate a
finite observed line from robust local cross-sectional support, and publish it
only when the fitted offset, direction, span, and support are jointly stable.
The candidate must be tested against thickness variation, holes, duplicate
parallel sheets, partial coverage, and genuinely separated near surfaces.  A
radius relaxation without that centerline/ambiguity gate is not promotable.

## What the self-healing replay proves

To separate pose selection from support gating, both finite retained poses for
`49001000` were replayed through the same V1 support kernel.  The accepted pose
(index 0, p95 `0.383 mm`) and the rejected finite pose (index 1, p95
`1.272 mm`, coverage `0.889`) produced identical scores at 3 mm: `2/1/3`,
F1 `0.50`.  At 5 mm, pose 0 produced `4/1/1`, F1 `0.80`, while pose 1
produced `3/1/2`, F1 `0.6667`.

This rules out alternate-pose selection as the reason the one-scan healing
looked good.  The accepted pose is the better of the two; the material gain
comes from allowing more scan support around its native axes.  The remaining
miss at 5 mm is still the axis/reference convention offset, so the self-heal
is a deterministic support-gate effect followed by a correspondence limit,
not unexplained stochastic recovery.

The replay is retained in
`gold11_v11_self_healing_pose_replay_v1/RESULT.json` (SHA-256
`dbba7a24420f98b03989c5820f6b1f33301ad931063736e9b521eade02eeac5c`).

No production or floor release follows from this diagnostic.  Full Gold11
historical comparability remains unestablished, and the original frozen V11
attempt remains untouched.
