# F1 13-corpus optimization checkpoint — 2026-07-17

## Honest retained checkpoint

The retained scan-only development checkpoint is nested leave-one-physical-part-out
v8:

- 13 scans, 71 manually reviewed bend axes
- 52 predictions: 30 TP, 22 FP, 41 FN
- precision 0.576923, recall 0.422535, F1 0.487805
- exact aligned parts: 0/13

This is not an 80% result and must not be described as one. Same-corpus fitting
reaches F1 0.821918, but that is resubstitution capacity evidence, not zero-shot
accuracy.

## Additional scan adjudication

Five old CAD/scan match sets were rendered in six high-resolution views and
passed through a deterministic scan-frame gate. The gate samples 201 points per
axis, measures nearest scan support within 3 mm, and crops accepted axes to the
longest contiguous supported run. It rejects lines with under 10% support or
under 8 mm supported span.

- 16 provisional axes reviewed
- 11 accepted, 5 rejected, 0 unreviewed
- two 28870000 axes rejected as mixed-frame (roughly 350 mm from the scan)
- 47266000 diagonal rejected as a planar-face crossing
- 48963008 P03 rejected as a mostly unsupported diagonal crossing
- 48991006 P09 rejected as a scan-supported planar-face line with only about
  16 degrees two-sided normal separation

Only 44211000, 48963008, and 48991006 have complete training-only inventories.
They remain explicitly non-authoritative and never enter an outer test fold.

Strict scan-only execution on those three scans succeeded sequentially at
1.00–1.15 GiB peak RSS per scan. Candidate generation peaked at 0.44 GiB.
Raw candidates cover all 11 accepted axes; clustered line objects cover 10/11.

## Rejected selector experiments

All results below use the original 13 outer test scans with physical-part
grouping and no part identity, expected count, CAD, or manual truth exposed to
candidate generation.

| Experiment | TP | FP | FN | F1 | Exact |
|---|---:|---:|---:|---:|---:|
| retained v8 reproduction | 30 | 22 | 41 | 0.487805 | 0/13 |
| nested auxiliary weight | 28 | 26 | 43 | 0.448000 | 0/13 |
| raw-candidate baseline | 28 | 31 | 43 | 0.430769 | 0/13 |
| junction/parallel arrangement features | 26 | 25 | 45 | 0.426230 | 0/13 |
| sequential nested two-sided veto | 30 | 22 | 41 | 0.487805 | 0/13 |
| sparse topology completion | 30 | 23 | 41 | 0.483871 | 0/13 |

The rejected two-sided fields were removed from the default v8 feature vector;
the baseline reproduces exactly. Optional experiments remain isolated behind
CLI flags and cannot affect the retained default.

## Material blocker requiring professor guidance

The candidate pool has an 84.5% birth ceiling (60/71), while the classifier can
fit this corpus above 80% but does not generalize across physical parts. More
independent weighting and feature tuning now increases variance rather than
accuracy. The unresolved semantic decision is how to distinguish, from scan
geometry alone:

1. a real fold between two planar sheet regions,
2. a free boundary/part outline with strong point support,
3. a seam or planar face ridge,
4. a vertical tab-to-wall fold versus a cut tab edge, and
5. multiple aliases belonging to one physical bend.

Professor review should operate on raw scan cross-sections and the accepted/
rejected examples, especially 48991006 P06–P09, and provide a deterministic
local semantic rule or labels for both sides of each transition. Screenshots or
count-only feedback are insufficient for the next solver tranche.

F1 remains research-only. Runtime/main promotion is prohibited until the
enterprise gates, including visual mismatch acceptance, pass.
