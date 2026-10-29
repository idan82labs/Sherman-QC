# Shaked F1 residual surface-ownership blocker — 2026-07-18

## Outcome

No additional count-free pruning rule was accepted for the four residual
false-positive lines in `05_49191032`, `09_49191037`, and `11_48340007`.
The frozen scan-mesh cache supplies convincing local bend evidence for all
four, while the recomputed triangle-region graph contains true-bend aliases
for every plausible ownership, free-boundary, or developability gate tested.

This is a no-patch result for the four errors left after the accepted v18
surface-graph pruning pass.  It preserves the v18 Shaked checkpoint and the
corrected original-13 checkpoint.  The Shaked labels remain assistant-reviewed,
user-review-pending development evidence; they are not enterprise ground truth.

## Residual lines

Metrics below were recomputed deterministically from the source PLY.  Pair
contact occupancy is the fraction of 40 along-axis bins supported by face-pair
contacts within 3 mm.  Normal-incidence error is the angle between the selected
axis and the cross product of its two fitted surface normals; an ideal plane
intersection would be near zero.

| Case | Candidate | Surface pair | Span mm | Bend angle | Transition score | Pair contact occupancy | Region hops | Normal-incidence error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `05_49191032` | `SMA1` | 7–59 | 35.810 | 122.01° | 0.634 | 0.850 | 3 | 8.275° |
| `05_49191032` | `SMA5` | 7–79 | 40.862 | 93.82° | 0.625 | 0.575 | 2 | 21.822° |
| `09_49191037` | `SMA1` | 111–215 | 20.131 | 98.65° | 0.585 | 0.525 | 2 | 30.587° |
| `11_48340007` | `SMA2` | 1–21 | 30.245 | 92.62° | 0.579 | 0.450 | 4 | 23.115° |

All four are single-source scan-mesh line objects.  That fact is not sufficient
for pruning: several exact Shaked bends are also single-source mesh objects.

## Exact true-bend aliases

The strongest alias is in `11_48340007`:

| Property | False `SMA2` (1–21) | True `SMA4` (1–3) |
|---|---:|---:|
| Minimum surface gap | 2.23 mm | 2.22 mm |
| Finite support / endpoint support | 0.92 / 0.667 | 0.92 / 0.667 |
| Region-graph hops | 4 | 4 |
| Maximum intermediate-region area | 992.903 mm² | 992.903 mm² |
| Total intermediate-region area | 1052.012 mm² | 1048.865 mm² |
| Pair contact occupancy at 3 mm | 0.450 | 0.375 |
| Normal-incidence error | 23.115° | 35.763° |

Both routes share the central sequence `1 → 116 → 78 → 132 → 0`; only the
terminal minor-region branch differs.  The false candidate therefore has
*better* plane-incidence and pair-contact numbers than the matched true line.
This rules out a safe local threshold on those quantities.

Other counterexamples are equally material:

- True `07_49191038/SMA3` has 53.304° normal-incidence error, worse than three
  residual false lines.
- True `09_49191037/SMA8` has 0.425 pair-contact occupancy, below the false
  `09_49191037/SMA1` value of 0.525.
- True `09_49191037/SMA2` needs six region hops, so rejecting long/nonlocal
  paths would remove recovered recall.
- False `05_49191032/SMA5` and true `05_49191032/SMA2` are both two-hop paths
  from surface 7 through a small curved-band region.  The topology describes
  the same pattern on both branches.

## Rejected deterministic gates

### Axis must follow the surface-plane intersection

A 20° maximum normal-incidence error would catch three of the four residual
lines, but it also rejects 12 matched Shaked lines.  Across the selected mesh
set the threshold audit was:

| Error threshold | False lines caught | Matched true lines lost |
|---:|---:|---:|
| 20° | 4 | 12 |
| 25° | 2 | 9 |
| 30° | 2 | 7 |
| 40° | 1 | 5 |

The extra false line at 20° is outside this three-case residual tranche.  The
large true-line errors show that a correct axis is often attached to the wrong
fitted mesh-surface pair.  Surface IDs cannot currently be treated as physical
sheet ownership.

### Require continuous face-pair contact

A contact-occupancy ceiling of 0.575 would catch three residual lines, but true
occupancies extend below 0.1.  Contact continuity is dominated by scan
occlusion, surface fragmentation, and the extractor's 30 mm proximity search.

### Reject remote or weak region-graph paths

Hop count, direct boundary length, bottleneck boundary length, intermediate
area, and inverse-boundary path weight all overlap the matched distribution.
The case-11 alias above is effectively identical on these features.

### Reject free-boundary lines

Median distance to an open mesh boundary was 8.985, 7.473, 22.396, and
4.243 mm for the four residuals.  In particular, the case-09 false marker is
not near an open triangle boundary.  Visual "part outline" semantics therefore
does not map to a reliable open-mesh-boundary predicate.

### Prune the weaker branch of a shared-surface fan

This would remove real multi-flange arrangements.  The case-11 false branch
shares hub surface 1 with three true branches, while its unary evidence lies
inside the true range.  Case 05 also contains both a true and two false branches
from hub surface 7.

## Frozen-schema limitation

The mesh-v4 payload serializes surface IDs, normals, areas, candidate axes, and
local scan evidence.  It does not serialize face membership, oriented shell
ownership, surface centroids, free-boundary membership, or region-graph paths.
Those quantities were recomputed from the PLY for this audit; even with that
extra information, the residuals were not separable without true-bend loss.

## Corrected original-13 audit

Both accepted conservative post-filters were applied read-only to every stored
prediction in:

`output/experiments/f1-10of13/grouped-holdout-selector-v19-mesh-v4-tight`

- `prune_weak_parallel_outlines`: zero removals.
- `prune_surface_graph_outlines`: zero removals.

The corrected original-13 result therefore remains TP 31 / FP 27 / FN 40,
F1 0.480620, exact 0/13.  No new rule from this investigation was integrated,
so there is no hidden original-corpus change.

## Required new invariant

Further safe progress requires a physical sheet-ownership representation that
survives the scan mesh's opposite skins, rounded bands, holes, and fragmented
plane regions.  A useful invariant must explain why the symmetric terminal
branches in case 11 represent different semantics while preserving the known
true six-hop and high-incidence-error bends.  This is the remaining professor-
level question; another tuned unary or surface-ID threshold is not justified.
