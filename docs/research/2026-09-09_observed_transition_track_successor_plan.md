# Scan-derived transition tracks — proposed successor, not implemented

Date2026-09-09. Independent source/design review by the finite-instance reviewer;
root records it for work after the first frozen mesh-workflow diagnostic. No
physical/Gold data was read for this design. Frozen R0 remains unchanged.

## Problem and scoped meaning

Native V1 deliberately partitions a cylindrical candidate into boundary-certified
search windows. The full-raw wrapper currently fits each window and rejects
competing curved rows before any observation association. Its emitted instance
tally is therefore window-dependent. Grouping every common CAD candidate would
also be wrong. The roadmap target is stage-appropriate observed transitions,
not CAD totals, successful fits or reconstruction of every manufacturing event.

## Proposed mechanism

Preserve fitted windows as evidence fragments with explicit native membership,
immutable raw arc/flange IDs and measured geometry. A new association stage may
propose neighbors from native membership, but must decide using scan evidence.
Equivalent use of the same curved support is duplicate evidence; incompatible
use remains ambiguous. Shared planar context alone neither merges nor rejects
distinct curved roots. Accepted groups require a consistent joint measured fit,
not a transitive chain of pairwise similarities.

Join separated fragments only with positive original-row bridge support under
a predeclared bounded sampling model. Assess cylindrical coordinates `(s, r*theta)`
and local angular coverage, not only global marginal spans. CAD continuity must
not fill missing scan observations. A flange hole may split search windows while
the observed curved transition remains connected.

Retain support as an interval/segment union, never a padded bounding segment.
The measured carrier can have a hole even when neighboring curved support is
connected. Publish separate window, fragment, resolved-track and unresolved-group
counts. A scoped observed count is available only when its association contract
is resolved; stage/scope, unit, registration, side and uncertainty limitations
remain attached. Missing acquisition evidence cannot distinguish occlusion from
missing material, so connectivity is an operational contract, not material proof.

## Needed record

An `observed_transition_track` should retain observation ID, source window IDs,
native candidate IDs, scan hash/raw curved support ownership, reusable flange
context, jointly measured axis/radius/sweep and residuals, supported intervals,
carrier pieces/gap reasons, association evidence and unresolved alternatives.
Do not parse encoded window-ID strings to recover provenance.

## Fixed adversarial proof before evaluation

- Identical scan with one versus several artificial search windows: same count.
- Flange-hole window split with positive curved scan bridge: one track.
- Hole through carrier: no exported line across the hole.
- Separate collinear tabs with matching radii/shared parent flange: remain distinct.
- Missing scan band without bridge: explicit unresolved association.
- Duplicate windows/raw rows and row/window shuffles: no overcount/order effects.
- Distinct bends sharing only planar context: both retained.
- Reused curved rows with incompatible flange pairs: ambiguity, no double count.
- Pairwise-compatible axis/radius drift chain: joint inconsistency prevents merge.
- Multiple CAD bends but only one observed curved transition: never copy CAD count.

Independent identifiability review: identical separated coaxial scan patches can
describe either one occluded transition or two separate collinear tabs. Without
additional observed termination/connection evidence, no scan-only estimator can
resolve those physical counts differently. The separate-tab control above must
therefore supply that independent evidence, or require unmerged fragments with
an unresolved count—not a fabricated physical count of two. A neighborhood graph
can define a declared sampled-component count, not prove material connectivity.
`observed_bend_count` must remain unknown until its separate physical semantics
are justified; exposing a scoped sampled-track count is not closure of the95%
physical count target.

The minimum additive source seam is before curved-support conflict clearing in
`backend/cad_guided_local_measurement_v1.py`; V1 itself stays frozen. Preserve
failed/subminimal fragments and the complete raw search-row union as well as
successful fits, or one-to-many window-partition invariance cannot hold. Distinct
observed bridge rows and one fresh joint fit must justify association; shared
flange context, residual ceilings and pairwise chains are insufficient. Tests
must include disconnected two-dimensional cylindrical support despite matching
one-dimensional axial/angular marginal spans. This remains design, not execution.

Keep Gold outside association/carrier choice. Any single-line export from a
multi-piece track requires a fixed, independently reviewed convention before
evaluation and disclosure of lost support. A union-aware evaluator would be a
separately versioned metric, not silently comparable to the existing line scorer.

## Registration work identified in parallel

The retained dense connected test loses fit-subset rank after correspondence
and trimming changes. Rare geometric constraints can be removed even when
all-supplied-cloud rank remains nonzero. A mechanism investigation should separate
initialization, finite gate, sample selection and refinement stages; more points
alone are not a proven remedy. [Gelfand et al.2003](https://pixl.cs.princeton.edu/pubs/Gelfand_2003_GSS/stabicp.pdf)
motivates selecting constraints that stabilize weak rigid-motion directions.
Its near-correct-overlap assumptions do not provide global correspondence or
acquisition-side truth. No new implementation or parameter sweep is authorized by
this document alone; each successor still needs a fixed budget and audit.
