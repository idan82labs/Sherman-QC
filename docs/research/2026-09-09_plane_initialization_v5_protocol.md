# Noisy finite-plane initialization V5: source protocol

Date: 2026-09-09. Draft source, not executed or promoted. Root owns
`backend/cad_guided_plane_initialization_v5.py`. Frozen V4/V2 source, prior
terminal attempts, original data and scorer remain unchanged. This protocol
does not independently authorize physical input or a native run.

## Engine connection and scope

Two pure-array entrypoints are additive:

- `prefilter_plane_initializations_v5`: independently owned noisy scan supports,
  complete imported CAD-face triangle unions, bounded plane search and finite
  assigned-face screening. No accepted registration.
- `register_whole_part_planes_v5`: same owned initialization, then unchanged
  V2 whole-surface refinement, observability, coverage, residual and generated-
  pose ambiguity gates on a prospectively bounded subset of starts.

Both require explicit `position_error_bounds_mm` and the exact unit assumption
`PROVISIONAL_COMMON_MM_NOT_CALIBRATED`. Error bounds are stipulated exploratory
assumptions, not inferred scanner calibration. They do not supply CAD nominal
measurements, a pose, acquisition-side identity, Gold or an observed bend count.
No file IO, native import, server startup, held-out data or external service is
part of this module. External CPU/wall/process-tree supervision is mandatory.

## Owned inputs and support bridge

The V5 extractor's owned-snapshot validation binds all original scan points,
triangle raw-row indices and per-row error declarations before geometry work.
Its result digests must match those snapshots exactly. Every core and closure,
including refusals and unscheduled expansions, survives as complete bounded
metadata. Potentially large arrays remain internally owned; output carries their
shape, dtype and hash, not a claim that raw-row maps are embedded in JSON.

Each qualified core/closure becomes one import-local plane-search witness with
its separate original support, fitted normal, witnessed anchored offset,
conditional normal cone, actual raw support RMS and observed centroid. Centroids
are used only by the existing within-scan scheduling rule. A core and closure
with the same rows remain alternative evidence; neither is a physical plane or
bend count. They are not collapsed using approximate XYZ similarity.

CAD fitting reuses the already reviewed V4 complete-face union fitter. It uses
every triangle of each import-local face and rejects a nonplanar whole face;
it does not segment a curved face until it resembles a plane. Its area TLS plane
passes through the retained area centroid. The V5 anchored equation is therefore
`n dot (x-area_centroid)=0`. Its zero model cone describes this nominal tessellated
plane, not certified analytic CAD or metrological accuracy. Complete face-row,
source-triangle and raw geometry digests are retained.

Before the first finite query, every selected scan support is rechecked against
the exact unique union of its original triangle vertex IDs, its count and both
digests. Every selected CAD support is rechecked against the whole input face
union and source-triangle identity. This is in-process owned-array evidence,
not file/native authentication; the later byte-workflow wrapper must bind that.

## Prospective finite-screening schedule

The plane-search helper already records its incomplete bank, triplet, pair,
unsigned and sign policies. This module adds two explicit schedules before their
corresponding work; it does not return the successful prefix of an overrun.

1. Collect the ascending IDs of all unsigned assignments with at least one
   `PROPOSED` normal-sign witness. Choose at most **32** evenly spaced integer
   rank strata including both endpoints; one slot uses the middle.
2. Screen **every proposed sign** for each selected unsigned assignment. There
   are at most eight per assignment, hence at most **256** screened hypotheses.
   Retain all unselected proposed signs as `NOT_SCHEDULED`, and all search-level
   refusals as nominal-witness refusals, not proven assignment infeasibility.
3. For every selected hypothesis, check **all original raw rows of each of its
   three supports** against the complete finite assigned CAD face union. Rows
   shared between supports or repeated under another pose are charged again.
   Evaluate all three supports even when the first geometrically fails.
4. Use unchanged V2 unambiguous finite-surface queries and the maintained
   correspondence distance / minimum scan coverage thresholds. Finite-edge or
   vertex proximity is not an infinite-plane zero residual. Full boundary and
   ambiguous-row counters are retained. Initial assigned faces are not final
   measurement matches or acquisition-side identity.
5. Only byte-identical proper rotation/translation pairs group passing starts.
   Retain every grouped hypothesis origin; nearby poses are not merged.
6. Choose at most **48** evenly spaced rank strata of those exact passing groups
   for V2 refinement. Keep every omitted group and its pose provenance explicitly
   unsearched. Selection is fixed before refinement; final residuals, Gold and
   evaluator answers never choose the initial schedule.

Any selected surface build/query or unexpected numerical/resource failure
invalidates all usable starts. No downstream fit survives incomplete selected
candidate evaluation. An empty completed schedule is an initialization failure,
not absence of a plane/bend. `search_complete` is the inherited field for the
**selected bounded policy only**; the additional `search_exhaustive=false` and
`global_pose_uniqueness=NOT_ESTABLISHED` apply even when registration succeeds.
All completed-search hypothesis identities enter the ledger before per-support
validation. A late failure retains unstarted identities and explicitly counts
any exact passing groups found before fatal invalidation; it does not report them
as surviving starts or invent a completed refinement-omission count.

Normal alignment uses the search helper's nominal Wahba witness. Failure of that
witness can miss another orientation inside heterogeneous cones. Three unoriented
planes also cannot generally resolve chirality. Finite support and whole-part
surface verification are necessary downstream checks, not global proofs.

## Unchanged registration and measurement boundaries

The complete original scan supports are used for initialization screening even
when the caller explicitly supplies a subset of original raw IDs for global
registration. That selection must be unique, at least36 and at most200,000 rows;
without a selection V2's at-most200,000-row scan admission still applies. The
evidence distinguishes full prefilter rows from selected global-fit rows and
binds both digests. No hidden thinning or membership transfer is permitted.
The V5 selection seam revalidates the owned snapshot's shape, count, lower/upper
raw-ID bounds and uniqueness before indexing. This locally closes an inherited
V4 caller-to-snapshot negative-index race without changing the frozen V4 source.

Global registration reuses V2 spatial samples, coarse/fine refinement, finite
triangle diagnostics and generated-pose class selection. No new threshold,
fine-only update or PCA fallback is substituted. A numerical failure in any
scheduled candidate blocks publication, even if another candidate passed.

A conditional whole-part fit is not a stable datum: it may absorb deformation.
Finite triangulation is approximate, errors uncalibrated, scale unfitted and
physical units provisional. Measurement remains a later full-raw scan operation;
this module never generates nominal bend values or an observed count. The
byte-to-measurement V5 wrapper and connected matrix are separate, not yet present
by virtue of this source document.

## Hard limits

V5 noisy extraction and declared plane-search caps remain their existing values.
The new initialization caps are:8,192 retained sign records;32 unsigned screening
assignments;2,048 cached face identities;500,000 indexed face triangles;32,768
finite query calls;64 million charged support rows;65,536 rows per chunk;512
million node visits;256 million triangle pairs;32 million normal pairs;4 billion
surface work units;131,071 face-build nodes;10 million face-build triangle visits;
48 refined groups;96 MiB detailed report. Configuration snapshots are type-checked
and bounded against independent literal maxima before geometry.

The full selected row/call obligation is preflighted before any assigned-face BVH
or query. Other operation budgets remain cumulatively enforced during work.
Ordinary exceptions fail closed; process-control exceptions are not swallowed.
Oversized or nonfinite reports yield a small whole-report refusal envelope, not
a usable partial report. It is outside a zero detailed-report budget.

First pure source/helper gate is prospective **60s wall /45s worker CPU /1GiB
sampled group RSS /1MiB stdout**, serial under root's isolated harness. Native
connected and physical runs need new bounded protocols and independent pre-run
audits. No started limits may be changed.

## Required independent controls before connected execution

Freeze independently constructed actual-array positives with unrelated scan and
CAD tessellations, noise, pose and partial-patch centroids. Require an actually
correct generated finite start, then a correct whole-part pose under declared
error limits; a refusal does not satisfy a positive geometry case. Keep separate
plane/symmetric/reflection/absent-support/uncertain cases with honest refusals.

Check complete core/closure raw lineage, whole CAD face union including holes,
all-sign unsigned scheduling, repeated-row charging, exact passing-group origins,
over48 group omissions, preflight before native-free surface work, every selected
case/candidate denominator, caller mutation, stale/incomplete support rejection,
late failure after positive work and finite report/candidate invalidation. Tiny
mock tests establish contract paths only; actual-array positives must exercise
the real extractor, search and finite-surface code. Root must read and independently
audit the full test source before execution. None is claimed executed here.
