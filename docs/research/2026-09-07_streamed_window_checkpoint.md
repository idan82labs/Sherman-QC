# Bounded large-cloud ingestion into declared local analysis windows

Date: 2026-09-07. Continuation of the
[locked roadmap](2026-09-07_production_generalization_roadmap.md#locked-execution-baseline--user-confirmed-2026-09-07)
and [same-face relation checkpoint](2026-09-07_same_face_tangency_checkpoint.md).
Source and invented inputs only. No scanner files, native detector, CAD records,
services, held-outs, email or control artifacts are accessed.

Continuation: the [dense all-row constraint checkpoint](2026-09-07_dense_axis_tube_checkpoint.md)
adds a separate conditional measurement path for a supplied axis/support family.
It audits every supplied support row but does not alter this dispatcher's
256-row window cap or discover dense bend regions.

## Scope chosen before the scale run

This step addresses large-cloud INGESTION and bounded local engine dispatch.
It does not implement dense full-scan discovery. The existing engine still accepts
at most 256 RAW rows per invocation, and that cap is not raised or evaded through
downsampling. Simply tiling a dense scan would change neighborhoods and graph
components, possibly hiding an incompatible continuation just outside a tile.

The new [stream bridge](../../research/geometry/streamed_window_bend_candidates.py)
accepts predeclared axis-aligned observed-coordinate windows and a cooperative
iterator of NumPy Nx3 chunks. Windows are caller stipulations, not verified CAD
events, registered locations, construction labels supplied to the engine, or scan
observations of expected bends. This is a useful finite-ROI seam toward the primary
CAD-assisted workflow, not a replacement scan-only product direction.

The exact completeness claim is: after valid normal EOF, each supported window
contains every supplied RAW row satisfying its fixed observed closed-box predicate,
and the unchanged engine completes its declared LOCAL search on that multiset.
Neither EOF nor a clear window boundary proves physical capture coverage.

## Atomic stream and local-boundary behavior

- Bounds are copied and the validated local policy is snapshotted before reading
  the producer. A separate empty-engine call validates policy without fitting
  geometry; it is not counted as a nonempty full-path result.
- Every chunk is shape/type/size checked before conversion. Every coordinate is
  checked for finiteness BEFORE spatial filtering, including off-window rows.
  Unsupported integer-to-float precision is rejected explicitly.
- No nonempty geometry analysis starts before normal EOF. A late NaN, iterator
  exception, row limit or chunk limit returns no geometry result, even if a good
  bend was seen earlier. Error messages do not reproduce producer exception text.
- Each window retains every selected raw coordinate and stream-row identity up
  to 256. Overflow clears its buffer permanently; duplicates consume this cap.
  Ingestion continues to validate/count the remaining stream, then the entire
  batch abstains. There is no first-256 sample rescue or automatic subdivision.
- All windows are checked before any nonempty engine call. A boundary-censored
  or oversized window prevents a good-window prefix from being returned. Later
  engine/output failures also clear all batch analyses. Per-window execution
  dispositions are diagnostics, not partial qualified results.
- Every canonical local coordinate maps to ALL matching global stream-row IDs.
  IDs survive rechunking of the same row sequence, not reordering or another scan.
  Overlapping windows remain independent alternatives; their candidate counts
  must not be added as physical events.

The returned counts separate rows inside ANY window, rows outside ALL windows,
and summed window memberships. The last quantity double-counts overlaps by design.
Input row count covers validated/processed chunks and excludes a rejected chunk;
nonempty engine-call counts are attempts. All frozen box bounds remain in failed
and empty-window receipts. Returned payload bytes reset to zero on failure,
while separate serialized/discarded-byte counters preserve performed work.
Retained coordinates are scalar copies, not NumPy row views that could keep entire
source chunks alive. No scan-sized point array, global distance matrix or global
row-ID set is accumulated.

## Why boundary censor is not global equivalence

Every selected raw point, not just a chosen candidate, must have observed box-face
clearance strictly greater than the declared margin plus an engineering allowance.
The declared margin must be at least

```text
max(max_neighborhood_radius, max_link_distance + 2*epsilon)
```

The added allowance is `512*machine_epsilon*(local_coordinate_magnitude + box_diagonal)`.
It conservatively exceeds the local engine's one/two distance-guard additions
at the radius caps. Equality is censored. This guards against obvious window cuts
through a sampled neighborhood or continuation without changing the final fit's
epsilon, removing inconvenient points, or asserting a missing physical bend.

It is still a LOCAL observed-window policy. No corrected/latent ROI membership is
claimed; the code does not inspect an outside epsilon shell to prove such a claim.
The floating-point allowance is not a directed-rounded numerical certificate.
Absence of samples near a boundary does not establish absence of material there.

There is also no claim that the result equals a hypothetical global-core result.
For example, with graph k=3, three interior points give a locally insufficient
graph. Adding a very distant fourth point can change the global cardinality
branch even though no cross-window capped edge exists. Global coordinate-scale
distance guards can also differ from local ones. A halo alone is insufficient to
assert global engine equivalence, physical component completeness or bend count.

## Explicit new aggregate budgets

This is a new dispatcher budget, not an unchanged whole-job budget:

| Consumer limit | Maximum |
| --- | ---: |
| Validated stream rows | 2,000,000 |
| Rows in one supplied chunk | 65,536 |
| Processed chunks, including empty chunks | 4,096 |
| Declared windows | 16 |
| Retained raw rows per window | 256 |
| Sum of serialized window-analysis payload bytes | 16 MiB |

All inherited per-window caps stay unchanged. At 16 windows, their aggregate
ceilings can reach 16,384 seed queries, 32,768 unique plane fits, 32,256 plane pairs,
4,096 associations and 4,096 center-assessment attempts. Per-association two-radius
cut limits still apply. This multiplied work is disclosed, not described as the
old single-core budget. Local engines run sequentially, not in parallel.

Consumer storage is bounded by a small number of chunks, 16*256 retained raw
memberships, bounded local-engine workspace and the result-payload budget plus
one pending local payload. Routing work is O(stream_rows * window_count); it uses
no all-pairs scan operation. The byte budget covers compact serialized WINDOW
payloads, not the small outer receipt metadata or peak temporary serialization
workspace/Python object graphs. One full local result and its transient JSON
string/UTF-8 bytes exist before the size check. Exhaustion clears all batch analyses.

The consumer cannot stop an iterator from blocking inside next() or allocating
an oversized chunk before yielding it. The chunk cap applies before consumer
conversion, not before producer allocation. A worker-isolated, cancellable parser
and process-tree resource enforcement remain required for a live ingestion path.
The source test uses a cooperative synthetic producer in one geometry process;
it is not durable job, cancellation, parser-security or floor-latency qualification.

## Fixed nonempty resource challenge

Before the first analyzer run, the scale fixture is fixed at TWO MILLION rows,
nominal chunk size 8,192 and eight windows spaced 50 mm apart. There are two
signed single-radius bends, two missing-middle-arc bends, two concentric-radius
mixtures and two adjacent U/Z profiles. These contribute 860 selected raw rows;
the other 1,999,140 are deterministic background outside every window.

Foreground rows are interleaved across the full stream, not only delivered as a
convenient first-chunk prefix. Each window uses analytically chosen fixed box
offsets `[-16,-4,-16]` to `[16,4,16]`, with boundary margin 2 mm. This choice was
checked from fixture bounds before results: the adjacent profile reaches z=11,
so the initially considered upper bound 12 would leave only 1 mm clearance.
No threshold or fixture is adjusted after examining a geometry result.

Acceptance requires nonempty direct-core oracles, exact local output and global
row-provenance equality, complete constructed-support matches with known-radius
containment, and transparent evidence that nonempty core calls occur only after
EOF. Test construction metadata never enters the dispatcher or local engine.
Separate controls cover 257 selected rows, raw duplicates,
boundaries/cropped continuation, overlaps, chunk partitions, policy mutation,
late off-window invalid coordinates and interrupted producers. The declared
resource target is at most 120 seconds and 512 MiB peak process RSS for this
synthetic test, within the unchanged 16 GiB project process-tree ceiling.

This intentionally does NOT claim two million DENSE analyzed bend samples: only
860 enter local geometry. The oversized/dense-window negative is part of the
reported outcome. Automatic ROI selection, full-density support, global topology,
registration and physical evidence remain unfinished.

## Verification and next action

Root reviewed both new test sources before execution; independent reviewers
checked the source/mathematical scope and wrote the contract/scale suites. Both
new files pass: **79 checks (72 contract + 7 scale/failure checks) in 0.95 s**;
external wall time 1.02 s, maximum process RSS 69,664,768 bytes (66.4 MiB), no
geometry child process. The command was:

```sh
/usr/bin/time -l env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_streamed_window_bend_candidates.py tests/test_streamed_window_scale.py -q -s
```

The fixed scale challenge consumed 2,000,000 rows in 245 chunks, selected and
retained all 860 raw rows in eight nonempty windows, and returned 1,278,398 compact
window-payload bytes. Every local output exactly equals its isolated direct-core
oracle; all canonical-to-stream row mappings are exact. Its work receipt records
8 direct oracle calls + 8 dispatched nonempty calls + 1 empty validation call.
The latter is not a geometry fit.

| Constructed-support outcome | Count |
| --- | ---: |
| Constructed events / complete-support conditional matches | 10 / 10 |
| Matched single-radius intervals checked | 8 |
| Matched two-radius band intervals checked | 4 |
| Matched containment failures | 0 |
| All associations / matched / unmatched | 16 / 10 / 6 |
| Unmatched conditional alternatives | 2 |
| Other single intervals left unscored | 8 |

The single-radius checks cover radius 2 controls and the adjacent events' separate
radii 2 and 3; mixture bands are checked at 2 and 2.2. Construction labels score
only exact full observed-support matches. Eight other proposed single intervals
(including the mixtures' single-model alternatives) are not assigned fabricated
truth. Raw statuses remain 10 conditional SINGLE, 2 conditional TWO and 4
ABSTAINED. The additive tangency audit retains eight matched single-face controls
and excludes the two unmatched single-face explanations; it does not remove
candidates or establish physical tangency. These are not physical precision or
false-positive rates.

The timed resource fixture took 0.551994 s for construction, eight direct oracles,
the full streamed dispatch and constructed-support containment checks. This
interval ends before the later exact-output/provenance assertions and receipt
serialization; the 0.95 s suite time includes those. Its sampled process-lifetime
RSS high-water mark was 62,930,944 bytes (60.0 MiB), not a local delta or the final
process peak. Both fixed 120 s / 512 MiB soft targets pass. This measures eight
windows and 8,192-row chunks, NOT worst-case work at every declared cap, repeated
benchmark statistics, hardware acceleration, real parser performance or scanner
latency. The 16-window / 65,536-row chunk limits are policy bounds, not a tested
maximum-load geometry qualification.

These are exact inputs from already-used construction families, not a new noise
or generalization matrix. The new evidence is stream routing, raw provenance,
resource behavior and unchanged local diagnostic behavior on nonempty inputs.

Both distinct and exactly repeated 257-row windows abstain for the entire batch
without any nonempty core call. Late off-window NaN, iterator interruption,
boundary censor, exhausted work/output budgets and incomplete local engines do
not return a successful prefix. Reused producer arrays cannot mutate retained
points, copied windows or snapshotted policy. Exact integer precision endpoints,
empty chunk limits, overlap receipts and the local/global cardinality
counterexample also pass.

The first execution also passed all 79 checks (0.97 s; external 1.04 s and
69,500,928-byte maximum RSS). Its resource interval was 0.567094 s and sampled
lifetime RSS 62,685,184 bytes. Before the final repeat, only a NEW test receipt
field was clarified: the older adjacent helper's `max_plane_evaluations` actually
reports performed fits, so this receipt renames it `plane_evaluation_count` and
checks it against the unchanged 2,048 policy cap. The observed count was 256 per
adjacent window. No geometry, fixture, threshold, resource target or dispatcher
source was changed in response to test results; there were no failed executions.

Final SHA-256 identities:

```text
research/geometry/streamed_window_bend_candidates.py
13eda1efcf16f8d730d4779140b95c086dee3ecfb1e77a0d7b133041662b7994
tests/test_streamed_window_bend_candidates.py
37cfd7542625053e3a6efb552d01f951abe6bebecddca06ca362cecd4705702d
tests/test_streamed_window_scale.py
58b9a53450fe2c4354b4a2d13e4e561fe1bf64d570b1964a2ce0bbe182f8c1a5
```

The ten named inherited engine files listed in the preceding same-face
checkpoint retain their recorded SHA-256 identities. No previous 86/200-check
or older full suite is represented as rerun by this checkpoint.

Next, use this bounded seam to make the dense-window decision explicit before
component-aware CAD-assisted mapping: a proposal/measurement separation would
need to retain all raw measurement evidence and propagate any compression error,
not simply raise the local cap or treat subsampling as complete support. Dense
connected scans are still unsupported here. Exact capture/scanner metadata,
calibration, event authority and separately permitted physical evaluation remain
required. No production promotion, UI or service changes occurred; C15, terminal
EXP35, M5/M6, locked acceptance gates and the paused goal remain unchanged.
