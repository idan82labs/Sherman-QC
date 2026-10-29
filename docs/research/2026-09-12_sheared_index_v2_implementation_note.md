# Minimal standalone sheared index V2 — source-only note

2026-09-12. Source-only: no implementation, execution or performance-rule freeze. Only named V1/primitive sources read. Primitive qualification is separate; physical query orientations remain unknown.

## Minimal boundary

Create a V2 module explicitly importing unchanged V1 and the primitive. Its internal subclass calls V1 construction unchanged: owned raw/support snapshots, hashes, stable world-axis sort,256-row leaves, tree membership, minimum raw IDs and <=7 seeds. Reuse V1 `_evaluate` and staged binary64 arithmetic. No monkeypatch, schema disguise, coordinate replacement or pose fit.

Post-build, use one global represented shear. Pack z low/high and residual low/high in `(node_slots,4)` binary64 storage:32 bytes/node. Reuse existing world boxes/start/end/minimum IDs. Global axes/slopes and fixed metadata are separate. Only transient ShearedBox objects are needed. Two nonempty children call merge_boxes; a single-child parent copies with charged work. Never merge empty padding or fabricate row counts.

Pre-reserve packed storage, immutable-freeze copy peak, leaf gather and primitive WORKING_BYTES. Charge gather/writes/copies/reports separately. Partial failure invalidates the index, retaining attempted counters. V2's wrapper preserves whole-clear, including late report failure, with new implementation/bound metadata and unchanged arithmetic identity.

V1 query needs a narrow explicit copy, not just `_bounds` override: child priority currently uses the same bounds as pruning (V1:301–309). Carry both returned intersection bounds and original world bounds. Prune only with intersection bounds; calculate `max(current_min-world_low,world_high-current_max)` and child-ID ties using world bounds, retaining the original stack/seed/traversal rule. Reuse original min/max raw-ID equality logic. The primitive returns the original rounded world interval, so no second corner computation is needed. Preserve query invalidation on BaseException. The changed traversal code, packing and wrapper are the actual future integration delta.

## Fixed raw-only chooser

Use only the existing sorted seed IDs, at most7. Enumerate their at-most35 unordered triples in lexicographic raw-ID order. Form two differences and their cross product with fixed separate operations; choose the finite nonzero cross with greatest max-absolute component, first triple on ties. Select its largest-absolute normal component k (lowest axis on ties), remaining axes i,j in ascending order; set slopes a=-n[i]/n[k], b=-n[j]/n[k]. The dominant denominator bounds slopes by1. Validate the represented result, never clamp an unsupported ratio. Exact-zero/unsupported cross cases use a declared zero-shear world fallback, not SVD, iterative fitting or query-dependent retries. Rank-one seed populations are a known fallback/efficacy negative.

Any admitted shear is safe only through every member's residual enclosure, not physical-plane estimation. Precharge unsuccessful triples too; freeze tie/fallback identity. No CAD, Gold, future query or benchmark outcome chooses it.

## Accounting and overhead forecast

The primitive callback increments separate attempted `sheared_bound_steps` and the existing cumulative `coordinate_visits` charge; an over-cap parent charge must remain visible even when the primitive does not return its numeric_steps. Never reset this budget between leaves/nodes/queries. Split diagnostics into chooser/build/merge/query/packing work, with sum invariants. Charge one existing node visit per visited node as before. This is added abstract work, not V1's old coordinate formula or CPU instructions.

For S support rows, L nonempty leaves and M two-child merges, primitive construction adds `61S+78L+75M`; each bound adds390 steps. Gather/packing/chooser work is additional. At S=900000, L=3516 and M=3515, that construction alone is55,437,873 steps. Packed padded tree8192 slots uses262,144 bytes. A full query visits at most8191 nodes, adding up to3,194,490 bound steps; twelve add38,333,880. These are source forecasts, not measured visits or efficacy.

A rough savings condition is `9*(E1-E2) > added_build +390*B2 - original_corner_work1`, before other differing costs. E denotes evaluated rows and B bound calls. V1 corner work is not zero: two programs each perform3 subtracts,3 multiplies and3 adds, plus sign selection, checks and zero handling. Independently audit/freeze the full common ledger rather than silently counting only18 scalar operations or V1's reported coordinate counter. Even ignoring query-bound overhead,55.4M extra construction costs about6.16M evaluated-row equivalents. Short batches cannot easily repay this.

The earlier eight-population/12-query20% aggregate proposal is NOT adopted. Flat, near-roundoff, line-fallback and plateau controls may still scan every leaf with added costs. Root decides thresholds after qualification/review.

## Bounded efficacy discriminator

Propose four fixed32768-row populations,12 literal queries each, V1 then V2 serially: exact oblique plane; same plane with resolved dyadic thickness and two finite extremal offsets; a slab with repeated extrema plateaus; near-axis/full-dimensional negative control. Include signed normal and near-normal directions, fixed origins/constants; freeze exact formulas/IDs before running. Scalar/blocked independent original-stage oracle checks endpoint bytes and minimum raw-ID ties. Keep every negative and refusal. Report cumulative build+query costs at query1,3,12 from the same batch, not warm reruns or selected best prefixes; retain complete12-query result as denominator.

Prospective envelope:30CPU/45wall seconds,2GiB tree,one heavy job,1MiB log/8MiB report. No run authority or forecasted pass. PRE binds source, four generators/query lists, accounting, independent oracle and terminal handling. Output is the complete cost/break-even table, not a universal gain, cap increase, physical run or broad32+8 gate.
