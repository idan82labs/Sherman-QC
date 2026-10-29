# Repeated membership cap: source diagnosis, not a recovered physical result

Date:2026-09-10. Source-only bounded audit by `/root/cad_support_lineage`, with
root checking the named V7/V5 seams. No execution or physical/cache/Gold reads.
This refines the membership row in the
[failure-factor audit](2026-09-10_gold11_failure_factor_audit.md).

## Exact cause and limits of the evidence

`backend/cad_guided_noisy_mesh_planes_v7.py:271` charges each core's base and
halo membership lengths separately. Lines305–315 already identify exact closure
duplicates by digest plus full array equality BEFORE charging a new closure's
complete triangle-row length. Thus a cache of identical closures is not a missing
optimization that can simply remove the observed20M cap failure.

Distinct but overlapping closures repeat logical memberships. The released
physical record establishes19,518,222 charged memberships and insufficient room
for the next closure, not its exact overlap with existing sets. Recovery magnitude
cannot be inferred from source or available released aggregates.

V5's `_support_rows` unions ORIGINAL triangle-referenced raw IDs, not coordinates,
and charges a separate raw-membership counter. `_fit` separately charges complete
support-row work before unchanged TLS/cone analysis. Compressing triangle
membership storage cannot silently remove those work or output obligations.

If the exact original logical-membership20M cap remains the admission rule, an
equivalent result above20M cannot be admitted. CPU caching can improve cost but
does not change that logical counter. Available RAM is not the cause or a reason
to claim the original cap was passed.

## Candidate representation direction — design only

A separately versioned immutable membership-block pool could share original
triangle-ID blocks with exact membership bitsets across nonidentical supports.
Preserve ordered support descriptors, full cardinalities and original-array
digests; digest collisions require exact equality. A bounded accessor reconstructs
sorted original IDs and supplies complete raw unions to unchanged V5 geometry.

This changes representation accounting and needs a NEW frozen protocol with
explicit finite limits, not a relabeled unchanged20M pass. Keep separate ledgers:

- Logical memberships across all supports, including repeats.
- Unique stored memberships and bytes.
- Lookup/comparison/encoding/traversal work.
- Per-call and cumulative decoded rows.
- Existing raw-membership, fitting, metadata and retained-output limits.

No unlimited traversal or expanded output is permitted merely because compact
storage fits. Failure late in encoding/decoding must still invalidate the whole
candidate with honest attempted/discarded evidence.

First qualification should independently construct overlapping/nonidentical sets,
exact duplicates, collisions, holes/disconnected IDs, repeated XYZ with distinct
raw identities, immutable round trips and every budget/late-failure boundary.
Then compare complete small-mesh predecessor/successor ledgers and numerical
results. Set-transport tests alone do not qualify autonomous mesh discovery.

No implementation or new budget is selected here. This seam is not a reason to
delay the next informative all11 comparison until universal support exists.
