# V4 supplied-plane initialization: source-only design

This additive proposal follows the retained negative V2 connected result and
negative V3 one-start comparison. It changes initialization, not fine iterations,
the frozen sources, acceptance gates or any physical receipt. See the
[mechanism checkpoint](2026-09-09_surface_registration_v2_mechanism_checkpoint.md).
No execution or positive engine result is asserted here.

## Contract

`backend/cad_guided_plane_hypotheses_v4.py` consumes exact tuples of frozen
`ObservedPlaneV4` and `PlaneEquationV4` records. Plane equations use unit normals
and offsets in each supplied millimetre frame. Scan records additionally declare
support count, RMS plane residual and two descending in-plane support extents.
These summaries are not raw-row provenance, verified material side or finite
support overlap. The helper imports only NumPy and standard-library modules.
Normal length must be within 1e-8 of unity; the admitted normal and offset are
rescaled together, preserving the plane. Mutable collections, numeric strings,
Booleans and NumPy scalar fields are rejected rather than silently coerced.

The returned JSON-native `hypotheses` contain proper rotations, translations,
both triplet intersections, every scan/CAD plane ID and normal-sign assignment,
and engineering residual/conditioning diagnostics. The entire result states
`candidates_only=true`, `accepted_registration=false`, and
`production_use_allowed=false`. It never chooses a pose, measures a bend or
claims a count, conformance, calibrated uncertainty or floor readiness.

## Critical mathematics

Let the scan triplet equations be `N_s x=d_s` and CAD equations `N_c x=d_c`,
with one plane normal per row. For each bijection and sign vector `sigma`, solve
the proper orthogonal Procrustes problem `R n_s ≈ sigma n_c`. If
`H=N_s.T @ (diag(sigma) @ N_c)=U S V.T`, then
`R=V diag(1,1,det(V U.T)) U.T`. No fitted scale or reflection is allowed.

For **exact** normal alignment, substituting `x_c=R x_s+t` gives
`N_c t=d_c-sigma*d_s`. With imperfect congruence, solving that expression after
normal fitting is origin-dependent. Instead the implementation solves the two
intrinsic intersections `a_s=solve(N_s,d_s)`, `a_c=solve(N_c,d_c)` and sets
`t=a_c-R a_s`. It maps intersections exactly and is equivariant to independent
rigid frame changes in exact arithmetic. The direct offset discrepancy remains
an explicitly origin-dependent diagnostic, never an admission gate.

Both normal matrices require `sigma_min/sigma_max ≥ .02`. Two independent
normals leave translation along their cross product unobserved; parallel or
nearly dependent triplets are refused. No centroid or partial-cloud bounding
box supplies that missing coordinate. An intersection may lie far outside all
observed finite patches; neither this construction nor the rank ratio verifies
finite support or correct correspondence. Conditioning and residual ceilings
are engineering guards, not confidence bounds or numerical certificates.

## Frozen mechanism defaults and accounting

Scan admission declares at least 12 support positions, RMS at most .1 mm and
second in-plane extent at least .25 mm. These values are not selected using the
constructed fixture's pose or physical labels. Normal alignment maximum angular
residual is .02 rad; the two intersection solves have maximum absolute backward
residual at most 1e-8 mm. Policy must be explicit in any successor evaluation.

IDs are sorted only for deterministic traversal. All scan 3-combinations, CAD
3-combinations, six CAD bijections and eight normal signs are enumerated. No
support consumption, first-match choice, inner/outer preference, geometry-based
deduplication or candidate ranking is performed. Duplicate geometric records
with distinct IDs remain alternatives. Duplicate IDs within a frame are invalid.
Support exclusions and primary geometric rejection counters remain visible.
Policy and budget snapshots are returned. Input support counts and extents stay
explicitly `UNVERIFIED_DECLARATIONS`; no equal-support or independence conclusion
is inferred from them, and their rows are never reused as a measurement result.

Exact-type integer hard caps are 12 scan planes, 24 CAD planes, 100,000 signed
assignments and 4,096 retained hypotheses. Callers may tighten but not raise
them. Zero caps are legal. The complete combinatorial assignment count is checked
before triplet/SVD work, even for later rank-rejected combinations. Every complete
search accounts for all planned signed assignments. Hypothesis overflow or a
numerical failure clears the entire prefix; no successful partial set escapes.
An external process watchdog is still required for any scheduled evaluation.

## Next independent gates

Before integration: independently posed/scaled/frame-shifted and imperfect-normal
equations; sign and ordering invariance; exact proper Procrustes/offset oracle;
two-plane, parallel/near-singular and reflected-normal controls; repeated
inner/outer and disconnected coplanar alternatives; support-admission boundaries;
typed immutable input and allocation-preflight controls; exact caps and failure
after a retained prefix; JSON serialization and honest denominator assertions.

A separate scan-mesh plane/lineage adapter and CAD-plane proposal source must
provide this input without supplying a pose oracle. Every resulting start still
needs unchanged finite-surface residual, coverage, observability and competing
pose admission. That connected all-case gate—not this source artifact—decides
whether initialization actually improves the engine.
