# Two-plane finite-context proposal V1: pure qualification

Date: 2026-09-10. Prospective source-only invented-input gate. This is a new
mechanism qualification, not a retry of any physical or synthetic terminal run.
The consumed Gold11 V3 negative and its blocked artifact release remain unchanged.

## Engineering reason and mathematical boundary

Six completed physical cases produced 18,541 proposed triplet poses but zero
jointly supported starts. A separate exact-physical-configuration constructed
case has only two qualified normal families at the declared .05 mm error bound.
This supports investigating additional proposal families, not asserting that a
correct physical pose was generated, nor weakening noise, support or rank gates.

The additive helper accepts caller-qualified, caller-scheduled ordered plane
pairs. Two independent exact normal correspondences with equal included angle
fix a proper central rotation. Translation satisfies two plane equations and
therefore retains one axial degree of freedom. In anchored coordinates:

`y = R (x - source_origin) + target_origin + t0 + lambda u`.

`M t0 = b`, `t0 = M.T (M M.T)^-1 b`, and `u` is the normalized cross product of
the two signed target normals. Both source and target systems must satisfy their
declared rank/cone margins. Noisy central rotations are seeds, not enclosures of
all feasible poses. Offsets, frame origins and normal-sign changes must transform
consistently. The fixed independent pose-action ceiling is 1e-8 mm.

Whole supplied geometry and each complete assigned support contribute axial
extrema and midpoint votes. These are placement hypotheses, not certified bounds,
finite matches, observed bends, measurements or uniqueness. Asymmetric partial
coverage can miss the truth; the test explicitly requires that limitation to
remain visible. Neither this gate nor a helper pass qualifies pair scheduling,
native geometry, whole-part registration, free-circle fitting or physical F1.

## Exact source closure and independent tests

Freeze these five sources by SHA-256 in a separate different-author pre-audit:

- `backend/cad_guided_two_plane_context_v1.py`
- `tests/test_cad_guided_two_plane_context_v1.py`
- `scripts/check_cad_guided_measurement.py`
- `docs/research/2026-09-10_two_plane_context_proposal_design.md`
- this protocol.

The helper and test were authored by different reviewers. Root read all 628
helper lines and all 610 test lines before adding the harness route. Expected
outcome: 110 tests, zero skips. Ordinary stdlib/NumPy/pytest only; tests load the
single helper directly, not the backend package or server. No native imports,
raw scan/CAD, retained prediction, Gold, catalog, email or held-out access.

Independent oracles cover exact rational plane equations; four constructed rigid
poses and sign/order/origin variants; non-axis rational frame changes; unit
rescaling; original-row/triangle extrema witnesses; weak caps without fabricated
normal qualification; partial and disconnected contexts; exact deduplication and
cache accounting; malformed lineage; rank/cone refusals; quota boundaries; owned
snapshot shape races; and whole-result clearing after late failures.

The 8192-slot test checks scheduler preflight only, not full-cap geometry work.
Injected exceptions and snapshot boundary mocks qualify those specific contracts,
not concurrent mutation protection, native runtime or end-to-end performance.

## Execution, budgets and decision

After independent exact-source PRE-GREEN, root executes once:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite two_plane_context_v1
```

60 s monitored wall, 45 s worker CPU, 1 GiB sampled process-group RSS, 1 MiB
captured output; private HOME/TMPDIR, all numeric thread settings 1. One geometry
job at a time. Limits cannot change after STARTED. The old supervisor bytes are
preserved at `source_snapshots/two_plane_context_pre_eae849f8/check_cad_guided_measurement.py`
with SHA-256 `eae849f8da3930913290c02d0aa41ef49c0f50ffd1cc9a93f51b847ac66f0c83`.
The new route changes only suite selection, five-file bindings, budget selection
and retained receipts. It does not change the legacy supervisor mechanism.

Retain STARTED/OUTPUT/COMPLETED, all test failures and pre/post source hashes in a
fresh owned attempt directory. No favorable prefix, skipped oracle or exit-zero
alone establishes success. Any failure, omitted test, skip, stop or source drift
is a terminal negative needing diagnosis and a justified new frozen successor.
Independent retained-only post-audit must confirm all 110 outcomes and the exact
five-file closure before integration. It does not require replaying the suite.

Known supervisor limitations remain: monitored wall excludes some setup/hash/
receipt IO and cleanup; RSS is sampled, not an allocation or continuous peak
guarantee. This pure no-subprocess test does not qualify escaped native process
containment. A CPU termination is a failure even if the parent stop field is null.

## Next connected requirement

A pure pass is only permission to continue integrating the proposal mechanism.
The engine successor needs a label-independent bounded pair schedule, genuine
two-support finite-screen adapter, joint candidate/quota accounting and unchanged
whole-part refinement/ambiguity/rank checks. It must qualify the actual .05 mm /
free-circle operating configuration on independently constructed noisy, partial,
ambiguous and missing-support cases. No fabricated third plane, CAD-measured
answer, Gold-guided pose or threshold relaxation. A physical successor requires
its own frozen protocol and independent pre-run audit; this document grants none.
