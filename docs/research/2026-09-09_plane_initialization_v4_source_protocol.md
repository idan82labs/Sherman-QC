# V4 initialization prerequisites — prospective source/synthetic gates

Date:2026-09-09. Engineering reason: the retained V3 fine-only native comparison
fails its advancement criterion; changing local descent did not recover the
correct pose. This successor investigates label-independent plane-derived starts,
not more iterations or relaxed registration gates. Frozen R0/V1/V2/V3 attempts
and original inputs remain untouched.

## Scope and fixed decisions

Only invented immutable PLY bytes and independently specified plane equations
are admitted in these prerequisite suites. No original CAD/scan, Gold, email,
held-out, app startup, physical pose or external service enters these tests.
The new triangle parser preserves original vertex and triangle-row identity from
the same fully validated bytes; nontriangular polygons refuse, never triangulate.
Mesh adjacency is not certified material connectivity. The plane helper consumes
explicitly unverified support summaries, not secretly verified raw measurements.

Critical method and exact defaults are in the
[V4 design](2026-09-09_plane_hypotheses_v4_design.md). Three independently
constraining plane normals are required. Proper normal alignment plus mapping
the two triplets' intrinsic intersections is the hypothesis; no centroid fills
an unconstrained translation. Every sign/permutation alternative and rejection
denominator is retained within fixed input/assignment/output caps. A budget or
numeric failure clears the entire proposal prefix. No hypothesis is an accepted
registration, observed bend or measured deviation.

## Frozen gate order and limits

1. Independent source review of the byte parser and all50 invented controls,
   then one named `mesh_topology_v4` harness run. Acceptance: every selected
   assertion passes, unchanged source hashes, no resource stop. Invalid/late
   bodies withhold all arrays; valid rows, winding, duplicate faces and canonical
   hashes match independent expected values across all three PLY encodings.
2. Independent source/math review of the supplied-plane helper and its separate
   analytical tests, then one named `plane_hypotheses_v4` harness run. Acceptance:
   every selected assertion passes, unchanged source hashes, no resource stop.
   Tests must cover proper poses, independent frame/unit changes including
   imperfect normal congruence, all orthogonal alternatives, signs/order,
   underconstrained planes, support/rank gates, strict type admission and fatal
   whole-prefix clearing. Independent truth is used only in assertions.

Each suite separately has60s wall/45s worker CPU/1GiB sampled process-group RSS/
1MiB combined output, one worker, BLAS/OMP threads1. The named harness binds its
own source, selected tests, all imported project sources and this protocol before
execution, retaining STARTED/output/COMPLETED in a new exclusive receipt directory.
No change to a started cap, case set or assertion; a failure remains retained and
any repair is documented and independently audited before a new receipt.

These are conditional implementation gates, not a positive engine experiment.
Passing them permits development of a scan-derived finite-patch proposal adapter
and a separately frozen connected comparison. It does **not** permit claiming
native or physical pose recovery, count/F1 gain, historical comparability,
uncertainty calibration or production promotion. The connected comparison must
freeze extraction, seed/assignment/refinement budgets, all constructed cases and
unchanged finite-surface acceptance/error criteria before execution. Gold stays
outside inference and threshold selection.
