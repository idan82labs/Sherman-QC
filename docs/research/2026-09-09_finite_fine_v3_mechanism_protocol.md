# Fine finite-surface Gauss–Newton V3 — additive mechanism experiment

This protocol follows the retained V2 connected native1/3 failure. The complete
V1 physical baseline, V2 source and all prior terminal receipts remain unchanged.
No new physical input or Gold is used. The autonomous CAD-assisted goal is active.

## Hypothesis and exact mathematical change

V2 uses fixed-nearest-point Kabsch in both coarse and fine stages. For many
flange-interior points whose normals are perpendicular to an axis, axial sliding
does not change their finite surface distance while their projections stay in
the face. Fixed-point Kabsch nevertheless assigns those rows axial stiffness.
If only a few cap rows constrain that direction, the resulting correction can
be much smaller than the cap residual. This is a falsifiable mechanism hypothesis,
not a causal conclusion from the negative native matrix.

Keep coarse V2 exactly. For the fine step, let c be the mean of the actual
retained physical points and L their RMS radius. The normalized rigid-action
Jacobian is B_i=[-skew(p_i-c)/L | I], with unknown z=[L*omega, displacement].
For unambiguous FACE-interior closest points use residual n_i dot(p_i-q_i)
and Jacobian n_i^T B_i. For EDGE/VERTEX closest points use residual p_i-q_i
and all three rows of B_i. One weight min(1,delta/||p_i-q_i||) belongs to each
physical point; all three boundary equations share it. Do not normalize by the
expanded equation count or apply coordinatewise Huber weights. Solve by SVD with
the retained1e-10 cutoff; recover omega=z_rotation/L, never a fitted scale.

Use the existing centered rigid-motion caps and exact Rodrigues composition.
Every backtracking trial re-queries actual finite distances on precisely the
same retained fine-input row indices (without claiming original raw-row provenance),
without reselecting, retrimming or discarding
new ambiguous/out-of-gate rows. Accept only a nonincreasing same-set mean Huber
loss within the existing1e-14 floating-point slack. At most eight trials and
the unchanged20 fine iterations in the invented diagnostic. All queries count
against the existing cumulative budgets; fatal failures propagate.

Boundary fixed-q residual is a point-target upper-bound surrogate, not the
exact active-edge projector. FACE normal residual is local, not finite geometry
outside the triangle. Actual finite-loss acceptance is therefore essential.
The expanded optimizer matrix never substitutes for final geometric observability;
the unchanged V2 FACE-only supplied/actual-trim/weighted normal-rank gates apply.

[Low2004](https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf)
provides the local linearized normal-step and centering/scaling rationale.
[Rusinkiewicz–Levoy2001](https://www.cs.princeton.edu/~smr/papers/fasticp/fasticp_paper.pdf)
discusses planar sliding and minimization variants given a rough initial alignment.
The mixed boundary surrogate and finite-loss safeguard above are our derived
engineering extension. Neither source guarantees recovery from a wrong basin.

## Fixed gates, limits and promotion sequence

1. Independent source/math review and focused helper tests: finite-difference
   Jacobians, frame/action covariance, shared vector weights, sparse-cap/large-
   flange control, finite-boundary backtracking, no trial-row removal, fatal
   query propagation and invalid chart/pose refusal.60s wall/45s CPU/1GiB/1MiB.
2. One new two-arm diagnostic in
   `tests/test_cad_guided_finite_fine_comparison_v3.py`: original invented nominal
   STEP/analytic PLY, same32768 CAD samples and fixed generated PCA candidate20.
   One unchanged V2 coarse call, then one V2 fine control and one V3 fine challenger
   from identical copied coarse states. Each has its own unchanged cumulative
   query budget.60s wall/45s CPU/1GiB/1MiB; one heavy job. Both arm outcomes remain
   and oracle evaluation occurs only after both terminate. No physical access.
3. Candidate-level advance criterion: both executions complete; challenger meets
   every unchanged V2 final geometric gate, maximum oracle action error<=.20mm,
   and strictly smaller error than control. Diagnostic pytest success alone is
   not this criterion. Candidate20 is a diagnostic identity, never production
   start selection or known-pose injection. A failure requires retained diagnosis,
   not longer iterations until success.
4. Only after evidence justifies it: new public24-start registration and connected
   three-case native workflow under separately frozen/audited source and budgets,
   retaining original nominal measurement/endpoint ceilings and required refusals.
   No single-start result is public-registration or engine-output qualification.
5. A separately frozen audited all-Gold11 development evaluation and controlled
   integration are still required. Historical comparability stays unestablished;
   physical measurement truth/calibration and observed physical count remain absent.

The named bounded harness captures source hashes and fixed limits before each
synthetic run and retains every output/refusal. No run has started under this
new protocol at its creation. No original full-matrix policy, geometric gate,
reference, scoring or tolerance is changed. This separate candidate diagnostic
explicitly executes only fixed start20; it does not qualify the full24-start search.
