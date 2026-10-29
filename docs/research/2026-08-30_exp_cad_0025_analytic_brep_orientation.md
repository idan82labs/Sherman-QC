# EXP_CAD_0025 — analytic B-rep orientation corroboration

## Executive result

EXP25 independently corroborates the EXP24 pose selection using analytic Parasolid B-rep topology rather than unresolved DisplayList samples. On both N011 and N012, the same frozen proper pose supports all 15 through-thickness cutout carriers and all six asymmetric crown carriers. The distinct 180-degree runner supports only 7/15 carriers and 0/6 asymmetric carriers.

Result:

`GPI-C_POSITIVE_INDEPENDENT_ANALYTIC_BREP_ORIENTATION_CORROBORATION_NOT_F1_PROGRESS`

This authorizes the absolute proper orientation for these two source-bound scans. It does not measure F1 and does not authorize production promotion.

## Independent source-only construction

EXP25 never reads EXP24, any unresolved DisplayList sample, evaluator truth, labels, expected counts, scanner-frame metadata, or filenames as identity.

It:

1. binds the authenticated SLDPRT and pinned cadmpeg hashes to frozen EXP17;
2. selects the unique active analytic shell containing all eight native bend faces;
3. extracts cylindrical faces with exactly two coaxial circle boundaries separated by the source sheet thickness of 2 mm;
4. requires the frozen 15-carrier inventory exactly;
5. enforces each face's analytic axial and angular trim;
6. scores deterministic 0.75 mm voxel-capped scan points within a 0.35 mm carrier tube;
7. ranks only pre-existing proper EXP22 poses that preserve strict 8/8 support on all four native bends;
8. never creates or refines a transform.

The 15 carriers comprise four flange holes, four base holes, the central cutout, four trimmed upper-slot arcs, and two upper holes. The latter six form the asymmetric crown group.

## Measurements

| Case | Selected | All carriers | Asymmetric | Total support | Runner | Runner all / asym | Support ratio |
|---|---|---:|---:|---:|---|---:|---:|
| N011 | SO3_PCA_02/04 class | 15/15 | 6/6 | 1,842 | SO3_PCA_21/23 | 7/15 / 0/6 | 3.176× |
| N012 | SO3_PCA_02 | 15/15 | 6/6 | 2,236 | SO3_PCA_21 | 7/15 / 0/6 | 3.869× |

## Robustness gates

The selected class is unchanged across:

- carrier tubes of 0.30, 0.35, and 0.40 mm on both scans: 6/6 passes;
- deterministic even and odd point partitions: 4/4 passes;
- every single-carrier omission: 30/30 passes;
- two independent physical acquisitions: 2/2 select the same class;
- separate primary and replay runs: 2/2 full semantic payloads and 2/2 PNG hashes byte-exact.
- focused unit and decoded-source integration tests: 5 passed.

## Native-resolution visual ruling

Both 3000×2400 overlays were inspected at original resolution. In the selected panels, yellow support traces the central cutout, circular holes, trimmed upper slot arcs, and two upper holes; the magenta analytic trim curves and centers sit on those physical features. The runner panels retain only symmetric flange/base carriers and lose the entire asymmetric group. Side and top views show that support is confined to through-thickness carrier neighborhoods rather than leaking across broad planar surfaces.

## Scientific interpretation

EXP24 and EXP25 are representation-independent corroborations, not statistically independent samples: both use the same CAD and scans. Together they remove the prior source-only pose-ownership blocker for N011/N012. They do not demonstrate defect generalization.

The next unchanged action is prospective application of this gate to the remaining source-bound scans where sufficient asymmetric analytic carriers exist. If a part lacks a unique supported carrier class, inspection must be symmetry-orbit invariant or abstain. Only after that prospective gate passes should held-out good/bad truth be opened for an actual accuracy/F1 measurement.
