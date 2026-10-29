# V33 PNFM V1 terminal source-free result

Date: 2026-08-14

`V33_CANDIDATE_ALIGNED_SURFEL_VARIFOLD_V1` is terminally retired at the first
frozen nuisance mismatch. The terminal token is `V33_NUISANCE_STOP`.

## What survived before retirement

- Static mechanism audit: `24/24`, token `V33_PNFM_MECHANISM_LOCKED`.
- Clean 60-degree fold: `1/1` as expected.
- Three physical-surface-equivalent remeshes (all diagonals flipped,
  checkerboard diagonals, skew lattice): each `1/1` as expected.
- Axially twisted normal-fan adversary: `0/0` as expected.
- Complementary-skin fragmentation positive: `1/1` as expected.
- Phase-window fragmentation positive: `1/1` as expected.
- SN01--SN05 nuisance cases: each `0/0` as expected.
- Every completed fixture was manually inspected at native 4096 x 4096
  resolution before the next fixture was invoked.

## Terminal mismatch

Fixture: `SN06_NONMONOTONE_BEAD_0_70_35_90`

- Expected pre-NMS/final: `0/0`.
- Observed pre-NMS/final: `1/1`.
- Decision: `PNFM_PASS`.
- Result SHA-256:
  `e19c3fdd203e7ecae0022f203da480d607a2532d428d245c771155b1e19bb415`.
- Cache SHA-256:
  `7145b0275ef2aea8de650453c6da2f5cea4a2031458ceb01b70b7f1ea4384731`.
- Post-invocation image SHA-256:
  `fbdb3667bb7e9d91169682a7007decea545bd2a7d70310a2bb0477d29176c6a3`.

The native image was inspected after the mismatch. It is not corrupt. It
visibly shows a smooth non-monotone bead with two broad outer normal
populations and a rounded center. The frozen candidate lies on the bead and
the low/core/high populations span its full axial length. Those properties
make the unordered projective normal-fan moments look fold-like even though
the cross-section turns through the non-monotone sequence `0, 70, 35, 90`
degrees. The emitted red line is therefore a physically coherent output of
the frozen PNFM hypothesis but a false bend under the fixture definition.

The decisive limitation is representational: PNFM records projective
position-normal mass and radial multiplicity, but it does not preserve the
ordered signed progression of normals across the candidate-normal
cross-section. SN06 and a true fold can share the accepted moment summary
while differing in that ordered progression.

For reference, the frozen V32 architecture rejected the same nuisance using
`V32_NORMAL_TURN_MEASURE`. Reusing or tuning that threshold inside V33 after
seeing SN06 would be a prohibited rescue, not an independent test.

## Integrity and stop action

- Gate-state SHA-256:
  `533a5c5ad097c6e3ae7007b78493ee61f9671e4d31d2c2c7ceb13dc137336e0a`.
- Engine SHA-256:
  `44c9931fc6ee74e3e17c704fa1afdb789222c1df7e2357b8479911263b6bfd66`.
- Lock SHA-256:
  `382a44d513b4f76135f24adea53488cf3862f8335dbe02e3fd7449af19edffad`.
- Gate driver SHA-256:
  `26102788c10435427940b7e77f217cad65ed62d923548ebc37d78de8cf15b54e`.
- Repository HEAD at execution: `5df325e33e9883353cbdefaed107742ba5c406f1`.

No V33 fixture was rerun, replaced, repaired, or reordered. No threshold,
predicate, renderer, candidate, or cache record was changed after invocation.
SN07--SN12, SF26--SF27, and SD01--SD04 remain marked
`NOT_RUN_AFTER_TERMINAL_FAILURE`. V16 remains retained and unchanged. V33 is
research-only and is not eligible for runtime/main or production.

## Successor constraint

Any successor must be a genuinely new locked architecture. It must explain
before execution how it distinguishes a monotone material fold from the
SN06 non-monotone bead without using gold, part identity, filenames, expected
counts, CAD, or post-hoc per-fixture constants. V33 may be used only as frozen
negative evidence; it may not be rescued.

