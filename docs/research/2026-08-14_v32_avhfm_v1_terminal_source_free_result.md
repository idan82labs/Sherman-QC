# V32 AVHFM V1 terminal source-free result

## Ruling

`V32_AXIAL_VORONOI_HARMONIC_FOLD_MEASURE_V1` is terminally retired on its first authorized physical invocation.

- Fixture: `SF01_FOLD60_MESH_C2`
- Pre-NMS lines: `0` (expected `1`)
- Final lines: `0` (expected `1`)
- First terminal reason: `V32_AXIAL_LEAKAGE_MEASURE`
- Terminal token: `V32_PHASE_INVARIANCE_STOP`
- Architecture invocations consumed: `1`
- No phase variant, nuisance fixture, metamorphic, replay, or stress execution was reached.
- No rerun, threshold rescue, V32b, or replacement positive is authorized or performed.

## Frozen observed accounting

- Candidate-wide graph: `7,738` vertices, `44,894` edges
- PCG iterations: `99`
- Axial events: `97`
- Total current stage measure: `24.0s`
- Axial-locality stage measure: `12.0s` (required `19.5s`)
- Local reasons: `48 × V32_AXIAL_LEAKAGE_MEASURE`, `49 × V32_TAIL_COVERAGE_MEASURE`
- Final accepted measure: `0.0s`
- Longest accepted run: `0.0s`
- Maximum gap: `24.0s`

## Native-resolution visual adjudication

All three 4096×4096 post-invocation artifacts were inspected individually.

1. `harmonic_current_multiview.png`: the global potential crosses the intended fold, but current-bearing mesh edges form a dense axial/diagonal lattice along the otherwise valid fold.
2. `axial_measure_audit.png`: the current predicate covers the full 24s audit interval, while the edge-wise axial-leakage predicate retains exactly 12s and therefore fails well before the local fold predicates can establish a certificate.
3. `final_overlay.png`: the unchanged V16 candidate line visibly lies on one clean, continuous fold, yet V32 emits no rescue.

The visual contract therefore independently makes the numeric false negative terminal: a visibly valid continuous fold was rejected.

## Diagnosis

The failure is not the V31 station-phase defect. It is a new discretization defect in V32's edge-wise axial leakage primitive. On a triangulated surface, a diagonal edge can carry legitimate cross-fold potential difference while its geometric edge vector also has a substantial axial component. The frozen leakage formula counts that diagonal component as axial detour. Consequently, ordinary mesh orientation is conflated with current routing along the bend axis.

This explains the simultaneous evidence:

- the harmonic field visibly follows the correct fold;
- the full interval passes the current floor/balance stage;
- only half the interval passes axial leakage;
- the original clean positive is rejected before any phase variant is reached.

Under the Pro ruling, this first-positive mismatch retires V32 and leaves the phase-free harmonic-flux hypothesis unsupported. Any successor must be a genuinely different source-only family; changing the leakage threshold, projecting diagonal edges differently inside V32, rerunning, or adding a favorable phase would be an unauthorized rescue.

## Integrity hashes

- Precommit manifest: `44e0a6e839b5fa8b6719012ec3064f0244146700c96f4a86c126319957e234cf`
- Canonical result: `88391a703c60165825e419620498e73a42e65d00b44b789bd3ed1169ba7b2f52`
- Binary cache: `4e5bf66ffb1ff8ed854a27ed1bb1f62b216e874ff1c9184f636cdfc9d0ecc95b`
- Terminal gate state: `4dd92192083619840fb7f5146e04b91d3cb676726893a07b00b8f21fedd98736`
- Harmonic-current visual: `8e563c3e19f9581f81acc9a20a9d9e18be2a1abd0d99231cdd51a279d4a80661`
- Axial-measure visual: `817303b8593e80a697e5e60495316ef38ee471bcd6ad26858afa14c826b3ba12`
- Final-overlay visual: `37867210e915470035276933f08e644ec87d7ae44c49c0d25bb3c97029dcb29a`
