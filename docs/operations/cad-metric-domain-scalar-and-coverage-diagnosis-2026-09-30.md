# Metric-domain attempt: scalar transport and remaining coverage

2026-09-30. Derived-only diagnosis of the terminal metric-domain V4 attempt2.
Only released result/extraction/rejection/diagnostic records and source were
inspected. No raw CAD, private source snapshot, scan or native execution occurred.
Frozen V5 and its consumed attempt remain unchanged.

## Scalar defect: proved geometry refused by Python type boundary

The attempt retains 98 extraction candidates, 19 folded displays and two flat
previews. Eight cases refuse at `V5_INVALID_NATIVE_DOMAIN_CERTIFICATE`:
02,05,06,07,09,14,16,19. This gate is before native unfolding, so those refusals
must not be reported as new geometric impossibility or manufacturing failures.

Source trace:

- `certify_rectangle_segments` initializes `residual` as a Python float from join
  norms. Its side `deviation` comes from NumPy coordinate arithmetic.
- `residual=max(residual,deviation)` returns the NumPy scalar when a nonzero side
  deviation exceeds the current join residual. The proved receipt therefore may
  contain `np.float64` in `maximum_residual_mm`.
- `axis_rows_v5` accepts that field only when `type(value) in (int,float)`.
  A finite NumPy float within the declared tolerance fails this exact-type gate.
- JSON serialization and loading converts the retained value into Python float,
  so the persisted receipt conceals the live in-memory scalar type.

Independent pure reproduction: a four-segment metric boundary with corners
`(0,0),(3,0),(3+5e-8,40),(0,40)` and source tolerance `1e-7 mm` is correctly
tolerance-qualified `PROVED`, with residual `np.float64(5.000000014e-8)`.
Its native strict-type check fails; its JSON-roundtrip strict-type check passes.
No native shape is needed to construct this counterexample.

Additionally, replaying **all eight** released JSON extraction records through
the unchanged frozen `axis_rows_v5` passes, producing respectively
3,12,16,12,4,1,9,7 candidates. That rules out persisted malformed domain content
as the immediate adapter cause for these records. The raw live scalar types were
not retained; diagnosis rests on the exact source path, reproducible pure failure
and universal persisted replay success, not invented direct observation.

The separately versioned fix should normalize supported real scalars to built-in
finite floats at the certificate transport boundary, after excluding Boolean,
complex, array, string and nonfinite values and enforcing unchanged numeric
tolerances. Alternatively an extractor successor can publish only built-in
numeric scalars. Do not bypass residual validation or widen geometry tolerances.
Test both live producer→adapter and persisted JSON→adapter parity, including
nonzero side deviations, join deviations, negative/over-limit residuals, NaN,
Boolean and array forgeries. An adapter-only repair needs its own source freeze
and independent PRE; do not rerun or edit this terminal attempt.

## Typed certificates resolve the remaining trim gate

The extension recovers one previous baseline pair, case13 faces82/83; total
candidate count rises97→98. Against baseline127, the remaining29 groups divide
exactly into24 ordinary trim pairs, three rolled/high-turn multiwire pairs and
two ordinary-looking27degree pairs lacking both tangent flange rails.

| Case | Remaining ordinary trim pairs | Certificate reason on both skins |
| --- | --- | --- |
| 04 | 2/3,5/6,55/56 | `PCURVE_NOT_COEFFICIENT_CERTIFIED_AXIS_LINE` |
| 11 | 6/56,54/55 | same |
| 12 | 13/14 | same |
| 13 | 1/86,4/5,38/63,54/57 | same;82/83 is now recovered |
| 16 | 1/2,4/174 | same |
| 17 | 3/21,5/22,23/106,51/80,53/59 | same |
| 19 | 1/240,8/9,42/44,81/179,82/178,86/369,206/207 | same |

All48 affected skins have one wire, declared source tolerance `1e-7 mm`, and
reach a B-spline pcurve whose poles fail the exact constant-coordinate test.
These are no longer unexplained four-edge-versus-five-edge failures: the
coefficient gate is the concrete immediate cause. Because extraction stops at
the first failing pcurve, `pcurve_types` is a visited prefix, not a complete
boundary inventory; do not infer all remaining pcurve types from it.

The gate demands exactly one coordinate constant across all control poles after
mapping cylinder UV to metric coordinates `(R*u,v)`. It currently cannot
distinguish a genuinely curved/nonrectangular boundary from a representation
whose coordinate variation is tiny and safely bounded by source tolerance.
The released receipt does not retain coefficient ranges/residuals, so that
distinction remains **unknown**. These24 groups cannot be promoted from the
reason string or baseline candidate count alone.

Next source mechanism: publish bounded coefficient diagnostics for each pcurve
(degree, knots/multiplicities, trim interval, positive weight range, metric pole
coordinate ranges and maximum deviation from a candidate side). Positive rational
coefficients allow convex-hull bounds for constant-coordinate deviation on the
certified complete interval. A separately reviewed axis-slab certificate could
permit deviation no larger than the **unchanged** source physical tolerance,
provided interval continuity, endpoint support, simple-loop topology, side
coverage, nonintersection and no-hole requirements remain proven. Do not replace
the coefficient certificate with samples, decimal rounding or a wider cutoff.
Genuinely curved cuts require a separate actual trim-domain mechanism.

The three rolled/high-turn pairs (10:1/15,14:1/25,15:1/23) now have a precise
`MULTIPLE_WIRES_OR_HOLES_UNSUPPORTED` certificate, each with at least two wires.
The published count stops after detecting the second wire; it is not a full
wire census. Their unequal154degree sweeps or178degree turn and absent accepted
rails remain additional model limitations; multiple wires must be preserved as
material exclusions, not discarded.

The27degree pairs10:26/27 and14:21/22 have proved four-line rectangular domains
but only one accepted tangent plane rail per skin. Their rejection is not a trim
proof failure. Extending them requires native-connected curved transition/bend
chains with explicit operation-identity ambiguity, not deletion of the rail gate.

## Scientific and execution checkpoint

Candidate totals are not reference bend counts. The scalar defect explains an
adapter availability regression, not extraction accuracy. Recovering98→127
candidate count would not establish90percent precision or recall; blind,
independent technical reference geometry and frozen one-to-one matching remain
required. Manufacturing truth, verified units and physical metrology remain
separate gates.

Only this report was authored. Existing standing CAD-only authority permits a
separately versioned in-scope scalar transport repair and coefficient-diagnostic
successor. Any new physical evaluation requires its own freeze and independent
gates. Keep every negative result and the current terminal attempt unchanged.
