# V67 NCMM pair-domain geometric erratum E1

## Outcome

The V67 implementation remains source-only and all detector constants remain
unchanged. The original frozen source-free fixture text contains two positive
pair-boundary cases that cannot exist under its own simultaneous endpoint
constraints:

- `F22A_ANGLE_EQ` requires a positive pair at angular separation `pi/24`.
- `F22G_VARIATION_EQ` requires a positive pair at endpoint-distance ratio
  `3/4`.

They are not detector failures and cannot be repaired by a different mesh or
fixture triangulation. They are contradictions in the fixture specification.

## Proof: F22A

The endpoint frame lock requires, at each endpoint,

`e/d in [7/8, 9/8]` and `|n|/d <= 1/8`.

Therefore the difference between the two normalized endpoint separation
vectors is at most

`sqrt((1/4)^2 + (1/4)^2) * d`.

The same specification requires `d/L <= 1/16`, so this difference is at most

`sqrt(2) / 64 * L = 0.02210 L`.

For two straight candidate lines separated by angle `theta`, projection on the
mean axis makes the endpoint separation vectors differ by

`2 tan(theta/2) L`.

At `theta = pi/24`, this is `0.13109 L`, almost six times the allowed maximum.
Consequently no pair can satisfy both `F22A` and the endpoint frame lock.

## Proof: F22G

The smallest endpoint magnitude permitted by the endpoint frame lock is
`7d/8`. The largest is

`sqrt((9d/8)^2 + (d/8)^2)`.

Thus the endpoint-distance ratio is always at least

`(7/8) / sqrt((9/8)^2 + (1/8)^2) = 0.77302`,

which is strictly greater than `3/4`. Equality at `3/4` and a below-boundary
case cannot enter the positive pair domain.

## Narrow correction

This erratum changes no production predicate, constant, candidate decision,
normal-cycle profile, matching rule, or resource cap. It changes fixture
expectations only:

- `F22A` and `F22B` verify that the requested 7.5-degree boundary is shadowed
  by the stricter endpoint domain and therefore keeps the pair.
- `F22G` and `F22H` verify analytically that the requested 0.75 boundary is
  shadowed by the endpoint domain.
- Feasible exact boundaries for overlap, upper separation, and endpoint
  direction change remain end-to-end positive/negative tests.

This is marked `GEOMETRIC_ERRATUM_E1` in the fixture lock. It is not evidence
for accuracy and does not authorize changing thresholds after seeing real
results.

## Synthetic-corpus status

All previously generated rectilinear synthetic parts remain quarantined. They
are visibly unlike the real LiDAR corpus and created misleading false-positive
pressure. No synthetic candidate result is used by V67 selection, preflight,
or any generalization claim.
