# V65 SFBOC V1 source-free fixture outcome

Date: 2026-08-15

Architecture: `V65_SHARED_FOLD_BASIN_OWNERSHIP_CERTIFICATE_V1`

Lock: `V65_SFBOC_V1_MECHANISM_LOCKED`

Decision: **TERMINAL RETIRE — locked scientific fixture contradiction**

## What was executed

The locked primitive was implemented as a research-only pre-NMS, deletion-only ownership consolidator in:

- `backend/f1_shared_fold_basin_ownership_certificate_v65.py`

The first preregistered source-free fixture was then executed without changing any locked threshold:

- `SF01_SHARP_ONE_FOLD_THREE_ROUTES`
- one equal-resolution, equal-area 70-degree dihedral mesh
- three 80h candidate segments at transverse arc positions -3h, +1h and +4h
- the ordinary-NMS eligibility veto forced false, exactly as required

All three eligible pairs were evaluated at all nine locked stations. The result was:

- positive edges: 0 / 3
- `SAME_BASIN` stations: 0 / 27
- `SPLIT_BASIN` stations: 0 / 27
- `INDETERMINATE` stations: 27 / 27
- reason at every station: `AMBIGUOUS_NORMAL`
- mutations: 0

This violates SF01's required result (one three-line cluster, two suppressions, and at least 8/9 `SAME_BASIN` stations on every positive pair). Per the locked protocol, failure of any one of the twelve fixtures retires V65 and prevents opening the prospective real-scan lane.

## Why this is a specification contradiction, not an implementation tuning issue

The locked mesh-normal construction assigns a crease vertex the area-weighted mean of the two incident face-normal projectors. For equal-area faces with unit normals separated by dihedral angle theta, the largest eigenvalue is:

`lambda_max = (1 + |cos(theta)|) / 2`

At theta = 70 degrees:

`lambda_max = (1 + cos(70 degrees)) / 2 = 0.6710100717...`

The locked ordered-profile rule requires the largest eigenvalue of every averaged path projector to be at least 0.85. A material path crossing the sharp dihedral necessarily samples the shared crease neighborhood, so the fixture necessarily becomes `INDETERMINATE` there. The required SF01 positive result and the locked 0.85 confidence rule cannot both hold for the stated equal-resolution sharp-dihedral construction.

The deterministic unit diagnostic in `tests/test_f1_shared_fold_basin_ownership_certificate_v65.py` independently verifies the eigenvalue identity and the threshold violation.

## Integrity decision

No threshold was weakened, no crease was rounded, no asymmetric face density was introduced, and no consumed real case was inspected or reused to rescue V65. The remaining eleven fixtures and the prospective real-scan lane were not executed because the protocol says the first fixture failure is terminal.

V52 remains the retained frozen baseline. V65 is research evidence only and must not enter runtime/main or production.
