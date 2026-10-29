# Connected depth V2: localized support-cover failure — 2026-09-11

## Finding and scope

The retained L-R and L-A failures are **existing-cover eligibility failures**, not
observed cap contamination, bad V2 circle fits, wrong sweep signs, or failed V2
endpoint localization. Every admitted localized context produced a measurement.
The cap filters consistently removed all 100 cap-only vertices. Each localized
case then had several overlapping but non-nested raw-role records; no single
record contained the whole component's arc union. The exact-existing-cover rule
therefore refused its one component before any V2 cover refit.

The current V2 implementation obeyed its narrower declared contract. This is a
limitation of that contract for context-dependent boundary support, not evidence
that it should publish a best member, discard inconvenient members, or silently
assemble an unqualified union.

This analysis reads only the explicitly released terminal attempt
`docs/research/cad_engine_v1_synthetic_checks/two_plane_depth_v2_connected-haebyimq`:
`COMPLETED.json`, `ARTIFACT_INVENTORY.json`, `OUTPUT.txt`, all eight measurement
results, and the two localized registration results. Reads were capped and
regular-file/no-follow/nonblocking checked. Scientific payload sizes and hashes
were checked against the inventory; measurement decoding rejected duplicate JSON
keys/nonfinite JSON constants. Analysis used standard-library JSON, hashing,
bitmaps, integer set operations and scalar summaries only. No engine import,
refit, raw-coordinate/physical input, private cache, native call, test or replay
was accessed. Only this new document was written; candidate sources and the old
BLOCKED native PRE were preserved.

This is a failure-mechanism analysis, not the separate whole-source/archive POST.
No hypothetical union was fitted or asserted to pass. Source predicates and
unreached oracle details remain mapped in
[the source map](2026-09-11_two_plane_depth_endcap_diagnostic_source_map.md).

## Terminal denominator and exact failed assertion

The terminal receipt records exit 1, process group 49810 confirmed empty, no
budget/primary/cleanup error, all 97 source hashes consistent before/after, and a
complete 32-file science inventory. `OUTPUT.txt` contains 6 PASS / 1 FAIL in
5027.15 seconds; the one failed pytest outcome is the complete packet assertion,
not a setup/import failure. Worker CPU is 5023.520858 seconds, sampled/worker peak
RSS 1,253,244,928 bytes and parent wall 5027.271550166886 seconds. The original
6000 CPU / 6595 work / 6600 parent / 4 GiB / 1 MiB output limits were not hit.

All four registration-stage oracles passed. V2 remains 2/4 required curved
positives, U-R and U-A only. All four same-context V1 diagnostic oracles failed;
they are retained baselines, not additional required positives. The retained
older composite comparator also has 2/4 positives and its historical import-
closure limitation remains disclosed. This run is neither a 4/4 success nor
physical/native qualification or F1 improvement.

For both localized V2 records, the error is exactly:

`AssertionError: ['NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS']`

The first `assert usable` in
`tests/test_cad_guided_support_cover_connected_v2.py:378` fails because no
`support_groups` row is `MEASURED_CONDITIONAL`. The following were **not reached**:
radius/sweep accuracy and anti-nominal checks, measurement/association oracle
checks, nonempty localized prediction assertion, generator endpoints and axis
endpoints. Publication and safety checks did complete successfully. An accurate
number in retained per-context evidence does not substitute for a published V2
measurement or count as those unreached checks passing.

| Case | Completed measured windows | Distinct raw-role records | Complete-plan components | Existing cover | V2 cover refit rows | V2 predictions / positive oracle |
|---|---:|---:|---:|---|---:|---|
| U-R | 48/48 | 18 | 1 | Yes | 261 | 1 / PASS |
| U-A | 48/48 | 24 | 1 | Yes | 261 | 1 / PASS |
| L-R | 12/12 | 5 | 1 | No | 0 | 0 / FAIL at first assertion |
| L-A | 16/16 | 7 | 1 | No | 0 | 0 / FAIL at first assertion |

Every window has `status=MEASURED_CONTEXT` and empty reasons. Every source V1
record was subsequently cleared by
`AMBIGUOUS_SHARED_RAW_OR_DUPLICATED_POSITION_SUPPORT`; there are 18/24/5/7 such
records respectively. V2 preserves that nested report and may only publish its
own freshly qualified group projection. Canonical JSON equality of each nested
V1 report to its separately retained V1 result and the nested SHA were checked
independently for all four cases, not just accepted from the pair flag.

The positive U-R group reports radius 5.300000000000005 mm and sweep
89.99999999999999 degrees; U-A reports radius 5.000048286597504 mm and canonical
signed sweep -90.49958264287886 degrees. U-A's passing negative sign illustrates
the preserved magnitude oracle, not a manufacturing handedness conclusion.
Both U cases publish one finite sample-supported instance, but their nonlocalized
fixture oracle does not check the localized truth endpoint predicate.

## Cap exclusion worked on these retained contexts

For **all 12 L-R and all 16 L-A filters**, the retained values are:

| Filter evidence | Value in every localized context |
|---|---:|
| `cap_proposed_triangle_count` | 196 |
| `conflicting_triangle_count` | 0 |
| `cap_like_triangle_count` after conflict veto | 196 |
| `excluded_row_count` | 100 |
| `kept_row_count` | 550 |
| Decoded excluded bitmap | Exactly raw IDs 550–649 |

The original fixture source assigns IDs 0–549 to the inner sheet, with axial
station `row_id % 11`; IDs 550–649 are the two added cap-only radial layers at
the lower end. Its 50 welded cap/sheet boundary IDs `11*i` remain eligible as
sheet points, as required by the all-incident-triangles rule. The decoded
exclusion bitmap leaves all 550 sheet vertices; it does not remove those welded
boundary vertices. No fixed arc or flange role in either localized result
contains any ID from 550–649.

Thus the earlier provisional possibility of a conflicting proposal preventing
cap removal is ruled out **for these retained filters**. It remains a general
conservative behavior of the algorithm, not the mechanism observed here. No
proposal pruning, extra cap deletion or filter-cone relaxation is supported by
this result.

Both localized owned preparations contain 17 qualified supports, all carrying
the literal provenance `seeded_v1:SEEDED_REGION`, with composite plane IDs.
L-R classifies IDs 1 and 6 as cap-like; L-A classifies ID 1. The complete 196-
triangle/150-vertex cap support in both cases has triangle-row SHA
`5efd0e59649682691a4dc6e6864ee76fe3bd2e19b4b3eb380bb824d18332985b`
and vertex-row SHA
`cc84f7fb26ea3a5f11e90861efb7889167f0512d1d37ff8e124e8b02aa61c9be`.
L-R's second cap proposal has 186 triangles/143 vertices and contributes no
additional excluded raw rows beyond the complete cap.

The cap normal-angle bounds are 0.017300477057378416 radians (L-R ID 1),
0.02156000764636228 (L-R ID 6), and 0.017354210387023564 (L-A ID 1).
L-R ID 1's retained context-axis angles range from 0.00014446658652499952 to
0.0016636115281725395 radians; L-A ID 1 ranges from
0.00002789327821957017 to 0.00030010517725981945. All pass the unchanged complete-
cone 15-degree role gate. These are context axes and stipulated proposal cones,
not verified scanner pose/normal uncertainty.

Global exclusion is identical while ROI exclusion differs: L-R removes 9–34
cap-only rows inside each nominal crop and retains 369 or 370 local rows; L-A
removes 14–49 and retains 370. This is not inconsistent accounting: the 100-row
filter is intersected with each different context's ROI. The filter did not
cause the differing sheet memberships described next.

The two localized registration records retain 48 and 44 completed candidate
evaluations, from which 12 and 16 measurement contexts qualify. Scheduler/context
V2 completed 252 and 254 ordered pair matches. Both retain
`accepted_registration=False`, no single transform, no verified stable datum,
unverified acquisition side and unknown global pose uniqueness. The serialized
registration and owned-role evidence hashes were recomputed and matched to both
measurement arms. A completed context remains a conditional row-selection
hypothesis, not an accepted physical pose.

## Exact set-level rejection: non-nested boundary support

Let A_i, F_i0, F_i1 be a record's original arc and canonically ordered flange raw
ID sets. All localized records share the same seven-field fit policy. The two
flange roles remain disjoint in raw-ID space, and the union has no arc/flange or
flange/flange raw-ID overlap. Every role has a large common intersection.

The complete plans do not report `CROSS_ROLE_POSITION_CONFLICT` or incompatible
policies. Each reports only `NO_EXISTING_WHOLE_COMPONENT_RAW_SUPPORT_COVER`.
Their `member_comparisons=0` is consistent with rejection at the candidate
union-equality check, before any flange-permutation candidate could be tested.

| Case | Arc/flange0/flange1 union sizes | Intersection sizes | All variable IDs' axial indices | Largest member arc set |
|---|---|---|---|---:|
| L-R | 195 / 53 / 53 | 184 / 46 / 48 | Only 0 and 10 | 190 |
| L-A | 209 / 55 / 55 | 171 / 45 / 45 | Only 0 and 10 | 190 |

These counts alone prove that no existing arc set equals the arc union, regardless
of flange permutation or radius fit. The source requires equality to the entire
component union, not merely a maximal member or near-equality. The shared arc
intersections also establish why all records belong to one ownership component
in each case. No favorable disconnected subset was excluded from the denominator.

The exact member ledger below abbreviates IDs only for readability; full IDs
remain in the hash-bound payloads. Missing counts refer to the whole role union,
not to CAD/full physical coverage.

| Case / record prefix | Context indices | Arc/F0/F1 rows | Missing union rows A/F0/F1 |
|---|---|---|---|
| L-R `70e5b174` | 0,1,2,3,4,28,35,37 | 190/50/50 | 5/3/3 |
| L-R `117b8337` | 29 | 190/49/50 | 5/4/3 |
| L-R `382a39b6` | 30 | 189/50/51 | 6/3/2 |
| L-R `ec326726` | 36 | 190/50/50 | 5/3/3 |
| L-R `4071487b` | 38 | 190/50/51 | 5/3/2 |
| L-A `efc31a72` | 0,1,29 | 190/50/50 | 19/5/5 |
| L-A `dcd0723a` | 2,3,4 | 190/50/50 | 19/5/5 |
| L-A `7f70342c` | 16,17,18 | 190/50/50 | 19/5/5 |
| L-A `2e870412` | 19,37 | 190/50/50 | 19/5/5 |
| L-A `cfd2bfd0` | 24,25,26 | 190/50/50 | 19/5/5 |
| L-A `032d229d` | 27 | 190/50/50 | 19/5/5 |
| L-A `513f481c` | 28 | 190/50/50 | 19/5/5 |

For L-R the variable arc IDs are exactly
`143,153,197,208,209,220,263,286,296,341,351`; variable flange IDs are
`54,55,66,77,87,88,98` and `407,417,439,440,450`. Every one has remainder 0 or
10 modulo 11. The same check covers all 38/10/10 variable arc/flange IDs in L-A.
These are the construction's z=0 or z=30 stations; no interior axial station
changes role membership between these records. The 100 cap-only exclusions are
constant, so this variability is downstream context-conditioned sheet selection.

The source mechanism is compatible with finite boundary sensitivity: both
`_crop_rows` and `_measure` apply context-transformed nominal span gates with
numerical-only guards; measured fit accuracy does not certify the pose used for
those gates. Small differing context axes/offsets can change boundary inclusion
while fitting the same interior geometry. The recorded ID differences and
source predicates establish the boundary-selection limitation. This analysis
did **not** reconstruct raw points or re-evaluate every per-row inequality, so
it does not assign each individual exclusion exclusively to the axial gate
rather than another context-conditioned predicate. The immediate cover rejection
does not require that stronger claim.

L-R's plan charged 1450 graph-row visits, 754 graph links and 3902 conservative
cover-row checks; L-A charged 2030, 1121 and 5460. Neither exhausted a set-work
cap. Both plans are `COMPLETE`, top-level measurement computation is `COMPLETE`,
and the one group is `UNAVAILABLE` with null cover ID, measurements and finite
support. Both `cover_refit_row_visits` counters are zero. Therefore no V2
all-point common-model or finite-local-coverage evaluation was performed on the
301-row L-R union or the 319-row L-A union.

## What the retained numbers establish—and do not

The localized per-context fits survive in source associations. Across all 12
L-R contexts, radius ranges from 5.299893975102891 to 5.300223132204016 mm and
sweep from 90.00067473968765 to 90.00122346357756 degrees. The largest retained
circle residual is 0.0017323518930343633 mm. Across all 16 clean L-A contexts,
radius ranges from 4.999999999999958 to 5.000000000000047 mm and sweep from
90.5 to 90.50000000000001 degrees; maximum circle residual is about
1.014e-14 mm. These are consistent with the respective construction's quantity
targets and distinguish this failure from obvious nominal-answer substitution.

They do not prove that one fit to the union would pass. Individually compatible
subsets can have incompatible centers/axes or combined residuals, and adding
points can alter the fitted geometry and finite-support disposition. Original
V1 canonical deviation fields also survive, but its canonical measurement and
finite objects were erased by ownership veto; no erased carrier is reconstructed
or counted. All quantities remain conditional on position units, selected roles
and single-side/model assumptions, with no calibrated uncertainty interval.

There is no V2 radius, signed sweep, finite segment or generator in either
localized output to call numerically inaccurate. Likewise, there is no observed
failure of local gap/slab gates or the .05 mm noisy L-R/.02 mm clean L-A endpoint
oracles. Those gates/oracles are unreached, not passed and not diagnosed as the
cause. Observed manufacturing count stays null and conformance indeterminate.

## Smallest general successor to consider, not an implementation release

The narrow correction seam is support aggregation. A separately versioned
**whole-component raw-role union proposal** can address non-nested supports
without changing the plane seeder, cap filtering, registration, radius solver,
sampling policies or endpoint oracle. It must not pretend an existing member
contains the union. The current exact-cover V2 and this terminal RED remain
immutable evidence.

Minimum conditions for that future proposal:

1. Preserve the maximal component built from all arc-related claims, including
   failed members; preserve cross-role exact-position veto and all lineage.
   Determine one globally consistent unordered-flange assignment. Ambiguous
   role mappings/cycles or contradictory ownership refuse the entire component;
   pairwise compatibility alone is not proof.
2. Form each role from **all** original raw IDs after that assignment, with no
   invented rows, favorable subcomponent, intersection trimming, discarded
   boundary observations, averaged answers, CAD radius, or selected best member.
3. Preflight union sizes/work/report bounds; perform one fresh all-point
   canonical plane/circle/tangency fit and the unchanged finite-support checks
   on the complete union. Any failure leaves the entire component unavailable.
   Near cylinders, bridging memberships, disjoint spans and duplicate-position
   aliases cannot be waived because the original members fitted separately.
4. Preserve every original context, raw role, status and association. Publish
   only a successful union's conditional geometric carrier, never concatenated
   nested/member predictions. The claim must change from
   `EXISTS_COMPLETED_EXISTING_COVER_CONTEXT` to an explicit common-sample-model
   statement over the union of completed context supports. No one pose/ROI,
   physical surface identity, manufacturing count, joint configuration, inner/
   outer verification or physical trim is thereby established.

The smallest independent simulation should first use a clean invented extrusion
with two or more non-nested endpoint-supported role sets: no existing cover,
unchanged current V2 refusal, and a candidate union admitted only through the
complete fresh fit and unchanged finite/endpoints contract. Add conflicting
flange roles, a near-cylinder/bridge alternative, disconnected axial support,
duplicate positions, perturbation and late whole-refusal controls. Close/ambiguous
geometries that fit one residual tube remain conditional sample-model evidence,
not known distinct or identical events. Freeze these semantics and denominators
before executing a new successor. No improvement or run-time saving is inferred
from this unexecuted correction direction.

A separately authorized prospective **native integrity diagnostic** may be
planned after this classification under its new protocol, preserving the 2/4
RED and any honest measurement refusal. That is not conversion of this gate to
qualification GREEN. The old native PRE remains blocked/unchanged; it does not
authorize that diagnostic, a replay, source edits or physical access.

## Retained evidence identifiers

All science paths below are under the named attempt's `ARTIFACTS/`. The complete
inventory records 123,963,276 bytes across 32 files; this analysis independently
checked the following ten scientific payloads, not the whole archive/source
closure. Original error/timing output was read in full within its 1 MiB cap.

| Record | Bytes | SHA256 |
|---|---:|---|
| `COMPLETED.json` | 13101 | `9807c4a1e002a9339249d66b682854fcb91c715e194f7de39e2d14f5016fddc4` |
| `ARTIFACT_INVENTORY.json` | 6126 | `8badfb2f1bbf74c65493c6f47e76567b1a1b1f01db69aeb41d414f6eb4922e82` |
| `OUTPUT.txt` | 18415 | `7f890dbc250dda1f9d4014ff97985549b2307ef6e077f3e6208757a617d625a8` |
| `MEASUREMENT_RESULT_00_V1.json` | 629296 | `57d13432209c1a9c5b1b916e775a1ae1ac77210c0a779f4afd62952dfdef508d` |
| `MEASUREMENT_RESULT_00_V2.json` | 671700 | `c005444577db7470296a66c2035230948410418e1814470d9e3ffda66c63dd88` |
| `MEASUREMENT_RESULT_01_V1.json` | 717392 | `8a99ef17a8963fcf86c8872b1c883780153bd8e3681c81c40a09b006fd5addab` |
| `MEASUREMENT_RESULT_01_V2.json` | 761721 | `bc00c3c02a055d2ef2fb88d46ae207982ff0647f9d8c43c30d4784e9ab433945` |
| `MEASUREMENT_RESULT_02_V1.json` | 154190 | `1a382e29e785e73e4a8d8a19d99684bf651cd3d6c51b5012a16d037bf7f1c2a1` |
| `MEASUREMENT_RESULT_02_V2.json` | 163691 | `42e5fc8ef34f2d57d5cd0f4307a0b7d12e8e863c50b24da69e9ef23cce5f0467` |
| `MEASUREMENT_RESULT_03_V1.json` | 204235 | `84175f88c2b7743f7831fcddad75188cebe7146ec6c21f82f229ca30ff520bc3` |
| `MEASUREMENT_RESULT_03_V2.json` | 215914 | `8e3a32f3ab406e67f97bdd50a59210582a0a4110ca322c832d756f1222b5e5bc` |
| `REGISTRATION_RESULT_02_SHARED.json` | 27855463 | `75f31edc73ec1dca098b9f1889afe788dd522ae9b907df9f1590a1df68e40ee7` |
| `REGISTRATION_RESULT_03_SHARED.json` | 27701744 | `ae6808826a1ac61a9e810570bfe331ba92bbf571341c6633173096b45d5beb0d` |

L-R registration-JSON/owned-role evidence digests checked against both arms are
`5d8ff02b37d88ce59a788c2e7c89541ac4b44c343cf274ddf43688c375a918d9` /
`a9dce4b56cd7dd62cccf77dc2430d7841ceb5319510e157a74c5c7325584f8e0`;
L-A equivalents are
`0cf0c188a98c56a63bf924f6a2e71b796f9ddbeab56f9c4f628967389a2a1b0a` /
`37e83622bcc797d04b4b3bd0e40cf1091ce43e659f5349d6c36e41bcc735f1c1`.
These bind retained in-process evidence, not external scan-file authenticity or
physical model truth.
