# CAD extraction improvement: native rational boundary chains

2026-09-30. Controlled development extractor V10 / corpus campaign V9.
Same twenty approved STEP byte hashes / 23 aliases and unchanged frozen
127-positive AI technical reference. No scans, Gold11, email, held-outs or
unrelated data; no deployment, runtime-default or reference edits.

## Measured improvement

| Metric | Retained baseline | Original-curve successor | Boundary-chain successor |
| --- | ---: | ---: | ---: |
| Known positive capture, all20 | 98/127 = 77.17% | 113/127 = 88.98% | **120/127 = 94.49%** |
| Known misses | 29 | 14 | **7** |
| Unmatched emitted predictions | 0 | 0 | **0** |
| Complete12 recall | 77/97 = 79.38% | 88/97 = 90.72% | **94/97 = 96.91%** |
| Complete12 micro F1 | 88.51% | 95.14% | **98.43%** |
| Complete12 precision | 100% | 100% | **100%** |
| Complete12 exact count | 6/12 = 50% | 7/12 = 58.33% | **10/12 = 83.33%** |
| Complete12 part-macro F1 | 72.98% | 83.99% | **89.58%** |
| Available flat previews | 6/20 | 6/20 | **6/20** |

The candidate adds22 verified regions, preserves every original98 dictionary
and all existing flat outputs, and reduces known misses by22/29 (75.86%). The
later chain step preserves all113 original-curve incumbent dictionaries exactly.
Known-positive capture has crossed90%, but **whole-corpus90% accuracy is not
established**: eight references remain partial with447 unresolved faces. The
whole-corpus optimistic recall ceiling is94.49%, F1 ceiling97.17%; these are
bounds, not measured full-corpus point estimates. No physical Gold11 F1 claim.
AI technical reference matching is not qualified human manufacturing approval.

## Every selected CAD retained

Every emitted feature matches one frozen native face-union reference. Partial
denominators can grow; their100% known capture is not completeness.

| Case | Reference | Known | Baseline matched | V10 matched | Remaining |
| --- | --- | ---: | ---: | ---: | ---: |
|01|Partial|6|6|6|0|
|02|Partial|3|3|3|0|
|03|Complete|1|1|1|0|
|04|Partial|3|0|3|0|
|05|Complete|12|12|12|0|
|06|Complete|16|16|16|0|
|07|Complete|12|12|12|0|
|08|Partial|8|8|8|0|
|09|Complete|4|4|4|0|
|10|Partial|3|1|1|2|
|11|Partial|3|1|3|0|
|12|Complete|4|3|4|0|
|13|Complete|5|1|5|0|
|14|Partial|3|1|1|2|
|15|Complete|1|0|0|1|
|16|Complete|11|9|11|0|
|17|Complete|5|0|3|2|
|18|Complete|12|12|12|0|
|19|Complete|14|7|14|0|
|20|Partial|1|1|1|0|
|Total|12complete/8partial|127|98|120|7|

## Mechanism, math and honest geometry limits

The first graph-strip campaign was terminal negative: exact joins of decomposed
Bezier segments accepted no additional regions. That result is preserved.
V9 then retained original exactly-active rational B-splines. Positive normalized
bases plus ordered angular coefficients establish monotone end graphs; positive
weights alone do not. V10 extends this to complete native edge chains, with
two isoangular rail chains and two end chains. Both end chains must cover every
retained safe component. Vertical connectors and all internal joins are excluded
as angular bands; no bounding-box fill, coefficient snapping or relaxed tolerance.

The marker path intersects all paired skin components using world radial vectors
in the stored carrier frame, handling axis reversal explicitly. It never uses
reference labels. A review caught the legacy epsilon-union potentially filling
a tiny excluded join band; exact pairwise intersection and an adversarial test
replaced it before physical execution. Same-solid ownership, opposite material
normals and shared native end caps remain mandatory.

Sources: `backend/cad_finite_native_curve_graph_domain_v3.py`,
`scripts/extract_step_topology_bend_axes_v10.py`, controlled runner
`scripts/evaluate_vendor_cad_corpus_v9.py`; existing default paths unchanged.
The sufficient rational graph argument builds on total positivity of spline
bases, e.g. [de Boor's primary paper](https://ftp.cs.wisc.edu/Approx/tp.pdf).
Native traversal/domain assumptions follow the
[OCCT WireExplorer contract](https://dev.opencascade.org/doc/occt-7.6.0/refman/html/class_b_rep_tools___wire_explorer.html).

New markers represent **strict interior support**, not full finite bend extent.
For chains, they may move from nominal half-angle to the midpoint of a common
safe component; they are not manufacturing bend centerlines. Native units are
provisionally mm, not independently calibrated. Large kernel vertex/edge
tolerances remain explicit, not hidden as metrology accuracy. Dimensions,
angle, complete boundary shape, neutral-axis/K-factor and unfolding accuracy
are unscored. Detection-only receipts cannot enter the rectangular unfolder.

## Verification and immutable evidence

- Authored chain qualification attempt1:32passes/3 failed test expectations
  about intentionally eroded axial endpoints; retained. Attempt2:35passes.
  After exact marker-intersection hardening, fresh attempt3:36passes plus4
  subtests, outer0/known cleanup,0.6885seconds,329104KiB. All60s/4GiB caps fixed.
- Final campaign all20 outer0/known cleanup,30.1967 summed seconds,604000KiB
  peak, fixed60s/4GiB percase and1230s aggregate. All failures/refusals retained.
- Unchanged scorer completed once, outer0/cleanup known,0.0855seconds under
  fixed30s/1GiB. Same matching ontology, full census and failure semantics.
- Root inspected all20 four-view panels at the previous step, then reopened all
  five changed final panels. Other15 PNGs are byte-identical, not merely equal
  counts. No omitted native face. [Visual findings](cad-native-chain-visual-review-2026-09-30.md).
  Whole-face reference colors are not inference; actual dashed endpoints are
  separately pinned. Labels may overlap or show occluded geometry.

Final campaign:
`artifacts/vendor-cad-corpus-native-chain-v9-2026-09-30-attempt1/`.
FROZEN SHA `8572b2256b87c4d14267dd587d62a3703371299bab9d1361ace241c58a3484b3`;
PRE `4521f94c75edcd4c20ed3c161e912df56d56fb83a0a936da2f9c009bf4e9bad6`;
POST `10370430f5b445d2fa0cc4d77592cb6c47d7837e2f8952a88825dbbb93d19ecf`.
Final score:
`artifacts/cad-bend-reference-score-native-chain-v3-2026-09-30-attempt1/`.
FROZEN `dec8e1b9d4eedb59978158fa76b62370bebdb2e265bba5cce6265e110003264d`;
PRE `71c2af96ce19a9d497ce5902e3022b8e9c8f59f7763af3df01179f804221026e`;
SCORE `014b4ec72b5b8179ef46a643444a256f7657c889371ed0ebefd150c85d39d013`.
Gallery:
`artifacts/cad-bend-native-chain-gallery-v3-2026-09-30-attempt1/`;
bundle `833cfc9f14350eb07d07968239bbe27b3b71b8e87843c1124b6a3b194fa4d0f8`.
Independent final score/gallery POST SHA
`2043b6d75a9af02b9b1f53d6cc425f9d0b2712de44dd98dfbe02d40124921f43`.
A separate face-union counter reproduced every case. All120 actual marker pairs
match extraction, bundle and render manifest; all twenty PNG hashes verified.
POST accepts evidence integrity with limitations, not manufacturing readiness.

## Reproduction and next gates

Terminal attempts must not be reopened. A reproduction needs a documented
engineering reason, a fresh output path, source/archive freeze and independent
PRE; restore archived sources into an isolated environment if later files have
changed. The frozen commands and source/runtime pins are the executable record.
Qualification entrypoint: `scripts/run_cad_graph_strip_authored_v3.py --attempt
FRESH_AUTHORED_PATH --reason REASON`. Native campaign entrypoint:
`scripts/evaluate_vendor_cad_corpus_v9.py --freeze --attempt FRESH_PATH
--inventory-sha256 713c14c33c045744c1d2f9ee79a92d713fdc09d1c3d14457e3df82659a8a72ff`,
then `--execute` with that new frozen hash and independent PRE path/hash.
Derived scoring/render supervisors retain their own exact commands and limits.

Remaining7: three broad rolled regions with holes/multiple wires, two
planar-to-rolled returns lacking the two-planar-flange rule, and two case17
ordinary regions blocked by `RAIL_CHAIN_JOIN_BACKTRACK`. Retained refusal
receipts do not quantify that failed join, so do not label it a floating-point
artifact or loosen the sign test without new diagnostic proof.

Next gates: preserve complete refusal geometry in a successor; handle qualified
rail-chain uncertainty, rolled/hole domains, resolve partial reference regions,
and extend exact finite-boundary unfolding with an independent flat oracle.
The90% known-detection milestone does not satisfy95% exact count, worst-case
coverage, CAD-agnostic unfolding, calibrated metrology or factory release.
