# F1 V12 mesh-route and conservative consensus-midpoint tranche

Date: 2026-08-13

## Outcome

V12 is retained as a source-only research selector. It fixes a route-plumbing
defect for bounded-decimation meshes and applies a conservative representative
repair that cannot add bends. The new mailed scan `34499000_B.ply` scored an
exact 3 TP / 0 FP / 0 FN (F1 1.0) against a manual visual precommit frozen
before inference. Native-resolution inspection confirmed that all three orange
model axes track the three cyan manual bend axes and avoid the large window,
slots, reliefs, sheet faces, and silhouettes.

## What failed first

The V11 unified call incorrectly sent this 942,394-vertex mesh through the
non-full-mesh V6 fallback because bounded decimation had been used. That
produced 0 TP / 4 FP / 3 FN under the strict four-millimetre gate, even though
visual inspection showed two long axes just outside tolerance. The existing V9
bounded-mesh route improved the frozen result to 2 TP / 1 FP / 1 FN (F1
0.666667).

A first V12 repair snapped a selected cylindrical representative completely to
a nearby three-source scan-mesh representative. It made the new scan exact but
failed the development regression: one historical true positive was lost and
F1 moved from 0.668852 to 0.666667. That version was rejected.

## Retained rule

For bounded-decimation meshes, V12 uses the existing V9 complexity route. A
selected axis may move only when all its supporting sources are cylindrical,
and a nearby scan-mesh-adjacency line has at least three source families,
two-sided transition score at least 0.85, axis angle at most 4 degrees,
transverse separation at most 5 mm, and shorter-span overlap at least 0.80.
The output representative is the endpoint-wise midpoint of the selected and
consensus axes, not the consensus line itself. Therefore the selected axis can
move by at most 2.5 mm under this rule. Full-resolution meshes continue through
the V11 representative-repair and short-fragment-pruning route. Non-mesh cases
remain on V6.

The rule consumes no part identity, sender count, expected bend count, CAD,
drawing, labels, gold, benchmark feedback, or runtime/production state.

## Evidence

- New mailed scan `34499000_B.ply`: 3 TP / 0 FP / 0 FN, F1 1.0, exact.
- Dense-mesh development regression (20 cases): V9 and retained V12 are
  identical at 102 TP / 74 FP / 27 FN, F1 0.668852; zero true-positive loss.
- Untouched `50552000_04.ply`: 4 TP / 0 FP / 0 FN, F1 1.0, exact.
- Prior untouched `488...` scan: 4 TP / 0 FP / 0 FN, F1 1.0, exact.
- Focused selector tests: 11 passed across V9, V10, V11, and V12.

The `34499000_B` outcome is not an independent confirmation of the midpoint
repair because its failure analysis motivated the repair. A different untouched
scan is still required for independent V12 confirmation.

## Independent confirmation and V13 follow-on

An untouched second mailed scan, `38169000_04.ply`, was downloaded after V12
was frozen. Its sender count was quarantined. Manual visual inspection of the
raw six-view plus a PCA-aligned cross-section froze three closely spaced long
return bends before inference. V12 abstained (0 predictions), producing 0 TP /
0 FP / 3 FN. Candidate diagnostics showed this was a selector failure rather
than a representation failure: the pool contained long axes within 1.3--1.8 mm
of all three frozen bends, while the highest V9 score was 0.423, just below the
fixed 0.45 threshold.

V13 adds one guarded action: only when the complex bounded-mesh precision pass
returns no axes, rerun the same selection at threshold 0.35. This condition did
not occur in any of the 20 dense development meshes, so their aggregate output
and F1 remain unchanged. On the untouched scan V13 recovered 2 TP / 0 FP / 1
FN (F1 0.8). Native-resolution overlay inspection confirms both recovered axes
track real return bends and do not drift onto the circular opening, lanced
slots, holes, or sheet boundary. The main-sheet-to-first-return bend remains a
clean false negative; no further threshold relaxation was accepted because it
rapidly introduces aliases.

V13 is therefore retained as a bounded recall improvement, not as an exact
solution. The next optimization target is source-only discrimination of the
remaining long parallel bend from same-edge cylindrical aliases.

## V14 continuous seed-family coverage

The remaining M001 bend was present in the candidate pool as
`cylindrical_standard::CB3_899`: 537.3 mm span, 1.27 mm transverse error,
1.15 degree axis error, complete finite support, complete longest support run,
complete endpoint support, and two cylindrical source families. It belonged to
a seed family not represented by V13's two selected axes.

A broad unrepresented-family recovery was rejected because it added one false
positive on development case C028. The distinguishing evidence was continuity:
the C028 alias had finite support 0.36 and longest-run support 0.20, while the
new scan's true axis had 1.0 for finite support, longest run, and endpoints.

V14 therefore runs only after V13's guarded no-output retry successfully
returns at least one line. It adds at most one highest-scoring representative
per still-unrepresented nonempty seed family, requiring span at least 70% of
the scan bounding span, selector score at least 0.15, at least two source
families, all three support-continuity fractions at least 0.90, two-sided normal
angle at least 45 degrees, and transverse angular-bin fraction at least 0.75.
The rule uses neither the number of already selected axes nor an expected bend
count.

On untouched `38169000_04.ply`, V14 recovered the remaining long return bend
and scored 3 TP / 0 FP / 0 FN, F1 1.0. Native-resolution overlay inspection
shows three distinct orange tracks on the three cyan return axes, with no line
on holes, slots, faces, or silhouettes. Across all 20 dense development meshes,
the V14 trigger remained inactive and aggregate behavior stayed exactly 102 TP /
74 FP / 27 FN, F1 0.668852. The earlier dense mailed `34499000_B.ply` also
remained exact at 3 TP / 0 FP / 0 FN.

Because `38169000_04` motivated V13 and V14, its exact result validates the
mechanism but is not an independent V14 generalization confirmation. Another
untouched scan is required before broadening the rule further.

## Independent confirmation and retained V15 tranche

The next mailed scan, `50566000_04 1.ply` (SHA-256
`8bb74bda6756561760924902992b2b8a73273f6dc41b136f5afcf347441a4bfa`),
had no prior repository footprint. Only its PLY was downloaded. The sender's
bend count was visible in the intake email but quarantined and not used; the
linked drawing, CAD, labels, and expected geometry were not opened.

Before inference, raw six-view inspection and a separate dominant-plane
intersection audit froze five physical axes: four longitudinal web/return
folds and one short transverse closing fold at the fork junction. Hole edges,
slots, free perimeters, parallel faces, and scan-boundary aliases were
explicitly rejected. The manual precommit was serialized before candidate
generation or selector execution.

Frozen V14 failed on this independent scan at 3 TP / 3 FP / 2 FN (F1
0.545455). Visual review showed that its six outputs represented only three
physical folds: two duplicate lanes on each long web fold and two competing
lanes on the short transverse fold. Both fork-arm folds were absent. This was
therefore a selector failure, not a representation failure: the pool contained
scan-supported plane intersections within 1.2--1.4 mm of the two missed axes.

V15 retains three bounded source-only corrections:

1. Nonmaximum suppression uses 7 degrees and 6 mm instead of 6 degrees and
   5 mm. Across the 20 bounded development meshes this alone preserves all
   102 true positives while reducing false positives from 74 to 36.
2. A missing member of a paired plane-fold family may be recovered only when
   two scan-supported plane intersections share one incident face, are each at
   least 40% of the part span, have complete finite/longest-run/endpoint
   support, at least 400 points and 10 mm cross-support width on both incident
   faces, p90 line distance at most 2 mm, intersection angle at least 75
   degrees, mutual axis angle at most 2 degrees, span ratio at least 0.90, and
   mutual transverse separation at least 15% of the part envelope. The new
   fork pair is separated by 17.77% of span. A development look-alike at 9.05%
   is rejected.
3. A selected scan-mesh-adjacency representative tilted by 6--8 degrees may be
   replaced by a nearby, continuous, at-least-two-source cylindrical-band
   representative only when it overlaps at least 75%, lies within 4 mm, and
   improves two-sided transition score by at least 0.10. No development case
   triggered this narrow swap.

The final V15 result on `50566000_04 1.ply` is exact: 5 TP / 0 FP / 0 FN,
precision 1.0, recall 1.0, F1 1.0. Native-resolution overlay inspection in
four views confirms that each orange dashed axis lies on a cyan supported fold
axis, including both fork-arm folds and the short transverse junction fold.
No model line lands on the central obround opening, fork slots, end-tab slots,
or free sheet edges.

Development and held-out evidence after freezing V15:

- 20 bounded development cases: V14 102 TP / 74 FP / 27 FN (F1 0.668852);
  V15 102 TP / 36 FP / 27 FN (F1 0.764045), zero true-positive loss.
- `34499000_B.ply`: exact 3/3.
- `38169000_04.ply`: exact 3/3, so the closely spaced return stack survives
  the wider NMS gate.
- `50566000_04 1.ply`: exact 5/5 independent confirmation.
- `50552000_04.ply`: 4 TP / 2 FP / 0 FN (F1 0.8). Visual inspection confirms
  all four true folds are present but two long free-edge/alias lanes remain.
  V15 improves this case by removing one V14 false line, but it is not exact
  and remains the next precision target.
- Focused V12--V15 tests: 15 passed.

V15 is retained for research. It remains source-only and consumes no part
identity, sender/expected bend count, CAD, drawing, labels, gold, or production
feedback.

## Artifacts

- Selector: `scripts/select_f1_mesh_route_v12.py`
- Tests: `tests/test_select_f1_mesh_route_v12.py`
- Regression: `output/research/f1-new-parts-20260813/v11/v12_bounded_development_regression.json`
- Frozen manual: `output/research/f1-new-parts-20260813/v11/holdout_344/manual_visual_precommit.json`
- Prediction: `output/research/f1-new-parts-20260813/v11/holdout_344/v12_prediction.json`
- Score: `output/research/f1-new-parts-20260813/v11/holdout_344/v12_score.json`
- Native overlay: `output/research/f1-new-parts-20260813/v11/holdout_344/34499000_B_v12_vs_manual.png`
- V15 selector: `scripts/select_f1_mesh_route_v15.py`
- V15 tests: `tests/test_select_f1_mesh_route_v15.py`
- V15 validator: `scripts/validate_f1_mesh_route_v15.py`
- V15 development regression: `output/research/f1-new-parts-20260813/v15_bounded_development_regression.json`
- New-scan frozen manual: `output/research/f1-new-parts-20260813/v14/holdout_50566/manual_visual_precommit.json`
- New-scan V15 prediction: `output/research/f1-new-parts-20260813/v14/holdout_50566/v15_prediction.json`
- New-scan V15 score: `output/research/f1-new-parts-20260813/v14/holdout_50566/v15_score.json`
- New-scan V15 overlay: `output/research/f1-new-parts-20260813/v14/holdout_50566/50566000_04_v15_vs_manual.png`

## V16 unified routing correction and retained result

The standalone V15 executable was a bounded-complex experiment, but it was
also invoked directly on `U02` (`50552000_04.ply`). That scan's frozen pool
diagnostics say `scan_mesh_adjacency.available=true` and
`bounded_decimation_used=false`: it is a native full-resolution mesh and must
remain on the V11 full-mesh lane. The earlier V15 `4 TP / 2 FP` result therefore
measured an invalid route application, not the intended unified selector.

V16 makes route ownership executable and testable:

- native full-resolution mesh -> V11 full-mesh consensus and fragment pruning;
- bounded small pool -> bounded recall plus conservative consensus repair;
- bounded complex pool -> retained V15 7-degree/6-mm NMS, retry/coverage,
  paired-plane recovery, tilted-axis repair, and conservative consensus repair;
- unavailable mesh -> V6 non-mesh lane.

The corrected `U02` result is exact `4 TP / 0 FP / 0 FN`. The three other
recent untouched confirmations remain exact: `U03` 3/3, `U04` 3/3, and `U05`
5/5. Aggregate recent-held-out evidence is therefore 15 TP, 0 FP, 0 FN. Manual
inspection of four native-resolution, four-view overlays confirms that every
predicted axis lies on a visible structural fold. In particular, `U04` keeps
all three closely spaced return folds and `U05` keeps the short transverse
junction fold without capturing the nearby opening. `U02` no longer contains
the two free-edge/alias lanes produced by the out-of-scope V15 invocation.

On the frozen 35-scan / 25-part-family development corpus, V16 preserves all
132 V12 true positives and removes 38 false positives: V12 is 132 TP / 98 FP /
44 FN (precision 0.573913, recall 0.75, F1 0.650246); V16 is 132 TP / 60 FP /
44 FN (precision 0.6875, recall 0.75, F1 0.717391). Exact cases rise 3 -> 4.
The minimum-family F1 remains 0.0, so V16 is retained as the research baseline
but is not promotable.

The only zero-F1 family is `48333006` (`C017`). Source-derived mesh-adjacency
candidates `SMA1` and `SMA3` match its two trusted axes to sub-millimetre and
sub-degree precision, but V11 selects neither. The representation exists; the
next bounded problem is full-mesh adjacency selection, not geometry recovery.

Operational evidence: two independent V16 serializations are byte-identical
for each of U02--U05. Selector-only execution on the 616-object U05 pool takes
1.02 seconds wall time with 392,773,632-byte maximum RSS and a
356,648,040-byte peak memory footprint on this workstation. This is downstream
of the already-bounded candidate generator and does not alter generator memory
or source serialization contracts.

V16 remains source-only. It consumes no part identity, expected bend count,
CAD, drawing, labels, gold, production feedback, or benchmark feedback during
inference.

### V16 artifacts

- Unified selector: `scripts/select_f1_mesh_route_v16.py`
- Routing tests: `tests/test_select_f1_mesh_route_v16.py`
- Unified validator: `scripts/validate_f1_mesh_route_v16.py`
- 35-scan regression: `output/research/f1-new-parts-20260813/v16_unified_development_regression.json`
- Recent predictions and determinism: `output/research/f1-new-parts-20260813/v16/`
- Native overlays: `output/research/f1-new-parts-20260813/v16/overlays/`
