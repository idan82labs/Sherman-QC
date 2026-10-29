# Finite curved-end bend detection successor

2026-09-30. User authorized higher CAD extraction score and visual render review.
The cohort remains the exact 20 STEP hashes / 23 aliases from the released CAD
inventory. The 127-positive technical reference snapshot remains unchanged.
No scans, Gold, incoming email, held-outs or additional datasets are in scope.

## Engineering reason and candidate

The measured V7 extraction baseline matches 98/127 known bends (77.17%).
Post-score diagnosis identifies 24 known misses whose cylindrical carriers are
recognized but whose curved end boundaries fail a sufficient rectangle proof.
Two return bends and three rolled regions have separate unsupported geometry.
This successor implements finite connected graph strips with curved end
boundaries, preserving the original native footprint and provenance. Positive
rational Bezier weights and monotone angular controls prove end curves are
single-valued. Separated axial coefficient hulls prove a positive common
interior core. Ordered native occurrence coverage, native validity, same-solid
ownership, opposite material normals, shared end caps and tangent flange rails
remain required. Arbitrary polygons, holes, rolls and curved continuations are
unsupported by this first extension.

Strict boundary closure is checked first. A separately named native-join
qualifier can certify an interior core when each end chain is one native edge,
all internal Bezier joins are exact, adjacent edges share the same native vertex
and actual chart gaps fit the source kernel precision. It excludes uncertain
angular side strips and retains the original curves; full boundary geometry is
explicitly unproved in that mode. Source tolerance is at most 2.5e-6 native
assumed mm; the intrinsic chart budget is four times that tolerance and at most
1e-5. Two endpoint tolerance balls account for twice the source bound, and the
short cylinder chord-to-arc ratio is at most two. Native radius must exceed
four times source tolerance and the actual chart gap must fit the budget, so
this conversion applies only to a small, coherent, non-wrapped join. No
coefficient snapping, parameter extrapolation or unknown material filling occurs.

Original rectangle detections are retained verbatim. Supplemental features may
not overlap their native faces. New core markers describe a certified subset of
bend support; they do not claim complete bend extent or dimension accuracy.
Only original schema7 rectangle receipts can enter the existing unfolding path.
Graph strips have explicit detection-only receipts and unavailable unfolding.

Native active spline intervals are copied and exactly restricted before the
OCCT Bezier decomposition. All coefficients, weights and original ordered
boundaries remain in the receipts. The algorithm has fixed span/pole/topology
budgets, with explicit refusal. No reference labels, filename rules or target
counts enter inference.

## Qualification and evaluation

Before native corpus execution: bounded authored tests for a sloping end-cap
bend, unchanged rectangular bend, reversed loop/frame and closed tube; negative
curve, hole, material-normal, disconnected-cap and budget fixtures. Independent
source/math review, exact source/runtime closure, fresh freeze and non-author
PRE approval are required. Previous attempts remain terminal.

Limits are fixed: one native job under the shared admission lock; 60 seconds /
4 GiB sampled tree RSS per case, 1,230 seconds per campaign, source at most
5 MB, aggregate per-case artifacts at most 40 MB including logs and source,
logs at most 1 MB. Reserve terminal receipt space and check budgets after exit.
Unknown cleanup stops following cases while retaining every selected case.

Success requires all 98 incumbent native feature groups and all six existing
flat previews retained, positive additional exact one-to-one reference matches,
and no new unmatched predictions. Report known-positive capture, complete-case
precision/recall/F1, exact count accuracy and unchanged partial-reference bounds.
Failure, refusal and zero-output cases stay in the score. A weaker or negative
result remains terminal and is not promoted.

After independent POST, generate derived comparison panels for all 20 CADs.
Root inspects every panel, emphasizing newly recovered face regions, surviving
holes and trim silhouettes, support markers and remaining missed regions.
Visual agreement supports diagnosis; it does not establish manufacturing blank
accuracy. Product/public defaults remain a separate integration gate.

## Mathematical source

OCCT's exact B-spline to Bezier decomposition retains the source curve and
orientation using adjacent knot spans, following Bohm's knot-insertion method:
[official conversion reference](https://dev.opencascade.org/doc/refman/html/class_geom2d_convert___b_spline_curve_to_bezier_curve.html).
Ordered wire traversal must be checked against all oriented edge occurrences:
[official wire explorer reference](https://dev.opencascade.org/doc/occt-7.6.0/refman/html/class_b_rep_tools___wire_explorer.html).
Positive weights make a rational Bezier coordinate a normalized positive
Bernstein average. Ordered angular controls make its derivative nonnegative
through the covariance of control value and Bernstein index; strict endpoint
progress gives strict interior progress. This is a sufficient conservative
certificate, not a curve fit. The independent audit and authored adversarial
tests are bound to the frozen candidate before evaluation.
