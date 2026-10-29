# Partial CAD reference metric bounds

Source-only mathematical extension; no matching, label or frozen scorer change.

Let T be matched reviewed positives across complete and partial parts, M their
unmatched positives, F unmatched predictions on complete parts, U unscored
predictions on partial parts, and N=T+F+U. Any truth completion can match x of U
predictions, 0<=x<=U, and add y>=0 unknown missed bends. It has TP=T+x,
FP=F+U-x, FN=M+y. These formulas require the same exact one-to-one face ownership
and immutable reviewed-positive assumptions as the frozen scorer.

Precision lies in [T/N,(T+U)/N]. Recall and F1 have lower infimum zero with
unbounded unknown truth, but their upper bounds are respectively
(T+U)/(T+U+M) and 2(T+U)/(2(T+U)+F+M). Both increase with x and decrease with y;
their maxima occur at optimistic x=U,y=0. This remains an outer bound even when
that optimistic completion is geometrically impossible. Unknown truth makes
the lower bound vacuous, not necessarily the upper bound.

With all references complete, U=0 and y=0: report exact point metrics. Zero
denominator ratios are null, never a fabricated perfect score. Partial lower
zero is an infimum, not necessarily attainable. These are technical CAD bounds,
not physical truth, manufacturing approval or unseen-family accuracy.

The synthetic test enumerates all T,M,F,U in 0..3, every x in 0..U and y in 0..8,
checking all defined ratios against the analytic bounds. No corpus inputs or
predictions are used. Independent nonauthor review remains required before use.
