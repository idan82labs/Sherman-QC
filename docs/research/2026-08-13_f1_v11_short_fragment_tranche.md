# F1 V11 short-fragment tranche

## Outcome

The untouched `50552000_04.ply` native visual precommit froze four base-to-wall bend axes before inference. Strict-blind V10 selected five axes: all four frozen bends plus one short internal corner fragment. V10 scored TP=4, FP=1, FN=0, precision=0.8, recall=1.0, F1=0.888889.

Native-resolution review rejected the fifth line as a conventional bend. It was 29.6828 mm long, supported exclusively by `cylindrical_standard` and `cylindrical_wide`, and was 39.5% of the median selected span. The four true bend axes were 70.7314–95.9080 mm long.

V11 adds one bounded source-only rule after V10: prune a selected line only when every supporting family is cylindrical and its span is below 45% of the median selected span. It cannot add bends and does not use identity, expected count, CAD, drawings, or labels.

## Evidence

- `50552000_04`: V11 TP=4, FP=0, FN=0, F1=1.0; native four-view overlay visually clean.
- Prior `48811000_A` optimization case: V11 TP=4, FP=0, FN=0, F1=1.0; no V11 prune, V10 representative repair preserved.
- Fifteen full-resolution-mesh development cases: V10 and V11 both TP=30, FP=24, FN=17, F1=0.594059. V11 made zero prunes and lost zero true positives.
- Focused V10/V11 tests: 6 passed.

## Interpretation

This funds the V11 tranche as a bounded precision improvement, but `50552000_04` is optimization evidence because its V10 false positive motivated V11. V11 still requires a different untouched scan with a frozen native-visual precommit for independent confirmation.
