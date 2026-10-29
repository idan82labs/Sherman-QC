# Sheared index V2: correctness passed, efficacy failed

2026-09-12. Invented-only development evidence, not a Gold11 score or physical
measurement. All sixteen ordinary tests passed in the sole execution; the
separately frozen performance rule failed. Do not integrate this candidate or
rerun it unchanged. Independent POST acceptance is recorded separately below.

## Complete selected comparison

Every row below includes one construction and all twelve queries on32,768
invented points. Both arms completed every query with equal endpoint bytes and
original raw-ID witnesses. No favorable prefix replaces this full denominator.
Common work is the frozen abstract coordinate ledger plus37 per actual old
world-bound call, not all hardware operations.

| Population | V1 common work | V2 common work | V1 build+query CPU s | V2 build+query CPU s | V1/V2 evaluated rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact flat | 3,396,022 | 5,622,307 | 0.028538 | 0.516222 | 276,784 /212,272 |
| Resolved unique extrema | 3,402,712 | 3,439,963 | 0.029558 | 0.217197 | 277,552 /19,504 |
| Repeated plateau | 3,397,882 | 6,388,147 | 0.029697 | 0.638230 | 277,040 /277,040 |
| Folded three planes | 1,237,422 | 3,668,475 | 0.011062 | 0.253062 | 42,580 /38,740 |
| Total | 11,434,038 | 19,118,892 | 0.098855 | 1.624711 | 873,956 /547,556 |

Resolved-case row evaluations fell about93%, yet construction/bounding overhead
made total common work1.095% higher. Folded-case work was196.46% higher. Neither
required work reduction occurred. Aggregate instrumented CPU was16.44 times V1,
so the separate no-regression condition also failed. These are fixed one-shot
synthetic timings, not statistical hardware-speed estimates.

The mechanism does produce sound tighter bounds, but does not amortize their
cost here. Active construction adds roughly2.126 million steps per population;
each visited bound then adds412 declared steps including adapters. Query CPU,
not just construction, dominates the observed slowdown. Plateau rows did not
decrease at all. Exact-flat rounding/tie cases remain expensive. Smaller leaf
counts alone would have given the wrong promotion decision.

## Custody and actual execution

Root fully accepted exact independent PRE8a419f before the sole invocation
a3bcfb; actual outer completiona4d367 closed session34031 with exit0 and final
publication/outcome gates true. Parent through-inventory wall2.149006s;
worker CPU2.091687s, peakRSS65,552,384B;16 collected/passed phase triplets,
zero failures, skips, xfails, xpasses, errors or deselections. Node digest:
`50a545be29526718fb83e6d6d6ffcb808e55857d2dfec1d400cb09e2dc5e5f58`.
External process check26edc5 found no owned worker28877 or matching runner.
Only after exit/quiescence did root inspect terminal records2ac7b8.

Execution record only, NOT a retry instruction:

```text
.venv/bin/python -I -S -B -u scripts/check_raw_extrema_index_v2.py --execute --pre-sha256 1c89afd8bafc2f6e9cd090119f66d94c4b117eb7ff14ff84bd13d3a3605a8e14
```

Exact attempt: `cad_engine_v1_synthetic_checks/raw_extrema_index_v2-492swhwk`.

| Retained file | Bytes | SHA256 |
| --- | ---: | --- |
| STARTED.json | 3,112 | da628d2b0bdcc516ce72b8f30b15612fff1f8b0bccb8fd05c74b5ebcd2505ab4 |
| OUTPUT.txt | 64,370 | 5c7324b16a02fe9bd296f6e380cf092ddbb9ab11418ea27699b6100010c18a7c |
| COMPLETED.json | 7,294 | ee811d3e0fed8cef41cd736c8a71079f6dc375ee22c0c18289dce96d95059a48 |

Ten-source map2966271d92c983a242493fa5a55a4e51f2c5adfd90c68c327b0640484541c2cc,
140,821B, and runtimea767bda3636799a4c226140f9f0d1a1e714620ea3ecd88c27d49a274f52fe976
were unchanged. PRE13,361B/1c89afd8bafc2f6e9cd090119f66d94c4b117eb7ff14ff84bd13d3a3605a8e14.
Root verified the separate11-payload source/PRE archive in bedac1:
`cad_engine_v1_synthetic_checks/raw_extrema_index_v2_frozen.6H0VQ8xo/SOURCE.tar`,
174,592B/7b067bedf8d270cd7fadbceff91ed6bfec63ecf306481af755fe82d7be31a453.
Payload154,182B/inventory117706dbdd5dfc5c72e754df72ec99f0bd30f756a4ab3d946f07b114151ccaaf.
The new supervisor cache remains opaque and is neither inspected nor removed.

## Independent acceptance and next decision

Root fully read and accepted the independent
[POST](2026-09-12_raw_extrema_index_v2_source_post_run_audit.md),10,358B/
670f62e50ee9b4c4865d8e21ceb57eec452126b69aa56948db5217c0200345dd, in a9733f,
and the [efficacy diagnosis](2026-09-12_raw_extrema_index_v2_efficacy_diagnosis.md),
8,759B/ae1269d6b4cb0cd4b57487b023e3b4eda7ad450f5e154ec7e1b765322ce710dd,
in e40b74. No discrepancy was found. Terminal inventory digest is
8c83dc631636b364c1609fcdb8a94240d603dedbe8db94f8573a053e5914cffd.
The complete negative disposition is not contingent on finding a favorable
alternative. Frozen source/test/protocol/PRE/attempts remain unchanged.

A cheaper spatial-ordering contingency was source-reviewed before these results,
but its estimated up-front key cost also cannot amortize on the cheap twelve-
query folded case. Do not implement it blindly or change this experiment's
decision. The next source analysis must target total repeated-query cost and
derive a feasible accounting model before another separately frozen experiment.

Connected source inspection also identifies a mathematical reuse opportunity:
the context recomputes `R.T @ u` and `dot(u,t0)` for each signed CAD assignment,
while its exact cache includes all represented direction/origin/constant bits.
In the ideal two-plane frame model those queries share a source-pair axial
direction up to sign and a zero axial minimum-norm translation component.
This is an algebraic observation, NOT evidence that the retained physical
queries have identical bytes. Snapping coefficients, setting constants to zero
or borrowing an old measured extremum would violate the unchanged program.
Only a separately reviewed conservative reuse bound with actual original-row
witness evaluation could exploit related queries without that substitution.

Gold11 remains audited TP0/FP0/FN71 per arm, recall0 and F1undefined; no physical
engine-output improvement is claimed. The autonomous roadmap continues.
