# Gold11 V7 result: completed execution, no engine-output improvement

Date: 2026-09-12. Root acceptance of the independent execution/custody,
scoring and scientific-diagnosis reviews. This is consumed-development evidence,
not unseen-family validation, manufacturing approval or production release.

## Result

The full paired run finished once with exit 0 after about 103 minutes. All eleven
cases in both arms were retained: 22 durable case/arm outcomes, 21 returned
registration refusals and one CPU-terminated worker with an explicit supervisor
refusal. No returned workflow reached bend measurement.

| Metric | Fresh control | Fresh challenger | Difference |
| --- | ---: | ---: | ---: |
| True positives | 0 | 0 | 0 |
| False positives | 0 | 0 | 0 |
| False negatives | 71 | 71 | 0 |
| Precision | undefined | undefined | undefined |
| Recall | 0 | 0 | 0 |
| Localized F1 | undefined | undefined | undefined |
| Minimum-family recall | 0 | 0 | 0 |
| Exact emitted-support-count proxy accuracy | 0 | 0 | 0 |
| Physical observed counts unavailable | 11/11 | 11/11 | 0 |

The frozen scorer returns null precision when there are no predictions, and
therefore null F1. We do not replace that policy after seeing the result.
There is no numeric new F1 and no improvement. Zero emitted supports do not mean
the scan contains zero bends. The 71 scoring misses are not 71 measured physical
defects. Physical angle/radius/deviation truth remains unavailable.

Historical F1 0.4580152672 still has unresolved source/configuration/reference
comparability. This fresh same-input paired result neither reconstructs nor
beats that historical result. The frozen R0 and all previous attempts remain
unchanged.

## Why it failed

These immediate causes are mutually exclusive over all 22 outcomes:

| Stopping mechanism | Outcomes | Cases/interpretation |
| --- | ---: | --- |
| Exact raw-extrema coordinate-work limit | 10 | Five scans, both arms; roughly 26.6–27.1 million successful repeated query-row evaluations before the next charge refuses |
| Completed finite proposal search, no passing start | 4 | Two scans, both arms; not an exhaustive proof that no physical pose exists |
| Scan-plane triangle-membership work limit | 2 | One scan, both arms; partial proposals correctly unavailable |
| Competitive-triangle certificate interval too wide | 2 | One scan, both arms; numerical support limitation, not established physical mismatch |
| Near-tie storage limit | 2 | Two challenger outcomes; represented triangles are not necessarily distinct physical surfaces |
| Completed residual/context rejection | 1 | Control context p95 upper 0.695647 mm exceeds the declared 0.5 mm gate |
| CPU termination, final geometry unknown | 1 | Control worker exited -24 near the 1860 s CPU limit; no final result or exact final CPU measurement |

Five returned outcomes also report INVALID_NATIVE_SOLID. These additional native
blockers are not silently reclassified as the immediate registration reason.
Family failure patterns and all 22 exact ledgers are in the
[scientific diagnosis](2026-09-12_gold11_mesh_development_v7_scientific_diagnosis.md).
No evidence here establishes overtraining or undertraining of a fitted model;
these are failures of a deterministic bounded geometry pipeline.

One important distinction: CG191 control had a supported candidate context
(context p95 upper 0.383340 mm), but a later candidate's fatal certificate query
invalidated the incomplete overall evaluation. This is not proof every proposed
pose was bad, and it does not permit salvaging a favorable prefix after the run.

The pre-run work qualified input handling, custody and bounded failure behavior;
it did not establish that these physical geometries would yield measurements.
The five returned cases shared with the earlier diagnostic are byte-identical
to its published result bindings. Running the remaining cohort supplied the
complete denominator and additional failure patterns, not a new mechanism that
could repair already-known failures. Another unchanged run is not justified.

## Independent acceptance and execution evidence

Root fully read and accepted:

- [Physical POST](2026-09-12_gold11_mesh_development_v7_post_run_audit.md),
  19,667 B, SHA256 bbc9d364eeca9887e5513b11ef14ec6c0bbae7a2309fe0551e36de33ed31d094.
- [Scoring POST](2026-09-12_gold11_mesh_development_v7_scoring_post_run_audit.md),
  13,120 B, SHA256 0b298457de24604647191d2b9d17a0803907df9566be1c9c06724c0a4f89230c.
- Scientific diagnosis linked above, 20,741 B,
  SHA256 eaaef192cb689bf1e7cb0d390aa3a47ce85138d38862332f93e46ef0d25ac271.

The initial combined display was truncated; separate complete reads 2812bf and
72716b supplied the scoring and scientific documents. Physical POST was complete
in 338aa6; final three-document hash verification ab121b matches these pins.

Sole original invocation e0f986:

```text
.venv/bin/python -I -S -B -u scripts/run_gold11_mesh_development_v7.py --execute
```

This is an execution record, NOT a retry command: the attempt is terminal and
must not be rerun, resumed, overwritten or rescored. Actual final result 6ce55b
closed session 75648 with exit 0. Fresh external check 75df5d established no
remaining owned writers. All 70 live/frozen source files, runtime metadata,
PRE/archive and source/input/result/projection/scorer bindings reconcile. The
241-file/42-directory tree retains every selected outcome and the unchanged
71-reference snapshot. Final inventory SHA256:
5cc5670cb38a60a04380deca616bab8efc11273ce6da5a6a294edae5330c5cd1.

Parent recorded wall 6180.037832 s and lifetime peak RSS 1,750,417,408 B.
Observed outer sampled tree-plus-parent peak was 2,595,110,912 B; sampling is
not a continuous memory guarantee. One killed worker lacks terminal CPU and
sampler telemetry. The coordinator cache is retained opaque outside the attempt,
not audited or deleted. Terminal status markers remain pending-POST by design;
this external acceptance does not rewrite them.

## Next justified engineering step

Prioritize the common exact-extrema bottleneck before another physical run.
The current hierarchy bounds projections with world-axis boxes; these ignore
correlations within oblique planar supports. Investigate a separately versioned
oriented/sheared bound that preserves original raw rows, exact staged binary64
leaf values, raw-ID ties and all-or-nothing failure semantics. This is a concrete
source-level hypothesis, not a demonstrated explanation of the physical query
directions: the retained aggregate counters do not reveal those directions.

Independent math, adversarial fixture and integration reviews are assigned.
The proof must enclose reconstruction error and rounding in the original
arithmetic program, including subnormals; a rotated box or algebraically
rearranged dot product alone is insufficient. Added construction/bounding work
and memory must be charged. Exactly planar normal queries may remain worst-case
because their extrema differ only at rounding scale; preserve that negative.

Advance only on exact output/witness equivalence and useful total-work reduction
on a fixed, independently constructed mechanism-specific matrix under declared
budgets. This does not require every physical case to succeed. If the bound
cannot deliver useful savings, retain that result and choose another mechanism.
Proven changes must connect to the maintained opt-in workflow and a new frozen,
independently audited physical successor before a new consumed-Gold11 run.

Other blockers remain explicit: finite proposal coverage, triangle certificate
resolution/tie representation/CPU, native solid support and the separation of
pose admission from shape conformance. Removing one capacity refusal does not
promise downstream measurement success. No tolerances, datums or nominal CAD
measurements will be invented to make outputs appear.

The standing autonomous scope remains the September 9 raw-file hashes and
unchanged selected-reference snapshot. No new email parts, held-outs, hardware,
deployment or paid services are involved. The benchmark follow-up stays active
because no numeric F1 exists; the broader roadmap is not complete.
