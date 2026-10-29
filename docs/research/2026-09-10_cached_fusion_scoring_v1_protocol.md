# Cached fusion scoring V1: prospective pure gate

Date:2026-09-10. Invented11case/6family/71line records only. No actual cached
prediction, physical input or Gold coordinate access/execution is authorized here.

## Contract and engineering reason

The independently qualified retained fusion helper reconstructs the original
four variants but does not score them. This additive scorer directly passes
their unchanged selected endpoints/order/display IDs to the original matcher,
without fabricating a legacy-CAD envelope or a measured-surface carrier contract.
Original fusion uses SHORTER-span duplicate overlap; scoring retains BOTH-span
overlap>=.60,<=5degrees and<=4mm, with unchanged one-to-one minimum-cost assignment.
Assignment is by row index, preserving duplicate display IDs without deduplication.
All four variants are reported; no best-variant selection or reference adjustment.

The caller must freeze/authenticate exact payload/reference/source bytes and
cohort identities. The public helper validates11cases/6families/71references,
not their historical provenance. References are detached and validated through
the frozen reference parser. Gold never enters reconstruction, inference, pose
or unit estimation; this is post-inference cached geometry only.

Whole-case reconstruction failures clear all four variants but retain all
reference FNs. A separately declared4096prediction cap applies per variant/case;
an over-cap variant clears that whole case in that lane, not its other valid
variants. Embedded conflicting family IDs refuse all four scoring lanes while
preserving the separate reconstruction artifact. Optional family ID must be
an exact string before comparison: nonstring/custom equality is not invoked.
Unexpected matcher failure propagates as failed scoring, never invented empties.
Undefined precision/F1 follow the frozen aggregate helper. Cached line-count
equality and exact localized sets remain distinct from physical observed counts.

## Independent controls

`tests/test_gold11_cached_fusion_scoring_v1.py`:548lines,20functions/101prospective
outcomes. Independent author `/root/production_architecture_review`; root read
all source/test. `/root/cad_measurement_integration_review` independently reviewed
the scorer and caught the optional-family nonstring equality issue, repaired
before freeze; fourteen adversarial controls cover both envelopes and no hooks.

Private-package fixture loads inspected pure helpers. Only exact legacy matcher
definitions are AST-extracted; ordinary small invented cases use actual SciPy
assignment and actual fusion. Original CLI/default-data code never executes.
Tests cover direct function identity, all4reports/denominators, nil metrics,
shorter-versus-both-span matching, row-index assignment with duplicate IDs,
greedy-order effects, alias precedence, family minima, whole-case refusals,
unit/frame conflicts, snapshot ownership, bad references/cohorts and late matcher
failure. Eight explicit mocked reconstruction/count-only matching controls check
4096/4097 cap admission without thousands of geometric comparisons.

## Exact gate and limits

Independent pre-run review must freeze nine files: new scorer, fusion helper,
cached scorer, mesh scorer, localized aggregate, legacy matcher, test, protocol
and supervisor. Count101outcomes/no skips before one execution:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite cached_fusion_scoring_v1
```

60s monitored wall/45s workerCPU/1GiB sampled groupRSS/1MiB stdout, numerical
threads1,privateHOME/TMPDIR. No other heavy job. Retain STARTED/OUTPUT/COMPLETED,
all outcomes and before/after hashes. Failure/skip/missing outcome/source/resource
issue is terminal; no retry or altered started budgets/oracles. Independent
retained post-audit before use. Supervisorb40bc9bc is preserved byte-identically;
legacy monitored-wall/receipt/cleanup and sampled-memory limitations remain.

A pass qualifies only the pure scoring adapter. A real cached-control run needs
its own exact22prediction/reference hashes, protocol and independent release.
No physical score, historical comparability, fresh inference, calibrated
uncertainty, observed count or production/floor release follows from this gate.
