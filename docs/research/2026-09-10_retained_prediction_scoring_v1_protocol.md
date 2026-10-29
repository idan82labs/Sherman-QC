# Retained-prediction scorer V1 — pure qualification protocol

Date: 2026-09-10. Prospective source/invented-data gate only. No cached physical
prediction, raw scan/CAD, reference catalog, inference or actual scoring run is
authorized by this protocol. Those require a separately frozen exact-cohort gate.

## Mechanism and comparison boundary

The additive `scripts/gold11_retained_prediction_scoring_v1.py` compares complete
declared maps for eleven cases, six families and 71 reference lines. It preserves
the frozen matcher and modern-engine eligibility from `gold11_mesh_scoring_v1`.
The new lane validates retained CAD prediction schema/direct endpoints without
claiming they are scan-measured carriers, fresh inference or the historical
scan/CAD fusion. Caller-authenticated file identities are a later requirement;
cardinality is not authentication. No transformation, cropping, fusion, filtering
or Gold-dependent selection occurs in this helper.

One malformed cached row invalidates its entire case under the declared failure
policy. Every selected reference remains a false negative when output is absent.
Valid empty cached lists do not establish observed absence. Unexpected matcher
failures propagate as scorer failures. Undefined metrics and deltas stay null.
Original scorer, reference rules, and all historical results remain unchanged.

## Independently authored obligations

Reviewer `/root/cad_measurement_integration_review` owns only the new test module;
`/root/production_architecture_review` authored the scorer, and root integrates.
The prospective matrix is 74 pure cases on invented inputs: unequal per-case
reference counts totaling 71; manually calculated pooled/family/case metrics;
unchanged angle/offset/both-span boundaries and one-to-one assignment; malformed
schema, endpoints, aliases, duplicate IDs and explicit convention conflicts;
complete-map/reference admission; modern refusal/delegation; metadata isolation;
undefined scores; mutation isolation; unexpected matcher-error propagation.

The 4096/4097 boundary is tested at private normalization, not by constructing
a 4096-square assignment matrix. No large benchmark or physical-data fixture is
allowed. Exact collection count and all expected values require root source
review and independent pre-run audit before execution. No imports/tests have
been run at protocol authoring time.

## Freeze, resource limits and gate

Freeze source, independent test, this protocol, harness and full import closure:
the new scorer, `gold11_mesh_scoring_v1.py`, `evaluate_f1_against_manual_gold.py`,
and `f1_physical_evaluation_contract_v2.py`. The test uses an isolated synthetic
scripts namespace; no app initialization or backend/native import. Any additional
executable/source-read dependency found in final review joins the hash closure.

One root-owned serial execution under 60 s wall, 45 s worker CPU, 1 GiB sampled
process-tree RSS and 1 MiB stdout. Numerical libraries single-threaded, isolated
Python/pytest, private HOME/TMPDIR, disabled plugin discovery/cache/conftest.
The exact harness route is:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite retained_prediction_scoring_v1
```

Only independent exact-bundle GREEN clears this first execution. Retain exclusive
STARTED/OUTPUT/COMPLETED and before/after hashes. All selected cases must pass,
zero unexplained skips/stops/source drift, followed by independent retained-only
post-audit. Any failed attempt is terminal; no altered fixture/cap or silent retry.
Passing qualifies only this pure scorer contract. It establishes no new physical
score, historical provenance, manufacturing count, metrology or product promotion.
