# Gold11 historical comparison binding review

Date: 2026-09-09. Decision class: bounded, read-only provenance review for a new
consumed-development comparison. **Historical comparability: UNESTABLISHED.**

This review does not execute inference, reconstruct predictions, score geometry,
open raw CAD/scan files or the Gold container, reopen terminal attempts, or change
Gold, frozen R0, production, or manufacturing authority. The only new file is
this narrative. The user's standing development mandate permits the root task to
prepare new independently audited protocols; the unknowns below do not revoke
that permission or require indefinite historical-artifact searching.

## Scope bound to the completed input review

The released receipt
`docs/research/gold11_input_review_v1_attempt/OUTPUT.txt` has verified SHA-256:

```text
0d1c5abcc534bd6d377296575941b0cab0dcf7bf8dec8b4bdd3f6c036a709c3f
```

Its `reference_binding` records:

```text
reference container snapshot: 4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36
selected 11 records:          d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc
historical_reference_byte_equality: UNESTABLISHED
```

These are current-review bindings, not proof of which reference bytes the
August scorer consumed. The container was not reopened here. Selected metadata
was joined by scan path/hash to the named August slice and reports; the two
excluded STL cases were not inspected. No new email, held-out cases, or other raw
cohorts were accessed.

## What the retained historical record actually says

The retained fusion diagnostic and August comparability audit agree on the same
11 scans, six families, and 71 reference lines:

| Lane | TP | FP | FN | Predictions | F1 | Exact count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scan-only A0 input | 19 | 14 | 52 | 33 | 0.36538461538461536 | 0/11 |
| Scan-supported CAD input | 20 | 24 | 51 | 44 | 0.34782608695652173 | 1/11 |
| Retained A0 + gated-CAD union | 30 | 30 | 41 | 60 | 0.4580152671755725 | 1/11 |

Prediction totals are the reported TP + FP, not a new scoring result. The union's
minimum family recall is 0.20, precision is 0.50, recall is
0.4225352112676056, and exact localized set accuracy is 0/11. The source's
per-case `exact` field means exact localized set, not mere count equality.

The CAD-only input above is **not** the separate pure-native-STEP candidate
reported near F1 0.354 in the August deep dive. Nor is the retained union the
aggressive three-source candidate reported near F1 0.456.

The original end-to-end `summary.json` reports 13 successful jobs and no failures.
Its 13-case count statistics are not the matched Gold11 union's localization or
exact-count metrics. Its root has no executable, configuration, environment, or
source-bundle binding. A declared blind-inference contract is present in the
selected case metadata; that declaration is not a reconstruction of the old
environment.

### Exact selected case denominator

The following counts are retained records, not recomputed results:

| Gold11 case | Family | Reference record | Gold | A0 input | Gated CAD input | Union |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CG08506AE97D | 10839000 | corpus/03_10839000 | 6 | 1 | 4 | 4 |
| CG191DEC7F81 | 49001000 | corpus/10_49001000 | 5 | 3 | 1 | 3 |
| CG1E0BBA2933 | 38172000 | shaked_xv/C05 | 10 | 5 | 9 | 11 |
| CG4BDD1371B6 | 37361005 | corpus/05_37361005 | 4 | 2 | 3 | 3 |
| CG4FAF273093 | 49024000 | corpus/11_49024000 | 3 | 1 | 3 | 3 |
| CG64C412CC73 | 40500000 | corpus/08_40500000 | 7 | 5 | 1 | 6 |
| CG65CD575B1C | 38172000 | corpus/06_38172000 | 6 | 1 | 2 | 2 |
| CG84C8FA12D2 | 10839000 | corpus/01_10839000 | 8 | 5 | 11 | 11 |
| CGE4B0549E41 | 10839000 | corpus/02_10839000 | 4 | 1 | 2 | 2 |
| CGEBA9A869AC | 38172000 | corpus/07_38172000 | 8 | 5 | 1 | 6 |
| CGF41F49066A | 38172000 | shaked_xv/C06 | 10 | 4 | 7 | 9 |

The selected scan-frame audit has 44 accepted CAD lines across these cases,
consistent with the CAD-only input total. Its declared support constants are
201 samples, 3 mm nearest-neighbor distance, minimum supported fraction 0.10,
and minimum contiguous supported span 8 mm. That audit does not independently
establish scan units, metrological datum, carrier semantics, or calibration.

## Union geometry was not materialized by the retained fusion script

`scripts/diagnose_f1_cad_scan_fusion_gold11.py` reads two geometry inputs per case:

```text
<a0-root>/<a0-case-token>/prediction_owned_regions.json
<cad-prediction-root>/<Gold11-case-token>/semantic_result.json
```

The A0 token is joined through
`<a0-pool-manifest>.cases[].source_scan_sha256 -> case_token`; it is not the
Gold11 token by implication. The CLI requires the pool manifest, A0 root, CAD
prediction root, Gold catalog, slice, and output as explicit arguments, with no
default paths for those roots. The named retained diagnostic, slice, stage graph,
summary, and scan-frame audit do not record those CLI values or exact input-file
hashes. The source inspection did not locate an invocation binding them.

The fusion script normalizes and preserves A0 input order, then visits CAD rows
in their input order. It adds a CAD row only if it is not a duplicate of any row
already in the union. Added rows receive the `CADF_` ID prefix. It scores that
in-memory union, but writes only aggregate/per-case counts and scores. It does
**not** write one union prediction-geometry file per case. Its report's aggregate
case rows also omit the individual matching geometry.

The connected exporter `scripts/score_f1_cad_guided_gold13_e2e.py` does materialize
`semantic_result.json`, but those files are CAD input predictions, not the union.
The exporter also requires an explicit output/prediction root. No guessed root,
recursive output/data search, or fallback to a different similarly named run was
used in this review. Thus an exact retained 11-file union allowlist, or the
necessary 22 input-file allowlist, is **not established** here. This is not a
claim that no other copies exist anywhere.

If exact input paths are later supplied by an in-scope retained command or
manifest, freeze their current bytes and ordered content before any computation.
Reconstructing the union would then be a **new, explicitly identified replay**,
not discovery of historically materialized union files. It must not rerun old
inference or pretend to restore missing historical source/environment evidence.

## Matcher and source binding limits

The source imports `Thresholds`, `pair_metrics`, and `match_lines` from
`scripts/evaluate_f1_against_manual_gold.py`. Inspection establishes the current
algorithm and defaults:

- Undirected axis angle at most 5 degrees.
- Symmetric transverse midpoint offset at most 4 mm.
- Finite overlap at least 0.60 of **both** Gold length and prediction length.
- One-to-one minimum-cost matching after inadmissible pairs receive a large
  sentinel cost.
- Endpoint precedence: `span_endpoints`, then direct `line_start`/`line_end`, then
  anchor/axis/span construction. No unit/frame/carrier verification is performed
  by that legacy matcher itself.

The union's duplicate rule instead requires angle at most 5 degrees, transverse
offset at most 4 mm, and overlap at least 0.60 of the **shorter** span. Do not
silently substitute that weaker overlap predicate for scorer admissibility.
The report's `matching_constants` describes fusion duplicate constants; the
both-span scoring requirements are recovered from source, not fully recorded in
that report.

The retained stage graph binds five stages to four unique source files. Three
current hashes match their recorded values; the native STEP extractor differs:

```text
recorded STEP: ca26c7e11cfa98e2f381cfd6e99cba4c3a42e49b9cd44f12c1a88e5c851de9e1
current STEP:  f089f71466e0f9bba47139c06efdf8bf651e7707fb05c78ca2195f06146b1314
```

Current matching hashes for the other three unique graph sources are:

```text
f5d91f841e15570d26557ba04793872889234a8b25de09a358d36ec416a643e1  scripts/apply_f1_cad_guided_scan_frame_gate.py
cc34184166118fb2d00bd4b15edeabf6aaab3e69ea6f7c8e33b9394fffba5b52  scripts/diagnose_f1_cad_scan_fusion_gold11.py
55d2a420c3af34fad52e6ef5f7a6a82b69fc99a64fadf0e938ba39f7691032bc  scripts/run_blind_cad_ply_benchmark.py
```

The stage graph does not bind the imported geometric matcher, aggregation
module, reference bytes, emitted predictions, all transitive dependencies, or
runtime environment. Current hashes, suitable for a **new** prospective replay
freeze but not supplied as historical proof, include:

```text
1e9420558b91f9b136fb32ed012b56c64cdaeecf7599ee1ef9fe336915d75c62  scripts/evaluate_f1_against_manual_gold.py
28522ee7c63d6886304042a97c736397ea6f043bf77f064cb42efc17a8b1a9b7  scripts/f1_physical_evaluation_contract_v2.py
3d5237489535e14101377b89f2023f830a99c4479c01ffa05ff6dc6e4810d0e4  scripts/build_f1_cad_guided_gold13_slice.py
```

Source drift blocks a claim of an exactly restored historical executable, but
would not alone prevent a new same-reference rescore of genuinely recovered,
byte-frozen old output geometry. Those two claims must remain distinct.

## Proposed immutable evidence allowlist

These exact record paths were read/hash-checked. The hashes are current-byte
bindings observed in this review, not tamper-proof historical attestations.
Multi-cohort containers may supply only the selected Gold11 metadata/records;
their whole-file hash does not authorize unrelated case extraction.

```text
4d02f3f48bb9894cbb4429eccce2c9c519a40d389dc10d929f80c75a45bc705f  docs/research/artifacts/2026-08-25_f1_retained_stage_graph_v1.json
e92234dffdf34317c19a9d0ca9fd50f30874baf20ac36b9082d0085095c922b6  output/research/f1-cad-guided-gold13-e2e-20260822/summary.json
a12b550310537a372f2ba53f8f2e992d5cb3cd75cb95d733ae66a8a761cea7db  output/research/f1-cad-guided-gold13-e2e-20260822/FUSION_GOLD11_DIAGNOSTIC.json
d7515160d8f55ddb2ba13d87c6e080a07022c03a30c50315a45316680cf27d3c  output/research/f1-cad-guided-gold13-e2e-20260822/SCAN_FRAME_GATE_AUDIT.json
7fd1cf5c14e842430220668b5a221a5c1b46d45fb3ac11e7ef72267985e41098  data/bend_corpus/evaluation_slices/cad_guided_gold13_physical_20260822.json
655bb3c0bed8467c4f9e60a67b9f765015543b2f8b1db93ebb35daf06508f579  docs/research/artifacts/2026-08-27_f1_cad_assisted_comparability_audit_v1.json
```

This allowlist binds historical **claims and metadata**, not an executable
comparison lane. It must not be represented as a completed 11-case prediction
allowlist. None of the named records supplied an archived Gold geometry digest
or historical/current reference-byte equality proof.

## Minimum honest next step

Proceed with a new independently audited, frozen, same-input development
comparison on the exact consumed Gold11 cohort and unchanged selected71
references under the user's standing mandate. Freeze baseline/challenger source,
configuration, adapter policy, evaluator, all inputs, case/family denominator,
failure treatment, and unit/frame/carrier assumptions before running. Keep Gold
outside inference and outside any pose, unit, or carrier choice. Retain every
case, refusal, error, emitted prediction, and match result in both lanes.

Frozen R0 remains unchanged and identifiable, but the completed input review
showed that its vertex-only parser rejects all 11 mesh PLY schemas. A useful
fresh paired baseline therefore needs an explicitly frozen mesh-compatible
control/adapter policy, with unchanged versus adapted R0 named honestly. Merely
beating R0's all-refusal input behavior is not proof of beating the historical
retained engine. A new control must not be retrospectively relabeled as the
August incumbent or silently substituted for frozen R0.

Until the missing historical output/reference bindings are genuinely recovered,
report these statements separately:

1. Historical recorded milestone: F1 0.4580152671755725, TP30/FP30/FN41, 1/11 exact
   count, six-family minimum recall 0.20; historical byte comparability unknown.
2. New paired development result: recomputed baseline and challenger on the
   same frozen current protocol, including all cases and failures.
3. Qualification: no unseen-family, manufacturing metrology, production, or
   floor-release approval follows from either diagnostic result.

If a future recovered-output replay reproduces the old counts, describe that as
score-consistent replay evidence. Matching aggregate numbers alone cannot prove
historical reference-byte identity or recreate missing environment authority.
