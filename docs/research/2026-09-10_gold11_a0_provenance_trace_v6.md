# Gold11 A0 provenance trace V6 — 2026-09-10

## Outcome

**An explicit documented A0 root is now identified for byte verification:**

`output/research/f1-gold27-20260803/structured_union_exact_v1/`

The [released structured-union report](STRUCTURED_UNION_GOLD37_E2E_20260805.md)
at lines22–24 names its `a0_evidence_manifest.json` and the A0 disposition
`FALSIFIED_REPRESENTATION_NOT_PROMOTABLE`; line283 names `batch_summary.json`
under the same root. This is an address declaration, **not yet proof that its
prediction bytes equal the eleven later-use A0 witnesses recovered in V5**.
Neither metadata file nor any target prediction was accessed in this pass.

No root was selected by matching a line count, score or geometry. This is the
document/source-declared A0 namespace. The original fusion `--a0-root` value is
still not recovered from an actual August22 invocation.

## Source-defined path and evidence schema

`scripts/build_f1_a0_evidence_manifest.py:13–18` explicitly defines:

- `BASE = ROOT / "output/research/f1-gold27-20260803"`;
- `DEFAULT_OUTPUT = BASE / "structured_union_exact_v1/a0_evidence_manifest.json"`;
- `exact_batch_summary = BASE / "structured_union_exact_v1/batch_summary.json"`.

Its returned manifest names baseline
`GOLD37_STRUCTURED_UNION_A0_20260805` and records artifact paths, sizes and
SHA-256 digests. The document's phrase **eleven A0 evidence artifacts** refers
to the eleven entries in `DEFAULT_FILES`, **not the eleven approved Gold11
scans or their A0 prediction files**. No such equivalence is inferred.

`scripts/build_f1_transition_probe_postlock_bundles.py:156–163` defines
`structured_root / "cases" / token / "prediction_owned_regions.json"`
and explicitly labels it the A0 prediction. Its metadata bundle records the
prediction-byte hash at line207. The single-case structured-union writer also
defines `output_dir / "prediction_owned_regions.json"` at lines382/431 of
`scripts/run_f1_structured_union_e2e.py`.

Combining the **documented root** with this **source-defined schema** supplies
the following exact next-file address pattern, without directory guessing:

`output/research/f1-gold27-20260803/structured_union_exact_v1/cases/<approved-token>/prediction_owned_regions.json`

The approved token order remains:
`C003,C021,C011,C008,C022,C015,C009,C001,C002,C010,C012`.
Their exact approved scan SHA and later-use prediction SHA associations remain
in V2 and V5 respectively. No file was opened to see whether a count looked right.

## Bounded search/read ledger

The separately authorized search covered only `scripts` and top-level
`docs/research/*.md`, using filename-returning literal/reference searches for
`--a0-root`, `prediction_owned_regions.json`, root identifiers and A0 evidence
declarations. It did not search output/data/pool contents. Sixteen substantive
source/doc files were inspected, within the maximum20:

1. `scripts/run_f1_structured_union_baseline.py`
2. `scripts/run_f1_gold27_mesh_crossfit.py`
3. `scripts/run_f1_gold27_unseen_profile.py`
4. `scripts/run_f1_gold27_routed_profile.py`
5. `scripts/score_f1_gold27_predictions.py`
6. `scripts/run_f1_local_plane_fragment_probe_replays.py`
7. `scripts/validate_f1_v2_cross_parent_rung_mapping_preflight.py`
8. `scripts/build_f1_transition_probe_postlock_bundles.py`
9. `scripts/build_professor_f1_structured_inference_dropoff_20260804.py`
10. `scripts/run_f1_structured_union_e2e.py`
11. `docs/research/STRUCTURED_UNION_GOLD37_E2E_20260805.md`
12. `docs/research/PROFESSOR_F1_TRANSITION_ATOMIZATION_DROPOFF_20260805.md`
13. `docs/research/PROFESSOR_F1_TRANSITION_ATOMIZATION_RESPONSE_20260805.md`
14. `docs/research/PROFESSOR_F1_V2_CORRECTED_MAPPING_PREFLIGHT_AUTHORIZATION_RESPONSE_20260805.md`
15. `docs/research/2026-08-29_exp_cad_0017_native_patch_axes.md`
16. `scripts/build_f1_a0_evidence_manifest.py`

The other source hits supplied caller-required roots or unrelated version
defaults, not accepted A0 identity evidence. The EXP0017 narrative hit was an
`A00`/root text false lead and was discarded. One overly broad source/doc snippet
request incidentally returned other-case **published narrative** lines; this was
reported to root and the subsequent reads were narrowed to address lines. Those
case facts were not used, copied into the recovered mapping or followed into
data. No raw/Gold/candidate/prediction geometry or unrelated payload was accessed.

The original inference, pool and sealed-truth addresses occurring in source were
treated only as source text; no target was statted, hashed or opened. No module
was imported, no source executed and no metric computed. Only this additive V6
note was written; previous traces and frozen runtime/test/result files were
preserved.

## Current-byte bindings for the address lead

| Source/document | Current SHA-256 |
|---|---|
| `STRUCTURED_UNION_GOLD37_E2E_20260805.md` | `390eb4844ca739eb524cf7789261248f8ac53f7b33fde6b52cb1900e4c8f0ac3` |
| `build_f1_a0_evidence_manifest.py` | `210f055a4c1038e9e6dea33c8ce331b82fe78a605ad943df4763e6b0c1b3c2d3` |
| `build_f1_transition_probe_postlock_bundles.py` | `f57ded8d5624a727aac0527ebbd275639ea9e587a20998388a0ca238b6e9eb69` |
| `run_f1_structured_union_e2e.py` | `1a8f895a2e8a510f34ac732b3ceb1d7458c340a392268d84c23a1a2d938add8d` |
| Previous V5 trace, unchanged | `7d7063f9c698cf2eeef6b68b3972db07753a264e55ba4f10d254ba6274809b90` |

These are current-byte observations, not independently dated historical source
attestations. The A0 target metadata/predictions were not hashed in this pass.

## Next independently bounded check

Read only the exact named A0 evidence manifest/batch-summary metadata, projecting
case tables to the approved eleven SHA/token associations before analysis.
Then, under the separately granted hash-only payload scope, compare the eleven
source-defined prediction paths to the V5 digest table. Preserve missing files,
hash/identity mismatches and counts exactly. Do not extend the check to another
root or token to rescue the expected33 total.

Even an eleven-for-eleven current-byte match would establish equality with the
later EXP_CAD_0008 A0 witnesses, not independently prove the original August22
execution or historical/reference-coordinate equality. Union reconstruction
and scoring require a newly frozen, separately reviewed protocol. Historical
retained0.458 and rejected EXP0008 reported0.46875 remain distinct from new-engine
output progress or promotion.
