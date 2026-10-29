# Sherman QC F1 — CIBEC physical qualification input contract V1

Date: 2026-08-25

This contract is used only after the two-authority CIBEC charter is signed and the 14-case source-free qualification passes. It does not authorize detector execution or opening the sequestered new corpus.

## Physical annotation qualification

Run:

```bash
.venv/bin/python3.11 scripts/score_f1_cibec_physical_qualification_v1.py \
  --input CIBEC_PHYSICAL_QUALIFICATION_INPUT.json \
  --output CIBEC_PHYSICAL_QUALIFICATION_RESULT.json
```

The input has schema `CIBEC_V1_PHYSICAL_ANNOTATION_QUALIFICATION_INPUT` and exactly 13 cases. Each case contains:

- `annotation_a`, `annotation_b`, and `adjudicated`: canonical `SCAN_BEND_EVENTS_V1` objects bound to the same opaque case, scan, ontology, and renderer hashes;
- `annotation_a_replay_identical`, `annotation_b_replay_identical`, and `adjudicated_replay_identical`: all must be `true`;
- `export_latency_seconds`: six primary/replay machine-export timings;
- `jobs_attempted: 6`, `jobs_succeeded: 6`;
- `peak_rss_bytes`, strictly below 2 GiB;
- `native_visual_passed: true` and `native_visual_audit_sha256`, binding the adjudicator's native-resolution audit artifact;
- `charter_rule_justification_sha256` on exactly the three opaque cases corresponding, in custodian records only, to the required written-justification parts. It is `null` on the other ten.

The top level additionally requires `hash_manifest_verified: true` and `leakage_audit_prohibited_field_or_asset_count: 0`.

The scorer rejects malformed denominators. A PASS requires every professor numerical, semantic, visual, deterministic-replay, latency, memory, job-success, manifest, and leakage gate.

## CAD primitive-grouping qualification

Run:

```bash
.venv/bin/python3.11 scripts/score_f1_cad_grouping_qualification_v1.py \
  --input CIBEC_CAD_GROUPING_QUALIFICATION_INPUT.json \
  --output CIBEC_CAD_GROUPING_QUALIFICATION_RESULT.json
```

The input has schema `CIBEC_V1_CAD_GROUPING_QUALIFICATION_INPUT` and exactly 13 cases: 12 trusted overlays and the one preregistered registration abstention. Every case contains two independently produced `CAD_EVENT_CORRESPONDENCE_V1` maps, two replay-identical flags, four primary/replay load-export timings, four job attempts, and peak RSS.

Each trusted case also contains `native_v9_primitive_ids`, the complete unique inventory over which all unordered primitive pairs are classified. A primitive omitted from an event is still in the pairwise denominator. A primitive not in the frozen inventory is rejected. The abstaining case is excluded only from grouping precision/recall/F1; its replay, jobs, latency, memory, manifest, and leakage evidence remains mandatory.

The top level requires the same verified manifest and zero-leakage count. PASS requires grouping precision, recall, and F1 each at least 0.90 plus all operational gates.

## Custody rule

Do not put filenames, part identities, family labels, expected counts, current predictions, CAD correspondence, or truth into detector-facing artifacts. The blind allocator uses a custodian-only salt to create opaque IDs and keeps scan hashes plus morphology membership in a separate custodian map.
