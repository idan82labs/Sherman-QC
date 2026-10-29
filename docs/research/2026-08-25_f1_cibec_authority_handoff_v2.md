# Sherman QC F1 — CIBEC_V1 two-authority handoff V2

Date: 2026-08-25

## Purpose

This is the shortest valid path from the current unsigned evaluation proposal to independently annotated physical evidence. It does not authorize detector inference, alter the retained F1, or open the new scans.

Review payload SHA-256:

`653fa43170663b8ea9f06f1ae053bdb093ba6aa8345b262a2223897c5ec095ee`

## What the two authorities review

1. Neutral source-free morphology, with corrected profile typography:
   `output/research/cibec-v1-neutral-authority-pack-v4/`
2. Exact proposed counting rules and S01–S14 marker positions:
   `output/research/cibec-v1-authority-candidate-v2/`
3. The especially consequential cases:
   S04 (three events), S05 (four), S06 (two disconnected spans), S07 (three), S12 (two), S13 (cropped partial span), and S14 (scan line retained while CAD abstains).

Candidate V2 changes only presentation. Its charter and expectation JSON are byte-identical to Candidate V1 and retain the same review hash.

## Required human action

The manufacturing authority and a distinct countersigning authority must independently accept, edit, or reject every disputed rule/marker. If either edits the charter or marker key, regenerate the candidate pack and obtain two acceptances of the new hash.

Only after both agree, edit:

`output/research/cibec-v1-authority-candidate-v2/AUTHORITY_ATTESTATION_TEMPLATE.json`

For each role, enter the real name, set `decision` to `ACCEPT`, and copy the exact review hash into `signed_payload_sha256`. The two names must identify distinct people. Codex must not invent either identity or acceptance.

## Mechanical sequence after real signatures

```bash
.venv/bin/python3.11 scripts/validate_cibec_v1_authority_review.py
.venv/bin/python3.11 scripts/materialize_cibec_v1_signed_authority.py
.venv/bin/python3.11 scripts/run_cibec_v1_source_free_qualification.py
```

Required outcomes:

- validator: `READY_TO_MATERIALIZE_SIGNED_CIBEC`;
- materializer: `SIGNED_ARTIFACTS_MATERIALIZED`;
- source-free qualification: `PASS` on all 14 semantics, replay, malformed-input, ownership, and abstention gates.

## Physical sequence after source-free PASS

1. Run the fixed 13 consumed cases through two independent primary annotators plus adjudicator.
2. Score with `scripts/score_f1_cibec_physical_qualification_v1.py`. Each annotator and the adjudicated artifact require a primary/replay pair (six successful exports per case), a bound native-resolution audit hash, peak-RSS evidence, and zero-leakage/hash-manifest attestations. Exactly three opaque cases must carry the required written-justification hashes. Every numerical, visual, latency, replay, job, unresolved, and non-line gate must pass.
3. Complete two independent CAD primitive grouping maps for the complete 13-case denominator and score with `scripts/score_f1_cad_grouping_qualification_v1.py`. Each trusted case must carry the full native V9 primitive inventory; the single registration-abstaining case is explicitly excluded from agreement only, while its replay, latency, memory, job, manifest, and leakage gates remain mandatory. Precision, recall, and F1 must each be at least 0.90.
4. From metadata only, verify at least 30 new scans across at least 15 opaque morphology groups. `research/f1_custody_v1.py` returns `CORPUS_CAPACITY_ABSTAIN` and creates no case mapping below either minimum.
5. Allocate the sufficient corpus with `scripts/build_f1_blind_custody_assignment_v1.py`. Provide a custodian-only blinding salt. Keep scan hashes, morphology membership, and that salt in the custodian-only map; none may appear in public annotation assignments.
6. Annotate and sequester the new corpus. Do not run the detector on it during corpus construction.

## Current state

The V2 validator is fail-closed and currently reports only:

- `MANUFACTURING_AUTHORITY_NAME_REQUIRED`;
- `COUNTERSIGNING_AUTHORITY_NAME_REQUIRED`.

No physical asset or prediction was opened while producing this handoff.
