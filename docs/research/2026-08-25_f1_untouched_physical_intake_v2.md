# Sherman QC untouched physical intake V2

Date: 2026-08-25

## Purpose

This intake separates data custody from model development. It allows a data
custodian to validate and hash new PLY scans while withholding filenames,
part identities, morphology groups, manufacturing instances, scan sessions,
calibration identities, truth, expected counts, predictions, and blinding salt
from the model lane.

The model implementer must not create or sign the private input. A custodian
who is not a model implementer prepares it and retains the private receipt.

## Private input contract

The JSON input uses schema `F1_PHYSICAL_INTAKE_CUSTODIAN_V2` and contains:

- cohort `UNTOUCHED_QUALIFICATION`;
- distinct custodian and model-implementer role identifiers;
- a secret-derived 64-character lowercase hexadecimal blinding-salt hash;
- the frozen acquisition-protocol SHA-256;
- one case record per physical scan with a relative PLY path, opaque SHA-256
  tokens for morphology, manufactured instance, scan session and calibration,
  units fixed to millimetres, completed coverage/quality review, and a declared
  relationship to prior designs.

No truth, bend count, prediction, CAD mapping, filename-derived identity, or
part number is permitted in a case record.

## Qualification arithmetic

Repeated captures of the same manufactured object do not inflate confirmation
capacity. A morphology family qualifies only with at least two independently
manufactured instances. Readiness requires at least 30 qualifying independent
instances across at least 15 such families. Smaller valid deliveries are
accepted as capacity-pending and may be combined later by the custodian.

## Validation

The gate rejects:

- malformed, empty, unsupported, or unbounded PLY inputs;
- missing XYZ vertex properties;
- duplicate scan content or scan-session tokens;
- any scan hash already present in development-consumed custody;
- path traversal, non-millimetre units, incomplete visual quality review,
  forbidden fields, or role overlap.

The public manifest exposes only opaque case IDs, scan hashes, byte/vertex
counts, PLY format, units, and completed quality flags. The private receipt
retains all custody mappings.

## Command

```bash
.venv/bin/python3.11 scripts/prepare_f1_physical_intake_v2.py \
  --private-input /custodian/private_intake.json \
  --intake-root /custodian/scans \
  --lineage-registry data/bend_corpus/custody/physical_lineage_v2.json \
  --workspace-root . \
  --public-output /shared/opaque_intake.json \
  --custodian-output /custodian/private_receipt.json
```

An exit status of 0 means the 30-instance/15-family gate is ready for a
custodian seal. Exit status 2 means the intake is valid but capacity remains
insufficient. Validation errors fail without publishing either output.

## Current status

The local workspace contains 38 unique physical scans, all development
consumed. Untouched eligible capacity remains 0/30 scans and 0/15 verified
two-instance morphology families. The pending August 17–20 geometry has not
been materialized locally, so it has not been opened or counted.
