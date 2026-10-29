# EXP_CAD_0013 — Canonical registration replay result

## Ruling

**REJECT. GPI-C_INFRASTRUCTURE_NEGATIVE_NONDETERMINISTIC.** Fail-closed serialization removed rejected-candidate noise, but it did not make the registration semantics reproducible.

## Evidence

- 13/13 deterministic CAD reference point and normal hashes matched.
- 13/13 datum-selection booleans matched.
- 2/2 rejected datum transforms were correctly null.
- Only 11/13 semantic case payloads were byte-identical.
- Only 12/13 bend-local reversal decision sets matched.
- `RG4F8188B4D1` changed from `{CAD_EVENT_0003, CAD_EVENT_0008, CAD_EVENT_0011}` to `{CAD_EVENT_0003, CAD_EVENT_0011}`.
- Its selected datum transform moved by 3.3414 mm despite identical source bytes, deterministic reference sampling, and unchanged global transform.
- Peak RSS was 1.147 GiB, below the 16 GiB limit.

## Interpretation

This is not harmless floating-point formatting. Sequential Open3D RANSAC chose a different large coplanar datum representation, which propagated into a different local bend verdict. Canonicalization must not conceal that change.

The smallest valid successor is a source-only deterministic plane extractor operating on the exact CAD triangle normals and offsets already carried by the deterministic reference cloud. No gold, labels, expected counts, or score feedback may enter that replacement.

No F1 or physical-generalization claim is made.
