# F1 V17 mesh-adjacency axial-hysteresis terminal STOP

## Ruling

`AUTHORIZE_MESH_ADJACENCY_AXIAL_HYSTERESIS_RECALL_V17` is **STOPPED**.

The authorized predicate was implemented exactly once, without a threshold
sweep or rescue tuning. It was falsified during the mandatory label-closed
U02--U05 regression, before the one-shot 35-scan development-gold evaluation.
The 35-scan gold regression was therefore not consumed.

V16 remains the retained frozen research baseline. Runtime/main and production
remain closed.

## Frozen implementation

- Predicate: `scripts/select_f1_full_mesh_adjacency_hysteresis_v17.py`
- Unified runner: `scripts/select_f1_mesh_route_v17.py`
- Independent validator: `scripts/validate_f1_mesh_route_v17.py`
- Unit/invariant tests:
  - `tests/test_select_f1_full_mesh_adjacency_hysteresis_v17.py`
  - `tests/test_validate_f1_mesh_route_v17.py`
- Frozen V11 duplicate predicate SHA-256:
  `104e2f4c8bb3b3b545e3699442cb38582b362d18ec9e3ed84b3be61f08dadd58`
- Professor ruling archive:
  `docs/research/2026-08-14_gpt56_sol_pro_v17_adjacency_hysteresis_ruling_verbatim.md`
  (`759deb8e0c0e492afe6e8c0327163b8c6b6909669068605172e022938e4eddf7`)

The implementation is append-only, imports the exact frozen V11/V4 `_similar`
duplicate predicate, uses ordinary binary64 comparisons, and ranks eligible
rescues by the preregistered lexicographic key. It changes neither V11 nor V16.

## Test evidence

The implementation/invariant suite passed:

```text
29 passed in 0.98s
```

The suite covers every eligibility boundary, invalid/missing/nonfinite P90,
incumbent-first suppression, candidate-ID tie-breaking, candidate-order
permutation, an independent predicate/ranking reconstruction, frozen duplicate
predicate hashing, and unchanged V11/V16 route tests.

## Label-closed activation preflight

The preflight used source-only candidate pools and frozen V9/V4 models. It did
not read manual labels or development gold.

| Scan | Frozen route | V17 rescues | V16/V17 bytes |
|---|---|---|---|
| C017 | native full V11 | SMA3, SMA1 | changed (intended activation) |
| U02 | native full V11 | SMA1, SMA2 | **changed (mandatory failure)** |
| U03 | bounded V15 | none | identical |
| U04 | bounded V15 | none | identical |
| U05 | bounded V15 | none | identical |

U02 was a frozen exact `4 TP / 0 FP / 0 FN` V16 case. The Pro ruling explicitly
required its complete V17 serialization to remain byte-identical. V17 instead
appended two lines and changed the count from four to six. This alone is a
terminal falsification gate.

Frozen U02 SHA-256 comparison:

- V16: `0acc64bfcfb0ca66194c8f5f49466d59d632b454a7236d5e82f3b36fb5ebe9dc`
- V17: `6eb1436ee3fe4af0d43a2207421063595683c9c01e34167234d2ceb6bd20a693`

U03, U04, and U05 V17 outputs are byte-identical to V16.

## Mandatory native-resolution visual audit

Artifact:
`output/research/f1-new-parts-20260813/v17_preflight/U02_v17_rescue_native_audit.png`

Artifact SHA-256:
`73092531cc33279ee0318b89ca3d131a35f38faecf1fa3f88ed8698f5e669d31`

The four-view audit shows all four frozen V16 incumbents in blue and both
unshortened V17 rescues in red:

- `SMA1` occupies a terminal/perimeter corner segment. Its endpoints lie on the
  exterior terminal geometry rather than spanning a distinct material fold.
- `SMA2` occupies a short exterior/thickness-edge segment. It is not an
  independent fold and would duplicate already-explained wall/base structure.

Both match the professor's predicted failure mode: long or accurately sampled
perimeter/thickness/corner evidence can satisfy the axial certificate without
being a counted bend.

## Why the 35-scan gold regression was not run

The locked U02 no-op gate is independent of labels and had already failed. The
professor's response says any failed gate is architecture STOP and forbids
changing the constants, priority, or duplicate semantics. Opening development
gold after terminal source-only falsification would add no admissible evidence
and risk post-hoc tuning. The one-shot development evaluation remains unspent.

## Preserved baseline and next scientific question

V16 remains at 132 TP / 60 FP / 44 FN on the frozen 35-scan development corpus
(precision 0.6875, recall 0.75, F1 0.717391), with U02--U05 at 15 TP / 0 FP /
0 FN and byte-identical deterministic replay.

The failure establishes that axial persistence, localization, continuity, and
two-sided normal evidence are insufficient on their own. The missing source-only
capability is an endpoint/ownership discriminator that separates independent
interior material folds from terminal silhouettes, openings, thickness faces,
and corner-cycle closures. Any next intervention requires a fresh architecture
identity and fresh GPT-5.6 Sol Pro authorization; V17 may not be rescued.
