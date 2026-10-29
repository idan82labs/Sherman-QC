# F1 V19 PTCP_V1 terminal source-free STOP

Date: 2026-08-14  
Objective preserved: improve blind scan-only bend-line F1 and generalization.

## Ruling and bounded implementation

GPT-5.6 Sol Pro authorized exactly
`PERSISTENT_TUBE_COMPONENT_PARITY_V1` (`PTCP_V1`). The complete ruling is
preserved verbatim in
`docs/research/2026-08-14_gpt56_sol_pro_ptcp_v1_ruling_verbatim.md`.

PTCP_V1 was implemented as an isolated research-only source-free/native-mesh
sidecar. It does not modify V16, runtime, main, production, gold, labels, CAD,
or any expected-count input. For each eligible candidate it constructs the
frozen candidate frame, clips the native mesh to the specified 16-sided prism,
builds geometry-keyed triangle fans and component adjacency, measures exact
triangle distances, and runs the prescribed batched H0 union-find filtration.
The cache contract is one 64-byte header plus one fixed 224-byte candidate
record; no source point, vertex list, face list, or source cloud is serialized.

Frozen implementation hashes at terminal execution:

- core: `2fd2b48aa20b2a7e831e830312d5212013ad6ad785d48ba769205f2378cf16f1`
- construction tests: `a76cce7b2c5d43a10ef5264ea15b1cd82608943464ecdcc488e8a0c3694f5eee`
- visual-audit renderer: `5f48bc497f400495e924842563d569b484ed0134edf1e382f70d17695e2dda3f`
- preserved ruling file: `0c890e47d2a90ef58fcb30a798a2cad22624977c71b24a8d016bd1db16abeca2`
- ruling source text before the archive's final newline:
  `c6e5eedac19c41e5c3a6a1e7dc9127958774534215871eeb7f3cb3eacb344984`

## Execution provenance

The system Python environment did not contain `pytest`, so that launch did not
execute a test. The project `.venv/bin/pytest` was then used.

Two early fixture-conformance defects were excluded before the frozen outcome:
the first separating-wall fixture did not touch either skin; its replacement
lay outside the frozen radius-80 prism and was clipped away. The valid fixture
uses native wall vertices at radius approximately 65, inside the frozen ports,
and produces the intended connecting fan. No architecture predicate was changed
in response to source-free semantics.

One representation-only correction was made before the final frozen run:
candidate-frame coordinates were canonicalized at `2^-40 * s` after a rigid
transform changed `death_q` from 8193 to 8192 by one floating-point ULP. This
rounding is far below the ruling's `s/1024` geometry quantum and did not change
any constant, gate, mask, persistence ratio, port, or construction rule.

## Source-free construction result

The final valid run passed both required positive constructions:

- two separated skins with the prescribed persistent two-component interval;
- the bounded finite-thickness positive variant.

It also passed all eight mandatory negative constructions, including a
connecting wall, one-skin-only evidence, extra persistent component, rim/free
edge, nearby panel pairing, seam, corner-cycle, and persistence-duration
failure. The semantic record stayed fixed under the tested rigid transforms:

- `retain=true`;
- `birth_q=1025`, `death_q=8193`;
- port masks `FF/FF/FF/FF/00/00/00/00`;
- 564 fans, 240 components, 2,792 edges.

The 0.5x and 2.0x scale metamorphs produced the same complete record bytes.

The source-free gate nevertheless failed its explicit rigid-metamorphic byte
identity requirement. Geometry, decisions, masks, filtration indices, fan
count, component count, and edge count remained identical, but the topology
hash changed:

| presentation | topology hash |
|---|---|
| base | `f459faad25d8a81855de10e8fa9f96d4` |
| 17-degree rigid transform, inverse normalized | `ade1d959da3b8db2f5fae21025849a5a` |
| 53-degree rigid transform, inverse normalized | `4c0172eb1e4cf191c59610d06fec5143` |
| 101-degree rigid transform, inverse normalized | `301b1af6a940f9328803ea0350dae10f` |

The frozen suite stopped on this failure (`3 passed, 1 failed` with `-x`), as
required. Later orientation/permutation/replay cases were not executed because
the Pro ruling says any source-free failure terminates PTCP_V1 before U02.

## Native-resolution visual audit

The exact 4096 x 2304 audit artifact is:

`output/research/f1-new-parts-20260813/v19_ptcp_source_free/hash_failure_audit_4096.png`

SHA-256:
`ecbc18ebc4a4c46cfce32cee6ffafa6af4c01d757859af350dd22c09160fc3cc`

It was manually inspected at native source resolution and in a readable scaled
view. The cyan base and pink/amber/violet inverse-normalized rigid-transform
meshes coincide in axial-by-arm-A, axial-by-arm-B, and transverse projections.
No displaced fan, extra component, missing boundary, or visible port change is
apparent. Maximum inverse-normalization vertex residuals are
`1.172e-13`, `7.105e-14`, and `5.859e-14` model units.

The visual evidence therefore distinguishes the failure sharply: the observed
two-skin geometry and all semantic decisions are visually/geometrically
equivalent, while the canonical topology bytes are not. This is a deterministic
canonicalization/fingerprint failure, not evidence that the geometry moved or
that the positive became a different physical construction.

## Terminal ruling

`PTCP_V1_SOURCE_FREE_STOP`

The locked Pro protocol makes the metamorphic byte/hash mismatch terminal.
There is no topology-hash patch, canonicalization rescue, threshold alteration,
rerun, V19b, U02 activation test, C017 source-only test, or gold opening.

V16 remains retained and frozen at 35 scans / 25 families, 132 TP / 60 FP /
44 FN, precision 0.6875, recall 0.75, F1 0.717391, and 4/35 exact-count cases.
PTCP_V1 is preserved as a negative result and is not promoted.
