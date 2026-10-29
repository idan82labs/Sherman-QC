# V66 TFTOC real-scan one-shot: terminal retirement

Date: 2026-08-15

Architecture: `V66_TRANSVERSE_FACE_TURN_OWNERSHIP_CERTIFICATE_V1`

## Decision

**TERMINALLY RETIRE V66. Do not tune, rerun, replace an input, or rescue this architecture.**

The frozen four-case source-only execution completed once. V66 suppressed no candidates on any case and failed the locked end-to-end metamorphic contract on one case. Pseudo-truth remained sealed and no accuracy score was computed.

## Bound execution

- Preflight manifest: `output/research/f1-tftoc-v66-u01-u04-preflight-v1-20260815/manifest.json`
- Preflight manifest SHA-256: `d05c3401e9fa39c33c77b90bbcd9449a64da4c0cc837e25cfa1bd87bd341bff4`
- One-shot summary: `output/research/f1-tftoc-v66-u01-u04-one-shot-v1-20260815/summary.json`
- One-shot summary SHA-256: `d928cba646204c6e42cca9bc6e84a5bd9582fe52444f846318c9a264bc192e6e`
- Wall time: 376.4469087500038 seconds
- Peak RSS reported by macOS: 3,075,145,728 bytes
- Cases: 4 bounded real corpus meshes
- Incoming candidate counts: 4, 5, 9, 3
- Total V66 suppressions: 0
- Changed cases: 0

## Exact gate result

Three cases passed the replay/metamorphic checks. The nine-candidate case `S9567D0D3F33D80F7C915` failed:

- identity replay was byte-identical;
- V66 mutation semantics were invariant under similarity, order/winding, and exact PL refinement;
- V66 found zero eligible pairs and zero positive pairs;
- V66 made zero mutations;
- the ordinary final NMS output and audit were not invariant under the locked 0.8x similarity transform.

This is a route-level failure even though the V66 certificate itself was a no-op. The proposed architecture therefore provides no measurable benefit and does not compose with the frozen downstream route under the required similarity contract.

## Visual adjudication

There is no V66 visual delta to adjudicate: V66 changed zero cases and every identity output equals the retained V52 output. Rendering V52 and V66 overlays would produce identical pictures and cannot convert this failure into evidence of improvement. Accordingly, the native-resolution changed-case visual gate is vacuous and the pseudo-truth gate remains unopened.

This is deliberately recorded as **NO_VISUAL_DELTA / NO_BENEFIT**, not as a visually inferred success or failure.

## Scientific interpretation

The source-free fixtures established only that TFTOC's local mathematical predicates behaved as designed on constructed meshes. The real meshes supplied no candidate pair eligible for that mechanism. Separately, the ordinary final NMS exposed absolute-distance scale sensitivity on one case. V66 therefore neither addresses the observed false-positive burden nor satisfies the end-to-end invariance requirement.

## Next action

Prepare a fresh post-V66 architecture-selection packet. It must include this terminal result, the quarantined synthetic-data finding, and the requirement that the next intervention produce an auditable real-scan visual delta without using gold, labels, CAD, expected counts, filenames/identity, benchmark feedback, or post-hoc rescue tuning.
