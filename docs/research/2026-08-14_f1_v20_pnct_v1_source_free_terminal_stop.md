# V20 PNCT_V1 source-free terminal STOP

Date: 2026-08-14  
Lane: research-only, source-only  
Retained baseline: V16 unchanged  
Real scans opened: no  
Locked labels opened: no

## Ruling consumed

The complete verified GPT-5.6 Sol Pro ruling is preserved at:

`docs/research/2026-08-14_gpt56_sol_pro_pnct_v1_ruling_verbatim.md`

It authorized exactly one terminal `PROJECTIVE_NORMAL_CHORD_TRANSPORT_V1`
tranche and prohibited V20b, threshold changes, fixture replacement, or rerun
rescue after any failed gate.

## Executed result

The one source-free execution is preserved under:

`output/research/f1-new-parts-20260813/v20_pnct_source_free/`

Observed gates:

- SF01 standard fragmented 90-degree offset fold: PASS.
- SF02 cropped offset fold: PASS.
- SF03 fragmented 60-degree / 6s offset fold: PASS.
- SF04 mandatory noncircular C2 offset fold: **FAIL**.
- Nine mandatory alias families: zero activations, PASS.
- Replay: byte-identical, PASS.
- Metamorphic serialization: **FAIL** for axis/seed reversal and some rigid transforms.
- P95 source-free candidate time: 0.1863 seconds, PASS.
- Peak process RSS: 43,384,832 bytes, PASS.
- Cache layout: 64 + 192*N bytes, PASS.

The summary is in `report.json`; the fixed cache is in `source_free.pnct`.

## Native-resolution visual finding

All four positive audits were rendered at 4096 by 4096 pixels and inspected.
The mandatory mismatch is:

`audits/SF04_noncircular_audit_4096.png`

The audit visibly shows a genuine two-skin, constant-offset noncircular fold.
Valid short chords and a smooth noncollapsed midpoint path are present where
both traces survive. However, deterministic 20% source-mesh deletion removes
different phase fragments at many axial stations. Only stations S02, S05, and
S08 retain all 12 bins and pass. The other eight stations have isolated or
multiple missing owned bins, so the frozen `8/11` station coverage gate fails.
This is the expected observability failure described by Pro: the physical
constant-thickness feature is present, but the opposing skin is not observable
often enough after fragmentation.

The standard-fold activation audit was also inspected. Its chords join the two
offset skins, the amber midpoint sweep is noncollapsed, both tails continue,
and there is no visible cap, opening wall, seam, cycle, unrelated-panel chord,
or three-layer branch.

## Terminal decision

PNCT_V1 is retired. Lowering `7/10` or `8/11`, changing deletion, modifying the
fixture, relaxing byte identity, or adding a V20b would be prohibited rescue.
U02, C017, the locked 35 scans, held-out U02-U05, runtime/main, and production
were not opened because the mandatory source-free gate did not pass.

V16 remains the retained research baseline with no changes.
