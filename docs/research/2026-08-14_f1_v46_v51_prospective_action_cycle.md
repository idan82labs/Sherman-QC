# F1 V46–V51 prospective action cycle

Date: 2026-08-14  
Status: research-only; no production/runtime promotion

## Contract

- Source PLY geometry only for every model decision.
- No CAD, drawing, manual bend location, filename feature, or part identity was exposed.
- Delivery bend counts were visible before execution and are therefore contamination-only context, not formal F1 gold.
- Every funded/retired mechanism was checked on native-resolution multiview line overlays.
- Each prospective failure created a new frozen successor; previous versions were not silently retuned.

## Scan evidence

| Case | Scan | SHA-256 | Delivery count context | Role |
|---|---|---|---:|---|
| P01 | `47247004_03.ply` | `8935b8a8e8aa429d046b0b93e7eb7598d79fc4e2ea4a160117300b16d957fb14` | 2 | V46 development |
| P02 | `50567000_05.ply` | `80472dfa8819ee30434ec759edee846d10f2ecb275ae2891315f97c3ae9d0863` | 2 | V46 prospective pass |
| P03 | `50539000_05.ply` | `85a7f8cca6bcd457f7e78fd7976c35a4c8901e1363317aeb883d4039c27aad7a` | 2 | V46 prospective failure / V47 development |
| P04 | `50553000_04.ply` | `581c7de10fc6b08790d7e31a6c7a280d99903cb89d689021b2410c8199ebe3b8` | 2 | V47 prospective failure / V48 development |
| P05 | `50537000_04.ply` | `ce3de3af51b685f59516a8bfac5458f809c4a267c75229b8353d6f06243b5486` | 2 | V48 prospective failure / V49 development |
| P06 | `50535000_04.ply` | `9c84cc623d9b8b699a780cb322be15031731354f03023f853f04d36037853f2d` | 2 | V49 prospective test; V50/V51 development |
| P07 | `47555020_1.ply` | `3b3e8c9f04e3688d58e4538acd0fa15850614feef8fb40d220a842f6ef1a5636` | 1 | Independent V51 validation |

## Version decisions

### V46 — elongated-lane consolidation

P01 reduced five raw predictions to two visually coherent longitudinal fold axes. P02 stayed at two. P03 prospectively falsified the route restriction and perpendicular-only microfragment rule: the same elongated geometry arrived through a small-pool route and retained four predictions. V46 is frozen as failed.

### V47 — route-independent elongated-lane consolidation

V47 removed the route restriction, suppressed any gated microfragment shorter than 5% of part length, and reused the retained V27 stitch tolerance. P01/P02/P03 became 2/2/2; U01–U04 stayed 5/4/3/4. P04 prospectively exposed a separate nested-partial duplicate mechanism, so V47 was retained but insufficient.

### V48 — nested-partial duplicate NMS

V48 suppresses a shorter line only when it is at most 60% of a longer line, within 10 degrees and 8 mm, and at least 70% covered. P04 reduced 3 to 2 and visually retained the two physical side-wall hinges. P05 prospectively removed one nested long duplicate but remained at three because two near-equal short cross-source lines represented one end hinge.

### V49 — short-span cross-source duplicate NMS

V49 is deliberately separate from V48. It applies only when both lines are short relative to the longest prediction (at most 15%), near-equal in length (at least 80%), from different source families, within 15 degrees and 2 mm, and at least 90% overlapped.

P05 changed 3 to 2 by retaining `plane_lattice::PLI_RP1_RP5` and suppressing `scan_mesh_adjacency::SMA4`:

- pair span / longest span: `0.080531770521`
- length ratio: `0.949517029837`
- axis angle: `14.647003533622°`
- transverse offset: `1.247743589142 mm`
- shorter-span overlap: `0.977601765037`

The native overlay shows one longitudinal fold and one perpendicular end fold.

### V50 — longest-incumbent ownership correction

P06 prospectively exposed an inherited V45 ordering bug. V16 produced a full `CB1_238` axis and a 14 mm `CB1_102` fragment of the same physical hinge. V45 accepted the short fragment first and then suppressed the full axis. V50 preserves V45 completion semantics but canonicalizes incumbents longest-first before NMS.

P06 now retains the 724.715 mm `CB1_238` full hinge and suppresses `CB1_102`. P01–P05 and U01–U04 remain unchanged after V47/V48/V49 composition.

### V51 — single-axis parallel-return recovery

After V50, P06 still had only the lower of two visually evident parallel transitions. The frozen candidate pool contained a distinct `CB2` cylindrical family on the upper transition. V51 activates only when exactly one long cylindrical axis survives and a different cylindrical family supplies a long, parallel, substantially overlapped candidate at a material offset.

P06 selected `cylindrical_standard::CB2_597`:

- selected support score: `59347.15882211622`
- selected length: `593.780261 mm`
- axis angle: `0.00044981476°`
- transverse offset: `15.209029158992 mm`
- shorter-span overlap: `1.0`

The final P06 overlay shows two distinct axes on the lower and upper web transitions. On independent P07, V51 stayed inactive (`incumbent_not_long_cylindrical_axis`) and preserved the single visually dominant hinge.

## Final counts

| Case | V51-composed count | V51 gate |
|---|---:|---|
| U01 | 5 | inactive |
| U02 | 4 | inactive |
| U03 | 3 | inactive |
| U04 | 4 | inactive |
| P01 | 2 | inactive |
| P02 | 2 | inactive |
| P03 | 2 | inactive |
| P04 | 2 | inactive |
| P05 | 2 | inactive |
| P06 | 2 | active; recovered `CB2_597` |
| P07 | 1 | inactive |

## Visual evidence

- P05 V49: `output/research/f1-v49-prospective-50537000-04-v1/visuals/P05_v49_overlay.png`
- P06 V16 before correction: `output/research/f1-v49-prospective-50535000-04-v1/visuals/P06_v16_overlay.png`
- P06 V45/V49 inherited failure: `output/research/f1-v49-prospective-50535000-04-v1/visuals/P06_v49_overlay.png`
- P06 V50 full-axis correction: `output/research/f1-v49-prospective-50535000-04-v1/visuals/P06_v50_v49_overlay.png`
- P06 V51 final: `output/research/f1-v49-prospective-50535000-04-v1/visuals/P06_v51_overlay.png`
- P07 V51 independent validation: `output/research/f1-v51-prospective-47555020-1-v1/visuals/P07_v51_overlay.png`

## Verification

Focused and inherited suite: `45 passed in 60.28s`.

The suite covers the no-label isolated runner, V41–V51 composition primitives, permutation determinism, retained stitching/completion behavior, and the new duplicate/recovery guards.

## Claim boundary and next action

This cycle materially improves prospective count agreement and visual bend placement, but it does **not** establish formal F1 because P01–P07 lack trusted manual bend-line annotations. V51 is retained as a research candidate after one independent safety validation. The next scientifically useful action is another untouched, more elaborate scan with manual visual adjudication before execution, followed by a location-labeled holdout as soon as trusted bend-line annotations are available.
