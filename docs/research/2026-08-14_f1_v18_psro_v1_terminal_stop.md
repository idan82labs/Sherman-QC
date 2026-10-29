# F1 V18 PSRO_V1 terminal source-only STOP

Date: 2026-08-14  
Objective preserved: improve blind scan-only bend-line F1 and generalization.

## Ruling and implementation

GPT-5.6 Sol Pro authorized exactly `PAIRED_SKIN_RIBBON_OWNERSHIP_V1`
(`PSRO_V1`). The complete ruling is preserved verbatim in
`docs/research/2026-08-14_gpt56_sol_pro_psro_v1_ruling_verbatim.md`.

PSRO_V1 was implemented as an isolated research-only native-mesh sidecar. It
adds no V17 thresholds or decisions and changes no runtime, main, production,
V16 candidate pool, or V16 output. It uses one transient Open3D ray scene,
raw pre-merge planar face owners, a bounded shared-edge transition-parent
graph, ten primary rays, two reciprocal center rays, and fixed 224-byte
candidate records plus a 64-byte scan header.

Frozen code hashes at execution:

- core: `8db85b027fa10daa87373542e7641c9bbc71ed9a8f44eae2e997bd35965ee8d4`
- runner: `8572a0ecd8c8558a1844ab518b2b2d7816e74b34e460a13578faedbc6d4ed2c1`
- construction tests: `26b5a9bddc9d9a5335234c2070ffff8e57ecc736c8fe74197cd6be5a1abbd3a0`

## Source-free construction gate

Six tests passed:

- finite-thickness paired inner/outer L-fold retains;
- symmetric candidate/mate presentation collapses to one ribbon;
- top-skin-to-thickness perimeter edge rejects;
- single miter without opposite-skin conjugate transition rejects;
- three rigid transforms preserve the semantic decision;
- whole-component face-orientation reversal preserves the decision;
- the serialized candidate record is exactly 224 bytes.

## Mandatory label-closed U02--U05 gate

U02 ran first. Three V11-rejected adjacency candidates were probed and zero
survived. The specifically locked candidates `SMA1` and `SMA2` both have
`retain=false`. The sidecar cache is separate, so the retained V16 axis and
route outputs remain byte-identical. Existing U02 V16 replay A and B both
compare byte-for-byte equal to the frozen V16 output.

U02 resource result:

- 4.39 seconds wall;
- 895,582,208-byte maximum RSS;
- 736-byte cache for three records, exactly `64 + 3 * 224`;
- zero PSRO appends.

U03, U04, and U05 remain on the frozen bounded-complex V15 route. PSRO is not
invoked, all have zero appends, and each emits only the 64-byte empty sidecar
header.

## Mandatory C017 source-only activation gate

C017 ran with gold still closed. Three V11-rejected adjacency candidates were
probed and zero survived:

| candidate | side masks | reciprocal | result |
|---|---:|---:|---|
| SMA3 | 30 / 1 | true / false | reject |
| SMA2 | 0 / 24 | false / false | reject |
| SMA1 | 9 / 12 | false / true | reject |

All fail mandatory endpoint/center station coverage or reciprocal ownership;
none has an admissible conjugate transition. The zero-activation gate is
terminal by preregistration. No native retained-ray audit is possible because
there is no activation.

A separate native-resolution four-view **failure** audit was nevertheless
rendered and manually inspected:

`output/research/f1-new-parts-20260813/v18_psro_preflight/C017/native_failure_audit.png`

SHA-256:
`05fddac158859f480a7525c4b49b1c11055c79f79f43bcc661c4a51dbfa620bf`

The three axes visually follow long, plausible structural transition bands.
Their opposite-surface hits are sparse and asymmetric, particularly at the
mandatory endpoints. The visual evidence is consistent with fragmented or
occluded scan-mesh face ownership breaking the exact station certificate; it
does not establish that the apparent folds are physically absent. That is a
representation failure, not authorization to relax PSRO after seeing C017.

C017 resource result:

- 2.04 seconds wall;
- 494,370,816-byte maximum RSS;
- 736-byte cache;
- zero PSRO appends.

Evidence hashes:

- U02 report: `0787549e6d92995e0634a8c3eba34fca0e4d8254e3feeb2fb99b81241e924fb1`
- U02 cache: `e5a07ab1a71899808e59ee830cb9a2cc83947d0bda632762f3fdf003301d56aa`
- C017 report: `fa28aac268b6d59acaee95d96937eff23819c57cab31731f2d8384f58aab1642`
- C017 cache: `c58ae9f74d608a695e2d4a6feb59b2068c2a1ec9ab0e9722f971b282a3ab34fa`

## Terminal ruling

`PAIRED_SKIN_RIBBON_OWNERSHIP_V1_STOP`

The construction is sound enough to reject the known U02 perimeter/thickness
aliases, but its exact reciprocal four-owner certificate is too strict for the
intended C017 scan evidence. Per the locked ruling there is no threshold
alteration, station replacement, rerun rescue, or V18b.

Development gold, labels, expected counts, CAD, benchmark feedback, and fresh
held-out gold were not opened. V16 remains retained and frozen at 132 TP / 60
FP / 44 FN, F1 0.717391. PSRO_V1 is not promoted.
