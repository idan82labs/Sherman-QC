# E3 mirror identity: a mirrored part can be inspected as VALID with PASS decisions (2026-10-28)

> **Research and diagnosis only.** Live E3 (`e3-1.14.0`, inspect fingerprint `8805148f…`), contracts, pins and the ledger were not changed. `engine.py` (`bb593889…`) and `config.py` (`71c47872…`) were last modified at the 01:58 re-pin, before this work.
> - Inputs: G1 regression tier (50 synthetic DEV cases), the A1 cohort (26 runnable scans, 14 SKUs) and the E2E part 10839000 (A1 `DEV-EA3E4820DF`: full, scan_1, scan_2). No FRESH, DEV-CHECK, TEST-B, N1, HOLDOUT10, EHO4, COH or sealed input was used.
> - Runs went through `sherman_slot.sh` with at most 2 slots, nice 10 and `OMP_NUM_THREADS=1`. Engine limits were production: 180 s timeout, 170 s budget, K=2 resample net. Peak RSS was 1.26 GiB. There were 0 timeouts.
> - Machine-readable record: `e3_mirror/results.json` (every run, live and prototype). Tools: `e3_mirror/tools/`. Prototype patch: `e3_mirror/e3_mirror_identity_proto.patch`. Plots: `e3_mirror/*.png`.

## Answer

**Yes, it is a real risk.**

**10839000.** The part is chiral, and a mirror image of it is a different part. E3 still accepts its mirrored scans as VALID:
- **All 9 mirrored 10839000 scans are VALID**: full, scan_1 and scan_2, each mirrored in x, y and z.
- They carry **12–18 PASS characteristics each, 142 in total**.

**Prevalence.**

| Corpus | Mirrored scans of chiral parts | VALID | PASS characteristics |
|---|---|---|---|
| G1 | 74 | **17 (23 %)** | 17, one per case |
| A1 | 48 | **9 (19 %)** | 142, all on 10839000 |

- No case reached a decided whole-part PASS. Every accepted 10839000 scan has the part's real out-of-tolerance bends (FAIL), as in the E2E. A mirrored 10839000 that is in tolerance would get ALL_MEASURED_IN_TOLERANCE. That last point is inferred from the identical bend readings; no such scan exists in the corpus.
- **Wrong-part swaps:** 0 of 30 were accepted with a PASS. One tiny part declared as a larger one came back VALID with nothing measured.

**Fix.** A part-agnostic "mirror identity check" was prototyped and measured on a scratch copy:
- **0 of 122** mirrored scans of chiral parts are accepted; live accepts 26.
- **0 new refusals and 0 changed characteristics** on all 76 genuine G1/A1 native scans.
- Mirrors of achiral parts (the same part) stay VALID: 30 of 30.
- **Cost:** 11 of 76 scans pay extra time. A1 mean is +4.6 s and the maximum +45.5 s (one scan, 81 → 130 s). The A1 maximum wall rises from 124 s to 130 s.
- **The thresholds are in-sample.** A held-out confirmation is needed before any release.

## 1. Is 10839000 mirror-symmetric?

**No. It is chiral, but only by one cutout.**

**E1's view of the final stage S12.** The best MIRROR candidates have RMS 0.331 mm and 0.333 mm, and a 180° rotation has 0.334 mm. All are just above E1's 0.30 mm certification threshold, so E1 does not certify the stage as mirror-symmetric.

**Independent check.** The mirrored CAD was fitted onto the CAD by ICP over proper rotations (24 starts, `tools/cad_chirality.py`).

| Measure | Value |
|---|---|
| Best proper rotation | 179.95° |
| RMS | 0.344 mm |
| P99 | 0.28 mm |
| P99.9 | 6.0 mm |
| Max | 8.5 mm |
| Points more than 1 mm off | 0.51 % |

**What tells the part from its mirror** (`e3mirror_cad_10839000.png`, checked):
- **One ~15 × 28 mm rectangular cutout** in the side wall at y ≈ 49 mm. The opposite wall (y ≈ 361 mm) is solid at that place.
- The 4 small holes in each wall sit at mirror-symmetric positions. Under the 180° fit their edges are at most 2.2 mm off.
- Every formed stage S1–S12 keeps the cutout (a cluster of about 1,080 points more than 1.5 mm off). Only the flat S0 is achiral.

![10839000 vs its mirror image](e3_mirror/e3mirror_cad_10839000.png)

A mirrored 10839000 is the frame with the cutout in the other wall. Its bends and flanges read the same as the real part's, but it is the wrong part. **VALID is wrong here.**

## 2. Why did MIRRORED_PART_SUSPECTED not fire?

The trace uses live e3-1.14.0 on the A1 compile of 10839000 (`engine.py` ~509–563). The E2E used its own wizard compile, with the same class of outcome.

| | Native full scan | x-mirrored full scan |
|---|---|---|
| Datum score | 0.970 | 0.954 |
| Explained | 0.9996 | 0.9974 |
| Total | 2.011 | 1.993 |
| Datum hole mismatch | 0 | 0 |
| Pose margin | 0.0064 | 0.0075 |
| Twin hole evidence | 12/3 vs 7/9 | 11/3 vs 7/9 |
| Result | VALID, 17 PASS / 4 FAIL | **VALID, 12/12 bends, 18 PASS / 4 FAIL** |

1. **The check never ran.** It is registered only when the best pose looks *suspicious*: `hole_mismatch > 0`, or `explained < 0.85`, or `datum score < 0.90`. The values are 0, 0.9974 and 0.954, so the trigger is false. The y- and z-mirrors behave the same way, with 17 and 18 PASS.
2. **Had it run, it would still not refuse.** A diagnostic copy forces the mirrored re-registration and keeps the live decision (`diag_instrumentation.patch`). Over the 3 passes:
   - Mirrored-fit totals are 1.998, 2.008 and 2.010; proper-fit totals are 1.993, 1.995 and 2.003.
   - The gain is +0.005, +0.013 and +0.007. The rule needs `mirror_margin` 0.08.
   - On the genuine scan the same gain is −0.007, −0.0001 and +0.003. **The totals cannot tell the two apart.**
   - The stage is not certified symmetric (0.331 > 0.30), so the self-map conversion and its hole rule do not apply either.
3. **The evidence is in the holes of every panel, not the datum.** The cutout is on a side wall, so the datum-only `hole_mismatch` is 0 under both images. The all-panel hole evidence (`_hole_evidence`, already used for twins) does separate them:
   - Mirrored scan: proper pose 11/3, mirrored pose 12/2. The mirrored pose wins by 2 in all 3 passes.
   - Genuine scan: the proper pose wins by 1 to 2.

**The G1 cases fail for a different reason** (17 accepted, all STAGE_1, STAGE_K or ORDER_SWAP). The proper pose still explains 0.86–0.96 of a mirrored partial part, above the 0.85 trigger. The mirrored fit then beats it by 0.033–0.15 in total and by 0.043–0.136 in explained share. Each of these cases had exactly one decided PASS, always a bend-line position.

**F1, the wrong part reported as UNTRUSTED_REGISTRATION.** A scan with no plane matching the CAD's datum returns `NO_DATUM_PLANE_MATCHED` before any wrong-part test runs (`engine.py` ~468). This is fail-closed (INVALID, never PASS). Only the label and the operator text are affected, and it is the most common wrong-part outcome (13/30 below).

## 3. Prevalence

**How the mirrors were built.** The scan coordinates and normals were negated in one axis and the triangle winding was reversed (`tools/mirror_scan.py`; PLY and binary STL). The 32 G1 wrong/mirrored items in the gate are characteristics of only 2 cases: HT-006 (mirrored) and CR-006 (wrong SKU). Both are refused natively, so this study builds its own mirrors.

**Which mirrors count.** A mirror is counted only when the shape is distinguishable from its mirror image (`tools/chirality_fast.py`, `chirality_formed.py`). That needs both of these:
- The shown stage has a cluster of more than 20 points deviating by more than 1.5 mm under every mirror candidate.
- The final part is chiral.

When the whole part is mirror-symmetric including holes, a mirrored partial scan is the same physical object as a different formed set of that part. Examples are 15651004 (B1 ↔ B2) and 23553000. E3 correctly decodes those as UNEXPECTED_FORMED, so they are scored separately as "same part". All 40 mirrored G1 cases are chiral by holes and notches. G1 flat stages are their own mirror image and were not mirrored.

**Live outcomes.** A1 uses live-engine runs. For G1 the diagnostic copy's decision equals live: those runs stay far from the budget, and the copy only measures.

| Mirrored scans | n | VALID | VALID with PASS (PASS characteristics) | Decided part PASS | Refused |
|---|---|---|---|---|---|
| G1, chiral (x for all 40 mirrored cases, plus y and z for the 17 with E1 mirror RMS > 0.3 mm or no defined stage) | 74 | **17** | **17 (17)** | 0 | 54 MIRRORED, 3 WRONG_PART |
| A1, chiral (16 scans × x, y, z) | 48 | **9** (all 10839000) | **9 (142)** | 0 | 30 MIRRORED, 9 WRONG_PART |
| A1, mirror of an achiral part (10 scans × 3) | 30 | 30 (correct) | 27 (402) | 6 | 0 |

- **Mirror axis:** no influence beyond noise. A scan accepted on one axis is accepted on all three, except G1 UC-006, where the z-mirror was refused.
- **10839000 in A1:** full scan 17–18 PASS, scan_1 12, scan_2 17–18, on every axis.
- **The G1 misses:**
  - STAGE_1: LB-003, HT-003, CR-003, TD-003 ×3, PN-003 ×3.
  - STAGE_K: TD-005 ×3.
  - ORDER_SWAP: UC-006 x and y, PN-006 ×3.
  - Every complete-part case (COMPLETE_NOMINAL, COMPLETE_ERRORS, MISSING_BEND, EXTRA_BEND, PARTIAL_BEND) was refused.

**Wrong-part swaps** (a scan of X declared as Y). There were 30 pairs: 16 A1 pairs, mostly the nearest-size SKU, and 14 G1 pairs, mostly same family with other parameters.
- **Results:** 15 WRONG_PART_SUSPECTED, 13 UNTRUSTED_REGISTRATION (`NO_DATUM_PLANE_MATCHED`), 1 MIRRORED_PART_SUSPECTED and **1 VALID**.
- **The VALID case:** 44211000_A (36 mm) declared as 28870000 (81 mm). It came back INCOMPLETE with 0/1 bends measured and nothing decided; its datum score was 0.47 and its explained share 0.74.
- **0/30 VALID with PASS.** Wrong-part swaps are not a PASS risk in this sample. The one VALID-but-empty result could get its own wrong-part rule later, for example a low datum score with no bend found. That was not prototyped.

## 4. The fix: a mirror identity check (prototype)

The patch is `e3_mirror/e3_mirror_identity_proto.patch`, against live `engine.py` and `config.py`. It was measured from a scratch shadow tree. Everything is part-agnostic: it uses only E1's compiled symmetry candidates and E3's own fit evidence.

1. **Wider trigger.** The mirrored scan is also registered when either of these holds:
   - The best proper pose explains less than **0.99** of the scan. The live trigger is 0.85.
   - The decoded stage is *near-achiral*: an E1 MIRROR candidate is above the certification threshold but within `near_twin_rms_mm` (1.0 mm). The decoded stage is the nearest compiled stage, or the final stage when there is none.
2. **Mirror poses for near-achiral stages.** When the scan is otherwise well explained, the mirror poses are seeded from the two best proper poses through those candidates (S·T·MIRROR, refined on the mirrored scan). This runs instead of the full mirrored search. It is cheaper, and it found the right mirror pose where the coarse search had missed it.
3. **Conversion margin.** A certified mirror self-map converts a better mirrored fit into a proper pose at 0.03 instead of 0.08, so that only what still differs counts.
4. **New refusal rules,** applied after the live rule (`_mirror_identity`):
   - **Holes.** The best confirmed-minus-contradicted hole score over every fitted panel, within the ambiguity margin of each side's best, favours the mirror by at least `hole_evidence_min_diff` (2). Its total may not be lower by more than the ambiguity margin, and its explained share may not be lower.
   - **Otherwise, unless the holes favour the proper pose:**
     - The mirrored total beats the best proper total by more than **0.03**. This applies only when the mirrored fit decodes an exact stage: an inexact decode of a symmetric part produced a genuine gain of 0.079.
     - Or the mirrored fit explains at least **0.02** more of the scan, with a total that is not lower.
   - When neither image explains 0.6 of the scan, the wrong-part check decides.

**Measured.** The prototype ran on all 76 genuine natives and on every mirror or swap that live accepted. Everything live already refused stays refused, because the prototype only adds refusals.

| | Live e3-1.14.0 | Prototype |
|---|---|---|
| Mirrored chiral scans accepted VALID (G1 + A1) | **26 / 122** (159 PASS characteristics) | **0 / 122** |
| Same-part mirrors (achiral) VALID | 30 / 30 | 30 / 30, characteristics identical |
| Wrong-part swaps VALID with PASS | 0 / 30 | 0 / 30 |
| Genuine natives: new refusals | – | **0 / 76** |
| Genuine natives: characteristics changed, PASS lost | – | **0 / 76, 0** |

**Margins on genuine scans,** over 67 passes where the new rules were evaluated:
- Total gain at most 0.009 with an exact stage, against the 0.03 margin.
- Explained gain at most 0.0005, against 0.02.
- Hole gain at most 0, against 2.

**Margins on the mirrored scans that fired:**
- Total gain 0.033–0.071. Weakest: TD-003 z at 0.033; its explained gain of 0.049 also fires.
- Explained gain 0.043–0.08.
- Hole gain +2 to +3 for 10839000.

All 26 refusals happened on the primary pass.

**Runtime.**
- **G1:** 6/50 scans pay; mean +0.95 s, maximum +21 s.
- **A1:** 5/26 scans pay; mean +4.6 s per scan.
  - 10839000: +21–24 s (full scan 97 → 121 s).
  - 50757000_02_X: +45.5 s (81 → 130 s). It is near-achiral and explains 0.98, so it pays the full mirrored search on all 3 passes.
- A1 maximum wall: 124.1 s live, 130.3 s prototype, under the 170 s budget.
- Measured with 2 slots busy.

![Signals](e3_mirror/e3mirror_signals.png)

Left panel (checked): on G1 and A1 the explained-share gain and the total gain separate genuine scans from mirrored partial scans. The accepted 10839000 mirrors sit among the genuine points, at an explained gain of only +0.002 to +0.007. Right panel: for the near-achiral parts the hole score separates them. A few mirrored passes sit far left; there the coarse search missed the mirror pose, which the seeding fixes.

## Caveats and owner decisions

- **In-sample thresholds.** 0.99, 0.03 and 0.02 were chosen while looking at these G1/A1 runs. This is development evidence, not generalization evidence. A frozen, held-out confirmation is needed before promotion.
- **Single-feature evidence.** For near-achiral parts the decision rests on the distinguishing feature, here one cutout, being captured. If the scan does not see it, the prototype does not refuse. A stricter option was not prototyped: for near-achiral stages, withhold PASS when the hole scores tie. It would have cost nothing on this corpus, where genuine near-achiral passes favoured the proper pose by 1 to 3.
- **Contract tension, owner decision needed.**
  - `_mirror_self_maps` says E3 must never claim MIRRORED_PART_SUSPECTED for a stage that E1 certifies as mirror-symmetric. The live code already refuses such stages through datum holes (CR-003).
  - The prototype also refuses after conversion when the mirrored datum still fits better by more than 0.03. That happens on G1 LB-003 and HT-003, where small datum features break E1's 0.30 mm RMS symmetry.
  - G1's truth requires MIRRORED_PART_SUSPECTED for every mirrored part.
- **Live fragility.** The genuine native 15651004_01_3RD_SCAN gets a mirrored-fit gain of **0.079** under the live rule, which refuses at 0.08. Live is 0.001 away from a false MIRRORED refusal on a genuine part. The prototype excludes that case through its exact-stage condition.
- **Adoption cost.** `engine.py` and `config.py` are hashed into the zs pins. Adopting the fix means a version bump, a re-pin of both lanes, the G1 and A1 gates, the dense set and the E2E, as in the 2026-10-28 re-pin.
- **Wrong part.** The F1 label and the one VALID-but-empty swap are not safety issues. Changing the operator text for `NO_DATUM_PLANE_MATCHED` ("does not match this part's CAD, or the datum was not captured") is a UI/contract decision.

## Reproduce

Scratch: `/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/e3mirror`. It holds the diagnostic tree `repo/`, the prototype tree `repo_fix/`, the job lists and `runs/`.
1. `tools/make_jobs.py`, then `tools/batch.py JOBS runs --workers 2`. Each job runs `run_e3m.py` through `sherman_slot.sh` at nice 10.
2. `tools/analyze.py` writes `summary.json` and `collected.json`.
3. `tools/plot_cad.py` and `tools/plot_signals.py` make the PNGs.

There were 512 run records: 258 diagnostic, 120 live and 134 prototype (1 of them superseded). The E2E-part smoke runs are in `runs_e2e/`.

## Frozen for held-out (written 2026-10-28, before any held-out run)

**Prototype.** `e3_mirror/e3_mirror_identity_proto.patch` (sha256 `b25f0abe…c73764`). It is applied to live e3-1.14.0, whose `engine.py` is `bb593889…` and `config.py` is `71c47872…`.
- Resulting `engine.py`: **`9f82ff2de32f8b2c5acab25c1f0c67252361f40a2bf4dc0ee2e46bcc55a6155b`**.
- Resulting `config.py`: **`d1f827707fc048e7302477aef8674dcadb9a5b5ccc1b218fe5388ae6e98b2417`**.
- All other inspect files are identical to live.
- The scratch tree `repo_fix/` was made read-only.

**Thresholds (frozen, not to be retuned):**

| Parameter | Value |
|---|---|
| `mirror_identity` | 1 |
| `mirror_identity_explained` (trigger) | 0.99 |
| `mirror_identity_margin` (total, exact decoded stage only; also the conversion margin) | 0.03 |
| `mirror_identity_explained_gain` | 0.02 |
| `near_twin_rms_mm` (near-achiral trigger and seeding) | 1.0 |
| `hole_evidence_min_diff` | 2 |
| `ambiguity_margin` (pools) | 0.03 |
| Wrong-part guard (`wrong_part_explained_min`) | 0.6 |
| Seeds | top 2 proper poses × near-mirror maps |

The live rule stays as it is: `mirror_margin` 0.08, behind the 0.85 / 0.90 suspicion trigger.

**Held-out set.** The E3-survey DEV cohort (33 scans), minus every scan whose bytes (sha256) equal an A1 scan. Every scan is DEV_CONSUMED / CONSUMED, in the frozen dev split and not in test_b. No FRESH, DEV-CHECK, TEST-B, N1, HOLDOUT10, EHO4 or sealed data is used.
- Other compiled DEV scans: the only further DEV_CONSUMED scan with a compiled part, `sc_4d277fbe7854`, is marked unusable and is outside the frozen dev split, so it is excluded.
- Other E2E parts: none. The E2E's only other real part is 23553000, already in A1; the HAT part is synthetic.

**Runs.** Each held-out native gets three runs: live, prototype, and the diagnostic copy for margins. Each scan is also mirrored in x, y and z and run on live and on the prototype. A mirror counts as a wrong part only when the final part is chiral and so is the shown shape, as in section 3.

**Pass condition:**
- 0 mirrored scans of chiral parts accepted VALID with any PASS under the prototype;
- 0 new refusals on genuine held-out natives.

A failure is reported as it is, without tuning.

## Held-out confirmation (frozen prototype, no retuning): **FAIL**

**The set.** It is 12 DEV-cohort scans whose bytes differ from every A1 scan. All are DEV_CONSUMED, in the dev split and outside test_b.
- 4 are re-exports of A1 physical scans, as PLY: 15651004 SECOND and 3RD, 11979003_C, and 47266000_F.
- 1 is a new scan of an A1 part: 38172000 FULL SCAN 1.
- 7 are scans of 6 parts never used for tuning: 49201000_C, 45796000_A, 42288005 SCAN_1 and SCAN_2, 47959001, 37126000_D, 28289002_B01.

**The runs.** 108 in all:
- each native on live e3-1.14.0, the frozen prototype, and the diagnostic copy;
- each scan mirrored in x, y and z, on live and on the prototype.

Peak RSS was 1.29 GiB and there were 0 timeouts. Raw runs are in the scratch `runs_heldout/`.

| | Live e3-1.14.0 | Prototype (frozen) |
|---|---|---|
| Mirrored chiral scans accepted VALID with PASS (21 runs: 49201000 ×3, 42288005 2×3, 47959001, 38172000, 11979003 and 47266000 ×3 each) | **9 runs, 30 PASS** (49201000 ×3 with 6 each; 42288005 ×6 with 2 each) | **2 runs, 12 PASS** (49201000_C x and z, 6 each) |
| Genuine natives: new refusals | – | **0 / 12** |
| Genuine natives: characteristics changed | – | **0 / 12** |
| Same-part mirrors (the part is its own mirror image; 15 runs) | 12 VALID, 3 WRONG_PART (28289002) | 11 VALID, **1 new refusal** (37126000_D, y), 3 MIRRORED (28289002) |

**Pass condition: not met.** It needed 0 mirrored scans accepted with a PASS. Two scans of 49201000_C were still accepted, each with 6 PASS. The genuine-native half of the condition holds.

**What 49201000_C shows.** E1 certifies its stage S2 as mirror-symmetric (0.085 mm RMS). The only chiral features are two small clusters, of about 45 points each, deviating by up to 3.4 mm, probably small hole offsets.
- On the x and z mirrors, E3 sees no difference: total gain 0.000, explained gain −0.01, and 0/0 holes under both images.
- On the z mirror the prototype does not even run the check. The scan explains at least 0.99, and a certified stage gets no near-mirror trigger.
- The y mirror was caught (holes 2 > 0).
- So the blind spot is a stage that E1 certifies as symmetric but that carries small chiral features. Neither live nor the prototype can tell those apart.

**What 37126000_D (y) shows.** This is a new refusal of a scan that is physically the genuine part, since the part is its own mirror image.
- The proper search failed (explained 0.744), and the certified self-map conversion found no new pose.
- The prototype's total-margin rule then refused it: gain 0.276, explained gain 0.256.
- Live accepted the same scan, VALID / REVIEW_REQUIRED with 0 PASS.

**Fixed as intended.** 42288005 SCAN_1 and SCAN_2 are chiral, and live accepts all 6 of their mirrors with 2 PASS each. The prototype refuses all 6 through the hole rule (5 > 3).

**Margins on genuine natives.** The new rules were evaluated on 14 passes.
- Total gain: at most 0.002 with an exact stage; 0.079 with an inexact stage, which is the guarded case.
- Explained gain: at most 0.0.
- Hole gain: at most 0.

**Runtime.** 1 of 12 natives paid extra: 49201000_C, +10.5 s (59.3 → 70.7 s). All the others were within ±1 s.

**The 15651004_01_3RD_SCAN near-miss under the live rule** (genuine scans whose mirrored-fit gain is within 0.01 of the 0.08 refusal margin, from forced re-registration):
- Held-out: **1 / 12** genuine natives, the 3RD_SCAN PLY re-export, with gain **0.0789** on the primary pass. The live trigger was active, so this is a real exposure. Its mirrored versions, which are the same part, reach 0.072.
- G1 + A1: the only case is the A1 STL of the same scan (0.079, 0.079, 0.078). Every other genuine pass was below 0.07, apart from certified-symmetric cases, where the conversion takes over.
- The prototype never refuses this scan, because of its exact-stage condition.

**Conclusion.** The frozen prototype closes the mirror gap on chiral parts whose difference E3 can see: 10839000, 42288005 and the G1 partial stages. It does not close it on stages E1 certifies as symmetric that carry small chiral features (49201000_C). It also adds one false refusal on an achiral part whose proper registration failed (37126000_D). Both are design questions for the owner, not threshold tuning.
- **Withholding PASS.** On a stage that E1 certifies as mirror-symmetric, PASS would be withheld unless the features on the mirror image are positively confirmed. Today the E1 contract says such a stage is never called mirrored.
- **The total-margin rule after a failed conversion.** It would be skipped on certified stages.

Neither was implemented or measured here.
