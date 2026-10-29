# TEST-B one-time result: zero-shot lanes (2026-10-25)

> **Ledger.** This is the programme-sealed TEST-B, consumed by rule (OD-1). It is **not** an unseen-family accuracy claim. It was run exactly once, under the pre-registration below. **TEST-B is now CONSUMED.** Never re-run it, and never tune on it.

## Custody chain

| Step | Record |
|---|---|
| Pre-registration | `PREREG.json`, sha256 `86043f50d431f077b74f6a5028db905cfac91cec3a785f964a3732a32d1363a3`. Freeze-ledger NOTE seq 12, `frz_cd236c31b560`, entry `1c38750c…` |
| Run binding | Freeze-ledger RUN seq 13, `frz_5008ef49bcc8`, entry `37deae20…`. Appended before the first lane process started |
| Runs | 61 scans × 3 lanes = 183 processes. Each went through `tools/guarded_slot.sh <sid> --final-testb` (gold guard, one slot, OMP 1). The lane re-checks the guard and checks the pinned hashes. Result: **183/183 status OK, pins_ok, 0 failed scans**. Outputs are in `runs_TESTB_CONSUMED/` |
| Scoring | Done once: `score_testb.py` → `RESULTS.json` (sha256 `7e2792c9…5b91b`) and `RESULTS_per_case.json` (`9cd96fc7…d4bf`) |
| Results binding | Freeze-ledger NOTE seq 14, `frz_a56c9989cbc7`, entry `d2829fda…` |

Fixed in the pre-registration:
- **Code fingerprint** `af87df2f…1441c`, covering 60 files: the lane, Z3, Z2, D1, the guard, the E3 metrology modules and the eval writer.
- **Config** `config_v1.json`, sha256 `dec044e8…f4a`.
- **Threshold rule:** a fixed 0.5 on the GBM probability.
- **Scorer:** the frozen `score_case` matcher (5° / 4 mm / 0.6, Hungarian, apex locus, raw frame).
- **Metrics:** family bootstrap, 10 000 resamples, seed 20261025.
- **Failed scans** count as all-FN.
- **Gold of record:** `ai_bend_lines_zs_v1_SEALED`, the OD-2 re-audit; 61 label shas pinned. The old FRESH labels are a secondary column.

## Results (pre-registered)

Gold of record (re-audit): 61 scans / 50 families / 317 gold lines. Zero-bend controls: none.

| Lane | P | R | **micro F1 [95 % CI]** | **family-macro [95 % CI]** | TP / FP / FN | Runtime per scan, median / P95 / max |
|---|---|---|---|---|---|---|
| **Primary: zs_lane_v1** (Z3 + Z2 pooled GBM + D1 snap) | 0.872 | 0.839 | **0.855** [0.764, 0.926] | **0.855** [0.782, 0.918] | 266 / 39 / 51 | 8.0 / 20.2 / 24.0 s |
| Secondary 1: Z3 alone | 0.890 | 0.814 | 0.850 [0.746, 0.928] | 0.857 [0.787, 0.917] | 258 / 32 / 59 | 3.0 / 10.6 / 20.3 s |
| Secondary 2: Z2 alone | 0.800 | 0.808 | 0.804 [0.716, 0.875] | 0.785 [0.690, 0.872] | 256 / 64 / 61 | 4.6 / 9.4 / 10.7 s |
| A0 baseline (phase 1, old labels) | 0.722 | 0.198 | 0.310 [0.236, 0.376] | 0.252 | 65 / 25 / 264 | 229 / 432 / 600 s |

Notes on the table:
- Runtime is the per-process wall time on a single thread. It includes interpreter start-up and the model load. Peak RSS was at most 5.9 GiB for zs.
- Every lane failed on 0 scans.

**Secondary column: old FRESH labels** (2026-10-17, 15° floor, not re-audited; 329 gold lines):

| Lane | micro F1 [CI] | family-macro [CI] | TP / FP / FN |
|---|---|---|---|
| zs | 0.893 [0.811, 0.952] | 0.875 [0.804, 0.936] | 283 / 22 / 46 |
| Z3 | 0.889 [0.791, 0.952] | 0.877 [0.810, 0.935] | 275 / 15 / 54 |
| Z2 | 0.835 [0.753, 0.902] | 0.803 [0.706, 0.887] | 271 / 49 / 58 |

**Verdict under the pre-registered decision rule.**
- The primary micro F1 of 0.855 is **≥ 0.85**, and it lies inside the pre-registered expected range of 0.84–0.92.
- Family-macro is also 0.855.
- The 95 % CI [0.764, 0.926] reaches below 0.85. With 61 scans and 50 families, this is a **narrow pass, not a confident one**.
- Against their pre-registered ranges, Z3 (0.850) fell inside [0.76, 0.88], but Z2 (0.804) fell slightly below [0.82, 0.92].
- Integrating Z2 still helps over Z3 alone (+0.005 micro). But the DEV margin of +0.030 shrank on TEST-B, and that difference is within noise.

**Re-audit vs old labels.** All three lanes score about 0.04 higher against the old FRESH labels.
- The re-audit changed the line count on 5 scans: 50278001 ×2 went from 9 to 5, 49291000 from 9 to 8, 36192000 from 5 to 4, and 23976000 from 3 to 1.
- Mostly this merges a fold split by windows into one line, which is the GOLD_SET_V2 window rule. The lanes still emit the split fragments, so those become SPAN FPs and FNs.
- The re-audit is the stricter gold, and the gold of record.

## Post-hoc error analysis of the TEST-B misses

> **POST-HOC.** Everything in this section was done after scoring, on consumed TEST-B outputs. It feeds future work only. No lane, threshold or rule is changed because of it, and nothing here may be used to tune or re-run TEST-B.

**zs error budget** (diagnose: nearest counterpart within 15° / 15 mm):
- FN 51: SPAN 28, MISSED 16, OFFSET 7.
- FP 39: SPAN 24, SPURIOUS 13, TAKEN 2.
- Of the SPAN errors, 18 FNs have a prediction *longer* than gold and 10 have one shorter. 20 SPAN FPs are *shorter* than gold and 4 are longer.
- The errors fall on 25 of 61 scans. 36 scans are perfect.

| Error class | Scans | Lines (FN / FP) | What happens | General fix to try on new data |
|---|---|---|---|---|
| **1. "Same two plates?" span convention** | 26687021 ×2, 50278001 ×2, ~8 singles | ~28 FN / ~24 FP | Two ways to fail. **Over-merge:** short separate tabs on one base (gold B2–B4 and B7–B10 of 26687021 are 24–65 mm tabs) come out as one 150–180 mm line. **Under-merge:** a 433 mm fold broken by windows (50278001) comes out as 256 + 70 + 70 mm fragments, which the old labels also had as separate lines. | A per-station flange-continuity test along the line, run on the selected lines (a 2-flat fit on both sides at each station). Split where a flank disappears; join collinear fragments only when both flanks continue through the gap at the same plate offset. On DEV a blind gap-merge was noise (+0.003 at 2 mm, −0.005 at 20 mm), so it needs geometric evidence, not a gap length. |
| **2. Tiny bends next to a long fold** | 49553000 | 6 FN | Six 6–7 mm bends 12 mm from a 100 mm line. The lane emits the long neighbour line instead. | Short-bend birth: a lower length floor at high slice support, and slicing along short local axes. |
| **3. Birth failure on a small part** | 44001000 (50 × 17 × 18 mm, 20 k vertices) | 6 FN | Z3 found 0 axis families: Gauss-map peaks fall below the 20 mm² area floor on a coarse small mesh. Z2 found 0 lines. | Make the area floor scale-relative (fraction of total area only) and use the D1 thickness-normalised peak test. |
| **4. Selector or flange-graph miss** | 49566000 | 2 FN | Z3 had 152 candidates but the selector kept none; Z2 found 0 lines. | Inspect the overlay. Candidate stream recall is likely fine, so this is a selection problem. |
| **5. Spurious lines** | 17139000 (4), 38539000 ×2 (2 each), 5 singles | 13 FP | Cut-wall and seam lines on a part with features, plus twins. | The discriminative strip-evidence test proposed by Z3 (monotone normal sweep at the measured radius). |

Comparison with DEV:
- The DEV OOF budget for zs was FN MISSED 13 / SPAN 13 / OFFSET 5 and FP SPAN 22 / SPURIOUS 8.
- TEST-B has relatively **more SPAN errors**, i.e. the window/tab convention. That is the class where the re-audit gold differs most from the conventions the selector was trained on.
- The **drop from DEV 0.944 to TEST-B 0.855** matches the audit's honest estimate of ~0.88 (0.84–0.92) and its touched-vs-untouched gap.
