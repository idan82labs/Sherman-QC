# Final held-out evaluation, 2026-10-28: pre-registration

Written in Stage 1, before any marking and before any engine run on these scans. After this file's sha256 is
recorded in the freeze ledger, nothing below may change. A change needs a new owner decision, recorded as a
ledger NOTE before the step it affects. Each such change is reported as a deviation.

## 1. Authority and ledger

- **Owner authorization (2026-10-28):** "Use them now, no-CAD first". This is a one-time use of the remaining held-out scans.
- **Freeze ledger** (`docs/production/runlog/freeze/LEDGER.json`):
  - seq 35, NOTE `frz_3f205ab798e9`: the authorization, plus the set manifest sha256.
  - seq 36, FREEZE `frz_231378b860b1`: the engines, matcher and bootstrap.
  - This file is bound by the next NOTE.
- **Protocol model:** `zeroshot_2026-10-25/newscans_N1`. Its records are freeze seqs 17–19, gold-ledger seqs 131–138 and the labels root `ai_bend_lines_newscans_n1_v1`.
- **After the run:** every scored scan and the canary's batch are CONSUMED. They may never again serve as held-out, tuning, calibration or selection data.

## 2. Set

The set comes from `SET_MANIFEST.json`. It is PRIVATE: it names the parts and the V/X sides, and markers never read it.

| Sub-set | Scans scored | Families (part_id[:5]) | Source |
|---|---|---|---|
| HOLDOUT10 | 9 | 9 | aug30 daily scans not in the TRAIN18 allowlist, minus 41880000_D |
| N001–N016 | 14 (N001–N014) | 7 V/X pairs | 2026-08-17/20 good/bad pairs |
| EHO4 | 4 | 2 V/X pairs | 2026-09-02 email holdout: `raw/EHO-00{1..4}.ply` |
| **Total** | **27** | **18** | |

Some members are listed but not scored:
- **N015/N016 (30820000_A01, V/X):** consumed at ledger seq 34. The N015 bytes are the seq-34 demo input, sha `f81ebf01…`. Both sides are left out, so the whole family is out.

Some items are excluded and untouched:
- 41880000_D, under a separate owner exclusion;
- the Eyal50 quarantine, which includes the only CAD of 16231000_03;
- rhino-preview face scans;
- every `*_SEALED*` path.

Two families were seen before. These are group5 IDs: the first five digits of the part number.
- **38173** (N001/N002) shares its group with DEV 38173000_02.
- **26687** (N005/N006) shares its group with FRESH 26687021.

They stay in the primary metric and are reported separately (§5).

**CAD:** no member has a STEP file. 23 have an SLDPRT. This is a no-CAD run, so E3 and the CAD path are not run.

**Prior research use:** the opaque N IDs appear in `output/research/autoresearch-loop/exp-cad-0015..0021`, from CAD-registration research. None of the scored scans is in the zero-shot programme INVENTORY. No zero-shot lane was trained, tuned or selected on any of them.

## 3. Arms (frozen at ledger seq 36)

| Arm | Lane | Code fingerprint | Pin sha256 | Models |
|---|---|---|---|---|
| **Primary** | zs_lane_v3 rev 3 | `871f23fa0f46…` | `711bd079a042…` | zs `ade6cde4…`, z3 `62e99d13…` |
| Comparator | zs_lane_v1 | `c712c15c0843…` | `0087fb2383a4…` | zs `ccc28773…` |

- Both pins passed `domains.production.zeroshot.pin.verify` (an empty result) on 2026-10-28.
- Both lanes run with a fixed threshold of 0.5 and no CAD inputs. The run tools are:
  - v3: `research/zeroshot/zs_lane_v3/run_cohort.sh` with `--strict-pins`;
  - v1: `docs/production/runlog/zeroshot_2026-10-25/integration/run_cohort.sh <tsv> <out> zs 0 1`.
- Every arm × scan pair runs through the gold guard (`tools/guarded_slot.sh`, sha `95716619…`). The guard admits a scan only when its gold is FROZEN.
- **One process at a time:** OMP_NUM_THREADS=1, nice 15, process-tree RSS < 16 GiB.
- **Exactly one run per arm and scan.** There is no re-run and no change to the threshold, rule, config or pin. A crash is recorded, never retried.
- **RUN entry first.** A freeze-ledger RUN entry is appended before the first lane process. It binds:
  - the scan TSV sha256;
  - the gold-ledger sha256;
  - the commands.
- If a pin no longer verifies at run time, the run does not start. The owner decides what happens next.

## 4. Gold protocol (frozen before any engine run)

1. **Packets.** Each scan is a byte copy with a neutral random ID (`gb_xxxxxx`), shuffled, mode 0444, with a uniform mtime. Markers get only `packets/batches.json`, the scan copies and `packets/MARKER_INSTRUCTIONS.md`, which carries the N1 MARKER_HOWTO rules. A packet holds no engine output, no CAD, no CAD-derived lines, no names and no V/X sides.
2. **Hidden canary.** One known-answer consumed DEV scan is shuffled in among the items and marked like every other item.
   - It is chosen by the N1 criteria: GOLD_SET_V2, label complete, full stage, PLY, not a previous canary, not an A1 tuning case, and a family group not used by earlier canaries or by this set.
   - Its identity is recorded only in `PRIVATE_KEY.json`. The sha256 of that file is bound in a ledger NOTE before marking starts.
3. **Two independent blind vision markers, A and B.**
   - Each is a separate agent context. They share no notes and do not see each other's folders.
   - Each marks every item, the canary included, with the `research.bench_labeling` `prep` → `pairs` → `lines` (apex locus) → missed-bend sweep → `notes.md` steps, under the GOLD_SET_V2 gold rule.
   - Markers never read the manifest, the private key, the ledgers, labels, engine output or CAD.
4. **Canary QC.** Each marker's canary lines are scored against the canary's GOLD_SET_V2 gold with the frozen matcher.
   - If line-F1 is below 0.95, that marker's whole batch is redone by a fresh marker agent.
   - If the redo also fails, marking stops and the owner is told. No canary label is ever written.
5. **Adjudicator.** A third agent, who is neither marker and sees no engine output, matches A against B with the frozen matcher (`score_case`, 5° / 4 mm / 0.6). It then resolves each disagreement on the renders and sections, and writes `adjudication.json` per item. For each scan it records:
   - `label_complete`;
   - a `part_class` tag (bent / rolled / flat / mixed);
   - `zero_bend` (true / false);
   - partial-scan status.
6. **Freeze gold.**
   - The gold is written with `write-label` (asset path = the original scan, or a byte-identical guard-clean copy) to a new labels root, `data/bend_corpus/annotations/ai_bend_lines_final_eval_2026_10_28_v1`, with provenance sidecars.
   - Each scan is set to FROZEN in `gold/GOLD_LEDGER.jsonl`.
   - The gold-ledger sha256 is then bound in a freeze-ledger NOTE.
   - This NOTE comes **before** the RUN entry. Gold is never edited after an engine output exists.
7. **Sealed human truth is not used.** That truth is the evaluator-only manifests and the EHO4 `sealed_truth/`. It is V/X pass/fail truth, not line truth. It stays closed for this line-F1 evaluation.

## 5. Metrics

**Primary metric.** It is computed for the zs_lane_v3 arm over all scored scans whose gold is label-complete, with `research/zeroshot/zs_lane_v1/scoring.py` (`score_scan`, `aggregate`).
- **Line-F1 micro:** the sum of TP/FP/FN over the scans, as in the frozen convention.
- **Family-macro line-F1:** the mean over families of each family's F1.
- **95 % CIs** for both: a percentile bootstrap over families (unit = part_id[:5]), n = 10000, seed = 20261018, passed explicitly.

**Secondary metrics.** All of these are descriptive.
1. Per sub-set: micro, macro and TP/FP/FN for HOLDOUT10, N001–N014 and EHO4.
2. **Zero-bend scans.** Their scorer disposition is ZERO_BEND_CONTROL, so they get no F1. Report:
   - the number of zero-bend scans;
   - the false lines on each one;
   - the micro including control FP lines, meaning every line predicted on a zero-bend scan is added as an FP.
3. **False lines on rolled or flat parts:** the FP count per scan whose adjudicated `part_class` is rolled or flat, and their total.
4. **Unseen-family subset:** micro and macro without group5 38173 and 26687.
5. **Comparator.** Run the same metrics for zs_lane_v1, plus a paired family bootstrap of Δmicro and Δmacro (v3 − v1), with the same n and seed.
6. For each matched line, the angle difference (engine included angle − gold), as in N1, with joggles flagged.
7. Wall time and peak RSS per process.
8. Marker A vs B agreement (line-F1) and each marker's canary F1.
9. **V/X pairs.** For each pair, the engine line count on V and on X, given as context only. It is not a defect metric.

**Decision rule.** None is pre-registered here: the evaluation is descriptive.
- This run may also serve as the zs_lane_v3 fresh-scan promotion test under `v3/PREREG_FRESH_TEMPLATE.json` (ledger seqs 21, 28, 32). That needs an owner decision, recorded as a NOTE before the RUN entry. In that case the template's rule applies unchanged.

## 6. What counts as a failure (never dropped)

- **Engine failures count as misses.** These are:
  - a crash, timeout, refusal or pin mismatch;
  - a non-OK status;
  - a missing or unreadable `eval_lines.json`.
  
  Each is scored as an empty prediction: every gold line is an FN, and a zero-bend scan gets 0 FP. Each failure is listed with its reason.
- Each arm runs once per scan. A failed scan stays failed.
- **Matcher exclusion.** If the matcher returns `EXCLUDED_BEFORE_SCORING` for a scan, that is reported as an evaluation failure. It is never a silent drop. The scan's gold lines are counted as FN in a reported "strict" micro.
- **Over-budget scans.** A scan over the v3 face budget runs on v3's pinned resampling, as frozen. A scan that exceeds the RSS limit is a failure. It is not re-run with more memory.

## 7. Exclusion rules (decided before any engine output)

- A scan leaves the scored set only by an adjudicator finding, written before the gold freeze:
  - the file is unreadable or corrupt;
  - or it is not a sheet-metal part.
  
  The reason goes in its adjudication record. A partial scan stays in, marked for formed bends only.
- A scan whose gold is not label-complete is reported but left out of the micro. This follows the frozen scorer.
- The set stays as listed. The engine runs exclude 30820000_A01, 41880000_D, Eyal50 and every `*_SEALED*` path.
- Nothing is tuned, calibrated or selected on these scans, and that includes thresholds and lane parameters. No result is used to change an engine before this report is published.

## 8. Prerequisites found in Stage 1 (open before Stage 2)

These must be resolved by an owner decision. They are not workarounds.

1. **Path guard.** `research/bench_labeling/custody.check_path` and `research/zeroshot/z3_slicing/custody.check_path` refuse paths that contain `heldout`/`holdout`, the opaque `N001`–`N016` IDs or the EHO4 part numbers. That covers:
   - this run folder (`…_heldout`), and so the packet copies;
   - the N originals;
   - the EHO `raw/` originals, under `…prospective-holdout…`.
   
   Neither `prep`/`lines`/`write-label` nor the lanes can read these scans until either:
   - the owner approves a guard-clean location for the byte copies;
   - or the owner approves a guard amendment tied to this ledger authorization.
2. **Missing slot script.** `tools/guarded_slot.sh` executes `/private/tmp/claude-501/sherman_slot.sh`, which no longer exists, because the scratch area was wiped. It must be restored before any guarded run.
3. **Gold-ledger tooling.** The scans must be registered in `gold/GOLD_LEDGER.jsonl` (REGISTERED → MARKING → ADJUDICATING → FROZEN), as `tools/gold_ledger_newscans.py` did for N1.
