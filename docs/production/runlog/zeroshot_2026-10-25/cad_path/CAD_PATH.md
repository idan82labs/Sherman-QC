# CAD path: zs_lane_v1 alongside E3 (DEV design study, 2026-10-25)

> **Ledger.** Every number here is consumed DEV data: the SPLIT_FROZEN dev scans with a compiled CAD, through the gold guard with `cad_path=True`. The guard refused the 2 OD-3 DEV-CHECK cases present before any per-scan file was read. zs probabilities are out of fold, so no CAD was used in training. The gold was only read by the frozen scorer. Nothing was written to gold; `gold_guard verify` gives 130 entries, chain intact.

## Result in one paragraph

- **No engine change was made.** Option 4 (registration seeding) failed its gate.
  - Seeds added on top of the search: no runtime or coverage gain, and the poses moved. That gave 46 characteristic changes and 7 new PASS decisions that cannot be verified.
  - Seeds replacing the search: median runtime −43 %, but P95 is unchanged. One VALID scan was lost, and one scan was registered on a wrong pose that still passed validity and published a new PASS. That is a false-PASS path.
- **Option 2 (gap-fill) is the clear win.** E3 lines plus zs lines for the bends E3 withholds, and zs lines on refused scans, took line micro-F1 from **0.955 to 0.987** (family-macro 0.845 → 0.972). FP stayed at 2. No E3 characteristic changes, so it creates 0 PASS.
- **Option 1 (cross-check)** is cheap and safe. It flags 3 of 49 VALID scans, and all 3 already carry non-PASS characteristics.
- **Option 3 (CAD priors):** the count cap is the one useful prior (zs F1 0.966 → 0.975). The K + 2 family cap cuts the zs P95 search time from 14.5 to 11.7 s.
- **Where the work belongs.** Options 1 and 2 sit after E3, in the runner/C06 layer, with E3 untouched. See "Recommended production design".

## Cohort and baseline

- **Cohort.** 51 DEV scans with a compiled CAD (E1), after the guard:
  - 49 line scans, 33 families, 302 gold lines;
  - 2 zero-bend controls.
- **E3 baseline.** e3-1.12.0 was re-run on all 51 in this study (arm `base`).
  - The stored runs lacked sc_186463746daa, so its numbers replace the integration README's "E3 alone 0.939". On the stored runs, E3 alone is 269/2/33 = 0.939.
  - VALID 49/51. The 2 refusals are both correct:
    - sc_2d3e4a623a31: MIRRORED. A true mirror (FIXES.md §3).
    - sc_3bde9996f14d: WRONG_PART, explained 0.236. Its gold says no bends are present: it is an unformed scan of a 1-bend part.
  - Runtime: median 15.2 s, P95 90.9 s. Registration (coarse + fine + refine + articulation + mirror check) is 91 % of E3 wall time, 1014 of 1111 s.
- **Scorer.** The frozen matcher (5° / 4 mm / 0.6, OUTER_APEX, raw frame) with family bootstrap (2000 draws).

| Arm (DEV, 49 line scans / 302 gold) | P | R | micro F1 [95 % CI] | family-macro [CI] | TP / FP / FN | ctrl FP lines | Can create a PASS? |
|---|---|---|---|---|---|---|---|
| E3 alone (e3-1.12.0, re-run) | 0.993 | 0.921 | 0.955 [0.898, 0.986] | 0.845 [0.728, 0.943] | 278 / 2 / 24 | 0 | (authority) |
| zs alone (no CAD) | 0.955 | 0.977 | 0.966 [0.942, 0.987] | 0.948 | 295 / 14 / 7 | 1 | n/a (display) |
| **Opt 3:** zs + count cap N | 0.977 | 0.974 | **0.975** [0.957, 0.990] | 0.945 | 294 / 7 / 8 | 1 | no |
| **Opt 3:** zs + one-sided window [−6°, +15°] + N | 0.977 | 0.970 | 0.973 [0.955, 0.988] | 0.941 | 293 / 7 / 9 | 0 | no |
| Opt 3: zs + symmetric window ±6° + N (reference) | 0.976 | 0.944 | 0.960 | 0.937 | 285 / 7 / 17 | 0 | no |
| **Opt 1:** E3 ∪ zs lines no CAD bend explains | 0.986 | 0.934 | 0.959 | 0.865 | 282 / 4 / 20 | 0 | no (flag → REVIEW) |
| **Opt 2:** E3 + zs for withheld bends only | 0.993 | 0.970 | 0.982 [0.957, 0.997] | 0.941 | 293 / 2 / 9 | 0 | no |
| **Opt 2:** E3 + zs for withheld bends + zs on refused scans | 0.993 | 0.980 | **0.987** [0.969, 0.998] | **0.972** [0.940, 0.999] | 296 / 2 / 6 | 1 | no |
| Opt 4 `add` (seeds added to E3's search) | | | 0.952 | 0.843 | 276 / 2 / 26 | 0 | **yes: 7 new PASS** |
| Opt 4 `only` (seeds replace coarse search, diagnostic) | | | 0.930 | 0.836 | 264 / 2 / 38 | 0 | **yes: 8 new PASS, 1 on a wrong pose** |

Records:
- `results/options_offline_base_rerun.json`: options 1–3 on the re-run E3. `results/options_offline.json` holds the same on the stored runs.
- `results/seed_eval.json`
- `results/e3_seed_arms.json`, with per-scan runs in `results/e3_seed_arms_runs.jsonl`.

## Option 1: detection cross-check (zs bends the recipe doesn't explain → REVIEW)

**Rule.** On a VALID inspection, a zs line (count-capped) is *explained* when it lies within 5°, 6 mm and ≥ 0.5 of its own span of one of these:
- an E3 published line;
- any bend's decoded-stage CAD locus;
- any stage model's formed-bend locus.

All are mapped to raw through E3's own registration. The stage loci matter for partial scans whose stage E3 left undetermined. An unexplained zs line raises the scan-level flag `UNEXPECTED_FEATURE_REVIEW`. It never changes a characteristic and can only move a scan verdict toward REVIEW.

**Numbers:**
- **False-alarm cost on correct parts.** 3 of 49 VALID scans were flagged, with 6 flags. 4 of the flags lie on real bends: they come from degenerate-pose mis-mapping on 49207000, a partial-stage scan and an UNKNOWN bend. 2 are zs false positives.
  - All 3 flagged scans already had UNKNOWN, REVIEW or FAIL characteristics. **The incremental scan-level REVIEW rate on DEV is 0/49.**
- **Sensitivity (leave one bend out).** Remove one CAD bend from the recipe and check whether its zs line gets flagged: **252/258 = 97.7 %**. The 6 misses are collinear or near-twin bends, explained by another bend's locus.
- **Wrong and mirrored parts.** E3 already refuses these on DEV (1 mirror, 1 unformed). The cross-check is a second, independent line of evidence; it is not needed for safety.
- **Before a REVIEW gate is used live,** the useful synthetic checks are:
  - an extra bend on the right part;
  - a wrong part whose datum fits.

**Assessment.** Safe, because it can only add REVIEW. It is cheap: the zs lane costs about 6 s and can run in parallel with E3. It earns its place as a detector for unexpected features or extra bends, which E3's recipe-driven decode cannot report.

## Option 2: gap-fill (located lines where E3 withholds)

**Rule.**
- **Withheld bend.** On a VALID inspection, take a bend whose BEND_LINE_POSITION is NOT_MEASURED (REGISTRATION_UNTRUSTED, AMBIGUOUS_POSE, OCCLUDED) and whose state is not NOT_YET_FORMED. It gets the closest count-capped zs line that:
  - matches its CAD locus (mapped through E3's T) within 5°, 6 mm and 0.5 overlap;
  - matches no published line.
- **Refused scan.** On a refused inspection (MIRRORED, WRONG_PART, UNTRUSTED), the count-capped zs lines are shown with no CAD association.
- **What the lines are.** These are zs geometry, not CAD and not a measurement. They are tagged `source = ZS_LANE_V1, decision = REVIEW`, and the characteristics are untouched.

**Numbers:**
- 21 withheld or UNKNOWN bends on 7 VALID scans; zs located 15 of them, and all 15 are TPs (FP stays 2).
  - The 6 misses: 3 on 49207000, where the zs lines sit beyond 6 mm of a mapped nominal under its degenerate pose; 2 UNKNOWN bends on the twin-ambiguous sc_c0ce5ecd2ba3; 1 UNKNOWN bend on 49783000.
  - The association offsets on the degenerate 49207000 pose are 0.6–4.1 mm. That is why the association radius is 6 mm, not the 4 mm matcher radius.
- Line F1 0.955 → 0.982 (withheld bends only) → 0.987 (plus refused scans). Family-macro 0.845 → 0.972.
- The refused-scan display adds one FP line on the unformed control sc_3bde9996f14d. So on a refused scan the lines must read "detected bends, unverified, part not matched".

**Assessment.** A clear gain in located bends shown to the operator, at no cost to the inspection contract. E3 PASS/FAIL is unchanged, so 0 false PASS by construction.

## Option 3: CAD priors into zs

From the `cad_prior` prototype (`../integration/cad_prior/CAD_PRIOR_PROTO.json`), re-scored here on the same cohort.

| Prior | F1 | Effect |
|---|---|---|
| count cap ≤ n_bends | 0.966 → **0.975** | Halves FP (14 → 7) for 1 TP |
| one-sided angle window [−6°, +15°] + cap | 0.973 | Neutral; it drops one control FP line |
| symmetric ±6° window + cap | 0.960 | **Harmful.** Real parts are under-bent by +7 to +13° (152–157° on 145° nominal) |
| CAD t + family cap K + 2 (Z3 re-run) + window + cap | 0.960 | Same F1 as the window arm. It is the runtime prior |

Search time (Z3 load + detect, single thread):

| | Default median / P95 / max | With CAD t + K + 2: median / P95 / max |
|---|---|---|
| Z3 | 2.2 / 8.7 / 20.3 s | 2.0 / 5.7 / 6.8 s |
| Full zs lane | 6.3 / 14.5 / 26.8 s | 5.8 / 11.7 / 12.8 s |

**Use in production.**
- The count cap applies at full stage only.
- CAD t plus K + 2 is the search-time cut.
- No symmetric window, ever. The one-sided window is optional.

## Option 4: registration seeding from zs ↔ CAD line matching

**Method** (`tools/seeds.py`). This is rotation-invariant graph matching.
- **Admitting a correspondence.** A pair of zs lines (i, j) ↔ a pair of CAD formed-bend lines (a, b) is admitted when:
  - the undirected inter-line angle agrees within 4°;
  - the inter-line distance agrees within 3 mm;
  - each zs span is no longer than 1.15 × its CAD span + 5 mm.
- **Transform.** Each admitted pair gives a proper rigid transform:
  - rotation: Kabsch on the two directions and their cross product;
  - translation: least-squares perpendicular residuals;
  - parallel pairs use the offset vector instead.
- **Scoring and refinement.** Each hypothesis is scored by one-to-one inliers (4°, 3 mm, overlap ≥ 0.3), refined on its inliers and deduplicated. It is tried against every stage model's formed bends.
- **Mirrored pass.** Gets its own seeds, from the mirrored lines.
- **Inputs.** Only lines and transforms reach E3: no names, no ids. Harness: `tools/e3_seeded.py`, which monkeypatches E3's coarse step at run time. The engine files were not edited.

**Seed quality against E3's own final pose** (48 VALID scans, stored runs):
- 34/48 get a seed. Parts with 1–2 bends, or with few zs lines, cannot fix a rigid pose from lines.
- Within 2° / 2 mm of E3's pose: **23/48 at top-1, 28/48 within the top 4.**
- Top-1 is often the 180° symmetric image. A bend-line graph is usually more symmetric than the datum's outline and holes.
- Seed time: median 0.04 s, max 7.1 s. On top of that comes the zs lane: median 5.8 s.

**E3 runs** (51 scans, e3-1.12.0, the 3 arms run concurrently in 3 geometry slots):

| Arm | VALID | Line F1 | Wall median / P95 / max / sum | Registration sum | Characteristic changes vs base | New PASS vs base |
|---|---|---|---|---|---|---|
| base | 49/51 | 0.955 | 15.2 / 90.9 / 117.9 / 1111 s | 1014 s | – | – |
| add (seeds + full search) | 49/51 | 0.952 | 14.1 / 96.8 / 117.1 / 1113 s | 1015 s | 46 in 9 scans | **7** |
| only (seeds replace coarse peaks; full search only when there is no seed) | 48/51 | 0.930 | **8.7** / 90.6 / 119.5 / 871 s | 776 s | 156 in 12 scans | **8** |

**What went wrong:**
1. **`add` is not neutral.** The seeds compete for the coarse → fine and refine slots (`coarse_candidates_fine = 14`, `pose_hypotheses = 6`) and push real coarse peaks out.
   - On sc_6eba02723d4b, E3 settled on a pose with a *lower* total (1.447 vs 1.487). Three BEND_LINE_POSITION FAILs became PASS, and one flange FAIL became PASS.
   - On sc_6e5d5a2f60f5 (total 1.338 vs 1.519), an angle FAIL at 152.9° became PASS at 90.1°, and a bend state changed.
   - With no characteristic truth on DEV, these PASSes cannot be shown correct. Under the zero-false-PASS rule they count against the option.
2. **`only` is unsafe.**
   - On sc_08686dc1d5d4 all 4 seeds were wrong: a partial scan matched to the S12 lines. E3 registered on datum score 0.313 and explained 0.813, where base had 0.998 and 1.000.
   - That pose **passed validity**, because explained ≥ the wrong-part threshold. It decoded 6 bend states differently and published a BEND_LINE_POSITION PASS that base had as NOT_APPLICABLE.
   - On sc_186463746daa the flipped seed won, and the scan went VALID → WRONG_PART.
3. **Twin detection needs the search.** E3's ambiguity check (`engine._ambiguity`) only sees the poses in the searched hypothesis list. Replacing the search with seeds makes AMBIGUOUS_POSE and the twin agreement depend on whether the seeds happen to contain the twin. Seeds can therefore never replace the search.
4. **No coverage headroom.** On DEV E3 is VALID on every scan it should be. The two refusals are correct.
   - The "hard registration" class is now registered: 49207000 is VALID, explained 0.856, since D2.
   - What remains there is **in-plane degeneracy**: positions are withheld because the datum's edges and holes leave a direction free. A seed cannot change that, because `refine_datum` decides degeneracy from the datum alone.
   - Fixing the free direction from bend lines would make BEND_LINE_POSITION self-referential: the pose would be fitted to the very lines whose position is being measured. So bend-line-borne poses can only ever support display (option 2), never a position PASS.
   - On 49207000 the best seed lies 0.2° and **7.1 mm** from E3's pose, along the free direction. This is consistent with that direction being unobservable from the datum.
5. **Runtime.** The P95 is set by the mirror check, which re-runs the whole search on the flipped scan (about 45 % of wall time on the slow cases), and by 2-bend parts that get no seed. Seeding does not touch either.

**Verdict.** Option 4 does not meet "clear gains, no regression":
- line F1 goes down;
- coverage is equal or lower;
- P95 is unchanged;
- new PASSes appear, one of them on a wrong pose.

**Engine left untouched.** The fingerprint is equal before and after this study: engine `5fbed707…`, compile `9831f9ae…`, eval `8aa6860c…`, contracts `06401977…`. So A1 (GOLD_SET_V1 0.990/0.974, per-bend 192/203) and G1 (PASS, 0/553 false PASS) are unchanged by construction, and no re-run was needed. No `engine_before.tgz` was taken, because nothing was changed.

## Recommended production design

```
scan ──► E3 (authority, unchanged) ─────────────► result.json (PASS/FAIL/REVIEW/UNKNOWN per characteristic)
   └──► zs_lane_v1 (+ CAD priors: count cap, CAD t, K+2)  ─► zs lines (raw frame)        │
                                                         └──────────► C06 overlay ◄──────┘
                                                                      • opt 2: zs line per withheld bend (REVIEW, display)
                                                                      • opt 2: zs lines on refused scans ("unverified")
                                                                      • opt 1: unexplained zs line → scan flag REVIEW
```

1. **E3 stays the only source of characteristics and decisions.** Nothing from zs enters E3's inputs.
2. **The zs lane runs in parallel with E3,** with the CAD priors: count cap at full stage, CAD t, K + 2. Its P95 is 11.7 s, well under E3's.
3. **A post-E3 overlay in C06 does the rest.** It uses only result.json (raw_to_frame, stage decision, bend rows), CompiledPartV1 and the zs lines. It:
   - attaches a gap-fill line to each withheld bend;
   - attaches the unverified lines to refused scans;
   - raises the `UNEXPECTED_FEATURE_REVIEW` flag.

   The overlay can only *lower* a scan verdict, to REVIEW. Its lines carry `source = ZS_LANE_V1` and are never characteristics.
4. **Contract.** This needs a job/presentation-level field, not an `inspection_result` change. The engine version and the inspection contract stay as they are. The overlay field and the scan flag need an owner decision, as with the earlier ENGINE_IO amendments.

**Why not inside E3?**
- E3 would have to import the research lane and its GBM pickles, which widens E3's code fingerprint and input surface for no metrological gain.
- The overlay needs nothing that result.json does not already publish.
- This deviates from the literal "implement in the engine if option 2 wins". The owner can still choose an E3 `--aux-lines` input instead; the overlay algorithm is `tools/options_offline.py` (`opt2_*`, `opt1_*`).

## Next steps

1. **Owner decision:** the C06 overlay (options 1 and 2) and its job-level field and REVIEW flag. Then productionize the zs lane in the runner: package Z2, Z3 and the GBM with a pinned fingerprint, guarded like E3.
2. **Pre-register the overlay** (association 5° / 6 mm / 0.5; count cap) and test it once on the 5 OD-3 DEV-CHECK cases, which have never been run, and on new deliveries with STEP. DEV here is consumed.
3. **Synthetic tests for option 1:**
   - an extra bend on the right part (it must flag);
   - a wrong part whose datum fits (it must flag, and E3 must still refuse or REVIEW).
4. **E3 runtime.** If search time matters, the target is the mirror check (about 45 % of the slow cases), for example by reusing the proper pass's plane segmentation. That is not seeding.
5. **Do not pursue registration seeding** unless E3 gets a way to enumerate certified twins of a chosen pose independently of the search; E1's symmetry candidates could provide one. Even then, seeds may only add hypotheses beyond the existing caps, never take their slots.

## Files

- `tools/build_cohort.py`: guarded per-scan facts (zs out-of-fold candidates, CAD stage lines, E3 rows). It reads `CADPATH_E3_DIR` when set.
- `tools/options_offline.py`: options 1–3 and the overlay algorithm.
- `tools/seeds.py`, `tools/seed_eval.py`: option 4 seeds and seed accuracy.
- `tools/e3_seeded.py`, `tools/seeded_driver.py`, `tools/compare_arms.py`: the option 4 E3 arms (base / add / only).
- `results/*.json`: all the numbers above. E3 run directories are in the session scratchpad (`cadpath_runs/{base,add,only}`).
