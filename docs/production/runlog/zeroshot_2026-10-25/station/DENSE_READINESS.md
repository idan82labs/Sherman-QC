# Station dense-scan readiness (2026-10-25)

> **Stand-in host.** Every number here comes from this Mac (Apple Silicon, 18 cores, 128 GB RAM, no NVIDIA GPU), standing in for the owner's station (NVIDIA RTX 5000 Pro, 256 GB RAM, one job at a time). Other agents shared the host: load average 3–24 during the runs. Each heavy run held one geometry slot (`sherman_slot.sh`) and ran alone among this task's jobs. Expect different absolute times on the station. Neither engine uses the GPU.
>
> **Measurement only.** No engine or lane code changed. E3 is frozen `e3-1.12.0`; zs is `zs_lane_v3` revision 2 (pin `ab31d061…`). TEST-B, DEV-CHECK, FRESH and N1 were not scored. N1 41018000 was run for timing only: its lines were discarded. Machine-readable results: `results.json`. Tools: `tools/`.

## A. Memory-cap amendment status (ENGINE_IO-A5): proposed, blocked on a re-pin

The owner decided to raise the per-step RSS cap from 4 GiB to 32 GiB. That cap lives in `contracts/production/line_job.ENGINE_RSS_LIMIT_BYTES`, with a C11 mirror in `apps/production_line/store/vocabulary.py`. Those were edited first, together with the supervisor ceiling (16 GiB), `SHERMAN_LINE_RSS_LIMIT_MB` (default 4096, max 16384), the runner messages and the tests. Then the named tests were run: **354 passed and 2 failed**: `test_prod_zeroshot_overlay_lane.py::test_pin_matches_the_frozen_research_config` and `::test_pin_verify_fails_closed_on_any_change`. `pin.verify` reported `CODE:contracts/production/line_job.py`.

- The whole `contracts/production` package is part of every zs lane pin, `line_job.py` included. That covers zs_lane_v1 (live, `af87df2f…`) and zs_lane_v3 rev 2 (`ab31d061…`).
- Any byte change to `line_job.py` makes the zs step fail closed with PIN_MISMATCH. **The overlay would go dark on every job** until both lanes are re-pinned.
- Re-pinning v3 changes the code fingerprint that the owner-confirmed `v3/PREREG_FRESH_TEMPLATE.json` binds. Re-pinning v1 changes the fingerprint recorded for consumed runs.

This task does not authorise either re-pin, so **every Task A code and test edit was reverted** (pins verify again; 356 + 45 named tests pass). A5 is recorded in `contracts/production/ENGINE_IO.md` §7.1 as **PROPOSED, not in force**, with the full rationale. A3 and A4 are taken by the roll design, so A5 is the next free id. Owner options:

1. **Re-pin both lanes for a contracts-only change.** Models, configs and thresholds stay identical. Record it in the ledger, update the PREREG primary-arm fingerprint, then apply A5 as written.
2. **Change the pin scope.** Exclude `line_job.py` (constants that the lane never uses) from the zs fingerprint. This is itself a re-pin, but it would decouple future limit changes.

The data below show the cap matters. At ~40 M faces, E3 peaks at 6.1–7.1 GiB and zs at 5.5–6.2 GiB. **With today's 4 GiB cap, both steps would be killed on every scan above roughly 20 M faces.**

## B. Method

**Dense meshes** (`tools/densify.py`):
- Conformal N-subdivision of the real scan: shared edge points, no T-junctions, original winding, N² triangles per face. Original vertices are bit-exact.
- New points are jittered inside the surface by ±0.2 lattice units (edge points along the edge, interior points in barycentric space). This breaks the regular lattice and can never invert a triangle.
- New points are displaced along their interpolated normal by N(0, 0.004564 mm), the scanner card's `noise_sigma_mm`.
- Output layout is Geomagic-style: binary PLY with float xyz and normals, uchar/int faces.
- N is the integer that gives the face count closest to each target. Actual sizes are 3.8–46 M faces.
- **Limit:** this upsamples the existing piecewise-linear surface. It adds no detail that a denser sensor would resolve, so it measures compute scaling and density sensitivity, not accuracy.

**Control:** the same N with no jitter and no noise (A N=3, B N=2). It gives the same E3 decision changes as the jittered mesh (0 differences between the two), so **the E3 changes come from point density itself, not from the jitter or noise**.

**Scans:** 4 programme-DEV scans with compiled recipes (custody DEV_CONSUMED):
- A sc_3c93a0bf6f97 (11979003, 0.49 M faces)
- B sc_6eba02723d4b (23553000, 0.94 M)
- C sc_e080ae094f35 (49204020, 1.41 M)
- D sc_203b95d7a753 (49201000, 3.96 M)

Plus N1 41018000 (3.37 M), zs only, for timing. It has no compiled part, so it ran with a labelled synthetic CAD-facts file (`tools/n1_facts_timing_only.json`).

**E3:** the frozen CLI with the eval-runner environment and a 32 GiB RSS limit. The time limit was **diagnostic**: 1800 s, with `--time-budget-s` 1790, so that dense runs finish and can be compared. Production is 180 / 170 s. Every run with a wall time below 170 s would give the same output under production limits, because its budget never expired.

**zs:** the production `lane_cli` with `SHERMAN_ZS_LANE_PIN=zs_lane_v3`. Pins are verified fail-closed and unpinned modules are refused, the production equivalent of `--strict-pins`. The research lane CLI was not usable here: its gold guard refuses a derived mesh as NOT_REGISTERED, and derived meshes must not be registered in the gold ledger.

## C. Results (1 thread, one job at a time)

Column key:
- **E3 dec Δ:** characteristics whose decision differs from the original-density run.
- **zs ↔ orig:** zs lines matched to the original-density lines within 6 mm.
- **zs DEV TP:** frozen-matcher score against DEV gold of record.

| Set | Mesh | Faces (M) | E3 s | E3 GiB | E3 top stages (s) | E3 summary | E3 dec Δ | zs s | zs GiB | zs budget step s | zs cell mm | zs lines | zs ↔ orig | zs DEV TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | original | 0.49 | 6 | 0.3 | registration 4, articulated 1 | REVIEW_REQUIRED | — | 6 | 1.0 | — | — | 5 | — | 5/7 FP 0 |
| A | N=3 | 4.41 | 15 | 1.0 | articulated 6, registration 5 | REVIEW_REQUIRED | 3 | 14 | 2.4 | 2.0 | 0.160 | 7 | 4/5 | 7/7 FP 0 |
| A | N=5 | 12.25 | 43 | 2.0 | mirror_check 21, articulated 8 | REVIEW_REQUIRED | 3 | 16 | 2.9 | 5.7 | 0.214 | 5 | 2/5 | 5/7 FP 0 |
| A | N=6 | 17.63 | 37 | 2.5 | articulated 19, metrology 7 | REVIEW_REQUIRED | 3 | 20 | 3.3 | 8.0 | 0.212 | 4 | 1/5 | 4/7 FP 0 |
| A | N=9 | 39.68 | 144 | 6.1 | mirror_check 76, articulated 35 | REVIEW_REQUIRED | 4 | 31 | 5.7 | 19.9 | 0.234 | 5 | 1/5 | **2/7 FP 3** |
| B | original | 0.94 | 27 | 0.6 | mirror_check 12, registration 11 | OUT_OF_TOLERANCE_FOUND | — | 7 | 1.5 | — | — | 16 | — | 16/16 FP 0 |
| B | N=2 | 3.77 | 37 | 0.9 | mirror_check 16, registration 11 | OUT_OF_TOLERANCE_FOUND | **7** | 11 | 2.4 | 1.8 | 0.266 | 16 | 15/16 | 16/16 FP 0 |
| B | N=3 | 8.49 | 49 | 1.6 | mirror_check 22, registration 12 | OUT_OF_TOLERANCE_FOUND | **7** | 14 | 2.5 | 3.9 | 0.289 | 16 | 16/16 | 16/16 FP 0 |
| B | N=5 | 23.59 | 94 | 4.0 | mirror_check 35, articulated 20 | OUT_OF_TOLERANCE_FOUND | **7** | 20 | 3.6 | 10.4 | 0.330 | 10 | 8/16 | **10/16** FP 0 |
| B | N=7 | 46.23 | 159 | 7.1 | mirror_check 56, articulated 41 | OUT_OF_TOLERANCE_FOUND | **7** | 31 | 6.2 | 20.0 | 0.318 | 6 | 4/16 | **6/16** FP 0 |
| C | original | 1.41 | 15 | 0.6 | articulated 7, registration 6 | ALL_MEASURED_IN_TOLERANCE | — | 7 | 1.7 | — | — | 12 | — | 12/12 FP 0 |
| C | N=2 | 5.65 | 32 | 1.0 | articulated 18, registration 8 | ALL_MEASURED_IN_TOLERANCE | 0 | 13 | 2.2 | 3.6 | 0.302 | 12 | 11/12 | 12/12 FP 0 |
| C | N=3 | 12.71 | 57 | 2.0 | articulated 37, registration 9 | ALL_MEASURED_IN_TOLERANCE | 0 | 16 | 3.1 | 7.0 | 0.338 | 11 | 11/12 | 11/12 FP 0 |
| C | N=4 | 22.60 | 68 | 3.6 | articulated 39, metrology 10 | ALL_MEASURED_IN_TOLERANCE | 0 | 21 | 3.5 | 11.1 | 0.341 | 8 | 8/12 | 8/12 FP 0 |
| C | N=5 | 35.32 | 140 | 5.0 | articulated 101, metrology 16 | ALL_MEASURED_IN_TOLERANCE | 0 | 30 | 5.5 | 19.0 | 0.361 | 10 | 9/12 | 10/12 FP 0 |
| D | original | 3.96 | 16 | 1.0 | registration 7, articulated 6 | REVIEW_REQUIRED | — | 9 | 2.5 | 1.8 | 0.258 | 2 | — | 2/2 FP 0 |
| D | N=2 | 15.85 | 36 | 2.2 | articulated 18, registration 8 | REVIEW_REQUIRED | 0 | 14 | 3.2 | 6.1 | 0.294 | 2 | 2/2 | 2/2 FP 0 |
| D | N=3 | 35.66 | 68 | 5.0 | articulated 39, metrology 10 | REVIEW_REQUIRED | 0 | 23 | 5.1 | 14.6 | 0.333 | 2 | 2/2 | 2/2 FP 0 |
| N1 41018000 | original | 3.37 | — | — | — | — | — | 26 | 2.6 | 1.7 | 0.517 | (discarded) | — | not scored |
| N1 41018000 | N=2 | 13.50 | — | — | — | — | — | 25 | 3.2 | 5.9 | 0.578 | (discarded) | — | not scored |
| N1 41018000 | N=3 | 30.37 | — | — | — | — | — | 30 | 4.2 | 12.8 | 0.563 | (discarded) | — | not scored |

### E3: time and memory

**Time** grows about linearly with face count, at roughly 3–4 s per M faces at 1 thread. It reaches 140–159 s at 35–46 M faces. That is above the 45 s P95 target and **within 10–30 s of the 170 s production budget**. A 50 M-face scan would likely hit the budget on this host, giving UNKNOWN(TIMEOUT) or validity TIMEOUT. Where the time goes:
- Registration runs on 0.8 mm voxels, so it stays flat at 5–15 s.
- The stages that scale with raw points do not: `articulated`, `mirror_check` (when it runs), and `metrology`.
- `mirror_check` ran on some dense meshes but not on the originals (A, for example). This alone adds 20–76 s.

**Memory** grows about linearly as well: 0.15 GiB per M faces, **7.1 GiB at 46 M**.

### E3: verdicts change with density (flagged)

The output does **not** stay the same on 2 of the 4 scans, and the no-noise control shows that density causes the change.

**A**: 3–4 decision changes. One is a **REVIEW→PASS upgrade** (B7 bend-line position). There are also PASS→REVIEW changes on B6/B7 angles. Angle shifts are up to 0.59°, about 2.7× U.

**B**: 7 changes, identical from N=2 (3.8 M faces) up to N=7:
- **4 FAIL→PASS**: B5 flange B, B6 flange A, B6 flange B, B7 flange A.
- 1 FAIL→REVIEW and 1 REVIEW→FAIL.
- 1 PASS→REVIEW (B5 angle).
- Flange-length shifts are 1.3–1.4 mm, about 10× U.
- The summary stays OUT_OF_TOLERANCE_FOUND.

**C and D**: 0 changes. Value deltas are 0.003 mm or less.

There is no truth to say which density is right. Either way, **an E3 verdict that depends on how densely the part was scanned is not acceptable**, and a FAIL→PASS flip is a false-PASS risk. E3 was qualified (G1 synthetic, physical gold) at about the scanner card's 0.40 mm spacing. The dense ZEISS scans fall outside that envelope.

### zs (v3 rev 2)

**Time** stays at 31 s or less at 46 M faces, far inside 90 s. The budget step (decimation to 3 M faces or fewer) is the part that grows: 20 s at 46 M.

**Memory reaches 5.1–6.2 GiB** at 30–46 M faces, which is **over today's 4 GiB zs cap**. In production today, the zs step would be killed and the job would get no overlay.

**Lines do not stay the same at high density.** The budget's cell size grows with input density:
- 0.16–0.27 mm at 4–5 M faces;
- 0.31–0.36 mm at 20–46 M faces.

This happens because a denser mesh occupies more grid cells, so the ladder has to coarsen further to fit 3 M faces. DEV recall drops as a result:
- B: 16/16 → 10/16 → 6/16;
- C: 12/12 → 8–10/12;
- A at 40 M: 2/7 with **3 FP**.

Up to about 10 M faces the lines match closely: B N=2/3 16/16, C N=2 12/12, D unchanged. So revision 2's budget makes zs **memory-safe and fast but not density-invariant**. The "rev2 thins above 3 M, so the lines match" expectation holds only near 3–10 M faces.

## D. Threads and determinism (set B, original and N=5 / 23.6 M faces)

| Run | 1 thread | 4 | 8 | 16 | Speed-up 16 vs 1 | Identical outputs |
|---|---|---|---|---|---|---|
| E3 original | 27.6 s | 27.1 | 27.0 | 26.8 | 1.03× | yes: 5 runs (incl. the main t1), stripped result.json and every engine file byte-identical |
| E3 23.6 M | 95.9 s | 94.8 | 93.0 | 95.4 | 1.01× | yes (5 runs) |
| zs original | 7.6 s | 7.5 | 7.4 | 7.3 | 1.05× | yes: lines sha identical (5 runs) |
| zs 23.6 M | 20.8 s | 20.0 | 19.9 | 21.2 | 0.98× | yes (5 runs) |

All BLAS and OpenMP variables (`OMP`, `OPENBLAS`, `MKL`, `NUMEXPR`, `VECLIB`) were set to N. **There is no speed-up**:
- E3 pins `cKDTree(workers=1)` by design (cloud.py: "results do not depend on the machine's core count");
- neither engine spends meaningful time in multithreaded BLAS.

**Determinism holds at every thread count.**

## E. Recommendations (proposals, none applied)

1. **Apply A5 (32 GiB) via one of the re-pin options in §A.** Dense scans need it: E3 reaches 7 GiB and zs 6 GiB at 46 M faces, both over 4 GiB. Projected at 50 M: about 8 GiB and 7 GiB, so 32 GiB leaves about 4× headroom, and both steps in parallel stay under 16 GiB on a 256 GB station. A5 cannot create a PASS by itself. **But see item 2:** today, dense input itself shifts E3 verdicts, including FAIL→PASS. A job that the cap used to kill would now complete with a density-dependent verdict.

2. **E3 needs a density budget.** Proposal: add a deterministic input-thinning step to E3's ingest, before `clouds`, that brings any scan back to the qualified envelope. For example, normal-aware voxel or quadric clustering to the scanner card's 0.40 mm point spacing, or to at most about 2 M points. This is the same idea as zs rev 2, but keyed to spacing rather than face count.
   - It is an E3 change: version bump (e3-1.13.0), re-run of G1, the A1 physical gates and the frozen recompile, and a ledger AMENDMENT.
   - It disables the overlay until both zs pins are re-pinned. It can be combined with the A5 re-pin into one re-pin event.
   - Expected effect: E3 time back to the 15–30 s range and verdicts equal to qualified-density behaviour.
   - **Until then, do not trust E3 verdicts on scans much denser than about 4 M faces** (B already shifts at 3.8 M).

3. **zs rev 2 budget: key it to spacing, not face count** (a revision 3; a re-freeze before the fresh-scan test, so the PREREG primary arm changes). For example, decimate to a fixed cell of about 0.25 mm, the DEV in-budget range, whatever the input density. Or raise `max_faces` now that memory allows: a 6 M budget at about 2× the time and memory would still be under 60 s and 10 GiB. Either way, re-check DEV OOF. **The fresh-scan promotion test should not run on ScanBox-density scans with rev 2 as-is**: its accuracy measured here falls with density.

4. **Time limits:**
   - **With** an E3 density budget, keep 180 s / 170 s (E3 would be about 15–30 s).
   - **Without** one, raise the E3 hard stop to 300 s with `--time-budget-s` 290 for 50 M scans. That would be a separate amendment; the 45 s P95 target is not reachable at 20 M faces or more.
   - zs 90 s is fine (31 s max) and the merge 60 s is unaffected.
   - Re-measure on the real station CPU.

5. **Default thread count: 1** (`OMP_NUM_THREADS=1` and friends, as today). More threads buy nothing for either engine, and 1 keeps the determinism argument simple. Use the spare cores as today: E3 and zs run in parallel. Real speed-ups would need code changes, for example running `mirror_check` and `articulated` hypotheses in a deterministic process pool. That would be an E3 version bump, not a setting.

## F. Reproduce

Meshes and runs are in `/private/tmp/claude-501/dense_ready/{meshes,runs}` (scratch, not kept in the repo). Commands:
- `.venv/bin/python docs/production/runlog/zeroshot_2026-10-25/station/tools/densify.py IN.ply OUT.ply N [--jitter 0 --sigma 0]`
- `tools/run_step.py e3 MESH COMPILED_DIR RECIPE OUT THREADS 1800`
- `tools/run_step.py zs MESH COMPILED_JSON OUT THREADS 1800`

Each runs through `/private/tmp/claude-501/sherman_slot.sh`. Then run `tools/analyze.py` and `tools/score_zs.py` (DEV sets only).
