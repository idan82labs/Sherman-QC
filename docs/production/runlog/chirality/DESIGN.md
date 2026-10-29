# Left/right (chirality) identity: design (2026-10-28)

Owner approval (2026-10-28, verbatim): "yes, go ahead with the design".

> **Status.** Part B (setup-screen warning and station badge) is display only and is in the working tree. Parts A and C are **research only**. Live E3 `e3-1.14.0`, the zs lanes, pins, `contracts/production` and the ledger were not changed. Releasing C needs an owner-approved gated re-pin (§5).

Background: `../zeroshot_2026-10-25/station/E3_MIRROR_IDENTITY.md`.
- Live E3 accepts mirrored scans of chiral parts as VALID with PASSes (10839000).
- The scene-level prototype failed held-out in two ways:
  - It missed 49201000_C. E1 certifies that stage mirror-symmetric at 0.085 mm RMS, but two small hole pairs make it chiral.
  - It falsely refused 37126000_D, an achiral part, after its proper registration failed.
- The cause is the same in both: the evidence was scene-level (totals, explained shares), while the difference is local.

## 1. Idea

Decide **at compile time, from the CAD**, whether each stage is superimposable on its mirror image, and if not, *where* the two differ. At inspection time E3 then checks exactly those places in the registered scan. A missing observation is never read as "absent".

## 2. A. Compile-time chirality map, `chirality_map_v0`

Code: `research/chirality/chirality_map.py`. It is a new artifact (`chirality_map_v0.json` + `.npz` probes) written beside the compiled part or to any directory. The compile output and the pinned schemas are untouched.

- **Input.** E1's stage sample clouds (the B-rep sampled on a 0.5 mm grid of each face) and the compiled panels (planes, outer boundaries, hole loops). Both come from E1's OCP read of the STEP, so nothing is re-tessellated and no scan is read.
- **Mirror fits.** The stage cloud is mirrored and fitted back onto itself by point-to-plane ICP from the 24 improper cube symmetries of its principal axes. It reuses E1's own symmetry-test `_icp`/`_score` read-only.
  - Each distinct optimum H is a *mirror hypothesis*: where a mirrored part would sit after E3's proper registration.
  - Hypotheses are kept when they overlap the CAD on at least half its samples, up to 6, best first.
- **Deviation and features.**
  - At full density, two deviations are computed:
    - d_A: CAD material that the mirror image lacks;
    - d_B: mirror-image material that the CAD lacks.
  - Samples more than 1.0 mm off are clustered with 3 mm linkage. A cluster of at least 20 samples is a **distinguishing feature**.
  - For each feature the map records:
    - **Location:** the centre in the stage frame and in the final frame.
    - **Panel:** the owning panel.
    - **Type and size:** HOLE / CUTOUT / OPENING / NOTCH_OR_TAB / FORM, from the B-rep hole loops and outline of that panel.
    - **Max deviation.**
    - **Expected scan evidence:**
      - *Proper-material* core probes: here the proper image has material and the mirror image has none.
      - *Mirror-material* core probes: the reverse.
      - Core probes are skin samples whose deviation is ≥ 2.0 mm.
      - A *ring* of shared skin samples 1–5 mm around the feature, used as the coverage reference.
  - Probes are stored in the final-pose frame of their panel, so E3 carries them with the articulated per-panel pose and bend-angle errors do not move them.
- **Classes** (geometry only, no part numbers):
  - **ACHIRAL**: some mirror fit leaves no distinguishing feature.
  - **CHIRAL_GROSS**: the best fit still leaves more than 5 % of the samples deviating.
  - **CHIRAL_BY_FEATURES**: otherwise.
  - A feature is *verifiable* when it has at least 8 core and 8 ring probes. A chiral stage without any verifiable feature can never be confirmed, so it always withholds.
- **Cost.** 0.2–12 s per stage single-threaded, RSS ≤ 0.6 GiB. The largest real part, 13 stages, takes 3.6 min.

| Example | E1 symmetry test | Map |
|---|---|---|
| 10839000 S12 | not certified (0.331 mm) | CHIRAL_BY_FEATURES. Under the 0.31 mm fit, the distinguishing feature is one 30.5 × 17.3 mm cutout: the proper part is open in wall P1, the mirror image in P2. 300 probes per role. ![](chirality_map_10839000.png) |
| 49201000_C S2 | **certified symmetric (0.085 mm)** | CHIRAL_BY_FEATURES. Two Ø ~6.8 mm holes on P0 sit at z = +90 / −80 on the part and at +80 / −90 on its mirror image. 49–51 core probes each. ![](chirality_map_49201000_C.png) |
| 37126000_D | — | ACHIRAL at every stage |

## 3. B. Setup-screen warning and station badge (shipped, display only)

Apps side only. No pinned contract changes, and nothing in `compiled_part.json`.
- **When the map is computed.** After release, the map is computed for the released compile, final stage only, in a supervised subprocess with a 300 s / 4 GiB limit. It runs on the first wizard or Vendor Setup read.
- **Storage.** `<data_root>/recipes/chirality/<compiled_part_sha256>/`.
- **API.**
  - `GET /api/production/parts/{id}/chirality`
  - `POST …/chirality/retry`
  - `GET /api/production/recipes/{id}/chirality` (read-only, for the station).
- **Wizard and Vendor Setup.** An amber "Left/right-sensitive part / חלק רגיש לימין/שמאל" warning in English and Hebrew (RTL), with the features listed. Vendor Setup also shows "L/R n" markers on the folded CAD view.
- **Station.** A "Check left/right variant by eye / בדקו בעין את גרסת ימין/שמאל" badge, shown for CHIRAL_BY_FEATURES parts.
- **Failure.** A failure never blocks onboarding or release; the UI shows "not computed".
- Screenshots are in `screens/`. Tests: `tests/test_chirality_display.py` and `frontend/react/scripts/check-chirality-ui.mjs`.
- **Limitations.**
  - The map is not started at release itself.
  - The station never starts a computation.
  - Only the final stage is computed.
  - The station's `B4` "Not measured" label overlaps in the fixture screenshot. That is pre-existing, not from the badge.

## 4. C. E3 enforcement, candidate `e3-1.15.0-rc` (scratch only)

Patch: `e3_1_15_0_rc_chirality.patch`, against live `engine.py` `bb593889…` and `config.py` `71c47872…`. It adds `inspect/chirality.py`, which is byte-identical to `research/chirality/check.py`.

- **Map.** `--chirality-map PATH`, default `<compiled_dir>/chirality_map_v0.json`. It is verified against `compiled_part_sha256` and the probe sha.
- **Check.** It runs after registration, articulation, decode and metrology, on the decoded stage. Every probe is carried onto the scan with its owning panel's fitted pose. A probe is *hit* when a scan point lies within 0.8 mm.
  - **Per feature and skin side:**
    - A role is PRESENT when ≥ 50 % of its probes are hit.
    - It is ABSENT when ≤ 10 % are hit **and** the ring on the same side is ≥ 50 % hit. Not scanned is not absent.
    - Proper votes: proper-material PRESENT, or mirror-material ABSENT. Mirror votes are the opposite.
    - The feature is PROPER, MIRROR, CONFLICT or UNOBSERVED.
  - **Per hypothesis:** PROPER when ≥ 1 feature is PROPER and none is MIRROR or CONFLICT. MIRROR when ≥ 1 is MIRROR and none is PROPER or CONFLICT.
- **Decision.**
  - **ACHIRAL stage:** nothing changes.
  - **Every hypothesis PROPER** (`PROPER_CONFIRMED`): the normal verdicts stand.
  - **Any hypothesis MIRROR:** `MIRRORED_PART_SUSPECTED (CHIRAL_FEATURES_MATCH_MIRROR:H<i>:<n>)`.
  - **Anything else** (unobserved, conflicting, unmapped stage, missing or mismatched map): **every PASS is withheld** as REVIEW. U is raised into the guard band, the characteristic gets the note `LEFT_RIGHT_UNCONFIRMED`, and the result gets a warning naming the reason. FAIL and undecided characteristics are untouched.
- **Achiral guard.**
  - When the live mirror rule would refuse a scan whose decoded stage the map certifies ACHIRAL, the result is `UNTRUSTED_REGISTRATION (MIRROR_FIT_BETTER_ACHIRAL_STAGE:…)` instead of a mirror refusal. That stage is its own mirror image, so a better mirrored fit means the proper search failed.
  - This is the 37126000_D failure mode.
- **Never creates a PASS.** The live mirror rule stays as an extra path. The new logic only refuses or raises U, and the K=2 resample net still only withholds.
- **Config.** `chirality_check=1` and `chirality_achiral_guard=1` are added; both are hashed into `config_sha256`. `ENGINE_VERSION` becomes `e3-1.15.0-rc`.
- **Runtime.** The map load and probe test take well under 1 s per pass.

## 5. Proposed contract amendment and re-pin plan (not applied)

`contracts/production` is hashed into the zs pins, so this is a proposal only.

**Amendment ENGINE_IO-A3 (chirality).**
1. **Compile output.** `CompiledPartV1` gains `chirality: ChiralityRefV1 | None`, holding:
   - `{rel_path: "chirality_map_v0.json", sha256, probes_rel_path, probes_sha256, method, thresholds_sha256, part_class}`;
   - per `StageModelV1`: `chirality_class: Literal["ACHIRAL","CHIRAL_GROSS","CHIRAL_BY_FEATURES"]`.
   - Both are included in `compiled_part_sha256`, so a map can never be paired with another compile.
   - E1 writes the map in the compile (new phase "chirality"; +2–220 s, under the 120 s budget only after the coarse screen is ported to the compile's decimated clouds; see open item 3).
2. **Symmetry contract change.**
   - Today `SymmetryTestV1` says a MIRROR candidate within 0.3 mm means E3 *must not* claim MIRRORED_PART_SUSPECTED.
   - Amend: that prohibition applies only to stages whose `chirality_class == ACHIRAL`.
   - A certified-symmetric but CHIRAL_BY_FEATURES stage (49201000_C) may be refused on feature evidence.
3. **Result vocabulary.**
   - The validity reason `CHIRAL_FEATURES_MATCH_MIRROR:*`.
   - The registration reason `MIRROR_FIT_BETTER_ACHIRAL_STAGE:*`.
   - The characteristic note and warning `LEFT_RIGHT_UNCONFIRMED`.
   - An optional result block `left_right: {decision, reason, map_sha256}`.
   - `check_result_against_compiled` gains one rule: a result with `left_right.decision ∉ {ACHIRAL, PROPER_CONFIRMED}` has no PASS.
4. **Recipe.** No change. The setup warning stays apps-side.

**Re-pin plan.** One event, like the 2026-10-28 re-pin, after owner sign-off:
1. **Contracts and compiler.** Contract amendment A3 and its tests. E1 `e1-1.2.0` writes the map, and every frozen recipe is recompiled. The frozen-recompile gate becomes "byte-identical except the new `chirality` block".
2. **Engine.** E3 `e3-1.15.0` from this patch, with `--chirality-map` dropped in favour of the compiled reference.
3. **Pins.** Re-pin zs v1 / v3 rev3 (`engine.py` and `config.py` change), update PREREG fingerprints, and add a ledger entry. The Rust adapter's `freeze_config` must be re-run.
4. **Gates:**
   - G1: 0 false PASS, WRONG_OR_MIRRORED 32/32.
   - A1 natives: unchanged except the LEFT_RIGHT withholds listed in RESULTS.
   - The dense set, runtime (G2(e) under the existing waiver), the real-part E2E, and the synthetic chirality suite (RESULTS.md) re-run on the release build.
5. **Rollback.** Restore the archived `engine.py`, `config.py` and contracts, and re-pin.

## 6. Freeze

`FREEZE.json` was written before any synthetic evaluation run. It records:
- the map thresholds (sha `3fa1a3ed…`) and the check thresholds (sha `22f6f6e7…`);
- the occlusion rule;
- the pass conditions;
- the sha256 of every code file, the corpus and the plan.

The evaluation was rerun in full on 2026-10-29 from `work/` after the scratch wipe (same frozen code; see RESULTS.md provenance). Smoke runs on seeds 9001–9003 (10839000 S12, 49201000_C S2, G1 UC-001 S2) were pipeline debugging only, before the freeze. Nothing was retuned after the freeze.
