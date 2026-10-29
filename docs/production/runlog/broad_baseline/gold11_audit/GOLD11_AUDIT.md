# Gold11 label audit, 2026-10-17: convention offset or real error?

**Question.** The broad baseline (`BROAD_BASELINE_2026-10-17.md` §5.4) says Gold11's reference lines
are off the bend on all 6 Gold11 scans it triaged (27 of 49 lines) and that 11 real bends are
missing. One agent per scan produced that claim, with no second verifier. This audit tests it
adversarially. The main hypothesis is that many of the "errors" are a **line-locus convention**: the
human line drawn at the virtual sharp, the mold line or the inner skin, while the AI and engine lines
sit on the outer-skin apex.

**Short answer.**

- **The triage holds, and it understated the problem.** Three blind auditors judged every one of the
  71 Gold11 lines. Only **13 are OK**. **20 are parallel, systematic offsets** (CONVENTION_OFFSET),
  **23 are real placement or extent errors** and **15 are not bends at all**. **31 formed bends have no
  line.**
- **The convention hypothesis, as stated, is refuted.** No line sits toward the outer virtual sharp.
  The offsets lie on the concave side and are spread over five different loci, from 1.0 to 5.9 mm.
  The 12 offsets of 3.2 mm or less are harmless under the 4 mm gate. The 8 offsets beyond 4 mm lie
  past the arc centre, on a flange face or in free air, which is not a bend locus.
- **A real convention exists, but only on 2 scans, and it is the inner surface.** There, STEP bend
  axes were deliberately moved onto the inner surface. That places lines 2.5–5.7 mm from the apex,
  and 3 of those 8 lines fall outside the gate.
- **Provenance explains the offsets.** 45 of the 71 "human" lines (8 of the 11 scans) are
  bit-identical to the drafts of an untracked script that tags them `human_two_point`:
  - **4 scans** are built from F1 detector regions;
  - **2 scans** are built from plane-lattice lines;
  - **2 scans** are STEP axes shifted onto the inner surface.

  The offsets come from how the lines were generated, not from a person's clicking habit.
- **The current gate is unreachable.** Against today's Gold11, a perfect outer-apex engine scores
  **F1 0.388**. That is lower than the legacy best lane (0.458), and G2(a) cannot be measured on
  these labels.
- **The proposed relabel.** `gold11_relabel_v1` uses 10 AI relabels, each passed by 2 verifiers,
  plus 1 human set. Against it the best lane scores **0.510** and the runtime **0.141**. Both still
  fail G2(a). A perfect engine would now score 0.955.
- **Recommendation.** Make the outer-skin apex the single line locus for every tier, and keep the
  frozen matcher. Replace Gold11 with `gold11_relabel_v1` for gate purposes once the owner signs off.

> **Ledger: consumed development.** Gold11 development cases and the BENCHMARK_V2 CONSUMED broad set
> only. None of this is unseen-family accuracy or promotion evidence. Nothing FRESH or sealed was
> opened, and nothing from HOLDOUT10, N001–N016, EHO4, authorization artifacts or
> `incoming_codex_fetch_2026-10-17`. No human gold file was modified. The catalog sha256 is still
> `4ad243c1…`, and `human_bend_lines_v1` is untouched.

## 1. Bottom line

1. **The label errors are real. They are not, in the main, a convention.**
   - Of 71 Gold11 lines, the 3-auditor majority finds:

     | Verdict | Lines |
     |---|---|
     | OK | 13 |
     | CONVENTION_OFFSET | 20 |
     | REAL_ERROR | 23 |
     | NOT_A_BEND | 15 |

   - **31 formed bends are missing.** 15 are ordinary folds. 12 are joggle halves and 4 are
     22–30° kinks; these 16 count only under the owner's conventions.
   - **What costs score.** Lines an exact apex line could never match:

     | Verdict | Lines that cannot match |
     |---|---|
     | CONVENTION_OFFSET | 8 |
     | REAL_ERROR | 17 |
     | NOT_A_BEND | 15 |
     | **Total** | **40 of 71** |

   - **Shares of the 71 defects that cost score** (those 40 lines plus the 31 missing bends):
     convention offsets 11 %, real errors 24 %, not a bend 21 %, missing bends 44 %.
2. **The specific hypothesis, "human at the virtual sharp or mold line", is refuted.**
   - **Direction.** 19 of the 20 offset lines sit on the concave side. So do 45 of the 60 lines
     that have an apex line within 15 mm. None is consistently on the outer virtual sharp, which
     lies 1.1–1.3 mm convex of the apex on the 90° folds.
   - **Inner-skin cluster (5 lines).** These sit at the inner crease or inner virtual sharp,
     1.0–1.6 mm from the apex.
   - **Inner-skin-of-thick-sheet or bend-axis cluster (7 lines).** These sit 2.5–3.2 mm from the
     apex. All 12 lines in these two clusters still match an apex line.
   - **Beyond the arc centre (8 lines).** At 4.1–5.7 mm, these lie on a flange's inner face or in
     free air. No bend locus explains them.
   - **sc_cec08dfca9ab.** All 6 of its lines, which are not script output, lie 4.8–5.9 mm into free
     air on the concave side. All 3 auditors call them REAL_ERROR.
   - **The only deliberate convention found is "inner surface".** On CG4FAF273093, STEP axes were
     "shifted to the inner-surface arc midlines" (2.5–2.6 mm from the apex, all 3 lines match). On
     CG191DEC7F81, STEP axes were "offset onto the observed bend surfaces" (2.8–5.7 mm, 3 of the 5
     lines miss).
3. **Gold11's "human" tier is mostly engine-derived or CAD-derived, and partly circular.** 45 of 71
   lines are identical to the bit (0.000 mm) to the drafts of
   `scripts/adjudicate_remaining_verified_bends.py`.
   - **The script.** It is untracked, from 16 Jul, and writes `source: human_two_point`. It builds
     the lines in three ways:

     | Built from | Scans | Lines | Edits applied |
     |---|---|---|---|
     | F1 zero-shot decomposition regions | CG84C8FA12D2, CGE4B0549E41, CG4BDD1371B6, CGEBA9A869AC | 24 | extensions, ±7 mm copies, joined fragments, a closed rectangle |
     | Plane-lattice intersection lines | CG65CD575B1C, CG64C412CC73 | 13 | +14.000 mm copies; 4 hand-entered vertical "corner folds" that are wall free ends |
     | STEP cylinder axes, hand-aligned and moved onto the inner surface | CG191DEC7F81, CG4FAF273093 | 8 | – |

   - **They reached the catalog unchanged,** as `authoritative_user_verified`.
   - **The rest.** 20 lines are Codex multiview (2 scans). Only CG08506AE97D's 6 lines are not
     script output.
4. **Rescored on the existing predictions, with the frozen matcher and no engine re-run:**

   | Gold11 truth | Best lane F1 (P / R) | Runtime F1 | Perfect apex engine F1 |
   |---|---|---|---|
   | Human Gold11, as published | 0.458 (0.500 / 0.423) | 0.074 | **0.388** |
   | **gold11_relabel_v1** | **0.510** (0.633 / 0.427) | **0.141** | 0.955 |
   | Diagnostic: human snapped to the apex | 0.626 | 0.148 | 0.625 |
   | Diagnostic: only OK + CONVENTION lines snapped | 0.504 | 0.074 | 0.488 |

   - **The locus convention alone is worth little:** +0.046 for the best lane and 0 for the
     runtime.
   - **The full broad set (57 scans) with gold11_relabel_v1 substituted:**

     | Cohort | Best lane F1 | Runtime F1 |
     |---|---|---|
     | All 57 scans | 0.445 → **0.460** | 0.114 → **0.129** |
     | CAD available | 0.474 → 0.493 | 0.154 → 0.172 |
5. **Recommendation** (owner sign-off required):
   - Adopt the **outer-skin apex** (the midline of the outer bend arc) as the only line locus. This
     is what the AI gold, bench_labeling `lines` and the engines' scan-fitted lines already use.
   - **Do not widen the 4 mm gate.**
   - **Replace Gold11 for gate purposes with gold11_relabel_v1.** Refit B2 on CG65CD575B1C and
     re-verify it, then make that scan AI too.
   - Until sign-off, G2(a) is **unmeasurable** on today's Gold11. It is not failed by the engine.

## 2. What was done, and what was not redone

- **Audit workflow** (`wf_37989420-76c`, sherman-gold11-label-audit-continue):
  - **Auditors.** 3 blind auditors per scan, giving 33 audits. The run re-used 5 results from the
    cache of the killed run `wf_9c8183ed-61a` instead of redoing them. Each auditor judged every
    line and listed the missing folds. Each also measured the offset of every line from the outer
    apex in its own perpendicular sections, which it rendered itself.
  - **Relabels.** 11 parallel AI relabels, one per scan, made with bench_labeling (`claude-vision-ai-v1`).
  - **Verifiers.** 2 independent verifiers per relabel, giving 22 verdicts.
  - **Re-use.** Prep and lines runs that already existed were copied byte for byte and hash-checked;
    see each `relabel/<scan>/reused_from.json` or `run_record.json`.
- **This roll-up re-ran nothing:** no auditor, relabel, verifier, prep, lines, write-label or engine
  step. It did three things:
  - **Read the agent outputs verbatim** from the run's transcripts. This reproduces the workflow's
    own compact record exactly on 11 of 11 scans: every auditor's verdict counts, missing counts,
    offsets and `relabel_pass`.
  - **Cross-checked every human line mechanically** against the relabel's apex lines, and against
    the virtual-sharp lines in `relabel/<scan>/lines.json`. The auditors' offsets agree with these
    to a median of 0.03 mm (p90 0.18 mm, max 0.67 mm, over 56 lines).
  - **Rescored existing predictions** with `research/broad_baseline/score.py broad`. That run took
    under 3 s and 0.1 GiB.
- **Two read-only provenance checks:**
  - catalog lines compared with the provisional-script drafts;
  - bend counts compared with the STEP v10 counts in `cad_crosscheck_raw/`.

## 3. Per-scan majority verdicts

The provenance column comes from the identity check (§1.3):

- **script** means the lines are bit-identical to `adjudicate_remaining_verified_bends.py` output.
  The part in brackets says what the script built them from: F1 regions, plane-lattice lines, or
  STEP axes moved onto the inner surface.
- **Codex** means Codex multiview gold.
- **clicks** means the lines were drawn as two clicked points each.

In the "Missing" column, the majority figure comes first and the three auditors' counts follow in
brackets. "Unan." is the number of lines all 3 auditors agree on.

| Gold11 | Scan | Part, stage | Provenance | Lines | OK | Conv. offset | Real error | Not a bend | Missing (a1/a2/a3) | Unan. | Relabel lines | Verifiers | v1 uses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CG08506AE97D | sc_cec08dfca9ab | 10839000 partial 2 | clicks | 6 | 0 | 0 | 6 | 0 | 4 (4/4/4) | 6/6 | 10 | pass, pass | AI |
| CG191DEC7F81 | sc_7a1773eca9fa | 49001000 PRELIM | script (STEP, inner surface) | 5 | 0 | 5 | 0 | 0 | 0 (0/0/0) | 3/5 | 5 | pass, pass | AI |
| CG1E0BBA2933 | sc_4f8188b4d1a8 | 38172000 X | Codex | 10 | 3 | 3 | 2 | 2 | 4 (4/4/4) | 7/10 | 12 | pass, pass | AI |
| CG4BDD1371B6 | sc_d1732de18023 | 37361005 full | script (F1 regions) | 4 | 0 | 2 | 2 | 0 | 0 (0/0/0) | 2/4 | 4 | pass, pass | AI |
| CG4FAF273093 | sc_70b39f530f00 | 49024000 full | script (STEP, inner surface) | 3 | 0 | 3 | 0 | 0 | 0 (0/0/0) | 3/3 | 3 | pass, pass | AI |
| CG64C412CC73 | sc_152f1b1aacb2 | 40500000 partial 2 | script (plane lattice) | 7 | 0 | 0 | 3 | 4 | 4 (3/4/4) | 7/7 | 7 | pass, pass | AI |
| CG65CD575B1C | sc_07062eee17aa | 38172000 partial 1 | script (plane lattice) | 6 | 2 | 0 | 2 | 2 | 0 (0/0/2) | 6/6 | 6 | **fail, fail** | **human** |
| CG84C8FA12D2 | sc_71cf0b216f43 | 10839000 full | script (F1 regions) | 8 | 2 | 2 | 4 | 0 | 4 (4/4/4) | 6/8 | 12 | pass, pass | AI |
| CGE4B0549E41 | sc_08686dc1d5d4 | 10839000 partial 1 | script (F1 regions) | 4 | 0 | 0 | 2 | 2 | 6 (6/6/6) | 2/4 | 8 | pass, pass | AI |
| CGEBA9A869AC | sc_f2475286312b | 38172000 partial 2 | script (F1 regions) | 8 | 0 | 3 | 2 | 3 | 5 (5/7/5) | 6/8 | 10 | pass, pass | AI |
| CGF41F49066A | sc_2714fb7e0c0b | 38172000 V | Codex | 10 | 6 | 2 | 0 | 2 | 4 (4/4/3) | 9/10 | 12 | pass, pass | AI |
| **All 11** | | | | **71** | **13** | **20** | **23** | **15** | **31** | **57/71** | **89** | 10 of 11 | 10 AI + 1 human |

**What each scan's gold gets wrong** (majority, with the auditors' measured offset from the outer
apex):

- **CG08506AE97D.**
  - HB1–HB6 sit 4.8–5.9 mm into free air on the concave side, past the arc centre. They are
    parallel and full length.
  - No locus convention explains them. A single frame transform does not repair them either; see
    `frame_hypothesis.json`.
  - Missing: two joggles, 4 bends.
- **CG191DEC7F81.**
  - All 5 lines are on the inner side of the bend.
  - M01 and M04 sit at 2.8 and 3.2 mm, so they still match an apex line.
  - M02, M03 and M05 sit at 5.1–5.7 mm, on the base's inner face past the inner tangent line, so
    they cannot match.
- **CG1E0BBA2933.**
  - B02, B04 and B06 sit at 1.0–1.4 mm, at the inner virtual sharp, which is harmless.
  - B09 and B10 are on the tab flat at 3.4 mm. They are still matchable.
  - B07 and B08 are 15 mm lines on butt seams.
  - Missing: 2 end-flange hinges (44 mm) and 2 slot-tab jogs of 22–25°.
- **CG4BDD1371B6.**
  - M02 and M03 sit on the inner crease (1.6 and 1.1 mm), which is harmless.
  - M01 sits on the lip's inner face (4.4 mm).
  - M04 runs along the window top edges (5.2 mm).
- **CG4FAF273093.** All 3 lines are on the inner-skin apex, 2.5–2.6 mm from the apex with t =
  2.8 mm. All 3 match.
- **CG64C412CC73.**
  - M03–M06 are wall free ends or straight wall.
  - M01, M02 and M07 carry over-long extents taken from engine plane-lattice lines. M07 fails the
    span test.
  - Missing: 2 top tabs (16 mm) and 2 end lips (21–22 mm).
- **CG65CD575B1C.**
  - M01 and M04 are 295–300 mm lines on 78–86 mm tab folds; only 26–29 % of each line is on the
    fold.
  - M03 and M06 are exact copies of M02 and M05, shifted 14.000 mm onto free flange edges.
  - One auditor, and the relabel, also find two 30° ears.
- **CG84C8FA12D2.**
  - M05–M08 are slanted 0.6–2.4° across the 40 mm wall face, 5.9–11.7 mm from the apex at
    midspan.
  - M01 and M02 sit on the flange's inner skin at 4.1–4.5 mm, and 31–33 % of each runs past the
    fold.
  - Missing: two joggles (about 2 mm step), 4 bends. The relabel's 12 lines equal the STEP count
    of 12.
- **CGE4B0549E41.**
  - M01 and M02 sit at 2.3–3.3 mm, skewed. 27–29 % of each runs past the fold.
  - M03 and M04 close a rectangle and are not folds.
  - Missing: 2 return folds (133 mm) and two joggles (4 bends).
- **CGEBA9A869AC.**
  - M01, M02 and M07 sit at 4.6–5.1 mm in free air. They are verbatim copies of F1 regions OB1,
    OB2 and OB6.
  - M03 and M05 are 254–262 mm lines on 82 mm tab folds; M05 is on the wrong side of a cutout,
    14.8 mm away.
  - M04, M06 and M08 are free edges.
  - Missing: the end up-flange (201 mm), 2 ears (43 mm) and the 2 arrow-slot tab folds.
- **CGF41F49066A.**
  - 6 of its lines are OK.
  - B05 sits on the bend axis at 2.5 mm. B07 sits near it at 3.1 mm, skewed 0.6° and about 11 mm
    over-long at each end. Both still match.
  - B06 and B08 are 15 mm seam corners.
  - Missing: 2 end-flange folds (44 mm) and 2 window-tab kinks of about 30° (STEP C5/C7).

**STEP count check** (counts only, full scans only): on the 5 non-staged Gold11 scans, the relabel's
bend count equals the STEP v10 count on all 5. The human count equals it on only 2 (CG4BDD1371B6 and
CG4FAF273093).

## 4. Convention offsets, real errors and missing bends

### 4.1 Which errors cost score

A line costs score when even an exact outer-apex line of the same fold cannot match it under the
frozen matcher (5°, 4 mm, 60 % of both spans).

| Majority verdict | Lines | Still matchable by an apex line (score-neutral) | Not matchable (an apex engine loses an FN and an FP) |
|---|---|---|---|
| OK | 13 | 13 | 0 |
| CONVENTION_OFFSET | 20 | 12 (offset ≤ 3.2 mm) | **8** (4.1–5.7 mm) |
| REAL_ERROR: wrong place or wrong extent | 23 | 6 | **17** |
| NOT_A_BEND: free edge, seam or flat face | 15 | 0 | **15** |
| **All lines** | **71** | **31** | **40** |
| MISSING_BEND: a formed fold with no line | 31 (relabel: 33) | – | **31** |

The six REAL_ERROR lines that still match are:

- **CG1E0BBA2933 B09 and B10:** 3.4 mm offset, with about 0.6 mm of slack left;
- **CG64C412CC73 M01 and M02:** over-long, but the span overlap is still 0.69–0.73;
- **CGE4B0549E41 M01 and M02.**

These pass only because the matcher is tolerant.

**The REAL_ERROR lines split into three kinds:**

- **Placement only (10):** CG08506AE97D HB1 and HB2, CG4BDD1371B6 M01 and M04, CG1E0BBA2933 B09
  and B10, and CG84C8FA12D2 M05–M08.
- **Extent only (5):** CG64C412CC73 M01, M02 and M07, and CG65CD575B1C M01 and M04.
- **Both (8):** CG08506AE97D HB3–HB6, CGE4B0549E41 M01 and M02, and CGEBA9A869AC M03 and M05.

**The missing bends split into three kinds:**

- **15 ordinary folds.** These are 16–201 mm lips, hinges, tabs, return folds and ears.
- **12 joggle halves.** These come from 2 joggles each on CG08506AE97D, CG84C8FA12D2 and
  CGE4B0549E41, counted under the owner rule that a joggle is 2 bends.
- **4 small kinks of 22–30°.** These are on CG1E0BBA2933 and CGF41F49066A, counted under the owner
  rule that small formed bends count.

The relabel also draws 2 ears of 30° on CG65CD575B1C. Only one auditor listed them, but both relabel
verifiers confirmed that the scan has 6 formed bends.

### 4.2 Offset distribution (mm from the outer apex, auditor median per line)

| Verdict | n | min | p25 | median | p75 | max | over 4 mm |
|---|---|---|---|---|---|---|---|
| CONVENTION_OFFSET | 20 | 0.99 | 2.27 | 2.95 | 4.61 | 5.70 | 8 |
| REAL_ERROR | 23 | 0.24 | 2.76 | 5.10 | 5.78 | 14.81 | 14 |
| OK | 13 | 0.14 | 0.46 | 0.73 | 1.11 | 2.71 | 0 |

The 20 CONVENTION_OFFSET values, in order:

> 0.99, 1.06, 1.08, 1.36, 1.61 | 2.49, 2.50, 2.60, 2.60, 2.80, 3.10, 3.20 | 4.09, 4.54, 4.58,
> 4.68, 5.10, 5.14, 5.45, 5.70

**All 20 are parallel and full length.**

- They lie within 1.0° of the apex line.
- The offset varies by at most 1.23 mm between the line's two ends.
- At least 67 % of every line lies on the fold.

**They fall into three clusters:**

| Offset | Lines | Where the line sits | Physical locus |
|---|---|---|---|
| 1.0–1.6 mm | 5 | Inner crease or inner virtual sharp (t − 0.4·Ri): CG1E0BBA2933 B02, B04, B06; CG4BDD1371B6 M02, M03 | Yes, the inner skin |
| 2.5–3.2 mm | 7 | Inner-skin apex of thick sheet (CG4FAF273093, t = 2.81, deliberately placed there by the script); flange-side inner tangent (CG191DEC7F81 M01, M04; t = 2.0); bend axis, i.e. the arc centre (CGF41F49066A B05, B07) | Yes, but it uses up to 80 % of the gate |
| 4.1–5.7 mm | 8 | Past the arc centre. On a flange's inner face: CG191DEC7F81 M02, M03, M05 (STEP axes the script moved onto the observed surface) and CG84C8FA12D2 M01, M02 (F1 regions). In free air: CGEBA9A869AC M01, M02, M07 (F1 regions OB1, OB2, OB6 verbatim) | **None** |

The mechanical side test splits each offset in the section into two parts: the component toward the
virtual sharp and the lateral component. 19 of the 20 offsets lie on the concave side. The one
nominally convex value (CG191DEC7F81 M03) is an artefact of the unscanned base outer skin; all 3
auditors place that line on the base's inner face.

**The CONVENTION_OFFSET label was applied unevenly across scans.**

- **The same kind of line got both labels.** CGEBA9A869AC's free-air lines (4.6–5.1 mm, parallel,
  concave) were called CONVENTION_OFFSET by all 3 auditors. CG08506AE97D's free-air lines
  (4.8–5.9 mm) were called REAL_ERROR by all 3.
- **How to read the label.** Treat CONVENTION_OFFSET as a *parallel systematic offset*, not as a
  recognised drafting convention.
- **Only 12 of the 20 are both a physical locus and score-neutral.** Those 12 are the 1.0–3.2 mm
  clusters.

### 4.3 What the locus diagnostic shows

The diagnostic snaps human lines onto the nearest relabel apex line and keeps each human line's
extent.

**Snap rule:**

- The nearest apex line must be within 5° and 8 mm.
- 8 mm is twice the gate. It falls in the empty gap of the measured offsets: the largest
  same-fold offset is 7.55 mm, and the next is 8.90 mm, which is a free edge.
- The full rule is in `truth/human_snapped_apex_manifest.json`.

| Variant | Lines moved | Best lane F1 | Runtime F1 | Perfect apex engine F1 |
|---|---|---|---|---|
| Human Gold11 | 0 | 0.458 | 0.074 | 0.388 |
| Snap only the lines the majority called OK or CONVENTION_OFFSET | 33 | 0.504 (+0.046) | 0.074 (+0) | 0.488 |
| Snap every line within the window | 54 | 0.626 (+0.168) | 0.148 (+0.074) | 0.625 |

- **The convention offsets are worth +0.046 F1 to the best lane and nothing to the runtime.**
- **Moving every misplaced line onto the apex is worth +0.168.** Most of that comes from lines the
  auditors called REAL_ERROR: CG08506AE97D's 6 lines and CG84C8FA12D2's slanted wall lines.
- **The snapped variant still keeps 15 not-a-bend lines, the extent errors and 31 missing bends.**
  That is why even a perfect engine scores only 0.625 against it.
- **This is a diagnostic only, never truth.**

## 5. Agreement: 3 auditors and the earlier single triage agent

**Between the 3 auditors:**

- **Line verdicts.** 57 of 71 are unanimous, 14 are 2-of-3, and none lacks a majority.
- **On 4 scans all 3 auditors give identical verdict counts.** On the other 7, two of the three
  agree.
- **The 14 disagreements are always between neighbouring classes:**

  | Disagreement | Lines | Which lines |
  |---|---|---|
  | OK vs CONVENTION_OFFSET | 7 | all ≤ 3.2 mm, so score-neutral |
  | CONVENTION_OFFSET vs REAL_ERROR | 5 | CG84C8FA12D2 M01, M02; CGE4B0549E41 M01, M02; CGF41F49066A B07 |
  | REAL_ERROR vs NOT_A_BEND | 2 | CGEBA9A869AC M04, M06 |

- **No line was called OK by one auditor and REAL_ERROR or NOT_A_BEND by another.**
- **Missing-bend counts** are identical on 7 of 11 scans. On the other 4 they differ by 1–2.

**Against the triage, on the 6 scans it covered (49 lines):**

| | Single triage agent | 3-auditor majority on the same 6 scans |
|---|---|---|
| Lines judged not OK | 27 | 38 (the triage's 27 plus 11 more) |
| The triage's 27 "wrong" lines | – | **23 confirmed** (REAL_ERROR or NOT_A_BEND), **4 recast as CONVENTION_OFFSET**, **0 overturned to OK** |
| Missing bends | 11 | 25 |

- **The 4 recast lines are CG84C8FA12D2 M01, M02 and CGEBA9A869AC M01, M07.** All 4 sit at
  4.1–5.1 mm, so they still cost score.
- **The 11 extra non-OK lines:**
  - CG1E0BBA2933 B02, B04, B06, B09 and B10;
  - CG64C412CC73 M01, M02 and M07, the engine-seeded extents the triage noted but did not count;
  - CGEBA9A869AC M02, which the triage noted;
  - CGF41F49066A B05 and B07.

  9 of them are still matchable by an apex line; CG64C412CC73 M07 and CGEBA9A869AC M02 are not.
- **The 25 missing bends** are the triage's 11, plus 12 joggle halves and kinks under the owner's
  conventions, plus CGEBA9A869AC's 2 arrow-slot tab folds, which the triage had booked as
  misplaced lines.
- **The triage's verdicts hold. Its two estimates do not:**
  - "A perfect engine would score about 0.61 on today's labels" is in fact **0.388**. The 5
    untriaged scans are defective too (11/15/11 instead of the assumed 22/0/0), and the triaged
    scans lack more bends than the triage counted.
  - "The best lane would score about 0.63 on corrected labels" is in fact **0.510**. The relabel
    adds 18 real bends that the lanes mostly miss. It also removes 4 seam "TPs" where engine false
    lines happened to match mislabelled seams.

## 6. Rescored F1 tables

All rows come from `score.py broad` on the existing `runs/best_f1` and `runs/runtime` predictions,
using the frozen matcher and the raw-scan frame. CIs come from resampling scans (10,000 resamples,
seed 20261017), the same method as the baseline report. The human row reproduces the published
TP/FP/FN exactly in both lanes.

### 6.1 Gold11 (11 scans)

| Gold11 truth | Gold lines | Best TP/FP/FN | P | R | **F1** | F1 CI | Macro | Runtime TP/FP/FN | P | R | **F1** | F1 CI | Perfect apex engine TP/FP/FN, F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Human (published) | 71 | 30/30/41 | 0.500 | 0.423 | **0.458** | 0.282–0.621 | 0.452 | 3/7/68 | 0.300 | 0.042 | **0.074** | 0.000–0.171 | 31/58/40, **0.388** |
| **gold11_relabel_v1** (10 AI + 1 human) | 89 | 38/22/51 | 0.633 | 0.427 | **0.510** | 0.416–0.594 | 0.487 | 7/3/82 | 0.700 | 0.079 | **0.141** | 0.021–0.281 | 85/4/4, **0.955** |
| Sensitivity: AI relabel on all 11 | 89 | 38/22/51 | 0.633 | 0.427 | 0.510 | 0.416–0.594 | 0.487 | 7/3/82 | 0.700 | 0.079 | 0.141 | 0.021–0.281 | 89/0/0, 1.000 |
| Diagnostic: human snapped to the apex | 71 | 41/19/30 | 0.683 | 0.577 | 0.626 | 0.515–0.711 | 0.606 | 6/4/65 | 0.600 | 0.085 | 0.148 | 0.026–0.282 | 50/39/21, 0.625 |
| Diagnostic: only OK + CONVENTION snapped | 71 | 33/27/38 | 0.550 | 0.465 | 0.504 | 0.351–0.634 | 0.484 | 3/7/68 | 0.300 | 0.042 | 0.074 | 0.000–0.171 | 39/50/32, 0.488 |

- **Paired change in F1 against human Gold11** (95 % CI):

  | Variant | Best lane | Runtime |
  |---|---|---|
  | gold11_relabel_v1 | −0.130 to +0.249 | −0.026 to +0.205 |
  | Snapped to the apex | +0.033 to +0.324 | 0.000 to +0.205 |

- **G2(a)** requires F1 ≥ 0.80 and precision ≥ 0.90. It **fails in every row for both lanes**.
  - On gold11_relabel_v1, the best lane is 0.29 F1 short and 0.27 short in precision.
  - Only gold11_relabel_v1 makes the gate reachable: a perfect apex engine scores 0.955 there,
    against 0.388 on today's labels.
- **The relabel is not a free gift to the engine. It moves the score both ways:**
  - **Up** where engine lines were correct but the gold was off-apex: CG84C8FA12D2 0.211 → 0.696,
    CGEBA9A869AC 0.143 → 0.500, CG08506AE97D 0.000 → 0.429, CG4BDD1371B6 0.571 → 0.857.
  - **Down** where the human gold had forgiven engine errors:
    - **Seams.** Engine false lines on seams had been matching seam labels: CGF41F49066A
      0.737 → 0.476 and CG1E0BBA2933 0.762 → 0.609.
    - **Extents.** Over-long engine extents had matched engine-seeded gold: CG64C412CC73
      0.462 → 0.308. A trimmed short-fold extent now fails the 60 % span test: CG4FAF273093
      0.667 → 0.333.
    - **Missing bends.** Real bends were missing from the gold: CGE4B0549E41 0.667 → 0.400.

### 6.2 Per scan (TP/FP/FN, F1)

| Gold11 | Best: human | Best: relabel_v1 | Best: snapped | Runtime: human | Runtime: relabel_v1 | Perfect apex engine vs human |
|---|---|---|---|---|---|---|
| CG08506AE97D | 0/4/6, 0.000 | 3/1/7, 0.429 | 3/1/3, 0.600 | 0/0/6, 0.000 | 0/0/10, 0.000 | 0/10/6, 0.000 |
| CG191DEC7F81 | 1/2/4, 0.250 | 1/2/4, 0.250 | 1/2/4, 0.250 | 1/0/4, 0.333 | 1/0/4, 0.333 | 2/3/3, 0.400 |
| CG1E0BBA2933 | 8/3/2, 0.762 | 7/4/5, 0.609 | 8/3/2, 0.762 | 0/0/10, 0.000 | 0/0/12, 0.000 | 8/4/2, 0.727 |
| CG4BDD1371B6 | 2/1/2, 0.571 | 3/0/1, 0.857 | 3/0/1, 0.857 | 1/0/3, 0.400 | 1/0/3, 0.400 | 2/2/2, 0.500 |
| CG4FAF273093 | 2/1/1, 0.667 | 1/2/2, 0.333 | 2/1/1, 0.667 | 1/2/2, 0.333 | 1/2/2, 0.333 | 3/0/0, 1.000 |
| CG64C412CC73 | 3/3/4, 0.462 | 2/4/5, 0.308 | 3/3/4, 0.462 | 0/0/7, 0.000 | 0/0/7, 0.000 | 2/5/5, 0.286 |
| CG65CD575B1C | 2/0/4, 0.500 | (human kept) 2/0/4, 0.500 | 2/0/4, 0.500 | 0/0/6, 0.000 | 0/0/6, 0.000 | 2/4/4, 0.333 |
| CG84C8FA12D2 | 2/9/6, 0.211 | 8/3/4, 0.696 | 7/4/1, 0.737 | 0/5/8, 0.000 | 4/1/8, 0.471 | 2/10/6, 0.182 |
| CGE4B0549E41 | 2/0/2, 0.667 | 2/0/6, 0.400 | 2/0/2, 0.667 | 0/0/4, 0.000 | 0/0/8, 0.000 | 2/6/2, 0.333 |
| CGEBA9A869AC | 1/5/7, 0.143 | 4/2/6, 0.500 | 3/3/5, 0.429 | 0/0/8, 0.000 | 0/0/10, 0.000 | 0/10/8, 0.000 |
| CGF41F49066A | 7/2/3, 0.737 | 5/4/7, 0.476 | 7/2/3, 0.737 | 0/0/10, 0.000 | 0/0/12, 0.000 | 8/4/2, 0.727 |
| **All 11** | **30/30/41, 0.458** | **38/22/51, 0.510** | **41/19/30, 0.626** | **3/7/68, 0.074** | **7/3/82, 0.141** | **31/58/40, 0.388** |

### 6.3 Broad set (57 scored scans) with gold11_relabel_v1 substituted

"Subst." is the case list v3 with the 11 Gold11 gold specs replaced. Every other scan is scored
exactly as published, using v3's frozen gold.

| Cohort | Scans | Gold v3 → subst. | Best v3 TP/FP/FN | Best subst. TP/FP/FN | Best F1 v3 → subst. | Runtime v3 → subst. TP/FP/FN | Runtime F1 v3 → subst. |
|---|---|---|---|---|---|---|---|
| **All** | 57 | 305 → 323 | 115/97/190 | 123/89/200 | **0.445 → 0.460** | 21/43/284 → 25/39/298 | **0.114 → 0.129** |
| Gold11 | 11 | 71 → 89 | 30/30/41 | 38/22/51 | 0.458 → 0.510 | 3/7/68 → 7/3/82 | 0.074 → 0.141 |
| Not Gold11 | 46 | 234 | 85/67/149 | unchanged | 0.440 | unchanged | 0.125 |
| **CAD available** | 36 | 208 → 226 | 91/85/117 | 99/77/127 | **0.474 → 0.493** (P 0.517 → 0.5625) | 21/43/187 → 25/39/201 | **0.154 → 0.172** |
| No CAD | 21 | 97 | 24/12/73 | unchanged | 0.361 | unchanged | 0.000 |
| Full scans | 44 | 228 → 236 | 93/64/135 | 96/61/140 | 0.483 → 0.489 | 16/35/212 → 20/31/216 | 0.115 → 0.139 |
| Staged scans | 13 | 77 → 87 | 22/33/55 | 27/28/60 | 0.333 → 0.380 | 5/8/72 → 5/8/82 | 0.111 → 0.100 |

**Uncertainty:**

- **Scan-resampled CIs after the substitution:**

  | Cohort | Best lane | Runtime |
  |---|---|---|
  | All 57 | 0.400–0.516 | 0.064–0.200 |
  | CAD available | 0.424–0.556 | 0.090–0.258 |

- **Paired change against v3 on all 57 scans:** −0.032 to +0.071 for the best lane and −0.008 to
  +0.060 for the runtime. The substitution moves the broad headline by less than its own noise.
- **The bar the recipe-guided engine must beat** on the CAD cohort becomes F1 0.493 and precision
  0.5625 for the best lane, and F1 0.172 for the runtime. If owner sign-off is withheld, the
  published figures stand: 0.474 / 0.517 and 0.154.

**A second change on disk (not part of this audit):**

- One non-Gold11 gold file changed after v3 was frozen. The AI gold for sc_2233372b8df7 is now
  revision 2, which adds the two 15° joggle bends at the owner's request.
- Rescoring with the label files exactly as they are now gives a best-lane F1 of 0.458 (All) and
  0.490 (CAD), and a runtime F1 of 0.129 and 0.171. Both lanes miss the 2 new lines.
- That is the only non-Gold11 row that differs from the published scores; the script checks this.

## 7. Recommendation

### 7.1 Line-locus convention: the outer-skin apex

- **Define the locus once, for every tier.**
  - A bend line is the midline of the **outer** bend arc, on the visible scanned surface.
  - It is the outer-skin point on the bisector of the two flanges, and it spans the formed fold.
  - The AI gold, bench_labeling `lines` (`label_line`, `locus: apex`) and this relabel already use
    it.
  - Keep the frozen matcher as it is: 5°, 4 mm, 60 % of both spans.
- **Why this locus and not another:**
  - **It is observable and unique.** It lies on the scanned surface, needs no radius or
    plane-intersection fit, and is what the bench tool measures.
    - The virtual sharp is in free air and depends on plane fits.
    - The bend axis needs a radius fit.
    - The inner skin is often only partly scanned; on CG191DEC7F81 M01 it is missing over about
      half the fold.
    - Where one skin is missing, the apex is derived from the other skin plus the thickness t.
  - **It is where engines already put their lines.** Matched scan-fitted lines sit about 1 mm
    from it (median 1.08 mm on the broad set).
  - **There is no convention worth preserving.** Gold11's offsets spread over five loci, from 1.0
    to 5.9 mm. 37 of its 71 lines are detector-derived (F1 regions or the plane lattice), and 20
    are Codex lines.
    - The one deliberate convention is "on the inner surface", used on CG4FAF273093 and
      CG191DEC7F81 (8 lines).
    - It cannot be seen where the inner skin is unscanned.
    - On tight or obtuse folds it lands past the 4 mm gate: CG191DEC7F81 M02, M03 and M05 sit at
      5.1–5.7 mm.
  - **The alternatives use up the tolerance.** The virtual sharp sits about 0.41·(Ri + t) away,
    1.1–1.3 mm into free air. The inner skin sits t away, 1.0–2.8 mm on these parts. The bend axis
    sits Ri + t away, about 2.5–3 mm here, which is up to about 80 % of the 4 mm gate.
- **Converting another locus.** Re-measure on the scan; do not apply a global shift, because the
  offsets in Gold11 are not constant. Where the outer skin is unscanned, derive the apex from the
  inner skin and the measured thickness t, and flag it.
- **Do not widen the gate to absorb label offsets.**

### 7.2 Should Gold11 be replaced for gate purposes? Yes, with gold11_relabel_v1. Owner sign-off required.

**Why:**

1. **Today's Gold11 cannot measure G2(a).**
   - A perfect apex engine scores 0.388 against it (P 0.348), below the gate.
   - The legacy best lane beats that perfect engine, 0.458 vs 0.388. The gold rewards the same
     omissions the engine makes.
2. **The labels are defective on every scan.**
   - 58 of 71 lines are not OK by majority vote, and 40 of them cannot be matched by a correct
     line.
   - 31 formed folds are missing.
   - The majority is strong: 57 of 71 lines are unanimous, and no line is split.
3. **The "human" tier is mostly script-generated** (§1.3). 37 lines come from engine detector
   output, which is circular for an engine yardstick. 8 come from hand-aligned STEP axes. Only
   CG08506AE97D's 6 lines were clicked, and all 6 are 4.8–5.9 mm off.
4. **gold11_relabel_v1 is better supported than the gold it replaces:**
   - 3 auditors' measurements, 2 independent verifiers per relabel, and apex lines fitted on the
     scan;
   - its bend counts match STEP on 5 of 5 full scans, where the human gold matches on 2;
   - against it, a perfect engine scores 0.955. That will be 1.000 once CG65CD575B1C is relabelled.

**Conditions for sign-off:**

- **(a) The owner reviews the 11 relabel contact sheets and close-ups** in `relabel/<scan>/render/`.
- **(b) CG65CD575B1C:** refit B2, the −Y flange, 0.6–0.9 mm inward onto the outer arc, then
  re-verify and swap it to AI. The bend count and extents already passed. The sensitivity row shows
  no score change on today's predictions.
- **(c) Confirm the conventions applied in v1:**
  - joggle = 2 bends: 12 bends;
  - 22–30° kinks and ears count: 4 kinks, plus 2 ears on CG65CD575B1C once it is swapped;
  - a window or relief inside a fold gives one line.
- **(d) Retag the 8 script-derived scans.** Change them from `authoritative_user_verified` to
  something like `LEGACY_ENGINE_DERIVED`.
  - Keep the legacy Gold11 for history (the 0.458 lineage). Quote both sets of numbers until
    sign-off, and never tune on either.
- **(e) Re-score the Jev A/B with `truth/gold11_relabel_v1_gold_override.json`.** This needs no new
  calls; see `jev_ab/JEV_AB.md`.

**Not recommended:** using the snapped human gold as truth. It is a diagnostic only, and it keeps 15
not-a-bend lines, the extent errors and 31 missing bends.

## 8. Caveats

- **Independence.** The auditors, the relabeller and the verifiers are all AI agents of the same
  family. Agreement between them is not full independence.
  - The independent checks are limited:
    - mechanical offsets agree with the auditors to 0.03 mm (median);
    - STEP counts match on the 5 full scans;
    - the provenance identity check is exact.
  - Positions on partial scans have no CAD check.
- **The relabel used the auditors' findings as hints.** On some scans its extents are the median of
  the auditors' measured fold ends, for example CG4FAF273093 and CGEBA9A869AC. So it is not fully
  independent of the audit.
- **16 of the 31 missing bends depend on owner conventions:** 12 joggle halves and 4 small kinks.
  Without those conventions, 15 remain.
- **Gold11 is small.** It has 11 scans, so its F1 CIs are about ±0.09–0.17 wide.
  - The relabel's change against human Gold11 is not significant for either lane: both paired CIs
    include 0 (§6.1).
  - Only the snapped diagnostic's gain for the best lane has a paired CI that excludes 0.
- **CONVENTION_OFFSET** marks a parallel, systematic offset. On 8 of its 20 lines it is not a
  physical locus (§4.2).

## 9. Files

All paths are under `docs/production/runlog/broad_baseline/gold11_audit/`.

| File | Contents |
|---|---|
| `audit_results.json` | The full audit: 33 auditor outputs verbatim (every line's verdict, offset, coverage and reasons, plus missing bends), the 11 relabel reports, the 22 verifier verdicts, per-line majority, mechanical cross-check, triage comparison, provenance identity check, STEP count check and roll-up |
| `truth/gold11_relabel_v1.json` | The truth variant: per-scan source, verifier verdicts, label paths and sha256 values |
| `truth/gold11_relabel_v1_gold_override.json` | Map from case_id to gold spec, for `--gold-override` |
| `truth/case_list_gold11__{human,relabel_v1,all11_ai_relabel,human_snapped_apex,human_snapped_apex_conv}.json` | The Gold11 case lists |
| `truth/case_list_broad_v3__gold11_relabel_v1.json` | The full broad case list with gold11_relabel_v1 substituted |
| `truth/human_snapped_apex*/`, `truth/human_snapped_apex_manifest.json` | The snapped diagnostic gold and the per-line snap log |
| `rescore/<variant>/scores_{best_f1,runtime}.json` | score.py reports |
| `rescore/RESCORE_SUMMARY.json` | All tables, CIs, ceilings and reproduction checks |
| `relabel/<scan>/`, `prep/<scan>/` | Relabel lines, pairs and renders; bench prep |
| `auditor{1,2,3}/<scan>/`, `renders/<scan>/` | Auditor evidence and the shared section renders |
| `scripts/` | `build_audit_results.py`, `build_truth_variants.py`, `rescore_gold11_variants.py` and `g11a_common.py` |

The AI relabel labels are in `data/bend_corpus/annotations/ai_bend_lines_gold11_relabel_v1/<asset>.json`,
with provenance in `provenance/`. To rebuild, run the three scripts in this order:

```
OMP_NUM_THREADS=1 .venv/bin/python -B docs/production/runlog/broad_baseline/gold11_audit/scripts/build_audit_results.py
OMP_NUM_THREADS=1 .venv/bin/python -B docs/production/runlog/broad_baseline/gold11_audit/scripts/build_truth_variants.py
OMP_NUM_THREADS=1 .venv/bin/python -B docs/production/runlog/broad_baseline/gold11_audit/scripts/rescore_gold11_variants.py
```

`build_audit_results.py` reads the workflow transcripts, which are kept outside the repo;
`audit_results.json` keeps their content.
