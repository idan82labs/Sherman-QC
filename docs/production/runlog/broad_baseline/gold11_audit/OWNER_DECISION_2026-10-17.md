# Owner decision, 2026-10-17: the Gold11 gate reference is the vision relabel

## Decision

1. **Gold11 for gates is `gold11_relabel_v1`.** This is the vision relabel that Claude marked from the scans (annotator `claude-vision-ai-v1`). Every one of the 11 scans was checked by two independent blind verifiers. Machine drafts are not used as Gold11 reference lines.
   - Labels: `data/bend_corpus/annotations/ai_bend_lines_gold11_relabel_v1/` (11 files)
   - Manifest: `truth/gold11_relabel_v1.json` (status OFFICIAL, AI relabel on 11/11 scans, 89 lines)
   - Case lists: `truth/case_list_gold11__relabel_v1.json` and `truth/case_list_broad_v3__gold11_relabel_v1.json`
2. **The human Gold11 is kept untouched, as a historical diagnostic only.** This covers the catalog `data/bend_corpus/annotations/bend_line_gold_catalog_v1.json` (sha256 `4ad243c1…`) and `data/bend_corpus/annotations/human_bend_lines_v1/`. Neither was modified. The human Gold11 no longer decides any gate.
3. **The line locus is the outer-skin bend apex,** the midline of the outer arc. Other conventions:
   - A joggle is 2 bends.
   - Formed bends of 15–30° count.
   - A fold that is interrupted by windows or reliefs but folds the same two plates is one line.
4. **The matcher is unchanged:** axis angle within 5°, transverse offset within 4 mm, span overlap of at least 60%, one-to-one Hungarian assignment (`research/broad_baseline/score.py`).

## What completed it (CG65CD575B1C / sc_07062eee17aa)

- **Why it was the last scan.** In the first relabel, line B2 (the −Y flange) failed both verifiers. It sat about 0.9 mm from the outer apex, toward the virtual sharp.
- **What causes it.** The fold's apex bows by about 3.5 mm over its 305 mm length, because the web edge strip is warped. The default apex rule puts a straight line on the outer envelope of the arc. That line touches the skin only where the bow comes closest.
- **The fix (r2, `relabel/sc_07062eee17aa_r2/`)** had two parts:
  - The web plane was refit on a local web top-skin patch.
  - The new opt-in override `apex_mode: station_median` was used (`research/bench_labeling/README_OVERRIDES.md` §4). It is additive: bends that do not use it are unchanged.
- **Result.** B2 is now a median of 0.45 mm from the measured apex (r1: 0.92 mm), with no systematic offset. The best least-squares straight line scores 0.56 mm. B1 and B3–B6 are byte-identical to r1.
- **Verification.** Two independent blind verifiers passed it (`relabel/sc_07062eee17aa_r2/VERIFICATION.json`).
  - Both passed all 6 lines and found no missed bend.
  - Verifier 2 called B2 a marginal pass. Over the first ~10 mm at the −X end of the bow, B2 is 1.5–2.3 mm from the apex. Any straight line is about 1.8 mm off there.
- **Label.** `data/bend_corpus/annotations/ai_bend_lines_gold11_relabel_v1/a962166c3019434c10628722.json`, revision 2.

## Scores against the 11/11 reference

These rescore the existing predictions only. No engine was run. Outputs are in `rescore/relabel_11of11/`.

| Set | Best lane F1 (TP/FP/FN) | Runtime F1 (TP/FP/FN) | Perfect-apex ceiling |
|---|---|---|---|
| Gold11, 11/11 (89 lines) | 0.510 (38/22/51) | 0.141 (7/3/82) | 1.000 (89/0/0) |
| Broad 57 with the relabel substituted | 0.458 (123/89/202) | 0.129 (25/39/300) | 0.615 (best) / 0.457 (runtime)* |

\* Only the Gold11 part of the broad set has an apex-relabel gold. For the broad ceiling, the 11 Gold11 rows are scored as perfect and the 46 other scans keep the engine's own scores.

Compared with the 10/11 proposal, the engine scores do not change. On this scan, the engine's 2 matched flange lines match both the human gold and the relabel, so the Gold11 F1 stays 0.510 for best and 0.141 for runtime. The perfect-apex ceiling rises from 0.955 to 1.000.

## Superseded

- The 10/11 PROPOSED truth files were copied to `truth/superseded_10of11/` before the update.
- `case_list_gold11__all11_ai_relabel.json` and the snapped-apex diagnostics pin this scan's label at revision 1 by sha256. They are now historical.
