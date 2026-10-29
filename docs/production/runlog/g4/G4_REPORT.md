# G4 scoring report (wave 4, E3 arm): PARTIAL — gate FAIL

Scored 2026-10-19 by the scorer step. Engine `e3-1.9.0`, packet `packet_e3/`, protocol E4-G4-2. Evaluator:
`domains.production.engine.eval.g4.evaluate_g4` against `canary_key_SEALED.json` (via symlink, not copied).

## Headline

**Gate: FAIL.** The decision does not depend on the missing reviews. Among the close-ups already adjudicated,
two measured bends do not land and one bend is mistagged, none of them canaries. The gate needs a lands rate of 1.0
and 0 mistags, and more verdicts cannot undo a failure that two independent primary reviewers have confirmed.

**This is a partial review and cannot certify a PASS.** The reviewer returns handed to the scorer were truncated
partway through scan `scan_459f38302d`, during primary A's B6 record. Only 14 of 52 scans have both primary reviews:
64 of 399 close-ups, plus 13 single-primary records for `scan_459f38302d`. The other 38 scans, 335 close-ups, have
no verdicts. That leaves 30 of the 40 canaries unscored, so the evaluator marks the review `REVIEW_INVALID` and
`VERDICTS_INCOMPLETE` on top of the substantive failures.

## Canary check (review validity)

- **10 of 10 canaries in the reviewed close-ups were flagged by both primary reviewers** and by the adjudicated
  verdict. No canary passed and no reviewer missed one (`misses_by_reviewer = {}`).
- The 30 canaries in the unreviewed 38 scans cannot be checked. For the packet as a whole, review validity is
  **not established**. It is incomplete, not failed.
- Canary identities and kinds are left out of this report on purpose, because 38 scans are still unreviewed.
  They are in the scorer-only file `g4_canary_check_partial_SEALED.json` (mode 0400). That file also holds the full
  evaluator output, whose `unflagged_images` list names those canaries. For the same reason no unsealed
  `g4_gate_report.json` was written.

## Gate metrics on the adjudicated non-canary close-ups (E3)

| Metric | All reviewed | Physical dev | Synthetic | Gate |
|---|---|---|---|---|
| MEASURED close-ups LANDS | **25/27 = 92.6 %** | 13/15 = 86.7 % | 12/12 = 100 % | 100 % |
| MEASURED bends LANDS (every tile lands) | **22/24 = 91.7 %** | 11/13 = 84.6 % | 11/11 = 100 % | 100 % |
| Mis-tags (tag_ok = NO) | **1** | 1 | 0 | 0 |
| NOT_YET_FORMED no-line justified | 27/27 = 100 % | 19/19 | 8/8 | 100 % |
| UNKNOWN justified | n/a: none of the 5 UNKNOWN close-ups is in the reviewed subset | | | 100 % |
| AMBIGUOUS (E1-justified, no question) | 4 in packet, rule-justified | | | 0 not justified |

- Synthetic sample requirement: met (26 synthetic scans, 16 incomplete).
- Tie-breaks: 6 close-ups in `scan_151dc615d7` decided primary disagreements. 2 in `scan_24698f2bcd` were
  redundant because the primaries already agreed. 0 unresolved.

### Failing bends (non-canary, adjudicated)

| Scan | Image | Bend (displayed) | Class | Verdict | Evidence |
|---|---|---|---|---|---|
| scan_151dc615d7 | img_54fc081015ef | B5 `bu_a756c2e2336e1530784d` (tip hook) | MEASURED, UNEXPECTED_FORMED | **NOT_LANDS** (A, B and tie-break) | The line and apex dot sit 1.8–2.6 mm off the outer apex, beyond the material, against a tolerance of max(2, Ri+t) ≈ 2.0–2.2 mm. The fitted flange arrows point into empty space. |
| scan_151dc615d7 | img_f96503b78ccc | B11 `bu_0625a6c8bc441be83a83` (mirror tip hook) | MEASURED, UNEXPECTED_FORMED | **NOT_LANDS** (A, B and tie-break) | The same failure mode, mirrored: dot about 2.4–2.6 mm from the apex, line 1.7–1.9 mm beyond the scan edge. |
| scan_3db9fefd2b | img_e69b9ef7cc6f | B7 `bu_a481c4126f40dded2f52` | NOT_YET_FORMED | **UNKNOWN_JUSTIFIED_MISTAG** (A and B agree) | The flat call is justified. At the bold-label spot the recipe CAD shows `1 · B1 · 4dca39`, and the CAD label for B7 sits about 7 mm below the line. Both reviewers think the cause is more likely the renderer's label placement than an engine mis-tag. This needs a renderer and engine check before any re-review. |

The two hook failures are on one physical scan, both on 8 mm end hooks of the second (V-bottom) bend. The pattern
fits the apex being placed on the wrong side of a short hook, not a random failure. Bends that are not in the
reviewed subset were not judged.

## Sensitivity: how "UNDETERMINABLE, recorded no" is encoded

Some reviewers said TAG_OK was **UNDETERMINABLE** because the recipe CAD panel was blank or did not show the bend,
but recorded `no` because their return schema had no UNDETERMINABLE value.

- `g4_verdicts.jsonl` stores these as `UNDETERMINABLE`, the protocol answer the reviewers stated. This applies to
  `scan_151dc615d7` primary A on B5, B6, B10, B11 and B14, and to the 12 joggle tiles of `scan_459f38302d`.
- A raw-`no` variant gives the **same** gate numbers on the adjudicated set, because the `scan_151dc615d7` cases
  went to tie-break and the `scan_459f38302d` cases have only one primary.
- Once primary B's returns for `scan_459f38302d` arrive, the blank-CAD joggle tiles (B1–B4, 12 images) need the
  same decision. Recorded as `no`, they would add up to 4 mis-tagged bends. That is a renderer gap (no recipe CAD at
  the final-stage pose), not evidence against the engine.

The same decoding applies to `lands`. On no-line close-ups (NOT_YET_FORMED), reviewers wrote `yes` or `no` to mean
unknown_justified YES or NO, because their schema had no NOT_APPLICABLE. The file stores `lands = NOT_APPLICABLE`
with `unknown_justified`. One `occluded` on a no-line close-up (`scan_151dc615d7` B3, primary A) is stored as
unknown_justified NO. The tie-break settled it as YES.

## Legacy arm (visual baseline, informational, not gated)

**Not computed.** No legacy reviewer returns reached the scorer. The legacy packet has 22 scans and 279 close-ups.
Its lands rate, mis-tag count and UNKNOWN justification remain open.

## Reviewer conduct notes (no effect on validity)

- Two reviewers disclosed that, while looking up the verdict schema, they saw the aggregate engine-vs-bar F1 table
  in `G4_RENDER.md`. It contains no per-image or canary information. Neither opened a `*_SEALED*` file.
- One reviewer overwrote another agent's scratch helper (`scratchpad/g4r2/crop.py`). Its owner has since
  rewritten it.

## Files

- `g4_verdicts.jsonl`: 149 validated `G4VerdictV1` lines (128 PRIMARY for 64 close-ups in 14 complete scans,
  13 single-PRIMARY lines for `scan_459f38302d`, and 8 TIEBREAK). Reasons are condensed from the reviewer returns. Image sha256 values come
  from the review manifest, and every subject `bend_uid` was checked against it.
- `g4_canary_check_partial_SEALED.json` (0400, scorer only): the canary check and the full evaluator output.

## To complete G4

1. Re-deliver the full reviewer returns for the 38 remaining scans and for primary B of `scan_459f38302d`, then
   append them and re-run the evaluator command in `G4_RENDER.md`.
2. Score the legacy packet the same way, with `--no-synthetic-requirement`.
3. The gate stays FAIL until the B5/B11 hook apex placement is fixed in the engine and the B7 label placement is
   explained. Either change means a new render and a new blind review.
