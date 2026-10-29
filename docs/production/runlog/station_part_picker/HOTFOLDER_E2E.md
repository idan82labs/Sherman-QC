# Hotfolder E2E simulation with edge cases (2026-10-28)

Run: `docs/production/runlog/e2e/20261028T-hotfolder-e2e/` (`steps.json`, `hotfolder.json`, screenshots).
Command, under one geometry slot:

```
/private/tmp/claude-501/sherman_slot.sh node scripts/production/run_e2e_real.mjs --run 20261028T-hotfolder-e2e --hotfolder
```

Result: **19 PASS, 1 FAIL**. The failure is W3b with the known signature `{"B10":{"missing":true,"labels":[]},"B7":{"missing":true,"labels":[]}}` (0 markers, `labels []`; ticket `docs/production/tickets/T-2026-10-26-W3B-STUDIO-MARKER-FLAKE.md`). W3b gates nothing downstream. Every other step passed, including the unchanged real E2E steps (R-full, R-scan_1, R-scan_2, R-hebrew, Z1) and the part picker step P1. **Zero false PASS** (`hotfolder.json: zero_false_pass = true`).

## How it was run

- The servers start from a temporary app copy (C04 `engine_bundle build --profile app`). The data root and inbox are temp dirs under `$TMPDIR`; no real station data is touched.
- Everything goes through the **real line service**: its hotfolder poller, the real resolver, the runner, E3 `e3-1.13.0` and the zs overlay. The station is driven in Chromium at 1920x1080. No job was created through the API. The only API calls were the operator actions (assign and dismiss), the part registration and the recipe retirement.
- Inputs are the allowed 10839000 scans the existing E2E already uses (full, scan_1, scan_2), as read-only custody copies. The variants were all built in the run's temp work dir:
  - re-exports: one extra PLY header comment, so a new sha256 with the same geometry;
  - a mirror image in x;
  - a version densified 4x (3,452,460 points);
  - a truncated copy (the first 60 % of the bytes);
  - JPEG bytes under a `.ply` name;
  - a 40-facet ASCII STL under a `.ply` name;
  - the nominal scan of a synthetic E2 HAT part (`generate_part_set`), standing in for another part.
  The helper is `scripts/production/hotfolder_e2e_support.py`. No sealed or forbidden data was used.
- How each file was written:
  - most files: a hidden temp name, then a rename, with the mtime aged 5 s (as `dropFile` does);
  - H1b: a slow write straight into the final name, in 2 MiB chunks every 0.4 s, with one 8 s stall partway (an SMB hiccup);
  - H1c and H9: a `.done` sidecar next to the file;
  - the zero-byte file: created in place.

## Expected vs actual

"Where it ended" is the scan row's `stored_rel_path` (it exists in every case) and the inbox afterwards. Unless a row says otherwise, the inbox was empty after the step (Z1: inbox emptied).

| # | Case (file name) | Expected | Actual: job / verdict | UI text | Audit events | File ended |
|---|---|---|---|---|---|---|
| H1a | Clean name `10839000_01_FULL_SCAN 1.ply` (R-full) | auto-run | job completed; VALID, OUT_OF_TOLERANCE_FOUND, CONSISTENT, 12/12 | "FIX" banner | SCAN_RECEIVED, JOB_QUEUED, JOB_STARTED, JOB_COMPLETED | `archive/scans/…` |
| H1b | Slow growing write into the final name, with an 8 s stall (`…FULL_SCAN 14.ply`) | never ingested while partial; one job on the complete bytes | 1 job, completed; VALID, OUT_OF_TOLERANCE_FOUND; the archived sha = the full file's sha | "FIX 4 to fix 8 to check" | same 4 kinds, once | `archive/` |
| H1c | Temp name + rename + `.done` {sha256, size_bytes, declared_stage: 10} (`…SCAN_2 5.ply`) | auto-run; stage declared from the sidecar | 1 job, `declared_stage {10, SIDECAR}`; E3 declared 10 = inferred 10, FINAL_INCOMPLETE (FINAL recipe, 2 bends not formed), OUT_OF_TOLERANCE_FOUND | "FIX 2 to fix 6 to check" | once | `archive/`; `.done` consumed |
| H2 | Unknown name `10839 000_01_FULL_SCAN 9.ply` | held back -> operator picks -> runs | P1: held UNMATCHED_SKU, 10839000 suggested first, picked in the Hebrew UI, completed VALID / OUT_OF_TOLERANCE_FOUND, "Part assigned by operator" | ribbon "Scan held back … Choose part (2)" | QUARANTINE_ASSIGNED (one, after a double click) | `archive/` |
| H3 | Ambiguous `10839000_01 23553000 FULL_SCAN 1.ply` (23553000 registered, never released) | held with suggestions -> pick | held AMBIGUOUS_SKU after 4 s; one suggestion, 10839000 (RESOLVER_CANDIDATE; the unreleased SKU is not offered); picked in the UI; the job was killed with the service (H13), recovered, completed on attempt 2: VALID / OUT_OF_TOLERANCE_FOUND | "File name matches several parts" | SCAN_QUARANTINED, QUARANTINE_ASSIGNED, JOB_STARTED, JOB_RECOVERED, JOB_STARTED, JOB_COMPLETED | `archive/` |
| H4a | Revision mismatch `10839000_02_FULL_SCAN 1.ply` | held | held REVISION_MISMATCH after 4 s; revision 02 read from the name, 10839000 suggested (FILENAME_SKU); dismissed (RESCAN) | "Revision differs from the released recipe" | SCAN_QUARANTINED, QUARANTINE_DISMISSED | `archive/` (kept) |
| H4b | Part not released `23553000_B_FULL_SCAN 1.ply` | held, not assignable | held PART_NOT_READY; no suggestion; assign -> **409 PART_NOT_READY**; dismissed | "Part recipe not ready" | SCAN_QUARANTINED, QUARANTINE_DISMISSED | `archive/` |
| H4c | Retired part `10839000_01_SCAN_2 7.ply`, after 10839000@v1 was retired | held, not offered, assign refused | held PART_NOT_READY (NO_ACTIVE_RECIPE); released-parts search empty; assign by SKU -> **409 PART_NOT_READY**, by recipe_id -> **409 RECIPE_RETIRED**; the picker shows "No released part matches"; dismissed in the UI (OTHER + note) | "Part recipe not ready" | SCAN_QUARANTINED, QUARANTINE_DISMISSED | `archive/` |
| H5 | Wrong part under a right name: synthetic HAT scan as `…FULL_SCAN 11.ply` | E3 WRONG_PART_SUSPECTED, never PASS | completed; **UNTRUSTED_REGISTRATION** (NO_DATUM_PLANE_MATCHED), INVALID_INSPECTION, 0/12 measured; the overlay is also INVALID_INSPECTION. Never PASS; see finding F1 | "Rescan · Scan could not be aligned to the CAD" | 4 kinds, once | `archive/` |
| H6 | Mirrored part: the real full scan mirrored in x, `…FULL_SCAN 12.ply` | E3 MIRRORED_PART_SUSPECTED, never PASS | completed; **VALID**, OUT_OF_TOLERANCE_FOUND, CONSISTENT, 12/12. Not a PASS, but **the mirror was not detected**; see finding F2 | "FIX 3 to fix 9 to check" | 4 kinds, once | `archive/` |
| H7 | Duplicate: the R-full bytes again as `…FULL_SCAN 1 (1).ply` | no new job | no new row and no new job (still 1 job for those bytes); SCAN_DUPLICATE on the original scan | — | SCAN_DUPLICATE | source removed from the inbox; the bytes already archived |
| H8a | Truncated PLY (60 %), `…FULL_SCAN 4.ply` | content quarantine, not assignable | PLY_TRUNCATED after 62 s (60 s grace); assign -> **409 NOT_RESOLVABLE**; dismissed | "Scan file is incomplete" | SCAN_RECEIVED, SCAN_QUARANTINED, QUARANTINE_DISMISSED | `quarantine/<date>/<sha>.ply` + `.reason.json` |
| H8b | Zero-byte file, created in place (`…FULL_SCAN 6.ply`) | visible to the operator, never inspected | **new**: EMPTY_SCAN after 121 s (`undersize_grace_s`); before this change it stayed in the inbox forever | "Scan file is empty" | SCAN_RECEIVED, SCAN_QUARANTINED | `quarantine/…` |
| H9a | Non-PLY bytes (JPEG) under `.ply` (`…FULL_SCAN 5.ply`) | content quarantine | UNSUPPORTED_FORMAT after 62 s; dismissed | "Unsupported scan format" | SCAN_QUARANTINED | `quarantine/…` |
| H9b | ASCII STL under `.ply` (`…FULL_SCAN 15.ply`) | content decides: a mesh; never PASS | archived as `.stl`, matched, E3: **INSUFFICIENT_CAPTURE** (SCAN_POINTS:3), INVALID_INSPECTION | "Rescan · Not enough of the part was scanned" | 4 kinds | `archive/…/<sha>.stl` |
| H9c | Foreign `.txt` in the inbox | never touched | left untouched and never registered (logged once, counted in the heartbeat as `ignored_files`); the test removed it at the end | — | none | inbox (untouched) |
| H9d | Sidecar sha mismatch: `.done` {sha256: 0…0} (`…SCAN_1 8.ply`) | content quarantine | SIDECAR_INVALID after 62 s; assign -> **409 NOT_RESOLVABLE**; dismissed | "Scan sidecar file is invalid" | SCAN_QUARANTINED, QUARANTINE_DISMISSED | `quarantine/…`; `.done` consumed |
| H10 | Over density: 3,452,460 points (`…FULL_SCAN 13.ply`) | INVALID_INSPECTION, never PASS | **UNSUPPORTED_INPUT** `DENSITY_ABOVE_VALIDATED_RANGE:points=3452460>3000000`, INVALID_INSPECTION | "Rescan · Scan file not supported" | 4 kinds | `archive/` |
| H11 | BEFORE BENDING name `BEFORE BENDING  10839000_01.ply` (a scan_1 re-export) | auto-run; stage declared | matched, completed; **no declared stage** (see decision D1); E3 inferred 8/12, FINAL_INCOMPLETE, OUT_OF_TOLERANCE_FOUND | "FIX 2 to fix 5 to check" | 4 kinds | `archive/` |
| H12 | Several files at once: H5, H6, H9b, H10, H11, H14, H1c, plus H1b and H3 in the queue | queued, one job at a time | 9 jobs, all completed; running intervals from the event log **never overlap** (`hotfolder.json: intervals`) | — | — | — |
| H13 | Line service SIGKILLed mid-job (3 s into H3's E3 run) and mid-write (H1b growing), then restarted | no loss, no double processing | every file produced exactly 1 job; the killed job got JOB_RECOVERED and completed on attempt 2; H1b was ingested once, with its complete bytes | — | JOB_RECOVERED | — |
| H14 | Hebrew + spaces `10839000_01 סריקה מלאה 1.ply` | name kept, auto-run | name stored unchanged; matched; completed VALID / OUT_OF_TOLERANCE_FOUND. P1's `סריקה ללא שם 1.ply` was held and dismissed | "FIX 4 to fix 8 to check" | 4 kinds | `archive/` |
| H15 | Archive month dir set read-only (0555, temp root only) while `10839000_02_SCAN_2 3.ply` dropped | stays in the inbox; heals after the fix; exactly once | while denied: in the inbox, row `received`, nothing stored. After chmod 0755 and the 60 s back-off (**new**): placed and held REVISION_MISMATCH; 1 SCAN_RECEIVED, 0 jobs | — | SCAN_RECEIVED, SCAN_QUARANTINED | `archive/` |

Disk full (ENOSPC) was not simulated on a real filesystem. It is covered by `tests/test_prod_hotfolder_edges.py::test_disk_full_while_copying_is_retried`: the copy fails 3 times, no `.sherman-partial` is left behind, and the retry after the back-off ingests the file once.

## Screenshots (looked at)

- Ribbon with 7 held scans: `../e2e/20261028T-hotfolder-e2e/42-hf-ribbon-held.png`
- Picker list of all held scans with reasons (English): `43-hf-picker-held-list.png`
- Content problem, cannot be assigned: `44-hf-picker-content-problem.png`
- Ambiguous name, confirm: `45-hf-picker-ambiguous-confirm.png`
- Job results:
  - `46-hf-H1b_slow_write.png`
  - `47-hf-H1c_sidecar.png`
  - `48-hf-H3_ambiguous.png`
  - `49-hf-H5_wrong_part.png` (Rescan)
  - `50-hf-H6_mirrored.png`
  - `51-hf-H10_density.png` (Rescan)
  - `52-hf-H11_before_bending.png`
  - `53-hf-H14_hebrew_space.png`
  - `54-hf-H9b_stl_named_ply.png` (Rescan)
- Retired part, nothing to pick: `55-hf-picker-retired.png`
- P1 in Hebrew/RTL, with the bidi fix for the scan-stats line: `35-picker-he-suggested.png` … `41-picker-en-result.png`

## Backend behaviour added for these cases (hotfolder, `apps/production_line/ingest/hotfolder.py`)

1. **Undersized and zero-byte files.** A file below `min_size_bytes` whose signature has not changed for `undersize_grace_s` (default 120 s) is quarantined as EMPTY_SCAN. Before this change it waited in the inbox forever and the operator never saw it. A file that is still growing is never affected.
2. **Retry after back-off instead of a permanent give-up.** Before this change, 3 failed ingests meant "give up until the file changes", so a file stayed stuck after a transient archive failure (permission denied, disk full) was fixed. Now the poller backs off for `failed_retry_s` (default 60 s) and then retries. The file stays in the inbox until it is placed: it is never lost.
3. **Orphan `.done` sidecars** are removed after `truncated_grace_s`. An orphan is a sidecar whose scan was already ingested before the marker arrived, which would otherwise stay in the inbox. This applies only in `move` mode.
4. **Foreign files** (any other extension) are still never touched. They are now logged once and counted in the poller heartbeat (`ignored_files`).

Tests: `tests/test_prod_hotfolder_edges.py` (7 tests) passes. The existing `tests/test_prod_hotfolder.py` (34 tests) passes unchanged.

## Findings and owner decisions

- **F1, wrong part (E3).** The synthetic other part under the right name came back as UNTRUSTED_REGISTRATION (NO_DATUM_PLANE_MATCHED), not WRONG_PART_SUSPECTED. It is INVALID and never a PASS, but the operator text reads "could not be aligned", not "wrong part". This is for the E3 research agent; no engine change was made here.
- **F2, mirrored part (E3).** The x-mirrored real full scan was inspected as **VALID** with all 12 bends measured (OUT_OF_TOLERANCE_FOUND). MIRRORED_PART_SUSPECTED did not fire. It happened not to be a PASS only because the part is out of tolerance anyway. A mirrored part that is in tolerance would currently be at risk of a false PASS. This needs E3 attention: either 10839000 is near mirror-symmetric, or the mirror check misses it. **Not changed here** (engine and pins are off limits).
- **D1, BEFORE BENDING -> declared stage. Not implemented, stopped as instructed.** `contracts/production/line_job.py` `DeclaredStageV1` (hashed into the zs pins) says the declared stage is "*Never derived from filename tokens*". Mapping `scan_label` (BEFORE_BENDING, SCAN_2, …) to `declared_stage` would contradict that contract. The supported ways to declare a stage are the `.done` sidecar (`declared_stage`, shown working in H1c) and the operator's stage choice in the part picker. Owner decision needed: whether a filename label may ever declare a stage. If so, the contract text must change, which means a pin re-cut.
- **D2, non-scan files.** Foreign extensions stay in the inbox untouched; the operator only sees them in the logs and heartbeat. Owner decision: should they be moved to a quarantine folder and shown on the station?
- **D3, permission-denied window.** While the archive is unwritable the scan row sits in `received` and the station shows nothing. The heartbeat and logs carry the error. Owner decision: surface "archive not writable" on the station?
- Pre-existing station UI defect, seen in `49-hf-H5_wrong_part.png` and not caused by this work: on an INVALID result the bend list's "Not measured" chip overlaps the "Not measured" cell text. Not fixed here (out of scope).
