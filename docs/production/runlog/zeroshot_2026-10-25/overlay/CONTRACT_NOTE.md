# Contract note: `zeroshot_overlay_v1` (job-level zero-shot overlay), 2026-10-25

**Status.** Implemented under the owner's standing "execute" direction (orchestrator decision: CAD_PATH.md options 1 + 2, post-E3). Freeze ledger: NOTE entry (see the ledger seq in the run report).

## What it is

A new, versioned, job-level document next to E3's result. It is **not** part of `ShopfloorInspectionResultV1`: `contracts/production` (frozen code) is unchanged, as are E3's characteristics, decisions, summary, the inspection contract and `JobResultSummaryV1`.

| | |
|---|---|
| Schema | `sherman_zeroshot_overlay_v1` (pydantic `domains.production.zeroshot.overlay.ZeroshotOverlayV1`, `extra="forbid"`) |
| Overlay version | `zs-overlay-1.1.0` (1.1 adds the joggle angle rule and `angle_checks`, from `angle_check/ANGLE_CHECK.md`) |
| Job output | `<job>/zeroshot/zeroshot_overlay_v1.json` (also `zs_lines.json`, the zs step output, and `overlay_status.json`, the step record) |
| Station API | `GET /api/line/jobs/{job_id}/zeroshot-overlay` (`line.read`; 409 job not completed, 404 `OVERLAY_MISSING`, 500 `OVERLAY_INVALID` / `OVERLAY_INCONSISTENT`; ETag) |
| Station screen | "Detections (unverified)" panel under the bend list (dashed, amber / grey, never green); an unexpected-bend flag is a REVIEW reason; a lowered verdict shows CHECK |
| PDF | Section 3 "Detections (unverified)" (Hebrew `זיהויים (לא מאומתים)`), plus a dashed note under the summary when the verdict was lowered |

## Fields

- `lane`: `zs_lane_v1`, variant, runtime version (`zs-prod-1.0.0`), pinned code fingerprint (`af87df2f…`, = frozen `config_v1.json`), model sha256 (`ccc28773…`), the CAD priors used, wall time, candidate counts.
- Identity: `scan_sha256`, `engine_result_sha256` (sha256 of the exact `engine/result.json` bytes), `engine_version`, `inspection_validity`. The API, the station and the PDF drop an overlay whose identity does not match the job.
- Verdict: `e3_summary_state`, `effective_summary_state`, `verdict_lowered`, `review_reasons`.
- `lines[]`: `line_id` (`Z01`…), `kind` (`WITHHELD_BEND` | `UNEXPECTED_BEND` | `REFUSED_SCAN_DETECTION`), `decision` (`REVIEW` | `UNVERIFIED`, never PASS/FAIL), `source = ZS_LANE_V1`, `bend_uid` (withheld bends only), raw-frame endpoints, E3-frame endpoints (VALID only), length, included angle, selector score, association metrics.
- `lines[].angle_status`: `SHOWN` or `NOT_MEASURED_JOGGLE` (then `incl_deg` is null: a zs angle is never shown as a number on a joggle bend, nor on a line of unknown bend on a part that has joggles).
- `flags[]`: `UNEXPECTED_FEATURE_REVIEW` with the unexplained line ids.
- `angle_checks[]`: `INDEPENDENT_ANGLE_DISAGREES` (bend, E3 angle, zs angle, delta) when E3 FAILed a MEASURED included angle on a non-joggle bend and the associated zs line's angle differs by more than 5°. A REVIEW reason only: the FAIL and the summary state stay. `joggle_bend_uids` lists the joggle bends.
- `withheld_bends`, `withheld_bends_located`, `notes`.

## Rules (pre-registered in CAD_PATH.md; association 5° / 6 mm / overlap 0.5; dedupe with the frozen matcher 5° / 4 mm / 0.6)

1. **VALID, option 2.** A formed (not NOT_YET_FORMED) bend whose position E3 does not publish gets the closest zs line associated with its decoded-stage CAD outer-apex locus (mapped to raw through E3's registration) that matches no published line: `WITHHELD_BEND`, `REVIEW`.
2. **VALID, option 1.** A zs line explained by no E3 published line, no decoded-stage bend locus and no stage-model formed-bend locus is `UNEXPECTED_BEND` (`REVIEW`) and raises `UNEXPECTED_FEATURE_REVIEW`.
3. **Refused scans** (validity not VALID): all zs lines are `REFUSED_SCAN_DETECTION`, `UNVERIFIED`; no association, no flag; the scan stays refused (`INVALID_INSPECTION`).
4. **Joggles** (ANGLE_CHECK.md definition, from the CompiledPart only): two supported bends sharing a panel, opposite directions, equal sweep (within 1°) of at most 60°, axes at most 10 t apart.
5. **Verdict.** `effective = REVIEW_REQUIRED` only when E3 is `ALL_MEASURED_IN_TOLERANCE` or `STAGE_IN_TOLERANCE` and a flag exists; otherwise `effective = e3`. The model refuses any document that creates or upgrades a pass-like state. FAIL, REVIEW, INCOMPLETE and INVALID never change.

## Runtime and containment

- The zs step (`python -m domains.production.zeroshot.lane_cli`) runs in parallel with E3 under its own supervisor limits (90 s, the engine's 4 GiB RSS cap, so both steps stay under 8 GiB; one thread each). The merge step (`overlay_cli`, 60 s) runs after E3's result validated and before C06.
- Any zs or merge failure (crash, timeout, RSS, pin mismatch, unpinned code, garbage, other scan) means **no overlay**: the job, its result and its summary are unchanged. E3 failure, cancel or lease loss stop the zs step first. The zs group is recorded in `logs/active_step_zs.json` and reaped like the engine step.
- Blindness: the zs step gets only the staged `input/scan.<ext>` (other names are refused) and rotation-invariant CAD facts (bend count, thickness, number of axis directions) from the CompiledPart.
- The lane is enabled by the wiring only when the engine bundle carries `domains/production/zeroshot/lane_cli.py`; the bundle copies the pin's 60 source files, the pin and the model, and the zs step re-verifies all of them before loading the model (fail closed, e.g. after any E3 engine bump, until re-pinned).

## Known limits

- The job's `result_summary` (history chips, job lists) keeps E3's state; the lowered REVIEW shows on the station banner, the overlay panel and the PDF note for the job on screen.
- The overlay lines are listed, not drawn in the 3D viewer.
- The association thresholds were set on consumed DEV data (CAD_PATH.md); the honest check is still the 5 OD-3 DEV-CHECK cases and new deliveries with STEP.
