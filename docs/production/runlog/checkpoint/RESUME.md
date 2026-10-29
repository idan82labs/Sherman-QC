# Resuming the Sherman workflows after an account switch / session restart

A workflow's resume cache only lives inside one session. After a /login or restart, a plain
`resumeFromRunId` re-runs every agent live (this already happened twice). Do this instead:

1. **Stop** any still-running copies (TaskStop) so nothing runs twice.
2. **Checkpoint:** `python3 docs/production/runlog/checkpoint/make_cont.py`
   - Saves every finished agent result into `<job>_done/`.
   - Regenerates `<job>_cont.js` for: wave4, g3fix, gold_v2.
   - `checkpoint_status.json` lists the agents that were in flight; those re-run from scratch.
3. **Engine safety (wave 4 only).** If `a1:change:N` was in flight, it may have half-edited
   `domains/production/engine/**`. Before relaunching, restore
   `docs/production/runlog/a1/iter_N/engine_before.tgz` over the engine dir. Do this only when
   `iter_N/CHANGE.md` is missing or incomplete. Do NOT restore if `a1:change:N` finished.
4. **Relaunch** each job with `Workflow({scriptPath: "<this dir>/<job>_cont.js"})`, no resumeFromRunId.
   - Finished agents come back through a cheap loader agent.
   - Per-scan gold outcomes are also on disk in `gold_set_v2/outcomes/`.
5. **Geometry lock.** A stale lock dir `/private/tmp/claude-501/sherman_geom.lock` with no live heavy
   process (check `ps`) can be removed. Every heavy job runs under it.
6. **Agents outside workflows** (if any were running) are continued with SendMessage to their id.
   A new session cannot reach them, so restart the task with a prompt that says to reuse files on disk.

Run IDs at checkpoint time (2026-10-18):
| Job | Run ID | Script |
|---|---|---|
| wave4 | wf_bcbaf368-ceb | scratchpad/cont/wave4_v3.js |
| g3fix | wf_ac66546b-4b6 | |
| gold_v2 | wf_80a10842-890 | |

The unwrapped sources the generator uses are listed in make_cont.py JOBS.

## Paused state (2026-10-18 ~10:00, owner switching accounts)
- All three workflows are STOPPED.
  - Final checkpoint taken; see `checkpoint_status.json`.
  - The geometry lock was cleared; no heavy process was running.
- **wave4:** 44 agents done. Baseline and A1 iterations 1–3 are complete. `a1:change:4` was in flight but had NOT modified the engine (it is byte-identical to `iter_4/engine_before.tgz`), so no restore is needed.
- **g3fix:** 31 agents done. All the missing blind reviews are finished; only the freeze step remains.
- **gold_v2:** 18 agents done; the selection and some preps are finished.
- **To resume:** relaunch `wave4_cont.js`, `g3fix_cont.js` and `gold_v2_cont.js` from this folder. Run `make_cont.py` first only if something ran after this checkpoint.

## 2026-10-19: do NOT resume wave4's devcheck step
DEV-CHECK and FRESH were each evaluated ONCE on e3-1.9.0, by owner order (final_eval_2026-10-19, ledger seq 3-5).
If wave4 is ever resumed, remove its 'devcheck' agent first: it must never run DEV-CHECK a second time.
Engine work after iter 7 is the "coverage fixes" agent (final_eval_2026-10-19/FIXES.md), not the A1 loop.
