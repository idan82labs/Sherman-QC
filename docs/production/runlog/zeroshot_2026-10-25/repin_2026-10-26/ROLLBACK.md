# Rollback: bundled production re-pin of 2026-10-26

This undoes the bundle: E3 e3-1.13.0, ENGINE_IO-A5 (32 GiB), the zs_lane_v1 and zs_lane_v3 rev3 pins, and the dense thinning step.

## Snapshot taken before any change

- Archive: `/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/repin_2026_10_26/pre_repin_snapshot.tgz`
- sha256: `54428c33fa49cd1d51776d76198fbec7e8a94c183656b17419b13fb029102273`
- Per-file sha256 before the change: `pre_repin_files.sha256`, next to the archive. A copy is in this directory as `pre_repin_files.sha256`.
- The archive holds every existing file the bundle touches (24 files; list in `touch_list.txt`).
- Files the bundle creates did not exist before:
  - `tests/test_prod_engine_e3_1_13_candidate.py`
  - `tests/test_prod_engine_e3_1_13_dense_thin.py`
  - `docs/production/runlog/zeroshot_2026-10-25/repin_2026-10-26/archive/`

Live state before the change, checked 2026-10-26:

| Item | Value |
|---|---|
| E3 engine | `e3-1.12.0`, inspect fingerprint `5fbed707cd53a084b9f8aabb64c80b26d4d0ba21cb6ebd5ea70735281a64a9c7` |
| compile / eval / contracts | `9831f9ae…` / `8aa6860c…` / `064019772ff4…` |
| zs_lane_v1 pin | sha256 `e77f1f73…`, fingerprint `af87df2f…`, verify `[]` |
| zs_lane_v3 pin | sha256 `115b7077…` (rev 2, `ab31d061…`); verify reported exactly budget.py and core.py, as designed |
| research config_v3.json | rev 3, fingerprint `dc23a905…`, sha256 `8e002f3e…` |
| freeze ledger | 24 entries, verifies, anchor `cf01e709…` |

## How to roll back

Run from the repo root (`/Users/idantal/ShermanQC/FullProject`):

```
W=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/repin_2026_10_26
shasum -a 256 $W/pre_repin_snapshot.tgz       # must print 54428c33…
tar xzf $W/pre_repin_snapshot.tgz             # restores the 24 files byte-identically, LEDGER.json included
rm -f tests/test_prod_engine_e3_1_13_candidate.py tests/test_prod_engine_e3_1_13_dense_thin.py
shasum -a 256 -c $W/pre_repin_files.sha256    # every line OK
```

The ledger rule: if the ledger has already been appended to, do **not** restore LEDGER.json from the archive, because the ledger is append-only. Instead, restore every other file, then append a NOTE entry: "ROLLBACK of the 2026-10-26 re-pin; supersedes seq N..M". To restore the code without touching the ledger, use:

```
tar xzf $W/pre_repin_snapshot.tgz --exclude docs/production/runlog/freeze/LEDGER.json
```

Then check that every live fingerprint equals the table above:

```
.venv/bin/python -B -c "from domains.production.engine.eval import freeze as F; print(F.current_code_fingerprints())"
.venv/bin/python -B -c "from domains.production.zeroshot import pin as P; print(P.verify(pin=P.load_pin(lane='zs_lane_v1')))"   # []
```

If the engine bundle was rebuilt on the station, rebuild it from the restored tree.

The E3-only alternative is `patch -R -p1 < docs/production/runlog/zeroshot_2026-10-25/station/e3_1_13/e3_1_13_0_rc.patch`. It covers only the rc patch: the version, K=2 and the thinning edits must be reverted too, so restore from the archive instead.

## Supplementary snapshot

After the main snapshot, four more test files were edited for ENGINE_IO-A5. Their exact pre-edit bytes were reconstructed by reversing single, unique-string edits:
- `tests/test_prod_supervisor.py`
- `tests/test_prod_runner.py`
- `tests/test_prod_zeroshot_overlay_runner.py`
- `tests/test_prod_e2e_fast.py`

They are archived in `pre_repin_snapshot_supp.tgz` (sha256 `1ce83c94345e3b0ff36dbd5ff83e67bbe60fa7acde8a748264750f57a8c7492e`, next to the main archive). Restore them with `tar xzf …supp.tgz -C .`.

## Rollback executed: 2026-10-26

The dense gate failed (see REPIN.md), so both archives were restored and the two new test files removed. Checks after the restore:
- `shasum -c`: all 24 + 4 files OK.
- Fingerprints equal the table above.
- zs_lane_v1 verify `[]`; zs_lane_v3 is the rev 2 pin with its designed 2-file mismatch.
- Ledger verifies, with 24 entries before the two NOTEs.
- Post-rollback tests: overlay_lane, zs_lane_v3, contract_parity, settings and supervisor, 162 passed.

## Attempt 2 (density guard, applied 2026-10-26): how to roll back

**Snapshot taken before attempt 2:**
- Archive: `/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/repin_2026_10_26/v2/pre_v2_snapshot.tgz`
- sha256: `6f1879baca04555b287f047b1b934f2331a1f83e45935d27c4d3470625192af7`
- It holds 32 files: `touch_list_attempt2.txt`, with per-file sha256 in `pre_attempt2_files.sha256`.
- Those 32 files include the four A5 test files, `v3/PREREG_FRESH_TEMPLATE.json`, and the attempt-1 REPIN.md / results.json / ROLLBACK.md.

**Files attempt 2 created:**
- `tests/test_prod_engine_e3_1_13_candidate.py`
- `tests/test_prod_engine_e3_1_13_density_guard.py`
- `repin_2026-10-26/archive/` and `repin_2026-10-26/evidence/`

**Steps** (from the repo root):

```
tar xzf /private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/repin_2026_10_26/v2/pre_v2_snapshot.tgz --exclude docs/production/runlog/freeze/LEDGER.json
rm -f tests/test_prod_engine_e3_1_13_candidate.py tests/test_prod_engine_e3_1_13_density_guard.py
```

Then:
1. Check the result with `grep -v LEDGER pre_attempt2_files.sha256 | shasum -a 256 -c`.
2. Append a NOTE "ROLLBACK of the 2026-10-26 re-pin (seq 27-29)". Never rewrite the ledger.
3. Rebuild the station engine bundle.

After that the live tree is back at e3-1.12.0 `5fbed707…`, zs_lane_v1 `af87df2f…`, and A5 PROPOSED.
