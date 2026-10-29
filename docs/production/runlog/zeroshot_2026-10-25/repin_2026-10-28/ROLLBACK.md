# Rollback: e3-1.14.0 re-pin of 2026-10-28

This undoes the 2026-10-28 re-pin. It returns the live tree to the 2026-10-26 state:
- e3-1.13.0, inspect fingerprint `f319ed5c…`;
- zs_lane_v1 pin fingerprint `52fd21d6…`, verify `[]`;
- zs_lane_v3 rev 3 pin `49a16399…`, verify `[]`;
- contracts `27d55333…`;
- ledger at 29 entries, verify OK.

## Snapshot taken before any change

- Archive: `/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/repin_2026_10_28/pre_repin_snapshot.tgz`
- sha256: `9adc6b0b90daf09ce6768e8e128b86e41c45b6ddfc53e9cc97918434ee0bd37d`
- It holds the 13 files in `touch_list.txt`. Their pre-change sha256 values are in `pre_files.sha256` (both are in this directory).
- One file did not exist before the change: `tests/test_prod_engine_e3_1_14_pose_wells.py`, added by the rc patch.

## How to roll back

Run from the repo root:

```
W=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/repin_2026_10_28
shasum -a 256 $W/pre_repin_snapshot.tgz          # 9adc6b0b…
tar xzf $W/pre_repin_snapshot.tgz --exclude docs/production/runlog/freeze/LEDGER.json
rm -f tests/test_prod_engine_e3_1_14_pose_wells.py
grep -v LEDGER docs/production/runlog/zeroshot_2026-10-25/repin_2026-10-28/pre_files.sha256 | shasum -a 256 -c
```

**The ledger is append-only.**
- If nothing was appended yet, restore LEDGER.json from the archive too.
- Otherwise append a NOTE "ROLLBACK of the 2026-10-28 re-pin (seq N..M)".

**After a rollback,** rebuild the station engine bundle. Then re-run the Rust adapter's `freeze_config`: `config_v3r3_rs.json` must match the restored files again.
