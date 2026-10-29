# Production re-pin to zs_lane_v3: plan (prepared, NOT switched)

**Status, 2026-10-25.** The live overlay still runs **zs_lane_v1**:
- `domains/production/zeroshot/pin.py` has `DEFAULT_LANE = "zs_lane_v1"`.
- The engine bundle still carries only the v1 pin.

The v3 pin is prepared and can be selected **only explicitly**, with `pin.load_pin(lane="zs_lane_v3")` or the environment variable `SHERMAN_ZS_LANE_PIN=zs_lane_v3`. An unknown value fails closed and produces no overlay.

**Do not flip before the fresh-scan test passes.** That test is the one-time run in `PREREG_FRESH_TEMPLATE.json`: v3 is primary, v1 is the comparison, and gold is frozen before any run.

## What is prepared

### `domains/production/zeroshot/assets/`

| File | Contents |
|---|---|
| `zs_lane_v3.pin.json` | sha256 `39c4d3b72653907c0c5e8bfa1091bb886efb2318e974023c3fb182a123de4df3` |
| `zs_lane_v3_zs_gbm.pkl` | Byte-identical to `v3/models/`; sha256 `ade6cde4…ffeb` |
| `zs_lane_v3_z3_gbm.pkl` | Byte-identical to `v3/models/`; sha256 `62e99d13…e28c` |

The pin carries:
- the code fingerprint `e34e1ba7…5b4a` (69 pinned files, including the pin and the 2 models);
- `frozen_config` (`config_v3.json` sha `9860f46b…0417`);
- the two models;
- `lane_params`: the v2 post-processing, the fixes, the Z3 and Z2 settings and the threshold. The bundle carries no research JSON, so the parameters live in the pin.

It is written by `research/zeroshot/zs_lane_v3/make_pin.py`, which refuses to write if the code on disk differs from `config_v3.json`.

### Code changes

**`pin.py`:**
- `PIN_RELS` lists the v1 and v3 pins.
- `DEFAULT_LANE` stays v1.
- `PIN_ENV` and `selected_lane()` select a pin explicitly.
- `load_pin(root, lane=)` checks that the pin names its own lane.
- `pinned_models()`: `verify()` checks every pinned model, now including the v3 Z3-alone model.
- `pinned_files()` includes them.
- v1 behaviour is unchanged. `PIN_REL` is still the v1 pin.

**`lane_cli.py`:**
- The lines document carries `lane` from the pin.
- When the selected pin is zs_lane_v3, `run_lane_v3` runs:
  1. the v3 core, with the CAD thickness for Z3, the snap and the evidence;
  2. the family cap K + 2;
  3. the v2 selection and post-processing;
  4. EXT and Z3R;
  5. the count cap n_bends, highest line score first. A Z3R line keeps its own pooled probability, which is below 0.5, so it is the first to go.
- Otherwise the unchanged v1 `run_lane` runs.

**`overlay.py`:** `LaneStampV1.lane` accepts `zs_lane_v1 | zs_lane_v3`. Nothing else changed. The `source` stays `ZS_LANE_V1`, and the overlay version stays `zs-overlay-1.1.0` (see the owner decision below).

**E3 engine code is unchanged.** The fingerprints below equal those in ledger seq 15 and 16, recomputed after this work:

| Fingerprint | Value |
|---|---|
| engine | `5fbed707…` |
| compile | `9831f9ae…` |
| eval | `8aa6860c…` |
| contracts | `06401977…` |

## Checks run now (v3 pin selected non-default)

### Tests

All tests were named and run with `-p no:cacheprovider`.

**Overlay tests, run twice.** Files:
- `tests/test_prod_zeroshot_overlay_rules.py`
- `tests/test_prod_zeroshot_overlay_lane.py`
- `tests/test_prod_zeroshot_overlay_runner.py`

| Pin | Result |
|---|---|
| Default (v1) | 50 / 50 passed |
| `SHERMAN_ZS_LANE_PIN=zs_lane_v3` | 50 / 50 passed. The rules file stamps every synthetic zs document with the selected lane, and the lane file checks the selected pin against its frozen config. |

**`tests/test_prod_zs_lane_v3.py`:** 26 / 26 passed. It covers:
- the EXT and Z3R rules;
- the frozen parameters;
- that the fingerprint covers v3 and the reused v2 modules;
- that v1 and v2 are unchanged;
- that the config is v2 plus the fixes;
- that the v3 pin matches the config and fails closed on the second model or a v3 source edit;
- pin selection: default, environment variable and unknown value;
- lane_cli refusal and PIN_MISMATCH with the v3 pin;
- the count-cap order;
- that the overlay accepts a v3 stamp.

**zs lane contract tests:** 40 / 40 passed. Files: `lane_v1_contract`, `lane_v2_contract`, `d1_two_skin`, `z2_self_recipe`, `z3_slicing`, `z4_selector`.

### G1 regression tier with `SHERMAN_ZS_LANE_PIN=zs_lane_v3` (`v3/g1/scorecard.{json,md}`)

**Result.**
- Regression verdict **PASS**; no binding gate failed.
- **FALSE_PASS 0 / 553.** WRONG_OR_MIRRORED_NEVER_PASS 0 / 32. SYMMETRIC_NO_WRONG_ID 0. ENGINE_FAILURES 0 / 50.

**Comparison with the v1 overlay gate (`overlay/g1/scorecard.md`).** Every gate value is identical. Only the runtime info rows differ: P95 14.3 s against 13.8 s.

**Why this is expected.** G1 exercises E3 only, so the zs pin cannot change it. The run confirms E3 is unaffected.

**Run conditions.** 562 s, one geometry slot.

### Production-path rehearsal (`v3/rehearsal/REHEARSAL.json`; tool `v3/tools/repin_rehearsal.py`)

This is the substitute for an E2E with v3. The real production zs step and merge step (`lane_cli` and `overlay_cli`) ran with both pins.

**Cohort.** 50 consumed-DEV E3-eligible scans from the CAD-path study:
- the stored e3-1.12.0 `result.json` and the E1 CompiledPart for each scan;
- the scan staged byte-identical as `input/scan.<ext>`;
- each run through `guarded_slot.sh --cad-path` (no OD-3 case is in the cohort; 0 refused).

**Result.**

| | v1 pin | v3 pin |
|---|---|---|
| lane status OK | 50 / 50 | 50 / 50 |
| overlays written and valid | 50 | 50 |
| **false PASS** (overlay creates or upgrades a pass-like state) | **0** | **0** |
| verdicts lowered to REVIEW | 0 | 0 |
| lines | 288 | 288 |
| WITHHELD / UNEXPECTED / REFUSED-scan lines | 15 / 4 / 3 | 15 / 4 / 3 |
| fixes fired | n/a | EXT 1 (45796000, on an E3 REVIEW_REQUIRED scan), Z3R 0 |
| zs step wall median / max | 7.3 / 14.0 s | 7.7 / 16.4 s |
| zs step peak RSS max | 3.24 GiB | 3.36 GiB |
| over the runner limits (90 s / 4 GiB) | 0 | 0 |

- No scan changed its effective state between v1 and v3.
- 4 cohort scans have a pass-like E3 state.

This is an engineering rehearsal on consumed DEV, not an accuracy result.

### E2E: not run with v3, and why

The C04 app bundle copies exactly the **default** pin's files (`apps/production_line/engine_bundle.py`, `ZEROSHOT_PIN_REL`; asserted by `test_bundles_carry_exactly_the_pinned_lane`). Inside a bundle, selecting the v3 pin therefore fails closed: the v3 sources and models are absent, so `PIN_MISMATCH` is raised and no overlay is written.

Running the real E2E with v3 needs the bundle change in step 3 below, which is part of the flip itself. The rehearsal above covers the zs step and the merge on 50 real scans. The E2E is step 6 of the flip.

## Steps to flip the default to v3 (only after the fresh-scan test passes)

### 1. Preconditions

- The fresh-scan RESULTS are ledgered, under a PREREG made from `PREREG_FRESH_TEMPLATE.json`, and the owner-confirmed decision rule says the flip is allowed.
- The owner approves the flip.

### 2. Re-verify the frozen artefacts

The two commands below must print `[] []` and `e34e1ba7… True`:

```
.venv/bin/python -B -c "from domains.production.zeroshot import pin as P; p=P.load_pin(lane='zs_lane_v3'); print(P.verify(pin=p), P.unpinned_loaded(pin=p))"
.venv/bin/python -B -c "import json; from research.zeroshot.zs_lane_v3 import fingerprint as F; c=json.load(open('research/zeroshot/zs_lane_v3/config_v3.json')); print(F.lane_fingerprint()['sha256'], F.lane_fingerprint()==c['code_fingerprint'])"
```

### 3. Edits (one change set)

1. **`domains/production/zeroshot/pin.py`:** set `DEFAULT_LANE = "zs_lane_v3"`. Keep `PIN_REL`, which still means the v1 pin, for rollback.
2. **`domains/production/zeroshot/__init__.py`:** set `LANE_ID = "zs_lane_v3"` and `ZS_RUNTIME_VERSION = "zs-prod-1.1.0"`.
3. **`apps/production_line/engine_bundle.py`:** set `ZEROSHOT_PIN_REL = "domains/production/zeroshot/assets/zs_lane_v3.pin.json"`, so that bundles carry the v3 pinned files.
4. **`tests/test_prod_zeroshot_overlay_lane.py`:** in `test_default_pin_is_v1`, assert `zs_lane_v3` and rename the test. The bundle test already follows `DEFAULT_LANE`.
5. **`overlay/CONTRACT_NOTE.md`:** record the lane `zs_lane_v3` and the new fingerprint and model shas.
6. **Owner decision:** keep `OverlayLineV1.source = "ZS_LANE_V1"` as the generic zero-shot source tag (no UI change), or add `ZS_LANE_V3`. Adding it means changing `overlay.py`, `frontend/react/src/features/station/model/zeroshotOverlay.ts` and the PDF, plus a frontend type, lint and build check outside the repo copy.

### 4. Tests (named only)

```
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_prod_zeroshot_overlay_rules.py tests/test_prod_zeroshot_overlay_lane.py tests/test_prod_zeroshot_overlay_runner.py
SHERMAN_ZS_LANE_PIN=zs_lane_v1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_prod_zeroshot_overlay_rules.py tests/test_prod_zeroshot_overlay_lane.py tests/test_prod_zeroshot_overlay_runner.py   # rollback path
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_prod_zs_lane_v3.py tests/test_prod_zeroshot_lane_v1_contract.py tests/test_prod_zeroshot_lane_v2_contract.py
```

### 5. G1 regression tier (E3 unchanged; 0 false PASS)

The run takes one geometry slot and about 9 minutes. The tier is the frozen-scanner-card DEV regression tier used for the overlay gate. Regenerate it if the scratch copy is gone, as for `overlay/g1`.

```
/private/tmp/claude-501/sherman_slot.sh env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend .venv/bin/python -m domains.production.engine.eval run \
  --tier-dir /private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/zsprod/e2_regression_card \
  --out <scratch>/g1_v3_live --ledger docs/production/runlog/freeze/LEDGER.json \
  --ledger-anchor cf01e709079c0fcc348758c5196884bff7d20f7836b88c7616883c0dcec67e67
```

**Pass criteria:**
- the regression verdict is PASS;
- no binding gate fails;
- G1.FALSE_PASS is 0 / 553;
- WRONG_OR_MIRRORED_NEVER_PASS is 0.

### 6. Real-part E2E (app copy, Playwright; the whole driver under the geometry lock)

```
/private/tmp/claude-501/sherman_slot.sh node scripts/production/run_e2e_real.mjs --run <YYYYMMDD>T-real-zs-overlay-v3
```

The driver builds a temporary app copy (C04 `--profile app`, the frontend built inside that copy, never `npm run build` in the repo).

**Pass criteria:**
- `steps.json` is 13 / 13 PASS;
- every job carries an overlay whose `lane.lane == "zs_lane_v3"`, with the v3 fingerprint and model sha;
- `overlay_compare` reports identical job, API, screen and PDF;
- no overlay creates or upgrades a PASS, and the banner verdict equals E3's state or a lowered REVIEW;
- no green in the overlay panel.

### 7. Optional repeat of the rehearsal

Repeat it with the live default (no environment variable needed after the flip):

```
.venv/bin/python -B docs/production/runlog/zeroshot_2026-10-25/v3/tools/repin_rehearsal.py <scratch>/rehearsal_live
```

The expected result is 0 false PASS.

### 8. Ledger

Append a freeze-ledger NOTE: "zero-shot overlay re-pinned to zs_lane_v3". It should bind:
- the changed files' sha256;
- the v3 pin sha;
- the G1 scorecard result;
- the E2E run id;
- the fresh-scan RESULTS seq;
- the engine fingerprints, unchanged.

Use `domains.production.engine.eval.freeze.append_entry(kind="NOTE")`. NOTE is used because a FREEZE-kind entry is the E4 engine freeze.

## Rollback

Set `DEFAULT_LANE = "zs_lane_v1"` and `ZEROSHOT_PIN_REL` back to the v1 pin, then rebuild the bundle.

The v1 pin, its model and its 60 pinned files are untouched by all of this; their verify result is `[]`.


## Revision 2 (2026-10-25)

The v3 pin is now **revision 2**:
- fingerprint `ab31d061c6231f5f7581f4cf03d451e77dbc5667e0193e66f3acac955e30a03d`;
- pin sha256 `115b70773fa20626ee8ead279e9c648013bd827871decb19374ab77092f17521`;
- 71 pinned files;
- `lane_params.budget.max_faces` 3 000 000.

The `e34e1ba7…` value in step 2 now reads `ab31d061…`. The rehearsal was repeated with the revision-2 pin: 50 / 50 OK, 0 false PASS, max 16.6 s and 2.58 GiB (`rev2/rehearsal/`). `DEFAULT_LANE` stays `zs_lane_v1`.


## Revision 3 (2026-10-26): research freeze only, the production pin is NOT re-pinned

zs_lane_v3 is re-frozen as **revision 3**. Only the budget step changed: `budget.py` (the density-invariant fixed-grid resampling) and `core.py` (which passes `budget.cell_mm`).
- Code fingerprint: `dc23a905bc2cf83a2fff2a4e1270ca654e95b8d3d6f2077db97c50559c85476f`.
- `config_v3.json` sha256: `8e002f3e9dac696fc6072d2d6a3abca6d964bd9704a25b54dd8f3b084b08b8b4`.
- See `rev3/FREEZE_REV3.md`.

**The production pin was deliberately left at revision 2.** That pin is `domains/production/zeroshot/assets/zs_lane_v3.pin.json`: sha256 `115b7077…`, fingerprint `ab31d061…`, `revision: 2`. It stays there until a separate re-pin event. Consequences until then:
- With `SHERMAN_ZS_LANE_PIN=zs_lane_v3` selected, `pin.verify` reports exactly `CODE:research/zeroshot/zs_lane_v3/budget.py` and `CODE:research/zeroshot/zs_lane_v3/core.py`. The zs step fails closed with PIN_MISMATCH, so no overlay is written. This is the designed behaviour.
- `test_prod_zeroshot_overlay_lane.py::test_pin_matches_the_frozen_research_config` and `::test_pin_verify_fails_closed_on_any_change` fail **only** with the v3 pin selected. With the default v1 pin they pass, and so do the other overlay and contract tests: 64 / 64.
- `pin.DEFAULT_LANE` stays `zs_lane_v1`. The live overlay is unaffected.
- `tests/test_prod_zs_lane_v3.py` now asserts this state:
  - the pin equals the archived revision-2 config (`rev3/rev2_archive/config_v3_rev2.json`);
  - it fails closed on exactly the two revision-3 files;
  - the archived revision-2 `budget_rev2.py` and `core_rev2.py` match the pin's hashes.

### The re-pin event (separate; owner decision)

This event can be combined with the ENGINE_IO-A5 re-pin, because both change the v3 fingerprint.

1. `make_pin.py`:
   - It now writes `lane_params.budget = {"max_faces": 3000000, "cell_mm": 0.3, ...}` and `revision: 3` from `config_v3.json`.
   - `lane_cli.py` needs no change. It already hands the pin's whole `budget` to `core.configure`, which reads `cell_mm`; the test `test_rev2_run_lane_v3_passes_the_pinned_budget` checks this.
2. In `tests/test_prod_zs_lane_v3.py`, switch `test_v3_pin_matches_config_and_verifies` and `test_v3_pin_fails_closed_on_the_second_model` back to comparing the pin with the live `config_v3.json` and the live files.
3. Repeat steps 2, 4 and 7 of this plan, plus the production rehearsal, with the revision-3 pin. Then add a ledger NOTE that binds the new pin's sha.

Until that event, the fresh-scan run uses the **research** CLI. That is `run_cohort.sh` with `--strict-pins`, which checks `config_v3.json`: revision 3 passes it, at 125 / 125 pins_ok on DEV. This is the PREREG primary arm.
