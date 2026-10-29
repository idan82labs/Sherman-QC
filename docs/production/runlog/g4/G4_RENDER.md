# G4 render step (wave 4): blind review packets with secret canaries (2026-10-19)

Plan §5 G4 and §10 amendment 8, rendered with the E4a renderer (`domains/production/engine/eval/render.py`,
`g4.build_review_packet`). This step renders only. Review, tie-break and scoring are later workflow steps.
**Reviewers must not open this document's sealed files** (`*_SEALED*`). Nothing a reviewer opens (images,
`review_manifest.json`, `scan_manifest.json`, this document) names a canary or carries a metric.

## What was rendered

| Arm | Engine | Scans | Close-ups | Global images | Packet | Sealed key |
|---|---|---|---|---|---|---|
| **E3 (gated)** | final A1 engine `e3-1.9.0`, default config `cb14d5ca…` | 52 = 26 physical dev + 26 synthetic | 399 | 312 | `packet_e3/` | `canary_key_SEALED.json` |
| LEGACY (visual baseline, informational) | unchanged `backend/bend_analysis_worker.py` | 22 physical dev | 279 | 97 | `packet_legacy/` | `canary_key_legacy_SEALED.json` |

- **Final engine provenance.** The A1 loop stopped after iteration 7 (plateau rule). Today's code fingerprints
  (`a1/iter_7/tools/fp.py`) equal iteration 7's final ones: engine `6297d8e3…`, compile `6147b43c…`,
  eval `8aa6860c…`, contracts `78e31787…`. The E3 outputs rendered here are iteration 7's MEASURE outputs:
  physical from `scratchpad/a1_iter7m/phys/physical/E3/`, synthetic from the G1 regression run
  `a1_iter7m/g1_regression/`. Every `result.json` was checked to be `e3-1.9.0` with config `cb14d5ca…`
  (`tools/build_g4.py`). The engine was not re-run and not changed.
- **Physical dev scans (26).** These are every A1 case (`freeze/cases_A1.json`: G2_COMPLETE, G3_STAGED and
  BROAD; DEV-CHECK is excluded by construction and was never read) that has an E3 output.
  - 10 are incomplete per the frozen G3 truth table (a NOT_FORMED row) and 16 are complete.
  - Not renderable: `pc_b4719db3dd69` (47479000). Its recipe compile is BLOCKED, so it has no E3 output. It stays
    in the denominators elsewhere.
- **Synthetic sample (26 ≥ 20, 16 incomplete).** Chosen by scenario only, before any rendering:
  - per family (LB, UC, ZS, HT, CR, TD, PN): the COMPLETE_NOMINAL `-000`, the STAGE_1 `-003` and the STAGE_K
    `-005` (LB-005 is PARTIAL_BEND);
  - plus LB-002 and HT-002 (stage-0 blanks), TD-006 (missing bend), UC-006 (order swap) and UCS-000 (the
    symmetric U channel, AMBIGUOUS close-ups);
  - "incomplete" means the truth formed count is below the planned count.
- **E3 close-up classes:** MEASURED 292, NOT_YET_FORMED 102, AMBIGUOUS_E1_JUSTIFIED 4, INVALID_INSPECTION 1.
- **Legacy arm.** The unchanged worker was re-run serially under the lock on the 23 frozen LEGACY_LISTS cases.
  - The run went into a fresh, unsealed scratch work dir. The frozen E4b run outputs sit under a `builder_SEALED`
    path, which this step may not read.
  - All 23 statuses equal the frozen E4b statuses (15 OK, 8 WORKER_ERROR), and every per-bend "line measured"
    flag equals the frozen `legacy_control/*/legacy_case.json` (`legacy_rerun_status.json`). The rerun reproduces
    the frozen arm.
  - 22 of the 23 are rendered (`pc_b4719db3dd69` compile BLOCKED). The 8 WORKER_ERROR scans render as
    INVALID_INSPECTION (no aligned scan).
  - The legacy arm measures few lines (22 over the 15 OK scans), so most legacy close-ups show only the nominal
    line.

## Render protocol (amendment 8, as implemented by E4a)

- **Global views per scan:** ISO, TOP, FRONT, SIDE, plus DIMENSIONS (arrows without values) and HEATMAP.
- **Close-ups:** one per bend (tiled per 150 mm) at exactly **10 px/mm** (≥ 8), each with a **10 mm scale bar**.
  - **SCAN panel:** the measured line **solid on the outer apex**, the OML sharp **dashed**, the nominal
    **grey dashed**, and points coloured by segment.
  - **RECIPE CAD panel:** the stage model from the **same camera, window and scale**, labelled from the recipe.
  - **Mid-span SECTION:** shows the flange-length arrows to the fitted edge.
- **Titles and file names** are opaque (`review scan scan_xxxxxxxxxx`, `img_<12 hex>.png`). No case id, part
  number, metric or canary marker appears in any image, in the PNG metadata (stripped by the palette
  re-encode) or in any manifest.
- **Canaries:** at least 10 % of each packet's close-ups (rounded up) are secretly corrupted, chosen from a
  secret OS salt (`salt_source = OS_RANDOM`).
  - **LINE_SHIFT** moves the solid and dashed lines and the section markers across the bend by |s| ∈
    [max(8, LANDS tolerance + 3), 15] mm.
  - **LABEL_SWAP** exchanges two bends' labels throughout the SCAN panel; only the truthful RECIPE CAD panel
    reveals it.
  - Ambiguous-pose close-ups are never canaries.
  - The identities and counts are only in the sealed keys.
- **Images** were palette re-encoded (192 colours, as in E4b's run logs) **before** each sha256 was taken, so
  the manifests, the keys and the files agree.
- **Integrity check:** `evaluate_g4` on each packet with an empty verdict file reports no problems (key
  commitment and every image sha256 verified) and the synthetic sample requirement met. The verdict is
  NOT_EVALUABLE only because no verdicts exist yet.

## Review items

Each reviewer gets one scan directory: `review_e3/<scan_ref>/`. It holds hard links to the packet images and
`scan_manifest.json`, which lists per image its id, sha256, kind, displayed subject `bend_uid` and the questions.

- **Engine arm (returned for review):** `items_e3.json`, 52 scans.
- **Legacy arm (listed separately, the visual baseline, not gated):** `items_legacy.json`, 22 scans,
  directories `review_legacy/<scan_ref>/`.

Close-ups per scan (E3): scan_04a7874fc9 2, scan_106dbaff8c 4, scan_13847a5c42 1, scan_151dc615d7 16,
scan_1801c48120 2, scan_24698f2bcd 5, scan_29f000f60c 2, scan_2dda3382c0 4, scan_2e7aec96f0 1, scan_2ffcebb3bd 4,
scan_3db9fefd2b 10, scan_3e5559b7fb 4, scan_41499ab800 5, scan_41ae07e603 4, scan_459f38302d 30, scan_4a6a53e4bd 2,
scan_4e9713bba4 1, scan_50d26f3119 1, scan_53140dc9f8 12, scan_535db24bfa 1, scan_54eb817529 5, scan_5d117c48cb 4,
scan_5e64d10d30 4, scan_631aa9d2b1 2, scan_6b7cdf7fbd 3, scan_6bf59282c3 5, scan_703ad8e226 1, scan_711b64b588 7,
scan_84a76b9aa5 8, scan_8568865f8a 1, scan_8c164de3ab 3, scan_8d822b00d1 8, scan_9018e173b9 3, scan_98388ea8bd 4,
scan_b13e7c6f3e 29, scan_c02f03f322 3, scan_c57406c637 18, scan_c71ffbb05a 18, scan_c8793f57ff 5, scan_cbd42e95b1 4,
scan_cec889f044 10, scan_d143597775 16, scan_d1f6be9fb9 16, scan_d32e5a93ff 4, scan_d69c0e5484 3, scan_e3359f02b4 16,
scan_e4e4af5689 4, scan_e84fbd95b4 18, scan_e857143822 30, scan_ebb5510d67 16, scan_f84750da5b 18, scan_f873025d01 2.

Close-ups per scan (LEGACY): scan_0019d86ccd 5, scan_1c2e11bc3e 18, scan_2b239108f6 18, scan_3471a86382 30,
scan_7100b6dffe 16, scan_74c680102c 18, scan_79ef50a5c0 12, scan_7e4df62126 1, scan_8831692f15 16, scan_94bfeff3c2 1,
scan_951f77f48b 1, scan_99680366c4 3, scan_9f40cc4e3c 16, scan_a7e7f6aef8 30, scan_b718a8a071 16, scan_c277746009 9,
scan_d7808ec7f1 5, scan_d7c3865366 18, scan_d8f841611e 5, scan_e7643c858c 1, scan_e8a7ddc4a6 10, scan_fab0059574 30.

Scan references map to cases only in the sealed keys (`scans`).

## For the scorer (after the reviews; reviewers never do this)

`evaluate_g4` reads `<sealed_dir>/canary_key.json`. To score without copying the key, point a temporary
directory's `canary_key.json` at it with a symlink:

```
mkdir -p $TMPDIR/g4s && ln -sf $PWD/docs/production/runlog/g4/canary_key_SEALED.json $TMPDIR/g4s/canary_key.json
PYTHONPATH=.:backend .venv/bin/python -m domains.production.engine.eval g4 \
  --packet docs/production/runlog/g4/packet_e3 --sealed $TMPDIR/g4s \
  --verdicts docs/production/runlog/g4/g4_verdicts.jsonl --out docs/production/runlog/g4/g4_gate_report.json
```

- Verdict lines are `G4VerdictV1`: `image_id`, `image_sha256`, `bend_uid` exactly as listed in the manifest
  (the *displayed* subject), `reviewer_id`, `role` PRIMARY|TIEBREAK, `lands`, `tag_ok`, `unknown_justified`
  only for no-line close-ups, `reason`, `recorded_at`.
- Global images need no verdict.
- The review counts only if every primary reviewer flagged every canary.
- The legacy packet is scored the same way with `canary_key_legacy_SEALED.json` and
  `--no-synthetic-requirement`. Its rates are the visual baseline, never a gate.

## Context: final engine vs the bar (A1 iteration 7 measurement, same scans; not re-measured here)

| Same scans | New engine e3-1.9.0 | Best F1 lane (the bar) |
|---|---|---|
| Gold11 relabel (gold of record), 9 A1 Gold11 cases | F1 0.987 | F1 0.547 |
| Human Gold11 (historical diagnostic only) | F1 0.391 | F1 0.473 |
| BROAD frozen labels, 26 cases (label-capped, see a1/OWNER_DIRECTION.md) | micro 0.702 / family-macro 0.817 | micro 0.508 / family-macro 0.543 |
| GOLD_SET_V1 | micro 0.990 / family-macro 0.974 | n/a here |
| Per-bend correctness given CAD | 183/203 = 0.901, 3 false-PASS proxies | n/a |

The shipped runtime engine is not a comparator. The legacy arm above is the visual-baseline control only.

## Files

- `packet_e3/`: `images/` and `review_manifest.json` (reviewer packet, E3).
- `packet_legacy/`: the same layout for the LEGACY packet.
- `review_e3/<scan_ref>/` and `review_legacy/<scan_ref>/`: per-scan review directories (hard links).
- `items_e3.json`, `items_legacy.json`: the review item lists.
- `legacy_rerun_status.json`: the legacy rerun statuses against the frozen E4b statuses.
- `canary_key_SEALED.json`, `canary_key_legacy_SEALED.json`: **sealed**, mode 0400, scorer only.
- `tools/build_g4.py`, `tools/run.sh`: how this was built. Every heavy step ran serially under
  `/private/tmp/claude-501/sherman_geom.lock` with `OMP_NUM_THREADS=1`.
  - Peak RSS: E3 packet 1.9 GiB (281 s), legacy rerun 2.7 GiB (448 s), legacy packet 0.85 GiB (94 s).
