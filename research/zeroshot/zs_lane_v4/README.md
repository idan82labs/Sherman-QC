# zs_lane_v4

Research candidate for the zero-shot (no-CAD) bend-line detector. Production stays on zs_lane_v1. Promotion is the owner's decision after the pre-registered fresh-scan test (`docs/production/runlog/zs_v4/PREREG_FRESH.md`).

The development record (protocol, pool, cache, results, design log, freeze) is in `docs/production/runlog/zs_v4/`. Start with `NOTES.md` there.

## What runs (`pipeline.run`)

1. **Detection: zs_lane_v3 revision 3's stages** (`core.process`). These are the frozen modules, imported read-only:
   - the density-invariant face budget;
   - Z2 with the v2 fallback;
   - Z3 axis-family slicing with `fallback_frac_only`, through the exact bounded merge;
   - pooling;
   - the D1 outer-apex snap of every candidate;
   - the v2 evidence: station sweep, interruption runs and pair plate continuity.

   With the same thickness, the candidates, features and evidence are exactly v3's (`tools/validate_cache.py`).
2. **Support profiles** (`profiles.py`). Per 1 mm station along each candidate, extended over its pooled Z2 line, the frozen v2 station tests are recorded:
   - bend-zone material;
   - flank A and flank B plate;
   - normal sweep;
   - the tight apex-touch test.
3. **Features** (`features.py`):
   - the 26 v1 pooled features;
   - the 6 v2 sweep summaries;
   - support-profile features: touch fraction and the longest unsupported runs, on the candidate and on the extension its Z2 line proposes;
   - interruption counts;
   - neighbourhood: collinear segments, duplicates, connected or disconnected pairs;
   - the Z2 line's own attributes;
   - scan metadata: t, mesh spacing, counts, area.
4. **Optional, set by the frozen config:**
   - extent hypotheses (`hypotheses.py`): the Z2 extent, the union with it, joins and split pieces, each competing with its candidate;
   - the hem generator (`hem.py`): the two parallel skins and the crown of a 180° fold, which Z3 and Z2 cannot emit (paper `th:cor-floors` iv);
   - the Z3 re-slice stability check (`stability.py`).
5. **Selector:** one GBM (`model.py`). It is followed by `select.select_h` (hypotheses compete greedily by probability; a hypothesis may not swallow another selected candidate) or by `select.select`:
   - v2's sweep gate and plate-continuity join;
   - EXT in mode `off`, `v3` or `safe`;
   - optional matcher-equivalence NMS;
   - hem lines.

There is no part-specific logic: no CAD, labels, file names or part identities. The lane reads only the scan bytes, staged as `scan.<ext>`. The code is deterministic.

## Why `safe` EXT

v3's lanced-tab failure on `gb_68cd06` came from EXT. A 52.5 mm candidate absorbed a 413 mm Z2 line that runs along a row of 14 separate tabs. EXT extended the candidate to 413 mm, and EXT's dedupe then dropped every tab line that the extended line now "duplicated".

`safe` keeps v3's extension rule. It refuses an extension that would swallow another output line: separate segments on one line stay separate unless the plate-continuity join merges them. Test: `tests/test_zs_v4_components.py`.

## Files

| File | Role |
|---|---|
| `core.py` | detection stages, profiles, metadata |
| `profiles.py` | compact support profiles |
| `features.py` | selector features |
| `hypotheses.py` | extent hypotheses and their features |
| `select.py` | selection and post-processing (`select`, `select_h`) |
| `hem.py` | hem generator |
| `stability.py` | Z3 re-slice stability |
| `pipeline.py` | end-to-end runtime |
| `model.py` | GBM wrapper (pickled) |
| `lane.py` | CLI (`--strict-pins`) |
| `fingerprint.py` | code fingerprint |
| `freeze_config.py` | final fit on the whole pool and `config_v4.json` |
| `make_pin.py` | the pin file |
| `evalkit.py`, `lofo.py`, `train.py`, `variants.py`, `baselines.py`, `cache_io.py` | evaluation side: pool, frozen matcher, nested LOFO, arms |
