# 47959001 Current-Snapshot Count Context

This pass realigns `47959001` diagnostics to the current decomposition snapshot:

- Decomposition: `f1_known10_after_control_exact_tiebreak_20260426/47959001/decomposition_result.json`
- Scan path: `/Users/idant/Downloads/47959001_01PREL_scan1.ply`
- Current raw F1 owned regions: `1`
- Expected visible observed bends from prior manual context: `5`

The older `47959001_count_context_region_labels.json` and `47959001_junction_bend_arrangement.json` are preserved as historical diagnostics, but they were built against a stale flange-ID set containing `F20/F21`. Those IDs are not present in the current decomposition, so they are not valid direct rerun controls for the current branch.

## Current Results

Count-context summary:

- Output: `47959001_current_count_context_summary.json`
- Status: `owned_region_capacity_deficit`
- Region count: `1`
- Recovery deficit: `5` by current summary logic because the only raw region remains pending, not manually accepted.

Residual/interface support search:

- Output: `47959001_current_residual_interface_support_candidates.json`
- Status: `partial_candidate_pool`
- Candidate count: `148`
- Admissible candidate count: `1`
- Top admissible candidate: `RIC1 / F1-F5`

Local ridge trace:

- Output: `47959001_current_local_ridge_candidates.json`
- Status: `partial_ridge_signal`
- Ridge atom count: `48`
- Admissible ridge candidates: `1`

Junction arrangement:

- Output: `47959001_current_junction_bend_arrangement.json`
- Status: `underfit_arrangement`
- Best hypothesis: `H3_locked_0_plus_3_new`
- Best candidates: `JBL2 / F1-F5`, `JBL40 / F10-F5`, `JBL50 / F43-F5`
- Best candidates overlap heavily, so they do not explain four missing conventional bends.

Underfit ridge support split:

- Output: `47959001_current_underfit_ridge_support_splits.json`
- Status: `underfit_not_splittable_single_ridge_family`
- Source ridge atoms: `48`
- Candidate segments: `1`
- Admissible segments: `1`
- Ridge span: `92.216051 mm`
- Largest projection gap: `5.396338 mm`
- Split threshold: `10.1895 mm`
- Interpretation: the broad ridge is continuous along projection and does not naturally split into multiple support objects.

Strict scan-wide uncovered corridor search:

- Output: `47959001_current_uncovered_corridors_strict_excluding_owned_and_ridge.json`
- Render: `47959001_current_strict_uncovered_corridors_visual_check/uncovered_bend_corridors_top_candidates_1080p.png`
- Exclusions: current owned bend atoms plus `48` atoms from the broad local ridge
- Two-sided contact required: `true`
- Status: `no_uncovered_corridor_candidate`
- Candidate count: `186`
- Admissible candidates: `0`
- Visual verdict: top candidates are broad panel/edge bands, not separate bend supports.

Low-threshold scan-wide ridge search:

- Output: `47959001_current_scanwide_low_threshold_ridges_excluding_owned_and_ridge.json`
- Render: `47959001_scanwide_low_threshold_ridges_visual_check/ridge_candidates_top_1080p.png`
- Exclusions: current owned bend atoms plus the previous broad 48-atom ridge
- Score threshold: `0.48`
- Status: `partial_ridge_signal`
- Domain atoms: `1175`
- Ridge atoms: `793`
- Candidate ridges: `2`
- Admissible ridges: `1`
- Visual verdict: the admissible ridge is a huge 792-atom broad face/edge field, not a countable bend-line support.

Thin atom-edge transition search:

- Output: `47959001_current_transition_edge_candidates_excluding_owned_and_ridge.json`
- Render: `47959001_transition_edge_candidates_visual_check/transition_edge_candidates_top_1080p.png`
- Exclusions: current owned bend atoms plus the previous broad 48-atom ridge
- Normal jump threshold: `32°`
- Selected transition edges: `2414`
- Candidate components: `4`
- Admissible candidates: `1`
- Visual verdict: `TEC2` is a compact 8-edge local fragment between `F1/F5`; the dominant `TEC1` component is a huge 805-atom boundary/face flood. This does not recover the four missing bend-line supports.

Raw-point crease search below atom graph:

- Output: `47959001_raw_point_crease_candidates.json`
- Render: `47959001_raw_point_crease_candidates_top4_visual_check/raw_point_crease_candidates_top_1080p.png`
- Source PLY: `/Users/idant/Downloads/47959001_01PREL_scan1.ply`
- Sampled points: `43142`
- Candidate crease points: `1208`
- Candidate components: `33`
- Admissible line-like crease candidates: `4`
- Visual verdict: this is the first diagnostic that numerically matches the four missing supports after excluding the accepted bend. The top-4 render shows several plausible crease-line supports, but at least one candidate still appears partly scattered on panel/edge texture. Treat as a promising candidate-generation lane, not accepted F1 count evidence yet.

Raw-point to F1 bridge validation:

- Output: `47959001_raw_point_crease_bridge_validation.json`
- Render: `47959001_raw_point_crease_bridge_validation_visual_check/raw_point_crease_bridge_validation_1080p.png`
- Raw admissible candidates: `4`
- Bridge-safe candidates: `2`
- Safe candidates:
  - `RPC5`, selected pair `F11-F47`, score `3.092831`
  - `RPC6`, selected pair `F12-F45`, score `3.069415`
- Rejected candidates:
  - `RPC3`, rejected for corridor/contact failure against best pair `F1-F5`
  - `RPC4`, rejected for axis/corridor/contact failure against best pair `F43-F5`
- Visual verdict: bridge validation is conservative and useful. The safe green candidates sit on coherent side crease supports; the rejected red candidates are broader/scattered panel or edge texture. This is still diagnostic-only because it recovers two of four missing supports, not the full target.

## Visual Verdict

The 1080p arrangement render shows the selected lines clustered along one broad edge/ridge family. This is not a valid multi-bend recovery for the five-visible-bend target.

Render folder:

`/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/Status for Idan to test/feature_region_labels_20260427/47959001_current_arrangement_visual_check/`

## Decision

Current-snapshot `47959001` is a support-creation underfit case, not a raw-family suppression regression.

The raw-family suppression change should not be judged against the stale `F20/F21` arrangement. The correct current conclusion is:

- existing F1 snapshot is too sparse (`1` owned region),
- current residual/ridge diagnostics find only one broad support family,
- local arrangement cannot recover the missing four visible bends,
- projection-gap splitting also finds only one continuous support family,
- strict scan-wide corridor search finds no clean independent two-sided uncovered corridor after excluding the known raw support and broad ridge,
- low-threshold scan-wide ridge discovery floods into broad face/edge support and does not isolate missing bend lines,
- thin transition-edge tracing finds only one compact local fragment plus one huge boundary flood, not the missing set of countable bend lines,
- raw-point crease tracing below the atom graph produces four line-like candidates, which is the first promising recovery signal but still needs flange-pair/contact validation and cleaner rendering before promotion,
- raw-point bridge validation keeps two coherent crease supports and rejects two texture-like candidates, proving that bridge validation can reduce overcount risk but is not enough to recover the full 47959001 count yet,
- this remains offline diagnostic evidence only.

## Next Work

- Build a controlled support-birth experiment from bridge-safe raw creases only. Convert `RPC5/RPC6` into temporary diagnostic F1 support births, then rerun local arrangement/count recovery while keeping rejected raw candidates out.
- Investigate why `RPC3/RPC4` fail bridge validation: true texture rejection vs overly strict/flange-fragmented pair selection.
- Keep scan-quality/coverage gating active for cases where raw crease candidates remain scattered or unstable.
- Do not use stale flange-ID arrangements as regression controls.
- Use this current-snapshot file set for future `47959001` experiments.
