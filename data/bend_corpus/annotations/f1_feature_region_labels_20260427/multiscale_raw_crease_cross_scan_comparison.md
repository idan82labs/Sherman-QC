# Multiscale Raw-Crease Cross-Scan Comparison

Generated during the F1 feature-region diagnostic tranche.

## Purpose

Compare the same multiscale raw-point crease extraction and F1/flange-family bridge validation across:

- `47959001`: problematic partial/compact-neighbor case, human visible count `5`
- `49001000`: poor-coverage partial case, human visible count `5`
- `37361005`: cleaner compact known-count case, human count `4`

The question is whether the `47959001` failure is a global raw-crease threshold/support-creation failure or a scan/coverage-specific failure.

## Results

| scan | human target | raw admissible before dedupe | deduped raw candidates | family bridge-safe | visual verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `47959001` | `5` visible observed | `7` | `5` | `2` | Underfit. Only two safe supports; other candidates fail corridor/axis/two-sided validation. |
| `49001000` | `5` visible observed | `13` | `6` | `2` | Similar underfit pattern. Safe supports exist, but most candidates are weak, off-corridor, or not line-like enough. |
| `37361005` | `4` conventional | `16` | `6` | `6` | Raw support is available; failure mode is overfragmentation/edge-counting, not missing raw support. |

## Candidate-Level Notes

`47959001`:

- Safe: `STANDARD_RPC5`, `STANDARD_RPC6`
- Rejected: `FINE_RPC4`, `STANDARD_RPC3`, `STANDARD_RPC4`
- Main rejection reasons: axis mismatch, corridor far from candidate, weak/two-sided flange support, low flange-pair score

`49001000`:

- Safe: `COARSE_RPC3`, `COARSE_RPC2`
- Rejected: `FINE_RPC3`, `FINE_RPC1`, `COARSE_RPC4`, `FINE_RPC7`
- Main rejection reasons: not line-like enough, outside corridor, weak two-sided support, low flange-pair score

`37361005`:

- Safe: all six deduped candidates
- Visual issue: safe candidates trace perimeter/edge bend structure, but there are six candidates against a four-bend human target. This needs duplicate/edge-family arrangement logic, not more raw support discovery.

## Interpretation

The multiscale raw-point layer is not globally starved. On `37361005`, it finds strong bridge-safe crease evidence. The `47959001` and `49001000` pattern is different: raw extraction produces some candidates, but bridge validation keeps only two, and rejected candidates fail geometric support tests rather than arbitrary thresholds.

This supports the current decision:

- Do not relax bridge gates to force `47959001` or `49001000` to the manual count.
- Add scan-quality / partial-observed-support gating for cases where raw and bridge diagnostics stall at partial recovery.
- For cleaner compact parts such as `37361005`, focus next on sparse arrangement / duplicate suppression so six bridge-safe edges reduce to the four countable conventional bends.

## Review Artifacts

- `47959001`: `47959001_multiscale_raw_point_crease_family_validation_visual_check/raw_point_crease_bridge_validation_1080p.png`
- `49001000`: `49001000_multiscale_raw_point_crease_family_validation_visual_check/raw_point_crease_bridge_validation_1080p.png`
- `37361005`: `37361005_multiscale_raw_point_crease_family_validation_visual_check/raw_point_crease_bridge_validation_1080p.png`

## Next Work

1. Add a scan-support quality gate for partial/poor scans:
   - raw candidate count,
   - bridge-safe fraction,
   - rejected-candidate reasons,
   - F1 coverage/fragmentation,
   - partial-scan route.
2. Add a local sparse arrangement diagnostic for `37361005`:
   - merge same-family/same-perimeter duplicate support,
   - prefer maximal direct bend edges,
   - reject non-countable perimeter/edge fragments.
3. Keep `47959001` and `49001000` as abstain/limited-observed-support canaries until additional visible support is found by a new evidence source, not by weaker gates.
