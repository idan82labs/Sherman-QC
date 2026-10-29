# V5 released physical-input derived geometry review — 2026-09-30

## Disposition

**ACCEPT_DERIVED_DISCRETE_GEOMETRY_LEDGER_WITH_LIMITATIONS.** The three released
`AVAILABLE` outputs are internally consistent with their native-evidence oracle
receipts and pass the pure V5 preview contract. No blocker was found in the
reviewed lineage, discrete metric area, orientation, hole topology, seam, or
finite bend-location ledgers. This is green for the separately authorized local
preview/visual QA step, not a manufacturing-ready or physical-accuracy claim.

The reviewer authored the geometry oracle and oracle bridge. Consequently this
is an explicit derived-receipt and contract consistency review, **not an
independent self-source audit**. The non-author source audit and frozen PRE gate
are separate records. Browser visual QA is also separate and is not certified
by this document.

## Scope and custody

Campaign root:
`artifacts/vendor-cad-upload-unfold-five-part-paired-v2-2026-09-30-attempt1`.

The accepted `POST_RUN_AUDIT.json` has SHA-256:
`065109230af699521efc198712fb4cf2ff7254b528738781eb92e14cd64e6ec4`.
Its decision is
`ACCEPT_CONSUMED_DEVELOPMENT_PROCESS_AND_PROVENANCE_WITH_LIMITATIONS`.
The nine reviewed artifacts below are explicitly named in its 118-artifact
release ledger. Every reviewed byte hash was checked against that ledger.

Only `GEOMETRY_ORACLE.json`, `FLAT_PREVIEW.json`, and `CANDIDATES.json` for
case-01, case-04, and case-05 were reviewed, plus the accepted POST record.
No raw CAD was opened, no native/OCP calculation or unfolding was rerun, and no
geometry was regenerated. Pure contract validation and combinatorial ledger
checks do not establish facts missing from the captured evidence.

The parent campaign retains all five selected cases and all ten paired arms;
the accepted POST reports control availability 2/5 and challenger availability
3/5. This review covers the three available challengers only and does not remove
the other two cases from any campaign denominator. There is no F1 result or
independent manufacturing bend-count truth in this campaign.

## Executed checks

- All nine released hashes and candidate/preview source hashes agree with their
  frozen custody records.
- `vendor_unfold_contract_v4.validate_preview` passes each of the three payloads,
  using the `candidates` list within the candidate-file envelope. This is a pure
  contract check; it imports no native CAD geometry.
- Each external oracle JSON equals the preview's embedded geometry oracle
  exactly. All three state `VERIFIED_UNDER_DECLARED_GEOMETRY_MODEL`, with model
  `DISCRETE_NATIVE_METRIC_CHART` and bridge model
  `EXACT_NATIVE_FACE_EDGE_POLYGON_AND_V5_SOURCE_SLICES`.
- The oracle face IDs, selected native face IDs and exact source-slice face IDs
  match. Face and triangle counts are consistent with the retained output.
- Global source triangle parity is consistent across all selected source faces,
  rather than mixed across native faces; exported flat triangles have positive-Z
  winding under the contract. A parity of -1 denotes a consistent global reversal,
  not loss of face ownership or a mixed-winding result.
- Every face reports a passed discrete native metric polygon-area check. Its
  isometry residual is finite and below the declared fixed 1e-5 mm roundoff
  limit; flat versus native metric polygon area differs by less than the fixed
  1e-8 mm² allowance. Summed flat face area agrees with the total ledger area.
- Boundary-cycle/Euler hole checks remain present per original native face.
  Their reported hole cycles are retained; they are not a newly established
  manufacturing hole-count truth.
- All selected native shared-edge seams are accounted for, with unique edge
  identities, two selected owners and at least two matching seam nodes. The
  seam graph connects every selected native face. The ledger explicitly records
  `all_selected_shared_native_edges_checked: true`.
- Candidate count, flat bend-line count, oracle bend count and finite-width proof
  count agree. Finite bend endpoints and seams satisfy the declared 1e-5 mm
  correspondence roundoff limit. Original candidate identity/source ownership
  remains bound through the contract and bridge receipt.
- Every selected face has a separate numerical native BRep area observation.
  No continuous trim-area allowance was manufactured from the observed numerical
  difference: continuous reference/error-bound fields remain absent, and all
  continuous checks remain explicitly unestablished.

## Per-case results

| Case / SKU | Native faces | Selected faces | Bend markers | Selected shared seams | Flat triangles | Flat metric polygon area (mm²) | Source triangle parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 / 37361005 | 88 | 9 | 4 | 8 | 1,600 | 67,048.52827471809 | -1 |
| 04 / 38172000 | 164 | 25 | 12 | 24 | 2,444 | 76,729.24588447026 | +1 |
| 05 / 10839000 | 164 | 25 | 12 | 24 | 2,734 | 92,995.17614641290 | +1 |

Totals: **59 selected native faces, 28 bend markers, 56 shared native edges and
6,778 flat triangles** across the three reviewed available outputs.

| Case | Max face isometry residual (mm) | Max discrete area difference (mm²) | Max seam residual (mm) | Max finite bend-line residual (mm) | Max flat-vs-numerical-native-area difference (mm²) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01 | 8.6764e-13 | 6.1334e-11 | 5.6843e-14 | 6.4811e-13 | 0.01898107458 |
| 04 | 8.1053e-13 | 1.4552e-11 | 8.5265e-14 | 8.6701e-13 | 0.02624706375 |
| 05 | 5.9939e-13 | 6.5768e-11 | 8.9877e-14 | 6.2592e-13 | 0.05086669805 |

These small metric residuals demonstrate computational consistency within the
declared discrete model; **they are not physical accuracy or calibrated
uncertainty**. The numerical native-area differences in the final column are
observations, not accepted continuous-area error bounds.

Hole-cycle ledger (native face ID: extra boundary cycles):

- Case 01: `10:4, 18:6, 51:4, 69:3`.
- Case 04: `12:3, 19:2, 54:2, 116:2, 137:3, 147:2`.
- Case 05: `21:3, 122:1, 130:4, 143:5, 162:3`.

## Exact reviewed byte identities

All paths in this table are relative to the campaign root above.

| Artifact | SHA-256 |
| --- | --- |
| `case-01-37361005/CANDIDATES.json` | `8048f26ca691584d8e6f6eddd12c384478dc5e7e2329465449ef97835bc57820` |
| `case-01-37361005/FLAT_PREVIEW.json` | `4f8364f372094015e5581c4dcecdd1030f59c1189c477dc82cdebf5e08bbf9ae` |
| `case-01-37361005/GEOMETRY_ORACLE.json` | `11cd175357d68f0b35f56f0ce50743b74be65ef932b9033492186c2253857f71` |
| `case-04-38172000/CANDIDATES.json` | `2f0b5a04acf6e7e086f08e2d6b97770cd5ad789e85afa6957a81f522e4385a83` |
| `case-04-38172000/FLAT_PREVIEW.json` | `4d0179bc980ab8495509e9ef91551661ea66e736cc7e0708b17c3caa459c37bf` |
| `case-04-38172000/GEOMETRY_ORACLE.json` | `4202e9b2545afbade4ec22cf4f5bad5cf79a7f83833a61f0285eb49db4a6f1ee` |
| `case-05-10839000/CANDIDATES.json` | `1528469ca24955ec2a084966f23b98a0b89556f9e09f944a32b1d57682a8192b` |
| `case-05-10839000/FLAT_PREVIEW.json` | `4418eff4fb8a99ca6993eddad3bfb174857e9cb2539e3bce5a9b9e31ce3998e7` |
| `case-05-10839000/GEOMETRY_ORACLE.json` | `5f067cba9f53666675e7cfda7e3117239413b6c06cddc6d10d7a7ccaf27227a7` |

Input source hashes were compared as identities in these released records only;
the input files themselves were not accessed:

- Case 01: `81c428a0a55c531d67fcd68d74cac905c8b7b00e88277d836a932e954c7d636e`.
- Case 04: `4d512b5c565ab0e9e4d02e73fcaafd164ae82bcd977eba69198c8b0356f5644f`.
- Case 05: `ea3e4820dff568916164c79417925ead79f307fba8aff11c7b3aeda7375e0b37`.

## Remaining gates and limitations

All three outputs retain `continuous_area_check: NOT_ESTABLISHED`,
`continuous_trim_proof: false`, and `manufacturing_ready: false`.
Continuous trim error, neutral-axis/K-factor material behaviour, manufacturing
dimensions, physical calibration, physical bend-count truth and unseen-family
generalization are not established by these receipts. Numerical BRep quadrature
and its convergence observations do not close those gates.

The accepted five-case result is consumed-development preview availability, not
a physical engine F1 or an unbiased larger-corpus score. This review grants no
new raw-data access, native execution, deployment or manufacturing authority.
The next in-scope gate is separate browser visual integrity QA on these exact
released outputs, with all unavailable cases and their causes retained.
