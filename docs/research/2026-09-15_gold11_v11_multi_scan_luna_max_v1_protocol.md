# Gold11 V11 multi-scan Luna-max healing protocol — 2026-09-15

This is a separately versioned consumed-development evaluation.  The frozen
Gold11 V11 attempt remains terminal and is not reopened, edited, rescored, or
used as a mutable input cache.  The first implementation probe (before this
protocol was written) is diagnostic only; the run recorded against this
protocol is the reproducible score claim for this candidate.

## Scope and custody

- Exactly the September 9 reviewed eleven scans, six CAD files, and unchanged
  71-line reference snapshot are in scope.  The case authority is the literal
  `CASES` assignment in `scripts/review_gold11_inputs_v1.py`, SHA-256
  `41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955`.
- The catalog is byte-bound to
  `4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36` and
  the selected reference mapping to
  `d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc`.
- No new email parts, held-outs, unrelated data, hardware, deployment, or
  external services are accessed.  Original data, historical scores, and the
  terminal V11 artifacts remain unchanged.

## Candidate and source freeze

The candidate is `V11_TOPOLOGY_AXIS_RAW_SCAN_SUPPORT_RECOVERY_V2_PROVISIONAL_POSE_HEALING`:

| source | SHA-256 |
| --- | --- |
| `backend/cad_guided_v11_topology_recovery_v2.py` | `f745c3f8471971e2ccae95d319aff174857f41ad1db88ea8c9d0f302c3f0c413` |
| `scripts/run_gold11_v11_multi_scan_luna_max_v1.py` | `07ff813521c224a321c89692f4c43b7d64703311f8050ca99739fe1152d3fa54` |
| `backend/cad_guided_v11_topology_recovery_v1.py` | `2e7178213fa8075a76488ceee700c8666b24e0f7353f4babb80c447c97accb4d` |
| `backend/cad_guided_mesh_input_v1.py` | `c4d936e2f60837d7c5221c9a46b64abd68c4ccee01ea72e9fd3c4050bde47917` |
| `scripts/extract_step_topology_bend_axes.py` | `f089f71466e0f9bba47139c06efdf8bf651e7707fb05c78ca2195f06146b1314` |
| `scripts/build_additional_line_label_review.py` | `8549157ef23b9361c1a746391b03163dca6fa8b53dcda5f3892de8ec56456ae6` |
| `scripts/evaluate_f1_against_manual_gold.py` | `1e9420558b91f9b136fb32ed012b56c64cdaeecf7599ee1ef9fe336915d75c62` |
| `scripts/gold11_admission_geometry.py` | `cbff410bc217d8835f0519fff5b4107b7d1ebd120960cb8e2d6ac21954964771` |

The V2 seam preserves an accepted V11 context pose when one exists.  If none
exists, it may select a retained V11 registration candidate only when its
published finite-distance p95 upper bound is at most 2.0 mm and its represented
scan-coverage lower bound is at least 0.55.  Selection is ordered by p95, RMS,
descending coverage, and published candidate index.  A rejected context is
never relabelled accepted.  Native STEP axes are finite topology proposals;
the raw scan supplies all emitted endpoints and support statistics.

Fixed numerical policy: 50,000 engine rows, 3.0 mm support radius, 201 line
samples, the existing 8-MiB input preparation limits, one process at a time,
`OMP_NUM_THREADS=1`, and no threshold or scorer changes.  A malformed or
unsupported case is retained as an explicit abstention; no fallback pose is
fabricated and no case is retried within the run.

Gold is loaded only after all eleven inference attempts have returned.  It is
used solely by the unchanged one-to-one minimum-cost matcher.  Precision,
recall, F1, and bend-count capture are aggregated over all 71 references;
unavailable cases remain in that denominator.

## Promotion decision

This is consumed-development evidence.  It cannot establish unseen-family
generalization, calibrated uncertainty, manufacturing conformance, historical
comparability, or floor release.  Promotion requires an independent audit,
multi-part replay, adversarial noise/partial-coverage tests, and a maintained
engine integration behind an explicit controlled path.  A numeric score below
or above the historical 0.4580152672 is not a claim of beating it while
historical comparability remains unestablished.
