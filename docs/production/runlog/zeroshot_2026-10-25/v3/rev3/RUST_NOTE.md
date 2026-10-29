# zs_lane_v3 revision 3: what changed in the budget step (for the Rust port)

This note is for `research/zeroshot/zs_rust`. That crate was not edited by this revision. Its `src/budget.rs` still ports revision 2 (`decimate(v, f, max_faces)`, a geometric ladder from the mean edge length).

**Scope of the change.** Only the budget step changed: `research/zeroshot/zs_lane_v3/budget.py`, plus one argument passed from `core.py`. These are unchanged:
- the trigger: `max_faces` = 3 000 000, read from the file header exactly as in revision 2;
- the in-budget path: the scan is used unchanged;
- the point-cloud path: not budgeted;
- the output file: a binary PLY written by open3d, with float64 xyz and, if the input had vertex normals, float64 normals;
- every stage after the budget.

The reference is `budget.py` at code fingerprint `dc23a905…` (sha256 of the file: see `config_v3.json`, `code_fingerprint.files`). The steps below are in the order Python runs them. All arithmetic is float64 and all integer keys are int64, unless stated.

## 0. Config

`config_v3.json` → `budget` = `{"max_faces": 3000000, "cell_mm": 0.30, "rule": …}`.
- `core.configure` now also reads `budget.cell_mm` (`core.BUDGET_CELL_MM`).
- `budget_scan(path, max_faces, workdir, cell_mm)` falls back to the module constant `CELL_MM = 0.30` when `cell_mm` is `None`.
- The production `lane_cli` passes the pin's whole `budget` dict to `core.configure`, so a revision-3 pin needs no code change there.

Constants: `CELL_MM = 0.30`, `LADDER = 1.12`, `MAX_STEPS = 24`, `CHUNK = 2_000_000`, `REG = 0.05`, `SKIN_COS = -0.5`.

Removed constants (revision 2 only): the per-vertex normal pass, the 6-bin axis key, the mean-edge start and the `REG`-towards-cluster-mean vertex clustering. Nothing of revision 2's `decimate` survives except the face dedupe rule (step 4e) and the PLY write.

## 1. Load

Same as revision 2:
- `V`: float64 (n×3), from open3d.
- `F`: int32 triangles (an int64 copy only if `len(V) >= 2^31`).
- File normals: open3d `vertex_normals` as float64, only if `has_vertex_normals()` and their count equals `len(V)`. Otherwise there are none.

## 2. Crossings: `crossings(V, F, h, file_normals, count_only)`

1. `g = floor(V / h)` as int64, per vertex and axis. The grid is absolute: there is no origin offset.
2. Axis permutation `PERM = {0: (1, 2), 1: (2, 0), 2: (0, 1)}`, i.e. (b, c) for grid lines parallel to axis a.
3. Loop order: **for a in 0, 1, 2 → for each chunk of `CHUNK` faces in face order → for ob in 1..max(sb) → for oc in 1..max(sc)**.
   - Within a chunk:
     - `gb = g[f, b]` and `gc = g[f, c]`, each 3 columns;
     - `lo_b, hi_b` = row min/max of `gb`, and likewise `lo_c, hi_c` for `gc`.
   - Candidates are the faces with `hi_b > lo_b` **and** `hi_c > lo_c`, using strict inequality.
   - With `sb = hi_b − lo_b` and `sc = hi_c − lo_c`, a candidate takes part in offset (ob, oc) iff `sb >= ob && sc >= oc`.
   - The lattice point is `B = lo_b + ob`, `C = lo_c + oc`, `qb = B·h`, `qc = C·h`. That is an int64 multiplied by an f64; the int is converted to f64 first, as numpy does.
4. Barycentrics. Evaluate them exactly in this form so the results are bit-identical:
   ```
   x0b=p0[b]; x0c=p0[c]; e1b=p1[b]-x0b; e1c=p1[c]-x0c; e2b=p2[b]-x0b; e2c=p2[c]-x0c
   den = e1b*e2c - e2b*e1c
   ok  = |den| > 1e-18            ;  den := ok ? den : 1.0
   l1  = ((qb-x0b)*e2c - e2b*(qc-x0c)) / den
   l2  = (e1b*(qc-x0c) - (qb-x0b)*e1c) / den
   l0  = 1.0 - l1 - l2
   ok &= l0 >= 0 && l1 >= 0 && l2 >= 0      (inclusive: a lattice point on a shared edge is found twice)
   ```
5. With `count_only`, the function returns just the number of accepted crossings.
6. Otherwise, per accepted crossing:
   - Point: `P = l0*p0 + l1*p1 + l2*p2`, evaluated as `(l0*p0 + l1*p1) + l2*p2` per component. Then `P[b] = qb` and `P[c] = qc` are overwritten exactly.
   - Normal: `cr = cross(p1 − p0, p2 − p0)`, then `N = cr / max(‖cr‖, 1e-30)`.
   - Optional interpolated file normal: `FN = l0*n0 + l1*n1 + l2*n2`, with no normalisation here.
   - `a`, `B`, `C`, and the edge index `G = floor(P[a] / h)` as int64.
7. **Canonical order.** Concatenate all parts, then sort by the key tuple **(a, B, C, G, P.x, P.y, P.z, N.x, N.y, N.z)**, ascending and stable. This is the numpy `lexsort` with the last key primary.
   - Every later step uses this order.
   - The result therefore does not depend on face order, vertex order or chunking, because exact-tie rows are identical in every field.
   - The test `test_rev3_resampling_is_density_invariant_and_order_free` checks this on shuffled input.

## 3. Ladder (`decimate`)

`h_k = cell_mm * 1.12**k`, using Python float `**`, i.e. `pow`. Do not compute it as repeated multiplication.

1. `n0 = count(0)`, where `count(k)` is the number of crossings at `h_k`, memoised.
2. Predict `k`:
   - if `2*n0 <= max_faces`, then `k = 0`;
   - else `k = min(24, ceil(ln(2*n0/max_faces) / (2*ln 1.12)))`.
3. Correct upwards: `while k < 24 && 2*count(k) > max_faces: k += 1`.
4. Correct downwards: `while k > 0 && 2*count(k−1) <= max_faces: k -= 1`.
5. Build the full crossing set at `h = h_k`.

The result is the smallest k whose output is within the budget, provided the count is monotone. The count depends on the surface, not on its sampling.

`info.ladder` lists `[round(h_j, 5), count_j, 2*count_j]` for every evaluated j, sorted by j.

## 4. Resample: `resample(cx, h)`

Let n be the number of crossings, in canonical order.

**a. Cells.**
- Each crossing i has 4 cells, j = 0..3, with offsets `QUAD = ((-1,-1), (0,-1), (0,0), (-1,0))` in (b, c):
  - `cell[a] = G`;
  - `cell[b] = B + db`;
  - `cell[c] = C + dc`.
- Flatten row-major to index `4i + j`.
- Compute `rel = cell − min(cell over all 4n rows)` per axis, `dim = max(rel) + 1` per axis, and `ckey = (rel0*dim1 + rel1)*dim2 + rel2`.

**b. Skins.**
- `first[cell]` = the smallest flattened index 4i+j with that ckey (numpy `unique(return_index)`).
- Reference normal of a row = `N[first // 4]`.
- `skin = dot(N_row, ref) < -0.5`, as 0 or 1.

**c. Vertices.**
- `vkey = ckey*2 + skin`.
- Vertices are the sorted unique `vkey` values, and the vertex index is the rank.
- `vinv[4i+j]` is that rank.

**d. Vertex position (QEF).** Each sum below is accumulated **in flattened row order** (numpy bincount order).
- `cnt[v]` = the number of rows, and `mean[v] = Σ P_row / cnt`, per axis.
- `d_row = N·P`, computed with einsum as `N.x*P.x + N.y*P.y + N.z*P.z`.
- `M[v]_{ij} = Σ N_i N_j` over i ≤ j, filled symmetrically, and `rhs[v]_k = Σ N_k d_row`.
- `lam = 0.05 * (M00 + M11 + M22)/3 + 1e-12`.
- Then `rhs += lam*mean` and `M_ii += lam`.
- `X = solve(M, rhs)`. numpy uses LAPACK gesv: LU with partial pivoting. The Rust port already reproduces this for revision 2 (`dgesv per cluster`).
- If `‖X − mean‖ > h`, then `X = mean`.
- Normals, if file normals exist: `No[v] = Σ FN_row`, in row order, then `No /= max(‖No‖, 1e-30)`.

**e. Faces.**
- `q[i] = vinv[4i .. 4i+3]`.
- If `N_i[a] < 0`, reverse q[i] to `[q3, q2, q1, q0]`.
- `d02 = ‖X[q0] − X[q2]‖` and `d13 = ‖X[q1] − X[q3]‖`.
- If `d02 <= d13`, the two triangles are `(q0,q1,q2), (q0,q2,q3)`; otherwise they are `(q0,q1,q3), (q1,q2,q3)`.
- They go in rows `2i` and `2i+1`.
- Drop triangles with a repeated vertex.
- Dedupe on the sorted vertex triple, keeping the first occurrence. The kept rows stay in their original order: this is revision 2's rule (stable lexsort on the sorted triple, then `sort(o[new])`).
- Faces are written as int32.

**f. Info.**
- The keys are `faces_in`, `vertices_in`, `faces_out`, `vertices_out`, `cell_mm` (= `round(h, 5)`), `crossings`, `ladder` and `clamped_frac` (`round(mean(far), 5)`).
- After the write, `budget_scan` adds `applied`, `max_faces`, `method: "fixed_grid_dual_contouring"` and `wall`.
- Revision 2's `mean_edge_mm` key is gone.

## 5. What the port must not do

- **Do not keep revision 2's density-dependent start.** The start was `edge × sqrt(F / max_faces)`; revision 3 starts at the fixed `cell_mm`.
- **Do not take the grid origin from the data** (no `V.min`). The grid is `floor(x / h)`.
- **Do not sum the accumulations in face order or per chunk.** Every sum runs in the canonical crossing order (§2.7), then in flattened row order 4i+j.

## Reference numbers

These come from the Python reference on this host, through the lane's production path, and can be used to check the port:

| Mesh | Faces in | `cell_mm` | Faces out | Vertices out |
|---|---|---|---|---|
| A (sc_3c93a0bf6f97) N=3 | 4 408 443 | 0.3 | 1 520 588 (crossings 763 509) | 775 820 |
| B (sc_6eba02723d4b) N=7 | 46 234 832 | 0.336 | 2 505 295 (crossings 1 263 212) | 1 255 959 |
| D (sc_203b95d7a753) native | 3 962 601 | 0.336 | 2 595 757 (crossings 1 298 386) | 1 298 078 |

The 6 budgeted DEV scans' budget infos (`cell_mm`, `faces_out`, `crossings`, `ladder`) are in `rev3/oof/BUDGETED_DEV.json`.
