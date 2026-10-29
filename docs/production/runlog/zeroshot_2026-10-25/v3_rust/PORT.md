# zs_lane_v3 in Rust (`zs_rust`): port, equivalence and speed

> **Superseded as reference (2026-10-26, later the same day):** the crate is now portable (pure-Rust libm and linear algebra, no Accelerate) and the binary is the reference; see [PORTABLE.md](PORTABLE.md). The Accelerate build described below (`567101d3…`) is kept as `research/zeroshot/zs_rust/target/accelerate_567101d3/zs_rust` for comparison only; its configs and fingerprints below are historical.

**Date:** 2026-10-26. **Status:** research port. Nothing in production was changed. The production pin, `lane_cli`, gold, labels, E3 and the frozen Python lanes were not edited. No commits were made.

> **What this is.** The whole zero-shot bend lane v3 runs as one Rust binary, `zs_rust`. That covers mesh I/O, the face budget, Z2 (D1 thickness, segmentation, graph), Z3 (thickness, axis families, slicing, bounded merge, strip evidence), pooling, the snap, the evidence features, both GBMs, post-processing, EXT/Z3R and the `eval_lines` output.
> - It reproduces the frozen Python **bit for bit**. That holds for revision 2 (`ab31d061…`) and for revision 3 (`dc23a905…`, the fixed-grid budget), which is selected by the config.
> - It is called as a subprocess that exchanges files.
> - Python keeps only a thin adapter (guard, custody, pins, point-cloud fallback), the GBM exporter and the equivalence harness.

| | Result |
|---|---|
| **Equivalence, revision 2** | DEV `eval_lines`: **125/125 byte-identical** to the frozen Python revision 2 at 1, 4, 8 and 16 threads (max endpoint difference 0.0 mm, max score difference 0.0). Stage dumps: 125/125 identical. The 18 densified meshes (3.8–46 M faces): 18/18 identical, budget records included. |
| **Equivalence, revision 3** | DEV `eval_lines`: **125/125** identical to the frozen Python revision 3 at 1, 4, 8 and 16 threads. Budget step, mesh level: **24/24 meshes give bitwise-identical resampled V and F** and budget info (6 budgeted DEV scans and 18 dense meshes). Dense lane: 18/18 identical. |
| **DEV OOF** (revision 2 procedure on the Rust dump) | Fixed v3 fold models: **F1 0.9660**, macro 0.9534, TP/FP/FN 597/20/22, control FP 1. Refit arm: 0.9651, 595/19/24. Both are byte-identical to the Python revision-2 OOF files. |
| **CAD-prior path** (production `lane_cli`) | 6/6 DEV scans identical under revision 2 and 5/5 under revision 3 |
| **Determinism** | Output is byte-identical across 1/4/8/16 threads on all 125 DEV scans for both revisions, the 4 N1 scans and all 16 dense meshes × 3 budget modes |
| **Speed, DEV** (122 mesh scans, revision 2) | Python P50 6.5 s / P95 16.5 s. Rust at 1 thread: 2.60 s / 5.02 s. Rust at 16 threads: **0.59 s / 1.15 s** (max 1.65 s). Median speed-up per scan: 2.8× at 1 thread, 7.1× at 4, 9.9× at 8, **11.6× at 16**. |
| **Speed, dense** (46 M faces, revision 3) | Python 29.5 s / 5.2 GiB. Rust 7.4 s at 1 thread and **1.46 s at 16 threads / 2.3 GiB**. |
| **Peak RSS** | DEV: at most 1.5 GiB (Python 2.6 GiB). On the 9.6 M-vertex stress mesh: 2.5–2.7 GiB with the revision-2 budget (Python 3.25 GiB). Revision 3 at 46 M faces: 2.3 GiB (Python 5.2 GiB). |
| **Binary** | `567101d3f5874a1ac13327892213002d4494a7395527628886b8bb37106d58a9`, 1 916 288 bytes. A clean rebuild reproduces the same sha256. |

## 1. Where Python spent its time (profile before the port)

- **cProfile on the largest DEV/N1 scan (41018000, revision 1): 130 s.**
  - Z3's `merge_segments` took 99 s; revision 2's bounded merge later removed most of that.
  - Z2 took 12 s and the snap 6 s.
- **Revision 2 DEV CLI, 119 mesh scans (four host-stalled scans excluded):** 792 s in total.
  - Z2: 431 s (54 %).
  - Z3: 278 s (35 %).
  - Snap: 27 s. Evidence: 24 s. Budget: 14 s.
- **Threads did not help.** The Python E3/zs lanes ran at 0.98–1.05× with more threads: the numpy, scipy and open3d calls are short and the glue between them is serial.
- Files: `profile/` (cProfile) and `speed/dev_stage_breakdown.json` (Python and Rust stage sums).

## 2. What was ported (module map)

| Python (frozen) | Rust (`research/zeroshot/zs_rust/src/`) |
|---|---|
| `zs_lane_v3/core.py`: `process`, `select_lines`, `payload` | `lane.rs` (`run`, `payload`), `main.rs` (`cmd_lane`, job and record I/O) |
| `zs_lane_v3/budget.py`, revision 2 (normal-aware vertex clustering) and revision 3 (fixed-grid dual contouring) | `budget.rs`. `apply(v, f, &BudgetCfg)` is the only entry point: `decimate` for revision 2, `decimate_rev3` for revision 3. The config's `budget.cell_mm` selects revision 3. |
| `zs_lane_v3/z3_merge.py`: bounded `merge_segments`, `detect` | `z3.rs` (`merge_segments`, `detect`, slicing, chains, corners, tracks, `segment_line`, `refine_span`, `strip_evidence`, `merge_candidates`) |
| `zs_lane_v3/fixes.py`: EXT, Z3R | `lanecore.rs` (`extend`, `rescue`, `apply_fixes`) |
| `zs_lane_v2/core.py`: Z2 driver, pool, snap, features | `main.rs` (`z2_full_with`), `lanecore.rs` (`pool`, `snap_all`, `feats`) |
| `zs_lane_v2/z3s.py`: thickness, Gauss map, normal peaks, axis families | `z3prep.rs`, `mesh.rs` |
| `zs_lane_v2/postproc.py` | `lanecore.rs` (`postprocess`, `pair_geom`, `dedupe_joined`) |
| `zs_lane_v2/evidence.py` | `evidence.rs` |
| `z3_slicing/` (slicer, union, pooled, selector) | `z3.rs`, `lsap.rs` (linear assignment), `morph.rs` |
| `z2_self_recipe/segment.py`, `graph.py`, `run.py` | `z2seg.rs`, `z2graph.rs`, `z2load.rs` |
| `d1_two_skin/two_skin.py` | `d1.rs`, `z2load.rs` |
| E3 metrology used by Z2 (`domains/production/engine/inspect/metrology.py` etc.; read-only reference) | `z2graph.rs` (`skin_peaks`, `station_layout`, `skin_level`, `fit_side`, `station_sharp`, `reseat_plane`, `robust_line`, `sagitta`, `flat_start`, `intersect`) |
| sklearn HistGradientBoosting pickles | `export_models.py` writes `models/*.gbm.json` (format `sherman.zs_gbm.v1`); `gbm.rs` does inference |
| scipy `cKDTree` | `ckdtree.rs` (exact build and query: libc++ `nth_element`, scipy's heap, FMA distances) |
| open3d `KDTreeFlann` / radius queries | `kdtree.rs` |
| open3d `RaycastingScene` (Embree 4.3.3) | `raycast.rs` (Embree's robust Plücker test with aarch64 FMAs; closest hit, lowest primitive id on exact ties) |
| numpy `PCG64`, `choice`, argsort, reductions; Python float repr and `round` | `nprng.rs`, `num.rs`, `pyjson.rs` |
| numpy LAPACK/BLAS (macOS Accelerate) | `accel.rs` (FFI to the same Accelerate routines: dsyrk, dsyevd, dgesv, dgetrf, dgelsd, dgemm) |
| open3d / assimp mesh readers | `meshio.rs` (streaming binary-PLY fast path, generic PLY, STL with assimp's vertex-join rule) |

**Size.** About 10.9 k lines of Rust in 25 files. The release binary is 1 916 288 bytes (aarch64-apple-darwin, ad-hoc signed).

**Dependencies** (crates.io; `Cargo.lock` is pinned): rayon, serde, serde_json and sha2. The binary links only the system Accelerate framework and libSystem.

**Adapter (`research/zeroshot/zs_lane_v3_rs/`):**

| File | Role |
|---|---|
| `lane.py` | CLI: guard, custody and strict pins (exit 4); calls the binary; runs the point-cloud fallback (exit 10 → the Python revision-2 lane) |
| `core.py` | `run_binary(…, cad=…)` |
| `rs.py` | Locates the binary; reads `ZS_RUST_THREADS` |
| `fingerprint.py` | Code fingerprint over the crate sources, `Cargo.lock`, the binary sha256, the exported models, the adapter and the zs_lane_v3 file set |
| `freeze_config.py` | Writes `config_v3_rs.json` (revision 2) and, with `--revision 3`, `config_v3r3_rs.json` |
| `export_models.py` | Writes the GBM JSON exports |
| `dev_dump.py`, `oof.py` | DEV dump and out-of-fold evaluation |
| `rev2ref.py`, `rev2_snapshot/` | Byte copies of revision 2, checked against its hashes. Revision 3 was built in place in `zs_lane_v3/`, so these copies are the revision-2 reference. |

**Production CAD priors are implemented and checked.** `lane_cli.run_lane_v3` passes `core.process(thickness={"t_mm": CAD, "source": "CAD"}, max_families=K)` and caps the result at `n_bends`.
- The binary takes these priors through the job field `"cad": {"t_mm", "max_families"}`.
- The cap stays with the caller, as it does in `lane_cli`.

## 3. Equivalence

All comparisons are exact. Endpoints, scores, inclinations, `src`, `op` and `fix` are compared as JSON values. "Identical" means byte-identical, so the 1e-6 mm / 1e-9 tolerances of the brief were never needed.

| Check | Revision 2 | Revision 3 |
|---|---|---|
| DEV `eval_lines` vs Python DEV CLI (`v3/rev2/dev_cli`, `v3/rev3/dev_cli`), 1 / 4 / 8 / 16 threads | 125 / 125 / 125 / 125 identical | 125 / 125 / 125 / 125 identical |
| Max endpoint / score difference | 0.0 mm / 0.0 | 0.0 mm / 0.0 |
| DEV stage dumps (candidates, feature matrix, Z2 lines, evidence incl. splits) vs the Python revision-2 dump | 125/125, max abs 0.0 | n/a (identical code after the budget) |
| DEV OOF (frozen `rev2_eval.py` on the Rust dump) | `OOF_v3r2_lines.json` and `OOF_v3r2_per_case.json` byte-identical: F1 0.9660, 597/20/22, ctrl FP 1; refit 0.9651, 595/19/24 | n/a |
| Budget step, mesh level: resampled V and F bitwise, info dict | revision-2 decimation: 6 budgeted DEV scans inside the dumps; synthetic test | **24/24** (6 DEV + 18 dense) |
| Dense meshes (A–D and N1 densified, 3.8–46.2 M faces), lines and budget record vs a fresh Python run | 18/18, budget info 18/18 | 18/18, budget info 18/18 |
| CAD priors (synthetic t / K / n_bends facts) | 6/6 | 5/5 (3 budgeted) |
| Mesh reading vs open3d (V, F bitwise) | 122/122 mesh scans (3 point clouds detected) | same |
| Z3 alone vs `z3_merge.detect`; Z2 alone vs `run_z2` (intermediate gates) | 122/122; 122/122 | same |
| GBM JSON export vs the sklearn pickles | max difference 0 on the DEV feature matrices (1e-9 gate) | same models |
| Unit tests `tests/test_prod_zs_rust.py` | 13/13 pass (primitives, cKDTree ties, ray casts vs open3d, mesh readers, both budgets at 1 and 4 threads, GBM, Z3, Z2, whole lane at 1 and 4 threads, point cloud, fingerprint) | |

Evidence files are in `equivalence/` (`rev2_eval_lines_t*_vs_python_rev2.json`, `rev3_eval_lines_t*_vs_python_rev3.json`, `dump_stage_cmp.json`, `r3_budget_mesh.json`, `dense_rev2_lines.json`, `dense_rev3_lines.json`, `cad_rev2.json`, `cad_rev3.json`, `mesh_read.json`, `z3full_cmp.json`, `z2full_cmp.json`, `z2_segment_cmp.json`) and `oof/`. Summary counts are in `results.json`.

**How exactness was reached.** Each of the following was found by a stage-by-stage diff on DEV and fixed at the root:
- **numpy evaluation orders:**
  - pairwise summation in blocks of 8 accumulators;
  - sequential axis-0 reductions;
  - `einsum` over a length-3 axis = `fma(a2,b2,a0*b0)+a1*b1`;
  - an Accelerate `ddot` with a strided eigenvector column = `(p0+p2)+p1`.
- **scipy:** its FMA-contracted squared distances.
- **LAPACK:** the same Accelerate routines, called through FFI with numpy's argument conventions (`rcond = eps·max(m,n)` for lstsq; det from the LU factors).
- **Bit-exact ports of the primitives:** numpy's PCG64 and every `choice` variant (Lemire bounded draws, Floyd, the shuffle tail); numpy's introsort argsort; scipy's cKDTree build and tie order; Embree's ray–triangle test.
- **LLVM's `sincos` fusion** differs in the last bit, so sin and cos are separate calls.
- **Python's float repr and `round`** are reproduced in the JSON writer.
- **The budget:** numpy `bincount` order (each bin summed in input order) and `lexsort` stability, at every thread count.

**Determinism.** Every parallel step either works on disjoint data or merges in a fixed order. Examples: tree builds by `rayon::join` on disjoint slices; ordered collects; a union-find whose root is the minimum index; bincounts over disjoint bin ranges; speculative merge batches that take the first hit in scan order. Output is byte-identical at 1, 4, 8 and 16 threads (§3 table).

## 4. Speed

The host is an Apple M-series laptop (shared, so timings carry some noise). Every Rust run is one process per scan, through guard and slot, with strict pins. Wall time is the adapter's process wall; RSS is the binary's peak RSS (getrusage).

### DEV (122 mesh scans; Python = the frozen DEV CLI runs on the same scans)

| Run | n | P50 | P95 | Max | Peak RSS |
|---|---|---|---|---|---|
| Python revision 2 (3 host-stalled scans excluded) | 119 | 6.52 s | 16.5 s | 25.2 s | 2.63 GiB |
| Python revision 3 | 122 | 6.77 s | 18.0 s | 30.8 s | 2.72 GiB |
| Rust revision 2, 1 thread | 122 | 2.60 s | 5.02 s | 8.57 s | 1.35 GiB |
| Rust revision 2, 4 threads | 122 | 1.00 s | 2.05 s | 2.56 s | 1.38 GiB |
| Rust revision 2, 8 threads | 122 | 0.70 s | 1.39 s | 1.87 s | 1.43 GiB |
| **Rust revision 2, 16 threads** | 122 | **0.59 s** | **1.15 s** | **1.65 s** | 1.46 GiB |
| Rust revision 3, 1 / 4 / 8 / 16 threads | 122 | 2.58 / 1.00 / 0.70 / 0.59 s | 5.28 / 2.12 / 1.58 / 1.27 s | 8.17 / 2.52 / 1.83 / 1.53 s | ≤ 1.52 GiB |

**Thread scaling.** Python got 0.98–1.05× from more threads. Rust's median speed-up per scan against Python revision 2 is:

| Threads | Median per scan | Sum of walls |
|---|---|---|
| 1 | 2.8× | 2.8× |
| 4 | 7.1× | 7.3× |
| 8 | 9.9× | 10.2× |
| 16 | **11.6×** | 12.1× |

Against revision 3 the medians are 2.9×, 7.4×, 10.4× and 12.1×.

Most DEV scans are 0.2–1.4 M faces, so above 8 threads the serial stages in the remaining-stages note below limit further gains.

**Stage sums over DEV** (`speed/dev_stage_breakdown.json`):

| Stage | Python rev 2 (119 scans) | Rust, 1 thread | Rust, 16 threads |
|---|---|---|---|
| Z2 | 431 s | 201 s | 37.9 s |
| Z3 | 278 s | 69.7 s | 17.3 s |
| Snap | 27 s | 2.3 s | 0.6 s |
| Evidence | 24 s | 3.3 s | 1.0 s |
| Budget | 14 s | 8.7 s | 3.1 s |
| **Total** | **792 s** | **310 s** | **70.3 s** |

### N1 (4 new scans; timing only, nothing scored)

| Scan | Python rev 2 (FREEZE_REV2) | Python rev 3 (RUNTIME.json) | Rust rev 2, 1 / 4 / 8 / 16 threads | Rust rev 3, 1 / 4 / 8 / 16 threads | Rust RSS |
|---|---|---|---|---|---|
| 41018000_H (3.4 M faces, budgeted) | 26.1 s | 34.8 s | 6.67 / 3.07 / 2.48 / 2.47 s | 8.28 / 3.53 / 2.78 / 2.77 s | 1.3–1.5 GiB |
| 45351000_B | 18.3 s | 19.2 s | 5.72 / 2.30 / 1.59 / 1.33 s | 5.79 / 2.20 / 1.60 / 1.34 s | 0.95 GiB |
| 47085000_A | 8.9 s | 9.5 s | 3.05 / 1.21 / 0.85 / 0.72 s | 3.08 / 1.18 / 0.87 / 0.70 s | 0.65 GiB |
| 51101001_01 | 21.7 s | 24.1 s | 6.40 / 2.54 / 1.90 / 1.70 s | 6.50 / 2.57 / 1.95 / 1.71 s | 1.2 GiB |

Output is identical across thread counts on all four scans.

### Dense meshes (station stand-ins: densified DEV scans A–D and N1, 3.8–46.2 M faces)

- Python is a fresh run per mesh of the frozen lane (`equivalence/dense_rev*_lines.json`).
- Rust is the binary alone (`speed/dense_rust.json`).
- "Budget off" is the experimental `budget_override {"max_faces": null}`: Rust only, with no Python reference, and its lines differ by design.

| Mesh | Faces | Python rev 2 | Rust rev 2, 1 / 16 thr (RSS) | Python rev 3 | Rust rev 3, 1 / 4 / 8 / 16 thr | Rust rev 3 RSS | Budget off, 1 / 16 thr (RSS) |
|---|---|---|---|---|---|---|---|
| B_n2 | 3.8 M | 12.1 s, 2.46 GiB | 4.81 / 1.00 s (1.36 GiB) | 14.5 s, 2.48 GiB | 5.65 / 2.01 / 1.37 / 1.14 s | 1.39 GiB | 4.86 / 1.03 s (1.32 GiB) |
| A_n3 | 4.4 M | 14.8 s, 2.48 GiB | 5.23 / 1.28 s (1.53 GiB) | 12.0 s, 2.14 GiB | 4.54 / 1.64 / 1.15 / 0.97 s | 0.95 GiB | 5.13 / 1.25 s (1.58 GiB) |
| C_n2 | 5.7 M | 13.2 s, 2.40 GiB | 5.83 / 1.22 s (1.49 GiB) | 15.1 s, 2.52 GiB | 6.06 / 2.14 / 1.41 / 1.16 s | 1.37 GiB | 5.40 / 1.15 s (2.18 GiB) |
| B_n3 | 8.5 M | 14.8 s, 2.61 GiB | 5.70 / 1.25 s (1.61 GiB) | 16.0 s, 2.60 GiB | 5.99 / 2.06 / 1.42 / 1.14 s | 1.46 GiB | 6.64 / 1.59 s (2.75 GiB) |
| A_n5 | 12.2 M | 16.8 s, 2.96 GiB | 6.75 / 1.61 s (1.92 GiB) | 13.8 s, 2.69 GiB | 4.71 / 1.75 / 1.24 / 1.06 s | 1.15 GiB | 8.83 / 2.23 s (4.42 GiB) |
| C_n3 | 12.7 M | 16.4 s, 3.06 GiB | 7.66 / 1.66 s (1.97 GiB) | 22.0 s, 3.05 GiB | 6.64 / 2.27 / 1.51 / 1.20 s | 1.58 GiB | 8.40 / 1.90 s (4.49 GiB) |
| N1_n2 | 13.5 M | 25.4 s, 3.16 GiB | 8.07 / 2.19 s (2.00 GiB) | 36.9 s, 3.27 GiB | 8.72 / 3.50 / 2.77 / 2.62 s | 1.66 GiB | 15.3 / 5.71 s (4.57 GiB) |
| D_n2 | 15.9 M | 13.6 s, 3.17 GiB | 5.78 / 1.39 s (2.33 GiB) | 15.3 s, 2.99 GiB | 5.46 / 1.97 / 1.35 / 1.12 s | 1.55 GiB | 8.35 / 2.17 s (5.17 GiB) |
| A_n6 | 17.6 M | 19.9 s, 3.35 GiB | 7.96 / 1.89 s (2.34 GiB) | 14.8 s, 2.73 GiB | 4.73 / 1.83 / 1.26 / 1.06 s | 1.28 GiB | 11.1 / 3.04 s (6.01 GiB) |
| C_n4 | 22.6 M | 21.4 s, 3.54 GiB | 9.47 / 2.18 s (2.94 GiB) | 22.7 s, 3.33 GiB | 6.65 / 2.34 / 1.53 / 1.25 s | 1.79 GiB | 12.9 / 3.07 s (7.53 GiB) |
| B_n5 | 23.6 M | 20.7 s, 3.60 GiB | 8.26 / 2.03 s (2.92 GiB) | 21.1 s, 3.63 GiB | 6.22 / 2.24 / 1.54 / 1.25 s | 1.83 GiB | 13.5 / 3.38 s (8.08 GiB) |
| N1_n3 | 30.4 M | 30.1 s, 4.20 GiB | 10.6 / 2.71 s (3.32 GiB) | 44.4 s, 4.12 GiB | 9.51 / 3.76 / 2.90 / 2.85 s | 2.08 GiB | 31.8 / 11.1 s (9.34 GiB) |
| C_n5 | 35.3 M | 29.9 s, 5.45 GiB | 13.2 / 3.26 s (3.85 GiB) | 27.7 s, 4.41 GiB | 7.00 / 2.47 / 1.62 / 1.31 s | 2.07 GiB | 18.2 / 4.60 s (9.88 GiB) |
| D_n3 | 35.7 M | 22.5 s, 5.14 GiB | 9.39 / 2.57 s (3.81 GiB) | 22.3 s, 5.05 GiB | 5.87 / 2.21 / 1.51 / 1.23 s | 2.08 GiB | 16.9 / 4.68 s (10.1 GiB) |
| A_n9 | 39.7 M | 31.9 s, 5.70 GiB | 13.8 / 3.59 s (4.30 GiB) | 20.4 s, 5.04 GiB | 5.33 / 2.00 / 1.40 / 1.18 s | 1.80 GiB | 21.6 / 6.38 s (11.0 GiB) |
| **B_n7** | **46.2 M** | 31.3 s, 6.18 GiB | 12.6 / 3.75 s (4.53 GiB) | 29.5 s, 5.16 GiB | **7.35 / 2.67 / 1.80 / 1.46 s** | **2.32 GiB** | 24.4 / 6.14 s (13.1 GiB) |

**Revision 3 in Rust.** At 46 M faces the lane takes 1.46 s at 16 threads (20× Python) and peaks at 2.3 GiB.
- The budget step itself takes about 0.33 s at 16 threads and 2.3 s at 1 thread. Python's budget step takes 15 s for the same mesh.
- Mesh load peaks at about the decoded arrays (1.1 GB at 46 M faces). Python peaks at 4.4–5.7 GiB.

**Budget off.** Rust runs the full 46 M-face mesh in 6.1 s at 16 threads and 13 GiB. That is within the station's 32 GiB step cap, but not within this host's 16 GiB rule for anything larger.

### Stress meshes (DEV scans subdivided ×4)

| Mesh | Python rev 2 (FREEZE_REV2) | Rust rev 2, 1 / 4 / 8 / 16 threads | Rust RSS | Budget off, 1 / 16 threads (RSS) |
|---|---|---|---|---|
| sc_3b3e8c9f04e3 ×4 (6.9 M v / 13.7 M f) | 13.0 s, 3.29 GiB | 5.57 / 2.35 / 1.64 / 1.36 s | 1.9–2.0 GiB | 7.72 / 2.28 s (4.7 GiB) |
| sc_ead69b8181a8 ×4 (9.6 M v / 19.2 M f) | 20.3 s, 3.25 GiB | 7.98 / 3.42 / 2.39 / 1.95 s | **2.5–2.7 GiB** | 13.1 / 3.44 s (6.9 GiB) |

### What made it fast (all output-identical)

| Change | Effect |
|---|---|
| Native code for the glue | The Python stages were dominated by per-call overhead around many small numpy calls. At 1 thread, Rust is 2.8× faster per scan (median). |
| rayon wherever the work splits | Ray casts, kNN queries, normals, region growing, the graph's candidate pairs, Z3 per-family slicing, strip evidence, the snap, the evidence features, both budgets |
| Parallel tree builds | BVH, kd-tree and cKDTree builds by `rayon::join` on disjoint slices (the tree shape does not depend on the thread count) |
| Shared structures | One BVH, one kd-tree and one mesh shared by Z2, Z3, the snap and the evidence |
| Mesh load | A streaming binary-PLY reader (64 MB blocks, parallel decode); unused vertex normals are skipped; the full-density mesh is freed right after the budget |
| Budget, revision 2 | Exact parallel `bincount`: items bucketed by bin range in input order, each range summed in order |
| Budget, revision 3 | Parallel crossing generation; one parallel sort whose tie-break is Python's concatenation rank, which equals numpy's stable lexsort |
| `merge_segments` | Speculative batches of pair tests run in parallel; the first merging pair in the frozen scan order wins, so the result equals the sequential loop |
| Scan hash | The scan's sha256 runs on its own thread while the lane runs, using CommonCrypto (hardware SHA-256, 4× the sha2 crate) |
| Build | `opt-level=3`, `lto="fat"`, `codegen-units=1`, `panic="abort"` |

**Build-flag variants.**
- `-C target-cpu=native` was measured on an earlier build of the same code: 14 profile runs, same output. The sums were 41.3 s → 41.9 s at 1 thread and 12.08 s → 12.19 s at 16 threads, i.e. no gain. It is not used.
- PGO was not tried: it needs `llvm-tools`, a toolchain install. The expected gain is under 10 %.
- **For the x86 station, build with `-C target-cpu=x86-64-v3`** (AVX2 and FMA).
  - The code relies on explicit `mul_add` to reproduce numpy and Embree's fused operations.
  - Without the FMA target feature, every `mul_add` becomes a libm call: correct, but slow.
  - `x86-64-v4` (AVX-512) adds nothing measurable for this code.

### Remaining serial stages (next CPU work, not done)

Summed over DEV at 16 threads (70.3 s in total): `z3.normal_peaks` (8.9 s), `z2.segment.grow_merge` (8.9 s), `z3.detect.slice` (7.0 s) and `read_mesh` (6.5 s; ASCII PLY and STL go through the generic reader). Together these are about 45 % of the 16-thread wall and run mostly serially, so parallelising them is the next 1.5–1.8× on DEV.

## 5. GPU (not done; top three candidates with estimates)

At 16 threads the lane takes 0.6 s (DEV median) and 1.46 s for 46 M faces under revision 3. Every GPU kernel would also have to be made bit-exact, using IEEE intrinsics (`__fma_rn`, `__fmaf_rn`) and ordered reductions.

1. **Ray casting: the Z2 D1 probes, Z3 thickness and the BVH build.** About 11 % of the 16-thread wall on DEV (3.9 + 3.1 + 1.2 s of 70 s).
   - Approach: a CUDA LBVH with a custom intersection program that reproduces Embree's f32 Plücker test (`fmaf`) and the lowest-id tie rule. The RT cores' fixed triangle test cannot be used, because its rounding differs from Embree's.
   - Estimate: these stages 5–10× faster, about 8 % of wall saved.
2. **Z2 kNN and normal smoothing.** About 7 % of wall (`z2.knn_query` 4.5 s plus normals).
   - Approach: exact grid kNN on the GPU, keeping scipy's tie order and FMA distances.
   - Estimate: 10× on the kNN, about 5 % of wall. Region growing stays on the CPU (it is irregular and order-dependent).
3. **The full-resolution dense path** (budget off, or a future finer cell): the Z3 Gauss map, normal peaks and slicing on 10–40 M faces.
   - With the budget off at 46 M faces, Rust takes 6.1 s at 16 threads, and Z3 dominates.
   - Estimate: 3–5× on those stages. This only matters if the budget is lifted.

**Verdict.** GPU would save under about 25 % of wall. It would add a CUDA toolchain and a second exactness surface. It is not worth doing before the serial CPU stages above are parallelised.

## 6. Build, binary, reproducibility

- **Build command** (run through the slot):

  ```
  cd research/zeroshot/zs_rust && cargo build --release --locked
  ```

  Toolchain: rustc and cargo 1.99.0. Profile: opt 3, fat LTO, 1 codegen unit, panic=abort, generic target CPU. The `accelerate` feature is on by default.
- **Binary:** `research/zeroshot/zs_rust/target/release/zs_rust`.
  - sha256: **`567101d3f5874a1ac13327892213002d4494a7395527628886b8bb37106d58a9`**
  - 1 916 288 bytes.
  - Links `Accelerate.framework` and `libSystem` only.
- **`Cargo.lock` sha256:** `6fb84a918bbd2d69e4200b5753e4467787ab2013a0965ceab31700fe73ff5e57`.
- **Reproducibility.**
  - `cargo clean && cargo build --release --locked` reproduced the identical sha256.
  - A clean build into a different target directory differs in exactly 48 bytes: the 16-byte `LC_UUID` and 32 bytes of the ad-hoc code signature. Code and data are identical.
  - So the binary is reproducible at a fixed build path. For a path-independent sha, build at a fixed path in CI, or compare with the UUID and signature masked.
- **Pinned configs:**
  - `config_v3_rs.json`, revision 2: base fingerprint `ab31d061…`, config `1c3f8cb7…`, code fingerprint `437bf67d43432c1c1b8834fe17e887c889443022d9c37a899fd149c523e974b7`.
  - `config_v3r3_rs.json`, revision 3: base fingerprint `dc23a905…`, config `8e002f3e…`, code fingerprint `398477f004374994e0fe17397d8ef910e37f205811b7847e708993b8fc8011be`.
  - Both pin the binary sha256 and the model exports.
  - The adapter's `--strict-pins` refuses (exit 4) on any mismatch. This was exercised by accident: a source edit during a run made the adapter refuse every later scan, which is the intended behaviour.

## 7. Production: what is needed

None of this was done; it is a proposal.

1. **Platform: the first decision.** Exactness against the Python reference relies on macOS Accelerate. Accelerate supplies the LAPACK and BLAS results that numpy on this Mac uses (3×3 `eigh`, `solve`, `lstsq`, `det`, `syrk`, `gemm`).
   - The station (RTX 5000, 256 GB) is very likely x86 Linux or Windows. There, numpy itself uses OpenBLAS or MKL, so the Python reference also changes in the last bits from the Mac.
   - Options:
     - **(a) Pin the station's numpy/LAPACK and validate the Rust x86 build against Python on the station.** The crate builds without `accelerate` using its own 3×3 routines. Those routines must then be matched to the station's LAPACK, as was done for Accelerate here. Risk: near-tie flips in Z3's merge order, which needed Accelerate parity on the Mac.
     - **(b) Declare the Rust binary the reference** and re-freeze accuracy (DEV OOF) with it on the station.
   - (b) is simpler and does not depend on the platform. (a) keeps the Python lane as the oracle.
2. **Runner integration.** In `lane_cli.run_lane_v3`, a pin with `lane: "zs_lane_v3_rs"` would build the job (`scan`, `config` = the pinned config, `cad` = `{t_mm, max_families}` from the compiled part's facts) and call the binary through `zs_lane_v3_rs.core.run_binary`.
   - Keep the `n_bends` cap and the `ZSxx` renumbering in the runner, exactly as today.
   - Exit 10 means a point cloud: either keep the Python fallback or report `REFUSED_SCAN_DETECTION`.
   - Thread count: `ZS_RUST_THREADS` = the number of physical cores, since the output does not depend on it.
   - The runtime contract (lines schema, `runtime_version`) does not change.
3. **Shipping in the app bundle.**
   - The binary goes in `Contents/Resources/bin/zs_rust` (or `Contents/MacOS/`), signed with the app's Developer ID under the hardened runtime. No entitlements are needed: no JIT, no network.
   - Models go in `Resources/models/*.gbm.json` (§3: 0 difference to the pickles).
   - The pin goes in `Resources/pins/`.
   - The binary is self-contained: about 1.6 MB, linking only system frameworks.
   - Re-signing changes the binary's sha256, so the pin must record **the signed binary's sha256** (computed after signing) together with the unsigned build's sha256 and build provenance.
4. **Pin-asset format (proposal), `zs_lane_v3_rs.pin.json`:**

```json
{
  "schema": "sherman.zs_lane_v3_rs.pin.v1",
  "lane": "zs_lane_v3_rs", "revision": 3,
  "base": {"lane": "zs_lane_v3", "code_fingerprint": "dc23a905…", "config_sha256": "8e002f3e…"},
  "binary": {"path": "bin/zs_rust", "sha256_signed": "…", "sha256_unsigned": "…", "bytes": 0,
             "target": "aarch64-apple-darwin", "rustc": "1.99.0", "cargo_lock_sha256": "…",
             "profile": {"opt": 3, "lto": "fat", "codegen_units": 1, "panic": "abort", "target_cpu": "generic"},
             "features": ["accelerate"]},
  "models": {"zs": {"path": "models/zs_lane_v3_zs_gbm.gbm.json", "sha256": "…", "format": "sherman.zs_gbm.v1",
                    "source_sha256": "ade6cde4…"},
             "z3": {"path": "models/zs_lane_v3_z3_gbm.gbm.json", "sha256": "…", "format": "sherman.zs_gbm.v1",
                    "source_sha256": "62e99d13…"}},
  "config": {"path": "config/config_v3r3_rs.json", "sha256": "…"},
  "runtime": {"threads": "physical_cores", "exit_codes": {"0": "ok", "2": "error", "4": "pin mismatch (adapter)",
              "10": "point cloud"}, "point_cloud": "refuse | python_fallback"},
  "equivalence": {"dev_eval_lines_identical": "125/125", "dev_oof": {"f1": 0.966, "tp_fp_fn": [597, 20, 22]},
                  "evidence": "docs/production/runlog/zeroshot_2026-10-25/v3_rust/results.json"}
}
```

   The runner checks every sha256 before the call (as `--strict-pins` does now) and refuses on mismatch.
5. **Re-freeze discipline.** The code fingerprint covers the crate sources, `Cargo.lock` and the binary. Any rebuild means: rerun `freeze_config` (both revisions), `tests/test_prod_zs_rust.py`, and the DEV equivalence (`tools/bench.sh` + `cmp_eval.py`, `dev_dump` + `cmp_dumps.py` + `oof`).

## 8. Open issues and caveats

- **Point clouds.** Three DEV scans have no faces; the binary refuses them (exit 10) and the adapter runs the Python revision-2 lane. Point clouds are never budgeted, so revisions 2 and 3 agree on them.
- **Ray-cast ties.** On an exact tie of hit distance between two triangles (a shared edge), Embree's reported primitive depends on its traversal; the port reports the lowest id. No DEV or dense scan was affected (all outputs identical), but a pathological mesh could differ.
- **Order assumptions.** Z2's `break_triangles` relies on Python set iteration order over small ints and on numpy argsort. Both were ported as they behave in CPython 3.11 and numpy 2.4. A future CPython or numpy change would change the reference, not the port.
- **Budget-off runs.** These are experimental: a Rust-only flag with no Python reference. Their lines differ from the budgeted lines by design.
- **The revision-2 reference copies** (`rev2_snapshot/`) exist because revision 3 replaced `zs_lane_v3/budget.py` and `core.py` in place.
- **Revision-3 config fallback.** The point-cloud fallback uses revision 2 even under the revision-3 config. This is identical for point clouds.

## 9. Files

- **Crate:** `research/zeroshot/zs_rust/` (`Cargo.toml`, `Cargo.lock`, `src/`).
- **Adapter:** `research/zeroshot/zs_lane_v3_rs/`.
- **Tests:** `tests/test_prod_zs_rust.py` (13 tests; each is skipped if the binary is missing). Run them through the slot: `.venv/bin/python -B -m pytest -q tests/test_prod_zs_rust.py`.
- **This folder:**
  - `results.json` (all numbers);
  - `equivalence/` (eval lines, dumps, dense meshes, revision-3 budget meshes, CAD priors, mesh reads, Z2/Z3 stage checks);
  - `oof/`;
  - `speed/` (DEV, N1, dense, stress, stage breakdown);
  - `profile/`;
  - `tools/` (bench, comparisons, dense, stress, results builder).
