# zs_rust portable build: where the drift comes from, how it closes, the claim, timing

**Date:** 2026-10-28/29. Research only. DEV scans only (`integration/dev_scans.tsv`, 125 = 122 meshes + 3 point clouds).
Nothing pinned was edited: `research/zeroshot/zs_rust` (sources, Cargo.lock, binaries), the `zs_lane_v3` /
`zs_lane_v3_rs` packages and configs, the pins, the freeze ledger and `domains/`, `contracts/` are untouched. The
probe work runs in a copy (`research/zeroshot/zs_rust_det`). The deterministic-math Python reference lives in
`research/zeroshot/zs_lane_v3_det` and patches in memory only. No crate was fetched: `libm`, `serde_json` and the
others were already in the cargo cache.

## Claim supported by this evidence

> On all 122 DEV mesh scans, the portable Rust build gives byte-identical output (every line field: endpoints,
> score, inclination, src, op, parts, fix, status, scan sha256) to the Python reference lane when that lane runs
> with the same pinned math kernels. The kernels are the libm-crate transcendentals and the reference-LAPACK linear
> algebra, used instead of macOS Accelerate and Apple libm.
>
> Against the unmodified production Python (numpy on Accelerate plus Apple libm), the portable build gives:
> - identical bend-line sets on 125/125 scans under the frozen matcher;
> - identical line count, status, src, op and fix on every scan, plus one `parts` index swap on one scan;
> - unchanged production-faithful DEV OOF (fixed models: F1 0.9660, 597/20/22);
> - max endpoint deviation 3.7e-5 mm (rev 3; 1.6e-6 mm rev 2) and max score deviation 0.077.
>
> All of the residual deviation is caused by platform math libraries. The portable build is deterministic: 1 vs
> 16 threads gives 122/122 byte-identical results. It is bit-identical between arm64 and x86-64 (generic and v2).

Not supported:
- **Bit-identity to the production Python on the Mac.** That would need Apple libm and Accelerate reproduced bit
  for bit (section 2).
- **x86-64-v3.** See section 3.

## 1. Drift sources (step 1)

The probe is a copy of the crate in which each operation can be switched at run time between its portable
implementation and the platform routine numpy uses:
- `ZS_ACCEL`: Accelerate `$NEWLAPACK$ILP64` dsyrk / dgemm(v) / dsyevd / dgesv / dgetrf / dgelsd, with numpy's call conventions;
- `ZS_HOST_MATH`: Apple libm.

**Validation:** with every operation switched to the platform routine, the probe reproduces the Python reference
on **122/122** DEV scans.

### Leave-one-out from the all-platform build (rev 2, t16)

Each row switches exactly one operation to its portable version. The count is the number of scans whose line
list stays byte-identical to Python (`ablation/`).

| Operation made portable | Identical | | Operation made portable | Identical |
|---|---|---|---|---|
| `X.T@X` (dsyrk) | **11** | | acos | **55** |
| eigh (dsyevd 2×2/3×3) | **23** | | atan2 | **91** |
| solve (dgesv 2×2/3×3) | **43** | | cos | **112** |
| gemm, det, lstsq | 122 each | | sin | **118** |
| | | | exp | **120** |
| | | | tan, ln, hypot, pow | 122 each |
| all linear algebra portable, platform libm | 10 | | all libm portable, Accelerate linear algebra | 43 |

### Per-call view

Both sides are evaluated on every call along the platform trajectory (`call_mismatch_counts.json`):

| Operation | Calls that differ in the last bits |
|---|---|
| dsyrk | 89 % |
| dsyevd | 98 % (including eigenvector sign flips) |
| dgesv | 94 % |
| dgelsd | ≈100 % |
| dgemm | 97 % |
| dgetrf→det | 13 % |
| sin | 5 % |
| cos | 6 % |
| exp | 9 % |
| acos | 13 % |
| atan2 | 17 % |

Every libm difference is ≤ 1 ulp, except atan2 and tan at ≤ 2 ulp. Most of these differences are absorbed
downstream. The ones that reach the output are the operations in the leave-one-out table that drop below 122.

**FMA / summation order.** Neither is an independent source. Outside the six linear-algebra routines and the nine
libm functions, the portable code (pairwise sums, explicit `dfma` where numpy's einsum fuses, Accelerate's
small-n ddot order) gives exactly the native results: switching just these 15 operations restores 122/122. The
order differences that matter are inside Accelerate's kernels. The previous session established FMA chains and
undocumented blocking there.

## 2. Closing the gap (step 2)

**(a) Matching the platform is infeasible.**
- **Apple libm is not correctly rounded.** For 1 500 sampled disagreeing calls per function, checked with
  70-digit decimal arithmetic (`libm_correct_rounding.json`), Apple libm is the correctly rounded side for:
  - atan2 97 %, exp 99 %;
  - cos 38 %, sin 40 %;
  - acos 11 %, hypot 30 %, tan 2 %.

  So no correctly-rounded libm, musl or otherwise, reproduces Apple's results. Matching would mean reverse
  engineering a closed-source library.
- **Accelerate.** The previous session's probes showed FMA plus undocumented blocking in dsyrk above 64 rows, and
  3×3 dsyevd last bits that never all agree. The 89–98 % per-call disagreement above confirms that a reference port
  cannot match it.
- **numpy on other platforms.** The production Python is not portable either: numpy wheels on Windows/Linux use
  OpenBLAS and a different libm.

**(b) Replacing the platform math in both lanes works.** `research/zeroshot/zs_lane_v3_det` is the research
copy. It runs the live Python lane (rev 3 config) with:
- numpy/math transcendentals, `np.linalg.{eigh,solve,det,lstsq}` and sklearn's `expit` replaced by the kernels in
  `detmath`. `detmath` is a cdylib that compiles zs_rust's own `fmath.rs`, `num.rs` and `accel.rs` by path,
  unmodified.
- in-memory rewritten copies of the 11 lane functions that contain `X.T @ X`, `q.T @ q`, `n @ pk.T` or
  `LADDER ** k` (listed in `python_detmath_patch_record.json`).

No file under `zs_lane_v3`, `z2_self_recipe`, `z3_slicing`, `zs_lane_v2`, `d1_two_skin` or `domains/` is edited.
There were zero fallback calls to the original ufuncs.

| Comparison (DEV rev 3) | Result |
|---|---|
| Python det-math vs Rust portable, t16 | **122/122 byte-identical**, all line fields, max endpoint 0, max score 0 (`py_det_vs_portable_rev3_eval.json`) |
| Python det-math vs Rust portable, t1 | **122/122** (`rust_portable_t1_vs_python_detmath_rev3.json`) |
| Point clouds (3) | The binary refuses them on purpose (exit 10); production runs the Python fallback. Python det-math ran them (status OK) |
| Control: live Python (native math) vs frozen rev-3 DEV CLI | 125/125 identical |
| Control: Rust Accelerate build vs live Python | 122/122 identical |
| Control: Rust portable vs live Python | 10/122 identical, max endpoint 3.7e-5 mm (the known drift) |

**DEV OOF of the det-math reference.** It was not recomputed separately. A Python dev_dump run under the patch
plus `rev2_eval` would be needed. Its eval lines equal the portable build's on 122/122. The portable build's rev-2
OOF is fixed-model 0.9660 (unchanged from production) and refit 0.9643 (production 0.9651). So the det-math
reference is expected to have the same OOF, but this is not verified at the dump level.

## 3. Cross-architecture and x86-64-v3 (step 3)

- **arm64 == x86-64 generic == x86-64-v2.** Byte-identical eval lines and full stage dumps on 122/122 DEV scans,
  rev 2 and rev 3, at 1 and 16 threads. This is the previous session's run under Rosetta (`../PORTABLE.md` §4,
  `../portable_results.json`).
  - Re-run here: arm64 t1 vs t16 is 122/122 (`rust_portable_t1_vs_t16_rev3.json`).
  - **The fresh full-DEV x86 re-run was not done.** Rosetta stalled for over 30 minutes partway through this
    session (see "Blocked" below) and the run was abandoned. The x86 identity claim therefore rests on the
    2026-10-26 evidence.
- **x86-64-v3 on sc_766be04dc456** (`x86_64_v3_sc_766be04dc456.json`). Test conditions: rev 2, t1, dump mode,
  reference = pinned arm64 binary.
  - The pinned v3 binary, a fresh v3 rebuild, and a v3 rebuild with **`-C target-feature=-fma`** all differ:
    7 pooled candidates instead of 9. **Disabling FMA does not fix it.** This matches the earlier finding that any
    `+avx` build shows it.
  - A v3 build of the probe copy, with the same arithmetic but slightly different code (a dispatch layer and an
    env-gated log line in the caller), matches arm64 exactly.
  - Disassembly of the affected `min(la, lb).max(1e-9)` sequence (`vminsd` / `vcmpunordsd` / `vblendvpd` /
    `vmaxsd`) is correct under Intel semantics.
  - So the effect depends on code layout. It is not a floating-point semantics difference (FMA, contraction or
    rounding).
  - A Rosetta-2 AVX translation fault is the most likely cause, but **it cannot be proven without real x86
    hardware**. An lldb session against the Rosetta process stopped at launch and did not report registers.
  - Station guidance is unchanged: ship x86-64-v2, and clear v3 only by a full DEV replay on the station CPU.

## 4. Timing (step 4)

Conditions:
- DEV rev 3, 122 mesh scans, process wall (one process per scan, nice 15), same host and session, run serially.
- Python has OMP_NUM_THREADS=1. Python runs single-threaded, so only the 1-thread row exists for it.
- Rust Accelerate and portable builds were interleaved per scan and thread count (`timing_dev_rev3.json`, `runs/`).

| Run | Threads | P50 | P95 | Max | Sum | Peak RSS | Median per-scan speed-up vs Python |
|---|---|---|---|---|---|---|---|
| Python (production math) | 1 | 6.88 s | 18.08 s | 29.61 s | 964.8 s | 2.70 GiB | 1.00 |
| Python det-math (research) | 1 | 6.90 s | 17.85 s | 27.10 s | 940.8 s | 2.71 GiB | — |
| Rust native (Accelerate) | 1 | 2.62 s | 5.45 s | 8.14 s | 323.6 s | 1.35 GiB | 2.95× |
| Rust native (Accelerate) | 16 | 0.58 s | 1.23 s | 1.52 s | 71.9 s | 1.51 GiB | 12.97× |
| **Rust portable** | 1 | **2.63 s** | **5.30 s** | 8.51 s | 321.8 s | 1.34 GiB | **2.97×** |
| **Rust portable** | 16 | **0.57 s** | **1.25 s** | 1.49 s | 71.8 s | 1.50 GiB | **13.02×** |

Portable / Accelerate median per-scan ratio: 0.996 at 1 thread and 0.998 at 16 threads, so they are the same
speed. The +3.8 % at 1 thread in PORTABLE.md §8 was host noise.

## Blocked / caveats

- **Rosetta stalled, then recovered (correction).** x86_64 launches stalled for over 30 minutes: three processes sat
  in uninterruptible state, most likely after the killed lldb/debugserver session on a Rosetta process. They then
  finished on their own. A later `zs_rust version` under `arch -x86_64` completed and no zs_rust process is left
  running, so no admin action is needed. The full-DEV x86 re-run was abandoned during the stall and has **not**
  been redone; it can be run now.
- **zs_rust_det binary rebuilt mid-ablation.** `zs_rust_det/target/release/zs_rust` was rebuilt during the
  ablation, after the all-platform validation run, to add an env-gated log line. The arithmetic is unchanged.
- **Rebuilt x86-64-v3 binary hash differs.** A v3 rebuild into a different target dir has a different sha256 from
  the pinned one: `5d9e54c5…` vs `b1252219…`. Byte-reproducible builds across target-dir paths are not yet
  established.

## Files

- **Tools** (`tools/`):
  - `det_run.py`: Rust runner with probe env;
  - `ablate.sh`;
  - `cr_check.py`;
  - `py_run.py`: Python runner, native or det-math;
  - `timing_table.py`.
- **Code:**
  - `research/zeroshot/zs_rust_det/`: probe copy. `src/probe.rs`, `src/accel_host.rs`, dispatch in `fmath.rs` /
    `accel.rs`;
  - `research/zeroshot/zs_lane_v3_det/`: `detmath/` cdylib, `dm.py`, `patch.py`, `run.py`.
- **Run outputs:** eval lines and dumps stay in the session scratch directory (DEV-derived, not committed). The
  summaries are here.
