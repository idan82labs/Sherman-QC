# zs_rust made portable: one deterministic binary for macOS arm64, x86-64 and the Windows station

**Date:** 2026-10-26. **Status:** research build; the binary is now the reference for zs_lane_v3 (owner: "station is Windows"). Nothing was installed into production. `domains/production`, `contracts`, `lane_cli`, `pin.py` and the `zs_lane_v3` Python package were not edited. No commits. Numbers: [portable_results.json](portable_results.json). Predecessor: [PORT.md](PORT.md) (the Accelerate build `567101d3…`).

## Verdict

| Question | Answer |
|---|---|
| **Platform dependence** | Removed. No Accelerate, no CommonCrypto, no host libm: every transcendental goes through the pure-Rust `libm` crate, the linear algebra is hand-ported reference LAPACK, fused multiply-adds are explicit only, SHA-256 is pure Rust. The arm64 binary imports nothing but libSystem's process / thread / file calls (`nm -u`: no math symbols); the x86-64 builds import no math symbols either. |
| **Drift against Python / Accelerate** | Small numeric drift, **no change in any line set and no change in the production-faithful OOF**. DEV, both revisions: 125/125 line sets equal under the frozen matcher, same line count / src / op / fix on every scan; max endpoint difference 1.6e-6 mm (rev 2) and 3.7e-5 mm (rev 3); max score difference 0.077 (one line). DEV OOF with the fixed v3 fold models: **F1 0.9660, 597/20/22, control FP 1: unchanged**. The *refit* OOF arm moved by one line: 0.9651 → 0.9643 (595/19/24 → 594/19/25). Dense meshes (rev 3): 18/18 same lines, max endpoint 7.1e-6 mm, scores identical. |
| **Cross-architecture identity** | arm64 vs x86-64 under Rosetta: **byte-identical on all 125 DEV scans (122 run by the binary, 3 point clouds refused on both) and all 18 dense meshes, at 1 and 16 threads, eval lines and full stage dumps**, for the generic x86-64 build and the x86-64-v2 build (section 4). The DEV OOF computed from the x86-64 dump is byte-identical to the arm64 one. **Exception: the x86-64-v3 (AVX2/FMA) build differs on 1 of 122 DEV scans** (sc_766be04dc456) under Rosetta; see section 4.3. The station script therefore defaults to x86-64-v2. |
| **Windows build** | `cargo check` passes for `x86_64-pc-windows-msvc` and `x86_64-pc-windows-gnullvm` and all code generates; **linking needs the MSVC Build Tools + Windows SDK (licence) or an llvm-mingw toolchain (download)**, so it stopped there as instructed. The station builds it with `station/build_install.ps1`; the golden self-test decides there. Not executed on Windows. |
| **Station self-check** | `zs_rust selftest --golden DIR --config CFG` (≈1.5 s): numeric primitives, 4 bundled synthetic meshes, the whole lane at 1 and N threads, digests compiled into the binary. Exit 0 pass / 3 refuse. Passes on every Mac build for both revisions; refuses a mesh moved by 1 µm and an unknown config. |
| **Speed** (rev 3, same host, interleaved with the Accelerate build) | DEV median per-scan ratio portable / Accelerate: **1.04 at 1 thread, 0.95–0.96 at 4/8/16 threads**. Dense: 0.99 / 0.97 / 0.95 / 0.95. Peak RSS unchanged (≤ 1.5 GiB DEV, ≤ 2.3 GiB dense). |

## 1. What changed in the crate (`research/zeroshot/zs_rust`)

| Before (Accelerate build) | Now (portable) |
|---|---|
| `accel.rs`: FFI to Accelerate `dsyrk`, `dgemm`, `dsyevd`, `dgesv`, `dgetrf`, `dgelsd` | `accel.rs` in pure Rust. `X.T@X`: numpy pairwise sums. `A@B.T`: index-order dot products. `eigh`: reference LAPACK 3.9 `dsyevd('V','L')` path for n ≤ 25 (`dsytd2` → `dsteqr('I')` → `dorm2r`, with the 3.9 `dlartg`, `dlapy2`, `dlaev2`, `dlasr`, old `dnrm2`, `dlascl`). `solve`: `dgetrf2` (recursive, reciprocal scaling) + `dgetrs`. `det`: `dgetrf2` + umath_linalg's sign·exp(Σ log|u_ii|). `lstsq`: minimum-norm solution by one-sided Jacobi SVD with dgelsd's `rcond` truncation. |
| `f64::sin/cos/tan/acos/atan2/exp/ln/hypot/powf` (host libm), `powi` | `fmath.rs`: trait methods `dsin … dpowf` on the `libm` crate 0.2.16 (pure Rust musl port; its arch module only swaps exact ops: sqrt, fma, rint). `powi(2)` → `x*x`. `%` → `libm::fmod`. |
| `mul_add` (hardware or host `fma`) | `dfma`: hardware FMA when the target has it (aarch64; x86 with the `fma` feature), else `libm::fma` (run-time-detected hardware or software). IEEE fma is correctly rounded, so every path gives the same bits. The compiler never contracts. |
| CommonCrypto SHA-256 | `sha256hw.rs` (Armv8 SHA2 instructions via `std::arch`, run-time detected) on aarch64, else the `sha2` crate (SHA-NI on x86 when present). Same digest; checked against each other in the self-test and a unit test. |
| `getrusage` FFI, macOS layout | `cfg(unix)` getrusage (bytes on macOS, KiB elsewhere) and `cfg(windows)` `K32GetProcessMemoryInfo`. Reporting only, never part of an output. |
| no mmap | unchanged: reads are `std::fs` (no mmap or OS-specific I/O in the crate). |
| — | `golden.rs` + `golden/` (section 5); `selftest --golden`; `version --build` prints the math build. |

Parallel reductions were already in fixed order (PORT.md section 3); float reductions are never auto-vectorised by rustc (no reassociation), so there is no target-dependent SIMD reduction. Dependencies: rayon, serde, serde_json, sha2 and **libm 0.2.16** (the only new crates.io fetch). `Cargo.lock` sha256 `56a44631…31d51`. 11 988 lines of Rust.

## 2. Drift against Python / Accelerate (all reported, nothing tuned)

The Python reference is the frozen DEV CLI output (`v3/rev2/dev_cli`, `v3/rev3/dev_cli`); for the dense meshes it is the Accelerate build, which PORT.md proved byte-identical to Python on 18/18. Portable arm64 binary, 16 threads.

| Check | Revision 2 | Revision 3 |
|---|---|---|
| DEV scans run by the binary | 122 (+3 point clouds: Python fallback, not the binary) | 122 (+3) |
| Byte-identical line lists | 10/122 | 10/122 |
| Line count, status, `src`, `op`, `fix` changed | 0 scans | 0 scans |
| `parts` changed | 1 line (sc_3baf989b1612 ZS009: candidate index 9 → 10, two coincident candidates swapped order; same geometry to 6e-14 mm) | same |
| Scans whose `incl` differs in the last bits | 88 (≈1e-14 deg, one acos ulp) | 87 |
| Max endpoint difference | **1.6e-6 mm** (budgeted scan sc_147d3123a21d) | **3.7e-5 mm** (budgeted sc_147d3123a21d; rev-3 QEF solve) |
| Max score difference | **0.077** (sc_3baf989b1612, the swapped line); next 0.0071, 0.0027, 0.0017; 10 scans > 1e-9 | same |
| Frozen matcher (5°/4 mm/0.6, Hungarian) per scan: disposition, TP/FP/FN, matched pairs | **125/125 equal** (in-sample, full models: 600/16/19 both) | **125/125 equal** |
| DEV OOF, fixed v3 fold models (production-faithful) | **F1 0.9660, macro 0.9534, 597/20/22, control FP 1: unchanged** | n/a |
| DEV OOF, refit arm (retrains the GBMs on the dump) | **changed: 0.9651 → 0.9643**, 595/19/24 → 594/19/25, macro 0.9524 → 0.9511 | n/a |
| Dense meshes vs the Accelerate build (= Python), 1 and 16 threads | n/a | **18/18 same lines**, max endpoint 7.1e-6 mm, max score 0.0 |

The score moves are GBM tree splits: a last-bit change in a feature that sits on a split threshold changes the leaf. The refit arm retrains the models on the drifted feature matrix, so its trees (and one prediction) change; the production lane uses the fixed models, whose OOF did not change. The OOF output files themselves are not byte-identical to the Accelerate ones (the fixed arm's line coordinates carry the same ulp drift); `OOF_v3r2_*` of the portable run are in the scratch work dir, the summary is in portable_results.json.

**Where the drift comes from (ablation, DEV rev 2, 16 threads):**

| Variant | Byte-identical line lists | Max score diff | Refit OOF |
|---|---|---|---|
| libm only (Accelerate kept) | 43/122 | 1.1e-16 | 0.9651 (unchanged) |
| pure linear algebra only (host libm kept) | 10/122 | 0.077 | 0.9643 |
| both (the portable build) | 10/122 | 0.077 | 0.9643 |

The linear algebra carries the material drift. A scratch probe (3000 synthetic cases each, not shipped) shows why: Accelerate's kernels use fused multiply-adds and undocumented blocked orders. `X.T@X` equals a sequential FMA chain exactly for ≤ 33 rows and nothing tried (2–32 accumulators, k-blocks) above 64 rows; the 3×3 `dsyevd` port agrees with Accelerate on eigenvector signs in 2999/3000 cases but never in all last bits. Reproducing Accelerate bit for bit would mean reverse-engineering its kernels; the reference-LAPACK algorithms keep its conventions (signs, pivoting, ordering), which is why no line set changed.

## 3. Determinism

Byte identity (eval lines without timestamps / wall times, and the full stage dump without timings) at 1 vs 16 threads: 122/122 DEV and 18/18 dense on arm64 and on x86-64. Unchanged from PORT.md: every parallel step is disjoint or merged in a fixed order.

## 4. Cross-architecture identity (`tools/portable_identity.py`)

### 4.1 Builds

| Target | Flags | sha256 | Bytes | Golden self-test |
|---|---|---|---|---|
| aarch64-apple-darwin (reference) | generic | `5767bf697884a60a240442cbe684db0bf6b385f490582dbbaf01cb0f3ea56478` | 2 037 760 | pass |
| x86_64-apple-darwin | generic (SSE2; `libm::fma`) | `a7a738b294b678b1631af81109f2f87c7082d9c14f839a385f1a932f626301e9` | 2 324 592 | pass (Rosetta) |
| x86_64-apple-darwin | `-C target-cpu=x86-64-v2` | `ab65b3b3ef4b4c581427cb2d0c827f8fa5430c82652d212fff1e3d3b11ce88fd` | 2 353 360 | pass (Rosetta) |
| x86_64-apple-darwin | `-C target-cpu=x86-64-v3` | `b1252219a968b4e47ce6a84ba561c499f1351c133b5065c0050fdb16a8d7fb01` | 2 396 896 | pass (Rosetta), **but see 4.3** |
| x86_64-pc-windows-msvc | x86-64-v2 (station default) | built on the station | | run by the install script |

### 4.2 Results (x86-64 under Rosetta 2 on the Apple M5 Max)

| Comparison | DEV rev 3 (125) | Dense rev 3 (18) | DEV rev 2 (125) |
|---|---|---|---|
| arm64 t1 vs t16 | 122/122 eval + dump identical, 3 point clouds on both | 18/18 | |
| x86-64 generic t16 vs arm64 t16 | **122/122 identical** | **18/18** | **122/122**; OOF files byte-identical to arm64 |
| x86-64-v2 t1 vs arm64 t1, t16 vs t16 | **122/122 identical at t1 and t16** (v2 t1 vs t16 also 122/122) | **18/18 at t1 and t16** | **122/122** (t16); OOF files byte-identical to arm64 |
| x86-64-v3 t1 vs arm64 t1, t16 vs t16 | 121/122 (sc_766be04dc456 differs) | **18/18** | 121/122 (same scan); the OOF moves (fixed arm 597/19/22, F1 0.9668) |
| x86-64-v3 t1 vs t16 | 122/122 (deterministic, consistently different on that scan) | 18/18 | |

### 4.3 The x86-64-v3 difference

On sc_766be04dc456 (no Z3 candidates, 9 Z2 lines) the AVX build pools 7 candidates instead of 9: two Z2 lines are merged as duplicates because their overlap ratio comes out ≈1e11 instead of ≈0.35, i.e. the divisor `la.min(lb).max(1e-9)` evaluated to 1e-9. Facts established:

- Any build with `+avx` (even without FMA/AVX2) reproduces it; SSE2 and SSE4.2 builds (and arm64) do not.
- The compiled code is `vshufpd; vminsd; vcmpunordsd; vblendvpd` (the NaN-aware `f64::min`), then `vmaxsd` with 1e-9. The same instruction sequence in isolation (inline asm) evaluates correctly under Rosetta, and adding a print or calling `pool()` outside `lane::run` changes the code and removes the effect.
- The source is plain IEEE arithmetic; the pairs are far from every threshold (the correct ratio is 0.35 vs 0.5).

So it is either Rosetta 2's AVX translation or an AVX code-generation fault; **it cannot be decided without real x86 hardware**. The golden set does not trigger it. Consequence: the station build defaults to **x86-64-v2** (SSE4.2, no AVX), which is the code path proven identical here; FMA hardware is still used through `libm::fma`'s run-time detection. x86-64-v3 is available (`-TargetCpu x86-64-v3`) only after it reproduces the arm64 reference on the station's own CPU on a full DEV replay (section 6).

### 4.4 Windows

| Step | Result |
|---|---|
| `rustup target add x86_64-pc-windows-msvc x86_64-pc-windows-gnullvm` | done (toolchain only) |
| `cargo check --release --locked --target …` | pass, both |
| build, msvc, default linker | `link.exe` not found |
| build, msvc, `-C linker=rust-lld` | lld-link: no `kernel32.lib`, `ntdll.lib`, `userenv.lib`, `ws2_32.lib`, `dbghelp.lib` → needs the Windows SDK + MSVC CRT (licence acceptance): **stopped** |
| build, gnullvm, default linker | `x86_64-w64-mingw32-clang` not found |
| build, gnullvm, rust-lld (`ld.lld` flavour) | no `-lkernel32 -lmingw32 -lmingwex -lmsvcrt -lunwind …` (llvm-mingw, a download): **stopped** |

Nothing in the code is OS-specific beyond the RSS report (`cfg(windows)` FFI to kernel32, compiled by `cargo check`).

## 5. Station self-check: `selftest --golden`

```
zs_rust selftest --golden <dir> --config <config.json> --repo <root> [--threads N] [--report out.json]
```

1. **Primitives**, digested bit for bit over fixed pseudo-random inputs: the libm functions, f64/f32 fma and sqrt, eigh 2×2/3×3, solve, det, lstsq, `X.T@X`, numpy PCG64 / choice and Python float repr, and hardware SHA-256 against the sha2 crate.
2. **Mesh files** `golden/`: `g1_channel.ply` (binary float32, 2 bends), `g2_zprofile_rot.ply` (binary float64, rotated, bends 60° and 120°), `g3_hat.ply` (ASCII, 4 bends), `g4_channel_soup.stl` (STL, read as a triangle soup). Synthetic only, made by `golden/make_golden_meshes.py` (numpy, fixed seeds). 2.3 MB.
3. **Lane** on each mesh plus a forced face budget on g1 (revision 2 vs 3 budget), at 1 thread and at N threads: digest of the eval lines, budget record and the whole stage dump (Z2 lines, candidates, X, evidence with split runs, both GBM probabilities).

Expectations (`golden/golden.json`, sha256 `57790e43…`) are compiled into the binary, keyed by the config's lane-parameter digest (variants.zs, budget, z3_params, z2_fallback) and the model sha256s, so a config or model change is reported as that, not as drift. Exit 0 = pass, 3 = refuse (failures listed: `PRIMITIVE:*`, `MESH_FILE:*`, `MODEL_FILES`, `NO_GOLDEN_FOR_CONFIG_PARAMS:*`, `LANE:*`, `THREADS:*`). Runtime ≈1.3–1.6 s, ≈110 MB.

Verified: passes for both configs at 1+1 and 1+16 threads on arm64, x86-64 generic, x86-64-v2 and x86-64-v3; refuses (exit 3) a golden mesh with one vertex moved 1 µm (`MESH_FILE:g3_hat.ply` + `LANE:g3_hat`) and a config with a changed threshold (`NO_GOLDEN_FOR_CONFIG_PARAMS`). Limits: it guards the toolchain/CPU/math layer and file integrity; it did not catch the x86-64-v3 effect of 4.3, which is why v3 is not the default.

## 6. Windows station: build, install, start (`research/zeroshot/zs_rust/station/build_install.ps1`)

Prerequisites on the station: rustup with the pinned toolchain (rustc 1.99.0), the `x86_64-pc-windows-msvc` target, Visual Studio Build Tools + Windows SDK (licence accepted by the owner), the crates (`cargo fetch` with network, or `cargo vendor` from this repo). Not executed here: no Windows host and no PowerShell on the Mac, so the script is reviewed but unrun.

- **Install** (`-Mode Install`, default `-Revision 3 -TargetCpu x86-64-v2`): checks rustc, `Cargo.lock`, models, config and `golden.json` against the pin record → `cargo build --release --locked --target x86_64-pc-windows-msvc` with `RUSTFLAGS="-C target-cpu=x86-64-v2 -C link-arg=/Brepro -C link-arg=/PDBALTPATH:%_PDB%"` (deterministic PE timestamp, no build path in the binary) → checks the unsigned sha256 against the pin when one is recorded → checks the CPU runs it → stages `bin\zs_rust.exe`, config, models, pin, golden set in a repo-relative layout → optional signing → `selftest --golden` at 1 thread and at the physical-core count → writes `install_record.json` (every installed file's sha256) → swaps the staged tree into place. Any failure: exit 3, binary renamed `*.refused`, nothing installed.
- **Signing note.** `signtool sign /fd SHA256 /tr <TSA> /td SHA256 /sha1 <thumbprint>` changes the file, so the record keeps both `sha256_unsigned` (reproducible build output, the pin) and `sha256_signed` (what runs); the self-test runs after signing. No entitlements or network are needed.
- **Startup** (`-Mode Startup`): re-hashes every installed file against `install_record.json`, then `selftest --golden`; exit 0 lets the lane run, anything else refuses it.
- **First station build.** The Windows sha256 is unknown until the station builds it; the script records it and prints the line to put into the pin (`binaries["x86_64-pc-windows-msvc+x86-64-v2"].sha256`). Recommended before first production use: replay DEV on the station (the `portable_run.py` + `portable_identity.py` pair) against the arm64 outputs; that is also the only way to clear x86-64-v3.
- The runner integration (`lane_cli` reading the pin, `ZS_RUST_THREADS` = physical cores, exit 10 policy) remains the later step of PORT.md section 7.

## 7. Re-freeze: configs and pin records

- `freeze_config` re-run for both revisions: `research/zeroshot/zs_lane_v3_rs/config_v3_rs.json` and `config_v3r3_rs.json` now pin the portable arm64 binary, `Cargo.lock`, the crate sources **and the golden set** (`fingerprint.py` covers `golden/*`), and record `rust.portable`, `rust.math`, `rust.targets` (sha per built target) and `rust.golden`. Revision 2: config sha256 `a4e0a644…`, code fingerprint `96dbde32…`, base `ab31d061…` / `1c3f8cb7…`. Revision 3: the parallel re-pin rewrote `research/zeroshot/zs_lane_v3/config_v3.json` during this work (sha `9b173ea8…`, code fingerprint `dc23a905…` → `49a16399…`; only the four re-pinned production files changed, the lane parameters are byte-equal), so `freeze_config` now accepts that sha as revision 3 (`REV3_REPINNED`) and the config records base `49a16399…` / `9b173ea8…`; code fingerprint `dd230cd0…`. Both pass `verify_pins` (empty mismatch list) and the golden self-test. Both still pin the **arm64** binary: on the station, the pin record and `install_record.json` are the authority, not the Mac adapter fingerprint.
- Pin records in the PORT.md section 7 format, schema `sherman.zs_lane_v3_rs.pin.v1`: `research/zeroshot/zs_lane_v3_rs/pins/zs_lane_v3_rs.rev2.pin.json` and `.rev3.pin.json` (`make_pins.py`). They carry the station target (`x86_64-pc-windows-msvc+x86-64-v2`, sha256 to be recorded at the first station build), every built binary's sha256, rustc 1.99.0, `Cargo.lock` `56a44631…`, the models (`zs` `0489df88…`, `z3` `8dfbcfbf…`, unchanged), the config sha and fingerprint, the golden set and the equivalence summary. sha256: rev 2 `3f43c22f…`, rev 3 `80aec83f…`. Target status recorded per binary (x86-64-v3: "NOT cleared").
- **Caveat (not caused here):** the adapter fingerprint also hashes four live production files that the parallel re-pin edits (`contracts/production/line_job.py`, `domains/production/engine/inspect/{config,engine,metrology}.py`; they no longer match revision 2's frozen fingerprint). Any later edit there makes `--strict-pins` refuse again until `freeze_config` is re-run. The binary pin (binary, Cargo.lock, models, golden) does not depend on them; separating the two is recommended for the integration step. The point-cloud Python fallback now runs with those re-pinned files.
- The Accelerate build is kept at `research/zeroshot/zs_rust/target/accelerate_567101d3/zs_rust` for comparison; its old configs/fingerprints in PORT.md are historical.

## 8. Speed (revision 3; the same host; portable and Accelerate builds interleaved per scan)

Host load was higher than during PORT.md (a parallel agent was running), so absolute times are 1.3–1.6× PORT.md's; the ratios are the measurement.

**DEV, 122 mesh scans, process wall:**

| Threads | Accelerate P50 / P95 / max | Portable P50 / P95 / max | Median ratio portable/Accelerate | Sum ratio | Peak RSS (P / A) |
|---|---|---|---|---|---|
| 1 | 3.72 / 7.63 / 15.0 s | 3.88 / 8.21 / 15.5 s | **1.038** | 1.04 | 1.26 / 1.24 GiB |
| 4 | 1.54 / 3.34 / 5.01 s | 1.41 / 3.18 / 4.80 s | 0.947 | 0.95 | 1.35 / 1.38 GiB |
| 8 | 1.14 / 2.79 / 3.52 s | 1.10 / 2.64 / 3.52 s | 0.955 | 0.96 | 1.30 / 1.44 GiB |
| 16 | 0.97 / 2.38 / 3.53 s | 0.95 / 2.49 / 3.34 s | 0.961 | 0.96 | 1.48 / 1.40 GiB |

**Dense, 18 meshes (3.8–46 M faces):**

| Threads | Accelerate P50 / P95 / max | Portable P50 / P95 / max | Median ratio | B_n7 (46 M) A / P |
|---|---|---|---|---|
| 1 | 9.28 / 17.5 / 18.7 s | 9.26 / 11.5 / 18.3 s | 0.987 | 10.7 / 9.76 s |
| 4 | 3.81 / 5.91 / 7.13 s | 3.71 / 7.20 / 7.48 s | 0.974 | 4.82 / 4.19 s |
| 8 | 2.80 / 4.85 / 5.64 s | 2.62 / 4.66 / 5.55 s | 0.948 | 3.12 / 2.93 s |
| 16 | 2.45 / 4.92 / 5.68 s | 2.20 / 5.07 / 5.52 s | 0.954 | 2.64 / 2.77 s |

Peak RSS dense: ≤ 2.33 GiB both.

**The one regression, 1 thread on DEV (+3.8 % median, +18 s over 470 s):** the stage sums put most of it in stages whose code did not change — `z2.knn_query` +4.8 s, `z2.thickness_rays` +4.2 s, `bvh_build` +3.4 s (kd-tree and ray casting use only arithmetic and explicit FMA) — and only +2.3 s in `z2.graph` and +1.3 s in `z2.segment`, where the pure-Rust lstsq/eigh replace Accelerate. So about a quarter of it is the linear algebra (the one-sided Jacobi lstsq does a few sweeps where dgelsd did one factorisation) and the rest is code layout / host noise of the same size as the 4–5 % *gains* at 4–16 threads (Accelerate's per-call overhead for 3×3 problems is gone; the Armv8 SHA path replaces CommonCrypto at the same speed). Under the 5 % noise floor of this shared host; not acted on.

## 9. Files and how to reproduce

- Crate: `research/zeroshot/zs_rust/` — `src/fmath.rs`, `src/accel.rs` (rewritten), `src/sha256hw.rs`, `src/golden.rs` (new), `src/num.rs`, `src/lane.rs`, `src/main.rs` and the math call sites (edited), `Cargo.toml` (`libm`, no `accelerate` feature), `Cargo.lock`, `golden/` (meshes, `golden.json`, generator), `station/build_install.ps1`.
- Adapter: `research/zeroshot/zs_lane_v3_rs/` — `fingerprint.py` (golden files, target table), `freeze_config.py` (portable fields), `make_pins.py` and `pins/` (new), both configs re-frozen.
- Tests: `tests/test_prod_zs_rust.py` (section 10).
- This folder: `PORTABLE.md`, `portable_results.json`; tools `portable_run.py` (runner, guard + custody, interleaved binaries / threads, optional Rosetta), `portable_identity.py`, `portable_matcher.py`, `portable_dump_prep.py`, `portable_results.py`.
- Run outputs (eval lines, dumps, records) stay in the session scratch directory; they are derived from DEV scans and are not committed.

Build: `cargo build --release --locked` (arm64); `cargo build --release --locked --target x86_64-apple-darwin` (generic); `CARGO_TARGET_DIR=target/x86_64-v2 RUSTFLAGS="-C target-cpu=x86-64-v2" cargo build --release --locked --target x86_64-apple-darwin`. Golden expectations: `zs_rust selftest --golden golden --config <cfg> --repo <root> --write golden/golden.json` for each config, then rebuild (the file is compiled in).

## 10. Tests (`tests/test_prod_zs_rust.py`)

18 tests, **18 passed** (and 3 crate unit tests: eigh / solve / lstsq reconstruction, sha256hw vs sha2). Exact against numpy/scipy/open3d: RNG + argsort, cKDTree kNN with ties, ray casts, mesh readers. Within the measured drift against the frozen Python: rev-2 and rev-3 budget (same faces, vertices ≤ 1e-9 mm; synthetic max 1.5e-12), GBM exports (≤ 4 ulp; seen ≤ 2), Z3 / Z2 (≤ 1e-6; seen 3.3e-10 / 2.8e-14), whole lane (same lines and discrete fields, geometry ≤ 1e-6, score ≤ 0.1). Byte-identical 1 vs 4 threads (rev-3 budget, lane, dump). New: golden self-test passes for both configs; refuses a 1 µm mesh edit and an unknown config; x86-64 generic and v2 builds pass it under Rosetta (v3 deliberately not tested as a clearance); no host math / Accelerate / CommonCrypto in the sources. The rev-2 fingerprint test now checks the revision-2 bytes only for `zs_lane_v3` files, because the parallel re-pin changed the four production files that revision 2 also hashed.

## 11. Not verified

- Execution on Windows and on a real x86-64 CPU (all x86 evidence is Rosetta 2). The PowerShell script was not run (no Windows / pwsh here).
- The root cause of the x86-64-v3 difference (Rosetta vs compiler).
- Reproducibility of the Windows sha256 across station rebuilds (`/Brepro` is set; unproven).
- DEV only: no FRESH, DEV-CHECK, TEST-B, HOLDOUT10, EHO4 or sealed data were run or scored; N1 was not run.
