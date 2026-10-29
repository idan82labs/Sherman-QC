# V2 guarded runtime smoke — frozen pre-run audit

Date: 2026-09-09. No physical input, Gold, inference, scoring or source freeze.

Prerequisite paired-runner pure gate89passes/1unconditional skip is independently
post-GREEN; see `2026-09-09_gold11_mesh_runner_v2_pre_run_audit.md`.
This separately reviewed smoke has31 pure controls plus one actual opt-in test.
Source gate31pass/1skip, then actual32pass are mandatory. Outer60s wall/45s
worker CPU/1GiB sampled group RSS/1MiB stdout; child20s wall/10s CPU/512MiB
sampled tree RSS/1MiB log, inherited group. One child log at most1MiB is retained
in the root-owned fresh ARTIFACTS directory. Metadata at most4KiB precedes
success assertions, including failed/absent logs. No manifest is written.

All24 copied source files are admitted before publication, per-file2MiB and
aggregate48MiB. Child imports only the copied candidate and native runtime,
constructs policy objects and one gp_Pnt, and checks tiny owned-file cleanup.
No scan/CAD parser or inspection/Gold/control entrypoint executes.
Runtime-version metadata is explicitly deferred to the separate physical freeze.

Root reviewed complete source/protocol and the additive harness delta. The
initial unexecuted draft's lost-failure-log and metadata-wording findings were
closed before this first freeze. Existing terminal attempts remain untouched.
Previous used6d589 harness is archived byte-exact at
`source_snapshots/all_generated_v8_6d58919f/check_cad_guided_measurement.py`.

## Exact27-file closure

| Frozen file | SHA-256 |
| --- | --- |
| `scripts/run_gold11_mesh_development_v2.py` | `94d51b90faef0d88b15ed2d3d84d5b7d36ba94bfb6f63f90b9cc555a35e26422` |
| `backend/cad_guided_mesh_offline_v1.py` | `55243b2a557e77a9c50e27a7fdef9257d60ed2468bc23c1788c391d7a836bdba` |
| `backend/cad_guided_mesh_input_v1.py` | `c4d936e2f60837d7c5221c9a46b64abd68c4ccee01ea72e9fd3c4050bde47917` |
| `backend/cad_guided_global_registration_v1.py` | `7700541063c4bfb40c72fb66baf2a78fea1c631e5e3f93eabbb51edfdb59eff7` |
| `backend/cad_guided_cad_sampling_v1.py` | `f1d32159a50b455d6b9a0b5f646586f0d79e7f9c26e8fffbe67f97c5b74e181a` |
| `backend/cad_guided_native_features_v1.py` | `3804a583f5642df41cd3eb02be36b3f5024347b6514861d404389e85c1ccf98b` |
| `backend/cad_guided_native_features.py` | `f8c681027af231546974633a02f12038d522c604bec14196f2ae5d49b439c99f` |
| `backend/cad_guided_local_measurement_v1.py` | `5b07e8c14d3eededfb7cbdb2cb628e6d631a30662280fc2d39b2528d862b541b` |
| `backend/cad_guided_measurement.py` | `d3a11bde2b14c6ece39414360e3146bf60e3801190b9e04cd6c5178ed903f2cb` |
| `backend/cad_guided_instances.py` | `d30b3fc8579b0a2d44d37bfe093343b3f7d703661793784d78b50efbc38eacda` |
| `backend/cad_guided_mesh_offline_v2.py` | `fa30bc1ed575f9174414e3737eaaffecc6eafdcbcd9e3b1d68475f1af7ea8cbf` |
| `backend/cad_guided_mesh_offline_v8.py` | `afdfc7f859eb5fdbf0fa7d796cc06a4180d40c02f6b0e590362f83e3f339e7ec` |
| `backend/cad_guided_cad_surface_v2.py` | `b7364234a716f283f46246a452451f47c8efb07c18b2d37375c6fda0636da0d1` |
| `backend/cad_guided_global_registration_v2.py` | `5a1b039d17b6d5bb5840ebb16b5ab3d131ca3df1b6c186746804327921627dc3` |
| `backend/cad_guided_triangle_surface_v2.py` | `7b5333e18817d30905f3a1ace1cbf1e85129ad904324c5e6ccf2baff665a5cc1` |
| `backend/cad_guided_mesh_topology_v4.py` | `4d061b56a39aee6b7d9e8c04eb7af88b0573a7916d3d679c610c4aca81ee0734` |
| `backend/cad_guided_plane_initialization_v4.py` | `a4586cd02c490493472ea842622efeafe9ae18f246979ad4be364313bad72724` |
| `backend/cad_guided_plane_initialization_v8.py` | `cd225b4642816f76cfd4b8bd725e3b8ffa5d9f6177ddbfa9b03fb6a22128c28b` |
| `backend/cad_guided_mesh_plane_proposals_v4.py` | `55356d900ce933b4dd4ff9fc8d055fdb46038422b69b6e05f7f9509d73f5ed48` |
| `backend/cad_guided_plane_hypotheses_v4.py` | `7e69fce8bf79216342071aacfbbb7c0d73972c78334cc81d7f0fe15c831f1370` |
| `backend/cad_guided_noisy_mesh_planes_v5.py` | `9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75` |
| `backend/cad_guided_noisy_mesh_planes_v7.py` | `b0822289cf3abbde9c7cbf263b45c8d8ef4c4f45d7913e452c47a46ed881449f` |
| `backend/cad_guided_plane_search_v6.py` | `aa2f9067c5e4e65129c0c8c42d39b6d980c3e33fadac6daf881e544648292676` |
| `backend/cad_guided_report_envelope_v6.py` | `c7cc45a362c5972aff78265b2ce63fa4d41ec023b662640d7d197c23003ba28e` |
| `tests/test_gold11_mesh_guard_smoke_v2.py` | `c56623967119a201b2c4db296f3e950d728642e48f0b28ce8c0a575b438d8e3d` |
| `docs/research/2026-09-09_gold11_mesh_guard_smoke_v2_protocol.md` | `a3c42812eee1fe660f13fba392cfc2d609b12c3ada748add98bfb03b1c9ce736` |
| `scripts/check_cad_guided_measurement.py` | `4f2a778150fa106d7e9c8155333a12d010eace0020b2e9c39f9836b7be8cce19` |

## Independent release

Independent reviewer `/root/astra_repo_guidance_review` fully read all412 test
lines and150 protocol lines, checked source/supervision/guard callees and the
additive harness, and independently verified all24 literal runtime-source hashes.
The ordered leaf closure plus test/protocol/harness exactly matches27 files.
Both pre-run findings are closed: child logs survive failure in fresh0700
retained ARTIFACTS; bounded metadata precedes assertions; runtime metadata is
not falsely claimed. Pure31+1skip and actual32 denominators reconcile.
Decision: GREEN FOR ONE FIRST SOURCE31-PASS/1-SKIP GATE under60/45/1GiB/1MiB.
ONE actual32-case route is conditionally GREEN only after that prerequisite,
with no27-file drift and the unchanged same-group20/10/512MiB/1MiB child limits.
No inspection, raw, physical/control/Gold, retry or production release follows.
Any missing case, failure, stop or source drift remains terminal negative.

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite gold11_mesh_guard_smoke_v2_source
```

## Source prerequisite and conditional actual dispatch

`gold11_mesh_guard_smoke_v2_source-_a5ek9hf`:31 passed,1declared actual skip.
Parent0.15441545797511935s,workerCPU0.126306s,sampled48,807,936B,
workerpeak48,791,552B. No60/45/1GiB/1MiB stop; all27 frozen before/after hashes
match. OUTPUT`0e051214e3664e0a9b5a2cfe968b80251bfe7a17535fa1d5796dcb9807bd358c`.
Root verified the reported exact prerequisite, satisfying the independent
conditional release above. Dispatch one actual smoke under unchanged limits.

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite gold11_mesh_guard_smoke_v2
```

## Terminal actual result

`gold11_mesh_guard_smoke_v2-ttmsddml`:32 passed,0 skipped. Parent0.8728012498468161s,
pytest workerCPU0.157824s,workerpeak49,315,840B,complete sampled group395,116,544B.
Actual childwall0.7038477919995785s,sampled child345,833,472B,exit0,no stop.
The worker CPU number excludes child CPU; child10s RLIMIT was enforced but its
actual CPU consumption was not separately published. No invented aggregate CPU.
All27 source hashes match before/after, no outer60/45/1GiB/1MiB stop.
24 source copies total556,011B. Actual child log is retained at
`ARTIFACTS/SMOKE_OUTPUT.txt`,133B,
SHA`253fb233df3fd9dc18835b2d60586052dc20c9b4713a60ae9bc856c085c5f38a`.
Outer OUTPUT SHA`1da31e0502793897c8c34fad8f0eb09609ff5b57ea0e911952953070e6b98ea6`.

The same133B success payload also appears in a deliberately mocked source test;
the actual proof is the separate actual-child supervision receipt with resource
samples and the retained file, not that shared literal/hash alone. Independent
post-check verified both exact27-file closures, output hashes, actual sampled
receipt,24-source denominator and retained childlog: POST-AUDIT GREEN FOR
RUNTIME-IMPORT COMPATIBILITY ONLY. No physical execution is released by this
record alone; a separate exact-bundle pre-run audit remains mandatory.
