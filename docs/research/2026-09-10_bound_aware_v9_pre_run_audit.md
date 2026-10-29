# Bound-aware V9 focused gates — exact pre-run audit

Date: 2026-09-10. Status: independent source review GREEN; exact receipt
confirmation requested before either execution. No V9 attempt exists at freeze.
Root checked live source hashes against the prior independent reviewed hashes.
This receipt freezes both gates before the first; no common file changes between
runs. I = initializer (19 files), W = byte workflow (32 files); union 34 files.

## Scope and independent review

Reviewer `/root/cad_support_lineage` independently reviewed V9 source, tests,
protocol and harness, including complete dependency closures and prospective
62 initializer/44 workflow cases. Review closed directed coverage rounding,
selected-pose and numerical-label binding, the actual completed-candidate
`translation` key, and retained nonfinite-evidence refusal expectations before
first execution. Root integrates and executes; the reviewer does not author
runtime repairs. Review is AI technical evidence, not manufacturing authority.

Prerequisite V3 consumer attempt `bound_aware_registration_v3-xji4g3ux` passed
68/68 and retained-only post-audit GREEN; helper87 and whole-surface52 are earlier
frozen passes. None establishes physical recovery. The exact
[protocol](2026-09-10_bound_aware_v9_focused_protocol.md) defines the selected
obligations, old-positive invariants, mocked boundaries and stop conditions.

## Exact closure

| File | Gate | SHA-256 |
| --- | --- | --- |
| `backend/cad_guided_plane_initialization_v9.py` | IW | `c64ab591ee07b91d7f20cb0d663cc2bb2a98e5baa681e072f97353b01e82401d` |
| `backend/cad_guided_plane_initialization_v8.py` | IW | `cd225b4642816f76cfd4b8bd725e3b8ffa5d9f6177ddbfa9b03fb6a22128c28b` |
| `backend/cad_guided_plane_initialization_v4.py` | IW | `a4586cd02c490493472ea842622efeafe9ae18f246979ad4be364313bad72724` |
| `backend/cad_guided_global_registration_v3.py` | IW | `088c0b5e9fd940094fd44462e7e5c831299fe2c69b87418d09c362772263bae0` |
| `backend/cad_guided_global_registration_v2.py` | IW | `5a1b039d17b6d5bb5840ebb16b5ab3d131ca3df1b6c186746804327921627dc3` |
| `backend/cad_guided_global_registration_v1.py` | IW | `7700541063c4bfb40c72fb66baf2a78fea1c631e5e3f93eabbb51edfdb59eff7` |
| `backend/cad_guided_triangle_surface_v3.py` | IW | `8e3222bd425451e2e89b2fe0083db34a3702577018b3f557b1f026a86add9f10` |
| `backend/cad_guided_triangle_surface_v2.py` | IW | `7b5333e18817d30905f3a1ace1cbf1e85129ad904324c5e6ccf2baff665a5cc1` |
| `backend/cad_guided_triangle_distance_bounds_v3.py` | IW | `2404278ebd8872dafa6752d9bf977879e7d0a21ea41752ec740501b7044058d7` |
| `backend/cad_guided_mesh_plane_proposals_v4.py` | IW | `55356d900ce933b4dd4ff9fc8d055fdb46038422b69b6e05f7f9509d73f5ed48` |
| `backend/cad_guided_plane_hypotheses_v4.py` | IW | `7e69fce8bf79216342071aacfbbb7c0d73972c78334cc81d7f0fe15c831f1370` |
| `backend/cad_guided_noisy_mesh_planes_v5.py` | IW | `9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75` |
| `backend/cad_guided_noisy_mesh_planes_v7.py` | IW | `b0822289cf3abbde9c7cbf263b45c8d8ef4c4f45d7913e452c47a46ed881449f` |
| `backend/cad_guided_plane_search_v6.py` | IW | `aa2f9067c5e4e65129c0c8c42d39b6d980c3e33fadac6daf881e544648292676` |
| `backend/cad_guided_report_envelope_v6.py` | IW | `c7cc45a362c5972aff78265b2ce63fa4d41ec023b662640d7d197c23003ba28e` |
| `docs/research/2026-09-10_bound_aware_v9_focused_protocol.md` | IW | `af07fd1b9cf4c950d9fdfa95cc7b4b102137b5c9bd859e8ab115716384c276a5` |
| `scripts/check_cad_guided_measurement.py` | IW | `c4a9cf0eb698a3e66bfd4f81e7fa1b4e57b5714d8628cb5c85642351f8544ed7` |
| `tests/test_cad_guided_plane_initialization_v8.py` | I | `e78e5bc17705334b9e7dd1ebe1f70271ae1cdca0a118bbfbea24d95f633a684a` |
| `tests/test_cad_guided_plane_initialization_v9.py` | I | `7299463b28808763aabff55137758961da8d61053939b6d4f015ae5dbd859223` |
| `backend/cad_guided_mesh_offline_v9.py` | W | `82151485d6ac265e1c9d6538d7f7c2110451403c5e3119ca82506fb69d3b24af` |
| `backend/cad_guided_mesh_offline_v8.py` | W | `afdfc7f859eb5fdbf0fa7d796cc06a4180d40c02f6b0e590362f83e3f339e7ec` |
| `backend/cad_guided_mesh_offline_v2.py` | W | `fa30bc1ed575f9174414e3737eaaffecc6eafdcbcd9e3b1d68475f1af7ea8cbf` |
| `backend/cad_guided_mesh_offline_v1.py` | W | `55243b2a557e77a9c50e27a7fdef9257d60ed2468bc23c1788c391d7a836bdba` |
| `backend/cad_guided_mesh_input_v1.py` | W | `c4d936e2f60837d7c5221c9a46b64abd68c4ccee01ea72e9fd3c4050bde47917` |
| `backend/cad_guided_mesh_topology_v4.py` | W | `4d061b56a39aee6b7d9e8c04eb7af88b0573a7916d3d679c610c4aca81ee0734` |
| `backend/cad_guided_native_features.py` | W | `f8c681027af231546974633a02f12038d522c604bec14196f2ae5d49b439c99f` |
| `backend/cad_guided_native_features_v1.py` | W | `3804a583f5642df41cd3eb02be36b3f5024347b6514861d404389e85c1ccf98b` |
| `backend/cad_guided_measurement.py` | W | `d3a11bde2b14c6ece39414360e3146bf60e3801190b9e04cd6c5178ed903f2cb` |
| `backend/cad_guided_instances.py` | W | `d30b3fc8579b0a2d44d37bfe093343b3f7d703661793784d78b50efbc38eacda` |
| `backend/cad_guided_cad_surface_v2.py` | W | `b7364234a716f283f46246a452451f47c8efb07c18b2d37375c6fda0636da0d1` |
| `backend/cad_guided_cad_sampling_v1.py` | W | `f1d32159a50b455d6b9a0b5f646586f0d79e7f9c26e8fffbe67f97c5b74e181a` |
| `backend/cad_guided_local_measurement_v1.py` | W | `5b07e8c14d3eededfb7cbdb2cb628e6d631a30662280fc2d39b2528d862b541b` |
| `tests/test_cad_guided_mesh_offline_v2.py` | W | `10fdfeac837d32a4f61eff53e329fbfeb0b742fcf7d9834024819e6241bb9cfc` |
| `tests/test_cad_guided_mesh_offline_v9.py` | W | `b3bb60f5b5e5e2dceb5a1bb87e3b9290178218b169095b90f244a8f13c3ccd00` |

## Execution decision

Each gate: 60 s parent wall, 45 s worker CPU, 1 GiB sampled process-tree RSS,
1 MiB output; numeric libraries single-threaded and isolated private environment.
No native geometry/import flags, physical input, retained predictions, Gold,
email, application/database startup or other heavy job. Only reviewed pure
sources and invented fixture stubs are exercised. The actual two noisy-array
positives retain their original geometry and error limits.

After independent exact-receipt GREEN, root executes once in this order:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite plane_initialization_v9
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite mesh_offline_v9
```

The first must pass all62 tests before the second starts. Expected workflow
count is44. Every failure/skip/stop/source mismatch is retained and prevents
advancement. Exclusive STARTED/OUTPUT/COMPLETED retain before/after hashes.
No terminal rerun, cap change or post-start source/fixture changes. Independent
retained-only post-audit is required. The used harness is archived byte-identically
before a later route addition. This gate cannot authorize physical execution,
claim a Gold11 score, observed-bend count or integration promotion.

