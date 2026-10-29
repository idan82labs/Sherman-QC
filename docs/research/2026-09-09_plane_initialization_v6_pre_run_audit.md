# V6 initializer independent pre-run gate

Date:2026-09-09. GREEN for one43-case invented-array gate.

## Exact28-file source/provenance closure

| File | SHA-256 |
| --- | --- |
| `backend/cad_guided_global_registration_v1.py` | `7700541063c4bfb40c72fb66baf2a78fea1c631e5e3f93eabbb51edfdb59eff7` |
| `backend/cad_guided_global_registration_v2.py` | `5a1b039d17b6d5bb5840ebb16b5ab3d131ca3df1b6c186746804327921627dc3` |
| `backend/cad_guided_mesh_plane_proposals_v4.py` | `55356d900ce933b4dd4ff9fc8d055fdb46038422b69b6e05f7f9509d73f5ed48` |
| `backend/cad_guided_noisy_mesh_planes_v5.py` | `9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75` |
| `backend/cad_guided_noisy_mesh_planes_v6.py` | `1f5fe2fd447c445efd8c0e12f9947bfe484126fd15b2fe1be1149f2d5baf2f72` |
| `backend/cad_guided_noisy_mesh_planes_v7.py` | `b0822289cf3abbde9c7cbf263b45c8d8ef4c4f45d7913e452c47a46ed881449f` |
| `backend/cad_guided_plane_hypotheses_v4.py` | `7e69fce8bf79216342071aacfbbb7c0d73972c78334cc81d7f0fe15c831f1370` |
| `backend/cad_guided_plane_initialization_v4.py` | `a4586cd02c490493472ea842622efeafe9ae18f246979ad4be364313bad72724` |
| `backend/cad_guided_plane_initialization_v5.py` | `c37d2bfa916914c64af74e9edd5638756131914600222c63fe8bf77cfd71d2ee` |
| `backend/cad_guided_plane_initialization_v6.py` | `8845ce139102da1cd2c3f310ed80c37ffb955c1435ee64581ad83259a86c3275` |
| `backend/cad_guided_plane_search_v5.py` | `13898d4d736ba99be951ad228e19d47d8848082001786240a0b8a212d6ab9b98` |
| `backend/cad_guided_report_envelope_v6.py` | `c7cc45a362c5972aff78265b2ce63fa4d41ec023b662640d7d197c23003ba28e` |
| `backend/cad_guided_triangle_surface_v2.py` | `7b5333e18817d30905f3a1ace1cbf1e85129ad904324c5e6ccf2baff665a5cc1` |
| `docs/research/2026-09-09_mesh_plane_proposals_v4_protocol.md` | `2ca3469de02e83ca3fcb0a891b96df494edc95010f8a134029dece3d7c80e479` |
| `docs/research/2026-09-09_noisy_mesh_evidence_v6_protocol.md` | `f4d2c0985c19fc26b39ab213e4088e491d90332110f6612fc3581c0abdf61ae6` |
| `docs/research/2026-09-09_noisy_mesh_evidence_v7_protocol.md` | `37739668e75ab437b67398a334e463a2d81a6b42d37bfb98fd27af02321e83d0` |
| `docs/research/2026-09-09_noisy_mesh_planes_v5_protocol.md` | `039b6a3f46e170947039ee32a8ce871f768aea781b369b244fc0e96a9965b71b` |
| `docs/research/2026-09-09_noisy_plane_registration_design_notes.md` | `03e014332e4a9072f631dd449c6c6f55710fba175dbe19af6e7349eed7562e52` |
| `docs/research/2026-09-09_plane_hypotheses_v4_design.md` | `aa81a52ea7329a91623e757432f49febb5de6c7fea037379b371a46a3b2cd591` |
| `docs/research/2026-09-09_plane_initialization_connected_v4_protocol.md` | `f5463d6f94470153daca3035a1b84f7f35284ee2ce31ff0e5661cee2f9aaea82` |
| `docs/research/2026-09-09_plane_initialization_v4_source_protocol.md` | `3e000cb99a248195050aef78c2466f923326b70efe6e0c05a31abcf5e6f6eb6e` |
| `docs/research/2026-09-09_plane_initialization_v5_protocol.md` | `593833164918b8dbc555a2b1620c5cc53681f1be5b56cc5d17eec4fba653fb60` |
| `docs/research/2026-09-09_plane_initialization_v6_protocol.md` | `f1c5a38137d04d04cb27731ab6fef44dc6f892627a4d0578ab3078367a049e67` |
| `docs/research/2026-09-09_plane_search_v5_protocol.md` | `2b0983d9dc6612147d5b1ecc45303f44f819bd2f7f14ed5e30c3b366af4257b0` |
| `docs/research/2026-09-09_report_envelope_v6_protocol.md` | `4ebda90a4aefffd0d8c08bd4a2e2b7dfe5212342518a2dc379e1827a08812640` |
| `scripts/check_cad_guided_measurement.py` | `33b85e3434d81b66c9a1c3cb6f41aa40e382159d7aaf8be79faecd5c4b092132` |
| `tests/test_cad_guided_plane_initialization_v5.py` | `b1031fda243055cd38b7f35146912a186702d77e62fe590989e8b4981cb3de60` |
| `tests/test_cad_guided_plane_initialization_v6.py` | `0ec33703323399a6dd87dfa64fd73557f8becf7bef95e70d6996d3da089ca82f` |

Independent Astra reviewed complete source and all41 retained controls plus two
successor controls. Root inspected the mechanical copy and every delta. Geometry,
finite query support, scheduling, refinement, ambiguity/numerical-failure gates
and original1e-7/.01/.011mm positive limits are unchanged. Only V7 lossless
metadata transport, the V6 refusal boundary at three sites and distinct API/type
names change. No native loader or physical data is imported by this gate.

Astra independently verified helper307pass and codec100pass receipts with
unchanged hashes/no stop, and approved the final corrected protocolf1c5a381.
The earlier40pass/1fail initializer and78pass/13fail codec stay terminal.
The separate capacity run has completed3pass, no competing heavy process remains.

Command: `.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite plane_initialization_v6`.
Require all43 mandatory checks,no skips,60s wall/45s worker CPU/1GiB sampled
process-group RSS/1MiB stdout. Preserve first failure/stop/drift without retry.
This is not approval of a connected native or physical evaluation.

