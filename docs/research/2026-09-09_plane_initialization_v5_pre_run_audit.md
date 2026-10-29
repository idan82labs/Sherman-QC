# Plane initialization V5: first connected-array pre-run audit

Date: 2026-09-09. **GREEN for this named pure/invented-array gate only**, recorded
before execution. The separate full-size V5 capacity attempt has already failed
its expanded-metadata cap; this small connected gate does not erase or solve that
negative. No physical input, native workflow, Gold or product-default path runs.

## Exact nineteen-file closure

| File | SHA-256 |
| --- | --- |
| `backend/cad_guided_global_registration_v2.py` | `5a1b039d17b6d5bb5840ebb16b5ab3d131ca3df1b6c186746804327921627dc3` |
| `backend/cad_guided_global_registration_v1.py` | `7700541063c4bfb40c72fb66baf2a78fea1c631e5e3f93eabbb51edfdb59eff7` |
| `backend/cad_guided_triangle_surface_v2.py` | `7b5333e18817d30905f3a1ace1cbf1e85129ad904324c5e6ccf2baff665a5cc1` |
| `backend/cad_guided_mesh_plane_proposals_v4.py` | `55356d900ce933b4dd4ff9fc8d055fdb46038422b69b6e05f7f9509d73f5ed48` |
| `docs/research/2026-09-09_mesh_plane_proposals_v4_protocol.md` | `2ca3469de02e83ca3fcb0a891b96df494edc95010f8a134029dece3d7c80e479` |
| `backend/cad_guided_plane_hypotheses_v4.py` | `7e69fce8bf79216342071aacfbbb7c0d73972c78334cc81d7f0fe15c831f1370` |
| `docs/research/2026-09-09_plane_initialization_v4_source_protocol.md` | `3e000cb99a248195050aef78c2466f923326b70efe6e0c05a31abcf5e6f6eb6e` |
| `docs/research/2026-09-09_plane_hypotheses_v4_design.md` | `aa81a52ea7329a91623e757432f49febb5de6c7fea037379b371a46a3b2cd591` |
| `backend/cad_guided_plane_initialization_v4.py` | `a4586cd02c490493472ea842622efeafe9ae18f246979ad4be364313bad72724` |
| `docs/research/2026-09-09_plane_initialization_connected_v4_protocol.md` | `f5463d6f94470153daca3035a1b84f7f35284ee2ce31ff0e5661cee2f9aaea82` |
| `backend/cad_guided_noisy_mesh_planes_v5.py` | `9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75` |
| `docs/research/2026-09-09_noisy_mesh_planes_v5_protocol.md` | `039b6a3f46e170947039ee32a8ce871f768aea781b369b244fc0e96a9965b71b` |
| `docs/research/2026-09-09_noisy_plane_registration_design_notes.md` | `03e014332e4a9072f631dd449c6c6f55710fba175dbe19af6e7349eed7562e52` |
| `backend/cad_guided_plane_search_v5.py` | `13898d4d736ba99be951ad228e19d47d8848082001786240a0b8a212d6ab9b98` |
| `docs/research/2026-09-09_plane_search_v5_protocol.md` | `2b0983d9dc6612147d5b1ecc45303f44f819bd2f7f14ed5e30c3b366af4257b0` |
| `backend/cad_guided_plane_initialization_v5.py` | `c37d2bfa916914c64af74e9edd5638756131914600222c63fe8bf77cfd71d2ee` |
| `docs/research/2026-09-09_plane_initialization_v5_protocol.md` | `593833164918b8dbc555a2b1620c5cc53681f1be5b56cc5d17eec4fba653fb60` |
| `tests/test_cad_guided_plane_initialization_v5.py` | `b1031fda243055cd38b7f35146912a186702d77e62fe590989e8b4981cb3de60` |
| `scripts/check_cad_guided_measurement.py` | `501cab2a2d6c6cb1be565e970523d1257a0fa46ac6634b53831f064fbaff0f52` |

Root authored the new initializer/protocol, not the independent tests.
Independent reviewer `/root/astra_repo_guidance_review` fully read the final
source, protocol and relevant frozen query/refinement callees. Its final
source-GREEN applies to c37d2bfa /59383316. Independent test author
`/root/cad_measurement_integration_review` supplied the complete test file
without execution. Root independently read all final test code and inspected
the additive harness route: named file, complete source closure,60/45/1GiB
budget, isolated backend namespace and durable retention. No native flag is set.

## Closed review findings

The new V5 global-row selection now rechecks the owned copy's shape/count,
nonnegative and upper raw-ID bounds and uniqueness before indexing. It does not
inherit V4's negative-index copy race or alter frozen V4. All known search
hypotheses enter the ledger before per-support checks; later malformed support
cannot erase unstarted identities. Late query failure retains attempted passing
group counts, invalidates usable group transforms/starts and leaves uncomputed
omission count unknown. Each has a deterministic independent regression.

The declared32-unsigned/all-proposed-signs/48-exact-group quotas remain
nonexhaustive. Every selected support row is charged and checked against its
complete finite CAD face, including repeated rows under another assignment.
V2 geometry/refinement/coverage/residual/observability/ambiguity thresholds are
not modified. Incomplete scheduled numerical evaluation vetoes fit publication.

## Fixed positive and negative oracles

Two required public-array positives use three unequal, separated nonorthogonal
patches: inset7x7 scan meshes, unrelated two-triangle CAD face unions and9x9
CAD samples. Balanced normal noise has amplitude.001mm with independently
zero mean and first in-plane moments; declared epsilon is.005mm. Quarter-turn
and rational non-axis rigid frames are not passed to inference. Actual finite
starts must have clean-row action error<=1e-7mm; final clean-row pose error
<=.01mm and noisy-row error<=.011mm. The prior strict small-matrix policy
(distance.5, finite P95.05, coverage.95/.30,147 points/two coarse/two fine
iterations) is fixed before execution. A refusal fails a required positive.
These ceilings are engineered test criteria, not calibrated uncertainty.

Other actual-array controls cover one-plane abstention, immutable complete raw
support, and finite CAD holes. Clearly labeled mocks isolate all-sign scheduling,
64 exact groups/48 strata and retained omissions, repeated-row preflight,
late-query evidence, numerical/finite-surface failures, competing generated
poses and report refusal. Mocks do not establish public pose accuracy.
Every literal initialization cap is checked against widening/bypass.

## One serial execution

```sh
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite plane_initialization_v5
```

Limits:60s parent wall /45s worker CPU /1GiB sampled group RSS /1MiB combined
output. No other geometry worker is live; full-capacity session79417 terminated
with exit1 and is closed. Preserve this gate's first output and all before/after
hashes. Any test failure, stop, supervision error or source drift blocks
advancement. Do not resize/relabel the positive cases or loosen their criteria.

The prior capacity harness da3828c1 is archived under
`source_snapshots/noisy_mesh_capacity_v5_da3828c1/`; changing the live harness
after terminal capacity execution is not in-run drift. No frozen computational
source/test/protocol was edited.

A pass would establish these small noisy mesh-to-finite-registration cases only.
Full-size representation, byte-to-native-measurement validation and a frozen
all11 paired Gold11 development run remain outstanding. No F1/count, stable
datum, scanner calibration, global uniqueness or floor-release claim follows.
