# V7 exact scalar transport: independent pre-run gate

Date:2026-09-09. GREEN for one frozen100-case pure/small-array gate only.

## Exact13-file provenance/execution closure

| File | SHA-256 |
| --- | --- |
| `backend/cad_guided_noisy_mesh_planes_v5.py` | `9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75` |
| `docs/research/2026-09-09_noisy_mesh_planes_v5_protocol.md` | `039b6a3f46e170947039ee32a8ce871f768aea781b369b244fc0e96a9965b71b` |
| `docs/research/2026-09-09_noisy_plane_registration_design_notes.md` | `03e014332e4a9072f631dd449c6c6f55710fba175dbe19af6e7349eed7562e52` |
| `backend/cad_guided_noisy_mesh_planes_v6.py` | `1f5fe2fd447c445efd8c0e12f9947bfe484126fd15b2fe1be1149f2d5baf2f72` |
| `docs/research/2026-09-09_noisy_mesh_evidence_v6_protocol.md` | `f4d2c0985c19fc26b39ab213e4088e491d90332110f6612fc3581c0abdf61ae6` |
| `tests/test_cad_guided_noisy_mesh_planes_v5.py` | `56c12020f3ae77dfafb7525b94d9ab823a9e2f630b98537ef1bac1434f715580` |
| `docs/research/2026-09-09_noisy_mesh_evidence_v6_test_protocol.md` | `0743f15919825eace08885f89f1f0435f2d84f668df19c2bc975b74cf712accd` |
| `tests/test_cad_guided_noisy_mesh_evidence_v6.py` | `223e28884e251c35a0dcc03d2250b89ed691338e0443dcb5b6bde8400f56ac45` |
| `backend/cad_guided_noisy_mesh_planes_v7.py` | `b0822289cf3abbde9c7cbf263b45c8d8ef4c4f45d7913e452c47a46ed881449f` |
| `docs/research/2026-09-09_noisy_mesh_evidence_v7_protocol.md` | `37739668e75ab437b67398a334e463a2d81a6b42d37bfb98fd27af02321e83d0` |
| `docs/research/2026-09-09_noisy_mesh_evidence_v7_test_protocol.md` | `1dbc0f3edbbf083c0f7996e4ebf103867298b1d05ff03569a234d7fa52b4cfd2` |
| `tests/test_cad_guided_noisy_mesh_evidence_v7.py` | `f785d6f1856b76826ab1f72966ccbe5cca7df19ceb472c9d3805ed6eb9e84827` |
| `scripts/check_cad_guided_measurement.py` | `33b85e3434d81b66c9a1c3cb6f41aa40e382159d7aaf8be79faecd5c4b092132` |

Root read the entire V6 source and all V7 changes, implementation protocol,
original91-case test plus all successor changes and nine scalar regressions.
Independent Astra read the complete340-line V7 source,579-line test, both
protocols and exact harness route/closure delta. Source and fixture reviews
agree: only exact NumPy float64 is normalized encoder-locally after one visit,
preserving binary64 values and all internal types/arrays. Decoder remains
strict. Geometry, noise, witnesses, cones, quotas and final work limits stay.

The failed V6 gate remains78-pass/13-fail, unchanged. The correction of the
export oracle models actual V5 JSON transport; it does not weaken the separate
internal exact-byte/type comparator or any geometry assertion. Prior terminal
harness76a73cdb is archived byte-identically; no attempt is overwritten/retried.
This is a new source-repaired candidate with a new frozen test/protocol.

Command: `.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite noisy_mesh_evidence_v7`.
Exactly100 mandatory controls,no skips,60s wall/45s worker CPU/1GiB sampled
group RSS/1MiB stdout,one serial job. The clean worker environment has no
native/capacity opt-in and imports no app/physical input. All13 hashes are
checked before/after. Any first failure, stop or drift stays terminal.
No capacity or initializer dispatch is approved by this codec verdict alone.
