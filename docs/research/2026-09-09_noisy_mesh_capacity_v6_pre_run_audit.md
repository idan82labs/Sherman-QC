# Capacity campaign V6 with V7 transport: independent pre-run audit

Date:2026-09-09. Source prerequisite GREEN; full-size execution conditionally GREEN.

## Exact12-file freeze

| File | SHA-256 |
| --- | --- |
| `backend/cad_guided_noisy_mesh_planes_v5.py` | `9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75` |
| `docs/research/2026-09-09_noisy_mesh_planes_v5_protocol.md` | `039b6a3f46e170947039ee32a8ce871f768aea781b369b244fc0e96a9965b71b` |
| `docs/research/2026-09-09_noisy_plane_registration_design_notes.md` | `03e014332e4a9072f631dd449c6c6f55710fba175dbe19af6e7349eed7562e52` |
| `backend/cad_guided_noisy_mesh_planes_v6.py` | `1f5fe2fd447c445efd8c0e12f9947bfe484126fd15b2fe1be1149f2d5baf2f72` |
| `docs/research/2026-09-09_noisy_mesh_evidence_v6_protocol.md` | `f4d2c0985c19fc26b39ab213e4088e491d90332110f6612fc3581c0abdf61ae6` |
| `backend/cad_guided_noisy_mesh_planes_v7.py` | `b0822289cf3abbde9c7cbf263b45c8d8ef4c4f45d7913e452c47a46ed881449f` |
| `docs/research/2026-09-09_noisy_mesh_evidence_v7_protocol.md` | `37739668e75ab437b67398a334e463a2d81a6b42d37bfb98fd27af02321e83d0` |
| `tests/test_cad_guided_noisy_mesh_capacity_v5.py` | `48f4f99537c2456cb40a53ab6165f5c608c75e707274153f42e5a7d079d68795` |
| `docs/research/2026-09-09_noisy_mesh_capacity_v5_protocol.md` | `cd446f96a086ac820630df1aa796ea40fad827c0cc88883af843e7756c09b1c5` |
| `docs/research/2026-09-09_noisy_mesh_capacity_v6_protocol.md` | `d6e32b81f563a7a302c15e255c594dcbfef8990ae375bc0c78e7fb66d778d19a` |
| `tests/test_cad_guided_noisy_mesh_capacity_v6.py` | `68f7bb9d017cbc71d5c3183fbc25561af1ca42e245942152d4d79d8feda7641d` |
| `scripts/check_cad_guided_measurement.py` | `33b85e3434d81b66c9a1c3cb6f41aa40e382159d7aaf8be79faecd5c4b092132` |

Root inspected the original capacity construction and every successor delta.
Independent Astra reviewed the complete test/protocol, unchanged V5 comparison,
all12 hashes and isolated routes. It independently verified the V7 codec100-pass
receipt,13 stable hashes and no stop. All original geometry/noise/oracle/limits
remain unchanged; only the lossless transport, schema/work assertions and
versioned names change. The earlier V5 capacity and V6 codec negatives remain
terminal. This is not physical noise qualification or a changed historical result.

First command: `.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite noisy_mesh_capacity_v6_source`.
Required2pass/1declared large-case skip under60wall/45workerCPU seconds,
1GiB sampled process-group RSS/1MiB stdout. No native or physical input.

Conditional second command: `.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite noisy_mesh_capacity_v6`.
Only after the prerequisite has2pass/1skip, no stop, all12 unchanged hashes and
no other heavy job, execute all3 cases once under120wall/90workerCPU seconds,
4GiB sampled process-group RSS/1MiB stdout. The full fixture stays900000 used
raw rows/1800000 triangles and all original ten-face/whole-witness/lineage gates.
Record the prerequisite and final dispatch confirmation before that command.
Any first negative remains terminal; do not retry, weaken geometry or change
started caps. The successful codec alone does not assume this case passes.

## Prerequisite completion and full dispatch release

`noisy_mesh_capacity_v6_source-xzu8djk7` completed2pass/1declared skip,
parent0.75482645817101s, workerCPU0.723492s, sampled/workerpeak97206272B,
no budget stop, all12 source hashes before/after consistent. OUTPUT SHA256
`2d8ca2c548d5e2141b6e9c99a47c8befbc422a0c43981928adf60081605b73b3`.
Root confirms no competing heavy job and no edits to the frozen closure.
The independent conditional release is now satisfied for the single full
three-case run with the unchanged120/90/4GiB/1MiB limits above.
