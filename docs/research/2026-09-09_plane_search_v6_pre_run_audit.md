# V6 fair-search first pure gate — frozen pre-run audit

Date: 2026-09-09. Decision: GREEN for exactly one first `plane_search_v6` run.
Independent reviewer: `/root/astra_repo_guidance_review` (Astra).
Root read the complete source, tests, protocols and harness before dispatch.
The reviewer independently read all 902 test lines, both protocols, complete
predecessor diff and additive harness diff, and checked the exact eight hashes.
No test, native or physical execution occurred during this review.

Frozen decision: all 115 cases (77 preserved, 38 added) must pass; no intended
skip. Counts, geometry thresholds, complete failures and no-prefix semantics
are fixed. Failure, count mismatch, resource stop or source drift is terminal
and cannot be retried or relabeled under this protocol.

One serial job: 60 s parent wall, 45 s worker CPU, 1 GiB sampled process-group
RSS, 1 MiB stdout. Clean environment, isolated Python, disabled plugins and
conftest, one named test only. No application/native/physical input is admitted.
The supervisor retains STARTED/OUTPUT/COMPLETED and before/after hashes.
Its prior exact8adf bytes are archived under
`docs/research/source_snapshots/mesh_connected_planes_v6_8adf01e0/`.

| Frozen file | SHA-256 |
| --- | --- |
| `backend/cad_guided_plane_search_v5.py` | `13898d4d736ba99be951ad228e19d47d8848082001786240a0b8a212d6ab9b98` |
| `docs/research/2026-09-09_plane_search_v5_protocol.md` | `2b0983d9dc6612147d5b1ecc45303f44f819bd2f7f14ed5e30c3b366af4257b0` |
| `tests/test_cad_guided_plane_search_v5.py` | `add36429382e4a5ce50775ef9fd61b068838a083e466493452dcd3f881944c22` |
| `backend/cad_guided_plane_search_v6.py` | `aa2f9067c5e4e65129c0c8c42d39b6d980c3e33fadac6daf881e544648292676` |
| `docs/research/2026-09-09_plane_search_v6_protocol.md` | `063338fe298036bb92950ad434a83ac85d03a5a3213d17bc65aa759de9041031` |
| `docs/research/2026-09-09_plane_search_v6_test_protocol.md` | `a37176aaf1d61fa92dcee185b6b81097cc8af3c349c993cfdb32d6c5346de7d8` |
| `tests/test_cad_guided_plane_search_v6.py` | `737dcb2da67bfd0e59923b9eb4cd7c629063994be8c4d84110ec8a79b0b2077f` |
| `scripts/check_cad_guided_measurement.py` | `87478a2c49fbaa42a773a4b7041e4dc243ad4924143985ba60e4f38aaf278bac` |

Command, from the Git root:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite plane_search_v6
```

The fair-bin proof and exact invocation-local identity receive source GREEN.
The independent oracle reconstructs windows/interval masks, compatible ordering,
canonical keys and occurrence maps without runtime schedule decisions. Geometry,
proper rotations, intrinsic translations, quota/work maxima and late whole-result
invalidation remain unchanged. Python/transient memory remains externally bounded.

Future V7 initializer/native routes in the harness are expressly outside this
release. A pass establishes declared scheduling/math/accounting only, not
connected recovery, full-capacity process memory, physical F1 or product admission.
