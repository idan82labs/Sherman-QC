# Connected V12 source gate — independent PRE, 2026-09-10

Decision: **GREEN_FOR_ONE_FROZEN_CONNECTED_V12_SOURCE_GATE**.
Reviewer: independent AI agent `/root/production_architecture_review`, not the
author of the V12 candidate, final tests, harness or protocol. This is a source
release decision, not human approval, an observed pass or manufacturing authority.
Blocking findings: none for the one command below.

## Exact release and denominator

From the canonical `FullProject` root, release only one first execution:

```sh
.venv/bin/python -I -S -B scripts/check_cad_guided_connected_v12.py --suite source
```

The three fixed pytest files have independently AST-counted denominators:

| Suite | Test functions | Collected outcomes | Required disposition |
| --- | ---: | ---: | --- |
| Context measurement/stage/publication | 29 | 168 | 168 PASS |
| Owned surface-role contracts | 21 | 69 | 69 PASS |
| Constructed fixture/serialization/custody | 13 | 20 | 19 PASS, 1 intentional SKIP |
| Total | 63 | 257 | **256 PASS, 1 intentional SKIP** |

The sole permitted skip is `test_connected_six_case_three_arm_packet`. The parent
builds a fresh environment without `SHERMAN_RUN_CONNECTED_V12`; the source
worker never sets that opt-in. Its attempted-call control asserts refusal before
artifact-directory creation. Unexpected collection, another skip, any failed
outcome or budget/custody/source failure is a terminal negative, not a retry.
Counts are a mandatory independent POST condition, not merely pytest exit zero.

Do **not** run `--suite connected`, `--worker` directly, native/byte adapters,
the full V4 runner, any physical/cached/Gold input or another scientific gate
under this PRE. A source POST-GREEN and a separate exact connected PRE are still
required before the 18-arm packet. All earlier terminal negatives remain intact.

## Source review and critical findings

Read AGENTS/current status, the complete final context tests and publication
validator, the complete harness/protocol and constructed fixture test, and the
unchanged role-helper/test sources already independently reviewed. Prior full
V12 initializer/workflow/stage/canonical-fit review is bound below. The local
import graph was checked statically without importing target modules; every
reachable local backend dependency is in the 49-file ledger. The V3 runner is
only read as bounded source for its named CONFIG AST; it is never imported or
executed, so its dormant V10 dispatch is not an additional import dependency.
The older planar tests supply named source capsules only in the unreleased
connected branch; their whole test suites are not imported into collection.

The harness installs an inert `backend` namespace before pytest; the context
fixture temporarily replaces/restores that namespace and its isolated package.
No application/server startup is reached. NumPy/SciPy and pytest are admitted
environment dependencies, not hashes of a fully hermetic software environment.
OCP imports remain inside native functions not called by these source tests.
The gate includes real small invented-array free-circle fits and codec/role
selection, not merely syntax checks; aggregation tests explicitly mock crop,
per-context support discovery and the owned role selector. Fixture tests only
construct/check invented arrays, bounded source literals and tiny owned files.
There is no registration-packet or physical inference execution in this suite.

The original publication gap is repaired before execution: strict canonical
base64, represented bitmap length, zero padding, counts and raw-ID hashes;
the four-field source tuple; every filter/window/association reference;
window-association bijection; ordered flange roles; and all associated support
rows against excluded bits are checked. UNAVAILABLE records retain and validate
their role/association evidence. The 47 new negative cases include a later
alias's internally consistent but contradictory exclusion; a separate positive
preserves two valid filters. Ordinary malformed-object exceptions are uniformly
ValueError-family at the public boundary; fatal scientific prefixes remain
empty. This is in-process serialization consistency, not a signed external
authenticity proof or a repeat of surface classification.

No new canonical radius/sign/carrier blocker was found. Fixed raw-role fitting
is not given nominal radius as its answer; proper-pose/order cases and changed
nominal radius check that distinction. Cross-record arc/arc or arc/flange raw
position conflicts veto competing records, rather than selecting a favorable
alias. Conditional scan point estimates and sampled support carriers do not
publish an accepted pose, known bend count, verified material side, physical
trim, uncertainty enclosure or joint/continuous-pose completeness.

Owned role preparation binds actual scan support arrays to V7 metadata; hashes
are never inverted into rows. Axis-cache keys use exact canonical raw-axis
bytes. Aggregate role-membership/incidence and full-crop work remain bounded;
publication masks are limited to 64 million bit rows and association inspection
to 4 million support-row visits. These logical bounds and end-report byte caps
are not proofs of peak heap size. The role cone/normalization reserve remains an
explicit engineering policy, not directed-rounding physical cap certification.
Unknown/noncap incidence preserves sheet/cap seams; declared fixture cones do
not establish that real extraction will find those cones.

## Budget, custody and mandatory POST

Frozen source limits: 55 seconds work plus bounded cleanup within 60 seconds;
45 worker CPU seconds; sampled owned-group RSS 1,073,741,824 bytes; stdout
1,048,576 bytes. Numeric threads are one, bytecode and plugin autoload are off,
and pytest has no conftest/cache/config discovery. Exactly one worker is allowed;
an observed descendant, malformed monitor, stop or unconfirmed group-empty
cleanup vetoes success. This is sampled lifecycle supervision, not an OS memory
allocation cap or hostile-process guarantee.

Source binding/setup and final artifact hashing/receipt publication are outside
the supervisor work clock, as disclosed in the protocol; do not call this a
hard whole-command 60-second bound. The private invented cache is retained.
The fresh attempt must retain STARTED, OUTPUT, ARTIFACT_INVENTORY and COMPLETED
plus the empty `ARTIFACTS` directory. Its source-suite inventory must be complete
with no arm artifacts, unknown names or content reads of unknown artifacts.
Inventory runs only after group-empty confirmation, checks the directory type,
and uses regular no-follow leaves with before/after identity checks. This is
stopped-writer custody, not adversarial path-replacement atomic attestation.

Independent POST must verify observed successful parent exit, exact outcome
denominator and allowed skip, all receipt/output hashes and retained inventory,
worker CPU/high-water RSS plus sampled group evidence, no stop/error or cleanup
attention, and every before/after/current source hash against the ledger below.
An incomplete receipt or inventory is not success. No test, target import,
geometry execution, original input access or attempt access was performed by
this PRE reviewer; only source/text, stdlib AST/count and digest checks occurred.

## Exact frozen source ledger

49 distinct files; combined bytes at review: **1,345,758**. Every file was hashed
from its current source bytes, and local Python sources were parsed without
execution. This complete map must match STARTED and COMPLETED; do not substitute
a later source revision because it has the same filename.

| Source | SHA256 |
| --- | --- |
| `backend/cad_guided_cad_sampling_v1.py` | `f1d32159a50b455d6b9a0b5f646586f0d79e7f9c26e8fffbe67f97c5b74e181a` |
| `backend/cad_guided_cad_surface_v2.py` | `b7364234a716f283f46246a452451f47c8efb07c18b2d37375c6fda0636da0d1` |
| `backend/cad_guided_context_measurement_v1.py` | `a756289847125e94baa654b4c82b4c3daf5592b6529a1ec7f1a1072dd79575a9` |
| `backend/cad_guided_context_surface_roles_v1.py` | `c042408109d260888834ebacc9ad3b62256fda589f1ea8cbba31164b2bea9a62` |
| `backend/cad_guided_face_box_screen_v1.py` | `c8f66cc25f5c4cb6a6ba3c1e1beeb205540813796bb2ab30d72eb596cdf3abb7` |
| `backend/cad_guided_global_registration_v1.py` | `7700541063c4bfb40c72fb66baf2a78fea1c631e5e3f93eabbb51edfdb59eff7` |
| `backend/cad_guided_global_registration_v2.py` | `5a1b039d17b6d5bb5840ebb16b5ab3d131ca3df1b6c186746804327921627dc3` |
| `backend/cad_guided_global_registration_v3.py` | `088c0b5e9fd940094fd44462e7e5c831299fe2c69b87418d09c362772263bae0` |
| `backend/cad_guided_instances.py` | `d30b3fc8579b0a2d44d37bfe093343b3f7d703661793784d78b50efbc38eacda` |
| `backend/cad_guided_local_measurement_v1.py` | `5b07e8c14d3eededfb7cbdb2cb628e6d631a30662280fc2d39b2528d862b541b` |
| `backend/cad_guided_measurement.py` | `d3a11bde2b14c6ece39414360e3146bf60e3801190b9e04cd6c5178ed903f2cb` |
| `backend/cad_guided_mesh_input_v1.py` | `c4d936e2f60837d7c5221c9a46b64abd68c4ccee01ea72e9fd3c4050bde47917` |
| `backend/cad_guided_mesh_offline_v1.py` | `55243b2a557e77a9c50e27a7fdef9257d60ed2468bc23c1788c391d7a836bdba` |
| `backend/cad_guided_mesh_offline_v11.py` | `dbd60ebc8c2c159866c212506e46e35d56af17b196827ba5243c637c6dfe6dfa` |
| `backend/cad_guided_mesh_offline_v12.py` | `daec784c9652e8bfc7b152cd851515f73e2f8bfffc404c1ec89b616bbd2fdbe3` |
| `backend/cad_guided_mesh_offline_v2.py` | `fa30bc1ed575f9174414e3737eaaffecc6eafdcbcd9e3b1d68475f1af7ea8cbf` |
| `backend/cad_guided_mesh_offline_v8.py` | `afdfc7f859eb5fdbf0fa7d796cc06a4180d40c02f6b0e590362f83e3f339e7ec` |
| `backend/cad_guided_mesh_plane_proposals_v4.py` | `55356d900ce933b4dd4ff9fc8d055fdb46038422b69b6e05f7f9509d73f5ed48` |
| `backend/cad_guided_mesh_topology_v4.py` | `4d061b56a39aee6b7d9e8c04eb7af88b0573a7916d3d679c610c4aca81ee0734` |
| `backend/cad_guided_native_features.py` | `f8c681027af231546974633a02f12038d522c604bec14196f2ae5d49b439c99f` |
| `backend/cad_guided_native_features_v1.py` | `3804a583f5642df41cd3eb02be36b3f5024347b6514861d404389e85c1ccf98b` |
| `backend/cad_guided_noisy_mesh_planes_v5.py` | `9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75` |
| `backend/cad_guided_noisy_mesh_planes_v7.py` | `b0822289cf3abbde9c7cbf263b45c8d8ef4c4f45d7913e452c47a46ed881449f` |
| `backend/cad_guided_plane_hypotheses_v4.py` | `7e69fce8bf79216342071aacfbbb7c0d73972c78334cc81d7f0fe15c831f1370` |
| `backend/cad_guided_plane_initialization_v10.py` | `398c40055c00fd3432410242a7a2ee399d9d716ea8cb3c1755916904b6074331` |
| `backend/cad_guided_plane_initialization_v11.py` | `31bc1b9c3fe58b1befd7432134ce0640fa9243e23f51b51c28ca4ae72aa7610e` |
| `backend/cad_guided_plane_initialization_v12.py` | `dfcc7335a9564a76dedff9f11dd03eb8d5c38cece52fbf6d68bd48a91b78c2e3` |
| `backend/cad_guided_plane_initialization_v4.py` | `a4586cd02c490493472ea842622efeafe9ae18f246979ad4be364313bad72724` |
| `backend/cad_guided_plane_initialization_v8.py` | `cd225b4642816f76cfd4b8bd725e3b8ffa5d9f6177ddbfa9b03fb6a22128c28b` |
| `backend/cad_guided_plane_initialization_v9.py` | `c64ab591ee07b91d7f20cb0d663cc2bb2a98e5baa681e072f97353b01e82401d` |
| `backend/cad_guided_plane_search_v5.py` | `13898d4d736ba99be951ad228e19d47d8848082001786240a0b8a212d6ab9b98` |
| `backend/cad_guided_plane_search_v6.py` | `aa2f9067c5e4e65129c0c8c42d39b6d980c3e33fadac6daf881e544648292676` |
| `backend/cad_guided_report_envelope_v6.py` | `c7cc45a362c5972aff78265b2ce63fa4d41ec023b662640d7d197c23003ba28e` |
| `backend/cad_guided_stage_policies_v1.py` | `5c3d1252ac5f38f422f8a9a22c98a3d4db1094cd0956c928e9800f75e3ee826c` |
| `backend/cad_guided_triangle_distance_bounds_v3.py` | `2404278ebd8872dafa6752d9bf977879e7d0a21ea41752ec740501b7044058d7` |
| `backend/cad_guided_triangle_surface_v2.py` | `7b5333e18817d30905f3a1ace1cbf1e85129ad904324c5e6ccf2baff665a5cc1` |
| `backend/cad_guided_triangle_surface_v3.py` | `8e3222bd425451e2e89b2fe0083db34a3702577018b3f557b1f026a86add9f10` |
| `backend/cad_guided_two_plane_context_v1.py` | `9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a` |
| `backend/cad_guided_two_plane_schedule_v1.py` | `1fcd5bb06922b64693de235cbf1e6e7df6fdd9c43795c6bced27af6d2e4b866f` |
| `docs/research/2026-09-10_connected_v12_fixture_design.md` | `5818f76ba084ccf726c9a2fffce15991ee94c8b8a068da95573318f4ae2c048f` |
| `docs/research/2026-09-10_connected_v12_gate_protocol.md` | `671aa78a6d68fc2d42f34ed464337fadaa184d68d4ac61c3bfe7e52ac3fa85e4` |
| `scripts/check_cad_guided_connected_v12.py` | `0003b32fb4a605cfb82a6edd167aa8d2cc024303e50921928efd21666d4df694` |
| `scripts/run_gold11_mesh_development_v3.py` | `77bbcffbf10f05a22d0c72c8a645112e489469b4af92b4dbba3ca6243fcecea7` |
| `tests/test_cad_guided_connected_v12.py` | `fbbbbc148c564c222013c68f81922e2aed19bdfc1033c7677117a1186582bdfd` |
| `tests/test_cad_guided_context_measurement_v1.py` | `1567780921771510ac761ed8d41c3b55d82de731b5b1cf409e8c53466a40589b` |
| `tests/test_cad_guided_context_surface_roles_v1.py` | `930baccbfe1a878cdce4c1543e79b1a6111f739c37c6bfa47bc7499945d87e4d` |
| `tests/test_cad_guided_plane_initialization_v10.py` | `4882b701af3806fd03ddde9bb0b1af823bd0124e859e6655207f683db1563eb0` |
| `tests/test_cad_guided_plane_initialization_v11.py` | `16a646404b4cf35975956b717a9143b52d8bca1e7ec5467f40096aefbc0edf1e` |
| `tests/test_cad_guided_plane_initialization_v8.py` | `e78e5bc17705334b9e7dd1ebe1f70271ae1cdca0a118bbfbea24d95f633a684a` |
