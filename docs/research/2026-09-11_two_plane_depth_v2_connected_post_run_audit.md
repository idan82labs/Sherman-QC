# Two-plane depth connected V2 — exact terminal RED POST

Date: 2026-09-11. Independent reviewer: `position_region_implementation`.

## Decision and scope

**RED for the original four-case scientific qualification. Artifact-level integrity admission passes.** This is a completed, fully retained failed qualification, not an interrupted run or permission to replace the denominator with the successful subset.

The single attempt `two_plane_depth_v2_connected-haebyimq` ended with **6 PASS / 1 FAIL in 5027.15 s**. All four actual shared registrations passed their original registration oracle. Required support-cover V2 measurements passed U-R and U-A, but failed L-R and L-A. Thus the mandatory positive measurement denominator is **2/4**, not 4/4. All four same-context V1 diagnostic measurements refused. Those diagnostic negatives were permitted; the two required V2 negatives were not.

All 16 stages and all 32 scientific files are present. Execution, safety, publication, input-custody, inventory, source-custody, cleanup and resource checks pass within the stated scope below. This admits the exact retained packet as evidence, including its failures. It does not qualify native STEP/PLY operation, Gold11, real scans, physical scanner uncertainty, manufacturing count, accepted registration, production use, or a successor launch. No new run is released by this POST.

## Authorized evidence and audit method

The parent confirmed terminal session `88321`, exit 1, parent PID `49808`, worker/process-group ID `49810`, and no remaining group members before authorizing this audit. The reviewer subsequently made an independent bounded `ps` snapshot: neither named PID nor any member of process group 49810 was present. The terminal receipt separately records `process_group_confirmed_empty=true` and `requires_group_attention=false`.

The audit read only the authorized terminal attempt's four administrative files and 32 scientific files, the 97 frozen sources and their PRE ledger references, and the 98-payload source archive. File reads used `O_NOFOLLOW | O_NONBLOCK`, regular-file/size checks, bounded chunks, and stable descriptor/path identity and size/time checks before and after reading. JSON decoding rejected duplicate keys, nonfinite constants, and overflowing nonfinite floating-point values. The archive was inspected in memory without extraction; only regular, exact expected members were admitted. The retained private cache was not opened, listed, hashed, removed or executed.

No project import, pytest collection/test, geometry generation, numerical refit, registration, measurement, raw-dataset read, native operation, or scientific artifact rewrite occurred during this POST. Source inspection/AST parsing, bounded JSON inspection, hash reconstruction, fixed scalar arithmetic and process inspection were the only checks. Read-only utility drafts were corrected for optional failed-oracle fields, the `answer` wrapper, source `dict(...)` literals, scan-versus-CAD digest roles, and positive-publication geometry-visit semantics; the final complete checks passed. These audit-utility corrections changed neither the frozen experiment nor its results.

Limits were retained, not enlarged: 4096 B per START, 8 MiB per planar RESULT, 64 MiB per shared-registration RESULT, 32 MiB per measurement RESULT; 544 MiB RESULT allowance plus 65,536 B START allowance, or **570,490,880 B** total. Each audit pass was subject to those per-file and complete-set limits. Four administrative files were bounded separately, including the 1 MiB OUTPUT limit.

## Exact identity and source custody

Attempt directory, relative to the repository:

`docs/research/cad_engine_v1_synthetic_checks/two_plane_depth_v2_connected-haebyimq`

The attempt has exactly `ARTIFACTS`, `ARTIFACT_INVENTORY.json`, `COMPLETED.json`, `OUTPUT.txt`, and `STARTED.json`; the scientific directory has exactly the 32 declared names. No unexpected entry was admitted.

| Administrative file | Bytes | SHA-256 |
| --- | ---: | --- |
| STARTED.json | 12307 | 89d8309118ae2ff4bfd3e495dd6149185144942eacd4590ae65f30d0fd037ed0 |
| OUTPUT.txt | 18415 | 7f890dbc250dda1f9d4014ff97985549b2307ef6e077f3e6208757a617d625a8 |
| ARTIFACT_INVENTORY.json | 6126 | 8badfb2f1bbf74c65493c6f47e76567b1a1b1f01db69aeb41d414f6eb4922e82 |
| COMPLETED.json | 13101 | 9807c4a1e002a9339249d66b682854fcb91c715e194f7de39e2d14f5016fddc4 |

Exact PRE: [2026-09-11_two_plane_depth_v2_connected_pre_run_audit.md](2026-09-11_two_plane_depth_v2_connected_pre_run_audit.md), 20,796 B, SHA-256 `c187be7bd6fd1b6d9db703acb6b19494fd5106285c24934380f2974a75ee76fb`.

The complete source ledger was independently reconciled from the original 72-source PRE, the depth V1 PRE's replacements/additions yielding 91, and the current PRE's two replacements plus six additions yielding **97 sources: 77 Python + 20 Markdown, 2,047,553 payload bytes**. All 97 current size/hash pairs match the exact PRE and both receipt maps, with no omitted or extra map keys. All 77 Python files parsed as AST without imports. The current packet retains the old RED evidence and the qualified context prerequisite; it does not rewrite them.

The three current packet pins remain:

| Source | Bytes | SHA-256 |
| --- | ---: | --- |
| tests/test_cad_guided_two_plane_depth_connected_v2.py | 18484 | 8d84b443c265c4ebacb85d27654518393c36676884a2c7c3f1347d485c6247ee |
| scripts/check_cad_guided_two_plane_depth_connected_v2.py | 11760 | 3c804d8ff96bf422cde54a35501c409a0fc443c350c3ef9e103c8a8725ce18be |
| docs/research/2026-09-11_two_plane_depth_v2_connected_protocol.md | 8343 | a28b4e3820d8456942928d4a046e6084f138202d316955df8157efc0f85de5f2 |

The relevant integrated engine pins remain `cad_guided_mesh_offline_v12.py` 46,206 B / `7c444bab2c197b0a6d9949ec6a15162286edb78bdfd081137e2907f16152c33e`, `cad_guided_plane_initialization_v12.py` 73,393 B / `57a7b44dedaead564d219ceb039f10bcf31ab2cd51b6ea4e2af30a34f00a1d87`, and new `cad_guided_two_plane_context_v2.py` 11,130 B / `d2bdea468ebe6eca5bea9f125a6f7cb3bbf49894a65ab6861c81fd2e7fb7b663`. They remain the bytes covered by the context source prerequisite, [241-outcome GREEN POST](2026-09-11_two_plane_context_v2_source_post_run_audit.md), 5334 B / `c079b7d442650dba692882abd9c10b1415634fded2cd473cb9b1a6f9571159e6`. That prerequisite was not rerun here and is not the connected scientific qualification.

`tests/__init__.py` remains 24 B / `5332e0ec7edf6ab084943384a46bb543762f0401b94ea5c4276a27fc63ef211b`. Repository-root `__init__.py` is absent; both before/after guards and the independent current check agree.

The frozen archive is:

`docs/research/cad_engine_v1_synthetic_checks/two_plane_depth_v2_connected_frozen.chf6g8vp/SOURCE.tar`

It is **2,150,400 B**, SHA-256 `1c288c7d9968a9759470bd76a8fd9b959c2443d5189b5d4142ef196db8bf8b54`. Its 98 unique regular-file members are exactly the 97 sources plus the exact PRE, with **2,068,349 payload bytes**. Each payload is byte-equal to its approved current source/PRE buffer. No link, extra path, directory payload, or AppleDouble payload is present. The artifact/archive admission therefore has a complete declared Python import-source closure including the tests initializer; it is not a claim that Python dependencies or operating-system implementation are cryptographically frozen.

## All 16 stages, no favorable-prefix substitution

START/RESULT identity, kind, case index, case name, arm, input binding, monotonic start and process-CPU start agree for every pair. START dispositions are `STARTED`; all RESULT dispositions are `COMPLETED_STAGE`. The original stage order is preserved with increasing monotonic start values. All 16 aggregate entries and all 16 progress entries match their retained results, clocks and artifact hashes. All stage execution/safety errors are null. Oracle failures remain separate retained `AssertionError` records; the recorder correctly omits an `oracle` value when that oracle raises.

| Case | Stage / arm | Call CPU s | Call wall s | Original oracle outcome |
| --- | --- | ---: | ---: | --- |
| P-clean | planar V11 | 0.259312 | 0.259608334 | PASS, positive, action error 0 |
| P-clean | planar V12 | 0.105038 | 0.105097375 | PASS, positive, action error 0 |
| P-noisy | planar V11 | 0.105175 | 0.105228167 | PASS, explicitly allowed retained baseline refusal |
| P-noisy | planar V12 | 0.105965 | 0.106008750 | PASS, positive, action error 0 |
| U-R | shared registration | 1047.316910 | 1048.125053125 | PASS, 48 completed qualifying contexts |
| U-R | diagnostic measurement V1 | 1.708356 | 1.709566459 | Refusal; permitted diagnostic negative |
| U-R | required measurement V2 | 1.697320 | 1.698374500 | PASS |
| U-A | shared registration | 1152.529501 | 1153.371124667 | PASS, 48 completed qualifying contexts |
| U-A | diagnostic measurement V1 | 1.682612 | 1.683723042 | Refusal; permitted diagnostic negative |
| U-A | required measurement V2 | 1.779347 | 1.780431167 | PASS |
| L-R | shared registration | 1462.253261 | 1463.287234917 | PASS, 12 completed qualifying contexts |
| L-R | diagnostic measurement V1 | 1.381932 | 1.382873958 | Refusal; permitted diagnostic negative |
| L-R | required measurement V2 | 1.437055 | 1.438005875 | FAIL, no usable existing support-cover measurement |
| L-A | shared registration | 1344.835040 | 1345.801802875 | PASS, 16 completed qualifying contexts |
| L-A | diagnostic measurement V1 | 1.388659 | 1.389823084 | Refusal; permitted diagnostic negative |
| L-A | required measurement V2 | 1.432589 | 1.433562791 | FAIL, no usable existing support-cover measurement |

The six ordinary packet controls pass. The seventh pytest outcome is the aggregate connected packet and fails its unchanged all-four-positive assertion. There are no setup errors, skips, xfails, xpasses, truncated cases or replacement outcomes. `challenger_positive_cases=2` and the ordered comparison vector is `[true,true,false,false]`; the V1 diagnostic positive vector is `[false,false,false,false]`. The final L-A stages completed after the L-R negative, preserving failure continuation.

## Same input pins and binding semantics

The original fixture remains `tests/test_cad_guided_connected_v12.py`, 49,931 B / `fbbbbc148c564c222013c68f81922e2aed19bdfc1033c7677117a1186582bdfd`. The shared recorder/oracle helper remains `tests/test_cad_guided_support_cover_connected_v2.py`, 31,074 B / `c645c7f0ff96e726989eb929176a8b1286f45c1ef996c68b21473ec488707cb5`. The only fresh helper-instance adjustment is the disclosed 64 MiB registration retention allowance and matching aggregate allowance; its fixture and scientific oracle code are unchanged.

All twelve curved stage bindings agree with the expected source-literal array hashes. The actual registration input declarations, initialization input declarations, context helper declarations and composite extraction metadata agree with those same pins. This is a consistency check of retained evidence and frozen executable checks, not reconstruction or independent hashing of absent raw arrays.

Common CAD triangle / face / local-source-ID pins are respectively:

- `835d6e433a70fbd49d9f25b6b72a6a4e5cc5c052186ca7f93d688188b479a663`
- `dca2a38fba5ab6432517f89dace6e40fe48009aec5e8a4773e77e781c0d8a031`
- `8408baf40dd517218eb6e665efbd4564c85a91e220363421969d219309d1d41e`

| Case | Raw rows / scan triangles | Scan point SHA-256 | Scan triangle-row SHA-256 |
| --- | --- | --- | --- |
| U-R | 450 / 784 | e35413e299bb669522ced7da745d0eab44297047634bf7e21cee270972abf895 | 02ce0d286d7e0af44b341a50eeaa034e9e713f298bf40c3c17b8ed0cac8ccbc6 |
| U-A | 450 / 784 | 7ccd0153ee3e9ff38a696994bd04a470eb9948f45aa9779421a115c6a42aefe8 | 02ce0d286d7e0af44b341a50eeaa034e9e713f298bf40c3c17b8ed0cac8ccbc6 |
| L-R | 650 / 1176 | 994b3d8a4de940c6470393fdb60dc20d9fb368434c135bd33cbc496872cfce3c | 8c4f5c76ca41d6fd9d63a7523a4cace886392e307fe07b56140dd8b510c12de6 |
| L-A | 650 / 1176 | 89a9670d884ef4b1d1dc00a0d84b1456ea3118144c5a3c23e2f9ee20e3d998a8 | 8c4f5c76ca41d6fd9d63a7523a4cace886392e307fe07b56140dd8b510c12de6 |

The stipulated 0.05 mm binary64 bound-vector digests were independently reconstructed from the declared scalar/count: U cases `59279b7e7b76f979adf4502f769f91b2b91b17eb4a357bea557ec6e185ba0625`; L cases `f36dcf3643211a940aca70dad69af009d85dbf2617c7ffa51cd522abaa9e3f2c`. They match initialization and role declarations. This verifies the declared bound bytes, not scanner calibration. CAD input count remains 564 triangles; all four registration answers declare 284 unique CAD initialization vertices, with their digest matching the registration evidence. This is not the 32,768-sample native path.

Each shared registration's complete canonical JSON digest was independently recomputed and matched both measurement arms and their exact shared-registration artifact references. Each owned-role fingerprint digest was recomputed. Restoring the V7-format envelope's source-defined insertion order recovered the original composite metadata byte size and SHA-256; decoding/re-encoding its detached descriptor tree recovered both original lane metadata digests. No descriptor was inverted into raw geometry or treated as physical-plane proof.

| Case | Shared registration canonical SHA-256 | Owned-role fingerprint SHA-256 |
| --- | --- | --- |
| U-R | c39ebd27f67388fed7af041063d9f09fb3f19840defd0ea9e36a01a713cb87ce | c4ae35cc5c4627c538dc55da3fbfe7985b60bb30badff6f4669f8b799c4fe6b5 |
| U-A | 95b0deb0c59d83e78ffe868cc786b34174f2777423a3b0934badf47e1f602c9e | 76c9b1d5220a25e2c907cc171df5c8ee9d846b99480fcb373815ff14b9d76062 |
| L-R | 5d8ff02b37d88ce59a788c2e7c89541ac4b44c343cf274ddf43688c375a918d9 | a9dce4b56cd7dd62cccf77dc2430d7841ceb5319510e157a74c5c7325584f8e0 |
| L-A | 0cf0c188a98c56a63bf924f6a2e71b796f9ddbeab56f9c4f628967389a2a1b0a | 37e83622bcc797d04b4b3bd0e40cf1091ce43e659f5349d6c36e41bcc735f1c1 |

The same pre-call policy snapshot is retained for all four cases. It explicitly requests composite scan planes, enabled pairs, scheduler mode `v2` and the exact typed-default scheduler budget profile. Both registration and initialization publications match that requested configuration; its canonical digest is `4a93432adda76b5bd58ae5c499af1a145ff0b4bf16086db22f6b17534aebb888`. Initialization's full integer budget dictionary matches the requested snapshot without bool/int aliases. All four `binding_error` values are null and `two_plane_context_profile_bound` values are true.

All four actual context helpers publish `two_plane_context_v2`, `COMPLETE`, complete selected-policy evaluation, the exact 18-field default integer profile and exact scheduling-capacity schema. Per-pair capacity is 256, but global capacity remains 256 ordered matches / 1024 sign families / 8192 slots. Source/target frames, zero origins and provisional common-mm assumption match. The two-plane scheduler similarly publishes complete V2 status, the exact requested budget profile and matching equation hashes. Source-bank equation hashes were recomputed from the retained canonical rows; CAD equation hashes were reconstructed using the source-defined scalar normalization of published coefficients, not plane fitting or scheduler execution. Common CAD equation hash: `74c797b73a0da4e5735a3d4e4d94b16bdb39adb7a09c7900e8befa308dbb578b`.

| Case | Source-bank equation SHA-256 |
| --- | --- |
| U-R | 742d552954e0ab2be6f83d33480a9ad491995d853c65de8a98f33f02998a7ab4 |
| U-A | 79116f192a7ba6fe9d39f3ad189192289af91267520f19e19397c70d080b925a |
| L-R | bebce292ff326b07bd71457a3f191ed09e9b04820c393406b5a19c220eaf03a7 |
| L-A | d45cc8dc5bb86c7e08a4c43dae8cbfbfe394a820f9693ecd89b7533e2780bfec |

## Actual depth admission and registration recovery

All scheduled match records have sequential integer IDs, 32 reserved hypothesis records and one unsigned reservation per ordered target. Both CAD orders are present; canonicalized context schedule rows match every scheduled match. All cases reach **42 ordered targets for at least one source pair**, exceeding the predecessor V1 consumer's four-target capacity. Every signed-family identity, four-sign denominator, correspondence identity and eight-slot sequence matches the retained schedule. Proposed/rejected/absent counts reconcile to the complete slot denominator, and pair/triplet reservations reconcile to the unchanged joint 8192-record ceiling.

| Case | Ordered pair targets | Triplet records / unsigned reservation | Completed families / slots | Proposed / rejected / absent slots | Passing groups / selected groups | Qualifying measurement contexts |
| --- | ---: | --- | --- | --- | --- | ---: |
| U-R | 256 | 0 / 0 | 1024 / 8192 | 7880 / 128 / 184 | 155 / 48 | 48 |
| U-A | 256 | 0 / 0 | 1024 / 8192 | 7864 / 128 / 200 | 274 / 48 | 48 |
| L-R | 252 | 128 / 16 | 1008 / 8064 | 6380 / 0 / 1684 | 53 / 48 | 12 |
| L-A | 254 | 64 / 8 | 1016 / 8128 | 6396 / 0 / 1732 | 44 / 44 | 16 |

These are different denominators: a passing initialization start group is not automatically a qualifying registration context, and neither is a measured bend. All four registration reports are `CONDITIONAL_MEASUREMENT_CONTEXTS`, with candidate evaluation and numerical consumer completion true, accepted registration false and production use false. The 48-start selection remains bounded/nonexhaustive; adding residual proposals is not a guarantee of monotonic downstream success.

Compared with the immediate, completely retained depth V1 failure, the prior per-pair consumer mismatch is no longer the stopping point: all four cases now complete the context ledger and actual registration. This is a concrete integration recovery. It does not establish the required localized measurement improvement. The older composite 2/4 comparison remains historically useful but carries the separately documented omitted-tests-initializer custody limitation; it is not a clean causal complete-source control. This packet's result is also 2/4, not an improved aggregate measurement score over that historical observation.

## Measurement oracles and preserved refusals

All eight measurement answers have successful publication validation, null publication/input-custody errors, unchanged scan digest bindings, and `independent_registration_replication=false`. The four V2 nested V1 reports are independently byte-equal under the canonical encoding to their corresponding standalone V1 measurements, and their source-result SHA-256 values match. They remain `RETAINED_EVIDENCE_NOT_ADDITIONAL_PREDICTIONS`. The actual publication validator revalidated the cover plan without a second cover refit; reported publication geometry row visits are 261, 261, 0, 0. Those positive-case validation visits are not a measurement rerun and were not performed by this POST.

All measurement reports retain `observed_bend_count=null`, `production_use_allowed=false`, conformance `INDETERMINATE`, joint configuration and continuous pose-family coverage `NOT_ESTABLISHED`. No diagnostic negative or partial report is converted into a manufacturing verdict.

U-R and U-A each retain one `MEASURED_CONDITIONAL` support group and one prediction, with overall status `PARTIAL`. Fixed scalar checks independently reproduce their original oracle errors from the retained measurement fields:

| Case | Truth radius / sweep | Measured radius mm | Measured sweep magnitude deg | Radius error mm | Sweep error deg |
| --- | --- | ---: | ---: | ---: | ---: |
| U-R | 5.3 mm / 90 deg, clean | 5.300000000000005 | 89.99999999999999 | 5.329070518200751e-15 | 1.4210854715202004e-14 |
| U-A | 5 mm / 90.5 deg, noisy | 5.000048286597504 | 90.49958264287886 | 4.828659750444331e-05 | 0.00041735712113677437 |

The unchanged clean limits are 2e-5 mm and 2e-5 degrees; noisy limits are 0.02 mm and 0.1 degree. The independent scalar checks also retain the non-nominal substitution guards: U-R differs from nominal 5 mm by at least 0.25 mm, and U-A differs from nominal 90 degrees by at least 0.35 degrees. Both use `free_circle`, the original free-circle tangency compatibility role and `ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM`. Physical trim endpoints, radius/sweep uncertainty intervals and datum-relative deviation remain null. Every published group's association retains accepted registration false and correct measured-minus-nominal radius/sweep-magnitude arithmetic. These nonlocalized cases do not qualify localized endpoint accuracy.

All four V1 diagnostics retain `UNAVAILABLE`, zero predictions and reason `NO_NONCONFLICTING_CANONICAL_SCAN_MEASUREMENTS`. Their conditional support-record counts are 18, 24, 5 and 7; those are not counts of usable measured groups.

L-R and L-A each retain one V2 support group with status `UNAVAILABLE`, null measurements, zero predictions, group reason **`NO_EXISTING_WHOLE_COMPONENT_RAW_SUPPORT_COVER`**, and overall reason **`NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS`**. Their first unchanged usable-group assertion raises `AssertionError`. The later required localized raw generator/axis endpoint oracles therefore do not execute and cannot be counted as passing. No localized radius, sweep, sampled-support endpoint, physical trim or completeness qualification is established. The retained group reason identifies the published stopping condition only; this POST does not refit or prove a geometric root cause.

## Resource and terminal reconciliation

| Resource / event | Frozen limit | Observed |
| --- | ---: | ---: |
| Worker CPU seconds | 6000 | 5023.520858 |
| Work-stop wall seconds | 6595 | 5027.257908792235 |
| Parent terminal wall seconds | 6600 | 5027.271550166886 |
| Sampled group peak RSS bytes | 4294967296 | 1253244928 |
| OUTPUT bytes | 1048576 | 18415 |
| All RESULT bytes | 570425344 | 123945682 |
| All START bytes | 65536 | 17594 |
| Scientific bytes total | 570490880 | 123963276 |

The footer's worker peak RSS equals the supervisor's sampled peak. Sum of the 16 call CPU times is 5020.018072 s; sum of call wall times is 5023.677519086001 s. Those are nested scientific-call clocks, not the supervisor's full work/setup/finalization clock, and are not substitutes for it. The ordered receipt phase times are work stop 5027.257908792235, reaped 5027.2579105841, group checked 5027.271404667292, terminal decision 5027.271550166886 seconds. All are inside the appropriate frozen deadlines.

`budget_stop`, `primary_error` and `cleanup_error` are null. The only observed group PID is 49810; final emptiness is confirmed both in the receipt and independent process snapshot. The inventory is complete with no errors. Sampled RSS/deadlines are supervisory checks, not a hard instantaneous allocation guarantee or hard-real-time guarantee for OS I/O. No runtime or source limit was changed during this attempt or POST.

## Exact 32-file scientific ledger

All paths below are relative to the attempt's `ARTIFACTS/`. The independently recomputed values match the complete inventory and all aggregate RESULT references.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| MEASUREMENT_RESULT_00_V1.json | 629296 | 57d13432209c1a9c5b1b916e775a1ae1ac77210c0a779f4afd62952dfdef508d |
| MEASUREMENT_RESULT_00_V2.json | 671700 | c005444577db7470296a66c2035230948410418e1814470d9e3ffda66c63dd88 |
| MEASUREMENT_RESULT_01_V1.json | 717392 | 8a99ef17a8963fcf86c8872b1c883780153bd8e3681c81c40a09b006fd5addab |
| MEASUREMENT_RESULT_01_V2.json | 761721 | bc00c3c02a055d2ef2fb88d46ae207982ff0647f9d8c43c30d4784e9ab433945 |
| MEASUREMENT_RESULT_02_V1.json | 154190 | 1a382e29e785e73e4a8d8a19d99684bf651cd3d6c51b5012a16d037bf7f1c2a1 |
| MEASUREMENT_RESULT_02_V2.json | 163691 | 42e5fc8ef34f2d57d5cd0f4307a0b7d12e8e863c50b24da69e9ef23cce5f0467 |
| MEASUREMENT_RESULT_03_V1.json | 204235 | 84175f88c2b7743f7831fcddad75188cebe7146ec6c21f82f229ca30ff520bc3 |
| MEASUREMENT_RESULT_03_V2.json | 215914 | 8e3a32f3ab406e67f97bdd50a59210582a0a4110ca322c832d756f1222b5e5bc |
| MEASUREMENT_START_00_V1.json | 1324 | 1a51643e9ee0d06d4dcc8246f057e0fb86b11e0f757019081f1aa73022fd9fd0 |
| MEASUREMENT_START_00_V2.json | 1324 | 0273b3faf5aa29db4c294e9895a914eb47fcafe8fa3fbc0ef23f1e8c55a9e336 |
| MEASUREMENT_START_01_V1.json | 1325 | b38c96f5b5c16a3399e42707b00b69a4530af6976cb16b2428e4f528f3198d7e |
| MEASUREMENT_START_01_V2.json | 1325 | 3647d3f434cc0d16953365bbecc7c96b2bb388ac31a84c11d77c846578a03a72 |
| MEASUREMENT_START_02_V1.json | 1325 | 02a0de91491318b8d8fc1318f6c02fd2829a4bc8575c56d953dee142c2375f58 |
| MEASUREMENT_START_02_V2.json | 1325 | 43bf75668f14bc17e0fd5990329958759f18184ca51845e58cc7214529b6a2fc |
| MEASUREMENT_START_03_V1.json | 1324 | 423a5eddaa19cba4626cabcc419f5fd439cbf8fe62cd3cd8aaad9c11ad2efb08 |
| MEASUREMENT_START_03_V2.json | 1324 | af518683d125d4a1a37cf54f511debead62cffdfa96772333ca82bfc1dd215a6 |
| PLANAR_RESULT_00_V11.json | 260172 | aa3f5546d13556fd43ee70fd3bc966fc3250ffb16a941111428037dbb64bd17b |
| PLANAR_RESULT_00_V12.json | 260277 | 611c1dea622ae14cb4f629aee64d5582e43ee1562d2f67aa713e54434d18ff97 |
| PLANAR_RESULT_01_V11.json | 263516 | fd786c12ff8bf2ce8fe0b0301125e5c573c6cbeb2cde354862002230bd26ebdf |
| PLANAR_RESULT_01_V12.json | 262331 | 9affe8ed6fdf6986f7ae97927296c564b0b8cc22f222d6fc0c788495aed96dfc |
| PLANAR_START_00_V11.json | 744 | e8def82a14604e0183f0b7c3795216844de92599d19935989dfb1c2d488c0b77 |
| PLANAR_START_00_V12.json | 744 | 72dc92a8de90ad813b0be5ab6bc110af669bbcd0f3777c8b1bba2d4695c9d3f6 |
| PLANAR_START_01_V11.json | 744 | c78d3c5ef0a5c3cdd5a30954cd742bae2bca41e7bf494726a061a19e7608e264 |
| PLANAR_START_01_V12.json | 744 | d828b85ba0bd71cba6f7774b9c48726d3ef6c92ecf1951f7a79dbd4047d786eb |
| REGISTRATION_RESULT_00_SHARED.json | 31623165 | 2bcb4452e158a5528c3a7c048077b526b4db0a4e27d7618bcac54f2e706a4bdd |
| REGISTRATION_RESULT_01_SHARED.json | 32200875 | 0c49c93232f92acab8c9167067220ba37ef882e60051193bb4d36a5aa9fabfb0 |
| REGISTRATION_RESULT_02_SHARED.json | 27855463 | 75f31edc73ec1dca098b9f1889afe788dd522ae9b907df9f1590a1df68e40ee7 |
| REGISTRATION_RESULT_03_SHARED.json | 27701744 | ae6808826a1ac61a9e810570bfe331ba92bbf571341c6633173096b45d5beb0d |
| REGISTRATION_START_00_SHARED.json | 1002 | 15ef1beb58bda55038902a4324d3243755ca24d5725cdd69ad49267871b3c3ad |
| REGISTRATION_START_01_SHARED.json | 1007 | 6fb96b84311ed5e0fa9a64b77bfc315b1af3784f8111b93cc61b1ca38e764a99 |
| REGISTRATION_START_02_SHARED.json | 1007 | 1f41bd6b705add150221317bbccc4f31aa3e176266ec6f11f06c60e8ac8a3fa7 |
| REGISTRATION_START_03_SHARED.json | 1006 | 86226e3bd9b815883c0c23cdd95e4f28c63a43dfed107d571f6efba6edc4f882 |

## Terminal disposition

Preserve this attempt, its exact failed aggregate, all 32 records, the source archive, and both prior RED packets. The artifact-level custody/binding admission is positive; the original connected qualification remains **RED, 2/4 required measurement positives**. No scientific threshold, oracle, source denominator, endpoint requirement, support-cover rule or production default is relaxed. Any correction or later experiment requires its own separately frozen source, reviewed protocol, exact PRE and authorized launch; this POST grants none of those.
