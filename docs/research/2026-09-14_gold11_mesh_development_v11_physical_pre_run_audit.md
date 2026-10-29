# Gold11 mesh development V11 — independent physical PRE-run audit (2026-09-14)

Reviewer: `independent-ai-pre-auditor-v11` (Claude Code, `claude-fable-5-1`), non-author, working READ-ONLY in
`/Users/idantal/ShermanQC/FullProject`. The author of the candidate was not consulted. Every claim below was
recomputed with my own hashing/parsing (`.venv/bin/python -c`-style snippets using only `json`, `ast`, `hashlib`,
`tarfile`, `os.lstat`, `importlib.metadata`; plus `shasum`, `wc`, `diff`, `cat`, `sed`, `grep`, `ls -la`, `stat`,
`sysctl hw.memsize`, `ps`). No test, geometry, runner (`--freeze`/`--execute`/`main`) or waiter was run. No raw data
(`data/`, `output/`, `outputs/`, `.ply`, `.step`), Gold/reference container (`data/bend_corpus/annotations/*`),
private cache (`/private/tmp/gold11-*`, `/var/folders/*`), `.env`/credential, HOLDOUT/email material, or V8/V9/V10
attempt content was opened, stat'ed or hashed; the single permitted exception was reading the `source_sha256` map of
`docs/research/gold11_mesh_development_v10_attempt/FROZEN.json` for comparison. The 17 raw scan/CAD paths were
treated as declarations only (strings inside the frozen case authority) and were never opened.

## Decision

**GREEN_FOR_FROZEN_CONSUMED_DEVELOPMENT_RUN** — every required check (items 1–9) passed with the concrete values
below; `blocking_findings` is empty. Machine-readable decision: `PRE_RUN_AUDIT.json` in this directory,
`bundle_sha256 = 0a03a2984325c3092e2eaef21d29d5be49e2c626292bfa84984b9cc0447e4c65`.

This is an engineering audit of custody and declarations (frozen bytes, hashes, constants, limits, case scope,
runtime metadata, archive, waiter contract, process quiescence). It is **not** manufacturing approval, **not** a
factory or production qualification, **not** a score forecast, and it establishes no physical accuracy, calibration
or historical comparability. It authorizes only the single scoped `--execute` of this exact frozen attempt
(`docs/research/gold11_mesh_development_v11_attempt`, FROZEN.json on-disk sha256
`b4260576f1d6024233c410fd628958446bdea1cea1b5fdfa67394a885e2a86f5`) through the waiter
`scripts/gold11_v11_execute_waiter.py`, once, under the unchanged protocol. Any change to the attempt tree, the
frozen sources, the runtime or the archive before launch voids this audit.

## 1. FROZEN.json, snapshot and runner constants

| Item | Value |
| --- | --- |
| FROZEN.json | `docs/research/gold11_mesh_development_v11_attempt/FROZEN.json`, 61999 bytes (≤ 256 KiB), mode 0600, on-disk sha256 `b4260576f1d6024233c410fd628958446bdea1cea1b5fdfa67394a885e2a86f5` |
| Canonical bundle sha256 | `0a03a2984325c3092e2eaef21d29d5be49e2c626292bfa84984b9cc0447e4c65` = sha256 of `json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()` (61998 canonical bytes); recomputed after decoding FROZEN.json with the runner's own `decode()` semantics (duplicate-key hook, non-finite rejection) — identical to what `verify_pre_run_audit` computes via `digest(encoded(bundle))` (runner lines 536–556, 1923–1930); also equal to the manifest's `bundle_sha256_canonical` |
| schema | `GOLD11_MESH_DEVELOPMENT_V11` |
| source_sha256 entries | 107; three-way agreement (working tree == snapshot copy == FROZEN value) **107/107, 0 disagreements**; snapshot file set == key set (0 extra, 0 missing) |
| SOURCES tuple (runner, ast) | 107 distinct names, first element is `SOURCE_BASIS`; `set(SOURCES) == set(FROZEN.source_sha256)` **True** |
| SOURCE_BASIS / SHA | `scripts/run_gold11_mesh_development_v10.py` / `f7fd083f8b74a22e7fd1afe872ae8b9b2bfbc94f4b7024fc8c2b46494208da92`; working tree, V11 snapshot, FROZEN value and the terminal V10 attempt's FROZEN.json `source_sha256` entry are all this value (byte-identical V10 basis) |
| V10 → V11 frozen map | 96 common entries, **0 changed**, 0 removed, 11 new (`backend/cad_guided_{context_measurement_v4,measurement_v3,mesh_offline_v18,native_feature_bridge_v2,support_cover_measurement_v5}.py`, `scripts/{gold11_contact_scoring_v5,gold11_schedule_projection_v6,gold11_v11_execute_waiter,run_gold11_mesh_development_v11}.py`, the V11 protocol and delta-design documents) |
| Runner constants vs bundle | Inert evaluation: the 48 top-level `Assign` nodes of lines 55–533 were selected by `ast`, scanned against a call allowlist (`dict/tuple/list/Path/json.loads/json.dumps/str/str.startswith/str.endswith` only), exec'd in a namespace holding only `json`, `math`, `Path`; `arm_request` was re-implemented from the read source (lines 603–621). **All match**: schema, source_profile, config_delta_from_v10, pose_admission_declaration, measurement_policy_declaration, config, limits, catalog_sha256, selected_reference_sha256, arms, arm_definitions, comparator, position_error_declaration, certificate_caps, retained_control_policy, arm_requests (both arms), scoring_transport, two_plane_declaration, artifact_policy, historical_comparability (`UNESTABLISHED`), production_use_allowed (`false`). Bundle key set equals the 24 keys assembled by `freeze()` (lines 1124–1137) |
| arm request sha256 (canonical) | control `9ef36d73e228ffe31244bea873644f82f916ce2a66b18b4b1f6fcf4a064d57ea`; challenger `49fc41dabc2b4c97aae55193d4433df2c738c73d073def865ee900637963dd19` |
| Other pins | runner V11 `8432203ec6c475644b2f4bf25aaf12f613a6451bdfed48355997a8b279f2d978`; `REVIEW_SHA` `41a79da1…` (= frozen review copy); sampler `80316d00…` (= frozen); terminal-artifacts helper `3f4c5754…` (= frozen and = `artifact_policy.helper_sha256`); supervision `9eec779c…` (= frozen `scripts/gold11_owned_process_supervision_v3.py`); all 11 `SCORING_SOURCES` are in the bundle; `RELEASE_AUTHORIZED = True`, `TIMING_RELEASE_STATUS = REVIEWED_V11_EXACT_TIMING_RELEASE` |

## 2. CONFIG / LIMITS / CERTIFICATE_CAPS delta from the frozen V10 basis

Evaluated by the same inert AST method from the snapshot copy of the V10 basis (sha `f7fd083f…`, nodes LIMITS 266–275,
CERTIFICATE_CAPS 278–279, CONFIG 286–387) and the V11 runner (LIMITS 315–324, CERTIFICATE_CAPS 327–328, CONFIG 335–439).

- LIMITS: 19 leaves, **identical**. CERTIFICATE_CAPS: 3 leaves, **identical** (initialization 1 024 000 000, surface 128 000 000, registration 8 000 000 000).
- CONFIG: 183 → 188 leaves; **0 changed, 0 removed, exactly 5 added**, all under `layer_policy`:
  `layer_mode_bin_mm 0.05`, `layer_membership_band_mm 0.1`, `max_bridging_row_fraction 0.35`, `layer_refit_iterations 6`,
  `layer_mode_ambiguity_fraction 0.25`. The eight V1 fields are unchanged (`enabled true, layer_split_gap_mm 0.3,
  min_layer_rows 12, max_layers_reported 8, max_layer_normal_angle_deg 3.0, max_layer_gap_mm 6.0,
  min_projected_interval_overlap 0.5, max_layer_rms_mm 0.1`); V11 layer_policy has 13 fields.
- `refinement_allowance.cpu_budget_seconds = 1860.0 == LIMITS.case_cpu_seconds = 1860`; `cpu_guard_seconds = 1500.0`.
- The evaluated V11 CONFIG/LIMITS/CAPS equal the FROZEN.json copies.

## 3. Case authority (frozen snapshot copy only)

`docs/research/gold11_mesh_development_v11_attempt/source/scripts/review_gold11_inputs_v1.py`: 25616 bytes, sha256
`41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955` (required value). `CASES = json.loads(r'''…''')`
parsed exactly as the runner does (single `Assign`, `json.loads` call with one literal argument, `ast.literal_eval`,
duplicate-key/non-finite rejecting decode):

- 11 cases; 11 distinct `scan_sha256`; **6 distinct `cad_sha256`**; 17 distinct absolute raw paths (11 scan + 6 CAD).
- raw17 map `{path: sha256}` canonical sha256 **`2459f4e93fe4204889ade6f8a4150d3543ec9f96625f39aaec811f003db4d157`** (required value).
- 11-case array canonical sha256 `943268be4ee3d33301ab44dfb3e6f85fb0d34de852f788464f1d438ffe43813e` == canonical sha256 of `FROZEN.json["cases"]`; arrays equal and in the same order:
  CG08506AE97D, CG191DEC7F81, CG1E0BBA2933, CG4BDD1371B6, CG4FAF273093, CG64C412CC73, CG65CD575B1C, CG84C8FA12D2, CGE4B0549E41, CGEBA9A869AC, CGF41F49066A.
- Declared reference bindings (declarations only; the runner verifies them at scoring time): CATALOG_SHA
  `4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36`, SELECTED_SHA
  `d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc` — identical in the runner constants, FROZEN.json and the protocol.
- No raw file was opened or hashed.

## 4. Runtime metadata

Replicated `runtime_metadata()` (runner 1105–1114) exactly: `Path(sys.executable).resolve()`, `sys.version`,
`sys.platform`, `os.uname().machine`, and `importlib.metadata.Distribution.discover(name=…, path=[<.venv site-packages>])`
(each of numpy, scipy, cadquery-ocp resolves to exactly one distribution). Live == FROZEN **True**:
python_executable `/Users/idantal/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/bin/python3.11`
(`.venv/bin/python` → symlink `…/cpython-3.11-macos-aarch64-none/bin/python3.11` → resolves there);
python_version `3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]`; platform `darwin`; machine `arm64`;
numpy `2.4.6`; scipy `1.17.1`; cadquery-ocp `7.9.3.1.1`; attestation `VERSION_METADATA_NOT_BINARY_ATTESTATION`.

## 5. Attempt tree, limits and arm requests

- Attempt directory (mode 0755) top-level entries: exactly `FROZEN.json`, `source` — no STARTED/COMPLETED markers; the
  three outer records (`…_attempt_outer_exit.json`, `…_outer_pid.txt`, `…_outer_stdout.txt`) are absent
  (`os.path.lexists` False). `docs/research` contains only the attempt directory and the archive directory under the
  `gold11_mesh_development_v11` prefix.
- Walk with `lstat`: 7 directories (all 0755), **108 regular files (all 0600)** = FROZEN.json + 107 sources, **0 symlinks**, 0 other.
- Frozen limits vs the protocol table (`docs/research/2026-09-14_gold11_mesh_development_v11_protocol.md`, lines 140–153) — **all equal**:
  case wall/CPU 3540/1860 s; allowance guard 1500 s; worker sampled tree / coordinator / outer tree 8/4/12 GiB
  (8589934592 / 4294967296 / 12884901888); scan/CAD input 64/8 MiB (67108864 / 8388608); leaf log / result 1/128 MiB;
  compact per-case 8 MiB (8388608, also `scoring_transport.inner_projection_bytes` and `retained_schedule_wrapper_bytes`);
  scorer 180/120 s / 2 GiB (2147483648); scorer request 192 MiB (201326592); scorer result 32 MiB (33554432);
  coordinator log 2 MiB / parent read 180 s; campaign 104400 s / reserve 2400 s; stage-checkpoint lines 65536 bytes
  (`STAGE_LOG_DECLARATION.checkpoint_stdout_bytes`, config-delta text).
- Arm requests: control `scan_plane_mode v7`, `schedule_request.mode v1`, `context_request.mode v4`,
  `two_plane_context_mode v4`, `scan_plane_configuration null`; challenger `composite_v1`, `v2`, `v4`, `v4`, with seeded
  and composite budgets. Both arms: workflow `CAD_GUIDED_MESH_OFFLINE_V13`, implementation
  `CAD_GUIDED_MESH_OFFLINE_V18_CONTEXT_V4_COMPLETION_V1_KERNEL_V3_NATIVE_BRIDGE_V2_ALLOWANCE_V1`, `two_plane_enabled true`.
  Matches the protocol (lines 75–77).

## 6. External archive

Exactly one directory `docs/research/gold11_mesh_development_v11_physical_frozen.b9bcd723/` (mode 0700, no symlinks)
containing `SOURCE.tar` (mode 0600, **4389376 bytes**, sha256
**`94601f14ed4276cab8b01c7f1afa040d56a4432aa2b51f23319cf32b5ed54e3c`**) and `SOURCE.tar.members.sha256.json`
(mode 0600, 41667 bytes, sha256 `34ecdcc64f0883e546299bb025cfe350e2b962014cf8a592eca891dd152d5d31`).

Streamed every tar member read-only with `tarfile` and hashed each payload: 114 entries = 108 regular members + 6
directories (`source`, `source/tests`, `source/backend`, `source/docs`, `source/scripts`, `source/docs/research`, all
0755), 0 other types. Expected 108 members (`FROZEN.json` + `source/<107 paths>`): **108 ok, 0 missing, 0 extra,
0 mismatches** — every payload sha256 equals both the on-disk snapshot sha256 and the FROZEN.json `source_sha256` value;
`FROZEN.json` is archived with its on-disk sha256 `b4260576…`. Manifest (schema
`GOLD11_V11_FROZEN_SOURCE_ARCHIVE_MEMBER_PAYLOAD_MANIFEST_V1`, written 2026-09-14T02:17:14Z): 108 member records,
**108 agree** member for member (name, bytes, mode, declared/on-disk/payload sha256), 0 disagree, 0 names absent from
the tar; its own `missing_from_archive`, `extra_in_archive`, `mismatches`, `unexpected` lists are empty;
`source_count_archived = source_count_declared = 107`; `bundle_sha256_canonical = 0a03a298…` (matches). Member
metadata: modes 0600, uid/gid 501/20, single mtime 1789352170 (2026-09-14T02:16:10Z).

## 7. Documents

- Protocol `2026-09-14_gold11_mesh_development_v11_protocol.md` (sha256 `987e6a165c820ff5a77918dcf64ee2d04b86167556ec2ae8a050e238324ef1e9`, frozen) names the V11 mechanism (measurement kernel V3 two-mode band-membership flange layer separation plus its consumer bindings bridge V2 / context V4 / cover V5 / contact V5 / projection V6 / mesh offline V18) and the V10 basis sha `f7fd083f…` (lines 73–74); its case-authority, catalog and selected-reference hashes match check 3.
- Delta design `2026-09-14_gold11_mesh_development_v11_delta_design.md` (sha256 `468fbfc509893e5ac060ed326d731e6993819c46e4706ddfbac35b9169139f35`, frozen) names the same mechanism and basis hash (lines 9–10) and the exact five-field layer-policy delta (lines 27–31).
- Review response `2026-09-14_fable51_v11_review_response.md` (sha256 `af0bda6872a92872c56312c030f24ac0e4cfd4795ead3f6966b1553c47ad71de`, 27707 bytes): closure hash table has **24 rows; 24/24 match the current working tree** (sha256 and byte size). The 11 rows that are in the frozen bundle also equal their FROZEN.json values (`backend/cad_guided_measurement_v3.py 8fd5c3e6…`, `native_feature_bridge_v2 ecac003e…`, `context_measurement_v4 7d12b922…`, `support_cover_measurement_v5 7a0b40cf…`, `mesh_offline_v18 57ae0310…`, `gold11_contact_scoring_v5 dd0963c2…`, `gold11_schedule_projection_v6 b419903c…`, runner `8432203e…`, waiter `1a7be9fd…`, delta design `468fbfc5…`, protocol `987e6a16…`); the 13 test/narrative rows match the working tree and are outside the bundle by design.
- Coordination `2026-09-13_fable51_v11_codex_coordination.md` (sha256 `716ea68231e4fda11e3f0fbe0595fab75cda2f08fcc06835c34a5d1c5b26f218`): records root's acceptance of the bounded V10→V11 runner/waiter source delta at runner `8432203e…` / waiter `1a7be9fd…` (01:22–01:28 UTC section) and the final "Bounded candidate source closure is ACCEPTED at the 24 published pins" checkpoint (02:14 UTC section), pinning the review response at `af0bda68…` — equal to the current file. It explicitly states that this acceptance is not PRE, a score, or a launch authorization, and that PRE must be GREEN with no blockers before the single waiter execute.
- Layer-kernel Codex audit (sha256 `f0f12ce875af085a7c7ba8967046787908b74e5bee979d5be1d03f11793d3b22`, the hash the coordination record cites): both P1 and both P2 findings source-closed; bounded kernel acceptance retained at kernel `8fd5c3e6…` (= frozen) and test `9bf3cb07…` (= working tree). **No unresolved blocking finding.** Open non-blocking prose item: header :16 "fit quality never selects".
- Consumer-chain Codex audit (sha256 `b6ba7bf9e9b2821cf80feb36141e85cf5398095aecf5bcd019d25f59fdeca890`, the hash the coordination record cites): both original P2 findings closed at source; full-bytes fixture closures accepted at `8b0556cc…` (= working tree); final section "ACCEPTED in the requested source-review scope". **No unresolved blocking finding.** Open non-blocking prose items: fixture lines 32, 42, 348–350, 437 broader than the executable contract; "24 vs 26 invalid cases" wording. Its earlier runner pin `e8de63e4…` is a superseded draft (see non-blocking note 2).

## 8. Waiter contract (`scripts/gold11_v11_execute_waiter.py`)

sha256 `1a7be9fda3d9551e0606764f178f69967c68dfce9d5be576fcee8c6a883e84b9`, 9213 bytes; `diff` working tree vs frozen
copy: identical. Verified in the text: O_EXCL reservation of the three outer records beside the attempt directory
(`_write_exclusive`, lines 62–66, 101–103; pre-existing record → 97, reservation race → 96); `fcntl.flock(LOCK_EX|LOCK_NB)`
held on the reservation for the waiter's life (143; locked → 92); one-time token `secrets.token_hex(16)` passed only via
`GOLD11_V11_WAITER_TOKEN` and required to match the RESERVED line and the pending exit record's nonce (99, 108, 131–134,
155–159); atomic exit record (temp + `os.replace`, 68–72) carrying the ACTUAL `proc.wait()` return code with
`OBSERVED_BY_WAITING_PARENT`, runner pid and observer pid (175, 181–183); unobserved-on-failure semantics
(`UNOBSERVED_PENDING` default, `UNOBSERVED_WAITER_LAUNCH_FAILED`, `UNOBSERVED_WAITING_PARENT_FAILED:<Exception>` then
re-raise, 75–80, 119–122, 176–180); direct `--waiter` refusal without a reservation token/config (94/93/92/91) touching no
record; the exact command `[<ROOT>/.venv/bin/python, -I, -S, -B, -u, scripts/run_gold11_mesh_development_v11.py, --execute]`
with `OMP_NUM_THREADS=1`, cwd `<ROOT>`, stdin DEVNULL, stdout+stderr into the reserved stdout record (32–33, 166–170);
`PRISTINE_TREE = ["FROZEN.json", "PRE_RUN_AUDIT.json", "source"]` checked by entry NAMES only (34, 93–95); overlap
refusal pattern `run_gold11_|check_|pytest|cad_guided|_gold11_v\d+_exact_scoring` (37, 87–92).

## 9. No-overlap and memory

`ps -axo pid,pgid,etime,rss,command` twice, 399 s apart: snapshot 1 at **2026-09-14T02:17:54Z**, snapshot 2 at
**2026-09-14T02:24:33Z** (files `ps_snapshot_1.txt`, `ps_snapshot_2.txt`). Lines matching
`run_gold11_|check_|pytest|cad_guided|execute_waiter|dump_fullbytes|exact_scoring|python`:

- snapshot 1: `81241 81241 03-13:05:14 849824 /Users/idantal/tavnit/.venv/bin/python3 .venv/bin/tavnit serve --port 9017` (an unrelated project's service, not Sherman geometry/benchmark) — nothing else.
- snapshot 2: the same PID 81241 line, plus my own transient `/bin/zsh -c …` shell running the grep — nothing else.

No `run_gold11_*`, `check_*`, pytest on this repository, `cad_guided`, `execute_waiter` or `dump_fullbytes` process
was live at either snapshot. `sysctl hw.memsize` = 137438953472 bytes = **128 GiB**, above the 16 GiB repository
ceiling and the 12 GiB campaign outer limit.

## Non-blocking notes (none affects the decision)

1. The waiter's own overlap pattern does not name other-version `execute_waiter` or `dump_fullbytes` processes; this PRE's independent `ps` checks covered them (none live). Defense in depth only.
2. The consumer-chain audit's source-closure sections pinned an earlier runner draft (`e8de63e4…`); the frozen runner `8432203e…` was accepted by root in the coordination record and matches the review-response table, and its text carries the bridge-V2/kernel-V3 identifiers the audit required (ARM_DEFINITIONS 181–195, COMPARATOR 198).
3. Auditors' explicitly non-blocking prose items remain open (kernel header :16; full-bytes fixture prose 32/42/348–350/437, a file outside the frozen bundle; "24 vs 26" wording). Documentation wording only.
4. Runner comment at lines 441–443 says "request context V3" while the constants and FROZEN.json declare context V4 (`BOTH_SUCCESSOR_ARMS_EXPLICIT_CONTEXT_V4`, mode `v4`). Stale comment; values verified.
5. The full-bytes real-connected fixture exercised the control lane only; the challenger lane's modules (composite surface roles v1, two-plane schedule v2, seeded/composite budgets) are byte-identical to the terminal V10 bundle (96/96 common entries unchanged) and V11 changes nothing there.
6. Runtime binding is by resolved interpreter path and distribution versions; if the uv-managed interpreter or numpy/scipy/cadquery-ocp change before `--execute`, `verify_bundle` refuses (`FROZEN_RUNTIME_METADATA_CHANGED`). Advisory.
7. Archive metadata reported: single member mtime 1789352170 (2026-09-14T02:16:10Z), uid/gid 501/20, member mode 0600, directory members 0755; attempt directory 0755 versus archive directory 0700.
8. `PRE_RUN_AUDIT.json` is written outside the repository (this PRE never writes into the attempt tree). The author must copy it byte-for-byte to `docs/research/gold11_mesh_development_v11_attempt/PRE_RUN_AUDIT.json` (its sha256 is printed below and in my final summary); `verify_pre_run_audit` (runner 1923–1930) then requires `decision == GREEN_FOR_FROZEN_CONSUMED_DEVELOPMENT_RUN`, `bundle_sha256 == 0a03a298…`, a non-empty `reviewer` and `blocking_findings == []`, and the waiter's launcher requires the tree to be exactly `FROZEN.json`, `PRE_RUN_AUDIT.json`, `source`.

## Files written by this audit (all under `/Users/idantal/.claude/jobs/1519fdc2/tmp/pre_v11/`)

- `PRE_RUN_AUDIT.json` — the machine-readable decision (keys: decision, bundle_sha256, reviewer, blocking_findings, non_blocking_findings, checks, written_utc).
- `2026-09-14_gold11_mesh_development_v11_physical_pre_run_audit.md` — this narrative.
- Scratch: `ps_snapshot_1.txt`, `ps_snapshot_2.txt` (raw `ps` tables with UTC stamps); `scratch_check1_bundle_constants.json`, `scratch_check2_config_delta.json`, `scratch_check3_cases.json`, `scratch_check4_runtime.json`, `scratch_check5_tree.json`, `scratch_check6_archive.json`, `scratch_check7_docs_hashes.json` (per-check ledgers from which the JSON was assembled); `scratch_assembly_values.json` (assembly values).
- Disclosure: the Claude Code harness itself persisted two oversized outputs of my FROZEN.json parse command under `~/.claude/projects/-Users-idantal-ShermanQC-FullProject/…/tool-results/` (harness-generated copies of my own command output, not writes I directed); nothing else was written anywhere, and nothing was written inside the repository.

## Scope statement

This audit binds the frozen source/runtime bundle, the raw17 declaration and full11 order, the unchanged 71-line
reference bindings (by declared hash only), both arm requests, the allowance, the scorer transport, the limits, the
failure-handling declarations and the archive custody, and confirms process quiescence at the time of the check. It is
an engineering audit of custody and declarations — not manufacturing approval, not a production or factory release,
not a score forecast, and not evidence of physical measurement accuracy or historical comparability (the bundle itself
declares `historical_comparability: UNESTABLISHED`, `production_use_allowed: false`). It authorizes only the single
scoped `--execute` of this frozen attempt through `scripts/gold11_v11_execute_waiter.py`; independent POST remains
required before any result is accepted.
