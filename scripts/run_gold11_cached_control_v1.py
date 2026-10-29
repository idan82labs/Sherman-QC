"""One-shot, independently gated cached Gold11 control; no fresh inference.

--freeze reads named SOURCE/DOCUMENT files only, never cached or reference data.
--execute requires the exact frozen bundle's independent PRE_RUN_AUDIT.json.
Its one guarded worker reads only the declared22 cached files and fixed catalog,
projects the immutable11/6/71 reference selection, and scores ALL four variants.
No raw scan/CAD stat/read, manifest expansion, threshold search, resume or retry.
COMPLETED is provisional until independent post-audit; FINALIZATION_FAILED vetoes
it. Partial files and every failed attempt stay retained, never retried in place.
Current retained-byte reconstruction is NOT historical engine replay or QC proof.
Python audit hooks protect accidental I/O, not hostile native OS syscalls/users.
"""
import argparse
import ast
import hashlib
from importlib import metadata as import_metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import secrets
import stat
import sys
import time


ROOT = Path("/Users/idantal/ShermanQC/FullProject")
RUN = ROOT / "docs/research/gold11_cached_control_v1_attempt"
RUNNER = "scripts/run_gold11_cached_control_v1.py"
SUPPORT = "scripts/run_gold11_mesh_development_v2.py"
SUPPORT_SHA = "94d51b90faef0d88b15ed2d3d84d5b7d36ba94bfb6f63f90b9cc555a35e26422"
REVIEW = "scripts/review_gold11_inputs_v1.py"
REVIEW_SHA = "41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955"
PROTOCOL = "docs/research/2026-09-10_gold11_cached_control_v1_protocol.md"
CATALOG = "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
CATALOG_SHA = "4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36"
SELECTED_SHA = "d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc"
A0_ROOT = "output/research/f1-gold27-20260803/structured_union_exact_v1/cases"
CAD_ROOT = "output/research/f1-cad-guided-gold13-e2e-20260822/predictions_scan_frame_gated"
VARIANTS = ("scan_only_a0", "cad_only_gated", "intersection", "union")
LIMITS = dict(total_wall_seconds=90, worker_wall_seconds=80, terminal_reserve_seconds=10,
    worker_cpu_seconds=55, aggregate_cpu_seconds=60, global_tree_rss_bytes=1024**3,
    log_bytes=1024**2, prediction_file_bytes=16*1024**2, catalog_bytes=8*1024**2,
    selected_reference_bytes=4*1024**2, report_bytes=24*1024**2,
    scientific_artifact_bytes=32*1024**2, source_file_bytes=2*1024**2,
    source_total_bytes=4*1024**2, control_file_bytes=256*1024)
SOURCES = (
    RUNNER, SUPPORT, REVIEW, PROTOCOL,
    "scripts/gold11_cached_fusion_scoring_v1.py", "scripts/gold11_retained_fusion_v1.py",
    "scripts/gold11_retained_prediction_scoring_v1.py", "scripts/gold11_mesh_scoring_v1.py",
    "scripts/evaluate_f1_against_manual_gold.py", "scripts/f1_physical_evaluation_contract_v2.py",
    "scripts/diagnose_f1_cad_scan_fusion_gold11.py",
    "tests/test_gold11_retained_fusion_v1.py", "tests/test_gold11_cached_fusion_scoring_v1.py",
    "tests/test_gold11_cached_control_v1.py",
    "docs/research/2026-09-10_gold11_retained_control_recovery.md",
    "docs/research/2026-09-10_gold11_a0_provenance_trace_v7.md",
)
# Released recovery/V7 tables, NOT discovered from a pool or mutable manifest.
# case, A0 token, approved scan SHA, A0 file SHA, CAD file SHA, reported A0/CAD/union.
BINDINGS = (
    ("CG08506AE97D", "C003", "cec08dfca9ab309ff3eef42cd3dc72eb33168e1d43029f3a0658f2923e4e171b", "df42a588cf1747e358c9f8fd8704a5b3c7d01388e5b85cc6df271edf2680d060", "51d9238df430c2090eb5d5787e24dcdc90aa299dbcb1bf279b59d18d332a014e", (1, 4, 4)),
    ("CG191DEC7F81", "C021", "7a1773eca9fab616afef498a9f9831e67eacb17fcc0d170607490a9e9db37020", "5c0ef9e5743e854550e40dea7808ec11570b897a076dd883dab67e1ae1f9b06a", "ecd34f506c1510d42e0d4bc4f6e540613c9477dfe307b221a36d7fdae2f6631d", (3, 1, 3)),
    ("CG1E0BBA2933", "C011", "4f8188b4d1a844305134801c3bb5364270250416d2f4dc43f0b05d37cc7962b9", "2eac14dd768471a540460f60ac8a696ae04a7d0abfcb4880618bfd4965fb5353", "3d42db809281eeca6b8d1c5ae0f8c97ec2aea8f7f2d55c578ad058778521e6e8", (5, 9, 11)),
    ("CG4BDD1371B6", "C008", "d1732de180230bdbf0caa6eaf454a33071c206060aea276dce67a9abcf13b038", "bdefea6cfab50cec78907cc997b1ad28b1ac8755a1b448444017f7a6f6216d4a", "a2c5f4b000f65932c44b238414fd428ca0b2f9f9f31cf5e98e6fa238c48fcc3e", (2, 3, 3)),
    ("CG4FAF273093", "C022", "70b39f530f00a72dd99acf00b4a065f45bfe4f50185e3579c3ba26143f7d2ade", "af8ce6e7827b1adb9fea871ceb5406d28e2871301a145268290eca3708b2dcc9", "2fb45cc5366c9527b8f84fd09ab90beae70978b653c6cea6f8354046ef96be3b", (1, 3, 3)),
    ("CG64C412CC73", "C015", "152f1b1aacb20920a601f7a8197d4c89a111688325bcf557a6a731641f306614", "f464aa39f3a16d1dd6272fc08a735b38c215ca8b7ad5b32dcdfe93101db61176", "7fd5fd125403af4f965f22a61cbe7781290a6e1326763eea47d74ff88d4172cd", (5, 1, 6)),
    ("CG65CD575B1C", "C009", "07062eee17aa4e41721b93cdb69ddb4b752e6713a16c5aab0a10df9c86d07880", "6160c39dee6998208c9f333855e9e402ed486282dddee7494b43a0104b0837a7", "dfd6e66b5716a9c2ecb1b12798a8a1b2ae6b93e1fdeba3b7bfbad6502146464d", (1, 2, 2)),
    ("CG84C8FA12D2", "C001", "71cf0b216f43284e401c891091cd56c10321aa3eebccde204f8d206683999215", "3befcc1d963ce9ab70a915e174c3be5503c5a81a46fa16c9f8883c23db9b7914", "685a2b0ddc86af01e4179f450e7bf45a5ed471267db7e04cae01364c2785d966", (5, 11, 11)),
    ("CGE4B0549E41", "C002", "08686dc1d5d4c89725393907082e8cc4e4c0f5a21427697760ff44d29c7939fc", "eb0b050d970b48108c662f8e943002bfdaf6c3a01a978b4483ca1a6d0c83015f", "27713f575b55867de7d73cdfbba960a65bc40d25c2e51179831b02daab9066ad", (1, 2, 2)),
    ("CGEBA9A869AC", "C010", "f2475286312b7e11be773166afae5d7eaacd826f8b59cee84289d05b4a222b63", "9e602684ecd345ace1a840743f0412d14d9106629c1a760dd1b0133d83fa8299", "22209579f6ee4a965d0b6b96b6140af9e6e68aade335e9c43bce95de54618231", (5, 1, 6)),
    ("CGF41F49066A", "C012", "2714fb7e0c0b681196da62bf368e60c0635b97000f34df3241ca1b70ee56faba", "02b0e6447e688951294e56a672ff42271ca74ebe8232345e386917aa2ed3b612", "8f6286463ca37c7ebc76b124ab7d48c63faa53c411295a66374c205c612b5d34", (4, 7, 9)),
)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def decode(blob):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("DUPLICATE_JSON_KEY")
            result[key] = value
        return result
    def reject(_):
        raise ValueError("NONFINITE_JSON")
    return json.loads(blob, object_pairs_hook=object_pairs, parse_constant=reject)


def read_fixed(root, relative, cap, expected=None, *, allow_empty=False, opener=None):
    """Descriptor-relative fixed path; no symlinks in ANY path component."""
    if type(relative) is not str:
        raise ValueError("FIXED_RELATIVE_PATH_REQUIRED")
    parts = Path(relative).parts
    if (not parts or Path(relative).is_absolute() or relative != "/".join(parts)
            or any(part in ("", ".", "..") for part in parts)):
        raise ValueError("FIXED_RELATIVE_PATH_REQUIRED")
    opener = os.open if opener is None else opener
    directory = opener(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = None
    try:
        for part in parts[:-1]:
            child = opener(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = opener(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not int(not allow_empty) <= before.st_size <= cap:
            raise ValueError("NONREGULAR_OR_BYTE_BUDGET")
        blob = bytearray()
        while True:
            block = os.read(fd, min(1024**2, cap+1-len(blob)))
            if not block:
                break
            blob.extend(block)
            if len(blob) > cap:
                raise ValueError("BYTE_BUDGET")
        after = os.fstat(fd)
        keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in keys):
            raise ValueError("SOURCE_CHANGED_DURING_READ")
        blob = bytes(blob)
        if expected is not None and digest(blob) != expected:
            raise ValueError("DECLARED_HASH_MISMATCH")
        return blob
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


def publish_bytes(path, blob):
    """Complete bytes published atomically and exclusively; never replace."""
    path = Path(path)
    temporary = path.with_name("." + path.name + ".partial-" + secrets.token_hex(8))
    with temporary.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path, follow_symlinks=False)
    temporary.unlink()
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_json(path, value):
    blob = encoded(value)
    if len(blob) > LIMITS["control_file_bytes"]:
        raise ValueError("CONTROL_RECORD_BYTE_BUDGET")
    publish_bytes(path, blob)


def support_module(frozen=False):
    # Inspected V2 imports only stdlib at module initialization. Its native,
    # physical-input and old-run functions are NEVER called by this runner.
    base = RUN / "source" if frozen else ROOT
    blob = read_fixed(base, SUPPORT, LIMITS["source_file_bytes"], SUPPORT_SHA)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("_gold11_cached_supervision", base / SUPPORT)
    module = importlib.util.module_from_spec(spec)
    # Execute the exact verified buffer; do not reopen the source via a loader.
    exec(compile(blob, str(base / SUPPORT), "exec"), module.__dict__)
    return module


def cases_from_source(blob):
    if digest(blob) != REVIEW_SHA:
        raise ValueError("REVIEW_SOURCE_BINDING_MISMATCH")
    nodes = [node.value for node in ast.parse(blob).body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "CASES" for target in node.targets)]
    if len(nodes) != 1:
        raise ValueError("CASE_CONSTANT_SCHEMA")
    call = nodes[0]
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name) and call.func.value.id == "json"
            and call.func.attr == "loads" and len(call.args) == 1 and not call.keywords):
        raise ValueError("CASE_CONSTANT_SCHEMA")
    rows = decode(ast.literal_eval(call.args[0]))
    by_id = {row["case_token"]: row for row in rows}
    if len(rows) != 11 or set(by_id) != {row[0] for row in BINDINGS}:
        raise ValueError("FIXED_ELEVEN_CASE_BINDING")
    cases = []
    for case_id, token, scan_sha, a0_sha, cad_sha, counts in BINDINGS:
        reviewed = by_id[case_id]
        if reviewed["scan_sha256"] != scan_sha:
            raise ValueError("APPROVED_RAW_SHA_BINDING_MISMATCH")
        cases.append(dict(case_id=case_id, family_id=reviewed["part_key"], scan_sha256=scan_sha,
            reference_id=reviewed["reference_id"], reference_scan_path=reviewed["scan_path"],
            a0_token=token, a0_path=f"{A0_ROOT}/{token}/prediction_owned_regions.json",
            a0_sha256=a0_sha, cad_path=f"{CAD_ROOT}/{case_id}/semantic_result.json",
            cad_sha256=cad_sha, historical_reported_counts=dict(zip(
                ("scan_only_a0", "cad_only_gated", "union"), counts))))
    if len({row["family_id"] for row in cases}) != 6 or len({row["scan_sha256"] for row in cases}) != 11:
        raise ValueError("FIXED_FAMILY_OR_SCAN_DENOMINATOR")
    return cases


def runtime_metadata(support):
    versions = {}
    for name in ("numpy", "scipy"):
        distributions = list(import_metadata.Distribution.discover(name=name, path=[support.runtime_roots()[0]]))
        if len(distributions) != 1:
            raise ValueError("RUNTIME_DISTRIBUTION_METADATA_AMBIGUOUS")
        versions[name] = distributions[0].version
    return dict(python_executable=str(Path(sys.executable).resolve()), python_version=sys.version,
                platform=sys.platform, machine=os.uname().machine, distributions=versions,
                attestation="VERSION_METADATA_NOT_BINARY_ATTESTATION")


def freeze():
    support = support_module()
    with support.deadline(20):
        blobs = {name: read_fixed(ROOT, name, LIMITS["source_file_bytes"]) for name in SOURCES}
        if sum(map(len, blobs.values())) > LIMITS["source_total_bytes"]:
            raise ValueError("SOURCE_CLOSURE_BYTE_BUDGET")
        bundle = dict(schema="GOLD11_CACHED_CONTROL_RUN_V1", cases=cases_from_source(blobs[REVIEW]),
            limits=LIMITS, variants=list(VARIANTS), catalog_path=CATALOG, catalog_sha256=CATALOG_SHA,
            selected_reference_sha256=SELECTED_SHA, reference_line_count=71,
            source_sha256={name: digest(blob) for name, blob in blobs.items()},
            runtime_metadata=runtime_metadata(support), inference_execution="NOT_PERFORMED",
            historical_comparability="UNESTABLISHED", production_use_allowed=False)
        RUN.mkdir(mode=0o700, exist_ok=False)
        for name, blob in blobs.items():
            target = RUN / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            publish_bytes(target, blob)
        publish_json(RUN / "FROZEN.json", bundle)
    print(json.dumps(dict(status="FROZEN_NOT_EXECUTED", bundle_sha256=digest(encoded(bundle)), attempt=str(RUN))), flush=True)


def verify_bundle(support):
    bundle = decode(read_fixed(RUN, "FROZEN.json", LIMITS["control_file_bytes"]))
    if (bundle.get("schema") != "GOLD11_CACHED_CONTROL_RUN_V1" or bundle.get("limits") != LIMITS
            or bundle.get("variants") != list(VARIANTS) or bundle.get("catalog_path") != CATALOG
            or bundle.get("catalog_sha256") != CATALOG_SHA or bundle.get("selected_reference_sha256") != SELECTED_SHA
            or bundle.get("reference_line_count") != 71 or set(bundle.get("source_sha256", {})) != set(SOURCES)):
        raise ValueError("FROZEN_PROTOCOL_OR_SOURCE_DENOMINATOR_CHANGED")
    for name, sha in bundle["source_sha256"].items():
        read_fixed(ROOT, name, LIMITS["source_file_bytes"], sha)
        read_fixed(RUN / "source", name, LIMITS["source_file_bytes"], sha)
    if bundle.get("cases") != cases_from_source(read_fixed(RUN / "source", REVIEW, LIMITS["source_file_bytes"], REVIEW_SHA)):
        raise ValueError("FROZEN_CASE_BINDINGS_CHANGED")
    if bundle.get("runtime_metadata") != runtime_metadata(support):
        raise ValueError("FROZEN_RUNTIME_METADATA_CHANGED")
    return bundle


def pre_run_audit(bundle):
    audit = decode(read_fixed(RUN, "PRE_RUN_AUDIT.json", LIMITS["control_file_bytes"]))
    if (audit.get("decision") != "GREEN_FOR_FROZEN_CACHED_CONTROL_RUN"
            or audit.get("bundle_sha256") != digest(encoded(bundle))
            or type(audit.get("reviewer")) is not str or not audit["reviewer"].strip()
            or audit.get("blocking_findings") != []):
        raise ValueError("EXACT_INDEPENDENT_PRE_RUN_AUDIT_REQUIRED")
    return audit


def unavailable_cases(cases, reason):
    return {variant: {case["case_id"]: dict(status="UNAVAILABLE_NOT_SCORED", reason=reason,
        family_id=case["family_id"], scan_sha256=case["scan_sha256"], precision=None, recall=None,
        f1=None, true_positive=None, false_positive=None, false_negative=None,
        observed_bend_count=None) for case in cases} for variant in VARIANTS}


def authorize_worker(support):
    request = decode(sys.stdin.buffer.read(4097))
    if (type(request) is not dict or set(request) != {"token", "parent_pid"}
            or not support.is_hex(request.get("token"), 64) or type(request.get("parent_pid")) is not int):
        raise ValueError("ONE_USE_PARENT_DISPATCH_REQUIRED")
    bundle = verify_bundle(support)
    audit = pre_run_audit(bundle)
    expected = decode(read_fixed(RUN, "DISPATCH.json", 16384))
    started = decode(read_fixed(RUN, "STARTED.json", 16384))
    cache = RUN / "work"
    if (request["parent_pid"] != os.getppid() or expected.get("parent_pid") != os.getppid()
            or os.getpgrp() != os.getpid() or expected.get("bundle_sha256") != digest(encoded(bundle))
            or not secrets.compare_digest(expected.get("token_sha256", ""), digest(request["token"].encode()))
            or expected.get("cache_identity") != support.cache_identity(cache)
            or started.get("status") != "STARTED_NO_RETRY" or started.get("limits") != LIMITS
            or started.get("bundle_sha256") != digest(encoded(bundle))
            or started.get("audit_sha256") != digest(encoded(audit))
            or Path.cwd() != cache or os.environ.get("TMPDIR") != str(cache)
            or Path(__file__).resolve() != (RUN / "source" / RUNNER).resolve()
            or any(os.path.lexists(RUN / name) for name in (
                "COMPLETED.json", "FINAL_GATE.json", "FINALIZATION_FAILED.json", "UNAVAILABLE.json"))):
        raise ValueError("WORKER_PARENT_FROZEN_SOURCE_OR_CONTROL_BINDING")
    publish_json(RUN / "WORKER_CLAIMED.json", dict(worker_pid=os.getpid(), parent_pid=os.getppid(),
        dispatch_sha256=digest(encoded(expected)), no_retry=True))
    return bundle, cache


def install_input_guard(support, cache, cases):
    base_guard = support.worker_audit_hook(RUN / "source", cache, "scripts")
    exact = {str(ROOT / CATALOG), *(str(ROOT / case[key]) for case in cases for key in ("a0_path", "cad_path"))}
    # read_fixed uses dir_fd traversal; os.open emits only its relative component
    # to Python's audit hook. Keep input opening in a tiny checked seam below,
    # never grant a relative-open exemption to imported scoring code.
    state = {"permit": None}
    def guard(event, args):
        if event == "open" and state["permit"] is not None:
            permit, state["permit"] = state["permit"], None
            # Consume BEFORE returning: a nested audit-hook open cannot borrow
            # the grant. CPython may add close-on-exec to the supplied flags.
            value, mode, flags = args
            if (not isinstance(value, (str, bytes)) or mode is not None
                    or type(flags) is not int
                    or (os.fsdecode(value), flags & ~os.O_CLOEXEC) != permit):
                raise PermissionError("FIXED_INPUT_COMPONENT_GRANT_MISMATCH")
            return
        base_guard(event, args)
    sys.addaudithook(guard)
    def open_component(path, flags, **kwargs):
        if state["permit"] is not None:
            raise PermissionError("NESTED_FIXED_INPUT_GRANT_DENIED")
        state["permit"] = (os.fsdecode(path), flags & ~os.O_CLOEXEC)
        try:
            fd = os.open(path, flags, **kwargs)
            if state["permit"] is not None:
                os.close(fd)
                raise PermissionError("FIXED_INPUT_AUDIT_GRANT_NOT_CONSUMED")
            return fd
        finally:
            state["permit"] = None
    def input_read(relative, cap, sha):
        if type(relative) is not str or str(ROOT / relative) not in exact:
            raise PermissionError("UNDECLARED_INPUT_PATH")
        return read_fixed(ROOT, relative, cap, sha, opener=open_component)
    return input_read, base_guard


def selected_references(blob, cases):
    payload = decode(blob)
    if type(payload) is not dict or type(payload.get("cases")) is not list or len(payload["cases"]) > 4096:
        raise ValueError("CATALOG_CONTAINER_SCHEMA_OR_BUDGET")
    by_sha = {case["scan_sha256"]: case for case in cases}
    selected = {}
    for row in payload["cases"]:
        # No field from an unselected case other than its selection SHA is used.
        if type(row) is not dict:
            raise ValueError("CATALOG_CASE_CONTAINER_SCHEMA")
        case = by_sha.get(row.get("scan_sha256"))
        if case is None:
            continue
        key = case["case_id"]
        if (key in selected or row.get("case_id") != case["reference_id"]
                or row.get("scan_path") != case["reference_scan_path"]
                or str(row.get("part_id")) != case["family_id"]
                or type(row.get("bends")) is not list or not 1 <= len(row["bends"]) <= 71):
            raise ValueError("SELECTED_REFERENCE_IDENTITY_OR_SCHEMA")
        selected[key] = row
    if (set(selected) != {case["case_id"] for case in cases}
            or sum(len(row["bends"]) for row in selected.values()) != 71
            or digest(encoded(selected)) != SELECTED_SHA):
        raise ValueError("EXACT_SEPTEMBER9_REFERENCE_SNAPSHOT_MISMATCH")
    return selected


def scientific_artifact(cache, relative, blob, budget):
    if budget["bytes"] + len(blob) > LIMITS["scientific_artifact_bytes"]:
        raise ValueError("AGGREGATE_SCIENTIFIC_ARTIFACT_BYTE_BUDGET")
    # Reserve the full attempted bytes BEFORE publication. A failed write/link/
    # fsync may leave a retained partial: charge conservatively, never retry it.
    budget["bytes"] += len(blob)
    path = cache / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(path, blob)


def worker():
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["worker_cpu_seconds"],)*2)
    support = support_module(frozen=True)
    bundle, cache = authorize_worker(support)
    cases, budget = bundle["cases"], {"bytes": 0}
    input_read, locked_guard = install_input_guard(support, cache, cases)
    checks = []
    try:
        for case in cases:
            for role in ("a0", "cad"):
                check = dict(case_id=case["case_id"], role=role, path=case[role+"_path"],
                    expected_sha256=case[role+"_sha256"], scan_sha256=case["scan_sha256"])
                try:
                    blob = input_read(case[role+"_path"], LIMITS["prediction_file_bytes"], case[role+"_sha256"])
                    destination = f"inputs/{role}/{case['case_id']}.json"
                    scientific_artifact(cache, destination, blob, budget)
                    check.update(status="BYTE_VERIFIED_AND_SNAPSHOTTED", bytes=len(blob), snapshot=destination)
                    del blob
                except Exception as error:
                    check.update(status="INPUT_REFUSED", error_type=type(error).__name__, reason=str(error)[:160])
                checks.append(check)
                scientific_artifact(cache, f"checks/{case['case_id']}_{role}.json", encoded(check), budget)
        if len(checks) != 22 or any(check["status"] != "BYTE_VERIFIED_AND_SNAPSHOTTED" for check in checks):
            raise ValueError("CACHED_INPUT_PREFLIGHT_INCOMPLETE_NO_SCORING_OR_CATALOG_READ")
        # Hash the one fixed catalog before JSON projection; never follow paths
        # declared inside it. No prediction geometry was interpreted above.
        catalog_blob = input_read(CATALOG, LIMITS["catalog_bytes"], CATALOG_SHA)
        selected = selected_references(catalog_blob, cases)
        del catalog_blob
        selected_blob = encoded(selected)
        if len(selected_blob) > LIMITS["selected_reference_bytes"]:
            raise ValueError("SELECTED_REFERENCE_BYTE_BUDGET")
        scientific_artifact(cache, "SELECTED_REFERENCES.json", selected_blob, budget)
        ledger = dict(status="ALL22_CACHED_BYTES_AND_EXACT71_REFERENCES_VERIFIED", checks=checks,
            catalog_path=CATALOG, catalog_sha256=CATALOG_SHA, selected_reference_sha256=digest(selected_blob),
            case_count=11, family_count=6, reference_line_count=71,
            raw_scan_or_cad_opened=False, historical_reference_identity="UNESTABLISHED")
        scientific_artifact(cache, "INPUT_PREFLIGHT.json", encoded(ledger), budget)
        # Permanently remove original-input access before importing any scorer.
        sys.addaudithook(locked_guard)
        sys.path.insert(0, support.runtime_roots()[0])
        support.install_namespace("scripts", RUN / "source/scripts")
        a0, cad = {}, {}
        for case in cases:
            for role, mapping in (("a0", a0), ("cad", cad)):
                # Absolute cache paths work with the now-locked audit guard;
                # descriptor consistency/hash checks remain the reviewed helper.
                blob = support.bounded_read(cache / f"inputs/{role}/{case['case_id']}.json",
                    LIMITS["prediction_file_bytes"], case[role+"_sha256"])
                mapping[case["case_id"]] = decode(blob)
        from scripts.gold11_cached_fusion_scoring_v1 import score_gold11_cached_fusion_v1
        report = score_gold11_cached_fusion_v1(
            reference_records_by_case={case: row["bends"] for case, row in selected.items()},
            a0_results_by_case=a0, cad_results_by_case=cad,
            case_family_by_id={case["case_id"]: case["family_id"] for case in cases})
        comparisons = {}
        for case in cases:
            observed = report["reconstruction"]["cases"][case["case_id"]]["cached_line_counts"]
            comparisons[case["case_id"]] = {variant: dict(historical_reported=expected,
                reconstructed=observed[variant], equal=observed[variant] == expected)
                for variant, expected in case["historical_reported_counts"].items()}
        report["runner_provenance"] = dict(bundle_sha256=digest(encoded(bundle)),
            input_preflight_sha256=digest(encoded(ledger)), selected_reference_sha256=SELECTED_SHA,
            case_scan_sha256={case["case_id"]: case["scan_sha256"] for case in cases},
            historical_count_comparison=comparisons, count_comparison_policy="REPORT_MISMATCH_NEVER_REPAIR_OR_SELECT",
            historical_score_beaten_claim_allowed=False, raw_scan_or_cad_opened=False)
        blob = encoded(report)
        if len(blob) > LIMITS["report_bytes"]:
            raise ValueError("FULL_FOUR_VARIANT_REPORT_BYTE_BUDGET")
        scientific_artifact(cache, "SCORER_REPORT.json", blob, budget)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        publish_json(cache / "WORKER_COMPLETED.json", dict(status="COMPUTED_CACHED_CONTROL_NOT_INFERENCE",
            report_sha256=digest(blob), scientific_artifact_bytes=budget["bytes"],
            worker_cpu_seconds=usage.ru_utime+usage.ru_stime, worker_peak_rss_bytes=usage.ru_maxrss,
            case_count=11, variant_count=4, observed_bend_count=None, production_use_allowed=False))
        print(json.dumps(dict(status="COMPUTED_CACHED_CONTROL_NOT_INFERENCE", report_sha256=digest(blob))), flush=True)
    except Exception as error:
        # No invented empty predictions/FNs on a GLOBAL execution failure.
        # The scorer itself already retains per-case schema/refusal FNs.
        publish_json(cache / "WORKER_FAILED.json", dict(status="TERMINAL_NOT_SCORED", error_type=type(error).__name__,
            reason=str(error)[:160], completed_input_checks=checks,
            scientific_artifact_bytes_charged=budget["bytes"],
            variants=unavailable_cases(cases, "GLOBAL_CACHED_CONTROL_FAILURE"), no_retry=True))
        raise


def aggregate_cpu():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def verify_retained_artifacts(bundle, worker_receipt):
    """Recheck OWNED snapshots, never reopen original cached/catalog sources."""
    if (type(worker_receipt) is not dict
            or worker_receipt.get("status") != "COMPUTED_CACHED_CONTROL_NOT_INFERENCE"
            or type(worker_receipt.get("report_sha256")) is not str
            or len(worker_receipt["report_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in worker_receipt["report_sha256"])
            or type(worker_receipt.get("scientific_artifact_bytes")) is not int
            or not 0 <= worker_receipt["scientific_artifact_bytes"] <= LIMITS["scientific_artifact_bytes"]
            or type(worker_receipt.get("case_count")) is not int or worker_receipt["case_count"] != 11
            or type(worker_receipt.get("variant_count")) is not int or worker_receipt["variant_count"] != 4
            or type(worker_receipt.get("worker_cpu_seconds")) not in (int, float)
            or not 0 <= worker_receipt["worker_cpu_seconds"] <= LIMITS["worker_cpu_seconds"]
            or type(worker_receipt.get("worker_peak_rss_bytes")) is not int
            or not 0 < worker_receipt["worker_peak_rss_bytes"] <= LIMITS["global_tree_rss_bytes"]
            or worker_receipt.get("observed_bend_count", "MISSING") is not None
            or worker_receipt.get("production_use_allowed") is not False):
        raise ValueError("WORKER_COMPLETION_CONTRACT")
    cache, artifact_bytes, expected_checks = RUN / "work", 0, []
    for case in bundle["cases"]:
        for role in ("a0", "cad"):
            snapshot = f"inputs/{role}/{case['case_id']}.json"
            blob = read_fixed(cache, snapshot,
                LIMITS["prediction_file_bytes"], case[role+"_sha256"])
            check = dict(case_id=case["case_id"], role=role, path=case[role+"_path"],
                expected_sha256=case[role+"_sha256"], scan_sha256=case["scan_sha256"],
                status="BYTE_VERIFIED_AND_SNAPSHOTTED", bytes=len(blob), snapshot=snapshot)
            artifact_bytes += len(blob)
            artifact_bytes += len(read_fixed(cache, f"checks/{case['case_id']}_{role}.json",
                LIMITS["control_file_bytes"], digest(encoded(check))))
            expected_checks.append(check)
    artifact_bytes += len(read_fixed(cache, "SELECTED_REFERENCES.json", LIMITS["selected_reference_bytes"], SELECTED_SHA))
    ledger_blob = read_fixed(cache, "INPUT_PREFLIGHT.json", LIMITS["control_file_bytes"])
    ledger = decode(ledger_blob)
    expected_ledger = dict(status="ALL22_CACHED_BYTES_AND_EXACT71_REFERENCES_VERIFIED", checks=expected_checks,
        catalog_path=CATALOG, catalog_sha256=CATALOG_SHA, selected_reference_sha256=SELECTED_SHA,
        case_count=11, family_count=6, reference_line_count=71,
        raw_scan_or_cad_opened=False, historical_reference_identity="UNESTABLISHED")
    if encoded(ledger) != encoded(expected_ledger):
        raise ValueError("RETAINED_INPUT_PREFLIGHT_BINDING")
    blob = read_fixed(cache, "SCORER_REPORT.json", LIMITS["report_bytes"], worker_receipt["report_sha256"])
    artifact_bytes += len(ledger_blob) + len(blob)
    report = decode(blob)
    if (type(report) is not dict or report.get("schema") != "GOLD11_CACHED_FUSION_GEOMETRY_DIAGNOSTIC_V1"
            or report.get("status") != "COMPUTED_DEVELOPMENT_DIAGNOSTIC"
            or type(report.get("variants")) is not dict or set(report["variants"]) != set(VARIANTS)
            or any(type(report.get(key)) is not int or report[key] != count
                   for key, count in (("case_count", 11), ("family_count", 6), ("reference_line_count", 71)))
            or any(type(lane) is not dict for lane in report["variants"].values())
            or any(set(lane.get("case_dispositions", {})) != {case["case_id"] for case in bundle["cases"]}
                   for lane in report["variants"].values())
            or type(report.get("runner_provenance")) is not dict
            or report.get("runner_provenance", {}).get("bundle_sha256") != digest(encoded(bundle))
            or report["runner_provenance"].get("input_preflight_sha256") != digest(ledger_blob)
            or report["runner_provenance"].get("selected_reference_sha256") != SELECTED_SHA
            or report["runner_provenance"].get("case_scan_sha256") != {
                case["case_id"]: case["scan_sha256"] for case in bundle["cases"]}
            or report.get("inference_execution") != "NOT_PERFORMED"
            or report.get("variant_selection") != "NONE_ALL_FOUR_REPORTED"
            or report.get("observed_bend_count", "MISSING") is not None
            or report.get("production_use_allowed") is not False
            or report.get("historical_score_comparability") != "UNESTABLISHED"
            or artifact_bytes != worker_receipt["scientific_artifact_bytes"]):
        raise ValueError("FULL_BOUND_FOUR_VARIANT_ELEVEN_CASE_REPORT_REQUIRED")
    families = {case["case_id"]: case["family_id"] for case in bundle["cases"]}
    reconstruction = report.get("reconstruction")
    if (type(reconstruction) is not dict
            or reconstruction.get("schema") != "GOLD11_RETAINED_FUSION_RECONSTRUCTION_V1"
            or type(reconstruction.get("cases")) is not dict or set(reconstruction["cases"]) != set(families)
            or any(type(row) is not dict or type(row.get("variants")) is not dict
                   or set(row["variants"]) != set(VARIANTS) for row in reconstruction["cases"].values())):
        raise ValueError("COMPLETE_FOUR_VARIANT_RECONSTRUCTION_REQUIRED")
    for lane in report["variants"].values():
        if (set(lane.get("families", {})) != set(families.values())
                or lane.get("geometry", {}).get("case_count") != 11
                or lane["geometry"].get("reference_line_count") != 71
                or any(type(row) is not dict or row.get("family_id") != families[case]
                       for case, row in lane["case_dispositions"].items())):
            raise ValueError("FULL_CASE_FAMILY_REFERENCE_LANE_BINDING")
    return dict(report_sha256=digest(blob), scientific_artifact_bytes=artifact_bytes,
                selected_reference_sha256=SELECTED_SHA, input_preflight_sha256=digest(ledger_blob))


def audit_final_sources(bundle, audit, support, evidence):
    """All paths, including failed workers: record partial drift evidence."""
    evidence.update(current={}, frozen={}, errors={}, complete=False, unchanged=False)
    for label, base in (("current", ROOT), ("frozen", RUN / "source")):
        for name in SOURCES:
            try:
                evidence[label][name] = digest(read_fixed(base, name, LIMITS["source_file_bytes"]))
            except TimeoutError:
                raise
            except Exception as error:
                evidence["errors"][label+":"+name] = type(error).__name__
    evidence["bundle_unchanged"] = digest(read_fixed(RUN, "FROZEN.json", LIMITS["control_file_bytes"])) == digest(encoded(bundle))
    evidence["audit_unchanged"] = digest(read_fixed(RUN, "PRE_RUN_AUDIT.json", LIMITS["control_file_bytes"])) == digest(encoded(audit))
    evidence["runtime_metadata_unchanged"] = runtime_metadata(support) == bundle["runtime_metadata"]
    evidence["complete"] = True
    evidence["unchanged"] = (not evidence["errors"] and evidence["current"] == evidence["frozen"] == bundle["source_sha256"]
        and evidence["bundle_unchanged"] and evidence["audit_unchanged"] and evidence["runtime_metadata_unchanged"])


def final_budget_evidence(started, cpu_start):
    wall, cpu = time.monotonic()-started, aggregate_cpu()-cpu_start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return dict(parent_wall_seconds=wall, aggregate_cpu_seconds=cpu, parent_peak_rss_bytes=rss,
        reaped_children_peak_rss_bytes=children_rss,
        wall_budget_met=wall <= LIMITS["total_wall_seconds"],
        cpu_budget_met=cpu <= LIMITS["aggregate_cpu_seconds"],
        parent_rss_budget_met=rss <= LIMITS["global_tree_rss_bytes"],
        reaped_child_rss_budget_met=children_rss <= LIMITS["global_tree_rss_bytes"])


def budgets_met(evidence):
    return all(evidence[key] is True for key in (
        "wall_budget_met", "cpu_budget_met", "parent_rss_budget_met", "reaped_child_rss_budget_met"))


def retained_scientific_usage(cache):
    """Owned, fixed output directories only; count partial files conservatively."""
    files = {}
    for relative in ("", "inputs", "inputs/a0", "inputs/cad", "checks"):
        try:
            entries = os.scandir(cache / relative)
        except FileNotFoundError:
            continue
        with entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    if str(Path(relative) / entry.name) not in ("inputs", "inputs/a0", "inputs/cad", "checks"):
                        raise ValueError("UNDECLARED_RETAINED_OUTPUT_DIRECTORY")
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("NONREGULAR_RETAINED_OUTPUT")
                # Logs and these bounded worker control records have separate
                # caps. Everything else, including scientific partials, counts.
                if relative == "" and any(entry.name == name or entry.name.startswith("."+name+".partial-")
                        for name in ("LIVE_OUTPUT.txt", "WORKER_COMPLETED.json", "WORKER_FAILED.json")):
                    continue
                files[str(Path(relative) / entry.name)] = info.st_size
                if len(files) > 128:
                    raise ValueError("RETAINED_ARTIFACT_ENTRY_BUDGET")
    return dict(files=files, logical_bytes_including_partials=sum(files.values()),
                partial_hardlinks_counted_conservatively_twice=True)


def execute():
    started, cpu_start = time.monotonic(), aggregate_cpu()
    support = support_module()
    started_attempted = False
    try:
        with support.deadline(20):
            bundle = verify_bundle(support)
            audit = pre_run_audit(bundle)
            for name in ("STARTED.json", "COMPLETED.json", "FINAL_GATE.json",
                         "FINALIZATION_FAILED.json", "UNAVAILABLE.json",
                         "DISPATCH.json", "WORKER_CLAIMED.json"):
                try:
                    (RUN / name).lstat()
                except FileNotFoundError:
                    continue
                raise ValueError("STARTED_OR_TERMINAL_ATTEMPT_NO_RETRY")
            started_attempted = True
            publish_json(RUN / "STARTED.json", dict(status="STARTED_NO_RETRY", limits=LIMITS,
                bundle_sha256=digest(encoded(bundle)), audit_sha256=digest(encoded(audit))))
    except BaseException as error:
        # No data worker has run. An exclusive-publication loser must not add a
        # veto to another owner's existing STARTED; other publication failures
        # may follow a durable link and receive an additive terminal veto.
        if not started_attempted or isinstance(error, FileExistsError):
            raise
        try:
            with support.deadline(1):
                publish_json(RUN / "FINALIZATION_FAILED.json", dict(status="TERMINAL_VETO_NO_RELEASE",
                    failures=[dict(stage="STARTED_PUBLICATION", error_type=type(error).__name__)],
                    observations=final_budget_evidence(started, cpu_start),
                    variants=unavailable_cases(bundle["cases"], "STARTED_PUBLICATION_FAILURE"),
                    worker_dispatched=False, no_resume=True))
        except BaseException:
            pass  # Nonzero parent exit and missing final gate never release.
        return 1
    receipt, worker_receipt, artifacts, failures, source_audit = {}, {}, {}, [], {}
    usage, output_sha = {}, None
    cache = RUN / "work"
    def record_failure(stage, error):
        failures.append(dict(stage=stage, error_type=type(error).__name__,
            reason=str(error)[:160] if isinstance(error, (ValueError, TimeoutError, PermissionError)) else type(error).__name__))
    def remaining(reserve=0):
        return max(.001, LIMITS["total_wall_seconds"]-(time.monotonic()-started)-reserve)
    try:
        cache.mkdir(mode=0o700, exist_ok=False)
        token = secrets.token_hex(32)
        publish_json(RUN / "DISPATCH.json", dict(parent_pid=os.getpid(), token_sha256=digest(token.encode()),
            bundle_sha256=digest(encoded(bundle)), cache_identity=support.cache_identity(cache)))
        command = [str(ROOT / ".venv/bin/python"), "-I", "-S", "-B", "-u", str(RUN / "source" / RUNNER), "--worker"]
        with support.deadline(remaining(LIMITS["terminal_reserve_seconds"])):
            receipt = support.supervise_process(command, dict(token=token, parent_pid=os.getpid()),
                cache=cache, log_path=cache / "LIVE_OUTPUT.txt", wall_seconds=LIMITS["worker_wall_seconds"],
                rss_limit=LIMITS["global_tree_rss_bytes"], log_limit=LIMITS["log_bytes"], new_session=True)
        if receipt.get("exit_code") != 0 or receipt.get("budget_stop"):
            raise ValueError("CACHED_WORKER_FAILED_OR_BUDGET_STOP")
        with support.deadline(remaining(2)):
            worker_receipt = decode(read_fixed(cache, "WORKER_COMPLETED.json", LIMITS["control_file_bytes"]))
            artifacts = verify_retained_artifacts(bundle, worker_receipt)
    except BaseException as error:
        record_failure("WORKER_OR_RETAINED_ARTIFACT_VALIDATION", error)
    # Always attempt bounded source/snapshot drift evidence, even after failure.
    try:
        with support.deadline(remaining(2)):
            audit_final_sources(bundle, audit, support, source_audit)
            if not source_audit["unchanged"]:
                raise ValueError("FINAL_SOURCE_OR_SNAPSHOT_DRIFT")
    except BaseException as error:
        record_failure("FINAL_SOURCE_AUDIT", error)
    try:
        with support.deadline(remaining(2)):
            usage = retained_scientific_usage(cache)
            if usage["logical_bytes_including_partials"] > LIMITS["scientific_artifact_bytes"]:
                raise ValueError("RETAINED_SCIENTIFIC_ARTIFACT_BYTE_LIMIT")
            if artifacts and usage["logical_bytes_including_partials"] != artifacts["scientific_artifact_bytes"]:
                raise ValueError("RETAINED_PARTIAL_OR_UNACCOUNTED_ARTIFACTS")
    except BaseException as error:
        record_failure("RETAINED_BYTE_ACCOUNTING", error)
    try:
        with support.deadline(remaining(1)):
            output = read_fixed(cache, "LIVE_OUTPUT.txt", LIMITS["log_bytes"], allow_empty=True)
            if len(output) != receipt.get("forwarded_log_bytes"):
                raise ValueError("RETAINED_LOG_RECEIPT_BYTE_MISMATCH")
    except BaseException as error:
        record_failure("LOG_READ_OR_BINDING", error)
        output = encoded(dict(status="LOG_UNAVAILABLE_NOT_EMPTY_SUCCESS", failure=failures[-1]))+b"\n"
    try:
        with support.deadline(remaining()):
            publish_bytes(RUN / "OUTPUT.txt", output)
        output_sha = digest(output)
    except BaseException as error:
        record_failure("OUTPUT_PUBLICATION", error)
    gate = final_budget_evidence(started, cpu_start)
    if not budgets_met(gate):
        record_failure("PRE_TERMINAL_BUDGET", ValueError("WALL_CPU_OR_PARENT_RSS_LIMIT"))
    status = "TERMINAL_NOT_SCORED" if failures else "COMPUTED_PENDING_INDEPENDENT_POST_AUDIT"
    try:
        with support.deadline(remaining()):
            if failures:
                publish_json(RUN / "UNAVAILABLE.json", dict(status=status, failures=failures,
                    variants=unavailable_cases(bundle["cases"], "GLOBAL_CACHED_CONTROL_FAILURE"),
                    score_computed_claim_allowed=False, retained_partial_artifacts_are_not_a_completed_report=True))
            completed = dict(status=status, failures=failures, supervisor_receipt=receipt,
                worker_receipt=worker_receipt, artifacts=artifacts, output_sha256=output_sha,
                final_source_audit=source_audit, retained_scientific_usage=usage,
                budget_observation_before_terminal_publication=gate, no_resume=True,
                release_requires="OBSERVED_PARENT_EXIT_ZERO_AND_FINAL_GATE_AND_NO_VETO_AND_EXACT_INDEPENDENT_POST_AUDIT",
                inference_execution="NOT_PERFORMED", historical_comparability="UNESTABLISHED",
                observed_bend_count=None, production_use_allowed=False)
            publish_json(RUN / "COMPLETED.json", completed)
            final_gate = final_budget_evidence(started, cpu_start)
            if failures or not budgets_met(final_gate):
                raise ValueError("FAILURE_OR_POST_COMPLETED_RESOURCE_LIMIT")
            publish_json(RUN / "FINAL_GATE.json", dict(status="PARENT_GATE_PENDING_INDEPENDENT_POST_AUDIT",
                completed_sha256=digest(encoded(completed)), observations=final_gate,
                source_audit_sha256=digest(encoded(source_audit)), output_sha256=output_sha,
                report_sha256=artifacts["report_sha256"],
                measurement_boundary="MEASURED_BEFORE_FINAL_GATE_PUBLICATION_AND_RECHECKED_BEFORE_PARENT_EXIT"))
        # Cannot place after-own-fsync timing inside that immutable file. A late
        # exceedance is an additive veto; a crash/missing observed parent exit is
        # NEVER release. Cleanup/publication cannot guarantee persistence on OS,
        # storage, power or uncatchable-process failure.
        if not budgets_met(final_budget_evidence(started, cpu_start)):
            raise ValueError("POST_FINAL_GATE_RESOURCE_LIMIT")
        with support.deadline(remaining()):
            print(json.dumps(dict(status=status, attempt=str(RUN), release_allowed=False)), flush=True)
        if not budgets_met(final_budget_evidence(started, cpu_start)):
            raise ValueError("POST_STDOUT_RESOURCE_LIMIT")
        return 0
    except BaseException as error:
        record_failure("TERMINAL_PUBLICATION_OR_FINAL_GATE", error)
        try:
            # Bounded emergency retention may occur after the 90s cutoff. It
            # cannot admit the run and is explicitly excluded from any success.
            with support.deadline(1):
                publish_json(RUN / "FINALIZATION_FAILED.json", dict(status="TERMINAL_VETO_NO_RELEASE",
                    failures=failures, observations=final_budget_evidence(started, cpu_start),
                    variants=unavailable_cases(bundle["cases"], "TERMINAL_VETO")))
        except BaseException:
            pass  # Partial/absent final gate plus nonzero exit stays a failure.
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.worker:
        worker()
    else:
        raise SystemExit(execute())


if __name__ == "__main__":
    main()
