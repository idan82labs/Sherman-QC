#!/usr/bin/env python3
"""G5 no-regression gate for the overnight production-line build (plan §10, amendment 11).

The gate compares the working tree against a baseline recorded *before* any production-line
agent edited the repository. It has three parts:

1. **Manifest** - sha256 of every file under ``backend/``, ``domains/bend/`` and ``scripts/``
   (``__pycache__`` directories, ``*.pyc`` files and ``scripts/production/`` are excluded).
   A symlink (file or directory) is recorded as ``symlink:<target>`` and is not followed, so
   adding or retargeting one is a change. Every added, removed or modified path must match an allowlisted glob; anything
   else is an UNEXPECTED change and fails the gate.
2. **Frontend** - ``tsc --noEmit``, ``npm run lint`` and every
   ``frontend/react/scripts/check-vendor-*.mjs`` script. TypeScript and ESLint diagnostics are
   reduced to line-independent keys (file, code/rule, message) with counts, so a diagnostic that
   only moved does not count as new, while a fresh one (or one more copy of an old one) does.
3. **Backend** - the vendor-setup focused pytest files (synthetic, isolated, no server import),
   run with the plan's PYT form plus ``--rootdir <repo>`` so junit ids are file-qualified
   (``backend.tests.test_x::test_y``). A test is a NEW failure when it is failing now and was not
   failing in the baseline (including tests that did not exist in the baseline).

Honest semantics: a check that could not run (missing tool, browser script with no demo origin)
is reported as NOT_RUN and never becomes PASS. When a check ran in the baseline (PASS or FAIL)
but cannot run now, it is INCOMPLETE and the gate cannot PASS (exit 3). Only a NOT_RUN that
matches a NOT_RUN baseline (the browser scripts) passes through. A backend test that passed in
the baseline and is now skipped or missing is a weakening and fails the gate. Pre-existing
baseline failures are reported as ``preexisting`` and do not fail the gate; only NEW failures do.

Allowlist globs are segment-aware: ``*``, ``?`` and ``[...]`` never cross ``/``; ``**`` matches
any number of path segments (including none); a trailing ``/`` allows the whole subtree. The
file ``docs/production/runlog/g5_baseline/allowlist.txt`` is loaded by default when present.

Usage::

    .venv/bin/python scripts/production/g5_check.py                 # full gate, exit 1 on NEW failures
    .venv/bin/python scripts/production/g5_check.py --allow 'scripts/foo.py' --allowlist-file list.txt
    .venv/bin/python scripts/production/g5_check.py --only manifest --json report.json
    .venv/bin/python scripts/production/g5_check.py --self-test     # mutate-and-check unit tests
    .venv/bin/python scripts/production/g5_check.py --write-baseline --force   # re-record (owner only)

Exit codes: 0 = no new failures, 1 = new failures, unexpected manifest changes or a baseline
that does not match its pinned sha256, 2 = usage or baseline error (including a refused
re-baseline), 3 = INCOMPLETE (no new failures, but a check that ran in the baseline did not run).

Baseline integrity: the sha256 of ``manifest.json`` and ``results.json`` is pinned below
(``PINNED_BASELINE_SHA256``) and printed on every run; a mismatch fails the gate. Re-recording
needs ``--write-baseline --force`` *and* ``G5_OWNER_REBASELINE=1`` in the environment (owner
only); the owner then updates the pinned hashes. Allowlisted changed paths are printed by name
so it is visible what was licensed. Browser vendor scripts that are NOT_RUN in both baseline and
now are counted in the headline (e.g. ``PASS (5 NOT_RUN, matching baseline)``): they are not
covered by the gate.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "docs/production/runlog/g5_baseline"
FRONTEND_DIR = REPO_ROOT / "frontend/react"
PYTHON = REPO_ROOT / ".venv/bin/python"

MANIFEST_ROOTS: tuple[str, ...] = ("backend", "domains/bend", "scripts")
#: Directory prefixes (repo-relative, posix) excluded from the manifest.
MANIFEST_EXCLUDED_PREFIXES: tuple[str, ...] = ("scripts/production",)
MANIFEST_EXCLUDED_DIR_NAMES: frozenset[str] = frozenset({"__pycache__"})
MANIFEST_EXCLUDED_SUFFIXES: tuple[str, ...] = (".pyc",)

#: Vendor-setup focused backend tests. All are synthetic, import no server module and read no
#: real CAD/scan data (inspected 2026-10-17 before inclusion).
BACKEND_TESTS: tuple[str, ...] = (
    "backend/tests/test_vendor_setup_catalog.py",
    "backend/tests/test_vendor_setup_service.py",
    "backend/tests/test_vendor_flange_preview.py",
    "tests/test_vendor_machine_moves_contract.py",
    "tests/test_vendor_unfold_api_contract.py",
    "tests/test_vendor_general_measurements_contract.py",
)

#: check-vendor scripts that drive a browser against a local demo server (VENDOR_DEMO_ORIGIN)
#: backed by CAD-derived fixtures. They cannot run without that environment and are NOT_RUN
#: unless ``--include-browser`` is passed.
BROWSER_VENDOR_SCRIPTS: frozenset[str] = frozenset({
    "check-vendor-corpus-demo.mjs",
    "check-vendor-general-measurements.mjs",
    "check-vendor-touch-gestures.mjs",
    "check-vendor-unfold-demo.mjs",
    "check-vendor-unfold-production-preview.mjs",
})

PASS, FAIL, NOT_RUN, INCOMPLETE = "PASS", "FAIL", "NOT_RUN", "INCOMPLETE"
DEFAULT_ALLOWLIST = BASELINE_DIR / "allowlist.txt"
#: sha256 of the wave-0 baseline files (recorded 2026-10-17). Change only on an owner re-baseline.
PINNED_BASELINE_SHA256: dict[str, str] = {
    "manifest.json": "7792b56135b6f204466a3ff1322556db5897ee5bd6685201f3fd932d405e9685",
    "results.json": "b5720c64b6f072729c8dcf51b598f8942954172a7c32a1fc218a8c183354ccf8",
}
OWNER_REBASELINE_ENV = "G5_OWNER_REBASELINE"
SYMLINK_PREFIX = "symlink:"
#: junit id recorded when pytest produced test ids without a file-qualified classname.
UNQUALIFIED_JUNIT_ID = "<junit>::unqualified test ids (classname empty; check --rootdir)"
CHECK_TIMEOUT_S = 900


# --------------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_manifest(root: Path = REPO_ROOT, roots: Sequence[str] = MANIFEST_ROOTS) -> dict[str, str]:
    """Return ``{repo-relative posix path: sha256}`` for every file under ``roots``.

    Symlinks (to files or directories) are recorded as ``symlink:<link target>`` and never
    followed, so a new or retargeted link shows up as a change.
    """
    manifest: dict[str, str] = {}
    for top in roots:
        base = root / top
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            rel_dir = Path(dirpath).relative_to(root).as_posix()
            kept: list[str] = []
            for d in sorted(dirnames):
                rel = f"{rel_dir}/{d}"
                if d in MANIFEST_EXCLUDED_DIR_NAMES or _is_excluded_prefix(rel):
                    continue
                full_dir = Path(dirpath) / d
                if full_dir.is_symlink():
                    try:
                        manifest[rel] = SYMLINK_PREFIX + os.readlink(full_dir)
                    except OSError:
                        pass
                    continue
                kept.append(d)
            dirnames[:] = kept
            for name in filenames:
                if name.endswith(MANIFEST_EXCLUDED_SUFFIXES):
                    continue
                full = Path(dirpath) / name
                if full.is_symlink():
                    try:
                        manifest[f"{rel_dir}/{name}"] = SYMLINK_PREFIX + os.readlink(full)
                    except OSError:
                        pass
                    continue
                if not full.is_file():
                    continue
                try:
                    manifest[f"{rel_dir}/{name}"] = _sha256(full)
                except OSError:
                    # A file deleted mid-walk by a concurrent agent is simply absent.
                    continue
    return dict(sorted(manifest.items()))


def _is_excluded_prefix(rel: str) -> bool:
    return any(rel == p or rel.startswith(p + "/") for p in MANIFEST_EXCLUDED_PREFIXES)


@dataclass
class ManifestDiff:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    allowed: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        return {k: list(getattr(self, k)) for k in ("added", "removed", "modified", "allowed", "unexpected")}


def _segments_match(parts: Sequence[str], pats: Sequence[str]) -> bool:
    """Match path segments against glob segments; ``**`` spans zero or more segments."""
    if not pats:
        return not parts
    head, rest = pats[0], pats[1:]
    if head == "**":
        return any(_segments_match(parts[i:], rest) for i in range(len(parts) + 1))
    return bool(parts) and fnmatch.fnmatchcase(parts[0], head) and _segments_match(parts[1:], rest)


def glob_match(path: str, pattern: str) -> bool:
    """Segment-aware glob: ``*``/``?``/``[...]`` stay within one segment, ``**`` crosses any depth."""
    return _segments_match(path.split("/"), pattern.split("/"))


def is_allowed(path: str, allowlist: Iterable[str]) -> bool:
    """A path is allowed when it matches a glob or sits under an allowlisted directory (``dir/``)."""
    for pattern in allowlist:
        pattern = pattern.strip()
        if not pattern:
            continue
        if pattern.endswith("/"):
            if path.startswith(pattern):
                return True
            continue
        if glob_match(path, pattern):
            return True
    return False


def diff_manifest(baseline: Mapping[str, str], current: Mapping[str, str],
                  allowlist: Iterable[str] = ()) -> ManifestDiff:
    allow = list(allowlist)
    diff = ManifestDiff(
        added=sorted(set(current) - set(baseline)),
        removed=sorted(set(baseline) - set(current)),
        modified=sorted(p for p in set(baseline) & set(current) if baseline[p] != current[p]),
    )
    for path in sorted(diff.added + diff.removed + diff.modified):
        (diff.allowed if is_allowed(path, allow) else diff.unexpected).append(path)
    return diff


def load_allowlist(paths: Sequence[str], inline: Sequence[str]) -> list[str]:
    patterns = [p for p in inline if p.strip()]
    for file_name in paths:
        for line in Path(file_name).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                patterns.append(line)
    return patterns


# --------------------------------------------------------------------------------------------
# Diagnostic parsing
# --------------------------------------------------------------------------------------------

_TSC_RE = re.compile(r"^(?P<file>[^\s(][^(]*)\((?P<line>\d+),(?P<col>\d+)\): error (?P<code>TS\d+): (?P<msg>.*)$")
_TSC_GLOBAL_RE = re.compile(r"^error (?P<code>TS\d+): (?P<msg>.*)$")
_ESLINT_FILE_RE = re.compile(r"^(/|[A-Za-z]:\\).+")
_ESLINT_ROW_RE = re.compile(r"^\s+(?P<line>\d+):(?P<col>\d+)\s+(?P<sev>error|warning)\s+(?P<msg>.+?)(?:\s{2,}(?P<rule>[@\w/-]+))?\s*$")


def parse_tsc(output: str) -> Counter[str]:
    """Line-independent keys ``file|code|message`` with occurrence counts."""
    keys: Counter[str] = Counter()
    for raw in output.splitlines():
        line = raw.rstrip()
        match = _TSC_RE.match(line)
        if match:
            keys[f"{match['file'].strip()}|{match['code']}|{match['msg'].strip()}"] += 1
            continue
        match = _TSC_GLOBAL_RE.match(line)
        if match:
            keys[f"<global>|{match['code']}|{match['msg'].strip()}"] += 1
    return keys


def parse_eslint(output: str, root: Path = FRONTEND_DIR) -> Counter[str]:
    """Keys ``file|severity|rule|message`` from ESLint's stylish formatter, with counts."""
    keys: Counter[str] = Counter()
    current: str | None = None
    for raw in output.splitlines():
        if _ESLINT_FILE_RE.match(raw):
            current = raw.strip()
            try:
                current = Path(current).resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                pass
            continue
        match = _ESLINT_ROW_RE.match(raw)
        if match and current is not None:
            keys[f"{current}|{match['sev']}|{match['rule'] or '-'}|{match['msg'].strip()}"] += 1
    return keys


def new_keys(baseline: Mapping[str, int], current: Mapping[str, int]) -> dict[str, int]:
    """Keys whose count grew relative to the baseline (count = how many extra occurrences)."""
    return {k: n - int(baseline.get(k, 0)) for k, n in sorted(current.items()) if n > int(baseline.get(k, 0))}


def parse_junit(xml_text: str) -> dict[str, str]:
    """``{nodeid: passed|failed|error|skipped}`` from a pytest junit XML report."""
    outcomes: dict[str, str] = {}
    root = ET.fromstring(xml_text)
    unqualified = False
    for case in root.iter("testcase"):
        classname = case.get("classname", "") or ""
        if not classname:
            # Without a file-qualified classname, same-named tests in different files would
            # merge into one id and mask each other; flag it as an error instead of guessing.
            unqualified = True
        nodeid = f"{classname}::{case.get('name', '')}"
        if case.find("failure") is not None:
            outcome = "failed"
        elif case.find("error") is not None:
            outcome = "error"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        else:
            outcome = "passed"
        # A test that errors in teardown after passing is reported twice; the worst wins.
        if outcomes.get(nodeid) in ("failed", "error"):
            continue
        outcomes[nodeid] = outcome
    if unqualified:
        outcomes[UNQUALIFIED_JUNIT_ID] = "error"
    return outcomes


def align_legacy_ids(baseline: Mapping[str, str], current: Mapping[str, str]) -> dict[str, str]:
    """Map bare ``::name`` baseline ids (wave-0 baseline, recorded without ``--rootdir``) onto
    the file-qualified ids of the current run.

    The wave-0 baseline names were checked unique, so each bare id maps to every current id
    with that test name (each inherits the baseline outcome). A bare id with no current match
    stays as is and is reported as vanished if it passed.
    """
    by_name: dict[str, list[str]] = {}
    for key in current:
        cls, sep, name = key.partition("::")
        if sep and cls:
            by_name.setdefault(name, []).append(key)
    aligned: dict[str, str] = {}
    for key, outcome in baseline.items():
        cls, sep, name = key.partition("::")
        targets = by_name.get(name, []) if sep and not cls else []
        if not targets:
            aligned[key] = outcome
            continue
        for target in targets:
            # Never let a bare baseline id overwrite a qualified one; worst outcome wins.
            prev = aligned.get(target)
            if prev in ("failed", "error"):
                continue
            aligned[target] = outcome
    return aligned


def new_test_failures(baseline: Mapping[str, str], current: Mapping[str, str]) -> list[str]:
    bad = ("failed", "error")
    return sorted(t for t, o in current.items() if o in bad and baseline.get(t) not in bad)


def weakened_tests(baseline: Mapping[str, str], current: Mapping[str, str]) -> list[str]:
    """Tests that passed in the baseline and are now skipped (a silent weakening of the gate)."""
    return sorted(t for t, o in current.items() if o == "skipped" and baseline.get(t) == "passed")


def _not_run_status(baseline_status: object) -> str:
    """NOT_RUN passes through only when the baseline was NOT_RUN too; otherwise INCOMPLETE."""
    return NOT_RUN if baseline_status in (None, NOT_RUN) else INCOMPLETE


# --------------------------------------------------------------------------------------------
# Running checks
# --------------------------------------------------------------------------------------------

@dataclass
class CommandResult:
    name: str
    status: str
    exit_code: int | None
    output: str
    seconds: float
    note: str = ""

    def summary(self) -> dict[str, object]:
        return {"name": self.name, "status": self.status, "exit_code": self.exit_code,
                "seconds": round(self.seconds, 2), "note": self.note}


def _run(name: str, cmd: Sequence[str], cwd: Path, env: Mapping[str, str] | None = None) -> CommandResult:
    start = time.monotonic()
    merged = dict(os.environ)
    if env:
        merged.update(env)
    try:
        proc = subprocess.run(list(cmd), cwd=cwd, env=merged, capture_output=True, text=True,
                              timeout=CHECK_TIMEOUT_S)
    except FileNotFoundError as exc:
        return CommandResult(name, NOT_RUN, None, str(exc), time.monotonic() - start, "tool not found")
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "") if isinstance(exc.stdout, str) else ""
        return CommandResult(name, FAIL, None, out, time.monotonic() - start, f"timeout {CHECK_TIMEOUT_S}s")
    output = proc.stdout + ("\n" if proc.stdout and proc.stderr else "") + proc.stderr
    return CommandResult(name, PASS if proc.returncode == 0 else FAIL, proc.returncode, output,
                         time.monotonic() - start)


def run_tsc() -> CommandResult:
    return _run("tsc", [str(FRONTEND_DIR / "node_modules/.bin/tsc"), "--noEmit"], FRONTEND_DIR)


def run_lint() -> CommandResult:
    npm = shutil.which("npm")
    if npm is None:
        return CommandResult("lint", NOT_RUN, None, "", 0.0, "npm not found")
    return _run("lint", [npm, "run", "lint"], FRONTEND_DIR, {"NO_UPDATE_NOTIFIER": "1", "npm_config_update_notifier": "false"})


def vendor_scripts() -> list[Path]:
    return sorted((FRONTEND_DIR / "scripts").glob("check-vendor-*.mjs"))


def run_vendor_script(script: Path, include_browser: bool) -> CommandResult:
    name = script.name
    if name in BROWSER_VENDOR_SCRIPTS and not include_browser:
        return CommandResult(name, NOT_RUN, None, "", 0.0,
                             "browser script: needs a local demo server (VENDOR_DEMO_ORIGIN) with "
                             "CAD-derived fixtures; pass --include-browser to run")
    node = shutil.which("node")
    if node is None:
        return CommandResult(name, NOT_RUN, None, "", 0.0, "node not found")
    return _run(name, [node, str(script.relative_to(FRONTEND_DIR))], FRONTEND_DIR,
                {"PYTHONDONTWRITEBYTECODE": "1"})


def run_backend_tests(tests: Sequence[str] = BACKEND_TESTS) -> tuple[CommandResult, dict[str, str]]:
    present = [t for t in tests if (REPO_ROOT / t).is_file()]
    missing = [t for t in tests if t not in present]
    if not present:
        # Never fall through to a blanket pytest over the repo (forbidden by the run rules).
        outcomes = {f"{t}::<file missing>": "error" for t in missing}
        return CommandResult("backend_vendor_setup_pytest", FAIL, None, "", 0.0,
                             f"all test files missing: {missing}; pytest not run"), outcomes
    with tempfile.TemporaryDirectory(prefix="g5_junit_") as tmp:
        xml_path = Path(tmp) / "junit.xml"
        result = _run("backend_vendor_setup_pytest", [
            str(PYTHON), "-m", "pytest", "-c", "/dev/null", "--rootdir", str(REPO_ROOT),
            "--noconftest", "-p", "no:cacheprovider", *present, "-q", f"--junitxml={xml_path}",
        ], REPO_ROOT, {
            "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": ".:backend", "OMP_NUM_THREADS": "1",
        })
        outcomes = parse_junit(xml_path.read_text(encoding="utf-8")) if xml_path.is_file() else {}
    for test_file in missing:
        # A vanished test file is a failure, not a silent pass.
        outcomes[f"{test_file}::<file missing>"] = "error"
    if missing:
        result.note = f"missing test files: {missing}"
    return result, outcomes


# --------------------------------------------------------------------------------------------
# Baseline I/O
# --------------------------------------------------------------------------------------------

def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")


def collect_checks(include_browser: bool, only: set[str]) -> dict[str, object]:
    """Run frontend/backend checks; return a JSON-ready results document plus raw outputs."""
    results: dict[str, object] = {"raw": {}}
    raw: dict[str, str] = results["raw"]  # type: ignore[assignment]
    if "frontend" in only:
        tsc, lint = run_tsc(), run_lint()
        raw["tsc"], raw["lint"] = tsc.output, lint.output
        results["tsc"] = {**tsc.summary(), "diagnostics": dict(parse_tsc(tsc.output))}
        results["lint"] = {**lint.summary(), "diagnostics": dict(parse_eslint(lint.output))}
        vendor: dict[str, object] = {}
        for script in vendor_scripts():
            res = run_vendor_script(script, include_browser)
            raw[script.name] = res.output
            vendor[script.name] = res.summary()
        results["vendor_scripts"] = vendor
    if "backend" in only:
        res, outcomes = run_backend_tests()
        raw["backend_pytest"] = res.output
        results["backend_tests"] = {**res.summary(), "files": list(BACKEND_TESTS), "outcomes": outcomes}
    return results


def write_baseline(include_browser: bool, manifest: Mapping[str, str] | None = None) -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    captured = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if manifest is None:
        manifest = compute_manifest()
    _write_json(BASELINE_DIR / "manifest.json", {
        "schema": "g5_manifest_v1", "captured_utc": captured, "roots": list(MANIFEST_ROOTS),
        "excluded": {"prefixes": list(MANIFEST_EXCLUDED_PREFIXES), "dir_names": sorted(MANIFEST_EXCLUDED_DIR_NAMES),
                     "suffixes": list(MANIFEST_EXCLUDED_SUFFIXES),
                     "symlinks": "recorded as symlink:<target>, not followed"},
        "file_count": len(manifest), "files": dict(manifest),
    })
    results = collect_checks(include_browser, {"frontend", "backend"})
    raw = results.pop("raw")
    logs = BASELINE_DIR / "logs"
    logs.mkdir(exist_ok=True)
    for name, text in raw.items():  # type: ignore[union-attr]
        (logs / f"{name}.txt").write_text(text, encoding="utf-8")
    _write_json(BASELINE_DIR / "results.json", {"schema": "g5_results_v1", "captured_utc": captured, **results})


def baseline_hashes(directory: Path | None = None) -> dict[str, str]:
    directory = BASELINE_DIR if directory is None else directory
    return {name: _sha256(directory / name) for name in PINNED_BASELINE_SHA256 if (directory / name).is_file()}


def baseline_integrity(directory: Path | None = None,
                       pinned: Mapping[str, str] = PINNED_BASELINE_SHA256) -> dict[str, object]:
    """Compare the baseline files' sha256 with the pinned values; a mismatch is a FAIL."""
    actual = baseline_hashes(directory)
    mismatched = sorted(n for n, h in pinned.items() if actual.get(n) != h)
    return {"status": FAIL if mismatched else PASS, "sha256": actual, "pinned": dict(pinned),
            "mismatched": mismatched}


def load_baseline() -> tuple[dict[str, str], dict[str, object]]:
    manifest_path, results_path = BASELINE_DIR / "manifest.json", BASELINE_DIR / "results.json"
    if not manifest_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(f"G5 baseline missing under {BASELINE_DIR}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    results = json.loads(results_path.read_text(encoding="utf-8"))
    return manifest, results


# --------------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------------

def compare(baseline_results: Mapping[str, object], current: Mapping[str, object]) -> dict[str, object]:
    """Return ``{check: {status, new: [...], preexisting: [...]}}`` - only NEW failures fail."""
    report: dict[str, object] = {}

    for name, parser_key in (("tsc", "diagnostics"), ("lint", "diagnostics")):
        if name not in current:
            continue
        cur = current[name]  # type: ignore[index]
        base = baseline_results.get(name, {})  # type: ignore[union-attr]
        if cur["status"] == NOT_RUN:
            report[name] = {"status": _not_run_status(base.get("status")), "baseline": base.get("status"),
                            "note": cur.get("note", "")}
            continue
        added = new_keys(base.get(parser_key, {}), cur.get(parser_key, {}))
        # A FAIL with nothing parsable is new unless it reproduces an equally unparsable baseline
        # FAIL with the same exit code (e.g. a crash after a baseline that had real diagnostics).
        unparsed_new = (cur["status"] == FAIL and not cur.get(parser_key) and not added
                        and (base.get("status") != FAIL or bool(base.get(parser_key))
                             or cur.get("exit_code") != base.get("exit_code")))
        new: list[str] = [f"{k} (+{n})" for k, n in added.items()]
        if unparsed_new:
            new.append(f"{name} exited {cur['exit_code']} with no parsable diagnostics (baseline {base.get('status')})")
        report[name] = {"status": FAIL if new else PASS, "exit_code": cur["exit_code"],
                        "baseline_exit_code": base.get("exit_code"), "new": new,
                        "preexisting_count": sum(int(v) for v in base.get(parser_key, {}).values())}

    if "vendor_scripts" in current:
        base_vendor = baseline_results.get("vendor_scripts", {})  # type: ignore[union-attr]
        per: dict[str, object] = {}
        new_any: list[str] = []
        incomplete_any: list[str] = []
        for script, cur in current["vendor_scripts"].items():  # type: ignore[union-attr]
            base = base_vendor.get(script)
            if cur["status"] == NOT_RUN:
                status = _not_run_status(base and base["status"])
                per[script] = {"status": status, "baseline": base and base["status"], "note": cur["note"]}
                if status == INCOMPLETE:
                    incomplete_any.append(script)
            elif cur["status"] == FAIL and (base is None or base["status"] != FAIL):
                per[script] = {"status": FAIL, "baseline": base and base["status"], "new": True}
                new_any.append(script)
            else:
                per[script] = {"status": cur["status"], "baseline": base and base["status"], "new": False}
        for script in sorted(set(base_vendor) - set(current["vendor_scripts"])):  # type: ignore[arg-type]
            per[script] = {"status": FAIL, "baseline": base_vendor[script]["status"], "new": True,
                           "note": "script disappeared"}
            new_any.append(script)
        not_run = sorted(s for s, row in per.items() if row["status"] == NOT_RUN)  # type: ignore[index]
        report["vendor_scripts"] = {"status": FAIL if new_any else (INCOMPLETE if incomplete_any else PASS),
                                    "new": new_any, "incomplete": incomplete_any,
                                    "not_run": not_run, "not_run_count": len(not_run), "scripts": per}

    if "backend_tests" in current:
        cur = current["backend_tests"]  # type: ignore[index]
        base = baseline_results.get("backend_tests", {})  # type: ignore[union-attr]
        cur_out = cur.get("outcomes", {})
        base_out = align_legacy_ids(base.get("outcomes", {}), cur_out)
        new = new_test_failures(base_out, cur_out)
        vanished = sorted(t for t, o in base_out.items() if o == "passed" and t not in cur_out)
        weakened = weakened_tests(base_out, cur_out)
        if cur["status"] in (FAIL, NOT_RUN) and not cur_out:
            new.append(f"pytest {cur['status']} (exit {cur['exit_code']}) with no junit outcomes "
                       f"(collection/usage error or interpreter missing)")
        report["backend_tests"] = {
            "status": FAIL if (new or vanished or weakened) else PASS, "exit_code": cur["exit_code"],
            "new": new, "vanished_passing_tests": vanished, "weakened_tests": weakened,
            "preexisting": sorted(t for t, o in cur_out.items() if o in ("failed", "error") and t not in new),
            "counts": dict(Counter(cur_out.values())),
        }
    return report


def run_gate(allowlist: Sequence[str], include_browser: bool, only: set[str]) -> dict[str, object]:
    base_manifest, base_results = load_baseline()
    report: dict[str, object] = {"schema": "g5_report_v1", "baseline_captured_utc": base_results.get("captured_utc"),
                                 "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                 "allowlist": list(allowlist)}
    report["baseline_integrity"] = baseline_integrity()
    if "manifest" in only:
        diff = diff_manifest(base_manifest, compute_manifest(), allowlist)
        report["manifest"] = {"status": FAIL if diff.unexpected else PASS, **diff.to_dict()}
    current = collect_checks(include_browser, only & {"frontend", "backend"})
    current.pop("raw", None)
    report.update(compare(base_results, current))
    report["gate"], report["failing_checks"], report["incomplete_checks"] = gate_verdict(report)
    return report


def gate_verdict(report: Mapping[str, object]) -> tuple[str, list[str], list[str]]:
    """FAIL beats INCOMPLETE beats PASS. A NOT_RUN check (with a NOT_RUN baseline) is neutral."""
    rows = {k: v for k, v in report.items() if isinstance(v, dict) and "status" in v}
    failing = [k for k, v in rows.items() if v["status"] == FAIL]
    incomplete = [k for k, v in rows.items() if v["status"] == INCOMPLETE]
    return (FAIL if failing else INCOMPLETE if incomplete else PASS), failing, incomplete


def print_report(report: Mapping[str, object]) -> None:
    vendor = report.get("vendor_scripts")
    suffix = ""
    if isinstance(vendor, dict) and vendor.get("not_run_count"):
        suffix = f" ({vendor['not_run_count']} vendor scripts NOT_RUN, matching baseline: not covered)"
    print(f"G5 gate: {report['gate']}{suffix}  (baseline {report.get('baseline_captured_utc')})")
    integrity = report.get("baseline_integrity")
    if isinstance(integrity, dict):
        print(f"  baseline_integrity: {integrity['status']}  "
              + "  ".join(f"{n}={h}" for n, h in integrity["sha256"].items()))  # type: ignore[union-attr]
        for name in integrity["mismatched"]:  # type: ignore[union-attr]
            print(f"    MISMATCH vs pinned sha256: {name}")
    if report.get("incomplete_checks"):
        print(f"  INCOMPLETE (ran in baseline, not run now): {report['incomplete_checks']}")
    manifest = report.get("manifest")
    if isinstance(manifest, dict):
        print(f"  manifest: {manifest['status']}  added={len(manifest['added'])} removed={len(manifest['removed'])} "
              f"modified={len(manifest['modified'])} allowed={len(manifest['allowed'])} "
              f"UNEXPECTED={len(manifest['unexpected'])}")
        for path in manifest["unexpected"][:200]:
            print(f"    unexpected: {path}")
        for path in manifest["allowed"][:200]:
            print(f"    allowed (licensed): {path}")
    for name in ("tsc", "lint", "backend_tests"):
        entry = report.get(name)
        if isinstance(entry, dict):
            print(f"  {name}: {entry['status']}  new={len(entry.get('new', []))}"
                  + (f"  ({entry['note']})" if entry.get("note") else ""))
            for item in entry.get("new", [])[:50]:
                print(f"    NEW: {item}")
            for item in entry.get("vanished_passing_tests", [])[:50]:
                print(f"    VANISHED: {item}")
            for item in entry.get("weakened_tests", [])[:50]:
                print(f"    WEAKENED (passed -> skipped): {item}")
    if isinstance(vendor, dict):
        print(f"  vendor_scripts: {vendor['status']}  not_run={vendor.get('not_run_count', 0)}")
        for script, row in vendor["scripts"].items():
            flag = " NEW FAILURE" if row.get("new") else ""
            print(f"    {script}: {row['status']} (baseline {row.get('baseline')}){flag}")


# --------------------------------------------------------------------------------------------
# Self-test (mutate-and-check; no repository writes)
# --------------------------------------------------------------------------------------------

def self_test() -> int:
    failures: list[str] = []

    def check(cond: bool, label: str) -> None:
        if not cond:
            failures.append(label)

    with tempfile.TemporaryDirectory(prefix="g5_selftest_") as tmp:
        root = Path(tmp)
        for rel, text in {"backend/a.py": "a", "backend/__pycache__/a.cpython-311.pyc": "x",
                          "backend/b.pyc": "x", "scripts/s.py": "s", "scripts/production/p.py": "p",
                          "domains/bend/d.py": "d", "other/o.py": "o"}.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)
        (root / "backend/link.py").symlink_to(root / "backend/a.py")
        base = compute_manifest(root)
        check(sorted(base) == ["backend/a.py", "backend/link.py", "domains/bend/d.py", "scripts/s.py"],
              f"manifest scope {sorted(base)}")
        check(base.get("backend/link.py", "").startswith(SYMLINK_PREFIX), "symlink recorded as link target")
        check(diff_manifest(base, compute_manifest(root)).unexpected == [], "unchanged tree with symlink clean")
        (root / "backend/link.py").unlink()
        (root / "backend/link.py").symlink_to(root / "scripts/s.py")
        (root / "backend/dirlink").symlink_to(root / "other")
        relinked = compute_manifest(root)
        check(diff_manifest(base, relinked).modified == ["backend/link.py"], "retargeted symlink is a change")
        check(diff_manifest(base, relinked).added == ["backend/dirlink"], "new directory symlink is added")
        check("backend/dirlink/o.py" not in relinked, "directory symlinks are not followed")
        (root / "backend/link.py").unlink()
        (root / "backend/dirlink").unlink()
        base = compute_manifest(root)
        check(diff_manifest(base, compute_manifest(root)).unexpected == [], "unchanged tree must be clean")
        (root / "backend/a.py").write_text("mutated")
        (root / "scripts/new.py").write_text("n")
        (root / "domains/bend/d.py").unlink()
        (root / "scripts/production/p2.py").write_text("ignored")
        diff = diff_manifest(base, compute_manifest(root))
        check(diff.modified == ["backend/a.py"], f"modified {diff.modified}")
        check(diff.added == ["scripts/new.py"], f"added {diff.added}")
        check(diff.removed == ["domains/bend/d.py"], f"removed {diff.removed}")
        check(len(diff.unexpected) == 3, "all three changes unexpected without allowlist")
        diff = diff_manifest(base, compute_manifest(root), ["backend/a.py", "domains/"])
        check(diff.unexpected == ["scripts/new.py"], f"allowlist partial {diff.unexpected}")
        diff = diff_manifest(base, compute_manifest(root), ["scripts/*.py", "backend/*", "domains/bend/*"])
        check(diff.unexpected == [], "globs cover everything")
        check(not is_allowed("backend/a.py", ["backend/a"]), "prefix without slash must not match")
        check(not is_allowed("scripts/sub/x.py", ["scripts/*.py"]), "'*' must not cross '/'")
        check(is_allowed("scripts/sub/x.py", ["scripts/**/*.py"]), "'**' spans directories")
        check(is_allowed("scripts/x.py", ["scripts/**/*.py"]), "'**' matches zero directories")
        check(not is_allowed("scripts/x.sh", ["scripts/**/*.py"]), "'**' still checks the basename")
        check(is_allowed("scripts/a/b/c.sh", ["scripts/a/"]), "trailing slash allows the subtree")

    tsc_base = parse_tsc("src/a.ts(3,4): error TS2322: Type 'x' is not assignable.\n")
    tsc_moved = parse_tsc("src/a.ts(9,1): error TS2322: Type 'x' is not assignable.\n")
    check(new_keys(tsc_base, tsc_moved) == {}, "moved tsc error is not new")
    tsc_more = parse_tsc("src/a.ts(9,1): error TS2322: Type 'x' is not assignable.\n"
                         "src/a.ts(10,1): error TS2322: Type 'x' is not assignable.\n"
                         "src/b.ts(1,1): error TS7006: Parameter 'p' implicitly has an 'any' type.\n")
    check(len(new_keys(tsc_base, tsc_more)) == 2, f"tsc new keys {new_keys(tsc_base, tsc_more)}")

    lint_text = (f"\n{FRONTEND_DIR}/src/x.tsx\n  12:5  error  'y' is assigned a value but never used  "
                 "@typescript-eslint/no-unused-vars\n\n✖ 1 problem (1 error, 0 warnings)\n")
    lint = parse_eslint(lint_text)
    check(list(lint) == ["src/x.tsx|error|@typescript-eslint/no-unused-vars|'y' is assigned a value but never used"],
          f"eslint parse {dict(lint)}")
    check(new_keys(lint, parse_eslint(lint_text.replace("12:5", "40:2"))) == {}, "moved lint error is not new")
    check(len(new_keys({}, lint)) == 1, "fresh lint error is new")

    junit = ('<testsuites><testsuite><testcase classname="t.m" name="ok"/>'
             '<testcase classname="t.m" name="bad"><failure message="x"/></testcase>'
             '<testcase classname="t.m" name="err"><error message="x"/></testcase>'
             '<testcase classname="t.m" name="sk"><skipped/></testcase></testsuite></testsuites>')
    out = parse_junit(junit)
    check(out == {"t.m::ok": "passed", "t.m::bad": "failed", "t.m::err": "error", "t.m::sk": "skipped"}, f"junit {out}")
    check(new_test_failures({"t.m::ok": "passed", "t.m::bad": "failed"}, out) == ["t.m::err"], "pre-existing failure not new")
    check(new_test_failures({}, {"x::new": "failed"}) == ["x::new"], "failing new test is new")
    # Real -c /dev/null output: empty classnames. Same-named tests in two files must not merge
    # silently: the run is flagged as an error.
    junit_bare = ('<testsuites><testsuite><testcase classname="" name="dup"/>'
                  '<testcase classname="" name="dup"><failure message="x"/></testcase></testsuite></testsuites>')
    out_bare = parse_junit(junit_bare)
    check(out_bare.get(UNQUALIFIED_JUNIT_ID) == "error", f"empty classname flagged {out_bare}")
    check(new_test_failures({}, out_bare) != [], "unqualified ids produce a new failure")
    # Legacy bare baseline ids align onto qualified current ids.
    legacy = {"::a": "passed", "::b": "failed", "::gone": "passed"}
    cur_q = {"pkg.f1::a": "failed", "pkg.f2::b": "failed"}
    aligned = align_legacy_ids(legacy, cur_q)
    check(aligned == {"pkg.f1::a": "passed", "pkg.f2::b": "failed", "::gone": "passed"}, f"align {aligned}")
    rep_leg = compare({"backend_tests": {"outcomes": legacy}},
                      {"backend_tests": {"status": FAIL, "exit_code": 1, "outcomes": cur_q}})
    check(rep_leg["backend_tests"]["new"] == ["pkg.f1::a"], f"legacy new {rep_leg['backend_tests']['new']}")  # type: ignore[index]
    check(rep_leg["backend_tests"]["vanished_passing_tests"] == ["::gone"], "legacy vanished detected")  # type: ignore[index]
    # A same-named test added in a second file: the new copy's failure is not masked.
    dup = align_legacy_ids({"::a": "passed"}, {"pkg.f1::a": "passed", "pkg.f2::a": "failed"})
    check(new_test_failures(dup, {"pkg.f1::a": "passed", "pkg.f2::a": "failed"}) == ["pkg.f2::a"],
          "duplicate name in a new file still reported")
    # All test files missing: never a blanket pytest run.
    res_missing, out_missing = run_backend_tests(("does/not/exist_test_g5.py",))
    check(res_missing.status == FAIL and res_missing.exit_code is None and res_missing.seconds == 0.0,
          f"missing files -> FAIL without running pytest {res_missing.summary()}")
    check(out_missing == {"does/not/exist_test_g5.py::<file missing>": "error"}, f"missing outcomes {out_missing}")

    base_results = {"tsc": {"status": PASS, "exit_code": 0, "diagnostics": {}},
                    "vendor_scripts": {"a.mjs": {"status": PASS}, "b.mjs": {"status": FAIL}, "gone.mjs": {"status": PASS}},
                    "backend_tests": {"outcomes": {"t::a": "passed", "t::b": "failed"}}}
    current = {"tsc": {"status": FAIL, "exit_code": 2, "diagnostics": {}},
               "vendor_scripts": {"a.mjs": {"status": FAIL, "exit_code": 1, "note": ""},
                                  "b.mjs": {"status": FAIL, "exit_code": 1, "note": ""},
                                  "c.mjs": {"status": NOT_RUN, "exit_code": None, "note": "env"}},
               "backend_tests": {"status": FAIL, "exit_code": 1, "outcomes": {"t::b": "failed"}}}
    rep = compare(base_results, current)
    check(rep["tsc"]["status"] == FAIL, "tsc exit without parsable diagnostics must fail")  # type: ignore[index]
    check(sorted(rep["vendor_scripts"]["new"]) == ["a.mjs", "gone.mjs"], f"vendor new {rep['vendor_scripts']['new']}")  # type: ignore[index]
    check(rep["vendor_scripts"]["scripts"]["c.mjs"]["status"] == NOT_RUN, "NOT_RUN never becomes PASS")  # type: ignore[index]
    check(rep["backend_tests"]["vanished_passing_tests"] == ["t::a"], "vanished passing test detected")  # type: ignore[index]
    check(rep["backend_tests"]["status"] == FAIL, "vanished test fails gate")  # type: ignore[index]
    rep_ok = compare(base_results, {"backend_tests": {"status": FAIL, "exit_code": 1,
                                                      "outcomes": {"t::a": "passed", "t::b": "failed"}}})
    check(rep_ok["backend_tests"]["status"] == PASS, "only pre-existing failure -> PASS")  # type: ignore[index]

    # UNKNOWN never becomes PASS: a check that ran in the baseline but cannot run now.
    base_all = {"tsc": {"status": PASS, "exit_code": 0, "diagnostics": {}},
                "lint": {"status": PASS, "exit_code": 0, "diagnostics": {}},
                "vendor_scripts": {"ok.mjs": {"status": PASS}, "br.mjs": {"status": NOT_RUN}},
                "backend_tests": {"outcomes": {"t::a": "passed", "t::b": "passed"}}}
    not_run = {"tsc": {"status": NOT_RUN, "exit_code": None, "note": "tool not found"},
               "lint": {"status": NOT_RUN, "exit_code": None, "note": "npm not found"},
               "vendor_scripts": {"ok.mjs": {"status": NOT_RUN, "exit_code": None, "note": "node not found"},
                                  "br.mjs": {"status": NOT_RUN, "exit_code": None, "note": "browser"}}}
    rep_nr = compare(base_all, not_run)
    check(rep_nr["tsc"]["status"] == INCOMPLETE, "tsc NOT_RUN after baseline PASS -> INCOMPLETE")  # type: ignore[index]
    check(rep_nr["lint"]["status"] == INCOMPLETE, "lint NOT_RUN after baseline PASS -> INCOMPLETE")  # type: ignore[index]
    check(rep_nr["vendor_scripts"]["status"] == INCOMPLETE, "vendor PASS -> NOT_RUN must not summarise PASS")  # type: ignore[index]
    check(rep_nr["vendor_scripts"]["scripts"]["br.mjs"]["status"] == NOT_RUN, "NOT_RUN baseline passes through")  # type: ignore[index]
    check(gate_verdict(rep_nr)[0] == INCOMPLETE, f"gate with NOT_RUN checks must not PASS: {gate_verdict(rep_nr)}")
    check(gate_verdict({**rep_nr, "x": {"status": FAIL}})[0] == FAIL, "FAIL beats INCOMPLETE")
    rep_br = compare(base_all, {"vendor_scripts": {"ok.mjs": {"status": PASS, "exit_code": 0, "note": ""},
                                                   "br.mjs": {"status": NOT_RUN, "exit_code": None, "note": ""}}})
    check(gate_verdict(rep_br)[0] == PASS, "matching NOT_RUN baseline keeps gate PASS")
    check(rep_br["vendor_scripts"]["not_run"] == ["br.mjs"] and rep_br["vendor_scripts"]["not_run_count"] == 1,  # type: ignore[index]
          "NOT_RUN scripts are counted in the summary")
    rep_sk = compare(base_all, {"backend_tests": {"status": PASS, "exit_code": 0,
                                                  "outcomes": {"t::a": "skipped", "t::b": "passed"}}})
    check(rep_sk["backend_tests"]["weakened_tests"] == ["t::a"], "passed -> skipped is reported")  # type: ignore[index]
    check(rep_sk["backend_tests"]["status"] == FAIL, "passed -> skipped fails backend")  # type: ignore[index]
    rep_py = compare(base_all, {"backend_tests": {"status": NOT_RUN, "exit_code": None, "outcomes": {}}})
    check(rep_py["backend_tests"]["status"] == FAIL, "pytest not runnable must not PASS")  # type: ignore[index]
    # Crash with no parsable output after a baseline that FAILED with real diagnostics.
    base_diag = {"tsc": {"status": FAIL, "exit_code": 2, "diagnostics": {"a|TS1|m": 1}}}
    rep_crash = compare(base_diag, {"tsc": {"status": FAIL, "exit_code": 134, "diagnostics": {}}})
    check(rep_crash["tsc"]["status"] == FAIL, "crash after diagnostic baseline FAIL must fail")  # type: ignore[index]
    rep_crash2 = compare(base_diag, {"tsc": {"status": FAIL, "exit_code": 2, "diagnostics": {}}})
    check(rep_crash2["tsc"]["status"] == FAIL, "same exit code but diagnostics vanished must fail")  # type: ignore[index]
    rep_py0 = compare({"backend_tests": {"outcomes": {}}},
                      {"backend_tests": {"status": NOT_RUN, "exit_code": None, "outcomes": {}}})
    check(rep_py0["backend_tests"]["status"] == FAIL, "pytest NOT_RUN with empty baseline must not PASS")  # type: ignore[index]
    rep_same = compare(base_diag, {"tsc": {"status": FAIL, "exit_code": 2, "diagnostics": {"a|TS1|m": 1}}})
    check(rep_same["tsc"]["status"] == PASS, "identical pre-existing tsc failure -> PASS")  # type: ignore[index]
    rep_fixed = compare(base_diag, {"tsc": {"status": PASS, "exit_code": 0, "diagnostics": {}}})
    check(rep_fixed["tsc"]["status"] == PASS, "fixing a baseline failure is not a regression")  # type: ignore[index]

    with tempfile.TemporaryDirectory(prefix="g5_selftest_int_") as tmp:
        bdir = Path(tmp)
        (bdir / "manifest.json").write_text("m")
        (bdir / "results.json").write_text("r")
        good = {"manifest.json": _sha256(bdir / "manifest.json"), "results.json": _sha256(bdir / "results.json")}
        check(baseline_integrity(bdir, good)["status"] == PASS, "untouched baseline integrity PASS")
        (bdir / "results.json").write_text("tampered")
        integ = baseline_integrity(bdir, good)
        check(integ["status"] == FAIL and integ["mismatched"] == ["results.json"], f"tamper detected {integ}")
        check(gate_verdict({"baseline_integrity": integ})[0] == FAIL, "tampered baseline fails gate")
        (bdir / "manifest.json").unlink()
        check("manifest.json" in baseline_integrity(bdir, good)["mismatched"], "missing baseline file mismatches")  # type: ignore[operator]

    for item in failures:
        print(f"SELF-TEST FAIL: {item}")
    print(f"self-test: {'PASS' if not failures else 'FAIL'} ({len(failures)} failures)")
    return 0 if not failures else 1


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--allow", action="append", default=[], help="allowlisted path glob (repeatable; '*' stays within one segment, '**' spans "
                             "directories, trailing '/' = whole subtree)")
    parser.add_argument("--allowlist-file", action="append", default=[],
                        help="file of globs, one per line, '#' comments (in addition to the default "
                             "docs/production/runlog/g5_baseline/allowlist.txt)")
    parser.add_argument("--no-default-allowlist", action="store_true",
                        help="do not load docs/production/runlog/g5_baseline/allowlist.txt")
    parser.add_argument("--only", action="append", choices=("manifest", "frontend", "backend"),
                        help="restrict to these parts (repeatable); default all")
    parser.add_argument("--include-browser", action="store_true", help="also run Playwright check-vendor scripts")
    parser.add_argument("--json", help="write the full report as JSON to this path")
    parser.add_argument("--self-test", action="store_true", help="run built-in mutate-and-check tests and exit")
    parser.add_argument("--write-baseline", action="store_true",
                        help="(re)record the baseline; overwriting needs --force and G5_OWNER_REBASELINE=1")
    parser.add_argument("--force", action="store_true")
    parser.epilog = "exit codes: 0 PASS, 1 FAIL (new failures / unexpected changes / baseline hash mismatch), " \
                    "2 usage or baseline error, 3 INCOMPLETE"
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.write_baseline:
        if (BASELINE_DIR / "results.json").exists():
            if not args.force:
                print("baseline exists; refusing to overwrite without --force", file=sys.stderr)
                return 2
            if os.environ.get(OWNER_REBASELINE_ENV) != "1":
                print(f"refusing to re-baseline: owner only (set {OWNER_REBASELINE_ENV}=1)", file=sys.stderr)
                return 2
        write_baseline(args.include_browser)
        print(f"baseline written to {BASELINE_DIR}")
        for name, digest in baseline_hashes().items():
            print(f"  {name} sha256={digest}  (update PINNED_BASELINE_SHA256)")
        return 0
    try:
        files = list(args.allowlist_file)
        if not args.no_default_allowlist and DEFAULT_ALLOWLIST.is_file():
            files.insert(0, str(DEFAULT_ALLOWLIST))
        allowlist = load_allowlist(files, args.allow)
        report = run_gate(allowlist, args.include_browser, set(args.only or ("manifest", "frontend", "backend")))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"G5 baseline error: {exc}", file=sys.stderr)
        return 2
    print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return {PASS: 0, FAIL: 1, INCOMPLETE: 3}[str(report["gate"])]


if __name__ == "__main__":
    sys.exit(main())
