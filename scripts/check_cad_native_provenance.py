"""One bounded native worker: invented geometry only; no external input arguments.

Run with .venv/bin/python -I -S scripts/check_cad_native_provenance.py.
Fixed selectors avoid rerunning unchanged matrices.
No mode accepts external data paths. The reader and guided suites read only
invented files they create exclusively in the parent's isolated cache; the
original default suites are in-memory.
No site startup/.pth, inherited credentials, application import, network or
protected input. Python audit guards are defense in depth, not an OS sandbox.
"""
import json
import hashlib
import os
from pathlib import Path
import resource
import runpy
import selectors
import signal
import subprocess
import sys
import tempfile
import time

WALL_SECONDS = 300
CPU_SECONDS = 240
TREE_RSS_BYTES = 2 * 1024**3
OUTPUT_BYTES = 1024**2
ROOT = Path(__file__).resolve().parents[1]
CHECKS = ('cad_native_ocp_checks.py', 'cad_native_open3d_checks.py', 'cad_root_axis_math_checks.py')
READER_CHECK = 'cad_native_file_roundtrip_checks.py'
GUIDED_CHECK = 'cad_native_guided_checks.py'
REGISTERED_CHECK = 'cad_registered_guided_checks.py'
RERUN_CHECK = 'cad_guided_rerun_checks.py'
FROZEN_CONTROL_FILES = ('scripts/cad_guided_development_bundle.py',
                        'backend/cad_guided_development_control.py',
                        'docs/research/baselines/cad_datum_registered_r0_20260909.json')
FROZEN_CONTROL_DIGEST = '0460d13536cf28bd6bf2977c39d3e732d9c35c219070d9205ae5c314bdf12c46'
REGISTERED_SOURCE_FILES = (
    'backend/cad_import.py', 'backend/cad_guided_case_input.py',
    'backend/cad_guided_offline.py', 'backend/cad_guided_registration.py',
    'backend/cad_guided_measurement.py', 'backend/cad_guided_native_features.py',
    'backend/cad_guided_instances.py', 'backend/cad_guided_evaluation.py',
    'backend/cad_guided_paired_evaluation.py', 'backend/bend_inspection_pipeline.py',
    'backend/feature_detection/bend_detector.py',
    'scripts/cad_registered_guided_checks.py', 'scripts/cad_guided_rerun_checks.py',
    'scripts/check_cad_native_provenance.py',
)


def suite_files(suite):
    if suite == 'all':
        return CHECKS
    if suite == 'ocp-only':
        return CHECKS[:1]
    if suite == 'readers-only':
        return (READER_CHECK,)
    if suite == 'guided-only':
        return (GUIDED_CHECK,)
    if suite == 'registered-only':
        return (REGISTERED_CHECK,)
    if suite in ('rerun-input-only', 'frozen-control-only'):
        return (RERUN_CHECK,)
    raise ValueError('Unknown fixed native suite')


def bounded_output(block, total):
    """Never forward bytes beyond the cap, including the crossing chunk."""
    remaining = max(0, OUTPUT_BYTES - total)
    return block[:remaining], total + len(block), len(block) > remaining


def path_in_roots(path, roots):
    return any(path == root or path.startswith(root + os.sep) for root in roots)


def source_binding(suite):
    """Bind named source bytes, not a commit or an authority declaration."""
    if suite not in ('registered-only', 'rerun-input-only', 'frozen-control-only'):
        return None
    names = REGISTERED_SOURCE_FILES + (FROZEN_CONTROL_FILES if suite == 'frozen-control-only' else ())
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in names}


def install_frozen_control_importer():
    """One fixed, owned R0 snapshot; runtime metadata is checked, not attested."""
    module = runpy.run_path(str(ROOT / FROZEN_CONTROL_FILES[0]))
    bundle = module['read_bundle']()
    if bundle['bundle_sha256'] != FROZEN_CONTROL_DIGEST:
        raise ValueError('UNEXPECTED_FROZEN_CONTROL_IDENTITY')
    captured = module['validate_bundle'](bundle)
    if captured[FROZEN_CONTROL_FILES[0]] != (ROOT / FROZEN_CONTROL_FILES[0]).read_bytes():
        raise ValueError('FROZEN_BUNDLE_BOOTSTRAP_CHANGED')
    if bundle['payload']['runtime_metadata'] != module['runtime_metadata']():
        raise ValueError('FROZEN_CONTROL_RUNTIME_METADATA_CHANGED')
    module['install_captured_importer'](bundle)
    from backend.cad_guided_development_control import construct_policies
    return bundle['bundle_sha256'], construct_policies()


def install_offline_source_importer():
    """Normal Python modules from named source, never stale pyc or legacy app imports."""
    import importlib.abc
    import importlib.machinery
    import importlib.util

    paths = {'backend.' + Path(name).stem: str(ROOT / name)
             for name in REGISTERED_SOURCE_FILES
             if name.startswith('backend/') and name.count('/') == 1
             and name != 'backend/bend_inspection_pipeline.py'}

    class SourceOnly(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            return compile(self.get_data(self.path), self.path, 'exec', dont_inherit=True)

    class NamedOfflineFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'backend':
                spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
                spec.submodule_search_locations = [str(ROOT / 'backend')]
                return spec
            if fullname.startswith('backend.'):
                if fullname not in paths:
                    raise ModuleNotFoundError('Offline worker forbids unnamed backend import', name=fullname)
                return importlib.util.spec_from_file_location(fullname, paths[fullname],
                                                             loader=SourceOnly(fullname, paths[fullname]))
            return None

    sys.meta_path.insert(0, NamedOfflineFinder())


def worker(cache, suite='all'):
    selected_checks = suite_files(suite)
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    sys.dont_write_bytecode = True
    # -S on 3.11 does not adjust prefix to the venv; derive the explicit site path.
    site = str(Path(sys.executable).absolute().parent.parent / 'lib' / 'python3.11' / 'site-packages')
    sys.path.insert(0, site)
    allowed_sources = {str(ROOT / 'scripts' / name) for name in (*CHECKS, READER_CHECK, GUIDED_CHECK, REGISTERED_CHECK, RERUN_CHECK)}
    allowed_sources.update(str(ROOT / name) for name in REGISTERED_SOURCE_FILES)
    if suite == 'frozen-control-only':
        allowed_sources.update(str(ROOT / name) for name in FROZEN_CONTROL_FILES)
    allowed_sources.update(str(ROOT / p) for p in
                           ('backend/cad_import.py', 'backend/feature_detection/bend_detector.py',
                            'backend/cad_guided_native_features.py', 'backend/cad_guided_measurement.py',
                            'backend/cad_guided_registration.py', 'backend/cad_guided_instances.py',
                            'backend/cad_guided_evaluation.py',
                            'backend/bend_inspection_pipeline.py',
                            'domains/bend/services/runtime_semantics.py',
                            'research/geometry/analytic_bend_fixture.py'))
    root = str(ROOT)
    # macOS /var and /private/var name the same owned temp directory. Resolve
    # only this already-created cache, never an arbitrary incoming read path.
    cache_roots = (os.path.abspath(cache), str(Path(cache).resolve()))

    def audit(event, args):
        if event in ('subprocess.Popen', 'os.system', 'os.fork', 'os.forkpty', 'os.posix_spawn',
                     'os.exec', 'os.setpgid', 'os.setsid') or event.startswith('socket.'):
            raise RuntimeError('Worker process/network operation forbidden')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.path.abspath(os.fsdecode(args[0]))
            mode, flags = args[1:3]
            write = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            inside_cache = path_in_roots(path, cache_roots)
            if write and not inside_cache and path != '/dev/null':
                raise RuntimeError('Worker write outside isolated cache forbidden')
            if path.startswith(root + os.sep) and not (path.startswith(site + os.sep) or path in allowed_sources):
                raise RuntimeError('Worker repository read outside named source/package allowlist forbidden')
            if not inside_cache and any(part in ('.env', '.aws', '.ssh', '.codex') for part in Path(path).parts):
                raise RuntimeError('Worker private configuration access forbidden')
        if event in ('os.listdir', 'os.scandir'):
            if args and isinstance(args[0], (str, bytes, os.PathLike)):
                path = os.path.abspath(os.fsdecode(args[0]))
                if path.startswith(root) and not path.startswith(site):
                    raise RuntimeError('Worker repository enumeration forbidden')

    sys.addaudithook(audit)
    if suite in ('registered-only', 'rerun-input-only'):
        install_offline_source_importer()
    control_policies = None
    if suite == 'frozen-control-only':
        bundle_digest, control_policies = install_frozen_control_importer()
        print(json.dumps({'frozen_control_bundle_sha256': bundle_digest,
                          'engine_source': 'CAPTURED_BYTES_NOT_WORKING_TREE_OR_PYC',
                          'runtime_assurance': 'METADATA_ONLY_NOT_BINARY_ATTESTATION',
                          'historical_gold11_comparable': False}), flush=True)
    start = time.monotonic()
    results = {}
    for name in selected_checks:
        print(json.dumps({'stage_started': name}), flush=True)
        try:
            check = runpy.run_path(str(ROOT / 'scripts' / name))['run_checks']
            results[name] = (check(control_policies=control_policies)
                             if suite == 'frozen-control-only' else check())
        except Exception as exc:
            # Fixed tiny aggregate only, no data/path-bearing traceback dump.
            results[name] = {'error_type': type(exc).__name__, 'error': str(exc)[:500]}
        result = dict(results[name])
        if result.get('kind') == 'CAD_NATIVE_OPEN3D_FIXED_MATRIX_V1' and 'cases' in result:
            cases = result.pop('cases')
            result['cap_cases'] = [c for c in cases if c.get('stratum') == 'raw_cap_boundary']
            result['all_case_failures'] = [dict(case_id=c['case_id'], failure_codes=c['failure_codes'],
                                               lineage_reason=c.get('lineage_reason'))
                                         for c in cases if c['status'] != 'PASS']
            result['floor_roundoff_different_rows'] = sum(c.get('floor_reference', {}).get(
                'rounded_arithmetic_floor_different_rows', 0) for c in cases)
            result['all_legacy_geometry_exact'] = all(c.get('legacy_geometry_exact', False) for c in cases)
        print(json.dumps({'stage_result': name, 'result': result}, sort_keys=True), flush=True)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(json.dumps({'worker_wall_seconds': time.monotonic() - start,
                      'worker_cpu_seconds': usage.ru_utime + usage.ru_stime,
                      'worker_peak_rss_bytes': usage.ru_maxrss,
                      'native_worker_completed': True}, sort_keys=True), flush=True)
    return int(any('error_type' in result or result.get('status') == 'FAIL' for result in results.values()))


def parent(suite='all'):
    suite_files(suite)
    frozen_sources = source_binding(suite)
    start = time.monotonic()
    peak_rss = total_bytes = 0
    stop_reason = None
    with tempfile.TemporaryDirectory(prefix='sherman-invented-native-') as cache:
        env = {key: '1' for key in ('PYTHONDONTWRITEBYTECODE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                  'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
        env.update(HOME=cache, TMPDIR=cache, MPLCONFIGDIR=cache, XDG_CACHE_HOME=cache,
                   OPEN3D_DISABLE_WEB_VISUALIZER='True', PATH='/usr/bin:/bin', LANG='C.UTF-8')
        child = subprocess.Popen([sys.executable, '-I', '-S', str(Path(__file__).resolve()), '--worker', cache,
                                  suite],
                                 cwd=cache, env=env, start_new_session=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        selector = selectors.DefaultSelector()
        selector.register(child.stdout, selectors.EVENT_READ)
        try:
            while selector.get_map():
                for key, _ in selector.select(.1):
                    block = os.read(key.fd, 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                    else:
                        forwarded, total_bytes, overflow = bounded_output(block, total_bytes)
                        sys.stdout.buffer.write(forwarded)
                        sys.stdout.flush()
                        if overflow:
                            stop_reason = 'OUTPUT_BUDGET_EXCEEDED'
                            break
                if stop_reason:
                    os.killpg(child.pid, signal.SIGKILL)
                    break
                # macOS -g selects this new process group; descendants inherit it.
                sample = subprocess.run(['/bin/ps', '-o', 'rss=', '-g', str(child.pid)],
                                        capture_output=True, text=True, timeout=2)
                if sample.returncode == 0:
                    rss = sum(int(row) * 1024 for row in sample.stdout.split())
                    peak_rss = max(peak_rss, rss)
                elif child.poll() is None:
                    stop_reason = 'RSS_MONITOR_UNAVAILABLE'
                if time.monotonic() - start > WALL_SECONDS:
                    stop_reason = 'WALL_BUDGET_EXCEEDED'
                if peak_rss > TREE_RSS_BYTES:
                    stop_reason = 'RSS_BUDGET_EXCEEDED'
                if total_bytes > OUTPUT_BYTES:
                    stop_reason = 'OUTPUT_BUDGET_EXCEEDED'
                if stop_reason:
                    os.killpg(child.pid, signal.SIGKILL)
                    break
            child.wait(timeout=5)
        finally:
            selector.close()
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        final_sources = source_binding(suite)
        if final_sources != frozen_sources:
            stop_reason = 'SOURCE_CHANGED_DURING_EXECUTION'
        print(json.dumps({'parent_wall_seconds': time.monotonic() - start,
                          'sampled_process_group_peak_rss_bytes': peak_rss,
                          'exit_code': child.returncode, 'stop_reason': stop_reason,
                          'source_sha256': frozen_sources,
                          'source_binding_unchanged': final_sources == frozen_sources,
                          'budgets': {'wall_seconds': WALL_SECONDS, 'cpu_seconds': CPU_SECONDS,
                                      'process_tree_rss_bytes': TREE_RSS_BYTES}}, sort_keys=True))
        return int(stop_reason is not None or child.returncode != 0)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker' and sys.argv[3] in ('ocp-only', 'readers-only', 'guided-only', 'registered-only', 'rerun-input-only', 'frozen-control-only', 'all'):
        raise SystemExit(worker(sys.argv[2], suite=sys.argv[3]))
    if sys.argv[1:] not in ([], ['--ocp-only'], ['--readers-only'], ['--guided-only'], ['--registered-only'], ['--rerun-input-only'], ['--frozen-control-only']):
        raise SystemExit('Only fixed native suite selectors are accepted; no data paths')
    raise SystemExit(parent(suite=sys.argv[1][2:] if len(sys.argv) == 2 else 'all'))
