"""Pure byte-budget regression; no child, native library or app is loaded."""
import ast
import hashlib
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize('limit,total,block,forwarded,next_total,overflow', [
    (8, 0, b'abc', b'abc', 3, False),
    (8, 5, b'abc', b'abc', 8, False),
    (8, 7, b'abc', b'a', 10, True),
    (8, 8, b'abc', b'', 11, True),
    (8, 9, b'abc', b'', 12, True),
    (8, 8, b'', b'', 8, False),
])
def test_crossing_chunk_never_forwards_over_budget(limit, total, block, forwarded, next_total, overflow):
    path = Path(__file__).resolve().parents[1] / 'scripts/check_cad_native_provenance.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'bounded_output']
    assert len(nodes) == 1
    ns = {'OUTPUT_BYTES': limit}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    assert ns['bounded_output'](block, total) == (forwarded, next_total, overflow)
    assert len(forwarded) <= max(0, limit - total)


@pytest.mark.parametrize('suite,expected', [
    ('all', ('a', 'b', 'c')), ('ocp-only', ('a',)), ('readers-only', ('reader',)),
    ('guided-only', ('guided',)),
    ('registered-only', ('registered',)),
    ('rerun-input-only', ('rerun',)),
    ('/arbitrary/input.step', None), ('../data', None), ('', None), (None, None),
])
def test_suite_selection_is_fixed_and_never_accepts_data_paths(suite, expected):
    path = Path(__file__).resolve().parents[1] / 'scripts/check_cad_native_provenance.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'suite_files']
    assert len(nodes) == 1
    ns = {'CHECKS': ('a', 'b', 'c'), 'READER_CHECK': 'reader', 'GUIDED_CHECK': 'guided',
          'REGISTERED_CHECK': 'registered', 'RERUN_CHECK': 'rerun'}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    if expected is None:
        with pytest.raises(ValueError, match='Unknown fixed'):
            ns['suite_files'](suite)
    else:
        assert ns['suite_files'](suite) == expected


@pytest.mark.parametrize('path,expected', [
    ('/var/folders/owned/file.step', True),
    ('/private/var/folders/owned/file.step', True),
    ('/var/folders/owned', True),
    ('/private/var/folders/owned', True),
    ('/private/var/folders/owned-other/file.step', False),
    ('/var/folders/sibling/file.step', False),
    ('/Users/example/input.step', False),
])
def test_owned_cache_aliases_do_not_admit_siblings_or_other_paths(path, expected):
    source = Path(__file__).resolve().parents[1] / 'scripts/check_cad_native_provenance.py'
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'path_in_roots']
    assert len(nodes) == 1
    ns = {'os': os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    assert ns['path_in_roots'](path, ('/var/folders/owned', '/private/var/folders/owned')) is expected


def test_source_binding_hashes_actual_named_bytes_and_detects_edits(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'scripts/check_cad_native_provenance.py'
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name == 'source_binding']
    assert len(nodes) == 1
    namespace = {'ROOT': tmp_path, 'REGISTERED_SOURCE_FILES': ('invented.py',), 'hashlib': hashlib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    (tmp_path / 'invented.py').write_bytes(b'x = 1\n')
    before = namespace['source_binding']('rerun-input-only')
    assert before == {'invented.py': hashlib.sha256(b'x = 1\n').hexdigest()}
    (tmp_path / 'invented.py').write_bytes(b'x = 2\n')
    assert namespace['source_binding']('rerun-input-only') != before
    assert namespace['source_binding']('registered-only') != before


def test_other_fixed_native_suites_do_not_hash_new_candidate_sources(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'scripts/check_cad_native_provenance.py'
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name == 'source_binding']
    namespace = {'ROOT': tmp_path, 'REGISTERED_SOURCE_FILES': ('absent.py',), 'hashlib': hashlib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    assert namespace['source_binding']('ocp-only') is None
