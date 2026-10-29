"""Public job-list isolation using source-extracted SQL and synthetic storage.

Never import database.py or the legacy API module: only the named list/count
methods execute against a manufactured SQLite :memory: database. PostgreSQL
queries use a placeholder-translating stub, not a driver or live connection.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sqlite3
import sys
from types import ModuleType, SimpleNamespace
from typing import List, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
DATABASE_SOURCE = ast.parse((ROOT / 'backend/database.py').read_text())


def _methods(class_name):
    cls = next(node for node in DATABASE_SOURCE.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in {'list_jobs', 'count_jobs'}]
    assert {node.name for node in methods} == {'list_jobs', 'count_jobs'}
    namespace = {'Optional': Optional, 'List': List, 'JobRecord': SimpleNamespace}
    exec(compile(ast.Module(body=methods, type_ignores=[]),
                 '<source-extracted job list/count>', 'exec'), namespace)
    return namespace


@pytest.fixture(params=['SQLiteDatabaseManager', 'PostgreSQLDatabaseManager'])
def database(request, monkeypatch):
    # TestClient executes the async endpoint on its portal thread; each fixture
    # is still isolated and all requests below are sequential.
    connection = sqlite3.connect(':memory:', check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute('''CREATE TABLE jobs (
        job_id TEXT, status TEXT, progress REAL, stage TEXT, message TEXT,
        error TEXT, part_id TEXT, part_name TEXT, material TEXT, tolerance REAL,
        reference_path TEXT, scan_path TEXT, drawing_path TEXT, report_path TEXT,
        pdf_path TEXT, result_json TEXT, created_at TEXT, updated_at TEXT,
        completed_at TEXT
    )''')
    rows = [
        ('bend_private', 'completed', 'private', '09'),
        ('BEND_CASE', 'completed', 'shared', '08'),
        ('bEnD_mix', 'pending', 'shared', '07'),
        ('ordinary-a', 'completed', 'shared', '06'),
        ('bendXliteral', 'completed', 'shared', '05'),
        ('bend', 'completed', 'shared', '04'),
        ('ordinary-b', 'pending', 'shared', '03'),
        ('other', 'completed', 'other', '02'),
    ]
    connection.executemany(
        'INSERT INTO jobs (job_id,status,part_id,created_at) VALUES (?,?,?,?)', rows)
    calls = []

    class Cursor:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def execute(self, query, params):
            calls.append((query, list(params)))
            self.cursor = connection.execute(query.replace('%s', '?'), params)
            return self
        def fetchall(self):
            return self.cursor.fetchall()
        def fetchone(self):
            return self.cursor.fetchone()

    class Connection:
        def execute(self, query, params):
            return Cursor().execute(query, params)
        def cursor(self, **kwargs):
            return Cursor()

    @contextmanager
    def get_connection(self):
        yield Connection()

    # The extracted PostgreSQL method imports only this inert cursor marker.
    package = ModuleType('psycopg2')
    package.__path__ = []
    extras = ModuleType('psycopg2.extras')
    extras.RealDictCursor = object()
    monkeypatch.setitem(sys.modules, 'psycopg2', package)
    monkeypatch.setitem(sys.modules, 'psycopg2.extras', extras)
    methods = _methods(request.param)
    receiver = type('SyntheticDatabase', (), {
        'list_jobs': methods['list_jobs'], 'count_jobs': methods['count_jobs'],
        'get_connection': get_connection,
        '_row_to_job': lambda self, row: SimpleNamespace(**row),
        '_row_to_job_from_dict': lambda self, row: SimpleNamespace(**row),
    })()
    yield SimpleNamespace(db=receiver, connection=connection, calls=calls)
    connection.close()


def test_default_callers_retain_every_namespace(database):
    assert len(database.db.list_jobs()) == database.db.count_jobs() == 8
    assert all('SUBSTR' not in query for query, _ in database.calls)


@pytest.mark.parametrize('prefix', ['bend_', 'BEND_', 'bEnD_'])
def test_namespace_filter_is_literal_case_insensitive_and_precedes_pagination(database, prefix):
    jobs = database.db.list_jobs(excluded_job_prefix=prefix, limit=2, offset=1)
    assert [job.job_id for job in jobs] == ['bendXliteral', 'bend']
    assert database.db.count_jobs(excluded_job_prefix=prefix) == 5
    query, params = database.calls[0]
    assert query.index('SUBSTR') < query.index('ORDER BY') < query.index('LIMIT')
    assert params == [len(prefix), prefix, 2, 1]


@pytest.mark.parametrize('filters,expected', [
    ({'part_id': 'private'}, []),
    ({'part_id': 'shared', 'status': 'pending'}, ['ordinary-b']),
    ({'part_id': 'shared', 'status': 'completed'}, ['ordinary-a', 'bendXliteral', 'bend']),
    ({'status': 'completed'}, ['ordinary-a', 'bendXliteral', 'bend', 'other']),
    ({'status': 'failed'}, []),
])
def test_status_part_filters_and_counts_use_same_public_partition(database, filters, expected):
    kwargs = {**filters, 'excluded_job_prefix': 'bend_'}
    assert [job.job_id for job in database.db.list_jobs(**kwargs)] == expected
    assert database.db.count_jobs(**kwargs) == len(expected)


@pytest.mark.parametrize('prefix', ['public_%', "x' OR 1=1 --"])
def test_prefix_is_bound_as_literal_data_not_sql_or_like_pattern(database, prefix):
    database.connection.executemany('INSERT INTO jobs (job_id) VALUES (?)', [
        (prefix + 'hidden',), ('public_Xvisible',),
    ])
    assert database.db.count_jobs(excluded_job_prefix=prefix) == 9
    query, params = database.calls[-1]
    assert prefix not in query
    assert params == [len(prefix), prefix]


@pytest.fixture
def public_client(database, monkeypatch):
    for name, relative in [('apps', 'apps'), ('apps.api', 'apps/api'),
                           ('apps.api.services', 'apps/api/services')]:
        package = ModuleType(name)
        package.__path__ = [str(ROOT / relative)]
        monkeypatch.setitem(sys.modules, name, package)
    legacy = SimpleNamespace(
        db=database.db, DEFAULT_PAGINATION_LIMIT=50, MAX_PAGINATION_LIMIT=1000,
        validate_pagination=lambda limit, offset: (min(limit, 1000), offset),
        _json_safe=lambda value: value,
    )
    for name in ['get_progress', 'stream_progress', 'get_result', 'get_deviations',
                 'get_aligned_scan', 'get_reference_mesh', 'get_job_details',
                 'delete_job', 'get_job_dimensions', 'get_job_bends',
                 'get_job_correlations', 'get_enhanced_analysis', 'get_stats']:
        setattr(legacy, name, None)
    context = ModuleType('apps.api.legacy_context')
    context.legacy = legacy
    monkeypatch.setitem(sys.modules, context.__name__, context)
    name = 'apps.api.services.jobs_service'
    spec = importlib.util.spec_from_file_location(name, ROOT / 'apps/api/services/jobs_service.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    app = FastAPI()
    app.add_api_route('/api/jobs', module.list_jobs, methods=['GET'])
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize('offset,ids,has_more', [
    (0, ['ordinary-a', 'bendXliteral'], True),
    (2, ['bend', 'ordinary-b'], True),
    (4, ['other'], False),
    (10, [], False),
])
def test_public_pages_and_total_do_not_count_private_rows(public_client, offset, ids, has_more):
    response = public_client.get('/api/jobs', params={'limit': 2, 'offset': offset})
    assert response.status_code == 200
    payload = response.json()
    assert [job['job_id'] for job in payload['jobs']] == ids
    assert {key: payload[key] for key in ['total', 'count', 'offset', 'limit', 'has_more']} == {
        'total': 5, 'count': len(ids), 'offset': offset, 'limit': 2, 'has_more': has_more,
    }


def test_private_part_filter_reveals_no_rows_or_counts(public_client):
    response = public_client.get('/api/jobs', params={'part_id': 'private'})
    assert response.status_code == 200
    assert response.json() == {
        'jobs': [], 'total': 0, 'count': 0, 'offset': 0, 'limit': 50, 'has_more': False,
    }


def test_public_filters_and_query_validation_are_preserved(public_client):
    response = public_client.get('/api/jobs', params={'part_id': 'shared', 'status': 'pending'})
    assert response.status_code == 200
    assert [job['job_id'] for job in response.json()['jobs']] == ['ordinary-b']
    assert response.json()['total'] == 1
    assert public_client.get('/api/jobs?limit=0').status_code == 422
    assert public_client.get('/api/jobs?offset=-1').status_code == 422
