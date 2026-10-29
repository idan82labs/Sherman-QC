"""Active bend router contracts with synthetic identity, storage and worker only.

The real legacy context/readiness/auth modules are replaced BEFORE loading the
service/router. No server, credential store, native worker or research artifact
is imported/read. All files below are manufactured inside pytest's temp folder.
"""
from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import logging
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace
import uuid

import pytest
from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import FileResponse
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
# Execute only the real, import-free authentication dependency function below,
# with the credential lookup replaced. Never execute the legacy module body.
_LEGACY_AUTH_FUNCTION = next(
    node for node in ast.parse((ROOT / 'backend/_server_legacy_impl.py').read_text()).body
    if isinstance(node, ast.AsyncFunctionDef) and node.name == 'require_auth'
)
JOB_SURFACES = [
    ('GET', '/api/bend-inspection/{job_id}'),
    ('GET', '/api/bend-inspection/{job_id}/labels'),
    ('PUT', '/api/bend-inspection/{job_id}/labels'),
    ('GET', '/api/bend-inspection/{job_id}/pdf'),
    ('GET', '/api/bend-inspection/{job_id}/overlay/manifest.json'),
    ('GET', '/api/bend-inspection/{job_id}/overlay/overview.png'),
    ('GET', '/api/bend-inspection/{job_id}/overlay/issues/issue.png'),
    ('GET', '/api/bend-inspection/{job_id}/overlay/bender.png'),
    ('GET', '/api/bend-inspection/{job_id}/table'),
    ('DELETE', '/api/bend-inspection/{job_id}'),
    ('GET', '/api/aligned-scan/{job_id}.ply'),
    ('HEAD', '/api/aligned-scan/{job_id}.ply'),
    ('GET', '/api/reference-mesh/{job_id}.ply'),
    ('HEAD', '/api/reference-mesh/{job_id}.ply'),
]
OTHER_SURFACES = [
    ('POST', '/api/bend-inspection/analyze'),
    ('GET', '/api/bend-inspection/jobs'),
    ('GET', '/api/bend-inspection/readiness'),
    ('POST', '/api/bend-inspection/extract-cad'),
    ('POST', '/api/bend-inspection/analyze-from-catalog'),
    ('GET', '/api/parts/with-cad'),
]


def _load(name, relative, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    # Package shells prevent importing aggregate route/service initialization.
    for name, relative in [
        ('apps', 'apps'), ('apps.api', 'apps/api'),
        ('apps.api.services', 'apps/api/services'),
        ('apps.api.routes', 'apps/api/routes'),
    ]:
        package = ModuleType(name)
        package.__path__ = [str(ROOT / relative)]
        monkeypatch.setitem(sys.modules, name, package)

    accounts = {
        'one': SimpleNamespace(id=1, role='operator', is_active=True),
        'two': SimpleNamespace(id=2, role='operator', is_active=True),
        'admin': SimpleNamespace(id=3, role='admin', is_active=True),
        'viewer': SimpleNamespace(id=4, role='viewer', is_active=True),
        'inactive': SimpleNamespace(id=1, role='operator', is_active=False),
        'unknown-role': SimpleNamespace(id=1, role='owner', is_active=True),
        'bad-id': SimpleNamespace(id=True, role='operator', is_active=True),
    }
    security = HTTPBearer(auto_error=False)
    async def synthetic_identity_lookup(authorization):
        return accounts.get(authorization.removeprefix('Bearer '))
    auth_namespace = {
        'HTTPAuthorizationCredentials': HTTPAuthorizationCredentials,
        'AuthUser': SimpleNamespace, 'Depends': Depends, 'security': security,
        'HTTPException': HTTPException, 'auth_get_current_user': synthetic_identity_lookup,
    }
    exec(compile(ast.Module(body=[_LEGACY_AUTH_FUNCTION], type_ignores=[]),
                 '<source-extracted legacy require_auth>', 'exec'), auth_namespace)
    require_auth = auth_namespace['require_auth']

    calls = []
    async def get_labels(job_id: str):
        calls.append(('get-labels', job_id))
        return {'job_id': job_id, 'instances': []}
    async def put_labels(job_id: str, payload: dict = Body(...)):
        calls.append(('put-labels', job_id))
        return payload
    async def validate_file_size(*args):
        calls.append(('validate-upload',))

    class AsyncFile:
        def __init__(self, path, mode):
            self.path, self.mode = path, mode
        async def __aenter__(self):
            self.handle = self.path.open(self.mode)
            return self
        async def write(self, value):
            self.handle.write(value)
        async def __aexit__(self, *args):
            self.handle.close()

    uploads, outputs = tmp_path / 'synthetic_uploads', tmp_path / 'synthetic_results'
    uploads.mkdir()
    outputs.mkdir()
    cad = tmp_path / 'manufactured-cad.ply'
    cad.write_text('synthetic fixture, not physical geometry')
    part = SimpleNamespace(id='part-fixture', part_number='SYNTHETIC',
                           part_name='Synthetic', cad_file_path=str(cad),
                           default_tolerance_angle=1.0)
    def worker(*args):
        calls.append(('worker', args[0]))
        output_dir = args[7]
        report = output_dir / 'bend_inspection_report.json'
        report.write_text(json.dumps({'summary': {}, 'part_id': 'synthetic'}))
        table = output_dir / 'bend_inspection_table.txt'
        table.write_text('synthetic table')
        return {'ok': True, 'report_path': str(report), 'table_path': str(table)}

    legacy = SimpleNamespace(
        require_auth=require_auth, bend_inspection_results={}, OUTPUT_DIR=outputs,
        UPLOAD_DIR=uploads, logger=logging.getLogger('synthetic-bend-boundary'),
        validate_file_size=validate_file_size, MAX_MESH_FILE_SIZE=10000,
        _run_bend_worker_subprocess=worker,
        get_catalog=lambda: SimpleNamespace(get_part=lambda _: part, list_parts=lambda **_: []),
        _json_safe=lambda value: value, load_bend_runtime_config=lambda: None,
        load_cad_geometry=lambda *args, **kwargs: None,
        extract_cad_bend_specs=lambda **kwargs: [], aiofiles=SimpleNamespace(open=AsyncFile),
        asyncio=asyncio, shutil=shutil, uuid=uuid,
        get_bend_inspection_labels=get_labels, put_bend_inspection_labels=put_labels,
    )
    context = ModuleType('apps.api.legacy_context')
    context.legacy = legacy
    monkeypatch.setitem(sys.modules, context.__name__, context)
    readiness = ModuleType('apps.api.services.bend_readiness_service')
    readiness.load_bend_inspection_readiness = lambda: {'operating_mode': 'RESEARCH_ONLY', 'production_use_allowed': False}
    async def get_readiness():
        return readiness.load_bend_inspection_readiness()
    readiness.get_bend_inspection_readiness = get_readiness
    monkeypatch.setitem(sys.modules, readiness.__name__, readiness)
    service = _load('apps.api.services.bend_inspection_service', 'apps/api/services/bend_inspection_service.py', monkeypatch)
    routes = _load('apps.api.routes.bend_inspection', 'apps/api/routes/bend_inspection.py', monkeypatch)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, service=service, routes=routes,
                              legacy=legacy, calls=calls, accounts=accounts)


def _request(boundary, method, path, token=None, **kwargs):
    if token is not None:
        kwargs['headers'] = {'Authorization': f'Bearer {token}'}
    if method == 'PUT':
        kwargs.setdefault('json', {'instances': []})
    return boundary.client.request(method, path.format(job_id='bend_owned'), **kwargs)


@pytest.mark.parametrize('method,path', JOB_SURFACES + OTHER_SURFACES)
def test_every_bend_surface_rejects_unauthenticated_before_effects(boundary, method, path):
    response = _request(boundary, method, path)
    assert response.status_code == 401
    assert boundary.calls == []
    assert boundary.legacy.bend_inspection_results == {}


@pytest.mark.parametrize('method,path', JOB_SURFACES + OTHER_SURFACES)
def test_every_bend_surface_rejects_viewer(boundary, method, path):
    assert _request(boundary, method, path, 'viewer').status_code == 403
    assert boundary.calls == []


@pytest.mark.parametrize('method,path', JOB_SURFACES)
@pytest.mark.parametrize('token', ['two', 'admin'])
def test_other_operator_and_admin_cannot_access_owned_job(boundary, method, path, token, monkeypatch):
    boundary.legacy.bend_inspection_results['bend_owned'] = {'owner_user_id': 1, 'status': 'completed'}
    def no_storage(*args):
        raise AssertionError('Unauthorized request reached storage')
    monkeypatch.setattr(boundary.service, '_safe_read_json', no_storage)
    response = _request(boundary, method, path, token)
    assert response.status_code == 404
    assert boundary.calls == []
    assert 'bend_owned' in boundary.legacy.bend_inspection_results


@pytest.mark.parametrize('token,status', [('expired', 401), ('inactive', 403), ('unknown-role', 403), ('bad-id', 403)])
def test_current_account_state_and_identity_must_be_valid(boundary, token, status):
    assert _request(boundary, 'GET', '/api/bend-inspection/jobs', token).status_code == status


def test_query_credentials_are_not_authentication(boundary):
    assert boundary.client.get('/api/bend-inspection/jobs?authorization=Bearer%20one').status_code == 401


@pytest.mark.parametrize('owner', [None, True, '1', 2, -1])
def test_malformed_or_foreign_persisted_ownership_blocks_report_read(boundary, owner, monkeypatch):
    calls = []
    def metadata_only(path):
        calls.append(path.name)
        assert path.name == 'bend_inspection_meta.json'
        return {'owner_user_id': owner}
    monkeypatch.setattr(boundary.service, '_safe_read_json', metadata_only)
    assert _request(boundary, 'GET', '/api/bend-inspection/{job_id}', 'one').status_code == 404
    assert calls == ['bend_inspection_meta.json']


@pytest.mark.parametrize('job_id', ['../escape', '..', '.', 'not-bend', 'bend_a/b', 'bend_a\\b', 'bend_' + 'x' * 129])
def test_invalid_job_ids_fail_before_storage(boundary, job_id, monkeypatch):
    monkeypatch.setattr(boundary.service, '_safe_read_json', lambda *_: pytest.fail('Invalid ID touched storage'))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(boundary.service.require_bend_job_owner(job_id, boundary.accounts['one']))
    assert exc.value.status_code == 404


def _persisted(boundary, job_id, owner):
    folder = boundary.legacy.OUTPUT_DIR / job_id
    folder.mkdir()
    (folder / 'bend_inspection_meta.json').write_text(json.dumps({'owner_user_id': owner, 'created_at': '2026-01-01T00:00:00'}))
    (folder / 'bend_inspection_report.json').write_text(json.dumps({'part_id': job_id}))
    return folder


def test_list_returns_only_owned_memory_and_persisted_jobs(boundary, monkeypatch):
    boundary.legacy.bend_inspection_results.update({
        'bend_memory': {'owner_user_id': 1, 'status': 'completed'},
        'bend_foreign': {'owner_user_id': 2, 'status': 'completed'},
        'bend_legacy': {'status': 'completed'},
    })
    _persisted(boundary, 'bend_saved', 1)
    _persisted(boundary, 'bend_other_saved', 2)
    _persisted(boundary, 'bend_old_saved', None)
    safe_read = boundary.service._safe_read_json
    def owned_report_only(path):
        if path.name == 'bend_inspection_report.json':
            assert path.parent.name == 'bend_saved'
        return safe_read(path)
    monkeypatch.setattr(boundary.service, '_safe_read_json', owned_report_only)
    response = _request(boundary, 'GET', '/api/bend-inspection/jobs', 'one')
    assert response.status_code == 200
    assert {job['job_id'] for job in response.json()['jobs']} == {'bend_memory', 'bend_saved'}
    assert response.json()['total'] == 2


@pytest.mark.parametrize('method,path', [item for item in JOB_SURFACES if item[0] != 'DELETE'])
def test_owner_can_access_job_and_synthetic_artifacts(boundary, method, path):
    folder = _persisted(boundary, 'bend_owned', 1)
    for filename in ['bend_inspection_report.pdf', 'bend_overlay_manifest.json',
                     'bend_overlay_overview.png', 'issue.png', 'bender.png',
                     'aligned_scan.ply', 'reference_mesh.ply', 'bend_inspection_table.txt']:
        (folder / filename).write_text('synthetic bytes')
    response = _request(boundary, method, path, 'one')
    assert response.status_code == 200


@pytest.mark.parametrize('catalog', [False, True])
def test_upload_ownership_is_server_assigned_and_survives_completion(boundary, catalog):
    endpoint = '/api/bend-inspection/analyze-from-catalog' if catalog else '/api/bend-inspection/analyze'
    files = {'scan_file': ('fixture.ply', b'synthetic scan')}
    if not catalog:
        files['cad_file'] = ('fixture.ply', b'synthetic CAD')
    response = _request(boundary, 'POST', endpoint, 'one', files=files,
                        data={'part_id': 'synthetic', 'owner_user_id': '2'})
    assert response.status_code == 200, response.text
    job_id = response.json()['job_id']
    record = boundary.legacy.bend_inspection_results[job_id]
    assert record['owner_user_id'] == 1
    assert record['status'] == 'completed'
    metadata = json.loads((boundary.legacy.OUTPUT_DIR / job_id / 'bend_inspection_meta.json').read_text())
    assert metadata['owner_user_id'] == 1
    boundary.legacy.bend_inspection_results.clear()  # Simulated in-memory restart.
    assert _request(boundary, 'GET', f'/api/bend-inspection/{job_id}', 'one').status_code == 200
    assert _request(boundary, 'GET', f'/api/bend-inspection/{job_id}', 'two').status_code == 404
    assert _request(boundary, 'DELETE', f'/api/bend-inspection/{job_id}', 'two').status_code == 404
    assert _request(boundary, 'DELETE', f'/api/bend-inspection/{job_id}', 'one').status_code == 200


def test_owner_cannot_delete_running_job(boundary):
    boundary.legacy.bend_inspection_results['bend_owned'] = {'owner_user_id': 1, 'status': 'running'}
    assert _request(boundary, 'DELETE', '/api/bend-inspection/{job_id}', 'one').status_code == 409
    assert 'bend_owned' in boundary.legacy.bend_inspection_results


def test_unowned_legacy_memory_job_is_not_claimed_by_first_operator(boundary):
    boundary.legacy.bend_inspection_results['bend_owned'] = {'status': 'completed'}
    for token in ('one', 'admin'):
        assert _request(boundary, 'GET', '/api/bend-inspection/{job_id}', token).status_code == 404


def test_every_registered_job_route_has_ownership_dependency(boundary):
    observed = 0
    for route in boundary.routes.router.routes:
        if route.path.startswith(('/api/aligned-scan/', '/api/reference-mesh/')):
            observed += len(route.methods)
            assert any(dep.call is boundary.service.require_bend_job_owner_if_applicable for dep in route.dependant.dependencies)
            continue
        assert any(dep.call is boundary.service.require_bend_operator for dep in route.dependant.dependencies)
        if '{job_id}' in route.path:
            observed += len(route.methods)
            assert any(dep.call is boundary.service.require_bend_job_owner for dep in route.dependant.dependencies)
    assert observed == len(JOB_SURFACES)


@pytest.mark.parametrize('catalog', [False, True])
def test_upload_collision_does_not_overwrite_prior_user_job(boundary, catalog, monkeypatch):
    old_hex, new_hex = '1' * 32, '2' * 32
    old_dir = boundary.legacy.UPLOAD_DIR / f'bend_{old_hex}'
    old_dir.mkdir()
    sentinel = old_dir / 'scan.ply'
    sentinel.write_text('another operator synthetic input')
    values = iter([old_hex, new_hex])
    monkeypatch.setattr(boundary.service, 'uuid', SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=next(values))))
    files = {'scan_file': ('fixture.ply', b'new synthetic scan')}
    if not catalog:
        files['cad_file'] = ('fixture.ply', b'new synthetic CAD')
    endpoint = '/api/bend-inspection/analyze-from-catalog' if catalog else '/api/bend-inspection/analyze'
    response = _request(boundary, 'POST', endpoint, 'one', files=files, data={'part_id': 'synthetic'})
    assert response.status_code == 200, response.text
    assert response.json()['job_id'] == f'bend_{new_hex}'
    assert sentinel.read_text() == 'another operator synthetic input'


@pytest.mark.parametrize('existing', ['memory', 'results', 'uploads'])
def test_namespace_exhaustion_is_bounded_and_preserves_existing_job(boundary, existing, monkeypatch):
    value = '3' * 32
    job_id = f'bend_{value}'
    if existing == 'memory':
        boundary.legacy.bend_inspection_results[job_id] = {'owner_user_id': 2}
    else:
        root = boundary.legacy.OUTPUT_DIR if existing == 'results' else boundary.legacy.UPLOAD_DIR
        (root / job_id).mkdir()
    calls = []
    def collision():
        calls.append(True)
        return SimpleNamespace(hex=value)
    monkeypatch.setattr(boundary.service, 'uuid', SimpleNamespace(uuid4=collision))
    with pytest.raises(HTTPException) as exc:
        boundary.service._reserve_bend_upload_dir()
    assert exc.value.status_code == 503
    assert len(calls) == 5


SHARED_SURFACES = [
    ('GET', '/api/progress/{job_id}'),
    ('GET', '/api/progress/{job_id}/stream'),
    ('GET', '/api/result/{job_id}'),
    ('GET', '/api/deviations/{job_id}'),
    ('GET', '/api/aligned-scan/{job_id}.ply'),
    ('GET', '/api/reference-mesh/{job_id}.ply'),
    ('GET', '/api/jobs/{job_id}'),
    ('DELETE', '/api/jobs/{job_id}'),
    ('GET', '/api/jobs/{job_id}/dimensions'),
    ('GET', '/api/jobs/{job_id}/bends'),
    ('GET', '/api/jobs/{job_id}/correlations'),
    ('GET', '/api/jobs/{job_id}/enhanced-analysis'),
    ('GET', '/api/download/{job_id}/pdf'),
    ('GET', '/api/download/{job_id}/json'),
    ('GET', '/api/heatmap/{job_id}/bend_inspection_report.json'),
    ('GET', '/api/heatmap/{job_id}/bend_inspection_meta.json'),
    ('GET', '/api/heatmaps/{job_id}'),
    ('GET', '/api/files/{job_id}/cad.ply'),
    ('GET', '/api/files/{job_id}/scan.ply'),
]


def _assembled(boundary, monkeypatch, bend_first):
    """Load actual route definitions with all legacy implementation stubs wired."""
    jobs = ModuleType('apps.api.services.jobs_service')
    async def job_endpoint(job_id: str):
        boundary.calls.append(('shared-job', job_id))
        return {'job_id': job_id}
    async def no_job_endpoint():
        return {'jobs': []}
    for name in ['get_progress', 'stream_progress', 'get_result', 'get_deviations',
                 'get_job_details', 'delete_job', 'get_job_dimensions', 'get_job_bends',
                 'get_job_correlations', 'get_enhanced_analysis']:
        setattr(jobs, name, job_endpoint)
    jobs.list_jobs = jobs.get_stats = no_job_endpoint
    jobs.get_aligned_scan = boundary.service.get_aligned_scan
    jobs.get_reference_mesh = boundary.service.get_reference_mesh
    monkeypatch.setitem(sys.modules, jobs.__name__, jobs)
    reports = ModuleType('apps.api.services.reports_service')
    async def artifact(job_id: str, filename: str):
        boundary.calls.append(('shared-artifact', job_id, filename))
        return FileResponse(boundary.legacy.OUTPUT_DIR / job_id / filename)
    async def uploaded(job_id: str, filename: str):
        boundary.calls.append(('shared-upload', job_id, filename))
        return FileResponse(boundary.legacy.UPLOAD_DIR / job_id / filename)
    reports.download_pdf = reports.download_json = reports.list_heatmaps = job_endpoint
    reports.get_heatmap = artifact
    reports.get_uploaded_file = uploaded
    monkeypatch.setitem(sys.modules, reports.__name__, reports)
    jobs_router = _load('apps.api.routes.jobs', 'apps/api/routes/jobs.py', monkeypatch).router
    reports_router = _load('apps.api.routes.reports', 'apps/api/routes/reports.py', monkeypatch).router
    app = FastAPI()
    routers = [boundary.routes.router, jobs_router, reports_router] if bend_first else [jobs_router, reports_router, boundary.routes.router]
    for router in routers:
        app.include_router(router)
    return app


def _shared_files(boundary, job_id='bend_owned'):
    folder = _persisted(boundary, job_id, 1)
    for name in ['aligned_scan.ply', 'reference_mesh.ply']:
        (folder / name).write_text('synthetic output')
    uploads = boundary.legacy.UPLOAD_DIR / job_id
    uploads.mkdir()
    (uploads / 'scan.ply').write_text('synthetic raw scan')
    (uploads / 'cad.ply').write_text('synthetic raw CAD')


@pytest.mark.parametrize('bend_first', [False, True])
@pytest.mark.parametrize('method,path', SHARED_SURFACES)
@pytest.mark.parametrize('token,expected', [(None, 401), ('two', 404), ('one', 200)])
def test_assembled_router_orders_protect_every_shared_bend_alias(boundary, monkeypatch, bend_first, method, path, token, expected):
    _shared_files(boundary)
    app = _assembled(boundary, monkeypatch, bend_first)
    with TestClient(app) as client:
        headers = {'Authorization': f'Bearer {token}'} if token else {}
        response = client.request(method, path.format(job_id='bend_owned'), headers=headers)
    assert response.status_code == expected, response.text
    if token != 'one':
        assert boundary.calls == []


@pytest.mark.parametrize('bend_first', [False, True])
@pytest.mark.parametrize('method,path', SHARED_SURFACES)
def test_non_bend_shared_route_behavior_is_preserved(boundary, monkeypatch, bend_first, method, path):
    _shared_files(boundary, 'ordinary-qc-job')
    app = _assembled(boundary, monkeypatch, bend_first)
    with TestClient(app) as client:
        response = client.request(method, path.format(job_id='ordinary-qc-job'))
    assert response.status_code == 200, response.text


@pytest.mark.parametrize('path', [
    '/api/files/ordinary-job//absolute/path.ply',
    '/api/files/bend_owned//absolute/path.ply',
    '/api/files/bend_owned/nested/scan.ply',
])
def test_shared_upload_route_cannot_escape_authorized_bend_directory(boundary, monkeypatch, path):
    _shared_files(boundary)
    app = _assembled(boundary, monkeypatch, False)
    with TestClient(app) as client:
        response = client.get(path, headers={'Authorization': 'Bearer one'})
    assert response.status_code == 400
    assert boundary.calls == []


def test_case_insensitive_bend_alias_still_requires_auth(boundary, monkeypatch):
    app = _assembled(boundary, monkeypatch, False)
    with TestClient(app) as client:
        response = client.get('/api/heatmap/BEND_owned/bend_inspection_report.json')
    assert response.status_code == 401
    assert boundary.calls == []


def test_generic_delete_cannot_bypass_running_job_guard(boundary, monkeypatch):
    boundary.legacy.bend_inspection_results['bend_owned'] = {'owner_user_id': 1, 'status': 'running'}
    app = _assembled(boundary, monkeypatch, False)
    with TestClient(app) as client:
        response = client.delete('/api/jobs/bend_owned', headers={'Authorization': 'Bearer one'})
    assert response.status_code == 409
    assert boundary.calls == []


def test_live_foreign_ownership_takes_precedence_over_stale_persisted_owner(boundary, monkeypatch):
    _persisted(boundary, 'bend_owned', 1)
    boundary.legacy.bend_inspection_results['bend_owned'] = {'owner_user_id': 2, 'status': 'running'}
    monkeypatch.setattr(boundary.service, '_safe_read_json', lambda *_: pytest.fail('Foreign live job read stale metadata'))
    response = _request(boundary, 'GET', '/api/bend-inspection/jobs', 'one')
    assert response.json() == {'jobs': [], 'total': 0}


@pytest.mark.parametrize('bend_first', [False, True])
@pytest.mark.parametrize('method,path', SHARED_SURFACES + [
    ('HEAD', '/api/aligned-scan/{job_id}.ply'),
    ('HEAD', '/api/reference-mesh/{job_id}.ply'),
])
@pytest.mark.parametrize('job_id', ['temp_secret', 'TeMp_secret'])
def test_temporary_namespace_is_never_served_even_to_operator(boundary, monkeypatch, bend_first, method, path, job_id):
    async def no_credentials(*args, **kwargs):
        pytest.fail('Temporary namespace reached credential lookup')
    monkeypatch.setattr(boundary.legacy, 'require_auth', no_credentials)
    app = _assembled(boundary, monkeypatch, bend_first)
    with TestClient(app) as client:
        response = client.request(method, path.format(job_id=job_id),
                                  headers={'Authorization': 'Bearer one'})
    assert response.status_code == 404
    assert boundary.calls == []


@pytest.mark.parametrize('job_id', ['temp_secret', 'TEMP_secret', 'temp_../escape'])
def test_temporary_denial_precedes_request_path_or_credential_access(boundary, job_id):
    # No request attributes are available: the namespace is rejected first.
    with pytest.raises(HTTPException) as exc:
        asyncio.run(boundary.service.require_bend_job_owner_if_applicable(job_id, object()))
    assert exc.value.status_code == 404
