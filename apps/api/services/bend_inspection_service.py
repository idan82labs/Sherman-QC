from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from apps.api.legacy_context import legacy


bend_inspection_results = legacy.bend_inspection_results
OUTPUT_DIR = legacy.OUTPUT_DIR
UPLOAD_DIR = legacy.UPLOAD_DIR
logger = legacy.logger
validate_file_size = legacy.validate_file_size
MAX_MESH_FILE_SIZE = legacy.MAX_MESH_FILE_SIZE
_run_bend_worker_subprocess = legacy._run_bend_worker_subprocess
get_catalog = legacy.get_catalog
_json_safe = legacy._json_safe
load_bend_runtime_config = legacy.load_bend_runtime_config
load_expected_bend_overrides = legacy.load_expected_bend_overrides
load_cad_geometry = legacy.load_cad_geometry
extract_cad_bend_specs = legacy.extract_cad_bend_specs
aiofiles = legacy.aiofiles
asyncio = legacy.asyncio
shutil = legacy.shutil
uuid = legacy.uuid


def _resolve_bend_artifact_urls(artifacts: Optional[Dict[str, Any]], job_id: str) -> Dict[str, Any]:
    if not isinstance(artifacts, dict):
        return {}
    resolved: Dict[str, Any] = {}
    for key, value in artifacts.items():
        if isinstance(value, str):
            resolved[key] = value.replace('{job_id}', job_id)
        elif isinstance(value, list):
            resolved[key] = [item.replace('{job_id}', job_id) if isinstance(item, str) else item for item in value]
        else:
            resolved[key] = value
    return resolved


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _build_persisted_bend_job(job_id: str, include_report: bool = False) -> Optional[Dict[str, Any]]:
    output_dir = OUTPUT_DIR / job_id
    report_path = output_dir / 'bend_inspection_report.json'
    if not report_path.exists():
        return None

    report = _safe_read_json(report_path) or {}
    meta = _safe_read_json(output_dir / 'bend_inspection_meta.json') or {}

    report_part_id = str(report.get('part_id') or '').strip()
    part_id = str(meta.get('part_id') or report_part_id or job_id).strip()
    part_name = str(meta.get('part_name') or part_id).strip()

    file_time = datetime.fromtimestamp(report_path.stat().st_mtime).isoformat()
    created_at = meta.get('created_at') or meta.get('started_at') or file_time
    completed_at = meta.get('completed_at') or file_time
    processing_time_ms = meta.get('processing_time_ms')
    if processing_time_ms is None and isinstance(report.get('processing_time_ms'), (int, float)):
        processing_time_ms = report.get('processing_time_ms')

    payload: Dict[str, Any] = {
        'job_id': job_id,
        'status': str(meta.get('status') or 'completed'),
        'progress': int(meta.get('progress') or 100),
        'part_id': part_id,
        'part_name': part_name,
        'created_at': created_at,
        'completed_at': completed_at,
        'processing_time_ms': processing_time_ms,
    }
    if isinstance(meta.get('seed_case'), dict):
        payload['seed_case'] = meta.get('seed_case')
    artifacts = _resolve_bend_artifact_urls(meta.get('artifacts'), job_id)
    ref_mesh = output_dir / 'reference_mesh.ply'
    if ref_mesh.exists() and 'reference_mesh_url' not in artifacts:
        artifacts['reference_mesh_url'] = f'/api/reference-mesh/{job_id}.ply'
    if artifacts:
        payload['artifacts'] = artifacts
    if include_report:
        payload['report'] = report
    return payload


def _list_persisted_bend_jobs(status: Optional[str] = None, include_report: bool = False) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for report_path in OUTPUT_DIR.glob('bend_*/bend_inspection_report.json'):
        job_id = report_path.parent.name
        job = _build_persisted_bend_job(job_id, include_report=include_report)
        if not job:
            continue
        if status and job.get('status') != status:
            continue
        jobs.append(job)
    return jobs


async def get_aligned_scan(job_id: str):
    scan_path = OUTPUT_DIR / job_id / 'aligned_scan.ply'
    if not scan_path.exists():
        raise HTTPException(404, 'Aligned scan not available for this job')
    return FileResponse(str(scan_path), media_type='application/x-ply', filename='aligned_scan.ply')


async def get_reference_mesh(job_id: str):
    mesh_path = OUTPUT_DIR / job_id / 'reference_mesh.ply'
    if not mesh_path.exists():
        raise HTTPException(404, 'Reference mesh not available for this job')
    return FileResponse(str(mesh_path), media_type='application/x-ply', filename='reference_mesh.ply')


async def run_bend_inspection(
    background_tasks: BackgroundTasks,
    cad_file: UploadFile = File(..., description='CAD reference file (PLY, STL, STEP)'),
    scan_file: UploadFile = File(..., description='Scan point cloud file (PLY, STL, PCD)'),
    part_id: str = Form('PART_001'),
    part_name: str = Form('Unnamed Part'),
    default_tolerance_angle: float = Form(1.0),
    default_tolerance_radius: float = Form(0.5),
):
    cad_ext = Path(cad_file.filename).suffix.lower()
    scan_ext = Path(scan_file.filename).suffix.lower()

    valid_cad_exts = ['.stl', '.ply', '.obj', '.step', '.stp']
    valid_scan_exts = ['.stl', '.ply', '.obj', '.pcd']

    if cad_ext not in valid_cad_exts:
        raise HTTPException(400, f'CAD file must be STL/PLY/OBJ/STEP, got {cad_ext}')
    if scan_ext not in valid_scan_exts:
        raise HTTPException(400, f'Scan file must be STL/PLY/OBJ/PCD, got {scan_ext}')

    await validate_file_size(cad_file, MAX_MESH_FILE_SIZE, 'CAD')
    await validate_file_size(scan_file, MAX_MESH_FILE_SIZE, 'Scan')

    job_id = f'bend_{uuid.uuid4().hex[:8]}'
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    cad_path = job_dir / f'cad{cad_ext}'
    scan_path = job_dir / f'scan{scan_ext}'

    cad_content = await cad_file.read()
    async with aiofiles.open(cad_path, 'wb') as handle:
        await handle.write(cad_content)

    scan_content = await scan_file.read()
    async with aiofiles.open(scan_path, 'wb') as handle:
        await handle.write(scan_content)

    bend_inspection_results[job_id] = {
        'status': 'running',
        'progress': 0,
        'part_id': part_id,
        'part_name': part_name,
        'created_at': datetime.now().isoformat(),
    }

    background_tasks.add_task(
        run_bend_inspection_task,
        job_id,
        str(cad_path),
        str(scan_path),
        part_id,
        part_name,
        default_tolerance_angle,
        default_tolerance_radius,
    )

    return {'job_id': job_id, 'status': 'started', 'message': 'Bend inspection started'}


async def run_bend_inspection_task(
    job_id: str,
    cad_path: str,
    scan_path: str,
    part_id: str,
    part_name: str,
    tolerance_angle: float,
    tolerance_radius: float,
):
    import time

    start_time = time.time()
    result_store = bend_inspection_results.get(job_id)
    if not result_store:
        return

    try:
        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(exist_ok=True)

        def progress_callback(update: Any):
            try:
                result_store['progress'] = int(max(result_store.get('progress', 0), update.progress))
                result_store['stage'] = update.message or update.stage
            except Exception:
                pass

        result_store['progress'] = 2
        result_store['stage'] = 'Launching isolated bend worker...'

        worker_result = await asyncio.to_thread(
            _run_bend_worker_subprocess,
            job_id,
            cad_path,
            scan_path,
            part_id,
            part_name,
            tolerance_angle,
            tolerance_radius,
            output_dir,
            progress_callback,
        )
        if not worker_result.get('ok'):
            raise RuntimeError(worker_result.get('error') or 'Bend worker failed')

        report_path = Path(str(worker_result['report_path']))
        table_path = Path(str(worker_result['table_path']))
        if not report_path.exists():
            raise FileNotFoundError(f'Bend worker produced no report file: {report_path}')

        with report_path.open('r', encoding='utf-8') as handle:
            report_dict = _json_safe(json.load(handle))
        table_text = table_path.read_text(encoding='utf-8') if table_path.exists() else ''
        artifacts_payload = _resolve_bend_artifact_urls(worker_result.get('artifacts'), job_id)
        processing_time_ms = float(worker_result.get('processing_time_ms') or ((time.time() - start_time) * 1000.0))
        pipeline_details = _json_safe(worker_result.get('pipeline_details'))
        runtime_config = _json_safe(worker_result.get('runtime_config'))

        completed_at = datetime.now().isoformat()
        metadata = {
            'job_id': job_id,
            'status': 'completed',
            'progress': 100,
            'part_id': part_id,
            'part_name': part_name,
            'created_at': result_store.get('created_at'),
            'completed_at': completed_at,
            'processing_time_ms': processing_time_ms,
            'cad_path': cad_path,
            'scan_path': scan_path,
            'artifacts': artifacts_payload,
        }
        meta_path = output_dir / 'bend_inspection_meta.json'
        with open(meta_path, 'w', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2)

        result_store['status'] = 'completed'
        result_store['progress'] = 100
        result_store['stage'] = 'Complete'
        result_store['report'] = report_dict
        result_store['table'] = table_text
        result_store['completed_at'] = completed_at
        result_store['processing_time_ms'] = processing_time_ms
        result_store['pipeline_details'] = pipeline_details
        result_store['runtime_config'] = runtime_config
        if artifacts_payload:
            result_store['artifacts'] = artifacts_payload

        logger.info(
            'Bend inspection completed for job %s: %s/%s bends detected',
            job_id,
            report_dict.get('summary', {}).get('completed_bends', report_dict.get('summary', {}).get('detected', '?')),
            report_dict.get('summary', {}).get('expected_bends', report_dict.get('summary', {}).get('total_bends', '?')),
        )
    except Exception as exc:
        import traceback

        logger.error('Bend inspection failed for job %s: %s', job_id, exc)
        logger.error(traceback.format_exc())
        result_store['status'] = 'failed'
        result_store['error'] = str(exc)


async def list_bend_inspection_jobs(status: Optional[str] = Query(None, description='Filter by status'), limit: int = Query(20, ge=1, le=100)):
    jobs_by_id: Dict[str, Dict[str, Any]] = {}

    for job_id, result in bend_inspection_results.items():
        if status and result.get('status') != status:
            continue
        jobs_by_id[job_id] = {
            'job_id': job_id,
            'part_id': result.get('part_id'),
            'part_name': result.get('part_name'),
            'status': result.get('status'),
            'progress': result.get('progress'),
            'created_at': result.get('created_at'),
            'completed_at': result.get('completed_at'),
            'processing_time_ms': result.get('processing_time_ms'),
            'report': result.get('report'),
            'artifacts': result.get('artifacts'),
        }

    persisted = _list_persisted_bend_jobs(status=status, include_report=True)
    for job in persisted:
        current = jobs_by_id.get(job['job_id'])
        if current and current.get('status') == 'running':
            continue
        if current and current.get('status') == 'completed':
            if not isinstance(current.get('report'), dict) and isinstance(job.get('report'), dict):
                current['report'] = job['report']
            continue
        jobs_by_id[job['job_id']] = job

    jobs = list(jobs_by_id.values())

    def _sort_key(job: Dict[str, Any]) -> datetime:
        created = _parse_iso_datetime(job.get('created_at'))
        completed = _parse_iso_datetime(job.get('completed_at'))
        return created or completed or datetime.min

    jobs.sort(key=_sort_key, reverse=True)
    return {'jobs': jobs[:limit], 'total': len(jobs)}


async def get_bend_inspection_result(job_id: str):
    result = bend_inspection_results.get(job_id)
    if not result:
        persisted = _build_persisted_bend_job(job_id, include_report=True)
        if persisted:
            table_path = OUTPUT_DIR / job_id / 'bend_inspection_table.txt'
            if table_path.exists():
                with open(table_path, 'r', encoding='utf-8') as handle:
                    persisted['table'] = handle.read()
            return persisted
        raise HTTPException(404, 'Bend inspection job not found')

    return {
        'job_id': job_id,
        'status': result.get('status'),
        'progress': result.get('progress'),
        'stage': result.get('stage'),
        'part_id': result.get('part_id'),
        'part_name': result.get('part_name'),
        'created_at': result.get('created_at'),
        'completed_at': result.get('completed_at'),
        'report': result.get('report'),
        'table': result.get('table'),
        'pipeline_details': result.get('pipeline_details'),
        'runtime_config': result.get('runtime_config'),
        'artifacts': result.get('artifacts'),
        'error': result.get('error'),
        'processing_time_ms': result.get('processing_time_ms'),
    }


async def download_bend_inspection_pdf(job_id: str):
    pdf_path = OUTPUT_DIR / job_id / 'bend_inspection_report.pdf'
    if not pdf_path.exists():
        raise HTTPException(404, 'Bend inspection PDF not available')
    return FileResponse(str(pdf_path), media_type='application/pdf', filename=f'Bend_Inspection_{job_id}.pdf')


async def get_bend_overlay_manifest(job_id: str):
    manifest_path = OUTPUT_DIR / job_id / 'bend_overlay_manifest.json'
    if not manifest_path.exists():
        raise HTTPException(404, 'Bend overlay manifest not available')
    return FileResponse(str(manifest_path), media_type='application/json', filename='bend_overlay_manifest.json')


async def get_bend_overlay_overview(job_id: str):
    overview_path = OUTPUT_DIR / job_id / 'bend_overlay_overview.png'
    if not overview_path.exists():
        raise HTTPException(404, 'Bend overlay overview not available')
    return FileResponse(str(overview_path), media_type='image/png', filename='bend_overlay_overview.png')


async def get_bend_overlay_issue(job_id: str, filename: str):
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(400, 'Invalid bend issue filename')
    issue_path = OUTPUT_DIR / job_id / safe_name
    if not issue_path.exists():
        raise HTTPException(404, 'Bend issue image not available')
    return FileResponse(str(issue_path), media_type='image/png', filename=safe_name)


async def get_bend_overlay_file(job_id: str, filename: str):
    """Serve any overlay artifact (bender view PNGs, etc.) from the job output directory."""
    safe_name = Path(filename).name
    if safe_name != filename or '..' in filename:
        raise HTTPException(400, 'Invalid filename')
    file_path = OUTPUT_DIR / job_id / safe_name
    if not file_path.exists():
        raise HTTPException(404, 'Overlay file not available')
    media_type = 'image/png' if safe_name.endswith('.png') else 'application/json'
    return FileResponse(str(file_path), media_type=media_type, filename=safe_name)


async def get_bend_inspection_table(job_id: str):
    result = bend_inspection_results.get(job_id)
    if result and result.get('table'):
        return {'table': result['table']}

    table_path = OUTPUT_DIR / job_id / 'bend_inspection_table.txt'
    if table_path.exists():
        with open(table_path, 'r', encoding='utf-8') as handle:
            return {'table': handle.read()}

    raise HTTPException(404, 'Bend inspection results not found')


async def extract_cad_bends(
    cad_file: UploadFile = File(..., description='CAD file to extract bends from'),
    part_id: str = Form('PART_001'),
    default_tolerance_angle: float = Form(1.0),
    default_tolerance_radius: float = Form(0.5),
):
    cad_ext = Path(cad_file.filename).suffix.lower()
    valid_exts = ['.stl', '.ply', '.obj', '.step', '.stp']
    if cad_ext not in valid_exts:
        raise HTTPException(400, f'CAD file must be STL/PLY/OBJ/STEP, got {cad_ext}')

    await validate_file_size(cad_file, MAX_MESH_FILE_SIZE, 'CAD')
    temp_dir = UPLOAD_DIR / f'temp_{uuid.uuid4().hex[:8]}'
    temp_dir.mkdir(exist_ok=True)

    try:
        runtime_cfg = load_bend_runtime_config()
        cad_path = temp_dir / f'cad{cad_ext}'
        cad_content = await cad_file.read()
        async with aiofiles.open(cad_path, 'wb') as handle:
            await handle.write(cad_content)

        cad_vertices, cad_triangles, cad_source = load_cad_geometry(
            str(cad_path),
            cad_import_deflection=runtime_cfg.cad_import_deflection,
        )
        bends = extract_cad_bend_specs(
            cad_vertices=cad_vertices,
            cad_triangles=cad_triangles,
            tolerance_angle=default_tolerance_angle,
            tolerance_radius=default_tolerance_radius,
            cad_extractor_kwargs=runtime_cfg.cad_extractor_kwargs,
            cad_detector_kwargs=runtime_cfg.cad_detector_kwargs,
            cad_detect_call_kwargs=runtime_cfg.cad_detect_call_kwargs,
        )
        expected_overrides = load_expected_bend_overrides()
        expected_bend_count = int(expected_overrides.get(part_id, len(bends)))
        if expected_bend_count <= 0:
            expected_bend_count = int(len(bends))

        return {
            'part_id': part_id,
            'cad_source': cad_source,
            'total_bends': len(bends),
            'expected_bend_count': expected_bend_count,
            'bends': [bend.to_dict() for bend in bends],
            'message': f'Extracted {len(bends)} bend specifications from CAD',
        }
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


async def analyze_from_catalog(
    background_tasks: BackgroundTasks,
    part_id: str = Form(..., description='Part ID from catalog'),
    scan_file: UploadFile = File(..., description='Scan file to analyze'),
):
    catalog = get_catalog()
    part = catalog.get_part(part_id)
    if not part:
        raise HTTPException(404, f'Part not found: {part_id}')
    if not part.cad_file_path or not Path(part.cad_file_path).exists():
        raise HTTPException(400, f'Part {part.part_number} has no CAD file uploaded')

    scan_ext = Path(scan_file.filename).suffix.lower()
    valid_exts = ['.stl', '.ply', '.obj', '.pcd']
    if scan_ext not in valid_exts:
        raise HTTPException(400, f'Scan file must be STL/PLY/OBJ/PCD, got {scan_ext}')

    await validate_file_size(scan_file, MAX_MESH_FILE_SIZE, 'Scan')

    job_id = f'bend_{uuid.uuid4().hex[:8]}'
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    try:
        scan_path = job_dir / f'scan{scan_ext}'
        scan_content = await scan_file.read()
        async with aiofiles.open(scan_path, 'wb') as handle:
            await handle.write(scan_content)

        bend_inspection_results[job_id] = {
            'status': 'running',
            'progress': 0,
            'part_id': part.id,
            'part_name': part.part_name or part.part_number,
            'created_at': datetime.now().isoformat(),
        }

        background_tasks.add_task(
            run_bend_inspection_task,
            job_id,
            str(part.cad_file_path),
            str(scan_path),
            part.id,
            part.part_name or part.part_number,
            part.default_tolerance_angle,
            0.5,
        )

        return {
            'job_id': job_id,
            'status': 'started',
            'message': 'Catalog-based bend inspection started',
            'part_id': part.id,
            'part_number': part.part_number,
            'part_name': part.part_name,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error('Error in catalog analysis: %s', exc)
        if job_dir.exists():
            shutil.rmtree(job_dir)
        bend_inspection_results.pop(job_id, None)
        raise HTTPException(500, f'Analysis failed: {str(exc)}')


async def get_parts_with_cad():
    catalog = get_catalog()
    parts = catalog.list_parts(limit=1000, offset=0)

    parts_with_cad = []
    for part in parts:
        if part.cad_file_path and Path(part.cad_file_path).exists():
            bend_specs = catalog.get_bend_specs_for_part(part.id)
            parts_with_cad.append({
                'id': part.id,
                'part_number': part.part_number,
                'part_name': part.part_name,
                'name': part.part_name,
                'customer': part.customer,
                'has_bend_specs': len(bend_specs) > 0,
            })

    return {'parts': parts_with_cad, 'total': len(parts_with_cad)}


async def delete_bend_inspection_job(job_id: str):
    if job_id in bend_inspection_results:
        del bend_inspection_results[job_id]

    job_dir = UPLOAD_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)

    output_dir = OUTPUT_DIR / job_id
    if output_dir.exists():
        shutil.rmtree(output_dir)

    return {'message': f'Bend inspection job {job_id} deleted'}
