from __future__ import annotations

from typing import Optional

from fastapi import Query

from apps.api.legacy_context import legacy

get_progress = legacy.get_progress
stream_progress = legacy.stream_progress
get_result = legacy.get_result
get_deviations = legacy.get_deviations
get_aligned_scan = legacy.get_aligned_scan
get_reference_mesh = legacy.get_reference_mesh
get_job_details = legacy.get_job_details
delete_job = legacy.delete_job
get_job_dimensions = legacy.get_job_dimensions
get_job_bends = legacy.get_job_bends
get_job_correlations = legacy.get_job_correlations
get_enhanced_analysis = legacy.get_enhanced_analysis
get_stats = legacy.get_stats


async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status: pending, running, completed, failed"),
    part_id: Optional[str] = Query(None, description="Filter by part ID"),
    limit: int = Query(legacy.DEFAULT_PAGINATION_LIMIT, ge=1, le=legacy.MAX_PAGINATION_LIMIT, description="Max jobs to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """List general QC jobs without revealing the private bend namespace.

    Filter in both database queries before counting/pagination; post-filtering
    would disclose private counts and let private rows consume public pages.
    """
    limit, offset = legacy.validate_pagination(limit, offset)
    filters = {"status": status, "part_id": part_id, "excluded_job_prefix": "bend_"}
    jobs_list = legacy.db.list_jobs(**filters, limit=limit, offset=offset)
    total_count = legacy.db.count_jobs(**filters)
    jobs_data = [
        legacy._json_safe({
            "job_id": job.job_id,
            "status": job.status,
            "part_id": job.part_id,
            "part_name": job.part_name,
            "material": job.material,
            "tolerance": job.tolerance,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            "progress": job.progress,
        })
        for job in jobs_list
    ]
    return legacy._json_safe({
        "jobs": jobs_data,
        "total": total_count,
        "count": len(jobs_data),
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(jobs_data) < total_count,
    })
