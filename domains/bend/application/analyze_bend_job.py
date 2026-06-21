from __future__ import annotations

from typing import Any, Dict

from domains.bend.mappers.ui_contracts import map_bend_report_to_contract
from domains.bend.services.runtime_service import run_progressive_bend_inspection


def analyze_bend_job(*, cad_path: str, scan_path: str, part_id: str, tolerance_angle: float, tolerance_radius: float, runtime_config: Any):
    report, details = run_progressive_bend_inspection(
        cad_path=cad_path,
        scan_path=scan_path,
        part_id=part_id,
        tolerance_angle=tolerance_angle,
        tolerance_radius=tolerance_radius,
        runtime_config=runtime_config,
    )
    contract = map_bend_report_to_contract(part_id, report.to_dict())
    return report, details, contract
