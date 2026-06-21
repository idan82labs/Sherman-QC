from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from domains.bend.services.count_service import load_unary_models, score_case_payload
from domains.bend.services.runtime_semantics import is_explicit_observability_evidence


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def latest_unary_model_path(latest_pointer_path: Optional[Path] = None) -> Optional[Path]:
    latest = latest_pointer_path or (PROJECT_ROOT / 'output' / 'unary_models' / 'latest.json')
    if latest.suffix == '.joblib':
        return latest if latest.exists() else None
    if not latest.exists():
        return None
    try:
        payload = json.loads(latest.read_text(encoding='utf-8'))
    except Exception:
        return None
    candidate = payload.get('model_path') or payload.get('path')
    if candidate:
        model_path = Path(str(candidate))
        if model_path.is_dir():
            model_path = model_path / 'bend_unary_models.joblib'
        return model_path if model_path.exists() else None
    latest_dir = payload.get('latest_bootstrap_dir')
    if not latest_dir:
        return None
    model_path = Path(str(latest_dir)) / 'bend_unary_models.joblib'
    return model_path if model_path.exists() else None


def promote_structured_count_reporting(report_dict: Dict[str, Any], latest_pointer_path: Optional[Path] = None) -> Dict[str, Any]:
    model_path = latest_unary_model_path(latest_pointer_path)
    if model_path is None:
        return report_dict
    try:
        bundle = load_unary_models(model_path)
        annotated = score_case_payload(report_dict, bundle)
    except Exception:
        return report_dict

    structured = (annotated.get('structured_context') or {}).get('count_posterior') or {}
    median_completed = structured.get('median_completed_bends')
    if median_completed is None:
        return annotated

    summary = annotated.setdefault('summary', {})
    matches = list(annotated.get('matches') or [])
    expected_bends = int(summary.get('expected_bends') or summary.get('total_bends') or 0)
    direct_completed = int(
        summary.get(
            'countable_completed_bends',
            summary.get('completed_bends', summary.get('detected', 0)),
        ) or 0
    )
    direct_remaining = int(
        summary.get(
            'countable_remaining_bends',
            summary.get('remaining_bends', max(0, expected_bends - direct_completed)),
        ) or 0
    )
    median_completed = int(median_completed)
    def _match_countable(match: Dict[str, Any]) -> bool:
        if 'countable_in_regression' in match:
            return bool(match.get('countable_in_regression', True))
        return bool((match.get('cad_bend') or {}).get('countable_in_regression', True))

    countable_matches = [match for match in matches if _match_countable(match)]
    has_process_features = any(
        not _match_countable(match)
        for match in matches
    )
    observed_countable_ceiling = sum(
        1
        for match in countable_matches
        if is_explicit_observability_evidence(
            match.get('observability_state'),
            physical_completion_state=match.get('physical_completion_state'),
            status=match.get('status'),
        )
    )
    if has_process_features and observed_countable_ceiling > 0:
        median_completed = min(median_completed, observed_countable_ceiling)
    if has_process_features and direct_completed < expected_bends:
        median_completed = min(median_completed, max(direct_completed, observed_countable_ceiling))

    structured['median_completed_bends'] = median_completed
    summary['match_evidence_completed_bends'] = direct_completed
    summary['match_evidence_remaining_bends'] = direct_remaining
    summary['structured_completed_bends'] = median_completed
    summary['structured_remaining_bends'] = max(0, expected_bends - min(expected_bends, median_completed))
    summary['structured_count_source'] = 'posterior_median'
    summary['structured_count_mean'] = structured.get('mean_completed_bends')
    summary['structured_count_map'] = structured.get('map_completed_bends')
    summary['structured_count_delta_vs_match'] = median_completed - direct_completed

    brief = annotated.setdefault('operator_brief', {})
    headline = str(brief.get('headline') or '').strip()
    structured_suffix = (
        f' | Count model: {median_completed}/{expected_bends} complete'
        if expected_bends > 0
        else f' | Count model: {median_completed} complete'
    )
    evidence_suffix = f' | Direct evidence: {direct_completed} explicit bends'
    if structured_suffix not in headline:
        brief['headline'] = f"{headline}{structured_suffix}{evidence_suffix}".strip(' |')
    return annotated
