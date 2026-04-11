from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from domains.bend.services.runtime_semantics import normalize_observability_state


@dataclass
class UnaryExample:
    head: str
    label: int
    weight: float
    label_source: str
    part_key: str
    scan_name: str
    bend_id: str
    features: Dict[str, Any]


@dataclass
class UnaryHeadSummary:
    name: str
    sample_count: int
    positive_count: int
    negative_count: int
    label_sources: Dict[str, int]
    features: List[str]
    calibration_method: str
    holdout_metrics: Dict[str, float]
    feature_importances: List[Tuple[str, float]]


@dataclass
class UnaryTrainingSummary:
    source_dir: str
    case_count: int
    visibility_head: Optional[UnaryHeadSummary]
    state_head: Optional[UnaryHeadSummary]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _candidate_summary(match: Dict[str, Any]) -> Dict[str, Any]:
    context = match.get("measurement_context") or {}
    candidates = context.get("assignment_candidates") or []
    measurement_candidates = [c for c in candidates if c.get("candidate_kind") == "MEASUREMENT"]
    measurement_candidates = sorted(measurement_candidates, key=lambda item: float(item.get("assignment_score") or 0.0), reverse=True)
    selected_id = match.get("assignment_candidate_id")
    selected = next((c for c in candidates if c.get("candidate_id") == selected_id), None)
    if selected is None and match.get("assignment_candidate_kind") == "GLOBAL_DETECTION":
        selected = {
            "assignment_score": match.get("assignment_candidate_score"),
            "candidate_kind": "GLOBAL_DETECTION",
            "measurement_success": True,
        }
    null_score = _safe_float(match.get("assignment_null_score"))
    top_score = _safe_float(measurement_candidates[0].get("assignment_score")) if measurement_candidates else _safe_float(match.get("assignment_candidate_score"))
    second_score = _safe_float(measurement_candidates[1].get("assignment_score")) if len(measurement_candidates) > 1 else 0.0
    selected_score = _safe_float((selected or {}).get("assignment_score"), _safe_float(match.get("assignment_candidate_score")))
    return {
        "candidate_count": int(match.get("assignment_candidate_count") or len(candidates) or 0),
        "measurement_candidate_count": len(measurement_candidates),
        "selected_candidate_kind": str(match.get("assignment_candidate_kind") or (selected or {}).get("candidate_kind") or "NONE"),
        "selected_candidate_score": selected_score,
        "null_score": null_score,
        "candidate_margin_to_null": selected_score - null_score,
        "candidate_margin_to_second": top_score - second_score,
        "selected_center_distance_mm": _safe_float((selected or {}).get("center_distance_mm")),
        "selected_angle_gap_deg": _safe_float((selected or {}).get("cad_angle_gap_deg")),
        "selected_direction_alignment": _safe_float((selected or {}).get("direction_alignment"), 1.0),
        "selected_measurement_success": 1 if (selected or {}).get("measurement_success") else 0,
        "selected_measurement_confidence": _safe_float((selected or {}).get("measurement_confidence_score")),
        "selected_candidate_visibility": _safe_float((selected or {}).get("visibility_score"), _safe_float(match.get("visibility_score"))),
        "selected_candidate_evidence": _safe_float((selected or {}).get("evidence_score"), _safe_float(match.get("local_evidence_score"))),
    }


def extract_match_features(case_payload: Dict[str, Any], match: Dict[str, Any], *, head: Optional[str] = None) -> Dict[str, Any]:
    expectation = case_payload.get("expectation") or {}
    details = case_payload.get("details") or {}
    summary = case_payload.get("summary") or {}
    features: Dict[str, Any] = {
        "scan_state": str(expectation.get("state") or "unknown"),
        "part_key": str(case_payload.get("part_key") or "unknown"),
        "spec_schedule_type": str(case_payload.get("spec_schedule_type") or "unknown"),
        "cad_strategy": str(details.get("cad_strategy") or "unknown"),
        "scan_quality_status": str(details.get("scan_quality_status") or "unknown"),
        "bend_form": str(match.get("bend_form") or "unknown"),
        "measurement_mode": str(match.get("measurement_mode") or "unknown"),
        "measurement_method": str(match.get("measurement_method") or "unknown"),
        "assignment_source": str(match.get("assignment_source") or "NONE"),
        "assignment_candidate_kind": str(match.get("assignment_candidate_kind") or "NONE"),
        "target_angle": _safe_float(match.get("target_angle")),
        "target_radius": _safe_float(match.get("target_radius")),
        "measured_angle": _safe_float(match.get("measured_angle")),
        "measured_radius": _safe_float(match.get("measured_radius")),
        "angle_deviation": _safe_float(match.get("angle_deviation")),
        "radius_deviation": _safe_float(match.get("radius_deviation")),
        "line_center_deviation_mm": _safe_float(match.get("line_center_deviation_mm")),
        "line_length_deviation_mm": _safe_float(match.get("line_length_deviation_mm")),
        "arc_length_deviation_mm": _safe_float(match.get("arc_length_deviation_mm")),
        "match_confidence": _safe_float(match.get("confidence")),
        "observability_confidence": _safe_float(match.get("observability_confidence")),
        "visibility_score": _safe_float(match.get("visibility_score")),
        "surface_visibility_ratio": _safe_float(match.get("surface_visibility_ratio")),
        "local_support_score": _safe_float(match.get("local_support_score")),
        "side_balance_score": _safe_float(match.get("side_balance_score")),
        "assignment_confidence": _safe_float(match.get("assignment_confidence")),
        "local_evidence_score": _safe_float(match.get("local_evidence_score")),
        "local_point_count": _safe_float(match.get("local_point_count")),
        "observed_surface_count": _safe_float(match.get("observed_surface_count")),
        "scan_point_count": _safe_float(details.get("scan_point_count")),
        "scan_coverage_pct": _safe_float(details.get("scan_coverage_pct")),
        "scan_density_pts_per_cm2": _safe_float(details.get("scan_density_pts_per_cm2")),
        "expected_total_bends": _safe_float(expectation.get("expected_total_bends")),
        "reported_completed_bends": _safe_float(summary.get("completed_bends")),
        "reported_remaining_bends": _safe_float(summary.get("remaining_bends")),
    }
    features.update(_candidate_summary(match))
    if head == "visibility":
        # Do not leak the target back into the visibility head.
        features.pop("observability_state", None)
    return features


def _visibility_example(case_payload: Dict[str, Any], match: Dict[str, Any]) -> UnaryExample:
    observability_state = normalize_observability_state(
        match.get("observability_state"),
        physical_completion_state=match.get("physical_completion_state"),
        status=match.get("status"),
    )
    label = 1 if observability_state in {"FORMED", "UNFORMED"} else 0
    return UnaryExample(
        head="visibility",
        label=label,
        weight=1.0,
        label_source=f"observability_state::{observability_state}",
        part_key=str(case_payload.get("part_key") or "unknown"),
        scan_name=str(case_payload.get("scan_name") or "unknown"),
        bend_id=str(match.get("bend_id") or "unknown"),
        features=extract_match_features(case_payload, match, head="visibility"),
    )


def _state_example(case_payload: Dict[str, Any], match: Dict[str, Any]) -> Optional[UnaryExample]:
    expectation = case_payload.get("expectation") or {}
    scan_state = str(expectation.get("state") or "unknown").lower()
    status = str(match.get("status") or "NOT_DETECTED").upper()
    observability_state = normalize_observability_state(
        match.get("observability_state"),
        physical_completion_state=match.get("physical_completion_state"),
        status=status,
    )
    physical_state = str(match.get("physical_completion_state") or "UNKNOWN").upper()
    expected_total = expectation.get("expected_total_bends")
    expected_completed = expectation.get("expected_completed_bends")
    label = None
    weight = 1.0
    source = None

    if scan_state == "full" and expected_total is not None:
        label = 1
        weight = 2.0
        source = "full_scan_completion_truth"
    elif physical_state == "NOT_FORMED" or observability_state == "UNFORMED":
        label = 0
        weight = 1.25
        source = "runtime_observed_not_formed"
    elif status in {"PASS", "FAIL", "WARNING"} and observability_state == "FORMED":
        label = 1
        weight = 0.8 if scan_state == "partial" else 1.0
        source = "runtime_detected_positive"
    else:
        visibility = _safe_float(match.get("visibility_score"))
        null_score = _safe_float(match.get("assignment_null_score"))
        selected_score = _safe_float(match.get("assignment_candidate_score"))
        local_support = _safe_float(match.get("local_support_score"))
        local_points = _safe_float(match.get("local_point_count"))
        if (
            scan_state == "partial"
            and status == "NOT_DETECTED"
            and observability_state == "UNKNOWN"
            and null_score >= selected_score
            and (
                visibility >= 0.45
                or (local_support >= 0.22 and local_points >= 35)
            )
        ):
            label = 0
            weight = 0.6
            source = "partial_unknown_null_bootstrap"

    if label is None or source is None:
        return None
    return UnaryExample(
        head="state",
        label=int(label),
        weight=float(weight),
        label_source=source,
        part_key=str(case_payload.get("part_key") or "unknown"),
        scan_name=str(case_payload.get("scan_name") or "unknown"),
        bend_id=str(match.get("bend_id") or "unknown"),
        features=extract_match_features(case_payload, match, head="state"),
    )


def build_examples(case_payloads: Sequence[Dict[str, Any]]) -> Tuple[List[UnaryExample], List[UnaryExample]]:
    visibility_examples: List[UnaryExample] = []
    state_examples: List[UnaryExample] = []
    for payload in case_payloads:
        for match in payload.get("matches") or []:
            visibility_examples.append(_visibility_example(payload, match))
            state_example = _state_example(payload, match)
            if state_example is not None:
                state_examples.append(state_example)
    return visibility_examples, state_examples


def _make_base_estimator() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        random_state=17,
        n_estimators=140,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
    )


def _fit_calibrated_head(examples: Sequence[UnaryExample], name: str) -> Tuple[Optional[Dict[str, Any]], Optional[UnaryHeadSummary]]:
    if not examples:
        return None, None
    y = np.asarray([e.label for e in examples], dtype=np.int32)
    if len(np.unique(y)) < 2:
        return None, None
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform([e.features for e in examples])
    sample_weight = np.asarray([e.weight for e in examples], dtype=np.float64)
    groups = np.asarray([e.part_key for e in examples])
    positive_count = int((y == 1).sum())
    negative_count = int((y == 0).sum())
    class_balance = {
        0: len(examples) / max(1.0, 2.0 * negative_count),
        1: len(examples) / max(1.0, 2.0 * positive_count),
    }
    sample_weight = np.asarray(
        [weight * class_balance[int(label)] for weight, label in zip(sample_weight, y)],
        dtype=np.float64,
    )
    min_class = min(positive_count, negative_count)
    calibration_method = "sigmoid"
    if min_class >= 60:
        calibration_method = "isotonic"
    cv = 3 if min_class >= 6 else 2

    estimator = CalibratedClassifierCV(_make_base_estimator(), method=calibration_method, cv=cv)
    estimator.fit(X, y, sample_weight=sample_weight)

    holdout_metrics: Dict[str, float] = {}
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 3 and min_class >= 8:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=17)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        eval_estimator = CalibratedClassifierCV(_make_base_estimator(), method=calibration_method, cv=min(cv, 3))
        eval_estimator.fit(X[train_idx], y[train_idx], sample_weight=sample_weight[train_idx])
        probs = eval_estimator.predict_proba(X[test_idx])[:, 1]
        holdout_metrics["brier"] = round(float(brier_score_loss(y[test_idx], probs, sample_weight=sample_weight[test_idx])), 4)
        holdout_metrics["log_loss"] = round(float(log_loss(y[test_idx], probs, sample_weight=sample_weight[test_idx], labels=[0, 1])), 4)
        if len(np.unique(y[test_idx])) > 1:
            holdout_metrics["roc_auc"] = round(float(roc_auc_score(y[test_idx], probs, sample_weight=sample_weight[test_idx])), 4)
        holdout_metrics["holdout_group_count"] = int(len(np.unique(groups[test_idx])))
        holdout_metrics["holdout_sample_count"] = int(len(test_idx))

    base_estimators = [cc.estimator for cc in getattr(estimator, 'calibrated_classifiers_', []) if getattr(cc, 'estimator', None) is not None]
    if not base_estimators:
        base_estimators = [getattr(estimator, 'estimator', None)] if getattr(estimator, 'estimator', None) is not None else []
    importances = np.zeros(X.shape[1], dtype=np.float64)
    valid_importances = 0
    for base in base_estimators:
        if hasattr(base, 'feature_importances_'):
            importances += np.asarray(base.feature_importances_, dtype=np.float64)
            valid_importances += 1
    if valid_importances:
        importances /= valid_importances
    feature_importances = sorted(
        [(name, float(score)) for name, score in zip(vectorizer.feature_names_, importances)],
        key=lambda item: item[1],
        reverse=True,
    )[:20]
    label_sources: Dict[str, int] = {}
    for example in examples:
        label_sources[example.label_source] = label_sources.get(example.label_source, 0) + 1
    bundle = {
        'name': name,
        'vectorizer': vectorizer,
        'estimator': estimator,
        'features': list(vectorizer.feature_names_),
        'calibration_method': calibration_method,
    }
    summary = UnaryHeadSummary(
        name=name,
        sample_count=int(len(examples)),
        positive_count=positive_count,
        negative_count=negative_count,
        label_sources=label_sources,
        features=list(vectorizer.feature_names_),
        calibration_method=calibration_method,
        holdout_metrics=holdout_metrics,
        feature_importances=feature_importances,
    )
    return bundle, summary


def train_unary_models(case_payloads: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], UnaryTrainingSummary]:
    visibility_examples, state_examples = build_examples(case_payloads)
    visibility_bundle, visibility_summary = _fit_calibrated_head(visibility_examples, 'visibility_head')
    state_bundle, state_summary = _fit_calibrated_head(state_examples, 'state_head')
    bundle = {
        'visibility_head': visibility_bundle,
        'state_head': state_bundle,
        'version': 'bootstrap_gbdt_v1',
    }
    summary = UnaryTrainingSummary(
        source_dir='',
        case_count=len(case_payloads),
        visibility_head=visibility_summary,
        state_head=state_summary,
    )
    return bundle, summary


def save_unary_models(bundle: Dict[str, Any], summary: UnaryTrainingSummary, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / 'bend_unary_models.joblib')
    serializable = asdict(summary)
    (out / 'training_summary.json').write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding='utf-8')
    return out


def load_unary_models(path: str | Path) -> Dict[str, Any]:
    return joblib.load(path)


def _predict_head(bundle: Optional[Dict[str, Any]], feature_dicts: Sequence[Dict[str, Any]]) -> List[Optional[float]]:
    if not bundle or not feature_dicts:
        return [None] * len(feature_dicts)
    vectorizer: DictVectorizer = bundle['vectorizer']
    estimator = bundle['estimator']
    X = vectorizer.transform(feature_dicts)
    probs = estimator.predict_proba(X)[:, 1]
    return [float(p) for p in probs]


def _candidate_atom_id(match: Dict[str, Any], candidate: Dict[str, Any]) -> Optional[str]:
    kind = str(candidate.get("candidate_kind") or "").upper()
    if kind == "MEASUREMENT":
        idx = candidate.get("measurement_index")
        if idx is not None:
            return f"measurement::{int(idx)}"
    if kind == "GLOBAL_DETECTION":
        cid = candidate.get("candidate_id") or match.get("assignment_candidate_id") or match.get("bend_id")
        if cid:
            return f"global::{cid}"
    return None


def _annotate_confusable_atoms(payload: Dict[str, Any]) -> Dict[str, Any]:
    atom_map: Dict[str, Dict[str, Any]] = {}
    matches = payload.get("matches") or []
    for match in matches:
        context = match.get("measurement_context") or {}
        candidates = context.get("assignment_candidates") or []
        selected_id = match.get("assignment_candidate_id")
        candidate_atom_ids: List[str] = []
        selected_atom_id = None
        for candidate in candidates:
            atom_id = _candidate_atom_id(match, candidate)
            if not atom_id:
                continue
            candidate_atom_ids.append(atom_id)
            record = atom_map.setdefault(
                atom_id,
                {
                    "atom_id": atom_id,
                    "candidate_kind": str(candidate.get("candidate_kind") or "UNKNOWN"),
                    "measurement_index": candidate.get("measurement_index"),
                    "bend_ids": [],
                    "claim_count": 0,
                },
            )
            bend_id = str(match.get("bend_id") or "unknown")
            if bend_id not in record["bend_ids"]:
                record["bend_ids"].append(bend_id)
            record["claim_count"] = len(record["bend_ids"])
            if selected_id and candidate.get("candidate_id") == selected_id:
                selected_atom_id = atom_id
        match["candidate_confusable_atom_ids"] = sorted(set(candidate_atom_ids))
        match["selected_confusable_atom_id"] = selected_atom_id

    atoms = sorted(atom_map.values(), key=lambda item: (item["candidate_kind"], item["atom_id"]))
    payload["structured_context"] = {
        "confusable_atoms": atoms,
        "measurement_atom_count": sum(1 for atom in atoms if str(atom.get("candidate_kind")).upper() == "MEASUREMENT"),
        "confusable_group_count": sum(1 for atom in atoms if int(atom.get("claim_count") or 0) > 1),
    }
    return payload


def _apply_hard_exclusivity(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = _annotate_confusable_atoms(payload)
    matches = payload.get("matches") or []
    atom_claims: Dict[str, List[Dict[str, Any]]] = {}
    for match in matches:
        pred = match.get("unary_predictions") or {}
        formed_prob = _safe_float(pred.get("formed_probability"))
        atom_id = match.get("selected_confusable_atom_id")
        if atom_id:
            atom_claims.setdefault(str(atom_id), []).append(
                {
                    "bend_id": str(match.get("bend_id") or "unknown"),
                    "formed_probability": formed_prob,
                    "assignment_candidate_score": _safe_float(match.get("assignment_candidate_score")),
                }
            )

    owner_by_atom: Dict[str, Optional[str]] = {}
    for atom_id, claims in atom_claims.items():
        claims.sort(
            key=lambda item: (
                float(item.get("formed_probability") or 0.0),
                float(item.get("assignment_candidate_score") or 0.0),
                item.get("bend_id") or "",
            ),
            reverse=True,
        )
        owner_by_atom[atom_id] = claims[0]["bend_id"] if claims else None

    exclusivity_rejections = 0
    for match in matches:
        pred = match.get("unary_predictions") or {}
        formed_prob = _safe_float(pred.get("formed_probability"))
        atom_id = match.get("selected_confusable_atom_id")
        rejected = False
        if atom_id and owner_by_atom.get(str(atom_id)) not in {None, str(match.get("bend_id") or "")}:
            rejected = True
            exclusivity_rejections += 1
        match["structured_predictions"] = {
            "exclusive_formed_probability": round(0.0 if rejected else formed_prob, 4),
            "exclusive_rejected": rejected,
            "exclusive_atom_owner": owner_by_atom.get(str(atom_id)) if atom_id else None,
        }

    payload.setdefault("structured_context", {})
    payload["structured_context"].update(
        {
            "hard_exclusivity_atoms": owner_by_atom,
            "hard_exclusivity_rejections": exclusivity_rejections,
        }
    )
    return payload


def _candidate_options(match: Dict[str, Any]) -> List[Dict[str, Any]]:
    context = match.get("measurement_context") or {}
    candidates = context.get("assignment_candidates") or []
    options: List[Dict[str, Any]] = []
    seen_keys = set()
    formed_prob = _safe_float((match.get("unary_predictions") or {}).get("formed_probability"))
    visibility_prob = _safe_float((match.get("unary_predictions") or {}).get("visibility_probability"), 0.5)
    null_score = _safe_float(match.get("assignment_null_score"))
    null_bonus = max(0.0, 1.0 - visibility_prob)
    options.append(
        {
            "atom_id": None,
            "candidate_id": "null_candidate",
            "candidate_kind": "NULL",
            "score": null_score + null_bonus,
            "formed_probability": 0.0,
        }
    )
    for candidate in candidates:
        atom_id = _candidate_atom_id(match, candidate)
        candidate_id = str(candidate.get("candidate_id") or "")
        key = (candidate_id, atom_id)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        assignment_score = _safe_float(candidate.get("assignment_score"))
        evidence_score = _safe_float(candidate.get("evidence_score"), _safe_float(match.get("local_evidence_score")))
        visibility_score = _safe_float(candidate.get("visibility_score"), visibility_prob)
        measurement_conf = _safe_float(candidate.get("measurement_confidence_score"), _safe_float(match.get("assignment_confidence")))
        score = (
            assignment_score
            + 0.45 * formed_prob
            + 0.15 * evidence_score
            + 0.1 * visibility_score
            + 0.1 * measurement_conf
        )
        options.append(
            {
                "atom_id": atom_id,
                "candidate_id": candidate_id or str(atom_id or "candidate"),
                "candidate_kind": str(candidate.get("candidate_kind") or "UNKNOWN").upper(),
                "score": score,
                "formed_probability": formed_prob,
            }
        )
    if not any(opt["atom_id"] is not None for opt in options):
        selected_atom = match.get("selected_confusable_atom_id")
        selected_kind = str(match.get("assignment_candidate_kind") or "UNKNOWN").upper()
        if selected_atom:
            options.append(
                {
                    "atom_id": selected_atom,
                    "candidate_id": str(match.get("assignment_candidate_id") or selected_atom),
                    "candidate_kind": selected_kind,
                    "score": _safe_float(match.get("assignment_candidate_score")) + 0.45 * formed_prob,
                    "formed_probability": formed_prob,
                }
            )
    dedup: Dict[Tuple[Optional[str], str], Dict[str, Any]] = {}
    for option in options:
        key = (option.get("atom_id"), str(option.get("candidate_id")))
        prev = dedup.get(key)
        if prev is None or float(option["score"]) > float(prev["score"]):
            dedup[key] = option
    return sorted(dedup.values(), key=lambda item: float(item["score"]), reverse=True)


def _build_confusable_components(matches: Sequence[Dict[str, Any]]) -> List[List[str]]:
    bend_to_atoms: Dict[str, set[str]] = {}
    atom_to_bends: Dict[str, set[str]] = {}
    for match in matches:
        bend_id = str(match.get("bend_id") or "")
        if not bend_id:
            continue
        atoms = {
            str(candidate.get("atom_id"))
            for candidate in (_candidate_options(match) or [])
            if candidate.get("atom_id") is not None
        }
        bend_to_atoms[bend_id] = atoms
        for atom in atoms:
            atom_to_bends.setdefault(atom, set()).add(bend_id)
    remaining = set(bend_to_atoms.keys())
    components: List[List[str]] = []
    while remaining:
        start = remaining.pop()
        queue = [start]
        component = {start}
        while queue:
            bend = queue.pop()
            for atom in bend_to_atoms.get(bend, set()):
                for other_bend in atom_to_bends.get(atom, set()):
                    if other_bend not in component:
                        component.add(other_bend)
                        if other_bend in remaining:
                            remaining.remove(other_bend)
                        queue.append(other_bend)
        components.append(sorted(component))
    return components


def _decode_component_joint_assignments(component_matches: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    ordered = sorted(component_matches, key=lambda m: str(m.get("bend_id") or ""))
    options_by_bend = {str(match.get("bend_id")): _candidate_options(match) for match in ordered}
    best_score = -1e18
    best_assignment: Dict[str, Dict[str, Any]] = {}

    def _dfs(index: int, used_atoms: set[str], score_acc: float, chosen: Dict[str, Dict[str, Any]]) -> None:
        nonlocal best_score, best_assignment
        if index >= len(ordered):
            if score_acc > best_score:
                best_score = score_acc
                best_assignment = dict(chosen)
            return
        bend_id = str(ordered[index].get("bend_id") or "")
        options = options_by_bend.get(bend_id) or []
        for option in options:
            atom_id = option.get("atom_id")
            if atom_id is not None and str(atom_id) in used_atoms:
                continue
            next_used = used_atoms
            if atom_id is not None:
                next_used = set(used_atoms)
                next_used.add(str(atom_id))
            chosen[bend_id] = option
            _dfs(index + 1, next_used, score_acc + float(option.get("score") or 0.0), chosen)
        chosen.pop(bend_id, None)

    _dfs(0, set(), 0.0, {})
    return best_assignment


def _apply_joint_candidate_exclusivity(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = _annotate_confusable_atoms(payload)
    matches = payload.get("matches") or []
    atom_claim_count = {
        str(atom.get("atom_id")): int(atom.get("claim_count") or 0)
        for atom in (payload.get("structured_context") or {}).get("confusable_atoms", [])
    }
    match_by_bend = {str(match.get("bend_id") or ""): match for match in matches if match.get("bend_id")}
    components = _build_confusable_components(matches)
    joint_assignments: Dict[str, Dict[str, Any]] = {}
    for component in components:
        component_matches = [match_by_bend[bend_id] for bend_id in component if bend_id in match_by_bend]
        if not component_matches:
            continue
        component_atoms = {
            str(option.get("atom_id"))
            for match in component_matches
            for option in _candidate_options(match)
            if option.get("atom_id") is not None
        }
        if not any(atom_claim_count.get(atom_id, 0) > 1 for atom_id in component_atoms):
            continue
        joint_assignments.update(_decode_component_joint_assignments(component_matches))

    joint_rejections = 0
    for match in matches:
        bend_id = str(match.get("bend_id") or "")
        assigned = joint_assignments.get(bend_id)
        structured = match.setdefault("structured_predictions", {})
        if assigned is None:
            structured.update(
                {
                    "joint_assignment_candidate_id": structured.get("exclusive_atom_owner") or match.get("assignment_candidate_id"),
                    "joint_assignment_candidate_kind": match.get("assignment_candidate_kind"),
                    "joint_assignment_atom_id": match.get("selected_confusable_atom_id"),
                    "joint_assignment_score": round(float(match.get("assignment_candidate_score") or 0.0), 4),
                    "joint_exclusive_formed_probability": round(
                        float(structured.get("exclusive_formed_probability") or 0.0), 4
                    ),
                    "joint_exclusive_rejected": bool(structured.get("exclusive_rejected")),
                }
            )
            continue
        atom_id = assigned.get("atom_id")
        rejected = atom_id is None
        if rejected:
            joint_rejections += 1
        structured.update(
            {
                "joint_assignment_candidate_id": assigned.get("candidate_id"),
                "joint_assignment_candidate_kind": assigned.get("candidate_kind"),
                "joint_assignment_atom_id": atom_id,
                "joint_assignment_score": round(float(assigned.get("score") or 0.0), 4),
                "joint_exclusive_formed_probability": round(
                    0.0 if rejected else float(assigned.get("formed_probability") or 0.0), 4
                ),
                "joint_exclusive_rejected": rejected,
            }
        )

    payload.setdefault("structured_context", {})
    payload["structured_context"].update(
        {
            "joint_confusable_components": components,
            "joint_exclusivity_rejections": joint_rejections,
        }
    )
    return payload


def _poisson_binomial_distribution(probs: Sequence[float]) -> List[float]:
    dist = [1.0]
    for prob in probs:
        p = max(0.0, min(1.0, float(prob)))
        next_dist = [0.0] * (len(dist) + 1)
        for k, mass in enumerate(dist):
            next_dist[k] += mass * (1.0 - p)
            next_dist[k + 1] += mass * p
        dist = next_dist
    total = sum(dist)
    if total > 0:
        dist = [value / total for value in dist]
    return dist


def _count_probability_for_match(match: Dict[str, Any], scan_state: str) -> float:
    if not bool(match.get("countable_in_regression", True)):
        return 0.0
    prob = _safe_float(
        (match.get("structured_predictions") or {}).get(
            "joint_exclusive_formed_probability",
            (match.get("structured_predictions") or {}).get("exclusive_formed_probability"),
        )
    )
    if (
        str(scan_state or "").lower() == "partial"
        and str(match.get("status") or "").upper() == "NOT_DETECTED"
        and normalize_observability_state(
            match.get("observability_state"),
            physical_completion_state=match.get("physical_completion_state"),
            status=match.get("status"),
        ) == "UNKNOWN"
    ):
        # On partial scans, fully unobserved bends should not add positive count mass.
        # This keeps missing evidence from masquerading as formed-state evidence.
        return 0.0
    return prob


def _count_posterior_summary(matches: Sequence[Dict[str, Any]], scan_state: str = "unknown") -> Dict[str, Any]:
    countable_matches = [match for match in matches if bool(match.get("countable_in_regression", True))]
    probs = [_count_probability_for_match(match, scan_state) for match in countable_matches]
    dist = _poisson_binomial_distribution(probs)
    cumulative = 0.0
    median_k = 0
    for k, mass in enumerate(dist):
        cumulative += mass
        if cumulative >= 0.5:
            median_k = k
            break
    map_k = max(range(len(dist)), key=lambda idx: dist[idx]) if dist else 0
    mean_k = sum(idx * mass for idx, mass in enumerate(dist))
    return {
        "count_distribution": [round(float(value), 6) for value in dist],
        "median_completed_bends": int(median_k),
        "map_completed_bends": int(map_k),
        "mean_completed_bends": round(float(mean_k), 4),
    }


def score_case_payload(case_payload: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.loads(json.dumps(case_payload))
    matches = payload.get('matches') or []
    features = [extract_match_features(payload, match) for match in matches]
    vis_probs = _predict_head(bundle.get('visibility_head'), features)
    state_probs = _predict_head(bundle.get('state_head'), features)
    for match, v_prob, s_prob in zip(matches, vis_probs, state_probs):
        match['unary_predictions'] = {
            'visibility_probability': round(v_prob, 4) if v_prob is not None else None,
            'formed_probability': round(s_prob, 4) if s_prob is not None else None,
        }
    payload = _apply_hard_exclusivity(payload)
    payload = _apply_joint_candidate_exclusivity(payload)
    payload.setdefault("structured_context", {})
    scan_state = str((payload.get("expectation") or {}).get("state") or "unknown")
    payload["structured_context"]["count_posterior"] = _count_posterior_summary(matches, scan_state=scan_state)
    return payload
