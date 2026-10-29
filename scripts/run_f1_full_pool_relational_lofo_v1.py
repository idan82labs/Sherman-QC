#!/usr/bin/env python3
"""Nested-LOFO full-pool relational ownership diagnostic for physical V51."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, _endpoints, is_match, match_lines, pair_metrics
from scripts.f1_physical_evaluation_contract_v2 import score_localized_cases
from scripts.localize_cad_bends_on_scan_v1 import _atomic


NUMERIC_FEATURES = (
    "case_percentile_support_count","case_percentile_score","case_percentile_consensus_count",
    "case_percentile_finite_support_fraction","case_percentile_finite_support_longest_run_fraction",
    "case_percentile_endpoint_support_fraction","case_percentile_normal_second_eigenvalue",
    "case_percentile_normal_spread_deg_scan","case_percentile_corridor_points_per_mm",
    "case_percentile_transverse_angular_span_deg","case_percentile_cluster_member_count",
    "case_percentile_cluster_source_count","case_percentile_two_sided_transition_score",
    "case_percentile_two_sided_normal_angle_deg","case_percentile_two_sided_support_balance",
    "case_percentile_two_sided_min_support_fraction","case_percentile_two_sided_normal_coherence",
    "case_percentile_tight_normal_second_eigenvalue_r2mm","case_percentile_tight_to_wide_normal_second_ratio",
    "finite_support_fraction","finite_support_longest_run_fraction","endpoint_support_fraction",
    "two_sided_transition_score","two_sided_normal_angle_deg","two_sided_min_support_fraction",
    "two_sided_support_balance","two_sided_normal_coherence","corridor_points_per_mm",
    "cluster_member_count","cluster_source_count","support_count","consensus_count",
)
THRESHOLD_GRID = tuple(round(value,2) for value in np.arange(0.50,0.951,0.05))
MATCH_THRESHOLDS = Thresholds()


def _candidate_keys(row: dict[str,Any]) -> set[str]:
    return {str(row[key]) for key in ("candidate_id","bend_id","source_candidate_id") if row.get(key) is not None}


def _safe_metrics(left: dict[str,Any], right: dict[str,Any]) -> dict[str,float]:
    return pair_metrics(_endpoints(left),_endpoints(right))


def _duplicate(left: dict[str,Any],right: dict[str,Any])->bool:
    metric=_safe_metrics(left,right)
    return metric["axis_angle_deg"]<=5.0 and metric["transverse_offset_mm"]<=4.0 and metric["shorter_span_overlap"]>=0.60


def _relational_features(candidates:list[dict[str,Any]],baseline:list[dict[str,Any]],bbox_diagonal:float)->list[dict[str,float]]:
    baseline_keys={str(row.get("source_hypothesis_id")) for row in baseline}
    features=[]
    endpoints=[_endpoints(row) for row in candidates]
    lengths=[float(np.linalg.norm(value[1]-value[0])) for value in endpoints]
    for index,candidate in enumerate(candidates):
        values={name:float(candidate.get(name) or 0.0) for name in NUMERIC_FEATURES}
        source=str(candidate.get("representative_source") or candidate.get("candidate_source") or "unknown")
        values[f"source={source}"]=1.0
        values["line_length_bbox_fraction"]=lengths[index]/max(bbox_diagonal,1e-9)
        values["is_v51_selected"]=float(bool(_candidate_keys(candidate)&baseline_keys))
        selected_metrics=[_safe_metrics(candidate,row) for row in baseline]
        if selected_metrics:
            values["minimum_selected_axis_angle_deg"]=min(row["axis_angle_deg"] for row in selected_metrics)/90.0
            parallel=[row for row in selected_metrics if row["axis_angle_deg"]<=10.0]
            values["nearest_parallel_selected_offset_bbox_fraction"]=(min((row["transverse_offset_mm"] for row in parallel),default=bbox_diagonal)/max(bbox_diagonal,1e-9))
            values["maximum_parallel_selected_overlap"]=max((row["shorter_span_overlap"] for row in parallel),default=0.0)
            values["duplicate_of_v51_selected"]=float(any(row["axis_angle_deg"]<=5.0 and row["transverse_offset_mm"]<=4.0 and row["shorter_span_overlap"]>=0.60 for row in selected_metrics))
            values["near_parallel_selected_count"]=float(sum(row["axis_angle_deg"]<=5.0 and row["transverse_offset_mm"]<=20.0 and row["shorter_span_overlap"]>=0.30 for row in selected_metrics))
        else:
            values.update(minimum_selected_axis_angle_deg=1.0,nearest_parallel_selected_offset_bbox_fraction=1.0,maximum_parallel_selected_overlap=0.0,duplicate_of_v51_selected=0.0,near_parallel_selected_count=0.0)
        duplicate_siblings=0;parallel_siblings=0
        for other_index,other in enumerate(candidates):
            if other_index==index:continue
            metric=pair_metrics(endpoints[index],endpoints[other_index])
            duplicate_siblings+=int(metric["axis_angle_deg"]<=5.0 and metric["transverse_offset_mm"]<=4.0 and metric["shorter_span_overlap"]>=0.60)
            parallel_siblings+=int(metric["axis_angle_deg"]<=5.0 and metric["transverse_offset_mm"]<=20.0 and metric["shorter_span_overlap"]>=0.30)
        values["duplicate_sibling_count"]=float(duplicate_siblings)
        values["near_parallel_sibling_count"]=float(parallel_siblings)
        features.append(values)
    return features


def _load_cases(public:dict[str,Any],catalog:dict[str,Any],root:Path)->list[dict[str,Any]]:
    gold_by_sha={str(row["scan_sha256"]):row for row in catalog["cases"]}
    cases=[]
    for public_row in public["cases"]:
        gold=gold_by_sha.get(str(public_row["source_sha256"]))
        if gold is None or gold.get("truth_tier")!="codex_manual_multiview_count_backed":continue
        case_id=str(public_row["opaque_case_id"]);case_root=root/"primary"/case_id
        pool=json.loads((case_root/"pool"/f"{case_id}.json").read_text())
        parent=json.loads((case_root/"semantic_result.json").read_text())
        candidates=list(pool["line_objects"])
        bbox=np.asarray(pool["scan_bbox_mm"],dtype=float)
        if bbox.shape == (2, 3):
            bbox_diagonal=float(np.linalg.norm(bbox[1]-bbox[0]))
        elif bbox.shape == (3,):
            bbox_diagonal=float(np.linalg.norm(bbox))
        else:
            raise ValueError(f"{case_id}: unsupported scan_bbox_mm shape {bbox.shape}")
        features=_relational_features(candidates,list(parent["predictions"]),bbox_diagonal)
        labels=[]
        for candidate in candidates:
            line=_endpoints(candidate)
            labels.append(int(any(is_match(pair_metrics(_endpoints(bend),line),MATCH_THRESHOLDS) for bend in gold["bends"])))
        baseline_by_source={str(row.get("source_hypothesis_id")):row for row in parent["predictions"]}
        cases.append({"case_id":case_id,"family_id":str(gold["part_id"]),"gold":gold["bends"],"candidates":candidates,"features":features,"labels":labels,"baseline_predictions":parent["predictions"],"baseline_by_source":baseline_by_source})
    if len(cases)!=12:raise ValueError(f"expected 12 cases, got {len(cases)}")
    return cases


def _fit(cases:list[dict[str,Any]])->tuple[DictVectorizer,HistGradientBoostingClassifier]:
    vectorizer=DictVectorizer(sparse=False)
    feature_rows=[feature for case in cases for feature in case["features"]]
    labels=np.asarray([label for case in cases for label in case["labels"]],dtype=np.int64)
    matrix=vectorizer.fit_transform(feature_rows)
    positives=max(1,int(np.sum(labels)));negatives=max(1,len(labels)-positives)
    weights=np.where(labels==1,len(labels)/(2*positives),len(labels)/(2*negatives))
    model=HistGradientBoostingClassifier(learning_rate=0.05,max_iter=150,max_leaf_nodes=15,l2_regularization=1.0,min_samples_leaf=20,random_state=17)
    model.fit(matrix,labels,sample_weight=weights)
    return vectorizer,model


def _probabilities(vectorizer:DictVectorizer,model:HistGradientBoostingClassifier,case:dict[str,Any])->np.ndarray:
    return model.predict_proba(vectorizer.transform(case["features"]))[:,1]


def _as_prediction(case:dict[str,Any],candidate:dict[str,Any])->dict[str,Any]:
    baseline=next((case["baseline_by_source"][key] for key in sorted(_candidate_keys(candidate)) if key in case["baseline_by_source"]),None)
    if baseline is not None:return {**baseline,"selection_route":"full_pool_relational_lofo_v1_retained_incumbent"}
    endpoints=_endpoints(candidate)
    return {"bend_id":str(candidate["bend_id"]),"source_hypothesis_id":str(candidate.get("candidate_id") or candidate["bend_id"]),"line_start":endpoints[0].tolist(),"line_end":endpoints[1].tolist(),"selection_route":"full_pool_relational_lofo_v1_pool_birth","selector_score":None}


def _select(case:dict[str,Any],probabilities:np.ndarray,threshold:float)->list[dict[str,Any]]:
    order=sorted(range(len(probabilities)),key=lambda index:(-float(probabilities[index]),str(case["candidates"][index].get("candidate_id") or case["candidates"][index]["bend_id"])))
    selected=[]
    for index in order:
        if float(probabilities[index])<threshold:continue
        prediction=_as_prediction(case,case["candidates"][index]);prediction["selector_score"]=round(float(probabilities[index]),12)
        if any(_duplicate(prediction,prior) for prior in selected):continue
        selected.append(prediction)
    selected.sort(key=lambda row:str(row["bend_id"]))
    return selected


def _score(case:dict[str,Any],probabilities:np.ndarray,threshold:float)->dict[str,Any]:
    predictions=_select(case,probabilities,threshold);result=match_lines(case["gold"],predictions,MATCH_THRESHOLDS);result.update(case_id=case["case_id"],family_id=case["family_id"],predictions=predictions);return result


def _aggregate(rows:list[dict[str,Any]])->dict[str,Any]:
    metrics=score_localized_cases(rows);metrics["minimum_family_recall"]=min(row["true_positive"]/row["gold_count"] for row in rows);return metrics


def _threshold(inner_predictions:list[tuple[dict[str,Any],np.ndarray]])->float:
    best=None
    for threshold in THRESHOLD_GRID:
        metrics=_aggregate([_score(case,probabilities,threshold) for case,probabilities in inner_predictions])
        key=(float(metrics["f1"] or 0.0),float(metrics["precision"] or 0.0),threshold)
        if best is None or key>best[0]:best=(key,threshold)
    assert best is not None;return best[1]


def _crossfit(cases:list[dict[str,Any]])->dict[str,Any]:
    outer=[]
    for outer_index,outer_case in enumerate(cases):
        train=[case for index,case in enumerate(cases) if index!=outer_index]
        inner_predictions=[]
        for inner_index,inner_case in enumerate(train):
            inner_train=[case for index,case in enumerate(train) if index!=inner_index]
            vectorizer,model=_fit(inner_train);inner_predictions.append((inner_case,_probabilities(vectorizer,model,inner_case)))
        threshold=_threshold(inner_predictions)
        vectorizer,model=_fit(train)
        scored=_score(outer_case,_probabilities(vectorizer,model,outer_case),threshold)
        scored["selected_threshold"]=threshold;scored["outer_case_excluded_from_fit_and_threshold"]=True;outer.append(scored)
    return {"metrics":_aggregate(outer),"outer_cases":outer}


def _baseline(cases:list[dict[str,Any]])->dict[str,Any]:
    rows=[]
    for case in cases:
        row=match_lines(case["gold"],case["baseline_predictions"],MATCH_THRESHOLDS);row.update(case_id=case["case_id"],family_id=case["family_id"]);rows.append(row)
    return {"metrics":_aggregate(rows),"cases":rows}


def _stable_crossfit(result:dict[str,Any])->str:
    return hashlib.sha256((json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()).hexdigest()


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--public-manifest",type=Path,required=True);parser.add_argument("--gold-catalog",type=Path,required=True);parser.add_argument("--v51-root",type=Path,required=True);parser.add_argument("--output-root",type=Path,required=True);args=parser.parse_args()
    started=time.perf_counter();cases=_load_cases(json.loads(args.public_manifest.read_text()),json.loads(args.gold_catalog.read_text()),args.v51_root)
    primary=_crossfit(cases);replay=_crossfit(cases)
    if _stable_crossfit(primary)!=_stable_crossfit(replay):raise RuntimeError("nested LOFO replay mismatch")
    baseline=_baseline(cases);before,after=baseline["metrics"],primary["metrics"]
    keys=("precision","recall","f1","exact_count_accuracy","exact_localized_target_set_accuracy","minimum_family_recall")
    deltas={key:after[key]-before[key] for key in keys}
    for stage,result in (("primary",primary),("replay",replay)):
        for row in result["outer_cases"]:
            _atomic(args.output_root/stage/row["case_id"]/"semantic_result.json",{"schema":"SHERMAN_QC_EXP_SCAN_0002_RESULT_V1","experiment_id":"EXP_SCAN_0002","case_id":row["case_id"],"predictions":row["predictions"],"selected_threshold":row["selected_threshold"],"outer_case_excluded_from_fit_and_threshold":True,"gold_or_expected_counts_used_at_inference":False})
    report={"schema":"SHERMAN_QC_EXP_SCAN_0002_NESTED_LOFO_REPORT_V1","experiment_id":"EXP_SCAN_0002","evidence_class":"DEVELOPMENT_CONSUMED_PHYSICAL_CROSSFIT","promotion_eligible":False,"baseline":baseline,"candidate":primary,"deltas":deltas,"replay_sha256":_stable_crossfit(replay),"semantic_replay_byte_identical":True,"elapsed_seconds":time.perf_counter()-started,"success_criterion_passed":bool(deltas["f1"]>0 and all(deltas[key]>=0 for key in keys if key!="f1"))}
    _atomic(args.output_root/"REPORT.json",report)
    print(json.dumps({"baseline_f1":before["f1"],"candidate_f1":after["f1"],"deltas":deltas,"elapsed_seconds":report["elapsed_seconds"],"success":report["success_criterion_passed"]},sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
