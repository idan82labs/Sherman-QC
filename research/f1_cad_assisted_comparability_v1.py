"""Build an apples-to-apples audit of CAD-assisted versus scan-only evidence."""

from __future__ import annotations

from typing import Any, Mapping


def _f1(row: Mapping[str, Any]) -> float:
    tp = int(row.get("true_positive") or 0)
    fp = int(row.get("false_positive") or 0)
    fn = int(row.get("false_negative") or 0)
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def _metric_subset(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "case_count", "precision", "recall", "f1", "true_positive",
            "false_positive", "false_negative", "exact_count_accuracy",
            "exact_localized_target_set_accuracy", "minimum_family_recall",
        )
    }


def build(
    *,
    multisource: Mapping[str, Any],
    line_comparison: Mapping[str, Any],
    unified_gate: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = multisource["metrics"]
    scan_matched = metrics["scan_only"]
    retained = metrics["incumbent_union"]
    aggressive = metrics["triple_union"]
    pure_cad = line_comparison["cad_guided_matched_a0_subset"]
    scan_physical = unified_gate["evidence"]["scan_only"]
    cad_evidence = unified_gate["evidence"]["cad_assisted"]

    cross_corpus_valid = (
        scan_physical["hashes"]["dataset"] == cad_evidence["hashes"]["dataset"]
        and scan_physical["counts"]["physical_scans"] == cad_evidence["counts"]["physical_scans"]
        and scan_physical["counts"]["morphology_families"] == cad_evidence["counts"]["morphology_families"]
    )
    case_deltas = []
    for case in multisource["cases"]:
        scan = case["scan_only"]
        cad = case["incumbent_union"]
        case_deltas.append({
            "case_id": case["case_id"],
            "family_id": case["family_id"],
            "scan_only_f1": _f1(scan),
            "retained_cad_f1": _f1(cad),
            "f1_delta": _f1(cad) - _f1(scan),
            "tp_delta": int(cad["true_positive"]) - int(scan["true_positive"]),
            "fp_delta": int(cad["false_positive"]) - int(scan["false_positive"]),
            "fn_delta": int(cad["false_negative"]) - int(scan["false_negative"]),
        })
    return {
        "schema": "SHERMAN_QC_F1_CAD_ASSISTED_COMPARABILITY_AUDIT_V1",
        "status": "PASS_DIAGNOSIS",
        "headline_cross_corpus_comparison_valid": cross_corpus_valid,
        "headline_cross_corpus_reasons": [] if cross_corpus_valid else [
            "DIFFERENT_DATASET_HASHES",
            "DIFFERENT_SCORED_SCAN_DENOMINATORS",
            "DIFFERENT_SCORED_MORPHOLOGY_DENOMINATORS",
            "DIFFERENT_CANDIDATE_VERSIONS_AND_SCORERS",
        ],
        "cross_corpus_denominators": {
            "scan_only_scored_scans": scan_physical["counts"]["physical_scans"],
            "scan_only_scored_morphologies": scan_physical["counts"]["morphology_families"],
            "cad_assisted_scored_scans": cad_evidence["counts"]["physical_scans"],
            "cad_assisted_scored_morphologies": cad_evidence["counts"]["morphology_families"],
        },
        "misleading_headline_values": {
            "scan_only_gold12_experimental_candidate_f1": scan_physical["candidate_metrics"]["f1"],
            "cad_assisted_gold11_experimental_candidate_f1": cad_evidence["candidate_metrics"]["f1"],
            "interpretation": "NOT_AN_APPLES_TO_APPLES_MODALITY_COMPARISON",
        },
        "matched_gold11": {
            "scan_only_control": _metric_subset(scan_matched),
            "pure_native_step_cad_guided": _metric_subset(pure_cad),
            "retained_scan_plus_scan_supported_cad": _metric_subset(retained),
            "aggressive_three_source_union": _metric_subset(aggressive),
            "retained_cad_f1_delta_vs_scan_only": float(retained["f1"]) - float(scan_matched["f1"]),
            "retained_cad_tp_delta_vs_scan_only": int(retained["true_positive"]) - int(scan_matched["true_positive"]),
            "retained_cad_fp_delta_vs_scan_only": int(retained["false_positive"]) - int(scan_matched["false_positive"]),
            "retained_cad_fn_delta_vs_scan_only": int(retained["false_negative"]) - int(scan_matched["false_negative"]),
            "aggressive_f1_delta_vs_retained": float(aggressive["f1"]) - float(retained["f1"]),
            "aggressive_tp_delta_vs_retained": int(aggressive["true_positive"]) - int(retained["true_positive"]),
            "aggressive_fp_delta_vs_retained": int(aggressive["false_positive"]) - int(retained["false_positive"]),
            "aggressive_fn_delta_vs_retained": int(aggressive["false_negative"]) - int(retained["false_negative"]),
        },
        "case_deltas_retained_cad_vs_scan_only": case_deltas,
        "ruling": {
            "cad_assistance_helped_on_matched_data": float(retained["f1"]) > float(scan_matched["f1"]),
            "pure_cad_guidance_helped_on_matched_data": float(pure_cad["f1"]) > float(scan_matched["f1"]),
            "aggressive_union_outperformed_retained": float(aggressive["f1"]) > float(retained["f1"]),
            "production_generalization_proven": False,
            "gpi": "GPI-C_POSITIVE_MATCHED_CAD_DEVELOPMENT_NOT_GENERALIZATION",
        },
    }
