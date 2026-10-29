"""Pure paired scoring of already-computed finite sample-support predictions.

This module does not load inputs, invoke inference, verify declarations, or grant
evaluation/release authority. Both lanes are scored against the same supplied
references and complete, explicitly declared case/family inventory. In particular,
sample-supported instance counts are not physical manufacturing-event counts.
"""

if __package__:
    from .cad_guided_evaluation import score_support_case, score_support_cases, Thresholds
else:
    from cad_guided_evaluation import score_support_case, score_support_cases, Thresholds


MAX_CASES = 4096
MAX_ID_LENGTH = 256
_METRICS = (
    "true_positive", "false_positive", "false_negative", "precision", "recall", "f1",
    "exact_count_accuracy", "exact_localized_target_set_accuracy", "inference_complete_rate",
)


def _identifier(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_ID_LENGTH:
        raise ValueError(f"{label} must be a nonempty bounded string")
    return value


def _inventory(expected_cases, declared_families):
    if not isinstance(expected_cases, (list, tuple)) or not 1 <= len(expected_cases) <= MAX_CASES:
        raise ValueError("expected_cases requires 1..4096 explicit cases")
    if (not isinstance(declared_families, (list, tuple))
            or not 1 <= len(declared_families) <= MAX_CASES):
        raise ValueError("declared_families requires 1..4096 explicit families")
    families = [_identifier(value, "family_id") for value in declared_families]
    if len(set(families)) != len(families):
        raise ValueError("declared family identities must be unique")
    family_set, case_ids, part_families, cases = set(families), set(), {}, []
    for case in expected_cases:
        if not isinstance(case, dict) or set(case) != {"case_id", "part_id", "family_id"}:
            raise ValueError("each expected case requires only case_id, part_id, family_id")
        case_id = _identifier(case["case_id"], "case_id")
        part_id = _identifier(case["part_id"], "part_id")
        family_id = _identifier(case["family_id"], "family_id")
        if case_id in case_ids:
            raise ValueError("expected case identities must be unique")
        if family_id not in family_set:
            raise ValueError("every case family must be explicitly declared")
        if part_id in part_families and part_families[part_id] != family_id:
            raise ValueError("one part cannot belong to multiple families")
        case_ids.add(case_id)
        part_families[part_id] = family_id
        cases.append({"case_id": case_id, "part_id": part_id, "family_id": family_id})
    return cases, families, part_families


def _exact_mapping(mapping, case_ids, label):
    if (not isinstance(mapping, dict) or len(mapping) != len(case_ids)
            or any(not isinstance(key, str) for key in mapping) or set(mapping) != case_ids):
        raise ValueError(f"{label} must contain exactly the complete declared case set")


def _deltas(incumbent, challenger):
    return {name: (challenger[name] - incumbent[name]
                   if incumbent[name] is not None and challenger[name] is not None else None)
            for name in _METRICS}


def _ratio_check(numerator, denominator, target_numerator, target_denominator, *, defined=True):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "target_numerator": target_numerator,
        "target_denominator": target_denominator,
        "meets": (numerator * target_denominator >= denominator * target_numerator
                  if defined and denominator > 0 else None),
    }


def _diagnostic_targets(aggregate, family_aggregates):
    """Exact integer comparisons; these are sample-support diagnostics ONLY."""
    tp, fp, fn = (aggregate[key] for key in ("true_positive", "false_positive", "false_negative"))
    checks = {
        "sample_support_exact_count": _ratio_check(
            aggregate["exact_count_cases"], aggregate["case_count"], 95, 100),
        "precision": _ratio_check(tp, tp + fp, 90, 100, defined=aggregate["precision"] is not None),
        "recall": _ratio_check(tp, tp + fn, 90, 100, defined=aggregate["recall"] is not None),
        "f1": _ratio_check(2 * tp, 2 * tp + fp + fn, 90, 100, defined=aggregate["f1"] is not None),
    }
    family_checks = {}
    for family_id, row in family_aggregates.items():
        family_checks[family_id] = (
            _ratio_check(row["true_positive"], row["true_positive"] + row["false_negative"], 80, 100,
                         defined=row["recall"] is not None)
            if row is not None else _ratio_check(0, 0, 80, 100, defined=False))
    known = all(row is not None and row["recall"] is not None for row in family_aggregates.values())
    return {
        "metric_scope": "SAMPLE_SUPPORT_DIAGNOSTIC_NOT_PHYSICAL_BEND_TARGET_QUALIFICATION",
        "checks": checks,
        "family_recall_checks": family_checks,
        "minimum_family_recall": min(row["recall"] for row in family_aggregates.values()) if known else None,
        "diagnostic_targets_met": (all(check["meets"] is True for check in checks.values())
                                   and all(check["meets"] is True for check in family_checks.values())),
        "physical_target_qualification": "NOT_ESTABLISHED",
    }


def score_paired_support_cases(*, expected_cases, declared_families, reference_records_by_case,
                               incumbent_payloads_by_case, challenger_payloads_by_case,
                               reference_frame, reference_unit, reference_convention, thresholds=None):
    """Recompute a paired comparison after inference, with no omitted jobs.

    Missing payloads are errors, never fabricated empty predictions. A failed
    inference job must supply an explicit UNAVAILABLE payload; valid references
    then remain in its scoring denominator. Missing/invalid references invalidate
    evaluation rather than inventing zero truth. Family membership and reference
    compatibility are declarations, not independently verified provenance.
    """
    cases, families, part_families = _inventory(expected_cases, declared_families)
    ids = [case["case_id"] for case in cases]
    id_set = set(ids)
    for label, mapping in (("references", reference_records_by_case),
                           ("incumbent payloads", incumbent_payloads_by_case),
                           ("challenger payloads", challenger_payloads_by_case)):
        _exact_mapping(mapping, id_set, label)
    if thresholds is None:
        thresholds = Thresholds()
    if not isinstance(thresholds, Thresholds):
        raise ValueError("thresholds must be Thresholds")
    lanes = {}
    for name, payloads in (("incumbent", incumbent_payloads_by_case),
                           ("challenger", challenger_payloads_by_case)):
        rows = [score_support_case(
            case_id=case_id, reference_records=reference_records_by_case[case_id],
            prediction_payload=payloads[case_id], reference_frame=reference_frame,
            reference_unit=reference_unit, reference_convention=reference_convention,
            thresholds=thresholds) for case_id in ids]
        lanes[name] = {"case_scores": rows, "aggregate": score_support_cases(rows, expected_case_ids=ids)}
    row_maps = {name: {row["case_id"]: row for row in lane["case_scores"]}
                for name, lane in lanes.items()}
    family_rows = []
    family_aggregates = {name: {} for name in lanes}
    for family_id in families:
        family_cases = [case for case in cases if case["family_id"] == family_id]
        family_ids = [case["case_id"] for case in family_cases]
        scores = {name: (score_support_cases([row_maps[name][case_id] for case_id in family_ids],
                                             expected_case_ids=family_ids) if family_ids else None)
                  for name in lanes}
        for name in lanes:
            family_aggregates[name][family_id] = scores[name]
        family_rows.append({
            "family_id": family_id, "evidence_status": "SCORED" if family_ids else "NO_EVIDENCE",
            "case_ids": family_ids, "case_count": len(family_ids),
            "part_count": len({case["part_id"] for case in family_cases}),
            **scores,
            "paired_metric_deltas": (_deltas(scores["incumbent"], scores["challenger"])
                                      if family_ids else {name: None for name in _METRICS}),
        })
    return {
        "schema_version": 1,
        "comparison_scope": "PAIRED_FINITE_SAMPLE_SUPPORT_DEVELOPMENT",
        "reference_contract_status": "DECLARED_NOT_PHYSICALLY_QUALIFIED",
        "reference_frame": reference_frame, "unit": reference_unit, "line_convention": reference_convention,
        "matching_thresholds": {name: getattr(thresholds, name) for name in (
            "angle_deg", "transverse_offset_mm", "gold_span_overlap", "prediction_span_overlap")},
        "expected_cases": cases, "case_count": len(cases), "part_count": len(part_families),
        "declared_families": families, "per_family": family_rows,
        "absent_declared_families": [row["family_id"] for row in family_rows if not row["case_count"]],
        **lanes,
        "paired_metric_deltas": _deltas(lanes["incumbent"]["aggregate"], lanes["challenger"]["aggregate"]),
        "support_diagnostic_targets": {
            name: _diagnostic_targets(lane["aggregate"], family_aggregates[name]) for name, lane in lanes.items()},
        "physical_bend_count": None, "manufacturing_event_completeness": "NOT_ESTABLISHED",
        "evidence_verification": "NOT_PERFORMED", "physical_access_authorized": False,
        "promotion_status": "NOT_AUTHORIZED", "production_use_allowed": False,
        "prospective_qualification_status": "NOT_EVALUATED",
    }
