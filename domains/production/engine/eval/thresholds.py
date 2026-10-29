"""Frozen gate thresholds of the E4 evaluation harness (plan §5 and §10 amendments 3-8).

Every number a gate compares against lives in one of the documents below, never
inline in the scorer. Each document is plain JSON-able data with a ``version``;
:func:`thresholds_sha256` hashes its canonical JSON bytes (contracts
``canonical.py``) so the freeze ledger (:mod:`.freeze`) can pin it before the
accuracy loop starts (amendment 4a: "Thresholds never change mid-run"). A run
that is meant to qualify checks the hash it scores with against the ledger.

Documents:

- :data:`G1_THRESHOLDS` -- synthetic math qualification (plan §5 G1, amendment 5).
- :data:`G2_LINE_THRESHOLDS` -- the Gold line-localization metric (G2a,
  amendment 6a); the matcher constants and the aggregate equal
  ``scripts/evaluate_f1_against_manual_gold.py``.
- :data:`G2_PHYSICAL_THRESHOLDS` -- the other physical complete-scan gates
  G2(b-f) and the compile-rate rule of amendment 4d. E4a defines them so the
  pre-A1 freeze can pin them; the wave-3 real-part runner evaluates them.
- :data:`G3_THRESHOLDS` -- physical incomplete scans (plan §5 G3, amendment 7).
- :data:`G4_THRESHOLDS` -- the visual-verification protocol (amendment 8).

Population definitions (which characteristics a gate counts) are stated next to
each number and implemented in :mod:`.scorer` / :mod:`.g4`; ENGINE_EVAL.md
repeats them in prose. Changing any value here changes the hash and therefore
needs a new ledger entry before A1 resumes; after the pre-A1 freeze a changed
document makes every scorecard non-qualifying (:func:`.freeze.check_against_ledger`).
"""
from __future__ import annotations

import copy
from typing import Any

from contracts.production.canonical import sha256_hex

__all__ = [
    "THRESHOLDS_SCHEMA",
    "G1_THRESHOLDS",
    "G2_LINE_THRESHOLDS",
    "G2_PHYSICAL_THRESHOLDS",
    "G3_THRESHOLDS",
    "G4_THRESHOLDS",
    "ALL_THRESHOLDS",
    "thresholds_sha256",
    "thresholds_bundle",
    "GIB",
]

THRESHOLDS_SCHEMA = "sherman_e4_thresholds_v1"
GIB = 1024**3

#: Plan §5 G1 plus amendment 5 (G1 integrity). All rates are fractions in [0, 1].
G1_THRESHOLDS: dict[str, Any] = {
    "schema": THRESHOLDS_SCHEMA,
    "document": "G1",
    "version": "E4-G1-2 (2026-10-17)",
    "source": "PRODUCTION_LINE_PLAN.md §5 G1; §10 amendments 3, 4, 5",
    # Complete parts (every planned bend formed, defined part, no extra bend) scanned
    # with the FULL two-setup robot program at the scanner model's native spacing.
    "accuracy_complete_full_visibility": {
        "FLANGE_LENGTH": {"unit": "mm", "abs_bias_max": 0.05, "p95_abs_error_max": 0.20},
        "INCLUDED_ANGLE": {"unit": "deg", "abs_bias_max": 0.10, "p95_abs_error_max": 0.30},
        "BEND_LINE_POSITION": {"unit": "mm", "abs_bias_max": None, "p95_abs_error_max": 0.30},
    },
    # MEASURED / scorable characteristics of defined parts, FULL plan ("realistic
    # occlusion"), native density. AMBIGUOUS_POSE on an E1-symmetric case is excused.
    "coverage_min": 0.90,
    # Count of false PASS decisions (E2 truth, not the engine's U). A PASS is false when
    # (a) the characteristic is scorable and truth_in_tolerance is False, or (b) it is on
    # E2's never-PASS list (out-of-tolerance sides, every characteristic of a wrong or
    # mirrored part), or (c) its bend is truly NOT_YET_FORMED (E2's never-FAIL list: the
    # true angle deviation is the full sweep, the flange and line do not exist yet).
    "false_pass_max_count": 0,
    "false_pass_rule": "scorable and truth out of tolerance | on E2 characteristics_never_pass | "
                       "on a truly NOT_YET_FORMED bend (E2 characteristics_never_fail)",
    # FAIL while truth_in_tolerance is True, over MEASURED truly-in-tolerance rows.
    "false_fail_max_rate": 0.005,
    # |actual - truth| <= U over MEASURED scorable rows (native density, all plans).
    "u_calibration_min_rate": 0.95,
    # Median U of MEASURED scorable rows on FULL-plan cases, per unit.
    "median_u_full_visibility_max": {"mm": 0.15, "deg": 0.25},
    # PASS or FAIL (not REVIEW) among MEASURED rows whose true deviation is near
    # zero (within near_fraction of the tolerance limit on its side) or far out
    # (beyond far_fraction of it).
    "decisiveness": {"min_rate": 0.90, "near_fraction": 0.5, "far_fraction": 1.5},
    # OLS slope of measured deviation (actual - expected) on true deviation
    # (truth - nominal) per kind, over MEASURED scorable rows. Needs at least
    # min_points rows of which min_large_points have |true dev| >= large_fraction
    # of the tolerance limit on their side; otherwise NOT_EVALUABLE.
    "slope": {"min": 0.95, "max": 1.05, "min_points": 10, "min_large_points": 5, "large_fraction_of_tol": 0.5,
              "kinds": ["FLANGE_LENGTH", "INCLUDED_ANGLE", "BEND_LINE_POSITION", "BACKGAUGE_LENGTH"]},
    # Binary formed/unformed per bend on truth-unambiguous incomplete-part cases; an
    # engine UNKNOWN (incl. AMBIGUOUS_POSE) counts as an error (amendment 5).
    "stage_state_accuracy_min": 0.98,
    # Backgauge sides (E1's side) with |true deviation| >= min_abs_dev_mm must be
    # decided FAIL or REVIEW.
    "backgauge_detection": {"min_abs_dev_mm": 0.5, "min_rate": 1.0},
    # Truth-frame line check (an automated LANDS/TAG_OK analogue used for the
    # confident-wrong-ID gate): a measured OML line is on its own bend when the
    # undirected angle <= angle_deg_max and the symmetric transverse offset <=
    # max(offset_floor_mm, Ri + t) to the bend's true OML sharp.
    "line_check": {"angle_deg_max": 5.0, "offset_floor_mm": 2.0, "offset_adds_ri_plus_t": True},
    # Identity from the registration (works without published lines): D = raw_to_frame
    # (after the unit factor) composed with inv(truth raw_to_cad) must be about the
    # identity. Reference points: the panel outer boundaries at the stage pose. POSE_OK
    # when D rotates <= rotation_deg_max and moves the points <= rms_displacement_mm_max
    # (RMS); POSE_SYMMETRIC_FLIP when D matches an E1 ROTATION self-map of the stage
    # model within the candidate limits (its bend_permutation names the wrong ids);
    # otherwise POSE_WRONG. An identity check, not an accuracy check.
    "pose_check": {"rotation_deg_max": 5.0, "rms_displacement_mm_max": 4.0, "candidate_rotation_deg_max": 5.0,
                   "candidate_rms_mm_max": 4.0, "candidate_kinds": ["ROTATION"],
                   "reference_points": "panel outer boundaries at the stage pose (<= 2000 points)"},
    "runtime": {"p95_wall_s_max": 45.0, "rss_max_bytes": 4 * GIB, "min_points": 900_000},
    # Amendment 3: the regression tier (about 50 cases) runs on every A1 change in at most 20 min.
    "tier_budget_s": {"regression": 1200.0},
    # Standard run conditions (plan §10 amendment 1; ENGINE_IO §4). Anything else is
    # recorded and makes the scorecard non-qualifying.
    "run_conditions": {"engine_argv_tail": ["-m", "domains.production.engine.inspect"],
                       "engine_timeout_s": 180.0, "engine_rss_limit_bytes": 4 * GIB, "engine_time_budget_s": 170.0,
                       "engine_extra_env": "none"},
    # Amendment 4b: the quantities the scanner model card must measure. A MODEL_CARD
    # that keeps an ASSUMED value for any listed field is partial (non-qualifying)
    # unless the freeze ledger holds an owner WAIVER for that model and those fields.
    "scanner_model": {
        "qualifying_provenance": "MODEL_CARD",
        "amendment_4b_quantities": {
            "noise": ["noise_sigma_mm"],
            "point_spacing": ["point_spacing_mm"],
            "edge_flare": ["edge_flare_mm", "edge_flare_probability"],
            "grazing_dropout": ["grazing_start_deg", "grazing_cutoff_deg"],
            "specular_dropout": ["specular_cone_deg", "specular_dropout_probability"],
            "duplicate_skin_offset": ["duplicate_skin_offset_mm"],
        },
        "measured_statuses": ["MEASURED", "UPPER_BOUND", "LOWER_BOUND"],
    },
    "suite": {
        "min_cases": 200,
        "min_families": 7,
        "required_family_features": ["stage_0", "stage_1", "acute", "large_radius", "short_flange", "long_flange"],
        "qualifying_seedset": "QUAL",
        "qualifying_tier": "full",
        "qualifying_scanner_provenance": "MODEL_CARD",
    },
    # A1's per-change signal: the binding gates whose *truth* population exists in the
    # scored tier. A gate whose truth population is empty there (e.g. no degraded scan
    # plans in the regression tier) is NOT_APPLICABLE_TO_TIER and ignored by
    # regression_verdict; an engine-side empty population stays NOT_EVALUABLE.
    "regression_verdict": "binding gates with a non-empty truth population in the tier",
    "bootstrap": {"resamples": 2000, "seed": 20261017, "cluster": "family", "ci": 0.95},
    "percentile_method": "numpy linear interpolation (Hyndman-Fan type 7)",
    "gate_statistic": "point estimate on the full population; family-cluster bootstrap CI reported, not gated",
}

#: Amendment 6a: the Gold line metric, scored in the raw-scan frame.
G2_LINE_THRESHOLDS: dict[str, Any] = {
    "schema": THRESHOLDS_SCHEMA,
    "document": "G2_LINE",
    "version": "E4-G2-LINE-2 (2026-10-17)",
    "source": "PRODUCTION_LINE_PLAN.md §5 G2a; §10 amendment 6a; scripts/evaluate_f1_against_manual_gold.py",
    "matcher": {
        "axis_angle_deg": 5.0,
        "transverse_offset_mm": 4.0,
        "gold_span_overlap": 0.60,
        "prediction_span_overlap": 0.60,
        "matching": "one_to_one_minimum_cost",
    },
    # Aggregate exactly as the script's evaluate(): precision and recall are 0.0 when
    # their denominator is empty, F1 is 0.0 when precision + recall = 0. An engine that
    # emits no line on usable Gold scores F1 = 0 (FAIL); NOT_EVALUABLE only without usable Gold.
    "aggregate": "script evaluate(): p = tp/(tp+fp) or 0.0, r = tp/(tp+fn) or 0.0, f1 = 2pr/(p+r) or 0.0",
    "declared_locus": "OUTER_APEX",
    # The apex is the measured OML sharp moved along the measured bisector by
    # (Ri + t)(1/sin(theta/2) - 1) at the MEASURED included angle (E1's loci use the same formula).
    "apex_rule": "measured OML + measured bisector * |nominal apex offset| * f(theta_measured) / f(theta_nominal), "
                 "f(theta) = 1/sin(theta/2) - 1",
    "diagnostic_loci": ["OML_VIRTUAL_SHARP"],
    "f1_min": 0.80,
    "precision_min": 0.90,
    "transform_residual_max_mm": 0.01,
    "label": "recipe-guided vs blind union, not like-for-like",
}

#: Plan §5 G2(b-f) plus amendments 4d and 6(c, d, f): physical complete scans of the
#: authorized development parts. Evaluated by the wave-3 real-part runner (not E4a).
G2_PHYSICAL_THRESHOLDS: dict[str, Any] = {
    "schema": THRESHOLDS_SCHEMA,
    "document": "G2_PHYSICAL",
    "version": "E4-G2-PHYS-1 (2026-10-17)",
    "source": "PRODUCTION_LINE_PLAN.md §5 G2(b-f); §10 amendments 4d, 4e, 4f, 6",
    "label": "consumed development parts (engineering ledger); never unseen-family accuracy",
    "compile_rate_min": 0.80,  # amendment 4d: of the authorized STEPs; failures stay in every denominator
    "coverage": {"bend_angle_measured_min": 0.90, "flange_side_length_measured_min": 0.70},  # G2b
    "angle_nominal_relative": {  # G2c, known-good parts (ANGLE_TRUTH_CONTROL_SET_20260404 stable + secondary)
        "mae_deg_max": 1.0, "no_worse_than_legacy": True,
        "population": "bends both arms MEASURE (amendment 6c)",
        "engine_coverage_at_least_legacy": True, "report": "per part, with histograms",
    },
    "flange_nominal_relative": {  # G2d
        "p95_abs_mm_max": 1.0, "exclusions": "none (amendment 6d)",
        "large_deviation_mm": 1.0, "large_deviation_evidence": "zoomed section crop each, reported separately",
    },
    "runtime": {"p95_wall_s_max": 45.0, "legacy_control_p95_s": 46.0, "crashes_max": 0,
                "every_failure": "explicit UNKNOWN or INVALID"},  # G2e
    "repeatability": {  # G2f, amendment 6f
        "parts": ["15651004 x3", "42288005 x2", "47959001 (PLY and STL)"],
        "p95_abs_between_repeats_mm_max": 0.3, "p95_abs_between_repeats_deg_max": 0.3,
        "better_than_legacy": True,
    },
    "dev_check": {"family_fraction": "about one third, including one staged family", "seen_by_a1": False,
                  "runs": 1, "gates": "G2 and G3 must also hold on it"},  # amendment 4f
}

#: Plan §5 G3 and amendment 7: physical staged scans against a frozen visual truth table.
G3_THRESHOLDS: dict[str, Any] = {
    "schema": THRESHOLDS_SCHEMA,
    "document": "G3",
    "version": "E4-G3-1 (2026-10-17)",
    "source": "PRODUCTION_LINE_PLAN.md §5 G3; §10 amendment 7",
    "truth_table": {"reviewers": 2, "tiebreak": 1, "labels": ["FORMED", "NOT_FORMED", "NOT_DETERMINABLE"],
                    "built_before_engine_output": True, "frozen": True},
    "state_match_rate_min": 1.0,  # every bend
    "unknown_matches_only": "NOT_DETERMINABLE",
    "unformed_reported_as_defect_max": 0,
    "blank_stage": {"stage": 0, "all_bends": "NOT_YET_FORMED"},
    "formed_count_equals_expected_completed": "where expected_completed exists (scoring only, never an engine input)",
    "formed_bends_meet": ["G2_PHYSICAL.coverage", "G2_PHYSICAL.angle_nominal_relative",
                          "G2_PHYSICAL.flange_nominal_relative"],
}

#: Amendment 8: renders, LANDS/TAG_OK, reviewers and canaries.
G4_THRESHOLDS: dict[str, Any] = {
    "schema": THRESHOLDS_SCHEMA,
    "document": "G4",
    "version": "E4-G4-2 (2026-10-17)",
    "source": "PRODUCTION_LINE_PLAN.md §5 G4; §10 amendment 8",
    "global_views": ["ISO", "TOP", "FRONT", "SIDE"],
    "closeup_min_px_per_mm": 8.0,
    "closeup_px_per_mm": 10.0,
    "scale_bar_mm": 10.0,
    "lands_rule": {"offset": "max(2 mm, Ri + t) from the crease", "offset_floor_mm": 2.0,
                   "min_length_fraction": 0.80, "parallel": True},
    # A LINE_SHIFT moves the line by |s| drawn from [max(shift_mm_min, tol + lands_margin_mm),
    # shift_mm_max], tol = the bend's LANDS tolerance max(2, Ri + t); when that range is empty
    # the close-up gets a LABEL_SWAP instead (or none). Ambiguous-pose close-ups are never canaries.
    "canaries": {"min_fraction": 0.10, "shift_mm_min": 8.0, "shift_mm_max": 15.0, "lands_margin_mm": 3.0,
                 "kinds": ["LINE_SHIFT", "LABEL_SWAP"]},
    "reviewers": {"primary": 2, "tiebreak": 1},
    # AMBIGUOUS is justified only by E1's symmetry test: an UNKNOWN(AMBIGUOUS_POSE) bend is
    # justified iff it is in an ambiguous_bend_groups orbit of the decoded stage model (no
    # reviewer question); otherwise it is unjustified without a reviewer question.
    "ambiguity_rule": "AMBIGUOUS_POSE justified iff the bend is in an E1 ambiguous_bend_groups orbit of the decoded "
                      "stage; without a decoded stage, of the stage whose formed set is the bends not reported "
                      "NOT_YET_FORMED (of any stage model only when none matches)",
    # Close-ups without a measured line that reviewers cannot justify from a picture: reported, not gated.
    "rule_classes_not_reviewer_justified": {
        "REGISTRATION_RULE": "position withheld as REGISTRATION_UNTRUSTED (amendment 9)",
        "UNSUPPORTED_RULE": "bend excluded by E1's per-feature UNSUPPORTED rule (amendment 4d)",
        "INVALID_INSPECTION": "the whole scan is INVALID (judged by G1 on synthetic and G2(e) on physical scans)",
    },
    "gate": {"lands_rate_min": 1.0, "mistag_max": 0, "unknown_justified_rate_min": 1.0,
             # every other close-up without a measured line (NOT_YET_FORMED, UNKNOWN, a formed
             # bend whose position was not measured) must be visually justified
             "no_line_justified_rate_min": 1.0, "canary_flag_rate_min": 1.0,
             "gated_sources": ["E3"], "baseline_sources": ["LEGACY"], "self_test_sources": ["ORACLE"],
             "test_seed_packets_count": False},
    # Plan §5 G4: synthetic parts get the same review on a sample of at least 20 cases,
    # including incomplete ones (physical dev scans: every scan, per the G2/G3 lists).
    "synthetic_sample": {"min_scans": 20, "min_incomplete_scans": 1},
}

ALL_THRESHOLDS: dict[str, dict[str, Any]] = {
    "G1": G1_THRESHOLDS,
    "G2_LINE": G2_LINE_THRESHOLDS,
    "G2_PHYSICAL": G2_PHYSICAL_THRESHOLDS,
    "G3": G3_THRESHOLDS,
    "G4": G4_THRESHOLDS,
}


def thresholds_sha256(doc: dict[str, Any]) -> str:
    """sha256 of the canonical JSON bytes of one threshold document."""
    return sha256_hex(doc)


def thresholds_bundle() -> dict[str, Any]:
    """Deep copy of every document with its hash (what the freeze ledger records)."""
    return {name: {"sha256": thresholds_sha256(doc), "document": copy.deepcopy(doc)} for name, doc in ALL_THRESHOLDS.items()}
