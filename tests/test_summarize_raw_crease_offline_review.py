import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_raw_crease_offline_review.py"
    spec = importlib.util.spec_from_file_location("summarize_raw_crease_offline_review", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


summary = _load_module()


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _quality(part_id, decision, gate="manual_review", reasons=None, safe=0, raw=0):
    return {
        "part_id": part_id,
        "decision": decision,
        "recommended_gate": gate,
        "reason_codes": reasons or [],
        "feature_snapshot": {
            "raw_admissible_before_dedupe": raw,
            "raw_deduped_candidate_count": raw,
            "bridge_safe_candidate_count": safe,
            "bridge_safe_fraction": round(safe / max(raw, 1), 4),
            "safe_flange_pair_count": safe,
            "safe_flange_pair_counts": {},
        },
    }


def _arrangement(part_id, status, selected, safe, dupes=0):
    return {
        "part_id": part_id,
        "status": status,
        "reason_codes": [status],
        "safe_candidate_count": safe,
        "selected_family_count": selected,
        "suppressed_duplicate_count": dupes,
        "selected_candidate_ids": [f"RPC{i}" for i in range(selected)],
        "selected_family_source_candidate_ids": [[f"RPC{i}"] for i in range(selected)],
    }


def _write_case(root, part_id, quality, arrangement):
    _write(root / f"{part_id}_raw_crease_scan_support_quality.json", quality)
    _write(root / f"{part_id}_raw_crease_sparse_arrangement.json", arrangement)


def test_limited_observed_support_holds_exact_count(tmp_path):
    _write_case(
        tmp_path,
        "partial",
        _quality("partial", "limited_observed_support", "scan_quality_gate", ["bridge_safe_below_human_target"], safe=2, raw=5),
        _arrangement("partial", "sparse_arrangement_underfit", selected=1, safe=2),
    )

    report = summary.summarize_raw_crease_offline_review(
        cases=[{"part_id": "partial", "target_count": 5, "track": "partial"}],
        annotations_dir=tmp_path,
    )

    row = report["cases"][0]
    assert row["final_offline_recommendation"] == "hold_exact_count_scan_quality_limited"
    assert "scan_support_below_target" in row["final_recommendation_reasons"]


def test_sparse_exact_candidate_routes_to_arrangement_review(tmp_path):
    _write_case(
        tmp_path,
        "clean",
        _quality("clean", "arrangement_needed", "sparse_arrangement_gate", ["duplicate_flange_pair_support"], safe=6, raw=6),
        _arrangement("clean", "sparse_arrangement_exact_candidate", selected=4, safe=6, dupes=2),
    )

    report = summary.summarize_raw_crease_offline_review(
        cases=[{"part_id": "clean", "target_count": 4, "track": "known"}],
        annotations_dir=tmp_path,
    )

    row = report["cases"][0]
    assert row["final_offline_recommendation"] == "candidate_for_sparse_arrangement_review"
    assert row["selected_family_count"] == 4


def test_existing_exact_is_preserved_when_raw_lane_abstains(tmp_path):
    _write_case(
        tmp_path,
        "exact",
        _quality("exact", "insufficient_support_abstain", "observe_only", ["no_bridge_safe_support"], safe=0, raw=2),
        _arrangement("exact", "no_safe_support", selected=0, safe=0),
    )

    report = summary.summarize_raw_crease_offline_review(
        cases=[{"part_id": "exact", "target_count": 3, "track": "known"}],
        annotations_dir=tmp_path,
        existing_f1_exact_by_part={"exact": 3},
    )

    row = report["cases"][0]
    assert row["final_offline_recommendation"] == "preserve_existing_f1_exact_raw_lane_abstains"
    assert "existing_f1_exact_matches_target" in row["final_recommendation_reasons"]


def test_rolled_or_mixed_transition_blocks_conventional_sparse_promotion(tmp_path):
    _write_case(
        tmp_path,
        "rolled",
        _quality("rolled", "promotion_needs_review", "manual_review", ["bridge_safe_matches_human_target"], safe=2, raw=5),
        _arrangement("rolled", "sparse_arrangement_underfit", selected=1, safe=2, dupes=1),
    )

    report = summary.summarize_raw_crease_offline_review(
        cases=[{"part_id": "rolled", "target_count": 2, "track": "known"}],
        annotations_dir=tmp_path,
        manual_validation={
            "records": [
                {
                    "part_key": "rolled",
                    "part_family": "rolled_or_rounded_mixed_transition",
                    "notes": "rounded part",
                }
            ]
        },
    )

    row = report["cases"][0]
    assert row["part_family"] == "rolled_or_rounded_mixed_transition"
    assert row["final_offline_recommendation"] == "hold_rolled_mixed_transition_semantics_needed"
    assert "rolled_or_mixed_transition_semantics" in row["final_recommendation_reasons"]


def test_markdown_contains_case_table(tmp_path):
    _write_case(
        tmp_path,
        "clean",
        _quality("clean", "arrangement_needed", safe=6, raw=6),
        _arrangement("clean", "sparse_arrangement_exact_candidate", selected=4, safe=6),
    )
    report = summary.summarize_raw_crease_offline_review(
        cases=[{"part_id": "clean", "target_count": 4, "track": "known"}],
        annotations_dir=tmp_path,
    )

    markdown = summary.render_markdown(report)

    assert "| clean | 4 | known | arrangement_needed | sparse_arrangement_exact_candidate | 4 | 6 | 0 | candidate_for_sparse_arrangement_review |" in markdown
