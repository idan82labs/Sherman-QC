import hashlib
import json
from pathlib import Path

from scripts.finalize_f1_v51_new_corpus_gate import TOKEN, finalize


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixtures(tmp_path: Path, *, wrong_marker: bool = False):
    cases = [f"O{i:03d}" for i in range(1, 31)]
    records = []
    for pass_name in ("primary", "replay"):
        records.extend(
            {
                "case_id": case_id,
                "pass": pass_name,
                "point_budget": 200_000,
                "forbidden_cache_keys": [],
                "elapsed_seconds": 12.0,
                "peak_stage_rss_bytes": 512_000_000,
            }
            for case_id in cases
        )
    execution = _write(
        tmp_path / "execution.json",
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v51_new_untouched_corpus_blind_execution",
            "token": TOKEN,
            "primary_success_count": 30,
            "replay_success_count": 30,
            "semantic_replay_identity": {case_id: True for case_id in cases},
            "records": records,
        },
    )
    score = _write(
        tmp_path / "score.json",
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v51_new_untouched_corpus_locked_score",
            "token": TOKEN,
            "aggregate_integers": {"case_count": 30, "family_count": 15, "exact_case_count": 30, "true_positive": 60, "false_positive": 0, "false_negative": 0},
            "aggregate_ratios": {"exact_count_accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "minimum_family_recall": 1.0},
            "cases": [{"case_id": case_id, "gold_count": 2} for case_id in cases],
        },
    )
    artifact = tmp_path / "artifact.png"
    artifact.write_bytes(b"immutable visual")
    audit_rows = []
    for index, case_id in enumerate(cases):
        audit_rows.append(
            {
                "case_id": case_id,
                "prediction_only_path": str(artifact),
                "prediction_only_sha256": _sha(artifact),
                "scored_overlay_path": str(artifact),
                "scored_overlay_sha256": _sha(artifact),
                "axis_stack_path": str(artifact),
                "axis_stack_sha256": _sha(artifact),
                "render_admissible_marker_count": 1 if wrong_marker and index == 0 else 2,
                "reviewer": "auditor-1",
                "ruling": "coherent",
            }
        )
    audit = _write(
        tmp_path / "audit.json",
        {"schema_version": 1, "kind": "sherman_qc_f1_v51_new_untouched_corpus_visual_audit", "token": TOKEN, "cases": audit_rows},
    )
    return execution, score, audit


def test_terminal_gate_passes_all_frozen_requirements(tmp_path):
    execution, score, audit = _fixtures(tmp_path)
    report = finalize(execution, score, audit)
    assert report["passed"] is True
    assert report["metrics"]["render_admissible_marker_accuracy"] == 1.0
    assert all(report["gates"].values())


def test_one_marker_count_miss_fails_point_99_gate(tmp_path):
    execution, score, audit = _fixtures(tmp_path, wrong_marker=True)
    report = finalize(execution, score, audit)
    assert report["passed"] is False
    assert report["metrics"]["render_admissible_marker_accuracy"] == 29 / 30
    assert report["gates"]["render_admissible_marker_accuracy"] is False
