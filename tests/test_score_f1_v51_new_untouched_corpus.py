from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.score_f1_v51_new_untouched_corpus import TOKEN, score


def _write(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line(bend_id: str, y: float) -> dict:
    return {"bend_id": bend_id, "line_start": [0.0, y, 0.0], "line_end": [100.0, y, 0.0]}


def test_generic_scorer_enforces_30_cases_15_families_and_exact_match(tmp_path: Path) -> None:
    cases = []
    families = {}
    prediction_root = tmp_path / "predictions"
    for family_index in range(15):
        family_id = f"F{family_index + 1:03d}"
        case_ids = []
        for within_family in range(2):
            case_id = f"O{family_index * 2 + within_family + 1:03d}"
            case_ids.append(case_id)
            cases.append({"case_id": case_id, "manual_bends": [_line("G1", 0.0)]})
            _write(
                prediction_root / case_id / "semantic_result.json",
                {
                    "kind": "sherman_qc_f1_v51_single_axis_parallel_return_recovery_result",
                    "source_only": True,
                    "predictions": [_line("P1", 0.0)],
                },
            )
        families[family_id] = case_ids
    truth_path = tmp_path / "truth.json"
    family_path = tmp_path / "families.json"
    truth_sha = _write(truth_path, {"schema_version": 1, "token": TOKEN, "cases": cases})
    family_sha = _write(
        family_path, {"schema_version": 1, "token": TOKEN, "families": families}
    )
    report = score(
        truth_path=truth_path,
        truth_sha256=truth_sha,
        family_path=family_path,
        family_sha256=family_sha,
        prediction_root=prediction_root,
    )
    assert report["aggregate_integers"] == {
        "case_count": 30,
        "family_count": 15,
        "exact_case_count": 30,
        "true_positive": 30,
        "false_positive": 0,
        "false_negative": 0,
    }
    assert all(report["semantic_gate"].values())


def test_generic_scorer_rejects_wrong_sealed_hash(tmp_path: Path) -> None:
    truth_path = tmp_path / "truth.json"
    family_path = tmp_path / "families.json"
    _write(truth_path, {"schema_version": 1, "token": TOKEN, "cases": []})
    family_sha = _write(
        family_path, {"schema_version": 1, "token": TOKEN, "families": {}}
    )
    try:
        score(
            truth_path=truth_path,
            truth_sha256="0" * 64,
            family_path=family_path,
            family_sha256=family_sha,
            prediction_root=tmp_path,
        )
    except ValueError as exc:
        assert "sealed input hash mismatch" in str(exc)
    else:
        raise AssertionError("wrong truth hash was accepted")
