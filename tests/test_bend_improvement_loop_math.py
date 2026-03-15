"""Math/regression tests for bend improvement loop selection logic."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bend_improvement_loop as loop  # noqa: E402


def _outcome(
    name: str,
    state: str,
    total: int,
    detected: int,
    passed: int,
    failed: int = 0,
    warnings: int = 0,
    seq: int = 1,
) -> loop.ScanOutcome:
    ratio = float(detected / max(1, total))
    return loop.ScanOutcome(
        config_name=name,
        part_key="P1",
        scan_file=f"{name}_{state}_{seq}.ply",
        state=state,
        sequence=seq,
        total_bends=total,
        cad_total_bends=total,
        detected=detected,
        passed=passed,
        failed=failed,
        warnings=warnings,
        progress_ratio=ratio,
        processing_time_ms=10.0,
    )


def _scan(name: str, state: str, seq: int, mtime: float = 1.0) -> loop.ScanFile:
    return loop.ScanFile(
        path=Path(f"/tmp/{name}.ply"),
        state=state,
        sequence=seq,
        sequence_confident=True,
        mtime=mtime,
    )


def test_bootstrap_mean_ci_singleton_is_exact():
    values = np.asarray([0.73], dtype=np.float64)
    mean, lcb, ucb = loop._bootstrap_mean_ci(values, sample_count=400, seed=42)
    assert mean == 0.73
    assert lcb == 0.73
    assert ucb == 0.73


def test_stable_seed_offset_is_deterministic():
    a = loop._stable_seed_offset("candidate_a", modulus=1_000_000)
    b = loop._stable_seed_offset("candidate_a", modulus=1_000_000)
    c = loop._stable_seed_offset("candidate_b", modulus=1_000_000)
    assert a == b
    assert 0 <= a < 1_000_000
    assert 0 <= c < 1_000_000
    assert a != c


def test_score_candidate_produces_conservative_confidence_metrics():
    outcomes = [
        _outcome("cand", "full", total=10, detected=10, passed=10, seq=1),
        _outcome("cand", "partial", total=10, detected=6, passed=6, seq=1),
        _outcome("cand", "partial", total=10, detected=8, passed=7, failed=1, seq=2),
    ]
    score = loop.score_candidate("cand", outcomes, bootstrap_samples=600, seed=123)

    assert 0.0 <= score.score <= 100.0
    assert 0.0 <= score.completion_recall <= 1.0
    assert 0.0 <= score.in_spec_precision <= 1.0
    assert score.utility_lcb95 <= score.mean_scan_utility <= score.utility_ucb95
    assert score.bootstrap_samples >= 200
    assert score.promotion_eligible is True


def test_regression_gate_rejects_overdetect_regression():
    incumbent = loop.score_candidate(
        "incumbent",
        [
            _outcome("incumbent", "full", total=10, detected=10, passed=10),
            _outcome("incumbent", "partial", total=10, detected=6, passed=6, seq=1),
            _outcome("incumbent", "partial", total=10, detected=7, passed=7, seq=2),
        ],
        bootstrap_samples=500,
        seed=7,
    )
    challenger = loop.score_candidate(
        "challenger",
        [
            _outcome("challenger", "full", total=10, detected=12, passed=10),  # over-detect
            _outcome("challenger", "partial", total=10, detected=9, passed=7, failed=2, seq=1),
            _outcome("challenger", "partial", total=10, detected=10, passed=7, failed=3, seq=2),
        ],
        bootstrap_samples=500,
        seed=8,
    )
    ok, reasons = loop.apply_regression_gate(challenger, incumbent)
    assert ok is False
    assert any("overdetect_penalty_regressed" in r for r in reasons)


def test_regression_gate_allows_clear_improvement():
    incumbent = loop.score_candidate(
        "incumbent",
        [
            _outcome("incumbent", "full", total=10, detected=8, passed=7),
            _outcome("incumbent", "partial", total=10, detected=5, passed=4, failed=1, seq=1),
            _outcome("incumbent", "partial", total=10, detected=6, passed=5, failed=1, seq=2),
        ],
        bootstrap_samples=500,
        seed=1,
    )
    challenger = loop.score_candidate(
        "challenger",
        [
            _outcome("challenger", "full", total=10, detected=10, passed=10),
            _outcome("challenger", "partial", total=10, detected=6, passed=6, seq=1),
            _outcome("challenger", "partial", total=10, detected=8, passed=8, seq=2),
        ],
        bootstrap_samples=500,
        seed=2,
    )
    ok, reasons = loop.apply_regression_gate(challenger, incumbent)
    assert ok is True
    assert reasons == []


def test_select_scans_for_evaluation_prefers_partial_and_full_when_limited():
    scans = [
        _scan("p2", "partial", seq=2, mtime=2.0),
        _scan("f1", "full", seq=99, mtime=3.0),
        _scan("p1", "partial", seq=1, mtime=1.0),
    ]
    chosen = loop._select_scans_for_evaluation(scans, max_scans_per_part=2)
    names = [s.path.stem for s in chosen]
    assert "p1" in names
    assert "f1" in names
    assert len(chosen) == 2


def test_select_scans_for_evaluation_no_limit_returns_all():
    scans = [
        _scan("p1", "partial", seq=1, mtime=1.0),
        _scan("f1", "full", seq=99, mtime=2.0),
    ]
    chosen = loop._select_scans_for_evaluation(scans, max_scans_per_part=0)
    assert len(chosen) == len(scans)
