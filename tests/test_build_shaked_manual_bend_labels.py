from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_shaked_manual_bend_labels as labels  # noqa: E402


def test_selection_ledger_has_unique_count_exact_axes() -> None:
    ledger = json.loads(labels.DEFAULT_LEDGER.read_text(encoding="utf-8"))
    assert len(ledger["cases"]) == 12
    assert sum(case["expected_bend_count"] for case in ledger["cases"]) == 59
    for case in ledger["cases"]:
        selected = [str(value) for value in case["selected_ids"]]
        assert len(selected) == case["expected_bend_count"]
        assert len(set(selected)) == len(selected)


def test_build_case_is_deterministic_and_bounded() -> None:
    ledger = json.loads(labels.DEFAULT_LEDGER.read_text(encoding="utf-8"))
    for case in ledger["cases"]:
        first = labels.build_case(
            case,
            scan_root=labels.DEFAULT_SCAN_ROOT,
            review_root=labels.DEFAULT_REVIEW_ROOT,
        )
        second = labels.build_case(
            case,
            scan_root=labels.DEFAULT_SCAN_ROOT,
            review_root=labels.DEFAULT_REVIEW_ROOT,
        )
        assert first == second
        assert first["authoritative_ground_truth"] is False
        assert first["accepted_bend_count"] == first["sender_bend_count"]
        assert len(json.dumps(first)) < 250_000
        for bend in first["manual_bends"]:
            assert "source_candidate" not in bend
            assert "source_points" not in bend
            assert bend["scan_support"]["supported_span_mm"] >= 7.0
            assert bend["frame_integrity_decision"] in {
                "pass",
                "documented_short_bend_override",
            }
