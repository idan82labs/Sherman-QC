import json
from pathlib import Path

from research.f1_cad_assisted_comparability_v1 import build


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_frozen_cad_comparison_rejects_cross_corpus_headline_and_recovers_matched_gain():
    result = build(
        multisource=_load("output/research/f1-step-topology-normal-verified-v9-20260822/MULTISOURCE_GOLD11_DIAGNOSTIC.json"),
        line_comparison=_load("output/research/f1-step-topology-normal-verified-v9-20260822/LINE_COMPARISON.json"),
        unified_gate=_load("docs/research/artifacts/2026-08-25_f1_current_lane_separated_unified_gate_v1.json"),
    )

    assert result["headline_cross_corpus_comparison_valid"] is False
    assert result["matched_gold11"]["retained_cad_f1_delta_vs_scan_only"] > 0.09
    assert result["matched_gold11"]["aggressive_tp_delta_vs_retained"] == 6
    assert result["matched_gold11"]["aggressive_fp_delta_vs_retained"] == 21
    assert result["ruling"]["cad_assistance_helped_on_matched_data"] is True
    assert result["ruling"]["pure_cad_guidance_helped_on_matched_data"] is False
    assert result["ruling"]["production_generalization_proven"] is False

