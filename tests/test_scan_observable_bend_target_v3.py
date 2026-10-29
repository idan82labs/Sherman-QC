import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data/bend_corpus/annotations/scan_observable_bend_target_v3.json"


def test_target_is_observation_equivariant_and_identity_blind():
    payload = json.loads(TARGET.read_text(encoding="utf-8"))
    assert payload["target_id"] == "SCAN_OBSERVABLE_BEND_TRANSITION_V3"
    equivalence = payload["equivalence_rule"]
    assert "identical" in equivalence["statement"].lower()
    assert "provenance" in equivalence["forbidden_requirement"].lower()
    forbidden = set(payload["inference_inputs"]["forbidden"])
    assert "part identity or filename semantics" in forbidden
    assert "good/bad role" in forbidden
    assert "expected bend count" in forbidden
    assert "CAD or drawings" in forbidden
    assert "gold labels or benchmark feedback" in forbidden


def test_target_preserves_prior_benchmarks_and_requires_visual_mismatch_review():
    payload = json.loads(TARGET.read_text(encoding="utf-8"))
    versioning = payload["benchmark_versioning"]
    assert versioning["existing_registry_immutable"] is True
    assert versioning["existing_gold_catalog_immutable"] is True
    order = payload["adjudication"]["order"]
    assert order.index("freeze finite manual axes before detector output is opened") < order.index(
        "run the detector once"
    )
    assert "inspect every false positive and false negative visually" in order[-1]
