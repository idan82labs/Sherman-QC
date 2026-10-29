from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
PACK = ROOT / "output/research/cad-line-ontology-authority-pack-v1"


def test_review_app_is_bound_to_complete_frozen_pack():
    manifest = json.loads((PACK / "MANIFEST.json").read_text())
    html = (PACK / "review_app.html").read_text()
    refs = [event["event_ref"] for case in manifest["cases"] for event in case["events"]]
    assert len(refs) == 63 and len(set(refs)) == 63
    assert "c52c7399cd7b3263e8f2dd3b9191c0dc356b7175a4ea9e5cb1cf5069a89e2c3f" in html
    assert "F1_CAD_LINE_ONTOLOGY_AUTHORITY_DECISION_V1" in html
    assert "MANUFACTURING_AUTHORITY" in html and "COUNTERSIGNING_AUTHORITY" in html
    assert "expected_count" not in html and "gold" not in html.lower()
    assert all((PACK / case["image_relative"]).is_file() for case in manifest["cases"])


def test_review_app_exports_validator_fields_and_enforces_independent_storage():
    html = (PACK / "review_app.html").read_text()
    for field in ("event_ref", "event_status", "line_convention", "rationale", "replacement_line"):
        assert field in html
    assert "sherman-qc-authority-review-v1:${role}" in html
    assert "Do not discuss choices" in html

