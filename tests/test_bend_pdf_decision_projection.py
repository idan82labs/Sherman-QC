"""Source-isolated export assembly checks; no PDFs, images or geometry are made.

The real selected functions/class are AST-loaded with flowable, renderer and
filesystem stubs. These checks cover text/metadata/control flow, not ReportLab
layout, fonts, native rendering, image validity or physical generalization.
"""
from __future__ import annotations

import ast
import copy
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import types
from xml.sax.saxutils import escape

import pytest

from contracts.ui.bend_results import BendDecisionContract
from domains.bend.application.finalize_bend_report import finalize_bend_report


SOURCE = Path(__file__).resolve().parents[1] / "backend/bend_report_generator.py"


@dataclass
class Flowable:
    kind: str
    value: object
    style: object = None

    def setStyle(self, style):
        self.style = style


@pytest.fixture
def export_module():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    selected_names = {
        "_export_decision_evidence", "_export_qualification", "_production_use_allowed",
        "_safe_slug", "augment_report_with_overlay_geometry", "_write_json",
        "render_bend_overlay_artifacts", "_BendPDF", "generate_bend_inspection_pdf",
        "build_bend_artifact_bundle",
    }
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected_names
    ]
    assert {node.name for node in selected} == selected_names
    builds = []
    writes = {}
    stamped = []

    class MemoryPath:
        """Path-shaped fixture which never asks the real filesystem anything."""
        def __init__(self, value):
            self.value = str(value)

        def __str__(self):
            return self.value

        def __truediv__(self, value):
            return MemoryPath(self.value.rstrip("/") + "/" + str(value))

        @property
        def name(self):
            return self.value.rsplit("/", 1)[-1]

        def exists(self):
            return self.value in writes

        def mkdir(self, **_kwargs):
            pass

        def write_text(self, value, **_kwargs):
            writes[self.value] = value

        def write_bytes(self, value):
            writes[self.value] = value

    class Document:
        def __init__(self, *_args, **_kwargs):
            pass

        def build(self, story, **callbacks):
            builds.append({"story": story, "callbacks": callbacks})

    class Renderer:
        def render_bend_status_overlay(self, **_kwargs):
            return b"SYNTHETIC-NOT-AN-IMAGE"

    def stamp(value):
        stamped.append(value)
        return b"STAMPED:" + value

    module = types.ModuleType("source_isolated_bend_pdf_export")
    module.__dict__.update({
        "__file__": str(SOURCE), "Path": MemoryPath, "datetime": datetime,
        "json": json, "re": re, "escape": escape, "finalize_bend_report": finalize_bend_report,
        "BendDecisionContract": BendDecisionContract,
        "Mapping": Mapping,
        "colors": types.SimpleNamespace(HexColor=lambda value: value, white="#FFFFFF"),
        "A4": (595.0, 842.0), "mm": 72.0 / 25.4,
        "getSampleStyleSheet": lambda: {key: {} for key in ("Heading1", "Heading2", "BodyText")},
        "ParagraphStyle": lambda *args, **kwargs: kwargs,
        "Paragraph": lambda text, style: Flowable("paragraph", text, style),
        "Table": lambda rows, **kwargs: Flowable("table", rows),
        "TableStyle": lambda style: style,
        "Spacer": lambda *args: Flowable("spacer", args),
        "Image": lambda *args, **kwargs: Flowable("image", args),
        "PageBreak": lambda: Flowable("page-break", None),
        "SimpleDocTemplate": Document, "Model3DSnapshotRenderer": Renderer,
        "_stamp_research_only": stamp,
        "_load_reference_positions": lambda _path: [],
        "_as_point": lambda _value: None,
        "RESEARCH_WATERMARK": "RESEARCH ONLY - NOT PRODUCTION QC",
        "BendArtifactBundle": lambda **kwargs: types.SimpleNamespace(**kwargs),
    })
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(code, str(SOURCE), "exec", dont_inherit=True), module.__dict__)
    module.builds = builds
    module.writes = writes
    module.stamped = stamped
    return module


def _report():
    matches = [
        {"bend_id": "B1", "status": "PASS", "observability_state": "FORMED",
         "target_angle": 90.0, "measured_angle": 89.0, "angle_deviation": -1.0,
         "radius_deviation": 0.25, "line_center_deviation_mm": 2.0,
         "action_item": "UNTRUSTED-MANUFACTURING-INSTRUCTION"},
        {"bend_id": "B2", "status": "FAIL", "observability_state": "UNFORMED"},
        {"bend_id": "B3", "status": "PASS"},
    ]
    return finalize_bend_report({
        "part_id": "SYNTHETIC<&>", "matches": matches,
        "summary": {"completed_bends": 999, "expected_bends": 1000, "remaining_bends": 1},
        "operator_brief": {"headline": "UNSUPPORTED-PRODUCTION-ACCEPTANCE"},
        "qualification_context": {"production_use_allowed": True, "promotion_status": "QUALIFIED", "source": "legacy"},
    })


def _flatten(value):
    if isinstance(value, Flowable):
        return _flatten(value.value)
    if isinstance(value, (list, tuple)):
        return "\n".join(_flatten(item) for item in value)
    return "" if value is None else str(value)


def _build(module, report):
    result = module.generate_bend_inspection_pdf(report, {}, "/synthetic/report.pdf", "/synthetic")
    assert result == "/synthetic/report.pdf"
    assert module.writes == {}, "assembling flowables must not create an artifact in this test"
    return _flatten(module.builds[-1]["story"])


def test_pdf_uses_observed_counts_and_quarantines_model_estimates(export_module):
    report = _report()
    before = copy.deepcopy(report)
    text = _build(export_module, report)
    assert "Observed formed: <b>1/3</b>" in text
    assert "Observed unformed" in text and "Unknown" in text
    assert "Completed:" not in text and "Remaining" not in text
    assert "Model estimate - not observed completion" in text
    assert "Legacy estimate: 999/1000; estimated remaining: 1" in text
    assert "UNSUPPORTED-PRODUCTION-ACCEPTANCE" not in text
    assert "UNTRUSTED-MANUFACTURING-INSTRUCTION" not in text
    assert "Registration: <b>UNVERIFIED</b>" in text
    assert "render evidence: <b>UNAVAILABLE</b>" in text
    assert "NOT QUALIFIED FOR PRODUCTION ACCEPT/REJECT" in text
    assert "90.0 / 89.0" in text and "-1.0°" in text and "+0.25 mm" in text
    assert "Experimental\ncomparison" in text
    assert "SYNTHETIC&lt;&amp;&gt;" in text
    assert report == before


def test_missing_contract_does_not_infer_observations_from_legacy_status(export_module):
    report = _report()
    del report["decision_contract"]
    text = _build(export_module, report)
    assert "Observed formed: <b>Unavailable</b>" in text
    assert "Registration: <b>UNAVAILABLE</b>" in text
    table = export_module._BendPDF()._bend_table(report["matches"], 500)
    assert {row[1] for row in table.value[1:]} == {"Unavailable"}
    assert export_module._production_use_allowed(report) is False


@pytest.mark.parametrize("mutation", ["boolean", "negative", "partition", "partial", "state", "version"])
def test_malformed_contract_is_unavailable_not_coerced(export_module, mutation):
    report = _report()
    decision = report["decision_contract"]
    if mutation == "boolean":
        decision["observed_counts"]["formed"] = True
    elif mutation == "negative":
        decision["observed_counts"]["unknown"] = -1
    elif mutation == "partition":
        decision["observed_counts"]["total"] = 9
    elif mutation == "partial":
        decision["observed_counts"]["partially_observed"] = 4
    elif mutation == "state":
        decision["bends"][0]["state"] = ["FORMED"]
    else:
        decision["schema_version"] = True
    assert export_module._export_decision_evidence(report)["observed_counts"] is None
    assert "Observed formed: <b>Unavailable</b>" in _build(export_module, report)


@pytest.mark.parametrize("mutation", [
    "duplicate_row_id", "empty_row_id", "integer_row_id", "extra_row", "missing_row",
    "state_count_mismatch", "undercounted_total", "known_detail_conflict",
    "abstained_known_state", "authorized", "registration_enum", "render_enum",
    "missing_reasons", "missing_row_reasons", "invalid_reason", "duplicate_raw_id",
    "empty_raw_id", "missing_raw_id", "malformed_raw_row", "missing_matches",
    "countability_mismatch", "partial_count_mismatch",
])
def test_export_rejects_adversarial_contract_and_target_correspondence(export_module, mutation):
    report = _report()
    decision = report["decision_contract"]
    counts = decision["observed_counts"]
    rows = decision["bends"]
    if mutation == "duplicate_row_id":
        rows[1]["bend_id"] = rows[0]["bend_id"]
    elif mutation == "empty_row_id":
        rows[0]["bend_id"] = " "
    elif mutation == "integer_row_id":
        rows[0]["bend_id"] = 1
    elif mutation == "extra_row":
        rows.append({"bend_id": "EXTRA", "state": "UNKNOWN", "observation_detail": "UNKNOWN", "reasons": []})
    elif mutation == "missing_row":
        rows.pop()
        counts.update(unknown=0, total=2)
    elif mutation == "state_count_mismatch":
        counts.update(formed=2, unknown=0)
    elif mutation == "undercounted_total":
        counts.update(unknown=0, total=2)
    elif mutation == "known_detail_conflict":
        rows[0]["observation_detail"] = "OBSERVED_NOT_FORMED"
    elif mutation == "abstained_known_state":
        decision["registration"] = "ABSTAINED"
    elif mutation == "authorized":
        decision["production_disposition"] = "AUTHORIZED"
    elif mutation == "registration_enum":
        decision["registration"] = "VERIFIED"
    elif mutation == "render_enum":
        decision["render_evidence"] = "QUALIFIED"
    elif mutation == "missing_reasons":
        del decision["reasons"]
    elif mutation == "missing_row_reasons":
        del rows[0]["reasons"]
    elif mutation == "invalid_reason":
        decision["reasons"] = [7]
    elif mutation == "duplicate_raw_id":
        report["matches"][1]["bend_id"] = "B1"
    elif mutation == "empty_raw_id":
        report["matches"][0]["bend_id"] = ""
    elif mutation == "missing_raw_id":
        del report["matches"][0]["bend_id"]
    elif mutation == "malformed_raw_row":
        report["matches"][0] = "not a row"
    elif mutation == "missing_matches":
        del report["matches"]
    elif mutation == "countability_mismatch":
        report["matches"][0]["countable_in_regression"] = False
    else:
        rows[2]["observation_detail"] = "PARTIALLY_OBSERVED"
    projected = export_module._export_decision_evidence(report)
    assert projected["kind"] == "BEND_DECISION_EVIDENCE_UNAVAILABLE"
    assert projected["observed_counts"] is None
    assert projected["production_disposition"] == "NOT_AUTHORIZED"
    assert "Observed formed: <b>Unavailable</b>" in _build(export_module, report)
    _, manifest = export_module.augment_report_with_overlay_geometry(report, "/synthetic/not-opened")
    assert manifest["decision_contract"]["observed_counts"] is None
    assert manifest["qualification_context"]["production_use_allowed"] is False


@pytest.mark.parametrize(("top", "nested", "expected_formed", "expected_total"), [
    (None, False, 0, 2),
    (False, True, 0, 2),
    (True, False, 1, 3),
])
def test_valid_countable_subset_matches_frontend_nullish_precedence(
    export_module, top, nested, expected_formed, expected_total,
):
    report = _report()
    report["matches"][0]["countable_in_regression"] = top
    report["matches"][0]["cad_bend"] = {"countable_in_regression": nested}
    report = finalize_bend_report(report)
    decision = export_module._export_decision_evidence(report)
    assert decision["kind"] == "BEND_DECISION_CONTRACT_V1"
    assert decision["observed_counts"]["formed"] == expected_formed
    assert decision["observed_counts"]["total"] == expected_total
    assert len(decision["bends"]) == 3


def test_valid_partial_and_abstained_evidence_is_preserved(export_module):
    report = _report()
    report["matches"][2]["observability_detail_state"] = "PARTIALLY_OBSERVED"
    report = finalize_bend_report(report)
    partial = export_module._export_decision_evidence(report)
    assert partial["observed_counts"]["partially_observed"] == 1
    report["alignment"] = {"alignment_abstained": True}
    report = finalize_bend_report(report)
    abstained = export_module._export_decision_evidence(report)
    assert abstained["registration"] == "ABSTAINED"
    assert abstained["observed_counts"] == {
        "formed": 0, "unformed": 0, "unknown": 3, "total": 3, "partially_observed": 1,
    }


@pytest.mark.parametrize("flag", [0, 1, "false", [], {}])
@pytest.mark.parametrize("location", ["top", "nested"])
def test_malformed_effective_countability_is_unavailable(export_module, flag, location):
    report = _report()
    if location == "top":
        report["matches"][0]["countable_in_regression"] = flag
    else:
        report["matches"][0]["countable_in_regression"] = None
        report["matches"][0]["cad_bend"] = {"countable_in_regression": flag}
    report = finalize_bend_report(report)
    assert export_module._export_decision_evidence(report)["observed_counts"] is None
    assert "Observed formed: <b>Unavailable</b>" in _build(export_module, report)


@pytest.mark.parametrize("top", [True, False])
def test_inactive_nested_malformed_flag_does_not_override_valid_top_level(export_module, top):
    report = _report()
    report["matches"][0]["countable_in_regression"] = top
    report["matches"][0]["cad_bend"] = {"countable_in_regression": "false"}
    report = finalize_bend_report(report)
    projected = export_module._export_decision_evidence(report)
    assert projected["kind"] == "BEND_DECISION_CONTRACT_V1"
    assert projected["observed_counts"]["total"] == (3 if top else 2)


@pytest.mark.parametrize("alignment", [False, 0, "", [], {"alignment_abstained": 0}, {"alignment_abstained": "false"}, {"alignment_abstained": []}])
def test_malformed_raw_alignment_is_unavailable(export_module, alignment):
    report = _report()
    report["alignment"] = alignment
    report = finalize_bend_report(report)
    assert export_module._export_decision_evidence(report)["observed_counts"] is None
    assert "Observed formed: <b>Unavailable</b>" in _build(export_module, report)


def test_raw_abstention_rejects_stale_unverified_decision(export_module):
    report = _report()
    report["alignment"] = {"alignment_abstained": True}
    assert report["decision_contract"]["registration"] == "UNVERIFIED"
    assert export_module._export_decision_evidence(report)["observed_counts"] is None
    report = finalize_bend_report(report)
    assert export_module._export_decision_evidence(report)["registration"] == "ABSTAINED"


@pytest.mark.parametrize("alignment", [None, {}, {"alignment_abstained": None}, {"alignment_abstained": False}])
def test_valid_optional_alignment_does_not_invent_abstention(export_module, alignment):
    report = _report()
    report["alignment"] = alignment
    projected = export_module._export_decision_evidence(report)
    assert projected["registration"] == "UNVERIFIED"
    assert projected["observed_counts"]["formed"] == 1


def test_manifest_carries_evidence_and_overrides_stale_qualification(export_module):
    report = _report()
    _, manifest = export_module.augment_report_with_overlay_geometry(report, "/synthetic/not-opened")
    assert manifest["decision_contract"]["observed_counts"] == report["decision_contract"]["observed_counts"]
    assert manifest["decision_contract"]["production_disposition"] == "NOT_AUTHORIZED"
    assert manifest["qualification_context"]["production_use_allowed"] is False
    assert manifest["qualification_context"]["source"] == "legacy"
    assert report["qualification_context"]["production_use_allowed"] is True
    assert export_module._production_use_allowed(manifest) is False
    assert export_module.writes == {}


@pytest.mark.parametrize("decision", [None, {}, {"kind": "BEND_DECISION_CONTRACT_V1"}, {"kind": "FUTURE", "production_disposition": "AUTHORIZED"}])
def test_no_contract_variant_bypasses_detached_artifact_warning(export_module, decision):
    manifest = {
        "decision_contract": decision,
        "qualification_context": {"production_use_allowed": True},
        "issue_tiles": [{"bend_id": "B1", "image": "synthetic_issue.png"}],
    }
    assert export_module._production_use_allowed(manifest) is False
    export_module.render_bend_overlay_artifacts(manifest, "/synthetic/not-opened", "/synthetic")
    image_values = [value for name, value in export_module.writes.items() if name.endswith(".png")]
    assert len(image_values) == 6
    assert all(value.startswith(b"STAMPED:") for value in image_values)
    assert len(export_module.stamped) == 6


def test_bundle_refreshes_manifest_and_pdf_after_completed_render_only(export_module):
    report = _report()
    bundle = export_module.build_bend_artifact_bundle(report, "/synthetic/not-opened", "/synthetic")
    manifest = json.loads(export_module.writes["/synthetic/bend_overlay_manifest.json"])
    assert bundle.manifest == manifest
    assert manifest["decision_contract"]["render_evidence"] == "UNVERIFIED"
    assert manifest["decision_contract"]["registration"] == "UNVERIFIED"
    assert manifest["decision_contract"]["production_disposition"] == "NOT_AUTHORIZED"
    assert "render evidence: <b>UNVERIFIED</b>" in _flatten(export_module.builds[-1]["story"])
    assert not any(name.endswith(".pdf") for name in export_module.writes)


def test_legacy_bundle_stays_unavailable_even_after_render(export_module):
    report = _report()
    del report["decision_contract"]
    bundle = export_module.build_bend_artifact_bundle(report, "/synthetic/not-opened", "/synthetic")
    assert bundle.manifest["decision_contract"]["observed_counts"] is None
    assert bundle.manifest["decision_contract"]["kind"] == "BEND_DECISION_EVIDENCE_UNAVAILABLE"


def test_missing_render_receipt_cannot_reuse_stale_availability(export_module):
    report = finalize_bend_report(_report(), render_artifacts_available=True)
    export_module.render_bend_overlay_artifacts = lambda **_kwargs: {
        "overview_path": None, "issue_paths": [],
    }
    bundle = export_module.build_bend_artifact_bundle(report, "/synthetic/not-opened", "/synthetic")
    assert bundle.manifest["decision_contract"]["render_evidence"] == "UNAVAILABLE"
    assert "render evidence: <b>UNAVAILABLE</b>" in _flatten(export_module.builds[-1]["story"])


def test_failed_render_does_not_assemble_a_pdf(export_module):
    def failure(**_kwargs):
        raise RuntimeError("synthetic render failure")

    export_module.render_bend_overlay_artifacts = failure
    with pytest.raises(RuntimeError, match="synthetic render failure"):
        export_module.build_bend_artifact_bundle(_report(), "/synthetic/not-opened", "/synthetic")
    assert export_module.builds == []
    assert export_module.writes == {}


def test_pdf_warning_is_registered_for_every_page(export_module):
    _build(export_module, _report())
    callbacks = export_module.builds[-1]["callbacks"]
    assert set(callbacks) == {"onFirstPage", "onLaterPages"}
    emitted = []
    canvas = types.SimpleNamespace(
        saveState=lambda: None, restoreState=lambda: None,
        setFont=lambda *_args: None, setFillColor=lambda *_args: None,
        drawString=lambda _x, _y, text: emitted.append(text),
    )
    for callback in callbacks.values():
        callback(canvas, object())
    assert emitted == ["RESEARCH ONLY - NOT PRODUCTION QC - DISPOSITION NOT AUTHORIZED"] * 2
