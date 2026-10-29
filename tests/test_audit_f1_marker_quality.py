import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_f1_marker_quality.py"
    spec = importlib.util.spec_from_file_location("audit_f1_marker_quality", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def _write_case(
    root: Path,
    *,
    part_id: str = "demo",
    bends=None,
    owned_regions=None,
    adjacency=None,
):
    case_dir = root / part_id
    render_dir = case_dir / "render"
    render_dir.mkdir(parents=True)
    if bends is None:
        bends = [
            {
                "bend_id": "OB1",
                "anchor": [0.0, 0.0, 0.0],
                "metadata": {
                    "render_anchor_source": "owned_atom_medoid",
                    "owned_support_points": [[0.0, 0.0, 0.0]],
                    "owned_atom_ids": ["A1"],
                    "incident_flange_ids": ["F1", "F2"],
                },
            }
        ]
    if owned_regions is None:
        owned_regions = [
            {
                "bend_id": "OB1",
                "incident_flange_ids": ["F1", "F2"],
                "owned_atom_ids": ["A1"],
            }
        ]
    (render_dir / "owned_region_manifest.json").write_text(json.dumps({"bends": bends}), encoding="utf-8")
    (case_dir / "decomposition_result.json").write_text(
        json.dumps(
            {
                "part_id": part_id,
                "atom_graph": {
                    "local_spacing_mm": 2.0,
                    "adjacency": adjacency or {"A1": []},
                },
                "owned_bend_regions": owned_regions,
            }
        ),
        encoding="utf-8",
    )
    return case_dir / "decomposition_result.json"


def _write_summary(path: Path, *, exact=2, render_count=1, source="control_guard", accepted_marker_count=None, semantics=None):
    candidate = {
        "exact_bend_count": exact,
        "bend_count_range": [exact, exact],
        "candidate_source": source,
        "render_owned_bend_region_count": render_count,
    }
    if accepted_marker_count is not None:
        candidate["accepted_render_marker_count"] = accepted_marker_count
    if semantics is not None:
        candidate["accepted_render_semantics"] = semantics
    (path.parent / "summary.json").write_text(
        json.dumps({"candidate": candidate}),
        encoding="utf-8",
    )


def test_marker_quality_audit_passes_clean_case(tmp_path):
    _write_case(tmp_path)

    report = audit.audit_marker_quality(paths=[tmp_path])

    assert report["case_count"] == 1
    assert report["issue_count"] == 0
    assert report["cases"][0]["status"] == "pass"


def test_marker_quality_audit_flags_floating_marker(tmp_path):
    _write_case(
        tmp_path,
        bends=[
            {
                "bend_id": "OB1",
                "anchor": [3.0, 0.0, 0.0],
                "metadata": {
                    "render_anchor_source": "centroid",
                    "owned_support_points": [[0.0, 0.0, 0.0]],
                    "owned_atom_ids": ["A1"],
                    "incident_flange_ids": ["F1", "F2"],
                },
            }
        ],
    )

    report = audit.audit_marker_quality(paths=[tmp_path])

    issue_names = {issue["issue"] for issue in report["issues"]}
    assert "anchor_source_not_owned_atom_medoid" in issue_names
    assert "marker_anchor_not_on_owned_support" in issue_names


def test_marker_quality_audit_flags_duplicate_marker_cluster(tmp_path):
    _write_case(
        tmp_path,
        bends=[
            {
                "bend_id": "OB1",
                "anchor": [0.0, 0.0, 0.0],
                "metadata": {
                    "render_anchor_source": "owned_atom_medoid",
                    "owned_support_points": [[0.0, 0.0, 0.0]],
                    "owned_atom_ids": [f"A{i}" for i in range(20)],
                    "incident_flange_ids": ["F1", "F2"],
                },
            },
            {
                "bend_id": "OB2",
                "anchor": [8.0, 0.0, 0.0],
                "metadata": {
                    "render_anchor_source": "owned_atom_medoid",
                    "owned_support_points": [[8.0, 0.0, 0.0]],
                    "owned_atom_ids": ["B1", "B2", "B3"],
                    "incident_flange_ids": ["F2", "F1"],
                },
            },
        ],
        owned_regions=[
            {
                "bend_id": "OB1",
                "incident_flange_ids": ["F1", "F2"],
                "owned_atom_ids": [f"A{i}" for i in range(20)],
            },
            {
                "bend_id": "OB2",
                "incident_flange_ids": ["F1", "F2"],
                "owned_atom_ids": ["B1", "B2", "B3"],
            },
        ],
    )

    report = audit.audit_marker_quality(paths=[tmp_path])

    issue_names = {issue["issue"] for issue in report["issues"]}
    assert "selected_pair_duplicate_marker_cluster" in issue_names


def test_marker_quality_audit_flags_small_cross_pair_marker_cluster(tmp_path):
    _write_case(
        tmp_path,
        bends=[
            {
                "bend_id": "OB1",
                "anchor": [0.0, 0.0, 0.0],
                "metadata": {
                    "render_anchor_source": "owned_atom_medoid",
                    "owned_support_points": [[0.0, 0.0, 0.0]],
                    "owned_atom_ids": ["A1", "A2", "A3", "A4", "A5"],
                    "incident_flange_ids": ["F1", "F2"],
                },
            },
            {
                "bend_id": "OB2",
                "anchor": [32.0, 0.0, 0.0],
                "metadata": {
                    "render_anchor_source": "owned_atom_medoid",
                    "owned_support_points": [[32.0, 0.0, 0.0]],
                    "owned_atom_ids": ["B1", "B2", "B3", "B4"],
                    "incident_flange_ids": ["F3", "F4"],
                },
            },
        ],
        owned_regions=[
            {
                "bend_id": "OB1",
                "incident_flange_ids": ["F1", "F2"],
                "owned_atom_ids": ["A1", "A2", "A3", "A4", "A5"],
            },
            {
                "bend_id": "OB2",
                "incident_flange_ids": ["F3", "F4"],
                "owned_atom_ids": ["B1", "B2", "B3", "B4"],
            },
        ],
    )

    report = audit.audit_marker_quality(paths=[tmp_path], tiny_atom_count_max=5)

    issue_names = {issue["issue"] for issue in report["issues"]}
    assert "tiny_cross_pair_marker_cluster" in issue_names


def test_marker_quality_audit_flags_accepted_exact_render_mismatch(tmp_path):
    result_path = _write_case(tmp_path)
    _write_summary(result_path, exact=2, render_count=1)

    report = audit.audit_marker_quality(paths=[tmp_path])

    issue_names = {issue["issue"] for issue in report["issues"]}
    assert "accepted_exact_not_rendered_by_owned_regions" in issue_names


def test_marker_quality_audit_allows_scan_context_accepted_render_for_guarded_exact(tmp_path):
    result_path = _write_case(tmp_path)
    _write_summary(
        result_path,
        exact=2,
        render_count=1,
        accepted_marker_count=0,
        semantics="scan_context_only_raw_f1_debug_not_final",
    )

    report = audit.audit_marker_quality(paths=[tmp_path])

    issue_names = {issue["issue"] for issue in report["issues"]}
    assert "accepted_exact_not_rendered_by_owned_regions" not in issue_names
