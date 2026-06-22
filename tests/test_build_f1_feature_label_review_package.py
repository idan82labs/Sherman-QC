import importlib.util
import json
from pathlib import Path

from PIL import Image


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_f1_feature_label_review_package.py"
    spec = importlib.util.spec_from_file_location("build_f1_feature_label_review_package", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = _load_module()


def _png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (160, 120), "white").save(path)


def test_builds_feature_label_template_and_copies_focus_images(tmp_path):
    focus = tmp_path / "source" / "patch_graph_owned_region_OB1.png"
    overview = tmp_path / "source" / "overview.png"
    _png(focus)
    _png(overview)
    blocker_path = tmp_path / "blockers.json"
    blocker_path.write_text(
        json.dumps(
            {
                "blockers": [
                    {
                        "part_key": "10839000",
                        "scan_name": "10839000_01_full_scan.ply",
                        "truth_conventional_bends": 8,
                        "truth_raised_form_features": 2,
                        "truth_total_features": 10,
                        "algorithm_range": [9, 10],
                        "owned_region_diagnostics": [
                            {
                                "bend_id": "OB1",
                                "provisional_role": "candidate_raised_form_or_fragment",
                                "diagnostic_tags": ["small_support"],
                                "angle_deg": 90.0,
                                "support_mass": 100.0,
                                "owned_atom_count": 4,
                                "span_length_mm": 12.0,
                                "incident_flange_ids": ["F1", "F2"],
                                "anchor": [0.0, 0.0, 0.0],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "part_key": "10839000",
                        "scan_path": "/tmp/10839000_01_full_scan.ply",
                        "render_artifacts": {
                            "overview_path": str(overview),
                            "focus_paths": [str(focus)],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = builder.build_feature_label_review_package(
        blocker_diagnostics_json=blocker_path,
        validation_summary_jsons=[summary_path],
        output_dir=tmp_path / "out",
    )

    package = payload["packages"][0]
    template = json.loads(Path(package["label_template"]).read_text())
    assert template["region_labels"][0]["bend_id"] == "OB1"
    assert template["region_labels"][0]["human_feature_label"] is None
    assert "raised_form_feature" in template["label_choices"]
    assert Path(template["region_labels"][0]["focus_image"]).exists()
    assert Path(package["contact_sheet"]).exists()


def test_focus_path_mapping_uses_bend_id_suffix():
    mapped = builder._focus_path_by_bend_id(
        {
            "focus_paths": [
                "/tmp/patch_graph_owned_region_OB8.png",
                "/tmp/patch_graph_owned_region_OB12.png",
            ]
        }
    )

    assert mapped["OB8"].endswith("OB8.png")
    assert mapped["OB12"].endswith("OB12.png")
