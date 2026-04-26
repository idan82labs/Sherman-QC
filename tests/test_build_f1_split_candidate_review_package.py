import importlib.util
import json
from pathlib import Path

from PIL import Image


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_f1_split_candidate_review_package.py"
    spec = importlib.util.spec_from_file_location("build_f1_split_candidate_review_package", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = _load_module()


def _png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (160, 120), "white").save(path)


def test_builds_split_candidate_template_and_contact_sheet(tmp_path):
    render_dir = tmp_path / "render"
    _png(render_dir / "patch_graph_owned_region_OB2.png")
    _png(render_dir / "patch_graph_owned_region_OB1.png")
    diagnostics = tmp_path / "split.json"
    diagnostics.write_text(
        json.dumps(
            {
                "part_id": "10839000",
                "scan_path": "/tmp/full.ply",
                "capacity_deficit": 1,
                "regions": [
                    {
                        "bend_id": "OB2",
                        "split_candidate": True,
                        "split_candidate_rank": "strong",
                        "max_projection_gap_mm": 12.7,
                        "max_gap_to_median_gap_ratio": 4.2,
                        "largest_gap_left_atom_count": 25,
                        "largest_gap_right_atom_count": 11,
                    },
                    {
                        "bend_id": "OB1",
                        "split_candidate": True,
                        "split_candidate_rank": "moderate",
                        "max_projection_gap_mm": 15.8,
                        "max_gap_to_median_gap_ratio": 2.9,
                        "largest_gap_left_atom_count": 33,
                        "largest_gap_right_atom_count": 3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = builder.build_split_candidate_review_package(
        split_diagnostics_json=diagnostics,
        render_dir=render_dir,
        output_dir=tmp_path / "out",
    )

    assert payload["candidates"] == ["OB2", "OB1"]
    assert Path(payload["contact_sheet"]).exists()
    template = json.loads(Path(payload["label_template"]).read_text())
    assert template["candidate_labels"][0]["bend_id"] == "OB2"
    assert template["candidate_labels"][0]["human_split_label"] is None
    assert "split_into_two_features" in template["split_label_choices"]


def test_focus_path_mapping_ignores_non_ob_files(tmp_path):
    render_dir = tmp_path / "render"
    _png(render_dir / "patch_graph_owned_region_OB2.png")
    _png(render_dir / "patch_graph_owned_region_atom_projection.png")

    mapped = builder._focus_path_by_bend_id(render_dir)

    assert list(mapped) == ["OB2"]
