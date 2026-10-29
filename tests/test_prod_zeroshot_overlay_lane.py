"""The packaged zs lane (``domains.production.zeroshot``): pin, blindness, CAD priors, bundling. Synthetic only.

No scan is read and the lane's detection is never run here: these checks hash source files, read the C01
example CompiledPart and the pin, and exercise the refusal paths of ``lane_cli`` (which stop before any scan
byte is opened).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "contracts" / "production" / "examples"


FROZEN_CONFIGS = {"zs_lane_v1": "research/zeroshot/zs_lane_v1/config_v1.json",
                  "zs_lane_v3": "research/zeroshot/zs_lane_v3/config_v3.json"}


def test_pin_matches_the_frozen_research_config():
    # the selected pin: zs_lane_v1 by default, zs_lane_v3 when SHERMAN_ZS_LANE_PIN=zs_lane_v3
    from domains.production.zeroshot import pin as PIN

    pin = PIN.load_pin(REPO)
    frozen = json.loads((REPO / FROZEN_CONFIGS[pin["lane"]]).read_text())
    assert pin["code_fingerprint"] == frozen["code_fingerprint"]
    assert pin["model"]["sha256"] == frozen["variants"]["zs"]["model_sha256"]
    assert pin["threshold"] == frozen["variants"]["zs"]["threshold"] == 0.5
    assert pin["cad_priors"]["angle_window"] is None
    if pin["lane"] == "zs_lane_v3":
        assert pin["extra_models"]["z3"]["sha256"] == frozen["variants"]["zs"]["z3_model_sha256"]
        assert pin["lane_params"]["variant"]["fix"] == frozen["variants"]["zs"]["fix"]
        assert pin["lane_params"]["variant"]["post"] == frozen["variants"]["zs"]["post"]
    assert PIN.verify(REPO) == []  # repo files == pinned files; vendored models == frozen models


def test_default_pin_is_v1():
    from domains.production.zeroshot import pin as PIN

    assert PIN.DEFAULT_LANE == "zs_lane_v1" and PIN.PIN_REL == PIN.PIN_RELS["zs_lane_v1"]


def test_pin_verify_fails_closed_on_any_change(tmp_path):
    from domains.production.zeroshot import pin as PIN

    pin = PIN.load_pin(REPO)
    for rel in PIN.pinned_files(pin):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, tmp_path / rel)
    assert PIN.verify(tmp_path) == []
    victim = "research/zeroshot/z3_slicing/slicer.py"
    (tmp_path / victim).write_text((tmp_path / victim).read_text() + "\n# edited\n")
    assert f"CODE:{victim}" in PIN.verify(tmp_path)
    shutil.copyfile(REPO / victim, tmp_path / victim)
    model = tmp_path / pin["model"]["rel_path"]
    model.chmod(0o644)
    model.write_bytes(model.read_bytes() + b"x")
    assert any(b.startswith("MODEL:") for b in PIN.verify(tmp_path))
    model.unlink()
    assert any(b.startswith("MODEL_MISSING:") for b in PIN.verify(tmp_path))


def test_cad_priors_are_rotation_invariant_facts_only():
    from domains.production.zeroshot.lane_cli import cad_facts

    doc = json.loads((EXAMPLES / "compiled_part_v1.example.json").read_text())
    facts = cad_facts(doc)
    assert facts == {"n_bends": 4, "t_mm": 2.0, "K": 1, "max_families": 3}  # 4 parallel bends: K = 1 -> K + 2
    doc["bends"][0]["supported"] = False
    assert cad_facts(doc)["n_bends"] == 3
    doc["bends"][1]["axis"]["direction"] = [1.0, 0.0, 0.0]
    assert cad_facts(doc)["K"] == 2


@pytest.mark.parametrize("name", ["10839000_01_FULL_SCAN 1.ply", "SN0042.ply", "scan_job.ply", "scan.obj"])
def test_lane_refuses_any_scan_but_the_staged_name(tmp_path, name):
    from domains.production.zeroshot.lane_cli import main

    scan = tmp_path / name
    scan.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
    out = tmp_path / "zeroshot" / "zs_lines.json"
    rc = main(["--scan", str(scan), "--compiled", str(EXAMPLES / "compiled_part_v1.example.json"), "--out", str(out)])
    doc = json.loads(out.read_text())
    assert rc == 1 and doc["status"] == "REFUSED_INPUT" and doc["lines"] == []


def test_lane_pin_mismatch_writes_no_lines(tmp_path, monkeypatch):
    from domains.production.zeroshot import lane_cli
    from domains.production.zeroshot import pin as PIN

    monkeypatch.setattr(PIN, "verify", lambda root=None, pin=None: ["CODE:research/zeroshot/z3_slicing/slicer.py"])
    (tmp_path / "input").mkdir()
    scan = tmp_path / "input" / "scan.ply"
    scan.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
    out = tmp_path / "zeroshot" / "zs_lines.json"
    rc = lane_cli.main(["--scan", str(scan), "--compiled", str(EXAMPLES / "compiled_part_v1.example.json"), "--out", str(out)])
    doc = json.loads(out.read_text())
    assert rc == 1 and doc["status"] == "PIN_MISMATCH" and doc["lines"] == []


def test_overlay_cli_writes_nothing_without_ok_lines(tmp_path):
    from domains.production.zeroshot.overlay_cli import main

    out = tmp_path / "zeroshot_overlay_v1.json"
    args = ["--result", str(EXAMPLES / "shopfloor_result_v1.example.json"), "--compiled",
            str(EXAMPLES / "compiled_part_v1.example.json"), "--zs", str(tmp_path / "missing.json"), "--out", str(out)]
    assert main(args) == 1 and not out.exists()
    (tmp_path / "zs.json").write_text(json.dumps({"status": "PIN_MISMATCH", "lines": []}))
    args[5] = str(tmp_path / "zs.json")
    assert main(args) == 1 and not out.exists()


def test_bundles_carry_exactly_the_pinned_lane():
    from apps.production_line import engine_bundle as EB
    from domains.production.zeroshot import pin as PIN

    warnings: list[str] = []
    files = EB._zeroshot_files(REPO, warnings)
    pin = PIN.load_pin(REPO, lane=PIN.DEFAULT_LANE)   # the bundle carries the default (live) pin
    assert sorted(files) == PIN.pinned_files(pin) and not warnings
    assert all(f.endswith(".py") or f in (EB.ZEROSHOT_PIN_REL, pin["model"]["rel_path"]) for f in files)
    assert not any(f.endswith(".json") and f.startswith("research/") for f in files)
    assert "domains/production/zeroshot/lane_cli.py" in EB.ZEROSHOT_ENTRY_FILES
    assert "domains/production/zeroshot/overlay_cli.py" in EB.ZEROSHOT_ENTRY_FILES


def test_wiring_enables_the_zs_step_only_with_a_bundled_lane(tmp_path):
    from types import SimpleNamespace

    from apps.production_line.wiring import runner_config_for

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    settings = SimpleNamespace(engine_bundle_dir=bundle, python_exe="/usr/bin/python3", data_root=tmp_path / "data",
                               job_timeout_s=180, rss_limit_bytes=4 * 1024**3, postprocess_timeout_s=120,
                               program_status_path=tmp_path / "status.json")
    assert runner_config_for(settings).zeroshot_module is None
    (bundle / "domains/production/zeroshot").mkdir(parents=True)
    (bundle / "domains/production/zeroshot/lane_cli.py").write_text("")
    cfg = runner_config_for(settings)
    assert cfg.zeroshot_module == "domains.production.zeroshot.lane_cli"
    assert cfg.zeroshot_rss_limit_bytes == cfg.engine_rss_limit_bytes
