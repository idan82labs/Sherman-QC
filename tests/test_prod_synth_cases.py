"""E2 case plans, per-case outputs, determinism and the CLI."""
from __future__ import annotations

import hashlib
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from contracts.production.canonical import canonical_json_bytes
from contracts.production.part_recipe import PartRecipeV1, ReleaseRequestV1, CadRef, DrawingRef, FlatRef, ReleasedBy
from domains.production.engine.synthetic.cases import (
    FULL_TEMPLATES,
    SEEDSETS,
    TIERS,
    CaseConflictError,
    SeedsetLockedError,
    generate_case,
    generate_tier,
    plan_cases,
    zeiss_scan_filename,
)
from domains.production.engine.synthetic.cli import main
from domains.production.engine.synthetic.families import FAMILIES
from domains.production.engine.synthetic.injection import Scenario
from domains.production.engine.synthetic.ply import read_ply
from domains.production.recipe.builder import build_part_recipe
from domains.production.recipe.sku import parse_scan_filename

FILES = ("part.step", "flat.dxf", "drawing.pdf", "recipe_draft.json", "recipe.json", "scan.ply", "truth.json", "truth_labels.npz", "case.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tier_sizes_families_and_subset_relation():
    reg = plan_cases("regression", "DEV")
    full = plan_cases("full", "DEV")
    assert len(reg) == 50 and len(full) >= 200 and len(full) == 7 * len(FULL_TEMPLATES) + 6
    assert {s.family for s in reg} == set(FAMILIES) == {s.family for s in full}
    assert len(FAMILIES) == 7
    full_by_id = {s.case_id: s for s in full}
    for s in reg:  # regression is a subset of full: same ids, seeds, scenarios
        f = full_by_id[s.case_id]
        assert (f.seed, f.scenario, f.features, f.plan, f.output) == (s.seed, s.scenario, s.features, s.plan, s.output)
    assert all(s.point_target is None for s in reg + full)  # G1 tiers scan at the model's native spacing
    rt = plan_cases("runtime", "DEV")
    assert len(rt) == 10 and all(s.point_target == 1_000_000 for s in rt)  # explicit, recorded override
    assert len(plan_cases("smoke", "DEV")) == 7
    with pytest.raises(ValueError):
        plan_cases("huge", "DEV")
    with pytest.raises(ValueError):
        plan_cases("full", "TEST")


@pytest.mark.parametrize("seedset", sorted(SEEDSETS))
@pytest.mark.parametrize("tier", ["regression", "full"])
def test_every_family_requests_the_amendment5_features(tier, seedset):
    """The templates request every amendment-5 feature per family, for every seed set
    (QUAL is planned only, never built: seeds and template rows, no geometry)."""
    for fam in FAMILIES:
        specs = [s for s in plan_cases(tier, seedset, allow_qual=True) if s.family == fam]
        scenarios = {s.scenario for s in specs}
        tags = {t for s in specs for t in s.features.tags()}
        assert {"STAGE_0", "STAGE_1"} <= scenarios, fam
        assert {"acute", "large_radius", "short_flange", "long_flange"} <= tags, fam


def test_stage_coverage_is_a_property_of_the_scan_not_of_the_recipe_stage(tmp_path, monkeypatch):
    """Amendment-5 coverage must hold by construction (E4's G1.SUITE gate fails on any
    gap): a stage-1 scan is a stage-1 case under a FINAL recipe too, since the recipe's
    inspection stage is a 25 % policy draw; a partial or out-of-order single bend is not."""
    from domains.production.engine.synthetic import cases as cases_mod
    from domains.production.engine.synthetic.cases import _features_present
    from domains.production.engine.synthetic.families import FeatureRequest, build_family

    design = build_family("HAT", np.random.default_rng(0), FeatureRequest())

    def features(**stage):
        block = {"formed_count": 1, "is_sequence_prefix": True, "partially_formed_bend_uids": [], "inspection_stage": "FINAL"}
        block.update(stage)
        return _features_present({"characteristics": {}, "stage": block, "expected": {"symmetric": False}}, design)

    assert features()["stage_1"] and features(inspection_stage="IN_PROCESS")["stage_1"]
    assert not features(is_sequence_prefix=False)["stage_1"]  # one bend formed out of order
    assert not features(partially_formed_bend_uids=["bu_x"])["stage_1"]  # the first bend only partly formed
    assert features(formed_count=0)["stage_0"] and not features(formed_count=0)["stage_1"]
    wrong = features(formed_count=None, is_sequence_prefix=None)  # a wrong part has no stage
    assert not wrong["stage_0"] and not wrong["stage_1"]
    # generated: the policy draw releases a STAGE_1 scan FINAL; it still covers stage 1
    monkeypatch.setattr(cases_mod, "_inspection_stage_policy", lambda sc, rng: replace(sc, inspection_stage="FINAL"))
    entry = generate_case(replace(_spec("E2-DEV-HT-003"), point_target=8_000), tmp_path, ray_threads=1)
    assert entry["scenario"] == "STAGE_1" and entry["inspection_stage"] == "FINAL"
    assert entry["features_present"]["stage_1"] is True
    truth = json.loads((tmp_path / "E2-DEV-HT-003" / "truth.json").read_text())
    assert truth["stage"]["formed_count"] == 1 and truth["stage"]["missing_bend_uids_final"]  # FINAL_INCOMPLETE


def test_dev_and_qual_seeds_are_disjoint():
    seeds = {}
    for seedset in SEEDSETS:  # seeds only (no geometry); QUAL needs the explicit opt-in even for that
        seeds[seedset] = {s.seed for tier in TIERS for s in plan_cases(tier, seedset, allow_qual=True)}
        ids = [s.case_id for tier in ("full", "runtime", "smoke") for s in plan_cases(tier, seedset, allow_qual=True)]
        assert len(ids) == len(set(ids))
    assert not seeds["DEV"] & seeds["QUAL"]
    assert len(seeds["DEV"]) == len({s.seed for tier in ("full", "runtime", "smoke") for s in plan_cases(tier, "DEV")})


def _spec(case_id: str, tier: str = "regression"):
    return next(s for s in plan_cases(tier, "DEV") if s.case_id == case_id)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2cases")
    entries = {}
    for cid in ("E2-DEV-UC-000", "E2-DEV-HT-005", "E2-DEV-HT-006"):  # complete, staged, mirrored
        entries[cid] = generate_case(_spec(cid), root)
    return root, entries


def test_case_directory_contents_and_hash_binding(generated):
    root, entries = generated
    for cid in entries:
        d = root / cid
        for name in FILES:
            assert (d / name).is_file(), name
        case = json.loads((d / "case.json").read_text())
        for name, digest in case["files"].items():
            assert _sha(d / name) == digest, name
        recipe = PartRecipeV1.model_validate_json((d / "recipe.json").read_bytes())
        assert canonical_json_bytes(recipe) == (d / "recipe.json").read_bytes()
        assert recipe.cad.sha256 == _sha(d / "part.step")
        assert recipe.flat is not None and recipe.flat.sha256 == _sha(d / "flat.dxf")
        assert recipe.drawing.sha256 == _sha(d / "drawing.pdf")
        assert recipe.confirmation == "DEV_AUTO_CONFIRMED"
        assert recipe.inspection_stage == case["inspection_stage"]
        truth = json.loads((d / "truth.json").read_text())
        assert [b.bend_uid for b in recipe.bends] == [b["bend_uid"] for b in truth["bends"]]
        assert truth["sku"] == recipe.sku == case["sku"]
        assert '"_tol"' not in (d / "truth.json").read_text()  # internal helper key never leaks


def test_recipe_is_reproducible_from_the_draft(generated):
    root, entries = generated
    for cid in entries:
        draft = json.loads((root / cid / "recipe_draft.json").read_text())
        bi = draft["builder_inputs"]
        rebuilt = build_part_recipe(
            part=bi["part"], spec_rows=bi["spec_rows"], draft=bi["draft"], release=ReleaseRequestV1.model_validate(bi["release"]),
            cad=CadRef.model_validate(bi["cad"]), drawing=DrawingRef.model_validate(bi["drawing"]), version=bi["version"],
            supersedes_version=None, released_by=ReleasedBy(**bi["released_by"]), released_at=bi["released_at"],
            flat=FlatRef.model_validate(bi["flat"]), cad_units_verified=bi["cad_units_verified"],
        )
        assert canonical_json_bytes(rebuilt) == (root / cid / "recipe.json").read_bytes()
        labels = [b["label"] for b in draft["summary"]["bends_in_order"]]
        assert labels == [f"B{i}" for i in range(1, len(labels) + 1)]


def test_scan_labels_and_registration_truth(generated):
    root, entries = generated
    for cid in entries:
        d = root / cid
        pts, faces, comments = read_ply(d / "scan.ply")
        truth = json.loads((d / "truth.json").read_text())
        case = json.loads((d / "case.json").read_text())
        assert (faces is not None) == (case["scan_format"] == "mesh")
        with np.load(d / "truth_labels.npz", allow_pickle=False) as z:
            labels = {k: z[k] for k in z.files}
        assert all(len(v) == len(pts) for v in labels.values())
        raw_to_cad = np.asarray(truth["scan"]["raw_to_cad"])
        assert np.allclose(raw_to_cad[:3, :3].T @ raw_to_cad[:3, :3], np.eye(3), atol=1e-9)
        real = labels["segment_kind"] != 255
        mapped = pts[real] @ raw_to_cad[:3, :3].T + raw_to_cad[:3, 3]
        err = np.linalg.norm(mapped - labels["ideal_cad"][real], axis=1)
        assert float(np.median(err)) < 0.08 and float(np.percentile(err, 99)) < 1.0
        # native density: the export's median nearest-neighbour spacing is the model's
        density = truth["scan"]["density"]
        model = truth["scan"]["scanner_model"]
        assert case["spec"]["point_target"] is None and density["policy"] == "NATIVE_SPACING" and density["override"] is False
        assert abs(density["nn_spacing_mm"] / model["point_spacing_mm"] - 1.0) <= 0.03
        from scipy.spatial import cKDTree

        nn = cKDTree(pts).query(pts, k=2)[0][:, 1]
        assert abs(float(np.median(nn)) / model["point_spacing_mm"] - 1.0) <= 0.035  # independent re-measurement
        # file order carries no label: outliers are not a contiguous tail, setups interleave
        out_idx = np.flatnonzero(labels["segment_kind"] == 255)
        assert len(out_idx) and out_idx[-1] - out_idx[0] > 10 * len(out_idx)
        assert not np.all(np.diff(labels["setup"][real].astype(int)) >= 0)


def test_staged_case_truth(generated):
    root, _ = generated
    truth = json.loads((root / "E2-DEV-HT-005" / "truth.json").read_text())
    st = truth["stage"]
    assert st["inspection_stage"] == "IN_PROCESS" and 0 < st["formed_count"] < len(truth["bends"])
    assert st["expected_stage_model_id"] == f"S{st['formed_count']}"
    assert truth["backgauge"] is not None and truth["backgauge"]["stage_index"] == st["formed_count"]
    states = list(truth["expected"]["bend_states"].values())
    assert states.count("FORMED") == st["formed_count"] and "NOT_YET_FORMED" in states
    assert "ALL_MEASURED_IN_TOLERANCE" in truth["expected"]["summary_never"]


def test_generation_is_deterministic_and_seed_sensitive(generated, tmp_path):
    root, _ = generated
    spec = _spec("E2-DEV-UC-000")
    generate_case(spec, tmp_path)
    for name in FILES:
        assert _sha(root / spec.case_id / name) == _sha(tmp_path / spec.case_id / name), name
    other = replace(spec, case_id="E2-DEV-UC-900", seed=spec.seed + 1)
    generate_case(other, tmp_path)
    assert _sha(tmp_path / other.case_id / "scan.ply") != _sha(tmp_path / spec.case_id / "scan.ply")
    # an existing directory with a different spec is never silently reused
    changed = replace(spec, point_target=50_000)
    with pytest.raises(FileExistsError):
        generate_case(changed, tmp_path)
    # the raycasting thread count is not part of the output (the fixture ran single-threaded)
    generate_case(spec, tmp_path / "threads", ray_threads=4)
    for name in ("scan.ply", "truth.json", "truth_labels.npz"):
        assert _sha(tmp_path / "threads" / spec.case_id / name) == _sha(root / spec.case_id / name), name


def test_zeiss_like_names_parse_with_the_c01_parser(generated):
    root, entries = generated
    names = []
    for cid in entries:
        case = json.loads((root / cid / "case.json").read_text())
        names.append((case["zeiss_filename"], case["sku"], case["revision"]))
    rng = np.random.default_rng(0)
    for stage, insp in ((0, "IN_PROCESS"), (2, "IN_PROCESS"), (None, "FINAL")):
        for rev in ("A", "01"):
            names.append((zeiss_scan_filename("SYN-HT-D123", rev, Scenario("x", stage, insp), rng), "SYN-HT-D123", rev))
    assert any(n.startswith("BEFORE BENDING  ") for n, _, _ in names)
    assert any("_FULL_SCAN" in n for n, _, _ in names) and any("_SCAN_2" in n for n, _, _ in names)
    for name, sku, rev in names:
        parsed = parse_scan_filename(name, {sku: rev})
        assert parsed.status == "OK", (name, parsed)
        assert parsed.sku == sku and parsed.revision_status == "MATCH", (name, parsed)
    # mutate-and-check: a wrong revision in the name is caught by the parser
    assert parse_scan_filename("SYN-HT-D123_B_SCAN_2 1.ply", {"SYN-HT-D123": "A"}).status == "REVISION_MISMATCH"


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def test_cli_guards_qual_and_resumes(tmp_path):
    code, _, err = _run(["generate", "--tier", "smoke", "--seedset", "QUAL", "--out", str(tmp_path / "q")])
    assert code == 2 and "QUAL" in err and not (tmp_path / "q").exists()
    code, out, err = _run(["plan", "--tier", "full", "--seedset", "QUAL", "--json"])  # no seeds without the opt-in
    assert code == 2 and out == "" and "QUAL" in err
    code, out, _ = _run(["part-set", "--seedset", "QUAL", "--out", str(tmp_path / "qps")])
    assert code == 2 and not (tmp_path / "qps").exists()
    code, out, _ = _run(["plan", "--tier", "regression", "--seedset", "DEV"])
    assert code == 0 and len(out.strip().splitlines()) == 50
    out_dir = tmp_path / "smoke"
    code, out, _ = _run(["generate", "--tier", "smoke", "--seedset", "DEV", "--out", str(out_dir), "--cases", "E2-DEV-SMLB-000"])
    assert code == 0 and "RESULT" in out
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["case_count"] == 1 and manifest["seedset"] == "DEV"
    code, out, _ = _run(["generate", "--tier", "smoke", "--seedset", "DEV", "--out", str(out_dir), "--cases", "E2-DEV-SMLB-000"])
    assert code == 0 and '"skipped"' in out
    inbox = tmp_path / "inbox"
    code, out, _ = _run(["export-inbox", "--cases", str(out_dir), "--out", str(inbox)])
    assert code == 0
    case = json.loads((out_dir / "E2-DEV-SMLB-000" / "case.json").read_text())
    exported = inbox / case["zeiss_filename"]
    assert exported.is_file() and exported.read_bytes() == (out_dir / "E2-DEV-SMLB-000" / "scan.ply").read_bytes()
    assert not list(inbox.glob(".*partial"))
    code, _, err = _run(["generate", "--tier", "smoke", "--seedset", "DEV", "--out", str(out_dir), "--families", "NOPE"])
    assert code == 2
    code, out, _ = _run(["scanner-model", "--out", str(tmp_path / "card.json")])
    assert code == 0 and json.loads((tmp_path / "card.json").read_text())["provenance"] == "ASSUMED"
    (tmp_path / "bad_card.json").write_text(json.dumps({"provenance": "MODEL_CARD", "noise_sigma_mm": 0.03}))
    code, _, err = _run(["generate", "--tier", "smoke", "--seedset", "DEV", "--out", str(tmp_path / "bad"),
                         "--scanner-model", str(tmp_path / "bad_card.json")])
    assert code == 2 and "must state every field" in err


def test_case_bytes_do_not_depend_on_the_python_hash_seed(tmp_path):
    """Cross-process determinism: ezdxf (and anything else) must not leak set order."""
    import os
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    digests = []
    for seed in ("1", "2"):
        out = tmp_path / f"run{seed}"
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": f"{repo}:{repo / 'backend'}", "OMP_NUM_THREADS": "1"}
        proc = subprocess.run(
            [sys.executable, "-m", "domains.production.engine.synthetic", "generate", "--tier", "smoke", "--seedset", "DEV",
             "--out", str(out), "--cases", "E2-DEV-SMTD-000", "--quiet"],
            cwd=repo, env=env, capture_output=True, text=True, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        case = out / "E2-DEV-SMTD-000"
        digests.append({p.name: _sha(p) for p in sorted(case.iterdir())})
    assert digests[0] == digests[1]


def test_part_set_for_onboarding_and_e2e(tmp_path):
    from domains.production.engine.synthetic.cases import PART_SET_SCANS, generate_part_set

    doc = generate_part_set("CHANNEL_WITH_RETURNS", "DEV", tmp_path, target_points=40_000)
    root = tmp_path / doc["set_id"]
    for name in ("part.step", "flat.dxf", "drawing.pdf", "recipe.json", "recipe_draft.json", "part_set.json"):
        assert (root / name).is_file()
    recipe = PartRecipeV1.model_validate_json((root / "recipe.json").read_bytes())
    assert recipe.inspection_stage == "FINAL" and recipe.sku == doc["sku"]
    assert [s["name"] for s in doc["scans"]] == list(PART_SET_SCANS) == doc["drop_order"]
    truths = {s["name"]: json.loads((root / s["dir"] / "truth.json").read_text()) for s in doc["scans"]}
    uids = [b.bend_uid for b in recipe.bends]
    for name, t in truths.items():
        assert [b["bend_uid"] for b in t["bends"]] == uids, name  # one part, one recipe
    errors = truths["errors"]
    label = {b["bend_uid"]: b["label"] for b in errors["bends"]}
    flange = [c for c in errors["characteristics"].values()
              if c["kind"] == "FLANGE_LENGTH" and label[c["bend_uid"]] == "B2" and c["true_deviation"] and abs(c["true_deviation"] - 1.5) < 1e-9]
    angle = [c for c in errors["characteristics"].values()
             if c["kind"] == "INCLUDED_ANGLE" and label[c["bend_uid"]] == "B3"]
    assert len(flange) == 1 and flange[0]["truth_in_tolerance"] is False
    assert angle[0]["true_deviation"] == pytest.approx(2.0, abs=1e-12)
    st = truths["stage_k"]["stage"]
    assert st["inspection_stage"] == "FINAL" and st["missing_bend_uids_final"] and st["expected_stage_model_id"] == f"S{st['formed_count']}"
    assert truths["stage_0"]["stage"]["formed_count"] == 0
    assert truths["wrong_part"]["wrong_sku"] is True
    assert (root / "scans/unknown_sku/scan.ply").read_bytes() == (root / "scans/nominal/scan.ply").read_bytes()
    unknown = next(s for s in doc["scans"] if s["name"] == "unknown_sku")
    assert parse_scan_filename(unknown["zeiss_filename"], {doc["sku"]: doc["revision"]}).status == "UNMATCHED"
    for s in doc["scans"]:
        if s["name"] != "unknown_sku":
            assert parse_scan_filename(s["zeiss_filename"], {doc["sku"]: doc["revision"]}).status == "OK", s["zeiss_filename"]
    code, out, _ = _run(["export-inbox", "--cases", str(root), "--out", str(tmp_path / "inbox")])
    assert code == 0 and [e["id"] for e in json.loads(out.split("RESULT\t", 1)[1])["exported"]] == list(PART_SET_SCANS)
    with pytest.raises(ValueError, match="bends"):
        generate_part_set("L_BRACKET", "DEV", tmp_path / "l")


def test_load_case_verifies_hashes(generated, tmp_path):
    import shutil

    from domains.production.engine.synthetic.cases import load_case

    root, _ = generated
    loaded = load_case(root / "E2-DEV-UC-000")
    assert loaded["truth"]["case_id"] == "E2-DEV-UC-000" and "scan.ply" in loaded["paths"]
    copy = tmp_path / "E2-DEV-UC-000"
    shutil.copytree(root / "E2-DEV-UC-000", copy)
    with open(copy / "scan.ply", "ab") as fh:  # tamper
        fh.write(b"\0")
    with pytest.raises(ValueError, match="scan.ply"):
        load_case(copy)
    assert load_case(copy, verify=False)["case"]["case_id"] == "E2-DEV-UC-000"


def test_package_imports_nothing_from_e1_e3_or_legacy():
    """Amendment 5: E2 truth is analytic and E2 imports nothing from E1 (compile) or E3 (inspect)."""
    import ast

    pkg = Path(__file__).resolve().parents[1] / "domains" / "production" / "engine" / "synthetic"
    allowed_roots = {
        "__future__", "argparse", "ast", "collections", "contextlib", "dataclasses", "hashlib", "io", "json", "math", "os",
        "pathlib", "re", "shutil", "sys", "tempfile", "time", "traceback", "typing", "zipfile",
        "numpy", "scipy", "OCP", "open3d", "ezdxf", "reportlab", "pydantic", "contracts", "domains",
    }
    for path in sorted(pkg.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import inside the package
                    continue
                names = [node.module or ""]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                assert root in allowed_roots, f"{path.name}: unexpected import {name}"
                if root == "domains":
                    assert name.startswith("domains.production.recipe"), f"{path.name}: {name} (only the C01 recipe domain)"
                if root == "contracts":
                    assert name.startswith("contracts.production"), f"{path.name}: {name}"


def test_stage_and_as_built_step_outputs(tmp_path):
    """Real STEP BReps for any stage of the tagged order, plus the as-built truth BRep."""
    from domains.production.engine.synthetic import brep
    from domains.production.engine.synthetic.cases import load_case

    spec = _spec("E2-DEV-HT-005")
    generate_case(replace(spec, point_target=30_000), tmp_path, write_as_built_step=True, write_stage_steps=True)
    root = tmp_path / spec.case_id
    loaded = load_case(root)  # hashes of the extra files are verified too
    n = len(loaded["truth"]["bends"])
    stage_files = sorted(k for k in loaded["case"]["files"] if k.startswith("stages/"))
    assert stage_files == [f"stages/S{k}.step" for k in range(n + 1)]
    cylinders = []
    for k in range(n + 1):
        shape = brep.read_step(root / "stages" / f"S{k}.step")
        brep.check_single_valid_solid(shape)
        cylinders.append(brep.face_type_counts(shape)["cylinder"])
    holes = cylinders[0]  # S0: only hole walls are cylinders
    assert cylinders == [holes + 2 * k for k in range(n + 1)]
    brep.check_single_valid_solid(brep.read_step(root / "as_built.step"))


def test_chunked_generation_merges_the_manifest(tmp_path):
    out = tmp_path / "chunks"
    for cid in ("E2-DEV-SMZS-000", "E2-DEV-SMLB-000"):
        code, _, _ = _run(["generate", "--tier", "smoke", "--seedset", "DEV", "--out", str(out), "--cases", cid, "--quiet"])
        assert code == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert [c["case_id"] for c in manifest["cases"]] == ["E2-DEV-SMLB-000", "E2-DEV-SMZS-000"]  # plan order
    assert manifest["case_count"] == 2 and set(manifest["coverage"]) == {"L_BRACKET", "Z_STEP"}


# ---------------------------------------------------------------------------
# resume identity, manifest provenance, blindness, policy guards
# ---------------------------------------------------------------------------


def _card(tmp_path, **changes):
    from domains.production.engine.synthetic.scanner import ScannerModel

    model = replace(ScannerModel(), name="card_measured", provenance="MODEL_CARD", notes="test card", **changes)
    path = tmp_path / f"card_{len(list(tmp_path.glob('card_*.json')))}.json"
    model.to_json(path)
    return model, path


def test_resume_never_reuses_a_case_from_another_scanner_model_or_option_set(tmp_path):
    """Finding: a resumed run must not skip a case generated with another model card
    (the frozen amendment-4b card would silently never reach the scans)."""
    out = tmp_path / "tier"
    spec = _spec("E2-DEV-SMLB-000", tier="smoke")
    first = generate_case(spec, out)
    assert first.get("skipped") is None
    case = json.loads((out / spec.case_id / "case.json").read_text())
    assert case["generation"]["scanner_model_provenance"] == "ASSUMED"
    assert generate_case(spec, out)["skipped"] is True  # same request: reused
    card, card_path = _card(tmp_path, noise_sigma_mm=0.08)
    with pytest.raises(CaseConflictError, match="scanner model"):
        generate_case(spec, out, scanner_model=card)
    with pytest.raises(CaseConflictError):
        generate_case(spec, out, write_stage_steps=True)  # options change the file set
    code, _, err = _run(["generate", "--tier", "smoke", "--seedset", "DEV", "--out", str(out), "--cases", spec.case_id,
                         "--scanner-model", str(card_path), "--quiet"])
    assert code == 2 and "does not match" in err
    old_scan = _sha(out / spec.case_id / "scan.ply")
    regenerated = generate_case(spec, out, scanner_model=card, force=True)
    assert regenerated.get("skipped") is None and _sha(out / spec.case_id / "scan.ply") != old_scan
    case = json.loads((out / spec.case_id / "case.json").read_text())
    truth = json.loads((out / spec.case_id / "truth.json").read_text())
    assert case["generation"]["scanner_model_sha256"] == card.sha256() == truth["scan"]["scanner_model_sha256"]
    assert truth["scan"]["scanner_model"]["provenance"] == "MODEL_CARD"
    assert generate_case(spec, out, scanner_model=card)["skipped"] is True


def test_a_manifest_holds_cases_of_one_scanner_model(tmp_path):
    out = tmp_path / "tier"
    m1 = generate_tier("smoke", "DEV", out, case_ids=["E2-DEV-SMLB-000"])
    card, _ = _card(tmp_path, noise_sigma_mm=0.05)
    with pytest.raises(CaseConflictError, match="another scanner model"):
        generate_tier("smoke", "DEV", out, case_ids=["E2-DEV-SMZS-000"], scanner_model=card)
    assert not (out / "E2-DEV-SMZS-000").exists()  # refused before generating anything
    assert json.loads((out / "manifest.json").read_text())["scanner_model_sha256"] == m1["scanner_model_sha256"]
    # a request mixing a new case with a mismatching existing one is refused before the new one is
    # built (here the new L_BRACKET case comes first in plan order)
    other = tmp_path / "other"
    generate_tier("smoke", "DEV", other, case_ids=["E2-DEV-SMZS-000"])
    with pytest.raises(CaseConflictError):
        generate_tier("smoke", "DEV", other, case_ids=["E2-DEV-SMLB-000", "E2-DEV-SMZS-000"], scanner_model=card)
    assert not (other / "E2-DEV-SMLB-000").exists()
    m2 = generate_tier("smoke", "DEV", out, case_ids=["E2-DEV-SMZS-000"])  # same model: merged
    assert [c["case_id"] for c in m2["cases"]] == ["E2-DEV-SMLB-000", "E2-DEV-SMZS-000"]
    assert {c["scanner_model_sha256"] for c in m2["cases"]} == {m2["scanner_model_sha256"]}
    m3 = generate_tier("smoke", "DEV", out, case_ids=["E2-DEV-SMLB-000", "E2-DEV-SMZS-000"], scanner_model=card, force=True)
    assert m3["scanner_model_sha256"] == card.sha256() and {c["scanner_model_sha256"] for c in m3["cases"]} == {card.sha256()}


def test_model_card_spacing_sets_the_scan_density(generated, tmp_path):
    """Native density follows the card's point spacing, not a point budget."""
    root, _ = generated
    spec = _spec("E2-DEV-UC-000")
    card, _ = _card(tmp_path, point_spacing_mm=0.7)
    generate_case(spec, tmp_path / "coarse", scanner_model=card)
    base = json.loads((root / spec.case_id / "truth.json").read_text())["scan"]
    coarse = json.loads((tmp_path / "coarse" / spec.case_id / "truth.json").read_text())["scan"]
    assert abs(coarse["density"]["nn_spacing_mm"] / 0.7 - 1.0) <= 0.03
    assert abs(base["density"]["nn_spacing_mm"] / 0.46 - 1.0) <= 0.03
    ratio = coarse["point_count"] / base["point_count"]
    assert 0.25 < ratio < 0.65  # ~ (0.46 / 0.7)^2 = 0.43
    # an explicit point-count override is recorded as such
    override = generate_case(replace(spec, case_id="E2-DEV-UC-950", point_target=20_000), tmp_path / "override")
    dens = json.loads((tmp_path / "override" / "E2-DEV-UC-950" / "truth.json").read_text())["scan"]["density"]
    assert dens["policy"] == "TARGET_POINTS_OVERRIDE" and dens["override"] is True and "never score" in dens["note"]
    assert 0.8 * 20_000 <= override["point_count"] <= 1.25 * 20_000


def test_recipe_side_files_name_neither_the_case_nor_its_scenario(generated):
    """Amendment 4 blindness: nothing E3 receives (recipe, CAD, flat) carries the case
    id, the template index or a scenario word; the mapping stays in case.json."""
    import re

    root, entries = generated
    scenario_words = ("COMPLETE_", "STAGE_0", "STAGE_1", "STAGE_K", "PARTIAL", "ORDER_SWAP", "MISSING", "EXTRA_BEND",
                      "MIRROR", "WRONG", "SPRINGBACK")
    skus = set()
    for cid in entries:
        d = root / cid
        case = json.loads((d / "case.json").read_text())
        recipe = PartRecipeV1.model_validate_json((d / "recipe.json").read_bytes())
        assert re.fullmatch(r"SYN-[0-9A-F]{10}", recipe.sku) and recipe.sku == case["sku"]
        skus.add(recipe.sku)
        index_token = f"{case['spec']['template_index']:03d}"
        for field in (recipe.sku, recipe.part_id, recipe.part_name, recipe.recipe_id, recipe.release_note or ""):
            assert cid not in field and "E2-" not in field and index_token not in field.replace(recipe.sku, ""), field
        for name in ("recipe.json", "recipe_draft.json", "flat.dxf", "part.step"):
            text = (d / name).read_bytes().decode("utf-8", errors="replace")
            assert cid not in text, name
            assert not any(w in text for w in scenario_words), name
        assert "case_id" not in json.loads((d / "recipe_draft.json").read_text())
        assert case["case_id"] == cid  # the truth side keeps the mapping
    assert len(skus) == len(entries)


def test_qual_seeds_are_locked_in_the_python_api(tmp_path):
    with pytest.raises(SeedsetLockedError):
        plan_cases("full", "QUAL")
    qual_spec = plan_cases("smoke", "QUAL", allow_qual=True)[0]
    with pytest.raises(SeedsetLockedError):
        generate_case(qual_spec, tmp_path)
    with pytest.raises(SeedsetLockedError):
        generate_tier("smoke", "QUAL", tmp_path / "q")
    from domains.production.engine.synthetic.cases import generate_part_set

    with pytest.raises(SeedsetLockedError):
        generate_part_set("HAT", "QUAL", tmp_path / "ps")
    assert not any(tmp_path.iterdir())  # nothing was generated


def test_internal_generation_errors_exit_3_not_2(tmp_path, monkeypatch):
    """GeometryError / KeyError from inside generation are internal failures (exit 3)."""
    from domains.production.engine.synthetic import cases as cases_mod
    from domains.production.engine.synthetic.geometry import GeometryError

    def boom(*args, **kwargs):
        raise GeometryError("no feasible draw after 50 attempts")

    monkeypatch.setattr(cases_mod, "generate_tier", boom)
    code, _, err = _run(["generate", "--tier", "smoke", "--seedset", "DEV", "--out", str(tmp_path / "x")])
    assert code == 3 and "GeometryError" in err

    def key_boom(*args, **kwargs):
        raise KeyError("internal")

    monkeypatch.setattr(cases_mod, "generate_tier", key_boom)
    assert _run(["generate", "--tier", "smoke", "--seedset", "DEV", "--out", str(tmp_path / "y")])[0] == 3


def test_recipe_inspection_stage_does_not_reveal_the_scenario_class():
    from domains.production.engine.synthetic.cases import _inspection_stage_policy

    rng = np.random.default_rng(0)
    seen = {}
    for name in ("COMPLETE_NOMINAL", "STAGE_K", "PARTIAL_BEND", "MISSING_BEND", "EXTRA_BEND", "MIRRORED"):
        stages = {_inspection_stage_policy(Scenario(name, None, "FINAL"), rng).inspection_stage for _ in range(200)}
        seen[name] = stages
    assert seen["COMPLETE_NOMINAL"] == {"FINAL", "IN_PROCESS"} and seen["STAGE_K"] == {"FINAL", "IN_PROCESS"}
    assert seen["PARTIAL_BEND"] == {"FINAL", "IN_PROCESS"} and seen["MIRRORED"] == {"FINAL", "IN_PROCESS"}
    assert seen["MISSING_BEND"] == {"FINAL"} and seen["EXTRA_BEND"] == {"FINAL"}  # G1: missing/extra in a FINAL scan


def test_order_swap_scenarios_are_always_out_of_order():
    """A swap inside one מהלך group is a legal reorder, never labelled ORDER_SWAP."""
    from domains.production.engine.synthetic.cases import _build_scenario, _out_of_order_swaps
    from domains.production.engine.synthetic.families import build_family, FeatureRequest
    from domains.production.engine.synthetic.injection import order_consistent
    from domains.production.engine.synthetic.truth import build_nominal

    spec = replace(_spec("E2-DEV-UC-000"), scenario="ORDER_SWAP")
    checked = 0
    for fam in ("U_CHANNEL", "HAT", "CHANNEL_WITH_RETURNS", "PANEL_WITH_HOLES_NOTCHES"):
        for seed in range(12):
            design = build_family(fam, np.random.default_rng(seed), FeatureRequest())
            triples = _out_of_order_swaps(design)
            for k, i, j in triples:
                formed = (set(design.sequence[:k]) - {design.sequence[i - 1]}) | {design.sequence[j - 1]}
                assert not order_consistent(design, formed)
            scenario, _ = _build_scenario(replace(spec, family=fam), design, build_nominal(design), np.random.default_rng(seed))
            if scenario.name == "ORDER_SWAP":
                inj = next(x for x in scenario.injections if x.kind == "ORDER_SWAP").params
                assert (scenario.stage, inj["formed_out"], inj["formed_in"]) in triples
                checked += 1
            else:
                assert not triples and scenario.name == "MISSING_BEND"
    assert checked > 20
    # a part whose bends are all one group has no out-of-order swap at all
    grouped = build_family("U_CHANNEL", np.random.default_rng(0), FeatureRequest())
    grouped = replace(grouped, groups=(tuple(grouped.sequence),))
    assert _out_of_order_swaps(grouped) == []


def test_skin_labels_are_contract_skins_of_the_physical_part(generated):
    """truth_labels 'skin' follows the ENGINE_IO §1 rule on the scanned part, for a
    mirrored part too: on the datum panel, contract-skin-0 points lie on the side of
    the datum normal with the larger +Z (ties +Y, then +X)."""
    root, entries = generated
    seen_mirrored = False
    for cid in entries:
        d = root / cid
        truth = json.loads((d / "truth.json").read_text())
        with np.load(d / "truth_labels.npz", allow_pickle=False) as z:
            labels = {k: z[k] for k in z.files}
        datum = truth["part"]["datum_panel"]
        idx = truth["scan"]["label_panels"].index(datum)
        n = np.asarray(next(p["skin0_normal_final"] for p in truth["part"]["panels"] if p["key"] == datum))
        if truth["mirrored"]:
            seen_mirrored = True
            n = np.asarray(truth["mirror_matrix_cad"])[:3, :3] @ n
        for axis in (2, 1, 0):  # contract skin-0 normal of the physical datum
            if abs(n[axis]) >= 1e-6:
                n0 = n if n[axis] > 0 else -n
                break
        on = (labels["segment_kind"] == 0) & (labels["segment_index"] == idx)
        h = labels["ideal_cad"][on].astype(np.float64) @ n0
        sk = labels["skin"][on]
        assert np.any(sk == 0) and np.any(sk == 1), cid
        assert float(h[sk == 0].min()) > float(h[sk == 1].max()) + 0.5 * truth["part"]["thickness_mm"], cid
    assert seen_mirrored
