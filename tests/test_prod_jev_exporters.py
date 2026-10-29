"""C10 Jev harness: onboarding and scan exporters, custody, determinism and leakage guards.

Synthetic records only. No file under the repo's data/ or research/ siblings is read
(the F1 parity test parses only the selector's *source* file, and skips if absent).
"""
from __future__ import annotations

import ast
import copy
import json
import os
import re
from pathlib import Path

import httpx
import pytest

from research.jev import export_onboarding_v1 as onb
from research.jev import export_scan_candidates_v1 as scan
from research.jev import state_v1 as S
from research.jev.labels_v1 import LabelRow, read_jsonl

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _refuse(*_a, **_k):
        raise AssertionError("network access is forbidden in C10 tests")

    monkeypatch.setattr(httpx.Client, "send", _refuse)


def _states(out: Path) -> list[dict]:
    return [r["state"] for r in read_jsonl(out / "states.jsonl")]


def _labels(out: Path) -> dict[str, LabelRow]:
    return {r["record_id"]: LabelRow.from_json(r) for r in read_jsonl(out / "labels.jsonl")}


def _state_by_id(out: Path) -> dict[str, dict]:
    return {r["record_id"]: r["state"] for r in read_jsonl(out / "states.jsonl")}


# --------------------------------------------------------------------------
# Recursive deny-list scan over every exported state
# --------------------------------------------------------------------------

_FORBIDDEN_TEXT = ("SYN-", "PART-REAL", "secret", ".ply", ".step", "bu_", "cad:", "manual:", "job-",
                   "crease-", "CAD_B", "DET_", "case-a", "2026-10", "reference_mesh_raw", "cad_model_mm")


def _walk(node, path="$"):
    if isinstance(node, dict):
        for k, v in node.items():
            yield path, k, v
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        raise AssertionError(f"list found at {path}")


def _assert_clean(states: list[dict]) -> None:
    forbidden_keys = S.INFERENCE_FORBIDDEN_KEYS | S.GEOMETRY_FORBIDDEN_KEYS
    for state in states:
        text = json.dumps(state)
        for needle in _FORBIDDEN_TEXT:
            assert needle not in text, needle
        for path, key, value in _walk(state):
            assert key not in forbidden_keys, f"{path}.{key}"
            assert not any(t in S.DENY_TOKENS for t in key.split("_")) or (path == "$" and key == "schema"), key
            assert not isinstance(value, (list, tuple)), f"{path}.{key}"
            if isinstance(value, str):
                assert len(value) <= 64 and not re.search(r"\d{4,}", value), value


@pytest.fixture()
def onboarding_input(tmp_path):
    return onb.write_synthetic_inputs(tmp_path / "in")


def test_onboarding_export_labels_and_recall_rows(tmp_path, onboarding_input):
    result = onb.export_onboarding([onboarding_input], tmp_path / "out")
    labels = _labels(tmp_path / "out")
    sources = {}
    for row in labels.values():
        sources[row.label_source] = sources.get(row.label_source, 0) + 1
    # 3 recipes x (4 confirmed, 1 not_a_bend, 1 wrong_location, 1 unreviewed, 1 unmarked graph, 1 manual miss)
    assert sources == {"operator_confirmed": 12, "operator_removed_not_a_bend": 3, "operator_wrong_location": 3,
                       "unreviewed": 3, "e1_unmarked_graph_bend": 3, "operator_manual_missing": 3}
    states = _state_by_id(tmp_path / "out")
    for rid, row in labels.items():
        assert (rid in states) == row.has_state
    assert result.manifest["counts"]["manual_corrections"] == 3
    assert result.manifest["authorization_sha256"] is None
    _assert_clean(list(states.values()))


def test_two_export_runs_are_byte_identical(tmp_path, onboarding_input):
    onb.export_onboarding([onboarding_input], tmp_path / "a")
    onb.export_onboarding([onboarding_input], tmp_path / "b")
    for name in ("states.jsonl", "labels.jsonl", "manifest.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name


def test_export_refuses_an_existing_out_dir_and_leaves_no_partial(tmp_path, onboarding_input):
    (tmp_path / "out").mkdir()
    with pytest.raises(onb.ExportError, match="already exists"):
        onb.export_onboarding([onboarding_input], tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"recipe": {"schema": "nope"}}) + "\n")
    with pytest.raises(onb.ExportError):
        onb.export_onboarding([bad], tmp_path / "out2")
    assert not (tmp_path / "out2").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".out2")]


def test_positive_and_negative_candidates_share_one_descriptor_schema(tmp_path, onboarding_input):
    """No presence leak: confirmed and removed CAD candidates have identical key sets."""
    onb.export_onboarding([onboarding_input], tmp_path / "out")
    states, labels = _state_by_id(tmp_path / "out"), _labels(tmp_path / "out")
    shapes: dict[str, set] = {}
    contexts = set()
    for rid, state in states.items():
        if state["candidate_source"] != "CAD_ONBOARDING":
            continue
        shape = tuple(sorted((blk, tuple(sorted(state.get(blk, {})))) for blk in S.TOP_LEVEL_BLOCKS))
        shapes.setdefault(labels[rid].label_source, set()).add(shape)
        contexts.add(json.dumps(state["context"], sort_keys=True))
    pos, neg = shapes["operator_confirmed"], shapes["operator_removed_not_a_bend"]
    assert pos == neg, (pos ^ neg)
    assert len(contexts) == 1  # same part-level context for every candidate of a part (seeds share t)


def test_unmarked_graph_rows_do_not_change_cad_candidate_states(tmp_path, onboarding_input):
    onb.export_onboarding([onboarding_input], tmp_path / "with")
    onb.export_onboarding([onboarding_input], tmp_path / "without", include_unmarked_graph=False)
    a, b = _state_by_id(tmp_path / "with"), _state_by_id(tmp_path / "without")
    assert set(b) < set(a)
    for rid, state in b.items():
        assert a[rid] == state


def test_dev_auto_confirmed_recipe_gives_weak_labels_only(tmp_path):
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(onb.synthetic_onboarding_bundle(5, sku="SYN-DEV", confirmation="DEV_AUTO_CONFIRMED")))
    onb.export_onboarding([path], tmp_path / "out")
    labels = _labels(tmp_path / "out").values()
    assert not [r for r in labels if r.label is not None]
    assert sum(1 for r in labels if r.weak_label_source == "dev_auto_confirmed_active") == 4


@pytest.mark.parametrize("change", [{"confirmation": "SOMETHING_ELSE"}, {"confirmation": "operator_confirmed"},
                                    {"released_at": None}, {"released_at": ""}])
def test_truth_fails_closed_on_untrusted_or_unreleased_recipes(tmp_path, change):
    bundle = onb.synthetic_onboarding_bundle(3)
    bundle["recipe"].update(change)
    if change.get("released_at", 1) is None:
        del bundle["recipe"]["released_at"]
    path = tmp_path / "b.json"
    path.write_text(json.dumps(bundle))
    result = onb.export_onboarding([path], tmp_path / "out")
    labels = list(_labels(tmp_path / "out").values())
    assert labels and all(r.label is None and r.weak_label is None for r in labels)
    assert all(r.has_state for r in labels)  # no recall rows from an untrusted recipe
    assert any(w.endswith(("UNTRUSTED_CONFIRMATION", "RECIPE_NOT_RELEASED")) for w in result.manifest["warnings"])


def test_missing_confirmation_defaults_to_operator_confirmed(tmp_path):
    bundle = onb.synthetic_onboarding_bundle(3)
    del bundle["recipe"]["confirmation"]
    path = tmp_path / "b.json"
    path.write_text(json.dumps(bundle))
    onb.export_onboarding([path], tmp_path / "out")
    assert sum(1 for r in _labels(tmp_path / "out").values() if r.label == 1) == 5  # 4 confirmed + 1 recall


@pytest.mark.parametrize("flags,error", [(["holdout"], onb.CustodyError), (["Sealed"], onb.CustodyError),
                                         (["one_shot", "consumed"], onb.CustodyError),
                                         (["whatever"], onb.ExportError), ("consumed", onb.ExportError)])
def test_bundle_provenance_flags_are_restricted(tmp_path, flags, error):
    bundle = onb.synthetic_onboarding_bundle(4)
    bundle["provenance_flags"] = flags
    path = tmp_path / "b.json"
    path.write_text(json.dumps(bundle))
    with pytest.raises(error):
        onb.export_onboarding([path], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_scan_export_refuses_excluding_provenance_flags(tmp_path, jobs):
    with pytest.raises(onb.CustodyError):
        scan.export_scan(tmp_path / "o", jobs_root=jobs, provenance_flags=["holdout"])


def test_recipe_without_draft_marks_exports_positives_only_with_a_warning(tmp_path):
    bundle = onb.synthetic_onboarding_bundle(1, with_compiled=False)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(bundle["recipe"]))
    result = onb.export_onboarding([path], tmp_path / "out")
    assert any("NO_NEGATIVES" in w for w in result.manifest["warnings"])
    assert result.manifest["counts"]["removed_without_geometry"] == 2
    labels = [r for r in _labels(tmp_path / "out").values() if r.has_state]
    assert {r.label_source for r in labels} == {"operator_confirmed"}


def test_flange_graph_relation_is_present_only_with_a_compiled_part(tmp_path):
    for with_compiled in (True, False):
        path = tmp_path / f"b{with_compiled}.json"
        path.write_text(json.dumps(onb.synthetic_onboarding_bundle(2, with_compiled=with_compiled)))
        onb.export_onboarding([path], tmp_path / f"o{with_compiled}")
        states = _states(tmp_path / f"o{with_compiled}")
        assert all(("cad_relation" in st) == with_compiled for st in states)


# --------------------------------------------------------------------------
# Custody
# --------------------------------------------------------------------------


@pytest.fixture()
def fake_repo(tmp_path):
    repo = tmp_path / "repo"
    for d in ("data", "outputs", "research/other", "research/jev", "free"):
        (repo / d).mkdir(parents=True)
    for i, d in enumerate(("data", "outputs", "research/other", "research/jev", "free")):
        (repo / d / "b.json").write_text(json.dumps(onb.synthetic_onboarding_bundle(i, sku=f"SYN-R{i}")))
    return repo


@pytest.mark.parametrize("sub", ["data", "outputs", "research/other"])
def test_protected_repo_dirs_need_authorization(fake_repo, tmp_path, sub):
    with pytest.raises(onb.CustodyError, match="authorization"):
        onb.export_onboarding([fake_repo / sub / "b.json"], tmp_path / "o", repo_root=fake_repo)
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"schema": onb.AUTHORIZATION_SCHEMA, "authorized_roots": [str(fake_repo / sub)]}))
    r = onb.export_onboarding([fake_repo / sub / "b.json"], tmp_path / "o", repo_root=fake_repo, authorization=auth)
    assert r.manifest["authorization_sha256"] and len(r.manifest["authorization_sha256"]) == 64
    # An authorization for another root does not cover this one.
    auth.write_text(json.dumps({"schema": onb.AUTHORIZATION_SCHEMA, "authorized_roots": [str(fake_repo / "free")]}))
    with pytest.raises(onb.CustodyError, match="authorized_roots"):
        onb.export_onboarding([fake_repo / sub / "b.json"], tmp_path / "o2", repo_root=fake_repo, authorization=auth)


@pytest.mark.parametrize("variant", ["DATA", "Data", "OUTPUTS", "Research/other"])
def test_case_variants_of_protected_dirs_are_refused(fake_repo, tmp_path, variant):
    """macOS volumes are case-insensitive: DATA/ is data/ and must not slip through."""
    path = fake_repo / variant / "b.json"
    with pytest.raises(onb.CustodyError, match="authorization"):
        onb.check_input_path(path, repo_root=fake_repo)
    with pytest.raises(onb.CustodyError):
        onb.export_onboarding([path], tmp_path / "o", repo_root=fake_repo)
    with pytest.raises(onb.CustodyError):
        onb.check_input_dir(path.parent, repo_root=fake_repo)


def test_case_variant_repo_root_prefix_is_refused(fake_repo):
    variant = Path(str(fake_repo).swapcase()) / "data" / "b.json"
    with pytest.raises(onb.CustodyError):
        onb.check_input_path(variant, repo_root=fake_repo)
    # Lexically outside a lower-cased repo root, still caught (lexically or by identity).
    with pytest.raises(onb.CustodyError):
        onb.check_input_path(fake_repo / "data" / "b.json", repo_root=Path(str(fake_repo).swapcase()))


def test_lexical_layer_alone_folds_case(tmp_path):
    """The pure string layer refuses case variants without any filesystem access."""
    root = Path("/Users/someone/ShermanQC/FullProject")
    for path in (root / "DATA" / "x.json", root / "Data" / "x.json", root / "Research" / "jev2" / "y.json",
                 Path(str(root).lower()) / "data" / "x.json", Path(str(root).upper()) / "OUTPUT" / "z"):
        assert onb._lexical_problem(path, root, None), path
    # The research/jev exemption is exact: a case variant is refused, never exempted.
    assert onb._lexical_problem(root / "research" / "jev" / "a.json", root, None) is None
    assert onb._lexical_problem(root / "Research" / "JEV" / "a.json", root, None)
    # authorized_roots match exactly as well.
    auth = onb.Authorization("0" * 64, (root / "data",))
    assert onb._lexical_problem(root / "data" / "x.json", root, auth) is None
    assert onb._lexical_problem(root / "DATA" / "x.json", root, auth)


def test_identity_check_catches_an_alias_the_strings_miss(fake_repo, tmp_path, monkeypatch):
    """With the lexical layer disabled, the (st_dev, st_ino) walk alone still refuses."""
    if not (fake_repo / "DATA").exists():
        pytest.skip("case-sensitive filesystem: DATA/ is not an alias of data/ here")
    monkeypatch.setattr(onb, "_lexical_problem", lambda *_a, **_k: None)
    with pytest.raises(onb.CustodyError, match="authorization"):
        onb.check_input_path(fake_repo / "DATA" / "b.json", repo_root=fake_repo)
    onb.check_input_path(fake_repo / "research" / "jev" / "b.json", repo_root=fake_repo)
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"schema": onb.AUTHORIZATION_SCHEMA, "authorized_roots": [str(fake_repo / "data")]}))
    loaded = onb.load_authorization(auth, repo_root=fake_repo)
    onb.check_input_path(fake_repo / "DATA" / "b.json", repo_root=fake_repo, authorization=loaded)
    with pytest.raises(onb.CustodyError, match="authorized_roots"):
        onb.check_input_path(fake_repo / "outputs" / "b.json", repo_root=fake_repo, authorization=loaded)


@pytest.mark.parametrize("where", ["research/other", "data", "Research/other"])
def test_authorization_file_must_itself_pass_custody(fake_repo, tmp_path, where):
    auth = fake_repo / where / "auth.json"
    if not auth.parent.exists():
        auth.parent.mkdir(parents=True)
    auth.write_text(json.dumps({"schema": onb.AUTHORIZATION_SCHEMA, "authorized_roots": [str(fake_repo / "data")]}))
    with pytest.raises(onb.CustodyError, match="authorization file"):
        onb.load_authorization(auth, repo_root=fake_repo)


@pytest.mark.parametrize("name", ["holdout_auth", "one_shot"])
def test_authorization_file_with_a_never_readable_component_is_not_opened(tmp_path, name, monkeypatch):
    d = tmp_path / name
    d.mkdir()
    (d / "auth.json").write_text("{}")
    opened = []
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: opened.append(self) or real(self))
    with pytest.raises(onb.CustodyError, match="never readable"):
        onb.load_authorization(d / "auth.json")
    assert opened == []
    # An authorized root that names a never-readable directory is refused too.
    ok = tmp_path / "auth.json"
    ok.write_text(json.dumps({"schema": onb.AUTHORIZATION_SCHEMA, "authorized_roots": [str(d)]}))
    with pytest.raises(onb.CustodyError, match="never readable"):
        onb.load_authorization(ok)


def test_unprotected_and_research_jev_paths_are_allowed(fake_repo, tmp_path):
    onb.export_onboarding([fake_repo / "free" / "b.json", fake_repo / "research/jev/b.json"], tmp_path / "o",
                          repo_root=fake_repo)


@pytest.mark.parametrize("name", ["HOLDOUT10", "sealed_truth", "EHO4", "one_shot", "N007"])
def test_never_readable_components_refused_even_with_authorization(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    (d / "b.json").write_text("{}")
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"schema": onb.AUTHORIZATION_SCHEMA, "authorized_roots": [str(tmp_path)]}))
    with pytest.raises(onb.CustodyError, match="never readable"):
        onb.export_onboarding([d / "b.json"], tmp_path / "o", authorization=auth)


def test_symlink_into_a_protected_dir_is_refused(fake_repo, tmp_path):
    link = tmp_path / "innocent.json"
    os.symlink(fake_repo / "data" / "b.json", link)
    with pytest.raises(onb.CustodyError, match="resolved"):
        onb.export_onboarding([link], tmp_path / "o", repo_root=fake_repo)


def test_the_same_record_twice_is_refused(tmp_path, onboarding_input):
    with pytest.raises(onb.ExportError, match="duplicate"):
        onb.export_onboarding([onboarding_input, onboarding_input], tmp_path / "o")


def test_real_repo_data_dir_is_refused_lexically_without_touching_it(tmp_path):
    with pytest.raises(onb.CustodyError):
        onb.check_input_path(REPO / "data" / "does-not-matter.json")
    with pytest.raises(onb.CustodyError):
        onb.check_input_path(REPO / "research" / "f1_multifamily_selector_v2" / "model.py")


# --------------------------------------------------------------------------
# Scan exporters
# --------------------------------------------------------------------------


@pytest.fixture()
def jobs(tmp_path):
    root = tmp_path / "jobs"
    for i, sku in enumerate(("SYN-SCAN-A", "SYN-SCAN-B")):
        engine = root / f"job-{i}" / "engine"
        engine.mkdir(parents=True)
        (engine / "result.json").write_text(json.dumps(scan.synthetic_e3_result(i, sku=sku)))
        (engine / "bend_inspection_report.json").write_text(json.dumps(scan.synthetic_legacy_report(i)))
    return root


def test_e3_rows_creases_and_weak_labels():
    part = scan.e3_records(scan.synthetic_e3_result(0), group_sku="SYN-SCAN-A")
    keys = [rec.candidate_key for rec, _ in part.records]
    # The NOT_YET_FORMED bend has no measured line: not a scan candidate, skipped and counted.
    assert sum(k.startswith("e3|bend|") for k in keys) == 2 and sum(k.startswith("e3|crease|") for k in keys) == 2
    assert part.counts["e3_bends_without_measured_line"] == 1 and part.counts["e3_bend_rows"] == 3
    weak = {rec.candidate_key: lab.weak_label_source for rec, lab in part.records}
    bends = [k for k in keys if k.startswith("e3|bend|")]
    assert [weak[k] for k in bends] == ["recipe_binding_bound_formed"] * 2
    assert weak["e3|crease|crease-1"] == "recipe_binding_unbound_far"  # 350 mm away
    assert weak["e3|crease|crease-2"] is None  # 2 mm from B2: not far
    assert all(lab.label is None for _, lab in part.records)  # no human labels supplied
    for rec, _ in part.records:
        _, state = S.build_state(rec)
        # The engine state is a (CAD-assisted) feature...
        assert "engine_bend_state" in state["cad_relation"] or "engine_crease_reason" in state["cad_relation"]
        assert "engine_bend_state" not in state["candidate"]
    assert all(lab.label_source == "unreviewed" for _, lab in part.records)  # ...never a label


def _key_paths(node, prefix=""):
    out = set()
    for k, v in node.items():
        out.add(f"{prefix}{k}")
        if isinstance(v, dict):
            out |= _key_paths(v, f"{prefix}{k}.")
    return out


def _string_values(node):
    out = set()
    for k, v in node.items():
        if isinstance(v, dict):
            out |= _string_values(v)
        elif isinstance(v, str):
            out.add((k, v))
    return out


def test_e3_cad_free_states_cannot_tell_planned_from_unplanned():
    """After the CAD-free drop, planned rows and creases share key paths and enum values."""
    for seed in range(4):
        part = scan.e3_records(scan.synthetic_e3_result(seed, n_bends=4, n_creases=3), group_sku="X")
        groups: dict[bool, list[dict]] = {True: [], False: []}
        for rec, _ in part.records:
            _, state = S.build_state(rec)
            groups[rec.candidate_key.startswith("e3|bend|")].append(S.ablate(state, cad_free=True))
        assert groups[True] and groups[False]
        planned_keys = {frozenset(_key_paths(st)) for st in groups[True]}
        crease_keys = {frozenset(_key_paths(st)) for st in groups[False]}
        assert planned_keys == crease_keys and len(planned_keys) == 1, planned_keys ^ crease_keys
        planned_enums = set().union(*(_string_values(st) for st in groups[True]))
        crease_enums = set().union(*(_string_values(st) for st in groups[False]))
        assert planned_enums == crease_enums


def test_e3_cad_free_ablation_removes_every_nominal_value():
    part = scan.e3_records(scan.synthetic_e3_result(0), group_sku="X")
    for rec, _ in part.records:
        _, state = S.build_state(rec)
        cad_free = json.dumps(S.ablate(state, cad_free=True))
        for needle in ("nominal", "deviation", "is_planned_bend", "planned_bends_count", "formed_fraction",
                       "prior_formed"):
            assert needle not in cad_free, needle


def test_e3_cad_free_blocks_do_not_depend_on_nominal_geometry():
    """Moving every nominal (CAD) polyline changes cad_relation only."""
    result = scan.synthetic_e3_result(0)
    moved = copy.deepcopy(result)
    for bend in moved["bends"]:
        bend["display"]["nominal_polyline_mm"] = [[p[0] + 7.0, p[1] - 33.0, p[2] + 2.0]
                                                  for p in bend["display"]["nominal_polyline_mm"]]
        bend["state_evidence"]["nominal_included_deg"] = 95.0
    a = [S.build_state(r)[1] for r, _ in scan.e3_records(result, group_sku="X").records]
    b = [S.build_state(r)[1] for r, _ in scan.e3_records(moved, group_sku="X").records]
    assert [S.ablate(x, cad_free=True) for x in a] == [S.ablate(x, cad_free=True) for x in b]
    assert [x.get("cad_relation") for x in a] != [x.get("cad_relation") for x in b]


def test_legacy_report_candidate_block_is_binding_free():
    part = scan.legacy_report_records(scan.synthetic_legacy_report(0), group_sku="X")
    assert part.counts["legacy_matches_without_detection"] == 1
    shapes = set()
    for rec, _ in part.records:
        _, state = S.build_state(rec)
        shapes.add(tuple(sorted(state["candidate"])))
        assert "observability_state" in state["cad_relation"] or rec.candidate_key.startswith("legacy|unmatched")
    assert len(shapes) == 1, shapes  # matched and unmatched share the CAD-free schema


def test_legacy_placeholder_matches_never_export_cad_geometry():
    """A CAD_LOCAL_NEIGHBORHOOD match carries the CAD line as its 'detected' line: skip it.

    Mutate-and-check: turning match 0 into a placeholder must give exactly the export of
    the same report with match 0 undetected (no CAD length, no weak label, no neighbour).
    """
    report = scan.synthetic_legacy_report(0, n_matches=3, n_unmatched=2)
    undetected = copy.deepcopy(report)
    undetected["matches"][0]["detected_line_start"] = undetected["matches"][0]["detected_line_end"] = None
    reference = scan.legacy_report_records(undetected, group_sku="X")
    ref_states = {rec.candidate_key: S.build_state(rec)[1] for rec, _ in reference.records}
    ref_weak = {rec.candidate_key: lab.weak_label_source for rec, lab in reference.records}
    for flag, copy_cad_line in (({"assignment_source": "CAD_LOCAL_NEIGHBORHOOD"}, True),
                                ({"detected_line_geometry_source": "CAD_REFERENCE_PLACEHOLDER"}, False),
                                ({}, True)):  # older report: no flag, detected line == CAD line
        mutated = copy.deepcopy(report)
        m = mutated["matches"][0]
        m.update(flag)
        if copy_cad_line:
            m["detected_line_start"], m["detected_line_end"] = list(m["cad_line_start"]), list(m["cad_line_end"])
        m["measured_radius"] = 2.0  # older reports: the target radius
        part = scan.legacy_report_records(mutated, group_sku="X")
        assert part.counts["legacy_matches_cad_placeholder"] == 1
        states = {rec.candidate_key: S.build_state(rec)[1] for rec, _ in part.records}
        weak = {rec.candidate_key: lab.weak_label_source for rec, lab in part.records}
        assert "legacy|match|CAD_B1" not in states  # neither a state nor a weak label
        assert states == ref_states and weak == ref_weak
        assert all(st["candidate"]["detected_line_length_mm"] != pytest.approx(100.0) for st in states.values())


def test_weak_far_uses_true_segment_distance():
    report = scan.synthetic_legacy_report(0, n_matches=2, n_unmatched=0)
    # CAD lines run along x at y=0 and y=60 (0..100 mm). A long detection crossing y=0 at x=80,
    # whose midpoint and endpoints are all >= 20 mm from both CAD lines, is not "far".
    report["unmatched_detections"] = [
        {"bend_id": "DET_X", "measured_angle": 120.0, "measured_radius": 3.0, "bend_line_length": 400.0,
         "bend_line_start": [80.0, -300.0, 0.0], "bend_line_end": [80.0, 20.0, 0.0], "confidence": 0.5},
        {"bend_id": "DET_C", "measured_angle": 120.0, "measured_radius": 3.0, "bend_line_length": 100.0,
         "bend_line_start": [300.0, 0.0, 0.0], "bend_line_end": [400.0, 0.0, 0.0], "confidence": 0.5},
    ]
    part = scan.legacy_report_records(report, group_sku="X")
    got = {rec.candidate_key: (S.build_state(rec)[1], lab) for rec, lab in part.records}
    assert got["legacy|unmatched|DET_X"][1].weak_label_source is None
    # Collinear but 200 mm along the axis: far, and not "within 10 mm" of a reference.
    collinear_state, collinear_label = got["legacy|unmatched|DET_C"]
    assert collinear_label.weak_label_source == "recipe_binding_unbound_far"
    assert collinear_state["cad_relation"]["is_within_10mm_of_reference"] is False


def test_legacy_weak_labels_follow_binding_and_distance():
    part = scan.legacy_report_records(scan.synthetic_legacy_report(0), group_sku="X")
    weak = {rec.candidate_key: lab.weak_label_source for rec, lab in part.records}
    assert weak["legacy|match|CAD_B1"] == "recipe_binding_bound_formed"
    assert weak["legacy|unmatched|DET_1"] == "recipe_binding_unbound_far"
    assert weak["legacy|unmatched|DET_2"] is None  # 3 mm from a CAD line


def test_legacy_marker_labels_apply_only_when_supplied_and_unambiguous():
    report = scan.synthetic_legacy_report(0)
    markers = {"CAD_B1": "human_marker_real_bend", "DET_1": "human_marker_not_bend_outline",
               "DET_2": "human_marker_uncertain"}
    part = scan.legacy_report_records(report, group_sku="X", marker_labels=markers)
    src = {rec.candidate_key: lab.label_source for rec, lab in part.records}
    assert src == {"legacy|match|CAD_B1": "human_marker_real_bend", "legacy|match|CAD_B2": "unreviewed",
                   "legacy|unmatched|DET_1": "human_marker_not_bend_outline",
                   "legacy|unmatched|DET_2": "human_marker_uncertain"}
    clash = copy.deepcopy(report)
    clash["unmatched_detections"][0]["bend_id"] = "CAD_B1"
    part = scan.legacy_report_records(clash, group_sku="X", marker_labels=markers)
    assert part.counts["legacy_marker_ambiguous_id"] == 2
    assert all(lab.label_source != "human_marker_real_bend" for _, lab in part.records)


def test_export_scan_reads_labels_only_from_an_explicit_root(tmp_path, jobs):
    marker = {"schema_version": 1, "kind": "sherman_qc_bend_instance_labels", "job_id": "j",
              "coordinate_frame": "reference_mesh_raw", "review_status": "complete", "manual_bends": [],
              "marker_reviews": [{"bend_id": "DET_1", "decision": "not_bend_outline"}]}
    # A label file inside the job workspace is ignored...
    (jobs / "job-0" / "bend_instance_labels.json").write_text(json.dumps(marker))
    r = scan.export_scan(tmp_path / "o1", jobs_root=jobs)
    assert not [row for row in _labels(tmp_path / "o1").values() if row.label_source != "unreviewed"]
    assert r.manifest["counts"]["jobs"] == 2
    # ...and used when supplied through --labels-root.
    labels_root = tmp_path / "labels" / "job-0"
    labels_root.mkdir(parents=True)
    (labels_root / "bend_instance_labels.json").write_text(json.dumps(marker))
    scan.export_scan(tmp_path / "o2", jobs_root=jobs, labels_root=tmp_path / "labels")
    srcs = [row.label_source for row in _labels(tmp_path / "o2").values()]
    assert srcs.count("human_marker_not_bend_outline") == 1
    # A wrong frame is refused, not silently used.
    marker["coordinate_frame"] = "cad_model_mm"
    (labels_root / "bend_instance_labels.json").write_text(json.dumps(marker))
    with pytest.raises(Exception, match="frame"):
        scan.export_scan(tmp_path / "o3", jobs_root=jobs, labels_root=tmp_path / "labels")


def test_export_scan_is_deterministic_and_clean(tmp_path, jobs):
    rows, gold = scan.synthetic_f1_pool(0)
    pool, gold_path = tmp_path / "pool.jsonl", tmp_path / "gold.json"
    pool.write_text("".join(json.dumps(r) + "\n" for r in rows))
    gold_path.write_text(json.dumps(gold))
    for name in ("a", "b"):
        scan.export_scan(tmp_path / name, jobs_root=jobs, f1_pool=pool, gold=gold_path)
    for name in ("states.jsonl", "labels.jsonl", "manifest.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name
    _assert_clean(_states(tmp_path / "a"))
    sources = {S_["candidate_source"] for S_ in _states(tmp_path / "a")}
    assert sources == {"SCAN_E3", "SCAN_LEGACY_REPORT", "SCAN_F1_POOL"}


def test_f1_pool_gold_labels_and_incomplete_gold():
    rows, gold = scan.synthetic_f1_pool(0)
    part = scan.f1_pool_records(rows, source_sha256="f" * 64, gold=scan.parse_gold(gold))
    src = sorted(lab.label_source for _, lab in part.records)
    assert src == ["gold_line_matched"] * 3 + ["gold_line_unmatched"] * 3
    gold["review_status"] = "in_progress"
    part = scan.f1_pool_records(rows, source_sha256="f" * 64, gold=scan.parse_gold(gold))
    assert sorted(lab.label_source for _, lab in part.records).count("gold_line_unmatched_incomplete") == 3
    gold["coordinate_frame"] = "cad_model_mm"
    with pytest.raises(Exception, match="frame"):
        scan.f1_pool_records(rows, source_sha256="f" * 64, gold=scan.parse_gold(gold))
    for rec, _ in part.records:
        _, state = S.build_state(rec)
        assert "part_id" not in json.dumps(state) and "secret" not in json.dumps(state)


def test_f1_state_keys_are_unique_and_pass_the_deny_list():
    keys = list(scan.F1_STATE_KEYS.values())
    assert len(set(keys)) == len(keys)
    S.build_state(S.CandidateRecord("a" * 64, "k", "SCAN_F1_POOL", {k: 0.5 for k in keys}))


def test_f1_feature_names_match_the_selector_source():
    """Parity with research/f1_multifamily_selector_v2/model.py (source text only, via ast)."""
    path = REPO / "research" / "f1_multifamily_selector_v2" / "model.py"
    if not path.is_file():
        pytest.skip("selector source not present")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "BASE_FEATURE_NAMES" for t in node.targets):
            names = tuple(ast.literal_eval(node.value))
    assert names == scan.F1_BASE_FEATURE_NAMES


def test_exporters_accept_the_contract_examples():
    """Field paths match the real C01 contracts (synthetic SYN-HAT-1 examples, contract-valid)."""
    ex = REPO / "contracts" / "production" / "examples"
    if not (ex / "shopfloor_result_v1.example.json").is_file():
        pytest.skip("contract examples not present")
    recipe = json.loads((ex / "part_recipe_v1.example.json").read_text())
    compiled = json.loads((ex / "compiled_part_v1.example.json").read_text())
    part = onb.onboarding_records({"recipe": recipe, "compiled_part": compiled})
    assert len(part.records) == len(recipe["bends"])
    for rec, lab in part.records:
        _, state = S.build_state(rec)
        assert lab.label_source == "operator_confirmed"
        assert state["candidate"]["bend_line_length_mm"] > 0
        assert state["cad_relation"]["reference_bends_count"] == len(compiled["bends"])
    for name, formed_weak in (("shopfloor_result_v1.example.json", 4), ("shopfloor_result_v1.in_process.example.json", 2),
                              ("shopfloor_result_v1.invalid.example.json", 0)):
        result = json.loads((ex / name).read_text())
        records = scan.e3_records(result, group_sku=result["sku"]).records
        states = [S.build_state(rec)[1] for rec, _ in records]
        _assert_clean(states)
        assert sum(1 for _, lab in records if lab.weak_label_source) == formed_weak, name
