"""TRUTH ORACLE -- NOT AN ENGINE. A harness self-test result synthesizer.

This module reads E2 **truth** and writes contract-valid E3 outputs
(``result.json`` plus the viewer files) whose values are truth plus a chosen
error model. It exists so the E4 harness (scorer, runner, renderer, canaries,
line metric) can be tested end to end before and independently of E3, and so
every gate can be shown to *fail* on a deliberately wrong "engine"
(mutate-and-check). Nothing it produces is evidence about the production
engine: results carry ``engine.engine_version = "e4-oracle-..."`` and the
warning ``E4_TRUTH_ORACLE_NOT_AN_ENGINE``; the scorecard marks any run that
contains them as not qualifying.

Two entry points:

- :func:`oracle_result` / :func:`write_oracle_output` (in-process), driven by an
  :class:`OracleSpec` (noise, bias, U, per-characteristic overrides, forced
  bend states, ambiguous pose, swapped lines, ignoring mirror/wrong-part
  detection).
- An E3-compatible command line (ENGINE_IO §4), for runner tests::

      python -m domains.production.engine.eval.oracle --compiled C --recipe R --scan S --out O --scan-sha256 H ...

  It finds the case's truth by the scan sha256 under ``$SHERMAN_E4_ORACLE_CASES``
  (an E2 tier directory) -- a lookup only a truth oracle may do -- and takes
  failure-injection switches from ``$SHERMAN_E4_ORACLE_MODE`` (comma list:
  ``sleep=S``, ``alloc_mb=N``, ``crash``, ``exit2``, ``exit3``, ``garbage``,
  ``no_result``, ``name_leak`` (writes the scan file name into the result, to prove the
  blindness check fails), ``modify_input`` (appends to its staged scan, to prove the
  runner's input check), ``noise=X``, ``seed=N``, ``bias_flange=X``, ``u_mm=X``).
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from contracts.production.canonical import canonical_json_bytes
from contracts.production.characteristics import make_characteristic_id
from contracts.production.compiled_part import CompiledPartV1
from contracts.production.consistency import check_result_against_compiled
from contracts.production.inspection_result import (
    BendDisplayV1,
    BendGeometryFileV1,
    BendRowV1,
    BendStateEvidenceV1,
    CharacteristicEvidenceV1,
    CharacteristicResultV1,
    CoverageV1,
    EngineInfoV1,
    InspectionValidityV1,
    RecipeRefV1,
    RegistrationV1,
    ResultContextV1,
    ScanInfoV1,
    ShopfloorInspectionResultV1,
    StageDecisionV1,
    StationStatsV1,
    UnplannedCreaseV1,
    ViewerAssetsV1,
    derive_stage_decision,
    derive_summary,
)
from contracts.production.layout import ENGINE_OUTPUT_FILES
from contracts.production.part_recipe import PartRecipeV1

from .engine_output import SEGMENT_BEND_BASE, SEGMENT_PANEL_BASE, apply4, stage_locus
from .plyio import read_ply, read_ply_header, sha256_file, write_mesh_ply, write_points_ply
from .truth import TruthCase, load_truth_case, scorable

__all__ = [
    "ORACLE_VERSION",
    "ORACLE_WARNING",
    "OracleSpec",
    "oracle_result",
    "write_oracle_output",
    "main",
]

ORACLE_VERSION = "e4-oracle-1.0.0"
ORACLE_WARNING = "E4_TRUTH_ORACLE_NOT_AN_ENGINE"
_CREATED_AT = "2026-10-17T00:00:00Z"


@dataclass(frozen=True)
class OracleSpec:
    """Error model of the oracle (all zero = exact truth)."""

    #: Gaussian error sigma per unit, added to truth (mm, deg).
    noise_mm: float = 0.0
    noise_deg: float = 0.0
    seed: int = 0
    #: Reported expanded uncertainty per unit.
    u_mm: float = 0.08
    u_deg: float = 0.15
    #: Additive bias per characteristic kind.
    bias: Mapping[str, float] = field(default_factory=dict)
    #: Multiplicative gain on the true deviation per kind (1.0 = honest; < 1 pulls to nominal).
    gain: Mapping[str, float] = field(default_factory=dict)
    #: characteristic_id -> {"actual": x} | {"error": e} | {"u": u} | {"state": "NOT_MEASURED", "reason": r}
    overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: bend_uid -> forced engine state (FORMED / NOT_YET_FORMED / UNKNOWN)
    bend_states: Mapping[str, str] = field(default_factory=dict)
    pose_ambiguous: bool = False
    #: Force an invalid validity state (e.g. UNTRUSTED_REGISTRATION).
    validity: str | None = None
    #: Swap the measured lines of two bends (a confident wrong id).
    swap_lines: tuple[str, str] | None = None
    #: Register the part in a symmetric flip: raw_to_frame = C . raw_to_cad with C an E1
    #: ROTATION self-map (within threshold, non-trivial bend permutation) of the truth stage;
    #: measured lines follow the flip (row x carries physical bend perm^-1(x)).
    pose_flip: bool = False
    #: Report mirrored / wrong parts as VALID with nominal actuals (to prove the never-PASS gate).
    ignore_part_identity: bool = False
    #: Report the truth's extra bends as unplanned creases.
    report_creases: bool = True
    write_assets: bool = True


def _unit_noise(spec: OracleSpec, unit: str) -> float:
    return spec.noise_deg if unit == "deg" else spec.noise_mm


def _u(spec: OracleSpec, unit: str) -> float:
    return spec.u_deg if unit == "deg" else spec.u_mm


def _expected_source(kind: str) -> str:
    return {"BACKGAUGE_LENGTH": "STAGE_MODEL_NOMINAL", "BEND_LINE_POSITION": "NOMINAL_LOCUS_REFERENCE",
            "GENERAL_MEASUREMENT": "CAD_DISTANCE"}.get(kind, "COMPILED_NOMINAL")


def _rng_for(spec: OracleSpec, key: str) -> np.random.Generator:
    import hashlib

    digest = hashlib.sha256(f"{spec.seed}|{key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _measure(nom, tc: TruthCase, spec: OracleSpec, *, side: str | None, defined: bool) -> CharacteristicResultV1:
    """One applicable characteristic of a bend whose state allows measuring."""
    cid = nom.characteristic_id
    src = _expected_source(nom.kind)
    ov = dict(spec.overrides.get(cid) or {})
    if ov.get("state") == "NOT_MEASURED":
        return CharacteristicResultV1.not_measured(nom, reason=ov.get("reason", "OCCLUDED"), expected_source=src)
    if nom.nominal is None:
        return CharacteristicResultV1.not_measured(nom, reason=nom.unavailable_reason or "NOMINAL_UNAVAILABLE",
                                                   expected_source=src)
    unit = nom.unit
    if "actual" in ov:
        actual = float(ov["actual"])  # a forced value (mutation tests), whatever the truth
    elif not defined:
        actual = float(nom.nominal)  # ignore_part_identity: pretend the part is the recipe part
    else:
        t = tc.characteristics.get(cid)
        if t is None:
            return CharacteristicResultV1.not_measured(nom, reason="EDGE_NOT_SEEN", expected_source=src)
        if t.kind == "BACKGAUGE_LENGTH":
            t = t.for_side(side)
        if not scorable(t):
            reason = "ADJACENT_BEND_NOT_MEASURED" if t.truth_undefined_reason == "ADJACENT_BEND_NOT_FORMED" else (
                "AMBIGUOUS_SEGMENTATION" if t.well_defined is False else "EDGE_NOT_SEEN")
            return CharacteristicResultV1.not_measured(nom, reason=reason, expected_source=src)
        gain = float(spec.gain.get(nom.kind, 1.0))
        true_dev = float(t.truth) - float(nom.nominal)
        actual = float(nom.nominal) + gain * true_dev
        actual += float(spec.bias.get(nom.kind, 0.0))
        sigma = _unit_noise(spec, unit)
        if sigma > 0:
            actual += float(_rng_for(spec, cid).normal(0.0, sigma))
    if "error" in ov:
        actual += float(ov["error"])
    u = float(ov.get("u", _u(spec, unit)))
    spread = max(u, 0.01)
    return CharacteristicResultV1.measured(
        nom, actual=actual, uncertainty=u, expected_source=src,
        stations=StationStatsV1(count=7, median=round(actual, 6), min=round(actual - spread / 2, 6),
                                max=round(actual + spread / 2, 6)),
        evidence=CharacteristicEvidenceV1(method="E4_TRUTH_ORACLE", support_points=1000, coverage_ratio=1.0,
                                          fit_rms_mm=0.0, notes=(ORACLE_WARNING,)),
    )


def _rigid(m: np.ndarray) -> tuple[tuple[float, ...], ...]:
    """Orthonormalise the rotation (SVD) so canonical rounding stays rigid."""
    u, _, vt = np.linalg.svd(m[:3, :3])
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = m[:3, 3]
    return tuple(tuple(round(float(v), 9) for v in row) for row in out)


def oracle_result(
    *,
    recipe: PartRecipeV1,
    compiled: CompiledPartV1,
    truth: TruthCase,
    scan_sha256: str,
    scan_format: str = "PLY_BINARY_LE",
    vertex_count: int = 0,
    face_count: int = 0,
    spec: OracleSpec = OracleSpec(),
    viewer: ViewerAssetsV1 | None = None,
    context: ResultContextV1 | None = None,
    extra_warnings: tuple[str, ...] = (),
) -> ShopfloorInspectionResultV1:
    """A contract-valid result from truth (see module docstring)."""
    tc = truth
    identity_ok = tc.defined_part or spec.ignore_part_identity
    if spec.validity is not None:
        validity_state = spec.validity
    elif not identity_ok:
        validity_state = "MIRRORED_PART_SUSPECTED" if tc.mirrored else "WRONG_PART_SUSPECTED"
    else:
        validity_state = "VALID"
    valid = validity_state == "VALID"
    pose_amb = bool(spec.pose_ambiguous and valid)

    # -- bend states --------------------------------------------------------------
    states: dict[str, str] = {}
    reasons: dict[str, tuple[str, ...]] = {}
    for b in compiled.bends:
        if not valid:
            states[b.bend_uid], reasons[b.bend_uid] = "UNKNOWN", ("INVALID_INSPECTION",)
            continue
        if not tc.defined_part:  # ignore_part_identity: claim every bend formed
            st = "FORMED"
        else:
            st = tc.bend(b.bend_uid).state or "UNKNOWN"
        st = spec.bend_states.get(b.bend_uid, st)
        if pose_amb and st != "NOT_YET_FORMED":
            st = "UNKNOWN"
        states[b.bend_uid] = st
        reasons[b.bend_uid] = (("AMBIGUOUS_POSE",) if pose_amb else ("OCCLUDED",)) if st == "UNKNOWN" else ()

    class _S:  # minimal row view for derive_stage_decision
        def __init__(self, b):
            self.bend_uid, self.sequence_index = b.bend_uid, b.sequence_index
            self.machine_move_index, self.state = b.machine_move_index, states[b.bend_uid]

    stage_fields = derive_stage_decision(
        inspection_stage=recipe.inspection_stage, declared_stage=None, bends=[_S(b) for b in compiled.bends],
        pose_ambiguous=pose_amb, validity_state=validity_state)
    k = stage_fields["inferred_stage"]
    stage_model_id = None
    if valid and k is not None:
        if "OUT_OF_ORDER_SUSPECTED" not in stage_fields["flags"]:
            stage_model_id = f"S{k}"
        else:
            formed = set(stage_fields["formed_bend_uids"])
            stage_model_id = next((s.stage_model_id for s in compiled.stages
                                   if s.kind == "OUT_OF_ORDER" and set(s.formed_bend_uids) == formed), None)
    stage_model = compiled.stage(stage_model_id) if stage_model_id else compiled.stage(f"S{len(compiled.bends)}")
    decodable = (valid and k is not None and 1 <= k <= len(compiled.bends) - 1
                 and not {"OUT_OF_ORDER_SUSPECTED", "AMBIGUOUS_POSE", "NOT_EVALUATED"} & set(stage_fields["flags"]))

    # -- symmetric flip (identity-check self-tests) --------------------------------------
    flip: tuple[np.ndarray, dict[str, str]] | None = None
    if spec.pose_flip and valid and tc.defined_part:
        formed = set(tc.formed_bend_uids or ())
        tstage = next((x for x in compiled.stages if set(x.formed_bend_uids) == formed), None)
        cand = None if tstage is None else next(
            (c for c in tstage.symmetry.candidates if c.kind == "ROTATION"
             and c.rms_mm <= tstage.symmetry.threshold_rms_mm and any(k != v for k, v in c.bend_permutation.items())),
            None)
        if cand is None:
            raise ValueError("pose_flip needs an E1 ROTATION self-map with a non-trivial bend permutation")
        flip = (np.asarray(cand.transform, dtype=np.float64).reshape(4, 4), dict(cand.bend_permutation))

    # -- characteristics and rows ----------------------------------------------------
    noms = {c.characteristic_id: c for c in compiled.characteristics}
    swap = dict(zip(spec.swap_lines, reversed(spec.swap_lines))) if spec.swap_lines else {}
    flip_source = {v: k for k, v in flip[1].items()} if flip is not None else {}
    rows = []
    for b in compiled.bends:
        st = states[b.bend_uid]
        chars = []
        for kind, side in (("FLANGE_LENGTH", "A"), ("FLANGE_LENGTH", "B"), ("INCLUDED_ANGLE", None),
                           ("BEND_LINE_POSITION", None)):
            nom = noms[make_characteristic_id(kind, bend_uid=b.bend_uid, side=side)]
            src = _expected_source(kind)
            if nom.recipe_requirement == "NOT_INSPECTED":
                chars.append(CharacteristicResultV1.not_applicable(nom, reason="NOT_INSPECTED_BY_RECIPE",
                                                                   expected_source=src))
            elif st == "NOT_YET_FORMED":
                chars.append(CharacteristicResultV1.not_applicable(nom, reason="BEND_NOT_YET_FORMED",
                                                                   expected_source=src))
            elif st == "UNKNOWN":
                why = "INVALID_INSPECTION" if not valid else reasons[b.bend_uid][0]
                chars.append(CharacteristicResultV1.not_measured(nom, reason=why, expected_source=src))
            else:
                chars.append(_measure(nom, tc, spec, side=side, defined=tc.defined_part))
        if decodable and b.sequence_index == k:
            bg = noms.get(make_characteristic_id("BACKGAUGE_LENGTH", bend_uid=b.bend_uid, stage_index=k))
            if bg is not None:
                chars.append(_measure(bg, tc, spec, side=bg.side, defined=tc.defined_part))
        nominal_line = stage_model.bend_lines[b.bend_uid]
        nl = (tuple(nominal_line.start_mm), tuple(nominal_line.end_mm))
        by_kind = {(c.kind, c.side): c for c in chars}
        position, angle = by_kind[("BEND_LINE_POSITION", None)], by_kind[("INCLUDED_ANGLE", None)]
        measured = None
        if position.state == "MEASURED":
            src_uid = swap.get(b.bend_uid, b.bend_uid)
            if flip is not None:
                src_uid = flip_source.get(src_uid, src_uid)
            line = None
            if tc.defined_part:
                line = tc.bend(src_uid).actual_oml
            if line is None:
                line = stage_locus(compiled, stage_model, src_uid, "oml_virtual_sharp")
            if flip is not None:
                line = apply4(flip[0], np.asarray(line, dtype=np.float64))
            measured = tuple(tuple(round(float(v), 6) for v in p) for p in line)
        mid = tuple((a + c) / 2 for a, c in zip(nl[0], nl[1]))
        incl = None
        if st in ("FORMED", "PARTIALLY_FORMED", "UNEXPECTED_FORMED") and angle.state == "MEASURED":
            incl = min(180.0, max(1e-3, float(angle.actual)))
        elif st == "NOT_YET_FORMED":
            incl = 180.0
        rows.append(BendRowV1(
            bend_uid=b.bend_uid, label=b.label, sequence_index=b.sequence_index,
            machine_move_index=b.machine_move_index, state=st, state_reasons=reasons[b.bend_uid],
            state_evidence=BendStateEvidenceV1(
                nominal_included_deg=b.nominal_included_deg, measured_included_deg=incl,
                formed_fraction=None if incl is None else round((180.0 - incl) / b.sweep_deg, 6)),
            characteristics=tuple(chars),
            display=BendDisplayV1(nominal_polyline_mm=nl, measured_polyline_mm=measured,
                                  label_anchor_mm=(mid[0], mid[1], mid[2] + 5.0)),
            notes=(ORACLE_WARNING,),
        ))

    creases = ()
    if valid and spec.report_creases and tc.defined_part:
        doc = json.loads((tc.case_dir / "truth.json").read_text(encoding="utf-8"))
        creases = tuple(
            UnplannedCreaseV1(crease_id=f"X{i + 1}", reason="EXTRA_BEND_SUSPECTED",
                              polyline_mm=(tuple(x["oml_virtual_sharp"]["start_mm"]), tuple(x["oml_virtual_sharp"]["end_mm"])),
                              included_deg=min(180.0, float(x["included_deg"])), note=ORACLE_WARNING)
            for i, x in enumerate(doc.get("extra_bends") or ()))

    stage = StageDecisionV1(inspection_stage=recipe.inspection_stage, stage_model_id=stage_model_id, **stage_fields)
    summary = derive_summary(validity_state=validity_state, bends=rows, stage_decision=stage.decision,
                             stage_status=stage.status, unplanned_creases=len(creases))
    if valid:
        if tc.raw_to_cad is None:
            raise ValueError(f"{tc.case_id}: truth has no raw_to_cad")
        raw_to_frame = tc.raw_to_cad if flip is None else flip[0] @ tc.raw_to_cad
        registration = RegistrationV1(method="E4_TRUTH_ORACLE", raw_to_frame=_rigid(raw_to_frame),
                                      datum_fitness=1.0, rmse_mm=0.0, pose_ambiguous=pose_amb)
        validity = InspectionValidityV1(state="VALID", registration=registration,
                                        coverage=CoverageV1(scan_point_count=vertex_count))
    else:
        validity = InspectionValidityV1(state=validity_state, reasons=(f"ORACLE:{validity_state}",),
                                        registration=RegistrationV1(method="E4_TRUTH_ORACLE", abstained=True))
    if viewer is None or not valid:
        viewer = ViewerAssetsV1(heatmap_band_mm=recipe.default_tolerances.surface_profile_mm,
                                heatmap_unavailable_reason="ORACLE_NO_ASSETS" if valid else "INVALID_INSPECTION")
    if viewer.deviations is not None and viewer.deviation_reference_stage != stage_model_id:
        viewer = viewer.model_copy(update={"deviations": None, "deviation_reference_stage": None,
                                           "heatmap_unavailable_reason": "NO_DECODED_STAGE_MODEL"})
    result = ShopfloorInspectionResultV1(
        created_at=_CREATED_AT, context=context or ResultContextV1(), sku=recipe.sku, part_name=recipe.part_name,
        recipe=RecipeRefV1(recipe_id=recipe.recipe_id, version=recipe.recipe_version,
                           recipe_sha256=recipe.recipe_sha256, content_sha256=recipe.content_sha256),
        scan=ScanInfoV1(sha256=scan_sha256, format=scan_format, vertex_count=vertex_count, face_count=face_count),
        engine=EngineInfoV1(engine_version=ORACLE_VERSION, compiled_part_sha256=compiled.compiled_part_sha256,
                            algorithm_versions={"oracle": "truth_plus_error_model"}),
        inspection_validity=validity, stage_decision=stage, summary=summary, bends=tuple(rows),
        unplanned_creases=creases, viewer_assets=viewer, warnings=(ORACLE_WARNING,) + tuple(extra_warnings),
    )
    problems = check_result_against_compiled(result, compiled, recipe)
    if problems:
        raise RuntimeError(f"oracle produced an inconsistent result: {problems}")
    return result


def _scan_format(path: Path) -> tuple[str, int, int]:
    if path.suffix.lower() == ".stl":
        raise ValueError("the oracle reads PLY scans only")
    header = read_ply_header(path)
    fmt = {"ascii": "PLY_ASCII", "binary_little_endian": "PLY_BINARY_LE",
           "binary_big_endian": "PLY_BINARY_BE"}[header.fmt]
    return fmt, header.count("vertex"), header.count("face")


def _nn_labels_and_deviation(points: np.ndarray, cloud: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    pts = np.asarray(cloud["points"], dtype=np.float64)
    tree = cKDTree(pts)
    dist, idx = tree.query(points, k=1)
    kind = np.asarray(cloud["segment_kind"])[idx]
    seg = np.asarray(cloud["segment_index"])[idx].astype(np.int64)
    labels = np.where(kind == 1, SEGMENT_BEND_BASE + seg, SEGMENT_PANEL_BASE + seg).astype(np.uint16)
    normals = np.asarray(cloud["normals"], dtype=np.float64)[idx]
    dev = np.einsum("ij,ij->i", points - pts[idx], normals).astype(np.float32)
    far = dist > 3.0
    labels[far] = 0
    dev[far] = np.nan
    return labels, dev


def write_oracle_output(
    out_dir: str | os.PathLike,
    *,
    recipe: PartRecipeV1,
    compiled: CompiledPartV1,
    compiled_dir: str | os.PathLike,
    truth: TruthCase,
    scan_path: str | os.PathLike,
    spec: OracleSpec = OracleSpec(),
    context: ResultContextV1 | None = None,
    unit_factor: float = 1.0,
    extra_warnings: tuple[str, ...] = (),
) -> ShopfloorInspectionResultV1:
    """Write a complete E3-shaped output directory (ENGINE_IO §4.1) from truth."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scan_path = Path(scan_path)
    fmt, nv, nf = _scan_format(scan_path)
    sha = sha256_file(scan_path)
    draft = oracle_result(recipe=recipe, compiled=compiled, truth=truth, scan_sha256=sha, scan_format=fmt,
                          vertex_count=nv, face_count=nf, spec=spec, context=context)
    viewer = draft.viewer_assets
    sid = draft.stage_decision.stage_model_id
    if spec.write_assets and draft.inspection_validity.state == "VALID":
        raw = read_ply(scan_path, with_faces=False).vertices * float(unit_factor)
        frame = apply4(np.asarray(draft.inspection_validity.registration.raw_to_frame, dtype=np.float64), raw)
        write_points_ply(out / ENGINE_OUTPUT_FILES["registered_scan"], frame, ("E4 truth oracle", "cad_model_mm"))
        stage = compiled.stage(sid) if sid else compiled.stage(f"S{len(compiled.bends)}")
        cdir = Path(compiled_dir)
        with np.load(cdir / stage.sample_cloud.rel_path, allow_pickle=False) as z:
            cloud = {k: z[k] for k in z.files}
        labels, dev = _nn_labels_and_deviation(frame, cloud)
        labels.astype("<u2").tofile(out / ENGINE_OUTPUT_FILES["segment_labels"])
        disp = out / "display"
        disp.mkdir(exist_ok=True)
        final = compiled.stage(f"S{len(compiled.bends)}")
        for key, st in (("stage_mesh", stage), ("final_mesh", final)):
            mesh = read_ply(cdir / st.display_mesh.rel_path)
            write_mesh_ply(out / ENGINE_OUTPUT_FILES[key], mesh.vertices, mesh.faces, ("E4 truth oracle",))
        fields = dict(registered_scan=ENGINE_OUTPUT_FILES["registered_scan"], point_count=int(len(frame)),
                      segment_labels=ENGINE_OUTPUT_FILES["segment_labels"], stage_mesh=ENGINE_OUTPUT_FILES["stage_mesh"],
                      final_mesh=ENGINE_OUTPUT_FILES["final_mesh"], bend_geometry=ENGINE_OUTPUT_FILES["bend_geometry"],
                      heatmap_band_mm=recipe.default_tolerances.surface_profile_mm)
        if sid is not None:
            dev.astype("<f4").tofile(out / ENGINE_OUTPUT_FILES["deviations"])
            fields.update(deviations=ENGINE_OUTPUT_FILES["deviations"], deviation_reference_stage=sid)
        else:
            fields.update(heatmap_unavailable_reason="NO_DECODED_STAGE_MODEL")
        viewer = ViewerAssetsV1(**fields)
    else:
        viewer = draft.viewer_assets.model_copy(update={"bend_geometry": ENGINE_OUTPUT_FILES["bend_geometry"]})
    result = oracle_result(recipe=recipe, compiled=compiled, truth=truth, scan_sha256=sha, scan_format=fmt,
                           vertex_count=nv, face_count=nf, spec=spec, viewer=viewer, context=context,
                           extra_warnings=extra_warnings)
    geo = BendGeometryFileV1.from_result(result)
    (out / ENGINE_OUTPUT_FILES["bend_geometry"]).write_bytes(canonical_json_bytes(geo.model_dump(mode="json")))
    (out / ENGINE_OUTPUT_FILES["log"]).write_text(f"{ORACLE_WARNING}\n", encoding="utf-8")
    tmp = out / ".result.json.tmp"
    tmp.write_bytes(canonical_json_bytes(result.model_dump(mode="json")))
    os.replace(tmp, out / ENGINE_OUTPUT_FILES["result"])
    return result


# ---------------------------------------------------------------------------
# E3-compatible command line (runner self-tests)
# ---------------------------------------------------------------------------


def _mode() -> dict[str, str]:
    out: dict[str, str] = {}
    for item in (os.environ.get("SHERMAN_E4_ORACLE_MODE") or "").split(","):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition("=")
        out[key.strip()] = value.strip()
    return out


def _find_case(scan_sha: str) -> Path | None:
    root = os.environ.get("SHERMAN_E4_ORACLE_CASES")
    if not root:
        return None
    for case_json in sorted(Path(root).glob("*/case.json")):
        try:
            doc = json.loads(case_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (doc.get("files") or {}).get("scan.ply") == scan_sha:
            return case_json.parent
    return None


def _emit(kind: str, payload: Mapping[str, Any]) -> None:
    sys.stdout.write(f"{kind}\t{json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n")
    sys.stdout.flush()


def _fail(out: Path, code: str, message: str) -> int:
    out.mkdir(parents=True, exist_ok=True)
    (out / "engine_error.json").write_text(json.dumps({"code": code, "message": message}) + "\n", encoding="utf-8")
    _emit("ERROR", {"code": code, "message": message})
    return 2


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m domains.production.engine.eval.oracle",
                                description="E4 TRUTH ORACLE (not an engine): E3-compatible CLI for harness self-tests")
    p.add_argument("--compiled", required=True, type=Path)
    p.add_argument("--recipe", required=True, type=Path)
    p.add_argument("--scan", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--scan-sha256", required=True)
    p.add_argument("--compiled-sha256")
    p.add_argument("--recipe-sha256")
    p.add_argument("--context", type=Path)
    p.add_argument("--declared-stage", type=int)
    p.add_argument("--declared-stage-source")
    p.add_argument("--unit-factor", type=float, default=1.0)
    p.add_argument("--time-budget-s", type=float, default=170.0)
    p.add_argument("--engine-bundle-id")
    p.add_argument("--config", type=Path)
    a = p.parse_args(argv)
    mode = _mode()
    out: Path = a.out
    if out.exists() and any(out.iterdir()):
        _emit("ERROR", {"code": "OUT_DIR_NOT_EMPTY", "message": str(out)})
        return 2
    _emit("PROGRESS", {"stage": "start", "progress": 0})
    if "sleep" in mode:
        time.sleep(float(mode["sleep"]))
    if "alloc_mb" in mode:
        block = bytearray(int(float(mode["alloc_mb"]) * 1024 * 1024))
        for i in range(0, len(block), 4096):
            block[i] = 1
        time.sleep(float(mode.get("hold_s", "1.0")))
    if "crash" in mode:
        os.kill(os.getpid(), 9)
    if "exit3" in mode:
        _emit("ERROR", {"code": "INTERNAL_ERROR", "message": "oracle exit3 mode"})
        return 3
    if not a.scan.is_file():
        return _fail(out, "SCAN_MISSING", str(a.scan))
    if sha256_file(a.scan) != a.scan_sha256:
        return _fail(out, "SCAN_HASH_MISMATCH", "scan bytes do not match --scan-sha256")
    if "exit2" in mode:
        return _fail(out, "COMPILE_BLOCKED", "oracle exit2 mode")
    from .engine_output import load_compiled, load_recipe

    try:
        compiled = load_compiled(a.compiled)
    except (OSError, ValueError) as exc:
        return _fail(out, "COMPILED_PART_MISSING", str(exc)[:300])
    if a.compiled_sha256 and compiled.compiled_part_sha256 != a.compiled_sha256:
        return _fail(out, "COMPILED_PART_HASH_MISMATCH", "compiled part hash differs from --compiled-sha256")
    recipe = load_recipe(a.recipe)
    if a.recipe_sha256 and recipe.recipe_sha256 != a.recipe_sha256:
        return _fail(out, "RECIPE_SHA_MISMATCH", "recipe hash differs from --recipe-sha256")
    case_dir = _find_case(a.scan_sha256)
    if case_dir is None:
        _emit("ERROR", {"code": "ORACLE_TRUTH_NOT_FOUND", "message": "set SHERMAN_E4_ORACLE_CASES"})
        return 3
    tc = load_truth_case(case_dir, verify_hashes=False)
    spec = OracleSpec(noise_mm=float(mode.get("noise", 0.0)), noise_deg=float(mode.get("noise", 0.0)),
                      seed=int(mode.get("seed", 0)), u_mm=float(mode.get("u_mm", 0.08)),
                      bias={"FLANGE_LENGTH": float(mode.get("bias_flange", 0.0))})
    out.mkdir(parents=True, exist_ok=True)
    if "garbage" in mode:
        (out / ENGINE_OUTPUT_FILES["result"]).write_text("{not json", encoding="utf-8")
        _emit("RESULT", {"result": "result.json"})
        return 0
    if "no_result" in mode:
        _emit("RESULT", {"result": "result.json"})
        return 0
    context = None
    if a.context is not None and a.context.is_file():
        context = ResultContextV1.model_validate(json.loads(a.context.read_text(encoding="utf-8")))
    leak = (f"scan_name:{a.scan.name}",) if "name_leak" in mode else ()
    result = write_oracle_output(out, recipe=recipe, compiled=compiled, compiled_dir=a.compiled.parent, truth=tc,
                                 scan_path=a.scan, spec=spec, context=context, unit_factor=a.unit_factor,
                                 extra_warnings=leak)
    if "modify_input" in mode:  # a misbehaving engine: write to its (read-only) input
        os.chmod(a.scan, 0o644)
        with open(a.scan, "ab") as fh:
            fh.write(b"\n")
    _emit("RESULT", {"summary_state": result.summary.state, "inspection_validity": result.inspection_validity.state,
                     "stage_status": result.stage_decision.status, "result": "result.json"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
