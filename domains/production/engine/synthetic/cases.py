"""Case plans (seed sets, tiers) and the per-case generator.

Seed sets (plan §10 amendment 4c):

- ``DEV``: for iteration (A1 loop, regression runs, debugging).
- ``QUAL``: **final-run only** (G1 is gated on QUAL). Seeds come from a
  different salt, so the two sets are disjoint by construction (asserted).
  Planning or generating QUAL needs ``allow_qual=True`` in the Python API
  (``--allow-qual`` on the CLI); otherwise :class:`SeedsetLockedError`.

Tiers (amendment 3):

- ``smoke``: one case per family at an explicit ~30k-point override, for tests.
- ``regression``: 7 cases per family + 1 symmetric = 50 cases at the scanner
  model's **native spacing**. It is the first 7 templates of ``full`` (same
  ids and seeds), so ``regression`` is a subset of ``full``.
- ``full``: 30 templates per family + 6 symmetric = 216 cases (>= 200), native
  spacing: the G1 suite.
- ``runtime``: 10 fixed large-part cases at an explicit ~1M-point override
  (runtime/RSS gate only; never scored for accuracy).

Scan density. ``point_target = None`` (``regression``, ``full``, part sets by
default) scans at the model's point spacing: the merge voxel is calibrated per
case so that the exported cloud's median nearest-neighbour distance equals
``ScannerModel.point_spacing_mm`` (within 3 %), the statistic the
amendment-4b model card measures on real exports; the point count is whatever
the part's size gives. An integer ``point_target`` is an explicit, recorded
override (``truth.scan.density.policy = "TARGET_POINTS_OVERRIDE"``).

Blindness (amendment 4). Nothing E3 receives names the case or its scenario:
SKUs, part ids and part names are opaque hashes of the seed, the recipe
carries no case id, and the recipe's ``inspection_stage`` is drawn by policy
rather than copied from the scenario (complete parts are released IN_PROCESS
25 % of the time, staged scans FINAL 25 % of the time; missing and extra bends
are always FINAL, as G1 requires). The case id <-> SKU mapping lives only in
``case.json`` and ``manifest.json``.

Every family's ``regression`` and ``full`` lists contain a stage-0 and a
stage-1 case, an acute bend, ``Ri >= 4t``, a flange <= 12 mm and a flange
>= 300 mm (amendment 5); :func:`coverage_report` verifies it on generated truth.
The features are properties of the scanned part, never of a random policy draw
(a stage-1 scan counts under a FINAL recipe too), so the templates guarantee the
coverage for every seed set, QUAL included.

Per case directory ``<out>/<case_id>/``:

=====================  ==========================================================
``part.step``          nominal folded CAD (OCP BRep, AP214, mm)
``flat.dxf``           nominal flat pattern (bend lines on BEND, angle notes)
``drawing.pdf``        synthetic drawing (flat + bend table)
``recipe_draft.json``  C01 builder inputs + human summary (SKU, order, groups, tolerances)
``recipe.json``        finalized PartRecipeV1 (canonical JSON)
``scan.ply``           virtual ZEISS scan, raw scanner frame (points or mesh)
``truth.json``         analytic truth (never give it to E3)
``truth_labels.npz``   per-scan-point truth labels (never give it to E3)
``case.json``          case metadata: spec, scenario, generation identity (scanner
                       model and scan plan hashes, options), ZEISS-like file name,
                       file hashes
=====================  ==========================================================

All files are byte-deterministic for a given (case id, seed, scanner model,
options); timings go to ``<out>/generation_log.jsonl`` only. A resumed run
skips a case only when its spec, generator version **and** generation identity
match; any other existing directory is refused unless ``force``.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .brep import (
    analytic_volume,
    build_pieces,
    check_single_valid_solid,
    fused_solid,
    mirror_shape,
    solid_volume,
    tessellate,
    write_step,
)
from .families import FAMILIES, FeatureRequest, build_family, random_placement
from .flat import write_drawing_pdf, write_flat_dxf
from .geometry import FoldedPart, FormState, GeometryError, PartDesign, apply_points, rigid_inverse
from .injection import Injection, Scenario, apply_scenario, far_parallel_edges, order_consistent
from .ply import write_ply
from .recipe import build_case_recipe, flange_inspection_overrides
from .scanner import SCAN_PLANS, ScannerModel, ScanResult, random_pose, scan_surface
from .symmetry import symmetry_test
from .truth import build_nominal, build_truth

__all__ = [
    "CasePolicyError",
    "CaseConflictError",
    "SeedsetLockedError",
    "opaque_sku",
    "generation_identity",
    "GENERATOR_VERSION",
    "CASE_SCHEMA",
    "MANIFEST_SCHEMA",
    "REVISIONS",
    "SEEDSETS",
    "TIERS",
    "FAMILY_CODES",
    "CaseSpec",
    "plan_cases",
    "generate_case",
    "generate_tier",
    "zeiss_scan_filename",
    "coverage_report",
    "case_seed",
    "PART_SET_SCHEMA",
    "PART_SET_SCANS",
    "generate_part_set",
    "load_case",
]

#: 1.1.1 (2026-10-17): ``features_present.stage_1`` is a property of the scan (exactly
#: the first bend of the tagged order formed, fully), no longer also of the recipe's
#: drawn inspection stage, so amendment-5 coverage holds by construction for every
#: seed set. Only ``case.json`` and the manifest coverage differ from 1.1.0; every
#: other case file is byte-identical.
GENERATOR_VERSION = "e2-synthetic-1.1.1"
CASE_SCHEMA = "sherman_synthetic_case_v1"
MANIFEST_SCHEMA = "sherman_synthetic_manifest_v1"
FAMILY_CODES = {
    "L_BRACKET": "LB",
    "U_CHANNEL": "UC",
    "Z_STEP": "ZS",
    "HAT": "HT",
    "CHANNEL_WITH_RETURNS": "CR",
    "TWO_DIRECTION_TAB": "TD",
    "PANEL_WITH_HOLES_NOTCHES": "PN",
}
#: Part revisions drawn for the recipes and the ZEISS-like names (letters and numeric).
REVISIONS: tuple[str, ...] = ("A", "B", "C", "01", "02")
#: Salt per seed set. QUAL is reserved for the final G1 run.
SEEDSETS = {"DEV": "e2-dev-2026-10-17", "QUAL": "e2-qual-2026-10-17-final-run-only"}
#: Seed sets that need an explicit opt-in (plan §10 amendment 4c: final run only).
LOCKED_SEEDSETS = frozenset({"QUAL"})
TIERS = ("smoke", "regression", "full", "runtime")
#: Scan density per tier: ``None`` = the scanner model's native spacing; an int is an
#: explicit point-count override (fast tests, runtime/RSS gate), recorded in the case.
_POINT_TARGET: dict[str, int | None] = {"smoke": 30_000, "regression": None, "full": None, "runtime": 1_000_000}
#: Initial point-count factor per scan plan (points ~ k * area / voxel^2), refined by later passes.
_PLAN_COUNT_FACTOR = {"FULL": 1.9, "TOP_ONLY": 1.0, "SPARSE": 0.6, "SINGLE": 0.3}
#: Initial guess of (median NN spacing / voxel) for native-density calibration, by
#: (several setups?, output). Refined per case by a secant iteration.
_NN_PER_VOXEL = {(True, "points"): 0.70, (True, "mesh"): 0.65, (False, "points"): 0.83, (False, "mesh"): 0.87}
_NN_TOLERANCE = 0.03
_MAX_DENSITY_PASSES = 5
#: Recipe inspection-stage policy (blindness): probability of the "other" stage per scenario class.
_COMPLETE_SCENARIOS = frozenset({"COMPLETE_NOMINAL", "COMPLETE_ERRORS", "COMPLETE_SPRINGBACK"})
_STAGED_SCENARIOS = frozenset({"STAGE_0", "STAGE_1", "STAGE_K", "PARTIAL_BEND", "ORDER_SWAP"})
_ALWAYS_FINAL_SCENARIOS = frozenset({"MISSING_BEND", "EXTRA_BEND"})
_CROSS_STAGE_PROBABILITY = 0.25


class CasePolicyError(Exception):
    """A request the generator refuses by policy (CLI exit 2), as opposed to an internal failure."""


class CaseConflictError(CasePolicyError, FileExistsError):
    """An existing case directory or manifest does not match the request (use a fresh
    directory, or ``force`` to regenerate)."""


class SeedsetLockedError(CasePolicyError, PermissionError):
    """QUAL seeds are final-run only (amendment 4c); pass ``allow_qual=True`` deliberately."""


def _require_seedset(seedset: str, allow_qual: bool) -> None:
    if seedset not in SEEDSETS:
        raise ValueError(f"unknown seed set {seedset!r}; known: {', '.join(SEEDSETS)}")
    if seedset in LOCKED_SEEDSETS and not allow_qual:
        raise SeedsetLockedError(
            f"{seedset} seeds are reserved for the final G1 run (plan §10 amendment 4c); pass allow_qual=True / --allow-qual"
        )


def _sha_doc(doc: Any) -> str:
    data = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def opaque_sku(seedset: str, seed: int, kind: str = "case") -> str:
    """Opaque synthetic SKU ``SYN-<10 hex>``: a hash of the seed-set salt and the
    seed, so it carries no family, template index or scenario (amendment 4)."""
    salt = SEEDSETS[seedset]
    return "SYN-" + hashlib.sha256(f"{salt}|sku|{kind}|{seed}".encode()).hexdigest()[:10].upper()


@dataclass(frozen=True)
class Template:
    scenario: str
    features: FeatureRequest = FeatureRequest()
    plan: str = "FULL"
    output: str = "points"
    exact: bool = False


F = FeatureRequest
_ROTATING = ("MISSING_BEND", "ORDER_SWAP", "EXTRA_BEND", "MIRRORED", "WRONG_SKU")
#: Per-family templates; the first 7 form the regression tier (index 6 rotates).
FULL_TEMPLATES: tuple[Template, ...] = (
    Template("COMPLETE_NOMINAL", F(), exact=True),
    Template("COMPLETE_ERRORS", F(acute=True), output="mesh"),
    Template("STAGE_0", F(large_radius=True)),
    Template("STAGE_1", F(short_flange=True)),
    Template("COMPLETE_ERRORS", F(long_flange=True)),
    Template("STAGE_K", F(), output="mesh"),
    Template("ROTATING", F()),
    Template("COMPLETE_SPRINGBACK", F()),
    Template("COMPLETE_ERRORS", F(large_radius=True)),
    Template("COMPLETE_NOMINAL", F(acute=True), plan="TOP_ONLY"),
    Template("STAGE_K", F(acute=True)),
    Template("PARTIAL_BEND", F()),
    Template("ORDER_SWAP", F()),
    Template("MISSING_BEND", F()),
    Template("EXTRA_BEND", F(), output="mesh"),
    Template("MIRRORED", F()),
    Template("WRONG_SKU", F()),
    Template("COMPLETE_ERRORS", F(short_flange=True)),
    Template("COMPLETE_ERRORS", F(), plan="SPARSE"),
    Template("STAGE_0", F(), output="mesh"),
    Template("STAGE_1", F(acute=True)),
    Template("COMPLETE_NOMINAL", F(long_flange=True), output="mesh"),
    Template("COMPLETE_SPRINGBACK", F(large_radius=True)),
    Template("STAGE_K", F(long_flange=True)),
    Template("COMPLETE_ERRORS", F(acute=True, large_radius=True)),
    Template("COMPLETE_NOMINAL", F(), plan="TOP_ONLY", output="mesh"),
    Template("PARTIAL_BEND", F(acute=True)),
    Template("COMPLETE_ERRORS", F()),
    Template("STAGE_K", F(), plan="TOP_ONLY"),
    Template("COMPLETE_NOMINAL", F(short_flange=True)),
)
SYMMETRIC_TEMPLATES: tuple[Template, ...] = (
    Template("COMPLETE_NOMINAL", F(symmetric=True), exact=True),
    Template("STAGE_1", F(symmetric=True)),
    Template("COMPLETE_ERRORS", F(symmetric=True)),
    Template("STAGE_0", F(symmetric=True)),
    Template("COMPLETE_NOMINAL", F(symmetric=True, acute=True)),
    Template("COMPLETE_NOMINAL", F(symmetric=True), plan="TOP_ONLY"),
)
_REGRESSION_COUNT = 7
#: Runtime cases use large parts (a flange >= 300 mm) so ~1M points keep a realistic spacing.
_RUNTIME_EXTRA = (("HAT", Template("STAGE_K", F(long_flange=True))),
                  ("PANEL_WITH_HOLES_NOTCHES", Template("COMPLETE_ERRORS", F(long_flange=True))),
                  ("TWO_DIRECTION_TAB", Template("COMPLETE_ERRORS", F(acute=True, long_flange=True))))
_MIN_SPACING_MM = 0.1


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    tier: str
    seedset: str
    family: str
    template_index: int
    scenario: str
    features: FeatureRequest
    plan: str
    output: str
    exact: bool
    seed: int
    #: ``None``: native scanner spacing; an int: explicit point-count override
    point_target: int | None
    sku: str

    def to_dict(self) -> dict[str, Any]:
        """Case identity as stored in ``case.json``. The tier is a property of the
        plan (recorded in ``manifest.json``), not of the case: ``regression`` is a
        subset of ``full`` and the same case must be byte-identical in both."""
        d = asdict(self)
        d.pop("tier")
        d["features"] = self.features.tags()
        return d


def case_seed(seedset: str, family: str, group: str, index: int) -> int:
    """Deterministic 63-bit seed of a case (salted per seed set)."""
    salt = SEEDSETS[seedset]
    digest = hashlib.sha256(f"{salt}|{family}|{group}|{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def plan_cases(tier: str, seedset: str, *, allow_qual: bool = False) -> list[CaseSpec]:
    """The ordered case list of a tier (no geometry is built). QUAL needs ``allow_qual``."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; known: {', '.join(TIERS)}")
    _require_seedset(seedset, allow_qual)
    target = _POINT_TARGET[tier]
    specs: list[CaseSpec] = []

    def add(family: str, group: str, code: str, index: int, tpl: Template, tier_name: str) -> None:
        scenario = tpl.scenario
        if scenario == "ROTATING":
            scenario = _ROTATING[FAMILIES.index(family) % len(_ROTATING)]
        seed = case_seed(seedset, family, group, index)
        specs.append(
            CaseSpec(
                case_id=f"E2-{seedset}-{code}-{index:03d}",
                tier=tier_name,
                seedset=seedset,
                family=family,
                template_index=index,
                scenario=scenario,
                features=tpl.features,
                plan=tpl.plan,
                output=tpl.output,
                exact=tpl.exact,
                seed=seed,
                point_target=target,
                sku=opaque_sku(seedset, seed),
            )
        )

    if tier == "smoke":
        for fam in FAMILIES:
            add(fam, "smoke", "SM" + FAMILY_CODES[fam], 0, Template("COMPLETE_NOMINAL", F()), tier)
        return specs
    if tier == "runtime":
        for fam in FAMILIES:
            add(fam, "runtime", "RT" + FAMILY_CODES[fam], 0, Template("COMPLETE_NOMINAL", F(long_flange=True)), tier)
        for i, (fam, tpl) in enumerate(_RUNTIME_EXTRA, start=1):
            add(fam, "runtime", "RT" + FAMILY_CODES[fam], i, tpl, tier)
        return specs
    count = _REGRESSION_COUNT if tier == "regression" else len(FULL_TEMPLATES)
    sym_count = 1 if tier == "regression" else len(SYMMETRIC_TEMPLATES)
    for fam in FAMILIES:
        for i in range(count):
            add(fam, "main", FAMILY_CODES[fam], i, FULL_TEMPLATES[i], tier)
    for i in range(sym_count):
        add("U_CHANNEL", "symmetric", "UCS", i, SYMMETRIC_TEMPLATES[i], tier)
    return specs


def zeiss_scan_filename(sku: str, revision: str, scenario: Scenario, rng: np.random.Generator, attempted: int | None = None) -> str:
    """A ZEISS-like export name (plan §1 real shapes). Informational only: the
    engine never sees it (amendment 4). The name follows what the operator
    meant to scan (``scenario.stage``: ``None`` = a finished part, even one with
    a missing bend), not the recipe's inspection stage. ``attempted``: bends
    the operator has worked on (formed or partially formed) in a mid-sequence
    scan."""
    stage = scenario.stage if attempted is None else attempted
    if scenario.stage is None:
        name = f"{sku}_{revision}_FULL_SCAN"
    elif stage == 0:
        return f"BEFORE BENDING  {sku}_{revision}.ply"
    else:
        name = f"{sku}_{revision}_SCAN_{stage}"
    roll = float(rng.random())
    if roll < 0.7:
        name += " 1"
    elif roll < 0.85:
        name += " 1 (1)"
    return name + ".ply"


# ---------------------------------------------------------------------------
# scenario construction
# ---------------------------------------------------------------------------


def _rngs(seed: int) -> dict[str, np.random.Generator]:
    names = ("geometry", "errors", "scan", "pose", "tolerance", "wrong", "placement", "naming", "symmetry", "policy")
    return {n: np.random.default_rng(np.random.SeedSequence([seed, i])) for i, n in enumerate(names)}


def _variation(exact: bool) -> list[Injection]:
    return [] if exact else [Injection("PROCESS_VARIATION", {"angle_sigma_deg": 0.15, "shift_sigma_mm": 0.03, "cut_sigma_mm": 0.03})]


def _free_flange_targets(design: PartDesign, nominal) -> list[tuple[int, str]]:
    """(sequence index, panel) pairs whose far boundary is a free edge on a non-datum panel."""
    out = []
    for si, key in enumerate(design.sequence, start=1):
        for letter in ("A", "B"):
            fb = nominal.flange[(key, letter)]
            panel = nominal.side_panel(key, letter)
            if panel == nominal.datum or fb.boundary_kind != "FREE_EDGE" or fb.kind_fraction < 0.999 or not fb.well_defined:
                continue
            if not far_parallel_edges(design, panel, key):
                continue
            out.append((si, panel))
    return out


def _signed(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.choice([-1.0, 1.0]) * rng.uniform(lo, hi))


def _build_scenario(spec: CaseSpec, design: PartDesign, nominal, rng: np.random.Generator) -> tuple[Scenario, PartDesign]:
    """Scenario for ``spec`` (may add a מהלך group to ``design`` for springback)."""
    n = len(design.sequence)
    var = _variation(spec.exact)
    name = spec.scenario
    if name == "ORDER_SWAP" and n < 2:
        name = "MISSING_BEND"
    if name == "STAGE_K" and n < 2:
        name = "PARTIAL_BEND"
    if name == "COMPLETE_NOMINAL":
        return Scenario(name, None, "FINAL", tuple(var)), design
    if name == "COMPLETE_ERRORS":
        inj = list(var)
        n_err = int(rng.integers(2, 4))
        kinds = list(rng.permutation(["ANGLE", "FLANGE_LENGTH", "BEND_LINE_SHIFT"]))[:n_err]
        targets = _free_flange_targets(design, nominal)
        for kind in kinds:
            si = int(rng.integers(1, n + 1))
            if kind == "ANGLE":
                inj.append(Injection("ANGLE", {"sequence_index": si, "delta_included_deg": _signed(rng, 0.2, 3.0)}))
            elif kind == "FLANGE_LENGTH" and targets:
                tsi, panel = targets[int(rng.integers(0, len(targets)))]
                inj.append(Injection("FLANGE_LENGTH", {"sequence_index": tsi, "panel": panel, "delta_mm": _signed(rng, 0.1, 1.5)}))
            elif kind == "BEND_LINE_SHIFT":
                inj.append(Injection("BEND_LINE_SHIFT", {"sequence_index": si, "delta_mm": _signed(rng, 0.1, 1.25)}))
        return Scenario(name, None, "FINAL", tuple(inj)), design
    if name == "COMPLETE_SPRINGBACK":
        if not design.groups:
            design = replace(design, groups=(tuple(design.sequence[: min(2, n)]),))
        inj = list(var)
        for gi in range(1, len(design.groups) + 1):
            inj.append(Injection("SPRINGBACK_GROUP", {"group_index": gi, "delta_included_deg": float(rng.uniform(0.5, 2.5))}))
        return Scenario(name, None, "FINAL", tuple(inj)), design
    if name == "STAGE_0":
        return Scenario(name, 0, "IN_PROCESS", tuple(var)), design
    if name == "STAGE_1":
        return Scenario(name, 1, "IN_PROCESS", tuple(var)), design
    if name == "STAGE_K":
        k = int(rng.integers(2, n)) if n >= 3 else 1
        inj = list(var)
        if rng.random() < 0.6:
            inj.append(Injection("BEND_LINE_SHIFT", {"sequence_index": k, "delta_mm": _signed(rng, 0.5, 1.5), "purpose": "backgauge error"}))
        return Scenario(name, k, "IN_PROCESS", tuple(inj)), design
    if name == "PARTIAL_BEND":
        k = int(rng.integers(0, n))
        inj = list(var) + [Injection("PARTIAL_BEND", {"sequence_index": k + 1, "fraction": float(rng.uniform(0.3, 0.7))})]
        return Scenario(name, k, "IN_PROCESS", tuple(inj)), design
    if name == "ORDER_SWAP":
        # a real order error: the bend formed early must rank strictly after the one
        # left out (bends of one מהלך group share a rank, so swapping inside a group
        # is a legal reorder, not an order error)
        triples = _out_of_order_swaps(design)
        if not triples:
            return _build_scenario(replace(spec, scenario="MISSING_BEND"), design, nominal, rng)
        k, i, j = triples[int(rng.integers(0, len(triples)))]
        inj = list(var) + [Injection("ORDER_SWAP", {"formed_out": i, "formed_in": j})]
        return Scenario(name, k, "IN_PROCESS", tuple(inj)), design
    if name == "MISSING_BEND":
        inj = list(var) + [Injection("MISSING_BEND", {"sequence_index": int(rng.integers(1, n + 1))})]
        return Scenario(name, None, "FINAL", tuple(inj)), design
    if name == "EXTRA_BEND":
        inj = list(var) + [Injection("EXTRA_BEND", {"fraction": float(rng.uniform(0.35, 0.65)), "sweep_deg": float(rng.uniform(15.0, 40.0)),
                                                     "direction": "UP" if rng.random() < 0.5 else "DOWN"})]
        return Scenario(name, None, "FINAL", tuple(inj)), design
    if name == "MIRRORED":
        return Scenario(name, None, "FINAL", tuple(var) + (Injection("MIRRORED", {"plane": "CAD x = 0"}),)), design
    if name == "WRONG_SKU":
        return Scenario(name, None, "FINAL", tuple(var) + (Injection("WRONG_SKU", {"scanned": "same family, other parameters"}),)), design
    raise ValueError(f"unknown scenario {name!r}")


def _out_of_order_swaps(design: PartDesign) -> list[tuple[int, int, int]]:
    """``(k, i, j)``: stage ``k`` with sequence bend ``i <= k`` left flat and ``j > k``
    formed instead, such that the formed set is out of order (``j``'s rank is above
    the lowest unformed rank)."""
    seq = list(design.sequence)
    n = len(seq)
    out = []
    for k in range(1, n):
        for i in range(1, k + 1):
            for j in range(k + 1, n + 1):
                formed = (set(seq[:k]) - {seq[i - 1]}) | {seq[j - 1]}
                if not order_consistent(design, formed):
                    out.append((k, i, j))
    return out


def _inspection_stage_policy(scenario: Scenario, rng: np.random.Generator) -> Scenario:
    """The recipe's inspection stage, drawn so that it does not reveal the scenario
    class to E3 (module docstring). Missing and extra bends stay FINAL (G1)."""
    draw = float(rng.random())
    if scenario.name in _ALWAYS_FINAL_SCENARIOS:
        stage = "FINAL"
    elif scenario.name in _STAGED_SCENARIOS:
        stage = "FINAL" if draw < _CROSS_STAGE_PROBABILITY else "IN_PROCESS"
    else:  # complete parts, mirrored, wrong part
        stage = "IN_PROCESS" if draw < _CROSS_STAGE_PROBABILITY else "FINAL"
    return replace(scenario, inspection_stage=stage)


def _tolerance_overrides(nominal, design: PartDesign, rng: np.random.Generator) -> dict[str, dict[str, Any]]:
    out = flange_inspection_overrides(nominal)
    if rng.random() < 0.25:
        key = design.sequence[int(rng.integers(0, len(design.sequence)))]
        entry = dict(out.get(key, {}))
        if rng.random() < 0.5:
            entry["flange_a_mm"] = {"plus": 0.3, "minus": 0.3}
        else:
            entry["angle_deg"] = {"plus": 1.5, "minus": 1.5}
        out[key] = entry
    return out


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def _write_json(path: Path, doc: Any) -> str:
    data = (json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Deterministic .npz (fixed zip timestamps; numpy's savez stamps wall time)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            arr_buf = io.BytesIO()
            np.lib.format.write_array(arr_buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, arr_buf.getvalue())
    return buf.getvalue()


def _checked_solid(fp: FoldedPart, what: str):
    pieces = build_pieces(fp)
    shape = fused_solid(pieces)
    check_single_valid_solid(shape)
    vol, ana = solid_volume(shape), analytic_volume(fp)
    if abs(vol - ana) > 1e-6 * max(1.0, ana):
        raise GeometryError(f"{what}: BRep volume {vol:.6f} != analytic {ana:.6f} (self-intersection)")
    return pieces, shape


def _scan(
    plan_name: str,
    output: str,
    point_target: int | None,
    surface_raw,
    model: ScannerModel,
    seed: int,
    area: float,
    ray_threads: int | None,
) -> tuple[ScanResult, dict[str, Any]]:
    """Scan at the requested density; returns the scan and its ``density`` record.

    ``point_target is None``: native density. The merge voxel is calibrated so the
    exported cloud's median nearest-neighbour spacing equals
    ``model.point_spacing_mm`` (secant iteration, within ``_NN_TOLERANCE``).
    Otherwise the voxel is calibrated to the point-count override. Every pass
    restarts the scan stream from the same seed, so the result is deterministic."""
    plan = SCAN_PLANS[plan_name]

    def run(voxel: float) -> ScanResult:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 99]))
        return scan_surface(surface_raw, model, plan, rng, spacing_mm=voxel, output=output, ray_threads=ray_threads)

    passes: list[tuple[float, float]] = []
    result: ScanResult | None = None
    if point_target is None:
        target = model.point_spacing_mm
        voxel = target / _NN_PER_VOXEL[(len(plan.setups) > 1, output)]
        for _ in range(_MAX_DENSITY_PASSES):
            voxel = max(_MIN_SPACING_MM, voxel)
            result = run(voxel)
            nn = float(result.stats["nn_spacing_mm"])
            passes.append((voxel, nn))
            if abs(nn / target - 1.0) <= _NN_TOLERANCE:
                break
            proportional = voxel * target / nn
            if len(passes) >= 2 and abs(passes[-1][1] - passes[-2][1]) > 1e-9:
                (v1, n1), (v2, n2) = passes[-2], passes[-1]
                step = v2 + (target - n2) * (v2 - v1) / (n2 - n1)
                # a secant step outside [0.5, 2] x the current voxel means a noisy response: fall back
                voxel = step if 0.5 * v2 <= step <= 2.0 * v2 else proportional
            else:
                voxel = proportional
        policy = "NATIVE_SPACING"
    else:
        voxel = max(_MIN_SPACING_MM, math.sqrt(_PLAN_COUNT_FACTOR[plan_name] * area / point_target))
        for _ in range(3):
            result = run(voxel)
            passes.append((voxel, float(result.stats["nn_spacing_mm"])))
            ratio = result.stats["points"] / point_target
            if 0.8 <= ratio <= 1.25:
                break
            voxel = max(_MIN_SPACING_MM, voxel * math.sqrt(ratio))
        policy = "TARGET_POINTS_OVERRIDE"
    assert result is not None
    nn = float(result.stats["nn_spacing_mm"])
    density = {
        "policy": policy,
        "override": point_target is not None,
        "model_point_spacing_mm": float(model.point_spacing_mm),
        "nn_spacing_mm": nn,
        "nn_spacing_vs_model": nn / float(model.point_spacing_mm),
        "voxel_mm": float(result.spacing_mm),
        "point_target": point_target,
        "points": int(result.stats["points"]),
        "passes": len(passes),
        "statistic": "median distance to the nearest other exported point (all points, outliers included)",
    }
    if point_target is not None:
        density["note"] = "explicit point-count override: density is not the scanner model's; never score accuracy on it"
    return result, density


def _write_part_files(
    out_dir: Path,
    design: PartDesign,
    nominal,
    solid_nom,
    *,
    sku: str,
    revision: str,
    part_name: str,
    inspection_stage: str,
    overrides: dict[str, dict[str, Any]],
):
    """Write part.step, flat.dxf, drawing.pdf, recipe.json and recipe_draft.json.

    These are the recipe-side (E1/E3-facing) files: they carry the opaque SKU and
    part name only, never the case id or scenario."""
    files: dict[str, str] = {}
    files["part.step"] = write_step(solid_nom, out_dir / "part.step", sku)
    files["flat.dxf"] = write_flat_dxf(design, out_dir / "flat.dxf")
    rows = []
    for k in design.sequence:
        b = design.bend(k)
        rows.append({
            "key": k, "label": f"B{design.sequence_index(k)}", "direction": nominal.final.contract_direction(k),
            "included_deg": 180.0 - b.sweep_deg, "sweep_deg": b.sweep_deg, "inner_radius_mm": b.inner_radius,
            "move": design.group_index(k) or "",
            "flange_a_mm": nominal.flange[(k, "A")].value_mm, "flange_b_mm": nominal.flange[(k, "B")].value_mm,
        })
    files["drawing.pdf"] = write_drawing_pdf(design, out_dir / "drawing.pdf", sku=sku, revision=revision,
                                             part_name=part_name, bend_rows=rows)
    case_recipe = build_case_recipe(
        design=design, nominal=nominal, sku=sku, revision=revision, part_name=part_name,
        cad_sha256=files["part.step"], drawing_sha256=files["drawing.pdf"], flat_sha256=files["flat.dxf"],
        inspection_stage=inspection_stage, bend_overrides=overrides,
    )
    recipe_bytes = case_recipe.recipe_bytes
    (out_dir / "recipe.json").write_bytes(recipe_bytes)
    files["recipe.json"] = hashlib.sha256(recipe_bytes).hexdigest()
    files["recipe_draft.json"] = _write_json(out_dir / "recipe_draft.json", case_recipe.draft_doc)
    return files, case_recipe


def _write_scan_files(
    out_dir: Path,
    *,
    case_id: str,
    design: PartDesign,
    nominal,
    case_recipe,
    scenario: Scenario,
    as_built,
    actual: FoldedPart,
    pieces_act,
    model: ScannerModel,
    pose: np.ndarray,
    scan_seed: int,
    plan_name: str,
    output: str,
    point_target: int | None,
    symmetry_rng: np.random.Generator,
    zeiss_name: str,
    ray_threads: int | None,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    """Scan the as-built part and write scan.ply, truth_labels.npz and truth.json."""
    files: dict[str, str] = {}
    surface_cad = tessellate(pieces_act, 0.005, 0.08)
    if as_built.mirror is not None:
        surface_cad = surface_cad.transformed(as_built.mirror)
    surface_raw = surface_cad.transformed(pose)
    scan, density = _scan(plan_name, output, point_target, surface_raw, model, scan_seed,
                          float(surface_cad.triangle_areas.sum()), ray_threads)
    files["scan.ply"] = write_ply(
        out_dir / "scan.ply", scan.points, scan.faces,
        comments=("generator sherman_qc_e2_synthetic", "units mm", "synthetic virtual ZEISS-like scan"),
    )
    ideal_cad = apply_points(rigid_inverse(pose), scan.ideal_raw)
    labels = {
        "segment_kind": scan.seg_kind,
        "segment_index": scan.seg_index,
        "skin": _contract_skin_labels(scan, actual),
        "view": scan.view,
        "setup": scan.setup,
        "flare": scan.flare,
        "ideal_cad": ideal_cad.astype(np.float32),
    }
    npz = _npz_bytes(labels)
    (out_dir / "truth_labels.npz").write_bytes(npz)
    files["truth_labels.npz"] = hashlib.sha256(npz).hexdigest()
    sym = None
    if not as_built.wrong_part:
        formed = set(as_built.formed_nominal)
        ref_fp = FoldedPart(design, FormState.nominal(design, [k for k in design.sequence if k in formed]), datum=nominal.datum)
        ref_surface = tessellate(build_pieces(ref_fp), 0.01, 0.1)
        lines = {k: ref_fp.bend_line_cad(k) for k in design.sequence if ref_fp.bend_geo[k].formed}
        sym = symmetry_test(ref_surface, lines, symmetry_rng, nthreads=ray_threads)
    scan_meta = {
        "file": "scan.ply",
        "format": "mesh" if scan.faces is not None else "points",
        "point_count": int(len(scan.points)),
        "face_count": None if scan.faces is None else int(len(scan.faces)),
        "frame": "raw_scanner_mm",
        "unit": "mm",
        # the model-card statistic: median nearest-neighbour distance of the exported points
        "spacing_mm": density["nn_spacing_mm"],
        "voxel_mm": density["voxel_mm"],
        "density": density,
        "scanner_model": model.to_dict(),
        "scanner_model_sha256": model.sha256(),
        "scan_plan": SCAN_PLANS[plan_name].to_dict(),
        "stats": scan.stats,
        "labels_file": "truth_labels.npz",
        "point_order": "random permutation (file order carries no label)",
        "zeiss_filename": zeiss_name,
    }
    label_dict = {"segment_kind": scan.seg_kind, "segment_index": scan.seg_index, "flat_skin": scan.skin}
    truth = build_truth(
        case_id=case_id, nominal=nominal, recipe=case_recipe.recipe, uid_by_key=case_recipe.uid_by_key,
        as_built=as_built, scenario=scenario, cad_to_raw=pose,
        labels=None if (as_built.wrong_part or as_built.mirror is not None) else label_dict,
        ideal_cad=np.where(np.isnan(ideal_cad), 0.0, ideal_cad), spacing_mm=density["nn_spacing_mm"], symmetry=sym,
        scan_meta=scan_meta,
    )
    files["truth.json"] = _write_json(out_dir / "truth.json", truth)
    return files, truth, scan_meta


def generation_identity(
    model: ScannerModel,
    plan_name: str,
    point_target: int | None,
    *,
    write_as_built_step: bool = False,
    write_stage_steps: bool = False,
) -> dict[str, Any]:
    """Everything besides the spec that changes a case's files. A resumed run reuses
    a case directory only when this (and the spec and generator version) match."""
    return {
        "scanner_model_sha256": model.sha256(),
        "scanner_model_name": model.name,
        "scanner_model_provenance": model.provenance,
        "scan_plan": plan_name,
        "scan_plan_sha256": SCAN_PLANS[plan_name].sha256(),
        "density_policy": "NATIVE_SPACING" if point_target is None else "TARGET_POINTS_OVERRIDE",
        "point_target": point_target,
        "options": {"as_built_step": bool(write_as_built_step), "stage_steps": bool(write_stage_steps)},
    }


def _scan_case_doc(*, case_id: str, spec: dict[str, Any], sku: str, revision: str, part_name: str, family: str, scenario: Scenario,
                   design: PartDesign, zeiss_name: str, scan_meta: dict[str, Any], plan_name: str, case_recipe, files: dict[str, str],
                   truth: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": CASE_SCHEMA,
        "generator_version": GENERATOR_VERSION,
        "case_id": case_id,
        "spec": spec,
        "generation": generation,
        "sku": sku,
        "revision": revision,
        "part_name": part_name,
        "family": family,
        "scenario": scenario.name,
        "inspection_stage": scenario.inspection_stage,
        "stage": scenario.stage,
        "injections": [i.to_dict() for i in scenario.injections],
        "bend_count": len(design.sequence),
        "thickness_mm": design.thickness,
        "zeiss_filename": zeiss_name,
        "scan_format": scan_meta["format"],
        "point_count": scan_meta["point_count"],
        "spacing_mm": scan_meta["spacing_mm"],
        "density_policy": scan_meta["density"]["policy"],
        "scan_plan": plan_name,
        "recipe_sha256": case_recipe.recipe.recipe_sha256,
        "cad_sha256": case_recipe.recipe.cad.sha256,
        "flat_sha256": case_recipe.recipe.flat.sha256 if case_recipe.recipe.flat else None,
        "features_present": _features_present(truth, design),
        "files": dict(sorted(files.items())),
        "custody": "SYNTHETIC: truth.json and truth_labels.npz are truth; never pass them, case.json or the ZEISS name to E3",
    }


def _existing_case(final_dir: Path, spec: CaseSpec, generation: dict[str, Any]) -> dict[str, Any]:
    """The ``case.json`` of an existing case directory when it matches the request
    (spec, generator version, generation identity); else :class:`CaseConflictError`."""
    existing = final_dir / "case.json"
    doc = None
    if existing.exists():
        try:
            doc = json.loads(existing.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            doc = None
    if doc is not None and (doc.get("spec"), doc.get("generator_version"), doc.get("generation")) == (
        spec.to_dict(), GENERATOR_VERSION, generation
    ):
        return doc
    differs = "unreadable case.json" if doc is None else ", ".join(
        name for name, a, b in (("spec", doc.get("spec"), spec.to_dict()),
                                ("generator version", doc.get("generator_version"), GENERATOR_VERSION),
                                ("scanner model / scan plan / options", doc.get("generation"), generation)) if a != b
    )
    raise CaseConflictError(f"{final_dir} exists and does not match the request ({differs}); use --force or a fresh --out")


def _part_name(sku: str) -> str:
    """Recipe part name: opaque (no family, case id or scenario)."""
    return f"Synthetic sheet-metal part {sku}"


def generate_case(
    spec: CaseSpec,
    out_root: str | os.PathLike,
    *,
    scanner_model: ScannerModel | None = None,
    force: bool = False,
    write_as_built_step: bool = False,
    write_stage_steps: bool = False,
    allow_qual: bool = False,
    ray_threads: int | None = None,
) -> dict[str, Any]:
    """Generate one case directory atomically; returns its manifest entry.

    ``write_as_built_step`` adds the scanned (as-built) BRep as ``as_built.step``
    (truth); ``write_stage_steps`` adds the nominal stage models S0..SN of the
    tagged order as ``stages/S<k>.step``. An existing directory is reused only
    when its spec, generator version and :func:`generation_identity` match;
    otherwise :class:`CaseConflictError` unless ``force``. QUAL specs need
    ``allow_qual``. ``ray_threads``: raycasting threads (``None``: from
    ``OMP_NUM_THREADS``); the output does not depend on it."""
    _require_seedset(spec.seedset, allow_qual)
    out_root = Path(out_root)
    final_dir = out_root / spec.case_id
    model = scanner_model or ScannerModel()
    generation = generation_identity(model, spec.plan, spec.point_target, write_as_built_step=write_as_built_step,
                                     write_stage_steps=write_stage_steps)
    if final_dir.exists() and not force:
        doc = _existing_case(final_dir, spec, generation)  # raises CaseConflictError on a mismatch
        return {"case_id": spec.case_id, "skipped": True, **_manifest_entry(doc)}
    t0 = time.time()
    rng = _rngs(spec.seed)
    tmp_dir = out_root / f".tmp-{spec.case_id}-{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    try:
        for attempt in range(50):
            design = build_family(spec.family, rng["geometry"], spec.features)
            if rng["placement"].random() < 0.5:
                design = replace(design, placement=tuple(tuple(r) for r in random_placement(rng["placement"]).tolist()))
            nominal = build_nominal(design)
            try:
                scenario, design = _build_scenario(spec, design, nominal, rng["errors"])
                nominal = build_nominal(design)
                _, solid_nom = _checked_solid(nominal.final, "nominal part")
                wrong_design = None
                if any(i.kind == "WRONG_SKU" for i in scenario.injections):
                    wrong_design = build_family(spec.family, rng["wrong"], spec.features)
                as_built, actual, pieces_act, solid_act = _as_built(design, scenario, rng["errors"], wrong_design)
                break
            except GeometryError:
                if attempt == 49:
                    raise
                continue
        scenario = _inspection_stage_policy(scenario, rng["policy"])
        revision = str(rng["naming"].choice(list(REVISIONS)))
        part_name = _part_name(spec.sku)
        overrides_rng = rng["tolerance"]
        files, case_recipe = _write_part_files(
            tmp_dir, design, nominal, solid_nom, sku=spec.sku, revision=revision, part_name=part_name,
            inspection_stage=scenario.inspection_stage, overrides=_tolerance_overrides(nominal, design, overrides_rng),
        )
        if write_as_built_step:
            shape = mirror_shape(solid_act, as_built.mirror) if as_built.mirror is not None else solid_act
            files["as_built.step"] = write_step(shape, tmp_dir / "as_built.step", spec.sku + " AS BUILT")
        if write_stage_steps:
            (tmp_dir / "stages").mkdir()
            for k, stage_fp in enumerate(nominal.stages):
                _, stage_solid = _checked_solid(stage_fp, f"stage S{k}")
                files[f"stages/S{k}.step"] = write_step(stage_solid, tmp_dir / "stages" / f"S{k}.step", f"{spec.sku} S{k}")
        pose = random_pose(rng["pose"])
        attempted = None if as_built.wrong_part else len(as_built.formed_nominal)
        zeiss_name = zeiss_scan_filename(spec.sku, revision, scenario, rng["naming"], attempted)
        scan_files, truth, scan_meta = _write_scan_files(
            tmp_dir, case_id=spec.case_id, design=design, nominal=nominal, case_recipe=case_recipe, scenario=scenario,
            as_built=as_built, actual=actual, pieces_act=pieces_act, model=model, pose=pose, scan_seed=spec.seed,
            plan_name=spec.plan, output=spec.output, point_target=spec.point_target, symmetry_rng=rng["symmetry"],
            zeiss_name=zeiss_name, ray_threads=ray_threads,
        )
        files.update(scan_files)
        case_doc = _scan_case_doc(
            case_id=spec.case_id, spec=spec.to_dict(), sku=spec.sku, revision=revision, part_name=part_name, family=spec.family,
            scenario=scenario, design=design, zeiss_name=zeiss_name, scan_meta=scan_meta, plan_name=spec.plan,
            case_recipe=case_recipe, files=files, truth=truth, generation=generation,
        )
        _write_json(tmp_dir / "case.json", case_doc)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(tmp_dir, final_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    entry = _manifest_entry(case_doc)
    entry["seconds"] = round(time.time() - t0, 3)
    return entry


def _as_built(design: PartDesign, scenario: Scenario, rng: np.random.Generator, wrong_design: PartDesign | None):
    """As-built part, its FoldedPart (with mirror) and checked unmirrored pieces/solid."""
    as_built = apply_scenario(design, scenario, rng, wrong_part_design=wrong_design)
    actual = FoldedPart(as_built.design, as_built.state, datum=as_built.datum, mirror=as_built.mirror)
    # solids are always built unmirrored; a mirrored part reflects the result
    actual_plain = FoldedPart(as_built.design, as_built.state, datum=as_built.datum)
    pieces_act, solid_act = _checked_solid(actual_plain, "as-built part")
    return as_built, actual, pieces_act, solid_act


def _contract_skin_labels(scan: ScanResult, actual: FoldedPart) -> np.ndarray:
    """Per-point contract skin (ENGINE_IO §1) of the **physical** part: skin 0 is the
    face of the scanned part's datum panel whose CAD-frame normal has the larger
    +Z (ties +Y, then +X). For a mirrored part the reflected datum normal is used,
    for a wrong part its own datum; edge walls (2) and outliers (255) pass through."""
    out = scan.skin.copy()
    try:
        same = actual.contract_skin0_is_flat_skin0()
    except GeometryError:
        return out
    if not same:
        m0 = scan.skin == 0
        m1 = scan.skin == 1
        out[m0] = 1
        out[m1] = 0
    return out


def _features_present(truth: dict[str, Any], design: PartDesign) -> dict[str, bool]:
    """Amendment-5 features of one case, as properties of the scanned part only.

    ``stage_0``: no bend formed (the blank). ``stage_1``: exactly the first bend of
    the tagged order formed, fully (no partial bend). Neither depends on the
    recipe's inspection stage, which is a policy draw (a stage-1 scan under a
    FINAL recipe is still a stage-1 case: FINAL_INCOMPLETE), so every family's
    STAGE_0 / STAGE_1 templates give the coverage for any seed set."""
    t = design.thickness
    lengths = [c["nominal"] for c in truth["characteristics"].values() if c.get("kind") == "FLANGE_LENGTH" and c.get("nominal") is not None]
    stage = truth["stage"]
    return {
        "stage_0": stage["formed_count"] == 0,
        "stage_1": stage["formed_count"] == 1 and bool(stage["is_sequence_prefix"]) and not stage["partially_formed_bend_uids"],
        "acute": any(b.included_deg < 90.0 for b in design.bends),
        "large_radius": any(b.inner_radius >= 4.0 * t - 1e-9 for b in design.bends),
        "short_flange": any(x <= 12.0 for x in lengths),
        "long_flange": any(x >= 300.0 for x in lengths),
        "symmetric": bool(truth["expected"]["symmetric"]),
    }


def _manifest_entry(case_doc: dict[str, Any]) -> dict[str, Any]:
    gen = case_doc.get("generation") or {}
    return {
        "case_id": case_doc["case_id"],
        "family": case_doc["family"],
        "scenario": case_doc["scenario"],
        "inspection_stage": case_doc["inspection_stage"],
        "sku": case_doc["sku"],
        "point_count": case_doc["point_count"],
        "spacing_mm": case_doc.get("spacing_mm"),
        "density_policy": case_doc.get("density_policy"),
        "scan_format": case_doc["scan_format"],
        "scanner_model_sha256": gen.get("scanner_model_sha256"),
        "features_present": case_doc["features_present"],
        "files": case_doc["files"],
    }


def coverage_report(entries: Iterable[dict[str, Any]]) -> dict[str, dict[str, bool]]:
    """Per family: which amendment-5 features at least one case contains."""
    keys = ("stage_0", "stage_1", "acute", "large_radius", "short_flange", "long_flange")
    out: dict[str, dict[str, bool]] = {}
    for e in entries:
        fam = out.setdefault(e["family"], {k: False for k in keys})
        for k in keys:
            fam[k] = fam[k] or bool(e["features_present"].get(k))
    return out


def generate_tier(
    tier: str,
    seedset: str,
    out_root: str | os.PathLike,
    *,
    families: Iterable[str] | None = None,
    case_ids: Iterable[str] | None = None,
    limit: int | None = None,
    scanner_model: ScannerModel | None = None,
    force: bool = False,
    write_as_built_step: bool = False,
    write_stage_steps: bool = False,
    allow_qual: bool = False,
    ray_threads: int | None = None,
    progress=None,
) -> dict[str, Any]:
    """Generate (or resume) a tier into ``out_root`` and write ``manifest.json``.

    Chunked runs merge into an existing manifest of the same tier, seed set and
    generator version. Every case of one manifest must come from the same scanner
    model: a request whose model differs from the manifest's is refused up front
    (:class:`CaseConflictError`) unless ``force`` regenerates every listed case."""
    _require_seedset(seedset, allow_qual)
    out = Path(out_root)
    model = scanner_model or ScannerModel()
    model_sha = model.sha256()
    all_specs = plan_cases(tier, seedset, allow_qual=allow_qual)
    fam_set = set(families) if families else None
    id_set = set(case_ids) if case_ids else None
    specs = [s for s in all_specs if (fam_set is None or s.family in fam_set) and (id_set is None or s.case_id in id_set)]
    if limit is not None:
        specs = specs[:limit]
    previous = _load_manifest(out / "manifest.json")
    merge_old: list[dict[str, Any]] = []
    if previous is not None and (previous.get("schema"), previous.get("generator_version"), previous.get("tier"),
                                 previous.get("seedset")) == (MANIFEST_SCHEMA, GENERATOR_VERSION, tier, seedset):
        requested = {s.case_id for s in specs}
        merge_old = [c for c in previous.get("cases", []) if c.get("case_id") not in requested]
        stale = sorted(c["case_id"] for c in merge_old if c.get("scanner_model_sha256") != model_sha)
        if stale:
            raise CaseConflictError(
                f"{out} already lists {len(stale)} case(s) generated with another scanner model "
                f"(e.g. {stale[0]}); one manifest holds one model: use a fresh --out, or regenerate every case with --force"
            )
    if not force:  # refuse a mismatching existing case before generating anything
        for spec in specs:
            if (out / spec.case_id).exists():
                _existing_case(out / spec.case_id, spec, generation_identity(
                    model, spec.plan, spec.point_target, write_as_built_step=write_as_built_step,
                    write_stage_steps=write_stage_steps))
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    log = out / "generation_log.jsonl"
    for i, spec in enumerate(specs, start=1):
        entry = generate_case(spec, out, scanner_model=model, force=force, write_as_built_step=write_as_built_step,
                              write_stage_steps=write_stage_steps, allow_qual=allow_qual, ray_threads=ray_threads)
        entries.append(entry)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"case_id": spec.case_id, "seconds": entry.get("seconds"), "skipped": entry.get("skipped", False)}) + "\n")
        if progress is not None:
            progress(i, len(specs), entry)
    clean = [{k: v for k, v in e.items() if k not in ("seconds", "skipped")} for e in entries]
    merged = {c["case_id"]: c for c in merge_old}
    merged.update({c["case_id"]: c for c in clean})
    order = {s.case_id: i for i, s in enumerate(all_specs)}
    clean = sorted(merged.values(), key=lambda c: order.get(c["case_id"], len(order)))
    models = sorted({c.get("scanner_model_sha256") for c in clean}, key=str)
    if models != [model_sha]:  # defensive: every listed case must carry this run's model
        raise CaseConflictError(f"manifest cases disagree on the scanner model: {models}")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generator_version": GENERATOR_VERSION,
        "tier": tier,
        "seedset": seedset,
        "seedset_policy": "QUAL is final-run only (plan §10 amendment 4c); DEV is for iteration",
        "scanner_model": model.to_dict(),
        "scanner_model_sha256": model_sha,
        "density_policy": "NATIVE_SPACING" if _POINT_TARGET[tier] is None else "TARGET_POINTS_OVERRIDE",
        "case_count": len(clean),
        "families": sorted({e["family"] for e in clean}),
        "coverage": coverage_report(clean),
        "cases": clean,
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


# ---------------------------------------------------------------------------
# part sets (one released part, several scans) for onboarding / E2E flows
# ---------------------------------------------------------------------------

PART_SET_SCHEMA = "sherman_synthetic_part_set_v1"
#: Scans of a part set, in inbox drop order.
PART_SET_SCANS = ("nominal", "errors", "stage_k", "stage_0", "unknown_sku", "wrong_part")
#: Amendment-10 E2E error scan: +1.5 mm on a free flange of B2, +2 deg on B3.
E2E_FLANGE_ERROR_MM = 1.5
E2E_ANGLE_ERROR_DEG = 2.0


def generate_part_set(
    family: str,
    seedset: str,
    out_root: str | os.PathLike,
    *,
    index: int = 0,
    scanner_model: ScannerModel | None = None,
    target_points: int | None = None,
    force: bool = False,
    allow_qual: bool = False,
    ray_threads: int | None = None,
) -> dict[str, Any]:
    """One part (STEP, flat, drawing, FINAL recipe) plus several scans of it.

    Layout ``<out>/<set_id>/``: the part files, ``part_set.json`` and
    ``scans/<name>/{scan.ply, truth.json, truth_labels.npz, case.json}`` for
    :data:`PART_SET_SCANS`. ``errors`` carries exactly +1.5 mm on a free flange
    side of B2 and +2.0 deg on B3 (no other variation), ``stage_k`` is an
    incomplete scan under the FINAL recipe (FINAL_INCOMPLETE), ``unknown_sku``
    is the nominal scan under an unknown SKU name (quarantine, then assign) and
    ``wrong_part`` is another part scanned under this SKU. The family needs >= 3
    bends (HAT, CHANNEL_WITH_RETURNS, TWO_DIRECTION_TAB, PANEL_WITH_HOLES_NOTCHES).
    Scans use the scanner model's native spacing unless ``target_points`` is
    given (an explicit, recorded override)."""
    if family not in FAMILY_CODES:
        raise ValueError(f"unknown family {family!r}")
    _require_seedset(seedset, allow_qual)
    code = FAMILY_CODES[family]
    set_id = f"E2-{seedset}-PS{code}-{index:02d}"
    seed = case_seed(seedset, family, "partset", index)
    sku = opaque_sku(seedset, seed, "partset")
    out_root = Path(out_root)
    final_dir = out_root / set_id
    if final_dir.exists() and not force:
        raise CaseConflictError(f"{final_dir} exists (use --force to regenerate)")
    model = scanner_model or ScannerModel()
    rng = _rngs(seed)
    design = nominal = solid_nom = None
    b2_panel = None
    for _ in range(100):
        cand = build_family(family, rng["geometry"], FeatureRequest())
        if rng["placement"].random() < 0.5:
            cand = replace(cand, placement=tuple(tuple(r) for r in random_placement(rng["placement"]).tolist()))
        if len(cand.sequence) < 3:
            raise ValueError(f"{family} has {len(cand.sequence)} bends; part sets need >= 3 (B3 angle error)")
        nom = build_nominal(cand)
        targets = [panel for si, panel in _free_flange_targets(cand, nom) if si == 2]
        if not targets:
            continue
        try:
            _, solid = _checked_solid(nom.final, "nominal part")
        except GeometryError:
            continue
        design, nominal, solid_nom, b2_panel = cand, nom, solid, targets[0]
        break
    if design is None:
        raise GeometryError(f"{family}: no part with a free flange on B2 after 100 draws")
    n = len(design.sequence)
    k = max(1, n // 2)
    revision = str(rng["naming"].choice(list(REVISIONS)))
    part_name = _part_name(sku)
    tmp_dir = out_root / f".tmp-{set_id}-{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    try:
        files, case_recipe = _write_part_files(
            tmp_dir, design, nominal, solid_nom, sku=sku, revision=revision, part_name=part_name,
            inspection_stage="FINAL", overrides=flange_inspection_overrides(nominal),
        )
        var = _variation(False)
        scenarios = {
            "nominal": Scenario("NOMINAL", None, "FINAL", tuple(var)),
            "errors": Scenario("ERRORS", None, "FINAL", (
                Injection("FLANGE_LENGTH", {"sequence_index": 2, "panel": b2_panel, "delta_mm": E2E_FLANGE_ERROR_MM}),
                Injection("ANGLE", {"sequence_index": 3, "delta_included_deg": E2E_ANGLE_ERROR_DEG}),
            )),
            "stage_k": Scenario("STAGE_K", k, "FINAL", tuple(var)),
            "stage_0": Scenario("STAGE_0", 0, "FINAL", tuple(var)),
            "wrong_part": Scenario("WRONG_SKU", None, "FINAL", tuple(var) + (Injection("WRONG_SKU", {"scanned": "same family, other parameters"}),)),
        }
        names = {
            "nominal": f"{sku}_{revision}_FULL_SCAN 1.ply",
            "errors": f"{sku}_{revision}_FULL_SCAN 2.ply",
            "stage_k": f"{sku}_{revision}_SCAN_{k} 1.ply",
            "stage_0": f"BEFORE BENDING  {sku}_{revision}.ply",
            "wrong_part": f"{sku}_{revision}_FULL_SCAN 3.ply",
            "unknown_sku": f"{opaque_sku(seedset, seed, 'partset-unknown')}_A_FULL_SCAN 1.ply",
        }
        scans_meta = []
        for i, name in enumerate(PART_SET_SCANS):
            scan_dir = tmp_dir / "scans" / name
            scan_dir.mkdir(parents=True)
            if name == "unknown_sku":
                src = tmp_dir / "scans" / "nominal"
                for f in ("scan.ply", "truth.json", "truth_labels.npz"):
                    shutil.copyfile(src / f, scan_dir / f)
                nominal_case = json.loads((src / "case.json").read_text(encoding="utf-8"))
                doc = {**nominal_case, "case_id": f"{set_id}/{name}", "scenario": "UNKNOWN_SKU_NAME", "zeiss_filename": names[name],
                       "note": "nominal scan bytes exported under an unknown SKU: expect quarantine (UNMATCHED_SKU), then assign the part's SKU"}
                _write_json(scan_dir / "case.json", doc)
                scans_meta.append({"name": name, "dir": f"scans/{name}", "zeiss_filename": names[name], "scenario": "UNKNOWN_SKU_NAME",
                                   "expected": {"quarantine_reason": "UNMATCHED_SKU", "after_assign": "same as nominal"}})
                continue
            srng = _rngs(case_seed(seedset, family, f"partset-{index}-{name}", 0))
            scenario = scenarios[name]
            wrong = build_family(family, srng["wrong"], FeatureRequest()) if name == "wrong_part" else None
            as_built, actual, pieces_act, _ = _as_built(design, scenario, srng["errors"], wrong)
            scan_files, truth, scan_meta = _write_scan_files(
                scan_dir, case_id=f"{set_id}/{name}", design=design, nominal=nominal, case_recipe=case_recipe, scenario=scenario,
                as_built=as_built, actual=actual, pieces_act=pieces_act, model=model, pose=random_pose(srng["pose"]),
                scan_seed=case_seed(seedset, family, f"partset-{index}-{name}", 1), plan_name="FULL", output="points",
                point_target=target_points, symmetry_rng=srng["symmetry"], zeiss_name=names[name], ray_threads=ray_threads,
            )
            spec = {"set_id": set_id, "scan": name, "family": family, "seedset": seedset, "index": index, "point_target": target_points}
            case_doc = _scan_case_doc(
                case_id=f"{set_id}/{name}", spec=spec, sku=sku, revision=revision, part_name=part_name, family=family,
                scenario=scenario, design=design, zeiss_name=names[name], scan_meta=scan_meta, plan_name="FULL",
                case_recipe=case_recipe, files=scan_files, truth=truth,
                generation=generation_identity(model, "FULL", target_points),
            )
            _write_json(scan_dir / "case.json", case_doc)
            expected = {
                "summary_never": truth["expected"]["summary_never"],
                "bend_states_contract": truth["expected"]["bend_states_contract"],
                "characteristics_never_pass": truth["expected"]["characteristics_never_pass"],
                "characteristics_must_be_unknown": truth["expected"]["characteristics_must_be_unknown"],
            }
            if name == "errors":
                expected["injected"] = {
                    cid: {"truth": c["truth"], "nominal": c["nominal"], "true_deviation": c["true_deviation"]}
                    for cid, c in truth["characteristics"].items()
                    if c.get("true_deviation") is not None and abs(c["true_deviation"]) > 1e-6
                }
            scans_meta.append({"name": name, "dir": f"scans/{name}", "zeiss_filename": names[name], "scenario": scenario.name,
                               "expected": expected})
        doc = {
            "schema": PART_SET_SCHEMA,
            "generator_version": GENERATOR_VERSION,
            "set_id": set_id,
            "family": family,
            "seedset": seedset,
            "sku": sku,
            "revision": revision,
            "part_name": part_name,
            "inspection_stage": "FINAL",
            "recipe_sha256": case_recipe.recipe.recipe_sha256,
            "cad_sha256": files["part.step"],
            "flat_sha256": files["flat.dxf"],
            "files": dict(sorted(files.items())),
            "e2e_errors": {"flange_on_B2_mm": E2E_FLANGE_ERROR_MM, "flange_panel": b2_panel, "angle_on_B3_deg": E2E_ANGLE_ERROR_DEG},
            "scanner_model_sha256": model.sha256(),
            "density_policy": "NATIVE_SPACING" if target_points is None else "TARGET_POINTS_OVERRIDE",
            "drop_order": list(PART_SET_SCANS),
            "scans": scans_meta,
            "custody": "SYNTHETIC: never pass truth.json, truth_labels.npz or case.json to E3",
        }
        _write_json(tmp_dir / "part_set.json", doc)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(tmp_dir, final_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return doc


def load_case(case_dir: str | os.PathLike, *, verify: bool = True) -> dict[str, Any]:
    """Read a case directory (``case.json`` + ``truth.json``); with ``verify`` every
    file hash listed in ``case.json`` is checked. Raises ``ValueError`` on a
    schema or hash mismatch (a tampered or partially copied case)."""
    root = Path(case_dir)
    case = json.loads((root / "case.json").read_text(encoding="utf-8"))
    if case.get("schema") != CASE_SCHEMA:
        raise ValueError(f"{root}: not an E2 case ({case.get('schema')!r})")
    if verify:
        for name, digest in case["files"].items():
            path = root / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"{root}: {name} is missing or does not match case.json")
    truth = json.loads((root / "truth.json").read_text(encoding="utf-8"))
    return {"case": case, "truth": truth, "paths": {name: str(root / name) for name in case["files"]}}
