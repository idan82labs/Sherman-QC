"""Loaders for E2 synthetic truth (``truth.json`` + ``case.json`` + ``manifest.json``).

The E2 suite writes, per case directory, ``case.json`` (spec, generation
identity, file hashes) and ``truth.json`` (schema ``sherman_synthetic_truth_v1``,
analytic truth keyed by the contract characteristic ids). This module turns
them into frozen, typed records for the scorer. It reads JSON only and imports
nothing from E2 (so the harness never depends on the generator's geometry code)
and nothing from E1 or E3.

Scoring rule (E2 truth contract, ENGINE_IO §1): a characteristic is
*scorable* when it is ``applicable``, its ``truth`` is not null and it is not
ill-defined (``well_defined`` is not false). A backgauge characteristic carries
truth for both sides (``per_side``); E1 attaches the contract characteristic to
one side, and only that side is scored (:meth:`TruthCase.backgauge_side`).

Undefined truth is never replaced by a nominal: ``truth`` stays ``None`` and
``truth_undefined_reason`` says why (WRONG_PART_SCANNED, MIRRORED_PART,
BEND_NOT_FORMED, ADJACENT_BEND_NOT_FORMED, NO_FAR_BOUNDARY).
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "TRUTH_SCHEMA",
    "CASE_SCHEMA",
    "MANIFEST_SCHEMA",
    "FORMED_LIKE",
    "TruthError",
    "TruthSide",
    "TruthCharacteristic",
    "TruthBend",
    "TruthCase",
    "load_truth_case",
    "load_manifest",
    "iter_tier_cases",
    "scorable",
]

TRUTH_SCHEMA = "sherman_synthetic_truth_v1"
CASE_SCHEMA = "sherman_synthetic_case_v1"
MANIFEST_SCHEMA = "sherman_synthetic_manifest_v1"
FORMED_LIKE = frozenset({"FORMED", "PARTIALLY_FORMED", "UNEXPECTED_FORMED"})


class TruthError(ValueError):
    """A case directory is not a valid, untampered E2 case."""


def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)


def _tol(doc: Mapping[str, Any] | None) -> tuple[float, float] | None:
    if not doc:
        return None
    return float(doc["plus"]), float(doc["minus"])


def _line(doc: Mapping[str, Any] | None) -> np.ndarray | None:
    if not doc:
        return None
    return np.asarray([doc["start_mm"], doc["end_mm"]], dtype=np.float64)


@dataclass(frozen=True)
class TruthSide:
    """One backgauge side (``truth.characteristics[<cid>].per_side[<A|B>]``)."""

    side: str
    nominal: float | None
    truth: float | None
    true_deviation: float | None
    truth_in_tolerance: bool | None
    well_defined: bool
    ill_defined_reason: str | None
    tolerance: tuple[float, float] | None
    observable: bool | None


@dataclass(frozen=True)
class TruthCharacteristic:
    characteristic_id: str
    kind: str
    unit: str
    bend_uid: str | None
    side: str | None
    stage_index: int | None
    nominal: float | None
    truth: float | None
    true_deviation: float | None
    truth_in_tolerance: bool | None
    tolerance: tuple[float, float] | None
    recipe_requirement: str | None
    applicable: bool
    not_applicable_reason: str | None
    truth_undefined_reason: str | None
    #: None for kinds without a well-definedness test (angle, position).
    well_defined: bool | None
    ill_defined_reason: str | None
    observable: bool | None
    #: Backgauge only: truth for each side that leads to unformed material.
    per_side: Mapping[str, TruthSide] = field(default_factory=dict)

    def for_side(self, side: str | None) -> "TruthCharacteristic":
        """A backgauge resolved to one side (identity for other kinds). An unknown
        side yields an unscorable record (truth None, reason BACKGAUGE_SIDE_UNKNOWN)."""
        if self.kind != "BACKGAUGE_LENGTH":
            return self
        s = self.per_side.get(side or "")
        if s is None:
            return TruthCharacteristic(
                characteristic_id=self.characteristic_id, kind=self.kind, unit=self.unit, bend_uid=self.bend_uid,
                side=side, stage_index=self.stage_index, nominal=None, truth=None, true_deviation=None,
                truth_in_tolerance=None, tolerance=self.tolerance, recipe_requirement=self.recipe_requirement,
                applicable=self.applicable, not_applicable_reason=self.not_applicable_reason,
                truth_undefined_reason="BACKGAUGE_SIDE_UNKNOWN", well_defined=False,
                ill_defined_reason="BACKGAUGE_SIDE_UNKNOWN", observable=None, per_side=self.per_side,
            )
        return TruthCharacteristic(
            characteristic_id=self.characteristic_id, kind=self.kind, unit=self.unit, bend_uid=self.bend_uid,
            side=s.side, stage_index=self.stage_index, nominal=s.nominal, truth=s.truth,
            true_deviation=s.true_deviation, truth_in_tolerance=s.truth_in_tolerance,
            tolerance=s.tolerance or self.tolerance, recipe_requirement=self.recipe_requirement,
            applicable=self.applicable, not_applicable_reason=self.not_applicable_reason,
            truth_undefined_reason=None if s.truth is not None else "NO_FAR_BOUNDARY",
            well_defined=s.well_defined, ill_defined_reason=s.ill_defined_reason, observable=s.observable,
            per_side=self.per_side,
        )


def scorable(c: TruthCharacteristic) -> bool:
    """E2 scoring rule: applicable, truth defined, not ill-defined."""
    return bool(c.applicable and c.truth is not None and c.well_defined is not False)


@dataclass(frozen=True)
class TruthBend:
    bend_uid: str
    label: str
    sequence_index: int
    machine_move_index: int | None
    #: Contract-vocabulary truth state (UNEXPECTED_FORMED for an out-of-order formed
    #: bend); None when undefined (wrong part).
    state: str | None
    inner_radius_mm: float
    thickness_mm: float
    nominal_included_deg: float
    #: As-built loci in cad_model_mm (datum cut features at nominal CAD position);
    #: None for a bend that is not formed or an undefined part.
    actual_oml: np.ndarray | None
    actual_outer_apex: np.ndarray | None

    @property
    def formed_like(self) -> bool | None:
        return None if self.state is None else self.state in FORMED_LIKE


@dataclass(frozen=True)
class TruthCase:
    case_id: str
    case_dir: Path
    family: str
    scenario: str
    seedset: str | None
    sku: str
    inspection_stage: str
    scan_plan: str
    scan_output: str | None
    density_policy: str
    point_count: int
    spacing_mm: float | None
    scanner_model_sha256: str | None
    scanner_model_provenance: str | None
    generator_version: str | None
    mirrored: bool
    wrong_sku: bool
    mirror_distinguishable_by_rms: bool | None
    #: E2 truth-side symmetry (diagnostic; E1's compile-time test is the authority).
    symmetric_e2: bool
    ambiguous_groups_e2: tuple[tuple[str, ...], ...]
    formed_bend_uids: tuple[str, ...] | None
    formed_count: int | None
    partially_formed_bend_uids: tuple[str, ...]
    is_sequence_prefix: bool | None
    order_consistent: bool | None
    out_of_order_bend_uids: tuple[str, ...]
    expected_stage_model_id: str | None
    missing_bend_uids_final: tuple[str, ...]
    extra_bend_count: int
    injections: tuple[Mapping[str, Any], ...]
    bends: tuple[TruthBend, ...]
    characteristics: Mapping[str, TruthCharacteristic]
    summary_never: frozenset[str]
    never_pass: frozenset[str]
    never_fail: frozenset[str]
    must_be_unknown: frozenset[str]
    ill_defined: frozenset[str]
    #: Raw scan frame -> cad_model_mm (registration truth); None when truth.json has none
    #: (never silently replaced by the identity).
    raw_to_cad: np.ndarray | None
    files: Mapping[str, str]

    @property
    def defined_part(self) -> bool:
        """The physical part is the recipe part (not mirrored, not a wrong part)."""
        return not (self.mirrored or self.wrong_sku)

    @property
    def complete(self) -> bool:
        """Every planned bend formed, none partial, no extra bend, defined part."""
        return bool(
            self.defined_part
            and self.formed_count is not None
            and self.formed_count == len(self.bends)
            and not self.partially_formed_bend_uids
            and self.extra_bend_count == 0
        )

    @property
    def full_visibility(self) -> bool:
        return self.scan_plan == "FULL"

    @property
    def native_density(self) -> bool:
        return self.density_policy == "NATIVE_SPACING"

    def bend(self, uid: str) -> TruthBend:
        for b in self.bends:
            if b.bend_uid == uid:
                return b
        raise KeyError(uid)

    def never_pass_key(self, c: TruthCharacteristic) -> bool:
        """Is ``c`` (side-resolved for a backgauge) on the never-PASS list?"""
        key = f"{c.characteristic_id}#{c.side}" if c.kind == "BACKGAUGE_LENGTH" else c.characteristic_id
        return key in self.never_pass

    def ill_defined_key(self, c: TruthCharacteristic) -> bool:
        key = f"{c.characteristic_id}#{c.side}" if c.kind == "BACKGAUGE_LENGTH" else c.characteristic_id
        return key in self.ill_defined


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _side(letter: str, doc: Mapping[str, Any]) -> TruthSide:
    obs = doc.get("observability") or {}
    return TruthSide(
        side=letter,
        nominal=_opt_float(doc.get("nominal_mm")),
        truth=_opt_float(doc.get("truth_mm")),
        true_deviation=_opt_float(doc.get("true_deviation_mm")),
        truth_in_tolerance=doc.get("truth_in_tolerance"),
        well_defined=bool(doc.get("well_defined")),
        ill_defined_reason=doc.get("ill_defined_reason"),
        tolerance=_tol(doc.get("tolerance")),
        observable=obs.get("observable"),
    )


def _characteristic(cid: str, doc: Mapping[str, Any]) -> TruthCharacteristic:
    kind = doc["kind"]
    per_side = {}
    if kind == "BACKGAUGE_LENGTH":
        per_side = {k: _side(k, v) for k, v in (doc.get("per_side") or {}).items()}
    obs = doc.get("observability") or {}
    return TruthCharacteristic(
        characteristic_id=cid,
        kind=kind,
        unit=doc.get("unit") or ("deg" if kind == "INCLUDED_ANGLE" else "mm"),
        bend_uid=doc.get("bend_uid"),
        side=doc.get("side"),
        stage_index=doc.get("stage_index"),
        nominal=_opt_float(doc.get("nominal")),
        truth=_opt_float(doc.get("truth")),
        true_deviation=_opt_float(doc.get("true_deviation")),
        truth_in_tolerance=doc.get("truth_in_tolerance"),
        tolerance=_tol(doc.get("tolerance")),
        recipe_requirement=doc.get("recipe_requirement"),
        applicable=bool(doc.get("applicable")),
        not_applicable_reason=doc.get("not_applicable_reason"),
        truth_undefined_reason=doc.get("truth_undefined_reason"),
        well_defined=doc.get("well_defined") if kind in ("FLANGE_LENGTH",) else None,
        ill_defined_reason=doc.get("ill_defined_reason"),
        observable=obs.get("observable"),
        per_side=per_side,
    )


def load_truth_case(case_dir: str | os.PathLike, *, verify_hashes: bool = True) -> TruthCase:
    """Read and type one E2 case directory.

    With ``verify_hashes`` every file hash listed in ``case.json`` is checked
    (a tampered or partially copied case raises :class:`TruthError`)."""
    root = Path(case_dir)
    try:
        case = json.loads((root / "case.json").read_text(encoding="utf-8"))
        truth = json.loads((root / "truth.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TruthError(f"{root}: unreadable case.json/truth.json ({exc})") from exc
    if case.get("schema") != CASE_SCHEMA:
        raise TruthError(f"{root}: not an E2 case ({case.get('schema')!r})")
    if truth.get("schema") != TRUTH_SCHEMA:
        raise TruthError(f"{root}: unsupported truth schema {truth.get('schema')!r}")
    if truth.get("case_id") != case.get("case_id"):
        raise TruthError(f"{root}: case.json and truth.json name different cases")
    files = dict(case.get("files") or {})
    if verify_hashes:
        for name, digest in files.items():
            path = root / name
            if not path.is_file() or _sha_file(path) != digest:
                raise TruthError(f"{root}: {name} is missing or does not match case.json")
    exp = truth.get("expected") or {}
    stage = truth.get("stage") or {}
    states = exp.get("bend_states_contract") or {}
    bends = []
    for b in truth.get("bends") or []:
        nom = b.get("nominal") or {}
        act = b.get("actual") or {}
        loci = act.get("loci") or {}
        bends.append(
            TruthBend(
                bend_uid=b["bend_uid"],
                label=b.get("label") or "",
                sequence_index=int(b["sequence_index"]),
                machine_move_index=b.get("machine_move_index"),
                state=states.get(b["bend_uid"]),
                inner_radius_mm=float(nom.get("inner_radius_mm") or 0.0),
                thickness_mm=float(nom.get("thickness_mm") or 0.0),
                nominal_included_deg=float(nom.get("included_deg") or 0.0),
                actual_oml=_line(loci.get("oml_virtual_sharp")),
                actual_outer_apex=_line(loci.get("outer_apex")),
            )
        )
    bends.sort(key=lambda r: r.sequence_index)
    chars = {cid: _characteristic(cid, doc) for cid, doc in (truth.get("characteristics") or {}).items()}
    spec = case.get("spec") or {}
    gen = case.get("generation") or {}
    scan = truth.get("scan") or {}
    raw = scan.get("raw_to_cad")
    raw_to_cad = None if raw is None else np.asarray(raw, dtype=np.float64).reshape(4, 4)
    formed = stage.get("formed_bend_uids")
    return TruthCase(
        case_id=str(case["case_id"]),
        case_dir=root,
        family=str(case.get("family") or truth.get("family")),
        scenario=str(case.get("scenario") or stage.get("scenario")),
        seedset=spec.get("seedset"),
        sku=str(case.get("sku") or truth.get("sku")),
        inspection_stage=str(stage.get("inspection_stage") or case.get("inspection_stage")),
        scan_plan=str(case.get("scan_plan") or gen.get("scan_plan")),
        scan_output=spec.get("output"),
        density_policy=str(case.get("density_policy") or gen.get("density_policy")),
        point_count=int(case.get("point_count") or scan.get("point_count") or 0),
        spacing_mm=_opt_float(case.get("spacing_mm")),
        scanner_model_sha256=gen.get("scanner_model_sha256"),
        scanner_model_provenance=gen.get("scanner_model_provenance"),
        generator_version=case.get("generator_version"),
        mirrored=bool(truth.get("mirrored")),
        wrong_sku=bool(truth.get("wrong_sku")),
        mirror_distinguishable_by_rms=exp.get("mirror_distinguishable_by_rms"),
        symmetric_e2=bool(exp.get("symmetric")),
        ambiguous_groups_e2=tuple(tuple(g) for g in exp.get("ambiguous_bend_groups") or ()),
        formed_bend_uids=None if formed is None else tuple(formed),
        formed_count=stage.get("formed_count"),
        partially_formed_bend_uids=tuple(stage.get("partially_formed_bend_uids") or ()),
        is_sequence_prefix=stage.get("is_sequence_prefix"),
        order_consistent=stage.get("order_consistent"),
        out_of_order_bend_uids=tuple(stage.get("out_of_order_formed_bend_uids") or ()),
        expected_stage_model_id=stage.get("expected_stage_model_id"),
        missing_bend_uids_final=tuple(stage.get("missing_bend_uids_final") or ()),
        extra_bend_count=len(truth.get("extra_bends") or ()),
        injections=tuple(truth.get("injections") or ()),
        bends=tuple(bends),
        characteristics=chars,
        summary_never=frozenset(exp.get("summary_never") or ()),
        never_pass=frozenset(exp.get("characteristics_never_pass") or ()),
        never_fail=frozenset(exp.get("characteristics_never_fail") or ()),
        must_be_unknown=frozenset(exp.get("characteristics_must_be_unknown") or ()),
        ill_defined=frozenset(exp.get("characteristics_ill_defined") or ()),
        raw_to_cad=raw_to_cad,
        files=files,
    )


def load_manifest(tier_dir: str | os.PathLike) -> dict[str, Any]:
    """The tier's ``manifest.json`` (schema checked)."""
    path = Path(tier_dir) / "manifest.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TruthError(f"{path}: unreadable manifest ({exc})") from exc
    if doc.get("schema") != MANIFEST_SCHEMA:
        raise TruthError(f"{path}: not an E2 manifest ({doc.get('schema')!r})")
    return doc


def iter_tier_cases(tier_dir: str | os.PathLike) -> Iterator[Path]:
    """Case directories of a tier in manifest (plan) order."""
    root = Path(tier_dir)
    for entry in load_manifest(root)["cases"]:
        yield root / entry["case_id"]
