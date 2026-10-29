"""``sherman_zeroshot_overlay_v1``: the post-E3 zero-shot overlay of one inspection job (versioned job-level field).

Contract note (``docs/production/runlog/zeroshot_2026-10-25/overlay/CONTRACT_NOTE.md``): the overlay is a
separate document, ``<job>/zeroshot/zeroshot_overlay_v1.json``, served by ``GET /api/line/jobs/{id}/zeroshot-
overlay``, shown by the station and listed by the PDF in its own "detections (unverified)" section. It is not part
of ``ShopfloorInspectionResultV1``: E3's result, its characteristics, its summary and the inspection contract are
untouched (``contracts/production`` is frozen code).

Inputs: E3's validated ``result.json`` (registration ``raw_to_frame``, decoded stage, bend rows), the CompiledPart
and the zs step's lines (raw scan frame, count-capped). Rules (CAD_PATH.md options 1 and 2, pre-registered
association 5 deg / 6 mm / prediction overlap 0.5; the published-line de-duplication uses the frozen matcher
5 deg / 4 mm / 0.6 / 0.6):

- **VALID, option 2 (gap-fill).** A bend that is not NOT_YET_FORMED and whose position E3 does not publish (no
  MEASURED BEND_LINE_POSITION with a measured line) gets the closest zs line that lies within the association
  of its decoded-stage CAD outer-apex locus mapped to raw through E3's own registration and that matches no
  published line: one ``WITHHELD_BEND`` line, ``decision REVIEW``.
- **VALID, option 1 (cross-check).** A zs line that no reference explains -- E3's published lines, every bend's
  decoded-stage locus, every stage model's formed-bend loci, all mapped through E3's registration -- is an
  ``UNEXPECTED_BEND`` line (``decision REVIEW``) and raises the flag ``UNEXPECTED_FEATURE_REVIEW``.
- **Refused (validity not VALID).** Every zs line is shown as ``REFUSED_SCAN_DETECTION``, ``decision
  UNVERIFIED`` ("detected bends, unverified, part not matched"); no CAD association, no flag; the scan stays
  refused.

Verdict: ``effective_summary_state`` is E3's summary state, except that a pass-like state
(ALL_MEASURED_IN_TOLERANCE / STAGE_IN_TOLERANCE) with an ``UNEXPECTED_FEATURE_REVIEW`` flag becomes
REVIEW_REQUIRED (``verdict_lowered``). Nothing else changes: FAIL, REVIEW, INCOMPLETE and INVALID stay, no line
ever carries PASS, and the model refuses any document that would create or upgrade a pass-like state.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

OVERLAY_SCHEMA = "sherman_zeroshot_overlay_v1"
OVERLAY_VERSION = "zs-overlay-1.1.0"
ANGLE_FLAG = "INDEPENDENT_ANGLE_DISAGREES"
#: E3 FAIL angle vs the zs angle on a non-joggle bend (angle_check/ANGLE_CHECK.md: 6 of 6 DEV cases were E3 errors).
ANGLE_DISAGREE_DEG = 5.0
#: Joggle (ANGLE_CHECK.md): two bends sharing a panel, opposite directions, equal sweep <= 60 deg, axes <= 10 t apart.
JOGGLE_MAX_SWEEP_DEG, JOGGLE_SWEEP_EQUAL_DEG, JOGGLE_AXIS_T = 60.0, 1.0, 10.0
SOURCE = "ZS_LANE_V1"
REVIEW_FLAG = "UNEXPECTED_FEATURE_REVIEW"
#: Pre-registered association of a zs line with a CAD locus (CAD_PATH.md option 1/2).
ASSOC_ANGLE_DEG, ASSOC_OFFSET_MM, ASSOC_OVERLAP = 5.0, 6.0, 0.5
#: Frozen matcher (published-line de-duplication).
MATCH_ANGLE_DEG, MATCH_OFFSET_MM, MATCH_OVERLAP = 5.0, 4.0, 0.6
PASS_LIKE_STATES = frozenset({"ALL_MEASURED_IN_TOLERANCE", "STAGE_IN_TOLERANCE"})
REVIEW_STATE = "REVIEW_REQUIRED"
SUMMARY_STATES = ("ALL_MEASURED_IN_TOLERANCE", "STAGE_IN_TOLERANCE", "OUT_OF_TOLERANCE_FOUND", "REVIEW_REQUIRED",
                  "INCOMPLETE", "INVALID_INSPECTION")

Vec3 = tuple[float, float, float]
LineKind = Literal["WITHHELD_BEND", "UNEXPECTED_BEND", "REFUSED_SCAN_DETECTION"]

__all__ = ["OVERLAY_SCHEMA", "OVERLAY_VERSION", "REVIEW_FLAG", "PASS_LIKE_STATES", "ZeroshotOverlayV1",
           "OverlayLineV1", "OverlayFlagV1", "LaneStampV1", "effective_state", "build_overlay", "load_overlay"]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AssocV1(_M):
    axis_angle_deg: float
    offset_mm: float
    overlap: float


class OverlayLineV1(_M):
    line_id: str = Field(min_length=1, max_length=16)
    kind: LineKind
    #: REVIEW (VALID scan: shown for review) or UNVERIFIED (refused scan). Never PASS / FAIL: not a measurement.
    decision: Literal["REVIEW", "UNVERIFIED"]
    source: Literal["ZS_LANE_V1"] = SOURCE
    bend_uid: str | None = None
    start_raw_mm: Vec3
    end_raw_mm: Vec3
    #: The same line in E3's frame (cad_model_mm) through E3's registration; VALID scans only.
    start_frame_mm: Vec3 | None = None
    end_frame_mm: Vec3 | None = None
    length_mm: float = Field(ge=0)
    #: zs included angle; None with ``angle_status NOT_MEASURED_JOGGLE`` (never shown as a number on a joggle).
    incl_deg: float | None = None
    angle_status: Literal["SHOWN", "NOT_MEASURED_JOGGLE"] = "SHOWN"
    score: float = Field(ge=0, le=1)
    assoc: AssocV1 | None = None

    @model_validator(mode="after")
    def _check(self) -> "OverlayLineV1":
        if (self.kind == "REFUSED_SCAN_DETECTION") != (self.decision == "UNVERIFIED"):
            raise ValueError("refused-scan detections are UNVERIFIED; VALID-scan overlay lines are REVIEW")
        if self.kind == "WITHHELD_BEND" and (self.bend_uid is None or self.assoc is None):
            raise ValueError("a WITHHELD_BEND line names its bend and association")
        if self.kind != "WITHHELD_BEND" and self.bend_uid is not None:
            raise ValueError("only a WITHHELD_BEND line names a bend")
        if self.angle_status == "NOT_MEASURED_JOGGLE" and self.incl_deg is not None:
            raise ValueError("a joggle angle is never shown as a number")
        return self


class AngleCheckV1(_M):
    """E3 FAILed an included angle that the independent zs angle disagrees with by more than 5 deg (REVIEW reason)."""

    code: Literal["INDEPENDENT_ANGLE_DISAGREES"] = "INDEPENDENT_ANGLE_DISAGREES"
    bend_uid: str
    e3_angle_deg: float
    zs_angle_deg: float
    delta_deg: float

    @model_validator(mode="after")
    def _check(self) -> "AngleCheckV1":
        if abs(self.delta_deg) <= ANGLE_DISAGREE_DEG:
            raise ValueError("an angle check needs a disagreement above 5 deg")
        return self


class OverlayFlagV1(_M):
    code: Literal["UNEXPECTED_FEATURE_REVIEW"]
    line_ids: tuple[str, ...] = Field(min_length=1)


class LaneStampV1(_M):
    #: The pinned lane that produced the lines: zs_lane_v1 (live default) or zs_lane_v3 (prepared re-pin).
    lane: Literal["zs_lane_v1", "zs_lane_v3"]
    variant: str
    runtime_version: str
    code_fingerprint: str = Field(min_length=64, max_length=64)
    model_sha256: str = Field(min_length=64, max_length=64)
    priors: dict[str, float | int]
    wall_s: float | None = None
    n_candidates: int | None = None
    n_selected: int | None = None
    n_lines: int


class ZeroshotOverlayV1(_M):
    schema_: Literal["sherman_zeroshot_overlay_v1"] = Field(default=OVERLAY_SCHEMA, alias="schema")
    overlay_version: str = OVERLAY_VERSION
    lane: LaneStampV1
    #: Identity: the overlay belongs to exactly this scan and this E3 result.
    scan_sha256: str = Field(min_length=64, max_length=64)
    engine_result_sha256: str = Field(min_length=64, max_length=64)
    engine_version: str
    inspection_validity: str
    e3_summary_state: Literal[SUMMARY_STATES]  # type: ignore[valid-type]
    effective_summary_state: Literal[SUMMARY_STATES]  # type: ignore[valid-type]
    verdict_lowered: bool
    review_reasons: tuple[Literal["UNEXPECTED_FEATURE_REVIEW", "INDEPENDENT_ANGLE_DISAGREES"], ...] = ()
    lines: tuple[OverlayLineV1, ...] = ()
    flags: tuple[OverlayFlagV1, ...] = ()
    #: REVIEW reasons from the independent angle on E3 angle FAILs. They never change the verdict.
    angle_checks: tuple[AngleCheckV1, ...] = ()
    joggle_bend_uids: tuple[str, ...] = ()
    withheld_bends: int = Field(ge=0)
    withheld_bends_located: int = Field(ge=0)
    notes: tuple[str, ...] = ()

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @model_validator(mode="after")
    def _check(self) -> "ZeroshotOverlayV1":
        if self.effective_summary_state != effective_state(self.e3_summary_state, bool(self.flags)):
            raise ValueError("effective_summary_state must be E3's state, lowered to REVIEW_REQUIRED only by a flag")
        if self.verdict_lowered != (self.effective_summary_state != self.e3_summary_state):
            raise ValueError("verdict_lowered must say whether the overlay changed the state")
        if self.effective_summary_state in PASS_LIKE_STATES and self.effective_summary_state != self.e3_summary_state:
            raise ValueError("the overlay never creates or upgrades a pass-like state")
        ids = [ln.line_id for ln in self.lines]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate overlay line ids")
        for f in self.flags:
            if not set(f.line_ids) <= {ln.line_id for ln in self.lines if ln.kind == "UNEXPECTED_BEND"}:
                raise ValueError("a flag references only UNEXPECTED_BEND lines")
        if set(self.review_reasons) != {f.code for f in self.flags} | {a.code for a in self.angle_checks}:
            raise ValueError("review_reasons are the flag and angle-check codes")
        if self.angle_checks and self.inspection_validity != "VALID":
            raise ValueError("angle checks only on a VALID inspection")
        if any(a.bend_uid in self.joggle_bend_uids for a in self.angle_checks):
            raise ValueError("no angle check on a joggle bend")
        refused = self.inspection_validity != "VALID"
        if refused and (self.flags or any(ln.kind != "REFUSED_SCAN_DETECTION" for ln in self.lines)):
            raise ValueError("a refused scan carries only unverified detections and no flag")
        if not refused and any(ln.kind == "REFUSED_SCAN_DETECTION" for ln in self.lines):
            raise ValueError("unverified detections only on a refused scan")
        if self.withheld_bends_located != sum(1 for ln in self.lines if ln.kind == "WITHHELD_BEND"):
            raise ValueError("withheld_bends_located counts the WITHHELD_BEND lines")
        if self.withheld_bends_located > self.withheld_bends:
            raise ValueError("more located than withheld bends")
        return self


def effective_state(e3_state: str, flagged: bool) -> str:
    """The only verdict change the overlay may make: a pass-like state with a flag -> REVIEW_REQUIRED."""
    return REVIEW_STATE if (flagged and e3_state in PASS_LIKE_STATES) else e3_state


# --------------------------------------------------------------------------- merge


def _metrics(ref: Any, pred: Any) -> dict[str, float]:
    import numpy as np

    from domains.production.engine.eval.line_metric import pair_metrics

    return pair_metrics(np.asarray(ref, float), np.asarray(pred, float))


def _assoc(ref: Any, z: Any) -> tuple[bool, dict[str, float]]:
    m = _metrics(ref, z)
    ok = (m["axis_angle_deg"] <= ASSOC_ANGLE_DEG and m["transverse_offset_mm"] <= ASSOC_OFFSET_MM
          and m["prediction_span_overlap"] >= ASSOC_OVERLAP)
    return ok, m


def _frozen_match(ref: Any, z: Any) -> bool:
    m = _metrics(ref, z)
    return (m["axis_angle_deg"] <= MATCH_ANGLE_DEG and m["transverse_offset_mm"] <= MATCH_OFFSET_MM
            and m["gold_span_overlap"] >= MATCH_OVERLAP and m["prediction_span_overlap"] >= MATCH_OVERLAP)


def joggle_bends(compiled: Any) -> set[str]:
    """Bends of a joggle pair (module constants; ANGLE_CHECK.md definition), from the CompiledPart only."""
    import numpy as np

    out: set[str] = set()
    bends = [b for b in compiled.bends if b.supported]
    t = float(compiled.thickness.t_mm)
    for a, b in ((a, b) for i, a in enumerate(bends) for b in bends[i + 1:]):
        if not ({a.parent_panel_id, a.child_panel_id} & {b.parent_panel_id, b.child_panel_id}):
            continue
        if a.direction is None or b.direction is None or a.direction == b.direction:
            continue
        if max(a.sweep_deg, b.sweep_deg) > JOGGLE_MAX_SWEEP_DEG or abs(a.sweep_deg - b.sweep_deg) > JOGGLE_SWEEP_EQUAL_DEG:
            continue
        u = np.asarray(a.axis.direction, float)
        u /= np.linalg.norm(u)
        d = np.asarray(b.axis.point_mm, float) - np.asarray(a.axis.point_mm, float)
        if np.linalg.norm(d - (d @ u) * u) <= JOGGLE_AXIS_T * t:
            out |= {a.bend_uid, b.bend_uid}
    return out


def _v3(p: Any) -> Vec3:
    return (round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4))


def _length(a: Any, b: Any) -> float:
    return round(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5, 3)


def build_overlay(result: Any, compiled: Any, zs: Mapping[str, Any], *, engine_result_sha256: str) -> ZeroshotOverlayV1:
    """Merge one job's overlay (module docstring). ``result`` is E3's ``ShopfloorInspectionResultV1``,
    ``compiled`` its ``CompiledPartV1``, ``zs`` the zs step's ``zs_lines.json`` with ``status OK``.

    Pure: reads nothing, writes nothing, never touches ``result``.
    """
    import numpy as np

    from domains.production.engine.eval.engine_output import (
        apply4,
        decoded_stage,
        frame_to_raw,
        measured_outer_apex,
        row_characteristic,
        stage_locus,
    )

    if zs.get("status") != "OK":
        raise ValueError(f"zs lines status is {zs.get('status')!r}, not OK")
    if zs.get("scan_sha256") != result.scan.sha256:
        raise ValueError("zs lines were computed on another scan")
    zl = list(zs.get("lines") or [])
    stamp = LaneStampV1(
        lane=zs["lane"], variant=str(zs.get("variant") or "zs"), runtime_version=str(zs["runtime_version"]),
        code_fingerprint=zs["code_fingerprint"], model_sha256=zs["model_sha256"],
        priors={k: v for k, v in (zs.get("priors") or {}).items() if isinstance(v, (int, float))},
        wall_s=zs.get("wall_s"), n_candidates=zs.get("n_candidates"), n_selected=zs.get("n_selected"),
        n_lines=len(zl),
    )
    validity = result.inspection_validity.state
    e3_state = result.summary.state
    joggles = joggle_bends(compiled)

    def angle(z: Mapping[str, Any], bend_uid: str | None) -> dict[str, Any]:
        # never a number on a joggle; a line with no known bend on a part with joggles may be one
        if (bend_uid in joggles) or (bend_uid is None and joggles):
            return {"incl_deg": None, "angle_status": "NOT_MEASURED_JOGGLE"}
        return {"incl_deg": z.get("incl_deg"), "angle_status": "SHOWN"}
    angle_checks: list[AngleCheckV1] = []
    seg = [np.asarray([z["start_mm"], z["end_mm"]], float) for z in zl]
    lines: list[OverlayLineV1] = []
    flags: list[OverlayFlagV1] = []
    notes: list[str] = []
    withheld = 0

    reg = result.inspection_validity.registration
    if validity != "VALID":
        for i, z in enumerate(zl):
            lines.append(OverlayLineV1(
                line_id=f"Z{i + 1:02d}", kind="REFUSED_SCAN_DETECTION", decision="UNVERIFIED",
                start_raw_mm=_v3(z["start_mm"]), end_raw_mm=_v3(z["end_mm"]),
                length_mm=_length(z["start_mm"], z["end_mm"]), **angle(z, None), score=float(z["score"])))
    elif reg is None or reg.raw_to_frame is None:
        notes.append("NO_REGISTRATION_TRANSFORM: E3 published no raw_to_frame; no CAD association possible")
    else:
        to_frame = np.asarray(reg.raw_to_frame, float).reshape(4, 4)
        unit = float(reg.unit_factor)
        stage = decoded_stage(result, compiled)
        published: list[np.ndarray] = []
        published_by: dict[str, np.ndarray] = {}
        nominal: dict[str, np.ndarray] = {}
        rows = list(result.bends)
        for row in rows:
            nominal[row.bend_uid] = frame_to_raw(result, stage_locus(compiled, stage, row.bend_uid, "outer_apex"))
            pos = row_characteristic(row, "BEND_LINE_POSITION")
            if (pos is not None and pos.state == "MEASURED" and row.display.measured_polyline_mm is not None
                    and row.state != "NOT_YET_FORMED"):
                apex, _src = measured_outer_apex(compiled, stage, row, fallback="NOMINAL")
                if apex is not None:
                    published.append(frame_to_raw(result, apex))
                    published_by[row.bend_uid] = published[-1]
        used: set[int] = set()
        n = 0
        # option 2: gap-fill each withheld bend with its closest associated zs line
        for row in rows:
            if row.state == "NOT_YET_FORMED":
                continue
            pos = row_characteristic(row, "BEND_LINE_POSITION")
            is_published = (pos is not None and pos.state == "MEASURED" and row.display.measured_polyline_mm is not None)
            if is_published:
                continue
            withheld += 1
            best: tuple[int, dict[str, float]] | None = None
            for i, s in enumerate(seg):
                if i in used or any(_frozen_match(p, s) for p in published):
                    continue
                ok, m = _assoc(nominal[row.bend_uid], s)
                if ok and (best is None or m["transverse_offset_mm"] < best[1]["transverse_offset_mm"]):
                    best = (i, m)
            if best is None:
                continue
            i, m = best
            used.add(i)
            n += 1
            z = zl[i]
            fr = apply4(to_frame, seg[i] * unit)
            lines.append(OverlayLineV1(
                line_id=f"Z{n:02d}", kind="WITHHELD_BEND", decision="REVIEW", bend_uid=row.bend_uid,
                start_raw_mm=_v3(z["start_mm"]), end_raw_mm=_v3(z["end_mm"]),
                start_frame_mm=_v3(fr[0]), end_frame_mm=_v3(fr[1]),
                length_mm=_length(z["start_mm"], z["end_mm"]), **angle(z, row.bend_uid), score=float(z["score"]),
                assoc=AssocV1(axis_angle_deg=round(m["axis_angle_deg"], 3), offset_mm=round(m["transverse_offset_mm"], 3),
                              overlap=round(m["prediction_span_overlap"], 3))))
        # option 1: zs lines the recipe does not explain -> REVIEW flag
        refs: list[np.ndarray] = list(nominal.values()) + published
        for st in compiled.stages:
            for uid in st.formed_bend_uids:
                refs.append(frame_to_raw(result, stage_locus(compiled, st, uid, "outer_apex")))
        unexplained: list[str] = []
        for i, s in enumerate(seg):
            if i in used or any(_assoc(r, s)[0] for r in refs):
                continue
            n += 1
            z = zl[i]
            fr = apply4(to_frame, s * unit)
            lid = f"Z{n:02d}"
            unexplained.append(lid)
            lines.append(OverlayLineV1(
                line_id=lid, kind="UNEXPECTED_BEND", decision="REVIEW",
                start_raw_mm=_v3(z["start_mm"]), end_raw_mm=_v3(z["end_mm"]),
                start_frame_mm=_v3(fr[0]), end_frame_mm=_v3(fr[1]),
                length_mm=_length(z["start_mm"], z["end_mm"]), **angle(z, None), score=float(z["score"])))
        if unexplained:
            flags.append(OverlayFlagV1(code=REVIEW_FLAG, line_ids=tuple(unexplained)))
        # independent angle on E3 angle FAILs (non-joggle bends): REVIEW reason only, the FAIL stays
        for row in rows:
            ang = row_characteristic(row, "INCLUDED_ANGLE")
            if (row.bend_uid in joggles or row.bend_uid not in published_by or ang is None or ang.state != "MEASURED"
                    or ang.decision != "FAIL" or ang.actual is None):
                continue
            best_m: tuple[int, float] | None = None
            for i, s in enumerate(seg):
                ok, m = _assoc(published_by[row.bend_uid], s)
                if ok and zl[i].get("incl_deg") is not None and (best_m is None or m["transverse_offset_mm"] < best_m[1]):
                    best_m = (i, m["transverse_offset_mm"])
            if best_m is None:
                continue
            zs_angle = float(zl[best_m[0]]["incl_deg"])
            delta = zs_angle - float(ang.actual)
            if abs(delta) > ANGLE_DISAGREE_DEG:
                angle_checks.append(AngleCheckV1(bend_uid=row.bend_uid, e3_angle_deg=round(float(ang.actual), 3),
                                                 zs_angle_deg=round(zs_angle, 3), delta_deg=round(delta, 3)))
    eff = effective_state(e3_state, bool(flags))
    return ZeroshotOverlayV1(
        lane=stamp, scan_sha256=result.scan.sha256, engine_result_sha256=engine_result_sha256,
        engine_version=result.engine.engine_version, inspection_validity=validity, e3_summary_state=e3_state,
        effective_summary_state=eff, verdict_lowered=eff != e3_state,
        review_reasons=tuple(sorted({f.code for f in flags} | {a.code for a in angle_checks})), lines=tuple(lines),
        flags=tuple(flags), angle_checks=tuple(angle_checks), joggle_bend_uids=tuple(sorted(joggles)),
        withheld_bends=withheld, withheld_bends_located=sum(1 for ln in lines if ln.kind == "WITHHELD_BEND"),
        notes=tuple(notes),
    )


def overlay_json_bytes(overlay: ZeroshotOverlayV1) -> bytes:
    import json

    return (json.dumps(overlay.model_dump(mode="json", by_alias=True), sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def load_overlay(path: str | Path, *, scan_sha256: str | None = None, e3_summary_state: str | None = None,
                 engine_result_sha256: str | None = None) -> ZeroshotOverlayV1 | None:
    """Validated overlay at ``path``, or None when absent, invalid or not this job's (display only: never raises)."""
    p = Path(path)
    try:
        if not p.is_file() or p.is_symlink():
            return None
        ov = ZeroshotOverlayV1.model_validate_json(p.read_bytes())
    except Exception:  # noqa: BLE001
        return None
    if scan_sha256 is not None and ov.scan_sha256 != scan_sha256:
        return None
    if e3_summary_state is not None and ov.e3_summary_state != e3_summary_state:
        return None
    if engine_result_sha256 is not None and ov.engine_result_sha256 != engine_result_sha256:
        return None
    return ov


def identity_problems(ov: ZeroshotOverlayV1, *, scan_sha256: str | None, e3_summary_state: str | None,
                      engine_result_sha256: str | None = None) -> list[str]:
    out: list[str] = []
    if scan_sha256 is not None and ov.scan_sha256 != scan_sha256:
        out.append("OVERLAY_SCAN_MISMATCH")
    if e3_summary_state is not None and ov.e3_summary_state != e3_summary_state:
        out.append("OVERLAY_SUMMARY_MISMATCH")
    if engine_result_sha256 is not None and ov.engine_result_sha256 != engine_result_sha256:
        out.append("OVERLAY_ENGINE_RESULT_MISMATCH")
    return out

