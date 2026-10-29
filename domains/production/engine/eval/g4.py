"""G4 visual verification: review packets, secret canaries, verdicts and the gate.

Protocol (plan §5 G4, §10 amendment 8; numbers in :data:`.thresholds.G4_THRESHOLDS`):

1. :func:`build_review_packet` renders, for each inspected scan, the global
   views (no metric values) and one close-up per bend (:mod:`.render`). At
   least 10 % of the close-ups are **canaries**: their measured line is moved
   across the bend by more than the bend's LANDS tolerance (LINE_SHIFT: ``|s|``
   drawn from ``[max(8, max(2, Ri + t) + 3), 15]`` mm; a bend whose range is
   empty gets a LABEL_SWAP instead), or the labels of two bends of one scan are
   exchanged everywhere in the SCAN panel (LABEL_SWAP; only the truthful RECIPE
   CAD panel reveals it). Canaries are rendered exactly like the others and get
   opaque file names; ambiguous-pose close-ups are never canaries. Reviewer
   images carry only opaque titles (never a case id or a jobs.json title).
2. The **reviewer packet** (``packet_dir``) holds the images and
   ``review_manifest.json``: per image its opaque id, sha256, kind, scan
   reference (opaque), the subject *as displayed* and the questions. It holds
   no case id, no metric, no decision and no canary information, only the sha256
   commitment of the sealed key.
3. The **sealed key** (``sealed_dir/canary_key.json``, written read-only and
   kept outside the packet) maps every image to its true subject, scan/case,
   arm (``source``), population (SYNTHETIC / PHYSICAL_DEV), the close-up class,
   the bend's LANDS tolerance, its E1 ambiguity membership and the canary
   corruption with its margin. Reviewers must never be given ``sealed_dir``.
4. Reviewers -- two fresh agents at max effort who did not write the engine
   and see no metrics, plus a third for ties -- append :class:`G4VerdictV1`
   records to ``g4_verdicts.jsonl`` (image sha256, bend_uid, verdict, reason).
5. :func:`evaluate_g4` verifies the key against its commitment, adjudicates
   (two primaries agree, else the tie-break), checks that **every canary was
   flagged by every primary reviewer** (otherwise the review does not count),
   and computes the gate on the non-canary close-ups of the gated arm (E3):
   100 % of MEASURED bends LAND, 0 mis-tagged, and every close-up without a
   measured line visually justified. LEGACY (the visual baseline) and ORACLE
   (harness self-test) images are reported, never gated. The synthetic part of
   the review needs at least 20 E3 scans including an incomplete one.

Close-up classes (:func:`closeup_class`): MEASURED (LANDS + TAG_OK);
NOT_YET_FORMED / UNKNOWN / FORMED_NO_LINE (the reviewer judges whether the
missing line is justified: flat, occluded or missing); AMBIGUOUS_* (an
AMBIGUOUS_POSE bend: amendment 8 justifies AMBIGUOUS only by E1's symmetry
test, so it is justified iff the bend is in an ``ambiguous_bend_groups`` orbit
of the decoded stage model -- without a decoded stage, of the stage model whose
formed set is the bends not reported NOT_YET_FORMED, and only when none matches
of any stage model -- no reviewer question, and an AMBIGUOUS outside every
orbit fails the gate); REGISTRATION_RULE (a formed bend whose line was
withheld as REGISTRATION_UNTRUSTED, amendment 9: not visually checkable,
counted and reported, TAG_OK still judged); UNSUPPORTED_RULE (a bend E1's
per-feature UNSUPPORTED rule excludes, amendment 4d: reported, TAG_OK judged);
INVALID_INSPECTION (the engine declared the whole scan invalid: no bend can be
reviewed; reported, never gated here -- G1 / G2(e) judge invalid outcomes).

LANDS: the solid line lies within max(2 mm, Ri + t) of the crease over at least
80 % of its length and is parallel to it. TAG_OK: its label matches the recipe
CAD render's label at that spot.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.production.common import UtcIsoTimestamp

from .plyio import sha256_file
from .render import (
    BendOverlay,
    Corruption,
    RenderScene,
    plan_closeups,
    render_closeup,
    render_global_views,
    scene_from_engine_output,
)
from .thresholds import G4_THRESHOLDS, thresholds_sha256

__all__ = [
    "PACKET_SCHEMA",
    "KEY_SCHEMA",
    "VERDICT_SCHEMA",
    "QUESTIONS",
    "CLOSEUP_CLASSES",
    "RenderJob",
    "G4VerdictV1",
    "closeup_class",
    "build_review_packet",
    "append_verdict",
    "load_verdicts",
    "evaluate_g4",
]

PACKET_SCHEMA = "sherman_g4_review_packet_v1"
KEY_SCHEMA = "sherman_g4_canary_key_v2"
VERDICT_SCHEMA = "sherman_g4_verdict_v1"

_TAG_QUESTION = "TAG_OK: does the bold subject label match the RECIPE CAD label at the same spot? YES / NO / UNDETERMINABLE."
QUESTIONS = {
    "CLOSEUP_MEASURED": [
        "LANDS: does the SOLID black line lie on the physical bend crease of the scan (within max(2 mm, Ri + t), "
        "parallel, over at least 80 % of its length)? Answer YES, NO, or OCCLUDED if the crease cannot be seen.",
        "TAG_OK: does the bold label next to the solid line equal the RECIPE CAD label at the same spot? YES / NO / "
        "UNDETERMINABLE.",
    ],
    "CLOSEUP_NOT_YET_FORMED": [
        "The engine reports this bend NOT YET FORMED (flat), so no measured line is drawn; the grey dashed line is "
        "where the bend will be. Is that justified: is the material along the grey dashed line really flat (unbent) "
        "in the scan? Answer unknown_justified YES or NO; lands = NOT_APPLICABLE.",
        _TAG_QUESTION,
    ],
    "CLOSEUP_UNKNOWN": [
        "The engine reports this bend UNKNOWN (reason in the header), so no measured line is drawn. Is that "
        "justified: is the bend really occluded or missing in the scan near the grey dashed line, so that its state "
        "cannot be judged? Answer unknown_justified YES or NO; lands = NOT_APPLICABLE.",
        _TAG_QUESTION,
    ],
    "CLOSEUP_FORMED_NO_LINE": [
        "The engine reports this bend formed but did not measure its line position (reason in the header). Is that "
        "justified: is the crease really occluded or missing in the scan near the grey dashed line? Answer "
        "unknown_justified YES or NO; lands = NOT_APPLICABLE.",
        _TAG_QUESTION,
    ],
    "CLOSEUP_AMBIGUOUS": [
        "Context only: the engine reports an ambiguous pose for this bend; whether that is justified is decided by "
        "the compile-time symmetry test, not from images. Record lands = NOT_APPLICABLE, tag_ok = UNDETERMINABLE "
        "and no unknown_justified.",
    ],
    "CLOSEUP_REGISTRATION_RULE": [
        "The engine withheld this bend's line because its in-plane registration is degenerate (header: "
        "REGISTRATION_UNTRUSTED); a picture cannot confirm that. Record lands = NOT_APPLICABLE and no "
        "unknown_justified.",
        _TAG_QUESTION,
    ],
    "CLOSEUP_UNSUPPORTED_RULE": [
        "The recipe compile marks this bend an unsupported feature (header: UNSUPPORTED_FEATURE), so no line is "
        "measured; a picture cannot change that rule. Record lands = NOT_APPLICABLE and no unknown_justified.",
        _TAG_QUESTION,
    ],
    "CLOSEUP_INVALID_INSPECTION": [
        "Context only: the engine declared this whole inspection invalid (header), so no bend is measured. Record "
        "lands = NOT_APPLICABLE, tag_ok = UNDETERMINABLE and no unknown_justified.",
    ],
    "GLOBAL": [
        "Context only: overview of the registered scan with every bend's line and label. Record a verdict only if "
        "something contradicts the close-ups (lands = NOT_APPLICABLE, tag_ok = UNDETERMINABLE).",
    ],
}
#: class -> question set
_QUESTION_SET = {
    "MEASURED": "CLOSEUP_MEASURED",
    "NOT_YET_FORMED": "CLOSEUP_NOT_YET_FORMED",
    "UNKNOWN": "CLOSEUP_UNKNOWN",
    "FORMED_NO_LINE": "CLOSEUP_FORMED_NO_LINE",
    "AMBIGUOUS_E1_JUSTIFIED": "CLOSEUP_AMBIGUOUS",
    "AMBIGUOUS_NOT_E1": "CLOSEUP_AMBIGUOUS",
    "REGISTRATION_RULE": "CLOSEUP_REGISTRATION_RULE",
    "UNSUPPORTED_RULE": "CLOSEUP_UNSUPPORTED_RULE",
    "INVALID_INSPECTION": "CLOSEUP_INVALID_INSPECTION",
}
CLOSEUP_CLASSES = frozenset(_QUESTION_SET)
#: Classes whose missing line the reviewers must justify (gated).
_REVIEWER_JUSTIFIED = ("NOT_YET_FORMED", "UNKNOWN", "FORMED_NO_LINE")
#: Classes whose label is judged (TAG_OK counted).
_TAGGED = ("MEASURED", "NOT_YET_FORMED", "UNKNOWN", "FORMED_NO_LINE", "REGISTRATION_RULE", "UNSUPPORTED_RULE")
_NOT_EVALUABLE_REASONS = {"REVIEW_INVALID", "VERDICTS_INCOMPLETE", "NO_MEASURED_BENDS", "NO_GATED_SCANS",
                          "SYNTHETIC_SAMPLE_INSUFFICIENT"}

Sha = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, doc: Any, *, read_only: bool = False) -> str:
    data = (json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    if read_only:
        os.chmod(path, 0o400)
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class RenderJob:
    """One inspected scan to review.

    E3 outputs are loaded from ``engine_dir``; another arm (the wave-3 legacy
    control arm) supplies a prebuilt neutral :class:`~.render.RenderScene` in
    ``scene`` instead, so it is rendered exactly the same way (amendment 8)."""

    engine_dir: Path | None
    compiled_json: Path | None
    recipe_json: Path | None
    #: Sealed identity (never shown to reviewers), e.g. the E2 case id or the dev scan name.
    case_ref: str
    #: Which arm produced the result: E3 (engine, gated), LEGACY (control arm, the visual
    #: baseline) or ORACLE (harness self-test). Only E3 is gated.
    source: str = "E3"
    #: Ignored in reviewer packets (images carry opaque titles only, so no case id can leak).
    title: str = ""
    scene: RenderScene | None = None
    #: SYNTHETIC (E2 case) or PHYSICAL_DEV (authorized development scan).
    population: str = "SYNTHETIC"
    #: The scanned part is incomplete (staged / partially formed); None when unknown.
    incomplete: bool | None = None

    def __post_init__(self) -> None:
        if self.source not in ("E3", "LEGACY", "ORACLE"):
            raise ValueError(f"unknown source {self.source!r}")
        if self.population not in ("SYNTHETIC", "PHYSICAL_DEV"):
            raise ValueError(f"unknown population {self.population!r}")


class G4VerdictV1(BaseModel):
    """One reviewer's verdict on one image (a line of ``g4_verdicts.jsonl``)."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["sherman_g4_verdict_v1"] = Field(default=VERDICT_SCHEMA, alias="schema")
    image_id: str = Field(min_length=1, max_length=64)
    image_sha256: Sha
    #: The subject bend_uid as listed in the review manifest (None for global images).
    bend_uid: str | None = Field(default=None, pattern=r"^bu_[0-9a-f]{20}$")
    reviewer_id: str = Field(min_length=1, max_length=64)
    role: Literal["PRIMARY", "TIEBREAK"]
    lands: Literal["YES", "NO", "OCCLUDED", "NOT_APPLICABLE"]
    tag_ok: Literal["YES", "NO", "UNDETERMINABLE"]
    unknown_justified: Literal["YES", "NO"] | None = None
    #: Derived summary: OK | NOT_LANDS | MISTAG | NOT_LANDS_MISTAG | OCCLUDED[_MISTAG] |
    #: UNKNOWN_JUSTIFIED[_MISTAG] | UNKNOWN_NOT_JUSTIFIED[_MISTAG] | CONTEXT[_MISTAG]
    verdict: str
    reason: str = Field(min_length=1, max_length=2000)
    recorded_at: UtcIsoTimestamp

    @staticmethod
    def derive(lands: str, tag_ok: str, unknown_justified: str | None) -> str:
        mistag = "_MISTAG" if tag_ok == "NO" else ""
        if lands == "NOT_APPLICABLE":
            if unknown_justified is None:
                return f"CONTEXT{mistag}"
            base = "UNKNOWN_JUSTIFIED" if unknown_justified == "YES" else "UNKNOWN_NOT_JUSTIFIED"
            return f"{base}{mistag}"
        if lands == "OCCLUDED":
            return f"OCCLUDED{mistag}"
        bad_l, bad_t = lands == "NO", tag_ok == "NO"
        return {(False, False): "OK", (True, False): "NOT_LANDS", (False, True): "MISTAG",
                (True, True): "NOT_LANDS_MISTAG"}[(bad_l, bad_t)]

    @model_validator(mode="after")
    def _consistent(self) -> "G4VerdictV1":
        if self.unknown_justified is not None and self.lands != "NOT_APPLICABLE":
            raise ValueError("unknown_justified is answered only when no measured line is shown")
        if self.verdict != self.derive(self.lands, self.tag_ok, self.unknown_justified):
            raise ValueError(f"verdict must be {self.derive(self.lands, self.tag_ok, self.unknown_justified)!r}")
        return self

    @classmethod
    def make(cls, **fields: Any) -> "G4VerdictV1":
        fields.setdefault("recorded_at", _now())
        fields["verdict"] = cls.derive(fields["lands"], fields["tag_ok"], fields.get("unknown_justified"))
        return cls(**fields)


def append_verdict(path: str | os.PathLike, verdict: G4VerdictV1) -> None:
    """Append one validated verdict line (JSONL)."""
    line = json.dumps(verdict.model_dump(mode="json", by_alias=True), sort_keys=True, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_verdicts(path: str | os.PathLike) -> tuple[list[G4VerdictV1], list[str]]:
    """All valid verdicts and the problems of the invalid lines (never silently dropped)."""
    out: list[G4VerdictV1] = []
    problems: list[str] = []
    p = Path(path)
    if not p.is_file():
        return out, [f"{p}: no verdict file"]
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(G4VerdictV1.model_validate(json.loads(line)))
        except (ValueError, TypeError) as exc:
            problems.append(f"line {i}: {str(exc).splitlines()[0][:200]}")
    return out, problems


# ---------------------------------------------------------------------------
# packet construction
# ---------------------------------------------------------------------------


def closeup_class(scene: RenderScene, b: BendOverlay) -> str:
    """The close-up class of a bend (module docstring)."""
    if scene.validity_state != "VALID":
        return "INVALID_INSPECTION"
    if b.measured_apex is not None:
        return "MEASURED"
    if "UNSUPPORTED_FEATURE" in b.state_reasons or b.position_reason == "UNSUPPORTED_FEATURE":
        return "UNSUPPORTED_RULE"
    if "AMBIGUOUS_POSE" in b.state_reasons or b.position_reason == "AMBIGUOUS_POSE":
        return "AMBIGUOUS_E1_JUSTIFIED" if scene.in_ambiguous_group(b.bend_uid) else "AMBIGUOUS_NOT_E1"
    if b.state == "NOT_YET_FORMED":
        return "NOT_YET_FORMED"
    if b.state == "UNKNOWN":
        return "UNKNOWN"
    if b.position_reason == "REGISTRATION_UNTRUSTED":
        return "REGISTRATION_RULE"
    return "FORMED_NO_LINE"


def _opaque(salt: str, *parts: Any) -> str:
    return "img_" + hashlib.sha256("|".join([salt, *map(str, parts)]).encode()).hexdigest()[:12]


def _shift_range(b: BendOverlay, cfg: Mapping[str, Any]) -> tuple[float, float] | None:
    lo = max(float(cfg["shift_mm_min"]), b.lands_tolerance_mm + float(cfg["lands_margin_mm"]))
    hi = float(cfg["shift_mm_max"])
    return (lo, hi) if lo <= hi else None


def build_review_packet(
    jobs: Sequence[RenderJob],
    packet_dir: str | os.PathLike,
    sealed_dir: str | os.PathLike,
    *,
    canary_fraction: float | None = None,
    seed: int | None = None,
    px_per_mm: float | None = None,
    include_globals: bool = True,
) -> dict[str, Any]:
    """Render the packet and the sealed key (module docstring). ``seed`` makes the
    canaries reproducible for **tests only**: the key records ``salt_source =
    TEST_SEED`` and :func:`evaluate_g4` refuses such a packet unless told it is a
    test. The default draws a secret salt from the OS."""
    packet = Path(packet_dir)
    sealed = Path(sealed_dir)
    if packet.resolve() == sealed.resolve() or sealed.resolve().is_relative_to(packet.resolve()):
        raise ValueError("the sealed key must live outside the reviewer packet")
    cfg = G4_THRESHOLDS["canaries"]
    frac = float(canary_fraction if canary_fraction is not None else cfg["min_fraction"])
    if frac < cfg["min_fraction"] - 1e-12:
        raise ValueError(f"canary fraction must be >= {cfg['min_fraction']}")
    salt = secrets.token_hex(16) if seed is None else hashlib.sha256(f"seed|{seed}".encode()).hexdigest()[:32]
    rng = np.random.default_rng(int(salt[:15], 16))
    images_dir = packet / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    staging = sealed / "staging"
    staging.mkdir(parents=True, exist_ok=True)

    scenes: list[tuple[str, RenderJob, RenderScene]] = []
    plans: list[tuple[int, Any]] = []
    for j, job in enumerate(jobs):
        scan_ref = "scan_" + hashlib.sha256(f"{salt}|scan|{job.case_ref}|{j}".encode()).hexdigest()[:10]
        title = f"review scan {scan_ref}"  # opaque: a jobs.json title is never drawn
        if job.scene is not None:
            scene = job.scene
            scene.title = title
        else:
            if job.engine_dir is None or job.compiled_json is None or job.recipe_json is None:
                raise ValueError(f"{job.case_ref}: a job needs an E3 output directory or a prebuilt scene")
            scene = scene_from_engine_output(job.engine_dir, job.compiled_json, job.recipe_json, title=title)
        scenes.append((scan_ref, job, scene))
        for plan in plan_closeups(scene, px_per_mm=px_per_mm):
            plans.append((j, plan))
    classes = [closeup_class(scenes[j][2], scenes[j][2].bend(plan.bend_uid)) for j, plan in plans]

    # -- choose canaries ---------------------------------------------------------------------
    # LINE_SHIFT needs a measured line and a shift range beyond the bend's LANDS tolerance;
    # LABEL_SWAP needs two untiled, label-judged close-ups of one scan (the RECIPE CAD panel
    # still shows the true label at that spot). Ambiguous-pose close-ups are never canaries.
    n_canary = max(1, math.ceil(frac * len(plans))) if plans else 0
    corruption: dict[int, Corruption] = {}
    swapped_with: dict[int, str] = {}

    def swappable(k: int) -> bool:
        return plans[k][1].tile_count == 1 and classes[k] in _TAGGED

    for i in (int(x) for x in rng.permutation(len(plans))):
        if len(corruption) >= n_canary:
            break
        if i in corruption:
            continue
        j, plan = plans[i]
        scene = scenes[j][2]
        b = scene.bend(plan.bend_uid)
        shift_range = _shift_range(b, cfg) if classes[i] == "MEASURED" else None
        partners = [k for k in range(len(plans)) if k not in corruption and k != i and plans[k][0] == j
                    and plans[k][1].bend_uid != plan.bend_uid and swappable(k)] if swappable(i) else []
        if shift_range is None and not partners:
            continue  # neither corruption is detectable here
        want_swap = bool(partners) and (shift_range is None or rng.random() < 0.5)
        if want_swap:
            k = partners[int(rng.integers(len(partners)))]
            a, other = b, scene.bend(plans[k][1].bend_uid)
            labels = {a.bend_uid: other.text, other.bend_uid: a.text}
            corruption[i] = Corruption("LABEL_SWAP", label_map=labels)
            corruption[k] = Corruption("LABEL_SWAP", label_map=labels)
            swapped_with[i], swapped_with[k] = other.bend_uid, a.bend_uid
        else:
            lo, hi = shift_range  # type: ignore[misc]
            mag = float(rng.uniform(lo, hi))
            corruption[i] = Corruption("LINE_SHIFT", shift_mm=mag if rng.random() < 0.5 else -mag)
    if len(corruption) < n_canary:
        raise ValueError(f"could place only {len(corruption)} of the {n_canary} required canaries")

    # -- render ---------------------------------------------------------------------------
    entries: list[dict[str, Any]] = []
    key_images: list[dict[str, Any]] = []
    for j, (scan_ref, job, scene) in enumerate(scenes):
        if include_globals:
            for rec in render_global_views(scene, staging / scan_ref, stem="global", show_values=False):
                image_id = _opaque(salt, scan_ref, rec["kind"])
                dst = images_dir / f"{image_id}.png"
                os.replace(staging / scan_ref / rec["file"], dst)
                entries.append({"image_id": image_id, "file": f"images/{dst.name}", "sha256": sha256_file(dst),
                                "kind": rec["kind"], "scan_ref": scan_ref, "subject": None,
                                "questions": QUESTIONS["GLOBAL"]})
                key_images.append({"image_id": image_id, "sha256": entries[-1]["sha256"], "kind": rec["kind"],
                                   "scan_ref": scan_ref, "case_ref": job.case_ref, "source": job.source,
                                   "population": job.population})
    for i, (j, plan) in enumerate(plans):
        scan_ref, job, scene = scenes[j]
        b = scene.bend(plan.bend_uid)
        image_id = _opaque(salt, scan_ref, plan.bend_uid, plan.tile_index)
        dst = images_dir / f"{image_id}.png"
        rec = render_closeup(scene, plan, staging / f"{image_id}.png", corruption=corruption.get(i))
        os.replace(staging / f"{image_id}.png", dst)
        sha = sha256_file(dst)
        shown_uid = swapped_with.get(i, b.bend_uid)
        shown = scene.bend(shown_uid)
        cls = classes[i]
        entries.append({
            "image_id": image_id, "file": f"images/{dst.name}", "sha256": sha, "kind": "CLOSEUP", "scan_ref": scan_ref,
            "subject": {"bend_uid": shown.bend_uid, "label": shown.label, "sequence_index": shown.sequence_index,
                        "display": rec["label_shown"], "state_shown": b.state,
                        "measured_line_shown": b.measured_apex is not None},
            "tile": {"index": plan.tile_index, "count": plan.tile_count}, "px_per_mm": plan.px_per_mm,
            "questions": QUESTIONS[_QUESTION_SET[cls]],
        })
        c = corruption.get(i)
        tol = b.lands_tolerance_mm
        key_images.append({
            "image_id": image_id, "sha256": sha, "kind": "CLOSEUP", "scan_ref": scan_ref, "case_ref": job.case_ref,
            "source": job.source, "population": job.population, "true_subject_uid": b.bend_uid,
            "displayed_subject_uid": shown.bend_uid, "state": b.state, "state_reasons": list(b.state_reasons),
            "position_reason": b.position_reason, "class": cls, "measured_line_shown": b.measured_apex is not None,
            "apex_source": b.apex_source, "lands_tolerance_mm": round(tol, 4),
            "in_e1_ambiguous_group": scene.in_ambiguous_group(b.bend_uid),
            "tile": {"index": plan.tile_index, "count": plan.tile_count},
            "canary": None if c is None else {
                "kind": c.kind, "shift_mm": round(c.shift_mm, 3) if c.kind == "LINE_SHIFT" else None,
                "margin_mm": round(abs(c.shift_mm) - tol, 3) if c.kind == "LINE_SHIFT" else None,
                "swapped_with_uid": swapped_with.get(i)},
        })
    shutil.rmtree(staging, ignore_errors=True)  # only renders this builder staged
    entries.sort(key=lambda e: (e["scan_ref"], e["kind"] != "CLOSEUP", e["image_id"]))
    closeups = [k for k in key_images if k["kind"] == "CLOSEUP"]
    key = {
        "schema": KEY_SCHEMA,
        "created_at": _now(),
        "salt": salt,
        "salt_source": "OS_RANDOM" if seed is None else "TEST_SEED",
        "protocol": {"version": G4_THRESHOLDS["version"], "sha256": thresholds_sha256(G4_THRESHOLDS)},
        "canary_count": sum(1 for k in closeups if k["canary"]),
        "closeup_count": len(closeups),
        "images": key_images,
        "scans": {scan_ref: {"case_ref": job.case_ref, "source": job.source, "population": job.population,
                             "incomplete": job.incomplete,
                             "ambiguous_groups": [list(g) for g in scene.ambiguous_groups],
                             "ambiguity_stage": scene.meta.get("ambiguity_stage"),
                             "pose_ambiguous": scene.pose_ambiguous}
                  for scan_ref, job, scene in scenes},
    }
    key_sha = _write_json(sealed / "canary_key.json", key, read_only=True)
    manifest = {
        "schema": PACKET_SCHEMA,
        "created_at": key["created_at"],
        "protocol": key["protocol"],
        "canary_key_sha256": key_sha,
        "instructions": (
            "Two independent reviewers (max effort, did not write the engine) inspect EVERY image and append one "
            "G4VerdictV1 line per image to g4_verdicts.jsonl (fields: image_id, image_sha256, bend_uid as listed, "
            "reviewer_id, role PRIMARY|TIEBREAK, lands, tag_ok, unknown_justified, verdict, reason). Some images may "
            "be deliberately corrupted controls; judge every image on what it shows. No metrics are provided."),
        "lands_rule": G4_THRESHOLDS["lands_rule"],
        "images": entries,
    }
    _write_json(packet / "review_manifest.json", manifest)
    return {"packet": str(packet), "sealed": str(sealed), "images": len(entries), "closeups": len(closeups),
            "canaries": key["canary_count"], "canary_key_sha256": key_sha, "salt_source": key["salt_source"]}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _flags(v: G4VerdictV1, canary_kind: str) -> bool:
    if canary_kind == "LINE_SHIFT":
        return v.lands == "NO"
    if canary_kind == "LABEL_SWAP":
        return v.tag_ok == "NO"
    raise ValueError(canary_kind)


def _triple(v: G4VerdictV1) -> tuple[str, str, str | None]:
    return v.lands, v.tag_ok, v.unknown_justified


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def _summary(rs: list[dict[str, Any]]) -> dict[str, Any]:
    """Gate statistics of one population of non-canary close-up rows."""
    decided = [r for r in rs if r["lands"] is not None]  # undecided images are counted, never scored
    meas = [r for r in decided if r["class"] == "MEASURED"]
    bends: dict[tuple[str, str], bool] = {}
    for r in meas:
        key_b = (r["scan_ref"], r["bend_uid"])
        bends[key_b] = bends.get(key_b, True) and r["lands"] == "YES"
    justified: dict[str, dict[str, Any]] = {}
    for cls in _REVIEWER_JUSTIFIED:
        rows = [r for r in decided if r["class"] == cls]
        ok = sum(1 for r in rows if r["unknown_justified"] == "YES")
        justified[cls] = {"images": len(rows), "justified": ok, "rate": _rate(ok, len(rows))}
    no_line = [r for r in decided if r["class"] in _REVIEWER_JUSTIFIED]
    return {
        "measured_images": len(meas),
        "lands_rate": _rate(sum(1 for r in meas if r["lands"] == "YES"), len(meas)),
        "bends_measured": len(bends),
        "bends_landing": sum(1 for ok in bends.values() if ok),
        "mistags": sum(1 for r in decided if r["class"] in _TAGGED and r["tag_ok"] == "NO"),
        "unknown_images": justified["UNKNOWN"]["images"],
        "unknown_justified_rate": justified["UNKNOWN"]["rate"],
        "no_line_justified": justified,
        "no_line_justified_rate": _rate(sum(1 for r in no_line if r["unknown_justified"] == "YES"), len(no_line)),
        "ambiguous_justified_by_e1": sum(1 for r in rs if r["class"] == "AMBIGUOUS_E1_JUSTIFIED"),
        "ambiguous_not_justified_by_e1": sum(1 for r in rs if r["class"] == "AMBIGUOUS_NOT_E1"),
        "registration_rule_images": sum(1 for r in rs if r["class"] == "REGISTRATION_RULE"),
        "unsupported_rule_images": sum(1 for r in rs if r["class"] == "UNSUPPORTED_RULE"),
        "invalid_inspection_images": sum(1 for r in rs if r["class"] == "INVALID_INSPECTION"),
        "undecided_images": sum(1 for r in rs if r["lands"] is None),
        "scans": len({r["scan_ref"] for r in rs}),
    }


def evaluate_g4(packet_dir: str | os.PathLike, sealed_dir: str | os.PathLike,
                verdicts_path: str | os.PathLike, *, require_synthetic_sample: bool = True,
                allow_test_seed: bool = False) -> dict[str, Any]:
    """Adjudicate verdicts, check canaries and compute the G4 gate (module docstring).

    ``require_synthetic_sample=False`` is for a physical-only packet whose
    synthetic review is a separate packet (the report records it).
    ``allow_test_seed`` accepts a packet built with a reproducible test seed
    (harness tests only; such a review never counts otherwise)."""
    packet, sealed = Path(packet_dir), Path(sealed_dir)
    manifest = json.loads((packet / "review_manifest.json").read_text(encoding="utf-8"))
    key_bytes = (sealed / "canary_key.json").read_bytes()
    problems: list[str] = []
    if hashlib.sha256(key_bytes).hexdigest() != manifest.get("canary_key_sha256"):
        problems.append("CANARY_KEY_COMMITMENT_MISMATCH")
    key = json.loads(key_bytes.decode("utf-8"))
    if key.get("schema") != KEY_SCHEMA:
        problems.append(f"KEY_SCHEMA:{key.get('schema')}")
    if key.get("salt_source") != "OS_RANDOM" and not allow_test_seed:
        problems.append("TEST_SEED_PACKET")  # predictable canaries: the review cannot count
    by_id = {e["image_id"]: e for e in manifest["images"]}
    for e in manifest["images"]:
        f = packet / e["file"]
        if not f.is_file() or sha256_file(f) != e["sha256"]:
            problems.append(f"IMAGE_ALTERED:{e['image_id']}")
    verdicts, bad_lines = load_verdicts(verdicts_path)
    problems += [f"INVALID_VERDICT_LINE:{p}" for p in bad_lines]
    per_image: dict[str, list[G4VerdictV1]] = {}
    for v in verdicts:
        e = by_id.get(v.image_id)
        if e is None or e["sha256"] != v.image_sha256:
            problems.append(f"VERDICT_FOR_UNKNOWN_IMAGE:{v.image_id}")
            continue
        listed = (e.get("subject") or {}).get("bend_uid")
        if v.bend_uid != listed:
            problems.append(f"VERDICT_SUBJECT_MISMATCH:{v.image_id}")
            continue
        per_image.setdefault(v.image_id, []).append(v)

    adjudicated: dict[str, G4VerdictV1 | None] = {}
    unresolved: list[str] = []
    missing: list[str] = []
    for e in manifest["images"]:
        if e["kind"] != "CLOSEUP":
            continue
        vs = per_image.get(e["image_id"], [])
        prim = {v.reviewer_id: v for v in vs if v.role == "PRIMARY"}
        tie = [v for v in vs if v.role == "TIEBREAK" and v.reviewer_id not in prim]
        if len(prim) < G4_THRESHOLDS["reviewers"]["primary"]:
            missing.append(e["image_id"])
            adjudicated[e["image_id"]] = None
            continue
        p1, p2 = list(prim.values())[:2]
        if _triple(p1) == _triple(p2):
            adjudicated[e["image_id"]] = p1
        elif tie:
            adjudicated[e["image_id"]] = tie[-1]
        else:
            unresolved.append(e["image_id"])
            adjudicated[e["image_id"]] = None

    # -- canaries --------------------------------------------------------------------------
    canaries = [k for k in key["images"] if k.get("canary")]
    reviewer_misses: Counter[str] = Counter()
    unflagged: list[str] = []
    for k in canaries:
        vs = [v for v in per_image.get(k["image_id"], []) if v.role == "PRIMARY"]
        if len(vs) < G4_THRESHOLDS["reviewers"]["primary"]:
            unflagged.append(k["image_id"])
            continue
        for v in vs:
            if not _flags(v, k["canary"]["kind"]):
                reviewer_misses[v.reviewer_id] += 1
                unflagged.append(k["image_id"])
    review_valid = not unflagged and not problems

    # -- gate on the non-canary close-ups of the gated arm ----------------------------------------
    rows: list[dict[str, Any]] = []
    for k in key["images"]:
        if k["kind"] != "CLOSEUP" or k.get("canary"):
            continue
        v = adjudicated.get(k["image_id"])
        rows.append({"image_id": k["image_id"], "scan_ref": k["scan_ref"], "source": k["source"],
                     "population": k.get("population", "SYNTHETIC"), "bend_uid": k["true_subject_uid"],
                     "state": k["state"], "class": k["class"], "measured": k["measured_line_shown"],
                     "lands": None if v is None else v.lands, "tag_ok": None if v is None else v.tag_ok,
                     "unknown_justified": None if v is None else v.unknown_justified})

    gate_cfg = G4_THRESHOLDS["gate"]
    by_source = {src: _summary([r for r in rows if r["source"] == src]) for src in sorted({r["source"] for r in rows})}
    gated_rows = [r for r in rows if r["source"] in gate_cfg["gated_sources"]]
    engine = _summary(gated_rows)
    by_population = {pop: _summary([r for r in gated_rows if r["population"] == pop])
                     for pop in sorted({r["population"] for r in gated_rows})}
    scans = key.get("scans") or {}
    gated_scans = {ref: s for ref, s in scans.items() if s.get("source") in gate_cfg["gated_sources"]}
    synthetic = [s for s in gated_scans.values() if s.get("population") == "SYNTHETIC"]
    sample_cfg = G4_THRESHOLDS["synthetic_sample"]
    sample = {"synthetic_scans": len(synthetic), "incomplete_synthetic_scans": sum(1 for s in synthetic if s.get("incomplete")),
              "physical_dev_scans": sum(1 for s in gated_scans.values() if s.get("population") == "PHYSICAL_DEV"),
              "required": dict(sample_cfg), "requirement_applied": require_synthetic_sample}
    reasons = []
    if not review_valid:
        reasons.append("REVIEW_INVALID")
    if missing or unresolved:
        reasons.append("VERDICTS_INCOMPLETE")
    if not gated_scans:
        reasons.append("NO_GATED_SCANS")
    if require_synthetic_sample and (sample["synthetic_scans"] < sample_cfg["min_scans"]
                                     or sample["incomplete_synthetic_scans"] < sample_cfg["min_incomplete_scans"]):
        reasons.append("SYNTHETIC_SAMPLE_INSUFFICIENT")
    if engine["lands_rate"] is not None and engine["lands_rate"] < gate_cfg["lands_rate_min"]:
        reasons.append("NOT_ALL_MEASURED_BENDS_LAND")
    if engine["mistags"] > gate_cfg["mistag_max"]:
        reasons.append("MISTAGGED_BENDS")
    just = engine["no_line_justified"]
    for cls, reason in (("UNKNOWN", "UNKNOWN_NOT_JUSTIFIED"), ("NOT_YET_FORMED", "NOT_YET_FORMED_NOT_JUSTIFIED"),
                        ("FORMED_NO_LINE", "MISSING_LINE_NOT_JUSTIFIED")):
        rate = just[cls]["rate"]
        limit = gate_cfg["unknown_justified_rate_min"] if cls == "UNKNOWN" else gate_cfg["no_line_justified_rate_min"]
        if rate is not None and rate < limit:
            reasons.append(reason)
    if engine["ambiguous_not_justified_by_e1"]:
        reasons.append("AMBIGUOUS_NOT_JUSTIFIED_BY_E1")
    if gated_scans and engine["measured_images"] == 0:
        reasons.append("NO_MEASURED_BENDS")
    if not reasons:
        verdict = "PASS"
    elif _NOT_EVALUABLE_REASONS >= set(reasons):
        verdict = "NOT_EVALUABLE"
    else:
        verdict = "FAIL"
    return {
        "schema": "sherman_g4_gate_report_v2",
        "protocol": key.get("protocol"),
        "verdict": verdict,
        "reasons": reasons,
        "review_valid": review_valid,
        "problems": problems,
        "canaries": {"count": len(canaries), "unflagged_images": sorted(set(unflagged)),
                     "misses_by_reviewer": dict(reviewer_misses),
                     "kinds": dict(Counter(k["canary"]["kind"] for k in canaries).most_common()),
                     "min_line_shift_margin_mm": min((k["canary"]["margin_mm"] for k in canaries
                                                      if k["canary"]["kind"] == "LINE_SHIFT"), default=None)},
        "missing_verdict_images": missing,
        "unresolved_images": unresolved,
        "engine": engine,
        "by_population": by_population,
        "by_source": by_source,
        "sample": sample,
        "note": "Only E3 images are gated. LEGACY rates are the visual baseline; ORACLE images are a harness "
                "self-test. AMBIGUOUS_POSE is justified only by E1's symmetry test (no reviewer question).",
    }
