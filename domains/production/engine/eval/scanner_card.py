"""Amendment-4b scanner model card: light statistics of authorized real scans -> E2 ScannerModel JSON.

The E2 virtual scanner (``domains.production.engine.synthetic.scanner.ScannerModel``)
ships with ASSUMED defaults. Amendment 4b requires a model card measured on at
least three authorized real scans (plan §9: ``../source-assets/02_CAD_QC_3D``,
development-consumed, read-only, one scan at a time). This module measures, per
scan, only light geometric statistics of the exported mesh (no registration,
no CAD, no labels):

- **point spacing**: median distance to the nearest other vertex, queried on an
  evenly spaced subset (the E2 statistic ``scanner.median_nn_spacing``);
- **noise floor**: robust sigma (MAD x 1.4826) of point-to-plane residuals in
  small (2 mm) same-skin patches (vertex normals within 14 deg of the seed's,
  so the opposite skin of the sheet never enters the fit), strictly planar
  patches only; the 5 mm and 10 mm residuals are reported as *waviness*;
- **skin separation**: the local distance between the two skins of the sheet
  (top-skin patch vs points with opposite normals); its robust spread across the
  part bounds the misregistration between the scan setups (plus the physical
  thickness tolerance), which is E2's ``duplicate_skin_offset_mm``;
- **duplicate skins**: share of same-skin patches whose residuals split into two
  layers (gap > 6 pooled sigma, minority share >= 15 %);
- **edge flare**: at free sheet edges (an edge-wall vertex whose perpendicular
  neighbours -- the two skins -- all lie behind it), how far the wall column under
  the vertex sticks out of the slab between the two skins, following only points
  contiguous with the slab; a column running more than 1 mm beyond the slab is
  another surface (a perpendicular flange or a bend) and the sample is excluded
  (counted). The share of edges with an excursion > max(0.05 mm, 4 sigma) and the
  95th percentile excursion among them (flares above 1 mm are not measurable);
- **outliers**: vertices whose 4th nearest neighbour is farther than 5 x the
  median spacing.

Grazing/specular/random dropout, edge erosion, the incidence gain and the sensor
geometry are **not identifiable** from an exported, merged mesh without the
sensor poses; the card keeps their E2 ASSUMED values and says so in ``notes``
(a MODEL_CARD must state every field, E2 enforces it).

**Field status.** The card notes end with a machine-readable
``FIELD_STATUS={...}`` object giving every E2 field one of MEASURED,
UPPER_BOUND (``duplicate_skin_offset_mm``: the skin-separation spread also holds
the thickness tolerance), LOWER_BOUND (``edge_flare_*``: flares above 1 mm and
the edge samples that continue into another surface are excluded) or ASSUMED.
:func:`field_status` parses it; the G1 scorecard (amendment 4b) is
non-qualifying (SCANNER_MODEL_PARTIAL) while an amendment-4b quantity -- noise,
point spacing, edge flare, grazing and specular dropout, duplicate-skin offset
-- is ASSUMED, unless the owner waives it in the freeze ledger.

:func:`build_card` maps the per-scan statistics (median over scans) onto the card
and returns it with an evidence document; :func:`validate_card` loads it with
E2's own ``ScannerModel.from_dict`` so the format can never drift. The evidence
of the committed card lives verbatim in ``docs/production/ENGINE_EVAL.md`` (a
fenced block tagged ``scanner-card-evidence``); :func:`load_evidence` reads it
from there or from a JSON file, so ``model-card --from-evidence
docs/production/ENGINE_EVAL.md --measured-on 2026-10-17`` rebuilds the
committed card byte for byte without reading any scan.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .plyio import read_ply, sha256_file

__all__ = [
    "CARD_TOOL_VERSION",
    "CARD_BUILD_VERSION",
    "MEASURED_FIELDS",
    "ASSUMED_FIELDS",
    "FIELD_STATUSES",
    "EVIDENCE_FENCE",
    "ScanStats",
    "measure_points",
    "measure_scan",
    "build_card",
    "field_status",
    "load_evidence",
    "stats_from_evidence",
    "validate_card",
]

#: Version of the measurement code (:func:`measure_points`), recorded in the evidence.
CARD_TOOL_VERSION = "e4a-scanner-card-1.0.0"
#: Version of the card assembly (:func:`build_card`: mapping, notes, FIELD_STATUS).
CARD_BUILD_VERSION = "e4a-scanner-card-build-1.1.0"
SCANNER_MODEL_SCHEMA = "sherman_synthetic_scanner_model_v1"
MEASURED_FIELDS = ("point_spacing_mm", "noise_sigma_mm", "duplicate_skin_offset_mm", "duplicate_skin_fraction",
                   "edge_flare_mm", "edge_flare_probability", "outlier_fraction")
#: Status of each measured field (every other E2 field is ASSUMED).
_MEASURED_STATUS = {"point_spacing_mm": "MEASURED", "noise_sigma_mm": "MEASURED",
                    "duplicate_skin_offset_mm": "UPPER_BOUND", "duplicate_skin_fraction": "MEASURED",
                    "edge_flare_mm": "LOWER_BOUND", "edge_flare_probability": "LOWER_BOUND",
                    "outlier_fraction": "MEASURED"}
FIELD_STATUSES = ("MEASURED", "UPPER_BOUND", "LOWER_BOUND", "ASSUMED")
_STATUS_MARK = "FIELD_STATUS="
#: Info string of the fenced evidence block in ENGINE_EVAL.md.
EVIDENCE_FENCE = "json scanner-card-evidence"
#: E2 fields that an exported mesh cannot identify; the card keeps E2's ASSUMED values.
ASSUMED_FIELDS = ("noise_incidence_gain", "edge_erosion_probability", "depth_jump_mm", "grazing_start_deg",
                  "grazing_cutoff_deg", "specular_cone_deg", "specular_dropout_probability",
                  "random_dropout_probability", "outlier_range_mm", "duplicate_skin_rotation_deg",
                  "duplicate_skin_patch_mm", "view_jitter_mm", "triangulation_angle_deg", "working_distance_mm")
_NN_SAMPLE = 50_000
#: A wall column contiguous beyond this distance from the skin slab is another surface
#: (a perpendicular flange or a bend), not a mixed-pixel flare; such samples are excluded.
_FLARE_CAP_MM = 1.0


@dataclass(frozen=True)
class ScanStats:
    name: str
    sha256: str | None
    vertex_count: int
    face_count: int
    has_normals: bool
    nn_spacing_mm: float
    noise_sigma_mm: float | None
    noise_patches: int
    waviness_5mm: float | None
    waviness_10mm: float | None
    skin_separation_median_mm: float | None
    skin_separation_sigma_mm: float | None
    skin_separation_p5_p95_mm: tuple[float, float] | None
    skin_separation_patches: int
    duplicate_layer_fraction: float | None
    duplicate_layer_gap_mm: float | None
    edge_samples: int
    #: edge samples excluded because the wall continues into another surface (> 1 mm beyond the slab)
    edge_samples_complex: int
    edge_flare_fraction: float | None
    edge_flare_p95_mm: float | None
    outlier_fraction: float
    seconds: float
    peak_rss_bytes: int


def _robust_sigma(r: np.ndarray) -> float:
    return float(np.median(np.abs(r - np.median(r))) * 1.4826)


def _plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = points.mean(axis=0)
    _, s, vt = np.linalg.svd(points - c, full_matrices=False)
    return c, vt[2], s


def _two_layers(r: np.ndarray) -> tuple[float, float, float]:
    """Best 2-means split of sorted residuals: (gap / pooled sigma, gap, minority share)."""
    rs = np.sort(r)
    n = len(rs)
    cs, cs2 = np.cumsum(rs), np.cumsum(rs * rs)
    best = (0.0, 0.0, 0.0)
    best_w = None
    for k in range(5, n - 5):
        na, nb = k, n - k
        ma, mb = cs[k - 1] / na, (cs[-1] - cs[k - 1]) / nb
        va = cs2[k - 1] / na - ma * ma
        vb = (cs2[-1] - cs2[k - 1]) / nb - mb * mb
        w = (va * na + vb * nb) / n
        if best_w is None or w < best_w:
            best_w = w
            pooled = float(np.sqrt(max(w, 1e-18)))
            best = ((mb - ma) / pooled, float(mb - ma), min(na, nb) / n)
    return best


def _area_uniform_seeds(v: np.ndarray, count: int, rng: np.random.Generator, voxel_mm: float = 1.0) -> np.ndarray:
    """One vertex per occupied ``voxel_mm`` voxel, then a random subset: seeds are
    uniform in surface area, so doubled (denser) regions are not over-sampled."""
    keys = np.floor(v / voxel_mm).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return rng.choice(first, min(count, len(first)), replace=False)


def _principal_axis(normals: np.ndarray) -> np.ndarray:
    """Sign-free dominant direction of a set of unit normals."""
    w, vec = np.linalg.eigh(normals.T @ normals)
    return vec[:, int(np.argmax(w))]


def _contiguous_excursion(h: np.ndarray, lo: float, hi: float, *, gap: float) -> float | None:
    """How far a wall column extends beyond the skin slab ``[lo, hi]``, following
    only points contiguous with the slab (consecutive heights closer than ``gap``).
    None when the column has no point inside the slab (not a sheet edge)."""
    hs = np.sort(h)
    inside = (hs >= lo - gap) & (hs <= hi + gap)
    if not inside.any():
        return None
    top = hs[inside].max()
    for x in hs[hs > top]:
        if x - top > gap:
            break
        top = x
    bottom = hs[inside].min()
    for x in hs[hs < bottom][::-1]:
        if bottom - x > gap:
            break
        bottom = x
    return float(max(0.0, top - hi, lo - bottom))


def measure_points(vertices: np.ndarray, normals: np.ndarray | None, *, name: str = "points",
                   sha256: str | None = None, face_count: int = 0, seed: int = 0, n_patches: int = 3000,
                   n_edge_seeds: int = 30000) -> ScanStats:
    """Statistics of one exported cloud (module docstring). Needs vertex normals
    for the same-skin, skin-separation and flare statistics (else they are None)."""
    from scipy.spatial import cKDTree

    t0 = time.time()
    v = np.asarray(vertices, dtype=np.float64)
    tree = cKDTree(v)
    q = v[np.unique(np.linspace(0, len(v) - 1, min(len(v), _NN_SAMPLE)).astype(np.int64))]
    d, _ = tree.query(q, k=2)
    spacing = float(np.median(d[:, 1]))
    d5, _ = tree.query(q, k=min(5, len(v)))
    outlier = float(np.mean(d5[:, -1] > 5.0 * spacing)) if len(v) >= 5 else 0.0
    rng = np.random.default_rng(seed)
    noise: list[float] = []
    wav5: list[float] = []
    wav10: list[float] = []
    seps: list[float] = []
    patches = 0
    dup_patches = 0
    dup_gaps: list[float] = []
    edges: list[float] = []
    complex_edges = 0
    if normals is not None:
        n = np.asarray(normals, dtype=np.float64)
        for s in _area_uniform_seeds(v, n_patches, rng):
            nb = np.asarray(tree.query_ball_point(v[s], 2.0))
            same = nb[(n[nb] @ n[s]) > 0.97] if len(nb) else nb
            if len(same) < 25:
                continue
            c, normal, sv = _plane(v[same])
            spread = np.degrees(np.arccos(np.clip(np.abs(n[same] @ normal), -1, 1)))
            if np.percentile(spread, 90) > 4.0:
                continue  # vertex normals say this is not a flat patch (bend, edge, hole wall)
            r = (v[same] - c) @ normal
            patches += 1
            ratio, gap, minority = _two_layers(r)
            if ratio > 6.0 and minority >= 0.15:
                dup_patches += 1
                dup_gaps.append(gap)
            elif sv[2] / max(sv[1], 1e-12) <= 0.05:
                noise.append(_robust_sigma(r))
            # skin separation: opposite-normal points straight below the patch (lateral <= 2 mm)
            deep = np.asarray(tree.query_ball_point(v[s], 6.0))
            opp = deep[(n[deep] @ n[s]) < -0.97] if len(deep) else deep
            if len(opp) >= 10:
                rel = v[opp] - c
                along = rel @ normal
                lateral = np.linalg.norm(rel - np.outer(along, normal), axis=1)
                below = along[lateral <= 2.0]
                if len(below) >= 10:
                    seps.append(abs(float(np.median(below))))
            for radius, sink in ((5.0, wav5), (10.0, wav10)):
                nb2 = np.asarray(tree.query_ball_point(v[s], radius))
                same2 = nb2[(n[nb2] @ n[s]) > 0.97]
                if len(same2) < 25:
                    continue
                c2, normal2, _ = _plane(v[same2])
                spread2 = np.degrees(np.arccos(np.clip(np.abs(n[same2] @ normal2), -1, 1)))
                if np.percentile(spread2, 90) > 4.0:
                    continue
                sink.append(_robust_sigma((v[same2] - c2) @ normal2))
        col_radius = max(0.75 * spacing, 0.2)
        for s in rng.choice(len(v), min(n_edge_seeds, len(v)), replace=False):
            nw = n[s]
            nb = np.asarray(tree.query_ball_point(v[s], 4.0))
            if len(nb) < 30:
                continue
            dots = n[nb] @ nw
            cand = nb[np.abs(dots) < 0.2]
            if len(cand) < 30:
                continue
            ns = _principal_axis(n[cand])
            ns = ns - nw * float(ns @ nw)
            if float(np.linalg.norm(ns)) < 0.5:
                continue
            ns /= np.linalg.norm(ns)
            top, bot = cand[(n[cand] @ ns) > 0.95], cand[(n[cand] @ ns) < -0.95]
            if len(top) < 10 or len(bot) < 10:
                continue
            skins = np.concatenate([top, bot])
            if np.mean((v[skins] - v[s]) @ nw < 0.3) < 0.97:
                continue  # not a free edge: both skins must lie behind the wall
            ht, hb = float(np.median(v[top] @ ns)), float(np.median(v[bot] @ ns))
            lo, hi = min(ht, hb), max(ht, hb)
            rel = v[nb] - v[s]
            lateral = np.linalg.norm(rel - np.outer(rel @ ns, ns), axis=1)
            column = nb[(dots > 0.7) & (lateral <= col_radius)]
            if not len(column):
                continue
            excursion = _contiguous_excursion(v[column] @ ns, lo, hi, gap=max(2.0 * spacing, 0.3))
            if excursion is None or excursion > _FLARE_CAP_MM:
                complex_edges += 1  # the wall continues into another surface, not a flare
                continue
            edges.append(excursion)
    floor = float(np.median(noise)) if noise else None
    thr = max(0.05, 4.0 * (floor or 0.0))
    ex = np.asarray(edges)
    flagged = ex[ex > thr] if ex.size else ex
    sep = np.asarray(seps)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ScanStats(
        name=name, sha256=sha256, vertex_count=int(len(v)), face_count=int(face_count),
        has_normals=normals is not None, nn_spacing_mm=spacing, noise_sigma_mm=floor, noise_patches=len(noise),
        waviness_5mm=float(np.median(wav5)) if wav5 else None, waviness_10mm=float(np.median(wav10)) if wav10 else None,
        skin_separation_median_mm=float(np.median(sep)) if sep.size else None,
        skin_separation_sigma_mm=_robust_sigma(sep) if sep.size >= 10 else None,
        skin_separation_p5_p95_mm=(float(np.percentile(sep, 5)), float(np.percentile(sep, 95))) if sep.size else None,
        skin_separation_patches=int(sep.size),
        duplicate_layer_fraction=(dup_patches / patches) if patches else None,
        duplicate_layer_gap_mm=float(np.median(dup_gaps)) if dup_gaps else None,
        edge_samples=int(ex.size), edge_samples_complex=int(complex_edges), edge_flare_fraction=float(len(flagged) / ex.size) if ex.size else None,
        edge_flare_p95_mm=float(np.percentile(flagged, 95)) if len(flagged) else (0.0 if ex.size else None),
        outlier_fraction=outlier, seconds=round(time.time() - t0, 2),
        peak_rss_bytes=int(rss if sys.platform == "darwin" else rss * 1024),
    )


def measure_scan(path: str | os.PathLike, *, seed: int = 0, n_patches: int = 3000,
                 n_edge_seeds: int = 30000) -> ScanStats:
    """Read one exported PLY (vertices + normals only) and measure it."""
    p = Path(path)
    sha = sha256_file(p)
    data = read_ply(p, with_faces=False)
    from .plyio import read_ply_header

    faces = read_ply_header(p).count("face")
    return measure_points(data.vertices, data.normals, name=p.name, sha256=sha, face_count=faces, seed=seed,
                          n_patches=n_patches, n_edge_seeds=n_edge_seeds)


def _assumed_defaults() -> dict[str, Any]:
    from domains.production.engine.synthetic.scanner import ScannerModel

    return ScannerModel().to_dict()


def build_card(stats: Sequence[ScanStats], *, name: str = "zeiss_geomagic_export_model_card_v1",
               min_scans: int = 3, measured_on: str = "", measured_by: str = CARD_TOOL_VERSION
               ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Card (E2 ScannerModel JSON, provenance MODEL_CARD) and its evidence document.

    Each measured field is the median over scans of the per-scan statistic:

    - ``point_spacing_mm`` = NN spacing; ``noise_sigma_mm`` = 2 mm noise floor
      (measured on flat same-skin patches of the merged export);
    - ``duplicate_skin_offset_mm`` = robust sigma of the skin separation (an
      UPPER_BOUND of the per-setup misregistration: it also holds the physical
      thickness tolerance);
    - ``duplicate_skin_fraction`` = share of two-layer patches;
    - ``edge_flare_probability`` = share of free-edge samples with flare;
      ``edge_flare_mm`` = their 95th percentile excursion / 0.95 (E2 draws the
      flare uniformly up to ``edge_flare_mm``); both LOWER_BOUNDs (flares above
      1 mm and samples that continue into another surface are excluded);
    - ``outlier_fraction`` = isolated-vertex share.

    ``measured_by`` is the measurement tool version (from the evidence when the
    card is rebuilt without reading scans).
    """
    if len(stats) < min_scans:
        raise ValueError(f"amendment 4b needs at least {min_scans} scans, got {len(stats)}")
    if any(not s.has_normals for s in stats):
        raise ValueError("every scan needs vertex normals for the same-skin statistics")

    def med(values: Sequence[float | None]) -> float:
        vals = [float(x) for x in values if x is not None]
        if not vals:
            raise ValueError("a measured field has no value on any scan")
        return float(np.median(vals))

    card = _assumed_defaults()
    measured = {
        "point_spacing_mm": med([s.nn_spacing_mm for s in stats]),
        "noise_sigma_mm": med([s.noise_sigma_mm for s in stats]),
        "duplicate_skin_offset_mm": med([s.skin_separation_sigma_mm for s in stats]),
        "duplicate_skin_fraction": med([s.duplicate_layer_fraction for s in stats]),
        "edge_flare_probability": med([s.edge_flare_fraction for s in stats]),
        "edge_flare_mm": med([s.edge_flare_p95_mm for s in stats]) / 0.95,
        "outlier_fraction": med([s.outlier_fraction for s in stats]),
    }
    card.update({k: round(v, 6) for k, v in measured.items()})
    card["name"] = name
    card["provenance"] = "MODEL_CARD"
    status = {k: _MEASURED_STATUS.get(k, "ASSUMED") for k in card if k not in ("schema", "name", "provenance", "notes")}
    scans = ", ".join(f"{s.name} ({(s.sha256 or '')[:12]})" for s in stats)
    w5 = med([s.waviness_5mm for s in stats])
    w10 = med([s.waviness_10mm for s in stats])
    floor = measured["noise_sigma_mm"]
    wav = ("close to the 2 mm floor (no large-scale waviness beyond white noise)" if max(w5, w10) <= 2.0 * floor
           else "above the 2 mm floor: a correlated error the white-noise model does not reproduce")
    excluded = [s.edge_samples_complex / max(1, s.edge_samples + s.edge_samples_complex) for s in stats]
    card["notes"] = (
        f"Amendment-4b model card ({CARD_BUILD_VERSION}; statistics by {measured_by}"
        f"{', ' + measured_on if measured_on else ''}). "
        f"Measured from the scans (median over {len(stats)} authorized development scans, plan §9; consumed "
        f"development data, light statistics only; status per field in FIELD_STATUS, some are bounds): "
        f"{', '.join(MEASURED_FIELDS)}. Scans: {scans}. "
        f"Retained E2 ASSUMED values (not identifiable from an exported merged mesh without sensor poses): "
        f"{', '.join(ASSUMED_FIELDS)}; amendment 4b also asks for grazing and specular dropout, so this card is "
        f"PARTIAL until those are measured (raw per-view exports or sensor poses) or the owner waives them. "
        f"noise_sigma_mm is the 2 mm same-skin noise floor of the exports; the 5 mm / 10 mm patch residuals "
        f"(median {w5:.4f} / {w10:.4f} mm) are {wav}. duplicate_skin_offset_mm is an UPPER BOUND (skin-separation "
        f"spread also holds the sheet thickness tolerance). edge_flare_mm and edge_flare_probability are LOWER BOUNDS: "
        f"only flares up to {_FLARE_CAP_MM:g} mm beyond the skins are measurable, and "
        f"{min(excluded):.0%}-{max(excluded):.0%} of the edge samples per scan were excluded because the wall continues "
        f"into another surface. Evidence: docs/production/ENGINE_EVAL.md (block {EVIDENCE_FENCE!r}). "
        f"{_STATUS_MARK}{json.dumps(status, sort_keys=True, separators=(',', ':'))}"
    )
    evidence = {
        "schema": "sherman_e4_scanner_card_evidence_v1",
        "tool": measured_by,
        "card_build": CARD_BUILD_VERSION,
        "measured_on": measured_on,
        "measured": {k: round(v, 6) for k, v in measured.items()},
        "field_status": status,
        "assumed_fields": list(ASSUMED_FIELDS),
        "scans": [asdict(s) for s in stats],
    }
    return card, evidence


def field_status(model: dict[str, Any]) -> dict[str, str] | None:
    """Per-field status of a scanner model (module docstring): the FIELD_STATUS
    object at the end of a MODEL_CARD's notes; every field ASSUMED for an ASSUMED
    model; None when a MODEL_CARD carries no parseable FIELD_STATUS."""
    if model.get("provenance") == "ASSUMED":
        return {k: "ASSUMED" for k in model if k not in ("schema", "name", "provenance", "notes")}
    notes = str(model.get("notes") or "")
    at = notes.rfind(_STATUS_MARK)
    if at < 0:
        return None
    try:
        doc, _ = json.JSONDecoder().raw_decode(notes[at + len(_STATUS_MARK):])
    except ValueError:
        return None
    if not isinstance(doc, dict) or any(v not in FIELD_STATUSES for v in doc.values()):
        return None
    return {str(k): str(v) for k, v in doc.items()}


def load_evidence(path: str | os.PathLike) -> dict[str, Any]:
    """An evidence document from a JSON file, or from the fenced block tagged
    ``scanner-card-evidence`` of a Markdown file (ENGINE_EVAL.md)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() != ".md":
        return json.loads(text)
    fence = "```" + EVIDENCE_FENCE
    start = text.find(fence)
    if start < 0:
        raise ValueError(f"{p}: no fenced block tagged {EVIDENCE_FENCE!r}")
    body_start = text.index("\n", start) + 1
    end = text.find("\n```", body_start)
    if end < 0:
        raise ValueError(f"{p}: unterminated evidence block")
    return json.loads(text[body_start:end])


def stats_from_evidence(doc: dict[str, Any]) -> list[ScanStats]:
    """Per-scan statistics back from an evidence document (rebuild a card without re-reading scans)."""
    out = []
    for row in doc["scans"]:
        row = dict(row)
        if row.get("skin_separation_p5_p95_mm") is not None:
            row["skin_separation_p5_p95_mm"] = tuple(row["skin_separation_p5_p95_mm"])
        out.append(ScanStats(**row))
    return out


def validate_card(card: dict[str, Any]) -> str:
    """Load the card with E2's ``ScannerModel.from_dict`` (rejects unknown or missing
    fields for a MODEL_CARD); returns the model's sha256."""
    from domains.production.engine.synthetic.scanner import ScannerModel

    return ScannerModel.from_dict(dict(card)).sha256()


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m domains.production.engine.eval.scanner_card",
                                 description="Amendment-4b scanner model card from authorized real scans")
    ap.add_argument("scans", nargs="*", type=Path)
    ap.add_argument("--from-evidence", type=Path,
                    help="rebuild the card from saved evidence (a JSON file or ENGINE_EVAL.md); no scan is read")
    ap.add_argument("--out", required=True, type=Path, help="card JSON (E2 ScannerModel format)")
    ap.add_argument("--evidence", type=Path, help="evidence JSON (per-scan statistics)")
    ap.add_argument("--name", default="zeiss_geomagic_export_model_card_v1")
    ap.add_argument("--measured-on", default="", help="measurement date recorded in the notes, e.g. 2026-10-17")
    a = ap.parse_args(argv)
    stats = []
    measured_by = CARD_TOOL_VERSION
    measured_on = a.measured_on
    if a.from_evidence is not None:
        doc = load_evidence(a.from_evidence)
        stats = stats_from_evidence(doc)
        measured_by = str(doc.get("tool") or CARD_TOOL_VERSION)
        measured_on = measured_on or str(doc.get("measured_on") or "")
    for path in a.scans:  # one scan at a time (plan §7)
        st = measure_scan(path)
        print(json.dumps({"scan": st.name, "seconds": st.seconds, "vertices": st.vertex_count}), flush=True)
        stats.append(st)
    card, evidence = build_card(stats, name=a.name, measured_on=measured_on, measured_by=measured_by)
    sha = validate_card(card)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if a.evidence:
        evidence["card_sha256"] = sha
        a.evidence.write_text(json.dumps(evidence, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"card": str(a.out), "scanner_model_sha256": sha}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
