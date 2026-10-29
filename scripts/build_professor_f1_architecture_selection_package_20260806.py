#!/usr/bin/env python3
"""Build the exact ten-file fresh F1 architecture-selection packet.

The packet is deliberately decision-only.  It cannot authorize execution,
open gold, or inherit any policy from the stopped local-plane-fragment family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "output/professor_package"
    / "Sherman_QC_F1_Fresh_Architecture_Selection_20260806_v1"
)
DEFAULT_AUDIT_PATH = (
    ROOT
    / "output/research/f1-new-architecture-selection-20260806"
    / "package_integrity_audit.json"
)
MAX_FILES = 10

SOURCE_BINDINGS = {
    "formal_lpf_stop": {
        "path": ROOT
        / "docs/research/PROFESSOR_F1_LOCAL_PLANE_FRAGMENT_FAMILY_STOP_RESPONSE_20260805.md",
        "sha256": "cd78057a14b7f763b7f503841ef7dc822f72de1fb207a8796dc74dddef6bc935",
    },
    "ordinary_compact_evidence": {
        "path": ROOT
        / "output/research/f1-gold27-20260803/v2_cross_parent_opposite_fragment_rung_ordinary_batch_v2_compact_evidence.json",
        "sha256": "df2911e6b2b31dc65224c678ce8214625e5d6f3e25fe70d9d9b3764b1679266f",
    },
    "enterprise_gate_spec": {
        "path": ROOT / "docs/release/ENTERPRISE_BEND_PROMOTION_GATES.md",
        "sha256": "25856776f6048501c5e364cdad3fd33d70f5c89c1fb8c4e05dfbb5c46a4125b6",
    },
    "enterprise_gate_snapshot": {
        "path": ROOT / "output/migration/enterprise-promotion-gates.json",
        "sha256": "2586bc09b7328900d5140857ece7553fd0188fddd29f2c462e01195aac13bfb8",
    },
}

REPOSITORY_BINDINGS = {
    "backend/zero_shot_bend_profile.py": "b263a46bb79c4bca3f2b279d528916b387b62e757220af1d54bf71f0618b712a",
    "scripts/build_f1_scan_only_candidate_pool.py": "7a6ce72684748b85dbddaa7cdfd9b6861368ced7ba8f086ee923de978f73a5f9",
    "scripts/trace_raw_point_crease_candidates.py": "602eb1df67fca0f4b408f2cdee63a2944d43c0c905acc2d4bc7ff275df27574d",
    "scripts/trace_multiscale_raw_point_crease_candidates.py": "cc4c4c9986ecda6d54c96272be22e81233d9863014d90dc7e60f344c1a74c205",
    "backend/f1_mesh_interface_geometry.py": "3fff57128c0316056c18e7053947da485e31aa594fb8f20c180ee6fd48371e5a",
    "scripts/run_f1_gold27_mesh_crossfit.py": "e9d8b22b7188a61ea9665d16a3642b883f797e168380ba7210960a606c4792fb",
    "domains/bend/services/heatmap_evidence_service.py": "f1b7b0dd13c1afc5e3db478e719e58642566d2c0e449c5ece3899b50970dd178",
    "scripts/correlate_f1_bends_with_surface_heatmap.py": "1265170409b59b2043531508bdf6e89668ae6350e937bdfd3e6b6c2e339192a8",
}

POSITIVE_TOKENS = (
    "SELECT_PAIRED_WEDGE_TRANSITION_STRIP_V1",
    "SELECT_MULTISCALE_RAW_NORMAL_TENSOR_CREASE_FIELD_V1",
    "SELECT_ROBUST_OSCULATING_JET_RIDGE_TRACE_V1",
)
STOP_TOKEN = "NO_ARCHITECTURE_SELECTED_EVIDENCE_INSUFFICIENT_STOP"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _validate_bound_sources() -> None:
    for label, binding in SOURCE_BINDINGS.items():
        path = Path(binding["path"])
        if not path.is_file():
            raise FileNotFoundError(f"missing bound source {label}: {path}")
        actual = _sha_file(path)
        if actual != binding["sha256"]:
            raise ValueError(
                f"bound source {label} differs from expected SHA-256: {actual}"
            )
    for relative, expected in REPOSITORY_BINDINGS.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing repository binding: {relative}")
        actual = _sha_file(path)
        if actual != expected:
            raise ValueError(
                f"repository binding {relative} differs from expected SHA-256: {actual}"
            )


def _validate_compact_evidence(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "aggregate_added_cross_parent_rung_count": 161,
        "frozen_target_count": 4_788,
        "target_component_image_present_count": 4_772,
        "target_component_image_absent_outside_tube_count": 16,
        "target_reached_by_added_rung_count": 122,
        "target_side_gate_advanced_count": 0,
        "target_novel_rung_attributable_certificate_count": 0,
        "maximum_peak_rss_bytes": 527_450_112,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"ordinary compact evidence mismatch for {key}")
    safety = {
        "truth_inputs_consumed": False,
        "gold_open_authorized": False,
        "designated_replays_authorized": False,
        "full_synthetic_resource_run_authorized": False,
        "ordinary_batch_authorized": False,
        "next_professor_packet_authorized": True,
    }
    for key, value in safety.items():
        if payload.get(key) is not value:
            raise ValueError(f"ordinary compact safety mismatch for {key}")


def _enterprise_memo() -> str:
    return """# Enterprise constraints and current promotion boundary

F1 remains offline research. This architecture-selection packet is not a release
review and cannot promote any implementation to runtime or main.

## Hash-bound gate evidence

- Enterprise specification SHA-256:
  `25856776f6048501c5e364cdad3fd33d70f5c89c1fb8c4e05dfbb5c46a4125b6`
- Current gate snapshot SHA-256:
  `2586bc09b7328900d5140857ece7553fd0188fddd29f2c462e01195aac13bfb8`
- Current status: `blocked`; 5 of 16 gates pass and 11 remain blocked.

The snapshot has 13 reviewed scans across 10 parts, no 30-part locked holdout,
and insufficient family coverage. Blind PLY-only exact count is 0/13; blind
CAD+PLY exact count is 3/12. These values are historical release evidence, not
inputs to the proposed representation preflight.

Release still requires at least 150 reviewed scans across 75 parts, at least 30
untouched holdout parts, at least 30 reviewed scans per declared family, bend
localization truth on at least 95% of release scans, PLY-only exact count at
least 90%, CAD+PLY exact count at least 95%, range containment at least 98%,
render-admissible instance accuracy at least 99%, deterministic 10/10 repeats,
P95 latency at most 300 seconds, and job success at least 99.5%.

The requested decision is narrower: choose one materially new scan-only
representation and prescribe one bounded gold-free preflight that can establish
whether its primitive is coherent enough to justify a later professor packet.
It may not claim generalization, accuracy, ownership, or release readiness.
"""


def _repository_memo() -> str:
    hashes = "\n".join(
        f"- `{path}` — `{digest}`"
        for path, digest in REPOSITORY_BINDINGS.items()
    )
    return f"""# Repository capability and contamination boundary

## Decision boundary

The existing F1 repository is not a neutral starting profile. The zero-shot
profile policies, raw-tracer thresholds, mesh selectors/topology defaults, and
heatmap consistency gates were developed with corpus feedback and are therefore
label-conditioned. None may be inherited by the selected architecture.

The stopped local-plane-fragment (LPF) representation is especially closed:
do not inherit its planes, fragments, pair axes, tubes, bins, thresholds,
certificate gates, graph components, rungs, ownership fields, or rescue logic.
Calling the same mechanism through a new module or token does not make it new.

## Narrow infrastructure that may be reused

Only implementation plumbing without semantic decisions may be reused:

- raw scan bytes and raw scan SHA-256 identity;
- the deterministic canonical sampling mechanism capped at 200,000 finite
  points, after binding raw coordinates and before recomputing fresh normals;
- compact numeric arrays and bounded integer/row IDs;
- the no-source-point-serialization and forbidden-key guards;
- isolated fresh-process execution, peak-RSS measurement, cache/schema identity,
  canonical JSON/digest, and deterministic-order utilities.

The current sampler sorts with supplied normals, so its normal-dependent ordering
is not reusable. Only its bounded deterministic mechanism may be adapted. Even
when neutral utilities are reused, the selected architecture must recompute
its normals, neighborhoods, scales, tensors, jets, transition evidence, traces,
and acceptance rules from a fresh preregistered configuration. Existing candidate
rows may not seed it.

## Explicitly excluded executable lanes

- CAD, nominal/reference geometry, CAD-vs-scan overlays, or CAD-derived truth;
- mesh reconstruction, mesh selectors, triangle topology, or mesh crossfit;
- screened Poisson or any global solver/optimization lane;
- existing tuned raw-point tracers or multiscale tracer acceptance thresholds;
- heatmap agreement as a birth, selection, or acceptance gate;
- LPF tuning, replay, rescue, threshold sweep, or semantic reuse.

The two old raw-tracer scripts are evidence that bounded point-native plumbing
exists, not implementations to select or tune. The mesh and heatmap files are
included only in this audit map to make the exclusion explicit.

## Line-level audit findings

- `zero_shot_bend_profile.py`: policy constants appear at lines 16–58;
  configured detector/candidate bundles at 224–308; part and scan-state hints at
  452–504; policy selection from 1318; tuned family gates at 1711–2065; and
  plane-lattice completion thresholds at 2710–2755. Its bounded detector input
  at 369–408 is coupled to a 192k policy and normal estimation.
- `build_f1_scan_only_candidate_pool.py`: the neutral 200k cap and sampler are at
  50 and 74–127, but supplied normals affect sorting. Forbidden-key validation
  is at 201–243, compact bounded tables at 313–369, and semantic plane/transition
  construction at 437–743 and 1125–1258.
- `trace_raw_point_crease_candidates.py`: tuned score/admissibility rules are at
  127–201 and sampled source points are serialized at 180–181. The executable is
  incompatible with the new schema.
- `trace_multiscale_raw_point_crease_candidates.py`: fixed scale recipes and
  duplicate gates are at 26–129; old tracer reuse is at 169–210.
- `run_f1_gold27_mesh_crossfit.py`: sealed manual/gold inputs at 39–42 and
  106–139 feed classifier and grid selection at 53–71 and 168–219.
- `heatmap_evidence_service.py`: its cache key uses path metadata/config rather
  than scan-content SHA at 15–30; CAD alignment/deviation occurs at 128–180.
- `correlate_f1_bends_with_surface_heatmap.py`: good/defective reference and
  manual inputs are required at 52–78; diagnostic gates are fixed at 101–108.

None of the eight audited files computes a raw scan file-bytes SHA-256, and none
evidences fresh-process RSS enforcement. Those are mandatory fresh bindings or
neutral external utilities, not claims about these implementations.

## Hash-bound repository audit surface

{hashes}
"""


def _research_memo() -> str:
    return """# Primary-source research memo for a materially new scan-only family

These papers justify primitive families, not Sherman-specific thresholds. No
paper result is treated as proof of bend accuracy or as authorization to reuse
the existing F1 implementations.

1. Niloy J. Mitra and An Nguyen, “Estimating Surface Normals in Noisy Point
   Cloud Data,” *Proceedings of the Nineteenth Annual Symposium on Computational
   Geometry*, 2003, pp. 322–328.
   Primary DOI: https://doi.org/10.1145/777792.777840
   Publisher: https://dl.acm.org/doi/10.1145/777792.777840
   Relevance: neighborhood size, density, curvature, and noise interact in local
   normal estimation; a fresh multiscale normal field must expose this dependence.

2. Mark Pauly, Richard Keiser, and Markus Gross, “Multi-scale Feature Extraction
   on Point-Sampled Surfaces,” *Computer Graphics Forum* 22(3), 2003, pp. 281–289.
   Primary DOI: https://doi.org/10.1111/1467-8659.00675
   Publisher: https://onlinelibrary.wiley.com/doi/10.1111/1467-8659.00675
   Relevance: point-native PCA classification across neighborhood scales supports
   an independently rebuilt multiscale crease field, but not inherited hysteresis.

3. Kris Demarsin, Denis Vanderstraeten, Tim Volodine, and Dirk Roose,
   “Detection of Closed Sharp Edges in Point Clouds Using Normal Estimation and
   Graph Theory,” *Computer-Aided Design* 39(4), 2007, pp. 276–283.
   Primary DOI: https://doi.org/10.1016/j.cad.2006.12.005
   Publisher: https://linkinghub.elsevier.com/retrieve/pii/S0010448506002260
   Relevance: normal discontinuities can support feature-line construction on
   point clouds; graph closure itself is not evidence of a physical bend.

4. Min Ki Park, Seung Joo Lee, and Kwan H. Lee, “Multi-scale Tensor Voting for
   Feature Extraction from Unstructured Point Clouds,” *Graphical Models* 74(4),
   2012, pp. 197–208.
   Primary DOI: https://doi.org/10.1016/j.gmod.2012.04.008
   Publisher: https://linkinghub.elsevier.com/retrieve/pii/S1524070312000288
   Relevance: tensor eigenstructure and scale behavior motivate a raw normal-
   tensor singular field without mesh or training data.

5. Frédéric Cazals and Marc Pouget, “Estimating Differential Quantities Using
   Polynomial Fitting of Osculating Jets,” *Computer Aided Geometric Design*
   22(2), 2005, pp. 121–146.
   Primary DOI: https://doi.org/10.1016/j.cagd.2004.09.004
   Publisher: https://linkinghub.elsevier.com/retrieve/pii/S016783960400113X
   Relevance: robustly fitted local jets provide a mathematically distinct route
   to normals, curvature, and curvature extrema directly from sampled points.

6. Yani Ioannou, Babak Taati, Robin Harrap, and Michael Greenspan, “Difference
   of Normals as a Multi-scale Operator in Unorganized Point Clouds,” *2012
   Second International Conference on 3D Imaging, Modeling, Processing,
   Visualization and Transmission*, 2012, pp. 501–508.
   Primary DOI: https://doi.org/10.1109/3DIMPVT.2012.12
   Publisher: https://ieeexplore.ieee.org/document/6375034/
   Relevance: differences between normals at two scales motivate false-first
   multiscale transition evidence computed on raw bounded scans.

7. Yutaka Ohtake, Alexander Belyaev, and Hans-Peter Seidel, “Ridge-Valley Lines
   on Meshes via Implicit Surface Fitting,” *ACM Transactions on Graphics* 23(3),
   2004, pp. 609–612.
   Primary DOI: https://doi.org/10.1145/1015706.1015768
   Publisher: https://dl.acm.org/doi/10.1145/1015706.1015768
   Relevance is limited to ridge/valley differential conditions and finite curve
   tracing. Its mesh and multilevel implicit-fitting pipeline is not an executable
   option here and does not authorize Poisson, mesh, or solver use.

The strongest Sherman preflight candidate is the paired wedge transition strip:
it combines multiscale raw normal-change maxima with a finite tangent trace and
an explicit two-sided raw-point certificate. Its one-side-deletion test provides
a direct false-first falsifier that is absent from scalar response maps alone.
"""


def _family_specs() -> str:
    return f"""# Three materially new executable architecture families

Exactly these three positive families are in scope. They share only neutral
bounded infrastructure and must start from raw sampled points.

## 1. `{POSITIVE_TOKENS[0]}` — recommended

Build a fresh multiscale normal-change field on the deterministic bounded raw
scan. Locate local maxima without using old tracer thresholds. From each stable
maximum, trace a finite tangent-aligned transition curve. Construct explicit,
disjoint left and right strips from raw points on opposite sides of the trace.

A certificate exists only when both independently measured strips satisfy the
same preregistered geometric schema. Deleting either strip and rebuilding must
destroy the certificate. The primitive is the trace plus its two raw support
strips—not a plane pair, fragment graph, component, mesh edge, or F1 bend birth.

Distinctive falsifier: one-side deletion kills the certificate. This makes
one-sided outlines, holes, scan boundaries, and residuals fail first.

## 2. `{POSITIVE_TOKENS[1]}`

Recompute local normals and normal covariance/tensor statistics across a small,
fixed scale lattice on the bounded raw scan. Retain a point-native crease
singular field only where tensor anisotropy/orientation behavior is stable across
scales and deterministic perturbations. Extract finite connected field traces
solely for preflight reporting.

No tensor-voting threshold, hysteresis, trace score, or neighborhood recipe from
the current F1 repository may be inherited. No mesh is formed.

Distinctive falsifier: the singular field must survive independent rebuild and
bounded point-order/permutation tests while disappearing under matched local
normal-direction scrambling.

## 3. `{POSITIVE_TOKENS[2]}`

Fit robust local polynomial osculating jets directly to bounded raw-point
neighborhoods under a fresh fixed degree/scale schema. Estimate curvature and
directional derivatives, identify ridge conditions, and trace finite curves
through locally coherent ridge points.

No global implicit surface, Poisson solve, mesh, old ridge tracer, plane/fragment
graph, training, or learned selection is allowed.

Distinctive falsifier: ridge traces must be stable to point ordering and a fixed
bounded resampling perturbation, while failing a preregistered local residual or
derivative-consistency test on unsupported regions.

## Shared semantic boundary

Every output is a representation/preflight primitive. None is an accepted bend,
owned bend, prediction, accuracy result, expected count, or runtime annotation.
"""


def _decision_matrix() -> str:
    return f"""# Exact architecture decision matrix

Return exactly one token and no compound selection.

| Token | Evidence advantage | Primary risk | Required preflight emphasis |
| --- | --- | --- | --- |
| `{POSITIVE_TOKENS[0]}` | Explicit two-sided raw support and deletion-causal certificate; strongest direct rejection of outlines/boundaries | Strip construction can silently encode scale or side bias | Independent rebuild, permutation, bilateral support, and delete-left/delete-right kill tests |
| `{POSITIVE_TOKENS[1]}` | Fully point-native multiscale field; robust literature basis for noise and feature scales | Scalar/tensor salience can overrespond to holes, borders, or texture | Scale stability, matched scrambling negative, boundary accounting, deterministic field topology |
| `{POSITIVE_TOKENS[2]}` | Mathematically distinct differential geometry representation with curvature/ridge semantics | Higher derivatives are noise-sensitive and may be resource-heavy | Residual conditioning, perturbation stability, derivative consistency, hard RSS/time caps |
| `{STOP_TOKEN}` | Correct when the attached evidence cannot support one bounded representation preflight | No architecture execution follows | State the missing evidence and stop |

The recommended token is `{POSITIVE_TOKENS[0]}` because its certificate embeds
an independently testable physical two-side transition premise and a sharp
one-side deletion falsifier. Recommendation is not authorization: Pro must choose.

Not valid responses: selecting multiple families; inventing a fourth choice;
requesting a CAD/reference, LPF, mesh, Poisson/solver, heatmap-gated, learned, or
existing-tracer lane; or authorizing an execution sequence beyond one preflight.
"""


def _preflight_contract() -> str:
    return """# Bounded gold-free preflight contract Pro must prescribe

For a positive selection, prescribe exactly one preflight and nothing after it.
The response must make the following fields implementation-ready.

## Immutable envelope

- Input lane: raw PLY scan points only, using a declared finite case set no
  larger than the existing C001–C037 research scan envelope.
- Input identity: raw scan byte SHA-256 plus deterministic sampler/config/schema
  hashes; no case-specific profile routing.
- Bound: at most 200,000 finite sampled points per scan.
- Execution: deterministic CPU, 1–8 fresh isolated processes, bounded per-part
  parallelism only after single-job measurement.
- Per-job memory: hard fail before 16 GiB peak RSS.
- Payload: compact numeric/ID/statistic tables; `source_points_serialized=false`;
  forbidden-key and cache-schema validation.

## Information prohibited from construction or selection

Gold, truth, expected bend counts, manual labels, CAD/reference geometry, family
answers, previous F1 candidates, label-conditioned profiles, mesh products,
heatmap gates, learning, threshold fitting, ownership, prediction births,
solver/Poisson stages, designated replays, synthetic rescue, runtime/main, and
production are closed.

## Required false-first structure

Pro must define one immutable representation configuration; complete untruncated
primitive/certificate tables; deterministic row ordering; independent rebuild;
point-order permutation equivalence; one family-specific destructive negative;
explicit rejection codes and denominators; serialized-size/RSS/time accounting;
and a validator that fails closed on missing or forged evidence.

For paired wedge strips, delete-left and delete-right rebuilds must each destroy
the corresponding certificate. For normal tensors, matched local normal-direction
scrambling must destroy the singular response. For jets, the chosen unsupported-
region residual/derivative perturbation must destroy ridge validity.

The preflight may test representation coherence and falsifiability only. It may
not use expected counts or report precision, recall, accuracy, bend ownership, or
generalization.

## Sole terminal statuses

- `PREFLIGHT_VALID_NEXT_PROFESSOR_PACKET`: the one preflight completed and its
  independent validator accepted all preregistered criteria. This authorizes only
  preparation of the next professor packet—not gold or another execution.
- `INVALID_NEW_REPRESENTATION_PREFLIGHT_STOP`: any criterion, source binding,
  resource cap, or tamper check fails; stop the selected family execution.
"""


def _evidence_index() -> str:
    source_rows = "\n".join(
        f"- `{label}`: `{Path(binding['path']).relative_to(ROOT)}` — "
        f"`{binding['sha256']}`"
        for label, binding in SOURCE_BINDINGS.items()
    )
    repo_rows = "\n".join(
        f"- `{relative}` — `{digest}`"
        for relative, digest in REPOSITORY_BINDINGS.items()
    )
    return f"""# Evidence and integrity index

## Authoritative source bindings

{source_rows}

## Repository audit bindings

{repo_rows}

## Packet construction assertions

- Exactly 10 individual files; no archive and no hidden attachment.
- Files 01 and 02 are byte-for-byte copies of their hash-bound sources.
- File 00 hashes files 01 through 09 after construction.
- The decision set has exactly three positive tokens and one fail-closed token.
- This packet contains no scan payload, credentials, `.env`, secret, gold label,
  CAD/reference model, mesh, executable detector, or runtime patch.
- The packet authorizes a decision response only; it executes no architecture.

An external machine-readable integrity audit is written beside the research
outputs rather than inside this directory so the upload set remains exactly ten
files. Recompute SHA-256 locally before upload and upload the files individually.
"""


def _prompt(hashes: Mapping[str, str]) -> str:
    attachment_hashes = "\n".join(
        f"- `{name}` — `{digest}`" for name, digest in sorted(hashes.items())
    )
    choices = "\n".join(f"- `{token}`" for token in (*POSITIVE_TOKENS, STOP_TOKEN))
    return f"""# GPT-5.6 Pro extended-thinking — fresh F1 architecture selection

Treat this as a fresh architecture-selection review. Read all nine evidence
attachments completely and verify their hashes before deciding. Prior chat
authorization does not carry into this packet.

The frozen local-plane-fragment family has a formal empirical STOP. A 37-case,
gold-free ordinary intervention added 161 rungs, reached 122 of 4,788 frozen
targets, advanced 0 side gates, and produced 0 novel attributable certificates.
Do not rescue, rename, tune, replay, or inherit that family.

## Your decision

Choose exactly one token:

{choices}

The first token is recommended, but the evidence must control your decision.
Do not select multiple families or synthesize a fourth choice.

If you choose one positive token, prescribe exactly one bounded, gold-free representation preflight
conforming to `08_BOUNDED_GOLD_FREE_PREFLIGHT_CONTRACT.md`.
Specify immutable inputs/config/schema, precise primitive and certificate tables,
false-first gates, the family-specific destructive negative, deterministic
rebuild/permutation tests, hard CPU/RSS/serialization caps, independent validator,
tamper tests, sole success/STOP statuses, and the exact evidence to include in
the next packet. Make it sufficiently concrete for one implementation pass.

Do not authorize execution beyond that single preflight. Do not open gold/truth,
expected counts, manual labels, CAD/reference, LPF, mesh selectors/topology,
Poisson or another solver, current tuned tracers, heatmap acceptance, learning,
threshold fitting, ownership, prediction births, replays, synthetic rescue,
runtime/main promotion, production, or a benchmark. A valid preflight result
authorizes only a new professor packet.

If the evidence is insufficient to prescribe exactly one such preflight, return
`{STOP_TOKEN}` and state the missing evidence. Do not substitute a broad research
agenda.

## Attachment SHA-256

{attachment_hashes}
"""


def _validate_package_contract(output_dir: Path) -> dict[str, Any]:
    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    if len(files) != MAX_FILES:
        raise ValueError(f"package must contain exactly {MAX_FILES} files")
    expected_names = {
        "00_GPT_PRO_PROMPT.md",
        "01_FORMAL_LPF_FAMILY_STOP.md",
        "02_ORDINARY_COMPACT_EVIDENCE.json",
        "03_ENTERPRISE_CONSTRAINTS.md",
        "04_REPOSITORY_CAPABILITY_BOUNDARY.md",
        "05_PRIMARY_SOURCE_RESEARCH.md",
        "06_MATERIALLY_NEW_FAMILY_SPECS.md",
        "07_EXACT_DECISION_MATRIX.md",
        "08_BOUNDED_GOLD_FREE_PREFLIGHT_CONTRACT.md",
        "09_EVIDENCE_INTEGRITY_INDEX.md",
    }
    names = {path.name for path in files}
    if names != expected_names:
        raise ValueError(f"unexpected package filenames: {sorted(names)}")
    if any(path.suffix.lower() in {".zip", ".tar", ".gz", ".env"} for path in files):
        raise ValueError("archive or secret-like file found in professor packet")

    prompt = (output_dir / "00_GPT_PRO_PROMPT.md").read_text(encoding="utf-8")
    matrix = (output_dir / "07_EXACT_DECISION_MATRIX.md").read_text(encoding="utf-8")
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in files
        if path.suffix in {".md", ".json"}
    )
    for token in (*POSITIVE_TOKENS, STOP_TOKEN):
        if token not in prompt or token not in matrix:
            raise ValueError(f"decision token missing from prompt/matrix: {token}")
    for doi in (
        "10.1145/777792.777840",
        "10.1111/1467-8659.00675",
        "10.1016/j.cad.2006.12.005",
        "10.1016/j.gmod.2012.04.008",
        "10.1016/j.cagd.2004.09.004",
        "10.1109/3DIMPVT.2012.12",
        "10.1145/1015706.1015768",
    ):
        if f"https://doi.org/{doi}" not in combined:
            raise ValueError(f"primary DOI missing: {doi}")
    for required in (
        "label-conditioned",
        "deterministic canonical sampling mechanism capped at 200,000 finite",
        "source_points_serialized=false",
        "one-side deletion",
        "exactly one bounded, gold-free representation preflight",
        "25856776f6048501c5e364cdad3fd33d70f5c89c1fb8c4e05dfbb5c46a4125b6",
        "2586bc09b7328900d5140857ece7553fd0188fddd29f2c462e01195aac13bfb8",
    ):
        if required not in combined:
            raise ValueError(f"required contract statement missing: {required}")
    hashes = {path.name: _sha_file(path) for path in files}
    return {
        "schema_version": 1,
        "kind": "f1_fresh_architecture_selection_package_integrity_v1",
        "valid": True,
        "packet_directory": str(output_dir.resolve()),
        "individual_file_count": len(files),
        "no_archive": True,
        "decision_only": True,
        "architecture_executed": False,
        "gold_open_authorized": False,
        "runtime_integration_authorized": False,
        "positive_tokens": list(POSITIVE_TOKENS),
        "fail_closed_token": STOP_TOKEN,
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": hashes[path.name]}
            for path in files
        ],
    }


def build_package(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR, audit_path: Path | None = DEFAULT_AUDIT_PATH
) -> dict[str, Any]:
    _validate_bound_sources()
    compact_source = Path(SOURCE_BINDINGS["ordinary_compact_evidence"]["path"])
    _validate_compact_evidence(compact_source)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    shutil.copyfile(
        SOURCE_BINDINGS["formal_lpf_stop"]["path"],
        output_dir / "01_FORMAL_LPF_FAMILY_STOP.md",
    )
    shutil.copyfile(
        compact_source,
        output_dir / "02_ORDINARY_COMPACT_EVIDENCE.json",
    )
    _write(output_dir / "03_ENTERPRISE_CONSTRAINTS.md", _enterprise_memo())
    _write(output_dir / "04_REPOSITORY_CAPABILITY_BOUNDARY.md", _repository_memo())
    _write(output_dir / "05_PRIMARY_SOURCE_RESEARCH.md", _research_memo())
    _write(output_dir / "06_MATERIALLY_NEW_FAMILY_SPECS.md", _family_specs())
    _write(output_dir / "07_EXACT_DECISION_MATRIX.md", _decision_matrix())
    _write(
        output_dir / "08_BOUNDED_GOLD_FREE_PREFLIGHT_CONTRACT.md",
        _preflight_contract(),
    )
    _write(output_dir / "09_EVIDENCE_INTEGRITY_INDEX.md", _evidence_index())

    evidence_hashes = {
        path.name: _sha_file(path)
        for path in sorted(output_dir.iterdir())
        if path.name != "00_GPT_PRO_PROMPT.md"
    }
    _write(output_dir / "00_GPT_PRO_PROMPT.md", _prompt(evidence_hashes))
    result = _validate_package_contract(output_dir)

    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT_PATH)
    args = parser.parse_args()
    print(
        json.dumps(
            build_package(output_dir=args.output_dir, audit_path=args.audit_path),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
