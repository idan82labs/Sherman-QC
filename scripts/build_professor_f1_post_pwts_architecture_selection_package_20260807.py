#!/usr/bin/env python3
"""Build the fresh post-PWTS architecture-selection packet.

The packet carries the exact nine-file PWTS failure capsule forward and asks
GPT-5.6 Pro to select one of two materially new, research-grounded families.
It authorizes architecture selection only: no geometry execution or gold access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_professor_f1_pwts_stop_decision_package_20260807 as stop_packet


DEFAULT_OUTPUT_DIR = (
    ROOT
    / "output/professor_package"
    / "Sherman_QC_F1_Post_PWTS_Architecture_Selection_20260807_v1"
)
DEFAULT_AUDIT_PATH = (
    stop_packet.PREFLIGHT_ROOT
    / "post-pwts-architecture-selection-package-v1"
    / "package_integrity_audit.json"
)
CLASSIFICATION_RESPONSE_PATH = (
    ROOT
    / "docs/research/PROFESSOR_F1_PWTS_STOP_CLASSIFICATION_RESPONSE_20260807.txt"
)
CLASSIFICATION_RESPONSE_SHA256 = (
    "ffd8c2a9259e7fcb4be96c966d771d4c8e2b73afa653cab5d7ea83bef8e980cf"
)
MAX_FILES = 10

INTEGRAL_TOKEN = "SELECT_CUBICAL_INTEGRAL_INVARIANT_PERSISTENCE_V1"
TENSOR_TOKEN = "SELECT_CELL_TENSOR_VOTING_SINGULAR_COMPLEX_V1"
JET_TOKEN = "SELECT_OSCULATING_JET_RIDGE_COMPLEX_V1"
STOP_TOKEN = "NO_POST_PWTS_ARCHITECTURE_SELECTED_EVIDENCE_INSUFFICIENT_STOP"
POSITIVE_TOKENS = (INTEGRAL_TOKEN, TENSOR_TOKEN, JET_TOKEN)
DECISION_TOKENS = (*POSITIVE_TOKENS, STOP_TOKEN)
LOCK_PACKET_TOKEN = "BUILD_SELECTED_ARCHITECTURE_PREFLIGHT_LOCK_PACKET_ONLY"
FORMAL_STOP_TOKEN = "FORMAL_POST_PWTS_ARCHITECTURE_STOP_NO_EXECUTION"
MISSING_EVIDENCE_TOKEN = "PROVIDE_ONE_NONEXECUTION_MISSING_EVIDENCE_ITEM_ONLY"
NEXT_STEP_TOKENS = (LOCK_PACKET_TOKEN, FORMAL_STOP_TOKEN, MISSING_EVIDENCE_TOKEN)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_classification_response() -> str:
    if not CLASSIFICATION_RESPONSE_PATH.is_file():
        raise FileNotFoundError(CLASSIFICATION_RESPONSE_PATH)
    observed = _sha_file(CLASSIFICATION_RESPONSE_PATH)
    if observed != CLASSIFICATION_RESPONSE_SHA256:
        raise ValueError(
            "preserved PWTS classification response differs from expected SHA-256"
        )
    text = CLASSIFICATION_RESPONSE_PATH.read_text(encoding="utf-8")
    required = (
        "CLASSIFICATION: PWTS_V1_PROTOCOL_INVALID_REPRESENTATION_UNRESOLVED",
        "NEXT_STEP: BUILD_FRESH_POST_PWTS_ARCHITECTURE_SELECTION_PACKET_ONLY",
        "PWTS and the already-stopped LPF family must not re-enter as candidates",
        "static worst-case primitive/table cardinality and evidence-volume analysis",
        "It carries no authority to execute geometry",
    )
    for value in required:
        if value not in text:
            raise ValueError(f"classification response lacks required ruling: {value}")
    return text


def _source_lock_stats() -> dict[str, Any]:
    _, rows = stop_packet._validate_source_lock()
    stats = {
        "case_count": len(rows),
        "finite_coordinate_count": sum(
            int(row["finite_coordinate_count"]) for row in rows.values()
        ),
        "exact_duplicate_count": sum(
            int(row["exact_duplicate_count"]) for row in rows.values()
        ),
        "raw_byte_count": sum(int(row["raw_byte_count"]) for row in rows.values()),
        "ply_formats": sorted({str(row["ply_format"]) for row in rows.values()}),
    }
    expected = {
        "case_count": 37,
        "finite_coordinate_count": 13_991_575,
        "exact_duplicate_count": 448_084,
        "raw_byte_count": 699_881_650,
        "ply_formats": ["ascii", "binary_little_endian"],
    }
    if stats != expected:
        raise ValueError("source-lock nonsemantic statistics mismatch")
    return stats


def _prompt(
    evidence_hashes: Mapping[str, str],
    classification_text: str,
    source_stats: Mapping[str, Any],
) -> str:
    attachments = "\n".join(
        f"- `{name}` — `{digest}`" for name, digest in sorted(evidence_hashes.items())
    )
    decisions = "\n".join(f"- `{token}`" for token in DECISION_TOKENS)
    next_steps = "\n".join(f"- `{token}`" for token in NEXT_STEP_TOKENS)
    return f"""# GPT-5.6 Pro extended-thinking — post-PWTS architecture selection

Treat this as a fresh, non-execution architecture-selection consultation. Read
all nine attachments completely, independently recompute their SHA-256 values,
and verify the cross-bindings before choosing. Prior-chat execution authority
does not carry into this packet.

The immediately preceding Pro ruling is preserved locally at SHA-256
`{CLASSIFICATION_RESPONSE_SHA256}` and began with exactly:

`CLASSIFICATION: PWTS_V1_PROTOCOL_INVALID_REPRESENTATION_UNRESOLVED`

`NEXT_STEP: BUILD_FRESH_POST_PWTS_ARCHITECTURE_SELECTION_PACKET_ONLY`

That ruling permits this packet and one architecture-selection response only.
It does not permit geometry execution, even if you select a positive family.

## Verbatim preceding Pro ruling

The byte range between the markers is exactly the preserved file whose SHA-256
is `{CLASSIFICATION_RESPONSE_SHA256}`. The trailing newline before the end
marker is part of the bound response.

-----BEGIN VERBATIM PWTS STOP CLASSIFICATION RESPONSE-----
{classification_text}-----END VERBATIM PWTS STOP CLASSIFICATION RESPONSE-----

## Integrity-bound PWTS failure capsule

The nine attachments are byte-identical copies of the prior classification
packet. External-file hashes, internal canonical object hashes, and references
must not be conflated:

| Artifact | External SHA-256 | Internal/content binding | Cross-reference |
| --- | --- | --- | --- |
| Prior PWTS ruling | `299e8d7f0a85c522ca4454587617d1f5df321c21d458bbda7cf7e2160d41f174` | none | locks the V1 configuration/schema and terminal semantics |
| Source manifest | `0d7d02698fdfc873ca56ff2248d7df9876a3e00ef1662bb73caab2d68f5b0bc4` | `a87b10e9b14119a543d5d114d84579007a1272616b29c317f505812c4f4667fd` | execution lock cites the internal binding |
| Configuration | `5d512e51911cff6ec4b44ce9bb17eead5852eb3647b15aaf10fd1dea490efd49` | same canonical hash | execution lock binds it |
| Schema | `f3a83fee7bc5c516041b585361e96376dec846da557e5f8530943be00dc87426` | same canonical hash | execution lock binds it |
| Execution lock | `6d0305daf64d5cf12268e20e68624f3de1287495e99ede6d900cb6e5ff19023a` | `2052b1f7c08e975049947368a97fbf5253344ce38c5c640c31d5a882c1054535` | checkpoint cites the internal binding |
| Orchestration STOP | `5493c57110d4158290cb707286fd283fb6e3568d466cdbab580ae093bd7183cf` | none | checkpoint cites this external hash; resource row cites stderr |
| Terminal checkpoint | `317291b705a9dc34e0c6c023900742e5a0b5d515e9f0878287b289349aa5027f` | `49572796e7ab3763711335b0d03c10191d0f9fc7375e150a190ce3a88d251ca0` | binds execution lock and orchestration STOP |
| C001 stderr | `7e874801f083d6dbc681abb02eec73661d86de0314efa3e4bbc71700413baf8d` | none | orchestration resource row cites it |
| Run-v1 launch failure | `213265c6bd42c383a8c777748fc5736b5d224e8f78d3984c18d96efe8ed20743` | none | separate zero-progress platform event |

## Exact frozen-fact ledger

- Preflight: `PWTS_REPRESENTATION_PREFLIGHT_V1`.
- Only C001/BASE_A attempted semantic construction; process 1 of 148.
- Locked seed-row cap: 1,024.
- Only supported count statement: `seed_count >= 1025`; exact count withheld.
- Failure: `StructuralCapError` before serialization.
- Terminal protocol status: `INVALID_NEW_REPRESENTATION_PREFLIGHT_STOP`.
- Zero semantic payload, zero scratch bytes, zero serialized bytes.
- 55.21574 CPU seconds; 35.56373308401089 wall seconds.
- Process-tree peak RSS: 844,136,448 bytes; no resource cap tripped.
- BASE_B, order permutation, independent validator, traces, strips,
  certificates, and bilateral-deletion arms did not complete.
- Gold/truth, expected counts, manual labels, CAD/reference, renders,
  benchmarks, runtime/main, and production were not consumed or authorized.
- Run-v1 was a separate pre-exec Darwin `RLIMIT_AS` launch failure with no
  semantic process and is not a preflight result.
- Run-v2 used verified fail-closed 10 ms process-group RSS supervision. This
  was not an in-kernel cgroup; a sub-sampling-interval transient overshoot is
  theoretically possible.
- The source lock contains exactly {source_stats['case_count']} scans C001–C037,
  {source_stats['finite_coordinate_count']:,} finite XYZ records,
  {source_stats['exact_duplicate_count']:,} exact duplicates, and
  {source_stats['raw_byte_count']:,} raw bytes across ASCII and binary-little-
  endian PLY. It performed no sampling or semantics and serialized no source
  points. These are 37 scans—not evidence of 37 parts or families.
- Enterprise release, runtime/main, production, gold, and benchmarks remain
  closed. This packet contains no release-performance metric.

## Claim boundary

| Class | Allowed statement |
| --- | --- |
| Observed | The frozen seed definition emitted at least 1,025 rows on C001 and therefore violated its frozen 1,024-row evidence envelope before serialization. |
| Inference | PWTS-V1's seed definition and evidence envelope are incompatible on C001; the protocol STOP is valid. |
| Unknown | Exact seed count, spatial pattern, overflow cause, physical-bend meaning, downstream trace/strip/certificate selectivity, deterministic equivalence, validator reproducibility, and bend accuracy. |

Do not infer any unknown. The architecture choice below must not use the
withheld seed count, a render, or a gold-relative outcome.

## Closed materially-new candidate set

PWTS and the stopped local-plane-fragment family are excluded. A threshold,
cap, truncation, tangent-tracing, strip, certificate, plane/fragment, or naming
change to either family is not a candidate. Pointwise DoN maxima, mesh/CAD,
heatmap, learned, and Poisson/global-reconstruction candidates are also closed.
Exactly three positive, finite-by-construction cell-complex families remain.

Every numeric envelope below is labeled
`PROPOSED_ENGINEERING_CONSTANT_NOT_LITERATURE_ESTABLISHED`. Selection does not
freeze or authorize any of it.

### Shared finite partition envelope

- `N <= 200,000` finite sampled XYZ points.
- Deterministic adaptive cubical partition with `L <= 32,768` leaves and at
  most `2L-1 = 65,535` partition-node rows.
- Every sampled point contributes to exactly one leaf aggregate. Reaching `L`
  leaves cells coarser; it drops no point, leaf, primitive, or rejection row.
- Candidate analysis uses at most `S=3` cell scales. Tensor candidate cell
  degree is at most `K_cell=32`.
- Exactly one MiB is reserved for fixed metadata, hashes, and logs.
- No coordinates, point arrays, neighborhood arrays, or point subsets appear
  in evidence. The independent validator rereads the raw input.
- Static work-buffer proof must be no more than 12 GiB before launch; measured
  process-tree RSS must remain below the enterprise 16 GiB ceiling.

### Candidate A — cubical integral-invariant persistence

Decision token: `{INTEGRAL_TOKEN}`

Primitive: a complete deterministic cubical partition; ball-intersection volume
and covariance-moment integral invariants at every leaf and scale; and a finite
filtered cubical persistence complex over the complete cell field. There are no
planes, fragments, pair axes, maxima seeds, tangent traces, strips, certificates,
or accepted bends. Persistence is a finite topology computation, not a surface
reconstruction or Poisson/mesh lane.

Primary mechanism support—not constants or Sherman accuracy:

- Pottmann et al., robust multiscale integral invariants, DOI
  `10.1016/j.cagd.2008.01.002`.
- Pottmann et al., principal curvatures from ball-integral moments, DOI
  `10.1016/j.cagd.2007.07.004`.
- Zomorodian and Carlsson, finite filtered-complex persistent homology, DOI
  `10.1007/s00454-004-1146-y`.

Engineering assumptions requiring a later lock packet: partition origin/split
priority, physical cell/ball scales, moment normalization, filtration scalar,
adjacency, homology dimension, coefficient field, persistence ordering/ties,
component semantics, and nonvacuity/selectivity gates.

Direct causal falsifier: replace every leaf second-moment tensor by its isotropic
same-trace tensor while preserving occupancy, trace, adjacency, and all scalar
mass. Rebuild the full field and persistence complex. Every primitive claimed
to arise from anisotropy must disappear. A survivor, missing negative, or zero
falsifier denominator is terminal STOP.

Failure modes: cell quantization, coarse leaves hiding narrow bends, holes and
scan boundaries creating persistent topology, density/mass dominating geometry,
filtration ties, cubical adjacency artifacts, global persistence cost, empty
output, or near-saturated output unrelated to physical bends.

| Complete integral-invariant evidence table | Maximum rows | Bytes/row | Maximum bytes |
| --- | ---: | ---: | ---: |
| partition node | 65,535 | 64 | 4,194,240 |
| multiscale field | `S*L = 98,304` | 80 | 7,864,320 |
| persistence primitive | 98,304 | 64 | 6,291,456 |
| rejection/negative | 98,304 | 48 | 4,718,592 |
| metadata/log reserve | 1 | 1,048,576 | 1,048,576 |
| **Complete maximum** | **360,448** | — | **24,117,184** |

Static hard evidence cap proposed for later adjudication: 32 MiB per complete
arm. Any omitted row, hidden tail, variable-width payload, or byte excess STOPs.

### Candidate B — cell tensor-voting singular complex

Decision token: `{TENSOR_TOKEN}`

Primitive: complete multiscale sign-invariant cell voting tensors, every directed
cell vote, every tensor eigenstructure/rejection row, and the induced finite
cell singular/persistence complex. It is not a pointwise maximum list, traced
curve, or PWTS/LPF derivative.

Primary mechanism support—not constants or Sherman accuracy:

- Park, Lee, and Lee, multiscale tensor voting/eigenstructure on unstructured
  point clouds, DOI `10.1016/j.gmod.2012.04.008`.
- Mitra and Nguyen, normal estimation sensitivity to neighborhood size,
  sampling density, curvature, and noise, DOI `10.1145/777792.777840`.

Engineering assumptions requiring a later lock packet: how cell tensors arise
from point aggregates, vote kernel and sign invariance, cell scale lattice,
neighbor degree, boundary/coverage gates, saliency/eigenvalue semantics,
filtration, singular-complex adjacency, and nonvacuity/selectivity gates.

Direct causal falsifier: replace every complete vote tensor by its isotropic
same-trace tensor while retaining occupancy, vote graph, scale, and trace. Rebuild
the full singular complex. All anisotropy-derived primitives must disappear.

Failure modes: holes/boundaries, density gradients, smooth cylinders, cell
coarsening, eigenvalue degeneracy, repeated bias across scales, vote-volume
growth, empty field, or saturated singular field unrelated to bends.

| Complete tensor evidence table | Maximum rows | Bytes/row | Maximum bytes |
| --- | ---: | ---: | ---: |
| partition node | 65,535 | 64 | 4,194,240 |
| multiscale tensor | `S*L = 98,304` | 96 | 9,437,184 |
| directed cell vote | `S*L*K_cell = 3,145,728` | 32 | 100,663,296 |
| singular/persistence primitive | 98,304 | 48 | 4,718,592 |
| rejection/negative | 98,304 | 48 | 4,718,592 |
| metadata/log reserve | 1 | 1,048,576 | 1,048,576 |
| **Complete maximum** | **3,506,176** | — | **124,780,480** |

Static hard evidence cap proposed for later adjudication: 128 MiB per complete
arm. Every vote is retained; no neighbor tail or rejected cell may be omitted.

### Candidate C — osculating-jet ridge complex

Decision token: `{JET_TOKEN}`

Primitive: one deterministic local polynomial height jet per occupied leaf,
complete fit conditioning/residual/differential rows, and a finite ridge/Morse
cell complex. It is not a plane-pair, pointwise seed list, or PWTS tangent trace.
No global implicit surface or mesh is constructed.

Primary mechanism support—not constants or Sherman accuracy:

- Cazals and Pouget, osculating jets for normals, curvatures, and curvature
  extrema from sampled points, DOI `10.1016/j.cagd.2004.09.004`.
- Cazals et al., classical principal-curvature ridge definition, DOI
  `10.1016/j.cagd.2006.04.002`.
- Jiao and Zha, parameterization/solver effects on differential fitting, DOI
  `10.1016/j.cagd.2009.02.006`.

Mandatory semantic ruling: a classical ridge is a principal-curvature extremum
along its principal direction, but an ideal constant-radius sheet-metal bend is
locally cylindrical and may form a curvature plateau with no unique strict
ridge centerline. Pro must decide whether a cell ridge/Morse complex has the
right physical semantics. Do not select this family by assuming the mismatch
away.

Engineering assumptions requiring a later lock packet: partition support,
local frame/tie rules, degree, robust estimator/loss, scale, kernel, solver,
iteration/convergence, conditioning/folding gates, signed versus sign-invariant
curvature, plateau policy, umbilics, ridge/Morse predicate, adjacency, and
nonvacuity/selectivity gates.

Direct causal falsifier: replace every fitted jet by its affine-only component
while preserving occupancy, first-order frame, adjacency, and fit domain. Rebuild
the ridge/Morse complex. Every curvature-derived primitive must disappear.

Failure modes: cylindrical plateaus, violation of a single smooth height-jet
model at sharp transitions, noise-amplified higher derivatives, holes/boundaries,
ill-conditioned Vandermonde systems, robust-fit non-determinism, umbilics,
cell-scale mismatch, empty output, or dense spurious ridge complexes.

| Complete jet evidence table | Maximum rows | Bytes/row | Maximum bytes |
| --- | ---: | ---: | ---: |
| partition node | 65,535 | 64 | 4,194,240 |
| occupied-leaf jet | 32,768 | 256 | 8,388,608 |
| neighborhood digest/count | 32,768 | 64 | 2,097,152 |
| ridge/Morse primitive | 32,768 | 64 | 2,097,152 |
| rejection/negative | 32,768 | 48 | 1,572,864 |
| metadata/log reserve | 1 | 1,048,576 | 1,048,576 |
| **Complete maximum** | **196,608** | — | **19,398,592** |

Static hard evidence cap proposed for later adjudication: 32 MiB per complete
arm. The apparent compactness does not resolve the scientific plateau risk or
robust-fit scratch/time risk.

## Candidate comparison

| Candidate | Distinct evidence advantage | Dominant scientific risk | Complete base bound |
| --- | --- | --- | ---: |
| Integral-invariant persistence | Robust multiscale moment field and finite persistence complex | Persistent topology/anisotropy may encode holes, boundaries, or density rather than bends | 24,117,184 B |
| Cell tensor singular complex | Direct multiscale orientation/singularity mechanism on unstructured points | Tensor singularity may saturate cylinders, boundaries, or repeated density bias | 124,780,480 B |
| Jet ridge complex | Differential curvature and conditioning semantics with the smallest table bound | Constant-radius bends may be plateaus, not classical ridges; derivatives are noise-sensitive | 19,398,592 B |

All three are finite by construction. None is claimed to detect bends.
Literature establishes mechanisms only; it does not select partition sizes,
thresholds, falsifier gates, evidence caps, or Sherman accuracy/generalization.

## Immutable contamination and safety boundary

- Input to any later specification may be raw PLY file bytes and finite XYZ only;
  filenames/paths/family/part identity may not influence semantics.
- At most 200,000 deterministic sampled points; CPU-only; single-job RSS measured
  before any bounded parallelism; fail below 16 GiB.
- No source normals/colors, gold/truth/expected counts, manual labels,
  CAD/reference geometry, renders, previous F1 candidates, family routing,
  label-conditioned profiles, threshold fitting, learning, mesh/topology,
  Poisson/global solvers, heatmap gates, ownership, prediction births, replay,
  synthetic rescue, runtime/main, production, benchmark, or accuracy reporting.
- No PWTS/LPF concepts may be renamed or inherited. Neutral byte hashing,
  canonical sampling, isolated process/resource monitoring, compact ID/statistic
  tables, and forbidden-key validation are the only reusable infrastructure.
- Full evidence must be complete and canonical. No top-k, reservoir, tail omission,
  post-hoc cap, or absent rejection denominator is allowed.
- `EXECUTION_AUTHORITY=NONE` for this packet and its response.

## Required answer

Return exactly two lines first:

`ARCHITECTURE: <exactly one architecture token>`

{decisions}

`NEXT_STEP: <exactly one next-step token>`

{next_steps}

Then give the shortest rigorous rationale. A positive architecture is logically
consistent only with `{LOCK_PACKET_TOKEN}`. The STOP architecture token is
consistent with `{FORMAL_STOP_TOKEN}` or `{MISSING_EVIDENCE_TOKEN}`.

If you choose a positive family, state exactly what non-execution evidence and
unresolved engineering decisions must enter a separate preflight-lock packet.
Do not prescribe or authorize an executable configuration in this response.
Do not combine the families, invent a fourth, revive PWTS/LPF, or treat any
`PROPOSED_ENGINEERING_CONSTANT_NOT_LITERATURE_ESTABLISHED` value as selected.

If no candidate has adequate non-execution support, return `{STOP_TOKEN}`
with `{FORMAL_STOP_TOKEN}`, or return it with `{MISSING_EVIDENCE_TOKEN}` and name
exactly one minimum non-execution item. This packet may culminate only in an
architecture selection; selection authorizes another Pro packet, not geometry execution.

## Attachment SHA-256

{attachments}
"""


def _validate_packet(output_dir: Path) -> dict[str, Any]:
    entries = sorted(output_dir.iterdir())
    if any(
        path.name.startswith(".") or path.is_symlink() or not path.is_file()
        for path in entries
    ):
        raise ValueError("packet may contain only regular non-hidden files")
    files = entries
    if len(files) != MAX_FILES:
        raise ValueError(f"expected exactly {MAX_FILES} individual files")
    expected_names = {
        "00_GPT_PRO_POST_PWTS_ARCHITECTURE_SELECTION.md",
        *stop_packet.SOURCE_BINDINGS.keys(),
    }
    if {path.name for path in files} != expected_names:
        raise ValueError("post-PWTS packet filename set mismatch")
    if any(path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"} for path in files):
        raise ValueError("archives are forbidden")

    prompt = (
        output_dir / "00_GPT_PRO_POST_PWTS_ARCHITECTURE_SELECTION.md"
    ).read_text(encoding="utf-8")
    for token in DECISION_TOKENS:
        if token not in prompt:
            raise ValueError(f"missing decision token: {token}")
    for token in NEXT_STEP_TOKENS:
        if token not in prompt:
            raise ValueError(f"missing next-step token: {token}")
    if len(POSITIVE_TOKENS) != 3 or len(DECISION_TOKENS) != 4:
        raise ValueError("candidate set must contain three positive tokens plus STOP")
    if any("PWTS" in token or "LPF" in token for token in POSITIVE_TOKENS):
        raise ValueError("stopped-family name appears in a positive token")
    for forbidden in (
        "SELECT_PAIRED_WEDGE_TRANSITION_STRIP",
        "SELECT_LOCAL_PLANE_FRAGMENT",
    ):
        if forbidden in prompt:
            raise ValueError(f"forbidden execution/family token present: {forbidden}")
    begin_marker = "-----BEGIN VERBATIM PWTS STOP CLASSIFICATION RESPONSE-----\n"
    end_marker = "-----END VERBATIM PWTS STOP CLASSIFICATION RESPONSE-----"
    if prompt.count(begin_marker) != 1 or prompt.count(end_marker) != 1:
        raise ValueError("classification-response marker count mismatch")
    embedded = prompt.split(begin_marker, 1)[1].split(end_marker, 1)[0]
    classification_text = CLASSIFICATION_RESPONSE_PATH.read_text(encoding="utf-8")
    if embedded != classification_text:
        raise ValueError("embedded classification response is not byte-exact")
    for required in (
        CLASSIFICATION_RESPONSE_SHA256,
        "PWTS_V1_PROTOCOL_INVALID_REPRESENTATION_UNRESOLVED",
        "BUILD_FRESH_POST_PWTS_ARCHITECTURE_SELECTION_PACKET_ONLY",
        "Observed",
        "Inference",
        "Unknown",
            "seed_count >= 1025",
            "13,991,575",
            "448,084",
            "699,881,650",
            "37 scans—not evidence of 37 parts or families",
        "PROPOSED_ENGINEERING_CONSTANT_NOT_LITERATURE_ESTABLISHED",
        "24,117,184",
        "124,780,480",
        "19,398,592",
        "2L-1 = 65,535",
        "S*L*K_cell = 3,145,728",
        "10.1016/j.cagd.2008.01.002",
        "10.1007/s00454-004-1146-y",
        "10.1016/j.gmod.2012.04.008",
        "10.1016/j.cagd.2004.09.004",
        "curvature plateau with no unique strict",
        "source normals/colors",
        "No top-k, reservoir, tail omission",
        "EXECUTION_AUTHORITY=NONE",
        "not geometry execution",
    ):
        if required not in prompt:
            raise ValueError(f"required architecture-selection evidence missing: {required}")

    hashes = {path.name: _sha_file(path) for path in files}
    return {
        "schema_version": 1,
        "package_id": "SHERMAN_QC_F1_POST_PWTS_ARCHITECTURE_SELECTION_20260807_V1",
        "individual_file_count": len(files),
        "no_archive": True,
        "architecture_selection_only": True,
        "geometry_execution_authorized": False,
        "gold_or_truth_consumed": False,
        "runtime_or_production_authorized": False,
        "execution_authority": "NONE",
        "classification_response_sha256": CLASSIFICATION_RESPONSE_SHA256,
        "decision_tokens": list(DECISION_TOKENS),
        "next_step_tokens": list(NEXT_STEP_TOKENS),
        "embedded_classification_response_byte_exact": True,
        "packet_sha256": hashes,
    }


def build_package(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    audit_path: Path | None = DEFAULT_AUDIT_PATH,
    verify_raw_sources: bool = False,
) -> dict[str, Any]:
    if verify_raw_sources:
        raise ValueError("architecture-selection builder may not read raw PLY files")
    stop_packet._validate_bound_sources()
    stop_validation = stop_packet._validate_stop_evidence(
        verify_raw_sources=False
    )
    classification_text = _validate_classification_response()
    source_stats = _source_lock_stats()

    output_dir.mkdir(parents=True, exist_ok=False)

    evidence_hashes: dict[str, str] = {}
    for name, binding in stop_packet.SOURCE_BINDINGS.items():
        destination = output_dir / name
        shutil.copyfile(Path(binding["path"]), destination)
        observed = _sha_file(destination)
        if observed != binding["sha256"]:
            raise ValueError(f"copied evidence SHA mismatch: {name}")
        evidence_hashes[name] = observed

    prompt_path = (
        output_dir / "00_GPT_PRO_POST_PWTS_ARCHITECTURE_SELECTION.md"
    )
    prompt_path.write_text(
        _prompt(evidence_hashes, classification_text, source_stats).rstrip() + "\n",
        encoding="utf-8",
    )
    result = _validate_packet(output_dir)
    result["source_validation"] = stop_validation
    result["source_lock_stats"] = source_stats

    if audit_path is not None:
        if audit_path.exists():
            raise FileExistsError(audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_bytes(_canonical_json_bytes(result) + b"\n")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT_PATH)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_package(
        output_dir=args.output_dir,
        audit_path=args.audit_path,
        verify_raw_sources=False,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
