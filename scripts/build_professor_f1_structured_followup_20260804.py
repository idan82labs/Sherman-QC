#!/usr/bin/env python3
"""Build the file-by-file GPT Pro follow-up for structured F1 inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "output/professor_package"
    / "Sherman_QC_F1_Structured_Inference_Followup_20260804"
)


FILES = {
    "01_prior_professor_response.json": ROOT
    / "output/professor_package/Sherman_QC_F1_Structured_Inference_Dropoff_20260804/04_professor_response.json",
    "02_candidate_pool_v2.json": ROOT
    / "output/research/f1-structured-union-20260804/45796000/candidate_pool_measured_v2/01_45796000.json",
    "03_typed_union_graph.json": ROOT
    / "output/research/f1-structured-union-20260804/45796000/exact_v1/typed_union_graph.json",
    "04_exact_cpsat_solution.json": ROOT
    / "output/research/f1-structured-union-20260804/45796000/exact_v1/cpsat_solution.json",
    "05_postlock_step_evaluation.json": ROOT
    / "output/research/f1-structured-union-20260804/45796000/exact_v1/postlock_step/line_evaluation.json",
    "06_structured_semantic_audit.png": ROOT
    / "output/research/f1-structured-union-20260804/45796000/baseline_v1/structured_semantic_audit.png",
    "07_candidate_pool_memory.txt": ROOT
    / "output/research/f1-structured-union-20260804/45796000/candidate_pool_measured_v2/build_time_l.txt",
    "08_solver_memory.txt": ROOT
    / "output/research/f1-structured-union-20260804/45796000/exact_v1/solver_time_l.txt",
    "09_10839000_memory.txt": ROOT
    / "output/enterprise_benchmark/10839000-no-render-memory-20260804-v2/time_l.txt",
    "10_10839000_determinism.json": ROOT
    / "output/enterprise_benchmark/10839000-no-render-memory-20260804-v2/determinism/determinism_summary.json",
    "11_blind_ply13_summary.json": ROOT
    / "output/enterprise_benchmark/blind-ply-bounded-20260804-v2/summary.json",
    "12_blind_cadply12_summary.json": ROOT
    / "output/enterprise_benchmark/blind-cad-ply-bounded-20260804-v2/result/summary.json",
    "13_manual_visual_audit.md": ROOT
    / "output/enterprise_benchmark/blind-ply-bounded-20260804-v2-review/MANUAL_VISUAL_AUDIT.md",
    "13b_enterprise_promotion_gates.json": ROOT
    / "output/migration/enterprise-promotion-gates-20260804-bounded.json",
    "14_f1_structured_union_graph.py": ROOT / "backend/f1_structured_union_graph.py",
    "15_solve_f1_structured_union_cpsat.py": ROOT
    / "scripts/solve_f1_structured_union_cpsat.py",
    "16_build_f1_scan_only_candidate_pool.py": ROOT
    / "scripts/build_f1_scan_only_candidate_pool.py",
    "17_test_f1_structured_union_cpsat.py": ROOT
    / "scripts/test_f1_structured_union_cpsat.py",
    "18_test_f1_structured_union_graph.py": ROOT
    / "backend/tests/test_f1_structured_union_graph.py",
    "19_test_candidate_pool_schema.py": ROOT
    / "tests/test_build_f1_scan_only_candidate_pool.py",
}


PROMPT = """# Sherman QC F1 structured inference — implementation follow-up

You are reviewing a research-only zero-shot bend detector. Continue directly
from `01_prior_professor_response.json`. Do not propose a new architecture from
scratch and do not treat this as an email. Inspect every uploaded file.

## What was implemented from your plan

- Union-of-lanes typed path graph; no global source router in this research path.
- Seven-label ontology and explicit non-counted corner objects.
- Bounded support schema: atom IDs/statistics, sparse atom adjacency, and compact
  plane table; no source points or part/truth/CAD metadata at inference.
- Deterministic CP-SAT with one-hot labels, relational chord/corner factors,
  exclusive compact-atom ownership, residual ownership, compatible continuation
  variables, endpoint degree <= 1, acyclic continuation-proposal validation,
  and connected bend count `sum(x)-sum(y)`.
- Exact sequential tie stages: primary integer energy, maximum non-residual
  owned support, maximum compatible continuations, then fixed-search canonical
  assignment. A primary-only no-good solve reports the next semantic energy.
- Pinned isolated OR-Tools process; runtime/main remains untouched.

## Locked results

- 45796000 pool: 1,070 candidates, 649 paths, 1,294 atoms, 9,959 atom edges,
  seven planes, zero serialized source points.
- 45796000 exact solve: OPTIMAL; eight counted bends, four non-counted corners;
  447 atoms assigned exclusively to non-residual paths; 847 residual atoms;
  integer margin 4,892,046; three byte-identical solver replays.
- Post-lock STEP scoring: 8 TP / 0 FP / 0 FN. This is a designed regression,
  not independent validation.
- Measured pool rebuild: 25.26 s, 913,408,000 bytes peak RSS, byte-identical.
- Solver: 0.42 s, 131,514,368 bytes peak RSS.
- Cold 10839000 no-render job: 10.19 s, 609,026,048 bytes peak RSS, three
  identical semantic replays. The old implementation exceeded 74.5 GiB.
- Fresh PLY-only blind slice: 13/13 jobs succeed; 0/13 accepted exact, 6/13
  ranges contain truth, 5/13 raw-F1 exact, 7/13 render-admissible exact, and
  0/13 manually visually acceptable after reviewing every contact sheet.
- Fresh CAD+PLY blind slice: 12/12 jobs succeed, 3/12 exact, strict blind
  contract rate 1.0. No runtime promotion is authorized.

## Known limitations that are deliberately not hidden

1. Paths with serialized atom incidence participate in exact ownership, but an
   independently generated path with no atom provenance can still be selected.
   Three of the eight 457 bend paths are in this category.
2. The graph accepts and solves continuation proposals but does not yet generate
   generic candidates from endpoint geometry/support.
3. Residual-birth column generation and learned potentials are not implemented.
4. The K(2,4) chord factor was designed with 457 visible and activates on none
   of the 37 Gold-27 development scans after the conservative restriction.
5. Existing runtime outputs still confuse outlines/free edges with bends and
   miss visible bends, as documented in `13_manual_visual_audit.md`.

## Required response

Give a code-level continuation, not a broad essay. Please answer these exact
questions in order:

1. Is the implemented exclusive-ownership formulation mathematically honest
   when zero-incidence paths remain admissible? Specify the smallest scan-only
   incidence completion that avoids inventing support or tuning on 457.
2. Give a deterministic continuation-proposal algorithm with explicit
   dimensionless residuals, engineering tolerances, canonical ordering, and a
   proof-friendly acyclic construction. It must work for fragmented true bends
   without joining distinct parallel bends.
3. Specify the first residual-birth column-generation slice we can implement
   before learned weights exist. Provide exact eligibility constraints and a
   defensible reduced-cost lower bound; do not use expected counts.
4. With only 27 labeled families and the fresh blind failures, define the
   minimum annotation contract and nested leave-one-family-out learning plan
   for class-specific ownership minima and low-capacity potentials. State what
   must remain engineering-fixed.
5. Audit `14_f1_structured_union_graph.py` and
   `15_solve_f1_structured_union_cpsat.py` for leakage, false exactness claims,
   invalid tie-breaking, missing constraints, or 457-specific logic. Cite
   functions/fields precisely and provide replacement pseudocode or patches.
6. Define the next two falsifiable ablations and their stop/go criteria. At
   least one must target outline/free-edge rejection and one must target
   missing-bend birth on family-held-out scans.

Do not authorize runtime promotion. Clearly separate OBSERVED, INFERENCE, and
UNKNOWN claims.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    prompt_path = PACKAGE / "00_GPT_PRO_PROMPT.md"
    prompt_path.write_text(PROMPT, encoding="utf-8")

    for destination_name, source in FILES.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, PACKAGE / destination_name)

    rows: list[dict[str, Any]] = []
    for path in sorted(PACKAGE.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        rows.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "submission_id": "sherman-qc-f1-structured-followup-20260804-v1",
        "submission_style": "gpt_pro_file_upload_not_email",
        "runtime_promotion_authorized": False,
        "file_count": len(rows),
        "files": rows,
    }
    (PACKAGE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"package": str(PACKAGE), "file_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
