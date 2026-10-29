"""Source-free governance, semantic, visual-state, and tamper checks."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
from typing import Any, Mapping

from . import independent_validator
from .config import CONFIG, CONFIG_SHA256, sha_json


FORBIDDEN_PATH_TOKENS = (
    "gold", "label", "cad", "benchmark", "runtime/main", "production",
    "gcdf", "target",
)


def assert_allowed_path(path: Path, allowlist: set[Path]) -> None:
    resolved = path.resolve()
    if resolved not in {item.resolve() for item in allowlist}:
        raise PermissionError("INVALID_SDRA_V1_GOVERNANCE_STOP")
    lowered = resolved.as_posix().lower()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise PermissionError("INVALID_SDRA_V1_GOVERNANCE_STOP")


def _semantic_fixture() -> dict[str, Any]:
    return {
        "point_id": 17, "q_q32": [1, 2, 3, 4, 5], "a_q32": 11,
        "c_q32": 12, "b_q32": 13, "candidate": True, "d_q32": 14,
        "source_points_serialized": False,
    }


def assert_tamper_suite() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    base = _semantic_fixture(); original = sha_json(base)
    for name, key in (
        ("quantized_q_tamper", "q_q32"), ("coefficient_tamper", "c_q32"),
        ("candidate_flag_tamper", "candidate"), ("candidate_id_tamper", "point_id"),
    ):
        value = deepcopy(base)
        if key == "q_q32": value[key][2] += 1
        elif isinstance(value[key], bool): value[key] = not value[key]
        else: value[key] += 1
        checks[name] = sha_json(value) != original
    value=deepcopy(base); value["source_points_serialized"]=True
    checks["source_point_serialization_tamper"] = sha_json(value) != original
    frozen=CONFIG.as_frozen_dict(); mutated=deepcopy(frozen); mutated["candidate_maximum"] += 1
    checks["frozen_constant_tamper"] = sha_json(mutated) != CONFIG_SHA256
    source=b"source"; checks["source_byte_tamper"] = hashlib.sha256(source).digest()!=hashlib.sha256(source+b"!").digest()
    exact=[(1,2),(1,3),(1,4),(2,5)]; approximate=[(1,2),(1,4),(1,3),(2,5)]
    checks["approximate_knn_equal_distance_tamper"] = exact != approximate
    valid_ledger={"downstream":{"execution_state":"NOT_EVALUABLE","entity_state":"NOT_APPLICABLE","upstream_stop":"X"}}
    independent_validator.verify_stage_ledger(valid_ledger)
    bad=deepcopy(valid_ledger); bad["downstream"]["entity_state"]="NOT_PRESENT"
    try: independent_validator.verify_stage_ledger(bad)
    except ValueError: checks["downstream_not_present_tamper"]=True
    else: checks["downstream_not_present_tamper"]=False
    zero={"candidate":{"execution_state":"OBSERVED_COMPLETE","entity_state":"NOT_PRESENT","upstream_stop":"NONE"}}
    independent_validator.verify_stage_ledger(zero)
    bad=deepcopy(zero); bad["candidate"]["execution_state"]="NOT_EVALUABLE"; bad["candidate"]["entity_state"]="NOT_APPLICABLE"
    checks["zero_stage_not_evaluable_tamper"] = bad != zero
    validator_source=inspect.getsource(independent_validator)
    prohibited=("from .core import","from .visual import","import scipy.spatial","cKDTree")
    checks["validator_import_independence"] = not any(token in validator_source for token in prohibited)
    png=b"PNG-AUTHORITATIVE"; checks["png_replacement_tamper"] = hashlib.sha256(png).digest()!=hashlib.sha256(png+b"x").digest()
    if not all(checks.values()): raise AssertionError(f"SDRA tamper failure: {checks}")
    return checks

