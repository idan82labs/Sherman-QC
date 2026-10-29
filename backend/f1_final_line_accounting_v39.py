"""Research-only V39 final-line accounting over V38.

V39 changes no decisions, NMS outcomes, or final-line content.  It exposes the
frozen fixture contract's second number as the total final line count while
retaining the rescue-only post-NMS count under an explicit field name.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from backend.f1_sharp_transition_route_certificate_v38 import select_unified_v38
from backend.f1_v34_artifact_scalar_normalizer import normalize_numpy_scalars


PROFILE = "V39_FINAL_LINE_ACCOUNTING_V1"
PRIMITIVE = "COUNT_ONLY_TOTAL_FINAL_LINE_ACCOUNTING_V1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def select_unified_v39(
    raw_source: dict[str, Any],
    raw_candidates: Sequence[dict[str, Any]],
    *,
    config_sha256: str,
    worker_count: int = 1,
) -> dict[str, Any]:
    base = select_unified_v38(
        raw_source,
        raw_candidates,
        config_sha256=config_sha256,
        worker_count=worker_count,
    )
    public = normalize_numpy_scalars({key: value for key, value in base.items() if key != "_ephemeral"})
    rescue_only = int(public["rescue_count"])
    final_count = len(public["final_lines"])
    public.update(
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v39_final_line_accounting_research_result",
            "profile": PROFILE,
            "primitive": PRIMITIVE,
            "v38_actions_retained": True,
            "v38_final_lines_retained": True,
            "count_only": True,
            "post_nms_rescue_only_count": rescue_only,
            "final_line_count": final_count,
            "rescue_count": final_count,
        }
    )
    cache_payload = {
        "profile": PROFILE,
        "source_sha256": public["source_sha256"],
        "config_sha256": config_sha256,
        "pre_nms_rescue_count": public["pre_nms_rescue_count"],
        "post_nms_rescue_only_count": rescue_only,
        "final_line_count": final_count,
        "decisions": public["decisions"],
    }
    cache = _canonical(cache_payload) + b"\n"
    public["cache_bytes"] = len(cache)
    public["cache_sha256"] = hashlib.sha256(cache).hexdigest()
    public["_ephemeral"] = {
        "candidate_visuals": base["_ephemeral"]["candidate_visuals"],
        "fixed_cache": cache,
        "base_result": base,
    }
    return public
