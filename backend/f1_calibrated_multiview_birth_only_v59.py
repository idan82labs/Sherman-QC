"""V59: preserve V52 exactly and append only V58 raw multiview births."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any


PROFILE = "F1_V59_CALIBRATED_MULTIVIEW_BIRTH_ONLY_V1"


def compose_v59(v52: dict[str, Any], v58: dict[str, Any]) -> dict[str, Any]:
    if v52.get("source_only") is not True or v58.get("source_only") is not True:
        raise ValueError("V59 requires source-only V52 and V58")
    incumbents = deepcopy(list(v52.get("predictions") or []))
    births = [
        deepcopy(row)
        for row in v58.get("predictions") or []
        if row.get("selection_route") == "calibrated_multiview_turn_consensus_v58"
    ]
    predictions = incumbents + births
    audit = {
        "retained_v52_count": len(incumbents),
        "raw_new_count": len(births),
        "v58_prunes_ignored_for_visual_safety": len(v58.get("pruned_v52") or []),
        "retained_v52_byte_preservation": True,
    }
    result = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v59_calibrated_multiview_birth_only_result",
        "profile": PROFILE,
        "source_sha256": v58.get("source_sha256"),
        "capability_sha256": v58.get("capability_sha256"),
        "v52_sha256": v52.get("v52_result_sha256"),
        "v58_sha256": v58.get("v58_result_sha256"),
        "predictions": predictions,
        "audit": audit,
        "expected_count_used": False,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "source_only": True,
        "research_only": True,
    }
    semantic = json.dumps({"profile": PROFILE, "predictions": predictions, "audit": audit}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result["v59_result_sha256"] = hashlib.sha256(semantic).hexdigest()
    return result
