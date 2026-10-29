"""Pinned one-part derived scoring; matching rules are delegated unchanged.

Run this derived worker under the frozen 30s/1GiB supervisor. It never reads CAD
or imports native geometry. Reference must be frozen before extraction launch.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LIMITS = {"seconds": 30, "rss_kib": 1024**2, "derived_bytes": 40_000_000}
PROFILE = "ONE_CAD_EMAIL_EXACT_FACE_SCORE_V1"


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    if path.is_symlink():
        raise ValueError("DERIVED_SYMLINK")
    with open(path, "rb") as stream:
        raw = stream.read(LIMITS["derived_bytes"]+1)
    if len(raw) > LIMITS["derived_bytes"]:
        raise ValueError("DERIVED_READ_BUDGET")
    return raw


def pinned(path, pin):
    raw = read(path)
    if sha(raw) != pin:
        raise ValueError("DERIVED_PIN_MISMATCH")
    return json.loads(raw), len(raw)


def write(path, value):
    raw = encode(value)
    if len(raw) > LIMITS["derived_bytes"]:
        raise ValueError("DERIVED_WRITE_BUDGET")
    with open(path, "xb") as stream:
        stream.write(raw)


def source_pins():
    names = ("scripts/score_cad_email_transfer_v1.py", "tests/test_score_cad_email_transfer_v1.py",
             "scripts/run_cad_email_score_v1.py", "tests/test_run_cad_email_score_v1.py",
             "scripts/evaluate_vendor_cad_upload_unfold_v2.py",
             "backend/cad_bend_reference_scoring_v1.py", "tests/test_cad_bend_reference_scoring_v1.py")
    return {name: sha((ROOT/name).read_bytes()) for name in names}


def freeze(path, candidate_pin, reference_pin, input_pin):
    for pin in (candidate_pin, reference_pin, input_pin):
        if not isinstance(pin, str) or len(pin) != 64 or any(x not in "0123456789abcdef" for x in pin):
            raise ValueError("SHA256_REQUIRED")
    write(path, {"profile": PROFILE, "limits": LIMITS, "source_pins": source_pins(),
                 "candidate_frozen_sha256": candidate_pin, "reference_sha256": reference_pin,
                 "input_sha256": input_pin, "matching": "UNCHANGED_EXACT_NATIVE_FACE_UNION",
                 "scope": "ONE_PART_TECHNICAL_TRANSFER_NOT_MANUFACTURING_OR_GENERALIZATION"})


def score(reference, extraction, terminal, post, plan):
    from backend.cad_bend_reference_scoring_v1 import score_parts
    if (post.get("decision") != "GREEN" or post.get("independent_non_author") is not True
            or post.get("mode") != "extraction"
            or post.get("frozen_sha256") != plan["candidate_frozen_sha256"]):
        raise ValueError("INDEPENDENT_EXTRACTION_POST_REQUIRED")
    if terminal.get("cleanup_known") is not True or terminal.get("mode") != "extraction":
        raise ValueError("EXTRACTION_CLEANUP_UNKNOWN")
    if reference.get("input_sha256") != plan["input_sha256"] or reference.get("reference_frozen") is not True:
        raise ValueError("REFERENCE_BINDING_REQUIRED")
    success = terminal.get("outer_exit") == 0 and not terminal.get("reason")
    part = dict(reference)
    # Structural single-transfer identity, never an extraction/matching feature.
    part.setdefault("part_id", "email-transfer-01")
    if success:
        if not isinstance(extraction, dict) or extraction.get("input_sha256") != plan["input_sha256"]:
            raise ValueError("EXTRACTION_INPUT_BINDING")
        axes = extraction.get("bend_axes")
        if not isinstance(axes, list):
            raise ValueError("EXTRACTION_AXES_REQUIRED")
        part["predictions"] = [{"prediction_id": axis["feature_id"], "face_ids": axis["face_indices"]}
                               for axis in axes]
        part["prediction_status"] = "ok"
    else:
        if extraction is not None:
            raise ValueError("FAILED_ABSTENTION_CANNOT_CARRY_PREDICTIONS")
        part.update(predictions=[], prediction_status="failed",
                    failure_reason=terminal.get("reason") or "NONZERO_EXTRACTION_EXIT")
    result = score_parts([part])
    result["transfer_scope"] = "ONE_PART_NOT_UNSEEN_FAMILY_ACCURACY"
    result["geometry_accuracy"] = "UNSCORED_FINITE_LOCATION_RADIUS_TRIM_AND_FULL_CENTERLINE"
    return result


def execute(plan_path, plan_pin, reference_path, terminal_path, terminal_pin,
            post_path, post_pin, output, extraction_path=None, extraction_pin=None):
    plan, size = pinned(plan_path, plan_pin)
    if plan.get("profile") != PROFILE or plan.get("limits") != LIMITS or plan.get("source_pins") != source_pins():
        raise ValueError("SCORER_PLAN_CHANGED")
    reference, count = pinned(reference_path, plan["reference_sha256"])
    size += count
    terminal, count = pinned(terminal_path, terminal_pin)
    size += count
    post, count = pinned(post_path, post_pin)
    size += count
    if post.get("terminal_sha256") != terminal_pin or post.get("extraction_sha256") != extraction_pin:
        raise ValueError("POST_OUTPUT_BINDING_REQUIRED")
    extraction = None
    if extraction_path is not None:
        extraction, count = pinned(extraction_path, extraction_pin)
        size += count
    result = score(reference, extraction, terminal, post, plan)
    result["bindings"] = {"scorer_plan_sha256": plan_pin, "reference_sha256": plan["reference_sha256"],
                          "terminal_sha256": terminal_pin, "post_sha256": post_pin,
                          "extraction_sha256": extraction_pin}
    if size+len(encode(result)) > LIMITS["derived_bytes"]:
        raise ValueError("CUMULATIVE_DERIVED_BUDGET")
    write(output, result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    for name in ("plan", "reference", "terminal", "post", "extraction", "output"):
        p.add_argument("--"+name, type=Path)
        if name != "output":
            p.add_argument("--"+name+"-sha256")
    p.add_argument("--candidate-frozen-sha256")
    p.add_argument("--input-sha256")
    a = p.parse_args()
    if a.freeze:
        freeze(a.plan, a.candidate_frozen_sha256, a.reference_sha256, a.input_sha256)
    else:
        execute(a.plan, a.plan_sha256, a.reference, a.terminal, a.terminal_sha256,
                a.post, a.post_sha256, a.output, a.extraction, a.extraction_sha256)


if __name__ == "__main__":
    main()
