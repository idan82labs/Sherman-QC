#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("raw_candidate_id") or candidate.get("candidate_id") or "")


def _best_pair(candidate: Mapping[str, Any]) -> str:
    pair = candidate.get("best_flange_pair") or candidate.get("best_pair") or ()
    return "-".join(str(value) for value in pair) if pair else "unknown"


def _reason_counts(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        if candidate.get("bridge_safe"):
            continue
        for reason in candidate.get("reason_codes") or candidate.get("rejection_reasons") or ():
            counts[str(reason)] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _support_decision(
    *,
    human_target: int | None,
    track: str,
    raw_deduped: int,
    bridge_safe: int,
    safe_fraction: float,
    safe_pair_count: int,
) -> tuple[str, list[str], str]:
    reasons: list[str] = []
    track = track.strip().lower()

    if bridge_safe <= 0:
        reasons.append("no_bridge_safe_support")
        return "insufficient_support_abstain", reasons, "observe_only"

    if human_target is not None:
        if bridge_safe < human_target:
            reasons.append("bridge_safe_below_human_target")
            if track in {"partial", "scan_limited", "observed"}:
                reasons.append("partial_or_observed_scan_support_incomplete")
            if safe_fraction < 0.5:
                reasons.append("low_bridge_safe_fraction")
            return "limited_observed_support", reasons, "scan_quality_gate"

        if bridge_safe > human_target:
            reasons.append("bridge_safe_exceeds_human_target")
            if safe_pair_count < bridge_safe:
                reasons.append("duplicate_flange_pair_support")
            return "arrangement_needed", reasons, "sparse_arrangement_gate"

        if safe_fraction >= 0.65:
            reasons.append("bridge_safe_matches_human_target")
            reasons.append("high_bridge_safe_fraction")
            return "promotion_candidate_support", reasons, "promotion_review"

        reasons.append("bridge_safe_matches_human_target")
        reasons.append("low_bridge_safe_fraction")
        return "promotion_needs_review", reasons, "manual_review"

    if raw_deduped > 0 and safe_fraction < 0.5:
        reasons.append("low_bridge_safe_fraction")
        return "limited_observed_support", reasons, "scan_quality_gate"

    if safe_pair_count < bridge_safe:
        reasons.append("duplicate_flange_pair_support")
        return "arrangement_needed", reasons, "sparse_arrangement_gate"

    reasons.append("bridge_support_available")
    return "promotion_needs_review", reasons, "manual_review"


def evaluate_raw_crease_scan_support_quality(
    *,
    raw_creases: Mapping[str, Any],
    bridge_validation: Mapping[str, Any],
    part_id: str | None = None,
    human_target: int | None = None,
    track: str = "unknown",
) -> Dict[str, Any]:
    candidates = list(bridge_validation.get("candidates") or ())
    safe_candidates = [candidate for candidate in candidates if candidate.get("bridge_safe")]
    raw_before = int(raw_creases.get("raw_admissible_candidate_count_before_dedupe") or raw_creases.get("raw_admissible_candidate_count") or 0)
    raw_deduped = int(raw_creases.get("admissible_candidate_count") or raw_creases.get("candidate_count") or len(raw_creases.get("candidates") or ()))
    bridge_count = int(
        bridge_validation.get("family_bridge_candidate_count")
        or bridge_validation.get("bridge_candidate_count")
        or len(candidates)
    )
    bridge_safe = int(
        bridge_validation.get("family_bridge_safe_candidate_count")
        or bridge_validation.get("bridge_safe_candidate_count")
        or len(safe_candidates)
    )
    safe_fraction = float(bridge_safe) / float(max(raw_deduped, 1))
    safe_pairs = [_best_pair(candidate) for candidate in safe_candidates]
    safe_pair_counts = dict(sorted(Counter(safe_pairs).items(), key=lambda item: (-item[1], item[0])))
    decision, reason_codes, recommended_gate = _support_decision(
        human_target=human_target,
        track=track,
        raw_deduped=raw_deduped,
        bridge_safe=bridge_safe,
        safe_fraction=safe_fraction,
        safe_pair_count=len(safe_pair_counts),
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "part_id": part_id or str(bridge_validation.get("part_id") or raw_creases.get("part_id") or "unknown"),
        "scan_path": bridge_validation.get("scan_path") or raw_creases.get("scan_path"),
        "track": track,
        "human_target": human_target,
        "decision": decision,
        "recommended_gate": recommended_gate,
        "reason_codes": reason_codes,
        "feature_snapshot": {
            "raw_admissible_before_dedupe": raw_before,
            "raw_deduped_candidate_count": raw_deduped,
            "bridge_candidate_count": bridge_count,
            "bridge_safe_candidate_count": bridge_safe,
            "bridge_safe_fraction": round(safe_fraction, 4),
            "safe_flange_pair_count": len(safe_pair_counts),
            "safe_flange_pair_counts": safe_pair_counts,
            "rejected_reason_counts": _reason_counts(candidates),
        },
        "safe_candidate_ids": [_candidate_id(candidate) for candidate in safe_candidates],
        "rejected_candidate_ids": [_candidate_id(candidate) for candidate in candidates if not candidate.get("bridge_safe")],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate raw-crease scan support quality for offline F1 promotion gating.")
    parser.add_argument("--raw-creases-json", required=True, type=Path)
    parser.add_argument("--bridge-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--part-id", default=None)
    parser.add_argument("--human-target", type=int, default=None)
    parser.add_argument("--track", default="unknown")
    args = parser.parse_args()
    report = evaluate_raw_crease_scan_support_quality(
        raw_creases=_load_json(args.raw_creases_json),
        bridge_validation=_load_json(args.bridge_json),
        part_id=args.part_id,
        human_target=args.human_target,
        track=args.track,
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "part_id": report["part_id"],
                "decision": report["decision"],
                "recommended_gate": report["recommended_gate"],
                "reason_codes": report["reason_codes"],
                "bridge_safe_candidate_count": report["feature_snapshot"]["bridge_safe_candidate_count"],
                "bridge_safe_fraction": report["feature_snapshot"]["bridge_safe_fraction"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
