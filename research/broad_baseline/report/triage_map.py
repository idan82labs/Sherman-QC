"""Map free-form triage verdict rows onto the scored FN / FP items of each lane.

Read-only. Inputs are the broad-baseline score files and the persisted triage
results (docs/production/runlog/broad_baseline/triage/triage_results_24.json).
Each scored false negative (gold id) and false positive (prediction id) of a
triaged scan gets the cause of the triage row that names it for that lane.
Rows that aggregate ids ("B1-B12 (all 12)", "CAD_B1,B2,B3 (3)") are expanded.
Anything left without a row is reported as UNMAPPED so it can be resolved by
hand (see MANUAL below) instead of being guessed.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/production/runlog/broad_baseline"

PRED_RE = re.compile(r"(CADF_CAD_B\d+|CAD_B\d+(?:\s*,\s*(?:CAD_)?B\d+)*|F1C_[0-9a-f]{4,})")
GOLD_TOKEN_RE = re.compile(r"\b(HB|M|B)(\d{1,2})\b")
GOLD_RANGE_RE = re.compile(r"\b(HB|M|B)(\d{1,2})\s*-\s*(?:HB|M|B)?(\d{1,2})\b")

# Hand resolutions for scored items whose triage row is phrased in a way the
# parser cannot attribute safely. Each entry cites the triage text it relies on.
MANUAL = {
    # sc_9bd69bc6535c best_f1: row "FP pred #0 (F1C_27e3...) / FN gold B4" is ENGINE_SPLIT_OR_MERGE
    # (one over-long line along the free edge); the parser handles it, kept for clarity.
}


def norm_lane(lane: str | None) -> set[str]:
    lane = (lane or "").strip()
    if lane == "best_f1":
        return {"best_f1"}
    if lane == "all":
        return {"best_f1", "runtime", "runtime_gated", "runtime_with_placeholders"}
    if lane in ("runtime_gated", "runtime_with_placeholders"):
        return {lane}
    if lane.startswith("runtime"):
        out = {"runtime"}
        low = lane.lower()
        if "gated" in low or "all 3" in low:
            out.add("runtime_gated")
        if "placeholder" in low or "all 3" in low:
            out.add("runtime_with_placeholders")
        return out
    return set()


def expand_preds(text: str) -> list[str]:
    out: list[str] = []
    for m in PRED_RE.finditer(text):
        tok = m.group(1)
        if tok.startswith("CAD_B") and "," in tok:
            for part in re.split(r"\s*,\s*", tok):
                part = part.strip()
                out.append(part if part.startswith("CAD_") else "CAD_" + part)
        else:
            out.append(tok)
    return out


def expand_golds(text: str) -> list[str]:
    # drop prediction tokens first so CAD_B3 is not read as gold B3
    stripped = PRED_RE.sub(" ", text)
    out: list[str] = []
    for m in GOLD_RANGE_RE.finditer(stripped):
        pre, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        width = len(m.group(2))
        for i in range(a, b + 1):
            out.append(f"{pre}{i:0{width}d}")
    stripped2 = GOLD_RANGE_RE.sub(" ", stripped)
    for m in GOLD_TOKEN_RE.finditer(stripped2):
        out.append(m.group(1) + m.group(2))
    return out


def parse_item(item: str) -> tuple[list[str], list[str]]:
    """Return (gold_ids_as_FN, pred_ids_as_FP) named by a row item."""
    fn: list[str] = []
    fp: list[str] = []
    for seg in item.split("/"):
        s = seg.strip()
        low = s.lower()
        if low.startswith("fn gold"):
            fn += expand_golds(s[len("fn gold"):])
        elif low.startswith("fp pred"):
            fp += expand_preds(s)
        elif low.startswith("fn "):
            fn += expand_golds(s[3:])
    return fn, fp


def same_gold(a: str, b: str) -> bool:
    ma, mb = GOLD_TOKEN_RE.fullmatch(a), GOLD_TOKEN_RE.fullmatch(b)
    if not ma or not mb:
        return a == b
    return ma.group(1) == mb.group(1) and int(ma.group(2)) == int(mb.group(2))


def same_pred(scored: str, named: str) -> bool:
    if scored == named:
        return True
    if scored.startswith("F1C_") and named.startswith("F1C_"):
        return scored[:12] == named[:12]  # 8 hex chars; one agent mistyped a suffix
    return False


def load():
    tri = json.loads((OUT / "triage/triage_results_24.json").read_text())
    scores = {lane: {c["case_id"]: c for c in json.loads((OUT / f"scores_{lane}.json").read_text())["cases"]}
              for lane in ("best_f1", "runtime")}
    return tri, scores


def map_all():
    tri, scores = load()
    mapped = []  # one record per scored error
    unmapped = []
    for res in tri["results"]:
        sid = res["scan_id"]
        rows = res["rows"]
        parsed = []
        for r in rows:
            fn, fp = parse_item(r.get("item", ""))
            parsed.append((norm_lane(r.get("lane")), fn, fp, r))
        for lane in ("best_f1", "runtime"):
            c = scores[lane][sid]
            for kind, ids in (("FN", c.get("unmatched_gold_ids") or []), ("FP", c.get("unmatched_prediction_ids") or [])):
                for i in ids:
                    hit = None
                    for lanes, fn, fp, r in parsed:
                        if lane not in lanes:
                            continue
                        if kind == "FN" and any(same_gold(i, g) for g in fn):
                            hit = r
                            break
                        if kind == "FP" and any(same_pred(i, p) for p in fp):
                            hit = r
                            break
                    rec = {"scan_id": sid, "lane": lane, "kind": kind, "id": i,
                           "cause": hit["cause"] if hit else None,
                           "row_item": hit["item"] if hit else None,
                           "note": (hit.get("note") or "")[:400] if hit else None}
                    (mapped if hit else unmapped).append(rec)
    return mapped, unmapped


if __name__ == "__main__":
    mapped, unmapped = map_all()
    print("mapped", len(mapped), "unmapped", len(unmapped))
    for u in unmapped:
        print("UNMAPPED", u["scan_id"], u["lane"], u["kind"], u["id"])
    cnt = Counter((m["lane"], m["kind"], m["cause"]) for m in mapped)
    for k, v in sorted(cnt.items()):
        print(k, v)
