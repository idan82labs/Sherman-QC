"""Build docs/production/runlog/benchmark_v2/BENCHMARK_V2.json from metadata only.

Reads: merged_inventory.json, label_prep/selection.json, label_prep/prep_runlog.json,
label_lines/<scan>/ directory listings (+ sha256 of lines.json), and the verifier
adjudication table (labels.tsv, transcribed from the workflow's label results).
Never opens scan geometry, sealed/held-out material or opaque dirs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
BV2 = REPO / "docs/production/runlog/benchmark_v2"
HOME = Path(os.path.expanduser("~"))
LABELS_TSV = Path(sys.argv[1])
OUT = BV2 / "BENCHMARK_V2.json"
# 2026-10-17 re-verification (round 1/2 lines dirs, verifier corrections, studio label writes).
REVERIFY = Path(sys.argv[2]) if len(sys.argv) > 2 else BV2 / "manifest_build/reverify_results_2026-10-17.json"
# 2026-10-17 owner-delegated final verdicts on the 11 scans left in NEEDS_HUMAN_REVIEW (round-3 lines dirs).
REVIEW11 = BV2 / "manifest_build/review11_verdicts_2026-10-17.json"
# 2026-10-17 synthetic LIDAR_DEMO files (legacy /api/live-scan/demo/load resamples), per-scan evidence; role EXCLUDED_NOT_A_SCAN.
SYNTH_EXCL = BV2 / "manifest_build/synthetic_demo_exclusions_2026-10-17.json"
# 2026-10-17 round-3 relabels of accepted scans (new lines dir, verifier counts, studio label rewrite).
RELABEL = BV2 / "manifest_build/relabel_r3_2026-10-17.json"
# 2026-10-17 owner-requested fix of the 42288005 FULL joggle miss found by the broad_baseline CAD cross-check (same schema).
RELABEL_EXTRA = [BV2 / "manifest_build/relabel_42288005_2026-10-17.json"]
REVIEW11_RANK = {"AI_GOLD_ACCEPTED": 0, "AI_GOLD_PARTIAL": 1, "REJECTED_UNUSABLE": 2, "REJECTED_NOT_A_SCAN": 3}

CONSUMED_PRIOR = {"Gold11", "Gold37", "physical15", "other-consumed"}
STAGED_HINTS = {"PARTIAL_SCAN_1", "PARTIAL_SCAN_2", "PRELIM", "STAGE0_BLANK"}
STUDIO_ROOTS = {
    "source-assets": (REPO.parent / "source-assets"),
    "data": REPO / "data",
    "uploads": REPO / "uploads",
    "live_scans": REPO / "live_scans",
    "output": REPO / "output",
}
RECIPE_EXT = {"stp", "step"}
NATIVE_CAD_EXT = {"sldprt", "x_t", "x_b", "xdl"}
MESH_CAD_EXT = {"ply", "stl"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expand(p: str) -> Path:
    return Path(p.replace("~", str(HOME), 1)) if p.startswith("~") else Path(p)


def ext_of(p: str) -> str:
    return p.rsplit(".", 1)[-1].lower() if "." in p else ""


def studio_entry(paths: list[str]) -> dict | None:
    for raw in paths:
        if ext_of(raw) != "ply":
            continue
        resolved = Path(os.path.realpath(expand(raw)))
        for source, root in STUDIO_ROOTS.items():
            try:
                rel = resolved.relative_to(Path(os.path.realpath(root))).as_posix()
            except ValueError:
                continue
            aid = hashlib.sha256(f"{source}:{rel}".encode()).hexdigest()[:24]
            return {
                "url_path": "/bend-labeling",
                "local_url": "http://127.0.0.1:3000/bend-labeling",
                "asset_id": aid,
                "source": source,
                "relative_path": rel,
                "search": resolved.name,
                "queue_filter": "All sources" if source == "output" else "Human queue",
                "indexed": True,
            }
    if not any(ext_of(r) == "ply" for r in paths):
        why = "not indexed: the studio lists .ply only and this scan exists only as " + ",".join(sorted({ext_of(r) for r in paths}))
    else:
        why = "not indexed: the .ply lives outside the studio roots (source-assets, data, uploads, live_scans, output), e.g. ~/Downloads"
    return {"url_path": "/bend-labeling", "indexed": False, "reason": why}


def load_labels() -> dict[str, dict]:
    out = {}
    cols = ["scan_id", "part_id", "n_bends", "agree", "disputed", "verifier_saw_images",
            "lines_yes", "lines_no", "lines_unclear", "missed_bends_visible",
            "missed_basis", "input_accepted", "verifier_summary"]
    for line in LABELS_TSV.read_text().splitlines():
        if not line.strip():
            continue
        vals = line.split("\t")
        row = dict(zip(cols, vals))
        for k in ["n_bends", "agree", "disputed", "lines_yes", "lines_no", "lines_unclear",
                  "missed_bends_visible"]:
            row[k] = int(row[k])
        row["verifier_saw_images"] = row["verifier_saw_images"] == "1"
        row["input_accepted"] = row["input_accepted"] == "1"
        out[row["scan_id"]] = row
    return out


def load_reverify() -> dict:
    if not REVERIFY.is_file():
        return {}
    return json.loads(REVERIFY.read_text())


def what_to_check(rv: dict) -> str:
    """Verifier guidance for a scan that did not pass: give-up reason, failing lines, corrections."""
    parts = []
    if rv.get("gave_up"):
        parts.append("LABELER GAVE UP: " + rv["give_up_reason"])
    for tag in ("verifier_a", "verifier_b"):
        v = rv.get(tag) or {}
        bad = [f"{x['label']} {x['lands']}: {x.get('reason', '')}" for x in v.get("lines", []) if x["lands"] != "yes"]
        if bad:
            parts.append(f"{tag} lines not landing: " + " | ".join(bad))
        if v.get("missed"):
            parts.append(f"{tag} missed bends: {v['missed']}")
        if v.get("correction"):
            parts.append(f"{tag} correction: " + v["correction"])
    return "\n".join(parts)


def adjudicate(row: dict, lines_dir: Path) -> dict:
    """Strict gate: an AI label is gold only if a verifier actually saw the renders."""
    nb = row["n_bends"]
    passed = (
        row["verifier_saw_images"]
        and row["lines_no"] == 0
        and row["lines_unclear"] == 0
        and row["missed_bends_visible"] == 0
        and row["lines_yes"] == nb
    )
    input_status = "AI_GOLD_ACCEPTED" if row["input_accepted"] else "NEEDS_HUMAN_REVIEW"
    if passed:
        status = "AI_GOLD_ACCEPTED"
        reason = "verifier inspected renders; every line lands; no missed bends"
    elif not row["verifier_saw_images"]:
        status = "NEEDS_HUMAN_REVIEW"
        reason = ("TOOL_FAILED_NO_IMAGES: lines step rejected pairs.json ({a,b} format) and the "
                  "verifier saw no renders; missed-bend count not assessed from line overlays")
    elif nb == 0 and row["missed_bends_visible"] > 0:
        status = "NEEDS_HUMAN_REVIEW"
        reason = "EMPTY_LABEL_BUT_BENDS_VISIBLE"
    else:
        status = "NEEDS_HUMAN_REVIEW"
        reason = "LINES_INCOMPLETE: some lines do not land / too short / missed bends"
    role = None
    if status == "AI_GOLD_ACCEPTED":
        role = "zero_bend_control" if nb == 0 else "bend_lines"
    lines_json = lines_dir / "lines.json"
    renders = sorted(p.name for p in (lines_dir / "render").glob("*.png")) if (lines_dir / "render").is_dir() else []
    disk = {
        "label_lines_dir": str(lines_dir.relative_to(REPO)),
        "lines_json_present": lines_json.is_file(),
        "lines_json_sha256": sha256_file(lines_json) if lines_json.is_file() else None,
        "pairs_json_sha256": sha256_file(lines_dir / "pairs.json") if (lines_dir / "pairs.json").is_file() else None,
        "render_png_count": len(renders),
    }
    stale = (not row["verifier_saw_images"]) and disk["lines_json_present"]
    return {
        "input_status": input_status,
        "final_status": status,
        "downgraded": input_status == "AI_GOLD_ACCEPTED" and status != "AI_GOLD_ACCEPTED",
        "reason": reason,
        "role": role,
        "n_bends_labelled": nb,
        "labeler_agreement": {"agree": row["agree"], "disputed": row["disputed"]},
        "verifier": {
            "saw_images": row["verifier_saw_images"],
            "lines_yes": row["lines_yes"],
            "lines_no": row["lines_no"],
            "lines_unclear": row["lines_unclear"],
            "missed_bends_visible": row["missed_bends_visible"],
            "missed_basis": row["missed_basis"],
            "summary": row["verifier_summary"],
        },
        "disk_at_manifest_time": disk,
        "lines_rendered_after_verification": stale,
    }


def main() -> None:
    inv_path = BV2 / "merged_inventory.json"
    sel_path = BV2 / "label_prep/selection.json"
    prep_path = BV2 / "label_prep/prep_runlog.json"
    inv = json.loads(inv_path.read_text())
    sel = json.loads(sel_path.read_text())
    prep = json.loads(prep_path.read_text()).get("runs", {})
    labels = load_labels()
    reverify = load_reverify()
    rv_results = {r["scan_id"]: r for r in reverify.get("results", [])}
    rv_studio = reverify.get("studio_label_writes", {}).get("final_by_scan", {})
    rv_refmesh = set(reverify.get("possible_cad_reference_meshes", {}).get("scan_ids", []))
    review11 = json.loads(REVIEW11.read_text()) if REVIEW11.is_file() else {}
    r11 = {r["scan_id"]: r for r in review11.get("results", [])}
    r11_counts = Counter()
    synth = json.loads(SYNTH_EXCL.read_text()) if SYNTH_EXCL.is_file() else {}
    sx = {r["scan_id"]: r for r in synth.get("results", [])}
    relabel = json.loads(RELABEL.read_text()) if RELABEL.is_file() else {}
    rl = {r["scan_id"]: r for r in relabel.get("results", [])}
    rl_src = {k: str(RELABEL.relative_to(REPO)) for k in rl}
    for extra in RELABEL_EXTRA:
        if extra.is_file():
            for r in json.loads(extra.read_text()).get("results", []):
                rl[r["scan_id"]] = r
                rl_src[r["scan_id"]] = str(extra.relative_to(REPO))

    files = inv["files"]
    by_sha = {f["sha256"]: f for f in files}
    selected = {s["sha256"]: s for s in sel["selected"]}
    skipped = {s["sha256"]: s for s in sel["skipped"]}
    sel_flags = {s["sha256"]: s.get("flags", []) for s in sel["selected"]}

    # --- part grouping (UNKNOWN / blank part ids are split per file) ---
    def part_key(f: dict) -> str:
        pid = f["part_id"]
        if pid in ("", "UNKNOWN"):
            return f"UNKNOWN:sc_{f['sha256'][:12]}" if f["kind"] == "scan" else f"UNKNOWN:{f['kind']}_{f['sha256'][:12]}"
        return pid

    groups: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        groups[part_key(f)].append(f)
    nod_by_part: dict[str, list[dict]] = defaultdict(list)
    for x in inv["not_on_disk"]:
        nod_by_part[x["part_id"] or "UNKNOWN:not_on_disk"].append(x)

    parts_out = {}
    review = []
    missing_studio = []
    counts = Counter()
    for pk in sorted(set(groups) | set(nod_by_part)):
        fl = groups.get(pk, [])
        usable = [f for f in fl if f["usable"]]
        eyal = [f for f in fl if f["eyal50"]]
        nod = nod_by_part.get(pk, [])
        cad = [f for f in usable if f["kind"] == "cad"]
        cad_ext = sorted({ext_of(f["paths"][0]) for f in cad if f["paths"]})
        drawings = [f for f in usable if f["kind"] == "drawing"]
        flats = [f for f in usable if f["is_flat"]]
        if any(e in RECIPE_EXT for e in cad_ext):
            cad_recipe = "STEP_ON_DISK_RECIPE_COMPILABLE_NOT_YET_COMPILED"
        elif any(e in NATIVE_CAD_EXT for e in cad_ext):
            cad_recipe = "NATIVE_CAD_ONLY_NEEDS_STEP_EXPORT"
        elif any(e in MESH_CAD_EXT for e in cad_ext):
            cad_recipe = "CAD_MESH_ONLY_NO_RECIPE"
        else:
            cad_recipe = "NO_CAD_ON_DISK"

        prior = sorted({f["prior_use"] for f in usable} | {x["prior_use"] for x in nod})
        scans_out = []
        for f in sorted((f for f in usable if f["kind"] == "scan"), key=lambda f: (f["paths"] or [""])[0]):
            sid = f"sc_{f['sha256'][:12]}"
            sk = skipped.get(f["sha256"])
            role = "benchmark_scan"
            role_note = None
            if not f["paths"]:
                role, role_note = "EXCLUDED_REDACTED_OPAQUE_COPY", "only copy is evaluator-opaque; path redacted"
            elif sk and "synthetic" in sk["reason"]:
                role, role_note = "EXCLUDED_SYNTHETIC_FIXTURE", sk["reason"]
            elif sk and ("alternate" in sk["reason"] or "twin" in sk["reason"]):
                role, role_note = "FORMAT_ALTERNATE", sk["reason"]
            elif "alternate format" in " ".join(f["why"]):
                role, role_note = "FORMAT_ALTERNATE", "; ".join(f["why"])
            elif sid in r11 and r11[sid]["final_status"] == "REJECTED_NOT_A_SCAN":
                role, role_note = "EXCLUDED_NOT_A_SCAN", "review11 verdict REJECTED_NOT_A_SCAN: " + r11[sid]["reason"]
            elif sid in sx:
                role, role_note = "EXCLUDED_NOT_A_SCAN", ("synthetic LIDAR_DEMO resample of " + sx[sid]["evidence"]["source_mesh"] +
                                                          " (see " + SYNTH_EXCL.name + "); not a physical scan")
            elif sk:
                role, role_note = "EXCLUDED_OTHER", sk["reason"]

            truth = []
            gold_detail = []
            for w in f["why"]:
                m = re.match(r"gold_lines: (\S+) tier=(\S+) frame=(\S+) n=(\d+)", w)
                if m:
                    src, tier, frame, n = m.groups()
                    human = tier == "authoritative_user_verified"
                    gold_detail.append({
                        "ref": src, "tier": tier, "label_frame": frame, "n_lines": int(n),
                        "labeller": "human (owner-verified)" if human else "AI (Codex manual multiview), not human",
                        "transform_recovery": (
                            "REQUIRED_PENDING: recover raw-to-reference transform (residual <= 0.01 mm) before scoring; plan s10 amendment 6(a)"
                            if frame == "reference_mesh_raw" else
                            "NOT_REQUIRED_IF_RAW_SHA_MATCHES: labels already in scan-raw frame; confirm scan sha256 equals the labelled file"
                        ),
                    })
                    truth.append("GOLD_LINES_HUMAN" if human else "GOLD_LINES_AI_LEGACY")
                m2 = re.match(r"Gold11 case (\S+) ref_frame=(\S+) refs=(\d+)", w)
                if m2:
                    gold_detail.append({"gold11_case": m2.group(1), "label_frame": m2.group(2), "n_refs": int(m2.group(3))})
                if w.startswith("human_bend_lines_v1"):
                    gold_detail.append({"studio_label_ids": w.split(" ", 1)[1].split(","),
                                        "labeller": "Codex multiview gold import (annotator field), not a human studio label"})
            lab = None
            if sid in labels:
                rv = rv_results.get(sid)
                v11 = r11.get(sid)
                x3 = rl.get(sid)
                lab = adjudicate(dict(labels[sid], **x3["tsv_override"]) if x3 else labels[sid],
                                 Path(x3["lines_dir"]) if x3 else Path(v11["final_lines_dir"]) if v11 else
                                 Path(rv["lines_dir"]) if rv and rv.get("lines_dir") else BV2 / "label_lines" / sid)
                if rv:
                    lab["reverify_2026_10_17"] = {"round": rv["round"], "status": rv["status"], "gave_up": rv["gave_up"],
                                                  "lines_dir": rv["lines_dir"]}
                    if lab["final_status"] != "AI_GOLD_ACCEPTED":
                        lab["what_to_check"] = what_to_check(rv)
                        if rv["gave_up"]:
                            lab["reason"] = "GAVE_UP_PREP_OR_TOOL_CANNOT_EXPRESS: " + lab["reason"]
                lab["studio_label"] = rv_studio.get(sid)
                if x3:
                    lab["studio_label"] = x3["studio_label"]
                    lab["relabel_r3"] = {k: x3[k] for k in ("lines_dir", "n_bends_before", "n_bends", "added", "reason", "evidence_dir")}
                if v11:
                    # owner-delegated final verdict overrides the strict-gate status
                    lab["final_status"] = v11["final_status"]
                    lab["reason"] = f"review11 {v11['verdict']}: " + v11["reason"]
                    lab["role"] = "bend_lines" if v11["final_status"] in ("AI_GOLD_ACCEPTED", "AI_GOLD_PARTIAL") else None
                    lab["label_complete"] = v11["label_complete"]
                    lab["studio_label"] = v11["studio_label"]
                    lab.pop("what_to_check", None)
                    lab["review11"] = {k: v11[k] for k in ("verdict", "real_bends", "labelled_bends", "drop_labels",
                                                          "label_complete", "final_lines_dir", "evidence_dir")}
                    r11_counts[v11["final_status"]] += 1
                if lab["final_status"] == "AI_GOLD_ACCEPTED":
                    truth.append("AI_GOLD_ACCEPTED")
                elif lab["final_status"] == "AI_GOLD_PARTIAL":
                    truth.append("AI_GOLD_PARTIAL")
                elif lab["final_status"].startswith("REJECTED"):
                    truth.append(lab["final_status"] + " (not truth)")
                else:
                    truth.append("AI_LABEL_PENDING_REVIEW (not truth)")
            if cad_recipe.startswith("STEP"):
                truth.append("CAD_RECIPE (nominal; not compiled yet)")
            truth = list(dict.fromkeys(truth))
            if "GOLD_LINES_HUMAN" in truth:
                primary = "GOLD_LINES_HUMAN"
            elif "AI_GOLD_ACCEPTED" in truth:
                primary = "AI_GOLD_ACCEPTED"
            elif "AI_GOLD_PARTIAL" in truth:
                primary = "AI_GOLD_PARTIAL"
            elif "GOLD_LINES_AI_LEGACY" in truth:
                primary = "GOLD_LINES_AI_LEGACY"
            elif any(t.startswith("CAD_RECIPE") for t in truth):
                primary = "CAD_RECIPE"
            else:
                primary = "NONE"
            hints = f["stage_hints"]
            staged = bool(set(hints) & STAGED_HINTS)
            pr = prep.get(sid, {})
            entry = {
                "scan_id": sid,
                "sha256": f["sha256"],
                "paths": f["paths"],
                "format": ext_of(f["paths"][0]) if f["paths"] else None,
                "rev": f["rev"],
                "origins": f["origins"],
                "prior_use": f["prior_use"],
                "prior_use_basis": f["why"],
                "role": role,
                "role_note": role_note,
                "stage_hints": hints,
                "staged_or_incomplete": staged,
                "quality_flags": sorted(set(sel_flags.get(f["sha256"], [])) | ({"LIDAR_DEMO_CAPTURE"} & set(hints))
                                        | ({"POSSIBLE_CAD_REFERENCE_MESH"} if sid in rv_refmesh else set())
                                        | ({"SYNTHETIC_RESAMPLE_NOT_A_SCAN"} if role == "EXCLUDED_NOT_A_SCAN" else set())),
                "scan_accuracy_scoring": ("EXCLUDE: may be a CAD reference mesh (uploads/*/reference.*), not a scan; zero-bend false-positive control only"
                                          if sid in rv_refmesh else
                                          ((r11.get(sid) or {}).get("scan_accuracy_scoring") or (sx.get(sid) or {}).get("scorer_action")
                                           or "EXCLUDE: not a physical scan")
                                          if role == "EXCLUDED_NOT_A_SCAN" else None),
                "truth_sources": truth or ["NONE"],
                "primary_truth": primary,
                "gold_detail": gold_detail,
                "ai_label_run": lab,
                "prep": {k: pr.get(k) for k in ("scan_kind", "scan_points", "thickness_mm", "n_patches", "unsegmented_fraction")} if pr else None,
                "studio": studio_entry(f["paths"]),
            }
            scans_out.append(entry)
            if role == "benchmark_scan":
                counts["scans"] += 1
                counts[f"scans_primary_truth_{primary}"] += 1
                if staged:
                    counts["scans_staged_or_incomplete"] += 1
                if lab:
                    counts["scans_ai_labelled"] += 1
                    counts[f"ai_final_{lab['final_status']}"] += 1
                    if lab["downgraded"]:
                        counts["ai_downgraded_from_accepted"] += 1
                    if lab["final_status"] == "AI_GOLD_ACCEPTED":
                        counts[f"ai_accepted_{lab['role']}"] += 1
                        counts["ai_accepted_bend_lines_total"] += lab["n_bends_labelled"]
                        st = lab.get("studio_label")
                        if st and st.get("written"):
                            counts["ai_accepted_studio_label_written"] += 1
                        else:
                            counts["ai_accepted_studio_label_missing"] += 1
                            missing_studio.append({"scan_id": sid, "n_bends": lab["n_bends_labelled"],
                                                   "error": (st or {}).get("error", "no write-label run recorded")})
                    elif lab["final_status"] == "AI_GOLD_PARTIAL":
                        counts["ai_partial_bend_lines_total"] += lab["n_bends_labelled"]
                    elif lab["final_status"] == "NEEDS_HUMAN_REVIEW":
                        review.append({
                            "scan_id": sid, "part_id": pk, "path": f["paths"][0],
                            "split": None,  # filled below
                            "reason": lab["reason"].split(":")[0],
                            "input_status": lab["input_status"],
                            "n_bends_proposed": lab["n_bends_labelled"],
                            "missed_bends_visible": lab["verifier"]["missed_bends_visible"],
                            "missed_basis": lab["verifier"]["missed_basis"],
                            "lines_rendered_after_verification": lab["lines_rendered_after_verification"],
                            "what_to_check": lab.get("what_to_check") or lab["verifier"]["summary"],
                            "gave_up": (lab.get("reverify_2026_10_17") or {}).get("gave_up", False),
                            "studio": entry["studio"],
                        })
            else:
                counts[f"scans_{role}"] += 1

        bench_scans = [s for s in scans_out if s["role"] == "benchmark_scan"]
        # --- split (part level, conservative) ---
        if not usable and eyal:
            split, split_basis = "EXCLUDED_EYAL50_QUARANTINE", "only Eyal-50 delivery files; locked-holdout membership not read"
        elif not usable and not nod:
            split, split_basis = "EXCLUDED_NO_USABLE_FILES", "no usable files"
        elif usable and {f["prior_use"] for f in usable} <= {"fresh"} and any(x["prior_use"] in CONSUMED_PRIOR for x in nod):
            split, split_basis = "CONSUMED", ("CONSUMED_UNVERIFIED: every on-disk file is fresh; consumed only because email attachments not on disk "
                                              "were matched to local files by filename (" + ", ".join(sorted({x["filename"] for x in nod if x["prior_use"] in CONSUMED_PRIOR})) +
                                              "). Conservative; owner may verify and reclassify to FRESH before any FRESH run, never after")
        elif any(p in CONSUMED_PRIOR for p in prior):
            split, split_basis = "CONSUMED", "at least one file of this part was used in development (" + ", ".join(p for p in prior if p in CONSUMED_PRIOR) + ")"
        elif "unknown" in prior:
            split, split_basis = "CONSUMED", "prior use unknown (no provenance); treated as consumed, never FRESH"
        elif eyal:
            split, split_basis = "HELD_EYAL50_OVERLAP", "fresh files but part also in Eyal-50 delivery (locked-holdout split unread); not evaluated until owner decides"
        else:
            split, split_basis = "FRESH", "every known file of this part is prior_use=fresh (never used in tuning)"
        for s in scans_out:
            if s["role"] != "benchmark_scan":
                s["eval_tier"] = None
                continue
            base = Path(s["paths"][0]).name
            alt_consumed = any(
                a["role"] == "FORMAT_ALTERNATE" and a["prior_use"] != "fresh" and base in (a["role_note"] or "")
                for a in scans_out
            )
            if split == "FRESH":
                s["eval_tier"] = "FRESH"
            elif split == "CONSUMED" and s["prior_use"] == "fresh" and not alt_consumed:
                s["eval_tier"] = "FRESH_SCAN_OF_CONSUMED_PART"
                counts["fresh_scan_of_consumed_part"] += 1
            else:
                s["eval_tier"] = split
            s["eval_tier_note"] = (
                "no recorded use of this scan's geometry, but the part is CONSUMED (" + split_basis + "); report separately from FRESH, also evaluated once"
                if s["eval_tier"] == "FRESH_SCAN_OF_CONSUMED_PART" else
                ("fresh file but a format alternate with identical geometry is consumed" if s["prior_use"] == "fresh" and alt_consumed else None)
            )
        for r in review:
            if r["part_id"] == pk:
                r["split"] = split
                r["eval_tier"] = next((s["eval_tier"] for s in scans_out if s["scan_id"] == r["scan_id"]), None)
        gold11 = any(s["prior_use"] == "Gold11" for s in scans_out)
        parts_out[pk] = {
            "split": split,
            "split_basis": split_basis,
            "prior_use": prior,
            "in_gold11": gold11,
            "in_gold37": any(s["prior_use"] == "Gold37" for s in scans_out),
            "in_physical15": any(s["prior_use"] == "physical15" for s in scans_out),
            "cad": {
                "files": [{"sha256": f["sha256"], "path": (f["paths"] or [None])[0], "format": ext_of(f["paths"][0]) if f["paths"] else None} for f in cad],
                "formats": cad_ext,
                "recipe_status": cad_recipe,
            },
            "flats": [{"sha256": f["sha256"], "path": (f["paths"] or [None])[0], "kind": f["kind"]} for f in flats],
            "drawings": [{"sha256": f["sha256"], "path": (f["paths"] or [None])[0]} for f in drawings],
            "scans": scans_out,
            "n_benchmark_scans": len(bench_scans),
            "not_on_disk": [{"filename": x["filename"], "kind": x["kind"], "source": x["source"], "download_status": x["download_status"], "prior_use": x["prior_use"]} for x in nod],
            "eyal50_quarantined_files": len(eyal),
        }
        counts[f"parts_split_{split}"] += 1
        if bench_scans:
            counts["parts_with_benchmark_scan"] += 1
            counts[f"parts_with_benchmark_scan_{split}"] += 1
            if cad:
                counts["parts_with_benchmark_scan_and_cad"] += 1
            if cad_recipe.startswith("STEP"):
                counts["parts_with_benchmark_scan_and_step"] += 1
            if not any(s["primary_truth"] != "NONE" for s in bench_scans):
                counts["parts_with_benchmark_scan_no_truth"] += 1
        if cad:
            counts["parts_with_cad"] += 1
        if split == "FRESH":
            counts["fresh_benchmark_scans"] += len(bench_scans)
            counts["fresh_scans_not_on_disk"] += sum(1 for x in nod if x["kind"] == "scan")

    real_parts = [k for k, v in parts_out.items() if not k.startswith("UNKNOWN:") and v["split"] not in ("EXCLUDED_EYAL50_QUARANTINE", "EXCLUDED_NO_USABLE_FILES")]
    summary = {
        "unique_part_ids_usable": len(real_parts),
        "unknown_part_scans_as_own_parts": sum(1 for k, v in parts_out.items() if k.startswith("UNKNOWN:sc_") and v["n_benchmark_scans"]),
        "benchmark_scans": counts["scans"],
        "parts_with_benchmark_scan": counts["parts_with_benchmark_scan"],
        "parts_with_benchmark_scan_and_cad": counts["parts_with_benchmark_scan_and_cad"],
        "parts_with_benchmark_scan_and_step": counts["parts_with_benchmark_scan_and_step"],
        "parts_with_cad_any": counts["parts_with_cad"],
        "scans_staged_or_incomplete": counts["scans_staged_or_incomplete"],
        "scans_primary_truth": {k.replace("scans_primary_truth_", ""): v for k, v in counts.items() if k.startswith("scans_primary_truth_")},
        "ai_label_run": {
            "scans_labelled": counts["scans_ai_labelled"],
            "input_reported_accepted": sum(1 for r in labels.values() if r["input_accepted"]),
            "final_ai_gold_accepted": counts["ai_final_AI_GOLD_ACCEPTED"],
            "of_which_bend_lines": counts["ai_accepted_bend_lines"],
            "of_which_zero_bend_controls": counts["ai_accepted_zero_bend_control"],
            "accepted_bend_lines_total": counts["ai_accepted_bend_lines_total"],
            "downgraded_from_accepted_no_images": counts["ai_downgraded_from_accepted"],
            "needs_human_review": counts["ai_final_NEEDS_HUMAN_REVIEW"],
            "accepted_with_studio_label_written": counts["ai_accepted_studio_label_written"],
            "accepted_without_studio_label": missing_studio,
            "reverify_2026_10_17": {"path": str(REVERIFY.relative_to(REPO)) if REVERIFY.is_file() else None,
                                    "scans_reverified": len(rv_results)},
            "possible_cad_reference_meshes_excluded_from_scan_accuracy": sorted(rv_refmesh),
            "ai_gold_partial": counts["ai_final_AI_GOLD_PARTIAL"],
            "ai_gold_partial_bend_lines_total": counts["ai_partial_bend_lines_total"],
            "review11_2026_10_17": {"path": str(REVIEW11.relative_to(REPO)) if REVIEW11.is_file() else None,
                                    "scans": len(r11), "by_status": dict(sorted(r11_counts.items(), key=lambda kv: REVIEW11_RANK.get(kv[0], 9))),
                                    "rejected_not_a_scan_excluded_from_benchmark": sorted(k for k, v in r11.items() if v["final_status"] == "REJECTED_NOT_A_SCAN"),
                                    "owner_flags": review11.get("owner_flags", [])},
            "synthetic_demo_exclusions_2026_10_17": {"path": str(SYNTH_EXCL.relative_to(REPO)) if SYNTH_EXCL.is_file() else None,
                                                     "excluded_not_a_scan": sorted(sx),
                                                     "newly_excluded": sorted(k for k in sx if k not in r11),
                                                     "gold_set_hits": synth.get("gold_set_hits", [])},
            "relabel_r3_2026_10_17": {"path": str(RELABEL.relative_to(REPO)) if RELABEL.is_file() else None,
                                      "scans": {k: f"{v['n_bends_before']} -> {v['n_bends']} bends" for k, v in rl.items()},
                                      "source_by_scan": rl_src},
        },
        "parts_by_split": {k.replace("parts_split_", ""): v for k, v in counts.items() if k.startswith("parts_split_")},
        "parts_with_benchmark_scan_by_split": {k.replace("parts_with_benchmark_scan_", ""): v for k, v in counts.items() if k.split("parts_with_benchmark_scan_")[-1] in ("CONSUMED", "FRESH", "HELD_EYAL50_OVERLAP")},
        "fresh_benchmark_scans_on_disk": counts["fresh_benchmark_scans"],
        "fresh_scan_of_consumed_part_on_disk": counts["fresh_scan_of_consumed_part"],
        "consumed_unverified_parts": sorted(k for k, v in parts_out.items() if v["split_basis"].startswith("CONSUMED_UNVERIFIED")),
        "fresh_parts_numeric_ids": sum(1 for k, v in parts_out.items() if v["split"] == "FRESH" and not k.startswith("UNKNOWN:")),
        "fresh_parts_with_cad": sum(1 for k, v in parts_out.items() if v["split"] == "FRESH" and v["cad"]["files"]),
        "fresh_parts_with_step": sum(1 for k, v in parts_out.items() if v["split"] == "FRESH" and v["cad"]["recipe_status"].startswith("STEP")),
        "fresh_parts_with_scans_pending_download": sorted(k for k, v in parts_out.items() if v["split"] == "FRESH" and any(x["kind"] == "scan" for x in v["not_on_disk"])),
        "fresh_scans_not_on_disk": counts["fresh_scans_not_on_disk"],
        "non_benchmark_scan_rows": {k.replace("scans_", ""): v for k, v in counts.items() if k.startswith("scans_EXCLUDED") or k.startswith("scans_FORMAT")},
        "not_on_disk_email_files": len(inv["not_on_disk"]),
        "not_on_disk_email_scans": sum(1 for x in inv["not_on_disk"] if x["kind"] == "scan"),
        "excluded_sealed_counts_from_inventory": inv["excluded_sealed_counts"],
        "eyal50_quarantined_files": inv["summary"].get("eyal50_quarantined_files"),
    }
    order = {"TOOL_FAILED_NO_IMAGES": 1, "EMPTY_LABEL_BUT_BENDS_VISIBLE": 0, "LINES_INCOMPLETE": 0}
    review.sort(key=lambda r: (r["split"] != "FRESH", order.get(r["reason"], 2), r["part_id"], r["scan_id"]))

    manifest = {
        "schema": "sherman.benchmark_v2.manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "metadata-only merge of the benchmark_v2 inventory, label_prep selection and label_lines verifier results; no scan geometry read",
        "inputs": {
            "merged_inventory": {"path": str(inv_path.relative_to(REPO)), "sha256": sha256_file(inv_path)},
            "label_prep_selection": {"path": str(sel_path.relative_to(REPO)), "sha256": sha256_file(sel_path)},
            "label_prep_runlog": {"path": str(prep_path.relative_to(REPO)), "sha256": sha256_file(prep_path)},
            "verifier_results": {"path": str(LABELS_TSV.resolve().relative_to(REPO)) if str(LABELS_TSV.resolve()).startswith(str(REPO)) else str(LABELS_TSV), "sha256": sha256_file(LABELS_TSV), "note": "workflow label results (58 scans) transcribed per scan into ai_label_run; 48 rows re-verified 2026-10-17"},
            "reverify_results": {"path": str(REVERIFY.relative_to(REPO)), "sha256": sha256_file(REVERIFY)} if REVERIFY.is_file() else None,
            "review11_verdicts": {"path": str(REVIEW11.relative_to(REPO)), "sha256": sha256_file(REVIEW11)} if REVIEW11.is_file() else None,
            "synthetic_demo_exclusions": {"path": str(SYNTH_EXCL.relative_to(REPO)), "sha256": sha256_file(SYNTH_EXCL)} if SYNTH_EXCL.is_file() else None,
            "relabel_r3": {"path": str(RELABEL.relative_to(REPO)), "sha256": sha256_file(RELABEL)} if RELABEL.is_file() else None,
            "relabel_extra": [{"path": str(x.relative_to(REPO)), "sha256": sha256_file(x)} for x in RELABEL_EXTRA if x.is_file()],
        },
        "rules": {
            "split_unit": "part (SKU). A part is FRESH only if every known file of it is prior_use=fresh. Any consumed or unknown-provenance file makes the whole part CONSUMED. Fresh-looking files of a consumed part stay CONSUMED.",
            "consumed_definition": "Gold11, Gold37, physical15, any other development/F1-tuning use (other-consumed), and unknown provenance.",
            "fresh_once_rule": "FRESH is evaluated ONCE, at the end of the accuracy loop, with one frozen engine config hash. FRESH results must never feed back into tuning, thresholds, label edits or case selection. If FRESH is looked at earlier, or the engine changes after the FRESH run, those parts become CONSUMED and the run is reported as such.",
            "gold11_note": "Gold11 is CONSUMED. The engine was tuned on it, so a green result there is expected and is not evidence of accuracy on unseen parts. Before scoring, each Gold label set in reference_mesh_raw must have its raw-to-reference transform recovered (residual <= 0.01 mm) and scoring happens in the raw-scan frame (plan s10 amendment 6(a)). Cases without a recoverable transform are listed and excluded before scoring.",
            "ai_gold_gate": "AI_GOLD_ACCEPTED requires that an independent verifier agent actually saw the rendered line overlays, every line lands (yes), none is no/unclear, and zero visible bends are missed. Labels whose verifier saw no images are NEEDS_HUMAN_REVIEW even when the upstream step reported accepted=true.",
            "review11_owner_delegated": "2026-10-17: the owner delegated the final verdict on the 11 scans left in NEEDS_HUMAN_REVIEW to the AI review (investigator, 2 blind verifiers, judge, final pass). Statuses: AI_GOLD_ACCEPTED, AI_GOLD_PARTIAL (label_complete=false: some real folds unlabelled; not usable as a line-F1 denominator), REJECTED_UNUSABLE, REJECTED_NOT_A_SCAN (role EXCLUDED_NOT_A_SCAN, out of the benchmark). Still AI-adjudicated, not human.",
            "synthetic_demo_exclusions": "2026-10-17: live_scans/*_LIDAR_DEMO_<stamp>.ply files are synthetic resamples of test_files/<part>/scan.stl made by the legacy /api/live-scan/demo/load endpoint (Open3D header, xyz only, ~14000 or 12000 points minus 6% dropout, N(0, 0.08 mm) jitter, mtime = name stamp). They are role EXCLUDED_NOT_A_SCAN, out of every benchmark count. Their AI label files stay on disk; scorers must skip the scan ids listed in the exclusions file.",
            "truth_tiers": {
                "GOLD_LINES_HUMAN": "tier authoritative_user_verified (owner-verified line references)",
                "AI_GOLD_ACCEPTED": "this run's AI vision labels that passed the strict gate; adjudicated by agents, not humans",
                "GOLD_LINES_AI_LEGACY": "earlier Codex manual-multiview labels (tiers codex_manual_multiview_*). Studio files under human_bend_lines_v1 are Codex imports, not human labels",
                "CAD_RECIPE": "nominal bends from a STEP CAD (V10 extractor / DEV_AUTO_CONFIRMED recipe). Gives design intent (count, nominal angle), not where a given scan's bend lies, and says nothing about staged scans until stage is decided. Not compiled by this manifest",
                "NONE": "no truth available",
            },
            "custody": "HOLDOUT10, sealed truth, opaque N001-N014, EHO4, authorization/one-shot artifacts and held-out manifests are absent: excluded upstream by name/manifest membership and never opened. Eyal-50 delivery files are quarantined (locked-holdout split unread).",
        },
        "summary": summary,
        "needs_human_review": review,
        "parts": parts_out,
    }
    OUT.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1))
    print("review", len(review))


if __name__ == "__main__":
    main()
