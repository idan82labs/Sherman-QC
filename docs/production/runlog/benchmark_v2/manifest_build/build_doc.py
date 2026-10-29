"""Render docs/production/BENCHMARK_V2.md from BENCHMARK_V2.json (no other inputs)."""
from __future__ import annotations

import collections as C
import hashlib
import json
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
MAN = REPO / "docs/production/runlog/benchmark_v2/BENCHMARK_V2.json"
OUT = REPO / "docs/production/BENCHMARK_V2.md"

m = json.loads(MAN.read_text())
S = m["summary"]
P = m["parts"]
man_sha = hashlib.sha256(MAN.read_bytes()).hexdigest()
ai = S["ai_label_run"]
pt = S["scans_primary_truth"]
r11 = ai.get("review11_2026_10_17") or {}
sxd = ai.get("synthetic_demo_exclusions_2026_10_17") or {}
rl3 = ai.get("relabel_r3_2026_10_17") or {}

RS = {
    "STEP_ON_DISK_RECIPE_COMPILABLE_NOT_YET_COMPILED": "STEP (not compiled)",
    "NATIVE_CAD_ONLY_NEEDS_STEP_EXPORT": "native only, needs STEP",
    "CAD_MESH_ONLY_NO_RECIPE": "mesh only",
    "NO_CAD_ON_DISK": "none",
}
AB = {"GOLD_LINES_HUMAN": "human", "AI_GOLD_ACCEPTED": "AI-accepted", "AI_GOLD_PARTIAL": "AI-partial", "GOLD_LINES_AI_LEGACY": "AI-legacy",
      "CAD_RECIPE": "CAD recipe", "NONE": "none"}


def short_name(k: str, scans: list[dict]) -> str:
    if not k.startswith("UNKNOWN"):
        return k
    p = scans[0]["paths"][0].split("/")
    return f"UNKNOWN ({p[-2]}/{p[-1]})"


L: list[str] = []
w = L.append

w("# Benchmark V2: broadened bend benchmark manifest")
w("")
w(f"Generated {m['generated_at'][:19]}Z from metadata only. Machine-readable manifest: "
  f"`docs/production/runlog/benchmark_v2/BENCHMARK_V2.json` (sha256 `{man_sha[:16]}…`). "
  "Inputs: `merged_inventory.json` (local files plus Sherman-domain email attachments, deduplicated by sha256), "
  "`label_prep/selection.json`, and the 58-scan AI labeling and verification results. No scan geometry was read to build it.")
w("")
w("Owner authorization (2026-10-17): broaden beyond Gold11 with all Sherman scans and CADs on this computer and the "
  "shermanep.com email attachments, and create AI-vision gold where no gold and no CAD exist. HOLDOUT10, sealed truth, "
  "opaque N001-N014, EHO4, authorization/one-shot artifacts and held-out manifests remain excluded.")
w("")

w("## Headline")
w("")
w(f"- **{S['benchmark_scans']} benchmark scans across {S['parts_with_benchmark_scan']} parts** "
  f"({S['parts_with_benchmark_scan'] - S['unknown_part_scans_as_own_parts']} with a SKU, {S['unknown_part_scans_as_own_parts']} without one). "
  f"Gold11 had 11 scans. {S['parts_with_benchmark_scan_and_cad']} of these parts have CAD on disk, but only "
  f"{S['parts_with_benchmark_scan_and_step']} have a STEP file that the V10 recipe compiler can read. The rest are SLDPRT or x_t and need a STEP export.")
w(f"- **Every part that has a scan on disk is CONSUMED, and FRESH has 0 scans on disk.** All {S['fresh_parts_numeric_ids']} FRESH parts are "
  f"CAD/drawing-only on disk. {S['fresh_scans_not_on_disk']} FRESH scan files (in {len(S['fresh_parts_with_scans_pending_download'])} parts) "
  "are still in email: they were not downloaded because of the MCP attachment size limit or SharePoint authentication. "
  f"In total, {S['not_on_disk_email_scans']} email scan attachments are not on disk. "
  "Until they are downloaded, nothing here can measure accuracy on unseen parts.")
w(f"- **Only {pt.get('GOLD_LINES_HUMAN', 0)} scans have human-verified gold lines.** "
  f"{pt.get('GOLD_LINES_AI_LEGACY', 0)} have earlier Codex (AI) labels, {pt.get('AI_GOLD_ACCEPTED', 0)} have AI gold accepted in this run, "
  f"{pt.get('CAD_RECIPE', 0)} have only a CAD recipe, and {pt.get('NONE', 0)} have no truth at all. "
  "Gold11 itself is 9 human label sets plus 2 Codex label sets (38172000 GOOD_V and DEFECTIVE_X).")
w(f"- **AI labeling: {ai['final_ai_gold_accepted']} of {ai['scans_labelled']} scans accepted.** {ai['of_which_bend_lines']} of them carry "
  f"{ai['accepted_bend_lines_total']} verified bend lines, and {ai['of_which_zero_bend_controls']} are zero-bend controls "
  "(rolled shells, one flat blank and one partial scan with no formed fold). "
  f"On 2026-10-17 {ai['reverify_2026_10_17']['scans_reverified']} scans were re-run and re-verified with renders under the strict rule "
  "(a line lands only on a real formed fold, parallel to it, covering at least ~80% of its visible length); every verifier in this manifest saw the line overlays, "
  f"and {ai['downgraded_from_accepted_no_images']} accepted labels lack images. "
  f"{ai['needs_human_review']} scans still need review: the owner delegated the final verdict on the last {r11.get('scans', 0)} to the AI review on 2026-10-17 "
  f"({', '.join(f'{n} {k}' for k, n in (r11.get('by_status') or {}).items())}; see below). "
  f"{ai['accepted_with_studio_label_written']} accepted scans have an `ai_bend_lines_v1` studio label file; "
  f"{len(ai['accepted_without_studio_label'])} exist only as `.stl` or `.obj` and have none yet.")
w("- **The Gold11 frame must be recovered before scoring.** 9 of the 11 Gold11 label sets were drawn in `reference_mesh_raw`. "
  "Plan §10 amendment 6(a) requires each case's raw-to-reference transform (residual ≤ 0.01 mm), with scoring in the raw-scan frame. "
  "This manifest marks those 9 as `REQUIRED_PENDING`. The 2 `scan_raw` cases need only a sha256 check of the labelled file.")
w("- **A green Gold11 is expected, not evidence.** The engine was tuned on Gold11, so it should come back green there. "
  "That shows the tuning reproduces; it says nothing about unseen parts.")
w("")

w("## Splits")
w("")
w("A split is assigned per **part** (SKU), never per scan, so a new scan of a tuned part cannot pass as unseen.")
w("")
w("| Split | Definition | Parts | Scans on disk |")
w("|---|---|---|---|")
w(f"| CONSUMED | Any file of the part is Gold11, Gold37, physical15, used in other development or F1 tuning (`other-consumed`), or of unknown provenance | "
  f"{S['parts_by_split'].get('CONSUMED', 0)} | {S['benchmark_scans'] - S['fresh_benchmark_scans_on_disk']} |")
w(f"| FRESH | Every known file of the part (local and email) is `prior_use=fresh`, and the part is not in the Eyal-50 delivery | "
  f"{S['parts_by_split'].get('FRESH', 0)} ({S['fresh_parts_numeric_ids']} with a SKU; {S['fresh_parts_with_cad']} with CAD, {S['fresh_parts_with_step']} with STEP) | "
  f"{S['fresh_benchmark_scans_on_disk']} ({S['fresh_scans_not_on_disk']} pending download) |")
w(f"| HELD_EYAL50_OVERLAP | Fresh files, but the part is also in the Eyal-50 delivery (locked-holdout split unread) | {S['parts_by_split'].get('HELD_EYAL50_OVERLAP', 0)} | 0 |")
w(f"| EXCLUDED_EYAL50_QUARANTINE | Only Eyal-50 delivery files (20 dev / 30 locked holdout; membership not read) | {S['parts_by_split'].get('EXCLUDED_EYAL50_QUARANTINE', 0)} | 0 |")
w("")
w(f"{len(S['consumed_unverified_parts'])} parts are `CONSUMED_UNVERIFIED`: {', '.join(S['consumed_unverified_parts'])}. Every file of these parts on disk is fresh. "
  "They count as consumed only because email copies that are not on disk were matched to local files by filename, "
  "and for 28289002 the only match is a generic drawing name (`שרטוט החלק.PDF`). They stay CONSUMED (conservative). "
  "The owner may verify them and move them to FRESH before the FRESH run, never after it.")
w("")
w(f"One scan is tagged `FRESH_SCAN_OF_CONSUMED_PART`: `17139000_01.ply` (`sc_90257b128d80`, part 17139000, one of the CONSUMED_UNVERIFIED parts). "
  "It has no recorded use. It is reported separately from FRESH and is also evaluated once. "
  "`37126000_D.ply` from email is not in this tier, because an identical-geometry STL of it in `source-assets` is consumed.")
w("")
w("### FRESH rule (binding)")
w("")
w("1. FRESH is evaluated **once**, at the end of the accuracy loop, with one frozen engine config hash recorded in the ledger before the run.")
w("2. FRESH results never feed back into tuning, thresholds, recipe or label edits, or case selection. Nobody looks at FRESH scans' engine output before that run.")
w("3. If FRESH is opened early, or the engine changes after the FRESH run, the affected parts become CONSUMED, and the report says so.")
w("4. FRESH labels (AI or human) may be drawn before the run, but only from geometry, never from engine output (the Bend Label Studio shows no predictions).")
w("5. CONSUMED results are engineering-ledger development evidence only and are never quoted as accuracy on unseen parts.")
w("")

w("## Truth tiers")
w("")
w("| Tier | Meaning | Scans (primary) |")
w("|---|---|---|")
w(f"| GOLD_LINES_HUMAN | `authoritative_user_verified` line references, verified by the owner | {pt.get('GOLD_LINES_HUMAN', 0)} |")
w(f"| AI_GOLD_ACCEPTED | This run's AI vision labels. An independent verifier agent inspected the renders, every line landed, and no visible bend was missed. **Adjudicated by agents, not humans.** | {pt.get('AI_GOLD_ACCEPTED', 0)} |")
w(f"| GOLD_LINES_AI_LEGACY | Earlier Codex manual-multiview labels (`codex_manual_multiview_*`). All 38 files in `data/bend_corpus/annotations/human_bend_lines_v1/` carry annotator `Codex multiview gold import`, so none are human studio labels | {pt.get('GOLD_LINES_AI_LEGACY', 0)} |")
w(f"| CAD_RECIPE | The part has a STEP file, so a V10 / DEV_AUTO_CONFIRMED recipe can be compiled. This gives design intent (count, nominal angle), not the scanned line position, and needs a stage decision for partial scans. Not compiled yet | {pt.get('CAD_RECIPE', 0)} |")
w(f"| NONE | No usable truth | {pt.get('NONE', 0)} |")
w("")
w("Each scan in the JSON lists all its `truth_sources`. Its `primary_truth` is the first that applies, in this order: human, AI-accepted, AI-legacy, CAD recipe. "
  "AI labels that did not pass are recorded as `AI_LABEL_PENDING_REVIEW (not truth)`.")
w("")

w("## Summary counts")
w("")
w("| Count | Value |")
w("|---|---|")
rows = [
    ("Unique SKUs with usable files (scan, CAD or drawing)", S["unique_part_ids_usable"]),
    ("Parts with a benchmark scan on disk", S["parts_with_benchmark_scan"]),
    ("Benchmark scans (deduplicated by sha256; format alternates, synthetic fixtures and redacted copies removed)", S["benchmark_scans"]),
    ("Parts with a scan and CAD", S["parts_with_benchmark_scan_and_cad"]),
    ("Parts with a scan and STEP (recipe-compilable)", S["parts_with_benchmark_scan_and_step"]),
    ("Parts with CAD of any kind (incl. CAD-only)", S["parts_with_cad_any"]),
    ("Staged or incomplete scans (PARTIAL_SCAN_1/2, PRELIM, STAGE0_BLANK)", S["scans_staged_or_incomplete"]),
    ("Scans with human gold lines", pt.get("GOLD_LINES_HUMAN", 0)),
    ("Scans with AI gold accepted (this run)", f"{ai['final_ai_gold_accepted']} ({ai['of_which_bend_lines']} with {ai['accepted_bend_lines_total']} lines; {ai['of_which_zero_bend_controls']} zero-bend)"),
    ("Scans needing human review (this run)", ai["needs_human_review"]),
    ("FRESH scans on disk / pending in email", f"{S['fresh_benchmark_scans_on_disk']} / {S['fresh_scans_not_on_disk']}"),
    ("Email attachments not on disk (all kinds / scans)", f"{S['not_on_disk_email_files']} / {S['not_on_disk_email_scans']}"),
    ("Scans with partial AI gold (label_complete=false)", ai.get("ai_gold_partial", 0)),
    ("Removed rows: format alternates / synthetic fixtures / not a scan (review11 + synthetic LIDAR demos) / redacted opaque copy",
     f"{S['non_benchmark_scan_rows'].get('FORMAT_ALTERNATE', 0)} / {S['non_benchmark_scan_rows'].get('EXCLUDED_SYNTHETIC_FIXTURE', 0)} / {S['non_benchmark_scan_rows'].get('EXCLUDED_NOT_A_SCAN', 0)} / {S['non_benchmark_scan_rows'].get('EXCLUDED_REDACTED_OPAQUE_COPY', 0)}"),
    ("Sealed rows dropped upstream (counted, never opened): scan / CAD / drawing",
     f"{S['excluded_sealed_counts_from_inventory']['scan']} / {S['excluded_sealed_counts_from_inventory']['cad']} / {S['excluded_sealed_counts_from_inventory']['drawing']}"),
    ("Eyal-50 quarantined files", S["eyal50_quarantined_files"]),
]
for k, v in rows:
    w(f"| {k} | {v} |")
w("")

w("## AI gold accepted (this run)")
w("")
w("| Scan | Part | File | Role | Lines | Verifier finding | lines.json sha256 | Studio label |")
w("|---|---|---|---|---|---|---|---|")
acc = []
for k, v in P.items():
    for s in v["scans"]:
        lab = s.get("ai_label_run")
        if lab and lab["final_status"] == "AI_GOLD_ACCEPTED":
            acc.append((k, s, lab))
acc.sort(key=lambda t: (t[2]["role"] != "bend_lines", t[0]))
for k, s, lab in acc:
    w(f"| `{s['scan_id']}` | {short_name(k, [s]) if k.startswith('UNKNOWN') else k} | {s['paths'][0].split('/')[-1]} | {lab['role']} | "
      f"{lab['n_bends_labelled']} | {lab['verifier']['summary']} | `{(lab['disk_at_manifest_time']['lines_json_sha256'] or '')[:12]}` | "
      f"{('`' + lab['studio_label']['asset_id'] + '`') if (lab.get('studio_label') or {}).get('written') else 'none: STL/OBJ only, needs a PLY export'} |")
w("")
w("Caveats before these go into scoring:")
w("- Zero-bend controls now carry `no_bends_present=true` in `pairs.json` and `lines.json` (fixed 2026-10-17). Use them only as false-positive controls, not in line-F1 denominators.")
w("- **Possible CAD reference meshes, kept out of scan-accuracy scoring:** " + ", ".join(f"`{x}`" for x in ai["possible_cad_reference_meshes_excluded_from_scan_accuracy"]) +
  ". These rolled shells come from `uploads/*/reference.*` (for `sc_2e584e9d1ffe`, `scan.ply` and `reference.ply` are byte-identical), so they may be CAD meshes rather than physical scans.")
w("- `sc_3bde9996f14d` (28870000_A_SCAN_1) is accepted as zero-bend. Its sibling full scan carries two shallow creases (about 24 and 18 deg), so check that this partial scan really shows no formed fold.")
w("- `sc_9567d0d3f33d`: `lines` was rerun on the on-record `{a,b}` pairs.json; the pairs and lines hashes now match, and the 7 bends are unchanged.")
w("- `sc_752e2bd54aad`: B2 covers about 72% of its fold. That is below the strict ~80% rule used in the 2026-10-17 re-verification; it was accepted earlier and was not re-verified.")
if sxd.get("excluded_not_a_scan"):
    w(f"- **Synthetic LIDAR demo files removed (2026-10-17):** {len(sxd['excluded_not_a_scan'])} `live_scans/*_LIDAR_DEMO_*.ply` files are resamples of `test_files/<part>/scan.stl` "
      "made by the legacy `/api/live-scan/demo/load` endpoint, not physical scans. They are `EXCLUDED_NOT_A_SCAN` and out of every count above; "
      f"{len(sxd.get('newly_excluded', []))} of them had been accepted as AI gold. Their label files stay on disk and scorers must skip them. "
      f"Evidence per scan: `{sxd['path']}`. Gold-set hits: {', '.join(sxd.get('gold_set_hits') or []) or 'none'}.")
for k, v in (rl3.get("scans") or {}).items():
    w(f"- `{k}` relabelled in round 3 ({v}); see `{(rl3.get('source_by_scan') or {}).get(k, rl3['path'])}`.")
w("- No studio label yet (STL or OBJ only; the studio indexes `.ply`): " + ", ".join(f"`{x['scan_id']}` ({x['n_bends']} lines)" for x in ai["accepted_without_studio_label"]) +
  ". Their gold is the verified `lines.json`; a PLY export with identical vertices is needed before `write-label`.")
w("- Scans that lived only in `~/Downloads` were copied byte-identically to `data/bend_corpus/incoming/2026-10-17_benchmark_v2_downloads/` so their labels have a stable studio asset id. "
  "`sc_361fc1049e39` has two label files, one per byte-identical copy (source-assets and the email copy under `data/`).")
w("")

w("## Review11: owner-delegated final verdicts (2026-10-17)")
w("")
w(f"The owner delegated the final verdict on the {r11.get('scans', 0)} scans that were left in NEEDS_HUMAN_REVIEW. Each got an investigator, two blind verifiers and a judge, all working from their own multi-angle renders "
  "and perpendicular sections, plus a final pass. Verdicts: `manifest_build/review11_verdicts_2026-10-17.json`. These are still AI-adjudicated, not human.")
w("")
w("| Scan | Part | File | Verdict | Real bends | Lines | Studio label | Reason |")
w("|---|---|---|---|---|---|---|---|")
r11rows = []
for k, v in P.items():
    for s in v["scans"]:
        lab = s.get("ai_label_run") or {}
        if lab.get("review11"):
            r11rows.append((k, s, lab))
r11rows.sort(key=lambda t: (t[2]["final_status"], t[0], t[1]["scan_id"]))
for k, s, lab in r11rows:
    st = lab.get("studio_label") or {}
    stud = f"`{st['asset_id']}`" if st.get("written") else ("none: " + (st.get("error") or st.get("reason") or "not written"))
    part = k if not k.startswith("UNKNOWN") else "UNKNOWN"
    status = lab["final_status"] + ("" if lab["review11"]["label_complete"] else " (label_complete=false)")
    w(f"| `{s['scan_id']}` | {part} | {s['paths'][0].split('/')[-1]} | {status} | {lab['review11']['real_bends']} | {lab['review11']['labelled_bends']} | {stud} | "
      f"{lab['reason'].split(': ', 1)[-1].replace('|', '/')} |")
w("")
for fl in r11.get("owner_flags", []):
    done = set(fl["scan_ids"]) <= set(sxd.get("newly_excluded") or [])
    w(f"- Flag ({', '.join('`' + x + '`' for x in fl['scan_ids'])}): {fl['issue']}" +
      (f" **Resolved 2026-10-17: all excluded as EXCLUDED_NOT_A_SCAN via `{sxd['path']}`.**" if done else ""))
w("")

w("## NEEDS_HUMAN_REVIEW: owner check in Bend Label Studio")
w("")
if not m["needs_human_review"]:
    w("None. Every scan that was waiting for review now has a review11 verdict (above).")
    w("")
_nhr_start = len(L)
w("Studio path: **`/bend-labeling`** (local: `http://127.0.0.1:3000/bend-labeling`, started with "
  "`PYTHON_BIN=/Users/idantal/ShermanQC/FullProject/.venv/bin/python scripts/run_bend_labeling_studio.sh`). "
  "The studio has no per-asset deep link. Type the file name shown below into the search box; the asset id is the one the studio computes "
  "(`sha256(source:relative_path)[:24]`). Saved labels go to `data/bend_corpus/annotations/human_bend_lines_v1/` in `source_ply_raw` "
  "coordinates. The studio shows no predictions, by design.")
w("")
w("Reason codes:")
w("- `LINES_INCOMPLETE`: the verifier saw renders, but some lines miss or are too short, or bends were missed. Fix it in the studio.")
w("- `EMPTY_LABEL_BUT_BENDS_VISIBLE`: no lines were drawn on a part that visibly has bends.")
w("- `TOOL_FAILED_NO_IMAGES`: nothing was ever rendered. Either label it in the studio, or rewrite pairs.json in list form, rerun `lines` (checking the Python exit status), and verify again.")
w("")
w("In the Missed column, `n/a` means not assessed. Counts marked (prep) come from the pre-label patch renders, not from line overlays.")
w("")
w("Some scans are not indexed by the studio. Those that exist only as `.stl`/`.obj` need a PLY export. Those in `~/Downloads` need copying under `data/` first; this manifest does not copy anything.")
w("")
w("| Scan | Part | Tier | Reason | Proposed | Missed | Studio |")
w("|---|---|---|---|---|---|---|")
for r in m["needs_human_review"]:
    st = r["studio"]
    if st and st.get("indexed"):
        loc = f"search `{st['search']}` (asset `{st['asset_id']}`)"
    else:
        loc = (st or {}).get("reason", "not indexed").replace("not indexed: ", "not indexed: ")
        loc = f"{loc}; path `{r['path']}`"
    tag = r["reason"]
    if r["input_status"] == "AI_GOLD_ACCEPTED":
        tag += ", was reported accepted"
    if r["lines_rendered_after_verification"]:
        tag += ", lines.json appeared after verification: re-verify"
    missed = "n/a" if r["missed_basis"] == "none" else f"{r['missed_bends_visible']}{' (prep)' if r['missed_basis'] == 'prep' else ''}"
    part = r["part_id"] if not r["part_id"].startswith("UNKNOWN") else "UNKNOWN"
    w(f"| `{r['scan_id']}` | {part} | {r['eval_tier']} | {tag} | {r['n_bends_proposed']} | {missed} | {loc} |")
w("")
w("`GAVE_UP_PREP_OR_TOOL_CANNOT_EXPRESS`: the labeler found that no patch pair from the current prep can give a landing line for every real fold. The usual cause is the global-plane gate in `lines.assign_full_resolution` dropping a warped plate. These need a re-prep, a tool change, or a hand label in the studio.")
w("")
w("### What to check (from the verifiers)")
w("")
for r in m["needs_human_review"]:
    part = r["part_id"] if not r["part_id"].startswith("UNKNOWN") else "UNKNOWN"
    w(f"**`{r['scan_id']}`** ({part}, `{r['path'].split('/')[-1]}`)")
    w("")
    for para in r["what_to_check"].split("\n"):
        if para.strip():
            w(f"> {para.strip()}")
            w(">")
    if L[-1] == ">":
        L.pop()
    w("")
w("Full verifier notes are also in `manifest_build/reverify_results_2026-10-17.json` and in the JSON under `parts.<part>.scans[].ai_label_run.what_to_check`.")
w("")

if not m["needs_human_review"]:
    del L[_nhr_start:]  # no review queue: drop the empty table and its instructions
w("## Parts with scans on disk")
w("")
w("The Truth column counts scans by primary truth. Pending = scan attachments still in email for that part.")
w("")
w("| Part | Split | Scans | Staged | Stage hints | CAD | Recipe | Flat | Drawings | Truth | Pending |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(P, key=lambda k: (k.startswith("UNKNOWN"), k)):
    v = P[k]
    bs = [s for s in v["scans"] if s["role"] == "benchmark_scan"]
    if not bs:
        continue
    tags = [t for t, f in (("Gold11", "in_gold11"), ("Gold37", "in_gold37"), ("physical15", "in_physical15")) if v[f]]
    split = v["split"] + (f" ({', '.join(tags)})" if tags else "")
    if any(s.get("eval_tier") == "FRESH_SCAN_OF_CONSUMED_PART" for s in bs):
        split += "; 1 fresh scan"
    hints = sorted({h for s in bs for h in s["stage_hints"]})
    tc = C.Counter(AB[s["primary_truth"]] for s in bs)
    truth = ", ".join(f"{n} {t}" for t, n in tc.most_common())
    pend = sum(1 for x in v["not_on_disk"] if x["kind"] == "scan")
    w(f"| {short_name(k, bs)} | {split} | {len(bs)} | {sum(s['staged_or_incomplete'] for s in bs)} | {', '.join(hints) or '-'} | "
      f"{','.join(v['cad']['formats']) or '-'} | {RS[v['cad']['recipe_status']]} | {len(v['flats'])} | {len(v['drawings'])} | {truth} | {pend or '-'} |")
w("")

w("## FRESH parts (no scans on disk yet)")
w("")
fr = sorted((k, v) for k, v in P.items() if v["split"] == "FRESH" and not k.startswith("UNKNOWN"))
rc = C.Counter(v["cad"]["recipe_status"] for k, v in fr)
w(f"{len(fr)} SKUs: {rc.get('STEP_ON_DISK_RECIPE_COMPILABLE_NOT_YET_COMPILED', 0)} with STEP, "
  f"{rc.get('NATIVE_CAD_ONLY_NEEDS_STEP_EXPORT', 0)} with native CAD only, and {rc.get('NO_CAD_ON_DISK', 0)} with no CAD on disk. "
  "Even without scans, the FRESH STEP files can test the compile gate (at least 80% of STEPs compile) once at the end. "
  "There are also 8 FRESH rows with no SKU (email drawings and images), listed in the JSON.")
w("")
w("FRESH parts whose scans are still in email:")
w("")
w("| Part | Scan attachments not on disk | Blocker | CAD on disk |")
w("|---|---|---|---|")
for k, v in fr:
    sc = [x for x in v["not_on_disk"] if x["kind"] == "scan"]
    if not sc:
        continue
    blk = ", ".join(sorted({x["download_status"].replace("NOT_DOWNLOADED_", "") for x in sc}))
    w(f"| {k} | {', '.join(x['filename'] for x in sc)} | {blk} | {RS[v['cad']['recipe_status']]} |")
w("")
w("All FRESH SKUs: " + ", ".join(k for k, _ in fr) + ".")
w("")

w("## Limits (read before quoting any number)")
w("")
w("- **AI labels are adjudicated by agents, not humans.** AI_GOLD_ACCEPTED means one labeler agent and one independent verifier agent agreed on rendered evidence. "
  f"No human has reviewed these {ai['final_ai_gold_accepted']} scans. Report results against them separately from GOLD_LINES_HUMAN.")
w("- Legacy Codex labels (42 scans) are also AI. Despite its `human_bend_lines_v1` directory name, that store has no human labels.")
w("- The first pass had 13 cases where accepted=true sat beside \"not verifiable\" notes. They were rerun and re-verified with renders on 2026-10-17; the strict gate still applies to every row.")
w("- Missed-bend counts taken from prep renders are estimates. `n/a` means not assessed, not zero.")
w("- Prior use comes from metadata (manifests, docs, hash matches). A file with no recorded use is `fresh` only to the extent those records are complete. Unknown provenance is treated as CONSUMED.")
w("- CAD recipes are listed as available but have not been compiled or reconciled with any scan. Staged scans need a per-bend FORMED/NOT_FORMED truth table (plan §10 amendment 7) before recipe truth applies to them.")
nod_pu = C.Counter(x["prior_use"] for v in P.values() for x in v["not_on_disk"] if x["kind"] == "scan")
w(f"- Email intake is incomplete: {S['not_on_disk_email_scans']} scan attachments are not on disk. By prior use: "
  + ", ".join(f"{k} {n}" for k, n in nod_pu.most_common()) + ". Only the fresh ones would add unseen evidence.")
w("- Eyal-50 (151 files) stays quarantined until the owner rules on its 20 dev / 30 locked-holdout split.")
w("")

w("## Custody")
w("")
w("- This manifest was built from metadata, directory listings and sha256 hashes. No scan geometry was read for it, and nothing from HOLDOUT10, sealed truth, opaque N001-N014, EHO4, one-shot/authorization artifacts or held-out manifests was opened. "
  "The upstream inventory dropped those rows by name and manifest membership (42 scans, 16 CAD files, 1 drawing counted only), and paths under `/opaque/` or `/evaluator_only/` are redacted.")
w("- Face/body scans and non-Sherman projects are not in the inventory.")
w("- Regenerate with `.venv/bin/python -I docs/production/runlog/benchmark_v2/manifest_build/build_manifest.py docs/production/runlog/benchmark_v2/manifest_build/verifier_results.tsv`, "
  "then run `build_doc.py` in the same directory. They read only `merged_inventory.json`, `label_prep/selection.json`, `label_prep/prep_runlog.json`, the `label_lines/` listings "
  "and `verifier_results.tsv` (the per-scan verifier verdicts transcribed from the workflow results).")
w("")

OUT.write_text("\n".join(L) + "\n")
print(OUT, len(L))
