"""make_jobs.py OUT_DIR: jobs for the prevalence study (G1 50 + A1 26 natives, mirrors, wrong-part swaps), all with
E3M_FORCE=1 (the diagnostic copy: live decision, mirrored fit always measured). Inputs come only from the e3stab
manifest's G1 and A1 entries (COH excluded)."""
import json, sys
from pathlib import Path
S = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
man = json.load(open(S / "e3stab/manifest.json"))
E = {e["id"]: e for e in man["unique_jobs"] if e["corpus"] in ("G1", "A1") and e.get("compile_status") != "BLOCKED"}
info = {}
for cid, e in E.items():
    if e["corpus"] == "G1":
        sc = json.load(open(S / f"zsprod/g1c/cases/{cid}.json"))["score"]["case"]
        cp = json.load(open(e["compiled_dir"] + "/compiled_part.json"))
        st = sc.get("expected_stage_model_id")
        sym = [s["symmetry"] for s in cp["stages"] if s["stage_model_id"] == st]
        mir = min([c["rms_mm"] for c in sym[0]["candidates"] if c["kind"] == "MIRROR"], default=None) if sym else None
        info[cid] = dict(scenario=sc.get("scenario"), stage=st, mirror_rms=mir)
    else:
        info[cid] = dict(scenario="A1", stage=None, mirror_rms=None)
F = {"E3M_FORCE": "1"}


def job(i, e, mirror=None, decl=None):
    d = E[decl] if decl else e
    return dict(id=i, scan=e["scan"], cdir=d["compiled_dir"], recipe=d["recipe"], mirror=mirror, env=F)


native, mirrors, swaps = [], [], []
for cid, e in E.items():
    native.append(job(f"N_{cid}", e))
    inf = info[cid]
    if e["corpus"] == "G1":
        if inf["scenario"] in ("STAGE_0", "MIRRORED", "WRONG_SKU") or inf["stage"] == "S0":
            continue  # flat blanks are their own mirror image; the existing wrong/mirrored cases stay native only
        axes = ("x", "y", "z") if (inf["mirror_rms"] is None or inf["mirror_rms"] > 0.3) else ("x",)
    else:
        axes = ("x", "y", "z")
    for a in axes:
        mirrors.append(job(f"M{a}_{cid}", e, mirror=a))
A1_PAIRS = [("pc_71cf0b216f43", "pc_d1732de18023"), ("pc_71cf0b216f43", "pc_07062eee17aa"), ("pc_d1732de18023", "pc_71cf0b216f43"),
            ("pc_2714fb7e0c0b", "pc_e080ae094f35"), ("pc_e080ae094f35", "pc_07062eee17aa"), ("pc_d45186653eed", "pc_6eba02723d4b"),
            ("pc_6eba02723d4b", "pc_d45186653eed"), ("pc_03395097cd96", "pc_70b39f530f00"), ("pc_70b39f530f00", "pc_03395097cd96"),
            ("pc_38091ac91d9d", "pc_9b9fd762f361"), ("pc_9b9fd762f361", "pc_38091ac91d9d"), ("pc_ff1667a2a80a", "pc_828bc34d9c8b"),
            ("pc_828bc34d9c8b", "pc_ff1667a2a80a"), ("pc_777c64c88967", "pc_07062eee17aa"), ("pc_361fc1049e39", "pc_38091ac91d9d"),
            ("pc_71cf0b216f43", "pc_777c64c88967")]
G1_PAIRS = [("E2-DEV-TD-000", "E2-DEV-TD-003"), ("E2-DEV-TD-003", "E2-DEV-TD-000"), ("E2-DEV-PN-000", "E2-DEV-PN-005"),
            ("E2-DEV-PN-003", "E2-DEV-PN-001"), ("E2-DEV-HT-000", "E2-DEV-HT-003"), ("E2-DEV-CR-000", "E2-DEV-CR-003"),
            ("E2-DEV-UC-000", "E2-DEV-UC-003"), ("E2-DEV-ZS-000", "E2-DEV-ZS-005"), ("E2-DEV-LB-001", "E2-DEV-LB-005"),
            ("E2-DEV-LB-005", "E2-DEV-LB-001"), ("E2-DEV-UC-001", "E2-DEV-UC-000"), ("E2-DEV-TD-001", "E2-DEV-TD-002"),
            ("E2-DEV-UC-003", "E2-DEV-ZS-005"), ("E2-DEV-HT-003", "E2-DEV-CR-005")]
for x, y in A1_PAIRS + G1_PAIRS:
    swaps.append(job(f"W_{x}__as__{y}", E[x], decl=y))
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for name, js in (("native", native), ("mirror", mirrors), ("swap", swaps)):
    with open(out / f"jobs_{name}.jsonl", "w") as fh:
        for j in js: fh.write(json.dumps(j) + "\n")
json.dump(info, open(out / "case_info.json", "w"), indent=1)
print(len(native), len(mirrors), len(swaps))
