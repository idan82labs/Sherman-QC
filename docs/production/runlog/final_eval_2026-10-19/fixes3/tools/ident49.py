"""DIAGNOSTIC (consumed FRESH 49207000): mapped CAD nominal lines vs every gold line, per bend."""
import json, sys, math
from pathlib import Path
import numpy as np
REPO = Path("/Users/idantal/ShermanQC/FullProject"); sys.path[:0] = [str(REPO), str(REPO / "backend")]
from domains.production.engine.eval.engine_output import decoded_stage, frame_to_raw, load_compiled, load_engine_output, stage_locus
cases = {c["case"]: c for c in json.loads(Path(sys.argv[1]).read_text())}
c = cases["fr_6fa053514df8"]; ed = Path(sys.argv[2])
eo = load_engine_output(str(ed)); comp = load_compiled(Path(c["compiled"]) / "compiled_part.json")
res = json.loads((ed / "result.json").read_text()); st = decoded_stage(eo.result, comp)
gold = c["gold"]
def metrics(a0, a1, g0, g1):
    da, dg = a1 - a0, g1 - g0; ang = math.degrees(math.acos(min(1, abs(da @ dg) / np.linalg.norm(da) / np.linalg.norm(dg))))
    u = dg / np.linalg.norm(dg); mid = (a0 + a1) / 2; r = mid - g0; off = np.linalg.norm(r - (r @ u) * u)
    return round(ang, 1), round(float(off), 1)
for b in res["bends"]:
    L = frame_to_raw(eo.result, stage_locus(comp, st, b["bend_uid"], "outer_apex"))
    a0, a1 = np.asarray(L[0], float), np.asarray(L[1], float)
    ms = sorted(((metrics(a0, a1, np.asarray(g["line_start"]), np.asarray(g["line_end"])), g["bend_id"]) for g in gold), key=lambda x: (x[0][0] > 5, x[0][1]))
    ang = next((ch["actual"] for ch in b["characteristics"] if ch["kind"] == "INCLUDED_ANGLE"), None)
    print(b["label"], "angle", None if ang is None else round(ang, 2), "decisions", [ch["decision"][:4] for ch in b["characteristics"]], "nearest gold:", ms[:2])
print("--- measured apex (engine) and ICP-mapped nominal vs gold (frozen matcher), per bend")
from domains.production.engine.eval.line_metric import match_lines
T = np.asarray(res["inspection_validity"]["registration"]["raw_to_frame"], float); Tinv = np.linalg.inv(T)
z = np.load("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/fixes2/icp49207.npz")
R, t = z["R"], z["t"]  # cad -> raw
meas, icp = [], []
for b in res["bends"]:
    d = b.get("diagnostics") or {}
    if "outer_apex_start_x" in d:
        p0 = np.array([d["outer_apex_start_x"], d["outer_apex_start_y"], d["outer_apex_start_z"], 1.0])
        p1 = np.array([d["outer_apex_end_x"], d["outer_apex_end_y"], d["outer_apex_end_z"], 1.0])
        meas.append({"bend_id": b["label"], "line_start": list((Tinv @ p0)[:3]), "line_end": list((Tinv @ p1)[:3])})
    L = stage_locus(comp, st, b["bend_uid"], "outer_apex")
    icp.append({"bend_id": b["label"], "line_start": list(R @ np.asarray(L[0], float) + t), "line_end": list(R @ np.asarray(L[1], float) + t)})
for name, preds in (("measured apex", meas), ("ICP nominal", icp)):
    m = match_lines(gold, preds)
    print(name, "tp/fp/fn", m["true_positive"], m["false_positive"], m["false_negative"], sorted((x["prediction_id"], x["gold_id"], round(x.get("transverse_offset_mm", -1), 2)) for x in m["matches"]))
