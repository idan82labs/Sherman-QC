"""DIAGNOSTIC (consumed FRESH 38968000): per-twin readings behind the twin-agreement decision."""
import json, sys, hashlib
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject"); sys.path[:0] = [str(REPO), str(REPO / "backend")]
import domains.production.engine.inspect.engine as E
from domains.production.engine.inspect import InspectOptions, inspect_scan, load_config, load_inputs
c = {x["case"]: x for x in json.loads(Path(sys.argv[1]).read_text())}["fr_c0ce5ecd2ba3"]
cap = {}
orig_agree, orig_spec = E._agree, E._matched_spec_ok
def agree(part, cid, rs):
    ok = orig_agree(part, cid, rs); cap.setdefault("agree", []).append((cid, rs, ok)); return ok
def spec(part, uid, kind, perms):
    ok = orig_spec(part, uid, kind, perms); cap.setdefault("spec", []).append((uid, kind, ok)); return ok
E._agree, E._matched_spec_ok = agree, spec
inp = load_inputs(Path(c["compiled"]), Path(c["recipe"]), Path(c["scan"]), hashlib.sha256(Path(c["scan"]).read_bytes()).hexdigest())
res = inspect_scan(inp, Path("unused"), load_config(None), InspectOptions(write_outputs=False)).result
lab = {b.bend_uid: b.label for b in res.bends}
for uid, kind, ok in cap.get("spec", []):
    if not ok: print("SPEC DIFFERS", lab[uid], kind)
for cid, rs, ok in cap.get("agree", []):
    print(lab[cid.split("/")[0]], cid.split("/", 1)[1], "agree" if ok else "WITHHELD", [tuple(round(v, 3) if isinstance(v, float) else v for v in r) for r in rs])
