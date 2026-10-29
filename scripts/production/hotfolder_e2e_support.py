"""Helpers for the hotfolder edge-case steps of ``run_e2e_real.mjs --hotfolder``.

Every output goes to the run's TEMPORARY work directory; inputs are the
read-only custody copies of the allowed 10839000 scans the real E2E already
uses, or synthetic E2 geometry. Nothing here reads the repo's data/ or any
sealed set, and nothing touches the engine.

Commands (each prints one JSON line):

- ``mirror SRC DST``: the scan mirrored in x (x, nx negated; triangle winding
  reversed), i.e. the mirror-image part.
- ``densify SRC DST --factor N --jitter MM``: every vertex repeated N times
  with a uniform jitter (vertex-only PLY), to exceed E3's validated density
  envelope (> 3,000,000 points).
- ``truncate SRC DST --frac F``: the first F of the bytes (an interrupted export).
- ``wrong-part WORK DST``: the nominal scan of a synthetic E2 HAT part
  (``generate_part_set``), i.e. another part, copied to DST.
- ``scan-report DATA_ROOT SHA...``: read-only report from the temp line DB:
  per sha the scan row (state, stored path and whether it exists, quarantine
  reason, resolution), its jobs (status, attempts, result summary, timings)
  and its events (kind, principal, payload).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np

_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4", "int32": "<i4",
          "uint": "<u4", "short": "<i2", "ushort": "<u2", "char": "i1"}


def _read_ply(path: Path) -> tuple[list[str], np.ndarray, np.ndarray | None, bytes]:
    raw = path.read_bytes()
    end = raw.index(b"end_header") + len(b"end_header")
    end = raw.index(b"\n", end) + 1
    header = raw[:end].decode("ascii").splitlines()
    assert header[0].strip() == "ply" and "binary_little_endian" in header[1], "binary little-endian PLY expected"
    n_v = n_f = 0
    vprops: list[tuple[str, str]] = []
    face_fmt = None
    cur = None
    for line in header:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            n_v, cur = int(parts[2]), "v"
        elif parts[:2] == ["element", "face"]:
            n_f, cur = int(parts[2]), "f"
        elif parts and parts[0] == "property" and cur == "v":
            vprops.append((parts[2], _TYPES[parts[1]]))
        elif parts and parts[0] == "property" and cur == "f":
            face_fmt = (_TYPES[parts[2]], _TYPES[parts[3]])
    vdt = np.dtype(vprops)
    verts = np.frombuffer(raw, dtype=vdt, count=n_v, offset=end).copy()
    faces = None
    rest = raw[end + vdt.itemsize * n_v:]
    if n_f and face_fmt is not None:
        fdt = np.dtype([("n", face_fmt[0]), ("v", face_fmt[1], 3)])
        faces = np.frombuffer(rest, dtype=fdt, count=n_f).copy()
        assert (faces["n"] == 3).all(), "triangle faces expected"
        rest = rest[fdt.itemsize * n_f:]
    return header, verts, faces, rest


def _write_ply(dst: Path, header: list[str], verts: np.ndarray, faces: np.ndarray | None, comment: str) -> None:
    out = [header[0], header[1], f"comment {comment}"] + [h for h in header[2:] if not h.startswith("end_header")] + ["end_header"]
    if faces is None:
        out = [h for h in out if not (h.startswith("element face") or h.startswith("property list"))]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(("\n".join(out) + "\n").encode("ascii"))
        fh.write(verts.tobytes())
        if faces is not None:
            fh.write(faces.tobytes())


def cmd_mirror(a: argparse.Namespace) -> dict:
    header, v, f, _ = _read_ply(Path(a.src))
    v["x"] = -v["x"]
    if "nx" in v.dtype.names:
        v["nx"] = -v["nx"]
    if f is not None:
        f["v"] = f["v"][:, ::-1].copy()
    _write_ply(Path(a.dst), header, v, f, "sherman-e2e mirrored in x (mirror-image part)")
    return {"vertices": int(len(v)), "faces": 0 if f is None else int(len(f))}


def cmd_densify(a: argparse.Namespace) -> dict:
    header, v, _f, _ = _read_ply(Path(a.src))
    rng = np.random.default_rng(7)
    out = np.repeat(v, a.factor)
    for k in ("x", "y", "z"):
        out[k] = out[k] + rng.uniform(-a.jitter, a.jitter, len(out)).astype(out[k].dtype)
    head = [h for h in header if not h.startswith("element vertex")]
    idx = next(i for i, h in enumerate(head) if h.startswith("property"))
    head.insert(idx, f"element vertex {len(out)}")
    _write_ply(Path(a.dst), head, out, None, f"sherman-e2e densified x{a.factor} jitter {a.jitter} mm")
    return {"vertices": int(len(out))}


def cmd_truncate(a: argparse.Namespace) -> dict:
    raw = Path(a.src).read_bytes()
    n = int(len(raw) * a.frac)
    Path(a.dst).write_bytes(raw[:n])
    return {"bytes": n, "of": len(raw)}


def cmd_wrong_part(a: argparse.Namespace) -> dict:
    from domains.production.engine.synthetic.cases import generate_part_set

    work = Path(a.work)
    meta = generate_part_set("HAT", "DEV", work, target_points=400000, ray_threads=1)
    set_dir = work / meta["set_id"]
    part_set = json.loads((set_dir / "part_set.json").read_text(encoding="utf-8"))
    nominal = next(s for s in part_set["scans"] if s["name"] == "nominal")
    shutil.copyfile(set_dir / nominal["dir"] / "scan.ply", a.dst)
    return {"set_id": meta["set_id"], "synthetic_sku": part_set["sku"], "family": "HAT"}


def cmd_scan_report(a: argparse.Namespace) -> dict:
    root = Path(a.data_root)
    db = root / "db" / "line.sqlite3"
    if not db.exists():
        cands = list(root.rglob("line.sqlite3"))
        db = cands[0]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    out: dict = {}
    for sha in a.sha:
        row = con.execute("SELECT * FROM scan_files WHERE sha256=?", (sha,)).fetchone()
        if row is None:
            out[sha] = None
            continue
        rel = row["stored_rel_path"]
        jobs = [dict(j) for j in con.execute(
            "SELECT job_id, execution_status, attempts, result_summary_json, failure_code, created_at, started_at, finished_at,"
            " declared_stage_json FROM jobs WHERE scan_sha256=? ORDER BY created_at", (sha,))]
        for j in jobs:
            j["result_summary"] = json.loads(j.pop("result_summary_json") or "null")
            j["declared_stage"] = json.loads(j.pop("declared_stage_json") or "null")
        events = [{"id": e["id"], "ts": e["ts"], "kind": e["kind"], "job_id": e["job_id"], "principal": json.loads(e["principal_json"] or "null"),
                   "payload": json.loads(e["payload_json"] or "{}")}
                  for e in con.execute("SELECT * FROM events WHERE scan_file_id=? ORDER BY id", (row["scan_file_id"],))]
        out[sha] = {
            "scan_file_id": row["scan_file_id"], "original_name": row["original_name"], "ingest_state": row["ingest_state"],
            "quarantine_reason": row["quarantine_reason"], "resolution": row["resolution"], "stored_rel_path": rel,
            "stored_exists": (root / rel).is_file(), "match": json.loads(row["match_json"] or "null"),
            "declared_stage": json.loads(row["declared_stage_json"] or "null"), "jobs": jobs, "events": events,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mirror"); p.add_argument("src"); p.add_argument("dst")
    p = sub.add_parser("densify"); p.add_argument("src"); p.add_argument("dst")
    p.add_argument("--factor", type=int, default=4); p.add_argument("--jitter", type=float, default=0.05)
    p = sub.add_parser("truncate"); p.add_argument("src"); p.add_argument("dst"); p.add_argument("--frac", type=float, default=0.6)
    p = sub.add_parser("wrong-part"); p.add_argument("work"); p.add_argument("dst")
    p = sub.add_parser("scan-report"); p.add_argument("data_root"); p.add_argument("sha", nargs="+")
    a = ap.parse_args(argv)
    fn = {"mirror": cmd_mirror, "densify": cmd_densify, "truncate": cmd_truncate, "wrong-part": cmd_wrong_part,
          "scan-report": cmd_scan_report}[a.cmd]
    print(json.dumps(fn(a), default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
