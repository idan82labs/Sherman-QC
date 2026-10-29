"""Render only released terminal corpus outputs, never CAD/native inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import textwrap

MAX_BYTES = 40_000_000


def read(path):
    if path.is_symlink():
        raise ValueError("DERIVED_SYMLINK_REFUSED")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES+1)
    if len(raw) > MAX_BYTES:
        raise ValueError("DERIVED_BYTE_BUDGET")
    return raw


def load(path):
    return json.loads(read(path))


def parse_folded_ply(raw):
    """Parse only the bounded exact binary display format emitted by _folded_ply."""
    marker = b"end_header\n"
    end = raw.find(marker, 0, 4096)
    if end < 0:
        raise ValueError("PLY_HEADER_REFUSED")
    header = raw[:end+len(marker)].decode("ascii").splitlines()
    expected = ["ply", "format binary_little_endian 1.0", None,
                "property float x", "property float y", "property float z", None,
                "property list uchar int vertex_indices", "end_header"]
    if len(header) != len(expected) or any(want is not None and got != want for got, want in zip(header, expected)):
        raise ValueError("PLY_SCHEMA_REFUSED")
    if not header[2].startswith("element vertex ") or not header[6].startswith("element face "):
        raise ValueError("PLY_COUNTS_REFUSED")
    nv, nt = int(header[2].split()[-1]), int(header[6].split()[-1])
    if not 0 < nv <= 600_000 or not 0 < nt <= 200_000:
        raise ValueError("PLY_GEOMETRY_BUDGET")
    body = raw[end+len(marker):]
    if len(body) != nv*12+nt*13:
        raise ValueError("PLY_LENGTH_MISMATCH")
    vertices = list(struct.iter_unpack("<fff", body[:nv*12]))
    triangles = []
    for count, a, b, c in struct.iter_unpack("<Biii", body[nv*12:]):
        if count != 3 or min(a,b,c) < 0 or max(a,b,c) >= nv:
            raise ValueError("PLY_TRIANGLE_REFUSED")
        triangles.append((a,b,c))
    return vertices, triangles


def validate_mesh(vertices, triangles):
    import numpy as np
    v, t = np.asarray(vertices, dtype=float), np.asarray(triangles)
    if (v.ndim != 2 or v.shape[1] != 3 or not 0 < len(v) <= 600_000 or not np.isfinite(v).all()
            or t.ndim != 2 or t.shape[1] != 3 or not 0 < len(t) <= 200_000
            or not np.issubdtype(t.dtype, np.integer) or t.min() < 0 or t.max() >= len(v)):
        raise ValueError("DERIVED_MESH_INVALID")
    return v, t


def flat_schema_adapter(vertices, triangles):
    """Accept the published flat one-dimensional arrays without coercing indices."""
    import numpy as np
    v, t = np.asarray(vertices), np.asarray(triangles)
    if v.ndim != 1 or t.ndim != 1 or not len(v) or not len(t) or len(v)%3 or len(t)%3:
        raise ValueError("FLAT_ARRAY_SHAPE_INVALID")
    if not np.issubdtype(t.dtype,np.integer) or np.issubdtype(t.dtype,np.bool_):
        raise ValueError("FLAT_TRIANGLE_INDICES_NOT_INTEGERS")
    return validate_mesh(v.reshape(-1,3),t.reshape(-1,3))


def flat_obj_bytes(vertices,triangles,bend_lines=()):
    v,t=validate_mesh(vertices,triangles)
    lines=["# Selected native CAD skin preview; not a neutral-axis manufacturing blank",
           "# Units assumed mm, unverified; physical accuracy UNSCORED"]
    lines.extend("v "+" ".join(repr(float(c)) for c in point) for point in v)
    lines.extend("f "+" ".join(str(int(c)+1) for c in face) for face in t)
    for line in bend_lines:
        feature=str(line["feature_id"]).replace("\n"," ").replace("\r"," ")
        lines.append("# bend "+feature+" line_start="+json.dumps(line["line_start"],allow_nan=False)+
                     " line_end="+json.dumps(line["line_end"],allow_nan=False))
    raw=("\n".join(lines)+"\n").encode("ascii")
    if len(raw)>MAX_BYTES:
        raise ValueError("OBJ_EXPORT_BYTE_BUDGET")
    return raw


def failure_text(reason):
    sanitized=" ".join(str(reason).split())
    return "\n".join(textwrap.wrap(sanitized[:180],width=38,break_long_words=True,break_on_hyphens=False))


def validate_post_release(attempt,post_path,post_sha256):
    post_raw=read(post_path)
    post=json.loads(post_raw)
    hashes=post.get("evidence",{}).get("derived_artifact_sha256",{})
    frozen_hash=hashlib.sha256(read(attempt / "FROZEN.json")).hexdigest()
    terminal_hash=hashlib.sha256(read(attempt / "CAMPAIGN_TERMINAL.json")).hexdigest()
    if (hashlib.sha256(post_raw).hexdigest()!=post_sha256
            or post.get("decision")!="ACCEPT_DIAGNOSTIC_WITH_LIMITATIONS"
            or post.get("independent_non_author") is not True or post.get("blocking_findings")!=[]
            or post.get("frozen_sha256")!=frozen_hash or hashes.get("FROZEN.json")!=frozen_hash
            or hashes.get("CAMPAIGN_TERMINAL.json")!=terminal_hash):
        raise ValueError("INDEPENDENT_POST_RELEASE_REQUIRED")
    # Only renderer inputs are read, never CAD snapshots or audit-log/native files.
    for i in range(20):
        folder=attempt/f"case-{i+1:02d}"
        if folder.is_symlink():
            raise ValueError("CASE_SYMLINK_REFUSED")
        for name in ("RESULT.json","CASE_TERMINAL.json","EXTRACTION.json","CANDIDATES.json","FLAT.json","folded.ply"):
            relative=f"case-{i+1:02d}/{name}"
            path=attempt/relative
            if path.exists() or relative in hashes:
                if relative not in hashes or hashlib.sha256(read(path)).hexdigest()!=hashes[relative]:
                    raise ValueError("DERIVED_POST_HASH_MISMATCH:"+relative)
    return frozen_hash


def nominal_lines(folder):
    if (folder / "CANDIDATES.json").exists():
        return load(folder / "CANDIDATES.json")
    if (folder / "EXTRACTION.json").exists():
        extraction = load(folder / "EXTRACTION.json")
        return [{"feature_id": f"AX{i+1}", "line_start": axis["canonical_line_start"],
                 "line_end": axis["canonical_line_end"]}
                for i,axis in enumerate(extraction.get("bend_axes",[]))]
    return []


def panel(ax, vertices, triangles, lines, flat):
    import numpy as np
    from matplotlib.collections import PolyCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    v, t = validate_mesh(vertices, triangles)
    if flat:
        ax.add_collection(PolyCollection(v[t, :2], facecolors="#c5d9e5", edgecolors="#647b89", linewidths=.08))
        ax.set_xlim(v[:,0].min(), v[:,0].max())
        ax.set_ylim(v[:,1].min(), v[:,1].max())
        ax.set_aspect("equal")
    else:
        ax.add_collection3d(Poly3DCollection(v[t], facecolors="#c5d9e5", edgecolors="#647b89", linewidths=.04, alpha=.9))
        center = (v.min(axis=0)+v.max(axis=0))/2
        radius = max(float(np.ptp(v, axis=0).max())/2, 1e-6)
        ax.set_xlim(center[0]-radius,center[0]+radius)
        ax.set_ylim(center[1]-radius,center[1]+radius)
        ax.set_zlim(center[2]-radius,center[2]+radius)
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=25, azim=-55)
    for line in lines:
        segment = np.asarray([line["line_start"], line["line_end"]], dtype=float)
        if segment.shape != (2,3) or not np.isfinite(segment).all():
            raise ValueError("DERIVED_BEND_LINE_INVALID")
        if flat:
            ax.plot(segment[:,0],segment[:,1], color="#c33124", linewidth=1.4)
            ax.text(*segment.mean(axis=0)[:2], str(line.get("display_label", line["feature_id"])), fontsize=7, color="#a32118")
        else:
            ax.plot(segment[:,0],segment[:,1],segment[:,2], color="#c33124", linewidth=1.7)
            ax.text(*segment.mean(axis=0), str(line.get("display_label", line["feature_id"])), fontsize=7, color="#a32118")
    ax.set_axis_off()


def render(attempt, out, post_path, post_sha256):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if attempt.is_symlink() or out.is_symlink():
        raise ValueError("DIRECTORY_SYMLINK_REFUSED")
    frozen, terminal = load(attempt / "FROZEN.json"), load(attempt / "CAMPAIGN_TERMINAL.json")
    frozen_hash=validate_post_release(attempt,post_path,post_sha256)
    if len(frozen["cases"]) != 20 or sum(len(c["aliases"]) for c in frozen["cases"]) != 23:
        raise ValueError("CORPUS_DENOMINATOR_CHANGED")
    # A terminal incomplete campaign can be rendered honestly, including absences.
    if terminal.get("status") not in {"COMPLETE", "INCOMPLETE"}:
        raise ValueError("TERMINAL_RELEASE_REQUIRED")
    out.mkdir(parents=False, exist_ok=False)
    manifest = {"accuracy": "UNSCORED", "manufacturing_ready": False,
                "input_frozen_sha256": frozen_hash,"post_sha256":post_sha256,
                "unique_models": 20, "alias_paths": 23, "cases": [], "visual_review_required": []}
    for i, case in enumerate(frozen["cases"]):
        folder = attempt / f"case-{i+1:02d}"
        if folder.is_symlink():
            raise ValueError("CASE_SYMLINK_REFUSED")
        result = load(folder / "RESULT.json") if (folder / "RESULT.json").exists() else {}
        receipt = load(folder / "CASE_TERMINAL.json") if (folder / "CASE_TERMINAL.json").exists() else {"status": "MISSING"}
        fig = plt.figure(figsize=(10,5), dpi=130)
        folded = fig.add_subplot(121, projection="3d")
        flat_ax = fig.add_subplot(122)
        errors = []
        flat_available = False
        export_path=None
        lines = nominal_lines(folder)
        labels = {row["feature_id"]: f"B{j+1}" for j,row in enumerate(lines)}
        lines = [{**row, "display_label": labels[row["feature_id"]]} for row in lines]
        for mode, ax in (("folded", folded), ("flat", flat_ax)):
            try:
                if mode == "folded":
                    v,t = parse_folded_ply(read(folder / "folded.ply"))
                    panel(ax,v,t,lines,False)
                else:
                    payload = load(folder / "FLAT.json")
                    if payload.get("status") != "AVAILABLE" or payload.get("source_sha256") != case["sha256"]:
                        raise ValueError(payload.get("reason") or "FLAT_UNAVAILABLE_OR_BINDING_INVALID")
                    vertices,triangles=flat_schema_adapter(payload["vertices"],payload["triangles"])
                    panel(ax,vertices,triangles,[{**row, "display_label": labels.get(row["feature_id"], row["feature_id"])} for row in payload["bend_lines"]],True)
                    export_path=f"case-{i+1:02d}-flat.obj"
                    with (out/export_path).open("xb") as stream:
                        stream.write(flat_obj_bytes(vertices,triangles,payload["bend_lines"]))
                    flat_available = True
            except (OSError,ValueError,KeyError,TypeError) as exc:
                errors.append({"panel": mode, "reason": str(exc)[:500]})
                ax.set_axis_off()
                text = f"{mode.upper()} UNAVAILABLE\n{failure_text(result.get(f'{mode}_reason') or exc)}"
                if mode == "folded":
                    if lines:
                        import numpy as np
                        coordinates = []
                        for line in lines:
                            segment = np.asarray([line["line_start"],line["line_end"]],dtype=float)
                            if segment.shape != (2,3) or not np.isfinite(segment).all():
                                raise ValueError("AXES_ONLY_LINE_INVALID")
                            coordinates.extend(segment)
                            ax.plot(*segment.T,color="#c33124",linewidth=1.4)
                        points = np.asarray(coordinates)
                        for dim,setter in enumerate((ax.set_xlim,ax.set_ylim,ax.set_zlim)):
                            low,high=points[:,dim].min(),points[:,dim].max()
                            padding=max((high-low)*.05,1.)
                            setter(low-padding,high+padding)
                        ax.view_init(elev=25,azim=-55)
                        text="AXES ONLY — CANNOT ASSESS SHAPE\n"+text
                    ax.text2D(.03,.5,text,transform=ax.transAxes,fontsize=9,wrap=True)
                else:
                    ax.text(.03,.5,text,transform=ax.transAxes,fontsize=9,wrap=True)
            ax.set_title(mode.upper()+" — selected-skin preview" if mode == "flat" else "FOLDED — native display mesh",fontsize=10)
        title = f"{i+1:02d} | {case['relative_path']} | aliases {len(case['aliases'])} | extraction {result.get('extraction_status','UNKNOWN')} / axes {result.get('axis_count','?')}"
        fig.suptitle(title,fontsize=10)
        fig.text(.02,.02,"DIAGNOSTIC • units assumed mm • accuracy UNSCORED • not manufacturing ready",fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.94))
        filename = f"case-{i+1:02d}-folded-flat.png"
        fig.savefig(out / filename)
        plt.close(fig)
        row = {"index": i, "source_sha256": case["sha256"], "aliases": case["aliases"], "image": filename,
               "result": result, "receipt": receipt, "render_errors": errors, "flat_rendered": flat_available,
               "display_label_by_feature_id": labels,
               "flat_export":export_path if flat_available else None,
               "flat_export_status":"AVAILABLE" if flat_available else "UNAVAILABLE"}
        manifest["cases"].append(row)
        if flat_available:
            manifest["visual_review_required"].append(filename)
    sheet, axes = plt.subplots(5,4,figsize=(24,15),dpi=100)
    for ax,row in zip(axes.flat,manifest["cases"]):
        ax.imshow(plt.imread(out / row["image"]))
        ax.set_axis_off()
    sheet.suptitle("All 20 unique CADs / 23 source aliases — actual derived outputs; failures retained",fontsize=15)
    sheet.tight_layout(rect=(0,0,1,.97))
    sheet.savefig(out / "CONTACT_SHEET.png")
    plt.close(sheet)
    for page in range(4):
        sheet,axes=plt.subplots(5,1,figsize=(12,30),dpi=120)
        for ax,row in zip(axes,manifest["cases"][page*5:(page+1)*5]):
            ax.imshow(plt.imread(out / row["image"]))
            ax.set_axis_off()
        sheet.suptitle(f"Corpus contact sheet {page+1}/4 — models {page*5+1:02d}–{page*5+5:02d}; failures retained",fontsize=13)
        sheet.tight_layout(rect=(0,0,1,.98))
        sheet.savefig(out / f"CONTACT_PAGE_{page+1}.png")
        plt.close(sheet)
    manifest["images_sha256"] = {p.name: hashlib.sha256(read(p)).hexdigest() for p in sorted(out.glob("*.png"))}
    manifest["exports_sha256"] = {p.name: hashlib.sha256(read(p)).hexdigest() for p in sorted(out.glob("*.obj"))}
    manifest["output_sha256"]={**manifest["images_sha256"],**manifest["exports_sha256"]}
    manifest["manifest_self_hash"]="External SHA-256 required; self-reference excluded"
    with (out / "GALLERY.json").open("x") as stream:
        json.dump(manifest,stream,sort_keys=True,indent=2)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--post",required=True,type=Path)
    parser.add_argument("--post-sha256",required=True)
    args=parser.parse_args()
    render(args.attempt,args.out,args.post,args.post_sha256)


if __name__ == "__main__":
    main()
