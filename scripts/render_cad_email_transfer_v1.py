"""Derived-only six-view single CAD review; meshes are display, not proof."""
from pathlib import Path
import argparse
import hashlib
import json
import io


def pinned_json(path, digest):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 40_000_000:
        raise ValueError("DERIVED_INPUT_BUDGET_OR_TYPE")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("DERIVED_PIN_MISMATCH")
    return json.loads(raw)


def render(evidence_path, evidence_sha, output, prediction_path=None, prediction_sha=None):
    if output.exists() or output.is_symlink() or output.with_suffix(".json").exists() or output.with_suffix(".json").is_symlink():
        raise ValueError("TERMINAL_OUTPUT_REUSE_REFUSED")
    evidence = pinned_json(evidence_path, evidence_sha)
    if (prediction_path is None) != (prediction_sha is None):
        raise ValueError("PREDICTION_PATH_AND_PIN_REQUIRED_TOGETHER")
    prediction = None if prediction_path is None else pinned_json(prediction_path, prediction_sha)
    if prediction is not None and prediction.get("input_sha256") != evidence["input_sha256"]:
        raise ValueError("PREDICTION_INPUT_MISMATCH")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    faces = evidence["faces"]
    if not isinstance(faces,list) or not 0<len(faces)<=10_000:
        raise ValueError("FACE_BUDGET")
    markers = [] if prediction is None else prediction["bend_axes"]
    if not isinstance(markers,list) or len(markers)>5_000:
        raise ValueError("MARKER_BUDGET")
    total_vertices=total_triangles=0
    for face in faces:
        mesh=face.get("display_mesh")
        if not mesh: continue
        vertices=np.asarray(mesh["vertices"],dtype=float)
        triangles=np.asarray(mesh["triangles_one_based"])
        if vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError("INVALID_MESH_VERTICES")
        if triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles) or not np.issubdtype(triangles.dtype,np.integer):
            raise ValueError("INVALID_MESH_TRIANGLES")
        if triangles.min()<1 or triangles.max()>len(vertices) or len(triangles)>200_000:
            raise ValueError("MESH_BUDGET_OR_INDICES")
        total_vertices+=len(vertices); total_triangles+=len(triangles)
        if total_vertices>2_000_000 or total_triangles>2_000_000:
            raise ValueError("AGGREGATE_MESH_BUDGET")
        if "area_centroid" in face:
            centroid=np.asarray(face["area_centroid"],dtype=float)
            if centroid.shape!=(3,) or not np.isfinite(centroid).all(): raise ValueError("INVALID_CENTROID")
    marker_points=[]
    for marker in markers:
        ends=np.asarray([marker["canonical_line_start"],marker["canonical_line_end"]],dtype=float)
        if ends.shape!=(2,3) or not np.isfinite(ends).all(): raise ValueError("INVALID_MARKER")
        marker_points.extend(ends)
    points = np.concatenate([np.asarray(f["display_mesh"]["vertices"], dtype=float)
                             for f in faces if f.get("display_mesh")])
    if not np.isfinite(points).all():
        raise ValueError("NONFINITE_DISPLAY_GEOMETRY")
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    outside_mesh_bounds=sum(bool(np.any(p<points.min(axis=0)) or np.any(p>points.max(axis=0))) for p in marker_points)
    display_points=points if not marker_points else np.concatenate([points,np.asarray(marker_points)])
    views = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(1,1,1),(-1,-1,-1)]
    fig, axs = plt.subplots(2,3,figsize=(15,10))
    for ax, direction in zip(axs.flat, views):
        z=np.asarray(direction,dtype=float); z/=np.linalg.norm(z)
        seed=np.eye(3)[int(np.argmin(np.abs(z)))]; x=np.cross(seed,z); x/=np.linalg.norm(x)
        basis=np.stack([x,np.cross(z,x),z],axis=1)
        polys,depths,colors,labels=[],[],[],[]
        for face in faces:
            mesh=face.get("display_mesh")
            if not mesh: continue
            v=(np.asarray(mesh["vertices"])-center)@basis
            t=np.asarray(mesh["triangles_one_based"],dtype=int)-1
            if len(t)>200_000 or t.min()<0 or t.max()>=len(v): raise ValueError("MESH_BUDGET_OR_INDICES")
            tri=v[t]; normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); length=np.linalg.norm(normal,axis=1)
            light=.55+.4*np.divide(np.abs(normal[:,2]),length,out=np.zeros(len(length)),where=length>0)
            surface=face["surface_class"]
            base=np.array([.86,.65,.28] if surface=="Cylinder" else [.43,.69,.75] if surface!="Plane" else [.77,.8,.83])
            colors.extend(base[None,:]*light[:,None]); polys.extend(tri[:,:,:2]); depths.extend(tri[:,:,2].mean(axis=1))
            if surface!="Plane" and "area_centroid" in face:
                labels.append(((np.asarray(face["area_centroid"])-center)@basis,str(face["id"])))
        order=np.argsort(depths)
        ax.add_collection(PolyCollection(np.asarray(polys)[order],facecolors=np.asarray(colors)[order],edgecolors="none"))
        projected=(display_points-center)@basis; lo,hi=projected[:,:2].min(axis=0),projected[:,:2].max(axis=0)
        pad=max(hi-lo)*.08+1e-6; ax.set_xlim(lo[0]-pad,hi[0]+pad); ax.set_ylim(lo[1]-pad,hi[1]+pad)
        for c,label in labels: ax.text(c[0],c[1],label,fontsize=5,color="#7b1414")
        for index,marker in enumerate(markers,1):
            endpoints=np.asarray([marker["canonical_line_start"],marker["canonical_line_end"]],dtype=float)
            if endpoints.shape!=(2,3) or not np.isfinite(endpoints).all(): raise ValueError("INVALID_MARKER")
            p=(endpoints-center)@basis
            ax.plot(p[:,0],p[:,1],color="#00b7d7",linewidth=2,linestyle="--")
            ax.text(*p[:,:2].mean(axis=0),f"B{index}",fontsize=7,color="#004a60")
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(f"View {direction}",fontsize=9)
    fig.suptitle("Email transfer CAD — native face IDs" + ("; V10 finite markers" if prediction is not None else "; no predictions"),fontsize=14)
    fig.text(.02,.015,"Orange: cylinders, not necessarily bends. Meshes aid review, not exact proof. Cyan: inferred finite interior markers, not full bend width.",fontsize=8)
    fig.tight_layout(rect=(0,.03,1,.96)); buffer=io.BytesIO(); fig.savefig(buffer,format="png",dpi=150); plt.close(fig)
    image=buffer.getvalue()
    if len(image)>40_000_000: raise ValueError("IMAGE_BUDGET")
    with output.open("xb") as stream: stream.write(image)
    manifest={"evidence_sha256":evidence_sha,"prediction_sha256":prediction_sha,"image_sha256":hashlib.sha256(image).hexdigest(),"prediction_access":prediction is not None,"marker_count":len(markers),"marker_endpoints_outside_mesh_bounds":outside_mesh_bounds,"marker_endpoints":[[m["canonical_line_start"],m["canonical_line_end"]] for m in markers],"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with output.with_suffix(".json").open("x") as stream: json.dump(manifest,stream,indent=2,sort_keys=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",required=True,type=Path); parser.add_argument("--evidence-sha256",required=True)
    parser.add_argument("--output",required=True,type=Path); parser.add_argument("--prediction",type=Path); parser.add_argument("--prediction-sha256")
    a=parser.parse_args(); render(a.evidence,a.evidence_sha256,a.output,a.prediction,a.prediction_sha256)
