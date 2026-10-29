"""Pinned released prediction/reference bundle for derived visual comparison."""
from pathlib import Path
import argparse
import hashlib
import json


def read(path,pin,limit=40_000_000):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>limit:
        raise ValueError("UNSAFE_DERIVED_INPUT")
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pin:
        raise ValueError("DERIVED_INPUT_PIN_MISMATCH")
    return json.loads(raw)


def release(directory,pin):
    post=read(directory/"POST_AUDIT.json",pin)
    if post.get("decision")!="ACCEPT_DIAGNOSTIC_WITH_LIMITATIONS" or post.get("independent_non_author") is not True or post.get("blocking_findings"):
        raise ValueError("PREDICTION_POST_RELEASE_REQUIRED")
    frozen=read(directory/"FROZEN.json",post["frozen_sha256"])
    cases=frozen.get("cases",[])
    if len(cases)!=20 or len({case["sha256"] for case in cases})!=20:
        raise ValueError("TWENTY_UNIQUE_SOURCE_CASES_REQUIRED")
    return post,frozen


def build(reference,reference_pin,baseline,baseline_post_pin,challenger,challenger_post_pin,out):
    refs=read(reference,reference_pin)
    if refs.get("reference_frozen") is not True or refs.get("independent_reviews_complete") is not True:
        raise ValueError("FROZEN_REFERENCE_REQUIRED")
    releases={"baseline":release(baseline,baseline_post_pin),"challenger":release(challenger,challenger_post_pin)}
    directories={"baseline":baseline,"challenger":challenger}
    if len(refs["parts"])!=20 or {p["part_id"] for p in refs["parts"]}!={f"case-{i:02d}" for i in range(1,21)}:
        raise ValueError("ALL_TWENTY_REFERENCES_REQUIRED")
    parts=[]
    for ref in sorted(refs["parts"],key=lambda p:p["part_id"]):
        name=ref["part_id"]
        part={key:ref[key] for key in ("part_id","source_sha256","evidence_sha256","reference_status","face_ids","reference_bends","unknown_face_ids")}
        for stage,(post,frozen) in releases.items():
            if frozen["cases"][int(name[-2:])-1]["sha256"]!=ref["source_sha256"]:
                raise ValueError("CASE_SOURCE_BINDING_DISAGREEMENT")
            pins=post["evidence"]["derived_artifact_sha256"]
            result=read(directories[stage]/name/"RESULT.json",pins[name+"/RESULT.json"])
            if result["source_sha256"]!=ref["source_sha256"]:
                raise ValueError("RESULT_SOURCE_DISAGREEMENT")
            predictions=[]
            if result["extraction_status"]=="COMPLETE":
                extraction=read(directories[stage]/name/"EXTRACTION.json",pins[name+"/EXTRACTION.json"])
                for row in extraction["bend_axes"]:
                    predictions.append({"prediction_id":row["feature_id"],"face_ids":row["face_indices"],
                        "line_start":row["canonical_line_start"],"line_end":row["canonical_line_end"],
                        "marker_extent_semantics":row["domain_receipt"].get("marker_extent_semantics","LEGACY_RECTANGULAR_SUPPORT")})
                status="ok"
            else:
                status="failed"
            part[stage]={"status":status,"predictions":predictions}
        parts.append(part)
    bundle={"schema":"cad_bend_visual_bundle_v1","reference_sha256":reference_pin,
        "baseline_post_sha256":baseline_post_pin,"challenger_post_sha256":challenger_post_pin,
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_overlays_are_not_inference":True,"parts":parts}
    with out.open("x") as stream:
        json.dump(bundle,stream,sort_keys=True,indent=2,allow_nan=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("reference","baseline","challenger","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("reference-sha256","baseline-post-sha256","challenger-post-sha256"):
        parser.add_argument("--"+name,required=True)
    a=parser.parse_args()
    build(a.reference,a.reference_sha256,a.baseline,a.baseline_post_sha256,a.challenger,a.challenger_post_sha256,a.out)
