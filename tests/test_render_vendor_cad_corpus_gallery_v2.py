"""Renderer source-only fixtures: no CAD, scans, OCP or native loading."""
import importlib.util
from pathlib import Path
import struct
import json
import pytest

SOURCE=Path(__file__).parents[1]/"scripts/render_vendor_cad_corpus_gallery_v2.py"
spec=importlib.util.spec_from_file_location("gallery_v2",SOURCE)
gallery=importlib.util.module_from_spec(spec)
spec.loader.exec_module(gallery)


def ply(count=3,indices=(0,1,2)):
    header=b"ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n"
    return header+struct.pack("<9f",0,0,0,1,0,0,0,1,0)+struct.pack("<Biii",count,*indices)


def test_exact_emitted_folded_ply_schema():
    vertices,triangles=gallery.parse_folded_ply(ply())
    assert vertices == [(0,0,0),(1,0,0),(0,1,0)]
    assert triangles == [(0,1,2)]


@pytest.mark.parametrize("raw",[b"fake",ply()[:-1],ply()+b"extra",ply(count=4),ply(indices=(0,1,3)),ply().replace(b"binary_little_endian",b"ascii              ")])
def test_malformed_or_unsupported_ply_refused(raw):
    with pytest.raises(ValueError):
        gallery.parse_folded_ply(raw)


def test_derived_symlink_refused(tmp_path):
    target=tmp_path/"fixture"
    target.write_bytes(b"fixture")
    link=tmp_path/"link"
    link.symlink_to(target)
    with pytest.raises(ValueError,match="SYMLINK"):
        gallery.read(link)


def test_derived_read_budget(tmp_path,monkeypatch):
    target=tmp_path/"fixture"
    target.write_bytes(b"fixture")
    monkeypatch.setattr(gallery,"MAX_BYTES",2)
    with pytest.raises(ValueError,match="BYTE_BUDGET"):
        gallery.read(target)


def test_no_cad_native_imports_or_reads_in_source():
    source=SOURCE.read_text()
    assert "import OCP" not in source and "import open3d" not in source
    assert "source.step" not in source and "source-assets" not in source
    assert 'folder / "folded.ply"' in source and 'folder / "FLAT.json"' in source


@pytest.mark.parametrize("vertices,triangles",[([[0,0,float("nan")]],[[0,0,0]]),([[0,0,0],[1,0,0],[0,1,0]],[[0,1,3]]),([[0,0,0],[1,0,0],[0,1,0]],[[0.,1.,2.]])])
def test_invalid_derived_mesh_refused(vertices,triangles):
    with pytest.raises(ValueError):
        gallery.validate_mesh(vertices,triangles)


def test_failed_adaptation_keeps_extracted_axes(tmp_path):
    (tmp_path/"EXTRACTION.json").write_text(json.dumps({"bend_axes":[{"canonical_line_start":[0,0,0],"canonical_line_end":[1,0,0]}]}))
    assert gallery.nominal_lines(tmp_path)==[{"feature_id":"AX1","line_start":[0,0,0],"line_end":[1,0,0]}]
    candidates=[{"feature_id":"B1","line_start":[0,1,0],"line_end":[1,1,0]}]
    (tmp_path/"CANDIDATES.json").write_text(json.dumps(candidates))
    assert gallery.nominal_lines(tmp_path)==candidates


def test_published_flattened_schema_adapter():
    v,t=gallery.flat_schema_adapter([0.,0.,0.,1.,0.,0.,0.,1.,0.],[0,1,2])
    assert v.shape==(3,3) and t.shape==(1,3)
    assert t.tolist()==[[0,1,2]]


@pytest.mark.parametrize("vertices,triangles",[([0,0],[0,1,2]),([0,0,0,1,0,0,0,1,0],[0,1]),([0,0,0,1,0,0,0,1,0],[0.,1.,2.]),([0,0,0,1,0,0,0,1,0],[False,True,False])])
def test_flattened_schema_never_blind_casts_indices(vertices,triangles):
    with pytest.raises(ValueError):
        gallery.flat_schema_adapter(vertices,triangles)


def test_long_failure_text_is_bounded_and_wrapped():
    text=gallery.failure_text("unbrokenfailure"*100+"\n\tmore")
    assert len(text)<190 and all(len(line)<=38 for line in text.splitlines())


def test_terminal_alone_is_not_post_release(tmp_path):
    pytest.importorskip("matplotlib")
    attempt=tmp_path/"attempt"
    attempt.mkdir()
    (attempt/"FROZEN.json").write_text(json.dumps({"cases":[]}))
    (attempt/"CAMPAIGN_TERMINAL.json").write_text(json.dumps({"status":"COMPLETE"}))
    post=tmp_path/"post.json"
    post.write_text(json.dumps({"decision":"GREEN"}))
    with pytest.raises(ValueError,match="POST_RELEASE"):
        gallery.render(attempt,tmp_path/"gallery",post,"0"*64)
    assert not (tmp_path/"gallery").exists()


def test_actual_post_schema_binds_every_renderer_input(tmp_path):
    import hashlib
    attempt=tmp_path/"attempt"
    attempt.mkdir()
    paths={"FROZEN.json":b"{}","CAMPAIGN_TERMINAL.json":b"{}","case-03/FLAT.json":b"{\"status\":\"AVAILABLE\"}"}
    hashes={}
    for relative,data in paths.items():
        path=attempt/relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        hashes[relative]=hashlib.sha256(data).hexdigest()
    post={"decision":"ACCEPT_DIAGNOSTIC_WITH_LIMITATIONS","independent_non_author":True,
          "blocking_findings":[],"frozen_sha256":hashes["FROZEN.json"],
          "evidence":{"derived_artifact_sha256":hashes}}
    post_path=tmp_path/"post.json"
    post_path.write_text(json.dumps(post))
    pin=hashlib.sha256(post_path.read_bytes()).hexdigest()
    assert gallery.validate_post_release(attempt,post_path,pin)==hashes["FROZEN.json"]
    (attempt/"case-03/FLAT.json").write_bytes(b"tampered")
    with pytest.raises(ValueError,match="POST_HASH_MISMATCH"):
        gallery.validate_post_release(attempt,post_path,pin)


def test_obj_export_preserves_actual_vectors_and_one_based_indices():
    raw=gallery.flat_obj_bytes([[0.,0.,0.],[1.25,0.,0.],[0.,2.5,0.]],[[0,1,2]])
    assert b"not a neutral-axis manufacturing blank" in raw
    assert b"v 1.25 0.0 0.0\n" in raw and b"v 0.0 2.5 0.0\n" in raw
    assert raw.endswith(b"f 1 2 3\n")
    raw=gallery.flat_obj_bytes([[0.,0.,0.],[1.25,0.,0.],[0.,2.5,0.]],[[0,1,2]],
                              [{"feature_id":"B1","line_start":[0,0,0],"line_end":[1.25,0,0]}])
    assert b'# bend B1 line_start=[0, 0, 0] line_end=[1.25, 0, 0]' in raw
