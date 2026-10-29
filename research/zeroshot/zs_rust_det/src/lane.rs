//! The whole lane in one process (zs_lane_v3 revision 2, research/zeroshot/zs_lane_v3/core.py + lane.py):
//! scan -> face budget -> Z2 -> Z3 -> pooling -> D1 snap -> v2 evidence -> GBM selection -> v2 post-processing ->
//! EXT / Z3R -> eval_lines.json (+ run record, + optional stage dump for the equivalence harness).
//! The gold guard, custody and pin checks stay with the caller (the Python adapter / lane_cli).
use crate::budget;
use crate::evidence;
use crate::gbm::Model;
use crate::kdtree::KdTree;
use crate::lanecore::{self, FixParams, OutLine, PCand, PostParams, Z2Line};
use crate::mesh::Mesh;
use crate::meshio;
use crate::num::*;
use crate::pyjson;

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const EXIT_POINT_CLOUD: i32 = 10;

pub struct LaneConfig {
    pub threshold: f64,
    pub post: PostParams,
    pub fix: FixParams,
    pub z2_fallback: bool,
    pub fallback_frac_only: bool,
    pub max_faces: Option<usize>,
    /// revision 3's grid cell (config budget.cell_mm); None = revision 2's budget
    pub cell_mm: Option<f64>,
    /// production CAD priors for Z3 (job "cad"); default = none
    pub cad: crate::Cad,
    pub zs_model: PathBuf,
    pub z3_model: PathBuf,
    pub raw: Value,
}

fn f(v: &Value, k: &str, d: f64) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(d)
}
fn b(v: &Value, k: &str, d: bool) -> bool {
    v.get(k).and_then(|x| x.as_bool()).unwrap_or(d)
}

pub fn load_config(path: &Path, repo: &Path) -> Result<LaneConfig, String> {
    let raw: Value = serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?)
        .map_err(|e| e.to_string())?;
    let v = &raw["variants"]["zs"];
    let post = &v["post"];
    if b(post, "split", false) || post.get("join_mode").and_then(|x| x.as_str()).unwrap_or("both") != "both" {
        return Err("config: only split=False and join_mode='both' are implemented".into());
    }
    if v.get("feats").and_then(|x| x.as_str()).unwrap_or("base") != "base" {
        return Err("config: only feats='base' is implemented".into());
    }
    let fx = &v["fix"];
    let models = &raw["rust"]["models"];
    let rp = |k: &str| -> Result<PathBuf, String> {
        let p = models[k]["path"].as_str().ok_or(format!("config lacks rust.models.{}.path", k))?;
        Ok(if Path::new(p).is_absolute() { PathBuf::from(p) } else { repo.join(p) })
    };
    Ok(LaneConfig {
        threshold: f(v, "threshold", 0.5),
        post: PostParams {
            join: b(post, "join", false),
            join_low: f(post, "join_low", 1.0),
            sweep_min: f(post, "sweep_min", 0.0),
            dedupe_join: b(post, "dedupe_join", false),
        },
        fix: FixParams {
            ext: b(fx, "ext", true),
            ext_ratio: f(fx, "ext_ratio", 1.3),
            ext_min_in: f(fx, "ext_min_in", 0.6),
            ext_dedupe: b(fx, "ext_dedupe", true),
            z3r: b(fx, "z3r", true),
            z3r_tau: f(fx, "z3r_tau", 0.9),
            z3r_sweep_min: f(fx, "z3r_sweep_min", 0.25),
        },
        z2_fallback: b(&raw, "z2_fallback", false),
        fallback_frac_only: raw["z3_params"].get("fallback_frac_only").and_then(|x| x.as_bool()).unwrap_or(false),
        max_faces: raw["budget"].get("max_faces").and_then(|x| x.as_u64()).map(|x| x as usize),
        cell_mm: raw["budget"].get("cell_mm").and_then(|x| x.as_f64()),
        cad: crate::Cad::default(),
        zs_model: rp("zs")?,
        z3_model: rp("z3")?,
        raw,
    })
}

/// sha256 of a file, pure Rust on every platform (no CommonCrypto / CNG): the Armv8 SHA2 instructions on aarch64
/// (sha256hw, detected at run time), else the sha2 crate (SHA-NI on x86-64 when present, detected at run time).
/// Same digest on every path.
pub fn sha256_file(p: &Path) -> Result<String, String> {
    use std::io::Read;
    let mut fh = std::fs::File::open(p).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; 4 << 20];
    if crate::sha256hw::available() {
        let mut h = crate::sha256hw::Hasher::new();
        loop {
            let n = fh.read(&mut buf).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        return Ok(h.finalize_hex());
    }
    let mut h = Sha256::new();
    loop {
        let n = fh.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// sha256 of bytes through the sha2 crate (reference path; the golden self-test checks sha256hw against it)
pub fn sha256_bytes(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

fn r3(x: f64) -> f64 {
    crate::z2graph::py_round(x, 3)
}

fn v3j(v: &V3) -> Value {
    json!([v[0], v[1], v[2]])
}

pub struct LaneResult {
    pub lines: Vec<OutLine>,
    pub status: &'static str,
    pub stages: Map<String, Value>,
    pub errors: Map<String, Value>,
    pub dump: Option<Value>,
}

/// core._process + select_lines
pub fn run(scan: &Path, cfg: &LaneConfig, want_dump: bool, splits: bool) -> Result<LaneResult, (i32, String)> {
    let t0 = Instant::now();
    let mut stages = Map::new();
    let errors = Map::new();
    // ---- budget
    let tb = Instant::now();
    let raw = crate::prof::time("read_mesh", || meshio::read_mesh_opt(scan, false)).map_err(|e| (2, e))?;
    if raw.f.is_empty() {
        return Err((EXIT_POINT_CLOUD, "POINT_CLOUD_UNSUPPORTED: scan has no faces".into()));
    }
    let meshio::RawMesh { v: rv, f: rf, .. } = raw;
    let (v, fcs, binfo) = match cfg.max_faces {
        None => (rv, rf, json!({"applied": false, "reason": "NO_BUDGET"})),
        Some(mf) => {
            let nf = meshio::face_count(scan);
            if nf.map(|n| n <= mf).unwrap_or(false) {
                (rv, rf, json!({"applied": false, "faces": nf.unwrap(), "max_faces": mf}))
            } else if rf.len() <= mf {
                let n = rf.len();
                (rv, rf, json!({"applied": false, "faces": n, "max_faces": mf}))
            } else {
                let bc = budget::BudgetCfg { max_faces: mf, cell_mm: cfg.cell_mm };
                let d = crate::prof::time("budget.decimate", || budget::apply(&rv, &rf, &bc));
                drop(rv); // the full-density mesh is not needed after the budget step
                drop(rf);
                let mut info = d.info.clone();
                info["applied"] = json!(true);
                info["max_faces"] = json!(mf);
                if cfg.cell_mm.is_some() {
                    info["method"] = json!("fixed_grid_dual_contouring");
                }
                info["wall"] = json!(r3(tb.elapsed().as_secs_f64()));
                (d.v, d.f, info)
            }
        }
    };
    stages.insert("budget".into(), binfo);
    // ---- shared structures: the Z3 mesh (faces with |cross| > 1e-12), one ray-cast scene when no face was
    // dropped (Z2's D1 probe and Z3's thickness then cast against the same triangles in the same order), one
    // kd-tree over the mesh vertices for Z3, the snap and the evidence
    let mesh = crate::prof::time("z3.mesh_from_arrays", || Mesh::from_arrays(v, &fcs));
    let scene = if mesh.f.len() == fcs.len() {
        Some(crate::prof::time("bvh_build", || crate::raycast::Scene::new(&mesh.v, &mesh.f)))
    } else {
        None
    };
    let tree = crate::prof::time("kdtree_build", || KdTree::new(&mesh.v));
    // ---- Z2 (first, as revision 2; Z2 and Z3 are independent)
    let z2o = crate::z2_full_with(&mesh.v, &fcs, cfg.z2_fallback, scene.as_ref());
    drop(fcs);
    let mut z2info = z2o.info.clone();
    z2info["wall"] = json!(r3(z2info["wall"].as_f64().unwrap()));
    stages.insert("z2".into(), Value::Object(z2info));
    let z2_lines: Vec<Z2Line> = z2o.lines.iter().map(|(a, b, i)| Z2Line { start: *a, end: *b, incl: *i }).collect();
    drop(z2o);
    // ---- Z3
    let t3 = Instant::now();
    let (z3j, z3c) = crate::z3_full_with(&mesh, cfg.fallback_frac_only, false, scene.as_ref(), Some(&tree), &cfg.cad);
    drop(scene);
    let t_mm = z3j["t_mm"].as_f64().unwrap();
    stages.insert("z3".into(), json!({"wall": r3(t3.elapsed().as_secs_f64()), "t_mm": t_mm, "n_candidates": z3c.len(),
        "area_total": z3j["area_total"], "n_families": z3j["families"].as_array().unwrap().len(),
        "family_fallback": z3j["family_fallback"], "rust": z3j["timing_s"]}));
    // ---- pool
    let tp = Instant::now();
    let (mut cands, x) = lanecore::pool(&z3c, &z2_lines);
    stages.insert("pool".into(), json!({"wall": r3(tp.elapsed().as_secs_f64()), "n": cands.len()}));
    // ---- snap (Z3 mesh points)
    let ts = Instant::now();
    crate::prof::time("snap", || lanecore::snap_all(&mut cands, &mesh.v, &mesh.vn, t_mm, &tree));
    stages.insert("snap_all".into(), json!({"wall": r3(ts.elapsed().as_secs_f64()), "n": cands.len()}));
    // ---- evidence
    let te = Instant::now();
    let spacing = crate::prof::time("evidence.spacing", || crate::mesh_spacing(&mesh));
    let ctx = evidence::Ctx::new(&mesh.v, &mesh.vn, &tree, t_mm, spacing);
    let cj = Value::Array(cands.iter().map(cand_evidence_json).collect());
    let mut ev = crate::prof::time("evidence.run", || evidence::run(&ctx, &cj, splits));
    if !splits {
        ev["splits"] = json!({});
    }
    stages.insert("evidence".into(), json!({"wall": r3(te.elapsed().as_secs_f64())}));
    // ---- selection
    let tsel = Instant::now();
    let mut lines: Vec<OutLine> = Vec::new();
    let mut prob = vec![];
    let mut p3 = vec![];
    if !cands.is_empty() {
        let zs = Model::load(&cfg.zs_model).map_err(|e| (2, e))?;
        let m3 = Model::load(&cfg.z3_model).map_err(|e| (2, e))?;
        prob = x.iter().map(|r| zs.proba(r)).collect();
        p3 = cands.iter().zip(&x).map(|(c, r)| if c.has_z3 { m3.proba(&r[..21]) } else { f64::NAN }).collect();
        let out = lanecore::postprocess(&cands, &prob, cfg.threshold, Some(&ev), &cfg.post);
        lines = lanecore::apply_fixes(out, &cands, &z2_lines, &prob, &p3, Some(&ev), &cfg.fix);
    }
    let ops: Map<String, Value> = ["keep", "split", "join"].iter().map(|k| (k.to_string(), json!(lines.iter().filter(|l| l.op == *k).count()))).collect();
    let fixes: Map<String, Value> = ["EXT", "Z3R"].iter().map(|k| (k.to_string(), json!(lines.iter().filter(|l| l.fix == Some(*k)).count()))).collect();
    stages.insert("select_post".into(), json!({"wall": r3(tsel.elapsed().as_secs_f64()), "n": lines.len(), "ops": ops, "fixes": fixes}));
    stages.insert("lane_wall_s".into(), json!(r3(t0.elapsed().as_secs_f64())));
    let dump = if want_dump {
        Some(json!({
            "stages": stages.clone(),
            "z2_lines": z2_lines.iter().map(|l| json!({"start": v3j(&l.start), "end": v3j(&l.end), "incl": l.incl})).collect::<Vec<_>>(),
            "candidates": cands.iter().map(cand_dump_json).collect::<Vec<_>>(),
            "X": x.iter().map(|r| r.iter().map(|&a| if a.is_nan() { Value::Null } else { json!(a) }).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "evidence": ev,
            "prob": prob.iter().map(|&a| json!(a)).collect::<Vec<_>>(),
            "prob_z3": p3.iter().map(|&a| if a.is_nan() { Value::Null } else { json!(a) }).collect::<Vec<_>>(),
            "status": "OK",
        }))
    } else {
        None
    };
    Ok(LaneResult { lines, status: "OK", stages, errors, dump })
}

fn cand_evidence_json(c: &PCand) -> Value {
    let mut m = Map::new();
    m.insert("start".into(), v3j(&c.start));
    m.insert("end".into(), v3j(&c.end));
    m.insert("incl".into(), json!(c.incl));
    if let (Some(a), Some(b)) = (c.da, c.db) {
        m.insert("da".into(), v3j(&a));
        m.insert("db".into(), v3j(&b));
    }
    if let Some(x) = c.flank_a {
        m.insert("flank_a".into(), json!(x));
    }
    if let Some(x) = c.flank_b {
        m.insert("flank_b".into(), json!(x));
    }
    Value::Object(m)
}

fn cand_dump_json(c: &PCand) -> Value {
    let mut m = Map::new();
    m.insert("start".into(), v3j(&c.start));
    m.insert("end".into(), v3j(&c.end));
    m.insert("src".into(), json!(c.src));
    m.insert("has_z3".into(), json!(c.has_z3));
    if let Some(a) = c.da {
        m.insert("da".into(), v3j(&a));
    }
    if let Some(a) = c.db {
        m.insert("db".into(), v3j(&a));
    }
    if let Some(x) = c.flank_a {
        m.insert("flank_a".into(), json!(x));
    }
    if let Some(x) = c.flank_b {
        m.insert("flank_b".into(), json!(x));
    }
    if let Some(x) = c.length {
        m.insert("length".into(), json!(x));
    }
    if let Some(x) = c.family {
        m.insert("family".into(), json!(x));
    }
    if let Some(x) = c.n_support {
        m.insert("n_support".into(), json!(x));
    }
    m.insert("incl".into(), json!(c.incl));
    m.insert("start_s".into(), v3j(&c.start_s));
    m.insert("end_s".into(), v3j(&c.end_s));
    m.insert("snap".into(), Value::Object(c.snap.clone()));
    Value::Object(m)
}

/// eval_lines.json (sherman.broad_baseline.eval_lines.v1) as zs_lane_v2.core.payload / bb_common.eval_payload
pub fn payload(scan: &Path, sha: &str, lines: &[OutLine], status: &str, lane: &str, conversion: Value) -> Value {
    let regs: Vec<Value> = lines
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let mut m = Map::new();
            m.insert("bend_id".into(), json!(format!("ZS{:03}", i + 1)));
            m.insert("line_start".into(), v3j(&l.start_s));
            m.insert("line_end".into(), v3j(&l.end_s));
            m.insert("score".into(), json!(l.score));
            m.insert("incl".into(), json!(l.incl));
            m.insert("src".into(), json!(l.src));
            m.insert("op".into(), json!(l.op));
            m.insert("parts".into(), json!(l.parts));
            if let Some(fx) = l.fix {
                m.insert("fix".into(), json!(fx));
            }
            Value::Object(m)
        })
        .collect();
    let failure = if status == "OK" { Value::Null } else { json!({"error": conversion.get("errors").cloned().unwrap_or(Value::Null)}) };
    let mut p = Map::new();
    p.insert("schema".into(), json!("sherman.broad_baseline.eval_lines.v1"));
    p.insert("lane".into(), json!(lane));
    p.insert("status".into(), json!(status));
    p.insert("frame".into(), json!("scan_raw"));
    p.insert("frame_definition".into(), json!("coordinates of the scan file exactly as stored (source_ply_raw == scan_raw)"));
    p.insert("units".into(), json!("scan file units (mm for every Sherman scan inspected so far; not re-estimated)"));
    p.insert("scan".into(), json!({"path": scan.to_string_lossy(), "sha256": sha}));
    p.insert("source".into(), json!({"kind": lane, "path": Value::Null, "sha256": Value::Null}));
    p.insert("conversion".into(), conversion);
    p.insert("line_count".into(), json!(regs.len()));
    p.insert("owned_bend_regions".into(), Value::Array(regs));
    p.insert("dropped".into(), json!([]));
    p.insert("failure".into(), failure);
    p.insert("created_at".into(), json!(utc_now()));
    Value::Object(p)
}

fn utc_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    // civil from days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m, d, rem / 3600, (rem % 3600) / 60, rem % 60)
}

pub fn write_json(path: &Path, v: &Value) -> Result<(), String> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, pyjson::dumps(v, 1)).map_err(|e| format!("{}: {}", path.display(), e))
}
