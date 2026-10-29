//! zs_rust: geometry kernels of the zero-shot bend lane v3, called by research/zeroshot/zs_lane_v3_rs as a
//! subprocess. Input: a job JSON naming .npy arrays and parameters; output: one JSON file.
//!
//!   zs_rust <command> --job JOB.json --out OUT.json [--threads N]
//!
//! Commands: z3 (Z3 slicing -> merged candidates), profile (D1 bend profiles for a batch of lines),
//! evidence (v2 station profiles, split runs, collinear pair evidence), segment (Z2 planar regions and flanges from
//! the kNN graph), version.
mod accel;
mod accel_host;
mod probe;
mod fmath;
mod golden;
mod sha256hw;
mod budget;
mod ckdtree;
mod nprng;
mod raycast;
mod d1;
mod evidence;
mod gbm;
mod kdtree;
mod lane;
mod lanecore;
mod lsap;
mod meshio;
mod mesh;
mod morph;
mod npy;
mod prof;
mod pyjson;
mod num;
mod z2graph;
mod z2load;
mod z2seg;
mod z3;
mod z3prep;

use num::V3;
use rayon::prelude::*;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const VERSION: &str = "zs_rust 0.1.0";

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}

fn load_v3(path: &Path) -> Result<Vec<V3>, String> {
    let (v, shape) = npy::read_f64(path)?;
    if shape.len() != 2 || shape[1] != 3 {
        return Err(format!("{}: expected (n,3) float array", path.display()));
    }
    Ok(v.chunks(3).map(|c| [c[0], c[1], c[2]]).collect())
}

fn load_f3(path: &Path) -> Result<Vec<[u32; 3]>, String> {
    let (v, shape) = npy::read_i64(path)?;
    if shape.len() != 2 || shape[1] != 3 {
        return Err(format!("{}: expected (m,3) int array", path.display()));
    }
    Ok(v.chunks(3).map(|c| [c[0] as u32, c[1] as u32, c[2] as u32]).collect())
}

fn jv3(v: &Value) -> V3 {
    let a = v.as_array().expect("3-vector");
    [a[0].as_f64().unwrap(), a[1].as_f64().unwrap(), a[2].as_f64().unwrap()]
}

fn job_path(job: &Value, base: &Path, key: &str) -> PathBuf {
    let p = PathBuf::from(job[key].as_str().unwrap_or_else(|| panic!("job lacks {}", key)));
    if p.is_absolute() {
        p
    } else {
        base.join(p)
    }
}

/// peak resident set size of this process in bytes (reporting only; never part of an output)
#[cfg(unix)]
fn rss_bytes() -> u64 {
    #[repr(C)]
    struct Timeval {
        tv_sec: i64,
        tv_usec: i64,
    }
    #[repr(C)]
    struct Rusage {
        ru_utime: Timeval,
        ru_stime: Timeval,
        ru_maxrss: i64,
        rest: [i64; 13],
    }
    extern "C" {
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }
    let mut ru: Rusage = unsafe { std::mem::zeroed() };
    if unsafe { getrusage(0, &mut ru) } != 0 {
        return 0;
    }
    // ru_maxrss: bytes on macOS, KiB on Linux and the other Unixes
    if cfg!(target_os = "macos") { ru.ru_maxrss as u64 } else { ru.ru_maxrss as u64 * 1024 }
}

/// peak working set of this process in bytes (Windows: K32GetProcessMemoryInfo, kernel32)
#[cfg(windows)]
fn rss_bytes() -> u64 {
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn K32GetProcessMemoryInfo(h: *mut std::ffi::c_void, c: *mut ProcessMemoryCounters, cb: u32) -> i32;
    }
    let mut c: ProcessMemoryCounters = unsafe { std::mem::zeroed() };
    c.cb = std::mem::size_of::<ProcessMemoryCounters>() as u32;
    if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } == 0 {
        return 0;
    }
    c.peak_working_set_size as u64
}

#[cfg(not(any(unix, windows)))]
fn rss_bytes() -> u64 {
    0
}

fn cmd_z3(job: &Value, base: &Path) -> Result<Value, String> {
    let t0 = Instant::now();
    let v = load_v3(&job_path(job, base, "V"))?;
    let f = load_f3(&job_path(job, base, "F"))?;
    let mesh = mesh::Mesh::from_arrays(v, &f);
    let t_load = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let tree = kdtree::KdTree::new(&mesh.v);
    let t_tree = t1.elapsed().as_secs_f64();
    let prm: z3::Params = serde_json::from_value(job["params"].clone()).map_err(|e| e.to_string())?;
    let t = job["t"].as_f64().ok_or("job lacks t")?;
    let axes: Vec<V3> = job["axes"].as_array().ok_or("job lacks axes")?.iter().map(jv3).collect();
    let want_debug = job["debug"].as_bool().unwrap_or(false);
    let (cands, n_raw, dbg, mut tm) = z3::detect(&mesh, &tree, &axes, &prm, t, want_debug);
    tm.insert("load".into(), json!(t_load));
    tm.insert("tree".into(), json!(t_tree));
    tm.insert("total".into(), json!(t0.elapsed().as_secs_f64()));
    let mut out = Map::new();
    out.insert("candidates".into(), Value::Array(cands.iter().map(z3::line_json).collect()));
    out.insert("raw_count".into(), json!(n_raw));
    out.insert("n_faces".into(), json!(mesh.f.len()));
    out.insert("timing_s".into(), Value::Object(tm));
    out.insert(
        "families".into(),
        Value::Array(
            dbg.iter()
                .map(|d| {
                    let mut m = Map::new();
                    m.insert("n_levels".into(), json!(d.n_levels));
                    m.insert("n_chains".into(), json!(d.n_chains));
                    m.insert("n_corners".into(), json!(d.n_corners));
                    m.insert("n_tracks".into(), json!(d.n_tracks));
                    m.insert("n_segments".into(), json!(d.n_segments));
                    m.insert("n_lines".into(), json!(d.n_lines));
                    if want_debug {
                        m.insert(
                            "corners".into(),
                            Value::Array(d.corners.iter().map(|c| json!([c.0, c.1, c.2, c.3])).collect()),
                        );
                        m.insert("seg_lines".into(), Value::Array(d.seg_lines.iter().map(z3::line_json).collect()));
                    }
                    Value::Object(m)
                })
                .collect(),
        ),
    );
    Ok(Value::Object(out))
}

/// Default z3s.Params (+ the lane's fallback_frac_only) as the slicing / peak parameter structs
pub fn z3_default_params() -> (z3::Params, z3prep::PeakParams) {
    (
        z3::Params {
            step_mm: 2.0, dp_eps_mm: 0.15, merge_deg: 4.0, min_flat_mm: 2.0, max_arc_mm: 25.0, min_bend_deg: 6.0,
            max_bend_deg: 165.0, flat_rms_mm: 0.25, link_pos_mm: 1.5, link_ang_deg: 7.0, link_bis_deg: 25.0,
            bridge_mm: 4.5, min_track_slices: 2, max_axis_dev_deg: 15.0, dedupe_pos_mm: 1.2,
        },
        z3prep::PeakParams {
            peak_sep_deg: 5.0, peak_win_deg: 2.5, peak_min_area_frac: 0.0005, peak_min_area_mm2: 20.0,
            peak_min_area_rel: 0.0, max_families: 20, family_merge_deg: 3.0,
        },
    )
}

/// Full Z3 on a mesh: thickness, families (with the fallback_frac_only retry), slicing ... NMS.
pub fn z3_full(mesh: &mesh::Mesh, fallback_frac_only: bool, want_debug: bool) -> Value {
    z3_full_lines(mesh, fallback_frac_only, want_debug).0
}

/// evidence.mesh_spacing(V, F, 20000): median edge length over a numpy-RNG sample of faces
pub fn mesh_spacing(mesh: &mesh::Mesh) -> f64 {
    if mesh.f.is_empty() {
        return 1.0;
    }
    let idx = nprng::Rng::new(0).choice_noreplace(mesh.f.len(), 20000.min(mesh.f.len()));
    let e: Vec<f64> = idx.iter().map(|&i| {
        let t = mesh.f[i as usize];
        num::norm3(&num::sub3(&mesh.v[t[0] as usize], &mesh.v[t[1] as usize]))
    }).collect();
    num::median(&e)
}

pub fn z3_full_lines(mesh: &mesh::Mesh, fallback_frac_only: bool, want_debug: bool) -> (Value, Vec<z3::Line>) {
    z3_full_with(mesh, fallback_frac_only, want_debug, None, None, &Cad::default())
}

/// Production CAD priors for Z3 (lane_cli.run_lane_v3: core.process(thickness={"t_mm": t, "source": "CAD"},
/// max_families=K)): an external sheet thickness replaces the ray-cast estimate, and the axis-family cap replaces
/// z3s.Params.max_families (20). None = the research lane.
#[derive(Clone, Debug, Default)]
pub struct Cad {
    pub t_mm: Option<f64>,
    pub max_families: Option<usize>,
}

pub fn z3_full_with(mesh: &mesh::Mesh, fallback_frac_only: bool, want_debug: bool, shared_scene: Option<&raycast::Scene>,
                    shared_tree: Option<&kdtree::KdTree>, cad: &Cad) -> (Value, Vec<z3::Line>) {
    let (prm, mut pp) = z3_default_params();
    if let Some(k) = cad.max_families {
        pp.max_families = k;
    }
    let t0 = Instant::now();
    let (th, th_json) = match cad.t_mm {
        Some(tc) => (z3prep::Thickness { t_mm: tc, confidence: f64::NAN, two_sided: f64::NAN }, json!({"t_mm": tc, "source": "CAD"})),
        None => {
            let own_scene;
            let scene: &raycast::Scene = match shared_scene {
                Some(s) => s,
                None => {
                    own_scene = prof::time("z3.bvh_build", || raycast::Scene::new(&mesh.v, &mesh.f));
                    &own_scene
                }
            };
            let th = prof::time("z3.thickness_rays", || z3prep::estimate_thickness(mesh, scene));
            let j = json!({"t_mm": th.t_mm, "confidence": th.confidence, "two_sided": th.two_sided});
            (th, j)
        }
    };
    let t = num::clip(th.t_mm, 0.5, 6.0);
    let t_th = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let g = prof::time("z3.gauss_samples", || z3prep::gauss_samples(mesh));
    let peaks = prof::time("z3.normal_peaks", || z3prep::normal_peaks(&g, &pp));
    let mut fams = prof::time("z3.axis_families", || z3prep::axis_families(&g, &peaks, &pp));
    let mut fallback = false;
    if fams.is_empty() && fallback_frac_only {
        let pp2 = z3prep::PeakParams { peak_min_area_mm2: 0.0, peak_min_area_rel: 0.0, ..pp };
        let peaks2 = z3prep::normal_peaks(&g, &pp2);
        fams = z3prep::axis_families(&g, &peaks2, &pp2);
        fallback = true;
    }
    let t_fam = t1.elapsed().as_secs_f64();
    let own_tree;
    let tree: &kdtree::KdTree = match shared_tree {
        Some(t) => t,
        None => {
            own_tree = prof::time("z3.kdtree_build", || kdtree::KdTree::new(&mesh.v));
            &own_tree
        }
    };
    let axes: Vec<V3> = fams.iter().map(|f| f.axis).collect();
    let (cands, n_raw, dbg, mut tm) = prof::time("z3.detect", || z3::detect(mesh, tree, &axes, &prm, t, want_debug));
    tm.insert("thickness".into(), json!(t_th));
    tm.insert("families".into(), json!(t_fam));
    let fam_json: Vec<Value> = fams.iter().map(|f| json!({"axis": f.axis.to_vec(), "score": f.score, "weight": f.weight, "arc_area": f.arc_area})).collect();
    let mut out = Map::new();
    out.insert("thickness".into(), th_json);
    out.insert("t_mm".into(), json!(t));
    out.insert("area_total".into(), json!(num::pw_sum(&mesh.fa)));
    out.insert("family_fallback".into(), json!(fallback));
    out.insert("families".into(), Value::Array(fam_json));
    out.insert("raw_count".into(), json!(n_raw));
    out.insert("candidates".into(), Value::Array(cands.iter().map(z3::line_json).collect()));
    out.insert("timing_s".into(), Value::Object(tm));
    if want_debug {
        out.insert("debug_families".into(), json!(dbg.iter().map(|d| json!({"n_corners": d.n_corners, "n_tracks": d.n_tracks})).collect::<Vec<_>>()));
    }
    (Value::Object(out), cands)
}

pub fn seg_default() -> z2seg::SegParams {
    z2seg::SegParams {
        smooth_iters: 2, seed_deg: 2.5, flat_deg: 4.0, seed_offset_mm: 0.12, min_seed_points: 30, grow_deg: 12.0,
        grow_dist_mm: 0.30, grow_iters: 60, merge_deg: 2.5, merge_dist_mm: 0.35, min_region_area_mm2: 12.0,
        pair_deg: 6.0, wall_width_t: 1.6, seed_release_deg: 6.0,
    }
}

pub struct Z2Out {
    pub lines: Vec<(V3, V3, f64)>,
    pub p: Vec<V3>,
    pub n: Vec<V3>,
    pub t: f64,
    pub info: Map<String, Value>,
}

fn z2_lines_for(p: &[V3], n: &[V3], nbr: &[u32], t: f64, pa: f64, seg: &z2seg::SegParams, gp: &z2graph::GraphParams) -> Vec<(V3, V3, f64)> {
    let so = prof::time("z2.segment", || z2seg::segment(p, n, nbr, 16, t, pa, seg));
    for (k, v) in &so.timing {
        prof::add(&format!("z2.segment.{}", k), *v);
    }
    let sc = z2graph::Scan { p, ns: &so.ns, nbr, k: 16, t, pa, regions: &so.regions, flanges: &so.flanges, fop: &so.fop };
    prof::time("z2.graph", || z2graph::run_lines(&sc, n, gp))
}

/// zs_lane_v2.core.run_z2 (with the Z2 fallback) on a mesh
pub fn z2_full(v: &[V3], f: &[[u32; 3]], fallback: bool) -> Z2Out {
    z2_full_with(v, f, fallback, None)
}

/// `scene`: a ray-cast scene over exactly these faces, shared with Z3 when no face was dropped there
pub fn z2_full_with(v: &[V3], f: &[[u32; 3]], fallback: bool, scene: Option<&raycast::Scene>) -> Z2Out {
    let t0 = Instant::now();
    let ld = z2load::load_scan(v, f, 450_000, scene);
    let tree = prof::time("z2.knn_build", || ckdtree::CKDTree::new(&ld.p));
    let knn: Vec<Vec<(f64, usize)>> = prof::time("z2.knn_query", || ld.p.par_iter().map(|x| tree.query(x, 17, f64::INFINITY)).collect());
    drop(tree);
    let mut nbr = Vec::with_capacity(ld.p.len() * 16);
    let mut d1: Vec<f64> = Vec::with_capacity(ld.p.len());
    for r in &knn {
        for j in 1..17 {
            nbr.push(r[j].1 as u32);
        }
        d1.push(r[1].0.sqrt());
    }
    drop(knn);
    let spacing = num::median(&d1);
    let seg = seg_default();
    let gp = z2graph::GraphParams::default();
    let mut lines = z2_lines_for(&ld.p, &ld.n, &nbr, ld.t, ld.pa, &seg, &gp);
    let mut info = Map::new();
    if fallback && lines.is_empty() {
        let col = |k: usize, q: f64| num::quantile(&ld.p.iter().map(|x| x[k]).collect::<Vec<_>>(), q);
        let ext = [col(0, 0.99) - col(0, 0.01), col(1, 0.99) - col(1, 0.01), col(2, 0.99) - col(2, 0.01)];
        let diag_mm = num::norm3(&ext);
        let sp = if spacing != 0.0 && !spacing.is_nan() { spacing } else { 0.5 };
        let mra = (4.0 * sp * sp).max(12.0f64.min(1e-3 * diag_mm * diag_mm));
        let msp = 10.0f64.max(30.0f64.min({ let q = diag_mm / sp.max(1e-3); q * q } * 1e-3)) as usize;
        let seg2 = z2seg::SegParams { min_region_area_mm2: mra, min_seed_points: msp, ..seg_default() };
        let gp2 = z2graph::GraphParams { min_span_mm: (3.0 * sp).max(8.0f64.min(0.08 * diag_mm)), ..z2graph::GraphParams::default() };
        lines = z2_lines_for(&ld.p, &ld.n, &nbr, ld.t, ld.pa, &seg2, &gp2);
        info.insert("fallback".into(), json!({"diag_mm": z2graph::py_round(diag_mm, 1), "spacing": z2graph::py_round(sp, 3),
            "min_region_area_mm2": mra, "min_seed_points": msp, "min_span_mm": gp2.min_span_mm, "n_lines": lines.len()}));
    }
    info.insert("wall".into(), json!(t0.elapsed().as_secs_f64()));
    info.insert("t_mm".into(), json!(ld.t));
    info.insert("n_lines".into(), json!(lines.len()));
    Z2Out { lines, p: ld.p, n: ld.n, t: ld.t, info }
}

fn cmd_z2full(job: &Value, _base: &Path) -> Result<Value, String> {
    let path = job["scan"].as_str().ok_or("job lacks scan")?;
    let raw = meshio::read_mesh(Path::new(path))?;
    if raw.f.is_empty() {
        return Err("POINT_CLOUD_UNSUPPORTED".into());
    }
    let z = z2_full(&raw.v, &raw.f, job["fallback"].as_bool().unwrap_or(true));
    let lines: Vec<Value> = z.lines.iter().map(|(a, b, i)| json!({"start": a.to_vec(), "end": b.to_vec(), "incl": i})).collect();
    Ok(json!({"lines": lines, "info": Value::Object(z.info), "t": z.t, "n_points": z.p.len()}))
}

fn cmd_z3full(job: &Value, _base: &Path) -> Result<Value, String> {
    let path = job["scan"].as_str().ok_or("job lacks scan")?;
    let raw = meshio::read_mesh(Path::new(path))?;
    if raw.f.is_empty() {
        return Err("POINT_CLOUD_UNSUPPORTED".into());
    }
    let mesh = mesh::Mesh::from_arrays(raw.v, &raw.f);
    Ok(z3_full(&mesh, job["fallback_frac_only"].as_bool().unwrap_or(true), false))
}

fn cmd_profile(job: &Value, base: &Path) -> Result<Value, String> {
    let t0 = Instant::now();
    let p = load_v3(&job_path(job, base, "P"))?;
    let n = load_v3(&job_path(job, base, "N"))?;
    let t = job["t"].as_f64().ok_or("job lacks t")?;
    let tree = kdtree::KdTree::new(&p);
    let lines: Vec<(V3, V3)> = job["lines"]
        .as_array()
        .ok_or("job lacks lines")?
        .iter()
        .map(|l| (jv3(&l[0]), jv3(&l[1])))
        .collect();
    let profs: Vec<Value> =
        lines.par_iter().map(|(a, b)| d1::bend_profile(&p, &n, &tree, a, b, t).to_json()).collect();
    let mut out = Map::new();
    out.insert("profiles".into(), Value::Array(profs));
    out.insert("timing_s".into(), json!({"total": t0.elapsed().as_secs_f64()}));
    Ok(Value::Object(out))
}

fn cmd_evidence(job: &Value, base: &Path) -> Result<Value, String> {
    let t0 = Instant::now();
    let v = load_v3(&job_path(job, base, "V"))?;
    let vn = load_v3(&job_path(job, base, "VN"))?;
    let t = job["t"].as_f64().ok_or("job lacks t")?;
    let spacing = job["spacing"].as_f64().ok_or("job lacks spacing")?;
    let tree = kdtree::KdTree::new(&v);
    let ctx = evidence::Ctx::new(&v, &vn, &tree, t, spacing);
    let res = evidence::run(&ctx, &job["cands"], job["splits"].as_bool().unwrap_or(false));
    let mut out = match res {
        Value::Object(m) => m,
        _ => Map::new(),
    };
    out.insert("timing_s".into(), json!({"total": t0.elapsed().as_secs_f64()}));
    Ok(Value::Object(out))
}

fn cmd_segment(job: &Value, base: &Path) -> Result<Value, String> {
    let t0 = Instant::now();
    let p = load_v3(&job_path(job, base, "P"))?;
    let n = load_v3(&job_path(job, base, "N"))?;
    let (nbr_v, shape) = npy::read_i64(&job_path(job, base, "NBR"))?;
    if shape.len() != 2 || shape[0] != p.len() {
        return Err("NBR must be (n, k)".into());
    }
    let k = shape[1];
    let nbr: Vec<u32> = nbr_v.iter().map(|&x| x as u32).collect();
    let t = job["t"].as_f64().ok_or("job lacks t")?;
    let pa = job["pa"].as_f64().ok_or("job lacks pa")?;
    let prm: z2seg::SegParams = serde_json::from_value(job["params"].clone()).map_err(|e| e.to_string())?;
    let out = z2seg::segment(&p, &n, &nbr, k, t, pa, &prm);
    let tag = job["out_tag"].as_str().unwrap_or("seg");
    let ns_flat: Vec<f64> = out.ns.iter().flat_map(|v| v.iter().cloned()).collect();
    npy::write_f64(&base.join(format!("{}_Ns.npy", tag)), &[out.ns.len(), 3], &ns_flat)?;
    npy::write_i64(&base.join(format!("{}_label.npy", tag)), &[out.label.len()], &out.label)?;
    npy::write_i64(&base.join(format!("{}_fop.npy", tag)), &[out.fop.len()], &out.fop)?;
    let mut ridx: Vec<i64> = Vec::new();
    let mut regs = Vec::new();
    for r in &out.regions {
        let off = ridx.len();
        ridx.extend(r.idx.iter().map(|&i| i as i64));
        regs.push(json!({"rid": r.rid, "off": off, "len": r.idx.len(), "plane": r.plane.to_json(), "area": r.area,
                         "width": r.width, "length": r.length, "flange": r.flange}));
    }
    npy::write_i64(&base.join(format!("{}_ridx.npy", tag)), &[ridx.len()], &ridx)?;
    let fl: Vec<Value> = out
        .flanges
        .iter()
        .map(|f| {
            json!({"fid": f.fid, "regions": f.regions, "normal": f.normal.to_vec(), "area": f.area,
                   "two_sided": f.two_sided, "width": f.width, "length": f.length})
        })
        .collect();
    let mut tm = Map::new();
    for (k_, v) in &out.timing {
        tm.insert(k_.clone(), json!(v));
    }
    tm.insert("total".into(), json!(t0.elapsed().as_secs_f64()));
    Ok(json!({"regions": regs, "flanges": fl, "files": {"Ns": format!("{}_Ns.npy", tag), "label": format!("{}_label.npy", tag),
              "fop": format!("{}_fop.npy", tag), "ridx": format!("{}_ridx.npy", tag)}, "timing_s": tm}))
}

/// Primitive checks against numpy / scipy (used by the equivalence harness and the tests).
fn cmd_selftest(job: &Value, base: &Path) -> Result<Value, String> {
    let mut out = Map::new();
    if let Some(ops) = job["rng"].as_array() {
        let mut res = Vec::new();
        for op in ops {
            let seed = op["seed"].as_u64().unwrap();
            let mut r = nprng::Rng::new(seed);
            let kind = op["op"].as_str().unwrap();
            let v: Value = match kind {
                "random" => json!(r.random_n(op["n"].as_u64().unwrap() as usize)),
                "choice_noreplace" => json!(r.choice_noreplace(op["pop"].as_u64().unwrap() as usize, op["size"].as_u64().unwrap() as usize)),
                "choice_p" | "choice_p_noreplace" => {
                    let (p, _) = npy::read_f64(&base.join(op["p"].as_str().unwrap()))?;
                    let size = op["size"].as_u64().unwrap() as usize;
                    if kind == "choice_p" { json!(r.choice_p(&p, size)) } else { json!(r.choice_p_noreplace(&p, size)) }
                }
                _ => Value::Null,
            };
            res.push(v);
        }
        out.insert("rng".into(), Value::Array(res));
    }
    if job.get("knn").map(|v| !v.is_null()).unwrap_or(false) {
        let kj = &job["knn"];
        let p = load_v3(&base.join(kj["P"].as_str().unwrap()))?;
        let k = kj["k"].as_u64().unwrap() as usize;
        let ub = kj["ub"].as_f64().unwrap_or(f64::INFINITY);
        let tree = ckdtree::CKDTree::new(&p);
        let q = if kj["Q"].is_string() { load_v3(&base.join(kj["Q"].as_str().unwrap()))? } else { p.clone() };
        let res: Vec<Vec<(f64, usize)>> = q.par_iter().map(|x| tree.query(x, k, ub)).collect();
        let idx: Vec<i64> = res.iter().flat_map(|r| r.iter().map(|x| x.1 as i64)).collect();
        let d: Vec<f64> = res.iter().flat_map(|r| r.iter().map(|x| x.0.sqrt())).collect();
        npy::write_i64(&base.join("knn_idx.npy"), &[q.len(), k], &idx)?;
        npy::write_f64(&base.join("knn_d.npy"), &[q.len(), k], &d)?;
        out.insert("knn".into(), json!({"idx": "knn_idx.npy", "d": "knn_d.npy"}));
    }
    if let Some(gj) = job.get("gbm").filter(|v| !v.is_null()) {
        let m = gbm::Model::load(Path::new(gj["model"].as_str().unwrap()))?;
        let (x, shape) = npy::read_f64(&base.join(gj["X"].as_str().unwrap()))?;
        let k = shape[1];
        let p: Vec<f64> = x.chunks(k).map(|r| m.proba(r)).collect();
        npy::write_f64(&base.join("gbm_p.npy"), &[p.len()], &p)?;
        out.insert("gbm".into(), json!("gbm_p.npy"));
    }
    if let Some(dj) = job.get("decimate").filter(|v| !v.is_null()) {
        let (v, f) = if let Some(sp) = dj["scan"].as_str() {
            let m = meshio::read_mesh(Path::new(sp))?;
            (m.v, m.f)
        } else {
            (load_v3(&base.join(dj["V"].as_str().unwrap()))?, load_f3(&base.join(dj["F"].as_str().unwrap()))?)
        };
        let bc = budget::BudgetCfg { max_faces: dj["max_faces"].as_u64().unwrap() as usize, cell_mm: dj["cell_mm"].as_f64() };
        let d = budget::apply(&v, &f, &bc);
        let vf: Vec<f64> = d.v.iter().flat_map(|x| x.iter().cloned()).collect();
        let ff: Vec<i64> = d.f.iter().flat_map(|x| x.iter().map(|&i| i as i64)).collect();
        npy::write_f64(&base.join("dec_V.npy"), &[d.v.len(), 3], &vf)?;
        npy::write_i64(&base.join("dec_F.npy"), &[d.f.len(), 3], &ff)?;
        out.insert("decimate".into(), d.info);
    }
    if let Some(arr) = job["argsort"].as_str() {
        let (v, _) = npy::read_f64(&base.join(arr))?;
        let o: Vec<i64> = num::np_argsort(&v).iter().map(|&i| i as i64).collect();
        npy::write_i64(&base.join("argsort_out.npy"), &[o.len()], &o)?;
        out.insert("argsort".into(), json!("argsort_out.npy"));
    }
    if let Some(mp) = job["mesh"].as_str() {
        let m = meshio::read_mesh(Path::new(mp))?;
        let vf: Vec<f64> = m.v.iter().flat_map(|x| x.iter().cloned()).collect();
        let ff: Vec<i64> = m.f.iter().flat_map(|x| x.iter().map(|&i| i as i64)).collect();
        npy::write_f64(&base.join("mesh_V.npy"), &[m.v.len(), 3], &vf)?;
        npy::write_i64(&base.join("mesh_F.npy"), &[m.f.len(), 3], &ff)?;
        out.insert("mesh".into(), json!({"nv": m.v.len(), "nf": m.f.len(), "normals": m.vn.is_some(),
                                          "face_count": meshio::face_count(Path::new(mp))}));
    }
    if job.get("rays").map(|v| !v.is_null()).unwrap_or(false) {
        let rj = &job["rays"];
        let v = load_v3(&base.join(rj["V"].as_str().unwrap()))?;
        let f = load_f3(&base.join(rj["F"].as_str().unwrap()))?;
        let (r, _) = npy::read_f64(&base.join(rj["R"].as_str().unwrap()))?;
        let rays: Vec<[f32; 6]> = r.chunks(6).map(|c| [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32, c[4] as f32, c[5] as f32]).collect();
        let sc = raycast::Scene::new(&v, &f);
        let res = sc.cast_many(&rays);
        let th: Vec<f64> = res.iter().map(|x| x.0 as f64).collect();
        let pid: Vec<i64> = res.iter().map(|x| x.1).collect();
        npy::write_f64(&base.join("ray_t.npy"), &[th.len()], &th)?;
        npy::write_i64(&base.join("ray_pid.npy"), &[pid.len()], &pid)?;
        out.insert("rays".into(), json!({"t": "ray_t.npy", "pid": "ray_pid.npy"}));
    }
    Ok(Value::Object(out))
}

fn cmd_lane(job: &Value, _base: &Path) -> Result<Value, String> {
    let scan = PathBuf::from(job["scan"].as_str().ok_or("job lacks scan")?);
    let repo = PathBuf::from(job["repo"].as_str().unwrap_or("."));
    let mut cfg = lane::load_config(Path::new(job["config"].as_str().ok_or("job lacks config")?), &repo)?;
    // engineering override (density studies only): {"budget_override": {"max_faces": N | null}}
    if let Some(o) = job.get("budget_override").filter(|v| v.is_object()) {
        cfg.max_faces = o["max_faces"].as_u64().map(|x| x as usize);
    }
    // production CAD priors (lane_cli.run_lane_v3): {"cad": {"t_mm": t, "max_families": K}}
    if let Some(c) = job.get("cad").filter(|v| v.is_object()) {
        cfg.cad = Cad { t_mm: c["t_mm"].as_f64(), max_families: c["max_families"].as_u64().map(|x| x as usize) };
    }
    let want_dump = job["dump"].is_string();
    let splits = job["splits"].as_bool().unwrap_or(false);
    let t0 = Instant::now();
    // the scan's sha256 (eval_lines input hash) on its own thread, overlapping the lane (same bytes, same digest)
    let sha_job = {
        let sp = scan.clone();
        std::thread::spawn(move || {
            let t = Instant::now();
            let r = lane::sha256_file(&sp);
            (r, t.elapsed().as_secs_f64())
        })
    };
    let res = match lane::run(&scan, &cfg, want_dump, splits) {
        Ok(r) => r,
        Err((code, msg)) => {
            eprintln!("zs_rust lane: {}", msg);
            std::process::exit(code);
        }
    };
    let tj = Instant::now();
    let (sha, sha_s) = sha_job.join().map_err(|_| "sha256 thread panicked".to_string())?;
    let sha = sha?;
    prof::add("scan_sha256(overlapped)", sha_s);
    prof::add("scan_sha256.wait", tj.elapsed().as_secs_f64());
    let tw = Instant::now();
    let label = job["lane"].as_str().unwrap_or("zs_lane_v3_rs:zs");
    let mut conv = match job.get("conversion") { Some(Value::Object(m)) => m.clone(), _ => Map::new() };
    conv.insert("errors".into(), Value::Object(res.errors.clone()));
    conv.insert("stages".into(), Value::Object(res.stages.clone()));
    if let Some(o) = job["out"].as_str() {
        let pay = lane::payload(&scan, &sha, &res.lines, res.status, label, Value::Object(conv));
        lane::write_json(Path::new(o), &pay)?;
    }
    if let (Some(d), Some(dump)) = (job["dump"].as_str(), res.dump) {
        let mut dump = dump;
        dump["scan_sha256"] = json!(sha);
        std::fs::write(d, serde_json::to_string(&dump).unwrap()).map_err(|e| e.to_string())?;
    }
    prof::add("write_outputs", tw.elapsed().as_secs_f64());
    let wall = t0.elapsed().as_secs_f64();
    let rec = json!({"timers_s": prof::snapshot(), "status": res.status, "n_lines": res.lines.len(), "lane_wall_s": wall, "stages": Value::Object(res.stages),
                     "peak_rss_gib": rss_bytes() as f64 / (1u64 << 30) as f64});
    if let Some(r) = job["record"].as_str() {
        lane::write_json(Path::new(r), &rec)?;
    }
    crate::probe::write_report();
    if false {
    }
    Ok(json!({"status": res.status, "n_lines": res.lines.len(), "scan_sha256": sha}))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: zs_rust <lane|z3|z3full|z2full|profile|evidence|segment|selftest|version> --job JOB.json --out OUT.json [--threads N]\n       zs_rust selftest --golden DIR --config CFG [--repo REPO] [--threads N] [--report OUT.json]");
        std::process::exit(64);
    }
    let cmd = args[1].clone();
    if cmd == "version" {
        println!("{}", VERSION);
        if args.iter().any(|a| a == "--build") {
            println!("{}", fmath::build_info());
        }
        return;
    }
    if cmd == "selftest" && args.iter().any(|a| a == "--golden") {
        std::process::exit(golden::main(&args));
    }
    let threads: usize = arg(&args, "--threads").and_then(|s| s.parse().ok()).unwrap_or(1);
    rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build_global().expect("rayon pool");
    let job_p = PathBuf::from(arg(&args, "--job").expect("--job"));
    let out_p = PathBuf::from(arg(&args, "--out").expect("--out"));
    let base = job_p.parent().unwrap_or(Path::new(".")).to_path_buf();
    let job: Value = serde_json::from_str(&std::fs::read_to_string(&job_p).expect("read job")).expect("job json");
    let t0 = Instant::now();
    let res = match cmd.as_str() {
        "z3" => cmd_z3(&job, &base),
        "profile" => cmd_profile(&job, &base),
        "evidence" => cmd_evidence(&job, &base),
        "segment" => cmd_segment(&job, &base),
        "selftest" => cmd_selftest(&job, &base),
        "z3full" => cmd_z3full(&job, &base),
        "z2full" => cmd_z2full(&job, &base),
        "lane" => cmd_lane(&job, &base),
        _ => Err(format!("unknown command {}", cmd)),
    };
    match res {
        Ok(Value::Object(mut m)) => {
            m.insert("version".into(), json!(VERSION));
            m.insert("threads".into(), json!(threads));
            m.insert("wall_s".into(), json!(t0.elapsed().as_secs_f64()));
            m.insert("peak_rss_bytes".into(), json!(rss_bytes()));
            std::fs::write(&out_p, serde_json::to_string(&Value::Object(m)).unwrap()).expect("write out");
        }
        Ok(_) => unreachable!(),
        Err(e) => {
            eprintln!("zs_rust {}: {}", cmd, e);
            std::process::exit(2);
        }
    }
}
