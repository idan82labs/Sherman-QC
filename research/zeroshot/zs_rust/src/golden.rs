//! `zs_rust selftest --golden DIR --config CFG [--repo REPO] [--threads N] [--report OUT] [--write OUT]`
//!
//! Station self-check, run at install and at every start. It refuses (exit 3) unless every digest equals the
//! expectation compiled into this binary (golden/golden.json, embedded at build time):
//!
//! 1. numeric primitives: the libm functions, fma (f64 / f32), the pure-Rust linear algebra (eigh, solve, det,
//!    lstsq), numpy's PCG64 draws and Python float repr, over fixed pseudo-random inputs, digested bit for bit;
//!    the hardware SHA-256 path against the sha2 crate;
//! 2. the bundled synthetic meshes (DIR: binary PLY float32 / float64, ASCII PLY, binary STL; no scan data), each
//!    file's sha256;
//! 3. the whole lane on each mesh (plus one forced face-budget case) at 1 thread and at N threads: eval lines,
//!    budget record, and the full stage dump (Z2 lines, candidates, feature matrix, evidence, both GBM
//!    probabilities), digested canonically (timings excluded).
//!
//! Expectations are keyed by the config's lane-parameter digest (variants.zs, budget, z3_params, z2_fallback) and
//! by the model files' sha256, so a config or model change is reported as such rather than as numeric drift.
//! `--write OUT` records the computed digests (to create golden.json; the binary must then be rebuilt).
use crate::fmath::{DetF32, DetF64};
use crate::{accel, lane, nprng, pyjson};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

const EXPECTED: &str = include_str!("../golden/golden.json");
pub const GOLDEN_SCHEMA: &str = "sherman.zs_rust.golden.v1";
/// mesh file, forced face budget (None = the config's)
const CASES: &[(&str, &str, Option<usize>)] = &[
    ("g1_channel", "g1_channel.ply", None),
    ("g2_zprofile_rot", "g2_zprofile_rot.ply", None),
    ("g3_hat", "g3_hat.ply", None),
    ("g4_channel_soup", "g4_channel_soup.stl", None),
    ("g1_channel_budget", "g1_channel.ply", Some(12_000)),
];

fn sha_hex(b: &[u8]) -> String {
    lane::sha256_bytes(b)
}

fn push_f64(buf: &mut Vec<u8>, x: f64) {
    buf.extend_from_slice(&x.to_bits().to_le_bytes());
}

/// digests of the numeric primitives over fixed inputs
pub fn primitives() -> Map<String, Value> {
    let mut out = Map::new();
    let mut r = nprng::Rng::new(20261026);
    let u = r.random_n(4096);
    let w = r.random_n(4096);
    // libm
    let mut b = Vec::new();
    for i in 0..u.len() {
        let x = (u[i] - 0.5) * 200.0;
        let y = (w[i] - 0.5) * 3.0;
        push_f64(&mut b, x.dsin());
        push_f64(&mut b, x.dcos());
        push_f64(&mut b, y.dtan());
        push_f64(&mut b, (2.0 * u[i] - 1.0).dacos());
        push_f64(&mut b, y.datan2(x));
        push_f64(&mut b, ((w[i] - 0.5) * 60.0).dexp());
        push_f64(&mut b, (u[i] * 1e3 + 1e-300).dln());
        push_f64(&mut b, x.dhypot(y));
        push_f64(&mut b, (1.0 + u[i]).dpowf((w[i] - 0.5) * 40.0));
        push_f64(&mut b, (x * y).to_radians().dsin().to_degrees());
    }
    out.insert("libm".into(), json!(sha_hex(&b)));
    // fma (f64 and f32), sqrt
    let mut b = Vec::new();
    for i in 0..u.len() {
        let (a, c) = (u[i] * 3.0 - 1.0, w[i] - 0.5);
        let bb = w[(i * 7 + 3) % w.len()] * 1e8 + 1.0;
        push_f64(&mut b, a.dfma(bb, c));
        push_f64(&mut b, (a * 1e-8).dfma(a * 1e8, -(a * a)));
        let f = (a as f32).dfma(bb as f32 * 1e-6, c as f32);
        b.extend_from_slice(&f.to_bits().to_le_bytes());
        push_f64(&mut b, (u[i] * 1e6).sqrt());
    }
    out.insert("fma_sqrt".into(), json!(sha_hex(&b)));
    // linear algebra
    let mut b = Vec::new();
    let mut k = 0usize;
    let mut nx = |s: f64| {
        k += 1;
        (u[k % u.len()] - 0.5 + w[(k * 13) % w.len()] * 1e-3) * s
    };
    for _ in 0..300 {
        let a: Vec<f64> = (0..9).map(|_| nx(10.0)).collect();
        let s3: Vec<f64> = (0..9).map(|q| { let (i, j) = (q / 3, q % 3); a[i * 3 + j] + a[j * 3 + i] }).collect();
        let (ev, evec) = accel::eigh(&s3, 3);
        ev.iter().chain(evec.iter()).for_each(|&x| push_f64(&mut b, x));
        let s2 = [s3[0], s3[1], s3[1], s3[4]];
        let (ev2, evec2) = accel::eigh(&s2, 2);
        ev2.iter().chain(evec2.iter()).for_each(|&x| push_f64(&mut b, x));
        if let Some(x) = accel::solve(&a, &[1.0, -2.0, 0.5], 3) {
            x.iter().for_each(|&v| push_f64(&mut b, v));
        }
        push_f64(&mut b, accel::det(&a, 3));
        let rows: Vec<f64> = (0..60).map(|_| nx(5.0)).collect();
        let rhs: Vec<f64> = (0..20).map(|_| nx(2.0)).collect();
        accel::lstsq(&rows, 20, 3, &rhs, 20.0 * f64::EPSILON).iter().for_each(|&v| push_f64(&mut b, v));
        let x3 = accel::xtx(&rows, 20, 3);
        x3.iter().for_each(|&v| push_f64(&mut b, v));
    }
    out.insert("linalg".into(), json!(sha_hex(&b)));
    // numpy PCG64 / choice, Python float repr
    let mut r2 = nprng::Rng::new(7);
    let ch = r2.choice_noreplace(100_000, 5_000);
    let mut b: Vec<u8> = ch.iter().flat_map(|&i| (i as u64).to_le_bytes()).collect();
    let reprs: Vec<Value> = u.iter().take(512).map(|&x| json!(x * 1e-3 / 7.0)).collect();
    b.extend_from_slice(pyjson::dumps(&Value::Array(reprs), 0).as_bytes());
    out.insert("rng_repr".into(), json!(sha_hex(&b)));
    // SHA-256: hardware path (where present) equals the sha2 crate
    let data: Vec<u8> = (0..3_000_017u32).map(|i| (i.wrapping_mul(2654435761) >> 11) as u8).collect();
    let soft = sha_hex(&data);
    let hw = if crate::sha256hw::available() {
        let mut h = crate::sha256hw::Hasher::new();
        h.update(&data[..1_000_003]);
        h.update(&data[1_000_003..]);
        h.finalize_hex()
    } else {
        soft.clone()
    };
    out.insert("sha256".into(), json!(if hw == soft { soft } else { format!("MISMATCH hw {} soft {}", hw, soft) }));
    out
}

/// lane-parameter digest of a config: what the lane's numbers depend on (pins and fingerprints excluded)
pub fn params_digest(raw: &Value) -> String {
    let p = json!({"variants.zs": raw["variants"]["zs"], "budget": raw["budget"], "z3_params": raw["z3_params"],
                   "z2_fallback": raw["z2_fallback"]});
    sha_hex(serde_json::to_string(&p).unwrap().as_bytes())
}

fn strip_wall(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.iter().filter(|(k, _)| k.as_str() != "wall").map(|(k, x)| (k.clone(), strip_wall(x))).collect()),
        other => other.clone(),
    }
}

fn case_digest(scan: &Path, cfg: &lane::LaneConfig) -> Result<(String, usize), String> {
    let res = lane::run(scan, cfg, true, true).map_err(|(c, m)| format!("lane exit {}: {}", c, m))?;
    let pay = lane::payload(scan, "", &res.lines, res.status, "golden", json!({}));
    let mut dump = res.dump.ok_or("no dump")?;
    let budget = strip_wall(&res.stages["budget"]);
    if let Value::Object(m) = &mut dump {
        m.remove("stages");
    }
    let canon = json!({"status": res.status, "lines": pay["owned_bend_regions"], "budget": budget, "dump": dump});
    Ok((sha_hex(serde_json::to_string(&canon).unwrap().as_bytes()), res.lines.len()))
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}

/// returns the process exit code: 0 pass, 3 refused (any mismatch), 2 usage / IO error
pub fn main(args: &[String]) -> i32 {
    let dir = PathBuf::from(arg(args, "--golden").unwrap_or_else(|| "golden".into()));
    let Some(cfg_p) = arg(args, "--config") else {
        eprintln!("selftest --golden needs --config CFG");
        return 2;
    };
    let repo = PathBuf::from(arg(args, "--repo").unwrap_or_else(|| ".".into()));
    let nthreads: usize = arg(args, "--threads").and_then(|s| s.parse().ok()).unwrap_or(16).max(1);
    let raw: Value = match std::fs::read_to_string(&cfg_p).map_err(|e| e.to_string()).and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string())) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("selftest: config {}: {}", cfg_p, e);
            return 2;
        }
    };
    let cfg0 = match lane::load_config(Path::new(&cfg_p), &repo) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("selftest: {}", e);
            return 2;
        }
    };
    let pd = params_digest(&raw);
    let models = json!({"zs": lane::sha256_file(&cfg0.zs_model).unwrap_or_else(|e| format!("MISSING {}", e)),
                        "z3": lane::sha256_file(&cfg0.z3_model).unwrap_or_else(|e| format!("MISSING {}", e))});
    let mut got = Map::new();
    got.insert("primitives".into(), Value::Object(primitives()));
    let mut meshes = Map::new();
    for (_, f, _) in CASES {
        if !meshes.contains_key(*f) {
            meshes.insert(f.to_string(), json!(lane::sha256_file(&dir.join(f)).unwrap_or_else(|e| format!("MISSING {}", e))));
        }
    }
    got.insert("meshes".into(), Value::Object(meshes));
    let mut cases = Map::new();
    let mut thread_mismatch = Vec::new();
    for (name, f, mf) in CASES {
        let mut cfg = lane::load_config(Path::new(&cfg_p), &repo).expect("config loaded above");
        if let Some(m) = mf {
            cfg.max_faces = Some(*m);
        }
        let scan = dir.join(f);
        let mut per_t = Vec::new();
        for t in [1usize, nthreads] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().expect("pool");
            let r = pool.install(|| case_digest(&scan, &cfg));
            per_t.push(match r {
                Ok((d, n)) => (d, n),
                Err(e) => (format!("ERROR {}", e), 0),
            });
        }
        if per_t[0].0 != per_t[1].0 {
            thread_mismatch.push(name.to_string());
        }
        cases.insert(name.to_string(), json!({"digest": per_t[0].0, "n_lines": per_t[0].1, "digest_threads_n": per_t[1].0}));
    }
    got.insert("cases".into(), Value::Object(cases.clone()));
    if let Some(w) = arg(args, "--write") {
        // merge into an existing expectation file (one set per lane-parameter digest)
        let mut base: Value = std::fs::read_to_string(&w).ok().and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| json!({"schema": GOLDEN_SCHEMA, "primitives": {}, "meshes": {}, "sets": {}}));
        base["schema"] = json!(GOLDEN_SCHEMA);
        base["primitives"] = got["primitives"].clone();
        base["meshes"] = got["meshes"].clone();
        let cs: Map<String, Value> = cases.iter().map(|(k, v)| (k.clone(), json!({"digest": v["digest"], "n_lines": v["n_lines"]}))).collect();
        base["sets"][&pd] = json!({"models": models, "config_note": raw.get("note").and_then(|n| n.as_str()).map(|s| s.chars().take(80).collect::<String>()),
                                    "budget": raw["budget"], "cases": cs});
        if let Err(e) = std::fs::write(&w, serde_json::to_string_pretty(&base).unwrap() + "\n") {
            eprintln!("selftest: write {}: {}", w, e);
            return 2;
        }
    }
    // compare with the compiled-in expectations
    let exp: Value = serde_json::from_str(EXPECTED).unwrap_or(Value::Null);
    let mut fails: Vec<String> = Vec::new();
    if exp["schema"] != json!(GOLDEN_SCHEMA) {
        fails.push("NO_EXPECTATIONS_COMPILED_IN".into());
    } else {
        for (k, v) in got["primitives"].as_object().unwrap() {
            if exp["primitives"][k] != *v {
                fails.push(format!("PRIMITIVE:{}", k));
            }
        }
        for (k, v) in got["meshes"].as_object().unwrap() {
            if exp["meshes"][k] != *v {
                fails.push(format!("MESH_FILE:{}", k));
            }
        }
        let set = &exp["sets"][&pd];
        if set.is_null() {
            fails.push(format!("NO_GOLDEN_FOR_CONFIG_PARAMS:{}", pd));
        } else {
            if set["models"] != models {
                fails.push("MODEL_FILES".into());
            }
            for (k, v) in &cases {
                if set["cases"][k]["digest"] != v["digest"] {
                    fails.push(format!("LANE:{}", k));
                }
                if v["digest_threads_n"] != v["digest"] {
                    fails.push(format!("THREADS:{}", k));
                }
            }
        }
    }
    let report = json!({"schema": "sherman.zs_rust.selftest.v1", "ok": fails.is_empty(), "failures": fails,
                        "version": crate::VERSION, "build": crate::fmath::build_info(), "threads": [1, nthreads],
                        "config": cfg_p, "params_digest": pd, "models": models, "got": got,
                        "thread_mismatch": thread_mismatch});
    let txt = serde_json::to_string_pretty(&report).unwrap();
    if let Some(r) = arg(args, "--report") {
        let _ = std::fs::write(&r, &txt);
    }
    println!("{}", txt);
    if report["ok"].as_bool().unwrap() {
        0
    } else {
        eprintln!("zs_rust selftest: REFUSED ({})", report["failures"]);
        3
    }
}
