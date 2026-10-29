//! Determinism probe (research copy only, never shipped): per-operation switch between the portable
//! implementation and the platform one numpy uses on this Mac (Apple libm, Accelerate), and a counting mode that
//! evaluates both and records where they differ.
//!
//! ZS_HOST_MATH = all | comma list of sin,cos,tan,acos,atan2,exp,ln,hypot,pow   (use Apple libm for these)
//! ZS_ACCEL     = all | comma list of xtx,gemm,eigh,solve,det,lstsq              (use Accelerate for these)
//! ZS_PROBE_OUT = path: evaluate both sides of every call, count bitwise mismatches, write JSON at exit of `lane`.
#![allow(dead_code)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

pub const MATH: [&str; 9] = ["sin", "cos", "tan", "acos", "atan2", "exp", "ln", "hypot", "pow"];
pub const OPS: [&str; 6] = ["xtx", "gemm", "eigh", "solve", "det", "lstsq"];

pub struct Cfg {
    pub host_math: u32,
    pub accel: u32,
    pub count: bool,
}

fn mask(var: &str, names: &[&str]) -> u32 {
    let v = std::env::var(var).unwrap_or_default();
    if v.trim() == "all" {
        return (1u32 << names.len()) - 1;
    }
    let mut m = 0;
    for t in v.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        let i = names.iter().position(|n| *n == t).unwrap_or_else(|| panic!("{}: unknown name {}", var, t));
        m |= 1 << i;
    }
    m
}

pub fn cfg() -> &'static Cfg {
    static C: OnceLock<Cfg> = OnceLock::new();
    C.get_or_init(|| Cfg {
        host_math: mask("ZS_HOST_MATH", &MATH),
        accel: mask("ZS_ACCEL", &OPS),
        count: std::env::var("ZS_PROBE_OUT").map(|s| !s.is_empty()).unwrap_or(false),
    })
}

#[inline]
pub fn host_math(id: usize) -> bool {
    cfg().host_math & (1 << id) != 0
}
#[inline]
pub fn accel(id: usize) -> bool {
    cfg().accel & (1 << id) != 0
}

static CALLS: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
static DIFF: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
static MAXULP: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
static SAMPLES: Mutex<Vec<(usize, Vec<f64>, f64, f64)>> = Mutex::new(Vec::new());

fn ulp_dist(a: f64, b: f64) -> u64 {
    let ia = a.to_bits() as i64;
    let ib = b.to_bits() as i64;
    let ka = if ia < 0 { i64::MIN - ia } else { ia };
    let kb = if ib < 0 { i64::MIN - ib } else { ib };
    (ka as i128 - kb as i128).unsigned_abs().min(u64::MAX as u128) as u64
}

/// one scalar math call evaluated both ways (counting mode)
pub fn math_hit(id: usize, port: f64, host: f64, args: &[f64]) {
    CALLS[id].fetch_add(1, Ordering::Relaxed);
    if port.to_bits() != host.to_bits() && !(port.is_nan() && host.is_nan()) {
        DIFF[id].fetch_add(1, Ordering::Relaxed);
        MAXULP[id].fetch_max(ulp_dist(port, host), Ordering::Relaxed);
        let mut s = SAMPLES.lock().unwrap();
        if s.len() < 20000 {
            s.push((id, args.to_vec(), port, host));
        }
    }
}

/// one linear-algebra call evaluated both ways (counting mode); id offset 9 + op
pub fn op_hit(op: usize, port: &[f64], host: &[f64]) {
    let id = 9 + op;
    CALLS[id].fetch_add(1, Ordering::Relaxed);
    let same = port.len() == host.len() && port.iter().zip(host).all(|(a, b)| a.to_bits() == b.to_bits());
    if !same {
        DIFF[id].fetch_add(1, Ordering::Relaxed);
        let m = port.iter().zip(host).map(|(a, b)| ulp_dist(*a, *b)).max().unwrap_or(u64::MAX);
        MAXULP[id].fetch_max(m, Ordering::Relaxed);
    }
}

pub fn write_report() {
    let Ok(p) = std::env::var("ZS_PROBE_OUT") else { return };
    if p.is_empty() {
        return;
    }
    let mut m = serde_json::Map::new();
    let names: Vec<&str> = MATH.iter().chain(OPS.iter()).cloned().collect();
    for (i, n) in names.iter().enumerate() {
        let c = CALLS[i].load(Ordering::Relaxed);
        if c > 0 {
            m.insert(n.to_string(), serde_json::json!({"calls": c, "differ": DIFF[i].load(Ordering::Relaxed), "max_ulp": MAXULP[i].load(Ordering::Relaxed)}));
        }
    }
    let s = SAMPLES.lock().unwrap();
    let samples: Vec<serde_json::Value> = s.iter().map(|(id, a, x, y)| serde_json::json!([MATH[*id], a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), x.to_bits(), y.to_bits()])).collect();
    let out = serde_json::json!({"host_math": cfg().host_math, "accel": cfg().accel, "counts": m, "math_samples": samples});
    std::fs::write(&p, serde_json::to_string(&out).unwrap()).expect("probe out");
}
