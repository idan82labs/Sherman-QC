//! The portable math kernels of zs_rust, compiled from its pinned source files (included by path, not copied):
//! libm-crate transcendentals (fmath.rs), numpy-order reductions (num.rs) and the pure-Rust linear algebra
//! (accel.rs). Exposed with a C ABI so the deterministic-math Python research reference calls exactly these.
#![allow(dead_code, clippy::missing_safety_doc)]
#[path = "../../../zs_rust/src/fmath.rs"]
mod fmath;
#[path = "../../../zs_rust/src/num.rs"]
mod num;
#[path = "../../../zs_rust/src/accel.rs"]
mod accel;
mod sha256hw {
    pub fn available() -> bool {
        false
    }
}
use fmath::DetF64;
use std::slice::{from_raw_parts as rp, from_raw_parts_mut as rpm};

/// unary: 0 sin 1 cos 2 tan 3 acos 4 exp 5 ln
#[no_mangle]
pub unsafe extern "C" fn dm_unary(f: i32, x: *const f64, y: *mut f64, n: usize) {
    let (x, y) = (rp(x, n), rpm(y, n));
    for i in 0..n {
        let v = x[i];
        y[i] = match f {
            0 => v.dsin(),
            1 => v.dcos(),
            2 => v.dtan(),
            3 => v.dacos(),
            4 => v.dexp(),
            5 => v.dln(),
            _ => f64::NAN,
        };
    }
}
/// binary: 0 atan2(a, b) 1 hypot(a, b) 2 pow(a, b)
#[no_mangle]
pub unsafe extern "C" fn dm_binary(f: i32, a: *const f64, b: *const f64, y: *mut f64, n: usize) {
    let (a, b, y) = (rp(a, n), rp(b, n), rpm(y, n));
    for i in 0..n {
        y[i] = match f {
            0 => a[i].datan2(b[i]),
            1 => a[i].dhypot(b[i]),
            2 => a[i].dpowf(b[i]),
            _ => f64::NAN,
        };
    }
}
#[no_mangle]
pub unsafe extern "C" fn dm_xtx(x: *const f64, rows: usize, k: usize, out: *mut f64) {
    rpm(out, k * k).copy_from_slice(&accel::xtx(rp(x, rows * k), rows, k));
}
#[no_mangle]
pub unsafe extern "C" fn dm_gemm_nt(a: *const f64, m: usize, b: *const f64, p: usize, k: usize, out: *mut f64) {
    rpm(out, m * p).copy_from_slice(&accel::gemm_nt(rp(a, m * k), m, rp(b, p * k), p, k));
}
/// batched eigh of `count` row-major n x n matrices
#[no_mangle]
pub unsafe extern "C" fn dm_eigh(m: *const f64, n: usize, count: usize, w: *mut f64, v: *mut f64) {
    let (m, w, v) = (rp(m, count * n * n), rpm(w, count * n), rpm(v, count * n * n));
    for c in 0..count {
        let (ww, vv) = accel::eigh(&m[c * n * n..(c + 1) * n * n], n);
        w[c * n..(c + 1) * n].copy_from_slice(&ww);
        v[c * n * n..(c + 1) * n * n].copy_from_slice(&vv);
    }
}
/// batched solve; returns the number of singular systems (their x is NaN)
#[no_mangle]
pub unsafe extern "C" fn dm_solve(a: *const f64, b: *const f64, n: usize, count: usize, x: *mut f64) -> usize {
    let (a, b, x) = (rp(a, count * n * n), rp(b, count * n), rpm(x, count * n));
    let mut bad = 0;
    for c in 0..count {
        match accel::solve(&a[c * n * n..(c + 1) * n * n], &b[c * n..(c + 1) * n], n) {
            Some(s) => x[c * n..(c + 1) * n].copy_from_slice(&s),
            None => {
                bad += 1;
                x[c * n..(c + 1) * n].fill(f64::NAN);
            }
        }
    }
    bad
}
#[no_mangle]
pub unsafe extern "C" fn dm_det(a: *const f64, n: usize) -> f64 {
    accel::det(rp(a, n * n), n)
}
#[no_mangle]
pub unsafe extern "C" fn dm_lstsq(a: *const f64, m: usize, n: usize, b: *const f64, rcond: f64, x: *mut f64) {
    rpm(x, n).copy_from_slice(&accel::lstsq(rp(a, m * n), m, n, rp(b, m), rcond));
}
