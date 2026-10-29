//! Platform-independent floating point: every transcendental goes through the pure-Rust `libm` crate (a port of
//! musl's libm), never the host C library (Apple libm, glibc, the Windows UCRT differ in the last bit). Fused
//! multiply-adds are explicit (`dfma`), never contracted by the compiler: the hardware instruction where the target
//! has one (aarch64; x86-64 with the `fma` feature, e.g. target-cpu=x86-64-v3), else libm's correctly rounded
//! software fma. IEEE 754 makes fma, sqrt, +, -, *, / correctly rounded, so all paths give the same bits.
//!
//! Methods are named d* so they cannot be shadowed by (or confused with) the std inherent methods that call the
//! host libm. `tests/test_prod_zs_rust.py` greps the sources for std transcendental calls.
#![allow(dead_code)]

use crate::probe;

#[inline(always)]
fn pick(id: usize, args: &[f64], port: impl Fn() -> f64, host: impl Fn() -> f64) -> f64 {
    let c = probe::cfg();
    let h = c.host_math & (1 << id) != 0;
    if c.count {
        let a = port();
        let b = host();
        probe::math_hit(id, a, b, args);
        return if h { b } else { a };
    }
    if h { host() } else { port() }
}
// host (Apple libm) calls, never inlined so LLVM cannot fuse sin+cos into sincos
#[inline(never)] fn h_sin(x: f64) -> f64 { x.sin() }
#[inline(never)] fn h_cos(x: f64) -> f64 { x.cos() }
#[inline(never)] fn h_tan(x: f64) -> f64 { x.tan() }
#[inline(never)] fn h_acos(x: f64) -> f64 { x.acos() }
#[inline(never)] fn h_atan2(y: f64, x: f64) -> f64 { y.atan2(x) }
#[inline(never)] fn h_exp(x: f64) -> f64 { x.exp() }
#[inline(never)] fn h_ln(x: f64) -> f64 { x.ln() }
#[inline(never)] fn h_hypot(x: f64, y: f64) -> f64 { x.hypot(y) }
#[inline(never)] fn h_pow(x: f64, y: f64) -> f64 { x.powf(y) }

pub trait DetF64: Copy {
    fn dsin(self) -> f64;
    fn dcos(self) -> f64;
    fn dtan(self) -> f64;
    fn dacos(self) -> f64;
    fn datan2(self, x: f64) -> f64;
    fn dexp(self) -> f64;
    fn dln(self) -> f64;
    fn dhypot(self, y: f64) -> f64;
    fn dpowf(self, y: f64) -> f64;
    fn dfma(self, b: f64, c: f64) -> f64;
}

impl DetF64 for f64 {
    #[inline]
    fn dsin(self) -> f64 {
        pick(0, &[self], || libm::sin(self), || h_sin(self))
    }
    #[inline]
    fn dcos(self) -> f64 {
        pick(1, &[self], || libm::cos(self), || h_cos(self))
    }
    #[inline]
    fn dtan(self) -> f64 {
        pick(2, &[self], || libm::tan(self), || h_tan(self))
    }
    #[inline]
    fn dacos(self) -> f64 {
        pick(3, &[self], || libm::acos(self), || h_acos(self))
    }
    /// self.atan2(x) = atan2(self, x), as std
    #[inline]
    fn datan2(self, x: f64) -> f64 {
        pick(4, &[self, x], || libm::atan2(self, x), || h_atan2(self, x))
    }
    #[inline]
    fn dexp(self) -> f64 {
        pick(5, &[self], || libm::exp(self), || h_exp(self))
    }
    #[inline]
    fn dln(self) -> f64 {
        pick(6, &[self], || libm::log(self), || h_ln(self))
    }
    #[inline]
    fn dhypot(self, y: f64) -> f64 {
        pick(7, &[self, y], || libm::hypot(self, y), || h_hypot(self, y))
    }
    #[inline]
    fn dpowf(self, y: f64) -> f64 {
        pick(8, &[self, y], || libm::pow(self, y), || h_pow(self, y))
    }
    /// fma(self, b, c) = self * b + c with one rounding
    #[inline]
    fn dfma(self, b: f64, c: f64) -> f64 {
        #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
        {
            self.mul_add(b, c)
        }
        #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
        {
            libm::fma(self, b, c)
        }
    }
}

pub trait DetF32: Copy {
    fn dfma(self, b: f32, c: f32) -> f32;
}

impl DetF32 for f32 {
    #[inline]
    fn dfma(self, b: f32, c: f32) -> f32 {
        #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
        {
            self.mul_add(b, c)
        }
        #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
        {
            libm::fmaf(self, b, c)
        }
    }
}

/// numpy float mod remainder base: C fmod (exact by definition; libm's for uniformity)
#[inline]
pub fn fmod(a: f64, b: f64) -> f64 {
    libm::fmod(a, b)
}

/// what the build fused and which math library it uses (selftest / version report)
pub fn build_info() -> serde_json::Value {
    let fma = if cfg!(any(target_arch = "aarch64", target_feature = "fma")) { "hardware (target feature)" } else { "libm::fma (hardware when detected at run time, else software; same bits)" };
    serde_json::json!({
        "target_arch": std::env::consts::ARCH, "target_os": std::env::consts::OS,
        "fma": fma, "libm": "libm crate (pure Rust, musl port)", "linalg": "pure Rust (accel.rs, LAPACK 3.9 algorithms)",
        "sha256": if crate::sha256hw::available() { "armv8 sha2 instructions (sha256hw)" } else { "sha2 crate" },
    })
}
