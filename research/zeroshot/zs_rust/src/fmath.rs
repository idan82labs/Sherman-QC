//! Platform-independent floating point: every transcendental goes through the pure-Rust `libm` crate (a port of
//! musl's libm), never the host C library (Apple libm, glibc, the Windows UCRT differ in the last bit). Fused
//! multiply-adds are explicit (`dfma`), never contracted by the compiler: the hardware instruction where the target
//! has one (aarch64; x86-64 with the `fma` feature, e.g. target-cpu=x86-64-v3), else libm's correctly rounded
//! software fma. IEEE 754 makes fma, sqrt, +, -, *, / correctly rounded, so all paths give the same bits.
//!
//! Methods are named d* so they cannot be shadowed by (or confused with) the std inherent methods that call the
//! host libm. `tests/test_prod_zs_rust.py` greps the sources for std transcendental calls.
#![allow(dead_code)]

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
        libm::sin(self)
    }
    #[inline]
    fn dcos(self) -> f64 {
        libm::cos(self)
    }
    #[inline]
    fn dtan(self) -> f64 {
        libm::tan(self)
    }
    #[inline]
    fn dacos(self) -> f64 {
        libm::acos(self)
    }
    /// self.atan2(x) = atan2(self, x), as std
    #[inline]
    fn datan2(self, x: f64) -> f64 {
        libm::atan2(self, x)
    }
    #[inline]
    fn dexp(self) -> f64 {
        libm::exp(self)
    }
    #[inline]
    fn dln(self) -> f64 {
        libm::log(self)
    }
    #[inline]
    fn dhypot(self, y: f64) -> f64 {
        libm::hypot(self, y)
    }
    #[inline]
    fn dpowf(self, y: f64) -> f64 {
        libm::pow(self, y)
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
