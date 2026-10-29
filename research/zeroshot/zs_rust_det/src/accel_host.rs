//! Accelerate (numpy's BLAS/LAPACK on this Mac, `$NEWLAPACK$ILP64`) called with numpy's conventions.
//! Research probe only (macOS).
#![allow(dead_code)]
use crate::fmath::DetF64;
type I = i64;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    #[link_name = "cblas_dsyrk$NEWLAPACK$ILP64"]
    fn cblas_dsyrk(o: i32, u: i32, t: i32, n: I, k: I, alpha: f64, a: *const f64, lda: I, beta: f64, c: *mut f64, ldc: I);
    #[link_name = "cblas_dgemm$NEWLAPACK$ILP64"]
    fn cblas_dgemm(o: i32, ta: i32, tb: i32, m: I, n: I, k: I, alpha: f64, a: *const f64, lda: I, b: *const f64, ldb: I, beta: f64, c: *mut f64, ldc: I);
    #[link_name = "cblas_dgemv$NEWLAPACK$ILP64"]
    fn cblas_dgemv(o: i32, t: i32, m: I, n: I, alpha: f64, a: *const f64, lda: I, x: *const f64, incx: I, beta: f64, y: *mut f64, incy: I);
    #[link_name = "dsyevd$NEWLAPACK$ILP64"]
    fn dsyevd(jobz: *const u8, uplo: *const u8, n: *const I, a: *mut f64, lda: *const I, w: *mut f64, work: *mut f64, lwork: *const I, iwork: *mut I, liwork: *const I, info: *mut I);
    #[link_name = "dgesv$NEWLAPACK$ILP64"]
    fn dgesv(n: *const I, nrhs: *const I, a: *mut f64, lda: *const I, ipiv: *mut I, b: *mut f64, ldb: *const I, info: *mut I);
    #[link_name = "dgetrf$NEWLAPACK$ILP64"]
    fn dgetrf(m: *const I, n: *const I, a: *mut f64, lda: *const I, ipiv: *mut I, info: *mut I);
    #[link_name = "dgelsd$NEWLAPACK$ILP64"]
    fn dgelsd(m: *const I, n: *const I, nrhs: *const I, a: *mut f64, lda: *const I, b: *mut f64, ldb: *const I, s: *mut f64, rcond: *const f64, rank: *mut I, work: *mut f64, lwork: *const I, iwork: *mut I, info: *mut I);
}
const ROW: i32 = 101;
const NOT: i32 = 111;
const TR: i32 = 112;
const UP: i32 = 121;

/// numpy X.T @ X: matmul's syrk path (RowMajor, Upper, Trans, P=k, N=rows, lda=k), then the upper triangle copied
pub fn xtx(x: &[f64], rows: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; k * k];
    unsafe { cblas_dsyrk(ROW, UP, TR, k as I, rows as I, 1.0, x.as_ptr(), k as I, 0.0, c.as_mut_ptr(), k as I) };
    for i in 0..k {
        for j in (i + 1)..k {
            c[j * k + i] = c[i * k + j];
        }
    }
    c
}

/// numpy A @ B.T (A m x k, B p x k, C-contiguous): gemv when p == 1 or m == 1, else gemm(NoTrans, Trans)
pub fn gemm_nt(a: &[f64], m: usize, b: &[f64], p: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * p];
    if m == 0 || p == 0 {
        return c;
    }
    unsafe {
        if p == 1 {
            cblas_dgemv(ROW, NOT, m as I, k as I, 1.0, a.as_ptr(), k as I, b.as_ptr(), 1, 0.0, c.as_mut_ptr(), 1);
        } else if m == 1 {
            // vector @ matrix (B.T is k x p with strides (1, k)): gemv on B (p x k row-major), no transpose
            cblas_dgemv(ROW, NOT, p as I, k as I, 1.0, b.as_ptr(), k as I, a.as_ptr(), 1, 0.0, c.as_mut_ptr(), 1);
        } else {
            cblas_dgemm(ROW, NOT, TR, m as I, p as I, k as I, 1.0, a.as_ptr(), k as I, b.as_ptr(), k as I, 0.0, c.as_mut_ptr(), p as I);
        }
    }
    c
}

fn colmajor(m: &[f64], n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i + j * n] = m[i * n + j];
        }
    }
    a
}

/// numpy.linalg.eigh (UPLO 'L'): dsyevd('V','L') with numpy's workspace query
pub fn eigh(m: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = colmajor(m, n);
    let nn = n as I;
    let mut w = vec![0.0f64; n];
    let mut info: I = 0;
    let mut qw = 0.0f64;
    let mut qi: I = 0;
    let m1: I = -1;
    unsafe { dsyevd(b"V".as_ptr(), b"L".as_ptr(), &nn, a.as_mut_ptr(), &nn, w.as_mut_ptr(), &mut qw, &m1, &mut qi, &m1, &mut info) };
    let lw = qw as I;
    let li = qi;
    let mut work = vec![0.0f64; lw.max(1) as usize];
    let mut iwork = vec![0 as I; li.max(1) as usize];
    unsafe { dsyevd(b"V".as_ptr(), b"L".as_ptr(), &nn, a.as_mut_ptr(), &nn, w.as_mut_ptr(), work.as_mut_ptr(), &lw, iwork.as_mut_ptr(), &li, &mut info) };
    if info != 0 {
        return (vec![f64::NAN; n], vec![f64::NAN; n * n]);
    }
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            v[i * n + j] = a[i + j * n];
        }
    }
    (w, v)
}

pub fn solve(a_rm: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = colmajor(a_rm, n);
    let nn = n as I;
    let one: I = 1;
    let mut ipiv = vec![0 as I; n];
    let mut x = b[..n].to_vec();
    let mut info: I = 0;
    unsafe { dgesv(&nn, &one, a.as_mut_ptr(), &nn, ipiv.as_mut_ptr(), x.as_mut_ptr(), &nn, &mut info) };
    if info != 0 { None } else { Some(x) }
}

pub fn det(a_rm: &[f64], n: usize) -> f64 {
    let mut a = colmajor(a_rm, n);
    let nn = n as I;
    let mut ipiv = vec![0 as I; n];
    let mut info: I = 0;
    unsafe { dgetrf(&nn, &nn, a.as_mut_ptr(), &nn, ipiv.as_mut_ptr(), &mut info) };
    if info != 0 {
        return 0.0;
    }
    let mut sign = 1.0f64;
    for i in 0..n {
        if ipiv[i] != (i as I) + 1 {
            sign = -sign;
        }
    }
    let mut logdet = 0.0f64;
    for i in 0..n {
        let mut e = a[i * n + i];
        if e < 0.0 {
            sign = -sign;
            e = -e;
        }
        logdet += e.dln();
    }
    sign * logdet.dexp()
}

/// numpy.linalg.lstsq(A, b, rcond)[0]: dgelsd with numpy's workspace query (lda = m, ldb = max(m, n))
pub fn lstsq(a_rows: &[f64], m: usize, n: usize, b: &[f64], rcond: f64) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if m == 0 {
        return vec![0.0; n];
    }
    let mut a = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            a[i + j * m] = a_rows[i * n + j];
        }
    }
    let ldb = m.max(n);
    let mut bb = vec![0.0f64; ldb];
    bb[..m].copy_from_slice(&b[..m]);
    let (mm, nn, one, lda, ldbb) = (m as I, n as I, 1 as I, m.max(1) as I, ldb as I);
    let mut s = vec![0.0f64; m.min(n)];
    let mut rank: I = 0;
    let mut info: I = 0;
    let mut qw = 0.0f64;
    let mut qi: I = 0;
    let m1: I = -1;
    unsafe { dgelsd(&mm, &nn, &one, a.as_mut_ptr(), &lda, bb.as_mut_ptr(), &ldbb, s.as_mut_ptr(), &rcond, &mut rank, &mut qw, &m1, &mut qi, &mut info) };
    let lw = qw as I;
    let mut work = vec![0.0f64; lw.max(1) as usize];
    let mut iwork = vec![0 as I; qi.max(1) as usize];
    unsafe { dgelsd(&mm, &nn, &one, a.as_mut_ptr(), &lda, bb.as_mut_ptr(), &ldbb, s.as_mut_ptr(), &rcond, &mut rank, work.as_mut_ptr(), &lw, iwork.as_mut_ptr(), &mut info) };
    bb[..n].to_vec()
}
