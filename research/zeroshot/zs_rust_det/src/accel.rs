//! numpy.linalg / matmul calls in pure Rust (no BLAS / LAPACK library, no platform dependence).
//!
//! Up to revision 3 of the port these went through macOS Accelerate (numpy's LAPACK on the Mac) for bit parity
//! with the Python lane. The portable build replaces them with hand ports of the reference LAPACK 3.9 algorithms
//! numpy calls, so the conventions (eigenvector signs, pivoting, eigenvalue order) are LAPACK's:
//!
//! * `X.T @ X`  -> pairwise sums per entry (numpy's summation; Accelerate's dsyrk blocking is not reproduced);
//! * `A @ B.T`  -> plain dot products in index order;
//! * `eigh`     -> dsyevd('V', 'L') for n <= 25: dsytd2 + dsteqr('I') + dorm2r (dormtr), reference BLAS order;
//! * `solve`    -> dgesv: dgetrf2 (recursive, reciprocal scaling) + dgetrs;
//! * `det`      -> dgetrf2 + umath_linalg's sign * exp(sum log|u_ii|);
//! * `lstsq`    -> minimum-norm least squares by one-sided Jacobi SVD with dgelsd's rcond truncation
//!                 (dgelsd's bidiagonal divide-and-conquer is not reproduced; same solution to rounding).
//!
//! Every operation is a plain IEEE expression with no fused multiply-add and no reassociation, so the bits are
//! the same on every target. Transcendentals go through `fmath` (pure-Rust libm).
#![allow(dead_code)]

use crate::fmath::DetF64;
use crate::num::pw_sum_by;

// LAPACK dlamch values (LAPACK 3.x, round-to-nearest)
const EPS: f64 = f64::EPSILON * 0.5; // dlamch('E') = 2^-53
const SAFMIN: f64 = f64::MIN_POSITIVE; // dlamch('S')

/// numpy `X.T @ X` for a C-contiguous (rows x k) matrix: row-major k x k result (symmetric by construction).
pub fn p_xtx(x: &[f64], rows: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; k * k];
    for i in 0..k {
        for j in i..k {
            let s = pw_sum_by(0, rows, &|r: usize| x[r * k + i] * x[r * k + j]);
            c[i * k + j] = s;
            c[j * k + i] = s;
        }
    }
    c
}

/// numpy `A @ B.T` for C-contiguous A (m x k) and B (p x k).
pub fn p_gemm_nt(a: &[f64], m: usize, b: &[f64], p: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * p];
    for i in 0..m {
        for j in 0..p {
            let mut s = 0.0;
            for q in 0..k {
                s += a[i * k + q] * b[j * k + q];
            }
            c[i * p + j] = s;
        }
    }
    c
}

// ------------------------------------------------------------------------------------------ reference LAPACK

#[inline]
fn sign(a: f64, b: f64) -> f64 {
    // Fortran SIGN(A, B): |A| with the sign of B (B = -0.0 counts as non-negative in LAPACK's compilers)
    if b >= 0.0 {
        a.abs()
    } else {
        -a.abs()
    }
}

/// dlapy2 (LAPACK 3.9): sqrt(x^2 + y^2) avoiding overflow
fn dlapy2(x: f64, y: f64) -> f64 {
    let x_nan = x.is_nan();
    let y_nan = y.is_nan();
    if x_nan {
        return x;
    }
    if y_nan {
        return y;
    }
    let xa = x.abs();
    let ya = y.abs();
    let w = xa.max(ya);
    let z = xa.min(ya);
    if z == 0.0 || w > f64::MAX {
        w
    } else {
        let q = z / w;
        w * (1.0 + q * q).sqrt()
    }
}

/// dlartg (LAPACK 3.9, before the 3.10 rewrite): plane rotation with r = sqrt(f^2 + g^2), c >= 0 when |f| > |g|
fn dlartg(f: f64, g: f64) -> (f64, f64, f64) {
    // safmn2 = base^int(log(safmin/eps)/log(base)/2) = 2^-484
    let safmn2 = f64::from_bits(((1023 - 484) as u64) << 52);
    let safmx2 = 1.0 / safmn2;
    if g == 0.0 {
        return (1.0, 0.0, f);
    }
    if f == 0.0 {
        return (0.0, 1.0, g);
    }
    let (mut f1, mut g1) = (f, g);
    let mut scale = f1.abs().max(g1.abs());
    let (mut cs, mut sn, mut r);
    if scale >= safmx2 {
        let mut count = 0;
        loop {
            count += 1;
            f1 *= safmn2;
            g1 *= safmn2;
            scale = f1.abs().max(g1.abs());
            if !(scale >= safmx2 && count < 20) {
                break;
            }
        }
        r = (f1 * f1 + g1 * g1).sqrt();
        cs = f1 / r;
        sn = g1 / r;
        for _ in 0..count {
            r *= safmx2;
        }
    } else if scale <= safmn2 {
        let mut count = 0;
        loop {
            count += 1;
            f1 *= safmx2;
            g1 *= safmx2;
            scale = f1.abs().max(g1.abs());
            if !(scale <= safmn2) {
                break;
            }
        }
        r = (f1 * f1 + g1 * g1).sqrt();
        cs = f1 / r;
        sn = g1 / r;
        for _ in 0..count {
            r *= safmn2;
        }
    } else {
        r = (f1 * f1 + g1 * g1).sqrt();
        cs = f1 / r;
        sn = g1 / r;
    }
    if f.abs() > g.abs() && cs < 0.0 {
        cs = -cs;
        sn = -sn;
        r = -r;
    }
    (cs, sn, r)
}

/// dlaev2: eigen-decomposition of [[a, b], [b, c]] -> (rt1, rt2, cs1, sn1), |rt1| >= |rt2|, (cs1, sn1) the unit
/// eigenvector of rt1
pub fn dlaev2(a: f64, b: f64, c: f64) -> (f64, f64, f64, f64) {
    let sm = a + c;
    let df = a - c;
    let adf = df.abs();
    let tb = b + b;
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        let q = ab / adf;
        adf * (1.0 + q * q).sqrt()
    } else if adf < ab {
        let q = adf / ab;
        ab * (1.0 + q * q).sqrt()
    } else {
        ab * 2f64.sqrt()
    };
    let (rt1, rt2, sgn1);
    if sm < 0.0 {
        rt1 = 0.5 * (sm - rt);
        sgn1 = -1.0;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else if sm > 0.0 {
        rt1 = 0.5 * (sm + rt);
        sgn1 = 1.0;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else {
        rt1 = 0.5 * rt;
        rt2 = -0.5 * rt;
        sgn1 = 1.0;
    }
    let (cs, sgn2) = if df >= 0.0 { (df + rt, 1.0) } else { (df - rt, -1.0) };
    let acs = cs.abs();
    let (mut cs1, mut sn1);
    if acs > ab {
        let ct = -tb / cs;
        sn1 = 1.0 / (1.0 + ct * ct).sqrt();
        cs1 = ct * sn1;
    } else if ab == 0.0 {
        cs1 = 1.0;
        sn1 = 0.0;
    } else {
        let tn = -cs / tb;
        cs1 = 1.0 / (1.0 + tn * tn).sqrt();
        sn1 = tn * cs1;
    }
    if sgn1 == sgn2 {
        let tn = cs1;
        cs1 = -sn1;
        sn1 = tn;
    }
    (rt1, rt2, cs1, sn1)
}

/// dlasr('R', 'V', direct, m rows, ncol columns): apply the plane rotations (c[j], s[j]) to columns j, j+1 of the
/// column-major m x * matrix z starting at column c0 (leading dimension ldz).
fn dlasr_rv(forward: bool, m: usize, ncol: usize, c: &[f64], s: &[f64], z: &mut [f64], ldz: usize, c0: usize) {
    if ncol < 2 {
        return;
    }
    let mut body = |j: usize| {
        let ct = c[j];
        let st = s[j];
        if ct != 1.0 || st != 0.0 {
            for i in 0..m {
                let a_j1 = (c0 + j + 1) * ldz + i;
                let a_j = (c0 + j) * ldz + i;
                let temp = z[a_j1];
                z[a_j1] = ct * temp - st * z[a_j];
                z[a_j] = st * temp + ct * z[a_j];
            }
        }
    };
    if forward {
        for j in 0..(ncol - 1) {
            body(j);
        }
    } else {
        for j in (0..(ncol - 1)).rev() {
            body(j);
        }
    }
}

/// dsteqr(compz = 'I'): eigenvalues d (ascending on return) and eigenvectors z (column-major n x n) of the
/// symmetric tridiagonal (d, e). Returns info (0 = converged).
fn dsteqr_i(n: usize, d: &mut [f64], e: &mut [f64], z: &mut [f64]) -> usize {
    for v in z.iter_mut() {
        *v = 0.0;
    }
    for i in 0..n {
        z[i * n + i] = 1.0;
    }
    if n <= 1 {
        return 0;
    }
    let eps = EPS;
    let eps2 = eps * eps;
    let safmin = SAFMIN;
    let safmax = 1.0 / safmin;
    let ssfmax = safmax.sqrt() / 3.0;
    let ssfmin = safmin.sqrt() / eps2;
    let maxit = 30;
    let nmaxit = n * maxit;
    let mut jtot = 0usize;
    let mut work = vec![0.0f64; 2 * (n - 1)];
    // 1-based indexing helpers over d (len n) and e (len n-1)
    let mut l1: usize = 1;
    let nm1 = n - 1;
    'outer: loop {
        if l1 > n {
            break;
        }
        if l1 > 1 {
            e[l1 - 2] = 0.0;
        }
        let mut m = n;
        if l1 <= nm1 {
            let mut mm = l1;
            let mut found = false;
            while mm <= nm1 {
                let tst = e[mm - 1].abs();
                if tst == 0.0 {
                    found = true;
                    break;
                }
                if tst <= (d[mm - 1].abs().sqrt() * d[mm].abs().sqrt()) * eps {
                    e[mm - 1] = 0.0;
                    found = true;
                    break;
                }
                mm += 1;
            }
            if found {
                m = mm;
            }
        }
        let mut l = l1;
        let lsv = l;
        let mut lend = m;
        let lendsv = lend;
        l1 = m + 1;
        if lend == l {
            continue 'outer;
        }
        // scale submatrix in rows and columns l to lend
        let mut anorm = 0.0f64;
        for i in l..=lend {
            let a = d[i - 1].abs();
            if anorm < a || a.is_nan() {
                anorm = a;
            }
        }
        for i in l..lend {
            let a = e[i - 1].abs();
            if anorm < a || a.is_nan() {
                anorm = a;
            }
        }
        let mut iscale = 0;
        if anorm == 0.0 {
            continue 'outer;
        }
        if anorm > ssfmax {
            iscale = 1;
            scale_range(d, l, lend, anorm, ssfmax);
            scale_range(e, l, lend - 1, anorm, ssfmax);
        }
        if anorm < ssfmin {
            iscale = 2;
            scale_range(d, l, lend, anorm, ssfmin);
            scale_range(e, l, lend - 1, anorm, ssfmin);
        }
        // choose between QL and QR iteration
        if d[lend - 1].abs() < d[l - 1].abs() {
            lend = lsv;
            l = lendsv;
        }
        if lend > l {
            // QL iteration: look for small subdiagonal element
            loop {
                let mut m = lend;
                if l != lend {
                    let mut mm = l;
                    while mm <= lend - 1 {
                        let tst = e[mm - 1].abs() * e[mm - 1].abs();
                        if tst <= (eps2 * d[mm - 1].abs()) * d[mm].abs() + safmin {
                            break;
                        }
                        mm += 1;
                    }
                    m = mm.min(lend);
                }
                if m < lend {
                    e[m - 1] = 0.0;
                }
                let mut p = d[l - 1];
                if m == l {
                    // eigenvalue found
                    d[l - 1] = p;
                    l += 1;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if m == l + 1 {
                    let (rt1, rt2, c, s) = dlaev2(d[l - 1], e[l - 1], d[l]);
                    work[l - 1] = c;
                    work[n - 1 + l - 1] = s;
                    let (cw, sw) = work.split_at(n - 1);
                    dlasr_rv(false, n, 2, &cw[l - 1..], &sw[l - 1..], z, n, l - 1);
                    d[l - 1] = rt1;
                    d[l] = rt2;
                    e[l - 1] = 0.0;
                    l += 2;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;
                // form shift
                let mut g = (d[l] - p) / (2.0 * e[l - 1]);
                let mut r = dlapy2(g, 1.0);
                g = d[m - 1] - p + (e[l - 1] / (g + sign(r, g)));
                let mut s = 1.0;
                let mut c = 1.0;
                p = 0.0;
                let mm1 = m - 1;
                let mut i = mm1;
                while i >= l {
                    let f = s * e[i - 1];
                    let b = c * e[i - 1];
                    let (c2, s2, r2) = dlartg(g, f);
                    c = c2;
                    s = s2;
                    r = r2;
                    if i != m - 1 {
                        e[i] = r;
                    }
                    g = d[i] - p;
                    r = (d[i - 1] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i] = g + p;
                    g = c * r - b;
                    work[i - 1] = c;
                    work[n - 1 + i - 1] = -s;
                    if i == l {
                        break;
                    }
                    i -= 1;
                }
                let mm = m - l + 1;
                let (cw, sw) = work.split_at(n - 1);
                dlasr_rv(false, n, mm, &cw[l - 1..], &sw[l - 1..], z, n, l - 1);
                d[l - 1] -= p;
                e[l - 1] = g;
            }
        } else {
            // QR iteration: look for small superdiagonal element
            loop {
                let mut m = lend;
                if l != lend {
                    let mut mm = l;
                    let mut found = false;
                    while mm >= lend + 1 {
                        let tst = e[mm - 2].abs() * e[mm - 2].abs();
                        if tst <= (eps2 * d[mm - 1].abs()) * d[mm - 2].abs() + safmin {
                            found = true;
                            break;
                        }
                        mm -= 1;
                    }
                    m = if found { mm } else { lend };
                }
                if m > lend {
                    e[m - 2] = 0.0;
                }
                let mut p = d[l - 1];
                if m == l {
                    d[l - 1] = p;
                    if l == 1 {
                        break;
                    }
                    l -= 1;
                    if l >= lend {
                        continue;
                    }
                    break;
                }
                if m + 1 == l {
                    let (rt1, rt2, c, s) = dlaev2(d[l - 2], e[l - 2], d[l - 1]);
                    work[m - 1] = c;
                    work[n - 1 + m - 1] = s;
                    let (cw, sw) = work.split_at(n - 1);
                    dlasr_rv(true, n, 2, &cw[m - 1..], &sw[m - 1..], z, n, l - 2);
                    d[l - 2] = rt1;
                    d[l - 1] = rt2;
                    e[l - 2] = 0.0;
                    if l < 2 + lend {
                        break;
                    }
                    l -= 2;
                    continue;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;
                let mut g = (d[l - 2] - p) / (2.0 * e[l - 2]);
                let mut r = dlapy2(g, 1.0);
                g = d[m - 1] - p + (e[l - 2] / (g + sign(r, g)));
                let mut s = 1.0;
                let mut c = 1.0;
                p = 0.0;
                let lm1 = l - 1;
                for i in m..=lm1 {
                    let f = s * e[i - 1];
                    let b = c * e[i - 1];
                    let (c2, s2, r2) = dlartg(g, f);
                    c = c2;
                    s = s2;
                    r = r2;
                    if i != m {
                        e[i - 2] = r;
                    }
                    g = d[i - 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i - 1] = g + p;
                    g = c * r - b;
                    work[i - 1] = c;
                    work[n - 1 + i - 1] = s;
                }
                let mm = l - m + 1;
                let (cw, sw) = work.split_at(n - 1);
                dlasr_rv(true, n, mm, &cw[m - 1..], &sw[m - 1..], z, n, m - 1);
                d[l - 1] -= p;
                e[lm1 - 1] = g;
            }
        }
        // undo scaling
        if iscale == 1 {
            scale_range(d, lsv, lendsv, ssfmax, anorm);
            scale_range(e, lsv, lendsv - 1, ssfmax, anorm);
        } else if iscale == 2 {
            scale_range(d, lsv, lendsv, ssfmin, anorm);
            scale_range(e, lsv, lendsv - 1, ssfmin, anorm);
        }
        if jtot >= nmaxit {
            return e.iter().filter(|&&x| x != 0.0).count().max(1);
        }
    }
    // selection sort of eigenvalues (ascending) with their vectors
    for ii in 2..=n {
        let i = ii - 1;
        let mut k = i;
        let mut p = d[i - 1];
        for j in ii..=n {
            if d[j - 1] < p {
                k = j;
                p = d[j - 1];
            }
        }
        if k != i {
            d[k - 1] = d[i - 1];
            d[i - 1] = p;
            for r in 0..n {
                z.swap((i - 1) * n + r, (k - 1) * n + r);
            }
        }
    }
    0
}

/// dlascl('G') on x[lo-1 .. hi] (1-based inclusive) from cfrom to cto, in LAPACK's overflow-safe steps
fn scale_range(x: &mut [f64], lo: usize, hi: usize, cfrom: f64, cto: f64) {
    if hi < lo {
        return;
    }
    let smlnum = SAFMIN;
    let bignum = 1.0 / smlnum;
    let mut cfromc = cfrom;
    let mut ctoc = cto;
    loop {
        let cfrom1 = cfromc * smlnum;
        let (mul, done);
        if cfrom1 == cfromc {
            mul = ctoc / cfromc;
            done = true;
        } else {
            let cto1 = ctoc / bignum;
            if cto1 == ctoc {
                mul = ctoc;
                done = true;
                cfromc = 1.0;
            } else if cfrom1.abs() > ctoc.abs() && ctoc != 0.0 {
                mul = smlnum;
                done = false;
                cfromc = cfrom1;
            } else if cto1.abs() > cfromc.abs() {
                mul = bignum;
                done = false;
                ctoc = cto1;
            } else {
                mul = ctoc / cfromc;
                done = true;
            }
        }
        for v in x[lo - 1..hi].iter_mut() {
            *v *= mul;
        }
        if done {
            break;
        }
    }
}

/// numpy.linalg.eigh (UPLO 'L') of a row-major symmetric n x n matrix: (w ascending, v row-major with
/// eigenvector j in column j). LAPACK dsyevd('V', 'L') for small n: dsytd2('L') -> dstedc('I') = dsteqr('I')
/// (n <= 25) -> dormtr('L', 'L', 'N') = dorm2r.
pub fn p_eigh(m: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    // column-major copy; only the lower triangle is referenced
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i + j * n] = m[i * n + j];
        }
    }
    let at = |i: usize, j: usize| i + j * n;
    if n == 1 {
        return (vec![a[0]], vec![1.0]);
    }
    // dsyevd's scaling of matrices with tiny / huge entries is not needed for geometry in mm (|a| in 1e-150..1e150)
    // ---- dsytd2('L')
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n - 1];
    let mut tau = vec![0.0f64; n - 1];
    for i in 0..(n - 1) {
        // dlarfg(n-i-1, a(i+1,i), a(i+2:n, i))
        let nn = n - i - 1;
        let mut alpha = a[at(i + 1, i)];
        let taui;
        if nn <= 1 {
            taui = 0.0;
        } else {
            // dnrm2 of x = a(i+2.., i) (reference BLAS before 3.10: scaled sum of squares)
            let xs: Vec<f64> = ((i + 2)..n).map(|r| a[at(r, i)]).collect();
            let xnorm = dnrm2_old(&xs);
            if xnorm == 0.0 {
                taui = 0.0;
            } else {
                let mut beta = -sign(dlapy2(alpha, xnorm), alpha);
                let safmin = SAFMIN / EPS;
                let mut knt = 0;
                if beta.abs() < safmin {
                    let rsafmn = 1.0 / safmin;
                    loop {
                        knt += 1;
                        for r in (i + 2)..n {
                            a[at(r, i)] *= rsafmn;
                        }
                        beta *= rsafmn;
                        alpha *= rsafmn;
                        if !(beta.abs() < safmin && knt < 20) {
                            break;
                        }
                    }
                    let xs: Vec<f64> = ((i + 2)..n).map(|r| a[at(r, i)]).collect();
                    let xnorm = dnrm2_old(&xs);
                    beta = -sign(dlapy2(alpha, xnorm), alpha);
                }
                taui = (beta - alpha) / beta;
                let sc = 1.0 / (alpha - beta);
                for r in (i + 2)..n {
                    a[at(r, i)] *= sc;
                }
                for _ in 0..knt {
                    beta *= safmin;
                }
                alpha = beta;
            }
        }
        a[at(i + 1, i)] = alpha;
        e[i] = a[at(i + 1, i)];
        if taui != 0.0 {
            a[at(i + 1, i)] = 1.0;
            let k = n - i - 1; // order of the trailing block A(i+1.., i+1..)
            let v: Vec<f64> = ((i + 1)..n).map(|r| a[at(r, i)]).collect();
            // dsymv('L', k, taui, A22, v, 0, y)
            let mut y = vec![0.0f64; k];
            for j in 0..k {
                let temp1 = taui * v[j];
                let mut temp2 = 0.0;
                y[j] += temp1 * a[at(i + 1 + j, i + 1 + j)];
                for ii in (j + 1)..k {
                    y[ii] += temp1 * a[at(i + 1 + ii, i + 1 + j)];
                    temp2 += a[at(i + 1 + ii, i + 1 + j)] * v[ii];
                }
                y[j] += taui * temp2;
            }
            // alpha = -1/2 taui ddot(y, v); y += alpha v
            let mut dot = 0.0;
            for j in 0..k {
                dot += y[j] * v[j];
            }
            let alph = -0.5 * taui * dot;
            for j in 0..k {
                y[j] += alph * v[j];
            }
            // dsyr2('L', k, -1, v, y, A22)
            for j in 0..k {
                if v[j] != 0.0 || y[j] != 0.0 {
                    let temp1 = -1.0 * y[j];
                    let temp2 = -1.0 * v[j];
                    for ii in j..k {
                        let p = at(i + 1 + ii, i + 1 + j);
                        a[p] = a[p] + v[ii] * temp1 + y[ii] * temp2;
                    }
                }
            }
            a[at(i + 1, i)] = e[i];
        }
        d[i] = a[at(i, i)];
        tau[i] = taui;
    }
    d[n - 1] = a[at(n - 1, n - 1)];
    // ---- dstedc('I') = dsteqr('I') for n <= 25
    let mut z = vec![0.0f64; n * n];
    let info = dsteqr_i(n, &mut d, &mut e, &mut z);
    if info != 0 {
        return (vec![f64::NAN; n], vec![f64::NAN; n * n]);
    }
    // ---- dormtr('L', 'L', 'N'): rows 2..n of Z := Q2 * Z(2:n, :) with Q2 = H(1) ... H(n-2) from A(2:n, 1:n-2)
    // dorm2r left / no-transpose: i = k down to 1 (k = n-1 reflectors of order nq-1)
    let mq = n - 1;
    for ri in (0..(n - 1)).rev() {
        let t = tau[ri];
        if t == 0.0 {
            continue;
        }
        // reflector v: v(0) = 1 at submatrix row ri (original row ri+1), v(r) = a(ri+1+r, ri) below
        let mi = mq - ri;
        let mut v = vec![0.0f64; mi];
        v[0] = 1.0;
        for r in 1..mi {
            v[r] = a[at(ri + 1 + r, ri)];
        }
        // dlarf('L', mi, n, v, tau, C = Z(ri+1.., :)): lastv / lastc scans
        let mut lastv = mi;
        while lastv > 0 && v[lastv - 1] == 0.0 {
            lastv -= 1;
        }
        if lastv == 0 {
            continue;
        }
        let row0 = ri + 1;
        let mut lastc = n;
        while lastc > 0 && (0..lastv).all(|r| z[(lastc - 1) * n + row0 + r] == 0.0) {
            lastc -= 1;
        }
        // w = C(1:lastv, 1:lastc)^T v  (dgemv 'T'); C -= tau v w^T (dger)
        let mut w = vec![0.0f64; lastc];
        for j in 0..lastc {
            let mut temp = 0.0;
            for r in 0..lastv {
                temp += z[j * n + row0 + r] * v[r];
            }
            w[j] = temp;
        }
        for j in 0..lastc {
            if w[j] != 0.0 {
                let temp = -t * w[j];
                for r in 0..lastv {
                    let p = j * n + row0 + r;
                    z[p] += v[r] * temp;
                }
            }
        }
    }
    // row-major out: v[i][j] = Z(i, j)
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            v[i * n + j] = z[i + j * n];
        }
    }
    (d, v)
}

/// reference BLAS dnrm2 before LAPACK 3.10 (scale / ssq update)
fn dnrm2_old(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 1 {
        return 0.0;
    }
    if n == 1 {
        return x[0].abs();
    }
    let mut scale = 0.0f64;
    let mut ssq = 1.0f64;
    for &xi in x {
        if xi != 0.0 {
            let absxi = xi.abs();
            if scale < absxi {
                let q = scale / absxi;
                ssq = 1.0 + ssq * (q * q);
                scale = absxi;
            } else {
                let q = absxi / scale;
                ssq += q * q;
            }
        }
    }
    scale * ssq.sqrt()
}

/// dgetrf2 (LAPACK 3.9, recursive) on the column-major m x n block of `a` at (r0, c0), leading dimension lda.
/// ipiv (0-based, relative to r0) has min(m, n) entries. Returns info (0 = non-singular).
fn dgetrf2(a: &mut [f64], lda: usize, r0: usize, c0: usize, m: usize, n: usize, ipiv: &mut [usize]) -> usize {
    let at = |i: usize, j: usize| (r0 + i) + (c0 + j) * lda;
    if m == 0 || n == 0 {
        return 0;
    }
    if m == 1 {
        ipiv[0] = 0;
        return if a[at(0, 0)] == 0.0 { 1 } else { 0 };
    }
    if n == 1 {
        // idamax: first index of the largest |a|
        let mut ip = 0;
        let mut mx = a[at(0, 0)].abs();
        for i in 1..m {
            let v = a[at(i, 0)].abs();
            if v > mx {
                mx = v;
                ip = i;
            }
        }
        ipiv[0] = ip;
        if a[at(ip, 0)] != 0.0 {
            if ip != 0 {
                a.swap(at(0, 0), at(ip, 0));
            }
            if a[at(0, 0)].abs() >= SAFMIN {
                let r = 1.0 / a[at(0, 0)];
                for i in 1..m {
                    a[at(i, 0)] *= r;
                }
            } else {
                for i in 1..m {
                    a[at(i, 0)] /= a[at(0, 0)];
                }
            }
            return 0;
        }
        return 1;
    }
    let mn = m.min(n);
    let n1 = mn / 2;
    let n2 = n - n1;
    let mut info = 0;
    // [A11; A21]
    let i1 = dgetrf2(a, lda, r0, c0, m, n1, &mut ipiv[..n1]);
    if info == 0 && i1 > 0 {
        info = i1;
    }
    // dlaswp on [A12; A22]
    for k in 0..n1 {
        let p = ipiv[k];
        if p != k {
            for j in n1..n {
                a.swap(at(k, j), at(p, j));
            }
        }
    }
    // A12 := inv(L11) A12 (dtrsm left lower no-trans unit)
    for j in n1..n {
        for k in 0..n1 {
            let bkj = a[at(k, j)];
            if bkj != 0.0 {
                for i in (k + 1)..n1 {
                    a[at(i, j)] -= bkj * a[at(i, k)];
                }
            }
        }
    }
    // A22 := A22 - A21 A12 (dgemm, alpha = -1, beta = 1)
    for j in n1..n {
        for l in 0..n1 {
            let temp = -1.0 * a[at(l, j)];
            for i in n1..m {
                a[at(i, j)] += temp * a[at(i, l)];
            }
        }
    }
    // factor A22
    let i2 = dgetrf2(a, lda, r0 + n1, c0 + n1, m - n1, n2, &mut ipiv[n1..mn]);
    if info == 0 && i2 > 0 {
        info = i2 + n1;
    }
    for i in n1..mn {
        ipiv[i] += n1;
    }
    // dlaswp on A21's columns for the second block's pivots
    for k in n1..mn {
        let p = ipiv[k];
        if p != k {
            for j in 0..n1 {
                a.swap(at(k, j), at(p, j));
            }
        }
    }
    info
}

/// numpy.linalg.solve(A, b) for a row-major n x n A and a vector b; None when singular (LinAlgError).
/// dgesv = dgetrf (dgetrf2 for n <= 64) + dgetrs.
pub fn p_solve(a_rm: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i + j * n] = a_rm[i * n + j];
        }
    }
    let mut ipiv = vec![0usize; n];
    if dgetrf2(&mut a, n, 0, 0, n, n, &mut ipiv) != 0 {
        return None;
    }
    let mut x = b[..n].to_vec();
    // dlaswp
    for k in 0..n {
        let p = ipiv[k];
        if p != k {
            x.swap(k, p);
        }
    }
    // dtrsm left lower no-trans unit
    for k in 0..n {
        if x[k] != 0.0 {
            for i in (k + 1)..n {
                x[i] -= x[k] * a[i + k * n];
            }
        }
    }
    // dtrsm left upper no-trans non-unit
    for k in (0..n).rev() {
        if x[k] != 0.0 {
            x[k] /= a[k + k * n];
            for i in 0..k {
                x[i] -= x[k] * a[i + k * n];
            }
        }
    }
    Some(x)
}

/// numpy.linalg.det of a row-major n x n matrix (umath_linalg: dgetrf, then sign * exp(sum log|u_ii|)).
pub fn p_det(a_rm: &[f64], n: usize) -> f64 {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i + j * n] = a_rm[i * n + j];
        }
    }
    let mut ipiv = vec![0usize; n];
    if dgetrf2(&mut a, n, 0, 0, n, n, &mut ipiv) != 0 {
        return 0.0;
    }
    let mut sign = 1.0f64;
    for i in 0..n {
        if ipiv[i] != i {
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

/// Minimum-norm least squares (numpy.linalg.lstsq(A, b, rcond)[0], dgelsd semantics) for a C-contiguous (m x n)
/// A with small n: one-sided (Hestenes) Jacobi SVD of A, singular values <= rcond * s_max treated as zero.
pub fn p_lstsq(a_rows: &[f64], m: usize, n: usize, b: &[f64], rcond: f64) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if m == 0 {
        return vec![0.0; n];
    }
    // columns of A (each length m) and V = I (n x n, column j = right singular vector j)
    let mut u: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a_rows[i * n + j]).collect()).collect();
    let mut v = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        v[j][j] = 1.0;
    }
    let dot = |x: &[f64], y: &[f64]| pw_sum_by(0, x.len(), &|i: usize| x[i] * y[i]);
    for _sweep in 0..60 {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let alpha = dot(&u[p], &u[p]);
                let beta = dot(&u[q], &u[q]);
                let gamma = dot(&u[p], &u[q]);
                if gamma == 0.0 || gamma.abs() <= EPS * (alpha.sqrt() * beta.sqrt()) {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta.abs() > 1e150 {
                    0.5 / zeta
                } else {
                    let t = 1.0 / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                    if zeta < 0.0 { -t } else { t }
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let up = u[p][i];
                    let uq = u[q][i];
                    u[p][i] = c * up - s * uq;
                    u[q][i] = s * up + c * uq;
                }
                for i in 0..n {
                    let vp = v[i][p];
                    let vq = v[i][q];
                    v[i][p] = c * vp - s * vq;
                    v[i][q] = s * vp + c * vq;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let sv: Vec<f64> = (0..n).map(|j| dot(&u[j], &u[j]).sqrt()).collect();
    let smax = sv.iter().cloned().fold(0.0f64, f64::max);
    let rc = if rcond > 0.0 { rcond } else { EPS };
    let tol = rc * smax;
    let mut x = vec![0.0f64; n];
    for j in 0..n {
        if sv[j] <= tol || sv[j] == 0.0 {
            continue;
        }
        // coefficient along right singular vector j: (u_j . b) / s_j^2 (u_j unnormalised = s_j * U_j)
        let ub = dot(&u[j], &b[..m]);
        let coef = (ub / sv[j]) / sv[j];
        for i in 0..n {
            x[i] += coef * v[i][j];
        }
    }
    x
}

/// numpy.linalg.lstsq(A, r, rcond=None)[0] for a C-contiguous (m x 3) A.
pub fn lstsq3(a_rows: &[[f64; 3]], r: &[f64]) -> [f64; 3] {
    let m = a_rows.len();
    let flat: Vec<f64> = a_rows.iter().flat_map(|x| x.iter().cloned()).collect();
    let rcond = f64::EPSILON * (m.max(3) as f64);
    let x = lstsq(&flat, m, 3, r, rcond);
    [x[0], x[1], x[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigh_reconstructs() {
        let m = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 1.0];
        let (w, v) = eigh(&m, 3);
        assert!(w[0] <= w[1] && w[1] <= w[2]);
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += v[i * 3 + k] * w[k] * v[j * 3 + k];
                }
                assert!((s - m[i * 3 + j]).abs() < 1e-12);
            }
        }
        let (w2, _) = eigh(&[2.0, 1.0, 1.0, 2.0], 2);
        assert!((w2[0] - 1.0).abs() < 1e-15 && (w2[1] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn solve_lstsq_det() {
        let a = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0];
        let x = solve(&a, &[1.0, 2.0, 3.0], 3).unwrap();
        for i in 0..3 {
            let s: f64 = (0..3).map(|j| a[i * 3 + j] * x[j]).sum();
            assert!((s - [1.0, 2.0, 3.0][i]).abs() < 1e-14);
        }
        assert!((det(&a, 3) - 18.0).abs() < 1e-12);
        let rows: Vec<f64> = (0..20).flat_map(|i| vec![1.0, i as f64]).collect();
        let y: Vec<f64> = (0..20).map(|i| 2.0 + 0.5 * i as f64).collect();
        let c = lstsq(&rows, 20, 2, &y, 1e-15);
        assert!((c[0] - 2.0).abs() < 1e-12 && (c[1] - 0.5).abs() < 1e-12);
        // rank deficient: minimum norm
        let rows: Vec<f64> = (0..10).flat_map(|_| vec![1.0, 1.0]).collect();
        let c = lstsq(&rows, 10, 2, &vec![2.0; 10], 1e-14);
        assert!((c[0] - 1.0).abs() < 1e-12 && (c[1] - 1.0).abs() < 1e-12);
    }
}

// ------------------------------------------------------------------ probe dispatch (research copy)
use crate::accel_host as h;
use crate::probe;

fn flat2(a: &(Vec<f64>, Vec<f64>)) -> Vec<f64> {
    a.0.iter().chain(a.1.iter()).cloned().collect()
}

pub fn xtx(x: &[f64], rows: usize, k: usize) -> Vec<f64> {
    let use_h = probe::accel(0);
    if probe::cfg().count {
        let (a, b) = (p_xtx(x, rows, k), h::xtx(x, rows, k));
        probe::op_hit(0, &a, &b);
        return if use_h { b } else { a };
    }
    if use_h { h::xtx(x, rows, k) } else { p_xtx(x, rows, k) }
}
pub fn gemm_nt(a: &[f64], m: usize, b: &[f64], p: usize, k: usize) -> Vec<f64> {
    let use_h = probe::accel(1);
    if probe::cfg().count {
        let (x, y) = (p_gemm_nt(a, m, b, p, k), h::gemm_nt(a, m, b, p, k));
        probe::op_hit(1, &x, &y);
        return if use_h { y } else { x };
    }
    if use_h { h::gemm_nt(a, m, b, p, k) } else { p_gemm_nt(a, m, b, p, k) }
}
pub fn eigh(m: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let use_h = probe::accel(2);
    if probe::cfg().count {
        let (x, y) = (p_eigh(m, n), h::eigh(m, n));
        probe::op_hit(2, &flat2(&x), &flat2(&y));
        return if use_h { y } else { x };
    }
    if use_h { h::eigh(m, n) } else { p_eigh(m, n) }
}
pub fn solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let use_h = probe::accel(3);
    if probe::cfg().count {
        let (x, y) = (p_solve(a, b, n), h::solve(a, b, n));
        probe::op_hit(3, &x.clone().unwrap_or_default(), &y.clone().unwrap_or_default());
        return if use_h { y } else { x };
    }
    if use_h { h::solve(a, b, n) } else { p_solve(a, b, n) }
}
pub fn det(a: &[f64], n: usize) -> f64 {
    let use_h = probe::accel(4);
    if probe::cfg().count {
        let (x, y) = (p_det(a, n), h::det(a, n));
        probe::op_hit(4, &[x], &[y]);
        return if use_h { y } else { x };
    }
    if use_h { h::det(a, n) } else { p_det(a, n) }
}
pub fn lstsq(a: &[f64], m: usize, n: usize, b: &[f64], rcond: f64) -> Vec<f64> {
    let use_h = probe::accel(5);
    if probe::cfg().count {
        let (x, y) = (p_lstsq(a, m, n, b, rcond), h::lstsq(a, m, n, b, rcond));
        probe::op_hit(5, &x, &y);
        return if use_h { y } else { x };
    }
    if use_h { h::lstsq(a, m, n, b, rcond) } else { p_lstsq(a, m, n, b, rcond) }
}
