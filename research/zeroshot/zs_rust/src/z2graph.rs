//! Port of z2_self_recipe/graph.py (candidate pairs, measure_edge, _station_refine, build_graph, nms,
//! suppress_facets, break_triangles), of the E3 metrology helpers it calls (domains/production/engine/inspect:
//! metrology._intersect / _flat_start / _robust_line / _sagitta / _reseat_plane / _station_layout / _fit_side /
//! _skin_level / _station_sharp, geom.skin_peaks, InspectConfig defaults) and of run.run_scan's line assembly with
//! the D1 snap. Risers (GraphParams.risers) are off in the lane and not ported.
#[allow(unused_imports)]
use crate::fmath::{DetF32, DetF64};
use crate::accel;
use crate::ckdtree::CKDTree;
use crate::d1;
use crate::kdtree::KdTree;
use crate::num::*;
use crate::z2seg::{dotn, linspace_idx, robust_plane, Flange, Plane, Region};
use crate::z3prep::convolve_same;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub struct GraphParams {
    pub min_sweep_deg: f64,
    pub max_sweep_deg: f64,
    pub near_mm: f64,
    pub slice_mm: f64,
    pub gap_bridge_mm: f64,
    pub arc_tol_deg: f64,
    pub arc_margin_deg: f64,
    pub corner_mm: f64,
    pub contact_mm: f64,
    pub min_arc_slices: i64,
    pub min_span_mm: f64,
    pub d1_snap: bool,
    pub min_cover: f64,
    pub local_depth_mm: f64,
    pub nms_offset_mm: f64,
    pub max_ro_t: f64,
    pub nms_deg: f64,
}

impl Default for GraphParams {
    fn default() -> Self {
        GraphParams {
            min_sweep_deg: 8.0, max_sweep_deg: 175.0, near_mm: 6.0, slice_mm: 2.0, gap_bridge_mm: 12.0,
            arc_tol_deg: 20.0, arc_margin_deg: 4.0, corner_mm: 6.0, contact_mm: 2.5, min_arc_slices: 3,
            min_span_mm: 8.0, d1_snap: true, min_cover: 0.25, local_depth_mm: 30.0, nms_offset_mm: 3.0,
            max_ro_t: 12.0, nms_deg: 5.0,
        }
    }
}

// InspectConfig defaults used here
const STATIONS_MIN: usize = 5;
const STATIONS_MAX: usize = 11;
const STATION_TARGET_MM: f64 = 12.0;
const MIN_STATION_POINTS: usize = 12;
const STATION_SKIN_REACH_MM: f64 = 0.6;
const SHARP_LINE_CONSISTENCY_MM: f64 = 0.15;
const MIN_STATION_FRACTION: f64 = 0.6;
const LOCAL_RESEAT_MIN_MM: f64 = 0.3;
const LOCAL_RESEAT_MAX_MM: f64 = 3.0;
const LOCAL_RESEAT_MAX_TILT_DEG: f64 = 6.0;
const LOCAL_RESEAT_MIN_SHARE: f64 = 0.25;
const FLAT_START_STEP_MM: f64 = 0.25;
const FLAT_START_TEST_MM: f64 = 0.5;
const FLAT_START_GAP_MM: f64 = 1.0;
const FLAT_START_REF_MM: f64 = 3.0;
const FLAT_START_DEV_MM: f64 = 0.02;
const FLAT_START_MAX_SHIFT_MM: f64 = 4.0;
const FLAT_START_CHUNK_MM: f64 = 25.0;

pub struct Scan<'a> {
    pub p: &'a [V3],
    pub ns: &'a [V3],
    pub nbr: &'a [u32],
    pub k: usize,
    pub t: f64,
    pub pa: f64,
    pub regions: &'a [Region],
    pub flanges: &'a [Flange],
    pub fop: &'a [i64],
}

fn unit(v: &V3) -> V3 {
    let n = norm3(v).max(1e-12);
    [v[0] / n, v[1] / n, v[2] / n]
}

/// metrology._intersect(a, b) -> (point, unit direction)
pub fn intersect(a: &Plane, b: &Plane) -> Option<(V3, V3)> {
    let d = cross3(&a.n, &b.n);
    let s = norm3(&d);
    if s < 1e-6 {
        return None;
    }
    let d = [d[0] / s, d[1] / s, d[2] / s];
    let m = [a.n[0], a.n[1], a.n[2], b.n[0], b.n[1], b.n[2], d[0], d[1], d[2]];
    let p = accel::solve(&m, &[a.d, b.d, 0.0], 3)?;
    Some(([p[0], p[1], p[2]], d))
}

fn bincount_ge(bins: &[usize], mask: &[bool], n: usize, k: usize) -> Vec<bool> {
    let mut c = vec![0usize; n];
    for (b, &m) in bins.iter().zip(mask) {
        if m {
            c[*b] += 1;
        }
    }
    c.iter().map(|&x| x >= k).collect()
}

fn thin_idx(n: usize, cap: usize) -> Vec<usize> {
    if n <= cap {
        (0..n).collect()
    } else {
        linspace_idx(n, cap)
    }
}

fn quantile_of(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(total_cmp);
    quantiles_sorted(&s, q)
}

/// points of `p` that have a point of `q` strictly within r (cKDTree query k=1, distance_upper_bound=r)
fn within(tree: &CKDTree, p: &[V3], r: f64) -> Vec<bool> {
    p.par_iter().map(|x| tree.query(x, 1, r)[0].0.is_finite()).collect()
}

// ------------------------------------------------------------------ geom / metrology helpers

/// geom.skin_peaks(h, t) -> (top, bottom)
fn skin_peaks(h: &[f64], t: f64) -> (f64, Option<f64>) {
    let hb = 0.05;
    let lo = h.iter().cloned().fold(f64::INFINITY, f64::min) - 1.0;
    let idx: Vec<i64> = h.iter().map(|&x| ((x - lo) / hb).floor() as i64).collect();
    let mx = *idx.iter().max().unwrap_or(&0);
    let mut hist = vec![0.0f64; (mx + 1).max(1) as usize];
    for &i in &idx {
        hist[i as usize] += 1.0;
    }
    let sm = convolve_same(&hist, &[1.0; 5]);
    let p1 = argmax(&sm);
    let h1 = lo + (p1 as f64 + 0.5) * hb;
    let c1 = sm[p1];
    let mut best: Option<(f64, f64)> = None;
    for sign in [1.0f64, -1.0] {
        let h2 = h1 + sign * t;
        let j = ((h2 - lo) / hb).floor() as i64;
        if j < 0 || j >= sm.len() as i64 {
            continue;
        }
        let a = (j - 3).max(0) as usize;
        let b = ((j + 4) as usize).min(sm.len());
        let win = &sm[a..b];
        let c2 = win.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if c2 >= 0.15 * c1 && (best.is_none() || c2 > best.unwrap().1) {
            let jj = a + argmax(win);
            best = Some((lo + (jj as f64 + 0.5) * hb, c2));
        }
    }
    match best {
        None => (h1, None),
        Some((b0, _)) => {
            if b0 < h1 {
                (h1, Some(b0))
            } else {
                (b0, Some(h1))
            }
        }
    }
}

fn station_layout(zlo: f64, zhi: f64) -> (Vec<f64>, f64) {
    let length = zhi - zlo;
    let n = round_even(length / STATION_TARGET_MM) as i64;
    let n = (STATIONS_MIN as i64).max((STATIONS_MAX as i64).min(n)) as usize;
    let w = length / n as f64;
    ((0..n).map(|i| zlo + w * (i as f64 + 0.5)).collect(), w)
}

fn skin_level(d: &[f64], target: f64, reach: f64, tol: f64, min_points: usize) -> Option<f64> {
    let win: Vec<f64> = d.iter().cloned().filter(|x| (x - target).abs() < reach).collect();
    if win.len() < min_points {
        return None;
    }
    let hb = 0.05;
    let lo = target - reach;
    let idx: Vec<i64> = win.iter().map(|&x| (((x - lo) / hb).floor() as i64).max(0)).collect();
    let mx = *idx.iter().max().unwrap();
    let mut hist = vec![0.0f64; (mx + 1) as usize];
    for &i in &idx {
        hist[i as usize] += 1.0;
    }
    let sm = convolve_same(&hist, &[1.0; 3]);
    let peak = lo + (argmax(&sm) as f64 + 0.5) * hb;
    let at_peak = win.iter().filter(|x| (*x - peak).abs() < tol).count();
    let at_target = win.iter().filter(|x| (*x - target).abs() < tol).count();
    if (at_target as f64) >= (min_points as f64).max(0.5 * at_peak as f64) {
        return Some(target);
    }
    if at_peak < min_points {
        return None;
    }
    let sel: Vec<f64> = win.iter().cloned().filter(|x| (x - peak).abs() < tol).collect();
    Some(median(&sel))
}

/// metrology._fit_side(pts, plane, t)[0]
fn fit_side(pts: &[V3], plane: &Plane, t: f64) -> Option<Plane> {
    let d = plane.distances(pts);
    let tol = 0.15;
    let reach = STATION_SKIN_REACH_MM.min(0.4 * t);
    for (target, inner) in [(0.0f64, false), (-t, true)] {
        let level = match skin_level(&d, target, reach, tol, MIN_STATION_POINTS) {
            Some(l) => l,
            None => continue,
        };
        let sel: Vec<V3> = pts.iter().zip(&d).filter(|(_, &x)| (x - level).abs() < tol).map(|(p, _)| *p).collect();
        if sel.len() < MIN_STATION_POINTS {
            continue;
        }
        let mut pl = robust_plane(&sel, 3)?;
        if dotn(&pl.n, pl.strided, &plane.n, plane.strided) < 0.0 {
            pl = pl.flipped();
        }
        return Some(if inner { pl.offset(t) } else { pl });
    }
    None
}

fn station_sharp(p1: &Plane, p2: &Plane, lp: &V3, a: &V3, zi: f64) -> Option<V3> {
    let (p0, d0) = intersect(p1, p2)?;
    let da = dot3(&d0, a);
    if da.abs() < 0.5 {
        return None;
    }
    let f = (zi - dot3(&sub3(&p0, lp), a)) / da;
    Some([p0[0] + f * d0[0], p0[1] + f * d0[1], p0[2] + f * d0[2]])
}

/// metrology._reseat_plane(pts, plane, t)
fn reseat_plane(pts: &[V3], plane: &Plane, t: f64) -> Option<Plane> {
    let need = MIN_STATION_POINTS * STATIONS_MIN;
    if pts.len() < need {
        return None;
    }
    let d = plane.distances(pts);
    let w = LOCAL_RESEAT_MAX_MM;
    let win: Vec<bool> = d.iter().map(|&x| x > -t - w && x < w).collect();
    if win.iter().filter(|&&x| x).count() < need {
        return None;
    }
    let dw: Vec<f64> = d.iter().zip(&win).filter(|(_, &b)| b).map(|(x, _)| *x).collect();
    let (top, bottom) = skin_peaks(&dw, t);
    let oml = if bottom.is_some() || top.abs() <= (top + t).abs() { top } else { top + t };
    if oml.abs() <= LOCAL_RESEAT_MIN_MM {
        return None;
    }
    let tol = 0.25f64.min(0.3 * t);
    let inner = d.iter().zip(&win).filter(|(x, &b)| b && (**x - oml).abs() < tol).count() < need;
    let mut pl = plane.offset(if inner { oml - t } else { oml });
    for _ in 0..3 {
        let dd = pl.distances(pts);
        let sel: Vec<V3> = pts.iter().zip(&dd).filter(|(_, &x)| x.abs() < tol).map(|(p, _)| *p).collect();
        if sel.len() < need {
            return None;
        }
        let q = robust_plane(&sel, 3)?;
        pl = if dotn(&q.n, q.strided, &plane.n, plane.strided) > 0.0 { q } else { q.flipped() };
    }
    let dd = pl.distances(pts);
    let cnt = dd.iter().filter(|x| x.abs() < tol).count();
    if (cnt as f64) < LOCAL_RESEAT_MIN_SHARE * pts.len() as f64 {
        return None;
    }
    let c = dotn(&pl.n, pl.strided, &plane.n, plane.strided);
    if (-1.0f64).max(1.0f64.min(c)).dacos().to_degrees() > LOCAL_RESEAT_MAX_TILT_DEG {
        return None;
    }
    Some(if inner { pl.offset(t) } else { pl })
}

/// metrology._robust_line(x, y) -> (a, b, kept)
fn robust_line(x: &[f64], y: &[f64]) -> Option<(f64, f64, Vec<bool>)> {
    let n = x.len();
    let mut keep = vec![true; n];
    let mut sol = [0.0f64; 2];
    for _ in 0..3 {
        let kc = keep.iter().filter(|&&k| k).count();
        if kc < 10 {
            return None;
        }
        let mut a = Vec::with_capacity(2 * kc);
        let mut b = Vec::with_capacity(kc);
        for i in 0..n {
            if keep[i] {
                a.push(1.0);
                a.push(x[i]);
                b.push(y[i]);
            }
        }
        let s = accel::lstsq(&a, kc, 2, &b, f64::EPSILON * (kc.max(2) as f64));
        sol = [s[0], s[1]];
        let r: Vec<f64> = (0..n).map(|i| y[i] - (sol[0] + sol[1] * x[i])).collect();
        let rk: Vec<f64> = (0..n).filter(|&i| keep[i]).map(|i| r[i]).collect();
        let md = median(&rk);
        let dev: Vec<f64> = rk.iter().map(|v| (v - md).abs()).collect();
        let mad = 1.4826 * median(&dev);
        let lim = (3.0 * mad).max(0.01);
        keep = r.iter().map(|v| v.abs() < lim).collect();
    }
    Some((sol[0], sol[1], keep))
}

/// metrology._sagitta(x, y): c L^2 / 8 of np.polyfit(u, y, 2)
fn sagitta(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let xmin = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let xmax = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let u: Vec<f64> = x.iter().map(|v| v - 0.5 * (xmin + xmax)).collect();
    // polyfit: lhs = vander(u, 3) = [u^2, u, 1]; scale = sqrt((lhs*lhs).sum(axis=0)); lhs /= scale
    let mut sc = [0.0f64; 3];
    for &ui in &u {
        let row = [ui * ui, ui, 1.0];
        for j in 0..3 {
            sc[j] += row[j] * row[j];
        }
    }
    let sc = [sc[0].sqrt(), sc[1].sqrt(), sc[2].sqrt()];
    let mut a = Vec::with_capacity(3 * n);
    for &ui in &u {
        let row = [ui * ui, ui, 1.0];
        for j in 0..3 {
            a.push(row[j] / sc[j]);
        }
    }
    let c = accel::lstsq(&a, n, 3, y, (n as f64) * f64::EPSILON);
    let c0 = c[0] / sc[0];
    let ptp = xmax - xmin;
    c0 * (ptp * ptp) / 8.0
}

/// metrology._flat_start(s, d, z, start0, depth, t, zlo, zhi)
fn flat_start(s: &[f64], d: &[f64], z: &[f64], start0: f64, depth: f64, t: f64, zlo: f64, zhi: f64) -> f64 {
    let (g, gap, step) = (FLAT_START_TEST_MM, FLAT_START_GAP_MM, FLAT_START_STEP_MM);
    let ref_min = 0.5 * FLAT_START_REF_MM;
    let limit = (start0 + FLAT_START_MAX_SHIFT_MM).min(depth - g - gap - ref_min);
    if FLAT_START_MAX_SHIFT_MM <= 0.0 || limit < start0 + step {
        return start0;
    }
    let band = 0.45 * t;
    let mut ss = Vec::new();
    let mut dd = Vec::new();
    let mut zz = Vec::new();
    for i in 0..s.len() {
        if s[i] >= start0 && s[i] <= depth && z[i] >= zlo && z[i] <= zhi {
            ss.push(s[i]);
            dd.push(d[i]);
            zz.push(z[i]);
        }
    }
    let need = MIN_STATION_POINTS * STATIONS_MIN;
    let m0: Vec<bool> = dd.iter().map(|x| x.abs() < band).collect();
    let m1: Vec<bool> = dd.iter().map(|x| (x + t).abs() < band).collect();
    let c0 = m0.iter().filter(|&&b| b).count();
    let c1 = m1.iter().filter(|&&b| b).count();
    let mut skins: Vec<(&Vec<bool>, f64, usize)> = Vec::new();
    if c0 >= need {
        skins.push((&m0, 0.0, c0));
    }
    if c1 >= need {
        skins.push((&m1, -t, c1));
    }
    if skins.is_empty() {
        return start0;
    }
    // max(skins, key=count): first maximal
    let mut bi = 0;
    for i in 1..skins.len() {
        if skins[i].2 > skins[bi].2 {
            bi = i;
        }
    }
    let (keep, level, _) = skins[bi];
    let mut s2 = Vec::new();
    let mut d2 = Vec::new();
    let mut z2 = Vec::new();
    for i in 0..ss.len() {
        if keep[i] {
            s2.push(ss[i]);
            d2.push(dd[i] - level);
            z2.push(zz[i]);
        }
    }
    let n_chunks = 1i64.max(round_even((zhi - zlo) / FLAT_START_CHUNK_MM) as i64) as usize;
    let chunk: Vec<usize> = z2
        .iter()
        .map(|&zv| ((((zv - zlo) / (zhi - zlo).max(1e-9) * n_chunks as f64) as i64).max(0) as usize).min(n_chunks - 1))
        .collect();
    let groups: Vec<Vec<usize>> = (0..n_chunks).map(|k| (0..chunk.len()).filter(|&i| chunk[i] == k).collect()).collect();
    let mut s0 = start0;
    while s0 <= limit + 1e-9 {
        let r0 = s0 + g + gap;
        let ref_len = FLAT_START_REF_MM.min(depth - r0);
        let mut res: Vec<f64> = Vec::new();
        let mut sags: Vec<f64> = Vec::new();
        let mut expected = 0.0f64;
        for gi in &groups {
            let sg: Vec<f64> = gi.iter().map(|&i| s2[i]).collect();
            let dg: Vec<f64> = gi.iter().map(|&i| d2[i]).collect();
            let refm: Vec<bool> = sg.iter().map(|&v| v >= r0 && v < r0 + ref_len).collect();
            let nref = refm.iter().filter(|&&b| b).count();
            if nref < 20 {
                continue;
            }
            let sr: Vec<f64> = sg.iter().zip(&refm).filter(|(_, &b)| b).map(|(v, _)| *v).collect();
            let dr: Vec<f64> = dg.iter().zip(&refm).filter(|(_, &b)| b).map(|(v, _)| *v).collect();
            let ptp = sr.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - sr.iter().cloned().fold(f64::INFINITY, f64::min);
            if ptp < 0.6 * ref_len {
                continue;
            }
            let line = match robust_line(&sr, &dr) {
                Some(l) => l,
                None => continue,
            };
            let kept: Vec<bool> = if line.2.iter().filter(|&&b| b).count() >= 10 { line.2.clone() } else { vec![true; nref] };
            let sk: Vec<f64> = sr.iter().zip(&kept).filter(|(_, &b)| b).map(|(v, _)| *v).collect();
            let dk: Vec<f64> = dr.iter().zip(&kept).filter(|(_, &b)| b).map(|(v, _)| *v).collect();
            sags.push(sagitta(&sk, &dk));
            for i in 0..sg.len() {
                if sg[i] >= s0 && sg[i] < s0 + g {
                    let r = dg[i] - (line.0 + line.1 * sg[i]);
                    if r > -band && r < 0.15 {
                        res.push(r);
                    }
                }
            }
            expected += (nref as f64) * g / ref_len;
        }
        if expected < 20.0 {
            break;
        }
        let judged = median(&sags).abs() <= FLAT_START_DEV_MM && (res.len() as f64) >= 0.5 * expected;
        if judged && median(&res) >= -FLAT_START_DEV_MM {
            return s0;
        }
        s0 += step;
    }
    start0
}

// ------------------------------------------------------------------ graph

#[derive(Clone, Debug)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
    pub ok: bool,
    pub reason: String,
    pub sweep_deg: f64,
    pub line_start: Option<V3>,
    pub line_end: Option<V3>,
    pub span_mm: f64,
    pub cover: f64,
    pub score: f64,
    pub geom: Option<Geom>,
}

#[derive(Clone, Debug)]
pub struct Geom {
    pub lp: V3,
    pub u: V3,
    pub na: (V3, bool),
    pub nb: (V3, bool),
    pub ro: f64,
    pub sweep: f64,
    pub dens: f64,
}

/// candidate_pairs(scan, prm): {(a, b): count} in sorted key order
fn candidate_pairs(sc: &Scan, prm: &GraphParams) -> BTreeMap<(usize, usize), usize> {
    let mut q: Vec<V3> = Vec::new();
    let mut lab: Vec<usize> = Vec::new();
    for f in sc.flanges {
        let mut bd: Vec<usize> = f
            .idx
            .iter()
            .cloned()
            .filter(|&i| (0..sc.k).any(|j| sc.fop[sc.nbr[i * sc.k + j] as usize] != f.fid as i64))
            .collect();
        if !bd.is_empty() {
            let mut seen = std::collections::HashSet::new();
            bd.retain(|&i| {
                let p = sc.p[i];
                let key = [p[0].floor() as i64, p[1].floor() as i64, p[2].floor() as i64];
                seen.insert(key)
            });
        }
        if bd.len() > 4000 {
            bd = linspace_idx(bd.len(), 4000).iter().map(|&k| bd[k]).collect();
        }
        for &i in &bd {
            q.push(sc.p[i]);
            lab.push(f.fid);
        }
    }
    let mut out = BTreeMap::new();
    if q.is_empty() {
        return out;
    }
    let tree = KdTree::new(&q);
    let r = prm.near_mm + 3.0 * sc.t;
    let lists: Vec<Vec<usize>> = q.par_iter().map(|x| tree.ball(x, r)).collect();
    for (i, lst) in lists.iter().enumerate() {
        for &j in lst {
            if j <= i {
                continue;
            }
            let (la, lb) = (lab[i], lab[j]);
            if la != lb {
                *out.entry((la.min(lb), la.max(lb))).or_insert(0) += 1;
            }
        }
    }
    out
}

fn runs(occ: &[bool], gap: i64) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    let mut last = 0usize;
    for (i, &o) in occ.iter().enumerate() {
        if !o {
            continue;
        }
        match start {
            None => {
                start = Some(i);
                last = i;
            }
            Some(s) => {
                if (i as i64) - (last as i64) - 1 <= gap {
                    last = i;
                } else {
                    out.push((s, last));
                    start = Some(i);
                    last = i;
                }
            }
        }
    }
    if let Some(s) = start {
        out.push((s, last));
    }
    out
}

/// outer_plane(scan, f, other_pts) -> (plane, seen_outer)
fn outer_plane(sc: &Scan, f: &Flange, other: &[V3]) -> (Plane, bool) {
    let mut best: Option<((bool, f64), Plane, bool)> = None;
    for &rid in &f.regions {
        let pl = &sc.regions[rid].plane;
        let s = median(&pl.distances(other));
        let (cand, seen) = if s < 0.0 {
            (pl.clone(), true)
        } else {
            (Plane { n: neg3(&pl.n), strided: false, d: -pl.d + sc.t, rms: pl.rms, count: pl.count, c: None }, false)
        };
        let area = sc.regions[rid].area;
        let key = (seen, area);
        let better = match &best {
            None => true,
            Some((bk, _, _)) => (key.0 as u8, key.1) > (bk.0 as u8, bk.1),
        };
        if better {
            best = Some((key, cand, seen));
        }
    }
    let b = best.unwrap();
    (b.1, b.2)
}

fn pts_of(p: &[V3], idx: &[usize]) -> Vec<V3> {
    idx.iter().map(|&i| p[i]).collect()
}

fn measure_edge(sc: &Scan, fa: &Flange, fb: &Flange, prm: &GraphParams, tree: &KdTree) -> Edge {
    let (p, ns, t) = (sc.p, sc.ns, sc.t);
    let mut e = Edge { a: fa.fid, b: fb.fid, ok: false, reason: String::new(), sweep_deg: 0.0, line_start: None,
                       line_end: None, span_mm: 0.0, cover: 0.0, score: 0.0, geom: None };
    let pa = pts_of(p, &fa.idx);
    let pb = pts_of(p, &fb.idx);
    let near = prm.near_mm + 3.0 * t;
    // _local(p, q, r): points of thin(p, 20000) within r of thin(q)
    let local = |pp: &[V3], qq: &[V3], r: f64| -> Vec<V3> {
        let tq: Vec<V3> = thin_idx(qq.len(), 3000).iter().map(|&i| qq[i]).collect();
        let tp: Vec<V3> = thin_idx(pp.len(), 20000).iter().map(|&i| pp[i]).collect();
        let tr = CKDTree::new(&tq);
        let w = within(&tr, &tp, r);
        let sel: Vec<V3> = tp.iter().zip(&w).filter(|(_, &b)| b).map(|(x, _)| *x).collect();
        if sel.len() >= 20 { sel } else { pp.to_vec() }
    };
    let pa_loc = local(&pa, &pb, near);
    let pb_loc = local(&pb, &pa, near);
    let mid_local = |f: &Flange, other: &Flange| -> Vec<V3> {
        let idx: Vec<usize> = thin_idx(f.idx.len(), 20000).iter().map(|&k| f.idx[k]).collect();
        let oq: Vec<V3> = thin_idx(other.idx.len(), 3000).iter().map(|&k| p[other.idx[k]]).collect();
        let tr = CKDTree::new(&oq);
        let qp = pts_of(p, &idx);
        let w = within(&tr, &qp, 3.0 * near);
        let mut sel: Vec<usize> = idx.iter().zip(&w).filter(|(_, &b)| b).map(|(i, _)| *i).collect();
        if sel.len() < 20 {
            sel = idx.clone();
        }
        let h = 0.5 * t;
        sel.iter().map(|&i| [p[i][0] - h * ns[i][0], p[i][1] - h * ns[i][1], p[i][2] - h * ns[i][2]]).collect()
    };
    let ma = mid_local(fa, fb);
    let mb = mid_local(fb, fa);
    let (pla, _ka) = outer_plane(sc, fa, &mb);
    let (plb, _kb) = outer_plane(sc, fb, &ma);
    let (lp, u) = match intersect(&pla, &plb) {
        Some(x) => x,
        None => {
            e.reason = "PARALLEL".into();
            return e;
        }
    };
    for (f, pts, other) in [(fa, &pa, &plb), (fb, &pb, &pla)] {
        if f.two_sided {
            continue;
        }
        if dotn(&f.normal, false, &other.n, other.strided).abs() > 20.0f64.to_radians().dsin() {
            continue;
        }
        let tp: Vec<V3> = thin_idx(pts.len(), 20000).iter().map(|&i| pts[i]).collect();
        let h: Vec<f64> = if tp.len() == 1 {
            vec![dotn(&tp[0], false, &other.n, other.strided)]
        } else {
            tp.iter().map(|x| dot3(x, &other.n)).collect()
        };
        if quantile_of(&h, 0.98) - quantile_of(&h, 0.02) < 1.3 * t + 0.5 {
            e.reason = "CUT_WALL".into();
            return e;
        }
    }
    let cs = clip(dotn(&pla.n, pla.strided, &plb.n, plb.strided), -1.0, 1.0);
    let sweep = cs.dacos().to_degrees();
    e.sweep_deg = sweep;
    if sweep < prm.min_sweep_deg || sweep > prm.max_sweep_deg {
        e.reason = "SWEEP_OUT_OF_RANGE".into();
        return e;
    }
    let local_centre = |pp: &[V3], q_loc: &[V3], r: f64| -> V3 {
        let tq: Vec<V3> = thin_idx(q_loc.len(), 3000).iter().map(|&i| q_loc[i]).collect();
        let tp: Vec<V3> = thin_idx(pp.len(), 20000).iter().map(|&i| pp[i]).collect();
        let tr = CKDTree::new(&tq);
        let w = within(&tr, &tp, r);
        let sel: Vec<V3> = tp.iter().zip(&w).filter(|(_, &b)| b).map(|(x, _)| *x).collect();
        let src = if sel.len() >= 20 { sel } else { pp.to_vec() };
        let col = |k: usize| median(&src.iter().map(|x| x[k]).collect::<Vec<_>>());
        [col(0), col(1), col(2)]
    };
    let ca = local_centre(&pa, &pb_loc, 3.0 * near);
    let cb = local_centre(&pb, &pa_loc, 3.0 * near);
    let mut wa = unit(&cross3(&u, &pla.n));
    if dot3(&sub3(&ca, &lp), &wa) < 0.0 {
        wa = neg3(&wa);
    }
    let mut wb = unit(&cross3(&u, &plb.n));
    if dot3(&sub3(&cb, &lp), &wb) < 0.0 {
        wb = neg3(&wb);
    }
    let rr_max = prm.corner_mm + 3.0 * t;
    let za: Vec<f64> = pa.iter().map(|x| dot3(&sub3(x, &lp), &u)).collect();
    let zb: Vec<f64> = pb.iter().map(|x| dot3(&sub3(x, &lp), &u)).collect();
    let mn = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let zlo = mn(&za).max(mn(&zb)) - 1.0;
    let zhi = mx(&za).min(mx(&zb)) + 1.0;
    if zhi - zlo < prm.min_span_mm {
        e.reason = "NO_COMMON_EXTENT".into();
        return e;
    }
    let nseg = 2i64.max(((zhi - zlo) / rr_max.max(1.0)).ceil() as i64);
    let mut cand: BTreeSet<usize> = BTreeSet::new();
    for k in 0..=nseg {
        let zz = zlo + (zhi - zlo) * (k as f64) / (nseg as f64);
        let c = [lp[0] + u[0] * zz, lp[1] + u[1] * zz, lp[2] + u[2] * zz];
        let r = rr_max.dhypot(0.5 * (zhi - zlo) / nseg as f64) + 0.5;
        for i in tree.ball(&c, r) {
            cand.insert(i);
        }
    }
    if cand.is_empty() {
        e.reason = "NO_POINTS".into();
        return e;
    }
    struct Pt {
        rel: V3,
        z: f64,
        rr: f64,
        fl: i64,
        nn: V3,
    }
    let mut pts: Vec<Pt> = Vec::new();
    for &i in &cand {
        let rel = sub3(&p[i], &lp);
        let z = dot3(&rel, &u);
        let cr = [rel[0] - z * u[0], rel[1] - z * u[1], rel[2] - z * u[2]];
        let rr = norm3(&cr);
        if z >= zlo && z <= zhi && rr < rr_max {
            pts.push(Pt { rel, z, rr, fl: sc.fop[i], nn: ns[i] });
        }
    }
    let st = prm.arc_tol_deg.to_radians().dsin();
    let e1 = (pla.n, pla.strided);
    let mut e2 = unit(&cross3(&u, &e1.0));
    if dotn(&plb.n, plb.strided, &e2, false) < 0.0 {
        e2 = neg3(&e2);
    }
    let ang_b = dotn(&plb.n, plb.strided, &e2, false).datan2(dotn(&plb.n, plb.strided, &e1.0, e1.1)).to_degrees();
    let mg = prm.arc_margin_deg;
    let q4 = 0.25 * ang_b;
    let n_pts = pts.len();
    let mut perp = vec![false; n_pts];
    let mut a_pt = vec![0.0; n_pts];
    let mut out_arc = vec![false; n_pts];
    let mut corner = vec![false; n_pts];
    let mut other = vec![false; n_pts];
    let mut mid = vec![false; n_pts];
    let mut arc = vec![false; n_pts];
    let mut near_a = vec![false; n_pts];
    let mut near_b = vec![false; n_pts];
    for (k, q) in pts.iter().enumerate() {
        perp[k] = dot3(&q.nn, &u).abs() < st;
        let n2 = dot3(&q.nn, &e2);
        let n1 = dot3(&q.nn, &e1.0);
        a_pt[k] = n2.datan2(n1).to_degrees();
        let a_in = (-n2).datan2(-n1).to_degrees();
        out_arc[k] = perp[k] && a_pt[k] > mg && a_pt[k] < ang_b - mg;
        let sa = dot3(&q.rel, &wa);
        let sb = dot3(&q.rel, &wb);
        corner[k] = sa < rr_max && sb < rr_max;
        other[k] = q.fl >= 0 && q.fl != fa.fid as i64 && q.fl != fb.fid as i64;
        mid[k] = perp[k] && ((a_pt[k] - 0.5 * ang_b).abs() < q4 || (a_in - 0.5 * ang_b).abs() < q4);
        arc[k] = mid[k] && corner[k] && !other[k];
        near_a[k] = q.fl == fa.fid as i64 && q.rr < rr_max;
        near_b[k] = q.fl == fb.fid as i64 && q.rr < rr_max;
    }
    let nb_bins = ((zhi - zlo) / prm.slice_mm).ceil() as i64;
    let bins: Vec<usize> = pts.iter().map(|q| ((((q.z - zlo) / prm.slice_mm) as i64).max(0)).min(nb_bins - 1) as usize).collect();
    let nbu = nb_bins.max(0) as usize;
    let occ_arc = bincount_ge(&bins, &arc, nbu, 2);
    let occ_a = bincount_ge(&bins, &near_a, nbu, 2);
    let occ_b = bincount_ge(&bins, &near_b, nbu, 2);
    let ev: Vec<bool> = (0..nbu).map(|i| occ_arc[i] && occ_a[i] && occ_b[i]).collect();
    let arc_points = arc.iter().filter(|&&b| b).count();
    let arc_slices = ev.iter().filter(|&&b| b).count() as i64;
    if arc_slices < prm.min_arc_slices {
        e.reason = "NO_STRIP".into();
        return e;
    }
    let gap = round_even(prm.gap_bridge_mm / prm.slice_mm) as i64;
    let tot = arc_slices as f64;
    let rs: Vec<(usize, usize)> = runs(&ev, gap)
        .into_iter()
        .filter(|&(a, b)| (ev[a..=b].iter().filter(|&&x| x).count() as f64) >= (prm.min_arc_slices as f64).max(0.1 * tot))
        .collect();
    if rs.is_empty() {
        e.reason = "NO_STRIP".into();
        return e;
    }
    let (b0, b1) = (rs[0].0, rs[rs.len() - 1].1);
    e.cover = mean_bool(&ev[b0..=b1]);
    let mut zz: Vec<f64> = (0..n_pts).filter(|&k| arc[k] && bins[k] >= b0 && bins[k] <= b1).map(|k| pts[k].z).collect();
    zz.sort_by(total_cmp);
    let s_lo = quantiles_sorted(&zz, 0.005);
    let s_hi = quantiles_sorted(&zz, 0.995);
    e.span_mm = s_hi - s_lo;
    if e.span_mm < prm.min_span_mm || e.cover < prm.min_cover {
        e.reason = "SHORT_OR_SPARSE".into();
        return e;
    }
    let nsum = [pla.n[0] + plb.n[0], pla.n[1] + plb.n[1], pla.n[2] + plb.n[2]];
    let bis = if norm3(&nsum) > 1e-9 { unit(&neg3(&nsum)) } else { neg3(&pla.n) };
    let half = 0.5 * ang_b;
    let lim = 3.0f64.max(0.2 * ang_b);
    let nbis: Vec<usize> = (0..n_pts)
        .filter(|&k| out_arc[k] && corner[k] && (a_pt[k] - half).abs() < lim && pts[k].z >= s_lo && pts[k].z <= s_hi)
        .collect();
    let mut ro: Option<f64> = None;
    let mut apex_off: Option<f64> = None;
    if nbis.len() >= 5 {
        let v: Vec<f64> = nbis.iter().map(|&k| dot3(&pts[k].rel, &bis)).collect();
        let ao = median(&v);
        apex_off = Some(ao);
        let kk = 1.0 / (sweep.min(170.0).to_radians() / 2.0).dcos() - 1.0;
        if kk > 1e-3 && ao > 0.0 {
            ro = Some(ao / kk);
        }
    }
    let mo: Vec<usize> = (0..n_pts)
        .filter(|&k| mid[k] && out_arc[k] && corner[k] && !other[k] && pts[k].z >= s_lo && pts[k].z <= s_hi)
        .collect();
    let mut ro_chord: Option<f64> = None;
    if mo.len() >= 10 {
        let perp_b = unit(&cross3(&u, &bis));
        let c2: Vec<f64> = mo.iter().map(|&k| dot3(&pts[k].rel, &perp_b)).collect();
        let chord = quantile_of(&c2, 0.95) - quantile_of(&c2, 0.05);
        ro_chord = Some(chord / (2.0 * (sweep.to_radians() / 4.0).dsin()).max(1e-6));
    }
    let r_test = if ro.is_some() { ro } else { ro_chord };
    if let Some(rt) = r_test {
        if rt > prm.max_ro_t * t {
            e.reason = "ROLLED".into();
            return e;
        }
    }
    let ro_use = match ro {
        Some(r) if 0.2 * t <= r && r <= 15.0 * t + 10.0 => r,
        _ => t.max(1.0) * 2.0,
    };
    let d_t = ro_use * (sweep.min(170.0).to_radians() / 2.0).dtan();
    let (lp2, valid, n_st) = station_refine(sc, fa, fb, &pla, &plb, &lp, &u, &wa, &wb, d_t, s_lo, s_hi, prm);
    if (valid as i64) < 3i64.max((MIN_STATION_FRACTION * n_st as f64).ceil() as i64) {
        e.reason = "LOW_DENSITY".into();
        return e;
    }
    let (back, base) = if apex_off.is_some() && ro.is_some() {
        (apex_off.unwrap(), lp)
    } else {
        (ro_use * (1.0 / (sweep.min(170.0).to_radians() / 2.0).dcos() - 1.0), lp2)
    };
    e.line_start = Some([
        (base[0] + u[0] * s_lo) + bis[0] * back,
        (base[1] + u[1] * s_lo) + bis[1] * back,
        (base[2] + u[2] * s_lo) + bis[2] * back,
    ]);
    e.line_end = Some([
        (base[0] + u[0] * s_hi) + bis[0] * back,
        (base[1] + u[1] * s_hi) + bis[1] * back,
        (base[2] + u[2] * s_hi) + bis[2] * back,
    ]);
    e.ok = true;
    e.geom = Some(Geom {
        lp: lp2,
        u,
        na: (pla.n, pla.strided),
        nb: (plb.n, plb.strided),
        ro: ro_use,
        sweep,
        dens: (arc_points as f64) * sc.pa / e.span_mm.max(1e-6),
    });
    e.score = e.cover * 1.0f64.min(e.span_mm / 20.0);
    e
}

/// _station_refine -> (lp, valid stations, n_stations)
fn station_refine(sc: &Scan, fa: &Flange, fb: &Flange, pla: &Plane, plb: &Plane, lp0: &V3, u: &V3, wa: &V3, wb: &V3,
                  d_t: f64, zlo: f64, zhi: f64, prm: &GraphParams) -> (V3, usize, usize) {
    let (p, t) = (sc.p, sc.t);
    let depth = prm.local_depth_mm;
    let mut lp = *lp0;
    let mut res: Vec<(Vec<usize>, Vec<f64>, Plane)> = Vec::new();
    for (f, pl, w) in [(fa, pla, wa), (fb, plb, wb)] {
        let idx = &f.idx;
        let pts = pts_of(p, idx);
        let rel: Vec<V3> = pts.iter().map(|x| sub3(x, &lp)).collect();
        let z: Vec<f64> = rel.iter().map(|r| dot3(r, u)).collect();
        let s: Vec<f64> = rel.iter().map(|r| dot3(r, w)).collect();
        let d = pl.distances(&pts);
        let start0 = d_t + 0.5;
        let start = flat_start(&s, &d, &z, start0, depth, t, zlo, zhi);
        let strip: Vec<bool> = (0..idx.len())
            .map(|i| z[i] > zlo - 1.0 && z[i] < zhi + 1.0 && s[i] > start && s[i] < depth && (d[i] + 0.5 * t).abs() < 0.5 * t + 0.6)
            .collect();
        let sidx: Vec<usize> = (0..idx.len()).filter(|&i| strip[i]).map(|i| idx[i]).collect();
        let sz: Vec<f64> = (0..idx.len()).filter(|&i| strip[i]).map(|i| z[i]).collect();
        let mut q: Option<Plane> = None;
        if sidx.len() >= MIN_STATION_POINTS * STATIONS_MIN {
            q = reseat_plane(&pts_of(p, &sidx), pl, t);
        }
        res.push((sidx, sz, q.unwrap_or_else(|| pl.clone())));
    }
    let (ia, za, mut pla2) = (res[0].0.clone(), res[0].1.clone(), res[0].2.clone());
    let (ib, zb, mut plb2) = (res[1].0.clone(), res[1].1.clone(), res[1].2.clone());
    if let Some((p0, d0)) = intersect(&pla2, &plb2) {
        let mut d0 = d0;
        if dot3(&d0, u) < 0.0 {
            d0 = neg3(&d0);
        }
        if dot3(&d0, u) > 3.0f64.to_radians().dcos() {
            let f = dot3(&sub3(&lp, &p0), &d0);
            lp = [p0[0] + f * d0[0], p0[1] + f * d0[1], p0[2] + f * d0[2]];
        } else {
            pla2 = pla.clone();
            plb2 = plb.clone();
        }
    }
    let need = MIN_STATION_POINTS;
    if ia.len() < need * 3 || ib.len() < need * 3 {
        return (lp, 0, STATIONS_MIN);
    }
    let (zs, wst) = station_layout(zlo, zhi);
    let n_st = zs.len();
    let sharps: Vec<V3> = zs
        .par_iter()
        .filter_map(|&zi| {
            let sa: Vec<usize> = (0..ia.len()).filter(|&k| (za[k] - zi).abs() <= 0.5 * wst).map(|k| ia[k]).collect();
            let sb: Vec<usize> = (0..ib.len()).filter(|&k| (zb[k] - zi).abs() <= 0.5 * wst).map(|k| ib[k]).collect();
            if sa.len() < need || sb.len() < need {
                return None;
            }
            let fpa = fit_side(&pts_of(p, &sa), &pla2, t)?;
            let fpb = fit_side(&pts_of(p, &sb), &plb2, t)?;
            station_sharp(&fpa, &fpb, &lp, u, zi)
        })
        .collect();
    let valid = sharps.len();
    if valid >= 3 {
        let offs: Vec<V3> = sharps
            .iter()
            .map(|q| {
                let r = sub3(q, &lp);
                let z = dot3(&r, u);
                [r[0] - z * u[0], r[1] - z * u[1], r[2] - z * u[2]]
            })
            .collect();
        let off = [
            median(&offs.iter().map(|o| o[0]).collect::<Vec<_>>()),
            median(&offs.iter().map(|o| o[1]).collect::<Vec<_>>()),
            median(&offs.iter().map(|o| o[2]).collect::<Vec<_>>()),
        ];
        let no = norm3(&off);
        if no > SHARP_LINE_CONSISTENCY_MM && no < 2.0 * t + 2.0 {
            lp = add3(&lp, &off);
        }
    }
    (lp, valid, n_st)
}

pub fn build_graph(sc: &Scan, prm: &GraphParams) -> Vec<Edge> {
    let cands = candidate_pairs(sc, prm);
    let tree = KdTree::new(sc.p);
    let keys: Vec<(usize, usize)> = cands.keys().cloned().collect();
    keys.par_iter().map(|&(a, b)| measure_edge(sc, &sc.flanges[a], &sc.flanges[b], prm, &tree)).collect()
}

fn line_metrics(a0: &V3, a1: &V3, b0: &V3, b1: &V3) -> (f64, f64, f64) {
    let da = unit(&sub3(a1, a0));
    let db = unit(&sub3(b1, b0));
    let ang = 1.0f64.min(dot3(&da, &db).abs()).dacos().to_degrees();
    // b.mean(0) - a.mean(0) for 2-row arrays: (r0 + r1) / 2 per column
    let mb = [(b0[0] + b1[0]) / 2.0, (b0[1] + b1[1]) / 2.0, (b0[2] + b1[2]) / 2.0];
    let ma = [(a0[0] + a1[0]) / 2.0, (a0[1] + a1[1]) / 2.0, (a0[2] + a1[2]) / 2.0];
    let md = sub3(&mb, &ma);
    let pr = dot3(&md, &da);
    let off = norm3(&[md[0] - da[0] * pr, md[1] - da[1] * pr, md[2] - da[2] * pr]);
    let za1 = dot3(&sub3(a1, a0), &da);
    let za = if za1 >= 0.0 { (0.0, za1) } else { (za1, 0.0) };
    let x = dot3(&sub3(b0, a0), &da);
    let y = dot3(&sub3(b1, a0), &da);
    let zb = if x <= y { (x, y) } else { (y, x) };
    let ov = 0.0f64.max(za.1.min(zb.1) - za.0.max(zb.0));
    (ang, off, ov / 1e-9f64.max((za.1 - za.0).min(zb.1 - zb.0)))
}

pub fn nms(edges: &[Edge], prm: &GraphParams, drop: &BTreeSet<usize>) -> Vec<Edge> {
    let mut c: Vec<&Edge> = edges.iter().filter(|e| e.ok && !drop.contains(&e.a) && !drop.contains(&e.b)).collect();
    c.sort_by(|x, y| total_cmp(&(-(x.score * 1000.0 + x.span_mm)), &(-(y.score * 1000.0 + y.span_mm))));
    let mut keep: Vec<Edge> = Vec::new();
    for e in c {
        let (s, t) = (e.line_start.unwrap(), e.line_end.unwrap());
        let dup = keep.iter().any(|k| {
            let (ang, off, ov) = line_metrics(&s, &t, &k.line_start.unwrap(), &k.line_end.unwrap());
            ang < prm.nms_deg && off < prm.nms_offset_mm && ov > 0.5
        });
        if !dup {
            keep.push(e.clone());
        }
    }
    keep
}

pub fn suppress_facets(sc: &Scan, edges: &[Edge], prm: &GraphParams) -> BTreeSet<usize> {
    let t = sc.t;
    let mut facets = BTreeSet::new();
    let ok: Vec<&Edge> = edges.iter().filter(|e| e.ok && e.geom.is_some()).collect();
    for x in sc.flanges {
        let px: Vec<V3> = thin_idx(x.idx.len(), 3000).iter().map(|&k| sc.p[x.idx[k]]).collect();
        for e in &ok {
            if x.fid == e.a || x.fid == e.b {
                continue;
            }
            let g = e.geom.as_ref().unwrap();
            let arc_len = (g.ro + t) * g.sweep.to_radians();
            if x.width > 1.5 * arc_len + 2.0 {
                continue;
            }
            let u = g.u;
            if dot3(&x.normal, &u).abs() > prm.arc_tol_deg.to_radians().dsin() {
                continue;
            }
            let ang_ab = clip(dotn(&g.na.0, g.na.1, &g.nb.0, g.nb.1), -1.0, 1.0).dacos().to_degrees();
            let mut ok_dir = false;
            for nx in [x.normal, neg3(&x.normal)] {
                let a1 = clip(dotn(&nx, false, &g.na.0, g.na.1), -1.0, 1.0).dacos().to_degrees();
                let a2 = clip(dotn(&nx, false, &g.nb.0, g.nb.1), -1.0, 1.0).dacos().to_degrees();
                if a1 + a2 < ang_ab + 6.0 && a1 > 2.0 && a2 > 2.0 {
                    ok_dir = true;
                }
            }
            if !ok_dir {
                continue;
            }
            let lim = g.ro + 2.0 * t + 1.0;
            let inside: Vec<bool> = px
                .iter()
                .map(|q| {
                    let r = sub3(q, &g.lp);
                    let z = dot3(&r, &u);
                    norm3(&[r[0] - z * u[0], r[1] - z * u[1], r[2] - z * u[2]]) < lim
                })
                .collect();
            if mean_bool(&inside) >= 0.8 {
                facets.insert(x.fid);
                break;
            }
        }
    }
    facets
}

pub fn break_triangles(kept: Vec<Edge>) -> Vec<Edge> {
    let mut edges = kept;
    loop {
        let mut by: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        let mut order: Vec<(usize, usize)> = Vec::new();
        for (i, e) in edges.iter().enumerate() {
            let k = (e.a.min(e.b), e.a.max(e.b));
            if !by.contains_key(&k) {
                order.push(k);
            }
            by.insert(k, i);
        }
        let mut adj: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
        for &(a, b) in &order {
            adj.entry(a).or_default().insert(b);
            adj.entry(b).or_default().insert(a);
        }
        let mut worst: Option<usize> = None;
        for &(a, b) in &order {
            let common: Vec<usize> = adj[&a].intersection(&adj[&b]).cloned().collect();
            for c in common {
                let tri = [by[&(a.min(b), a.max(b))], by[&(a.min(c), a.max(c))], by[&(b.min(c), b.max(c))]];
                let us: Vec<V3> = tri.iter().filter_map(|&i| edges[i].geom.as_ref().map(|g| g.u)).collect();
                let c10 = 10.0f64.to_radians().dcos();
                if us.len() == 3 && [(0, 1), (0, 2), (1, 2)].iter().all(|&(i, j)| dot3(&us[i], &us[j]).abs() < c10) {
                    let mut wi = tri[0];
                    for &ti in &tri[1..] {
                        if edges[ti].span_mm < edges[wi].span_mm {
                            wi = ti;
                        }
                    }
                    let rest: Vec<usize> = tri.iter().cloned().filter(|&ti| ti != wi).collect();
                    let short = edges[wi].span_mm < 0.5 * rest.iter().map(|&i| edges[i].span_mm).fold(f64::INFINITY, f64::min);
                    let thin = edges[wi].geom.as_ref().unwrap().dens
                        < 0.5 * rest.iter().map(|&i| edges[i].geom.as_ref().unwrap().dens).fold(f64::NEG_INFINITY, f64::max);
                    if !(short || thin) {
                        continue;
                    }
                    if worst.is_none() || edges[wi].span_mm < edges[worst.unwrap()].span_mm {
                        worst = Some(wi);
                    }
                }
            }
        }
        match worst {
            None => return edges,
            Some(w) => {
                edges.remove(w);
            }
        }
    }
}

/// run_scan's line set: (start, end, included angle rounded to 3 decimals)
pub fn run_lines(sc: &Scan, n_raw: &[V3], prm: &GraphParams) -> Vec<(V3, V3, f64)> {
    let edges = build_graph(sc, prm);
    let facets = suppress_facets(sc, &edges, prm);
    let kept = nms(&edges, prm, &facets);
    let kept = break_triangles(kept);
    let tree = if prm.d1_snap { Some(KdTree::new(sc.p)) } else { None };
    kept.iter()
        .map(|e| {
            let (mut a, mut b) = (e.line_start.unwrap(), e.line_end.unwrap());
            if let Some(tr) = &tree {
                let th = 180.0 - e.sweep_deg;
                let prof = d1::bend_profile(sc.p, n_raw, tr, &a, &b, sc.t);
                let gate = sc.t / (th.to_radians() / 2.0).dsin().max(1e-3) + 1.0;
                if let Some(d3) = d1::snap_offset(&prof, gate) {
                    a = add3(&a, &d3);
                    b = add3(&b, &d3);
                }
            }
            (a, b, py_round(180.0 - e.sweep_deg, 3))
        })
        .collect()
}

/// Python round(x, ndigits) (correctly rounded decimal, half to even on the exact value)
pub fn py_round(x: f64, nd: usize) -> f64 {
    format!("{:.*}", nd, x).parse::<f64>().unwrap_or(x)
}
