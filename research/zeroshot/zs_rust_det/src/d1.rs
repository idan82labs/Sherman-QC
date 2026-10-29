//! Port of d1_two_skin.two_skin.bend_profile (cross-section analysis around a candidate line). The candidate
//! window is gathered with a kd-tree ball that contains the Python selection; the selected points are then
//! filtered with the Python predicates in ascending point order, so every array the Python code builds is
//! reproduced element for element.
#[allow(unused_imports)]
use crate::fmath::{DetF32, DetF64};
use crate::accel;
use crate::kdtree::KdTree;
use crate::num::*;
use serde_json::{json, Map, Value};

fn unit(v: &V3) -> V3 {
    let n = norm3(v).max(1e-12);
    div3(v, n)
}

fn frame(axis: &V3) -> (V3, V3) {
    let u = unit(axis);
    let rf = if u[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let e1 = unit(&cross3(&u, &rf));
    let e2 = cross3(&u, &e1);
    (e1, e2)
}

fn line_intersect(m1: &V2, o1: f64, m2: &V2, o2: f64) -> Option<V2> {
    let a = [m1[0], m1[1], m2[0], m2[1]];
    if accel::det(&a, 2).abs() < 1e-6 {
        return None;
    }
    accel::solve(&a, &[o1, o2], 2).map(|x| [x[0], x[1]])
}

/// numpy.histogram(ang, bins=180, range=(0, 180), weights=w): blocked bincount (65536) as numpy does.
fn histogram180(ang: &[f64], w: &[f64]) -> Vec<f64> {
    let nb = 180usize;
    let mut total = vec![0.0f64; nb];
    let block = 65536usize;
    let mut s = 0;
    while s < ang.len() {
        let e = (s + block).min(ang.len());
        let mut blk = vec![0.0f64; nb];
        for i in s..e {
            let a = ang[i];
            if !(a >= 0.0 && a <= 180.0) {
                continue;
            }
            // f_indices = ((a - 0) / 180) * 180 ; edges are integers 0..180
            let f = ((a - 0.0) / 180.0) * 180.0;
            let mut idx = f as i64;
            if idx == nb as i64 {
                idx -= 1;
            }
            if a < idx as f64 {
                idx -= 1;
            }
            if a >= (idx + 1) as f64 && idx != nb as i64 - 1 {
                idx += 1;
            }
            blk[idx as usize] += w[i];
        }
        for k in 0..nb {
            total[k] += blk[k];
        }
        s = e;
    }
    total
}

pub struct Profile {
    pub ok: bool,
    pub reason: String,
    pub included_angle_deg: Option<f64>,
    pub bisector_2d: Option<V2>,
    pub outer_sharp_2d: Option<V2>,
    pub inner_sharp_2d: Option<V2>,
    pub mid_sharp_2d: Option<V2>,
    pub outer_apex_2d: Option<V2>,
    pub inner_apex_2d: Option<V2>,
    pub outer_apex_support: i64,
    pub inner_apex_support: i64,
    pub arm_points: Option<Vec<i64>>,
    pub arm_outer_fraction: Option<Vec<f64>>,
    pub local_t_mm: Option<f64>,
    pub e1: V3,
    pub e2: V3,
    pub origin: V3,
}

impl Profile {
    pub fn to_json(&self) -> Value {
        let o2 = |v: &Option<V2>| match v {
            Some(x) => json!([x[0], x[1]]),
            None => Value::Null,
        };
        let mut m = Map::new();
        m.insert("ok".into(), json!(self.ok));
        m.insert("reason".into(), json!(self.reason));
        m.insert("included_angle_deg".into(), json!(self.included_angle_deg));
        m.insert("bisector_2d".into(), o2(&self.bisector_2d));
        m.insert("outer_sharp_2d".into(), o2(&self.outer_sharp_2d));
        m.insert("inner_sharp_2d".into(), o2(&self.inner_sharp_2d));
        m.insert("mid_sharp_2d".into(), o2(&self.mid_sharp_2d));
        m.insert("outer_apex_2d".into(), o2(&self.outer_apex_2d));
        m.insert("inner_apex_2d".into(), o2(&self.inner_apex_2d));
        m.insert("outer_apex_support".into(), json!(self.outer_apex_support));
        m.insert("inner_apex_support".into(), json!(self.inner_apex_support));
        m.insert("arm_points".into(), json!(self.arm_points));
        m.insert("arm_outer_fraction".into(), json!(self.arm_outer_fraction));
        m.insert("local_t_mm".into(), json!(self.local_t_mm));
        m.insert("e1".into(), json!(self.e1.to_vec()));
        m.insert("e2".into(), json!(self.e2.to_vec()));
        m.insert("origin".into(), json!(self.origin.to_vec()));
        Value::Object(m)
    }
}

fn med(v: &[f64]) -> f64 {
    median(v)
}

/// bend_profile(points, normals, start, end, t) with the default radius / trim / min_arm_deg.
pub fn bend_profile(pts: &[V3], nrm: &[V3], tree: &KdTree, start: &V3, end: &V3, t: f64) -> Profile {
    let trim = 0.1f64;
    let min_arm_deg = 8.0f64;
    let axis = sub3(end, start);
    let l = norm3(&axis);
    let u = div3(&axis, l);
    let (e1, e2) = frame(&u);
    let mid = [0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1]), 0.5 * (start[2] + end[2])];
    let rr = 14.0f64.max(5.0 * t);
    let mut prof = Profile {
        ok: false,
        reason: String::new(),
        included_angle_deg: None,
        bisector_2d: None,
        outer_sharp_2d: None,
        inner_sharp_2d: None,
        mid_sharp_2d: None,
        outer_apex_2d: None,
        inner_apex_2d: None,
        outer_apex_support: 0,
        inner_apex_support: 0,
        arm_points: None,
        arm_outer_fraction: None,
        local_t_mm: None,
        e1,
        e2,
        origin: mid,
    };
    let hs = (0.5 - trim) * l;
    // the Python selection |s| < hs and |q| < R lies inside the ball of radius sqrt(hs^2 + R^2)
    let mut q: Vec<V2> = Vec::new();
    let mut nn: Vec<V3> = Vec::new();
    let mut r: Vec<f64> = Vec::new();
    if hs.is_finite() && hs > 0.0 {
        let rad = (hs * hs + rr * rr).sqrt() * (1.0 + 1e-9) + 1e-9;
        let idx = tree.ball(&mid, rad);
        for i in idx {
            let rel = sub3(&pts[i], &mid);
            let s = dot3(&rel, &u);
            if !(s.abs() < hs) {
                continue;
            }
            let n = nrm[i];
            let qq = [dot3(&rel, &e1), dot3(&rel, &e2)];
            let ri = norm2(&qq);
            if ri < rr && dot3(&n, &u).abs() < 0.5 {
                q.push(qq);
                nn.push(n);
                r.push(ri);
            }
        }
    }
    if q.len() < 40 {
        prof.reason = "TOO_FEW_POINTS".into();
        return prof;
    }
    let m2: Vec<V2> = nn
        .iter()
        .map(|n| {
            let v = [dot3(n, &e1), dot3(n, &e2)];
            let d = norm2(&v).max(1e-12);
            [v[0] / d, v[1] / d]
        })
        .collect();
    let r_in = 2.5f64.max(1.5 * t);
    let arm_idx: Vec<usize> = (0..q.len()).filter(|&i| r[i] > r_in).collect();
    if arm_idx.len() < 30 {
        prof.reason = "NO_ARMS".into();
        return prof;
    }
    let ang: Vec<f64> = arm_idx.iter().map(|&i| np_mod(m2[i][1].datan2(m2[i][0]).to_degrees(), 180.0)).collect();
    let wts: Vec<f64> = arm_idx.iter().map(|&i| (-r[i] / 6.0).dexp()).collect();
    let hist = histogram180(&ang, &wts);
    // np.convolve(np.r_[hist[-2:], hist, hist[:2]], k, mode="same")[2:-2]
    let mut ext = Vec::with_capacity(184);
    ext.push(hist[178]);
    ext.push(hist[179]);
    ext.extend_from_slice(&hist);
    ext.push(hist[0]);
    ext.push(hist[1]);
    let k = [0.25, 0.5, 1.0, 0.5, 0.25];
    let mut sm = vec![0.0f64; 180];
    for i in 0..180 {
        let c = i + 2;
        let mut acc = 0.0;
        for (j, kk) in k.iter().enumerate() {
            acc += ext[c + j - 2] * kk;
        }
        sm[i] = acc;
    }
    let p1 = argmax(&sm);
    let sm2: Vec<f64> = (0..180)
        .map(|i| {
            let a = (i as i64 - p1 as i64).abs();
            let dd = a.min(180 - a);
            if (dd as f64) > min_arm_deg {
                sm[i]
            } else {
                -1.0
            }
        })
        .collect();
    let p2 = argmax(&sm2);
    if sm2[p2] < 0.5f64.max(0.08 * sm[p1]) {
        prof.reason = "ONE_ARM".into();
        return prof;
    }
    let mut phis = Vec::new();
    for p in [p1, p2] {
        let pc = p as f64 + 0.5;
        let mut sn = Vec::new();
        let mut cs = Vec::new();
        for &a in &ang {
            let d0 = (a - pc).abs();
            let dd = d0.min(180.0 - d0);
            if dd < 4.0 {
                let a2 = (2.0 * a).to_radians();
                sn.push(lsin(a2));
                cs.push(lcos(a2));
            }
        }
        let ms = if sn.is_empty() { f64::NAN } else { mean(&sn) };
        let mc = if cs.is_empty() { f64::NAN } else { mean(&cs) };
        phis.push(np_mod(0.5 * ms.datan2(mc).to_degrees(), 180.0));
    }
    let c6 = 6.0f64.to_radians().dcos();
    let mut arms: Vec<(V2, V2, Vec<usize>)> = Vec::new();
    for phi in phis {
        let nvec = [lcos(phi.to_radians()), lsin(phi.to_radians())];
        let tvec = [-nvec[1], nvec[0]];
        let mut mem: Vec<usize> = arm_idx.iter().cloned().filter(|&i| dot2(&m2[i], &nvec).abs() > c6).collect();
        let along: Vec<f64> = mem.iter().map(|&i| dot2(&q[i], &tvec)).collect();
        let md = med(&along);
        let sgn = if md >= 0.0 { 1.0 } else { -1.0 };
        let a_dir = [sgn * tvec[0], sgn * tvec[1]];
        mem.retain(|&i| dot2(&q[i], &a_dir) > 0.0);
        if mem.len() >= 10 {
            let o: Vec<f64> = mem.iter().map(|&i| dot2(&q[i], &nvec)).collect();
            let rm: Vec<f64> = mem.iter().map(|&i| r[i]).collect();
            let mut ord: Vec<usize> = (0..mem.len()).collect();
            ord.sort_by(|&a, &b| total_cmp(&rm[a], &rm[b]));
            let kk = 5usize.max(mem.len() / 5).min(mem.len());
            let on: Vec<f64> = ord[..kk].iter().map(|&j| o[j]).collect();
            let o_near = med(&on);
            let lim = 1.5 * t + 0.5;
            mem = mem.iter().zip(&o).filter(|(_, &oo)| (oo - o_near).abs() < lim).map(|(&i, _)| i).collect();
        }
        arms.push((a_dir, nvec, mem));
    }
    let a1 = arms[0].0;
    let a2 = arms[1].0;
    let ssum = [a1[0] + a2[0], a1[1] + a2[1]];
    if norm2(&ssum) < 1e-3 {
        prof.reason = "FLAT".into();
        return prof;
    }
    let sl = norm2(&ssum).max(1e-12);
    let b = [-(ssum[0] / sl), -(ssum[1] / sl)];
    let cos_inc = clip(dot2(&a1, &a2), -1.0, 1.0);
    let theta = cos_inc.dacos().to_degrees();
    let mut outer_lines = Vec::new();
    let mut inner_lines = Vec::new();
    let mut fr = Vec::new();
    let mut npts = Vec::new();
    let mut loc_t = Vec::new();
    for (_a_dir, nvec, mem) in &arms {
        let m_out = if dot2(nvec, &b) > 0.0 { *nvec } else { [-nvec[0], -nvec[1]] };
        let mut o_out = Vec::new();
        let mut o_in = Vec::new();
        let mut n_out = 0usize;
        for &i in mem {
            let o = dot2(&q[i], &m_out);
            let on = dot2(&m2[i], &m_out);
            if on > 0.7 {
                o_out.push(o);
                n_out += 1;
            }
            if on < -0.7 {
                o_in.push(o);
            }
        }
        npts.push(mem.len() as i64);
        fr.push(if mem.is_empty() { 0.0 } else { n_out as f64 / mem.len() as f64 });
        let oo = if o_out.len() >= 8 { Some(med(&o_out)) } else { None };
        let oi = if o_in.len() >= 8 { Some(med(&o_in)) } else { None };
        if let (Some(x), Some(y)) = (oo, oi) {
            loc_t.push(x - y);
        }
        if oo.is_none() && oi.is_none() {
            prof.reason = "ARM_WITHOUT_SKIN".into();
            return prof;
        }
        let oo2 = match oo {
            Some(x) => x,
            None => oi.unwrap() + t,
        };
        let oi2 = match oi {
            Some(x) => x,
            None => oo2 - t,
        };
        outer_lines.push((m_out, oo2));
        inner_lines.push((m_out, oi2));
    }
    let vo = line_intersect(&outer_lines[0].0, outer_lines[0].1, &outer_lines[1].0, outer_lines[1].1);
    let vi = line_intersect(&inner_lines[0].0, inner_lines[0].1, &inner_lines[1].0, inner_lines[1].1);
    let (vo, vi) = match (vo, vi) {
        (Some(a), Some(c)) => (a, c),
        _ => {
            prof.reason = "PARALLEL_ARMS".into();
            return prof;
        }
    };
    let vm = [0.5 * (vo[0] + vi[0]), 0.5 * (vo[1] + vi[1])];
    let perp = [-b[1], b[0]];
    let w = 0.75f64.max(0.4 * t);
    let apex = |v0: &V2, sign: f64| -> (Option<V2>, i64) {
        let mut ax_c = Vec::new();
        for i in 0..q.len() {
            let dq = [q[i][0] - v0[0], q[i][1] - v0[1]];
            let lat = dot2(&dq, &perp);
            let ax = dot2(&dq, &b);
            let on = dot2(&m2[i], &b);
            if lat.abs() < w && sign * on > 0.7 && ax < 1.0 && ax > -rr / 2.0 {
                ax_c.push(ax);
            }
        }
        if ax_c.len() < 3 {
            return (None, ax_c.len() as i64);
        }
        let mm = med(&ax_c);
        (Some([v0[0] + b[0] * mm, v0[1] + b[1] * mm]), ax_c.len() as i64)
    };
    let (ao, nao) = apex(&vo, 1.0);
    let (ai, nai) = apex(&vo, -1.0);
    prof.ok = true;
    prof.included_angle_deg = Some(theta);
    prof.bisector_2d = Some(b);
    prof.outer_sharp_2d = Some(vo);
    prof.inner_sharp_2d = Some(vi);
    prof.mid_sharp_2d = Some(vm);
    prof.outer_apex_2d = ao;
    prof.inner_apex_2d = ai;
    prof.outer_apex_support = nao;
    prof.inner_apex_support = nai;
    prof.arm_points = Some(npts);
    prof.arm_outer_fraction = Some(fr);
    prof.local_t_mm = if loc_t.is_empty() { None } else { Some(med(&loc_t)) };
    prof
}

/// snap_line_to_outer_apex: the translation applied to the line (None: not moved)
pub fn snap_offset(prof: &Profile, max_shift: f64) -> Option<V3> {
    if !prof.ok {
        return None;
    }
    let target = match (prof.outer_apex_2d, prof.outer_sharp_2d) {
        (Some(a), _) => a,
        (None, Some(s)) => s,
        _ => return None,
    };
    let shift = target[0].dhypot(target[1]);
    if shift > max_shift {
        return None;
    }
    let (e1, e2) = (prof.e1, prof.e2);
    Some([target[0] * e1[0] + target[1] * e2[0], target[0] * e1[1] + target[1] * e2[1], target[0] * e1[2] + target[1] * e2[2]])
}
