//! Triangle mesh with face normals / areas and area-weighted vertex normals, computed exactly as
//! z3s.mesh_from_arrays / two_skin.mesh_from_arrays (same operation order).
use crate::num::*;

pub struct Mesh {
    pub v: Vec<V3>,
    pub f: Vec<[u32; 3]>,
    pub fnrm: Vec<V3>,
    pub fa: Vec<f64>,
    pub vn: Vec<V3>,
}

impl Mesh {
    pub fn from_arrays(v: Vec<V3>, f_in: &[[u32; 3]]) -> Mesh {
        let mut f = Vec::with_capacity(f_in.len());
        let mut cr_all = Vec::with_capacity(f_in.len());
        let mut nr_all = Vec::with_capacity(f_in.len());
        for t in f_in {
            let p0 = v[t[0] as usize];
            let cr = cross3(&sub3(&v[t[1] as usize], &p0), &sub3(&v[t[2] as usize], &p0));
            let nr = norm3(&cr);
            if nr > 1e-12 {
                f.push(*t);
                cr_all.push(cr);
                nr_all.push(nr);
            }
        }
        let fnrm: Vec<V3> = cr_all.iter().zip(&nr_all).map(|(c, &n)| [c[0] / n, c[1] / n, c[2] / n]).collect();
        let fa: Vec<f64> = nr_all.iter().map(|&n| 0.5 * n).collect();
        let mut vn = vec![[0.0f64; 3]; v.len()];
        for k in 0..3 {
            for (t, c) in f.iter().zip(&cr_all) {
                let x = &mut vn[t[k] as usize];
                x[0] += c[0];
                x[1] += c[1];
                x[2] += c[2];
            }
        }
        for x in vn.iter_mut() {
            let n = norm3(x).max(1e-300);
            *x = [x[0] / n, x[1] / n, x[2] / n];
        }
        Mesh { v, f, fnrm, fa, vn }
    }
}
