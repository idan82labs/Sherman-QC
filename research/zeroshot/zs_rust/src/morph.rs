//! Binary morphology and 8-connected labelling on small row-major grids, matching scipy.ndimage
//! (border_value 0) for the structures the lane uses.

/// ndimage.binary_closing(grid, structure=np.ones((2, 2))): dilation out[i,j] = OR g[i+a, j+b] (a, b in {0,1}),
/// then erosion out[i,j] = AND d[i-a, j-b]; outside the grid is false.
pub fn closing2x2(g: &[bool], ns: usize, na: usize) -> Vec<bool> {
    let at = |v: &[bool], i: isize, j: isize| -> bool {
        if i < 0 || j < 0 || i as usize >= ns || j as usize >= na {
            false
        } else {
            v[i as usize * na + j as usize]
        }
    };
    let mut dil = vec![false; ns * na];
    for i in 0..ns as isize {
        for j in 0..na as isize {
            dil[i as usize * na + j as usize] = at(g, i, j) || at(g, i + 1, j) || at(g, i, j + 1) || at(g, i + 1, j + 1);
        }
    }
    let mut ero = vec![false; ns * na];
    for i in 0..ns as isize {
        for j in 0..na as isize {
            ero[i as usize * na + j as usize] =
                at(&dil, i, j) && at(&dil, i - 1, j) && at(&dil, i, j - 1) && at(&dil, i - 1, j - 1);
        }
    }
    ero
}

/// ndimage.binary_dilation(grid, structure=generate_binary_structure(2, 1)) (4-neighbour cross)
pub fn dilate_cross(g: &[bool], ns: usize, na: usize) -> Vec<bool> {
    let at = |i: isize, j: isize| -> bool {
        if i < 0 || j < 0 || i as usize >= ns || j as usize >= na {
            false
        } else {
            g[i as usize * na + j as usize]
        }
    };
    let mut out = vec![false; ns * na];
    for i in 0..ns as isize {
        for j in 0..na as isize {
            out[i as usize * na + j as usize] = at(i, j) || at(i - 1, j) || at(i + 1, j) || at(i, j - 1) || at(i, j + 1);
        }
    }
    out
}

/// 8-connected component labels (0 = background, components numbered from 1 in raster order of first pixel).
pub fn label8(g: &[bool], ns: usize, na: usize) -> Vec<u32> {
    let mut parent: Vec<u32> = (0..(ns * na) as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for i in 0..ns {
        for j in 0..na {
            let k = i * na + j;
            if !g[k] {
                continue;
            }
            let nbrs: [(isize, isize); 4] = [(-1, -1), (-1, 0), (-1, 1), (0, -1)];
            for (di, dj) in nbrs {
                let ii = i as isize + di;
                let jj = j as isize + dj;
                if ii < 0 || jj < 0 || jj as usize >= na {
                    continue;
                }
                let kk = ii as usize * na + jj as usize;
                if g[kk] {
                    let a = find(&mut parent, k as u32);
                    let b = find(&mut parent, kk as u32);
                    if a != b {
                        parent[a.max(b) as usize] = a.min(b);
                    }
                }
            }
        }
    }
    let mut lab = vec![0u32; ns * na];
    let mut ids: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for k in 0..ns * na {
        if g[k] {
            let r = find(&mut parent, k as u32);
            let n = ids.len() as u32 + 1;
            let id = *ids.entry(r).or_insert(n);
            lab[k] = id;
        }
    }
    lab
}

pub fn labels_in_rows(lab: &[u32], na: usize, r0: usize, r1: usize) -> std::collections::BTreeSet<u32> {
    let mut s = std::collections::BTreeSet::new();
    for i in r0..r1 {
        for j in 0..na {
            let l = lab[i * na + j];
            if l != 0 {
                s.insert(l);
            }
        }
    }
    s
}

/// Is there an 8-connected component that has pixels in rows [l0, l1) and in rows [r0, r1)?
pub fn rows_connected8(g: &[bool], ns: usize, na: usize, l0: usize, l1: usize, r0: usize, r1: usize) -> bool {
    let lab = label8(g, ns, na);
    let left = labels_in_rows(&lab, na, l0, l1);
    let right = labels_in_rows(&lab, na, r0, r1);
    left.intersection(&right).next().is_some()
}
