//! numpy.random.default_rng(seed) reproduced: SeedSequence -> PCG64 (XSL-RR), Generator.random,
//! Generator.choice (with / without replacement, with / without p) as numpy 2.x implements them.
#![allow(dead_code)]

const INIT_A: u32 = 0x43b0d7e5;
const MULT_A: u32 = 0x931e8875;
const INIT_B: u32 = 0x8b51f9dd;
const MULT_B: u32 = 0x58f38ded;
const MIX_MULT_L: u32 = 0xca01f9dd;
const MIX_MULT_R: u32 = 0x4973f715;
const XSHIFT: u32 = 16;
const PCG_MULT: u128 = (0x2360ED051FC65DA4u128 << 64) | 0x4385DF649FCCF645u128;

fn hashmix(value: u32, hc: &mut u32) -> u32 {
    let mut v = value ^ *hc;
    *hc = hc.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hc);
    v ^ (v >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

fn seed_state(seed: u64) -> [u64; 4] {
    // entropy array of a non-negative int: 32-bit words, little first (0 -> [0])
    let mut ent: Vec<u32> = Vec::new();
    if seed == 0 {
        ent.push(0);
    }
    let mut s = seed;
    while s > 0 {
        ent.push((s & 0xffff_ffff) as u32);
        s >>= 32;
    }
    let mut pool = [0u32; 4];
    let mut hc = INIT_A;
    for i in 0..4 {
        pool[i] = if i < ent.len() { hashmix(ent[i], &mut hc) } else { hashmix(0, &mut hc) };
    }
    for i_src in 0..4 {
        for i_dst in 0..4 {
            if i_src != i_dst {
                let h = hashmix(pool[i_src], &mut hc);
                pool[i_dst] = mix(pool[i_dst], h);
            }
        }
    }
    for i_src in 4..ent.len() {
        for i_dst in 0..4 {
            let h = hashmix(ent[i_src], &mut hc);
            pool[i_dst] = mix(pool[i_dst], h);
        }
    }
    // generate_state(4, uint64) = 8 uint32 words viewed as little-endian uint64
    let mut hb = INIT_B;
    let mut w = [0u32; 8];
    for i in 0..8 {
        let mut d = pool[i % 4];
        d ^= hb;
        hb = hb.wrapping_mul(MULT_B);
        d = d.wrapping_mul(hb);
        d ^= d >> XSHIFT;
        w[i] = d;
    }
    let mut out = [0u64; 4];
    for i in 0..4 {
        out[i] = (w[2 * i] as u64) | ((w[2 * i + 1] as u64) << 32);
    }
    out
}

pub struct Rng {
    state: u128,
    inc: u128,
    has_u32: bool,
    u32buf: u32,
}

impl Rng {
    /// numpy.random.default_rng(seed)
    pub fn new(seed: u64) -> Rng {
        let v = seed_state(seed);
        let s: u128 = ((v[0] as u128) << 64) | v[1] as u128;
        let i: u128 = ((v[2] as u128) << 64) | v[3] as u128;
        let mut r = Rng { state: 0, inc: (i << 1) | 1, has_u32: false, u32buf: 0 };
        r.step();
        r.state = r.state.wrapping_add(s);
        r.step();
        r
    }
    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let st = self.state;
        let x = ((st >> 64) as u64) ^ (st as u64);
        let rot = (st >> 122) as u32;
        x.rotate_right(rot)
    }
    pub fn next_u32(&mut self) -> u32 {
        if self.has_u32 {
            self.has_u32 = false;
            return self.u32buf;
        }
        let n = self.next_u64();
        self.has_u32 = true;
        self.u32buf = (n >> 32) as u32;
        (n & 0xffff_ffff) as u32
    }
    /// Generator.random()
    pub fn random(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / 9007199254740992.0)
    }
    pub fn random_n(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.random()).collect()
    }
    fn interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;
        if max <= 0xffff_ffff {
            loop {
                let v = (self.next_u32() as u64) & mask;
                if v <= max {
                    return v;
                }
            }
        } else {
            loop {
                let v = self.next_u64() & mask;
                if v <= max {
                    return v;
                }
            }
        }
    }
    fn bounded_u64(&mut self, rng: u64) -> u64 {
        if rng == 0 {
            return 0;
        }
        if rng <= 0xffff_ffff {
            if rng == 0xffff_ffff {
                return self.next_u32() as u64;
            }
            let rng_excl = (rng as u32).wrapping_add(1);
            let mut m = (self.next_u32() as u64) * (rng_excl as u64);
            let mut left = (m & 0xffff_ffff) as u32;
            if left < rng_excl {
                let thr = (u32::MAX - rng as u32) % rng_excl;
                while left < thr {
                    m = (self.next_u32() as u64) * (rng_excl as u64);
                    left = (m & 0xffff_ffff) as u32;
                }
            }
            m >> 32
        } else {
            // bounded_lemire_uint64
            let rng_excl = rng.wrapping_add(1);
            let mut m = (self.next_u64() as u128) * (rng_excl as u128);
            let mut left = m as u64;
            if left < rng_excl {
                let thr = (u64::MAX - rng) % rng_excl;
                while left < thr {
                    m = (self.next_u64() as u128) * (rng_excl as u128);
                    left = m as u64;
                }
            }
            (m >> 64) as u64
        }
    }
    fn shuffle_int(&mut self, n: usize, first: usize, data: &mut [i64]) {
        let mut i = n;
        while i > first {
            i -= 1;
            let j = self.bounded_u64(i as u64) as usize;
            data.swap(i, j);
        }
    }

    /// Generator.choice(pop, size, replace=False) (shuffle=True)
    pub fn choice_noreplace(&mut self, pop: usize, size: usize) -> Vec<i64> {
        let cutoff = 50usize;
        if pop > 10000 && size > pop / cutoff {
            let mut idx: Vec<i64> = (0..pop as i64).collect();
            let first = if pop > size { pop - size } else { 1 }.max(1);
            self.shuffle_int(pop, first, &mut idx);
            idx[(pop - size)..].to_vec()
        } else {
            let mut idx = vec![0i64; size];
            let set_size = (1.2 * size as f64) as u64;
            let mut mask = set_size;
            // _gen_mask
            mask |= mask >> 1;
            mask |= mask >> 2;
            mask |= mask >> 4;
            mask |= mask >> 8;
            mask |= mask >> 16;
            mask |= mask >> 32;
            let set_size = 1 + mask;
            let mut hs = vec![u64::MAX; set_size as usize];
            for j in (pop - size)..pop {
                let val = self.bounded_u64(j as u64);
                let mut loc = val & mask;
                while hs[loc as usize] != u64::MAX && hs[loc as usize] != val {
                    loc = (loc + 1) & mask;
                }
                if hs[loc as usize] == u64::MAX {
                    hs[loc as usize] = val;
                    idx[j - pop + size] = val as i64;
                } else {
                    let mut loc = (j as u64) & mask;
                    while hs[loc as usize] != u64::MAX {
                        loc = (loc + 1) & mask;
                    }
                    hs[loc as usize] = j as u64;
                    idx[j - pop + size] = j as i64;
                }
            }
            self.shuffle_int(size, 1, &mut idx);
            idx
        }
    }

    /// Generator.choice(pop, size, replace=True, p=p)
    pub fn choice_p(&mut self, p: &[f64], size: usize) -> Vec<usize> {
        let cdf = cumsum_norm(p);
        let u = self.random_n(size);
        u.iter().map(|&x| searchsorted_right(&cdf, x)).collect()
    }

    /// Generator.choice(pop, size, replace=False, p=p)
    pub fn choice_p_noreplace(&mut self, p0: &[f64], size: usize) -> Vec<usize> {
        let mut p = p0.to_vec();
        let mut found: Vec<usize> = Vec::with_capacity(size);
        while found.len() < size {
            let x = self.random_n(size - found.len());
            for &f in &found {
                p[f] = 0.0;
            }
            let cdf = cumsum_norm(&p);
            let new: Vec<usize> = x.iter().map(|&v| searchsorted_right(&cdf, v)).collect();
            // np.unique(new, return_index=True), indices sorted: first occurrences, in order
            let mut seen = std::collections::HashSet::new();
            for v in new {
                if seen.insert(v) {
                    found.push(v);
                }
            }
        }
        found
    }
}

fn cumsum_norm(p: &[f64]) -> Vec<f64> {
    let mut c = Vec::with_capacity(p.len());
    let mut s = 0.0f64;
    for &x in p {
        s += x;
        c.push(s);
    }
    let last = *c.last().unwrap();
    for x in c.iter_mut() {
        *x /= last;
    }
    c
}

fn searchsorted_right(a: &[f64], v: f64) -> usize {
    let (mut lo, mut hi) = (0usize, a.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        if a[mid] <= v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}
