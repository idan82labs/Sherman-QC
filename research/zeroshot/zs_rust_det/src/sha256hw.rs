//! SHA-256 with the Armv8 SHA2 instructions (aarch64 only, detected at run time). The sha2 crate uses SHA-NI on
//! x86-64 by itself but needs its C-assembly `asm` feature for the Arm instructions; this module gives the Mac build
//! the same speed with std::arch intrinsics only. The digest is the standard SHA-256: `selftest --golden` and the
//! unit test compare it with the sha2 crate. Round structure adapted from the sha2 crate's aarch64 backend (MIT /
//! Apache-2.0, after mbedtls).
#![allow(dead_code)]

const K32: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

#[cfg(target_arch = "aarch64")]
pub fn available() -> bool {
    std::arch::is_aarch64_feature_detected!("sha2")
}
#[cfg(not(target_arch = "aarch64"))]
pub fn available() -> bool {
    false
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sha2")]
unsafe fn compress(state: &mut [u32; 8], blocks: &[u8]) {
    use std::arch::aarch64::*;
    let mut abcd = vld1q_u32(state[0..4].as_ptr());
    let mut efgh = vld1q_u32(state[4..8].as_ptr());
    for block in blocks.chunks_exact(64) {
        let abcd_orig = abcd;
        let efgh_orig = efgh;
        let mut s0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block[0..16].as_ptr())));
        let mut s1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block[16..32].as_ptr())));
        let mut s2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block[32..48].as_ptr())));
        let mut s3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block[48..64].as_ptr())));
        macro_rules! rounds4 {
            ($s:expr, $t:expr) => {{
                let tmp = vaddq_u32($s, vld1q_u32(K32[$t..].as_ptr()));
                let abcd_prev = abcd;
                abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
                efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);
            }};
        }
        rounds4!(s0, 0);
        rounds4!(s1, 4);
        rounds4!(s2, 8);
        rounds4!(s3, 12);
        let mut t = 16;
        while t < 64 {
            s0 = vsha256su1q_u32(vsha256su0q_u32(s0, s1), s2, s3);
            rounds4!(s0, t);
            s1 = vsha256su1q_u32(vsha256su0q_u32(s1, s2), s3, s0);
            rounds4!(s1, t + 4);
            s2 = vsha256su1q_u32(vsha256su0q_u32(s2, s3), s0, s1);
            rounds4!(s2, t + 8);
            s3 = vsha256su1q_u32(vsha256su0q_u32(s3, s0), s1, s2);
            rounds4!(s3, t + 12);
            t += 16;
        }
        abcd = vaddq_u32(abcd, abcd_orig);
        efgh = vaddq_u32(efgh, efgh_orig);
    }
    vst1q_u32(state[0..4].as_mut_ptr(), abcd);
    vst1q_u32(state[4..8].as_mut_ptr(), efgh);
}

#[cfg(not(target_arch = "aarch64"))]
unsafe fn compress(_state: &mut [u32; 8], _blocks: &[u8]) {
    unreachable!("sha256hw is aarch64 only")
}

/// Streaming SHA-256 (call only when `available()`).
pub struct Hasher {
    state: [u32; 8],
    buf: [u8; 64],
    nbuf: usize,
    len: u64,
}

impl Hasher {
    pub fn new() -> Hasher {
        Hasher {
            state: [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19],
            buf: [0; 64],
            nbuf: 0,
            len: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        assert!(available());
        self.len += data.len() as u64;
        if self.nbuf > 0 {
            let take = (64 - self.nbuf).min(data.len());
            self.buf[self.nbuf..self.nbuf + take].copy_from_slice(&data[..take]);
            self.nbuf += take;
            data = &data[take..];
            if self.nbuf == 64 {
                let b = self.buf;
                unsafe { compress(&mut self.state, &b) };
                self.nbuf = 0;
            }
        }
        let full = data.len() / 64 * 64;
        if full > 0 {
            unsafe { compress(&mut self.state, &data[..full]) };
        }
        let rest = &data[full..];
        self.buf[..rest.len()].copy_from_slice(rest);
        self.nbuf += rest.len();
    }

    pub fn finalize_hex(mut self) -> String {
        let bits = self.len.wrapping_mul(8);
        let mut pad = vec![0x80u8];
        let rem = (self.len % 64) as usize;
        let zeros = if rem < 56 { 55 - rem } else { 119 - rem };
        pad.extend(std::iter::repeat(0u8).take(zeros));
        pad.extend_from_slice(&bits.to_be_bytes());
        let l = self.len;
        self.update(&pad);
        self.len = l;
        debug_assert_eq!(self.nbuf, 0);
        self.state.iter().map(|w| format!("{:08x}", w)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn equals_sha2_crate() {
        if !available() {
            return;
        }
        let data: Vec<u8> = (0..200_003u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        for n in [0usize, 1, 55, 56, 63, 64, 65, 119, 120, 1000, 200_003] {
            let mut h = Hasher::new();
            let (a, b) = data[..n].split_at(n / 3);
            h.update(a);
            h.update(b);
            assert_eq!(h.finalize_hex(), format!("{:x}", Sha256::digest(&data[..n])), "n={}", n);
        }
    }
}
