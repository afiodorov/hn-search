//! Sign-bit binary quantization + Hamming distance.
//!
//! Packing must match the Python builder (`misc/build_search_artifacts.py`), which
//! uses `np.packbits(bits, bitorder="big")`: bit `i` lands in byte `i/8` at position
//! `7 - (i%8)` (MSB-first). Hamming is invariant to a consistent permutation, but we
//! keep the exact convention so codes built either side are interchangeable.

pub const DIM: usize = 768;
pub const CODE_BYTES: usize = DIM / 8; // 96

/// Quantize a 768-d vector to a 96-byte sign-bit code (bit = 1 iff value > 0).
pub fn quantize(vec: &[f32]) -> [u8; CODE_BYTES] {
    debug_assert_eq!(vec.len(), DIM);
    let mut code = [0u8; CODE_BYTES];
    for (i, &v) in vec.iter().enumerate() {
        if v > 0.0 {
            code[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    code
}

/// Hamming distance between two 96-byte codes (popcount of XOR), in u64 chunks.
#[inline]
pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), CODE_BYTES);
    debug_assert_eq!(b.len(), CODE_BYTES);
    let mut dist = 0u32;
    let mut i = 0;
    // 96 == 12 * 8, so the chunk loop consumes everything.
    while i + 8 <= CODE_BYTES {
        let x = u64::from_le_bytes(a[i..i + 8].try_into().unwrap());
        let y = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        dist += (x ^ y).count_ones();
        i += 8;
    }
    while i < CODE_BYTES {
        dist += (a[i] ^ b[i]).count_ones();
        i += 1;
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_sign_and_packing() {
        let mut v = vec![-1.0f32; DIM];
        v[0] = 1.0; // bit 0 -> byte 0, MSB
        v[9] = 2.0; // bit 9 -> byte 1, position 6 (value 0b0100_0000)
        let code = quantize(&v);
        assert_eq!(code[0], 0b1000_0000);
        assert_eq!(code[1], 0b0100_0000);
    }

    #[test]
    fn hamming_counts_diff_bits() {
        let a = [0u8; CODE_BYTES];
        let mut b = [0u8; CODE_BYTES];
        b[0] = 0b1010_0000;
        b[5] = 0b0000_0011;
        assert_eq!(hamming(&a, &b), 4);
    }
}
