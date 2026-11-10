//! Small seeded random generator (xoshiro256++), so results only depend on the seed.

pub(crate) struct Rng {
    s: [u64; 4],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        let mut state = seed;
        let s = [
            splitmix64(&mut state),
            splitmix64(&mut state),
            splitmix64(&mut state),
            splitmix64(&mut state),
        ];
        Rng { s }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [0, bound), without modulo bias (Lemire's method).
    pub(crate) fn below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        let mut m = (self.next_u64() as u128) * (bound as u128);
        if (m as u64) < bound {
            let threshold = bound.wrapping_neg() % bound;
            while (m as u64) < threshold {
                m = (self.next_u64() as u128) * (bound as u128);
            }
        }
        (m >> 64) as u64
    }

    /// A random permutation of 0..n.
    pub(crate) fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.below(i as u64 + 1) as usize;
            order.swap(i, j);
        }
        order
    }

    /// `count` distinct values from 0..n, sorted. Floyd's algorithm, so memory is O(count).
    pub(crate) fn sample_sorted(&mut self, n: usize, count: usize) -> Vec<usize> {
        debug_assert!(count <= n);
        let mut picked = std::collections::HashSet::with_capacity(count);
        for j in (n - count)..n {
            let t = self.below(j as u64 + 1) as usize;
            if !picked.insert(t) {
                picked.insert(j);
            }
        }
        let mut out: Vec<usize> = picked.into_iter().collect();
        out.sort_unstable();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_has_every_index_once() {
        let mut rng = Rng::new(7);
        let mut p = rng.permutation(1000);
        p.sort_unstable();
        assert!(p.iter().enumerate().all(|(i, &v)| i == v));
    }

    #[test]
    fn sample_is_sorted_and_distinct() {
        let mut rng = Rng::new(7);
        let s = rng.sample_sorted(10_000, 500);
        assert_eq!(s.len(), 500);
        assert!(s.windows(2).all(|w| w[0] < w[1]));
        assert!(*s.last().unwrap() < 10_000);
    }

    #[test]
    fn same_seed_same_numbers() {
        let a: Vec<u64> = (0..4).map(|_| Rng::new(1).next_u64()).collect();
        assert!(a.windows(2).all(|w| w[0] == w[1]));
        assert_ne!(Rng::new(1).next_u64(), Rng::new(2).next_u64());
    }
}
