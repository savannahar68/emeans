//! Nearest centroid search, the part where training spends almost all its time.
//!
//! Rows are split in fixed blocks. For each block, the dot products with a tile of
//! centroids come from one SIMD matrix multiply (`matrixmultiply::sgemm`), then
//! `|c|^2 - 2 x.c` is scanned for the smallest value, as faiss does. Blocks run in
//! parallel when the `parallel` feature is on. The block size does not depend on the
//! thread count, so results are the same with 1 thread or 64.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Metric {
    /// Smallest squared L2 distance.
    L2,
    /// Largest inner product.
    InnerProduct,
}

/// Where the rows of a search come from.
#[derive(Clone, Copy)]
pub(crate) enum Rows<'a> {
    /// `n * d` values, row major.
    Dense(&'a [f32]),
    /// The rows of `data` listed in `index`.
    Indexed(&'a [f32], &'a [usize]),
}

impl Rows<'_> {
    pub(crate) fn len(&self, d: usize) -> usize {
        match self {
            Rows::Dense(x) => x.len() / d,
            Rows::Indexed(_, index) => index.len(),
        }
    }

    pub(crate) fn row(&self, i: usize, d: usize) -> &[f32] {
        match self {
            Rows::Dense(x) => &x[i * d..(i + 1) * d],
            Rows::Indexed(x, index) => &x[index[i] * d..(index[i] + 1) * d],
        }
    }
}

/// Rows per block. A multiple of the sgemm micro kernel height.
const BLOCK_ROWS: usize = 128;
/// Upper bound on the dot product scratch of one block, in floats (1 MiB).
const TILE_FLOATS: usize = 1 << 18;

pub(crate) fn squared_norms(x: &[f32], d: usize) -> Vec<f32> {
    x.chunks_exact(d)
        .map(|row| row.iter().map(|v| v * v).sum())
        .collect()
}

/// For every row, the best centroid and its score.
///
/// L2 writes the squared distance, clamped at 0. Inner product writes the dot product.
/// Ties go to the lowest centroid id.
pub(crate) fn nearest(
    rows: Rows<'_>,
    d: usize,
    centroids: &[f32],
    centroid_norms: &[f32],
    metric: Metric,
    dist: &mut [f32],
    labels: &mut [u32],
) {
    let n = rows.len(d);
    debug_assert_eq!(dist.len(), n);
    debug_assert_eq!(labels.len(), n);
    let k = centroids.len() / d;
    let tile = (TILE_FLOATS / BLOCK_ROWS).clamp(1, k.max(1));
    let run = |block: usize, dist: &mut [f32], labels: &mut [u32], scratch: &mut Scratch| {
        search_block(
            rows,
            block * BLOCK_ROWS,
            d,
            centroids,
            centroid_norms,
            tile,
            metric,
            dist,
            labels,
            scratch,
        );
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        dist.par_chunks_mut(BLOCK_ROWS)
            .zip(labels.par_chunks_mut(BLOCK_ROWS))
            .enumerate()
            .for_each_init(Scratch::default, |scratch, (block, (dist, labels))| {
                run(block, dist, labels, scratch)
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = Scratch::default();
        for (block, (dist, labels)) in dist
            .chunks_mut(BLOCK_ROWS)
            .zip(labels.chunks_mut(BLOCK_ROWS))
            .enumerate()
        {
            run(block, dist, labels, &mut scratch);
        }
    }
}

#[derive(Default)]
struct Scratch {
    gathered: Vec<f32>,
    dots: Vec<f32>,
    best: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn search_block(
    rows: Rows<'_>,
    first: usize,
    d: usize,
    centroids: &[f32],
    centroid_norms: &[f32],
    tile: usize,
    metric: Metric,
    dist: &mut [f32],
    labels: &mut [u32],
    scratch: &mut Scratch,
) {
    let m = dist.len();
    let k = centroids.len() / d;
    let block: &[f32] = match rows {
        Rows::Dense(x) => &x[first * d..(first + m) * d],
        Rows::Indexed(x, index) => {
            scratch.gathered.clear();
            for &i in &index[first..first + m] {
                scratch.gathered.extend_from_slice(&x[i * d..(i + 1) * d]);
            }
            &scratch.gathered
        }
    };

    scratch.dots.resize(m * tile, 0.0);
    scratch.best.clear();
    scratch.best.resize(m, f32::INFINITY);
    labels.fill(0);

    for start in (0..k).step_by(tile) {
        let width = tile.min(k - start);
        dot_tile(
            m,
            d,
            width,
            block,
            &centroids[start * d..(start + width) * d],
            &mut scratch.dots,
        );
        scan_tile(
            &scratch.dots[..m * width],
            &centroid_norms[start..start + width],
            start as u32,
            metric,
            &mut scratch.best,
            labels,
        );
    }

    for i in 0..m {
        dist[i] = match metric {
            Metric::L2 => {
                let row = &block[i * d..(i + 1) * d];
                let norm: f32 = row.iter().map(|v| v * v).sum();
                (norm + scratch.best[i]).max(0.0)
            }
            // The scan kept -dot.
            Metric::InnerProduct => -scratch.best[i],
        };
    }
}

/// Lanes of the vectorized scan. 16 floats fill one AVX-512 register.
const LANES: usize = 16;

/// For each row of `dots` (one row per data row, one column per centroid of the tile),
/// keep the best score and its centroid id in `best` and `labels`.
///
/// L2 looks for the smallest `norm - 2 * dot`, inner product for the largest dot.
/// Ties go to the lowest id, the same as a plain loop.
///
/// The work is written over 16 lanes so the compiler turns it into SIMD. On x86 it is
/// compiled for AVX-512 and AVX2 too, and the best one is picked once at run time from
/// what the CPU supports. Other CPUs (and ARM, where NEON is always on) use the plain build.
fn scan_tile(
    dots: &[f32],
    norms: &[f32],
    start: u32,
    metric: Metric,
    best: &mut [f32],
    labels: &mut [u32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: the CPU supports AVX-512F, checked just above.
            return unsafe { scan_tile_avx512(dots, norms, start, metric, best, labels) };
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: the CPU supports AVX2 and FMA, checked just above.
            return unsafe { scan_tile_avx2(dots, norms, start, metric, best, labels) };
        }
    }
    scan_tile_any(dots, norms, start, metric, best, labels)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn scan_tile_avx512(
    dots: &[f32],
    norms: &[f32],
    start: u32,
    metric: Metric,
    best: &mut [f32],
    labels: &mut [u32],
) {
    scan_tile_any(dots, norms, start, metric, best, labels)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn scan_tile_avx2(
    dots: &[f32],
    norms: &[f32],
    start: u32,
    metric: Metric,
    best: &mut [f32],
    labels: &mut [u32],
) {
    scan_tile_any(dots, norms, start, metric, best, labels)
}

#[inline(always)]
fn scan_tile_any(
    dots: &[f32],
    norms: &[f32],
    start: u32,
    metric: Metric,
    best: &mut [f32],
    labels: &mut [u32],
) {
    let width = norms.len();
    let rows = dots
        .chunks_exact(width)
        .zip(best.iter_mut())
        .zip(labels.iter_mut());
    for ((row, best), label) in rows {
        match metric {
            Metric::L2 => scan_row::<true>(row, norms, start, best, label),
            Metric::InnerProduct => scan_row::<false>(row, norms, start, best, label),
        }
    }
}

/// Score of one centroid: `norm - 2 * dot` for L2, `-dot` for inner product, so that
/// lower is always better. `best` holds this score until the end of the block.
#[inline(always)]
fn score<const L2: bool>(dot: f32, norm: f32) -> f32 {
    if L2 { norm - 2.0 * dot } else { -dot }
}

#[inline(always)]
fn scan_row<const L2: bool>(
    row: &[f32],
    norms: &[f32],
    start: u32,
    best: &mut f32,
    label: &mut u32,
) {
    let mut lane_best = [f32::INFINITY; LANES];
    let mut lane_id = [0u32; LANES];
    let mut ids: [u32; LANES] = std::array::from_fn(|l| l as u32);
    let (dot_chunks, _) = row.as_chunks::<LANES>();
    let (norm_chunks, _) = norms.as_chunks::<LANES>();
    let tail = dot_chunks.len() * LANES;
    for (dots, norms) in dot_chunks.iter().zip(norm_chunks) {
        for l in 0..LANES {
            let s = score::<L2>(dots[l], norms[l]);
            let better = s < lane_best[l];
            lane_best[l] = if better { s } else { lane_best[l] };
            lane_id[l] = if better { ids[l] } else { lane_id[l] };
            ids[l] += LANES as u32;
        }
    }
    if tail > 0 {
        // Lowest score over the lanes, lowest id on ties.
        let (mut v, mut id) = (lane_best[0], lane_id[0]);
        for l in 1..LANES {
            if lane_best[l] < v || (lane_best[l] == v && lane_id[l] < id) {
                v = lane_best[l];
                id = lane_id[l];
            }
        }
        // Earlier tiles have lower ids, so only a strictly better score wins.
        if v < *best {
            *best = v;
            *label = start + id;
        }
    }
    for (j, (&dot, &norm)) in row[tail..].iter().zip(&norms[tail..]).enumerate() {
        let s = score::<L2>(dot, norm);
        if s < *best {
            *best = s;
            *label = start + (tail + j) as u32;
        }
    }
}

/// `dots[m x width] = block[m x d] * tile[width x d]^T`, all row major.
#[cfg(not(feature = "accelerate"))]
fn dot_tile(m: usize, d: usize, width: usize, block: &[f32], tile: &[f32], dots: &mut [f32]) {
    assert!(block.len() >= m * d && tile.len() >= width * d && dots.len() >= m * width);
    // SAFETY: the asserts above cover every element the strides reach. The tile is
    // read as its transpose by swapping its strides.
    unsafe {
        matrixmultiply::sgemm(
            m,
            d,
            width,
            1.0,
            block.as_ptr(),
            d as isize,
            1,
            tile.as_ptr(),
            1,
            d as isize,
            0.0,
            dots.as_mut_ptr(),
            width as isize,
            1,
        );
    }
}

#[cfg(feature = "accelerate")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// Same as above with Apple's Accelerate, which uses the matrix unit of Apple chips.
#[cfg(feature = "accelerate")]
fn dot_tile(m: usize, d: usize, width: usize, block: &[f32], tile: &[f32], dots: &mut [f32]) {
    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    const TRANS: i32 = 112;
    assert!(block.len() >= m * d && tile.len() >= width * d && dots.len() >= m * width);
    let (m, n, k) = (m as i32, width as i32, d as i32);
    // SAFETY: the assert above covers every element read or written.
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            TRANS,
            m,
            n,
            k,
            1.0,
            block.as_ptr(),
            k,
            tile.as_ptr(),
            k,
            0.0,
            dots.as_mut_ptr(),
            n,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn brute_force(x: &[f32], c: &[f32], d: usize) -> Vec<(f32, u32)> {
        x.chunks_exact(d)
            .map(|row| {
                let mut best = (f32::INFINITY, 0);
                for (j, cent) in c.chunks_exact(d).enumerate() {
                    let dist: f32 = row.iter().zip(cent).map(|(a, b)| (a - b) * (a - b)).sum();
                    if dist < best.0 {
                        best = (dist, j as u32);
                    }
                }
                best
            })
            .collect()
    }

    #[test]
    fn matches_brute_force() {
        let mut rng = Rng::new(3);
        // Sizes that do not divide the block or the tile.
        let (n, k, d) = (1000, 2100, 7);
        let x: Vec<f32> = (0..n * d).map(|_| rng.next_f64() as f32).collect();
        let c: Vec<f32> = (0..k * d).map(|_| rng.next_f64() as f32).collect();
        let norms = squared_norms(&c, d);
        let mut dist = vec![0.0; n];
        let mut labels = vec![0; n];
        nearest(
            Rows::Dense(&x),
            d,
            &c,
            &norms,
            Metric::L2,
            &mut dist,
            &mut labels,
        );
        let expected = brute_force(&x, &c, d);
        let mut same = 0;
        for i in 0..n {
            assert!((dist[i] - expected[i].0).abs() < 1e-4);
            same += (labels[i] == expected[i].1) as usize;
        }
        // Near ties can flip with rounding, but almost all labels must agree.
        assert!(same >= n - 5, "{same} of {n} labels agree");
    }

    #[test]
    fn indexed_rows_match_dense_rows() {
        let mut rng = Rng::new(4);
        let (n, k, d) = (300, 50, 16);
        let x: Vec<f32> = (0..n * d).map(|_| rng.next_f64() as f32).collect();
        let c: Vec<f32> = (0..k * d).map(|_| rng.next_f64() as f32).collect();
        let norms = squared_norms(&c, d);
        let index: Vec<usize> = (0..n).rev().collect();
        let mut d1 = vec![0.0; n];
        let mut l1 = vec![0; n];
        let mut d2 = vec![0.0; n];
        let mut l2 = vec![0; n];
        nearest(
            Rows::Dense(&x),
            d,
            &c,
            &norms,
            Metric::InnerProduct,
            &mut d1,
            &mut l1,
        );
        nearest(
            Rows::Indexed(&x, &index),
            d,
            &c,
            &norms,
            Metric::InnerProduct,
            &mut d2,
            &mut l2,
        );
        for i in 0..n {
            assert_eq!(l1[i], l2[n - 1 - i]);
            assert_eq!(d1[i], d2[n - 1 - i]);
        }
    }
}
