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
    search(
        rows,
        d,
        centroids,
        centroid_norms,
        metric,
        dist,
        labels,
        &mut [],
        &mut [],
    );
}

/// Like [`nearest`], and also writes the second best centroid and its score.
/// With a single centroid there is no second one: its score is infinite (negative
/// infinity for inner product) and its id is `u32::MAX`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn nearest_two(
    rows: Rows<'_>,
    d: usize,
    centroids: &[f32],
    centroid_norms: &[f32],
    metric: Metric,
    dist: &mut [f32],
    labels: &mut [u32],
    dist2: &mut [f32],
    labels2: &mut [u32],
) {
    debug_assert_eq!(dist2.len(), dist.len());
    debug_assert_eq!(labels2.len(), dist.len());
    search(
        rows,
        d,
        centroids,
        centroid_norms,
        metric,
        dist,
        labels,
        dist2,
        labels2,
    );
}

/// Output of one block. `dist2` and `labels2` are empty when only the best is wanted.
struct Out<'a> {
    dist: &'a mut [f32],
    labels: &'a mut [u32],
    dist2: &'a mut [f32],
    labels2: &'a mut [u32],
}

#[allow(clippy::too_many_arguments)]
fn search(
    rows: Rows<'_>,
    d: usize,
    centroids: &[f32],
    centroid_norms: &[f32],
    metric: Metric,
    dist: &mut [f32],
    labels: &mut [u32],
    dist2: &mut [f32],
    labels2: &mut [u32],
) {
    let n = rows.len(d);
    debug_assert_eq!(dist.len(), n);
    debug_assert_eq!(labels.len(), n);
    let k = centroids.len() / d;
    let tile = (TILE_FLOATS / BLOCK_ROWS).clamp(1, k.max(1));
    let mut dist2 = dist2.chunks_mut(BLOCK_ROWS);
    let mut labels2 = labels2.chunks_mut(BLOCK_ROWS);
    let blocks: Vec<Out<'_>> = dist
        .chunks_mut(BLOCK_ROWS)
        .zip(labels.chunks_mut(BLOCK_ROWS))
        .map(|(dist, labels)| Out {
            dist,
            labels,
            dist2: dist2.next().unwrap_or_default(),
            labels2: labels2.next().unwrap_or_default(),
        })
        .collect();
    let run = |block: usize, out: Out<'_>, scratch: &mut Scratch| {
        search_block(
            rows,
            block * BLOCK_ROWS,
            d,
            centroids,
            centroid_norms,
            tile,
            metric,
            out,
            scratch,
        );
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        blocks
            .into_par_iter()
            .enumerate()
            .for_each_init(Scratch::default, |scratch, (block, out)| {
                run(block, out, scratch)
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = Scratch::default();
        for (block, out) in blocks.into_iter().enumerate() {
            run(block, out, &mut scratch);
        }
    }
}

#[derive(Default)]
struct Scratch {
    gathered: Vec<f32>,
    dots: Vec<f32>,
    best: Vec<f32>,
    best2: Vec<f32>,
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
    out: Out<'_>,
    scratch: &mut Scratch,
) {
    let m = out.dist.len();
    let k = centroids.len() / d;
    let two = !out.dist2.is_empty();
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
    scratch.best2.clear();
    if two {
        // No id yet, so the first candidate does not push a fake id 0 to second place.
        out.labels.fill(u32::MAX);
        scratch.best2.resize(m, f32::INFINITY);
        out.labels2.fill(u32::MAX);
    } else {
        out.labels.fill(0);
    }

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
        let mut best = Best {
            best: &mut scratch.best,
            labels: &mut *out.labels,
            best2: &mut scratch.best2,
            labels2: &mut *out.labels2,
            two,
        };
        scan_tile(
            &mut scratch.dots[..m * width],
            &centroid_norms[start..start + width],
            start as u32,
            metric,
            &mut best,
        );
    }

    // The scan kept norm - 2 * dot for L2 and -dot for inner product.
    let finish = |row: usize, score: f32| match metric {
        Metric::L2 => {
            let row = &block[row * d..(row + 1) * d];
            let norm: f32 = row.iter().map(|v| v * v).sum();
            (norm + score).max(0.0)
        }
        Metric::InnerProduct => -score,
    };
    for i in 0..m {
        out.dist[i] = finish(i, scratch.best[i]);
        if two {
            out.dist2[i] = finish(i, scratch.best2[i]);
            if out.labels[i] == u32::MAX {
                out.labels[i] = 0;
            }
        }
    }
}

/// Best and second best score of each row so far, with their centroid ids.
/// The second pair is only kept when `two` is set.
struct Best<'a> {
    best: &'a mut [f32],
    labels: &'a mut [u32],
    best2: &'a mut [f32],
    labels2: &'a mut [u32],
    two: bool,
}

/// Lanes of the vectorized scan. 16 floats fill one AVX-512 register.
const LANES: usize = 16;

/// For each row of `dots` (one row per data row, one column per centroid of the tile),
/// keep the best score and its centroid id in `best`, and the second best too when
/// `best.two` is set.
///
/// L2 looks for the smallest `norm - 2 * dot`, inner product for the largest dot.
/// Ties go to the lowest id, the same as a plain loop.
///
/// The work is written over 16 lanes so the compiler turns it into SIMD. On x86 it is
/// compiled for AVX-512 and AVX2 too, and the best one is picked once at run time from
/// what the CPU supports. Other CPUs (and ARM, where NEON is always on) use the plain build.
fn scan_tile(dots: &mut [f32], norms: &[f32], start: u32, metric: Metric, best: &mut Best<'_>) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: the CPU supports AVX-512F, checked just above.
            return unsafe { scan_tile_avx512(dots, norms, start, metric, best) };
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: the CPU supports AVX2 and FMA, checked just above.
            return unsafe { scan_tile_avx2(dots, norms, start, metric, best) };
        }
    }
    scan_tile_any(dots, norms, start, metric, best)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn scan_tile_avx512(
    dots: &mut [f32],
    norms: &[f32],
    start: u32,
    metric: Metric,
    best: &mut Best<'_>,
) {
    scan_tile_any(dots, norms, start, metric, best)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn scan_tile_avx2(
    dots: &mut [f32],
    norms: &[f32],
    start: u32,
    metric: Metric,
    best: &mut Best<'_>,
) {
    scan_tile_any(dots, norms, start, metric, best)
}

#[inline(always)]
fn scan_tile_any(dots: &mut [f32], norms: &[f32], start: u32, metric: Metric, best: &mut Best<'_>) {
    let width = norms.len();
    let two = best.two;
    for (i, row) in dots.chunks_exact_mut(width).enumerate() {
        match (metric, two) {
            (Metric::L2, false) => {
                scan_row::<true>(row, norms, start, &mut best.best[i], &mut best.labels[i])
            }
            (Metric::InnerProduct, false) => {
                scan_row::<false>(row, norms, start, &mut best.best[i], &mut best.labels[i])
            }
            (Metric::L2, true) => scan_row_two::<true>(row, norms, start, best.row(i)),
            (Metric::InnerProduct, true) => scan_row_two::<false>(row, norms, start, best.row(i)),
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

/// The best two of one row: (score, id) pairs, compared by score, then by id.
struct Two<'a> {
    best: &'a mut f32,
    label: &'a mut u32,
    best2: &'a mut f32,
    label2: &'a mut u32,
}

impl Best<'_> {
    fn row(&mut self, i: usize) -> Two<'_> {
        Two {
            best: &mut self.best[i],
            label: &mut self.labels[i],
            best2: &mut self.best2[i],
            label2: &mut self.labels2[i],
        }
    }
}

impl Two<'_> {
    #[inline(always)]
    fn push(&mut self, s: f32, id: u32) {
        let before = |b: f32, b_id: u32| s < b || (s == b && id < b_id);
        if before(*self.best, *self.label) {
            *self.best2 = *self.best;
            *self.label2 = *self.label;
            *self.best = s;
            *self.label = id;
        } else if before(*self.best2, *self.label2) {
            *self.best2 = s;
            *self.label2 = id;
        }
    }
}

/// The best two of one row of the tile, merged into `two`.
///
/// Two plain scans are faster than one scan that keeps two per lane, which the
/// compiler does not vectorize. The second scan hides the winner by setting its dot
/// to minus infinity (a score of plus infinity), then puts it back.
#[inline(always)]
fn scan_row_two<const L2: bool>(row: &mut [f32], norms: &[f32], start: u32, mut two: Two<'_>) {
    let (mut best, mut id) = (f32::INFINITY, u32::MAX);
    scan_row::<L2>(row, norms, 0, &mut best, &mut id);
    if id == u32::MAX {
        return;
    }
    two.push(best, start + id);
    let kept = std::mem::replace(&mut row[id as usize], f32::NEG_INFINITY);
    let (mut next, mut next_id) = (f32::INFINITY, u32::MAX);
    scan_row::<L2>(row, norms, 0, &mut next, &mut next_id);
    row[id as usize] = kept;
    if next_id != u32::MAX {
        two.push(next, start + next_id);
    }
}

/// `dots[m x width] = block[m x d] * tile[width x d]^T`, all row major.
#[cfg(not(emeans_blas))]
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

#[cfg(emeans_blas)]
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

/// Same as above with a system BLAS: Accelerate on macOS (uses the matrix unit of Apple
/// chips), or MKL with the `mkl` feature. Each call runs on one thread, rayon already
/// uses every core.
#[cfg(emeans_blas)]
fn dot_tile(m: usize, d: usize, width: usize, block: &[f32], tile: &[f32], dots: &mut [f32]) {
    single_threaded_blas();
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

#[cfg(feature = "mkl")]
fn single_threaded_blas() {
    unsafe extern "C" {
        fn MKL_Set_Threading_Layer(layer: i32) -> i32;
    }
    const MKL_THREADING_SEQUENTIAL: i32 = 1;
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: plain call into MKL, made once before the first multiply.
    ONCE.call_once(|| unsafe {
        MKL_Set_Threading_Layer(MKL_THREADING_SEQUENTIAL);
    });
}

// Accelerate decides its own threads. Small calls like ours stay on the calling thread.
#[cfg(all(emeans_blas, not(feature = "mkl")))]
fn single_threaded_blas() {}

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
    fn second_best_matches_brute_force() {
        let mut rng = Rng::new(5);
        // A tile of more than one centroid lane, plus a tail.
        let (n, k, d) = (500, 2100, 5);
        let x: Vec<f32> = (0..n * d).map(|_| rng.next_f64() as f32).collect();
        let c: Vec<f32> = (0..k * d).map(|_| rng.next_f64() as f32).collect();
        let norms = squared_norms(&c, d);
        let (mut d1, mut l1) = (vec![0.0; n], vec![0; n]);
        let (mut d2, mut l2) = (vec![0.0; n], vec![0; n]);
        nearest_two(
            Rows::Dense(&x),
            d,
            &c,
            &norms,
            Metric::L2,
            &mut d1,
            &mut l1,
            &mut d2,
            &mut l2,
        );
        let mut same = 0;
        for (i, row) in x.chunks_exact(d).enumerate() {
            let mut all: Vec<(f32, u32)> = c
                .chunks_exact(d)
                .enumerate()
                .map(|(j, cent)| {
                    let dist = row.iter().zip(cent).map(|(a, b)| (a - b) * (a - b)).sum();
                    (dist, j as u32)
                })
                .collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            assert!(d1[i] <= d2[i]);
            assert!((d2[i] - all[1].0).abs() < 1e-4);
            same += (l1[i] == all[0].1 && l2[i] == all[1].1) as usize;
        }
        assert!(same >= n - 5, "{same} of {n} pairs agree");
    }

    #[test]
    fn second_best_is_empty_with_one_centroid() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let c = [0.0f32, 0.0];
        let norms = squared_norms(&c, 2);
        let (mut d1, mut l1) = (vec![0.0; 2], vec![9; 2]);
        let (mut d2, mut l2) = (vec![0.0; 2], vec![9; 2]);
        nearest_two(
            Rows::Dense(&x),
            2,
            &c,
            &norms,
            Metric::L2,
            &mut d1,
            &mut l1,
            &mut d2,
            &mut l2,
        );
        assert_eq!(l1, [0, 0]);
        assert_eq!(l2, [u32::MAX; 2]);
        assert!(d2.iter().all(|v| v.is_infinite()));
    }

    #[test]
    fn ties_keep_the_lower_id_first() {
        // Four equal centroids: the best two are ids 0 and 1, in that order.
        let x = [1.0f32; 3];
        let c = [0.0f32; 12];
        let norms = squared_norms(&c, 3);
        let (mut d1, mut l1) = (vec![0.0; 1], vec![9; 1]);
        let (mut d2, mut l2) = (vec![0.0; 1], vec![9; 1]);
        nearest_two(
            Rows::Dense(&x),
            3,
            &c,
            &norms,
            Metric::InnerProduct,
            &mut d1,
            &mut l1,
            &mut d2,
            &mut l2,
        );
        assert_eq!((l1[0], l2[0]), (0, 1));
        assert_eq!(d1[0], d2[0]);
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
