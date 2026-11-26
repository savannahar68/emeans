use emeans::{BatchLaw, Emeans, Error, Model};

fn blobs(n: usize, d: usize, centers: usize, seed: u64) -> Vec<f32> {
    // Small deterministic generator so the tests need no extra crate.
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    let means: Vec<f32> = (0..centers * d).map(|_| next() * 20.0).collect();
    let mut x = Vec::with_capacity(n * d);
    for i in 0..n {
        let c = i % centers;
        for j in 0..d {
            x.push(means[c * d + j] + next() - 0.5);
        }
    }
    x
}

fn mse(model: &Model, x: &[f32]) -> f64 {
    let (dist, _) = model.assign(x).unwrap();
    dist.iter().map(|&v| v as f64).sum::<f64>() / dist.len() as f64
}

#[test]
fn closing_centroids_are_means_of_last_pass_labels() {
    let (n, d, k) = (3000, 8, 10);
    let x = blobs(n, d, 10, 1);
    let model = Emeans::new(d, k).train(&x).unwrap();
    let labels = model.labels().unwrap();
    for j in 0..k {
        let rows: Vec<&[f32]> = x
            .chunks_exact(d)
            .zip(labels)
            .filter(|(_, l)| **l == j as i64)
            .map(|(r, _)| r)
            .collect();
        if rows.is_empty() {
            continue;
        }
        for c in 0..d {
            let mean = rows.iter().map(|r| r[c] as f64).sum::<f64>() / rows.len() as f64;
            assert!((model.centroid(j)[c] as f64 - mean).abs() < 1e-4);
        }
    }
}

#[test]
fn finds_well_separated_clusters() {
    // With random starting points and k equal to the number of blobs, two centroids
    // often share a blob (faiss gets stuck the same way). With 4 centroids per blob,
    // every blob gets some.
    let (n, d, k) = (5000, 32, 32);
    let x = blobs(n, d, 8, 2);
    let model = Emeans::new(d, k).train(&x).unwrap();
    // Each point is uniform in a unit cube around its center: one centroid per blob
    // gives an MSE of d / 12, more centroids do better.
    let one_per_blob = d as f64 / 12.0;
    assert!(mse(&model, &x) < one_per_blob, "mse {}", mse(&model, &x));
}

#[test]
fn same_seed_same_result() {
    let (n, d, k) = (4000, 16, 50);
    let x = blobs(n, d, 30, 3);
    let a = Emeans::new(d, k).seed(9).train(&x).unwrap();
    let b = Emeans::new(d, k).seed(9).train(&x).unwrap();
    assert_eq!(a.centroids(), b.centroids());
    assert_eq!(a.labels(), b.labels());
}

#[cfg(feature = "parallel")]
#[test]
fn thread_count_does_not_change_the_result() {
    let (n, d, k) = (20_000, 24, 100);
    let x = blobs(n, d, 60, 4);
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| Emeans::new(d, k).train(&x).unwrap())
    };
    let one = run(1);
    let many = run(8);
    assert_eq!(one.centroids(), many.centroids());
    assert_eq!(one.labels(), many.labels());
}

#[cfg(feature = "parallel")]
#[test]
fn thread_count_does_not_change_train_from() {
    // Odd width and odd thread counts so rows and tiles split unevenly.
    let (n, d, k) = (30_000, 100, 128);
    let x = blobs(n, d, 90, 5);
    let init = x[..k * d].to_vec();
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                Emeans::new(d, k)
                    .nepochs(5.0)
                    .seed(11)
                    .train_from(&x, &init)
                    .unwrap()
            })
    };
    let one = run(1);
    for threads in [3, 7, 32] {
        let many = run(threads);
        assert_eq!(one.centroids(), many.centroids(), "threads {threads}");
        assert_eq!(one.labels(), many.labels(), "threads {threads}");
    }
}

#[test]
fn dead_clusters_are_revived() {
    // Start every centroid on top of one point: most clusters get nothing at first.
    let (n, d, k) = (4000, 8, 40);
    let x = blobs(n, d, 40, 5);
    let init: Vec<f32> = (0..k).flat_map(|_| x[..d].iter().copied()).collect();
    let model = Emeans::new(d, k).train_from(&x, &init).unwrap();
    assert!(model.n_splits() > 0);
    let (_, labels) = model.assign(&x).unwrap();
    let mut used = vec![false; k];
    labels.iter().for_each(|&l| used[l as usize] = true);
    assert!(used.iter().filter(|u| **u).count() > k * 3 / 4);
}

#[test]
fn max_points_per_centroid_trains_on_a_subset() {
    let (n, d, k) = (5000, 4, 10);
    let x = blobs(n, d, 10, 6);
    let model = Emeans::new(d, k)
        .max_points_per_centroid(100)
        .train(&x)
        .unwrap();
    let labels = model.labels().unwrap();
    assert_eq!(labels.len(), n);
    assert_eq!(labels.iter().filter(|&&l| l >= 0).count(), 1000);
    assert_eq!(model.recipe().unwrap().epoch_rows.last(), Some(&1000));
}

#[test]
fn spherical_centroids_have_unit_norm() {
    let (n, d, k) = (3000, 16, 12);
    let mut x = blobs(n, d, 12, 7);
    for row in x.chunks_exact_mut(d) {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        row.iter_mut().for_each(|v| *v /= norm);
    }
    let model = Emeans::new(d, k).spherical(true).train(&x).unwrap();
    for j in 0..k {
        let norm = model.centroid(j).iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
    let (sim, _) = model.assign(&x).unwrap();
    assert!(sim.iter().all(|&s| s <= 1.0 + 1e-5));
}

#[test]
fn recipe_follows_the_laws() {
    let em = Emeans::new(64, 500);
    let r = em.recipe(200_000).unwrap();
    assert_eq!(r.batch_law, Some(BatchLaw::Throughput));
    assert_eq!(r.batch_size, 16384);
    assert_eq!(r.epoch_rows, vec![200_000; 5]);
    let r = Emeans::new(8, 500).recipe(200_000).unwrap();
    assert_eq!(r.batch_law, Some(BatchLaw::Quality));
    // Cost based revival by default, mass based only when asked or when k = 1.
    assert_eq!(r.unused_threshold, None);
    let r = Emeans::new(8, 500)
        .unused_threshold(0.2)
        .recipe(1000)
        .unwrap();
    assert_eq!(r.unused_threshold, Some(0.2));
    let one = |d| Emeans::new(d, 1).recipe(1000).unwrap().unused_threshold;
    assert_eq!(one(8), Some(0.1));
    assert_eq!(one(2), Some(0.0));
}

#[test]
fn batch_loop_matches_train_shape() {
    let (n, d, k) = (1000, 6, 7);
    let x = blobs(n, d, 7, 8);
    let em = Emeans::new(d, k).nepochs(2.0);
    let mut t = em.begin(n, &x[..k * d]).unwrap();
    let r = t.recipe().clone();
    for &rows in &r.epoch_rows {
        for start in (0..rows).step_by(r.batch_size) {
            let end = (start + r.batch_size).min(rows);
            // Give each batch in two pieces.
            let mid = (start + end) / 2;
            t.accumulate(&x[start * d..mid * d]).unwrap();
            t.accumulate(&x[mid * d..end * d]).unwrap();
            t.update().unwrap();
        }
        t.end_epoch().unwrap();
    }
    let model = t.finalize().unwrap();
    assert_eq!(model.epoch_stats().len(), 2);
    assert!(model.labels().is_none());
}

#[test]
fn wrong_inputs_are_errors() {
    let x = vec![0.0f32; 100];
    assert!(matches!(
        Emeans::new(3, 2).train(&x),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        Emeans::new(10, 20).train(&x),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        Emeans::new(10, 2).nepochs(0.0).train(&x),
        Err(Error::InvalidInput(_))
    ));

    let em = Emeans::new(10, 2);
    let mut t = em.begin(10, &x[..20]).unwrap();
    assert!(matches!(t.update(), Err(Error::WrongOrder(_))));
    assert!(matches!(t.end_epoch(), Err(Error::WrongOrder(_))));
    let too_many = vec![0.0f32; (t.batch_rows() + 1) * 10];
    assert!(matches!(t.accumulate(&too_many), Err(Error::WrongOrder(_))));
}

/// Gaussian rows around `centers` random centers, like the faiss tests.
fn gaussian_blobs(n: usize, d: usize, centers: usize, spread: f32, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut uniform = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    };
    let mut normal = move || {
        let (u, v) = (uniform(), uniform());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    };
    let means: Vec<f32> = (0..centers * d).map(|_| normal() * spread).collect();
    let mut x = Vec::with_capacity(n * d);
    for i in 0..n {
        let c = (i * 7919) % centers;
        for j in 0..d {
            x.push(means[c * d + j] + normal());
        }
    }
    x
}

#[test]
fn cost_based_revival_keeps_a_small_far_blob() {
    // 4000 rows near 0 and 40 rows near 20. One centroid starts far away and gets no
    // rows. Mass based revival moves the small blob's centroid, cost based keeps it.
    let d = 4;
    let mut x = gaussian_blobs(4000, d, 1, 0.0, 1);
    x.extend(gaussian_blobs(40, d, 1, 0.0, 2).iter().map(|v| v + 20.0));
    let mut init = x[..2 * d].to_vec();
    init.extend_from_slice(&x[x.len() - d..]);
    init.extend([-50.0; 4]);
    for (threshold, far_kept) in [(None, true), (Some(0.1), false)] {
        let mut em = Emeans::new(d, 4).nepochs(3.0);
        if let Some(t) = threshold {
            em = em.unused_threshold(t);
        }
        let model = em.train_from(&x, &init).unwrap();
        let near_far_blob =
            (0..4).any(|j| model.centroid(j).iter().all(|v| (v - 20.0).abs() < 1.0));
        assert_eq!(near_far_blob, far_kept, "threshold {threshold:?}");
        // The dead centroid moved.
        assert!(model.centroids().iter().all(|v| v.abs() < 30.0));
    }
}

/// Tight blobs of very different sizes: blob b holds 2400 / (b + 1) rows.
fn skewed_blobs(d: usize, blobs: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut uniform = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    };
    let mut x = Vec::new();
    for b in 0..blobs {
        let center: Vec<f32> = (0..d).map(|_| (uniform() * 10.0).floor() * 10.0).collect();
        for _ in 0..2400 / (b + 1) {
            x.extend(center.iter().map(|c| c + uniform() - 0.5));
        }
    }
    x
}

#[test]
fn cost_based_revival_leaves_no_empty_cluster_on_skewed_blobs() {
    // 0.2.0 left up to a third of the clusters empty here: twins on a small blob both
    // moved away, and clusters that lost their rows were never revived.
    let (d, k) = (8, 64);
    for seed in 1..=4 {
        let x = skewed_blobs(d, 24, seed);
        let n = x.len() / d;
        let init: Vec<f32> = (0..k)
            .flat_map(|i| {
                let r = (i * 7919) % n;
                x[r * d..(r + 1) * d].to_vec()
            })
            .collect();
        let model = Emeans::new(d, k).seed(seed).train_from(&x, &init).unwrap();
        let mut counts = vec![0; k];
        for &l in model.labels().unwrap() {
            counts[l as usize] += 1;
        }
        let empty = counts.iter().filter(|&&c| c == 0).count();
        assert_eq!(empty, 0, "seed {seed}: {counts:?}");
        // Every row ends near its own blob center, not halfway to another blob.
        let (dist, _) = model.assign(&x).unwrap();
        let mean = dist.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        assert!(mean < 1.0, "seed {seed}: mean error {mean}");
    }
}

/// k starting rows taken from the first `blobs` blobs of `gaussian_blobs`, so most
/// blobs start with no centroid.
fn crowded_start(x: &[f32], d: usize, k: usize, centers: usize, blobs: usize) -> Vec<f32> {
    (0..x.len() / d)
        .filter(|i| (i * 7919) % centers < blobs)
        .take(k)
        .flat_map(|i| x[i * d..(i + 1) * d].to_vec())
        .collect()
}

#[test]
fn revival_continues_in_the_last_pass() {
    // The first pass gives the starting centroids back exactly, so no split pays.
    // The last pass spreads rows around 30, and that donor's split pays.
    let init = [0.0, 1.0, 10.0, 30.0];
    let mut t = Emeans::new(1, 4)
        .nepochs(1.6)
        .batch_size(10)
        .alpha0(0.5)
        .begin(100, &init)
        .unwrap();
    let r = t.recipe().clone();
    for (pass, pattern) in [&init[..], &[27.0, 30.0, 33.0]].into_iter().enumerate() {
        assert!(pass < r.epoch_rows.len());
        while t.batch_rows() > 0 {
            let rows: Vec<f32> = pattern
                .iter()
                .copied()
                .cycle()
                .take(t.batch_rows())
                .collect();
            t.accumulate(&rows).unwrap();
            t.update().unwrap();
        }
        t.end_epoch().unwrap();
    }
    let model = t.finalize().unwrap();
    assert_eq!(model.epoch_stats()[0].n_splits, 0);
    assert!(model.epoch_stats()[1].n_splits > 0);
}

#[test]
fn handover_centroids_are_means_of_final_labels() {
    // One pass from a bad start, so many clusters move while the labels are taken.
    let (n, d, k) = (20_000, 8, 64);
    let x = gaussian_blobs(n, d, 64, 4.0, 4);
    let init = crowded_start(&x, d, k, 64, 8);
    let em = Emeans::new(d, k).nepochs(1.0).batch_size(500);
    let model = em.train_from(&x, &init).unwrap();
    assert!(model.n_splits() > 0);
    let labels = model.labels().unwrap();
    let mut sums = vec![0.0f64; k * d];
    let mut counts = vec![0usize; k];
    for (row, &l) in x.chunks_exact(d).zip(labels) {
        let l = l as usize;
        counts[l] += 1;
        for (s, &v) in sums[l * d..(l + 1) * d].iter_mut().zip(row) {
            *s += v as f64;
        }
    }
    for j in (0..k).filter(|&j| counts[j] > 0) {
        for c in 0..d {
            let mean = sums[j * d + c] / counts[j] as f64;
            assert!((model.centroid(j)[c] as f64 - mean).abs() < 1e-4);
        }
    }
    // From a start this bad, faiss main also ends about 2% worse with the handover
    // than without it. Check it stays near that.
    let (mut on, mut off) = (0.0, 0.0);
    for seed in 1..=4 {
        let x = gaussian_blobs(n, d, 64, 4.0, seed);
        let init = crowded_start(&x, d, k, 64, 8);
        let em = Emeans::new(d, k).nepochs(1.0).batch_size(500).seed(seed);
        on += mse(&em.train_from(&x, &init).unwrap(), &x);
        let em = em.closing_handover(false);
        off += mse(&em.train_from(&x, &init).unwrap(), &x);
    }
    assert!(on <= off * 1.05, "with {on}, without {off}");
}

#[test]
fn spherical_cost_based_revival_runs() {
    let (n, d, k) = (5000, 16, 40);
    let mut x = gaussian_blobs(n, d, 20, 3.0, 5);
    for row in x.chunks_exact_mut(d) {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        row.iter_mut().for_each(|v| *v /= norm);
    }
    let init: Vec<f32> = (0..k).flat_map(|_| x[..d].iter().copied()).collect();
    let model = Emeans::new(d, k)
        .spherical(true)
        .train_from(&x, &init)
        .unwrap();
    assert!(model.n_splits() > 0);
    for j in 0..k {
        let norm = model.centroid(j).iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}

#[test]
fn duplicate_rows_stay_in_the_lowest_cluster() {
    // Every centroid starts on the same row. No split can gain anything, so nothing
    // moves, ties go to the lower index, and the labels match a fresh search.
    let d = 4;
    let x: Vec<f32> = [1.0f32, 2.0, 3.0, 4.0].repeat(3);
    for seed in 0..4 {
        let model = Emeans::new(d, 2).seed(seed).train(&x).unwrap();
        assert_eq!(model.labels().unwrap(), &[0, 0, 0]);
        assert_eq!(model.assign(&x).unwrap().1, vec![0, 0, 0]);
    }
}
