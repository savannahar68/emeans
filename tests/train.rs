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
    assert_eq!(r.unused_threshold, 0.1);
    assert_eq!(
        Emeans::new(2, 5).recipe(1000).unwrap().unused_threshold,
        0.0
    );
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
