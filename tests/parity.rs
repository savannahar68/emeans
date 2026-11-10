//! Compare with faiss's Python NumpyEmeans on the same rows, order and starting centroids.
//! The files in tests/data come from scripts/make_parity_data.py.

use std::path::PathBuf;

use emeans::Emeans;

fn read(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn f32s(name: &str) -> Vec<f32> {
    read(name)
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn u32s(name: &str) -> Vec<u32> {
    read(name)
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn i32s(name: &str) -> Vec<i32> {
    read(name)
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn check(name: &str, nepochs: f64, spherical: bool) {
    let (d, k) = (12, 20);
    let x = f32s(&format!("{name}_x.f32"));
    let init = f32s(&format!("{name}_init.f32"));
    let order = u32s(&format!("{name}_order.u32"));
    let expected = f32s(&format!("{name}_centroids.f32"));
    let expected_labels = i32s(&format!("{name}_labels.i32"));
    let n = x.len() / d;

    let em = Emeans::new(d, k)
        .nepochs(nepochs)
        .spherical(spherical)
        .unused_threshold(0.0);
    let mut trainer = em.begin(n, &init).unwrap();
    let recipe = trainer.recipe().clone();
    let mut order = order.iter().map(|&i| i as usize);
    let mut labels = vec![-1i64; n];
    let mut batch = Vec::new();
    let mut batch_labels = Vec::new();
    for &rows in &recipe.epoch_rows {
        let ids: Vec<usize> = order.by_ref().take(rows).collect();
        for chunk in ids.chunks(recipe.batch_size) {
            batch.clear();
            for &i in chunk {
                batch.extend_from_slice(&x[i * d..(i + 1) * d]);
            }
            batch_labels.resize(chunk.len(), 0);
            trainer
                .accumulate_labeled(&batch, &mut batch_labels)
                .unwrap();
            for (&i, &l) in chunk.iter().zip(&batch_labels) {
                labels[i] = l;
            }
            trainer.update().unwrap();
        }
        trainer.end_epoch().unwrap();
    }
    let model = trainer.finalize().unwrap();

    let worst = model
        .centroids()
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "{name}: largest centroid difference {worst}");
    let same = labels
        .iter()
        .zip(&expected_labels)
        .filter(|(a, b)| **a == **b as i64)
        .count();
    assert_eq!(same, n, "{name}: {same} of {n} last pass labels agree");
}

#[test]
fn same_centroids_as_faiss_l2() {
    check("l2", 2.5, false);
}

#[test]
fn same_centroids_as_faiss_spherical() {
    check("spherical", 3.0, true);
}
