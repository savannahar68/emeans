# emeans

Rust port of [e-means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means), the k-means replacement from Meta FAIR that ships in faiss. It reaches a lower error than Lloyd's k-means in about a tenth of the passes over the data.

This crate ports the CPU version (`NumpyEmeans`). No Python, no BLAS install needed.

```toml
[dependencies]
emeans = "0.1"
```

```rust
use emeans::Emeans;

// x holds n vectors of dimension d, one after the other.
let model = Emeans::new(d, k).train(&x)?;

let centroids = model.centroids();          // k * d values
let labels = model.labels().unwrap();       // each row's cluster from the last pass
let (distances, nearest) = model.assign(&new_x)?;
```

Settings use a builder. The defaults are the same as in faiss:

```rust
let model = Emeans::new(d, k)
    .nepochs(3.0)               // passes over the data, default 5, can be 2.5
    .batch_law(BatchLaw::Quality)
    .seed(7)
    .spherical(true)            // cluster by direction, input must have norm 1
    .max_points_per_centroid(256)
    .train(&x)?;
```

## How it works

Each centroid is the ratio of two moving averages: the number of points it gets and the sum of those points. For every batch:

```
N_j <- (1 - alpha) * N_j + alpha * n_j
S_j <- (1 - alpha) * S_j + alpha * s_j
c_j  = S_j / N_j
```

The batch size and the starting `alpha` come from laws fitted in the paper, so you only pick the number of passes. `alpha` goes down along a cosine to a tenth of its start. Clusters that get almost no points are revived by splitting a busy one. At the end, each centroid is set to the exact mean of the points the last pass gave it.

## Data bigger than memory

Feed the batches yourself with `begin`. The rules are the same as in faiss: every row once per pass, in a new random order each pass, in batches of `recipe().batch_size` rows.

```rust
let mut t = Emeans::new(d, k).begin(n, &init_centroids)?;
let recipe = t.recipe().clone();
for &rows in &recipe.epoch_rows {
    for start in (0..rows).step_by(recipe.batch_size) {
        let batch = read_rows(start, (start + recipe.batch_size).min(rows)); // your loader
        t.accumulate(&batch)?;
        t.update()?;
    }
    t.end_epoch()?;
}
let model = t.finalize()?;
```

## Speed

The nearest centroid search is almost all of the time. It is done with a SIMD matrix multiply (`matrixmultiply`) on blocks of rows, spread over all cores with `rayon`. Results do not depend on the number of threads.

Training time on an Apple M4 (10 cores), random data, 5 passes, compared with faiss `NumpyEmeans` 1.13 on the same machine:

| n | d | k | faiss (Python) | emeans | emeans + `accelerate` |
|---|---|---|---|---|---|
| 200,000 | 64 | 500 | 0.69 s | 0.20 s | 0.13 s |
| 1,000,000 | 128 | 1,024 | 6.88 s | 3.10 s | 1.50 s |
| 1,000,000 | 32 | 4,096 | 9.59 s | 3.48 s | 1.99 s |
| 500,000 | 768 | 256 | 11.59 s | 2.94 s | 1.90 s |

The final error is the same in both. Run it yourself with `cargo run --release --example bench -- 1000000 128 1024`.

### Features

- `parallel` (on by default): use all cores. Turn it off with `default-features = false`.
- `accelerate` (macOS only): use Apple's Accelerate for the matrix multiply, which runs on the matrix unit of Apple chips.

Set the number of threads with `RAYON_NUM_THREADS`, or run inside your own rayon pool.

## Same results as faiss

`tests/parity.rs` runs the same rows, in the same order, from the same starting centroids as faiss and checks that the centroids match within 1e-4 and that every last pass label is the same. The test files come from `scripts/make_parity_data.py`.

Differences from faiss:

- The random generator is not numpy's, so with the same seed you get a different (but equally good) run.
- Batch sums are added in f64, faiss uses f32 for batches under 2^24 rows.
- Only the CPU version is ported. The GPU kernels, the multi GPU classes and the 8 and 16 bit formats are not.

## Paper

```bibtex
@misc{bessa2026emeans,
  title  = {{e-means}: outperforming {k-means} (in one-tenth the epochs)},
  author = {Bessa, Swann and Mazar{\'e}, Pierre-Emmanuel and Szilvasy, Gergely and
            Touvron, Hugo and Picard, David and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  year   = {2026},
  url    = {https://arxiv.org/abs/2611.09501}
}
```

## License

MIT, like faiss. See [LICENSE](LICENSE).
