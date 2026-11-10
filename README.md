# emeans

Rust port of [e-means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means) from faiss, with Python bindings. e-means is a k-means replacement from Meta FAIR that reaches a lower error in about a tenth of the passes ([paper](https://arxiv.org/abs/2611.09501)).

This ports the CPU version (`NumpyEmeans`). Results match the Python code: same input, order and starting centroids give the same centroids within 1e-4 (`tests/parity.rs`).

## What is faster

Almost all the time goes into finding the nearest centroid of each row. Here it is:

- a SIMD matrix multiply (`matrixmultiply`, AVX-512 / AVX2 / NEON picked at run time),
- a vectorized scan for the smallest distance, built for AVX-512, AVX2 and plain, picked at run time from the CPU,
- spread over all cores with `rayon`, with the same result for any number of threads.

No feature flag is needed for SIMD. Two optional features swap the matrix multiply:

- `mkl`: Intel MKL, 10-15% faster on Intel. Needs `libmkl_rt` (set `MKL_LIB_DIR` if it is not found).
- `accelerate` (macOS only): Apple's matrix unit, about 2x faster. Apple's library can take several hundred MB more memory on wide vectors (d=768), so it is not on by default.

Memory: on top of your data, training needs about 20-30 MB, plus 8 bytes per row for the labels.

## Numbers

5 passes on random data, time to train. faiss is `NumpyEmeans` from faiss-cpu 1.15. The final error is the same for all.

Intel Xeon 8481C, 8 vCPU (GCP c3-standard-8):

| n, d, k | faiss (Python) | emeans (Python) |
|---|---|---|
| 200k, 64, 500 | 1.38 s | 0.27 s |
| 1M, 128, 1024 | 16.77 s | 4.24 s |
| 1M, 32, 4096 | 6.94 s | 4.11 s |
| 500k, 768, 256 | 24.87 s | 4.08 s |

Apple M4, 10 cores:

| n, d, k | faiss (Python) | emeans (Python) | emeans + `accelerate` (Rust) |
|---|---|---|---|
| 200k, 64, 500 | 0.70 s | 0.17 s | 0.13 s |
| 1M, 128, 1024 | 7.01 s | 3.06 s | 1.50 s |
| 1M, 32, 4096 | 10.05 s | 3.04 s | 1.99 s |
| 500k, 768, 256 | 11.42 s | 3.08 s | 1.90 s |

Other Rust crates on the same Xeon, 5 iterations, time / error / peak memory:

| n, d, k | emeans | [fastkmeans-rs](https://crates.io/crates/fastkmeans-rs) | [kmeans_uni](https://crates.io/crates/kmeans_uni) |
|---|---|---|---|
| 200k, 64, 500 | 0.30 s / 4.502 / 63 MB | 0.56 s / 4.533 / 490 MB | 7.75 s / 4.533 / 59 MB |
| 1M, 128, 1024 | 4.52 s / 9.719 / 539 MB | 5.78 s / 9.751 / 2.68 GB | 128.8 s / 9.751 / 529 MB |

Run it yourself: `cargo run --release --example bench -- 1000000 128 1024`.

## Python

```
pip install emeans
```

Same class as faiss `NumpyEmeans`, so most code only needs a new import:

```python
from emeans import Emeans

em = Emeans(d, k)                     # nepochs=5, seed=1234, spherical=False, ...
centroids, labels = em.train(x)       # x: (n, d) array
distances, nearest = em.assign(x_new)
```

Training releases the GIL. For data bigger than memory, the faiss loop works as is: `em.begin(n, init)`, then `em.update(em.accumulate(batch))`, `em.end_epoch()`, `em.finalize()`.

## Rust

```toml
[dependencies]
emeans = "0.1"
```

```rust
use emeans::Emeans;

let model = Emeans::new(d, k).nepochs(5.0).train(&x)?; // x: n * d values, row major
let (distances, nearest) = model.assign(&x_new)?;
```

For data bigger than memory, use `Emeans::begin` and feed batches yourself (see the docs).

## Differences from faiss

- The random generator is not numpy's, so the same seed gives a different (equally good) run.
- Batch sums are added in f64.
- GPU kernels, multi GPU classes and 8/16 bit formats are not ported.

## License

MIT, like faiss. If you use e-means, please cite the [paper](https://arxiv.org/abs/2611.09501).
