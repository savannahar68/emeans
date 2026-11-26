# emeans

Rust port of [e-means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means) from faiss, with Python bindings. e-means is a k-means replacement from Meta FAIR that reaches a lower error in about a tenth of the passes ([paper](https://arxiv.org/abs/2611.09501)).

This ports the CPU version (`NumpyEmeans`), up to faiss commit [`1bafdcc`](https://github.com/facebookresearch/faiss/commit/1bafdcc965) (2026-11-25). With revival off, results match the Python code: same input, order and starting centroids give the same centroids within 1e-4 (`tests/parity.rs`).

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

5 passes on random data, time to train. faiss is `NumpyEmeans` from faiss-cpu (1.15 on Intel, 1.16 on ARM). The final error is the same for all.

Linux, Intel CPU, 8 cores (Xeon on Google Cloud):

| n, d, k | faiss (Python) | emeans (Python) |
|---|---|---|
| 200k, 64, 500 | 1.38 s | 0.27 s |
| 1M, 128, 1024 | 16.77 s | 4.24 s |
| 1M, 32, 4096 | 6.94 s | 4.11 s |
| 500k, 768, 256 | 24.87 s | 4.08 s |

Linux, ARM CPU, 8 cores (Google Axion on Google Cloud):

| n, d, k | faiss (Python) | emeans (Python) |
|---|---|---|
| 200k, 64, 500 | 0.87 s | 0.21 s |
| 1M, 128, 1024 | 9.68 s | 3.22 s |
| 1M, 32, 4096 | 6.48 s | 3.11 s |
| 500k, 768, 256 | 17.80 s | 2.93 s |

Mac, Apple M4, 10 cores:

| n, d, k | faiss (Python) | emeans (Python) | emeans + `accelerate` (Rust) |
|---|---|---|---|
| 200k, 64, 500 | 0.70 s | 0.17 s | 0.13 s |
| 1M, 128, 1024 | 7.01 s | 3.06 s | 1.50 s |
| 1M, 32, 4096 | 10.05 s | 3.04 s | 1.99 s |
| 500k, 768, 256 | 11.42 s | 3.08 s | 1.90 s |

Other Rust crates on the same Intel machine, 5 iterations, time / error / peak memory:

| n, d, k | emeans | [fastkmeans-rs](https://crates.io/crates/fastkmeans-rs) | [kmeans_uni](https://crates.io/crates/kmeans_uni) |
|---|---|---|---|
| 200k, 64, 500 | 0.30 s / 4.502 / 63 MB | 0.56 s / 4.533 / 490 MB | 7.75 s / 4.533 / 59 MB |
| 1M, 128, 1024 | 4.52 s / 9.719 / 539 MB | 5.78 s / 9.751 / 2.68 GB | 128.8 s / 9.751 / 529 MB |

Run it yourself: `cargo run --release --example bench -- 1000000 128 1024`.

These numbers are from 0.1.2. On a 16 core AMD machine (Azure), 0.2.0 trains within 20% of 0.1.2 either way, and reaches a lower error thanks to the new revival.

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
emeans = "0.2"
```

```rust
use emeans::Emeans;

let model = Emeans::new(d, k).nepochs(5.0).train(&x)?; // x: n * d values, row major
let (distances, nearest) = model.assign(&x_new)?;
```

For data bigger than memory, use `Emeans::begin` and feed batches yourself (see the docs).

## Cluster revival

Since 0.2.0, revival follows faiss main ([#5915](https://github.com/facebookresearch/faiss/pull/5915)). Each cluster tracks its removal cost, how much the error would grow without it (the gap between each row's nearest and second nearest centroid), and its long axis. Every 5 updates, the clusters that cost less than 0.3 times the median move to split the clusters with the longest axes, and the two halves go to the means of a Gaussian cut along that axis. Revival keeps going in the last pass, and `train` then fixes the labels of rows whose cluster moved after they were assigned (`closing_handover`, on by default).

Two changes on top of faiss main, since 0.2.1. A cluster does not move in the same round as the cluster that would take its rows, so two centroids sharing a blob cannot both leave it. A cluster that got no row in a whole pass counts as costing nothing. Without these, on tight blobs of very different sizes, up to half of the clusters ended empty (faiss main does the same).

On clustered test data, this gives 2-9% lower error than the old rule, the same gain as faiss main.

`unused_threshold(0.1)` brings back the old mass based rule, `unused_threshold(0.0)` turns revival off.

## Differences from faiss

- The random generator is not numpy's, so the same seed gives a different (equally good) run.
- Batch sums are added in f64.
- faiss-cpu 1.16 on pip predates the new revival. Pass `unused_threshold=0.1` to compare against it.
- Cost based revival keeps a cluster and the one that takes its rows from moving together, and treats a cluster with no row in a whole pass as dead (see above).
- GPU kernels, multi GPU classes and 8/16 bit formats are not ported.

## License

MIT, like faiss. If you use e-means, please cite the [paper](https://arxiv.org/abs/2611.09501).
