"""Write the files used by tests/parity.rs.

Runs faiss's NumpyEmeans with a fixed row order and fixed starting centroids, and
saves the inputs and the result as raw little endian files. Revival is off because
it draws from numpy's random generator, which the Rust code does not copy.

Usage: python scripts/make_parity_data.py <path to faiss/contrib> tests/data
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[1])
from e_means import NumpyEmeans  # noqa: E402

out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)


def run(name, n, d, k, nepochs, spherical):
    rng = np.random.default_rng(0)
    # Clustered data, so most labels are not near a tie.
    centers = rng.normal(size=(k, d)).astype("float32") * 4
    x = (centers[rng.integers(k, size=n)] + rng.normal(size=(n, d))).astype("float32")
    if spherical:
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    init = x[rng.choice(n, k, replace=False)].copy()

    em = NumpyEmeans(d, k, nepochs=nepochs, spherical=spherical, unused_threshold=0)
    em.begin(n, init)
    batch = em.recipe["batch_size"]
    orders = []
    labels = np.full(n, -1, dtype="int64")
    for epoch, rows in enumerate(em.recipe["epoch_rows"]):
        order = rng.permutation(n)[:rows]
        orders.append(order)
        for start in range(0, rows, batch):
            idx = order[start:start + batch]
            stats = em.accumulate(x[idx])
            if "labels" in stats:
                labels[idx] = stats["labels"]
            em.update(stats)
        em.end_epoch()
    centroids = em.finalize()

    x.tofile(out / f"{name}_x.f32")
    init.tofile(out / f"{name}_init.f32")
    np.concatenate(orders).astype("uint32").tofile(out / f"{name}_order.u32")
    centroids.tofile(out / f"{name}_centroids.f32")
    labels.astype("int32").tofile(out / f"{name}_labels.i32")
    print(name, em.recipe["batch_size"], em.recipe["alpha0"], em.recipe["epoch_rows"])


run("l2", 2000, 12, 20, 2.5, False)
run("spherical", 2000, 12, 20, 3, True)
