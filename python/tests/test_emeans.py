from pathlib import Path

import numpy as np
import pytest

from emeans import Emeans

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


def blobs(n, d, centers, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.random((centers, d), dtype=np.float32) * 20
    return means[np.arange(n) % centers] + rng.random((n, d), dtype=np.float32) - 0.5


def test_train_and_assign():
    x = blobs(5000, 32, 8)
    em = Emeans(32, 32)
    centroids, labels = em.train(x)
    assert centroids.shape == (32, 32) and centroids.dtype == np.float32
    assert labels.shape == (5000,) and labels.dtype == np.int64
    dist, nearest = em.assign(x)
    assert dist.shape == (5000,) and nearest.shape == (5000,)
    assert dist.mean() < 32 / 12
    assert em.recipe["batch_size"] > 0
    assert len(em.obj) == 5


def test_float64_and_strided_input_are_converted():
    x = blobs(2000, 8, 4).astype(np.float64)
    a = Emeans(8, 4, seed=3).train(x)[0]
    b = Emeans(8, 4, seed=3).train(np.asfortranarray(x))[0]
    np.testing.assert_array_equal(a, b)


def test_max_points_per_centroid():
    x = blobs(5000, 4, 10)
    _, labels = Emeans(4, 10, max_points_per_centroid=100).train(x)
    assert (labels >= 0).sum() == 1000


def test_spherical():
    x = blobs(3000, 16, 12)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    em = Emeans(16, 12, spherical=True)
    centroids, _ = em.train(x)
    np.testing.assert_allclose(np.linalg.norm(centroids, axis=1), 1, atol=1e-5)
    sim, _ = em.assign(x)
    assert sim.max() <= 1 + 1e-5


def test_errors():
    with pytest.raises(ValueError):
        Emeans(8, 4).train(np.zeros((10, 7), dtype=np.float32))
    with pytest.raises(ValueError):
        Emeans(8, 4, batch_law="fast")
    with pytest.raises(RuntimeError):
        Emeans(8, 4).assign(np.zeros((1, 8), dtype=np.float32))


@pytest.mark.parametrize("name,nepochs,spherical", [("l2", 2.5, False), ("spherical", 3, True)])
def test_same_as_faiss(name, nepochs, spherical):
    # Files written by scripts/make_parity_data.py from faiss's NumpyEmeans.
    d, k = 12, 20
    x = np.fromfile(DATA / f"{name}_x.f32", dtype=np.float32).reshape(-1, d)
    init = np.fromfile(DATA / f"{name}_init.f32", dtype=np.float32).reshape(k, d)
    order = np.fromfile(DATA / f"{name}_order.u32", dtype=np.uint32).astype(np.int64)
    expected = np.fromfile(DATA / f"{name}_centroids.f32", dtype=np.float32).reshape(k, d)

    # The same loop as in the faiss README.
    em = Emeans(d, k, nepochs=nepochs, spherical=spherical, unused_threshold=0)
    em.begin(len(x), init)
    batch = em.recipe["batch_size"]
    pos = 0
    for rows in em.recipe["epoch_rows"]:
        ids = order[pos:pos + rows]
        pos += rows
        for start in range(0, rows, batch):
            em.update(em.accumulate(x[ids[start:start + batch]]))
        em.end_epoch()
    np.testing.assert_allclose(em.finalize(), expected, atol=1e-4)
