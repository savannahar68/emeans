"""Fast e-means clustering, written in Rust.

The class follows faiss's ``NumpyEmeans`` (faiss/contrib/e_means), so code
written for it works with a changed import:

    from emeans import Emeans

    em = Emeans(d, k)
    centroids, labels = em.train(x)
    distances, nearest = em.assign(x_new)
"""

import numpy as np

from ._emeans import Native

__all__ = ["Emeans", "NumpyEmeans"]


def _f32(x, name):
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("%s must be a 2D array, got shape %s" % (name, x.shape))
    return x


class Emeans:
    """e-means on the CPU. Only ``d`` and ``k`` are required.

    ``nepochs`` is the number of passes over the data (can be fractional).
    The batch size and the starting step size come from fitted laws unless
    ``batch_size`` or ``alpha0`` are given. ``batch_law`` is ``"auto"``,
    ``"throughput"`` or ``"quality"``. ``unused_threshold=None`` means 0.1,
    or 0 (no revival) when d <= 2. With ``spherical=True``, the input rows
    must already have norm 1 and ``assign`` returns similarities.

    Training runs without the GIL, on all cores. Set ``RAYON_NUM_THREADS``
    before the first call to use fewer.
    """

    def __init__(
        self,
        d,
        k,
        nepochs=5,
        batch_size=None,
        alpha0=None,
        batch_law="auto",
        seed=1234,
        verbose=False,
        assignment_chunk_size=None,
        capture_assignments=True,
        spherical=False,
        *,
        unused_threshold=None,
        max_points_per_centroid=None,
    ):
        self.d = int(d)
        self.k = int(k)
        self.spherical = bool(spherical)
        self._native = Native(
            self.d,
            self.k,
            float(nepochs),
            None if batch_size is None else int(batch_size),
            None if alpha0 is None else float(alpha0),
            batch_law,
            int(seed),
            bool(verbose),
            65536 if assignment_chunk_size is None else int(assignment_chunk_size),
            bool(capture_assignments),
            self.spherical,
            None if unused_threshold is None else float(unused_threshold),
            None if max_points_per_centroid is None else int(max_points_per_centroid),
        )
        self.centroids = None
        self.assignments = None

    def train(self, x, init_centroids=None):
        """Train on ``x`` (n, d). Return the centroids (k, d) and each row's last pass label."""
        x = _f32(x, "x")
        init = None if init_centroids is None else _f32(init_centroids, "init_centroids")
        self._native.train(x, init)
        self.centroids = self._native.centroids()
        self.assignments = self._native.labels()
        return self.centroids, self.assignments

    def assign(self, x):
        """Squared distance and id of each row's nearest centroid (similarity when spherical)."""
        return self._native.assign(_f32(x, "x"))

    # Batch loop, for data that does not fit in memory. Same calls as faiss:
    #   em.begin(n, init); em.update(em.accumulate(batch)); em.end_epoch(); em.finalize()

    def begin(self, global_n, init_centroids):
        self._native.begin(int(global_n), _f32(init_centroids, "init_centroids"))

    def accumulate(self, x):
        """Add rows to the current batch. Returns their nearest centroid ids."""
        return self._native.accumulate(_f32(x, "batch"))

    def update(self, stats=None):
        """Apply the current batch. ``stats`` is accepted and ignored, for faiss compatibility."""
        self._native.update()

    def end_epoch(self):
        self._native.end_epoch()

    def finalize(self):
        self._native.finalize()
        self.centroids = self._native.centroids()
        return self.centroids

    def current_centroids(self):
        return self._native.current_centroids()

    @property
    def recipe(self):
        return self._native.recipe()

    @property
    def iteration_stats(self):
        return self._native.iteration_stats()

    @property
    def obj(self):
        stats = self.iteration_stats
        return np.array([s["obj"] for s in stats], dtype=np.float64) if stats else None

    @property
    def n_splits(self):
        return self._native.n_splits()


# Same name as the faiss class, to ease switching.
NumpyEmeans = Emeans
