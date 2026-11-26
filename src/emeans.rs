use std::time::Instant;

use crate::error::{Result, invalid, wrong_order};
use crate::knn::{self, Metric, Rows};
use crate::recipe::{self, BatchLaw, Recipe};
use crate::rng::Rng;

// Independent random streams derived from the one seed.
const STREAM_INIT: u64 = 1;
const STREAM_SUBSET: u64 = 2;
const STREAM_SPLITS: u64 = 3;
const STREAM_EPOCH: u64 = 1 << 32;

fn stream(seed: u64, id: u64) -> Rng {
    Rng::new(seed ^ id.wrapping_mul(0xD1B5_4A32_D192_ED03))
}

/// Settings of an e-means run. Only `d` and `k` are required.
///
/// ```
/// use emeans::Emeans;
///
/// let (n, d, k) = (2_000, 8, 16);
/// let x: Vec<f32> = (0..n * d).map(|i| ((i * 7919) % 1000) as f32 / 1000.0).collect();
///
/// let model = Emeans::new(d, k).nepochs(3.0).seed(42).train(&x).unwrap();
/// assert_eq!(model.centroids().len(), k * d);
///
/// let (distances, labels) = model.assign(&x[..10 * d]).unwrap();
/// assert_eq!(labels.len(), 10);
/// # let _ = distances;
/// ```
#[derive(Clone, Debug)]
pub struct Emeans {
    d: usize,
    k: usize,
    nepochs: f64,
    batch_law: BatchLaw,
    batch_size: Option<usize>,
    alpha0: Option<f64>,
    seed: u64,
    unused_threshold: Option<f32>,
    max_points_per_centroid: Option<usize>,
    spherical: bool,
    capture_assignments: bool,
    closing_handover: bool,
    chunk_size: usize,
    verbose: bool,
}

impl Emeans {
    /// `d` is the dimension of the vectors and `k` the number of clusters.
    pub fn new(d: usize, k: usize) -> Self {
        Emeans {
            d,
            k,
            nepochs: 5.0,
            batch_law: BatchLaw::Auto,
            batch_size: None,
            alpha0: None,
            seed: 1234,
            unused_threshold: None,
            max_points_per_centroid: None,
            spherical: false,
            capture_assignments: true,
            closing_handover: true,
            chunk_size: 65536,
            verbose: false,
        }
    }

    /// Number of passes over the data, can be fractional. Default 5.
    pub fn nepochs(mut self, nepochs: f64) -> Self {
        self.nepochs = nepochs;
        self
    }

    /// Law used to pick the batch size. Default [`BatchLaw::Auto`].
    pub fn batch_law(mut self, law: BatchLaw) -> Self {
        self.batch_law = law;
        self
    }

    /// Fixed batch size instead of the batch law. The step size still follows the step law.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Fixed starting step size instead of the step law.
    pub fn alpha0(mut self, alpha0: f64) -> Self {
        self.alpha0 = Some(alpha0);
        self
    }

    /// Seed of the starting centroids, the shuffles and the revivals. Default 1234.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Switch to mass based revival: revive clusters whose mass falls below this fraction
    /// of the mean, as faiss did before. 0 turns revival off.
    ///
    /// Without this call, revival is cost based. Each cluster tracks how much the error
    /// would grow without it (the gap between each point's nearest and second nearest
    /// centroid) and its long axis. Every 5 updates, the clusters that cost less than
    /// 0.3 times the median move to split the clusters with the longest axes. A cluster
    /// with no row in a whole pass costs nothing.
    pub fn unused_threshold(mut self, threshold: f32) -> Self {
        self.unused_threshold = Some(threshold);
        self
    }

    /// Train on a random subset of `max * k` rows when there are more, like `faiss.Kmeans`.
    /// Rows left out get label -1.
    pub fn max_points_per_centroid(mut self, max: usize) -> Self {
        self.max_points_per_centroid = Some(max);
        self
    }

    /// Keep the centroids on the unit sphere and assign by inner product.
    /// The input vectors should already have norm 1.
    pub fn spherical(mut self, spherical: bool) -> Self {
        self.spherical = spherical;
        self
    }

    /// Keep each row's label from the last pass (8 bytes per row). Default true.
    pub fn capture_assignments(mut self, capture: bool) -> Self {
        self.capture_assignments = capture;
        self
    }

    /// With cost based revival, a cluster can move during the last pass after some rows
    /// were already given to it. When this is on (the default) and the labels are kept,
    /// `train` fixes those rows at the end: they go to their second nearest centroid, or
    /// are searched again if that one moved too. Each centroid is then the mean of its
    /// final rows.
    pub fn closing_handover(mut self, on: bool) -> Self {
        self.closing_handover = on;
        self
    }

    /// Rows searched at once. Only changes memory use. Default 65536.
    pub fn chunk_size(mut self, rows: usize) -> Self {
        self.chunk_size = rows;
        self
    }

    /// Print one line per pass on stderr.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    fn check(&self) -> Result<()> {
        if self.d == 0 || self.k == 0 {
            return invalid("d and k must be at least 1");
        }
        if self.k > u32::MAX as usize {
            return invalid("k must fit in 32 bits");
        }
        if !(self.nepochs.is_finite() && self.nepochs > 0.0) {
            return invalid("nepochs must be a positive number");
        }
        if self.batch_size == Some(0) {
            return invalid("batch_size must be at least 1");
        }
        if let Some(a) = self.alpha0
            && !(a > 0.0 && a <= 1.0)
        {
            return invalid("alpha0 must be in (0, 1]");
        }
        if let Some(t) = self.unused_threshold
            && !(t.is_finite() && t >= 0.0)
        {
            return invalid("unused_threshold must be >= 0");
        }
        if self.max_points_per_centroid == Some(0) {
            return invalid("max_points_per_centroid must be at least 1");
        }
        if self.chunk_size == 0 {
            return invalid("chunk_size must be at least 1");
        }
        Ok(())
    }

    /// The plan a run on `n` rows would follow.
    pub fn recipe(&self, n: usize) -> Result<Recipe> {
        self.check()?;
        if n == 0 {
            return invalid("need at least one row");
        }
        let (batch_law, batch_size) = match self.batch_size {
            Some(b) => (None, b.min(n)),
            None => {
                let law = self.batch_law.resolve(self.d);
                (
                    Some(law),
                    recipe::batch_size_for(n, self.k, self.nepochs, law),
                )
            }
        };
        let alpha0 = self
            .alpha0
            .unwrap_or_else(|| recipe::step_law(n, self.k, self.nepochs, batch_size));
        let epoch_rows = recipe::epoch_rows(n, self.nepochs);
        if epoch_rows.contains(&0) {
            return invalid("nepochs is too small: the run would see no rows");
        }
        let epoch_steps: Vec<usize> = epoch_rows.iter().map(|r| r.div_ceil(batch_size)).collect();
        let unused_threshold = match self.unused_threshold {
            Some(t) => Some(t),
            None if self.k > 1 => None,
            None if self.d <= recipe::NO_REVIVAL_MAX_D => Some(0.0),
            None => Some(recipe::UNUSED_THRESHOLD),
        };
        Ok(Recipe {
            batch_law,
            batch_size,
            alpha0,
            nepochs: self.nepochs,
            steps_per_pass: n.div_ceil(batch_size),
            total_steps: epoch_steps.iter().sum(),
            epoch_rows,
            epoch_steps,
            unused_threshold,
            seed: self.seed,
            chunk_size: self.chunk_size,
        })
    }

    fn rows_of(&self, x: &[f32]) -> Result<usize> {
        self.check()?;
        if !x.len().is_multiple_of(self.d) {
            return invalid(format!(
                "data length {} is not a multiple of d = {}",
                x.len(),
                self.d
            ));
        }
        Ok(x.len() / self.d)
    }

    /// Train on `x`, `n * d` values in row major order, with random starting centroids.
    pub fn train(&self, x: &[f32]) -> Result<Model> {
        self.train_inner(x, None)
    }

    /// Train on `x` starting from the given `k * d` centroids.
    pub fn train_from(&self, x: &[f32], init_centroids: &[f32]) -> Result<Model> {
        self.train_inner(x, Some(init_centroids))
    }

    fn train_inner(&self, x: &[f32], init: Option<&[f32]>) -> Result<Model> {
        let n = self.rows_of(x)?;
        let limit = self
            .max_points_per_centroid
            .map(|m| m.saturating_mul(self.k));
        let keep = match limit {
            Some(limit) if n > limit => {
                if self.verbose {
                    eprintln!(
                        "clustering {n} points to {} centroids: sampling {limit} points \
                         (max_points_per_centroid={})",
                        self.k,
                        self.max_points_per_centroid.unwrap()
                    );
                }
                stream(self.seed, STREAM_SUBSET).sample_sorted(n, limit)
            }
            _ => return self.train_rows(x, n, init),
        };
        let mut subset = Vec::with_capacity(keep.len() * self.d);
        for &i in &keep {
            subset.extend_from_slice(&x[i * self.d..(i + 1) * self.d]);
        }
        let mut model = self.train_rows(&subset, keep.len(), init)?;
        if let Some(sampled) = model.labels.take() {
            let mut labels = vec![-1i64; n];
            for (&i, &label) in keep.iter().zip(&sampled) {
                labels[i] = label;
            }
            model.labels = Some(labels);
        }
        Ok(model)
    }

    fn train_rows(&self, x: &[f32], n: usize, init: Option<&[f32]>) -> Result<Model> {
        let d = self.d;
        let picked;
        let init = match init {
            Some(init) => init,
            None if n < self.k => {
                return invalid(format!("need at least k = {} rows, got {n}", self.k));
            }
            None if n == self.k => x,
            None => {
                let order = stream(self.seed, STREAM_INIT).permutation(n);
                picked = order[..self.k]
                    .iter()
                    .flat_map(|&i| x[i * d..(i + 1) * d].iter().copied())
                    .collect::<Vec<f32>>();
                &picked
            }
        };
        let mut trainer = self.begin(n, init)?;
        let mut labels = self.capture_assignments.then(|| vec![-1i64; n]);
        let handover = labels.is_some() && self.closing_handover && trainer.removal_cost();
        let mut second = if handover {
            vec![u32::MAX; n]
        } else {
            Vec::new()
        };
        let (mut batch_labels, mut batch_second) = (Vec::new(), Vec::new());
        let batch_size = trainer.recipe.batch_size;
        let epochs = trainer.recipe.epoch_rows.clone();
        let mut order = Vec::new();
        for (epoch, &rows) in epochs.iter().enumerate() {
            order = stream(self.seed, STREAM_EPOCH + epoch as u64).permutation(n);
            let last = epoch + 1 == epochs.len();
            if last && handover {
                trainer.closing = Some(Closing::new(self.k));
            }
            for batch in order[..rows].chunks(batch_size) {
                match labels.as_mut() {
                    Some(labels) if last => {
                        batch_labels.resize(batch.len(), 0);
                        let second_out = if handover {
                            batch_second.resize(batch.len(), 0);
                            Some(batch_second.as_mut_slice())
                        } else {
                            None
                        };
                        trainer.accumulate_rows(
                            Rows::Indexed(x, batch),
                            Some(&mut batch_labels),
                            second_out,
                        )?;
                        for (&i, &label) in batch.iter().zip(&batch_labels) {
                            labels[i] = label;
                        }
                        if handover {
                            for (&i, &s) in batch.iter().zip(&batch_second) {
                                second[i] = s;
                            }
                        }
                        if let Some(closing) = trainer.closing.as_mut() {
                            closing.done += batch.len();
                        }
                    }
                    _ => trainer.accumulate_rows(Rows::Indexed(x, batch), None, None)?,
                }
                trainer.update()?;
            }
            trainer.end_epoch()?;
        }
        if let Some(labels) = labels.as_mut()
            && handover
        {
            trainer.close_handover(x, &order[..epochs[epochs.len() - 1]], labels, &second);
        }
        let mut model = trainer.finalize()?;
        model.labels = labels;
        Ok(model)
    }

    /// Start a run where you feed the batches yourself, for data that does not fit in memory.
    ///
    /// `n` is the total number of rows and `init_centroids` holds `k * d` values. For each
    /// entry of `recipe().epoch_rows`, feed that many rows in a fresh random order, in
    /// batches of `recipe().batch_size` rows (the last batch of a pass takes what is left).
    /// Call [`Trainer::update`] after each batch, [`Trainer::end_epoch`] after each pass,
    /// then [`Trainer::finalize`].
    pub fn begin(&self, n: usize, init_centroids: &[f32]) -> Result<Trainer> {
        let recipe = self.recipe(n)?;
        let (d, k) = (self.d, self.k);
        if init_centroids.len() != k * d {
            return invalid(format!(
                "init_centroids has {} values, expected k * d = {}",
                init_centroids.len(),
                k * d
            ));
        }
        let mut centroids = init_centroids.to_vec();
        if self.spherical {
            for row in centroids.chunks_exact_mut(d) {
                normalize(row);
            }
        }
        let mut split_rng = stream(self.seed, STREAM_SPLITS);
        // Revival places each split along the donor's long axis, so every mode that
        // revives tracks the axes. Only cost based revival needs the second nearest.
        let removal =
            (recipe.unused_threshold != Some(0.0)).then(|| Removal::new(k, d, &mut split_rng));
        let mut trainer = Trainer {
            d,
            k,
            spherical: self.spherical,
            verbose: self.verbose,
            n,
            recipe,
            positions: centroids.clone(),
            centroids,
            usage: vec![recipe::INIT_USAGE; k],
            norms: vec![0.0; k],
            batch_sums: vec![0.0; k * d],
            batch_counts: vec![0.0; k],
            batch_objective: 0.0,
            pending: 0,
            pass_counts: vec![0.0; k],
            pass_sums: None,
            pass_objective: 0.0,
            step: 0,
            epoch: 0,
            epoch_rows_done: 0,
            splits_before: 0,
            n_splits: 0,
            split_rng,
            removal,
            closing: None,
            epoch_stats: Vec::new(),
            start: Instant::now(),
            dist: Vec::new(),
            labels: Vec::new(),
            dist2: Vec::new(),
            labels2: Vec::new(),
            margin: Vec::new(),
        };
        trainer.start_epoch();
        if self.verbose {
            let law = match trainer.recipe.batch_law {
                Some(law) => format!("{law:?} batch law"),
                None => "batch given".to_string(),
            };
            eprintln!(
                "Clustering {n} points in {d}D to {k} clusters, {} epochs (batch {}, alpha0 {:.3}, {law})",
                trainer.recipe.nepochs, trainer.recipe.batch_size, trainer.recipe.alpha0
            );
        }
        Ok(trainer)
    }
}

/// Statistics of one pass.
#[derive(Clone, Debug, PartialEq)]
pub struct EpochStats {
    /// Sum of squared distances, measured while the centroids move (not the final error).
    /// In spherical mode, the sum of `2 - 2 * similarity`.
    pub objective: f64,
    /// Seconds since the start of training.
    pub time: f64,
    /// `k * sum(count^2) / rows^2`: 1 when the clusters are perfectly balanced.
    pub imbalance_factor: f64,
    /// Clusters revived during this pass.
    pub n_splits: usize,
}

/// A run in progress, fed batch by batch. Created by [`Emeans::begin`].
pub struct Trainer {
    d: usize,
    k: usize,
    spherical: bool,
    verbose: bool,
    n: usize,
    recipe: Recipe,
    // Moving average positions, and the centroids used for search (unit length in spherical mode).
    positions: Vec<f32>,
    centroids: Vec<f32>,
    usage: Vec<f32>,
    norms: Vec<f32>,
    batch_sums: Vec<f64>,
    batch_counts: Vec<f64>,
    batch_objective: f64,
    pending: usize,
    pass_counts: Vec<f64>,
    pass_sums: Option<Vec<f64>>,
    pass_objective: f64,
    step: usize,
    epoch: usize,
    epoch_rows_done: usize,
    splits_before: usize,
    n_splits: usize,
    split_rng: Rng,
    removal: Option<Removal>,
    closing: Option<Closing>,
    epoch_stats: Vec<EpochStats>,
    start: Instant,
    dist: Vec<f32>,
    labels: Vec<u32>,
    dist2: Vec<f32>,
    labels2: Vec<u32>,
    margin: Vec<f32>,
}

/// What revival needs to know about each cluster.
struct Removal {
    /// Per cluster `[U / scale | R]`: the moving average of `p * y` over the cluster's
    /// rows (`y` = row - centroid, `p` = `y` along the axis), and the moving average of
    /// the removal cost `d2 - d1`.
    stats: Vec<f32>,
    /// U decays as this one number instead of row by row.
    scale: f64,
    /// The current batch's `[sum p * y | sum d2 - d1]`.
    batch: Vec<f32>,
    /// Unit long axis of each cluster, `k * d` values.
    axes: Vec<f32>,
    /// Where each cluster's rows would go without it: the second nearest centroid of
    /// the last row it got. `u32::MAX` until it gets one.
    heir: Vec<u32>,
    /// Rows given since each cluster last got one.
    quiet: Vec<usize>,
}

impl Removal {
    fn new(k: usize, d: usize, rng: &mut Rng) -> Self {
        let mut axes: Vec<f32> = Vec::with_capacity(k * d);
        let mut row = vec![0.0f64; d];
        for _ in 0..k {
            row.iter_mut().for_each(|v| *v = rng.normal());
            let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            axes.extend(row.iter().map(|v| (v / norm) as f32));
        }
        Removal {
            stats: vec![0.0; k * (d + 1)],
            scale: 1.0,
            batch: vec![0.0; k * (d + 1)],
            axes,
            heir: vec![u32::MAX; k],
            quiet: vec![0; k],
        }
    }

    /// `scale * |U_j|`.
    fn axis_norm(&self, j: usize, d: usize) -> f64 {
        let u = &self.stats[j * (d + 1)..j * (d + 1) + d];
        self.scale * u.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt()
    }
}

/// When each cluster moved during the last pass, counted in rows of that pass.
/// Only kept by `train`, for the closing handover.
struct Closing {
    /// Rows of the last pass assigned so far.
    done: usize,
    /// Rows done when each cluster last moved, -1 if it did not move.
    moved_at: Vec<i64>,
    /// Every split of the pass: (donor, rows done at the time, revived cluster).
    splits: Vec<(usize, usize, usize)>,
}

impl Closing {
    fn new(k: usize) -> Self {
        Closing {
            done: 0,
            moved_at: vec![-1; k],
            splits: Vec::new(),
        }
    }
}

impl Trainer {
    /// The plan of this run.
    pub fn recipe(&self) -> &Recipe {
        &self.recipe
    }

    /// The current centroids, `k * d` values (unit length in spherical mode).
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Clusters revived so far.
    pub fn n_splits(&self) -> usize {
        self.n_splits
    }

    /// Rows the next [`Trainer::update`] expects in total.
    pub fn batch_rows(&self) -> usize {
        match self.recipe.epoch_rows.get(self.epoch) {
            Some(&rows) => self.recipe.batch_size.min(rows - self.epoch_rows_done),
            None => 0,
        }
    }

    /// Add `rows` (a multiple of `d` values) to the current batch.
    ///
    /// A batch can be given in several pieces. All pieces of one batch are searched
    /// against the same centroids, the ones from before the batch.
    pub fn accumulate(&mut self, rows: &[f32]) -> Result<()> {
        self.check_rows(rows)?;
        self.accumulate_rows(Rows::Dense(rows), None, None)
    }

    /// Like [`Trainer::accumulate`], and also writes each row's nearest centroid into `labels`.
    pub fn accumulate_labeled(&mut self, rows: &[f32], labels: &mut [i64]) -> Result<()> {
        self.check_rows(rows)?;
        if labels.len() != rows.len() / self.d {
            return invalid("labels must have one entry per row");
        }
        self.accumulate_rows(Rows::Dense(rows), Some(labels), None)
    }

    fn check_rows(&self, rows: &[f32]) -> Result<()> {
        if !rows.len().is_multiple_of(self.d) {
            return invalid(format!(
                "batch length {} is not a multiple of d = {}",
                rows.len(),
                self.d
            ));
        }
        Ok(())
    }

    fn removal_cost(&self) -> bool {
        self.recipe.unused_threshold.is_none()
    }

    fn accumulate_rows(
        &mut self,
        rows: Rows<'_>,
        mut labels_out: Option<&mut [i64]>,
        mut second_out: Option<&mut [u32]>,
    ) -> Result<()> {
        let d = self.d;
        let n = rows.len(d);
        if self.epoch >= self.recipe.epoch_rows.len() {
            return wrong_order("all passes are done, call finalize");
        }
        let expected = self.batch_rows();
        if self.pending + n > expected {
            return wrong_order(format!(
                "this batch takes {expected} rows, got {}",
                self.pending + n
            ));
        }
        if n == 0 {
            return Ok(());
        }
        if self.pending == 0 {
            self.batch_sums.fill(0.0);
            self.batch_counts.fill(0.0);
            self.batch_objective = 0.0;
            if !self.spherical {
                self.norms = knn::squared_norms(&self.centroids, d);
            }
        }
        let metric = if self.spherical {
            Metric::InnerProduct
        } else {
            Metric::L2
        };
        let two = self.removal_cost();
        let chunk = self.recipe.chunk_size;
        let mut objective = 0.0f64;
        for begin in (0..n).step_by(chunk) {
            let end = (begin + chunk).min(n);
            let part = match rows {
                Rows::Dense(x) => Rows::Dense(&x[begin * d..end * d]),
                Rows::Indexed(x, index) => Rows::Indexed(x, &index[begin..end]),
            };
            self.dist.resize(end - begin, 0.0);
            self.labels.resize(end - begin, 0);
            if two {
                self.dist2.resize(end - begin, 0.0);
                self.labels2.resize(end - begin, 0);
                knn::nearest_two(
                    part,
                    d,
                    &self.centroids,
                    &self.norms,
                    metric,
                    &mut self.dist,
                    &mut self.labels,
                    &mut self.dist2,
                    &mut self.labels2,
                );
            } else {
                knn::nearest(
                    part,
                    d,
                    &self.centroids,
                    &self.norms,
                    metric,
                    &mut self.dist,
                    &mut self.labels,
                );
            }
            for &dist in &self.dist {
                objective += if self.spherical {
                    2.0 - 2.0 * dist as f64
                } else {
                    dist as f64
                };
            }
            if self.removal.is_some() {
                // The removal cost is d2 - d1. Inner product keeps similarities, and
                // faiss scores them as -2 * similarity.
                self.margin.clear();
                self.margin.extend((0..end - begin).map(|i| {
                    let margin = match (two, self.spherical) {
                        (false, _) => 0.0,
                        (true, false) => self.dist2[i] - self.dist[i],
                        (true, true) => 2.0 * (self.dist[i] - self.dist2[i]),
                    };
                    if margin.is_finite() { margin } else { 0.0 }
                }));
            }
            add_rows(
                part,
                d,
                &self.labels,
                if two { &self.labels2 } else { &[] },
                &self.margin,
                &self.centroids,
                self.removal.as_mut(),
                &mut self.batch_sums,
                &mut self.batch_counts,
            );
            if let Some(out) = labels_out.as_deref_mut() {
                for (o, &l) in out[begin..end].iter_mut().zip(&self.labels) {
                    *o = l as i64;
                }
            }
            if let Some(out) = second_out.as_deref_mut() {
                out[begin..end].copy_from_slice(&self.labels2);
            }
        }
        self.batch_objective += objective;
        self.pending += n;
        Ok(())
    }

    /// Apply the current batch to the centroids. The batch must be complete.
    pub fn update(&mut self) -> Result<()> {
        let expected = self.batch_rows();
        if expected == 0 {
            return wrong_order("all passes are done, call finalize");
        }
        if self.pending != expected {
            return wrong_order(format!(
                "this batch takes {expected} rows, got {}",
                self.pending
            ));
        }
        let alpha = recipe::scheduled_alpha(self.recipe.alpha0, self.step, self.recipe.total_steps);
        self.update_removal(alpha);
        self.apply_batch(alpha);
        if (self.step + 1).is_multiple_of(recipe::CHECK_UNUSED_EVERY) {
            self.revive_unused();
        }
        self.refresh_search();
        self.step += 1;
        self.epoch_rows_done += expected;
        self.pending = 0;
        Ok(())
    }

    /// The moving average update, written on positions c = S / N and masses N:
    ///
    /// ```text
    /// N_new = (1 - alpha) * N + alpha * n_j
    /// c_new = c + (alpha * n_j / N_new) * (s_j / n_j - c)
    /// ```
    fn apply_batch(&mut self, alpha: f64) {
        let d = self.d;
        self.pass_objective += self.batch_objective;
        for (p, &c) in self.pass_counts.iter_mut().zip(&self.batch_counts) {
            *p += c;
        }
        if let Some(pass_sums) = self.pass_sums.as_mut() {
            for (p, &s) in pass_sums.iter_mut().zip(&self.batch_sums) {
                *p += s;
            }
        }
        // A step of exactly 1 replaces the mass of occupied clusters and leaves the others alone.
        let replace = alpha == 1.0;
        let alpha = alpha as f32;
        if !replace {
            let decay = 1.0 - alpha;
            for u in self.usage.iter_mut() {
                *u *= decay;
            }
        }
        for j in 0..self.k {
            let count = self.batch_counts[j];
            if count == 0.0 {
                continue;
            }
            let retained = if replace { 0.0 } else { self.usage[j] };
            let mass = retained + alpha * count as f32;
            let weight = alpha * count as f32 / mass;
            let sums = &self.batch_sums[j * d..(j + 1) * d];
            let centroid = &mut self.positions[j * d..(j + 1) * d];
            for (c, &s) in centroid.iter_mut().zip(sums) {
                let mean = (s / count) as f32;
                *c = if weight == 1.0 {
                    mean
                } else {
                    *c + weight * (mean - *c)
                };
            }
            self.usage[j] = mass;
        }
    }

    /// Fold the batch into the removal costs and the long axes (Oja's rule).
    fn update_removal(&mut self, alpha: f64) {
        let Some(removal) = self.removal.as_mut() else {
            return;
        };
        let (k, d) = (self.k, self.d);
        let decay = 1.0 - alpha;
        let (decay32, alpha32) = (decay as f32, alpha as f32);
        for j in 0..k {
            let at = j * (d + 1) + d;
            removal.stats[at] = decay32 * removal.stats[at] + alpha32 * removal.batch[at];
        }
        let mut scale = removal.scale * decay;
        // Fold the scale back into U before it gets too small for f32.
        if scale <= 2f64.powi(-32) {
            for row in removal.stats.chunks_exact_mut(d + 1) {
                row[..d].iter_mut().for_each(|v| *v *= scale as f32);
            }
            scale = 1.0;
        }
        removal.scale = scale;
        let weight = (alpha / scale) as f32;
        let given = self.pending;
        for j in 0..k {
            if self.batch_counts[j] == 0.0 {
                removal.quiet[j] += given;
                continue;
            }
            removal.quiet[j] = 0;
            let rows = j * (d + 1)..(j + 1) * (d + 1);
            let (u, b) = (&mut removal.stats[rows.clone()], &mut removal.batch[rows]);
            for (u, &b) in u[..d].iter_mut().zip(&b[..d]) {
                *u += weight * b;
            }
            let norm = u[..d]
                .iter()
                .map(|&v| v as f64 * v as f64)
                .sum::<f64>()
                .sqrt();
            if norm > 0.0 {
                for (a, &u) in removal.axes[j * d..(j + 1) * d].iter_mut().zip(&u[..d]) {
                    *a = (u as f64 / norm) as f32;
                }
            }
            b.fill(0.0);
        }
    }

    /// Move clusters that do little to split busy ones.
    fn revive_unused(&mut self) {
        let splits = match self.recipe.unused_threshold {
            None => self.choose_by_cost(),
            Some(t) if t > 0.0 => self.choose_by_mass(t),
            Some(_) => return,
        };
        if !splits.is_empty() {
            self.apply_splits(&splits);
        }
    }

    /// Every cluster with a removal cost below 0.3 times the median, or not positive,
    /// moves, cheapest first. The donors are the other clusters, longest axis first.
    fn choose_by_cost(&self) -> Vec<(usize, usize)> {
        let (k, d) = (self.k, self.d);
        let removal = self
            .removal
            .as_ref()
            .expect("cost based revival tracks costs");
        // The cost of a cluster that stopped getting rows only fades, as fast as every
        // other cost, so it may never fall below the median. A cluster that got no row
        // in a whole pass owns none, and costs nothing.
        let pass = self.recipe.epoch_rows.iter().copied().max().unwrap_or(0);
        let cost: Vec<f64> = (0..k)
            .map(|j| {
                if removal.quiet[j] >= pass {
                    0.0
                } else {
                    removal.stats[j * (d + 1) + d] as f64
                }
            })
            .collect();
        let mut sorted = cost.clone();
        sorted.sort_by(f64::total_cmp);
        let median = if k % 2 == 1 {
            sorted[k / 2]
        } else {
            (sorted[k / 2 - 1] + sorted[k / 2]) / 2.0
        };
        let limit = recipe::REMOVAL_THRESHOLD * median;
        let mut cheap: Vec<usize> = (0..k)
            .filter(|&j| cost[j] <= 0.0 || cost[j] < limit)
            .collect();
        if cheap.is_empty() {
            return Vec::new();
        }
        cheap.sort_by(|&a, &b| cost[a].total_cmp(&cost[b]));
        // Each cost assumes the rest stay, so a cluster cannot move together with the
        // one that takes its rows. Otherwise two twins on a blob both leave it.
        let mut is_moved = vec![false; k];
        let mut kept = vec![false; k];
        let mut moved = Vec::with_capacity(cheap.len());
        for j in cheap {
            let heir = removal.heir[j] as usize;
            if kept[j] || (heir < k && is_moved[heir]) {
                continue;
            }
            is_moved[j] = true;
            moved.push(j);
            if heir < k {
                kept[heir] = true;
            }
        }
        let axis: Vec<f32> = (0..k).map(|j| removal.axis_norm(j, d) as f32).collect();
        let mut donors: Vec<usize> = (0..k)
            .filter(|&j| self.usage[j] > 0.0 && !is_moved[j])
            .collect();
        donors.sort_by(|&a, &b| axis[b].total_cmp(&axis[a]));
        // A pair moves only if the donor's split gains more than the move costs,
        // (2 / pi) |U| > R. Costs rise and gains fall along the list, so the pairs
        // that pay come first. A donor with no spread gains nothing, so even a free
        // move onto it would only make a copy (duplicate rows).
        let gain = 2.0 / std::f64::consts::PI;
        donors
            .into_iter()
            .zip(moved)
            .take_while(|&(donor, j)| {
                let gain = gain * axis[donor] as f64;
                gain > 0.0 && (cost[j] <= 0.0 || gain > cost[j])
            })
            .collect()
    }

    /// Clusters below `threshold` times the mean mass are revived. Donors are drawn
    /// without replacement in proportion to their mass (Gumbel top-k).
    fn choose_by_mass(&mut self, threshold: f32) -> Vec<(usize, usize)> {
        let k = self.k;
        let total: f32 = self.usage.iter().sum();
        let limit = total * threshold / k as f32;
        let unused: Vec<usize> = (0..k).filter(|&j| self.usage[j] < limit).collect();
        let donors: Vec<usize> = (0..k)
            .filter(|&j| self.usage[j] >= limit && self.usage[j] > 0.0)
            .collect();
        if unused.is_empty() || donors.is_empty() {
            return Vec::new();
        }
        let mut keys: Vec<(f64, usize)> = donors
            .iter()
            .map(|&j| {
                let u = self.split_rng.next_f64().max(f64::MIN_POSITIVE);
                ((self.usage[j] as f64).ln() - (-u.ln()).ln(), j)
            })
            .collect();
        keys.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        keys.into_iter().map(|(_, j)| j).zip(unused).collect()
    }

    /// Split each donor into itself and the revived cluster. The two copies go to
    /// `c -/+ s * u`, with `u` the donor's long axis and `s = sqrt(2 |U| / (pi N))`,
    /// the means of the two halves of a Gaussian along `u`.
    fn apply_splits(&mut self, splits: &[(usize, usize)]) {
        let d = self.d;
        let removal = self.removal.as_mut().expect("revival tracks the axes");
        let last_pass = self.pass_sums.is_some();
        for &(source, target) in splits {
            let norm = removal.axis_norm(source, d);
            let mass = self.usage[source] as f64;
            let spread = if norm > 0.0 && mass > 0.0 && norm.is_finite() && mass.is_finite() {
                (2.0 * (norm / mass) / std::f64::consts::PI).sqrt()
            } else {
                0.0
            };
            let mut collapsed = true;
            for i in 0..d {
                let c = self.positions[source * d + i] as f64;
                let offset = spread * removal.axes[source * d + i] as f64;
                let donor = (c - offset) as f32;
                let child = (c + offset) as f32;
                collapsed &= donor == child;
                self.positions[source * d + i] = donor;
                self.positions[target * d + i] = child;
            }
            if collapsed {
                // Two equal centroids: move the child by one ulp.
                let v = self.positions[target * d];
                let up = v.next_up();
                self.positions[target * d] = if up.is_finite() { up } else { v.next_down() };
            }
            let half = self.usage[source] / 2.0;
            self.usage[source] = half;
            self.usage[target] = half;

            let (from, to) = (source * (d + 1), target * (d + 1));
            removal.stats[from..from + d + 1]
                .iter_mut()
                .for_each(|v| *v /= 2.0);
            removal.stats.copy_within(from..from + d + 1, to);
            removal
                .axes
                .copy_within(source * d..(source + 1) * d, target * d);
            removal.heir[source] = target as u32;
            removal.heir[target] = source as u32;
            removal.quiet[target] = 0;

            match self.closing.as_mut() {
                Some(closing) => {
                    closing.moved_at[target] = closing.done as i64;
                    closing.splits.push((source, closing.done, target));
                }
                // Without the handover, a split in the last pass drops what both
                // clusters got before it, as their rows no longer match.
                None if last_pass && self.recipe.unused_threshold.is_none() => {
                    let sums = self.pass_sums.as_mut().unwrap();
                    for j in [source, target] {
                        sums[j * d..(j + 1) * d].fill(0.0);
                        self.pass_counts[j] = 0.0;
                    }
                }
                None => {}
            }
        }
        self.n_splits += splits.len();
    }

    fn refresh_search(&mut self) {
        self.centroids.copy_from_slice(&self.positions);
        if self.spherical {
            for row in self.centroids.chunks_exact_mut(self.d) {
                normalize(row);
            }
        }
    }

    fn start_epoch(&mut self) {
        self.epoch_rows_done = 0;
        self.splits_before = self.n_splits;
        self.pass_counts.fill(0.0);
        self.pass_objective = 0.0;
        let last = self.epoch + 1 == self.recipe.epoch_rows.len();
        self.pass_sums = last.then(|| vec![0.0; self.k * self.d]);
    }

    /// Finish one pass. Every row of the pass must have been given.
    pub fn end_epoch(&mut self) -> Result<()> {
        let Some(&rows) = self.recipe.epoch_rows.get(self.epoch) else {
            return wrong_order("all passes are done, call finalize");
        };
        if self.pending != 0 || self.epoch_rows_done != rows {
            return wrong_order(format!(
                "pass {} takes {rows} rows, got {}",
                self.epoch,
                self.epoch_rows_done + self.pending
            ));
        }
        let square: f64 = self.pass_counts.iter().map(|c| c * c).sum();
        let stats = EpochStats {
            objective: self.pass_objective,
            time: self.start.elapsed().as_secs_f64(),
            imbalance_factor: self.k as f64 * square / (rows as f64 * rows as f64),
            n_splits: self.n_splits - self.splits_before,
        };
        if self.verbose {
            let done: usize = self.recipe.epoch_rows[..=self.epoch].iter().sum();
            let total: usize = self.recipe.epoch_rows.iter().sum();
            eprintln!(
                "  Epoch {:.4} / {:.4}, iteration {} / {} ({:.2} s): objective={}",
                done as f64 / self.n as f64,
                total as f64 / self.n as f64,
                self.step,
                self.recipe.total_steps,
                stats.time,
                stats.objective
            );
        }
        self.epoch_stats.push(stats);
        self.epoch += 1;
        if self.epoch < self.recipe.epoch_rows.len() {
            self.start_epoch();
        }
        Ok(())
    }

    /// Fix the last pass labels after clusters moved during it, then rebuild the sums of
    /// the clusters involved from the fixed labels.
    ///
    /// `order` holds the rows of the last pass in the order they were given, `second` the
    /// second nearest centroid each row got with its label.
    fn close_handover(&mut self, x: &[f32], order: &[usize], labels: &mut [i64], second: &[u32]) {
        let Some(closing) = self.closing.take() else {
            return;
        };
        let (k, d) = (self.k, self.d);
        let moved_at = &closing.moved_at;
        if moved_at.iter().all(|&m| m < 0) {
            return;
        }

        // A row given to a cluster before that cluster moved goes to its second nearest,
        // unless that one also moved after the row was given.
        let mut stale = vec![false; labels.len()];
        let mut stale_rows = Vec::new();
        let mut search = Vec::new();
        for (rank, &i) in order.iter().enumerate() {
            let rank = rank as i64;
            if rank >= moved_at[labels[i] as usize] {
                continue;
            }
            stale[i] = true;
            stale_rows.push(i);
            let s = second[i] as usize;
            if s < k && moved_at[s] <= rank {
                labels[i] = s as i64;
            } else {
                search.push(i);
            }
        }
        if !search.is_empty() {
            let metric = if self.spherical {
                Metric::InnerProduct
            } else {
                Metric::L2
            };
            let norms = knn::squared_norms(&self.centroids, d);
            for part in search.chunks(self.recipe.chunk_size) {
                self.dist.resize(part.len(), 0.0);
                self.labels.resize(part.len(), 0);
                knn::nearest(
                    Rows::Indexed(x, part),
                    d,
                    &self.centroids,
                    &norms,
                    metric,
                    &mut self.dist,
                    &mut self.labels,
                );
                for (&i, &l) in part.iter().zip(&self.labels) {
                    labels[i] = l as i64;
                }
            }
        }

        // The other rows of a donor, given before its next split, go to the nearer of
        // the donor and the cluster that split made. The entry with donor k matches no
        // row and keeps the search below in bounds.
        let mut splits = closing.splits;
        splits.push((k, 0, 0));
        splits.sort_by_key(|&(source, born, _)| (source, born));
        let centroids = &self.centroids;
        let distance = |row: &[f32], j: usize| -> f64 {
            row.iter()
                .zip(&centroids[j * d..(j + 1) * d])
                .map(|(&a, &b)| {
                    let diff = (a - b) as f64;
                    diff * diff
                })
                .sum()
        };
        for (rank, &i) in order.iter().enumerate() {
            if stale[i] {
                continue;
            }
            let own = labels[i] as usize;
            let next = splits
                .partition_point(|&(source, born, _)| (source, born) <= (own, rank))
                .min(splits.len() - 1);
            let (source, born, child) = splits[next];
            if source != own || born <= rank || moved_at[child] != born as i64 {
                continue;
            }
            let row = &x[i * d..(i + 1) * d];
            let (near_own, near_child) = (distance(row, own), distance(row, child));
            if near_child < near_own || (near_child == near_own && child < own) {
                labels[i] = child as i64;
            }
        }

        let mut rebuild = vec![false; k + 1];
        for &(source, _, child) in &splits {
            rebuild[source] = true;
            rebuild[if source < k { child } else { k }] = true;
        }
        for &i in &stale_rows {
            rebuild[labels[i] as usize] = true;
        }
        let sums = self
            .pass_sums
            .as_mut()
            .expect("the last pass keeps its sums");
        for j in (0..k).filter(|&j| rebuild[j]) {
            sums[j * d..(j + 1) * d].fill(0.0);
            self.pass_counts[j] = 0.0;
        }
        for &i in order {
            let l = labels[i] as usize;
            if rebuild[l] {
                for (s, &v) in sums[l * d..(l + 1) * d]
                    .iter_mut()
                    .zip(&x[i * d..(i + 1) * d])
                {
                    *s += v as f64;
                }
                self.pass_counts[l] += 1.0;
            }
        }
    }

    /// Finish the run. Each centroid becomes the exact mean of the rows the last pass gave it.
    /// Centroids that got no row keep their position.
    pub fn finalize(mut self) -> Result<Model> {
        if self.epoch != self.recipe.epoch_rows.len() {
            return wrong_order(format!(
                "{} of {} passes done",
                self.epoch,
                self.recipe.epoch_rows.len()
            ));
        }
        let d = self.d;
        let sums = self.pass_sums.take().expect("the last pass keeps its sums");
        let mut mean = vec![0.0f64; d];
        for j in 0..self.k {
            let count = self.pass_counts[j];
            let centroid = &mut self.positions[j * d..(j + 1) * d];
            if count > 0.0 {
                for (m, &s) in mean.iter_mut().zip(&sums[j * d..(j + 1) * d]) {
                    *m = s / count;
                }
            } else {
                for (m, &c) in mean.iter_mut().zip(centroid.iter()) {
                    *m = c as f64;
                }
            }
            if self.spherical {
                normalize_f64(&mut mean);
            }
            for (c, &m) in centroid.iter_mut().zip(&mean) {
                *c = m as f32;
            }
        }
        Ok(Model {
            d,
            k: self.k,
            spherical: self.spherical,
            norms: knn::squared_norms(&self.positions, d),
            centroids: self.positions,
            labels: None,
            epoch_stats: self.epoch_stats,
            recipe: Some(self.recipe),
            n_splits: self.n_splits,
        })
    }
}

/// Trained centroids.
#[derive(Clone, Debug)]
pub struct Model {
    d: usize,
    k: usize,
    spherical: bool,
    centroids: Vec<f32>,
    norms: Vec<f32>,
    labels: Option<Vec<i64>>,
    epoch_stats: Vec<EpochStats>,
    recipe: Option<Recipe>,
    n_splits: usize,
}

impl Model {
    /// Wrap centroids you already have, for example loaded from disk, to call [`Model::assign`].
    pub fn from_centroids(d: usize, centroids: Vec<f32>, spherical: bool) -> Result<Model> {
        if d == 0 || centroids.is_empty() || !centroids.len().is_multiple_of(d) {
            return invalid("centroids must hold k * d values with k, d >= 1");
        }
        let k = centroids.len() / d;
        if k > u32::MAX as usize {
            return invalid("k must fit in 32 bits");
        }
        Ok(Model {
            d,
            k,
            spherical,
            norms: knn::squared_norms(&centroids, d),
            centroids,
            labels: None,
            epoch_stats: Vec::new(),
            recipe: None,
            n_splits: 0,
        })
    }

    /// Dimension of the vectors.
    pub fn d(&self) -> usize {
        self.d
    }

    /// Number of clusters.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Whether the model clusters by direction.
    pub fn is_spherical(&self) -> bool {
        self.spherical
    }

    /// The centroids, `k * d` values in row major order.
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// One centroid.
    pub fn centroid(&self, j: usize) -> &[f32] {
        &self.centroids[j * self.d..(j + 1) * self.d]
    }

    /// Take ownership of the centroids.
    pub fn into_centroids(self) -> Vec<f32> {
        self.centroids
    }

    /// Each training row's label from the last pass, -1 for rows left out by
    /// `max_points_per_centroid`. `None` when the labels were not kept or the model
    /// came from the batch loop.
    ///
    /// These labels come for free but are not always the nearest final centroid, since
    /// the centroids still moved during the last pass. Call [`Model::assign`] for exact ones.
    ///
    /// With more centroids than distinct rows, some clusters stay empty. Exact ties go
    /// to the lower index, so duplicate rows land in the lowest of the tied clusters.
    pub fn labels(&self) -> Option<&[i64]> {
        self.labels.as_deref()
    }

    /// Take ownership of the labels.
    pub fn take_labels(&mut self) -> Option<Vec<i64>> {
        self.labels.take()
    }

    /// Statistics of each pass.
    pub fn epoch_stats(&self) -> &[EpochStats] {
        &self.epoch_stats
    }

    /// The plan the run followed. `None` for [`Model::from_centroids`].
    pub fn recipe(&self) -> Option<&Recipe> {
        self.recipe.as_ref()
    }

    /// Clusters revived during training.
    pub fn n_splits(&self) -> usize {
        self.n_splits
    }

    /// Nearest centroid of each row of `x`, and its squared distance.
    /// In spherical mode, the highest inner product instead (higher is closer).
    pub fn assign(&self, x: &[f32]) -> Result<(Vec<f32>, Vec<i64>)> {
        if !x.len().is_multiple_of(self.d) {
            return invalid(format!(
                "data length {} is not a multiple of d = {}",
                x.len(),
                self.d
            ));
        }
        let n = x.len() / self.d;
        let metric = if self.spherical {
            Metric::InnerProduct
        } else {
            Metric::L2
        };
        let mut dist = vec![0.0f32; n];
        let mut labels = vec![0u32; n];
        knn::nearest(
            Rows::Dense(x),
            self.d,
            &self.centroids,
            &self.norms,
            metric,
            &mut dist,
            &mut labels,
        );
        Ok((dist, labels.into_iter().map(|l| l as i64).collect()))
    }
}

/// Add each row to its cluster's batch sums and count, and to the revival statistics
/// when there are some: `p * y` with `y` = row - centroid and `p` = `y` along the axis,
/// and the removal cost from `margin`.
///
/// Clusters are split in ranges that run in parallel. Each range scans every row and
/// adds its own clusters' rows in row order, so the result does not depend on the
/// number of threads.
#[allow(clippy::too_many_arguments)]
fn add_rows(
    rows: Rows<'_>,
    d: usize,
    labels: &[u32],
    second: &[u32],
    margin: &[f32],
    centroids: &[f32],
    removal: Option<&mut Removal>,
    sums: &mut [f64],
    counts: &mut [f64],
) {
    let (axes, batch, heir): (&[f32], &mut [f32], &mut [u32]) = match removal {
        Some(r) => (&r.axes, &mut r.batch, &mut r.heir),
        None => (&[], &mut [], &mut []),
    };
    let add = |first: usize,
               sums: &mut [f64],
               counts: &mut [f64],
               batch: &mut [f32],
               heir: &mut [u32]| {
        let last = first + counts.len();
        for (i, &label) in labels.iter().enumerate() {
            let label = label as usize;
            if label < first || label >= last {
                continue;
            }
            let (j, row) = (label - first, rows.row(i, d));
            counts[j] += 1.0;
            for (s, &v) in sums[j * d..(j + 1) * d].iter_mut().zip(row) {
                *s += v as f64;
            }
            if batch.is_empty() {
                continue;
            }
            let centroid = &centroids[label * d..(label + 1) * d];
            let axis = &axes[label * d..(label + 1) * d];
            let p: f32 = row
                .iter()
                .zip(centroid)
                .zip(axis)
                .map(|((&x, &c), &a)| (x - c) * a)
                .sum();
            let stats = &mut batch[j * (d + 1)..(j + 1) * (d + 1)];
            for ((s, &x), &c) in stats.iter_mut().zip(row).zip(centroid) {
                *s += p * (x - c);
            }
            stats[d] += margin[i];
            if !second.is_empty() && second[i] != u32::MAX {
                heir[j] = second[i];
            }
        }
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        // A few ranges per thread, so uneven ranges still balance.
        let k = counts.len();
        let per = k.div_ceil(4 * rayon::current_num_threads()).max(1);
        let batch_per = if batch.is_empty() { 0 } else { per * (d + 1) };
        let mut batches: Vec<&mut [f32]> = if batch_per == 0 {
            Vec::new()
        } else {
            batch.chunks_mut(batch_per).collect()
        };
        batches.resize_with(k.div_ceil(per), Default::default);
        let mut heirs: Vec<&mut [u32]> = if heir.is_empty() {
            Vec::new()
        } else {
            heir.chunks_mut(per).collect()
        };
        heirs.resize_with(k.div_ceil(per), Default::default);
        sums.par_chunks_mut(per * d)
            .zip(counts.par_chunks_mut(per))
            .zip(batches.into_par_iter())
            .zip(heirs.into_par_iter())
            .enumerate()
            .for_each(|(r, (((sums, counts), batch), heir))| {
                add(r * per, sums, counts, batch, heir)
            });
    }
    #[cfg(not(feature = "parallel"))]
    add(0, sums, counts, batch, heir);
}

/// Scale a row to norm 1. Divides by the largest value first so the norm cannot overflow.
/// A zero row stays zero.
fn normalize(row: &mut [f32]) {
    let max = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if max > 0.0 {
        row.iter_mut().for_each(|v| *v /= max);
    }
    let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        row.iter_mut().for_each(|v| *v /= norm);
    }
}

fn normalize_f64(row: &mut [f64]) {
    let max = row.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if max > 0.0 {
        row.iter_mut().for_each(|v| *v /= max);
    }
    let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 {
        row.iter_mut().for_each(|v| *v /= norm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_revival_moves_every_cheap_cluster() {
        // Same case as the faiss test: costs [1, 100, 0, 100], axis lengths [1, 3, 2, 4].
        // Clusters 2 and 0 move, cheapest first, to split 3 and then 1.
        let mut t = Emeans::new(1, 4)
            .nepochs(1.0)
            .batch_size(4)
            .begin(4, &[0.0, 1.0, -2.0, 10.0])
            .unwrap();
        t.usage = vec![1.0; 4];
        let removal = t.removal.as_mut().unwrap();
        for (j, (cost, axis)) in [(1.0, 1.0), (100.0, 3.0), (0.0, 2.0), (100.0, 4.0)]
            .into_iter()
            .enumerate()
        {
            removal.stats[j * 2] = axis;
            removal.stats[j * 2 + 1] = cost;
        }
        assert_eq!(t.choose_by_cost(), vec![(3, 2), (1, 0)]);
    }

    #[test]
    fn split_copies_sit_on_the_donor_axis() {
        let mut t = Emeans::new(2, 2).begin(10, &[1.0, 1.0, 5.0, 5.0]).unwrap();
        t.usage = vec![4.0, 0.0];
        let removal = t.removal.as_mut().unwrap();
        removal.axes = vec![1.0, 0.0, 0.0, 1.0];
        // |U| = 2 * pi, N = 4: s = sqrt(2 * 2 * pi / (pi * 4)) = 1.
        removal.stats[..3].copy_from_slice(&[2.0 * std::f32::consts::PI, 0.0, 0.5]);
        t.apply_splits(&[(0, 1)]);
        let expected = [0.0, 1.0, 2.0, 1.0];
        assert!(
            t.positions
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
        assert_eq!(t.usage, vec![2.0, 2.0]);
        let removal = t.removal.as_ref().unwrap();
        assert_eq!(&removal.stats[..3], &removal.stats[3..]);
        assert_eq!(removal.stats[2], 0.25);
        assert_eq!(removal.axes, vec![1.0, 0.0, 1.0, 0.0]);
    }
}
