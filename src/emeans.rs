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

    /// Revive clusters whose mass falls below this fraction of the mean. 0 turns it off.
    /// Default 0.1, or 0 when `d <= 2`.
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
        let unused_threshold =
            self.unused_threshold
                .unwrap_or(if self.d <= recipe::NO_REVIVAL_MAX_D {
                    0.0
                } else {
                    recipe::UNUSED_THRESHOLD
                });
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
        if x.len() % self.d != 0 {
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
        let mut batch_labels = Vec::new();
        let batch_size = trainer.recipe.batch_size;
        let epochs = trainer.recipe.epoch_rows.clone();
        for (epoch, &rows) in epochs.iter().enumerate() {
            let order = stream(self.seed, STREAM_EPOCH + epoch as u64).permutation(n);
            let last = epoch + 1 == epochs.len();
            for batch in order[..rows].chunks(batch_size) {
                match labels.as_mut() {
                    Some(labels) if last => {
                        batch_labels.resize(batch.len(), 0);
                        trainer
                            .accumulate_rows(Rows::Indexed(x, batch), Some(&mut batch_labels))?;
                        for (&i, &label) in batch.iter().zip(&batch_labels) {
                            labels[i] = label;
                        }
                    }
                    _ => trainer.accumulate_rows(Rows::Indexed(x, batch), None)?,
                }
                trainer.update()?;
            }
            trainer.end_epoch()?;
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
            split_rng: stream(self.seed, STREAM_SPLITS),
            epoch_stats: Vec::new(),
            start: Instant::now(),
            dist: Vec::new(),
            labels: Vec::new(),
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
    epoch_stats: Vec<EpochStats>,
    start: Instant,
    dist: Vec<f32>,
    labels: Vec<u32>,
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
        self.accumulate_rows(Rows::Dense(rows), None)
    }

    /// Like [`Trainer::accumulate`], and also writes each row's nearest centroid into `labels`.
    pub fn accumulate_labeled(&mut self, rows: &[f32], labels: &mut [i64]) -> Result<()> {
        self.check_rows(rows)?;
        if labels.len() != rows.len() / self.d {
            return invalid("labels must have one entry per row");
        }
        self.accumulate_rows(Rows::Dense(rows), Some(labels))
    }

    fn check_rows(&self, rows: &[f32]) -> Result<()> {
        if rows.len() % self.d != 0 {
            return invalid(format!(
                "batch length {} is not a multiple of d = {}",
                rows.len(),
                self.d
            ));
        }
        Ok(())
    }

    fn accumulate_rows(
        &mut self,
        rows: Rows<'_>,
        mut labels_out: Option<&mut [i64]>,
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
            knn::nearest(
                part,
                d,
                &self.centroids,
                &self.norms,
                metric,
                &mut self.dist,
                &mut self.labels,
            );
            for (i, (&dist, &label)) in self.dist.iter().zip(&self.labels).enumerate() {
                let label = label as usize;
                objective += if self.spherical {
                    2.0 - 2.0 * dist as f64
                } else {
                    dist as f64
                };
                self.batch_counts[label] += 1.0;
                let sums = &mut self.batch_sums[label * d..(label + 1) * d];
                for (s, &v) in sums.iter_mut().zip(part.row(i, d)) {
                    *s += v as f64;
                }
            }
            if let Some(out) = labels_out.as_deref_mut() {
                for (o, &l) in out[begin..end].iter_mut().zip(&self.labels) {
                    *o = l as i64;
                }
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
        self.apply_batch(alpha);
        if (self.step + 1) % recipe::CHECK_UNUSED_EVERY == 0 {
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

    /// Split busy clusters to replace the ones that get almost no rows.
    fn revive_unused(&mut self) {
        let threshold = self.recipe.unused_threshold;
        if threshold == 0.0 {
            return;
        }
        let (k, d) = (self.k, self.d);
        let total: f32 = self.usage.iter().sum();
        let limit = total * threshold / k as f32;
        let unused: Vec<usize> = (0..k).filter(|&j| self.usage[j] < limit).collect();
        let donors: Vec<usize> = (0..k)
            .filter(|&j| self.usage[j] >= limit && self.usage[j] > 0.0)
            .collect();
        let nsplit = unused.len().min(donors.len());
        if nsplit == 0 {
            return;
        }
        // Donors drawn without replacement in proportion to their mass (Gumbel top-k).
        let mut keys: Vec<(f64, usize)> = donors
            .iter()
            .map(|&j| {
                let u = self.split_rng.next_f64().max(f64::MIN_POSITIVE);
                ((self.usage[j] as f64).ln() - (-u.ln()).ln(), j)
            })
            .collect();
        keys.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        for (&(_, source), &target) in keys.iter().zip(&unused).take(nsplit) {
            let half = self.usage[source] / 2.0;
            let base = self.positions[source * d..(source + 1) * d].to_vec();
            let mut collapsed = true;
            for (i, &v) in base.iter().enumerate() {
                let offset = if i % 2 == 0 {
                    recipe::SPLIT_EPSILON
                } else {
                    -recipe::SPLIT_EPSILON
                };
                let donor = v * (1.0 - offset);
                let child = v * (1.0 + offset);
                collapsed &= donor == child;
                self.positions[source * d + i] = donor;
                self.positions[target * d + i] = child;
            }
            if collapsed {
                // A zero centroid splits into two equal ones: move the child by one ulp.
                let v = self.positions[target * d];
                let up = v.next_up();
                self.positions[target * d] = if up.is_finite() { up } else { v.next_down() };
            }
            self.usage[source] = half;
            self.usage[target] = half;
        }
        self.n_splits += nsplit;
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
        if d == 0 || centroids.is_empty() || centroids.len() % d != 0 {
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
        if x.len() % self.d != 0 {
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
