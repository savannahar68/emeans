//! Batch size and step size from the fitted laws, and the per pass schedule.
//!
//! The constants are copied from `faiss/contrib/e_means/base.py`.

/// Epsilon of the cosine schedule: the step goes down to this fraction of `alpha0`.
pub(crate) const ALPHA_MIN_FACTOR: f64 = 0.1;
/// Number of updates between two checks for unused clusters.
pub(crate) const CHECK_UNUSED_EVERY: usize = 5;
/// Starting mass of every cluster.
pub(crate) const INIT_USAGE: f32 = 1.0 / 65536.0;
/// Relative shift applied when a busy cluster is split in two.
pub(crate) const SPLIT_EPSILON: f32 = 1e-4;
/// Default revival threshold, as a fraction of the mean mass.
pub(crate) const UNUSED_THRESHOLD: f32 = 0.1;
/// Revival is off by default at or below this dimension.
pub(crate) const NO_REVIVAL_MAX_D: usize = 2;
/// `BatchLaw::Auto` picks the quality law at or below this dimension.
pub(crate) const AUTO_QUALITY_MAX_D: usize = 16;

// B = C * k^a * n^b * T^c
const THROUGHPUT_LAW: (f64, f64, f64, f64) = (0.035134, 0.134749, 0.876473, 0.813344);
const QUALITY_LAW: (f64, f64, f64, f64) = (0.00439175, 0.134749, 0.876473, 0.813344);
// alpha0 = C * k^a * B^b * n^c * T^e
const ALPHA_LAW: (f64, f64, f64, f64, f64) = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057);

/// How the batch size is picked when it is not given.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BatchLaw {
    /// `Quality` when `d <= 16`, `Throughput` otherwise.
    #[default]
    Auto,
    /// Bigger batches, faster training.
    Throughput,
    /// Batches about 8 times smaller, lower final error.
    Quality,
}

impl BatchLaw {
    pub(crate) fn resolve(self, d: usize) -> BatchLaw {
        match self {
            BatchLaw::Auto if d <= AUTO_QUALITY_MAX_D => BatchLaw::Quality,
            BatchLaw::Auto => BatchLaw::Throughput,
            other => other,
        }
    }
}

/// The training plan resolved for one run.
#[derive(Clone, Debug, PartialEq)]
pub struct Recipe {
    /// The law that picked the batch size, `None` when the batch size was given.
    pub batch_law: Option<BatchLaw>,
    /// Rows per update.
    pub batch_size: usize,
    /// Starting step size.
    pub alpha0: f64,
    /// Number of passes over the data, can be fractional.
    pub nepochs: f64,
    /// Updates in one full pass.
    pub steps_per_pass: usize,
    /// Updates over the whole run.
    pub total_steps: usize,
    /// Rows to feed in each pass. The last pass is always a full one.
    pub epoch_rows: Vec<usize>,
    /// Updates in each pass.
    pub epoch_steps: Vec<usize>,
    /// Clusters with a mass below this fraction of the mean are revived. 0 turns revival off.
    pub unused_threshold: f32,
    /// Seed of the starting centroids, the shuffles and the revivals.
    pub seed: u64,
    /// Rows searched at once. Only changes memory use.
    pub chunk_size: usize,
}

pub(crate) fn round_power_of_two(value: f64) -> usize {
    if value <= 1.0 {
        return 1;
    }
    let low = 2f64.powi(value.log2().floor() as i32);
    let high = 2.0 * low;
    if value / low < high / value {
        low as usize
    } else {
        high as usize
    }
}

pub(crate) fn step_law(n: usize, k: usize, nepochs: f64, batch_size: usize) -> f64 {
    let (c, a_k, a_b, a_n, a_t) = ALPHA_LAW;
    let alpha0 = c
        * (k as f64).powf(a_k)
        * (batch_size as f64).powf(a_b)
        * (n as f64).powf(a_n)
        * nepochs.powf(a_t);
    alpha0.clamp(1e-4, 1.0)
}

pub(crate) fn batch_size_for(n: usize, k: usize, nepochs: f64, law: BatchLaw) -> usize {
    let (c, b_k, b_n, b_t) = match law {
        BatchLaw::Quality => QUALITY_LAW,
        _ => THROUGHPUT_LAW,
    };
    let raw = c * (k as f64).powf(b_k) * (n as f64).powf(b_n) * nepochs.powf(b_t);
    let mut batch = round_power_of_two(raw);
    // The two fitted recipes cap the batch differently, kept as in faiss.
    batch = match law {
        BatchLaw::Quality => batch.min(n / 2),
        _ => batch.min(round_power_of_two(n as f64 / 2.0)),
    };
    batch.min(n).max(1)
}

/// Whole passes, then the fractional part, then a last full pass.
pub(crate) fn epoch_rows(n: usize, nepochs: f64) -> Vec<usize> {
    let whole = nepochs.floor();
    let remainder = ((nepochs - whole) * n as f64).round_ties_even() as usize;
    let whole = whole as usize;
    if whole == 0 {
        return vec![remainder];
    }
    let mut rows = vec![n; whole - 1];
    if remainder > 0 {
        rows.push(remainder);
    }
    rows.push(n);
    rows
}

/// Cosine decay from `alpha0` down to `ALPHA_MIN_FACTOR * alpha0`.
pub(crate) fn scheduled_alpha(alpha0: f64, step: usize, total_steps: usize) -> f64 {
    let fraction = step.min(total_steps) as f64 / total_steps.max(1) as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos());
    alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_of_two_rounding() {
        assert_eq!(round_power_of_two(0.3), 1);
        assert_eq!(round_power_of_two(5.0), 4);
        assert_eq!(round_power_of_two(6.0), 8);
        assert_eq!(round_power_of_two(1000.0), 1024);
    }

    #[test]
    fn epoch_rows_keep_last_pass_full() {
        assert_eq!(epoch_rows(10, 3.0), vec![10, 10, 10]);
        assert_eq!(epoch_rows(10, 2.5), vec![10, 5, 10]);
        assert_eq!(epoch_rows(10, 0.5), vec![5]);
        // Python rounds ties to even.
        assert_eq!(epoch_rows(5, 1.5), vec![2, 5]);
    }

    #[test]
    fn alpha_schedule_ends_at_a_tenth() {
        assert_eq!(scheduled_alpha(0.5, 0, 100), 0.5);
        assert!((scheduled_alpha(0.5, 100, 100) - 0.05).abs() < 1e-12);
    }

    // Values printed by faiss NumpyEmeans for the README example.
    #[test]
    fn matches_faiss_readme_example() {
        let law = BatchLaw::Auto.resolve(64);
        let batch = batch_size_for(200_000, 500, 5.0, law);
        assert_eq!(batch, 16384);
        assert_eq!(step_law(200_000, 500, 5.0, batch), 1.0);
    }
}
