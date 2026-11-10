//! e-means clustering: k-means quality in about a tenth of the passes.
//!
//! This is a Rust port of the CPU version of e-means from
//! [faiss/contrib/e_means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means),
//! described in the paper [e-means: outperforming k-means (in one-tenth the
//! epochs)](https://arxiv.org/abs/2611.09501).
//!
//! e-means is mini-batch k-means where each centroid is the ratio of two moving averages,
//! one for the number of points it gets and one for their sum. The batch size and the
//! starting step size come from fitted laws, so the only setting you usually touch is the
//! number of passes.
//!
//! ```
//! use emeans::Emeans;
//!
//! let (n, d, k) = (5_000, 16, 32);
//! let x: Vec<f32> = (0..n * d).map(|i| ((i * 2654435761usize) % 1000) as f32 / 1000.0).collect();
//!
//! let model = Emeans::new(d, k).train(&x)?;
//! let (distances, labels) = model.assign(&x)?;
//! assert_eq!(labels.len(), n);
//! # Ok::<(), emeans::Error>(())
//! ```
//!
//! Data that does not fit in memory can be fed batch by batch with [`Emeans::begin`].

mod emeans;
mod error;
mod knn;
mod recipe;
mod rng;

pub use crate::emeans::{Emeans, EpochStats, Model, Trainer};
pub use crate::error::{Error, Result};
pub use crate::recipe::{BatchLaw, Recipe};
