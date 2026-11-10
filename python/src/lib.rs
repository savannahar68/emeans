//! Python module `emeans._emeans`. The friendly class lives in `python/emeans/__init__.py`.

use emeans::{BatchLaw, Emeans, Error, Model, Trainer};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Distances (or similarities) and labels returned by `assign`.
type Assigned<'py> = (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i64>>);

fn to_py(err: Error) -> PyErr {
    match err {
        Error::InvalidInput(msg) => PyValueError::new_err(msg),
        Error::WrongOrder(msg) => PyRuntimeError::new_err(msg),
    }
}

fn rows<'a>(x: &'a PyReadonlyArray2<'_, f32>, d: usize, what: &str) -> PyResult<&'a [f32]> {
    if x.shape()[1] != d {
        return Err(PyValueError::new_err(format!(
            "{what} has {} columns, expected d = {d}",
            x.shape()[1]
        )));
    }
    x.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{what} must be C contiguous")))
}

fn matrix<'py>(py: Python<'py>, values: &[f32], d: usize) -> PyResult<Bound<'py, PyArray2<f32>>> {
    PyArray1::from_slice(py, values).reshape([values.len() / d, d])
}

#[pyclass(module = "emeans._emeans")]
struct Native {
    config: Emeans,
    d: usize,
    model: Option<Model>,
    trainer: Option<Trainer>,
}

#[pymethods]
impl Native {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        d: usize,
        k: usize,
        nepochs: f64,
        batch_size: Option<usize>,
        alpha0: Option<f64>,
        batch_law: &str,
        seed: u64,
        verbose: bool,
        chunk_size: usize,
        capture_assignments: bool,
        spherical: bool,
        unused_threshold: Option<f32>,
        max_points_per_centroid: Option<usize>,
    ) -> PyResult<Self> {
        let law = match batch_law {
            "auto" => BatchLaw::Auto,
            "throughput" => BatchLaw::Throughput,
            "quality" => BatchLaw::Quality,
            other => {
                return Err(PyValueError::new_err(format!(
                    "batch_law must be \"auto\", \"throughput\" or \"quality\", got {other:?}"
                )));
            }
        };
        let mut config = Emeans::new(d, k)
            .nepochs(nepochs)
            .batch_law(law)
            .seed(seed)
            .verbose(verbose)
            .chunk_size(chunk_size)
            .capture_assignments(capture_assignments)
            .spherical(spherical);
        if let Some(b) = batch_size {
            config = config.batch_size(b);
        }
        if let Some(a) = alpha0 {
            config = config.alpha0(a);
        }
        if let Some(t) = unused_threshold {
            config = config.unused_threshold(t);
        }
        if let Some(m) = max_points_per_centroid {
            config = config.max_points_per_centroid(m);
        }
        Ok(Native {
            config,
            d,
            model: None,
            trainer: None,
        })
    }

    #[pyo3(signature = (x, init_centroids=None))]
    fn train(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        init_centroids: Option<PyReadonlyArray2<'_, f32>>,
    ) -> PyResult<()> {
        let d = self.d;
        let data = rows(&x, d, "x")?;
        let init = match &init_centroids {
            Some(init) => Some(rows(init, d, "init_centroids")?),
            None => None,
        };
        let config = &self.config;
        let model = py
            .detach(|| match init {
                Some(init) => config.train_from(data, init),
                None => config.train(data),
            })
            .map_err(to_py)?;
        self.model = Some(model);
        self.trainer = None;
        Ok(())
    }

    fn assign<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Assigned<'py>> {
        let model = self.model()?;
        let data = rows(&x, self.d, "x")?;
        let (dist, labels) = py.detach(|| model.assign(data)).map_err(to_py)?;
        Ok((PyArray1::from_vec(py, dist), PyArray1::from_vec(py, labels)))
    }

    fn centroids<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
        match &self.model {
            Some(m) => Ok(Some(matrix(py, m.centroids(), self.d)?)),
            None => Ok(None),
        }
    }

    fn labels<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<i64>>> {
        let labels = self.model.as_ref()?.labels()?;
        Some(PyArray1::from_slice(py, labels))
    }

    fn n_splits(&self) -> usize {
        match (&self.model, &self.trainer) {
            (Some(m), _) => m.n_splits(),
            (None, Some(t)) => t.n_splits(),
            _ => 0,
        }
    }

    fn recipe<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let recipe = match (&self.trainer, &self.model) {
            (Some(t), _) => t.recipe(),
            (None, Some(m)) => match m.recipe() {
                Some(r) => r,
                None => return Ok(None),
            },
            _ => return Ok(None),
        };
        let dict = PyDict::new(py);
        let law = recipe.batch_law.map(|l| match l {
            BatchLaw::Quality => "quality",
            _ => "throughput",
        });
        dict.set_item("batch_law", law)?;
        dict.set_item("batch_size", recipe.batch_size)?;
        dict.set_item("alpha0", recipe.alpha0)?;
        dict.set_item("nepochs", recipe.nepochs)?;
        dict.set_item("steps_per_pass", recipe.steps_per_pass)?;
        dict.set_item("total_steps", recipe.total_steps)?;
        dict.set_item("epoch_rows", recipe.epoch_rows.clone())?;
        dict.set_item("epoch_steps", recipe.epoch_steps.clone())?;
        dict.set_item("unused_threshold", recipe.unused_threshold)?;
        dict.set_item("seed", recipe.seed)?;
        dict.set_item("assignment_chunk_size", recipe.chunk_size)?;
        Ok(Some(dict))
    }

    fn iteration_stats<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let Some(model) = &self.model else {
            return Ok(Vec::new());
        };
        model
            .epoch_stats()
            .iter()
            .map(|s| {
                let dict = PyDict::new(py);
                dict.set_item("obj", s.objective)?;
                dict.set_item("time", s.time)?;
                dict.set_item("imbalance_factor", s.imbalance_factor)?;
                dict.set_item("nsplit", s.n_splits)?;
                Ok(dict)
            })
            .collect()
    }

    fn begin(&mut self, n: usize, init_centroids: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        if self.trainer.is_some() || self.model.is_some() {
            return Err(PyRuntimeError::new_err(
                "create a new object for another run",
            ));
        }
        let init = rows(&init_centroids, self.d, "init_centroids")?;
        self.trainer = Some(self.config.begin(n, init).map_err(to_py)?);
        Ok(())
    }

    fn accumulate<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let d = self.d;
        let data = rows(&x, d, "batch")?;
        let trainer = self.trainer()?;
        let mut labels = vec![0i64; data.len() / d];
        py.detach(|| trainer.accumulate_labeled(data, &mut labels))
            .map_err(to_py)?;
        Ok(PyArray1::from_vec(py, labels))
    }

    fn update(&mut self) -> PyResult<()> {
        self.trainer()?.update().map_err(to_py)
    }

    fn end_epoch(&mut self) -> PyResult<()> {
        self.trainer()?.end_epoch().map_err(to_py)
    }

    fn finalize(&mut self) -> PyResult<()> {
        if self.model.is_some() && self.trainer.is_none() {
            return Ok(());
        }
        let trainer = self
            .trainer
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("call begin first"))?;
        self.model = Some(trainer.finalize().map_err(to_py)?);
        Ok(())
    }

    fn current_centroids<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let d = self.d;
        match (&self.trainer, &self.model) {
            (Some(t), _) => matrix(py, t.centroids(), d),
            (None, Some(m)) => matrix(py, m.centroids(), d),
            _ => Err(PyRuntimeError::new_err("call train or begin first")),
        }
    }
}

impl Native {
    fn model(&self) -> PyResult<&Model> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call train or finalize first"))
    }

    fn trainer(&mut self) -> PyResult<&mut Trainer> {
        self.trainer
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("call begin first"))
    }
}

#[pymodule]
fn _emeans(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Native>()?;
    Ok(())
}
