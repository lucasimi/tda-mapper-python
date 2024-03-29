use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2};
use numpy::ndarray::Axis;

pub mod utils;

type Scalar = f32;

#[pyclass]
struct VPTree {
    vpt: utils::vptree::VPTree<Vec<f64>>
}

fn dist_inf(a: &Vec<f64>, b: &Vec<f64>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .max_by(|x, y| x.total_cmp(y))
        .unwrap_or(f64::NAN) as f32
}

fn dist_lp(p: i32, a: &Vec<f64>, b: &Vec<f64>) -> f32 {
    let s: f64 = a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(p))
        .sum();
    s.powf(1.0f64 / (p as f64)) as f32
}

fn euclidean(a: &Vec<f64>, b: &Vec<f64>) -> f32 {
    dist_lp(2, a, b)
}

fn to_vec(arr: &PyArray2<f64>) -> Vec<Vec<f64>> {
    arr.readonly().as_array().axis_iter(Axis(0))
        .map(|row| row.to_vec())
        .collect()
}

#[pymethods]
impl VPTree {

    #[new]
    fn init(items: &PyArray2<f64>) -> Self {
        let v: Vec<Vec<f64>> = to_vec(items);
        VPTree { vpt: utils::vptree::build(v, euclidean) }
    }

    fn search(&self, target: &PyArray1<f64>, radius: Scalar) -> Vec<usize> {
        let v = target.readonly().to_vec();
        let u: Vec<usize> = match v {
            Ok(vv) => {
                let t = utils::vptree::search(&self.vpt, &vv, radius);
                t
            },
            Err(_) => Vec::new()
        };
        u
    }

}

#[pyfunction]
fn quick_select(vec: Vec<f64>, k: usize) -> Vec<f64> {
    let mut v = vec.clone();
    crate::utils::quickselect::quick_select(&mut v, k);
    return v;
}

/// The name of this function must match `lib.name` in `Cargo.toml`
#[pymodule]
fn tdamapper_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quick_select, m)?)?;
    m.add_class::<VPTree>()?;
    Ok(())
}
