use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2};
use numpy::ndarray::Axis;
use pyo3::types::PyDict;

pub mod utils;

type Scalar = f32;


#[pyclass]
struct VPTree {
    vpt: utils::vptree::VPTree<Vec<f64>, WKMetric>
}

struct Euclidean {}
struct Chebyshev {}

struct Lp {p: f64}

enum WKMetric {
    Euclidean(Euclidean),
    Chebyshev(Chebyshev),
    Lp(Lp)
}

impl utils::vptree::Metric<Vec<f64>> for WKMetric {
    fn apply(&self, x:&Vec<f64>, y:&Vec<f64>) -> utils::vptree::Scalar {
        match self {
            WKMetric::Euclidean(m) => m.apply(x, y),
            WKMetric::Chebyshev(m) => m.apply(x, y),
            WKMetric::Lp(m) => m.apply(x, y)
        }
    }
}

impl utils::vptree::Metric<Vec<f64>> for Euclidean {

    fn apply(&self, x:&Vec<f64>, y:&Vec<f64>) -> utils::vptree::Scalar {
        dist_lp(2f64, x, y)
    }

}

impl utils::vptree::Metric<Vec<f64>> for Chebyshev {

    fn apply(&self, x:&Vec<f64>, y:&Vec<f64>) -> utils::vptree::Scalar {
        dist_inf(x, y)
    }

}

impl utils::vptree::Metric<Vec<f64>> for Lp {

    fn apply(&self, x:&Vec<f64>, y:&Vec<f64>) -> utils::vptree::Scalar {
        dist_lp(self.p, x, y)
    }

}

fn dist_inf(a: &Vec<f64>, b: &Vec<f64>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .max_by(|x, y| x.total_cmp(y))
        .unwrap_or(f64::NAN) as f32
}

fn dist_lp(p: f64, a: &Vec<f64>, b: &Vec<f64>) -> f32 {
    let s: f64 = a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs().powf(p))
        .sum();
    s.powf(1.0f64 / p) as f32
}

fn to_vec(arr: &PyArray2<f64>) -> Vec<Vec<f64>> {
    arr.readonly().as_array().axis_iter(Axis(0))
        .map(|row| row.to_vec())
        .collect()
}

#[pymethods]
impl VPTree {

    #[new]
    //#[pyo3(signature = (items=&PyArray2::new(), metric="euclidean", **kwargs))]
    fn init(items: &PyArray2<f64>, metric: &str, kwargs: Option<&PyDict>) -> Self {
        let mtr: WKMetric = match metric {
            "euclidean" => WKMetric::Euclidean(Euclidean {}),
            "chebyshev" => WKMetric::Chebyshev(Chebyshev {}),
            "lp" => match kwargs {
                None => WKMetric::Euclidean(Euclidean {}),
                Some(dict) => {
                    match dict.get_item("p") {
                        Ok(Some(p)) => {
                            match p.extract::<f64>() {
                                Ok(p_f64) => WKMetric::Lp(Lp {p: p_f64}),
                                _ => WKMetric::Euclidean(Euclidean {})
                            }
                        }
                        _ => WKMetric::Euclidean(Euclidean {})
                    } 
                }
            },
            _ => WKMetric::Euclidean(Euclidean {})
        };

        let v: Vec<Vec<f64>> = to_vec(items);
        VPTree { vpt: utils::vptree::VPTree::build(v, mtr) }
    }

    fn ball_search(&self, point: &PyArray1<f64>, eps: Scalar) -> Vec<usize> {
        let v = point.readonly().to_vec();
        let u: Vec<usize> = match v {
            Ok(vv) => {
                let t = self.vpt.ball_search(&vv, eps);
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

    #[pyfn(m)]
    fn to_arr(py: Python, vec: Vec<f64>) -> &PyArray1<f64> {
        PyArray1::from_vec(py, vec)
    }



    m.add_function(wrap_pyfunction!(quick_select, m)?)?;
    m.add_class::<VPTree>()?;
    Ok(())
}
