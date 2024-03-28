use pyo3::prelude::*;

pub mod utils;

type Scalar = f32;

#[pyclass]
struct VPTree {
    vpt: utils::vptree::VPTree<f64>
}

fn absdist(n: &f64, m: &f64) -> Scalar {
    (*n - *m).abs() as f32
}

#[pymethods]
impl VPTree {

    #[new]
    fn init(items: Vec<f64>) -> Self {
        VPTree { vpt: utils::vptree::build(items, absdist) }
    }

    fn search(&self, target: f64, radius: Scalar) -> Vec<usize> {
        utils::vptree::search(&self.vpt, &target, radius)
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
fn tdamapper_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quick_select, m)?)?;
    m.add_class::<VPTree>()?;
    Ok(())
}
