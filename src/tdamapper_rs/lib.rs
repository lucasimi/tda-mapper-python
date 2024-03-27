use pyo3::prelude::*;

pub mod utils;
use crate::utils::quickselect;

#[pyfunction]
fn foo() {
    println!("Foo!");
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
    m.add_function(wrap_pyfunction!(foo, m)?)?;
    m.add_function(wrap_pyfunction!(quick_select, m)?)?;
    Ok(())
}
