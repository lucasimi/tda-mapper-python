use pyo3::prelude::*;

#[pyfunction]
fn foo() {
    println!("Foo!");
}

/// The name of this function must match `lib.name` in `Cargo.toml`
#[pymodule]
fn tdamapper_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(foo, m)?)?;
    Ok(())
}
