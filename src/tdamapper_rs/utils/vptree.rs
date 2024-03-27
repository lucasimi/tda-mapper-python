use crate::utils::quickselect;

pub fn foo() {
    let mut v = vec![1, 2, 3];
    quickselect::quick_select(&mut v, 0)
}