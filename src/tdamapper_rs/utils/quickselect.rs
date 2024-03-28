fn partition<T: PartialOrd>(vec: &mut [T], k: usize) -> usize {
    if k >= vec.len() {
        return vec.len();
    }
    vec.swap(k, 0);
    let mut higher: usize = 1;
    for j in 1..vec.len() {
        if vec[j] <= vec[0] {
            vec.swap(higher, j);
            higher += 1;
        }
    }
    vec.swap(0, higher - 1);
    higher
}

pub fn quick_select<T: PartialOrd>(vec: &mut [T], k: usize) -> () {
    if k >= vec.len() {
        return;
    }
    let mut arr = &mut vec[..];
    let mut idx = k;
    loop {
        let higher = partition(arr, idx);
        if higher == idx + 1 {
            return;
        }
        else if higher > idx + 1 {
            arr = &mut arr[..higher - 1];
        } else {
            arr = &mut arr[higher..];
            idx -= higher;
        }
    }
}

#[cfg(test)]
mod tests {

    use super::quick_select;
    use super::partition;


    fn generate(n: usize, r: std::ops::Range<i32>) -> Vec<i32> {
        let mut vec: Vec<i32> = Vec::with_capacity(n);
        for _ in 0..n {
            vec.push(fastrand::i32(r.clone()))
        }
        vec
    }

    quickcheck::quickcheck! {
        fn prop_partition(v: Vec<i32>, k: usize) -> bool {
            if v.is_empty() {
                return true;
            }
            let k_mod = k % v.len();
            let mut vec: Vec<i32> = v.clone();
            let val = vec[k_mod];
            let h = partition(&mut vec, k_mod);
            assert_eq!(vec[h - 1], val);
            for i in 0..h {
                if vec[i] > vec[h - 1] {
                    return false;
                }
            }
            for i in h..vec.len() {
                if vec[i] <= vec[h - 1] {
                    return false;
                }
            }
            return true;
        }

        fn prop_quick_select(v: Vec<i32>, k: usize) -> bool {
            if v.len() < 2 {
                return true;
            }
            let k_mod = k % v.len();
            let mut vec: Vec<i32> = v.clone();
            quick_select(&mut vec, k_mod);
            let val = vec[k_mod];
            for i in 0..k_mod {
                if vec[i] > val {
                    return false;
                }
            }
            for i in k_mod..vec.len() {
                if vec[i] < val {
                    return false;
                }
            }
            return true;
        }
    }

    #[test]
    fn test_partition_empty() {
        let mut v: Vec<i32> = vec![];
        assert_eq!(partition(&mut v, 0), 0);
        assert_eq!(partition(&mut v, 1), 0);
        assert_eq!(partition(&mut v, 2), 0);
    }

    #[test]
    fn test_partition_singleton() {
        let mut v: Vec<i32> = vec![0];
        assert_eq!(partition(&mut v, 0), 1);
        assert_eq!(partition(&mut v, 1), 1);
        assert_eq!(partition(&mut v, 2), 1);
    }

    #[test]
    fn test_partition_sample() {
        let mut v = vec![1, 0];
        let k = 0;
        let val = v[k];
        let h = partition(&mut v, k);
        assert_eq!(v[h - 1], val);
        for i in 0..h {
            assert!(v[i] <= v[h - 1]);
        }
        for i in h..v.len() {
            assert!(v[i] > v[h - 1]);
        }
    }

    #[test]
    fn test_partition_random() {
        let mut v = generate(1000, 0..10);
        let k = 500;
        let val = v[k];
        let h = partition(&mut v, k);
        assert_eq!(v[h - 1], val);
        for i in 0..h {
            assert!(v[i] <= v[h - 1]);
        }
        for i in h..v.len() {
            assert!(v[i] > v[h - 1]);
        }
    }

    #[test]
    fn test_quick_select_empty() {
        let mut v: Vec<i32> = vec![];
        quick_select(&mut v, 0);
        quick_select(&mut v, 1);
        quick_select(&mut v, 2);
    }

    #[test]
    fn test_quick_select_singleton() {
        let mut v: Vec<i32> = vec![0];
        quick_select(&mut v, 0);
        quick_select(&mut v, 1);
        quick_select(&mut v, 2);
    }

    #[test]
    fn test_quick_select_sample() {
        let mut v = vec![1, 0];
        let k = 0;
        quick_select(&mut v, k);
        for i in 0..k {
            assert!(v[i] <= v[k]);
        }
        for i in k..v.len() {
            assert!(v[i] >= v[k]);
        }
    }

    #[test]
    fn test_quick_select_sample_2() {
        let mut v = vec![33, 51, 93, 55, 96, 48, 94, 42, 74, 95];
        let k = 5;
        quick_select(&mut v, k);
        for i in 0..k {
            assert!(v[i] <= v[k]);
        }
        for i in k..v.len() {
            assert!(v[i] >= v[k]);
        }
    }

    #[test]
    fn test_quick_select_random() {
        let mut v = generate(10, 0..100);
        let k = v.len() / 2;
        quick_select(&mut v, k);
        for i in 0..k {
            assert!(v[i] <= v[k]);
        }
        for i in k..v.len() {
            assert!(v[i] >= v[k]);
        }
    }

}
