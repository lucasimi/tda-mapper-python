use crate::utils::quickselect;
use fastrand;

pub type Scalar = f32;

pub trait Metric<T> {
    fn apply(&self, x:&T, y:&T) -> Scalar;
}

impl<T, F> Metric<T> for F where F:Fn(&T, &T) -> Scalar {
    fn apply(&self, x:&T, y:&T) -> Scalar {
        (self)(x, y)
    }
}

#[derive(Debug)]
struct VP {
    center: usize,
    radius: Scalar
}

impl VP {

    fn new(center: usize) -> VP {
        VP { center: center, radius: 0.0 }
    }

}

impl PartialEq for VP {

    fn eq(&self, other: &Self) -> bool {
        self.radius == other.radius
    }

}

impl PartialOrd for VP {

    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.radius.partial_cmp(&other.radius)
    }

}

fn get_mid<T>(arr: &[T]) -> usize {
    arr.len() / 2
}

pub struct VPTree<T, M> {
    items: Vec<T>,
    vps: Vec<VP>,
    metric: M
}

impl<T, M: Metric<T>> VPTree<T, M> {

    pub fn build(items: Vec<T>, metric: M) -> VPTree<T, M> {
        let mut vps: Vec<VP> = items.iter().enumerate()
            .map(|(x, _)| VP::new(x))
            .collect();
        Self::build_iter(&items, &mut vps, &metric);
        VPTree { 
            items: items,
            vps: vps,
            metric: metric
        }
    }

    fn build_iter(items: &[T], vps: &mut [VP], metric: &M) -> () {
        let mut stack: Vec<&mut [VP]> = Vec::with_capacity(vps.len());
        if !vps.is_empty() {
            stack.push(vps);
        }
        while let Some(vec) = stack.pop() {
            Self::process_slice(items, vec, metric);
            let tail = &mut vec[1..];
            let mid_tail = get_mid(tail);
            let (left, right) = tail.split_at_mut(mid_tail);
            if !left.is_empty() {
                stack.push(left);
            }
            if !right.is_empty() {
                stack.push(right);
            }
        }
    }

    fn process_slice(items: &[T], vec: &mut [VP], metric: &M) -> () {
        let pivot: usize = fastrand::usize(0..vec.len());
        vec.swap(pivot, 0);
        let vp = &items[vec[0].center];
        for i in 0..vec.len() {
            let p = &items[vec[i].center];
            vec[i].radius = metric.apply(vp, p);
        }
        let tail = &mut vec[1..];
        if !tail.is_empty() {
            let tail_mid: usize = get_mid(tail);
            quickselect::quick_select(tail, tail_mid);
            vec[0].radius = tail[tail_mid].radius;
        }
    }

    pub fn ball_search(&self, target: &T, eps: Scalar) -> Vec<usize> {
        let mut results: Vec<usize> = vec![];
        let mut stack: Vec<&[VP]> = Vec::with_capacity(self.items.len());
        if !self.items.is_empty() {
            stack.push(&self.vps);
        }
        while let Some(vec) = stack.pop() {
            let center = &vec[0];
            let vp = &self.items[center.center];
            let dist = self.metric.apply(vp, target);
            let tail = &vec[1..];
            let tail_mid = get_mid(tail);
            if dist <= eps {
                results.push(center.center);
            }
            if dist < center.radius + eps {
                let left = &tail[..tail_mid];
                if !left.is_empty() {
                    stack.push(left);
                }
            } 
            if dist >= center.radius - eps {
                let right = &tail[tail_mid..];
                if !right.is_empty() {
                    stack.push(right);
                }
            }
        }
        results
    }

}



#[cfg(test)]
mod tests {

    use crate::utils::vptree::Scalar;
    use crate::utils::vptree::Metric;
    use crate::utils::vptree::VP;
    use crate::utils::vptree::VPTree;
    use std::collections::HashSet;

    use super::get_mid;

    fn absdist(n: &i32, m: &i32) -> Scalar {
        (*n).saturating_sub(*m).saturating_abs() as f32
    }

    fn search_naive<T>(vec: &[T], dist: &dyn Metric<T>, target: &T, eps: Scalar) -> Vec<usize> {
        return vec.iter().enumerate()
            .filter(|(_, x)| dist.apply(target, x) <= eps)
            .map(|(i, _)| i)
            .collect();
    }

    fn check_vptree<T, M: Metric<T>>(vpt: &VPTree<T, M>) -> bool {
        let mut stack: Vec<&[VP]> = Vec::with_capacity(vpt.items.len());
        if !vpt.items.is_empty() {
            stack.push(&vpt.vps);
        }
        while let Some(v) = stack.pop() {
            if !v.is_empty() {
                let mid = 1 + get_mid(&v[1..]);
                let vp = &vpt.items[v[0].center];
                for i in 1..mid {
                    let p = &vpt.items[v[i].center];
                    if vpt.metric.apply(vp, p) > v[0].radius {
                        return false;
                    }
                }
                for i in mid..v.len() {
                    let p = &vpt.items[v[i].center];
                    if vpt.metric.apply(vp, p) < v[0].radius {
                        return false;
                    }
                }
                let left = &v[1..mid];
                let right = &v[mid..];
                if !left.is_empty() {
                    stack.push(left);
                }
                if !right.is_empty() {
                    stack.push(right);
                }
            }
        }
        return true;
    }

    #[test]
    fn test_build_empty() {
        let vec = vec![];
        let vpt = VPTree::build(vec, absdist);
        assert!(vpt.items.is_empty());
        assert!(check_vptree(&vpt));
        assert!(vpt.ball_search(&0, 1.0).is_empty());
    }

    #[test]
    fn test_build_sample() {
        let vec = vec![-1, 4, -4, 1, 2, -3];
        let vpt = VPTree::build(vec, absdist);
        assert!(check_vptree(&vpt));
        assert_eq!(vpt.ball_search(&-1, 1.0).len(), 1);
        assert_eq!(vpt.ball_search(&4, 1.0).len(), 1);
        assert_eq!(vpt.ball_search(&-4, 1.0).len(), 2);
        assert_eq!(vpt.ball_search(&1, 1.0).len(), 2);
        assert_eq!(vpt.ball_search(&2, 1.0).len(), 2);
        assert_eq!(vpt.ball_search(&-3, 1.0).len(), 2);
        assert_eq!(vpt.ball_search(&0, 1.0).len(), 2);
    }

    quickcheck::quickcheck! {
        fn prop_build(vec: Vec<i32>) -> bool {
            let vpt = VPTree::build(vec, absdist);
            check_vptree(&vpt)
        }

        fn prop_search(vec: Vec<i32>, target: i32, eps: Scalar) -> bool {
            let vpt = VPTree::build(vec.clone(), absdist);
            let v1 = search_naive(&vec, &absdist, &target, eps);
            let v2 = vpt.ball_search(&target, eps);
            let h1: HashSet<usize> = HashSet::from_iter(v1);
            let h2: HashSet<usize> = HashSet::from_iter(v2);
            return h1 == h2;
        }
    }

}
