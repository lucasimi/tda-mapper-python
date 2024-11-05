import unittest
import numpy as np

from tdamapper.cover import KNNCover
from tdamapper.utils.metrics import euclidean
from tdamapper.utils.vptree_flat import VPTree


X = np.array([
    [ 99.30543214,  99.85036546],
    [ 98.82687659, 101.94362119],
    [ 99.260437  , 101.5430146 ],
    [ 99.5444675 , 100.01747916],
    [ 99.49034782,  99.5619257 ],
    [ 99.73199663, 100.8024564 ],
    [100.3130677 ,  99.14590426],
    [101.18802979, 100.31694261],
    [ 98.96575716, 100.68159452],
    [101.8831507 ,  98.65224094],
    [ 99.59682305, 101.22244507],
    [ 99.32566734, 100.03183056],
    [100.04575852,  99.81281615],
    [101.48825219, 101.89588918],
    [100.15494743, 100.37816252],
    [ 98.95144703,  98.57998206],
    [101.86755799,  99.02272212],
    [100.72909056, 100.12898291],
    [ 99.65208785, 100.15634897],
    [ 99.97181777, 100.42833187],
    [ 99.36567791,  99.63725883],
    [ 99.89678115, 100.4105985 ],
    [100.46566244,  98.46375631],
    [ 99.25524518,  99.17356146],
    [100.92085882, 100.31872765],
    [100.12691209, 100.40198936],
    [101.53277921, 101.46935877],
    [101.12663592,  98.92006849],
    [ 99.61267318,  99.69769725],
    [100.17742614,  99.59821906],
    [ 98.74720464, 100.77749036],
    [ 99.09270164, 100.0519454 ],
    [ 99.13877431, 101.91006495],
    [ 97.44701018, 100.6536186 ],
    [ 99.10453344, 100.3869025 ],
    [ 99.12920285,  99.42115034],
    [ 99.90154748,  99.33652171],
    [ 98.99978465,  98.4552289 ],
    [ 98.85253135,  99.56217996],
    [100.20827498, 100.97663904],
    [ 99.96071718,  98.8319065 ],
    [101.49407907,  99.79484174],
    [ 99.50196755, 101.92953205],
    [100.40234164,  99.31518991],
    [ 98.29372981, 101.9507754 ],
    [100.1666735 , 100.63503144],
    [ 99.32753955,  99.64044684],
    [100.14404357, 101.45427351],
    [ 99.3563816 ,  97.77659685],
    [ 98.50874241, 100.4393917 ],
    [ 99.36415392, 100.67643329],
    [102.38314477, 100.94447949],
    [100.44386323, 100.33367433],
    [100.94942081, 100.08755124],
    [100.85683061,  99.34897441],
    [ 99.58638102,  99.25254519],
    [ 99.08717777, 101.11701629],
    [ 98.729515  , 100.96939671],
    [ 99.48919486,  98.81936782],
    [100.57659082,  99.79170124],
    [100.06651722, 100.3024719 ],
    [ 98.36980165, 100.46278226],
    [100.37642553,  98.90059921],
    [ 98.83485016, 100.90082649],
    [ 98.70714309, 100.26705087],
    [100.76103773, 100.12167502],
    [100.39600671,  98.90693849],
    [101.13940068,  98.76517418],
    [ 98.89561666, 100.05216508],
    [ 98.38610215,  99.78725972],
    [ 99.68844747, 100.05616534],
    [101.76405235, 100.40015721],
    [100.97873798, 102.2408932 ],
    [100.94725197,  99.84498991],
    [102.26975462,  98.54563433],
    [100.95008842,  99.84864279],
    [101.17877957,  99.82007516],
    [100.52327666,  99.82845367],
    [100.77179055, 100.82350415],
    [ 98.68409259,  99.5384154 ]
])


x = np.array([ 99.73199663, 100.8024564 ])


class TestKNN(unittest.TestCase):

    def test_knn_search(self):
        knn_cover = KNNCover(neighbors=5, metric='euclidean')
        knn_cover.fit(X)
        neigh_ids = knn_cover.search(x)
        d = euclidean()
        dists = [d(x, X[j]) for j in neigh_ids]
        x_dist = d(x, X[5])
        self.assertTrue(x_dist in dists)

    def test_vptree(self):
        vptree = VPTree(X[:80], metric='euclidean', leaf_capacity=5)
        neigh = vptree.knn_search(x, 5)
        d = euclidean()
        dists = [d(x, y) for y in neigh]
        x_dist = d(x, X[5])
        self.check_vptree(vptree)
        self.assertTrue(x_dist in dists)

    def test_vptree_simple(self):
        XX = np.array([np.array([x, x/2]) for x in range(30)])
        vptree = VPTree(XX, metric='euclidean', leaf_capacity=5, leaf_radius=0.0)
        xx = np.array([3, 3/2])
        neigh = vptree.knn_search(xx, 2)
        d = euclidean()
        dists = [d(xx, y) for y in neigh]
        self.check_vptree(vptree)
        self.assertTrue(0.0 in dists)

    def check_vptree(self, vpt):
        data = vpt._get_dataset()
        dist = vpt._get_distance()
        leaf_capacity = vpt.get_leaf_capacity()
        leaf_radius = vpt.get_leaf_radius()

        def check_sub(start, end):
            v_radius, v_point, *_ = data[start]
            mid = (start + end) // 2
            for i in range(start + 1, mid):
                _, y, *_ = data[i]
                self.assertTrue(dist(v_point, y) <= v_radius)
            for i in range(mid, end):
                _, y, *_ = data[i]
                self.assertTrue(dist(v_point, y) >= v_radius)

        def check_rec(start, end):
            v_radius, *_ = data[start]
            if (end - start > leaf_capacity) and (v_radius > leaf_radius):
                check_sub(start, end)
                mid = (start + end) // 2
                check_rec(start + 1, mid)
                check_rec(mid, end)
        check_rec(0, len(data))
            
            