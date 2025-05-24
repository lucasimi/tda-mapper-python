from tdamapper.utils.unionfind import UnionFind


def test_list():
    data = [1, 2, 3, 4]
    uf = UnionFind(data)
    for i in data:
        assert i == uf.find(i)
    j = uf.union(1, 2)
    assert j == uf.find(1)
    assert j == uf.find(2)
    k = uf.union(3, 4)
    assert k == uf.find(3)
    assert k == uf.find(4)
