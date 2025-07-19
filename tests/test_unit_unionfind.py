import pytest

from tdamapper.utils.unionfind import UnionFind


@pytest.mark.parametrize(
    "datasets, expected_components",
    [
        ([[], []], 0),
        ([[], [1]], 1),
        ([[], [1, 2]], 1),
        ([[1, 2], [3, 4]], 2),
        ([[1, 2, 3], [3, 4]], 1),
    ],
)
def test_unionfind(datasets, expected_components):
    all_data = []
    for dataset in datasets:
        all_data.extend(dataset)
    uf = UnionFind(all_data)
    for dataset in datasets:
        if dataset:
            for x, y in zip(dataset, dataset[1:]):
                uf.union(x, y)

    components = set()
    for x in all_data:
        components.add(uf.find(x))

    assert len(components) == expected_components
