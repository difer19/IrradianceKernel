import numpy as np

from irradiance_kernel.diagnostics import make_spatial_groups
from irradiance_kernel.evaluation import metric_bundle


def test_metric_bundle_explains_weakest_class():
    metrics = metric_bundle(
        np.array([0, 0, 1, 1]),
        np.array([0, 0, 0, 0]),
        detailed=True,
    )
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["min_class_f1"] == 0.0
    assert [row["support"] for row in metrics["per_class"]] == [2, 2]


def test_spatial_groups_are_reproducible():
    coordinates = np.array(
        [[0, 0], [0, 1], [10, 10], [10, 11], [20, 20], [20, 21]], dtype=float
    )
    X = np.column_stack([coordinates, np.ones((6, 7))])
    first = make_spatial_groups(X)
    second = make_spatial_groups(X)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 3
