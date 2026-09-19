import numpy as np
import pytest
from sklearn.base import clone

from irradiance_kernel.discretizers import make_discretizer


@pytest.mark.parametrize("name", ["uniform", "kmeans", "dbscan", "agglomerative"])
def test_discretizer_orders_classes_and_handles_new_values(name):
    rng = np.random.default_rng(2021)
    y = np.r_[rng.normal(195, 0.8, 45), rng.normal(210, 0.8, 45), rng.normal(232, 0.8, 45)]
    discretizer = clone(make_discretizer(name)).fit(y)
    labels = discretizer.transform([190, 200, 220, 240])
    assert np.all(np.diff(labels) >= 0)
    assert len(discretizer.class_descriptions()) >= 2


def test_uniform_clips_values_outside_training_range():
    discretizer = make_discretizer("uniform").fit([0, 1, 2, 3, 4, 5])
    assert discretizer.transform([-10, 20]).tolist() == [0, 4]
