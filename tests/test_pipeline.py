import numpy as np
import pytest
from sklearn.base import clone

from irradiance_kernel.data import load_dataset
from irradiance_kernel.pipeline import IrradiancePipeline


@pytest.mark.parametrize("model", ["ksvc", "kannc", "kridge"])
@pytest.mark.parametrize("reducer", ["pca", "lda"])
def test_pipeline_smoke(model, reducer):
    frame = load_dataset("landsat").iloc[:180]
    X = frame.drop(columns="value").to_numpy()
    y = frame["value"].to_numpy()
    estimator = IrradiancePipeline(
        scaler="standard",
        discretizer="uniform",
        reducer=reducer,
        model=model,
        kernel="rbf",
    )
    fitted = clone(estimator).fit(X[:140], y[:140])
    expected = fitted.transform_target(y[140:])
    predicted = fitted.predict(X[140:])
    assert expected.shape == predicted.shape


def test_validation_values_do_not_define_uniform_edges():
    frame = load_dataset("landsat").iloc[:120]
    X = frame.drop(columns="value").to_numpy()
    y = frame["value"].to_numpy()
    estimator = IrradiancePipeline().fit(X[:100], y[:100])
    assert estimator.discretizer_.edges_[0] == pytest.approx(y[:100].min())
    assert estimator.discretizer_.edges_[-1] == pytest.approx(y[:100].max())


@pytest.mark.parametrize("model,module", [("ksvc", "KSVM"), ("kannc", "KANN")])
def test_pipeline_uses_original_repository_classes(model, module):
    frame = load_dataset("landsat").iloc[:160]
    X = frame.drop(columns="value").to_numpy()
    y = frame["value"].to_numpy()
    fitted = IrradiancePipeline(
        scaler="standard",
        discretizer="uniform",
        reducer="pca",
        model=model,
        kernel="linear",
    ).fit(X[:130], y[:130])
    assert type(fitted.model_).__module__ == module
