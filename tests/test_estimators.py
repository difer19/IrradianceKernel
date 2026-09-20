import joblib
import numpy as np
import pytest
from sklearn.base import clone

from irradiance_kernel.estimators import KANNC, KSVC, KRidgeClassifier


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(90, 5))
    y = np.digitize(X[:, 0] + 0.4 * X[:, 1], [-0.6, 0.4])
    return X, y


@pytest.mark.parametrize(
    "estimator",
    [
        KSVC(kernel="rbf", gamma=0.2, random_state=2021),
        KANNC(kernel="linear", max_iter=80, random_state=2021),
        KRidgeClassifier(kernel="rbf"),
    ],
)
def test_estimators_clone_predict_and_serialize(estimator, classification_data, tmp_path):
    X, y = classification_data
    fitted = clone(estimator).fit(X, y)
    predictions = fitted.predict(X[:8])
    assert predictions.shape == (8,)
    if hasattr(fitted, "decision_function"):
        assert fitted.decision_function(X[:8]).shape == (8, len(fitted.classes_))
    path = tmp_path / "model.joblib"
    joblib.dump(fitted, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(predictions, restored.predict(X[:8]))
