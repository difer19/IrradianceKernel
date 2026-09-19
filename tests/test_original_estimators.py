import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "references" / "original"))

from KANN import KANNC
from KSVM import KSVC


def test_original_classes_are_imported_directly():
    assert KSVC.__module__ == "KSVM"
    assert KANNC.__module__ == "KANN"


def test_original_classes_fit_and_predict():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(120, 9))
    y = np.repeat(np.arange(3), 40)
    estimators = [
        KSVC(C=1.0, kernel="linear", degree=2, gamma="scale", random_state=2021),
        KANNC(
            kernel="linear",
            degree=2,
            gamma=0.1,
            max_iter=50,
            early_stopping=True,
            random_state=2021,
        ),
    ]
    for estimator in estimators:
        prediction = clone(estimator).fit(X[:100], y[:100]).predict(X[100:])
        assert prediction.shape == (20,)
