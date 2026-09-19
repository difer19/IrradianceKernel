#!/usr/bin/env python3
"""Paso 1: verifica e instancia las clases originales sin modificarlas."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "references" / "original"
sys.path.insert(0, str(ORIGINAL))
sys.path.insert(0, str(ROOT / "src"))

from irradiance_kernel.constants import FEATURE_COLUMNS, TARGET_COLUMN
from irradiance_kernel.data import load_dataset
from KANN import KANNC
from KSVM import KSVC

EXPECTED_HASHES = {
    "KSVM.py": "5569fc5c5498f98742a2b2003e0f43e1d951157ba191443380c4453b14ba362f",
    "KANN.py": "5c2b954d4a15f32d5c546ef1d68a27e96b7449d61cbc35c9ec49724d6b2dc3c6",
    "KernelUtilities.py": "0e3414053a387bb73ccfac24de3a50eba7f70c79f5ff502c50119cc6cefb3053",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for filename, expected in EXPECTED_HASHES.items():
        actual = sha256(ORIGINAL / filename)
        if actual != expected:
            raise RuntimeError(f"{filename} fue modificado: {actual}")
        print(f"{filename}: copia exacta ({actual})")

    np.random.seed(2021)
    frame = load_dataset("landsat").iloc[:180]
    X = StandardScaler().fit_transform(frame[FEATURE_COLUMNS].to_numpy(float))
    values = frame[TARGET_COLUMN].to_numpy(float)
    edges = np.linspace(values.min(), values.max(), 6)
    y = np.clip(np.digitize(values, edges[1:-1]), 0, 4)

    estimators = {
        "KSVC original": KSVC(
            C=1.0, kernel="linear", degree=2, gamma="scale", random_state=2021
        ),
        "KANNC original": KANNC(
            kernel="linear",
            degree=2,
            gamma=0.1,
            max_iter=50,
            early_stopping=True,
            random_state=2021,
        ),
    }
    for name, estimator in estimators.items():
        fitted = clone(estimator).fit(X[:140], y[:140])
        prediction = fitted.predict(X[140:])
        print(
            f"{name}: fit/predict correcto; módulo={type(fitted).__module__}; "
            f"predicciones={prediction.shape}"
        )


if __name__ == "__main__":
    main()
