#!/usr/bin/env python3
"""Paso 2: prueba 3 modelos × 9 kernels sin generar artefactos finales."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irradiance_kernel.constants import FEATURE_COLUMNS, KERNELS, TARGET_COLUMN
from irradiance_kernel.data import load_dataset
from irradiance_kernel.evaluation import ExperimentConfig, evaluate_configuration


def main() -> None:
    frame = load_dataset("landsat").iloc[:180]
    X = frame[FEATURE_COLUMNS].to_numpy(float)
    y = frame[TARGET_COLUMN].to_numpy(float)
    splits = list(KFold(2, shuffle=True, random_state=2021).split(X))
    completed = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for model in ("ksvc", "kannc", "kridge"):
            for kernel in KERNELS:
                config = ExperimentConfig("standard", "uniform", "pca", model, kernel)
                result = evaluate_configuration(config, X, y, splits)
                detail = "OK" if result["status"] == "ok" else result["error"]
                print(f"{model:7} {kernel:14} {detail}")
                completed += result["status"] == "ok"
    print(f"Resultado: {completed}/27 configuraciones correctas")
    if completed != 27:
        raise RuntimeError("La matriz pequeña no terminó completamente")


if __name__ == "__main__":
    main()
