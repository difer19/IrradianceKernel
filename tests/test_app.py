import json
from pathlib import Path

import joblib

from app.app import create_app
from irradiance_kernel.data import load_dataset
from irradiance_kernel.pipeline import IrradiancePipeline


def _test_manifest(tmp_path):
    frame = load_dataset("landsat")
    X = frame.drop(columns="value").to_numpy()
    y = frame.value.to_numpy()
    models = []
    for index, kernel in enumerate(["linear", "rbf"], start=1):
        estimator = IrradiancePipeline(kernel=kernel).fit(X[:180], y[:180])
        model_path = tmp_path / f"model-{index}.joblib"
        joblib.dump(estimator, model_path)
        models.append(
            {
                "id": f"model-{index}",
                "dataset": "landsat",
                "rank": index,
                "config": {"model": "ksvc", "kernel": kernel, "discretizer": "uniform", "scaler": "standard", "reducer": "pca"},
                "cv_metrics": {},
                "holdout_metrics": {},
                "classes": estimator.class_descriptions_,
                "files": {"model": str(model_path), "confusion": "x.png", "curve": "y.png"},
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"models": models}), encoding="utf-8")
    return path


def test_routes(monkeypatch, tmp_path):
    manifest = _test_manifest(tmp_path)
    # Las rutas absolutas son válidas en el manifiesto de prueba.
    original_load = joblib.load
    monkeypatch.setattr(joblib, "load", lambda path: original_load(path if Path(path).is_absolute() else path))
    app = create_app({"TESTING": True, "MANIFEST_PATH": manifest})
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/api/models").status_code == 200
    assert client.get("/api/boundary").status_code == 200
    comparison = client.get("/api/compare?model_a=model-1&model_b=model-2")
    assert comparison.status_code == 200
    point = client.get(
        "/api/point?satellite=landsat&lat=1.5&lon=-77.9&model_a=model-1&model_b=model-2"
    )
    assert point.status_code == 200
    assert "classification_a" in point.get_json()
    invalid = client.get(
        "/api/point?satellite=landsat&lat=999&lon=0&model_a=model-1&model_b=model-2"
    )
    assert invalid.status_code == 400
