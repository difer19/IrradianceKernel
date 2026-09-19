"""Aplicación Flask para comparar modelos de clasificación de irradiancia."""

from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from irradiance_kernel.constants import FEATURE_COLUMNS, TARGET_COLUMN
from irradiance_kernel.data import load_dataset, web_mercator_to_wgs84, wgs84_to_web_mercator


def create_app(test_config=None):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(MANIFEST_PATH=PROJECT_ROOT / "artifacts" / "manifest.json")
    if test_config:
        app.config.update(test_config)

    def read_manifest():
        path = Path(app.config["MANIFEST_PATH"])
        if not path.exists():
            raise FileNotFoundError(
                "No existe artifacts/manifest.json. Ejecute python run_experiments.py primero."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def get_metadata(model_id):
        for model in read_manifest().get("models", []):
            if model["id"] == model_id:
                return model
        raise KeyError(f"Modelo desconocido: {model_id}")

    @lru_cache(maxsize=12)
    def load_model(model_id):
        metadata = get_metadata(model_id)
        path = PROJECT_ROOT / metadata["files"]["model"]
        return joblib.load(path)

    def validate_pair(model_a, model_b, satellite=None):
        left = get_metadata(model_a)
        right = get_metadata(model_b)
        if left["dataset"] != right["dataset"]:
            raise ValueError("Los modelos deben pertenecer al mismo satélite")
        if satellite and left["dataset"] != satellite:
            raise ValueError("El satélite no coincide con los modelos seleccionados")
        return left, right

    def public_metadata(model):
        return {
            key: value
            for key, value in model.items()
            if key not in {"train_rows", "holdout_rows"}
        }

    def map_points(metadata):
        dataset = metadata["dataset"]
        frame = load_dataset(dataset)
        estimator = load_model(metadata["id"])
        predictions = estimator.predict(frame[FEATURE_COLUMNS].to_numpy(dtype=float))
        latitudes, longitudes = web_mercator_to_wgs84(
            frame["latitude"].to_numpy(), frame["longitude"].to_numpy()
        )
        return [
            {
                "lat": round(float(lat), 6),
                "lon": round(float(lon), 6),
                "class": int(label),
                "value": float(value),
            }
            for lat, lon, label, value in zip(
                latitudes, longitudes, predictions, frame[TARGET_COLUMN]
            )
        ]

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/models")
    def models_api():
        manifest = read_manifest()
        return jsonify(
            {
                "models": [public_metadata(model) for model in manifest.get("models", [])],
                "datasets": manifest.get("datasets", {}),
                "created_with": manifest.get("created_with", {}),
            }
        )

    @app.get("/api/compare")
    def compare_api():
        model_a = request.args.get("model_a", "")
        model_b = request.args.get("model_b", "")
        left, right = validate_pair(model_a, model_b)
        return jsonify(
            {
                "model_a": public_metadata(left),
                "model_b": public_metadata(right),
                "points_a": map_points(left),
                "points_b": map_points(right),
            }
        )

    @app.get("/api/point")
    def point_api():
        satellite = request.args.get("satellite", "").lower()
        model_a = request.args.get("model_a", "")
        model_b = request.args.get("model_b", "")
        left, right = validate_pair(model_a, model_b, satellite)
        try:
            latitude = float(request.args["lat"])
            longitude = float(request.args["lon"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("lat y lon deben ser números") from exc
        if not math.isfinite(latitude) or not math.isfinite(longitude):
            raise ValueError("lat y lon deben ser finitos")
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("Coordenadas fuera del rango WGS84")

        x, y = wgs84_to_web_mercator(latitude, longitude)
        frame = load_dataset(satellite)
        distances = np.hypot(frame["latitude"].to_numpy() - x, frame["longitude"].to_numpy() - y)
        nearest_index = int(np.argmin(distances))
        row = frame.iloc[[nearest_index]]
        features = row[FEATURE_COLUMNS].to_numpy(dtype=float)
        point_lat, point_lon = web_mercator_to_wgs84(
            row.iloc[0]["latitude"], row.iloc[0]["longitude"]
        )

        def classification(metadata):
            label = int(load_model(metadata["id"]).predict(features)[0])
            description = next(
                (item for item in metadata["classes"] if int(item["label"]) == label), None
            )
            return {"model_id": metadata["id"], "class": label, "description": description}

        return jsonify(
            {
                "clicked": {"lat": latitude, "lon": longitude},
                "nearest": {
                    "row": nearest_index,
                    "lat": point_lat,
                    "lon": point_lon,
                    "distance_m": float(distances[nearest_index]),
                    "irradiance": float(row.iloc[0][TARGET_COLUMN]),
                },
                "classification_a": classification(left),
                "classification_b": classification(right),
            }
        )

    @app.get("/artifacts/<path:filename>")
    def artifact_file(filename):
        return send_from_directory(PROJECT_ROOT / "artifacts", filename)

    @app.errorhandler(KeyError)
    @app.errorhandler(ValueError)
    def bad_request(error):
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(FileNotFoundError)
    def missing_artifacts(error):
        return jsonify({"error": str(error)}), 503

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
