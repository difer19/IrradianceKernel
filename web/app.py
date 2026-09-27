"""Aplicación Flask para explorar los modelos de irradiancia exportados."""

from __future__ import annotations

from pathlib import Path
import math
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"
DATASETS_DIR = PROJECT_DIR / "datasets"
FEATURE_NAMES = [
    "latitude", "longitude", "band1", "band2", "band3",
    "band4", "band5", "band6", "band7",
]
SATELLITES = ("landsat", "modis")
WEB_MERCATOR_RADIUS = 6_378_137.0

app = Flask(__name__)
MODEL_REGISTRY: dict[str, dict[str, Any]] = {}


def _safe_number(value: Any) -> float | None:
    """Convierte escalares NumPy a JSON y representa NaN como null."""
    value = float(value)
    return value if math.isfinite(value) else None


def _epsg3857_to_wgs84(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convierte x/y EPSG:3857 a longitud/latitud para Leaflet."""
    longitude = np.degrees(x / WEB_MERCATOR_RADIUS)
    latitude = np.degrees(
        2.0 * np.arctan(np.exp(y / WEB_MERCATOR_RADIUS)) - np.pi / 2.0
    )
    return latitude, longitude


def _class_label(centers: np.ndarray, class_id: int) -> str:
    """Explica una clase con sus límites aproximados de irradiancia."""
    centers = np.asarray(centers, dtype=float)
    class_id = int(class_id)
    if len(centers) == 1:
        return f"Clase {class_id} (≈ {centers[0]:.2f})"
    boundaries = (centers[:-1] + centers[1:]) / 2.0
    if class_id == 0:
        return f"Clase 0 (≤ {boundaries[0]:.2f})"
    if class_id == len(centers) - 1:
        return f"Clase {class_id} (≥ {boundaries[-1]:.2f})"
    return (
        f"Clase {class_id} ({boundaries[class_id - 1]:.2f}–"
        f"{boundaries[class_id]:.2f})"
    )


def _classes_payload(centers: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"id": index, "label": _class_label(centers, index),
         "center": _safe_number(center)}
        for index, center in enumerate(np.asarray(centers, dtype=float))
    ]


def _score_matrix(scores: Any, n_classes: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack((-scores, scores))
    if scores.shape[1] == n_classes:
        return scores
    # Mantiene una forma válida aunque una implementación devuelva menos
    # columnas por una clase ausente en un conjunto concreto.
    padded = np.full((len(scores), n_classes), -np.inf, dtype=float)
    padded[:, :min(scores.shape[1], n_classes)] = scores[:, :n_classes]
    return padded


def _curve_payload(y_true: np.ndarray, scores: np.ndarray, n_classes: int) -> dict[str, Any]:
    one_hot = np.zeros((len(y_true), n_classes), dtype=int)
    one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1
    counts = np.bincount(y_true.astype(int), minlength=n_classes)
    nonzero = counts[counts > 0]
    minority = int(nonzero.min()) if len(nonzero) else 0
    majority = int(nonzero.max()) if len(nonzero) else 0
    imbalanced = (
        minority == 0 or minority / max(len(y_true), 1) < 0.10
        or majority / max(minority, 1) > 2
    )

    if imbalanced:
        x, y, _ = precision_recall_curve(one_hot.ravel(), scores.ravel())
        curve_type = "precision_recall"
        x_label, y_label = "Recall", "Precision"
    else:
        x, y, _ = roc_curve(one_hot.ravel(), scores.ravel())
        curve_type = "roc"
        x_label, y_label = "False positive rate", "True positive rate"

    # El navegador no necesita cientos de puntos repetidos para dibujarla.
    if len(x) > 180:
        selected = np.linspace(0, len(x) - 1, 180).astype(int)
        x, y = x[selected], y[selected]
    return {
        "type": curve_type,
        "x_label": x_label,
        "y_label": y_label,
        "x": [_safe_number(value) for value in x],
        "y": [_safe_number(value) for value in y],
    }


def _evaluate_holdout(artifact: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    """Calcula métricas sobre el mismo holdout fijo usado en el notebook."""
    pipeline = artifact["pipeline"]
    classes = np.asarray(artifact["classes"], dtype=int)
    n_classes = len(classes)
    indices = np.arange(len(frame))
    _, holdout_indices = train_test_split(
        indices, test_size=0.20, random_state=2021, shuffle=True
    )
    features = frame[FEATURE_NAMES].to_numpy(dtype=float)
    y_holdout = pipeline.transform_target(
        frame.loc[holdout_indices, "value"].to_numpy(dtype=float)
    ).astype(int)
    predictions = np.asarray(pipeline.predict(features[holdout_indices]), dtype=int)
    scores = _score_matrix(
        pipeline.decision_function(features[holdout_indices]), n_classes
    )
    labels = np.arange(n_classes)

    try:
        if n_classes == 2:
            auc = roc_auc_score(y_holdout, scores[:, 1])
        else:
            one_hot = np.zeros((len(y_holdout), n_classes), dtype=int)
            one_hot[np.arange(len(y_holdout)), y_holdout] = 1
            auc = roc_auc_score(one_hot, scores, average="macro")
    except ValueError:
        auc = np.nan

    return {
        "metrics": {
            "accuracy": _safe_number(accuracy_score(y_holdout, predictions)),
            "f1_macro": _safe_number(
                f1_score(y_holdout, predictions, average="macro", zero_division=0)
            ),
            "auc_ovr": _safe_number(auc),
            "mcc": _safe_number(matthews_corrcoef(y_holdout, predictions)),
        },
        "confusion_matrix": confusion_matrix(
            y_holdout, predictions, labels=labels
        ).astype(int).tolist(),
        "curve": _curve_payload(y_holdout, scores, n_classes),
        "holdout_size": int(len(holdout_indices)),
    }


def _map_points(artifact: dict[str, Any], frame: pd.DataFrame) -> list[dict[str, Any]]:
    pipeline = artifact["pipeline"]
    centers = np.asarray(artifact["target_centers"], dtype=float)
    features = frame[FEATURE_NAMES].to_numpy(dtype=float)
    predictions = np.asarray(pipeline.predict(features), dtype=int)
    # En los CSV originales los nombres de las columnas están invertidos:
    # `latitude` guarda X (longitud) y `longitude` guarda Y (latitud), ambos
    # en metros EPSG:3857. Las características del modelo se conservan tal
    # como fueron entrenadas; aquí solo corregimos la visualización geográfica.
    latitude, longitude = _epsg3857_to_wgs84(
        frame["latitude"].to_numpy(dtype=float),
        frame["longitude"].to_numpy(dtype=float),
    )
    points = []
    for index in range(len(frame)):
        class_id = int(predictions[index])
        points.append({
            "index": index,
            "lat": _safe_number(latitude[index]),
            "lon": _safe_number(longitude[index]),
            "x_epsg3857": _safe_number(frame.iloc[index]["latitude"]),
            "y_epsg3857": _safe_number(frame.iloc[index]["longitude"]),
            "irradiance": _safe_number(frame.iloc[index]["value"]),
            "predicted_class": class_id,
            "class_label": _class_label(centers, class_id),
        })
    return points


def load_registry() -> None:
    """Carga exclusivamente los artefactos existentes; nunca entrena desde la web."""
    MODEL_REGISTRY.clear()
    model_paths = sorted(MODELS_DIR.glob("*_best.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"No hay modelos *_best.joblib en {MODELS_DIR}")

    for model_path in model_paths:
        artifact = joblib.load(model_path)
        satellite = artifact["satellite"]
        if satellite not in SATELLITES:
            continue
        data_path = DATASETS_DIR / f"{satellite}_model.csv"
        frame = pd.read_csv(data_path)
        evaluation = _evaluate_holdout(artifact, frame)
        points = _map_points(artifact, frame)
        model_id = model_path.stem
        MODEL_REGISTRY[model_id] = {
            "id": model_id,
            "satellite": satellite,
            "configuration": artifact["configuration"],
            "artifact": artifact,
            "frame": frame,
            "metrics": evaluation["metrics"],
            "confusion_matrix": evaluation["confusion_matrix"],
            "curve": evaluation["curve"],
            "holdout_size": evaluation["holdout_size"],
            "classes": _classes_payload(artifact["target_centers"]),
            "points": points,
        }


def _model_or_error(model_id: str) -> dict[str, Any]:
    model = MODEL_REGISTRY.get(model_id)
    if model is None:
        raise KeyError(model_id)
    return model


def _summary(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": model["id"],
        "satellite": model["satellite"],
        "configuration": model["configuration"],
        "metrics": model["metrics"],
        "classes": model["classes"],
        "holdout_size": model["holdout_size"],
        "point_count": len(model["points"]),
    }


def _comparison_model(model: dict[str, Any]) -> dict[str, Any]:
    payload = _summary(model)
    payload.update({
        "points": model["points"],
        "confusion_matrix": model["confusion_matrix"],
        "curve": model["curve"],
    })
    return payload


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/models")
def api_models():
    return jsonify({"models": [_summary(model) for model in MODEL_REGISTRY.values()]})


@app.get("/api/compare")
def api_compare():
    model_a_id = request.args.get("model_a", "")
    model_b_id = request.args.get("model_b", "")
    if model_a_id == model_b_id:
        return jsonify({"error": "Selecciona dos modelos diferentes."}), 400
    try:
        model_a = _model_or_error(model_a_id)
        model_b = _model_or_error(model_b_id)
    except KeyError as error:
        return jsonify({"error": f"Modelo no encontrado: {error.args[0]}"}), 404
    return jsonify({
        "model_a": _comparison_model(model_a),
        "model_b": _comparison_model(model_b),
        "same_satellite": model_a["satellite"] == model_b["satellite"],
    })


@app.get("/api/point")
def api_point():
    satellite = request.args.get("satellite", "")
    model_ids = [request.args.get("model_a", ""), request.args.get("model_b", "")]
    if satellite not in SATELLITES:
        return jsonify({"error": "Satélite no válido."}), 400
    try:
        query_lat = float(request.args["lat"])
        query_lon = float(request.args["lon"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "lat y lon deben ser números."}), 400
    if not math.isfinite(query_lat) or not math.isfinite(query_lon):
        return jsonify({"error": "lat y lon deben ser finitos."}), 400

    try:
        selected_model = next(
            model for model in MODEL_REGISTRY.values()
            if model["satellite"] == satellite
        )
    except StopIteration:
        return jsonify({"error": "No hay un modelo para ese satélite."}), 404
    frame = selected_model["frame"]
    points = selected_model["points"]
    distances = np.asarray([
        (point["lat"] - query_lat) ** 2
        + ((point["lon"] - query_lon) * math.cos(math.radians(query_lat))) ** 2
        for point in points
    ])
    nearest_index = int(np.argmin(distances))
    nearest = points[nearest_index]
    distance_km = math.sqrt(float(distances[nearest_index])) * 111.32
    row_features = frame.iloc[[nearest_index]][FEATURE_NAMES].to_numpy(dtype=float)

    predictions = []
    for model_id in model_ids:
        if not model_id:
            continue
        try:
            model = _model_or_error(model_id)
        except KeyError:
            return jsonify({"error": f"Modelo no encontrado: {model_id}"}), 404
        predicted_class = int(model["artifact"]["pipeline"].predict(row_features)[0])
        centers = np.asarray(model["artifact"]["target_centers"], dtype=float)
        predictions.append({
            "model_id": model_id,
            "satellite": model["satellite"],
            "native_dataset": model["satellite"] == satellite,
            "predicted_class": predicted_class,
            "class_label": _class_label(centers, predicted_class),
        })

    return jsonify({
        "query": {"satellite": satellite, "lat": query_lat, "lon": query_lon},
        "point": nearest,
        "distance_km": round(distance_km, 4),
        "predictions": predictions,
        "note": (
            "El punto se busca entre las observaciones del satélite seleccionado. "
            "Si el segundo modelo pertenece a otro satélite, su predicción es una "
            "comparación exploratoria usando las mismas nueve variables."
        ),
    })


@app.errorhandler(KeyError)
def handle_key_error(error):
    return jsonify({"error": f"Clave no encontrada: {error.args[0]}"}), 404


load_registry()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
