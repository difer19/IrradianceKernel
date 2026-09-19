"""Diagnósticos adicionales que no alteran la selección experimental original."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .constants import ARTIFACT_DIR, FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN
from .data import load_dataset
from .evaluation import ExperimentConfig, metric_bundle
from .pipeline import IrradiancePipeline


def make_spatial_groups(X, n_groups: int = 3) -> np.ndarray:
    """Agrupa coordenadas contiguas para una validación geográfica sencilla."""
    coordinates = StandardScaler().fit_transform(np.asarray(X, dtype=float)[:, :2])
    return KMeans(n_clusters=n_groups, random_state=RANDOM_STATE, n_init=20).fit_predict(
        coordinates
    )


def spatial_cv_metrics(config: ExperimentConfig, X, y, n_splits: int = 3) -> dict:
    """Evalúa un pipeline dejando una zona geográfica completa fuera por fold."""
    groups = make_spatial_groups(X, n_splits)
    rows = []
    for fit_indices, validation_indices in GroupKFold(n_splits=n_splits).split(X, y, groups):
        estimator = IrradiancePipeline(**asdict(config), random_state=RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(X[fit_indices], y[fit_indices])
        expected = estimator.transform_target(y[validation_indices])
        predicted = estimator.predict(X[validation_indices])
        scores = estimator.decision_function(X[validation_indices])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            rows.append(metric_bundle(expected, predicted, scores, estimator.classes_))
    table = pd.DataFrame(rows)
    metrics = {}
    for name in [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "min_class_f1",
        "mcc",
        "auc_ovr_macro",
    ]:
        metrics[name] = float(table[name].mean())
        metrics[f"{name}_std"] = float(table[name].std(ddof=0))
    return metrics


def _save_class_report(model: dict, output_dir: Path) -> str:
    rows = model["holdout_metrics"]["per_class"]
    labels = [f"Clase {row['label']}\n(n={row['support']})" for row in rows]
    values = [row["f1"] for row in rows]
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    bars = axis.bar(labels, values, color="#087f75")
    axis.axhline(model["baseline_metrics"]["f1_macro"], color="#e58f65", linestyle="--", label="F1 macro línea base")
    axis.set(title="F1 por clase - holdout", ylabel="F1", ylim=(0, 1.05))
    axis.bar_label(bars, fmt="%.2f", padding=3)
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path = output_dir / f"{model['id']}-class-f1.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path.relative_to(output_dir.parents[1]))


def build_diagnostics(artifact_dir: str | Path = ARTIFACT_DIR) -> dict:
    """Enriquece el manifiesto existente sin cambiar modelos ni holdout."""
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_dir = artifact_dir / "reports"
    spatial_rows = []

    for model in manifest["models"]:
        frame = load_dataset(model["dataset"])
        X = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = frame[TARGET_COLUMN].to_numpy(dtype=float)
        train_indices = np.asarray(model["train_rows"], dtype=int)
        holdout_indices = np.asarray(model["holdout_rows"], dtype=int)
        estimator = joblib.load(artifact_dir.parent / model["files"]["model"])
        expected = estimator.transform_target(y[holdout_indices])
        predicted = estimator.predict(X[holdout_indices])
        scores = estimator.decision_function(X[holdout_indices])
        model["holdout_metrics"] = metric_bundle(
            expected, predicted, scores, estimator.classes_, detailed=True
        )

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X[train_indices], estimator.transform_target(y[train_indices]))
        dummy_prediction = dummy.predict(X[holdout_indices])
        model["baseline_metrics"] = metric_bundle(expected, dummy_prediction)
        model["baseline_metrics"].pop("auc_ovr_macro")

        config = ExperimentConfig(**model["config"])
        model["spatial_cv_metrics"] = spatial_cv_metrics(
            config, X[train_indices], y[train_indices]
        )
        spatial_rows.append(
            {
                "dataset": model["dataset"],
                "rank": model["rank"],
                "model_id": model["id"],
                **model["spatial_cv_metrics"],
            }
        )
        model["files"]["class_f1"] = _save_class_report(model, report_dir)

    leaderboard_rows = []
    for dataset in ("landsat", "modis"):
        table = pd.read_csv(artifact_dir / f"{dataset}_cv_results.csv")
        valid = table.query("status == 'ok'").copy()
        valid = valid.sort_values(
            ["f1_macro_mean", "mcc_mean", "auc_ovr_macro_mean", "accuracy_mean"],
            ascending=False,
        )
        for discretizer, group in valid.groupby("discretizer", sort=True):
            row = group.iloc[0]
            leaderboard_rows.append(
                {
                    "dataset": dataset,
                    "discretizer": discretizer,
                    "scaler": row.scaler,
                    "reducer": row.reducer,
                    "model": row.model,
                    "kernel": row.kernel,
                    "f1_macro_cv": float(row.f1_macro_mean),
                    "mcc_cv": float(row.mcc_mean),
                    "auc_cv": float(row.auc_ovr_macro_mean),
                    "accuracy_cv": float(row.accuracy_mean),
                }
            )

    spatial_path = artifact_dir / "spatial_validation.csv"
    leaderboard_path = artifact_dir / "leaderboard_by_discretizer.csv"
    pd.DataFrame(spatial_rows).to_csv(spatial_path, index=False)
    pd.DataFrame(leaderboard_rows).to_csv(leaderboard_path, index=False)
    manifest["diagnostics"] = {
        "note": "Diagnósticos posteriores a la selección; no cambian el ranking ni usan el holdout para elegir modelos.",
        "baseline": "DummyClassifier(strategy='most_frequent')",
        "spatial_validation": "GroupKFold de 3 zonas obtenidas con KMeans sobre las coordenadas del 80% de desarrollo.",
        "files": {
            "spatial_validation": str(spatial_path.relative_to(artifact_dir.parent)),
            "leaderboard_by_discretizer": str(leaderboard_path.relative_to(artifact_dir.parent)),
        },
    }
    manifest["schema_version"] = 2
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return manifest
