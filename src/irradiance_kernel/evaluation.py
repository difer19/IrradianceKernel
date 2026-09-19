"""Evaluación reproducible de todas las estructuras de pipeline."""

from __future__ import annotations

import itertools
import json
import platform
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, train_test_split

from .constants import (
    ARTIFACT_DIR,
    DISCRETIZERS,
    FEATURE_COLUMNS,
    KERNELS,
    MODELS,
    RANDOM_STATE,
    REDUCERS,
    SCALERS,
    TARGET_COLUMN,
)
from .data import DATASET_FILES, load_dataset, sha256_file
from .pipeline import IrradiancePipeline


@dataclass(frozen=True)
class ExperimentConfig:
    scaler: str
    discretizer: str
    reducer: str
    model: str
    kernel: str

    @property
    def config_id(self) -> str:
        return "__".join((self.scaler, self.discretizer, self.reducer, self.model, self.kernel))


def generate_configurations() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(*values)
        for values in itertools.product(SCALERS, DISCRETIZERS, REDUCERS, MODELS, KERNELS)
    ]


def _multiclass_auc(y_true, scores, score_classes) -> float:
    values = []
    for column, label in enumerate(score_classes):
        binary = (np.asarray(y_true) == label).astype(int)
        if np.unique(binary).size < 2:
            continue
        values.append(roc_auc_score(binary, scores[:, column]))
    return float(np.mean(values)) if values else float("nan")


def metric_bundle(y_true, y_pred, scores=None, score_classes=None, *, detailed=False) -> dict:
    labels = np.union1d(y_true, y_pred)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "min_class_f1": float(class_f1.min()) if class_f1.size else float("nan"),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auc_ovr_macro": float("nan"),
    }
    if scores is not None and score_classes is not None:
        metrics["auc_ovr_macro"] = _multiclass_auc(y_true, scores, score_classes)
    if detailed:
        metrics["per_class"] = [
            {
                "label": int(label),
                "support": int(class_support),
                "precision": float(class_precision),
                "recall": float(class_recall),
                "f1": float(label_f1),
            }
            for label, class_support, class_precision, class_recall, label_f1 in zip(
                labels, support, precision, recall, class_f1
            )
        ]
    return metrics


def evaluate_configuration(config, X_train, y_train, splits) -> dict:
    config_data = asdict(config)
    config_data["config_id"] = config.config_id
    fold_rows = []
    try:
        for fold, (fit_indices, validation_indices) in enumerate(splits, start=1):
            estimator = IrradiancePipeline(
                scaler=config.scaler,
                discretizer=config.discretizer,
                reducer=config.reducer,
                model=config.model,
                kernel=config.kernel,
                random_state=RANDOM_STATE,
            )
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X_train[fit_indices], y_train[fit_indices])
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            predictions = estimator.predict(X_train[validation_indices])
            scores = estimator.decision_function(X_train[validation_indices])
            score_seconds = time.perf_counter() - started
            expected = estimator.transform_target(y_train[validation_indices])
            row = metric_bundle(expected, predictions, scores, estimator.classes_)
            row.update(
                {
                    "fold": fold,
                    "fit_seconds": fit_seconds,
                    "score_seconds": score_seconds,
                    "n_classes": len(estimator.classes_),
                }
            )
            fold_rows.append(row)
        folds = pd.DataFrame(fold_rows)
        config_data.update({"status": "ok", "error": ""})
        for metric in [
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "min_class_f1",
            "mcc",
            "auc_ovr_macro",
            "fit_seconds",
            "score_seconds",
            "n_classes",
        ]:
            config_data[f"{metric}_mean"] = float(folds[metric].mean())
            config_data[f"{metric}_std"] = float(folds[metric].std(ddof=0))
    except Exception as exc:  # El informe debe conservar fallos de configuraciones individuales.
        config_data.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return config_data


def _slug(dataset: str, rank: int, config: ExperimentConfig) -> str:
    return f"{dataset}-top{rank}-{config.model}-{config.kernel}-{config.discretizer}-{config.scaler}-{config.reducer}"


def _class_balance_requires_pr(y) -> bool:
    _, counts = np.unique(y, return_counts=True)
    if not len(counts):
        return False
    return counts.min() / counts.sum() < 0.10 or counts.max() / counts.min() > 2.0


def _save_confusion(y_true, y_pred, labels, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, cmap="Blues", colorbar=False, ax=axis
    )
    axis.set_title("Matriz de confusión - holdout")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_curve(y_true, scores, classes, path: Path) -> str:
    use_pr = _class_balance_requires_pr(y_true)
    figure, axis = plt.subplots(figsize=(6.5, 5))
    for column, label in enumerate(classes):
        binary = (np.asarray(y_true) == label).astype(int)
        if np.unique(binary).size < 2:
            continue
        if use_pr:
            precision, recall, _ = precision_recall_curve(binary, scores[:, column])
            score = average_precision_score(binary, scores[:, column])
            axis.plot(recall, precision, label=f"Clase {label} (AP={score:.3f})")
        else:
            false_positive, true_positive, _ = roc_curve(binary, scores[:, column])
            score = auc(false_positive, true_positive)
            axis.plot(false_positive, true_positive, label=f"Clase {label} (AUC={score:.3f})")
    if use_pr:
        axis.set(xlabel="Recall", ylabel="Precisión", title="Curvas precisión-recall")
        curve_type = "precision_recall"
    else:
        axis.plot([0, 1], [0, 1], "--", color="#64748b", linewidth=1)
        axis.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title="Curvas ROC")
        curve_type = "roc"
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return curve_type


def _safe_json(value):
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_dataset_experiments(
    dataset: str,
    *,
    output_dir: str | Path = ARTIFACT_DIR,
    n_jobs: int = 1,
    max_configs: int | None = None,
) -> dict:
    """Ejecuta CV, selecciona por CV y evalúa los tres ganadores en holdout."""
    output = Path(output_dir)
    model_dir = output / "models"
    report_dir = output / "reports"
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(dataset)
    X = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = frame[TARGET_COLUMN].to_numpy(dtype=float)
    all_indices = np.arange(len(frame))
    train_indices, holdout_indices = train_test_split(
        all_indices, test_size=0.20, random_state=RANDOM_STATE, shuffle=True
    )
    X_train, y_train = X[train_indices], y[train_indices]
    splitter = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(X_train))
    configurations = generate_configurations()
    if max_configs is not None:
        configurations = configurations[:max_configs]

    rows = Parallel(n_jobs=n_jobs, verbose=10 if n_jobs != 1 else 0)(
        delayed(evaluate_configuration)(config, X_train, y_train, splits)
        for config in configurations
    )
    results = pd.DataFrame(rows)
    result_path = output / f"{dataset}_cv_results.csv"
    results.to_csv(result_path, index=False)
    successful = results[results["status"] == "ok"].copy()
    if successful.empty:
        raise RuntimeError(f"Ninguna configuración terminó correctamente para {dataset}")
    successful = successful.sort_values(
        ["f1_macro_mean", "mcc_mean", "auc_ovr_macro_mean", "accuracy_mean", "fit_seconds_mean"],
        ascending=[False, False, False, False, True],
        na_position="last",
    )

    artifacts = []
    rank = 1
    for row in successful.to_dict("records"):
        if rank > 3:
            break
        config = ExperimentConfig(
            row["scaler"], row["discretizer"], row["reducer"], row["model"], row["kernel"]
        )
        estimator = IrradiancePipeline(**asdict(config), random_state=RANDOM_STATE)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X_train, y_train)
            y_holdout = estimator.transform_target(y[holdout_indices])
            predictions = estimator.predict(X[holdout_indices])
            scores = estimator.decision_function(X[holdout_indices])
        except Exception:
            continue
        holdout_metrics = metric_bundle(
            y_holdout, predictions, scores, estimator.classes_, detailed=True
        )
        slug = _slug(dataset, rank, config)
        model_path = model_dir / f"{slug}.joblib"
        confusion_path = report_dir / f"{slug}-confusion.png"
        curve_path = report_dir / f"{slug}-curve.png"
        joblib.dump(estimator, model_path)
        _save_confusion(y_holdout, predictions, estimator.classes_, confusion_path)
        curve_type = _save_curve(y_holdout, scores, estimator.classes_, curve_path)
        artifacts.append(
            {
                "id": slug,
                "dataset": dataset,
                "rank": rank,
                "config": asdict(config),
                "cv_metrics": {
                    key.removesuffix("_mean"): row[key]
                    for key in [
                        "accuracy_mean",
                        "f1_macro_mean",
                        "mcc_mean",
                        "auc_ovr_macro_mean",
                        "fit_seconds_mean",
                        "score_seconds_mean",
                    ]
                },
                "holdout_metrics": holdout_metrics,
                "classes": estimator.class_descriptions_,
                "resolved_kernel": estimator.resolved_kernel(),
                "files": {
                    "model": str(model_path.relative_to(output.parent)),
                    "confusion": str(confusion_path.relative_to(output.parent)),
                    "curve": str(curve_path.relative_to(output.parent)),
                },
                "curve_type": curve_type,
                "train_rows": train_indices.tolist(),
                "holdout_rows": holdout_indices.tolist(),
            }
        )
        rank += 1
    if len(artifacts) < 3:
        raise RuntimeError(f"Solo se pudieron guardar {len(artifacts)} modelos para {dataset}")
    return {"dataset": dataset, "results": str(result_path), "artifacts": artifacts}


def run_all_experiments(*, n_jobs: int = 1, max_configs: int | None = None) -> dict:
    manifests = [
        run_dataset_experiments(name, n_jobs=n_jobs, max_configs=max_configs)
        for name in ("landsat", "modis")
    ]
    payload = {
        "schema_version": 1,
        "created_with": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "random_state": RANDOM_STATE,
        "datasets": {
            name: {
                "file": str(path.relative_to(path.parents[2])),
                "sha256": sha256_file(path),
                "rows": len(load_dataset(name)),
            }
            for name, path in DATASET_FILES.items()
        },
        "models": [artifact for manifest in manifests for artifact in manifest["artifacts"]],
        "result_files": {manifest["dataset"]: manifest["results"] for manifest in manifests},
    }
    manifest_path = ARTIFACT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(_safe_json(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    # Los diagnósticos se calculan después de cerrar la selección y nunca alteran el ranking.
    from .diagnostics import build_diagnostics

    return build_diagnostics(ARTIFACT_DIR)
