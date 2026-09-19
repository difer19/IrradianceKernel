"""Constantes compartidas por experimentos, notebook y aplicación."""

from pathlib import Path

RANDOM_STATE = 2021
FEATURE_COLUMNS = ["latitude", "longitude", *[f"band{i}" for i in range(1, 8)]]
TARGET_COLUMN = "value"

SCALERS = ("standard", "minmax", "normalizer")
DISCRETIZERS = ("uniform", "kmeans", "dbscan", "agglomerative")
REDUCERS = ("pca", "lda")
MODELS = ("ksvc", "kannc", "kridge")
KERNELS = (
    "linear",
    "poly",
    "rbf",
    "hyperbolic",
    "triangle",
    "radial_basic",
    "rquadratic",
    "canberra",
    "truncated",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
