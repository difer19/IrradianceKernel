"""Carga, validación y transformación geográfica de los datos."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DATA_DIR, FEATURE_COLUMNS, TARGET_COLUMN


DATASET_FILES = {
    "landsat": DATA_DIR / "landsat_model.csv",
    "modis": DATA_DIR / "modis_model.csv",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(name: str) -> pd.DataFrame:
    """Carga un dataset y comprueba su contrato mínimo."""
    key = name.lower()
    if key not in DATASET_FILES:
        raise ValueError(f"Dataset desconocido: {name!r}")
    frame = pd.read_csv(DATASET_FILES[key])
    expected = FEATURE_COLUMNS + [TARGET_COLUMN]
    if list(frame.columns) != expected:
        raise ValueError(f"Columnas inesperadas en {key}: {list(frame.columns)}")
    if frame.isna().any().any():
        raise ValueError(f"El dataset {key} contiene valores ausentes")
    return frame


def web_mercator_to_wgs84(x, y):
    """Convierte EPSG:3857 a (latitud, longitud) WGS84 sin dependencia GIS."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    longitude = np.degrees(x_arr / 6378137.0)
    latitude = np.degrees(2.0 * np.arctan(np.exp(y_arr / 6378137.0)) - np.pi / 2.0)
    if latitude.ndim == 0:
        return float(latitude), float(longitude)
    return latitude, longitude


def wgs84_to_web_mercator(latitude, longitude):
    """Convierte (latitud, longitud) WGS84 a EPSG:3857."""
    lat = np.clip(np.asarray(latitude, dtype=float), -85.05112878, 85.05112878)
    lon = np.asarray(longitude, dtype=float)
    x = 6378137.0 * np.radians(lon)
    y = 6378137.0 * np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))
    if x.ndim == 0:
        return float(x), float(y)
    return x, y
